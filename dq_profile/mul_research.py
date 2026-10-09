"""Paired, bounded fixed-mul experiments on the isolated diagnostic runtime.

This is a research protocol, not the canonical Local acceptance selector.
It reuses the Local pass, replay, quant RNG and metric definitions explicitly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from library.dq_mul_policy import MulPolicy, UNET_GROUPS, digest, module_group
from dq_profile.geometry import SourceGroupMap
from dq_profile.protocol import CandidateDefinition, canonical_sha256
from dq_profile.quant_context import ProfileQuantContext
from dq_profile.replay import seed_step_rng
from dq_profile.snapshot import TrainingSnapshot
from dq_profile.trainer_runtime import DiagnosticProfileRuntime, _combined_dq_metrics
from dq_profile.v2_calibration import fingerprint_tree, gradient_tail_rows
from dq_profile.v24_acceptance import _body_tail, _cluster_quantile, _index_rows
from dq_profile.research_budget import ResearchBudget, ResearchStop, read_json, write_json, file_hash


def candidate(name, unet, te=None, *, overrides=None, te_quantized=True):
    policy = {"base_mul": unet, "components": {"unet": unet, "te1": unet if te is None else te, "te2": unet if te is None else te}, "te_quantized": te_quantized, "group_overrides": [{"group_id": k, "range_mul": v} for k, v in (overrides or {}).items()]}
    return {"id": name, "policy": policy}


def initial_candidates():
    scope = []
    for mul in (2.70, 3.15, 3.75):
        scope += [candidate(f"U_{mul:.2f}", mul, te_quantized=False), candidate(f"UT_{mul:.2f}", mul)]
    cross = [candidate("U2.70_T3.75", 2.70, 3.75), candidate("U3.75_T2.70", 3.75, 2.70)]
    interventions = []
    for group in UNET_GROUPS:
        interventions += [candidate(f"low_restore_{group}", 2.70, 3.75, overrides={group: 3.75}), candidate(f"high_lower_{group}", 3.75, 3.75, overrides={group: 2.70})]
    return scope, cross, interventions


class ResearchQuantContext(ProfileQuantContext):
    def begin_pass(self, **kwargs):
        super().begin_pass(**kwargs)
        self.executed_quant = {}
        # Compare actual dropout sites/masks independent of the quant repeat.
        self._dropout_hasher = hashlib.sha256()

    def rand_for(self, x, *, module_name, invocation=None, repeat=None):
        before = self._invocations[module_name] if invocation is None else invocation
        result = super().rand_for(x, module_name=module_name, invocation=invocation, repeat=repeat)
        self.executed_quant.setdefault(module_name, []).append((before, self.repeat if repeat is None else repeat, tuple(x.shape)))
        return result

    def pass_trace(self):
        result = super().pass_trace()
        common = {k: v for k, v in self.executed_quant.items() if k.startswith("lora_unet_")}
        result.update(effective_quant_count=len(self.executed_quant), effective_quant_set_sha256=digest(sorted(self.executed_quant)), shared_unet_quant_addresses_sha256=digest(common))
        return result


def summarize_samples(samples, *, bins=4, iterations=2000, seed=2401):
    """Use the existing source weighting, quantiles and source bootstrap."""
    grouped = defaultdict(list)
    for row in samples:
        if not math.isfinite(row["relative_gradient_distance"]):
            raise ResearchStop("Nonfinite local gradient distance")
        grouped[row["candidate"]].append(row)
    results = []
    for name, rows in sorted(grouped.items()):
        all_groups, by_bin = _index_rows(rows, timestep_bins=bins)
        sources = sorted(all_groups)
        point = _body_tail(all_groups, by_bin, sources, field="relative_gradient_distance", timestep_bins=bins)
        median = _cluster_quantile(all_groups, sources, "relative_gradient_distance", .5)
        rng = np.random.default_rng(seed)
        draws = []
        for indexes in rng.integers(0, len(sources), size=(iterations, len(sources))):
            selected = [sources[i] for i in indexes]
            draw = _body_tail(all_groups, by_bin, selected, field="relative_gradient_distance", timestep_bins=bins)
            draws.append((draw["body"], draw["tail"], _cluster_quantile(all_groups, selected, "relative_gradient_distance", .5)))
        ci = np.quantile(np.asarray(draws), [.025, .975], axis=0)
        results.append({"candidate": name, "body": point["body"], "tail": point["tail"], "p50": median, "per_bin": point["per_bin"], "worst_bin": point["worst_timestep_bin"], "body_ci": ci[:, 0].tolist(), "tail_ci": ci[:, 1].tolist(), "p50_ci": ci[:, 2].tolist(), "images": len({r["image_key"] for r in rows}), "sources": len(sources), "observations": len(rows), "source_bootstrap_iterations": iterations})
    return results


def recipient_energy(reference, value, *, denominator=None, per_module=False):
    if set(reference.values) != set(value.values):
        raise ResearchStop("Gradient topology changed")
    totals = defaultdict(lambda: {"reference_sq": 0., "difference_sq": 0.})
    for name, left in reference.values.items():
        module_name = name.split(".", 1)[0]
        group = module_name if per_module else module_group(module_name)
        right = value.values[name]
        if left.shape != right.shape:
            raise ResearchStop("Gradient shape changed")
        totals[group]["reference_sq"] += float(torch.sum(left * left))
        totals[group]["difference_sq"] += float(torch.sum((right - left) ** 2))
    norm_sq = reference.norm_sq if denominator is None else denominator
    return {k: {**v, "contribution": v["difference_sq"] / max(norm_sq, 1e-30), "relative_distance": math.sqrt(v["difference_sq"] / max(v["reference_sq"], 1e-30))} for k, v in totals.items()}


class MulResearchRuntime(DiagnosticProfileRuntime):
    probe_address_version = "legacy-index-v1"
    collect_module_energy = False

    def _probe_addresses(self, image_index, bin_index, noise, image_key):
        if self.probe_address_version == "stable-image-v1":
            # Neither subset order nor dropout regime changes quant addresses.
            # The model RNG uses its own phase and never quantization repeats.
            return (f"tail-stable:{bin_index}:{noise}:{image_key}",
                    f"image:{image_key}|bin:{bin_index}|noise:{noise}")
        return (f"tail:{image_index}:{bin_index}:{noise}:{image_key}",
                f"image:{image_index}|bin:{bin_index}|noise:{noise}|key:{image_key}")

    def __init__(self, *, args, trainer):
        super().__init__(args=args, trainer=trainer)
        self.budget = args._dq_research_budget
        self.quant_context = ResearchQuantContext(self.protocol_seed)
        self.policies = {}
        self.expanded = {}
        self.reference_hashes = {}
        self.stage_reports = []
        self.pass_timings = []
        self._legacy_parity = False
        self.run_root = Path(args.dq_profile_run_dir)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _configure_candidate_quantization(self, network, candidate, range_mul, shadow):
        super()._configure_candidate_quantization(network, candidate, range_mul, shadow)
        if not self._legacy_parity and candidate.quantized:
            self.policies[candidate.name].apply(network)
        self.expected_quant = sorted(m.lora_name for m in network.unet_loras + network.text_encoder_loras if m.delta_q_enabled)

    def _extra_quantization_metrics(self, exported):
        accum = (exported or {}).get("accum", {})
        result = {}
        for scope in ("unet", "te"):
            metrics = _combined_dq_metrics({"accum": {scope: accum.get(scope)}})
            for key, value in metrics.items():
                result[f"{scope}_{key}"] = value
            accumulator = accum.get(scope)
            count = getattr(accumulator, "numel", None)
            result[f"{scope}_quant_numel"] = None if count is None else float(count)
        return result

    def _run_pass(self, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = super()._run_pass(**kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row, gradient, shadow = result
        if set(self.quant_context.executed_quant) != set(self.expected_quant):
            raise ResearchStop("Actual quantized modules differ from policy")
        if not math.isfinite(row["loss"]) or not math.isfinite(gradient.norm) or gradient.norm <= 0:
            raise ResearchStop("Nonfinite/zero reference or candidate gradient")
        row["seconds"] = seconds
        self.pass_timings.append({"stage": self.budget.stage, "candidate": row["candidate"], "seconds": seconds, "cuda_peak_allocated": torch.cuda.max_memory_allocated(), "cuda_peak_reserved": torch.cuda.max_memory_reserved(), "cpu_rss": psutil.Process().memory_info().rss})
        if len(self.pass_timings) % 20 == 0:
            write_json(self.run_root / "progress.json", {"stage": self.budget.stage, "completed_passes": len(self.pass_timings), "last": self.pass_timings[-1], "updated_at": time.time()})
        return result

    def _register(self, definitions, network):
        used_path = self.budget.root / "assignments_used.json"
        used = read_json(used_path) if used_path.exists() else {}
        for item in definitions:
            policy = MulPolicy.from_dict(item["policy"])
            mapping = policy.expand([m.lora_name for m in network.unet_loras + network.text_encoder_loras])
            identity = digest(mapping)
            if item["id"] in self.expanded and self.expanded[item["id"]]["sha256"] != identity:
                raise ResearchStop("Candidate ID reused with a different assignment")
            self.policies[item["id"]] = policy
            self.expanded[item["id"]] = {"declaration": item, "sha256": identity, "modules": mapping}
            used.setdefault(identity, item)
        if len(used) > self.budget.limits["unique_local_candidate_assignments"]:
            raise ResearchStop("Unique candidate limit reached")
        write_json(used_path, used)
        write_json(self.run_root / "candidate_assignments.json", self.expanded)

    def _restore(self, snapshot, ctx):
        snapshot.restore(network=ctx["accelerator"].unwrap_model(ctx["network"]), optimizer=ctx["optimizer"], scheduler=ctx["lr_scheduler"], scaler=getattr(ctx["accelerator"], "scaler", None), trainer=self.trainer, guardian=ctx["grad_norm_guardian"])

    def _measure(self, label, selected, definitions, snapshot, ctx, *, dropout=False, noise_offset=0, pairs=(), pilot=False):
        stage = label[:2]
        self.budget.set_stage(stage)
        network = ctx["accelerator"].unwrap_model(ctx["network"])
        if len({item["id"] for item in definitions}) != len(definitions):
            raise ResearchStop("Duplicate candidate IDs")
        self._register(definitions, network)
        canonical = {}
        aliases = {}
        for item in definitions:
            name = item["id"]
            first = canonical.setdefault(self.expanded[name]["sha256"], name)
            if first != name:
                aliases[name] = first
        measurement_identity = digest({"execution_contract": self.budget.approval["execution_contract_sha256"], "state": fingerprint_tree(snapshot.network_state), "replay": [s.digest for s in selected], "candidates": [(d["id"], self.expanded[d["id"]]["sha256"]) for d in definitions], "dropout": dropout, "noise_offset": noise_offset, "seed": self.protocol_seed, "pairs": list(pairs), "metric": "2.4.0", "pilot": pilot, "probe_address_version": self.probe_address_version})
        for cached_path in sorted((self.budget.root / "runs").glob(f"measure_*/{label}/result.json")):
            cached_result = read_json(cached_path)
            if cached_result.get("status") != "complete" or cached_result.get("measurement_identity") != measurement_identity:
                continue
            if not all((cached_path.parent / n).is_file() and file_hash(cached_path.parent / n) == h for n, h in cached_result.get("artifact_hashes", {}).items()) or len(cached_result.get("artifact_hashes", {})) != 4:
                raise ResearchStop("Completed stage artifacts changed or are missing")
            cached_recipients = [json.loads(line) for line in (cached_path.parent / "recipient_energy.jsonl").read_text(encoding="utf-8").splitlines()]
            for line in (cached_path.parent / "per_image.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["candidate"] == "no_quant":
                    key = (tuple(s.image_keys[0] for s in selected), row["image_key"], row["timestep_bin"], row["noise_replica"], row["probe_regime"])
                    self.reference_hashes[key] = row["gradient_hash"]
            self.stage_reports.append({**cached_result, "reused_from": str(cached_path)})
            print(f"RESEARCH_STAGE_REUSED {label}", flush=True)
            return cached_result, cached_recipients
        required = len(selected) * 4 * (3 + len(canonical) * 4)
        if pilot:
            required += 4 * sum(d["id"].startswith("U_") for d in definitions) + 2 * len(definitions)
        warm_seconds = np.quantile([r["seconds"] for r in self.pass_timings[-80:]], .9) if self.pass_timings else 1.5
        reserved = 3600 if stage in {"P2", "P3", "P4"} else 0
        self.budget.check(reserve_seconds=float(required * warm_seconds * 1.25 + 300 + reserved), reserve_fb=required, storage=True)
        self._restore(snapshot, ctx)
        first_pass_index = len(self.pass_timings)
        start = time.perf_counter()
        trainable_before = digest([(n, p.requires_grad) for n, p in network.named_parameters()])
        rows, recipients, interactions, parity = [], [], [], []
        regime = "training_dropout_on" if dropout else "structural_dropout_off"
        source_map = SourceGroupMap.load(self.args.dq_profile_source_group_map)
        by_id = {item["id"]: item for item in definitions}
        for image_index, source in enumerate(selected):
            image_key = source.image_keys[0]
            for bin_index in range(4):
                for noise_index in range(3):
                    noise = noise_index + noise_offset
                    probe = self._fixed_timestep_item(source, probe_replica=noise, bin_index=bin_index, bin_count=4, noise_scheduler=ctx["noise_scheduler"])
                    probe_id, seed_id = self._probe_addresses(image_index, bin_index, noise, image_key)
                    metadata = {"image_key": image_key, "source_group": source_map.resolve(image_key), "timestep_bin": bin_index, "noise_replica": noise, "probe_regime": regime, "probe_address_version": self.probe_address_version, "model_seed_address": seed_id}

                    def run(name, quant_repeat=0):
                        seed_step_rng(self.protocol_seed, seed_id, phase="v2_tail_structural_model", repeat=0)
                        is_quant = name != "no_quant"
                        mul = by_id[name]["policy"]["base_mul"] if is_quant else None
                        definition = CandidateDefinition(name, is_quant, None, None, mul, False)
                        return self._run_pass(replay=probe, candidate=definition, range_mul=mul, phase="v2_tail_probe", probe_or_step=probe_id, repeat=quant_repeat, dropout_enabled=dropout, shadow=False, update=False, do_auto_observation=False, absolute_step=snapshot.metadata["global_step"], epoch=snapshot.metadata["epoch"], **ctx)

                    reference, reference_gradient, _ = run("no_quant")
                    reference.update(metadata, quant_repeat=None, parameter_gradient_cosine=1., gradient_topology_matches=True)
                    rows.append(reference)
                    ref_key = (tuple(x.image_keys[0] for x in selected), image_key, bin_index, noise, regime)
                    previous_hash = self.reference_hashes.setdefault(ref_key, reference["gradient_hash"])
                    if previous_hash != reference["gradient_hash"]:
                        raise ResearchStop("Repeated no-quant reference changed across stages")
                    if noise_index >= 2:
                        continue
                    cached = {}
                    # Reuse only inside this exact state/probe/noise/dropout
                    # cell. Candidate labels are reporting metadata, not keys.
                    assignment_cache = {}
                    shared_addresses = {}
                    for item in definitions:
                        name = item["id"]
                        for quant_repeat in range(2):
                            assignment_key = (self.expanded[name]["sha256"], quant_repeat)
                            if assignment_key in assignment_cache:
                                original, gradient = assignment_cache[assignment_key]
                                row = {**original, "candidate": name, "range_mul": item["policy"]["base_mul"], "seconds": 0., "forward_backward_performed": False, "reused_from_candidate": original["candidate"]}
                            else:
                                row, gradient, _ = run(name, quant_repeat)
                                row["forward_backward_performed"] = True
                                assignment_cache[assignment_key] = (dict(row), gradient)
                            comparison = reference_gradient.cosine(gradient)
                            if not comparison["topology_matches"]:
                                raise ResearchStop("Candidate changed gradient topology")
                            for field in ("replay_digest", "noise_digest", "timestep_digest", "rng_digest_before", "rng_digest_after", "dropout_mask_digest"):
                                if row[field] != reference[field]:
                                    raise ResearchStop(f"Paired input/RNG mismatch: {field}")
                            address = shared_addresses.setdefault(quant_repeat, row["shared_unet_quant_addresses_sha256"])
                            if address != row["shared_unet_quant_addresses_sha256"]:
                                raise ResearchStop("Shared UNet quantization RNG addresses changed")
                            row.update(metadata, quant_repeat=quant_repeat, parameter_gradient_cosine=comparison["cosine"], gradient_topology_matches=True)
                            rows.append(row)
                            energies = recipient_energy(reference_gradient, gradient)
                            recipient_row = {**metadata, "candidate": name, "quant_repeat": quant_repeat, "reference_norm_sq": reference_gradient.norm_sq, "recipient_energy": energies}
                            background = item.get("background_id") or ("U2.70_T3.75" if name.startswith("low_restore_") else "UT_3.75" if name.startswith("high_lower_") else None)
                            if background is not None and (background, quant_repeat) in cached:
                                recipient_row["intervention_background"] = background
                                recipient_row["intervention_recipient_energy"] = recipient_energy(cached[background, quant_repeat], gradient, denominator=reference_gradient.norm_sq)
                                if self.collect_module_energy:
                                    recipient_row["intervention_module_energy"] = recipient_energy(cached[background, quant_repeat], gradient, denominator=reference_gradient.norm_sq, per_module=True)
                            recipients.append(recipient_row)
                            cached[name, quant_repeat] = gradient
                            if pilot and image_index == 0 and bin_index == 0 and noise_index == 0 and name.startswith("U_"):
                                self._legacy_parity = True
                                try:
                                    legacy, legacy_gradient, _ = run(name, quant_repeat)
                                finally:
                                    self._legacy_parity = False
                                if legacy["gradient_hash"] != row["gradient_hash"] or legacy["loss"] != row["loss"]:
                                    raise ResearchStop("Legacy UNet-only Local parity failed")
                                reverse, _, _ = run(name, quant_repeat)
                                if reverse["gradient_hash"] != row["gradient_hash"]:
                                    raise ResearchStop("Candidate repeat/order parity failed")
                                parity.append({"candidate": name, "repeat": quant_repeat, "legacy_gradient_hash": legacy["gradient_hash"], "exact": True})
                    if pilot and image_index == 0 and bin_index == 0 and noise_index == 0:
                        for item in reversed(definitions):
                            for quant_repeat in range(2):
                                repeated, repeated_gradient, _ = run(item["id"], quant_repeat)
                                if repeated["gradient_hash"] != next(r["gradient_hash"] for r in reversed(rows) if r["candidate"] == item["id"] and r.get("quant_repeat") == quant_repeat):
                                    raise ResearchStop("Reversed candidate order changed the result")
                    for pair in pairs:
                        for quant_repeat in range(2):
                            gradients = [cached[name, quant_repeat] for name in (pair["ab"], pair["a"], pair["b"], pair["base"])]
                            energy = 0.
                            for parameter in gradients[0].values:
                                residual = gradients[0].values[parameter] - gradients[1].values[parameter] - gradients[2].values[parameter] + gradients[3].values[parameter]
                                energy += float(torch.sum(residual * residual))
                            interactions.append({**metadata, "pair": pair, "quant_repeat": quant_repeat, "interaction_norm_over_noquant": math.sqrt(energy / max(reference_gradient.norm_sq, 1e-30))})
                    # Gradients are ephemeral and never saved as a tensor bank.
                    del cached, assignment_cache, reference_gradient
        measurement_seconds = time.perf_counter() - start
        if trainable_before != digest([(n, p.requires_grad) for n, p in network.named_parameters()]):
            raise ResearchStop("Trainable set changed")
        if fingerprint_tree(network.state_dict()) != fingerprint_tree(snapshot.network_state):
            raise ResearchStop("Local measurement changed model weights")
        self._restore(snapshot, ctx)
        if fingerprint_tree(ctx["optimizer"].state_dict()) != fingerprint_tree(snapshot.optimizer_state):
            raise ResearchStop("Optimizer state did not restore")
        samples = gradient_tail_rows(rows)
        for row in samples:
            row["probe_regime"] = regime
        analysis_start = time.perf_counter()
        summary = summarize_samples(samples)
        stage_root = self.run_root / label
        stage_root.mkdir(exist_ok=True)
        for name, data in (("per_image.jsonl", rows), ("gradient_tail.jsonl", samples), ("recipient_energy.jsonl", recipients), ("interactions.jsonl", interactions)):
            with (stage_root / name).open("w", encoding="utf-8") as stream:
                for row in data:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        executed_passes = len(self.pass_timings) - first_pass_index
        if executed_passes != required:
            raise ResearchStop("Actual pass count differs from the reserved measurement plan")
        result = {"protocol": "dq-mul-research-v1", "status": "complete", "measurement_identity": measurement_identity, "artifact_hashes": {n: file_hash(stage_root / n) for n in ("per_image.jsonl", "gradient_tail.jsonl", "recipient_energy.jsonl", "interactions.jsonl")}, "stage": label, "pilot": pilot, "regime": regime, "state_sha256": fingerprint_tree(snapshot.network_state), "probe_keys_sha256": digest([{k: row[k] for k in ("image_key", "timestep_bin", "noise_replica", "probe_regime", "replay_digest")} for row in rows if row["candidate"] == "no_quant"]), "candidate_definitions": definitions, "candidate_aliases": aliases, "unique_candidate_assignments": len(canonical), "summary": summary, "parity": parity, "forward_backward_calls": executed_passes, "measurement_seconds": measurement_seconds, "analysis_save_seconds": time.perf_counter() - analysis_start, "not_image_quality": True, "independent_images": len(selected), "bootstrap_definition": "existing v2.4 source bootstrap, 2000 draws; repeats are not independent images"}
        write_json(stage_root / "result.json", result)
        self.stage_reports.append(result)
        write_json(self.run_root / "timing.json", self.pass_timings)
        write_json(self.run_root / "progress.json", {"stage": label, "status": "complete", "completed_passes": len(self.pass_timings), "updated_at": time.time()})
        print(f"RESEARCH_STAGE_COMPLETE {label} passes={executed_passes} measured_seconds={measurement_seconds:.1f}", flush=True)
        return result, recipients

    def _prepare_research_boundary(self, **kw):
        if kw["global_step"] != kw["dq_delta_begin_step"] or kw["accelerator"].num_processes != 1:
            raise ResearchStop("Invalid shared warmup boundary")
        network = kw["accelerator"].unwrap_model(kw["network"])
        if type(network).__module__ != "dq_profile.copied_lora":
            raise ResearchStop("Research must use isolated diagnostic LoRA")
        snapshot = TrainingSnapshot.capture(network=network, optimizer=kw["optimizer"], scheduler=kw["lr_scheduler"], scaler=getattr(kw["accelerator"], "scaler", None), trainer=self.trainer, guardian=kw["grad_norm_guardian"], global_step=kw["global_step"], epoch=kw["epoch"], data_step=kw["data_step"])
        components = {"network": snapshot.network_state, "optimizer": snapshot.optimizer_state, "scheduler": snapshot.scheduler_state, "scaler": snapshot.scaler_state, "rng": snapshot.rng_state, "network_runtime": snapshot.network_runtime, "trainer": snapshot.trainer_state, "guardian": snapshot.guardian_state, "metadata": snapshot.metadata}
        fingerprints = {k: fingerprint_tree(v) for k, v in components.items()}
        prefix = read_json(Path(self.args.dq_profile_prefix_gate_file).parent / "summary.json")
        expected = prefix["snapshot"]["fingerprints"]
        differences = {k: {"prefix": expected.get(k), "research": v} for k, v in fingerprints.items() if expected.get(k) != v}
        write_json(self.run_root / "snapshot_parity.json", {"passed": not differences, "differences": differences, "fingerprints": fingerprints})
        if differences:
            raise ResearchStop("Research warmup differs from the validated prefix snapshot")
        torch.save(snapshot, self.run_root / "research_snapshot.pt")
        source_map = SourceGroupMap.load(self.args.dq_profile_source_group_map)
        sequence, capture = self._capture_batches(first_batch=kw["first_batch"], epoch_iterator=kw["epoch_iterator"], train_dataloader=kw["train_dataloader"], current_epoch=kw["current_epoch"], current_step=kw["current_step"], global_step=kw["global_step"], epoch=kw["epoch"], data_step=kw["data_step"], count=128, source_group_resolver=source_map.resolve, required_source_groups=self._source_group_order(source_map))
        ctx_names = ("accelerator", "network", "optimizer", "lr_scheduler", "grad_norm_guardian", "unet", "text_encoders", "tokenizers", "train_unet", "train_text_encoder", "training_model", "on_step_start", "weight_dtype", "noise_scheduler")
        ctx = {key: kw[key] for key in ctx_names}
        self._restore(snapshot, ctx)
        self._materialize_replay(sequence, accelerator=kw["accelerator"], vae=kw["vae"], vae_dtype=kw["vae_dtype"], weight_dtype=kw["weight_dtype"], noise_scheduler=kw["noise_scheduler"])
        selected, _ = self._select_probe_items(sequence)
        # The actual dataset count, never duplicated slots, controls coverage.
        if len(selected) != sum(len(s["image_files"]) for s in self.budget.manifest["sources"]):
            raise ResearchStop("Research requires full approved image coverage")
        inventory = []
        for module in network.unet_loras + network.text_encoder_loras:
            inventory.append({"name": module.lora_name, "group": module_group(module.lora_name), "rank": int(module.lora_down.weight.shape[0]), "down_shape": list(module.lora_down.weight.shape), "up_shape": list(module.lora_up.weight.shape), "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)})
        write_json(self.run_root / "module_inventory.json", {"modules": inventory, "counts": dict(Counter(x["group"] for x in inventory)), "trainable_names": [n for n, p in network.named_parameters() if p.requires_grad]})
        write_json(self.run_root / "probe_manifest.json", {"capture": capture, "selected": [{"image_key": s.image_keys[0], "source_group": source_map.resolve(s.image_keys[0]), "replay_digest": s.digest, "latent_shape": list(s.latents.shape)} for s in selected], "fingerprints": fingerprints})
        by_source = defaultdict(list)
        for item in selected:
            by_source[source_map.resolve(item.image_keys[0])].append(item)
        pilot_items = [items[len(items) // 2] for _, items in sorted(by_source.items())]
        remaining = [s for s in selected if all(s is not chosen for chosen in pilot_items)]
        if remaining:
            pilot_items.append(max(remaining, key=lambda s: int(s.latents.numel())))
        pilot_items = pilot_items[:self.budget.proposal["pilot"]["max_unique_images"]]
        if {source_map.resolve(s.image_keys[0]) for s in pilot_items} != set(by_source):
            raise ResearchStop("Pilot image cap cannot cover every source")
        return snapshot, ctx, selected, pilot_items

    def run_from_boundary(self, **kw):
        snapshot, ctx, selected, pilot_items = self._prepare_research_boundary(**kw)
        scope, cross, interventions = initial_candidates()
        self._measure("P1_pilot", pilot_items, [scope[-2], scope[-1]], snapshot, ctx, pilot=True)
        self._measure("P2_scope", selected, scope, snapshot, ctx)
        self._measure("P3_cross", selected, cross, snapshot, ctx)
        low_base, high_base = cross[0], scope[-1]
        p4, recipients = self._measure("P4_groups", selected, [low_base, high_base, *interventions], snapshot, ctx)
        vectors = defaultdict(lambda: defaultdict(float))
        for row in recipients:
            if row["candidate"].startswith("low_restore_"):
                for group, energy in row["intervention_recipient_energy"].items():
                    vectors[row["candidate"]][group] += energy["contribution"]
        def overlap(pair):
            a, b = (vectors[f"low_restore_{g}"] for g in pair)
            keys = set(a) | set(b)
            denominator = math.sqrt(sum(a[k] ** 2 for k in keys) * sum(b[k] ** 2 for k in keys))
            return sum(a[k] * b[k] for k in keys) / max(denominator, 1e-30)
        pairs = sorted(itertools.combinations(UNET_GROUPS, 2))
        first = min(pairs, key=lambda p: (-overlap(p), p))
        remaining_groups = sorted(set(UNET_GROUPS) - set(first))
        second = tuple(remaining_groups)
        pair_defs, pair_cells = [], []
        for index, groups in enumerate((first, second)):
            low = index == 0
            name = f"pair_{'low' if low else 'high'}_{index + 1}"
            base = low_base if low else high_base
            prefix = "low_restore_" if low else "high_lower_"
            new = candidate(name, 2.70 if low else 3.75, 3.75, overrides={g: 3.75 if low else 2.70 for g in groups})
            members = [next(d for d in interventions if d["id"] == prefix + g) for g in groups]
            pair_defs.extend([base, *members, new])
            pair_cells.append({"ab": name, "a": members[0]["id"], "b": members[1]["id"], "base": base["id"]})
        write_json(self.run_root / "pair_selection.json", {"rule": "recipient-energy overlap for low background; complementary unused groups for high background", "first_groups": first, "overlap": overlap(first), "second_groups": second, "cells": pair_cells, "not_causal_attribution": True})
        unique_pairs = list({item["id"]: item for item in pair_defs}.values())
        self._measure("P5_pairs", selected, unique_pairs, snapshot, ctx, pairs=pair_cells)
        scores = {r["candidate"]: r for r in p4["summary"]}
        low_choice = min((d for d in interventions if d["id"].startswith("low_restore_")), key=lambda d: (scores[d["id"]]["tail"], scores[d["id"]]["body"], d["id"]))
        high_choice = max((d for d in interventions if d["id"].startswith("high_lower_")), key=lambda d: (scores[d["id"]]["body"], d["id"]))
        chosen = [low_base, high_base, low_choice, high_choice]
        write_json(self.run_root / "confirmation_selection.json", {"rule": "retain one low-background tail-mitigating intervention and one high-background intervention with stronger deformation; no image-quality claim", "candidates": chosen, "new_noise_replicas": [3, 4, 5]})
        self._measure("P5_confirmation", selected, chosen, snapshot, ctx, noise_offset=3)
        self._measure("P5_dropout_confirmation", pilot_items, chosen, snapshot, ctx, dropout=True, noise_offset=3)
        final = {"protocol": "dq-mul-research-v1", "status": "complete", "P6_executed": False, "stages": self.stage_reports, "provisional_candidates": [low_choice, high_choice], "limitations": [f"{len(selected)}-image single-dataset local measurement", "same images with independent noise; no unseen-image claim", "dropout-on confirmation has pilot image coverage", "pair tests do not identify all interactions", "no training-quality or preference evaluation"]}
        write_json(self.run_root / "research_summary.json", final)
        write_json(self.run_root / "timing.json", self.pass_timings)
        self._restore(snapshot, ctx)
        return final
