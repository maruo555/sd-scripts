"""Preregistered hierarchical fixed-mul continuation, with paired RNG addresses."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from library.dq_mul_policy import MulPolicy, digest, module_groups
from dq_profile.metrics import ExactGradient
from dq_profile.mul_research import MulResearchRuntime, candidate
from dq_profile.protocol import CandidateDefinition
from dq_profile.replay import seed_step_rng
from dq_profile.research_budget import ResearchStop, read_json, write_json
from dq_profile.v2_calibration import fingerprint_tree
from dq_profile.v24_acceptance import _body_tail, _cluster_quantile, _index_rows


GRID = (2.70, 3.15, 3.45, 3.75, 4.05)
BACKGROUND = "component_C"


def intervention(name, *, groups=None, modules=None, mul=3.75):
    result = candidate(name, 2.70, 3.75, overrides={group: mul for group in (groups or ())})
    if modules:
        result["policy"]["module_overrides"] = {module: mul for module in modules}
    result["background_id"] = BACKGROUND
    return result


def background():
    return candidate(BACKGROUND, 2.70, 3.75)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_evidence(samples, name, control=BACKGROUND, *, bootstrap=2000):
    """Cluster resampling uses the same source draw on both sides of a contrast."""
    left = [row for row in samples if row["candidate"] == control]
    right = [row for row in samples if row["candidate"] == name]
    cell = lambda row: tuple(row[field] for field in ("image_key", "timestep_bin", "noise_replica", "quant_repeat"))
    if not left or {cell(row) for row in left} != {cell(row) for row in right}:
        raise ResearchStop("Unmatched cells in candidate selection")
    if len(left) != len(right) or len({cell(row) for row in left}) != len(left):
        raise ResearchStop("Repeated aliases cannot be treated as independent evidence")
    indexes = [_index_rows(rows, timestep_bins=4) for rows in (left, right)]
    sources = sorted(indexes[0][0])
    if sources != sorted(indexes[1][0]):
        raise ResearchStop("Unmatched sources in candidate selection")

    def scores(selected):
        results = []
        for all_groups, bins in indexes:
            tail = _body_tail(all_groups, bins, selected, field="relative_gradient_distance", timestep_bins=4)
            results.append(np.asarray([_cluster_quantile(all_groups, selected, "relative_gradient_distance", .5), tail["body"], tail["tail"]]))
        return results[1] - results[0]

    point = scores(sources)
    loo = {source: scores([s for s in sources if s != source]).tolist() for source in sources} if len(sources) > 1 else {}
    draws = []
    rng = np.random.default_rng(2401)
    for selected in rng.integers(0, len(sources), size=(bootstrap, len(sources))):
        draws.append(scores([sources[i] for i in selected]))
    ci = np.quantile(np.asarray(draws), [.025, .975], axis=0).T.tolist()
    return {"candidate": name, "control": control, "delta_p50": float(point[0]), "delta_body": float(point[1]), "delta_tail": float(point[2]),
            "paired_source_ci": dict(zip(("p50", "body", "tail"), ci)), "source_leave_one_out": loo,
            "negative_p50_leave_one_out": sum(value[0] < 0 for value in loo.values()), "sources": len(sources),
            "independent_images": len({row["image_key"] for row in left}), "bootstrap_draws": bootstrap,
            "not_image_quality": True}


def rank_evidence(items):
    # Screen on typical distortion and source consistency; Tail is reported,
    # never silently minimized as if it were an image-quality metric.
    return sorted(items, key=lambda item: (-item["negative_p50_leave_one_out"], item["delta_p50"], item["candidate"]))


def select_final_candidate(definitions, evidence_off, evidence_on, assignments, excluded):
    excluded_hashes = {assignments[name]["sha256"] for name in excluded}
    eligible, reasons = [], {}
    for definition in definitions:
        name = definition["id"]
        off, on = evidence_off[name], evidence_on[name]
        failures = []
        if assignments[name]["sha256"] in excluded_hashes:
            failures.append("duplicate_of_predeclared_training_control")
        if off["delta_p50"] >= 0 or off["paired_source_ci"]["p50"][1] >= 0:
            failures.append("independent_noise_p50_improvement_not_supported")
        if off["negative_p50_leave_one_out"] != off["sources"]:
            failures.append("independent_noise_source_sensitivity")
        if on["delta_p50"] >= 0 or on["negative_p50_leave_one_out"] < on["sources"] - 1:
            failures.append("dropout_p50_improvement_not_stable")
        reasons[name] = failures
        if not failures:
            changed_modules = sum(row["component"] == "unet" and row["mul"] != 2.70 for row in assignments[name]["modules"].values())
            eligible.append((definition, off["delta_p50"], changed_modules))
    if not eligible:
        return None, reasons
    best_delta = min(item[1] for item in eligible)
    # Within 10% of the strongest typical improvement, prefer less intervention.
    near = [item for item in eligible if item[1] <= best_delta * .9]
    chosen = min(near, key=lambda item: (item[2], item[1], item[0]["id"]))[0]
    return chosen, reasons


class HierarchicalMulRuntime(MulResearchRuntime):
    probe_address_version = "stable-image-v1"

    def _stage_evidence(self, label, definitions):
        rows = jsonl(self.run_root / label / "gradient_tail.jsonl")
        evidence = {item["id"]: paired_evidence(rows, item["id"]) for item in definitions if item["id"] != BACKGROUND}
        write_json(self.run_root / label / "paired_evidence.json", evidence)
        return evidence

    def _one_pass(self, source, definition, snapshot, ctx, *, bin_index, noise, quant_repeat=0,
                  dropout=True, update=False, addresses=None):
        self._restore(snapshot, ctx)
        probe = self._fixed_timestep_item(source, probe_replica=noise, bin_index=bin_index, bin_count=4, noise_scheduler=ctx["noise_scheduler"])
        probe_id, seed_id = addresses or self._probe_addresses(0, bin_index, noise, source.image_keys[0])
        seed_step_rng(self.protocol_seed, seed_id, phase="v2_tail_structural_model", repeat=0)
        quantized = definition is not None
        name = definition["id"] if quantized else "no_quant"
        mul = definition["policy"]["base_mul"] if quantized else None
        row, gradient, _ = self._run_pass(replay=probe, candidate=CandidateDefinition(name, quantized, None, None, mul, False),
            range_mul=mul, phase="v2_tail_probe", probe_or_step=probe_id, repeat=quant_repeat,
            dropout_enabled=dropout, shadow=False, update=update, do_auto_observation=False,
            absolute_step=snapshot.metadata["global_step"], epoch=snapshot.metadata["epoch"], hard_safety=update, **ctx)
        row.update(image_key=source.image_keys[0], timestep_bin=bin_index, noise_replica=noise, quant_repeat=quant_repeat,
                   model_seed_address=seed_id, probe_regime="training_dropout_on" if dropout else "structural_dropout_off")
        if row["forced_safety_abort"]:
            write_json(self.run_root / "failed_actual_update.json", row)
            raise ResearchStop("Actual update failed a hard safety check: " + str(row["invalid_reason"]))
        return row, gradient

    def _audit_updates(self, label, selected, definitions, snapshot, ctx, *, noise=3, bins=(3,), regimes=(True,),
                       addresses=None, quant_repeats=2, control_id=None, mask_variant=None):
        self._register(definitions, ctx["accelerator"].unwrap_model(ctx["network"]))
        control_id = control_id or definitions[0]["id"]
        if definitions[0]["id"] != control_id:
            raise ResearchStop("The update control must be evaluated first")
        count = len(selected) * len(bins) * len(regimes) * (1 + quant_repeats * len(definitions))
        used = getattr(self, "actual_update_calls", 0)
        if used + count > 400:
            raise ResearchStop("Actual-update inspection exceeds the 400-pass sublimit")
        self.budget.check(reserve_fb=count, reserve_seconds=count * 3 + 180, storage=True)
        network = ctx["accelerator"].unwrap_model(ctx["network"])
        rows = []
        for source in selected:
            for bin_index in bins:
                for dropout in regimes:
                    reference = None
                    control_updates = {}
                    for definition in (None, *definitions):
                        for repeat in range(1 if definition is None else quant_repeats):
                            row, gradient = self._one_pass(source, definition, snapshot, ctx, bin_index=bin_index, noise=noise,
                                                          quant_repeat=repeat, dropout=dropout, update=True, addresses=addresses)
                            values = {name: parameter.detach().float().cpu() - snapshot.network_state[name].float()
                                      for name, parameter in network.named_parameters() if parameter.requires_grad}
                            delta = ExactGradient(values, sum(float(torch.sum(value * value)) for value in values.values()))
                            if not math.isfinite(delta.norm):
                                raise ResearchStop("Nonfinite actual parameter update")
                            if definition is None:
                                reference = delta
                            comparison = reference.cosine(delta)
                            # A native skip produces the zero vector; its angle is undefined.
                            if not math.isfinite(comparison["cosine"]):
                                comparison["cosine"] = None
                            row.update(actual_parameter_update_norm=delta.norm, update_vs_no_quant=comparison,
                                       actual_gradient_norm=gradient.norm, dropout_mask_variant=mask_variant,
                                       update_control_candidate=control_id)
                            if definition is not None:
                                if definition["id"] == control_id:
                                    control_updates[repeat] = delta
                                paired = control_updates[repeat].cosine(delta)
                                if not math.isfinite(paired["cosine"]):
                                    paired["cosine"] = None
                                row["update_vs_control"] = paired
                                row["update_control_quant_repeat"] = repeat
                            rows.append(row)
                            self.actual_update_calls = getattr(self, "actual_update_calls", 0) + 1
                            write_json(self.run_root / (label + ".json"), {"status": "running", "expected_calls": count, "rows": rows})
        self._restore(snapshot, ctx)
        write_json(self.run_root / (label + ".json"), {"status": "complete", "calls": count, "cumulative_update_inspections": self.actual_update_calls,
            "same_complete_state_restored_before_every_pass": True, "rows": rows,
            "scope": "one-step probes at the diagnostic boundary; not a full-training quality prediction"})
        return rows

    def _replay_old_outlier(self, selected, snapshot, ctx):
        prior = Path(self.budget.proposal["prior_evidence_root"]) / "runs/measure_01"
        old_fingerprints = read_json(prior / "snapshot_parity.json")["fingerprints"]
        current = read_json(self.run_root / "snapshot_parity.json")["fingerprints"]
        if old_fingerprints != current:
            raise ResearchStop("Prior outlier cannot be replayed: complete boundary state changed")
        samples = jsonl(prior / "P5_dropout_confirmation/gradient_tail.jsonl")
        worst = max((row for row in samples if row["candidate"] == "low_restore_unet.attn2"), key=lambda row: row["relative_gradient_distance"])
        original_rows = jsonl(prior / "P5_dropout_confirmation/per_image.jsonl")
        original_rows = [row for row in original_rows if all(row[key] == worst[key] for key in ("image_key", "timestep_bin", "noise_replica"))]
        definitions = read_json(prior / "P5_dropout_confirmation/result.json")["candidate_definitions"]
        by_id = {item["id"]: item for item in definitions}
        self._register(definitions, ctx["accelerator"].unwrap_model(ctx["network"]))
        source = next(item for item in selected if item.image_keys[0] == worst["image_key"])
        results = []
        for original in original_rows:
            image_index = int(original["probe_or_step"].split(":", 2)[1])
            seed_id = f"image:{image_index}|bin:{worst['timestep_bin']}|noise:{worst['noise_replica']}|key:{worst['image_key']}"
            row, _ = self._one_pass(source, by_id.get(original["candidate"]), snapshot, ctx,
                bin_index=worst["timestep_bin"], noise=worst["noise_replica"], quant_repeat=original.get("quant_repeat") or 0,
                addresses=(original["probe_or_step"], seed_id))
            fields = ("gradient_hash", "loss", "replay_digest", "noise_digest", "timestep_digest", "rng_digest_before", "rng_digest_after", "dropout_mask_digest", "quant_rng_digest")
            differences = {field: {"old": original[field], "new": row[field]} for field in fields if original[field] != row[field]}
            results.append({"candidate": original["candidate"], "repeat": row["repeat"], "exact": not differences, "differences": differences})
            write_json(self.run_root / "P1_original_outlier_replay.json", {"results": results, "complete": len(results) == len(original_rows)})
            if differences:
                raise ResearchStop("Original outlier replay differs; inspect recorded evidence")
        self._restore(snapshot, ctx)
        # Carry the exact legacy model and quantization addresses into actual
        # updates. Stable-image probes alone do not inspect this old outlier.
        legacy_definitions = [by_id[name] for name in ("U2.70_T3.75", "low_restore_unet.attn2", "UT_3.75")]
        probe_id = original_rows[0]["probe_or_step"]
        comparisons = []
        regime_rows = {}
        for label, dropout, model_seed in (("off", False, seed_id), ("original_on", True, seed_id),
                                           ("additional_on", True, seed_id + "|additional_dropout_mask:1")):
            update_rows = self._audit_updates("P1_legacy_updates_" + label, [source], legacy_definitions, snapshot, ctx,
                noise=worst["noise_replica"], bins=(worst["timestep_bin"],), regimes=(dropout,),
                addresses=(probe_id, model_seed), quant_repeats=4, mask_variant=label)
            regime_rows[label] = {(row["candidate"], row["quant_repeat"]): row for row in update_rows}
            if label == "original_on":
                old_by_cell = {(row["candidate"], row.get("quant_repeat") or 0): row for row in original_rows}
                for row in update_rows:
                    original = old_by_cell.get((row["candidate"], row["quant_repeat"]))
                    if original is None:
                        continue
                    fields = ("gradient_hash", "loss", "replay_digest", "noise_digest", "timestep_digest", "rng_digest_before", "dropout_mask_digest", "quant_rng_digest")
                    differences = {key: {"old": original[key], "update": row[key]} for key in fields if original[key] != row[key]}
                    comparisons.append({"candidate": row["candidate"], "repeat": row["quant_repeat"], "exact_before_update": not differences, "differences": differences})
                write_json(self.run_root / "P1_legacy_update_replay_parity.json", {"status": "pass" if not any(row["differences"] for row in comparisons) else "fail", "comparisons": comparisons})
                if len(comparisons) != 7 or any(row["differences"] for row in comparisons):
                    raise ResearchStop("Original outlier actual-update pass did not reproduce its pre-update gradient")
        for key, off in regime_rows["off"].items():
            original_on, extra_on = (regime_rows[label][key] for label in ("original_on", "additional_on"))
            for field in ("replay_digest", "noise_digest", "timestep_digest", "quant_rng_digest"):
                if off[field] != original_on[field] or off[field] != extra_on[field]:
                    raise ResearchStop("Legacy update dropout controls changed: " + field)
            if off["rng_digest_before"] != original_on["rng_digest_before"]:
                raise ResearchStop("Legacy dropout OFF/ON initial model RNG differs")
        if not any(regime_rows["original_on"][key]["dropout_mask_digest"] != row["dropout_mask_digest"] for key, row in regime_rows["additional_on"].items()):
            raise ResearchStop("The additional dropout mask did not change")
        write_json(self.run_root / "P1_legacy_dropout_control_audit.json", {"status": "pass", "matched_cells": len(regime_rows["off"]), "regimes": list(regime_rows), "quantization_addresses_fixed_across_masks": True})
        return source

    def _matched_dropout_check(self, off_label, on_label):
        fields = ("candidate", "image_key", "timestep_bin", "noise_replica", "quant_repeat")
        off = {tuple(row[field] for field in fields): row for row in jsonl(self.run_root / off_label / "per_image.jsonl")}
        on = {tuple(row[field] for field in fields): row for row in jsonl(self.run_root / on_label / "per_image.jsonl")}
        if set(off) != set(on):
            raise ResearchStop("Dropout comparison coverage changed")
        for key, left in off.items():
            right = on[key]
            for field in ("replay_digest", "noise_digest", "timestep_digest", "rng_digest_before", "quant_rng_digest", "probe_or_step", "model_seed_address"):
                if left[field] != right[field]:
                    raise ResearchStop("Dropout comparison changed a control: " + field)
        write_json(self.run_root / "P5_dropout_control_audit.json", {"status": "pass", "matched_rows": len(off), "quant_rng_independent_of_dropout": True})

    def run_from_boundary(self, **kw):
        snapshot, ctx, selected, pilot = self._prepare_research_boundary(**kw)
        self.budget.set_stage("P1")
        c, a, h = background(), intervention("attn2_A", groups=["unet.attn2"]), candidate("uniform_375", 3.75)
        outlier_source = self._replay_old_outlier(selected, snapshot, ctx)
        self._audit_updates("P1_actual_updates", [outlier_source], [c, a, h], snapshot, ctx, bins=(0, 3), regimes=(False, True))
        self._measure("P1_stable_pilot", pilot, [candidate("U_3.75", 3.75, te_quantized=False), h], snapshot, ctx, pilot=True)

        uniform = [candidate(f"uniform_{int(round(mul * 100))}", mul) for mul in GRID]
        self._measure("P2_uniform", selected, uniform, snapshot, ctx)
        self._audit_updates("P2_uniform_actual_updates", [outlier_source], [uniform[0], uniform[1], h], snapshot, ctx, bins=(0, 3), regimes=(False, True))
        attn_curve = [c] + [intervention("attn2_A" if mul == 3.75 else f"attn2_{int(round(mul * 100))}", groups=["unet.attn2"], mul=mul) for mul in GRID if mul != 2.70]
        self._measure("P2_attn2_curve", selected, attn_curve, snapshot, ctx)
        self._stage_evidence("P2_attn2_curve", attn_curve)

        roles = [intervention("role_" + role, groups=["unet.attn2." + role]) for role in ("q", "k", "v", "out")]
        self._measure("P3_roles", selected, [c, *roles], snapshot, ctx)
        role_evidence = self._stage_evidence("P3_roles", roles)
        best_role = rank_evidence(list(role_evidence.values()))[0]["candidate"].removeprefix("role_")
        regions = [intervention("region_" + best_role + "_" + region, groups=[f"unet.attn2.{best_role}.{region}"]) for region in ("input", "middle", "output")]
        self.collect_module_energy = True
        _, recipient_rows = self._measure("P3_regions", selected, [c, *regions], snapshot, ctx)
        self.collect_module_energy = False
        region_evidence = self._stage_evidence("P3_regions", regions)
        best_region_id = rank_evidence(list(region_evidence.values()))[0]["candidate"]
        best_region = next(item for item in regions if item["id"] == best_region_id)
        selector = best_region["policy"]["group_overrides"][0]["group_id"]
        # Recipient response screens where to intervene next, not a causal
        # attribution. All selected exact-module interventions are measured.
        energies = defaultdict(lambda: defaultdict(list))
        for row in recipient_rows:
            if row["candidate"] == best_region_id:
                for name, values in row.get("intervention_module_energy", {}).items():
                    if selector in module_groups(name):
                        energies[name][row["source_group"]].append(values["contribution"])
        scores = {name: float(np.mean([np.mean(values) for values in sources.values()])) for name, sources in energies.items()}
        module_names = sorted(scores, key=lambda name: (-scores[name], name))[:4]
        if not module_names:
            raise ResearchStop("No exact modules found in selected region")
        modules = [intervention(f"module_{i+1:02d}", modules=[name]) for i, name in enumerate(module_names)]
        write_json(self.run_root / "P3_hierarchy_selection.json", {"role": best_role, "region": best_region_id, "module_screen_rule": "source-balanced mean intervention recipient energy within selected role and region", "module_scores": scores, "modules": module_names, "not_causal_attribution": True})
        self._measure("P4_modules", selected, [c, *modules], snapshot, ctx)
        module_evidence = self._stage_evidence("P4_modules", modules)

        fine_definitions = roles + regions + modules
        fine_evidence = {**role_evidence, **region_evidence, **module_evidence}
        best_fine_id = rank_evidence(list(fine_evidence.values()))[0]["candidate"]
        best_fine = next(item for item in fine_definitions if item["id"] == best_fine_id)
        fine_curve = [c]
        for mul in GRID:
            if mul == 2.70:
                continue
            item = deepcopy(best_fine)
            item["id"] = best_fine_id if mul == 3.75 else best_fine_id + f"_m{int(round(mul * 100))}"
            for group in item["policy"]["group_overrides"]:
                group["range_mul"] = mul
            for name in item["policy"].get("module_overrides", {}):
                item["policy"]["module_overrides"][name] = mul
            fine_curve.append(item)
        self._measure("P4_fine_curve", selected, fine_curve, snapshot, ctx)
        curve_evidence = self._stage_evidence("P4_fine_curve", fine_curve)
        curve_winner = next(item for item in fine_curve if item["id"] == rank_evidence(list(curve_evidence.values()))[0]["candidate"])

        ff = intervention("ff_single", groups=["unet.ff"])
        pair = intervention("attn2_ff_pair", groups=["unet.attn2", "unet.ff"])
        pair_defs = [c, a, ff, pair]
        pair_cells = [{"base": c["id"], "a": a["id"], "b": ff["id"], "ab": pair["id"]}]
        # One exact-module pair is informative only if both singles improved.
        promising = [row for row in rank_evidence(list(module_evidence.values())) if row["delta_p50"] < 0 and row["negative_p50_leave_one_out"] >= row["sources"] - 1]
        exact_pair = None
        if len(promising) >= 2:
            members = [next(item for item in modules if item["id"] == row["candidate"]) for row in promising[:2]]
            exact_names = [next(iter(item["policy"]["module_overrides"])) for item in members]
            exact_pair = intervention("exact_module_pair", modules=exact_names)
            pair_defs += [*members, exact_pair]
            pair_cells.append({"base": c["id"], "a": members[0]["id"], "b": members[1]["id"], "ab": exact_pair["id"]})
        self._measure("P4_pairs", selected, pair_defs, snapshot, ctx, pairs=pair_cells)
        self._stage_evidence("P4_pairs", pair_defs)

        finalists = [curve_winner, pair] + ([exact_pair] if exact_pair else [])
        confirm = [c, a, h, *finalists]
        write_json(self.run_root / "P5_preregistered_finalists.json", {"definitions": confirm, "rule": "best fine curve plus prespecified coarse pair and optional supported exact pair; independent noise next", "noise_replicas": [3, 4, 5]})
        self._measure("P5_independent_noise", selected, confirm, snapshot, ctx, noise_offset=3)
        off_evidence = self._stage_evidence("P5_independent_noise", confirm)
        self._measure("P5_matched_dropout_off", pilot, confirm, snapshot, ctx, noise_offset=3)
        self._measure("P5_matched_dropout_on", pilot, confirm, snapshot, ctx, noise_offset=3, dropout=True)
        self._matched_dropout_check("P5_matched_dropout_off", "P5_matched_dropout_on")
        on_evidence = self._stage_evidence("P5_matched_dropout_on", confirm)
        self._audit_updates("P5_actual_updates", pilot, confirm, snapshot, ctx, regimes=(False, True))
        winner, reasons = select_final_candidate(finalists, off_evidence, on_evidence, self.expanded, {BACKGROUND, a["id"], h["id"]})
        full = [uniform[0], uniform[1], uniform[3], c, a] + ([winner] if winner else [])
        final = {"protocol": "dq-mul-hierarchical-v1", "status": "complete", "stages": self.stage_reports,
            "full_training_candidates": full, "selection_reasons": reasons, "selected_F": winner,
            "candidate_assignment_hashes": {item["id"]: self.expanded[item["id"]]["sha256"] for item in full},
            "full_training_steps_per_condition": 13600, "P6_executed": False, "not_image_quality": True,
            "limitations": ["single 17-image dataset; no unseen-image validation", "8-image source-balanced dropout confirmation", "one-step update inspection does not prove long-run stability", "final visual evaluation is performed by the user"]}
        write_json(self.run_root / "research_summary.json", final)
        self._restore(snapshot, ctx)
        return final
