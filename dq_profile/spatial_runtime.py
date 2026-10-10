"""Standard spatial comparisons appended to the original OFF five-point Local pass."""
from __future__ import annotations

import json
import math
import hashlib

from dq_profile.trainer_runtime import DiagnosticProfileRuntime, _set_fake_quant
from dq_profile.protocol import CandidateDefinition
from dq_profile.quant_context import ProfileQuantContext
from dq_profile.replay import seed_step_rng
from dq_profile.v24_acceptance import analyze_local_profile
from dq_profile.v2_calibration import fingerprint_tree
from dq_profile.spatial import STANDARD_VERSION, UNIFORM_MULS, build_conditions, body_from_analysis, component_moments
from library.dq_mul_policy import MulPolicy, digest


class _PairedQuantContext(ProfileQuantContext):
    def begin_pass(self, **kwargs):
        super().begin_pass(**kwargs)
        # The quantization repeat changes rounding, not the matched model/dropout RNG.
        # Hash actual mask sites only, as in the validated research protocol.
        self._dropout_hasher = hashlib.sha256()


class SpatialDiagnosticRuntime(DiagnosticProfileRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quant_context = _PairedQuantContext(self.protocol_seed, rng_mode="stateless")
        self.spatial_rows = []
        self.spatial_refs = {}
        self.spatial_policies = {}
        self.spatial_collect = False
        self.spatial_regime = "off"

    def _configure_candidate_quantization(self, network, candidate, range_mul, shadow):
        if candidate.name not in self.spatial_policies:
            return super()._configure_candidate_quantization(network, candidate, range_mul, shadow)
        _set_fake_quant(network, self.args, enabled=candidate.quantized, range_mul=range_mul)
        self.spatial_policies[candidate.name].apply(network)

    def _run_pass(self, **kwargs):
        row, gradient, shadows = super()._run_pass(**kwargs)
        if not self.spatial_collect or kwargs["phase"] != "v2_tail_probe":
            return row, gradient, shadows
        if not math.isfinite(row["loss"]) or not math.isfinite(gradient.norm) or row.get("forced_safety_abort"):
            raise FloatingPointError("Nonfinite or invalid standard diagnostic probe")
        key = str(kwargs["probe_or_step"])
        observer = getattr(self.trainer, "_dataset_diagnostics", None)
        mse = getattr(observer, "raw_mse", None)
        if mse is not None and not math.isfinite(mse):
            raise FloatingPointError("Nonfinite raw MSE in standard diagnostic probe")
        if not kwargs["candidate"].quantized:
            if self.spatial_regime == "off" and key in self.spatial_refs:
                old = self.spatial_refs[key]
                for field in ("gradient_hash", "loss", "noise_digest", "replay_digest", "timestep_digest", "dropout_mask_digest"):
                    if old.get(field) != row.get(field):
                        raise ValueError(f"Repeated OFF reference changed: {field}")
            self.spatial_reference = (key, row, gradient, mse)
            self.spatial_refs.setdefault(key, dict(row))
        else:
            ref_key, ref, reference_gradient, reference_mse = self.spatial_reference
            if ref_key != key:
                raise ValueError("Spatial probe has no matched no-quant reference")
            for field in ("noise_digest", "replay_digest", "timestep_digest", "dropout_mask_digest", "rng_digest_before"):
                if ref.get(field) != row.get(field):
                    raise ValueError(f"Candidate/reference input or dropout mismatch: {field}")
            _, image_index, bin_index, noise, _ = key.split(":", 4)
            probe = kwargs["replay"]
            image_key = probe.image_keys[0] if probe.image_keys else probe.digest
            self.spatial_rows.append({"candidate": kwargs["candidate"].name, "regime": self.spatial_regime,
                "image_key": image_key, "source_group": self.spatial_source_map.resolve(image_key),
                "image_index": int(image_index), "timestep_bin": int(bin_index), "noise_replica": int(noise),
                "quant_repeat": kwargs["repeat"], "input_id": digest([image_key, key, probe.digest]),
                "noise_digest": row["noise_digest"], "dropout_mask_digest": row.get("dropout_mask_digest"),
                "reference_hash": ref["gradient_hash"], "quantized_hash": row["gradient_hash"],
                "reference_loss": ref["loss"], "quantized_loss": row["loss"],
                "reference_mse": reference_mse, "quantized_mse": mse,
                "components": component_moments(reference_gradient, gradient)})
        return row, gradient, shadows

    def _run_tail_probes(self, **ctx):
        if tuple(self.args.dq_profile_range_muls_resolved) != UNIFORM_MULS:
            raise ValueError("Standard spatial comparison requires the uniform five-point grid")
        selected, self.spatial_source_map = self._select_probe_items(ctx["sequence"])
        network = ctx["accelerator"].unwrap_model(ctx["network"])
        names = [m.lora_name for m in network.text_encoder_loras + network.unet_loras]
        fixed = build_conditions(None)[0][-1]
        MulPolicy.from_dict(fixed["policy"]).expand(names)  # Validate topology before measuring.
        self.spatial_collect = True
        try:
            original = super()._run_tail_probes(**ctx)
            summary = {"schema_version": "2.1.0", "profile": {"protocol": "v24-acceptance-local", "timestep_bins": self.args.dq_profile_timestep_bins},
                       "candidates": [c.to_dict() | {"candidate": c.name} for c in self.candidates]}
            uniform = analyze_local_profile(summary=summary, gradient_tail_rows=self._tail_probe_result["gradient_tail_rows"],
                dataset_id="standard", core_grid=UNIFORM_MULS)
            body = body_from_analysis(uniform)
            conditions, skipped = build_conditions(body, spatial=not getattr(self.args, "dq_profile_uniform_only", False))
            assignments = {}
            for item in conditions:
                policy = MulPolicy.from_dict(item["policy"])
                expanded = policy.expand(names)
                assignments[item["id"]] = {"declaration": policy.to_dict(), "assignments_sha256": digest(expanded)}
                self.spatial_policies[item["id"]] = policy
            extra = [item for item in conditions if item["kind"] != "uniform"]
            if extra:
                self._measure_additional(selected, extra, ctx, dropout=False)
            if getattr(self.args, "dq_profile_dropout_on", False):
                self._measure_additional(selected, conditions, ctx, dropout=True)
            snapshot = ctx["snapshot"]
            metadata = {"version": STANDARD_VERSION, "conditions": conditions, "skipped": skipped,
                "protocol_seed": self.protocol_seed,
                "body_mul": body, "body_selection": uniform["selection"], "assignments": assignments,
                "state_sha256": fingerprint_tree(snapshot.network_state), "step": snapshot.metadata["global_step"],
                "quantization": {"bits": self.args.dq_delta_bits, "stat": self.args.dq_delta_stat,
                    "mode": self.args.dq_delta_mode, "granularity": self.args.dq_delta_granularity,
                    "scope": "UNet+TE1+TE2"},
                "training_dropout": {"network": getattr(self.args, "network_dropout", None),
                    "rank": next((float(v.split("=", 1)[1]) for v in (getattr(self.args, "network_args", None) or [])
                                  if v.startswith("rank_dropout=")), None)},
                "source_map": self.spatial_source_map.manifest(), "dropout_on": bool(getattr(self.args, "dq_profile_dropout_on", False)),
                "bins": int(self.args.dq_profile_timestep_bins), "candidate_noises": int(self.args.dq_profile_safety_candidate_noise_replicas_resolved),
                "quant_repeats": int(self.args.dq_profile_safety_quant_repeats_resolved),
                "probe_address_version": "existing-local-address-v1", "not_image_quality": True,
                "additional_reference_probes": len(selected) * int(self.args.dq_profile_timestep_bins) * int(self.args.dq_profile_safety_candidate_noise_replicas_resolved) * (bool(extra) + bool(getattr(self.args, "dq_profile_dropout_on", False)))}
            target = self.artifacts.root
            reference_calls = len(selected) * int(self.args.dq_profile_timestep_bins) * int(self.args.dq_profile_safety_no_quant_noise_replicas_resolved)
            metadata["local_forward_backward_calls"] = len(self.spatial_rows) + reference_calls + metadata["additional_reference_probes"]
            metadata["quantized_probe_calls"] = len(self.spatial_rows)
            with (target / "spatial_observations.jsonl").open("w", encoding="utf-8") as stream:
                for row in self.spatial_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            (target / "spatial_contract.json").write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            return original  # Legacy acceptance statistics remain OFF uniform-only.
        finally:
            self.spatial_collect = False
            ctx["snapshot"].restore(network=network, optimizer=ctx["optimizer"], scheduler=ctx["lr_scheduler"],
                scaler=getattr(ctx["accelerator"], "scaler", None), trainer=self.trainer, guardian=ctx["grad_norm_guardian"])

    def _measure_additional(self, selected, conditions, ctx, *, dropout):
        snapshot = ctx["snapshot"]
        network = ctx["accelerator"].unwrap_model(ctx["network"])
        snapshot.restore(network=network, optimizer=ctx["optimizer"], scheduler=ctx["lr_scheduler"],
            scaler=getattr(ctx["accelerator"], "scaler", None), trainer=self.trainer, guardian=ctx["grad_norm_guardian"])
        self.spatial_regime = "on" if dropout else "off"
        run_context = {key: value for key, value in ctx.items() if key not in {"sequence", "snapshot"}}
        bins = int(self.args.dq_profile_timestep_bins)
        noises = int(self.args.dq_profile_safety_candidate_noise_replicas_resolved)
        repeats = int(self.args.dq_profile_safety_quant_repeats_resolved)
        for image_index, source in enumerate(selected):
            image_key = source.image_keys[0] if source.image_keys else source.digest
            for bin_index in range(bins):
                for noise in range(noises):
                    probe = self._fixed_timestep_item(source, probe_replica=noise, bin_index=bin_index,
                        bin_count=bins, noise_scheduler=ctx["noise_scheduler"])
                    key = f"tail:{image_index}:{bin_index}:{noise}:{image_key}"
                    seed_id = f"image:{image_index}|bin:{bin_index}|noise:{noise}|key:{image_key}"
                    definitions = [(self.candidates[0], 0)]
                    for item in conditions:
                        definitions.extend((CandidateDefinition(item["id"], True, None, None, item["mul"][0], False), repeat) for repeat in range(repeats))
                    for candidate, repeat in definitions:
                        seed_step_rng(self.protocol_seed, seed_id, phase="v2_tail_structural_model", repeat=0)
                        self._run_pass(replay=probe, candidate=candidate, range_mul=candidate.initial_range_mul,
                            phase="v2_tail_probe", probe_or_step=key, repeat=repeat, dropout_enabled=dropout,
                            shadow=False, update=False, do_auto_observation=False,
                            absolute_step=int(snapshot.metadata["global_step"]), epoch=int(snapshot.metadata["epoch"]), **run_context)
