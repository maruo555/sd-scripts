"""Standard-only Prefix QA and Local measurement from one warmup boundary.

Strict keeps its independent worker/snapshot checks. No measurements or Prefix
updates are removed here; only repeated warmup/model loading is avoided.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from dq_profile.diagnostic_eval import ModelEnvelope
from dq_profile.manifest import build_source_manifest
from dq_profile.protocol import canonical_sha256
from dq_profile.replay import ReplaySequence, replay_digest
from dq_profile.snapshot import TrainingSnapshot, capture_rng_state, restore_rng_state
from dq_profile.v2_calibration import fingerprint_tree, source_contract_from_manifest


def prefix_args_from_local(args, prefix_dir: Path):
    # Preserve a caller's budget callback identity; Prefix updates still count.
    callback = getattr(args, "_dq_profile_budget_callback", None)
    prefix = copy.deepcopy(args, {id(callback): callback} if callback is not None else {})
    prefix.dq_profile_protocol = "v2-prefix-smoke"
    prefix.dq_profile_shared_warmup = False
    prefix.dq_profile_snapshot_only = False
    prefix.dq_profile_standard_version = 1
    prefix.dq_profile_data_diagnostics = "off"
    prefix.dq_profile_dropout_on = False
    prefix.dq_profile_uniform_only = False
    prefix.dq_profile_te_quantized = False  # Prefix uses the preset's dq_delta_scope, as before.
    prefix.dq_profile_policy_grid_file = None
    prefix.dq_profile_policy_grid_resolved = {}
    prefix.dq_profile_prefix_gate_file = None
    prefix.dq_profile_range_muls = "3.15"
    prefix.dq_profile_requested_range_muls = "3.15"
    prefix.dq_profile_range_muls_resolved = (3.15,)
    prefix.dq_profile_branch_repeats = 1
    prefix.dq_profile_guardian_ablation = "common_only"
    prefix.dq_profile_probe_replicas_resolved = 0
    prefix.dq_profile_capture_steps = int(prefix.dq_profile_prefix_long_steps)
    prefix.dq_profile_sweep_steps = prefix.dq_profile_capture_steps
    prefix.dq_profile_run_dir = str(prefix_dir)
    prefix.dq_profile_output_dir = str(prefix_dir.parent)
    prefix.dq_profile_name = prefix_dir.name
    prefix.output_dir = str(prefix_dir)
    prefix.output_name = prefix_dir.name
    if hasattr(prefix, "dq_profile_prefix_kernel_policy"):
        prefix.dq_profile_prefix_kernel_policy["protocol"] = prefix.dq_profile_protocol
    short, long = int(prefix.dq_profile_prefix_short_steps), int(prefix.dq_profile_prefix_long_steps)
    updates = 2 * (short * 2 + long)
    payload = prefix.dq_profile_preflight
    payload.update({
        "branch_steps": long, "standard_probe_replicas": 0, "full_probe_replicas": 0,
        "estimated_standard_steps": updates, "estimated_full_steps": updates,
        "estimated_standard_epochs": updates / max(1, payload["steps_per_epoch"]),
        "estimated_full_epochs": updates / max(1, payload["steps_per_epoch"]),
    })
    payload["v2"] = {
        "protocol": prefix.dq_profile_protocol, "range_muls": [3.15],
        "sweep_steps": long, "branch_repeats": 1, "capture_steps": long,
        "snapshot_only": False, "warmup_reused": True,
        "cost_components": {"warmup_steps": 0, "prefix_branch_steps": updates, "structural_probe_steps": 0},
        "prefix_contract": {
            "execution_mode": prefix.dq_profile_execution_mode, "qa_depth": prefix.dq_profile_qa_depth,
            "short_steps": short, "long_steps": long, "checkpoints": sorted({0, 1, short // 2, short}),
        },
    }
    return prefix


def snapshot_fingerprints(snapshot):
    fields = {
        "network": "network_state", "optimizer": "optimizer_state", "scheduler": "scheduler_state",
        "scaler": "scaler_state", "rng": "rng_state", "network_runtime": "network_runtime",
        "trainer": "trainer_state", "guardian": "guardian_state", "metadata": "metadata",
    }
    values = {key: fingerprint_tree(getattr(snapshot, field)) for key, field in fields.items()}
    return {**values, "combined": canonical_sha256(values)}


def _replay_fingerprint(sequence):
    # Recompute from tensors, not cached digest strings, to detect in-place edits.
    return canonical_sha256([replay_digest(vars(item)) for item in sequence])


def validate_shared_gate(args, prefix_dir: Path):
    gate_path = prefix_dir / "calibration_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("schema_version") != "2.1.0" or gate.get("metric_definition_version") != "2.1.0"
            or gate.get("passed") is not True or gate.get("gate") not in {"pass_exact", "pass_numeric"}):
        raise RuntimeError("shared warmup Prefix gate failed; Local measurement was not started")
    expected = args.dq_profile_shared_source_contract_sha256
    saved_manifest = json.loads((prefix_dir / "source_manifest.json").read_text(encoding="utf-8"))
    inputs = [args.dataset_config]
    if getattr(args, "dq_profile_source_group_map", None):
        inputs.append(args.dq_profile_source_group_map)
    current, _ = build_source_manifest(Path(__file__).resolve().parents[1], quant_rng_mode="stateless", additional_files=tuple(inputs))
    if (not expected or gate.get("source_contract_sha256") != expected
            or source_contract_from_manifest(saved_manifest)["sha256"] != expected
            or source_contract_from_manifest(current)["sha256"] != expected):
        raise RuntimeError("source contract changed during shared warmup/Prefix; Local measurement was not started")
    args.dq_profile_prefix_gate_file = str(gate_path)
    args.dq_profile_prefix_gate_sha256 = canonical_sha256(gate)
    args.dq_profile_expected_source_contract_sha256 = expected
    args.dq_profile_current_source_contract_sha256 = expected
    return {"passed": True, "gate_file": str(gate_path), "gate_file_sha256": canonical_sha256(gate),
            "expected_source_contract_sha256": expected, "current_source_contract_sha256": expected}


def run_shared_prefix(*, runtime, snapshot, sequence, boundary_context):
    from dq_profile.trainer_runtime import DiagnosticProfileRuntime

    args = runtime.args
    if runtime.profile_protocol != "v24-acceptance-local" or args.dq_profile_execution_mode != "standard":
        raise ValueError("shared warmup is only supported by Standard Local")
    prefix_dir = runtime.artifacts.root / "shared_prefix"
    if prefix_dir.exists():
        raise FileExistsError(f"refusing to overwrite shared Prefix artifacts: {prefix_dir}")
    prefix_args = prefix_args_from_local(args, prefix_dir)
    if len(sequence) < prefix_args.dq_profile_capture_steps:
        raise RuntimeError("shared replay is shorter than the Prefix contract")
    prefix_sequence = ReplaySequence()
    for item in list(sequence)[:prefix_args.dq_profile_capture_steps]:
        prefix_sequence.append(copy.deepcopy(item))
    prefix_sequence.seal()
    if not all(item.materialized() for item in prefix_sequence):
        raise RuntimeError("shared Prefix requires materialized replay inputs")
    replay_before = _replay_fingerprint(sequence)
    prefix_replay_before = _replay_fingerprint(prefix_sequence)
    expected = snapshot_fingerprints(snapshot)
    accelerator = boundary_context["accelerator"]
    network = accelerator.unwrap_model(boundary_context["network"])
    restore_context = {
        "network": network, "optimizer": boundary_context["optimizer"],
        "scheduler": boundary_context["lr_scheduler"], "scaler": getattr(accelerator, "scaler", None),
        "trainer": runtime.trainer, "guardian": boundary_context["grad_norm_guardian"],
    }
    # Keep the caller's post-materialization RNG, too; Local then performs its
    # usual snapshot restores and per-probe seeding, unchanged.
    caller_rng = capture_rng_state()
    snapshot.restore(**restore_context)
    envelope = ModelEnvelope([network, boundary_context["unet"], *boundary_context["text_encoders"]], runtime)
    observer = getattr(runtime.trainer, "_dataset_diagnostics", None)
    runtime.trainer._dataset_diagnostics = None
    prefix = DiagnosticProfileRuntime(args=prefix_args, trainer=runtime.trainer)
    logging.getLogger(__name__).info("Shared warmup: validating Prefix before Local measurement")
    try:
        prefix.run_from_boundary(
            **boundary_context,
            _prepared_boundary=(snapshot, prefix_sequence, {
                "warmup_reused": True, "replay_source": "local_materialized_prefix",
                "retained_batches": len(prefix_sequence), "live_dataloader_reused_by_branches": False,
            }),
            _end_training=False,
        )
    except BaseException as error:
        prefix.artifacts.mark_failed(error)
        raise
    finally:
        try:
            snapshot.restore(**restore_context)
            envelope.restore()
            restored = TrainingSnapshot.capture(
                **restore_context, global_step=boundary_context["global_step"],
                epoch=boundary_context["epoch"], data_step=boundary_context["data_step"],
            )
            restored_fingerprints = snapshot_fingerprints(restored)
            if restored_fingerprints != expected:
                raise RuntimeError("shared warmup state restoration failed; Local measurement was not started")
        finally:
            runtime.trainer._dataset_diagnostics = observer
            restore_rng_state(caller_rng)
    if _replay_fingerprint(sequence) != replay_before or _replay_fingerprint(prefix_sequence) != prefix_replay_before:
        raise RuntimeError("shared warmup replay changed during Prefix; Local measurement was not started")
    source_gate = validate_shared_gate(args, prefix_dir)
    record = {
        "schema_version": "shared-warmup-v1", "passed": True,
        "warmup_executions": 1, "independent_warmup_parity_performed": False,
        "snapshot_fingerprints": expected, "restored_snapshot_fingerprints": restored_fingerprints,
        "replay_sha256": replay_before, "prefix_replay_sha256": prefix_replay_before,
        "prefix_replay_batches": len(prefix_sequence), "prefix_source_gate": source_gate,
        "dataset_diagnostics_observer_preserved": True,
    }
    runtime.artifacts.write_json("shared_warmup.json", record)
    logging.getLogger(__name__).info("Shared warmup: Prefix passed and boundary restored; starting Local measurement")
    return record
