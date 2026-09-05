"""Optional observer attached only to the isolated diagnostic trainer."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import numpy as np

from dq_profile.dataset_diagnostics import (SCHEMA, finite, gradient_metrics, identity,
    load_group_map, normalized_path, rebuild, tags, write_json)
from dq_profile.replay import seed_step_rng
from dq_profile.snapshot import (TrainingSnapshot, clone_state_to_cpu, _capture_network_runtime,
                                 _restore_network_runtime, _trainer_state)
from dq_profile.v2_calibration import fingerprint_tree


def scalar_tree(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return scalar_tree(value.tolist())
    if isinstance(value, torch.Tensor):
        return scalar_tree(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): scalar_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_tree(v) for v in value]
    if isinstance(value, float) and not finite(value):
        return None
    return value


def build_inventory(dataset_group):
    rows = []
    for di, dataset in enumerate(getattr(dataset_group, "datasets", [dataset_group])):
        for key, info in dataset.image_data.items():
            subset = dataset.image_to_subset[key]
            path = normalized_path(info.absolute_path)
            folder = normalized_path(getattr(subset, "image_dir", None) or Path(path).parent)
            subset_index = int(subset.subset_index)
            resolution = list(getattr(dataset, "resolution", None) or (dataset.width, dataset.height))
            row = {"image_id": identity(path), "sample_id": identity(path, di, subset_index, resolution),
                   "path": path, "name": Path(path).name, "folder_path": folder,
                   "folder_id": identity(folder), "folder_name": Path(folder).name,
                   "dataset_index": di, "subset_index": subset_index,
                   "resolution": resolution, "bucket_resolution": scalar_tree(info.bucket_reso),
                   "caption": info.caption, "tags": tags(info.caption),
                   "class_tokens": getattr(subset, "class_tokens", None),
                   "subset_group": getattr(subset, "group", None),
                   "num_repeats": info.num_repeats, "is_reg": info.is_reg,
                   "presented_count": 0, "updated_count": 0, "skipped_count": 0,
                   "first_seen_step": None, "last_seen_step": None}
            rows.append(row)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate diagnostic sample identity")
    return rows


def runtime_copy(value):
    # hashlib accumulators are mutable but deliberately not pickleable.
    memo = {}
    seen = set()
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if hasattr(item, "hexdigest") and hasattr(item, "copy"):
            memo[id(item)] = item.copy()
        elif isinstance(item, dict):
            for v in item.values():
                visit(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                visit(v)
        elif hasattr(item, "__dict__") and not isinstance(item, (torch.Tensor, torch.nn.Module)):
            visit(vars(item))
    visit(value)
    return copy.deepcopy(value, memo)


class ModelEnvelope:
    """Preserve children modes, buffers and DQ runtime objects outside state_dict.

    Network/optimizer/scaler/RNG/guardian are additionally covered by TrainingSnapshot.
    Parameter objects and DQ manager/context identities are never replaced.
    """
    def __init__(self, modules, runtime):
        self.modules = list(dict.fromkeys(m for root in modules for m in root.modules()))
        self.modes = [m.training for m in self.modules]
        self.buffers = [(b, b.detach().clone()) for m in self.modules for b in m.buffers(recurse=False)]
        self.flags = [(p, p.requires_grad) for m in self.modules for p in m.parameters(recurse=False)]
        self.attrs = []
        self.objects = {}
        for m in self.modules:
            attrs = {}
            for key, value in vars(m).items():
                if key.startswith(("delta_q_", "dq_")) or key == "multiplier":
                    if value is not None and hasattr(value, "__dict__") and not isinstance(value, torch.Tensor):
                        self.objects[id(value)] = (value, runtime_copy(vars(value)))
                        attrs[key] = (True, value)
                    else:
                        attrs[key] = (False, copy.deepcopy(value))
            self.attrs.append(attrs)
        self.runtime = runtime
        self.sequence = runtime._stats_sequence
        self.objects[id(runtime.quant_context)] = (runtime.quant_context, runtime_copy(vars(runtime.quant_context)))

    def restore(self):
        with torch.no_grad():
            for b, saved in self.buffers:
                b.copy_(saved)
        for p, flag in self.flags:
            p.requires_grad_(flag)
        for m, mode, attrs in zip(self.modules, self.modes, self.attrs):
            m.training = mode
            for key in list(vars(m)):
                if (key.startswith(("delta_q_", "dq_")) or key == "multiplier") and key not in attrs:
                    delattr(m, key)
            for key, (is_object, value) in attrs.items():
                setattr(m, key, value if is_object else copy.deepcopy(value))
        for obj, state in self.objects.values():
            obj.__dict__.clear()
            obj.__dict__.update(runtime_copy(state))
        self.runtime._stats_sequence = self.sequence
        if any(m.training != mode for m, mode in zip(self.modules, self.modes)) or any(not torch.equal(b, v) for b, v in self.buffers):
            raise RuntimeError("failed_state_restoration: module modes/buffers")


class DatasetDiagnostics:
    def __init__(self, args, dataset, network, unet, text_encoders, trainer):
        self.args, self.mode = args, args.dq_profile_data_diagnostics
        if args.dq_profile_protocol != "v24-acceptance-local":
            raise ValueError("dataset diagnostics currently requires v24-acceptance-local")
        if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
            raise ValueError("dataset diagnostics requires gradient_accumulation_steps=1")
        self.inventory = build_inventory(dataset)
        self.index = {(r["path"], r["subset_index"]): r for r in self.inventory}
        self.references, self.quant, self.inputs, self.bank = [], [], {}, []
        self.active = False
        self.raw_mse = None
        self.group_map = load_group_map(getattr(args, "dq_profile_group_map", None))
        self.initial = None
        self.restoration = {"status": "not_run"}
        if self.mode == "warmup":
            network_ids = {id(p) for p in network.parameters()}
            if any(p.requires_grad and id(p) not in network_ids for root in [unet, *text_encoders] for p in root.parameters()):
                raise ValueError("warmup diagnostics requires frozen base weights outside the saved network")
            self.initial = {"network": clone_state_to_cpu(network.state_dict()),
                            "runtime": _capture_network_runtime(network), "trainer": _trainer_state(trainer),
                            "buffers": [(b, clone_state_to_cpu(b)) for root in [unet, *text_encoders] for b in root.buffers()]}
            self.initial_hash = fingerprint_tree(self.initial["network"])

    def sample(self, batch):
        if len(batch["image_keys"]) != 1:
            raise ValueError("dataset diagnostics requires batch_size=1; observations cannot be split after averaging")
        key = (normalized_path(batch["image_keys"][0]), int(batch["subset_indices"][0]))
        if key not in self.index:
            raise ValueError(f"diagnostic replay sample is absent from loader inventory: {key}")
        return self.index[key]

    def observe_training(self, batch, step, updated):
        row = self.sample(batch)
        row["presented_count"] += 1
        row["updated_count" if updated else "skipped_count"] += 1
        if row["first_seen_step"] is None:
            row["first_seen_step"] = int(step)
        row["last_seen_step"] = int(step)

    def tap(self, prediction, target):
        if self.active:
            with torch.no_grad():
                self.target_dtype = str(target.dtype)
                self.raw_mse = float((prediction.detach().float() - target.detach().float()).square().mean().item())

    def record(self, probe, base, row, *, comparison=None, quant_repeat=None, model_seed_id=None):
        sample = self.sample(probe.batch)
        sample["source_group_id"] = base["source_group"]
        eid = identity(sample["sample_id"], probe.digest)
        values = {"sample_id": sample["sample_id"], "image_id": sample["image_id"], "eval_input_id": eid,
                  "source_group_id": base["source_group"], "bin": base["timestep_bin"],
                  "noise": base["noise_replica"], "timestep": base["timestep"],
                  "raw_mse": self.raw_mse, "objective_loss": row["loss"],
                  "invalid_reason": None if finite(self.raw_mse) else "nonfinite_raw_mse"}
        if comparison is None:
            self.references.append(scalar_tree({**values, "snapshot": "post"}))
            metadata = {k: v for k, v in probe.batch.items() if k in {
                "captions", "input_ids", "input_ids2", "input_ids1", "flippeds", "original_sizes_hw",
                "crop_top_lefts", "target_sizes_hw", "bucket_resos", "subset_indices", "network_multipliers", "network_multiplier", "loss_weights"}}
            self.inputs[eid] = {"eval_input_id": eid, "sample_id": sample["sample_id"], "digest": probe.digest,
                                "model_seed_id": model_seed_id, "batch": scalar_tree(metadata),
                                "bin": base["timestep_bin"], "noise": base["noise_replica"], "timestep": base["timestep"],
                                "noise_digest": row.get("noise_digest"), "target_dtype": self.target_dtype,
                                "token_truncation_status": "unknown",
                                "caption_changed_from_original": probe.batch.get("captions", [None])[0] != sample["caption"]}
            if self.mode == "warmup":
                self.bank.append((probe, dict(values), model_seed_id))
        else:
            self.quant.append(scalar_tree({**values, "mul": row["range_mul"], "quant_repeat": quant_repeat,
                **gradient_metrics(comparison),
                **{k: row.get(k) for k in ("clip_rate", "quant_error_ratio", "clip_error_rms", "round_error_rms")}}))

    def evaluate_initial(self, runtime, snapshot, **context):
        self.active = False
        self.post_state_hash = fingerprint_tree(snapshot.network_state)
        if self.mode != "warmup":
            return
        accelerator, network = context["accelerator"], context["network"]
        unwrapped = accelerator.unwrap_model(network)
        restore_args = dict(network=unwrapped, optimizer=context["optimizer"], scheduler=context["lr_scheduler"],
                            scaler=getattr(accelerator, "scaler", None), trainer=runtime.trainer,
                            guardian=context["grad_norm_guardian"])
        saved = TrainingSnapshot.capture(**restore_args, **{k: snapshot.metadata[k] for k in ("global_step", "epoch", "data_step")})
        envelope = ModelEnvelope([unwrapped, context["unet"], *context["text_encoders"]], runtime)
        before = fingerprint_tree(vars(saved))
        try:
            unwrapped.load_state_dict(self.initial["network"], strict=True)
            _restore_network_runtime(unwrapped, self.initial["runtime"])
            for key, value in self.initial["trainer"].items():
                if key != "_te_frozen_param_names":
                    setattr(runtime.trainer, key, clone_state_to_cpu(value))
            with torch.no_grad():
                for buffer, value in self.initial["buffers"]:
                    buffer.copy_(value)
            for probe, values, seed_id in self.bank:
                seed_step_rng(runtime.protocol_seed, seed_id, phase="v2_tail_structural_model", repeat=0)
                self.active, self.raw_mse = True, None
                pre, _, _ = runtime._run_pass(replay=probe, candidate=next(c for c in runtime.candidates if not c.quantized),
                    range_mul=None, phase="dataset_pre", probe_or_step=values["eval_input_id"], repeat=0,
                    dropout_enabled=False, shadow=False, update=False, do_auto_observation=False,
                    absolute_step=int(snapshot.metadata["global_step"]), epoch=int(snapshot.metadata["epoch"]),
                    diagnostic_forward_only=True, **context)
                self.references.append(scalar_tree({**values, "snapshot": "pre", "raw_mse": self.raw_mse,
                    "objective_loss": pre["loss"], "invalid_reason": None if finite(self.raw_mse) else "nonfinite_raw_mse"}))
        finally:
            self.active = False
            saved.restore(**restore_args)
            envelope.restore()
            after = TrainingSnapshot.capture(**restore_args, **{k: snapshot.metadata[k] for k in ("global_step", "epoch", "data_step")})
            after_hash = fingerprint_tree(vars(after))
            self.restoration = {"status": "passed" if before == after_hash else "failed", "before": before, "after": after_hash}
            if before != after_hash:
                raise RuntimeError("failed_state_restoration: dataset initial evaluation")
            self.bank.clear()
            self.initial = None

    def finish(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": SCHEMA, "mode": self.mode, "selector_input": False,
            "loss_metric": "raw_mse", "prediction_target": "velocity" if getattr(self.args, "v_parameterization", False) else "epsilon",
            "measurement_contract": self.args.dq_profile_measurement_contract,
            "max_images": self.args.dq_profile_max_images, "bins": self.args.dq_profile_timestep_bins,
            "muls": list(self.args.dq_profile_range_muls_resolved), "group_map": self.group_map,
            "initial_state_hash": getattr(self, "initial_hash", None), "post_state_hash": getattr(self, "post_state_hash", None), "state_restoration": self.restoration,
            "numerical_mode_parity": "not_validated_on_full_sdxl_run", "ci_status": "not_computed",
            "caption_policy": "frozen_local_replay", "mask_metric": "raw_full_latent_before_mask",
            "noise_options": {key: scalar_tree(getattr(self.args, key, None)) for key in ("noise_offset", "noise_offset_random_strength", "multires_noise_iterations", "multires_noise_discount", "adaptive_noise_scale", "ip_noise_gamma", "zero_terminal_snr")},
            "local_input_limitations": ["uses existing materialized Local noise and target; does not reselect inputs or regenerate progress-zero noise", "initial forward uses same autograd/checkpointing path; backward and optimizer updates omitted"],
            "pairing": "same sample/eval_input/bin/noise; candidate noise 0,1 only"}
        write_json(directory / "manifest.json", manifest)
        for name, rows in (("inventory", self.inventory), ("reference_probes", self.references),
                           ("quant_probes", self.quant), ("evaluation_inputs", self.inputs.values())):
            with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(scalar_tree(row), ensure_ascii=False, allow_nan=False) + "\n")
        rebuild(directory)
