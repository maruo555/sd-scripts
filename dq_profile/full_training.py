"""Budget and evidence adapter for opt-in, ordinary full-training research jobs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePath
import struct
import time

import torch
from safetensors import safe_open

from library.dq_mul_policy import digest
from dq_profile.research_budget import ResearchStop, file_hash, read_json, write_json


def training_args_record(args):
    """Preserve effective CLI values, including paths resolved by the trainer."""
    def serialize(value):
        if isinstance(value, PurePath):
            return str(value)
        raise TypeError(f"Unsupported training argument type: {type(value).__name__}")

    values = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    return json.loads(json.dumps(values, ensure_ascii=False, allow_nan=False, default=serialize))


def _floating_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            yield value.detach()
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating_tensors(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ResearchStop("Nonfinite scalar in optimizer state")


def assert_finite_tensors(values, label):
    groups = defaultdict(list)
    for tensor in values:
        groups[(tensor.device, tensor.dtype)].append(tensor)
    finite_norm_overflows = 0
    for tensors in groups.values():
        # One host synchronization per dtype/device, instead of one per tensor.
        norms = torch._foreach_norm(tensors)
        if not bool(torch.isfinite(torch.stack(norms)).all().item()):
            for tensor, norm in zip(tensors, norms):
                if bool(torch.isfinite(norm).item()):
                    continue
                # A norm reduction can overflow while every tensor element is
                # finite. Only actual nonfinite values fail this value check.
                if not bool(torch.isfinite(tensor).all().item()):
                    raise ResearchStop("Nonfinite " + label)
                finite_norm_overflows += 1
    return finite_norm_overflows


def check_backward(network, loss):
    if not math.isfinite(float(loss.detach())):
        raise ResearchStop("Nonfinite training loss")
    return assert_finite_tensors((p.grad.detach() for p in network.parameters() if p.grad is not None), "training gradients")


def backward_failure_evidence(network, accelerator, loss, global_step, reason):
    """Capture scalars and bad-gradient counts without changing training state."""
    loss_value = float(loss.detach())
    scaler = getattr(accelerator, "scaler", None)
    scale = float(scaler.get_scale()) if scaler is not None else None
    gradients = [(name, p.grad.detach()) for name, p in network.named_parameters() if p.grad is not None]
    bad = []
    for name, gradient in gradients:
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        bad.append({"name": name, "dtype": str(gradient.dtype), "device": str(gradient.device),
                    "shape": list(gradient.shape), "elements": gradient.numel(),
                    "nonfinite_elements": int((~finite).sum().item()),
                    "nan_elements": int(torch.isnan(gradient).sum().item()),
                    "positive_inf_elements": int(torch.isposinf(gradient).sum().item()),
                    "negative_inf_elements": int(torch.isneginf(gradient).sum().item())})
    return {"reason": str(reason), "recorded_at": time.time(), "scheduled_step": int(global_step) + 1,
            "stage": "after_backward_before_native_guardian_clip_optimizer",
            "loss_is_finite": math.isfinite(loss_value), "loss_if_finite": loss_value if math.isfinite(loss_value) else None,
            "scaler_scale": scale if scale is None or math.isfinite(scale) else None,
            "scaler_scale_is_finite": scale is None or math.isfinite(scale),
            "gradient_tensor_count": len(gradients), "nonfinite_gradient_tensors": bad,
            "failed_step_optimizer_update_performed": False}


def check_updated_state(network, optimizer):
    assert_finite_tensors((p.detach() for p in network.parameters()), "training parameters")
    assert_finite_tensors(_floating_tensors(optimizer.state), "optimizer state")


def training_state_fingerprint(network, optimizer):
    """Hash exact parameter/state bytes without changing values, devices or RNG."""
    fingerprint = hashlib.sha256()

    def token(value):
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        fingerprint.update(struct.pack("<Q", len(data)))
        fingerprint.update(data)

    def visit(value):
        if isinstance(value, torch.Tensor):
            token(["tensor", str(value.dtype), str(value.device), list(value.shape)])
            raw = value.detach().to(device="cpu", copy=True).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            fingerprint.update(struct.pack("<Q", len(raw)))
            fingerprint.update(raw)
        elif isinstance(value, dict):
            token(["dict", len(value)])
            for key, item in value.items():
                if isinstance(key, torch.Tensor):
                    token(["parameter_key", id(key)])
                else:
                    visit(key)
                visit(item)
        elif isinstance(value, (tuple, list)):
            token([type(value).__name__, len(value)])
            for item in value:
                visit(item)
        elif value is None or isinstance(value, (str, int, float, bool)):
            token([type(value).__name__, value])
        elif isinstance(value, (torch.dtype, torch.device)):
            token([type(value).__name__, str(value)])
        else:
            raise ResearchStop("Unsupported optimizer state type: " + type(value).__name__)

    for name, parameter in network.named_parameters():
        token(["parameter", name, id(parameter)])
        visit(parameter)
    visit(optimizer.state)
    for group in optimizer.param_groups:
        visit({key: value for key, value in group.items() if key != "params"})
        token(["group_parameters", [id(parameter) for parameter in group["params"]]])
    visit(optimizer.defaults)
    return fingerprint.hexdigest()


def read_checkpoint_metadata(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 32 * 1024 * 1024:
            raise ResearchStop("Unexpected checkpoint header length")
        header = json.loads(stream.read(size))
    return header.get("__metadata__", {})


def index_checkpoints(output, *, definition, expected_assignment, expected_steps=13600, complete=False):
    output = Path(output)
    rows = []
    for path in sorted(output.glob("*.safetensors")):
        metadata = read_checkpoint_metadata(path)
        if metadata.get("ss_dq_mul_assignments_sha256") != expected_assignment:
            raise ResearchStop("Checkpoint has the wrong mul policy: " + path.name)
        steps, epoch = int(metadata.get("ss_steps", -1)), int(metadata.get("ss_epoch", -1))
        if steps <= 0 or steps > expected_steps or steps != epoch * 340:
            raise ResearchStop("Checkpoint step is outside the agreed schedule")
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            if not tensors.keys():
                raise ResearchStop("Empty checkpoint")
            for name in tensors.keys():
                if not bool(torch.isfinite(tensors.get_tensor(name)).all().item()):
                    raise ResearchStop("Nonfinite saved checkpoint: " + path.name)
        variant = metadata.get("ss_avg_shadow_variant")
        if variant == "final_raw":
            provenance = "final_raw_before_final_averaging; trajectory may include earlier promote decisions"
        elif path.stem == "mul_" + definition["id"]:
            provenance = "trainer_final; consult avg promote log for averaging and prior trajectory"
        else:
            provenance = "epoch_checkpoint; consult avg promote log for raw/selected history"
        rows.append({"condition": definition["id"], "epoch": epoch, "step": steps, "variant": variant or "ordinary_save",
                     "provenance": provenance, "path": str(path.resolve()), "bytes": path.stat().st_size,
                     "sha256": file_hash(path), "policy_sha256": expected_assignment,
                     "saved_tensors_finite": True,
                     "saved_at": metadata.get("modelspec.date"), "policy": definition["policy"]})
    if complete:
        if not {1, *range(2, 41)}.issubset({row["epoch"] for row in rows}):
            raise ResearchStop("Missing epoch checkpoint(s)")
        if sum(row["step"] == expected_steps and row["variant"] == "final_raw" for row in rows) != 1:
            raise ResearchStop("Missing unique final_raw checkpoint")
        if not any(row["step"] == expected_steps and Path(row["path"]).stem == "mul_" + definition["id"] for row in rows):
            raise ResearchStop("Missing final checkpoint")
    write_json(output / "checkpoint_index.json", {"complete": complete, "condition": definition["id"], "rows": rows})
    return rows


class BudgetTrainingObserver:
    def __init__(self, budget, output, definition, expected_assignment, *, gradient_handling="strict"):
        if gradient_handling not in {"strict", "verified_native_skip"}:
            raise ResearchStop("Unknown research gradient handling mode")
        self.budget, self.output = budget, Path(output)
        self.definition, self.expected_assignment = definition, expected_assignment
        self.rows_path = self.output / "step_audit.jsonl"
        self.actual_updates = self.guardian_skips = self.amp_skips = 0
        self.finite_gradient_norm_overflows = 0
        self.gradient_handling = gradient_handling
        self.pending_native_skip = None
        self.verified_nonfinite_skips = 0
        self.consecutive_nonfinite_skips = 0
        self.started_at = time.time()
        self.last_step = 0

    def __call__(self, event, **state):
        if event == "start":
            self.network, self.optimizer = state["network"], state["optimizer"]
            self.accelerator = state["accelerator"]
            args = state["args"]
            if args.max_train_steps != 13600 or args.max_train_epochs != 40 or state["global_step"] != 0:
                raise ResearchStop("Full training must start at zero with the agreed 40-epoch schedule")
            if args.gradient_accumulation_steps != 1 or self.accelerator.num_processes != 1:
                raise ResearchStop("Full training requires one GPU and one accumulation step")
            if not state["policy"] or state["policy"]["assignments_sha256"] != self.expected_assignment:
                raise ResearchStop("Ordinary trainer resolved a different policy than diagnosis")
            self.mapping = state["policy"]["modules"]
            write_json(self.output / "resolved_policy.json", state["policy"])
            write_json(self.output / "effective_training_args.private.json", training_args_record(args))
            check_updated_state(self.network, self.optimizer)
        elif event == "before_training_batch":
            self.budget.consume_forward_backward("full_training")
        elif event == "before_shadow_forward":
            self.budget.consume_forward_backward("avg_shadow_forward_only")
        elif event == "after_shadow_forward":
            if not math.isfinite(float(state["loss"].detach())):
                raise ResearchStop("Nonfinite avg shadow score")
        elif event == "after_backward":
            if self.pending_native_skip is not None:
                raise ResearchStop("Previous nonfinite gradient step has not been verified")
            try:
                self.finite_gradient_norm_overflows += check_backward(self.network, state["loss"])
            except ResearchStop as exc:
                evidence = backward_failure_evidence(self.network, self.accelerator, state["loss"], state["global_step"], exc)
                if self.gradient_handling != "verified_native_skip" or str(exc) != "Nonfinite training gradients":
                    write_json(self.output / "backward_failure_evidence.json", evidence)
                    raise
                check_updated_state(self.network, self.optimizer)
                scale = evidence["scaler_scale"]
                if not evidence["scaler_scale_is_finite"] or (scale is not None and scale <= 0):
                    raise ResearchStop("Invalid loss scaler before native skip")
                evidence.update(status="pending_native_decision", before_state_sha256=training_state_fingerprint(self.network, self.optimizer))
                self.pending_native_skip = evidence
                self._record_native_skip(evidence)
            quantized = state["global_step"] >= 680
            for module in self.network.text_encoder_loras + self.network.unet_loras:
                expected = self.mapping[module.lora_name]
                if module.delta_q_range_mul != expected["mul"] or bool(module.delta_q_enabled) != bool(quantized and expected["enabled"]):
                    raise ResearchStop("Effective training mul or quantization scope changed")
        elif event == "after_training_step":
            check_updated_state(self.network, self.optimizer)
            if self.pending_native_skip is not None:
                self._verify_native_skip(state)
            else:
                self.consecutive_nonfinite_skips = 0
            self.last_step = int(state["global_step"])
            self.guardian_skips += int(state["guardian_skipped"])
            self.amp_skips += int(bool(state["amp_skipped"]))
            self.actual_updates += int(not state["guardian_skipped"] and not state["amp_skipped"])
            row = {**state, "actual_updates": self.actual_updates, "guardian_skips": self.guardian_skips, "amp_skips": self.amp_skips,
                   "finite_gradient_norm_overflows": self.finite_gradient_norm_overflows,
                   "verified_nonfinite_skips": self.verified_nonfinite_skips,
                   "consecutive_nonfinite_skips": self.consecutive_nonfinite_skips,
                   "lr": [float(group["lr"]) for group in self.optimizer.param_groups], "updated_at": time.time()}
            with self.rows_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if self.last_step % 20 == 0 or self.last_step == 1:
                write_json(self.output / "progress.json", row)

    def _record_native_skip(self, evidence):
        with (self.output / "nonfinite_gradient_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, allow_nan=False) + "\n")

    def _verify_native_skip(self, state):
        evidence = dict(self.pending_native_skip)
        try:
            if int(state["global_step"]) != evidence["scheduled_step"]:
                raise ResearchStop("Nonfinite gradient step identity changed")
            if not math.isfinite(float(state["loss"])):
                raise ResearchStop("Nonfinite training loss after native decision")
            if not state["guardian_skipped"] and not state["amp_skipped"]:
                raise ResearchStop("Native trainer did not skip the nonfinite gradient update")
            after_hash = training_state_fingerprint(self.network, self.optimizer)
            if after_hash != evidence["before_state_sha256"]:
                raise ResearchStop("Parameters or optimizer state changed on a nonfinite gradient step")
            scaler = getattr(self.accelerator, "scaler", None)
            scale = float(scaler.get_scale()) if scaler is not None else None
            if scale is not None and (not math.isfinite(scale) or scale <= 0):
                raise ResearchStop("Invalid loss scaler after native skip")
            if not state["guardian_skipped"] and (scale is None or evidence["scaler_scale"] is None or scale >= evidence["scaler_scale"]):
                raise ResearchStop("AMP skip did not reduce its loss scale")
            self.verified_nonfinite_skips += 1
            self.consecutive_nonfinite_skips += 1
            evidence.update(status="verified_native_skip", guardian_skipped=bool(state["guardian_skipped"]),
                            amp_skipped=state["amp_skipped"], after_state_sha256=after_hash,
                            state_finite_and_unchanged=True, scaler_scale_after=scale,
                            consecutive_nonfinite_skips=self.consecutive_nonfinite_skips)
            self._record_native_skip(evidence)
            self.pending_native_skip = None
            if self.consecutive_nonfinite_skips >= 3:
                raise ResearchStop("Three consecutive nonfinite gradient skips")
        except ResearchStop as exc:
            write_json(self.output / "native_skip_failure.json", {**evidence, "verification_failure": str(exc)})
            raise

    def finish(self):
        if self.last_step != 13600:
            raise ResearchStop("Training returned before the full schedule completed")
        rows = index_checkpoints(self.output, definition=self.definition, expected_assignment=self.expected_assignment, complete=True)
        write_json(self.output / "training_complete.json", {"status": "complete", "scheduled_steps": self.last_step,
            "actual_updates": self.actual_updates, "guardian_skips": self.guardian_skips, "amp_skips": self.amp_skips,
            "gradient_handling": self.gradient_handling, "verified_nonfinite_skips": self.verified_nonfinite_skips,
            "checkpoint_count": len(rows), "seconds": time.time() - self.started_at, "policy_sha256": self.expected_assignment})
        return rows
