"""Per-run, per-attempt training state for SDXL LoRA delta quantization."""
from dataclasses import dataclass
from functools import partial
import math

import torch

from library.dq_auto import DQAutoController, update_ema
from library.dq_mul_policy import MulPolicy, scope_includes
from library.rounding_util import quantize_delta


def _parts(scope):
    return ("unet", "te") if scope == "both" else (scope,)


@dataclass
class DQStepResult:
    attempt_step: int
    train_progress: float
    bits: int | None
    bits_changed_since_auto: bool
    auto_metrics: dict | None
    log_rows: list[dict]
    auto_row: dict | None


class DQTraining:
    """A controller is shared by the attached, in-scope training adapters of one run."""

    def __init__(self, config, network, *, total_steps, lr_warmup_steps, num_processes):
        self.values = config.values
        self.total_steps = total_steps
        self.schedule = config.bits_schedule
        self.bits = self.values["dq_delta_bits"]
        self.bits_changed_since_auto = False
        # The upstream pre-training sample runs before the first scheduled attempt.
        self.active_bits = self.bits
        self.triton_quantizer = None
        if self.values["dq_delta_use_triton"]:
            from library.dq_triton import quantize_delta_triton

            self.triton_quantizer = quantize_delta_triton
        elif self.values["dq_delta_triton_stats"]:
            import logging

            logging.getLogger(__name__).warning("--dq_delta_triton_stats needs --dq_delta_use_triton; using Torch statistics")
        self.fused_scopes = frozenset()
        self.packed_sinks = {}
        self.begin = self.values["dq_delta_begin"]
        self.begin_step = None
        if self.values["dq_delta_begin_after_lr_warmup"]:
            self.begin_step = (int(lr_warmup_steps * total_steps * num_processes)
                               if isinstance(lr_warmup_steps, float) else int(lr_warmup_steps))

        policy = MulPolicy.from_values(self.values)
        modules = [*network.unet_loras, *network.text_encoder_loras]
        assignments = policy.expand(module.lora_name for module in modules)
        fixed_overrides = any(value is not None for value in (policy.attn2, policy.te, policy.te1, policy.te2))
        self.assignment_record = policy.record(assignments) if fixed_overrides else None
        self.auto_controller = DQAutoController(self.values) if self.values["dq_delta_auto_range_mul"] else None
        self.common_mul = self.auto_controller.mul if self.auto_controller else policy.base
        self.log_scope = self.values["dq_delta_log_scope"]
        self.auto_scope = self.values["dq_delta_auto_scope"]
        for module in modules:
            row = assignments[module.lora_name]
            if scope_includes(self.values["dq_delta_scope"], row["component"]):
                part = "unet" if row["component"] == "unet" else "te"
                module.delta_quantizer = partial(self._quantize, part, row["mul"] if fixed_overrides else None)

        self.attempt = None
        self.progress = 0.0
        self.do_log = self.do_auto = self.full_stats = False
        self.totals = {}
        self.observers = {}
        self.error_rms_ema = self.error_ratio_ema = None
        self.last_result = None

    def begin_attempt(self, completed_steps):
        if self.attempt is not None:
            raise RuntimeError(f"DQ attempt {self.attempt} has not finished")
        self.attempt = completed_steps + 1
        self.last_result = None
        self.progress = completed_steps / self.total_steps
        started = completed_steps >= self.begin_step if self.begin_step is not None else self.progress >= self.begin
        bits = self.bits
        for threshold, scheduled_bits in self.schedule:
            if self.progress < threshold:
                break
            bits = scheduled_bits
        self.bits_changed_since_auto |= bits != self.bits
        self.bits = bits
        self.active_bits = bits if started else None

        v = self.values
        self.do_log = started and v["dq_delta_log"] and self.attempt % v["dq_delta_log_every"] == 0
        self.do_auto = started and v["dq_delta_auto_range_mul"] and self.attempt % v["dq_delta_auto_every"] == 0
        self.full_stats = self.do_log or (self.do_auto and v["dq_delta_auto_preset"] == "clip_rate_low_auto")
        scopes = set(_parts(self.log_scope)) if self.do_log else set()
        if self.do_auto:
            scopes.update(_parts(self.auto_scope))
        self.observers = {scope: partial(self._observe, scope) for scope in scopes}
        self.fused_scopes = frozenset(scopes) if self.full_stats and v["dq_delta_triton_stats"] else frozenset()
        self.packed_sinks = {scope:partial(self.add_packed,scope) for scope in self.fused_scopes}

    def _quantize(self, scope, fixed_mul, delta):
        if self.active_bits is None:
            return delta
        mul = self.common_mul if fixed_mul is None else fixed_mul
        if self.triton_quantizer is not None:
            return self.triton_quantizer(
                delta, bits=self.active_bits, range_mul=mul, observe=self.observers.get(scope),
                fused=scope in self.fused_scopes, collect_packed=self.packed_sinks.get(scope),
            )
        return quantize_delta(delta, bits=self.active_bits, range_mul=mul, observe=self.observers.get(scope))

    def _observe(self, scope, delta, output, normalized, scale, qmax):
        # GPU-only reductions here. Read numeric summaries at the scheduled attempt boundary.
        values = [normalized.new_tensor(delta.numel()), (normalized.abs() >= qmax).sum(dtype=torch.float32)]
        if self.full_stats:
            x = delta.float().reshape(-1)
            q = output.float().reshape(-1)
            values.extend((torch.dot(x, x), torch.dot(q, q), torch.dot(x, q)))
        self.add_packed(scope, torch.stack(values))

    def add_packed(self, scope, packed):
        if scope in self.totals:
            self.totals[scope].add_(packed)
        else:
            self.totals[scope] = packed

    def _metrics(self, scope):
        result = dict(numel=0, clip_rate_raw=None, input_rms=None, quant_error_rms_raw=None, quant_error_ratio_raw=None)
        sums = [self.totals[part] for part in _parts(scope) if part in self.totals]
        if not sums:
            return result
        total = sums[0] if len(sums) == 1 else sums[0] + sums[1]
        n = total[0]
        divisor = n.clamp_min(1)
        metrics = [n, total[1] / divisor]
        if len(total) == 5:
            metrics.extend([total[2] / divisor, (total[2] + total[3] - 2 * total[4]).clamp_min(0) / divisor])
        values = torch.stack(metrics).tolist()
        if values[0] > 0:
            result.update(numel=int(values[0]), clip_rate_raw=values[1])
            if len(values) == 4:
                rms, error = math.sqrt(values[2]), math.sqrt(values[3])
                result.update(input_rms=rms, quant_error_rms_raw=error, quant_error_ratio_raw=error / (rms + 1e-12))
        return result

    def finish_attempt(self, step, *, epoch, guardian_skipped):
        if step != self.attempt or self.attempt is None:
            raise RuntimeError(f"DQ attempt mismatch: completing {step}, open {self.attempt}")
        self.attempt = None
        self.observers.clear()  # Later validation and candidate scoring cannot append to this attempt.
        self.packed_sinks.clear()
        if guardian_skipped:
            self.fused_scopes = frozenset()
        self.last_result = None
        if guardian_skipped or not (self.do_log or self.do_auto):
            self.totals.clear()
            return None

        # Read each scheduled scope only once if log and controller observations coincide.
        measured = {}

        def metrics_for(scope):
            if scope not in measured:
                measured[scope] = self._metrics(scope)
            return measured[scope]

        auto = self.auto_controller
        auto_metrics = metrics_for(self.auto_scope) if self.do_auto else None
        changed = self.bits_changed_since_auto
        auto_row = None
        if auto_metrics is not None:
            auto_row = auto.check(epoch=epoch, step=step, progress=self.progress, bits=self.active_bits,
                                  bits_changed=changed, metrics=auto_metrics)
            self.common_mul = auto.mul
            if auto_metrics["clip_rate_raw"] is not None:
                self.bits_changed_since_auto = False

        rows = []
        if self.do_log:
            merged = metrics_for(self.log_scope)
            self.error_rms_ema = update_ema(self.error_rms_ema, merged["quant_error_rms_raw"])
            self.error_ratio_ema = update_ema(self.error_ratio_ema, merged["quant_error_ratio_raw"])
            for scope in _parts(self.log_scope):
                metrics = metrics_for(scope)
                rows.append({
                    "epoch": epoch, "attempt_step": step, "scope": scope, "target": "delta", "bits": self.active_bits,
                    "train_progress": self.progress, "common_mul_after": self.common_mul,
                    "log_scope_error_rms_ema": self.error_rms_ema, "log_scope_error_ratio_ema": self.error_ratio_ema,
                    "active_clip_band": auto.band if auto else "custom",
                    "active_clip_low": auto.band_limits[0] if auto else 0.0005,
                    "active_clip_high": auto.band_limits[1] if auto else 0.003,
                    "controller_clip_rate_ema": auto.clip_rate_ema if auto else None,
                    "auto_state_step": auto.last_step if auto else None,
                    **metrics,
                })
        self.totals.clear()
        self.last_result = DQStepResult(step, self.progress, self.active_bits, changed, auto_metrics, rows, auto_row)
        return self.last_result
