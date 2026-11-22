"""Automatic range selection for LoRA delta quantization, once per collected training attempt."""

import math

import torch


# Bump the settings/metric version if any of these policies changes.
CLIP_BANDS = {
    "default": (0.0005, 0.003),
    "high": (0.003, 0.005),
    "high_narrow": (0.0038, 0.0048),
    "mid": (0.002, 0.004),
    "low": (0.0005, 0.0022),
}
EMA_BETA = 0.95
MUL_UP, MUL_DOWN = 1.01, 0.995
MIN_MUL, MAX_MUL = 1.0, 6.0
MIN_PROGRESS, FREEZE_PROGRESS = 0.25, 0.55
BAD_STREAK = 3
QERR_RATIO_THRESHOLD, QERR_PER_CLIP_THRESHOLD = 0.25, 130.0


def update_ema(previous, current):
    if current is None:
        return previous
    if previous is None:
        return current
    return previous * EMA_BETA + (1.0 - EMA_BETA) * current


def _clamp_mul(mul):
    return max(MIN_MUL, min(MAX_MUL, mul))


class LowBand:
    """One-way transition from the low to the mid clip band."""

    def __init__(self):
        self.state = "observe"
        self.phase = "pre_min_progress"
        self.escaped = False
        self.can_escape = False
        self.bad_streak = 0
        self.error_ratio_ema = None

    def decide(self, progress, qerr_per_clip):
        bad = None
        if self.error_ratio_ema is not None and qerr_per_clip is not None:
            bad = self.error_ratio_ema >= QERR_RATIO_THRESHOLD and qerr_per_clip >= QERR_PER_CLIP_THRESHOLD
        self.can_escape = MIN_PROGRESS <= progress < FREEZE_PROGRESS and not self.escaped
        if progress < MIN_PROGRESS:
            self.state, self.phase = "observe", "pre_min_progress"
            decision, reason = "observe", "min_progress"
            self.bad_streak = 0
        elif self.escaped:
            self.state, self.phase = "mid_lock", "escaped"
            decision, reason = "mid_lock", "escaped_once"
            self.bad_streak = 0
        elif progress >= FREEZE_PROGRESS:
            self.state, self.phase = "frozen", "frozen"
            decision, reason = "frozen", "freeze_progress"
            self.bad_streak = self.bad_streak + 1 if bad else 0
        elif bad is None:
            self.state, self.phase = "observe", "active"
            decision, reason = "observe", "insufficient_qerr_stats"
            self.bad_streak = 0
        else:
            self.phase = "active"
            self.bad_streak = self.bad_streak + 1 if bad else 0
            decision, reason = ("bad_count", "low_bad") if bad else ("keep_low", "in_band")
            if self.bad_streak >= BAD_STREAK:
                self.escaped = True
                self.can_escape = False
                self.state, self.phase = "escape_to_mid", "escaped"
                decision, reason = "escape_to_mid", "bad_streak_met"
        return {"low_auto_decision": decision, "low_auto_reason": reason, "low_auto_bad": bad,
                "low_auto_phase": self.phase, "low_auto_can_escape": self.can_escape}


class DQAutoController:
    """Consumes confirmed auto-scope measurements. Its current mul is used by the next forward."""

    def __init__(self, values):
        self.scope = values["dq_delta_auto_scope"]
        preset = values["dq_delta_auto_preset"] or "default"
        self.low = LowBand() if preset == "clip_rate_low_auto" else None
        self.band = "low" if self.low else preset.removeprefix("clip_rate_")
        self.use_raw = values["dq_delta_auto_use_raw"]
        self.qerr_floor = max(1e-12, values["dq_delta_qerr_per_clip_floor"])
        self.warmup_updates = 0
        if values["dq_delta_auto_warmup"]:
            self.warmup_updates = values["dq_delta_auto_warmup_updates"] or math.ceil(2.0 / (1.0 - EMA_BETA))
        self.warmup_remaining = self.warmup_updates
        self.in_band_streak = 0
        self.last_step = 0
        self.clip_rate_ema = None
        self.mul = values["dq_delta_range_mul"]
        target = None
        init_applied = values["dq_delta_auto_init_range_mul_from_band"]
        if init_applied:
            low, high = self.band_limits
            target = (low + high) / 2.0
            # Match the legacy float32 erfinv input on CPU, including rounding of the target.
            prob = 1.0 - target / 2.0
            inv = torch.erfinv(torch.tensor(2.0 * prob - 1.0, dtype=torch.float32, device="cpu")).item()
            self.mul = _clamp_mul(math.sqrt(2.0) * inv)
        self.settings = {
            "enabled": True, "scope": self.scope, "preset": preset, "ema_beta": EMA_BETA,
            "mul_up": MUL_UP, "mul_down": MUL_DOWN, "min_mul": MIN_MUL, "max_mul": MAX_MUL,
            "every": values["dq_delta_auto_every"], "use_raw": self.use_raw, "warmup_updates": self.warmup_updates,
            "qerr_per_clip_floor": self.qerr_floor,
            "init": {"applied": init_applied, "mul": self.mul if init_applied else None, "clip_target": target},
            "low_auto": {"enabled": self.low is not None, "min_progress": MIN_PROGRESS, "freeze_progress": FREEZE_PROGRESS,
                         "bad_streak_threshold": BAD_STREAK, "qerr_ratio_threshold": QERR_RATIO_THRESHOLD,
                         "qerr_per_clip_threshold": QERR_PER_CLIP_THRESHOLD},
        }
        self.init_row = self._row("init", 0, 0)
        self.last_row = self.init_row

    @property
    def band_limits(self):
        return CLIP_BANDS[self.band]

    def _row(self, event_type, step, progress):
        return {
            "event_type": event_type, "epoch": None, "attempt_step": step, "scope": self.scope, "target": None, "bits": None,
            "clip_rate_raw": None, "clip_rate_ema": self.clip_rate_ema, "common_mul_before": None, "common_mul_after": None,
            "auto_applied": None, "warmup_active": None, "warmup_remaining_after": self.warmup_remaining,
            "auto_reason": None, "qerr_per_clip": None, "active_clip_band": self.band, "active_clip_low": self.band_limits[0],
            "active_clip_high": self.band_limits[1], "low_auto_state": self.low.state if self.low else None,
            "low_auto_decision": None, "low_auto_reason": None, "low_auto_bad": None,
            "low_auto_bad_streak": self.low.bad_streak if self.low else None, "train_progress": progress,
            "low_auto_phase": self.low.phase if self.low else None, "low_auto_can_escape": self.low.can_escape if self.low else None,
            "auto_quant_error_ratio_raw": None, "auto_quant_error_ratio_ema": self.low.error_ratio_ema if self.low else None,
            "low_auto_phase_after": self.low.phase if self.low else None,
        }

    def _adjust(self, raw):
        low, high = self.band_limits
        if self.clip_rate_ema > high and (not self.use_raw or raw > high):
            self.mul = _clamp_mul(self.mul * MUL_UP)
            return "clip_high"
        if self.clip_rate_ema < low and (not self.use_raw or raw < low):
            self.mul = _clamp_mul(self.mul * MUL_DOWN)
            return "clip_low"
        self.mul = _clamp_mul(self.mul)
        return "in_band"

    def check(self, *, epoch, step, progress, bits, bits_changed, metrics):
        if step <= self.last_step:
            raise ValueError(f"auto attempt {step} already closed (last {self.last_step})")
        raw = metrics["clip_rate_raw"]
        error = metrics["quant_error_ratio_raw"]
        before = self.mul
        if raw is not None:
            if bits_changed:
                self.clip_rate_ema = None
                self.warmup_remaining = self.warmup_updates
                self.in_band_streak = 0
                if self.low:
                    self.low.error_ratio_ema = None
            self.clip_rate_ema = update_ema(self.clip_rate_ema, raw)
        warmup = self.warmup_remaining > 0
        reason = "warmup" if warmup else "in_band"
        low_result = {}
        qerr_per_clip = None
        if raw is not None:
            if self.low:
                self.low.error_ratio_ema = update_ema(self.low.error_ratio_ema, error)
                if self.low.error_ratio_ema is not None:
                    qerr_per_clip = self.low.error_ratio_ema / max(self.clip_rate_ema, self.qerr_floor)
            if warmup:
                low, high = self.band_limits
                inside = low <= self.clip_rate_ema <= high
                self.in_band_streak = self.in_band_streak + 1 if inside else 0
                self.warmup_remaining = 0 if self.in_band_streak >= 3 else self.warmup_remaining - 1
                if self.low:
                    # The warmup observation does not replace the low controller's persistent phase.
                    low_result = {"low_auto_phase": "warmup", "low_auto_can_escape": False,
                                  "low_auto_reason": "warmup", "low_auto_decision": "observe"}
            else:
                if self.low:
                    low_result = self.low.decide(progress, qerr_per_clip)
                    if self.low.escaped:
                        self.band = "mid"
                reason = self._adjust(raw)
        row = self._row("check", step, progress)
        row.update(epoch=epoch, target="delta", bits=bits, clip_rate_raw=raw,
                   common_mul_before=before, common_mul_after=self.mul, auto_applied=self.mul != before,
                   warmup_active=warmup, auto_reason=reason, qerr_per_clip=qerr_per_clip,
                   auto_quant_error_ratio_raw=error, **low_result)
        self.last_row = row
        self.last_step = step
        return row
