from __future__ import annotations

import bisect
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _parse_te_index(description: str) -> Optional[int]:
    """Return zero-based Text Encoder index from lr description."""

    if not description:
        return None

    head = description.split()[0]
    if not head.startswith("textencoder"):
        return None

    suffix = head[len("textencoder") :]
    if not suffix:
        return 0

    if not suffix.isdigit():
        return None

    return int(suffix) - 1


class TrimmedMaxQueue:
    def __init__(self, maxlen: int, trim_ratio: float = 0.0) -> None:
        self.maxlen = maxlen
        self.trim_ratio = max(0.0, min(trim_ratio, 0.5))
        self._data: deque[float] = deque()
        self._sorted: List[float] = []

    def push(self, value: float) -> None:
        self._data.append(value)
        bisect.insort(self._sorted, float(value))
        if len(self._data) > self.maxlen:
            removed = self._data.popleft()
            idx = bisect.bisect_left(self._sorted, removed)
            if idx < len(self._sorted):
                self._sorted.pop(idx)

    def max(self) -> float:
        if not self._sorted:
            return 0.0
        if self.trim_ratio <= 0.0:
            return self._sorted[-1]
        trim_count = int(len(self._sorted) * self.trim_ratio)
        if trim_count <= 0:
            return self._sorted[-1]
        index = len(self._sorted) - trim_count - 1
        if index < 0:
            index = 0
        return self._sorted[index]


@dataclass
class TeThresholds:
    drop_threshold: float
    spread_limit: float
    trend_limit: float
    global_drop: float
    freeze_local: float
    freeze_global: float


@dataclass
class TePlateauConfig:
    local_window: int
    peak_window: int
    global_window: int
    peak_trim_ratio: float
    local_patience: int
    global_patience: int
    decay_mult: float
    freeze_patience: int
    ignore_steps: int
    cooldown: int
    thresholds: Dict[int, TeThresholds]
    log_path: Optional[str]
    log_interval: int
    global_alpha: float = 0.005
    flag_expire: Optional[int] = None
    flag_thaw_ratio: float = 0.9


@dataclass
class TePlateauState:
    te_id: int
    name: str
    parameters: List[torch.nn.Parameter]
    param_group_indices: List[int]
    param_group_lrs: Dict[int, float]
    state: str = "active"
    local_values: deque = field(default_factory=deque)
    local_peak: TrimmedMaxQueue = field(init=False)
    global_peak: TrimmedMaxQueue = field(init=False)
    global_ema: Optional[float] = None
    global_counter: int = 0
    local_counter: int = 0
    freeze_counter: int = 0
    flag_active: bool = False
    flag_step: int = -1
    flag_peak: float = 0.0
    cooldown: int = 0
    last_valid_grad: float = 0.0

    def initialise_queues(self, config: TePlateauConfig) -> None:
        self.local_values = deque(maxlen=config.local_window)
        self.local_peak = TrimmedMaxQueue(config.peak_window, config.peak_trim_ratio)
        self.global_peak = TrimmedMaxQueue(config.global_window, config.peak_trim_ratio)

    def current_lr(self) -> float:
        if not self.param_group_lrs:
            return 0.0
        return max(self.param_group_lrs.values())


class TePlateauController:
    def __init__(
        self,
        config: TePlateauConfig,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        lr_descriptions: Optional[Sequence[str]],
        te_indices: Sequence[int],
        network,
        logger,
        is_main_process: bool,
    ) -> None:
        self.config = config
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.is_main_process = is_main_process
        self.device = next(network.parameters()).device if list(network.parameters()) else torch.device("cpu")
        self.states: Dict[int, TePlateauState] = {}
        self.active = False
        self.log_buffer: List[str] = []
        self.last_log_flush_step: Optional[int] = None
        self.log_header_written = False

        self._build_states(network, lr_descriptions, te_indices)
        if self.states:
            self.active = True

    def _build_states(self, network, lr_descriptions, te_indices: Sequence[int]) -> None:
        if lr_descriptions is None:
            return

        group_map: Dict[int, List[int]] = {}
        for idx, desc in enumerate(lr_descriptions):
            te_idx = _parse_te_index(desc)
            if te_idx is None:
                continue
            if te_indices and te_idx not in te_indices:
                continue
            group_map.setdefault(te_idx, []).append(idx)

        if not group_map:
            return

        for te_idx, group_indices in group_map.items():
            params: List[torch.nn.Parameter] = []
            seen: set[int] = set()
            param_group_lrs: Dict[int, float] = {}
            for group_index in group_indices:
                param_group = self.optimizer.param_groups[group_index]
                param_group_lrs[group_index] = float(param_group.get("lr", 0.0))
                for param in param_group["params"]:
                    pid = id(param)
                    if pid in seen:
                        continue
                    params.append(param)
                    seen.add(pid)

            if not params:
                continue

            thresholds = self.config.thresholds.get(te_idx)
            if thresholds is None:
                continue

            state = TePlateauState(
                te_id=te_idx,
                name=f"TE{te_idx + 1}",
                parameters=params,
                param_group_indices=group_indices,
                param_group_lrs=param_group_lrs,
            )
            state.initialise_queues(self.config)
            self.states[te_idx] = state

    def is_active(self) -> bool:
        return self.active

    def update(self, step: int, grad_scale: float) -> List[Dict[str, str]]:
        if not self.active:
            return []

        events: List[Dict[str, str]] = []
        inv_scale = 1.0
        if grad_scale != 0.0 and math.isfinite(grad_scale):
            inv_scale = 1.0 / grad_scale

        for te_idx, state in self.states.items():
            thresholds = self.config.thresholds[te_idx]
            event = self._update_state(state, thresholds, step, inv_scale)
            if event is not None:
                events.append(event)

        if self.config.log_path and self.is_main_process:
            if self.config.log_interval <= 1 or (step % self.config.log_interval) == 0:
                self.flush_logs()

        return events

    def _update_state(
        self,
        state: TePlateauState,
        thresholds: TeThresholds,
        step: int,
        inv_scale: float,
    ) -> Optional[Dict[str, str]]:
        grad_norm = self._compute_grad_norm(state, inv_scale)
        if grad_norm is None:
            return None

        state.local_values.append(grad_norm)
        state.local_peak.push(grad_norm)

        median_local, spread_local, trend_ratio = self._compute_local_statistics(state)
        peak_recent = max(state.local_peak.max(), 1e-12)
        drop_ratio_local = median_local / peak_recent if peak_recent > 0 else 1.0

        # update global ema/peak trackers
        if state.global_ema is None:
            state.global_ema = grad_norm
        else:
            alpha = self.config.global_alpha
            state.global_ema = (1 - alpha) * state.global_ema + alpha * grad_norm

        state.global_peak.push(state.global_ema)
        global_peak = max(state.global_peak.max(), 1e-12)
        drop_ratio_global = state.global_ema / global_peak if global_peak > 0 else 1.0

        evaluation_enabled = step >= self.config.ignore_steps and state.cooldown <= 0

        if evaluation_enabled and state.state == "active":
            if drop_ratio_global <= thresholds.global_drop:
                state.global_counter += 1
            else:
                state.global_counter = 0

            if not state.flag_active and state.global_counter >= self.config.global_patience:
                state.flag_active = True
                state.flag_step = step
                state.flag_peak = global_peak

        # flag expiration check / reset when global ema recovers sufficiently
        if state.flag_active:
            expire_after = self.config.flag_expire or self.config.global_window
            should_reset = False
            if expire_after and step - state.flag_step >= expire_after:
                should_reset = True
            if not should_reset and state.global_ema >= state.flag_peak * self.config.flag_thaw_ratio:
                should_reset = True
            if should_reset:
                state.flag_active = False
                state.global_counter = 0

        event_type: Optional[str] = None

        if not evaluation_enabled:
            state.local_counter = 0
            state.freeze_counter = 0
        elif state.state == "active":
            if (
                drop_ratio_local <= thresholds.drop_threshold
                and spread_local <= thresholds.spread_limit * peak_recent
                and trend_ratio <= thresholds.trend_limit
            ):
                state.local_counter += 1
            else:
                state.local_counter = 0

            if state.flag_active and state.local_counter >= self.config.local_patience:
                self._apply_decay(state)
                event_type = "decay"
        elif state.state == "decayed":
            if (
                drop_ratio_local <= thresholds.freeze_local
                and drop_ratio_global <= thresholds.freeze_global
            ):
                state.freeze_counter += 1
            else:
                state.freeze_counter = 0

            if state.freeze_counter >= self.config.freeze_patience:
                self._apply_freeze(state)
                event_type = "freeze"


        if state.cooldown > 0:
            state.cooldown -= 1

        log_state = state.state
        if self.config.log_path and self.is_main_process:
            record = self._format_log_record(
                step,
                state.name,
                grad_norm,
                median_local,
                spread_local,
                trend_ratio,
                drop_ratio_local,
                drop_ratio_global,
                log_state,
                state.current_lr(),
                event_type or "",
            )
            self.log_buffer.append(record)

        if event_type is None:
            return None

        return {
            "type": event_type,
            "te": state.name,
            "lr": f"{state.current_lr():.6g}",
            "step": step,
        }

    def _compute_grad_norm(self, state: TePlateauState, inv_scale: float) -> Optional[float]:
        sum_sq = 0.0
        has_grad = False
        for param in state.parameters:
            grad = param.grad
            if grad is None:
                continue
            value = grad.detach()
            if inv_scale != 1.0:
                value = value * inv_scale
            sq = (value.float() * value.float()).sum().item()
            if not math.isfinite(sq):
                continue
            sum_sq += sq
            has_grad = True

        if not has_grad:
            grad_norm = state.last_valid_grad
        else:
            grad_norm = math.sqrt(sum_sq) if sum_sq > 0 else 0.0

        if not math.isfinite(grad_norm):
            grad_norm = state.last_valid_grad

        state.last_valid_grad = grad_norm
        return grad_norm

    def _compute_local_statistics(self, state: TePlateauState) -> Tuple[float, float, float]:
        if not state.local_values:
            return state.last_valid_grad, 0.0, float("inf")

        values = np.array(state.local_values, dtype=np.float32)
        median = float(np.median(values))
        if values.size == 1:
            return median, 0.0, 0.0

        pct90 = float(np.percentile(values, 90))
        pct10 = float(np.percentile(values, 10))
        spread = pct90 - pct10

        x = np.arange(values.size, dtype=np.float32)
        x_mean = float(x.mean())
        y_mean = float(values.mean())
        cov = float(np.dot(x - x_mean, values - y_mean))
        var = float(np.dot(x - x_mean, x - x_mean))
        slope = cov / var if var > 0 else 0.0
        peak_recent = max(state.local_peak.max(), 1e-12)
        trend_ratio = abs(slope) / peak_recent

        return median, spread, trend_ratio

    def _apply_decay(self, state: TePlateauState) -> None:
        for group_idx in state.param_group_indices:
            group = self.optimizer.param_groups[group_idx]
            new_lr = float(group.get("lr", 0.0)) * self.config.decay_mult
            group["lr"] = new_lr
            state.param_group_lrs[group_idx] = new_lr
            self._update_scheduler_lr(group_idx, new_lr)

        state.state = "decayed"
        state.flag_active = False
        state.local_counter = 0
        state.global_counter = 0
        state.cooldown = self.config.cooldown

    def _apply_freeze(self, state: TePlateauState) -> None:
        for group_idx in state.param_group_indices:
            group = self.optimizer.param_groups[group_idx]
            group["lr"] = 0.0
            self._update_scheduler_lr(group_idx, 0.0)
            state.param_group_lrs[group_idx] = 0.0
        for param in state.parameters:
            param.requires_grad_(False)
            param.grad = None

        state.state = "frozen"
        state.cooldown = self.config.cooldown
        state.freeze_counter = 0

    def _update_scheduler_lr(self, group_idx: int, new_lr: float) -> None:
        scheduler = self.lr_scheduler
        if hasattr(scheduler, "base_lrs") and len(scheduler.base_lrs) > group_idx:
            scheduler.base_lrs[group_idx] = new_lr
        if hasattr(scheduler, "_last_lr") and len(scheduler._last_lr) > group_idx:
            scheduler._last_lr[group_idx] = new_lr

    def _format_log_record(
        self,
        step: int,
        name: str,
        grad_norm: float,
        median: float,
        spread: float,
        trend: float,
        drop_local: float,
        drop_global: float,
        state: str,
        lr_value: float,
        event: str,
    ) -> str:
        return (
            f"{step},{name},{grad_norm:.6e},{median:.6e},{spread:.6e},{trend:.6e},"
            f"{drop_local:.6e},{drop_global:.6e},{state},{lr_value:.6e},{event}"
        )

    def flush_logs(self) -> None:
        log_path = self.config.log_path
        if not (log_path and self.log_buffer):
            return

        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not self.log_header_written:
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                self.log_header_written = True
        with open(log_path, "a", encoding="utf-8", newline="") as fp:
            if not self.log_header_written or fp.tell() == 0:
                fp.write(
                    "step,te_name,grad_norm,median_local,spread_local,trend_ratio,"
                    "drop_ratio_local,drop_ratio_global,state,lr,event\n"
                )
                self.log_header_written = True
            for line in self.log_buffer:
                fp.write(line + "\n")
        self.log_buffer.clear()

    def finalize(self) -> None:
        if self.config.log_path and self.log_buffer:
            self.flush_logs()
