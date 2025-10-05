import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch


logger = logging.getLogger(__name__)


LOG_FLUSH_FREQUENCY = 200


@dataclass
class TeScheduleConfig:
    te_index: int
    name: str
    mode: str  # "monitor" or "freeze"
    ema_fast_alpha: float
    ema_slow_alpha: float
    plateau_ratio: float
    plateau_patience: int
    min_step: int
    decay_factor: float
    decay_limit: int  # -1 for unlimited
    freeze_ratio: float
    freeze_patience: int


@dataclass
class TeScheduleState:
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    last_metric: float = 0.0
    last_ratio: float = 1.0
    plateau_counter: int = 0
    freeze_counter: int = 0
    decay_count: int = 0
    frozen: bool = False
    last_action: str = "init"


@dataclass
class TeLogEntry:
    step: int
    te_index: int
    te_name: str
    metric: float
    ema_fast: float
    ema_slow: float
    ratio: float
    lr: float
    action: str
    plateau_counter: int
    decay_count: int
    freeze_counter: int
    plateau_ready: bool

    @staticmethod
    def csv_header() -> Sequence[str]:
        return [
            "step",
            "te_index",
            "te_name",
            "metric",
            "ema_fast",
            "ema_slow",
            "ratio",
            "lr",
            "action",
            "plateau_counter",
            "decay_count",
            "freeze_counter",
            "plateau_ready",
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "te_index": self.te_index,
            "te_name": self.te_name,
            "metric": self.metric,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ratio": self.ratio,
            "lr": self.lr,
            "action": self.action,
            "plateau_counter": self.plateau_counter,
            "decay_count": self.decay_count,
            "freeze_counter": self.freeze_counter,
            "plateau_ready": self.plateau_ready,
        }

    def to_csv_row(self) -> List[str]:
        data = self.to_dict()
        return [str(data[key]) for key in self.csv_header()]


class TeAutoScheduler:
    """EMA-based plateau controller for Text Encoder LoRA training."""

    def __init__(
        self,
        configs: Sequence[TeScheduleConfig],
        optimizer,
        te_param_groups: Dict[int, Sequence[int]],
        te_parameters: Dict[int, Sequence[torch.nn.Parameter]],
        log_interval: int = 0,
        log_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.configs: Dict[int, TeScheduleConfig] = {cfg.te_index: cfg for cfg in configs}
        self.states: Dict[int, TeScheduleState] = {cfg.te_index: TeScheduleState() for cfg in configs}
        self.optimizer = optimizer
        self.te_param_groups = {idx: list(group_indices) for idx, group_indices in te_param_groups.items()}
        self.te_parameters = {idx: list(params) for idx, params in te_parameters.items()}
        self.verbose = verbose

        self.log_path: Optional[Path] = Path(log_path).expanduser() if log_path else None
        self.log_format: Optional[str] = None
        if self.log_path is not None:
            suffix = self.log_path.suffix.lower()
            if suffix == ".jsonl":
                self.log_format = "jsonl"
            else:
                self.log_format = "csv"
            if self.log_path.parent and not self.log_path.parent.exists():
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_interval = max(0, log_interval)
        self._log_buffer: List[TeLogEntry] = []
        self._log_header_written = False
        self._last_flushed_step: Optional[int] = None
        self._log_disabled = False

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------
    def collect_metrics(self, grad_scale: float = 1.0) -> Dict[int, float]:
        metrics: Dict[int, float] = {}
        inv_scale = 1.0
        grad_scale_f: Optional[float] = None
        if grad_scale is not None:
            try:
                grad_scale_f = float(grad_scale)
            except (TypeError, ValueError):
                grad_scale_f = None
        if grad_scale_f is not None and grad_scale_f not in (0.0, 1.0) and math.isfinite(grad_scale_f):
            inv_scale = 1.0 / grad_scale_f
        for te_idx, params in self.te_parameters.items():
            state = self.states.get(te_idx)
            if state is None or state.frozen:
                continue

            total = 0.0
            has_grad = False
            for param in params:
                grad = param.grad
                if grad is None:
                    continue
                has_grad = True
                grad_fp32 = grad.detach().float()
                if inv_scale != 1.0:
                    grad_fp32 = grad_fp32 * inv_scale
                if not torch.isfinite(grad_fp32).all():
                    grad_fp32 = torch.nan_to_num(grad_fp32, nan=0.0, posinf=0.0, neginf=0.0)
                total += grad_fp32.pow(2).sum().item()

            metric = math.sqrt(total) if has_grad and total > 0.0 else 0.0
            if not math.isfinite(metric):
                metric = 0.0
            metrics[te_idx] = metric
        return metrics

    # ------------------------------------------------------------------
    # Update per step
    # ------------------------------------------------------------------
    def step(self, step_index: int, metrics: Dict[int, float]) -> List[TeLogEntry]:
        entries: List[TeLogEntry] = []
        for te_idx, state in self.states.items():
            cfg = self.configs[te_idx]
            metric = metrics.get(te_idx, 0.0)
            state.last_metric = metric

            if state.frozen:
                entry = self._emit_entry(step_index, te_idx, metric, state.last_action)
                entries.append(entry)
                continue

            state.ema_fast = _update_ema(state.ema_fast, metric, cfg.ema_fast_alpha)
            state.ema_slow = _update_ema(state.ema_slow, metric, cfg.ema_slow_alpha)

            if state.ema_slow and state.ema_slow > 0:
                ratio = state.ema_fast / state.ema_slow
            else:
                ratio = 1.0
            if not math.isfinite(ratio):
                ratio = 0.0
            state.last_ratio = ratio

            plateau_ready = step_index >= cfg.min_step
            action = "monitor"

            if plateau_ready:
                if ratio < cfg.plateau_ratio:
                    state.plateau_counter += 1
                else:
                    state.plateau_counter = 0

                if cfg.mode == "freeze":
                    if ratio < cfg.freeze_ratio:
                        state.freeze_counter += 1
                    else:
                        state.freeze_counter = 0

                if state.plateau_counter >= cfg.plateau_patience:
                    if cfg.decay_limit < 0 or state.decay_count < cfg.decay_limit:
                        if self._apply_decay(te_idx, cfg.decay_factor):
                            state.decay_count += 1
                            state.plateau_counter = 0
                            state.freeze_counter = 0
                            action = "decay"
                        else:
                            action = "decay_skipped"
                    elif cfg.mode == "freeze":
                        self._apply_freeze(te_idx)
                        state.frozen = True
                        state.plateau_counter = 0
                        action = "freeze"
                elif (
                    cfg.mode == "freeze"
                    and cfg.decay_limit >= 0
                    and state.decay_count >= cfg.decay_limit
                    and state.freeze_counter >= cfg.freeze_patience
                ):
                    self._apply_freeze(te_idx)
                    state.frozen = True
                    state.freeze_counter = 0
                    action = "freeze"
            else:
                state.plateau_counter = 0
                state.freeze_counter = 0

            state.last_action = action
            entry = self._emit_entry(step_index, te_idx, metric, action, plateau_ready=plateau_ready)
            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_decay(self, te_idx: int, factor: float) -> bool:
        if factor <= 0.0:
            logger.warning("TE%d decay factor <= 0, skipping", te_idx + 1)
            return False

        updated = False
        for group_idx in self.te_param_groups.get(te_idx, []):
            group = self.optimizer.param_groups[group_idx]
            old_lr = group.get("lr", 0.0)
            new_lr = old_lr * factor
            group["lr"] = new_lr
            updated = True

        if self.verbose and updated:
            logger.info("TE%d learning rate decayed by %.4f", te_idx + 1, factor)

        return updated

    def _apply_freeze(self, te_idx: int) -> None:
        for param in self.te_parameters.get(te_idx, []):
            if param.requires_grad:
                param.requires_grad_(False)
            if param.grad is not None:
                param.grad = None

        for group_idx in self.te_param_groups.get(te_idx, []):
            group = self.optimizer.param_groups[group_idx]
            group["lr"] = 0.0

        if self.verbose:
            logger.info("TE%d training frozen", te_idx + 1)

    def _emit_entry(
        self,
        step_index: int,
        te_idx: int,
        metric: float,
        action: str,
        plateau_ready: bool = True,
    ) -> TeLogEntry:
        state = self.states[te_idx]
        cfg = self.configs[te_idx]

        ema_fast = state.ema_fast or 0.0
        ema_slow = state.ema_slow or 0.0
        ratio = state.last_ratio
        lr_values = [self.optimizer.param_groups[idx].get("lr", 0.0) for idx in self.te_param_groups.get(te_idx, [])]
        lr = lr_values[0] if lr_values else 0.0

        entry = TeLogEntry(
            step=step_index,
            te_index=te_idx,
            te_name=cfg.name,
            metric=metric,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ratio=ratio,
            lr=lr,
            action=action,
            plateau_counter=state.plateau_counter,
            decay_count=state.decay_count,
            freeze_counter=state.freeze_counter,
            plateau_ready=plateau_ready,
        )

        self._log(entry)
        return entry

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, entry: TeLogEntry) -> None:
        if self._log_disabled:
            return

        if self.log_path is None and not self.verbose:
            return

        should_log = False
        if self.log_interval > 0 and entry.step % self.log_interval == 0:
            should_log = True
        if entry.action in {"decay", "freeze"}:
            should_log = True
        if self.log_path is not None:
            should_log = True

        if self.verbose:
            logger.info(
                "[TE%d] step=%d metric=%.4e ratio=%.3f lr=%.4e action=%s",
                entry.te_index + 1,
                entry.step,
                entry.metric,
                entry.ratio,
                entry.lr,
                entry.action,
            )

        if self.log_path is not None:
            self._log_buffer.append(entry)
            if entry.step % LOG_FLUSH_FREQUENCY == 0 and entry.step != 0:
                if self._last_flushed_step != entry.step:
                    self.flush(force=True)

    def flush(self, force: bool = False) -> None:
        if self._log_disabled or self.log_path is None:
            return

        if not self._log_buffer:
            return

        if not force and len(self._log_buffer) < LOG_FLUSH_FREQUENCY:
            return

        last_step = self._log_buffer[-1].step
        mode = "a" if self.log_path.exists() or self._log_header_written else "w"
        try:
            with self.log_path.open(mode, encoding="utf-8", newline="") as fp:
                if self.log_format == "csv":
                    writer = csv.writer(fp)
                    if not self._log_header_written and mode == "w":
                        writer.writerow(TeLogEntry.csv_header())
                    for entry in self._log_buffer:
                        writer.writerow(entry.to_csv_row())
                    self._log_header_written = True
                elif self.log_format == "jsonl":
                    for entry in self._log_buffer:
                        fp.write(json.dumps(entry.to_dict()) + "\n")
                    self._log_header_written = True
                else:
                    for entry in self._log_buffer:
                        fp.write(
                            f"{entry.step}\t{entry.te_index}\t{entry.te_name}\t{entry.metric:.6e}\t{entry.ema_fast:.6e}\t{entry.ema_slow:.6e}\t{entry.ratio:.4f}\t{entry.lr:.6e}\t{entry.action}\n"
                        )
                    self._log_header_written = True
        except OSError as err:
            logger.warning(
                "TE monitor log write failed (%s). Disabling further file logging.",
                err,
            )
            self._log_disabled = True
            self._log_buffer.clear()
            return

        self._log_buffer.clear()
        self._last_flushed_step = last_step


def _update_ema(prev: Optional[float], value: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    if prev is None:
        return value
    return prev * (1.0 - alpha) + value * alpha


def build_te_param_maps(
    network,
    te_selection_indices: Iterable[int],
) -> Dict[int, List[torch.nn.Parameter]]:
    te_params: Dict[int, List[torch.nn.Parameter]] = {}
    attr_name = "_text_encoder_loras_by_encoder"
    if not hasattr(network, attr_name):
        return te_params

    lora_groups = getattr(network, attr_name)
    for idx, group in enumerate(lora_groups):
        if idx not in te_selection_indices:
            continue
        if not group:
            continue
        params: List[torch.nn.Parameter] = []
        for lora_module in group:
            params.extend(list(lora_module.parameters()))
        if params:
            te_params[idx] = params
    return te_params


def build_optimizer_te_group_map(optimizer, te_parameters: Dict[int, Sequence[torch.nn.Parameter]]) -> Dict[int, List[int]]:
    param_to_te: Dict[int, int] = {}
    for te_idx, params in te_parameters.items():
        for param in params:
            param_to_te[id(param)] = te_idx

    mapping: Dict[int, List[int]] = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        te_ids = set()
        for param in group.get("params", []):
            te_id = param_to_te.get(id(param))
            if te_id is not None:
                te_ids.add(te_id)
        if not te_ids:
            continue
        if len(te_ids) > 1:
            logger.warning("optimizer param_group %d spans multiple TE indices: %s", group_idx, sorted(te_ids))
        te_id = sorted(te_ids)[0]
        mapping.setdefault(te_id, []).append(group_idx)
    return mapping
