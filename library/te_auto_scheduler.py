import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch


logger = logging.getLogger(__name__)


# Number of buffered log entries before touching disk
LOG_FLUSH_FREQUENCY = 200


@dataclass
class TeScheduleConfig:
    """Configuration for a single Text Encoder schedule."""

    te_index: int
    name: str
    monitor_start_step: int = 0
    warmup_steps: int = 300
    baseline_beta: float = 0.9
    decay_threshold: float = 0.4
    freeze_threshold: float = 0.2
    decay_patience: int = 100
    freeze_patience: int = 100
    decay_factor: float = 0.5
    decay_max: int = 1
    min_baseline: float = 1e-6
    mode: str = "monitor"  # "monitor" or "freeze"


@dataclass
class TeScheduleState:
    baseline_ema: Optional[float] = None
    warmup_count: int = 0
    baseline_finalized: bool = False
    last_metric: float = 0.0
    last_score: float = 1.0
    patience_decay: int = 0
    patience_freeze: int = 0
    decay_applied: int = 0
    frozen: bool = False
    last_action: str = "init"


@dataclass
class TeLogEntry:
    step: int
    te_index: int
    te_name: str
    metric: float
    baseline: float
    score: float
    lr: float
    action: str
    warmup_remaining: int
    decay_counter: int
    freeze_counter: int
    decay_applied: int
    baseline_ready: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "te_index": self.te_index,
            "te_name": self.te_name,
            "metric": self.metric,
            "baseline": self.baseline,
            "score": self.score,
            "lr": self.lr,
            "action": self.action,
            "warmup_remaining": self.warmup_remaining,
            "decay_counter": self.decay_counter,
            "freeze_counter": self.freeze_counter,
            "decay_applied": self.decay_applied,
            "baseline_ready": self.baseline_ready,
        }

    @staticmethod
    def csv_header() -> Sequence[str]:
        return [
            "step",
            "te_index",
            "te_name",
            "metric",
            "baseline",
            "score",
            "lr",
            "action",
            "warmup_remaining",
            "decay_counter",
            "freeze_counter",
            "decay_applied",
            "baseline_ready",
        ]

    def to_csv_row(self) -> List[str]:
        data = self.to_dict()
        return [str(data[key]) for key in self.csv_header()]


class TeAutoScheduler:
    """Relative-warmup based controller for Text Encoder learning rate and freezing."""

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
        self.log_interval = max(0, log_interval)
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
        self._log_buffer: List[TeLogEntry] = []
        self._log_header_written = False
        self._last_flushed_step: Optional[int] = None
        self._log_disabled = False

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------
    def collect_metrics(self) -> Dict[int, float]:
        metrics: Dict[int, float] = {}
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

            if state.frozen:
                entry = self._emit_entry(step_index, te_idx, metric, "frozen")
                entries.append(entry)
                continue

            if step_index <= cfg.monitor_start_step:
                # Pre-monitor phase: optionally prime baseline but count as no action
                if state.baseline_ema is None:
                    state.baseline_ema = metric
                else:
                    state.baseline_ema = cfg.baseline_beta * state.baseline_ema + (1.0 - cfg.baseline_beta) * metric
                state.last_metric = metric
                state.last_score = 1.0
                state.last_action = "pre_monitor"
                entry = self._emit_entry(step_index, te_idx, metric, state.last_action)
                entries.append(entry)
                continue

            if state.warmup_count < cfg.warmup_steps:
                state.warmup_count += 1
                if state.baseline_ema is None:
                    state.baseline_ema = metric
                else:
                    state.baseline_ema = cfg.baseline_beta * state.baseline_ema + (1.0 - cfg.baseline_beta) * metric
                state.last_metric = metric
                state.last_score = 1.0
                state.last_action = "warmup"
                if state.warmup_count >= cfg.warmup_steps:
                    state.baseline_finalized = True
                    state.last_action = "baseline_finalized"
                entry = self._emit_entry(step_index, te_idx, metric, state.last_action)
                entries.append(entry)
                continue

            if not state.baseline_finalized:
                # Safeguard in case warmup_steps is 0
                state.baseline_finalized = True

            baseline = max(state.baseline_ema or 0.0, cfg.min_baseline)
            score = min(metric / baseline, 1.0) if baseline > 0 else 0.0

            state.last_metric = metric
            state.last_score = score

            action = "monitor"
            decay_triggered = False
            freeze_triggered = False

            if cfg.decay_patience <= 0:
                decay_condition_met = score < cfg.decay_threshold
                state.patience_decay = 0
            else:
                if score < cfg.decay_threshold:
                    state.patience_decay += 1
                else:
                    state.patience_decay = 0
                decay_condition_met = state.patience_decay >= cfg.decay_patience

            if cfg.mode == "freeze":
                if cfg.freeze_patience <= 0:
                    freeze_condition_met = score < cfg.freeze_threshold and state.baseline_ema >= cfg.min_baseline
                    state.patience_freeze = 0
                else:
                    if score < cfg.freeze_threshold and state.baseline_ema >= cfg.min_baseline:
                        state.patience_freeze += 1
                    else:
                        state.patience_freeze = 0
                    freeze_condition_met = state.patience_freeze >= cfg.freeze_patience
            else:
                freeze_condition_met = False

            if decay_condition_met and (cfg.decay_max < 0 or state.decay_applied < cfg.decay_max):
                decay_triggered = True

            if cfg.mode == "freeze":
                if freeze_condition_met:
                    freeze_triggered = True
                elif (cfg.decay_max >= 0 and state.decay_applied >= cfg.decay_max and decay_condition_met):
                    freeze_triggered = True

            if decay_triggered:
                if self._apply_decay(te_idx, cfg.decay_factor):
                    state.decay_applied += 1
                    state.patience_decay = 0
                    action = "decay"
                else:
                    state.last_action = "decay_skipped"
                    entry = self._emit_entry(step_index, te_idx, metric, state.last_action)
                    entries.append(entry)
                    continue

            if freeze_triggered and cfg.mode == "freeze":
                self._apply_freeze(te_idx)
                state.frozen = True
                state.last_action = "freeze"
                entry = self._emit_entry(step_index, te_idx, metric, state.last_action)
                entries.append(entry)
                continue

            state.last_action = action
            entry = self._emit_entry(step_index, te_idx, metric, action)
            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_decay(self, te_idx: int, factor: float) -> bool:
        if factor <= 0.0:
            logger.warning("TE%d decay factor <= 0, skipping lr decay", te_idx + 1)
            return False

        updated = False
        for group_idx in self.te_param_groups.get(te_idx, []):
            group = self.optimizer.param_groups[group_idx]
            old_lr = group.get("lr", 0.0)
            new_lr = old_lr * factor
            group["lr"] = new_lr
            updated = True

        if self.verbose and updated:
            logger.info("TE%d learning rate decayed by factor %.4f", te_idx + 1, factor)

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

    def _emit_entry(self, step_index: int, te_idx: int, metric: float, action: str) -> TeLogEntry:
        state = self.states[te_idx]
        cfg = self.configs[te_idx]

        baseline = state.baseline_ema or 0.0
        score = state.last_score
        warmup_remaining = max(cfg.warmup_steps - state.warmup_count, 0)

        lr_values = [self.optimizer.param_groups[idx].get("lr", 0.0) for idx in self.te_param_groups.get(te_idx, [])]
        lr = lr_values[0] if lr_values else 0.0

        entry = TeLogEntry(
            step=step_index,
            te_index=te_idx,
            te_name=self.configs[te_idx].name,
            metric=metric,
            baseline=baseline,
            score=score,
            lr=lr,
            action=action,
            warmup_remaining=warmup_remaining,
            decay_counter=self.states[te_idx].patience_decay,
            freeze_counter=self.states[te_idx].patience_freeze,
            decay_applied=self.states[te_idx].decay_applied,
            baseline_ready=self.states[te_idx].baseline_finalized,
        )

        self._log(entry)
        return entry

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, entry: TeLogEntry) -> None:
        if self._log_disabled:
            return

        if self.log_interval <= 0 and self.log_path is None and not self.verbose:
            return

        should_log = False
        if self.log_interval > 0 and entry.step % self.log_interval == 0:
            should_log = True
        if entry.action in {"decay", "freeze", "baseline_finalized"}:
            should_log = True
        if self.log_path is not None:
            should_log = should_log or True

        if not should_log:
            if self.verbose:
                logger.info(
                    "[TE%d] step=%d metric=%.4e score=%.3f lr=%.4e action=%s",
                    entry.te_index + 1,
                    entry.step,
                    entry.metric,
                    entry.score,
                    entry.lr,
                    entry.action,
                )
            return

        if self.verbose:
            logger.info(
                "[TE%d] step=%d metric=%.4e score=%.3f lr=%.4e action=%s",
                entry.te_index + 1,
                entry.step,
                entry.metric,
                entry.score,
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
                            f"{entry.step}\t{entry.te_index}\t{entry.te_name}\t{entry.metric:.6e}\t{entry.baseline:.6e}\t{entry.score:.4f}\t{entry.lr:.6e}\t{entry.action}\n"
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


def build_te_param_maps(
    network,
    te_selection_indices: Iterable[int],
) -> Dict[int, List[torch.nn.Parameter]]:
    """Utility to gather TE LoRA parameters per encoder index."""

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
    """Map optimizer param_group indices to TE indices."""

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
