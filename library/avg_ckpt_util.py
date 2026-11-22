"""CPU LoRA snapshots and the epoch window shared by live and promote averaging."""
import logging
from collections import deque

import torch

logger = logging.getLogger(__name__)


def filter_lora_state_dict(state_dict):
    return {key: value.detach().to(device="cpu", dtype=torch.float32, copy=True)
            for key, value in state_dict.items() if key.startswith("lora_")}


def average_state_dicts(states, mode="uniform"):
    """Pure averaging retains the legacy key intersection; training validates complete snapshots first."""
    if not states or mode not in {"uniform", "ema"}:
        raise ValueError("Expected a nonempty window and uniform or ema mode")
    first = states[0]
    common = set(first)
    for state in states[1:]:
        if state.keys() != first.keys():
            logger.warning("Average checkpoint keys differ: missing=%s unexpected=%s", first.keys()-state.keys(), state.keys()-first.keys())
        common.intersection_update(state)
    if mode == "uniform":
        return {key: torch.stack([state[key] for state in states]).mean(dim=0) for key in first if key in common}
    alpha = 2 / (len(states) + 1)
    result = {key: value.clone() for key, value in first.items() if key in common}
    for state in states[1:]:
        for key, value in result.items():
            value.mul_(1 - alpha).add_(state[key], alpha=alpha)
    return result


def check_lora_layout(state, target):
    target_keys = {key for key in target if key.startswith("lora_")}
    if not state or state.keys() != target_keys:
        raise ValueError(f"LoRA window keys differ: missing={sorted(target_keys - state.keys())}, unexpected={sorted(state.keys() - target_keys)}")
    for key, value in state.items():
        if value.shape != target[key].shape:
            raise ValueError(f"LoRA window shape mismatch for {key}: {value.shape} vs {target[key].shape}")


def apply_lora_snapshot(network, state):
    check_lora_layout(state, network.state_dict())
    network.load_state_dict(state, strict=False)


class AverageWindow:
    def __init__(self, size, mode):
        if size < 1 or mode not in {"uniform", "ema"}:
            raise ValueError("Expected a positive window and uniform or ema mode")
        self.entries = deque(maxlen=size)
        self.mode = mode

    @property
    def epochs(self):
        return tuple(epoch for epoch, _ in self.entries)

    def append(self, epoch, network):
        """Return the independent raw snapshot and, once the window is full, its single candidate."""
        raw = filter_lora_state_dict(network.state_dict())
        if not raw:
            raise ValueError("Cannot average a network without lora_ weights")
        if self.entries:
            last_epoch, last = self.entries[-1]
            if epoch <= last_epoch:
                raise ValueError(f"Epoch {epoch} already recorded or out of order (last={last_epoch})")
            check_lora_layout(raw, last)
        self.entries.append((epoch, raw))
        if len(self.entries) < self.entries.maxlen:
            return raw, None
        return raw, average_state_dicts([state for _, state in self.entries], self.mode)
