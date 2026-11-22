"""Epoch-end LoRA averaging. Scheduled checkpoints and samples are saved before this controller runs."""
import os
import tempfile
from pathlib import Path

import torch

from library.avg_ckpt_util import AverageWindow, apply_lora_snapshot
from library.promote_bank import PromoteBank


def reset_optimizer_moments(optimizer):
    for state in optimizer.state.values():
        state.setdefault("step", 0)
        for key in ("exp_avg", "exp_avg_sq", "exp_avg_max"):
            buffer = state.get(key)
            if isinstance(buffer, torch.Tensor):
                buffer.zero_()


def score_pair(network, raw, candidate, score, bank, device):
    devices = [device] if device.type == "cuda" else []
    # The inner fork restarts candidate scoring from the exact raw RNG state.
    # The outer fork also restores RNG if restoring raw weights raises.
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        try:
            with torch.random.fork_rng(devices=devices):
                raw_score = score(bank.items)
            apply_lora_snapshot(network, candidate)
            candidate_score = score(bank.items)
        finally:
            apply_lora_snapshot(network, raw)
    return raw_score, candidate_score


class EpochAveraging:
    def __init__(self, values):
        self.window = AverageWindow(values["avg_window"], values["avg_mode"])
        self.bank = PromoteBank(values["avg_shadow_bank_size"]) if values["avg_cp_mode"] == "promote" else None
        self.begin = values["avg_begin"]
        self.margin = values["avg_shadow_margin"]
        self.patience = values["avg_shadow_patience"]
        self.reset = values["avg_reset_stats"]
        self.keep_final_raw = values["avg_save_final_raw"]
        self.streak = 0
        self.final_raw = None

    def eligible(self, epoch, total_epochs):
        return epoch / total_epochs >= self.begin

    def finish_epoch(self, epoch, total_epochs, network, optimizer, score, device, total_timesteps):
        final_epoch = epoch == total_epochs
        candidate = None
        active = self.eligible(epoch, total_epochs)
        if active:
            raw, candidate = self.window.append(epoch, network)
            if final_epoch and self.keep_final_raw:
                self.final_raw = raw
        if self.bank is None:
            if candidate is not None:
                self._adopt(network, optimizer, candidate)
            return []

        raw_score = candidate_score = None
        margin_ok = adopted = False
        if not active:
            status = "before_avg_begin"
        elif not self.bank.finalize(final_epoch=final_epoch):
            status = "building_pool"
        elif candidate is None:
            status = "waiting_window"
        else:
            raw_score, candidate_score = score_pair(network, raw, candidate, score, self.bank, device)
            margin_ok = candidate_score < raw_score * (1.0 - self.margin)
            self.streak = self.streak + 1 if margin_ok else 0
            adopted = margin_ok and self.streak >= self.patience
            if adopted:
                self._adopt(network, optimizer, candidate)
            status = "scored"

        epochs = self.window.epochs
        row = {
            "row_type": "decision", "epoch": epoch, "status": status, "selected_candidate_mode": self.window.mode,
            "raw_proxy_loss": raw_score, "selected_proxy_loss": candidate_score, "margin_ok": margin_ok,
            "win_streak": self.streak, "promote_applied": adopted, "optimizer_reset_applied": adopted and self.reset,
            "construction_pool_size": self.bank.construction_size, "bank_size": len(self.bank.items), "window_count": len(epochs),
            **self.bank.summary(total_timesteps),
        }
        return [row] + [{"row_type": "window_epoch", "epoch": epoch, "window_position": i, "window_epoch": member}
                        for i, member in enumerate(epochs, 1)]

    def _adopt(self, network, optimizer, candidate):
        apply_lora_snapshot(network, candidate)
        if self.reset:
            reset_optimizer_moments(optimizer)


def save_final_raw(path, state, dtype, metadata):
    """Serialize the independent raw snapshot after the usual final checkpoint, without changing live weights."""
    from safetensors.torch import save_file
    from library.model_io import precalculate_safetensors_hashes

    path = Path(path)
    tensors = {key: (v if dtype is None else v.to(dtype=dtype)) for key, v in state.items()}
    metadata = dict(metadata, ss_avg_shadow_variant="final_raw")
    metadata["sshs_model_hash"], metadata["sshs_legacy_hash"] = precalculate_safetensors_hashes(tensors, metadata)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(fd)
    try:
        save_file(tensors, temp, metadata=metadata)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
