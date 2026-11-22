"""Bounded-memory effective rank measurements of LoRA factors."""
import logging

import torch

logger = logging.getLogger(__name__)


def _compute_lora_effective_rank_stats(lora, eps: float = 1e-12):
    w_down = getattr(lora, "lora_down", None)
    w_up = getattr(lora, "lora_up", None)
    if w_down is None or w_up is None:
        return None
    w_down = w_down.weight
    w_up = w_up.weight
    if w_down is None or w_up is None:
        return None

    with torch.no_grad():
        a = w_down.to(dtype=torch.float32)
        b = w_up.to(dtype=torch.float32)
        if a.dim() == 4:
            a = a.reshape(a.shape[0], -1)
        if b.dim() == 4:
            b = b.reshape(b.shape[0], b.shape[1])
        r = a.shape[0]
        if r <= 0:
            return None

        p = b.transpose(0, 1) @ b
        q = a @ a.transpose(0, 1)

        eye = torch.eye(r, device=q.device, dtype=q.dtype)
        diag_mean = torch.mean(torch.diag(q)) if r > 0 else torch.tensor(0.0, device=q.device, dtype=q.dtype)
        jitter = eps * (diag_mean.abs() + 1.0)
        l, info = torch.linalg.cholesky_ex(q + jitter * eye)
        if int(info.item()) != 0:
            eigvals, eigvecs = torch.linalg.eigh(q)
            eigvals = torch.clamp(eigvals, min=0.0)
            sqrt_q = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.transpose(0, 1)
            s = sqrt_q @ p @ sqrt_q
        else:
            s = l.transpose(0, 1) @ p @ l

        s = (s + s.transpose(0, 1)) * 0.5
        eigvals = torch.linalg.eigvalsh(s)
        eigvals = torch.clamp(eigvals, min=0.0)
        energy = eigvals.sum()
        energy_val = float(energy.item())
        scale = float(getattr(lora, "scale", 1.0))
        mult = float(getattr(lora, "multiplier", 1.0))
        energy_val *= (scale * mult) ** 2
        if energy_val <= eps:
            return {
                "module": lora.lora_name,
                "r": int(r),
                "sat": 0.0,
                "top1": 0.0,
                "energy": energy_val,
            }

        w = eigvals / (energy + eps)
        entropy = -(w * torch.log(w + eps)).sum()
        r_eff = torch.exp(entropy)
        sat = float(r_eff.item()) / float(r)
        top1 = float(torch.max(w).item())
        return {
            "module": lora.lora_name,
            "r": int(r),
            "sat": sat,
            "top1": top1,
            "energy": energy_val,
        }


_RANK_STATS_BATCH_MAX_BYTES = 64 * 1024 * 1024


def _rank_stats_batch_chunk_size(entries) -> int:
    """Keep temporary stacked rank-stat tensors within a bounded size."""
    if not entries:
        return 1

    _, _, a, b, r = entries[0]
    weight_numel = a.numel() + b.numel()
    # Conservatively account for both the stacked source dtype and its FP32
    # compute copy. Gram/eigendecomposition tensors are small for normal LoRA
    # ranks, but include headroom so high-rank custom networks remain bounded.
    weight_bytes = weight_numel * (max(a.element_size(), b.element_size()) + 4)
    workspace_bytes = 12 * r * r * 4
    bytes_per_item = max(1, weight_bytes + workspace_bytes)
    return max(1, _RANK_STATS_BATCH_MAX_BYTES // bytes_per_item)


def _compute_rank_metrics_from_grams(p: torch.Tensor, q: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    diag_mean = torch.diagonal(q, dim1=-2, dim2=-1).mean(dim=-1)
    jitter = eps * (diag_mean.abs() + 1.0)
    eye = torch.eye(r, device=q.device, dtype=q.dtype).unsqueeze(0)
    chol, info = torch.linalg.cholesky_ex(q + jitter[:, None, None] * eye)

    chol_s = (chol.transpose(1, 2) @ p) @ chol

    # Failed entries are replaced before the batched eigensolve so one bad
    # Cholesky factor cannot poison the group. `info` travels in the final host
    # transfer and only failed entries use the original scalar fallback.
    s = torch.where((info == 0)[:, None, None], chol_s, torch.zeros_like(chol_s))
    s = (s + s.transpose(1, 2)) * 0.5
    eigvals = torch.linalg.eigvalsh(s)
    eigvals = torch.clamp(eigvals, min=0.0)
    energy = eigvals.sum(dim=-1)
    weights = eigvals / (energy[:, None] + eps)
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
    effective_rank = torch.exp(entropy)
    top1 = torch.max(weights, dim=-1).values
    return torch.stack((energy, effective_rank, top1, info.to(dtype=torch.float32)), dim=-1)


def _compute_lora_effective_rank_stats_batched(loras, eps: float = 1e-12):
    """Compute effective-rank diagnostics in shape batches.

    The returned dictionaries and ordering match the scalar helper above. GPU
    results are copied to the host once per device rather than once per module.
    Unsupported custom module shapes retain the scalar behavior.
    """
    result_slots = [None] * len(loras)
    groups = {}

    with torch.no_grad():
        for index, lora in enumerate(loras):
            w_down_module = getattr(lora, "lora_down", None)
            w_up_module = getattr(lora, "lora_up", None)
            if w_down_module is None or w_up_module is None:
                continue

            a = w_down_module.weight
            b = w_up_module.weight
            if a is None or b is None:
                continue
            if a.dim() == 4:
                a = a.reshape(a.shape[0], -1)
            if b.dim() == 4:
                # LoRA up-convolutions are 1x1. Keep the existing reshape so a
                # non-standard kernel raises in the same way as the scalar path.
                b = b.reshape(b.shape[0], b.shape[1])

            r = a.shape[0]
            if r <= 0:
                continue

            if a.dim() != 2 or b.dim() != 2 or a.device != b.device:
                result_slots[index] = _compute_lora_effective_rank_stats(lora, eps=eps)
                continue

            key = (a.device, a.dtype, b.dtype, tuple(a.shape), tuple(b.shape))
            groups.setdefault(key, []).append((index, lora, a, b, int(r)))

        rank_group_counts = {}
        for entries in groups.values():
            rank_key = (entries[0][2].device, entries[0][4])
            rank_group_counts[rank_key] = rank_group_counts.get(rank_key, 0) + len(entries)

        # P/Q consolidation removes repeated tiny solver launches. Keep it only
        # while a conservative estimate of Gram matrices plus solver workspace
        # stays within the same temporary-memory budget as the shape batches.
        consolidate_rank_groups = {
            (device, r)
            for (device, r), count in rank_group_counts.items()
            if count * 12 * r * r * 4 <= _RANK_STATS_BATCH_MAX_BYTES
        }

        gram_groups = {}
        device_outputs = {}
        for entries in groups.values():
            chunk_size = _rank_stats_batch_chunk_size(entries)
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                a_batch = torch.stack([entry[2] for entry in chunk], dim=0).to(dtype=torch.float32)
                b_batch = torch.stack([entry[3] for entry in chunk], dim=0).to(dtype=torch.float32)
                r = chunk[0][4]

                p = b_batch.transpose(1, 2) @ b_batch
                q = a_batch @ a_batch.transpose(1, 2)
                del a_batch, b_batch
                rank_key = (q.device, r)
                if rank_key in consolidate_rank_groups:
                    gram_groups.setdefault(rank_key, []).append((chunk, p, q))
                else:
                    packed = _compute_rank_metrics_from_grams(p, q, r, eps)
                    device_outputs.setdefault(packed.device, []).append((chunk, packed))

        for (device, r), gram_batches in gram_groups.items():
            entries = [entry for chunk, _, _ in gram_batches for entry in chunk]
            if len(gram_batches) == 1:
                p = gram_batches[0][1]
                q = gram_batches[0][2]
            else:
                p = torch.cat([p_batch for _, p_batch, _ in gram_batches], dim=0)
                q = torch.cat([q_batch for _, _, q_batch in gram_batches], dim=0)
            packed = _compute_rank_metrics_from_grams(p, q, r, eps)
            device_outputs.setdefault(device, []).append((entries, packed))

        for outputs in device_outputs.values():
            host_values = torch.cat([packed for _, packed in outputs], dim=0).cpu().tolist()
            offset = 0
            for chunk, packed in outputs:
                chunk_values = host_values[offset : offset + packed.shape[0]]
                offset += packed.shape[0]
                for entry, values in zip(chunk, chunk_values):
                    index, lora, _, _, r = entry
                    raw_energy, effective_rank, top1, cholesky_info = values
                    if int(cholesky_info) != 0:
                        result_slots[index] = _compute_lora_effective_rank_stats(lora, eps=eps)
                        continue
                    scale = float(getattr(lora, "scale", 1.0))
                    mult = float(getattr(lora, "multiplier", 1.0))
                    energy_val = float(raw_energy) * (scale * mult) ** 2
                    if energy_val <= eps:
                        sat = 0.0
                        top1_val = 0.0
                    else:
                        sat = float(effective_rank) / float(r)
                        top1_val = float(top1)
                    result_slots[index] = {
                        "module": lora.lora_name,
                        "r": int(r),
                        "sat": sat,
                        "top1": top1_val,
                        "energy": energy_val,
                    }

    return [stats for stats in result_slots if stats is not None]


class RankObserver:
    """Measure UNet adapters after scheduled updates, using actual optimizer membership."""

    interval = 100

    def __init__(self, network, optimizer):
        self.modules = tuple(network.unet_loras)
        self.module_names = tuple(module.lora_name for module in self.modules)
        owners = {}
        for module in self.modules:
            for parameter in module.parameters():
                owners[id(parameter)] = "unet"
        for module in network.text_encoder_loras:
            index = module.text_encoder_index
            if index in (1, 2):
                for parameter in module.parameters():
                    owners[id(parameter)] = f"te{index}"
        self.group_indices = {component: [] for component in ("unet", "te1", "te2")}
        for index, group in enumerate(optimizer.param_groups):
            components = {owners.get(id(parameter)) for parameter in group["params"]}
            for component in self.group_indices:
                if component in components:
                    self.group_indices[component].append(index)

    def observe(self, epoch, attempt_step, scheduler, guardian_skipped=False):
        if guardian_skipped or attempt_step <= 0 or attempt_step % self.interval or not self.modules:
            return []
        try:
            values = _compute_lora_effective_rank_stats_batched(self.modules)
            if tuple(result["module"] for result in values) != self.module_names:
                raise ValueError("Missing rank statistics for an expected UNet adapter")
            all_rates = scheduler.get_last_lr()
            rates = {}
            for component, indices in self.group_indices.items():
                selected = [float(all_rates[i]) for i in indices]
                rates[f"{component}_lr_min"] = min(selected, default=None)
                rates[f"{component}_lr_max"] = max(selected, default=None)
            return [
                {"epoch": epoch, "attempt_step": attempt_step, "scope": "unet",
                 "module": value["module"], "rank_dim": value["r"], "rank_sat": value["sat"],
                 "rank_top1": value["top1"], "rank_energy": value["energy"], **rates}
                for value in values
            ]
        except (RuntimeError, ValueError, IndexError) as exc:
            logger.warning("Rank observation at attempt %d was omitted: %s", attempt_step, exc)
            return []
