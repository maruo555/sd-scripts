"""Fixed CPU inputs for comparing raw and averaged SDXL adapters, sampled from real training batches."""
from collections import Counter
from dataclasses import dataclass

import torch


def _copy_tree(value, device, *, compact=False):
    if isinstance(value, torch.Tensor):
        dtype = torch.float16 if compact and value.is_floating_point() else value.dtype
        return value.detach().to(device=device, dtype=dtype, copy=True)
    if isinstance(value, dict):
        return {k: _copy_tree(v, device, compact=compact) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_tree(v, device, compact=compact) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(v, device, compact=compact) for v in value)
    return value


def labels(values):
    return [str(value).strip() for value in values if value is not None and str(value).strip()]


@dataclass
class BankItem:
    batch: dict
    noisy_latents: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    huber_c: object
    weighting: object
    weighted_tokens: object

    @classmethod
    def capture(cls, batch, noisy_latents, target, timesteps, huber_c, weighting, weighted_tokens=None):
        if batch["loss_weights"].shape != (1,) or len(batch["image_keys"]) != 1 or len(batch["class_tokens"]) != 1:
            raise ValueError("promote expects one training image, class label and loss weight per batch")
        fields = ("image_keys", "class_tokens", "input_ids_list", "original_sizes_hw", "crop_top_lefts", "target_sizes_hw",
                  "loss_weights", "network_multipliers", "conditioning_images", "alpha_masks")
        saved = {key: _copy_tree(batch[key], "cpu", compact=(key == "alpha_masks")) for key in fields if key in batch}
        return cls(saved, _copy_tree(noisy_latents, "cpu", compact=True), _copy_tree(target, "cpu", compact=True),
                   _copy_tree(timesteps, "cpu"), _copy_tree(huber_c, "cpu"), _copy_tree(weighting, "cpu"),
                   _copy_tree(weighted_tokens, "cpu"))

    def materialize(self, device, dtype):
        noisy = self.noisy_latents.to(device=device, dtype=dtype, copy=True)
        target = self.target.to(device=device, dtype=dtype, copy=True)
        huber_c = self.huber_c
        if isinstance(huber_c, torch.Tensor):
            huber_c = huber_c.to(device=device, dtype=dtype, copy=True)
        return (_copy_tree(self.batch, device), noisy, target, _copy_tree(self.timesteps, device), huber_c,
                _copy_tree(self.weighting, device), _copy_tree(self.weighted_tokens, device))


class PromoteBank:
    def __init__(self, size):
        self.size = size
        self.pool_limit = max(size * 4, 32)
        self.pool = []
        self.items = []
        self.construction_size = 0

    def wants_item(self):
        return not self.items and len(self.pool) < self.pool_limit

    def add(self, item):
        if self.wants_item():
            self.pool.append(item)
            self.construction_size = len(self.pool)

    def finalize(self, *, final_epoch):
        if self.items:
            return True
        if len(self.pool) < self.size or (len(self.pool) < self.pool_limit and not final_epoch):
            return False
        # Dict insertion order is the first-seen class order. An empty class is a single fallback group.
        groups = {}
        for item in self.pool:
            key = "||".join(dict.fromkeys(labels(item.batch["class_tokens"]))) or "__no_class_tokens__"
            groups.setdefault(key, []).append(item)
        used_images = set()
        while len(self.items) < self.size:
            for candidates in groups.values():
                if not candidates:
                    continue
                chosen = 0
                for i, candidate in enumerate(candidates):
                    image = "||".join(dict.fromkeys(labels(candidate.batch["image_keys"])))
                    if not image or image not in used_images:
                        chosen = i
                        break
                item = candidates.pop(chosen)
                self.items.append(item)
                used_images.update(labels(item.batch["image_keys"]))
                if len(self.items) == self.size:
                    break
        self.pool.clear()
        return True

    def summary(self, total_timesteps):
        if not self.items:
            return {}
        result = {}
        for field, per_key in (("class_tokens", "bank_max_per_class_tokens"), ("image_keys", "bank_max_per_image_key")):
            counts = Counter(label for item in self.items for label in labels(item.batch[field]))
            result[f"bank_unique_{field}"] = len(counts)
            result[per_key] = max(counts.values(), default=0)
        times = [v for item in self.items for v in item.timesteps.reshape(-1).tolist()]
        low = sum(t < total_timesteps // 3 for t in times)
        high = sum(t >= 2 * total_timesteps // 3 for t in times)
        result.update(bank_timestep_low_count=low, bank_timestep_mid_count=len(times)-low-high, bank_timestep_high_count=high)
        return result
