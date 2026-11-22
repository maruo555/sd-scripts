"""Record the weighted training loss of one sample against its actual training subset."""

import math
from numbers import Integral

import torch


def _group_name(subset):
    return (getattr(subset, "group", None) or "").strip() or "__ungrouped__"


def _one(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return None


def _timestep(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    return int(value) if isinstance(value, Integral) else None


class GroupLossObserver:
    beta = 0.98

    def __init__(self, dataset_group):
        self.datasets = tuple(getattr(dataset_group, "datasets", ()))
        self.resolved_subsets = []
        self.subsets = {}
        self.history = {}
        for dataset_index, dataset in enumerate(self.datasets):
            size = getattr(dataset, "batch_size", None)
            if size is not None and size != 1:
                raise ValueError(f"group_loss_log requires batch size 1, dataset {dataset_index} uses {size}")
            for local_index, subset in enumerate(getattr(dataset, "subsets", ())):
                index = len(self.resolved_subsets)
                group = _group_name(subset)
                self.subsets[dataset_index, local_index] = (index, group)
                self.resolved_subsets.append({"subset_index": index, "dataset_index": dataset_index,
                                             "local_subset_index": local_index, "group": group})

    @staticmethod
    def validate_batch(batch):
        # Sample-first fields take precedence over auxiliary lists (e.g. layers).
        for key in ("loss_weights", "latents", "images", "input_ids", "captions",
                    "network_multipliers", "subset_indices", "image_keys"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim:
                size = value.shape[0]
            elif isinstance(value, (list, tuple)):
                size = len(value)
            else:
                continue
            if size == 1:
                return
            raise ValueError(f"group_loss_log requires batch size 1, {key} has {size}")
        raise ValueError("group_loss_log could not infer the runtime batch size")

    def _sample_metadata(self, batch):
        pair = _one(batch.get("subset_indices"))
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None, "__ungrouped__", None
        dataset_index, local_index = pair
        if not isinstance(dataset_index, Integral) or not 0 <= dataset_index < len(self.datasets):
            return None, "__ungrouped__", None
        dataset = self.datasets[dataset_index]
        index, group = self.subsets.get((dataset_index, local_index), (-1, "__ungrouped__"))
        image_key = _one(batch.get("image_keys"))
        info = getattr(dataset, "image_data", {}).get(image_key)
        resolution = getattr(info, "bucket_reso", None)
        bucket = None
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            if all(isinstance(n, Integral) and n > 0 for n in resolution):
                bucket = f"{resolution[0]}x{resolution[1]}"
        return index, group, bucket

    def observe(self, batch, timestep, loss, epoch, attempt_step, guardian_skipped=False):
        self.validate_batch(batch)
        if guardian_skipped or not math.isfinite(loss):
            return []
        index, group, bucket = self._sample_metadata(batch)
        previous = self.history.get(group)
        if previous is None:
            ema, count = loss, 1
        else:
            ema, count = previous[0] * self.beta + loss * (1.0 - self.beta), previous[1] + 1
        self.history[group] = (ema, count)
        return [{"epoch": epoch, "attempt_step": attempt_step, "group": group, "subset_index": index,
                 "timestep": _timestep(timestep), "bucket_reso": bucket, "loss": loss,
                 "group_ema_loss": ema, "group_observation_count": count}]
