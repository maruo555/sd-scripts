from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from library import self_distill_cache


class SelfDistillDataset(Dataset):
    def __init__(self, manifest_path: str, require_prompt_embeddings: bool = False):
        self.manifest_path = manifest_path
        self.entries = self_distill_cache.load_manifest(manifest_path)
        self.require_prompt_embeddings = require_prompt_embeddings

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        bundle = self_distill_cache.load_tensor_bundle(entry["tensors_path"])

        if self.require_prompt_embeddings:
            required = {
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            }
            missing = required.difference(bundle.keys())
            if missing:
                raise ValueError(
                    f"Cached prompt embeddings are required but missing in {entry['tensors_path']}: {sorted(missing)}"
                )

        item = {
            "record_id": entry["record_id"],
            "prompt_text": entry["prompt_text"],
            "negative_prompt": entry.get("negative_prompt", ""),
            "variant_type": entry["variant_type"],
            "seed": entry["seed"],
            "generation_settings": entry["generation_settings"],
            "preview_image_path": entry.get("preview_image_path"),
        }
        item.update(bundle)
        return item


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("SelfDistillDataset currently supports batch_size=1 only.")
    return batch[0]
