from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from library import self_distill_cache


class SelfDistillDataset(Dataset):
    def __init__(self, manifest_path: str, split: str = "train", require_teacher_conditioning: bool = True):
        self.manifest_path = manifest_path
        self.header, entries = self_distill_cache.load_manifest_with_header(manifest_path)
        self.entries = [entry for entry in entries if entry.get("split", "train") == split]
        self.require_teacher_conditioning = require_teacher_conditioning

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        bundle = self_distill_cache.load_tensor_bundle(entry["tensors_path"])
        conditioning_source = entry.get("conditioning_source", "teacher")

        required = {
            "base_prompt_embeds",
            "base_negative_prompt_embeds",
            "base_pooled_prompt_embeds",
            "base_negative_pooled_prompt_embeds",
            "target_timesteps",
            "x_t",
            "teacher_target",
            "base_target",
        }
        if self.require_teacher_conditioning or conditioning_source == "teacher":
            required.update(
                {
                    "teacher_prompt_embeds",
                    "teacher_negative_prompt_embeds",
                    "teacher_pooled_prompt_embeds",
                    "teacher_negative_pooled_prompt_embeds",
                }
            )

        missing = sorted(key for key in required if key not in bundle)
        if missing:
            raise ValueError(f"Missing cached tensors in {entry['tensors_path']}: {missing}")

        item = {
            "record_id": entry["record_id"],
            "prompt_text": entry["prompt_text"],
            "negative_prompt": entry.get("negative_prompt", ""),
            "variant_type": entry["variant_type"],
            "split": entry.get("split", "train"),
            "seed": entry["seed"],
            "conditioning_source": conditioning_source,
            "loss_role": entry.get("loss_role", "keep"),
            "generation_settings": entry["generation_settings"],
            "template": entry.get("template", ""),
            "template_index": entry.get("template_index", -1),
            "preview_image_path": entry.get("preview_image_path"),
        }
        item.update(bundle)
        return item


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("SelfDistillDataset currently supports batch_size=1 only.")
    return batch[0]
