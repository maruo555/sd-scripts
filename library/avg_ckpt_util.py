from typing import Dict, List, Optional, Tuple
import os
import re
import logging
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


def filter_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only LoRA weights from state_dict on CPU as float32."""
    return {k: v.detach().float().cpu() for k, v in state_dict.items() if k.startswith("lora_")}


def average_state_dicts(states: List[Dict[str, torch.Tensor]], mode: str = "uniform", metrics: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    assert len(states) > 0
    reference_keys = list(states[0].keys())
    keys = set(reference_keys)
    for sd in states[1:]:
        keys &= set(sd.keys())
    for sd in states:
        missing = [k for k in reference_keys if k not in sd]
        unexpected = [k for k in sd.keys() if k not in reference_keys]
        if missing or unexpected:
            logging.warning(f"[AVG] missing={missing} unexpected={unexpected}")
    if mode == "metric" and metrics is None:
        metrics = [1.0] * len(states)
    if mode == "uniform":
        avg = {k: torch.stack([sd[k] for sd in states if k in sd], dim=0).mean(dim=0) for k in keys}
    elif mode == "ema":
        alpha = 2 / (len(states) + 1)
        avg = {k: states[0][k].clone() for k in keys}
        for sd in states[1:]:
            for k in keys:
                avg[k].mul_(1 - alpha).add_(sd[k], alpha=alpha)
    else:  # metric
        total = sum(metrics)
        weights = [m / total for m in metrics]
        avg = {k: sum(w * sd[k] for sd, w in zip(states, weights) if k in sd) for k in keys}
    return avg


def load_lora_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, map_location="cpu")
    return filter_lora_state_dict(sd)


def save_lora_state_dict(
    path: str, state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None, metadata: Optional[Dict[str, str]] = None
):
    save_sd = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if dtype is not None:
            tensor = tensor.to(dtype)
        save_sd[key] = tensor

    ext = os.path.splitext(path)[1]
    if ext == ".safetensors":
        save_safetensors(save_sd, path, metadata=metadata)
    else:
        torch.save(save_sd, path)


def collect_last_checkpoints_with_epochs(output_dir: str, model_name: str, ext: str, n: int) -> List[Tuple[int, str]]:
    pattern = re.compile(re.escape(model_name) + r"-(\d{6})" + re.escape(ext) + "$")
    files = []
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), os.path.join(output_dir, f)))
    files.sort()
    return files[-n:]


def collect_last_checkpoints(output_dir: str, model_name: str, ext: str, n: int) -> List[str]:
    return [p for _, p in collect_last_checkpoints_with_epochs(output_dir, model_name, ext, n)]
