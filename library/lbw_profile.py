import json
import math
import os
from typing import Any, Dict, Optional

import torch


def _load_profile_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        if ext == ".toml":
            import toml

            return toml.load(f)
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to read YAML LBW profiles.") from exc
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported LBW profile format: {path}")


def load_profile(path: Optional[str]) -> Optional[Dict[str, float]]:
    if path is None:
        return None
    raw = _load_profile_file(path)
    profile = raw.get("profile", raw)
    normalized = {
        "input": float(profile.get("input", 1.0)),
        "mid": float(profile.get("mid", 1.0)),
        "output": float(profile.get("output", 1.0)),
        "conv": float(profile.get("conv", 1.0)),
        "default": float(profile.get("default", 1.0)),
    }
    return normalized


def group_for_lora_name(name: str, tensor: Optional[torch.Tensor] = None) -> str:
    if "input_blocks" in name:
        return "input"
    if "middle_block" in name or "mid_block" in name:
        return "mid"
    if "output_blocks" in name or "out." in name:
        return "output"
    if tensor is not None and tensor.ndim == 4 and tuple(tensor.shape[-2:]) != (1, 1):
        return "conv"
    return "default"


def scale_lora_state_dict(weights_sd: Dict[str, torch.Tensor], profile: Optional[Dict[str, float]]) -> Dict[str, torch.Tensor]:
    if profile is None:
        return weights_sd

    scaled = {}
    for key, value in weights_sd.items():
        if "lora_up.weight" in key or "lora_down.weight" in key:
            lora_name = key.rsplit(".", 2)[0]
            group = group_for_lora_name(lora_name, value)
            multiplier = profile.get(group, profile.get("default", 1.0))
            factor = math.sqrt(max(multiplier, 0.0))
            scaled[key] = value * factor
        else:
            scaled[key] = value
    return scaled


def apply_profile_to_network(network, profile: Optional[Dict[str, float]]) -> None:
    if profile is None:
        return

    state_dict = network.state_dict()
    for key, value in list(state_dict.items()):
        if "lora_up.weight" in key or "lora_down.weight" in key:
            lora_name = key.rsplit(".", 2)[0]
            group = group_for_lora_name(lora_name, value)
            multiplier = profile.get(group, profile.get("default", 1.0))
            state_dict[key] = value * math.sqrt(max(multiplier, 0.0))
    network.load_state_dict(state_dict, strict=False)
