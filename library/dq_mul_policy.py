"""Explicit fixed delta-quantization policies. No ordinary-training defaults."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping


UNET_GROUPS = ("unet.attn1", "unet.attn2", "unet.ff", "unet.other_projection")


def module_group(name: str) -> str:
    if name.startswith("lora_te1_"):
        return "te1"
    if name.startswith("lora_te2_"):
        return "te2"
    if not name.startswith("lora_unet_"):
        raise ValueError(f"Unknown LoRA component: {name}")
    if "_attn1_" in name:
        return "unet.attn1"
    if "_attn2_" in name:
        return "unet.attn2"
    if "_ff_" in name:
        return "unet.ff"
    if "_proj_in" in name or "_proj_out" in name:
        return "unet.other_projection"
    raise ValueError(f"Unclassified UNet LoRA: {name}")


def positive_mul(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("mul must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("mul must be a finite positive number")
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MulPolicy:
    base_mul: float
    components: tuple[tuple[str, float], ...] = ()
    group_overrides: tuple[tuple[str, float], ...] = ()
    te_quantized: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MulPolicy":
        unknown = set(value) - {"base_mul", "components", "group_overrides", "te_quantized"}
        if unknown:
            raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
        components = value.get("components", {})
        if set(components) - {"unet", "te1", "te2"}:
            raise ValueError("Unknown component override")
        groups = []
        for item in value.get("group_overrides", []):
            if set(item) != {"group_id", "range_mul"} or item["group_id"] not in UNET_GROUPS:
                raise ValueError("Unknown group override")
            groups.append((item["group_id"], positive_mul(item["range_mul"])))
        if len({name for name, _ in groups}) != len(groups):
            raise ValueError("Conflicting group overrides are not ordered assignments")
        enabled = value.get("te_quantized", True)
        if not isinstance(enabled, bool):
            raise ValueError("te_quantized must be a boolean, never a numeric mul")
        return cls(positive_mul(value["base_mul"]), tuple(sorted((k, positive_mul(v)) for k, v in components.items())), tuple(sorted(groups)), enabled)

    def expand(self, names: list[str]) -> dict[str, dict[str, Any]]:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate module names")
        components, overrides = dict(self.components), dict(self.group_overrides)
        found = set()
        result = {}
        for name in sorted(names):
            group = module_group(name)
            component = "unet" if group.startswith("unet.") else group
            found.add(group)
            result[name] = {"group": group, "component": component, "mul": overrides.get(group, components.get(component, self.base_mul)), "enabled": component == "unet" or self.te_quantized}
        if set(overrides) - found:
            raise ValueError(f"Unmatched group overrides: {sorted(set(overrides) - found)}")
        actual_components = {row["component"] for row in result.values()}
        if set(components) - actual_components:
            raise ValueError("Unmatched component overrides")
        return result

    def apply(self, network: Any, *, enabled: bool = True) -> dict[str, dict[str, Any]]:
        """Call AFTER the common setter/restore; never modifies trainable state."""
        modules = list(network.unet_loras) + list(network.text_encoder_loras)
        expanded = self.expand([m.lora_name for m in modules])
        for module in modules:
            row = expanded[module.lora_name]
            module.delta_q_range_mul = row["mul"]
            module.delta_q_enabled = bool(enabled and row["enabled"])
        return expanded
