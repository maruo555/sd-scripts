"""Explicit fixed delta-quantization policies. No ordinary-training defaults."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


UNET_GROUPS = ("unet.attn1", "unet.attn2", "unet.ff", "unet.other_projection")
ATTENTION_ROLES = ("q", "k", "v", "out")
UNET_REGIONS = ("input", "middle", "output")
FINE_GROUPS = tuple(
    group
    for attention in ("attn1", "attn2")
    for role in ATTENTION_ROLES
    for group in (f"unet.{attention}.{role}", *(f"unet.{attention}.{role}.{region}" for region in UNET_REGIONS))
)


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


def module_groups(name: str) -> tuple[str, ...]:
    """Applicable selectors in increasing specificity; coarse IDs stay stable."""
    coarse = module_group(name)
    groups = [coarse]
    if coarse in {"unet.attn1", "unet.attn2"}:
        role = next((role for role, suffix in (("q", "_to_q"), ("k", "_to_k"), ("v", "_to_v"), ("out", "_to_out_0")) if name.endswith(suffix)), None)
        if role:
            groups.append(f"{coarse}.{role}")
            region = next((region for region, token in (("input", "_input_blocks_"), ("middle", "_middle_block_"), ("output", "_output_blocks_")) if token in name), None)
            if region:
                groups.append(f"{coarse}.{role}.{region}")
    return tuple(groups)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate policy JSON key: {key}")
        result[key] = value
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MulPolicy:
    base_mul: float
    components: tuple[tuple[str, float], ...] = ()
    group_overrides: tuple[tuple[str, float], ...] = ()
    te_quantized: bool = True
    module_overrides: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_file(cls, path) -> "MulPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json_object))

    @classmethod
    def from_training_args(cls, args):
        path = getattr(args, "dq_delta_policy_file", None)
        fields = ("attn2", "te", "te1", "te2")
        direct = {field: getattr(args, f"dq_delta_range_mul_{field}", None) for field in fields}
        direct = {field: value for field, value in direct.items() if value is not None}
        if path and direct:
            raise ValueError("Direct mul overrides cannot be combined with dq_delta_policy_file")
        if not path and not direct:
            return None
        if (getattr(args, "dq_delta_bits", 0) or 0) <= 0 or getattr(args, "dq_delta_bits_sched", None):
            raise ValueError("Fixed mul settings require fixed dq_delta_bits")
        if getattr(args, "dq_delta_stat", None) != "rms" or getattr(args, "dq_quantize_z", False):
            raise ValueError("Fixed mul settings require RMS delta quantization")
        if getattr(args, "dq_delta_auto_range_mul", False):
            raise ValueError("Fixed mul policies cannot be combined with automatic mul tuning")
        if path:
            return cls.from_file(path)

        # Compile CLI overrides into the same declaration used by the network,
        # checkpoint metadata and resume guard. No diagnostic artifact is needed.
        direct = {field: positive_mul(value) for field, value in direct.items()}
        components = {}
        for component in ("te1", "te2"):
            value = direct.get(component, direct.get("te"))
            if value is not None:
                components[component] = value
        groups = []
        if "attn2" in direct:
            groups.append({"group_id": "unet.attn2", "range_mul": direct["attn2"]})
        return cls.from_dict({
            "base_mul": getattr(args, "dq_delta_range_mul", 3.0),
            "components": components,
            "group_overrides": groups,
            "te_quantized": True,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MulPolicy":
        if not isinstance(value, Mapping):
            raise ValueError("Policy must be an object")
        unknown = set(value) - {"base_mul", "components", "group_overrides", "te_quantized", "module_overrides"}
        if unknown:
            raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
        components = value.get("components", {})
        if not isinstance(components, Mapping):
            raise ValueError("components must be an object")
        if set(components) - {"unet", "te1", "te2"}:
            raise ValueError("Unknown component override")
        groups = []
        for item in value.get("group_overrides", []):
            if set(item) != {"group_id", "range_mul"} or item["group_id"] not in (*UNET_GROUPS, *FINE_GROUPS):
                raise ValueError("Unknown group override")
            groups.append((item["group_id"], positive_mul(item["range_mul"])))
        if len({name for name, _ in groups}) != len(groups):
            raise ValueError("Conflicting group overrides are not ordered assignments")
        enabled = value.get("te_quantized", True)
        if not isinstance(enabled, bool):
            raise ValueError("te_quantized must be a boolean, never a numeric mul")
        modules = value.get("module_overrides", {})
        if not isinstance(modules, Mapping) or any(not isinstance(name, str) for name in modules):
            raise ValueError("module_overrides must map exact module names to mul values")
        return cls(positive_mul(value["base_mul"]), tuple(sorted((k, positive_mul(v)) for k, v in components.items())), tuple(sorted(groups)), enabled, tuple(sorted((k, positive_mul(v)) for k, v in modules.items())))

    def to_dict(self) -> dict[str, Any]:
        result = {"base_mul": self.base_mul, "components": dict(self.components), "group_overrides": [{"group_id": k, "range_mul": v} for k, v in self.group_overrides], "te_quantized": self.te_quantized}
        if self.module_overrides:
            result["module_overrides"] = dict(self.module_overrides)
        return result

    def expand(self, names: list[str]) -> dict[str, dict[str, Any]]:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate module names")
        components, overrides, modules = dict(self.components), dict(self.group_overrides), dict(self.module_overrides)
        found = set()
        result = {}
        for name in sorted(names):
            group = module_group(name)
            component = "unet" if group.startswith("unet.") else group
            groups = module_groups(name)
            found.update(groups)
            mul = components.get(component, self.base_mul)
            for selector in groups:
                mul = overrides.get(selector, mul)
            result[name] = {"group": group, "component": component, "mul": modules.get(name, mul), "enabled": component == "unet" or self.te_quantized}
        if set(overrides) - found:
            raise ValueError(f"Unmatched group overrides: {sorted(set(overrides) - found)}")
        actual_components = {row["component"] for row in result.values()}
        if set(components) - actual_components:
            raise ValueError("Unmatched component overrides")
        if set(modules) - set(names):
            raise ValueError(f"Unmatched exact module overrides: {sorted(set(modules) - set(names))}")
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


def fixed_policy_resume_record(record):
    """Persist the normalized declaration and resolved assignments, not the input path."""
    if record is None:
        return None
    declaration = MulPolicy.from_dict(record["declaration"]).to_dict()
    assignments = record["assignments_sha256"]
    if not isinstance(assignments, str) or len(assignments) != 64 or any(c not in "0123456789abcdef" for c in assignments):
        raise ValueError("Invalid resolved mul assignment hash")
    if "modules" in record and digest(record["modules"]) != assignments:
        raise ValueError("Resolved mul assignments do not match their hash")
    return {"schema_version": 1, "declaration": declaration, "assignments_sha256": assignments}


def validate_fixed_policy_resume(train_state, current_record):
    """Keep legacy resumes unchanged; an explicit policy must match saved state."""
    expected = fixed_policy_resume_record(current_record)
    saved = train_state.get("dq_mul_policy")
    if saved is None and expected is None:
        return
    if saved is None:
        raise ValueError("Resume state has no fixed mul policy record. Start a new run instead of changing policy during resume.")
    if expected is None:
        raise ValueError("Resume state requires the same fixed mul settings (CLI overrides or dq_delta_policy_file).")
    if saved != expected:
        raise ValueError("Fixed mul policy or resolved module assignments differ from the resume state. Resume with the original declaration; equivalent expanded values alone are insufficient. Start a new run when switching declaration formats or settings.")



@contextmanager
def fixed_policy_resume_load_context():
    """Allow NumPy RNG array data during an explicitly requested policy resume.

    Older Accelerate versions load their NumPy RNG pickle with torch.load's
    current weights-only default. Keep that restriction and allow only the
    array reconstruction types it needs, scoped to the resume operation.
    """
    import numpy as np
    import torch

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    get_safe_globals = getattr(torch.serialization, "get_safe_globals", None)
    if safe_globals is None or get_safe_globals is None:
        yield
        return
    rng_array_types = (np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32)))
    already_allowed = set(get_safe_globals())
    missing = [value for value in rng_array_types if value not in already_allowed]
    with safe_globals(missing):
        yield
