"""Resolve fixed training multipliers once from the attached SDXL LoRA names."""
from dataclasses import dataclass
import hashlib
import json
import math

OVERRIDES = ("attn2", "te", "te1", "te2")
RESOLVER_VERSION = 1
QUANT_COMPONENTS = {"unet": {"unet"}, "te": {"te1", "te2"}, "both": {"unet", "te1", "te2"}}


def module_group(name: str) -> str:
    if name.startswith("lora_te1_"):
        return "te1"
    if name.startswith("lora_te2_"):
        return "te2"
    if not name.startswith("lora_unet_"):
        raise ValueError(f"Unknown SDXL LoRA component: {name}")
    if "_attn1_" in name:
        return "unet.attn1"
    if "_attn2_" in name:
        return "unet.attn2"
    if "_ff_" in name:
        return "unet.ff"
    if "_proj_in" in name or "_proj_out" in name:
        return "unet.other_projection"
    return "unet.other"


def scope_includes(scope: str, component: str) -> bool:
    """Quantization scope is independent of assignment, logging and auto-control."""
    return component in QUANT_COMPONENTS[scope]


def assignment_digest(assignments: dict) -> str:
    encoded = json.dumps(assignments, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MulPolicy:
    base: float
    attn2: float | None = None
    te: float | None = None
    te1: float | None = None
    te2: float | None = None

    def __post_init__(self):
        for key in ("base", *OVERRIDES):
            value = getattr(self, key)
            if key != "base" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"DQ {key} multiplier must be finite and positive: {value!r}")
            object.__setattr__(self, key, float(value))

    @classmethod
    def from_values(cls, values: dict):
        return cls(values["dq_delta_range_mul"], **{key: values.get("dq_delta_range_mul_" + key) for key in OVERRIDES})

    def expand(self, names) -> dict:
        result = {}
        per_encoder = {te: getattr(self, te) if getattr(self, te) is not None else self.te for te in ("te1", "te2")}
        expected = {te for te, mul in per_encoder.items() if mul is not None}
        if self.attn2 is not None:
            expected.add("unet.attn2")
        for name in sorted(names):
            if name in result:
                raise ValueError(f"Duplicate LoRA module name: {name}")
            group = module_group(name)
            expected.discard(group)
            component = "unet" if group.startswith("unet.") else group
            override = self.attn2 if group == "unet.attn2" else per_encoder.get(component)
            result[name] = {
                "group": group, "component": component, "mul": self.base if override is None else override, "enabled": True
            }
        if expected:
            raise ValueError(f"DQ multiplier overrides matched no attached LoRA for: {', '.join(sorted(expected))}")
        return result

    def record(self, assignments: dict) -> dict:
        return {
            "base": self.base,
            "overrides": {key: getattr(self, key) for key in OVERRIDES if getattr(self, key) is not None},
            "resolver_version": RESOLVER_VERSION,
            "module_names": sorted(assignments),
            "assignment_digest": assignment_digest(assignments),
        }


def restore_assignments(record: dict) -> dict:
    """Read resolved.dq.mul; callers report invalid or unsupported records as unavailable."""
    if type(record.get("resolver_version")) is not int or record["resolver_version"] != RESOLVER_VERSION:
        raise ValueError(f"Unsupported DQ multiplier resolver version: {record.get('resolver_version')!r}")
    overrides = record.get("overrides")
    names = record.get("module_names")
    if (not isinstance(overrides, dict) or set(overrides) - set(OVERRIDES)
            or not isinstance(names, list) or not all(isinstance(name, str) for name in names)):
        raise ValueError("Invalid DQ multiplier overrides or module_names")
    result = MulPolicy(record.get("base"), **overrides).expand(names)
    if assignment_digest(result) != record.get("assignment_digest"):
        raise ValueError("DQ multiplier assignment digest mismatch")
    return result
