"""LoRA component-strength helpers used only by image generation tools.

The public training code deliberately does not depend on this module.  A
strength specification has one of three shapes:

* ``(common,)``
* ``(text_encoder, unet)``
* ``(text_encoder_1, text_encoder_2, unet)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LoraComponentStrengths:
    te1: float
    te2: float
    unet: float
    value_count: int = 1

    def __post_init__(self):
        if self.value_count not in (1, 2, 3):
            raise ValueError("value_count must be 1, 2, or 3")
        for name, value in (("te1", self.te1), ("te2", self.te2), ("unet", self.unet)):
            if not math.isfinite(value):
                raise ValueError(f"{name} strength must be finite")

    @property
    def is_component_split(self) -> bool:
        return self.value_count > 1

    def as_spec(self, value_count: int | None = None) -> tuple[float, ...]:
        count = self.value_count if value_count is None else value_count
        if count == 1:
            if not (self.te1 == self.te2 == self.unet):
                raise ValueError("component strengths cannot be represented by one value")
            return (self.unet,)
        if count == 2:
            if self.te1 != self.te2:
                raise ValueError("split TE1/TE2 strengths cannot be represented by two values")
            return (self.te1, self.unet)
        if count == 3:
            return (self.te1, self.te2, self.unet)
        raise ValueError("value_count must be 1, 2, or 3")


def normalize_strength_spec(value, default: float = 1.0) -> tuple[float, ...]:
    """Normalize a GUI/config strength value to a tuple with 1--3 values."""

    if value is None:
        values = (float(default),)
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError("strength must contain 1, 2, or 3 comma-separated numbers")
        values = tuple(float(part) for part in parts)
    elif isinstance(value, (int, float)):
        values = (float(value),)
    else:
        values = tuple(float(part) for part in value)

    if len(values) not in (1, 2, 3):
        raise ValueError("strength must contain 1, 2, or 3 values")
    if any(not math.isfinite(part) for part in values):
        raise ValueError("strength values must be finite")
    return values


def component_strengths_from_spec(value) -> LoraComponentStrengths:
    values = normalize_strength_spec(value)
    if len(values) == 1:
        return LoraComponentStrengths(values[0], values[0], values[0], 1)
    if len(values) == 2:
        return LoraComponentStrengths(values[0], values[0], values[1], 2)
    return LoraComponentStrengths(values[0], values[1], values[2], 3)


def format_strength_value(value: float) -> str:
    """Shortest round-tripping decimal for a single finite strength."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"strength must be finite, got {value!r}")
    return repr(number).removesuffix(".0")


def parse_prompt_strengths(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(token.strip()) for token in value.split(","))
    except ValueError as exc:
        raise ValueError(f"invalid --am value in {value!r}") from exc
    if not result or not all(math.isfinite(val) for val in result):
        raise ValueError(f"--am requires finite numbers, got {value!r}")
    return result


def format_strength_spec(value) -> str:
    values = normalize_strength_spec(value)
    return ", ".join(format_strength_value(part) for part in values)


def serialize_strength_spec(value):
    values = normalize_strength_spec(value)
    return values[0] if len(values) == 1 else list(values)


def flatten_strength_specs(specs: Iterable[Sequence[float] | float]) -> tuple[float, ...]:
    """Flatten per-LoRA specs using the widest mode used by the condition."""

    normalized = [normalize_strength_spec(spec) for spec in specs]
    if not normalized:
        return ()
    target_count = max(len(spec) for spec in normalized)
    flattened = []
    for spec in normalized:
        strengths = component_strengths_from_spec(spec)
        flattened.extend(strengths.as_spec(target_count))
    return tuple(flattened)


def resolve_flat_strengths(
    values: Sequence[float] | None,
    network_count: int,
    *,
    repeat_last_legacy_value: bool = False,
) -> tuple[LoraComponentStrengths, ...]:
    """Resolve CLI/prompt values using N, 2N, or 3N values.

    One through N values keep the legacy common-strength behavior.  Missing
    CLI values default to 1.0; prompt-line values repeat the last value, which
    preserves the historical ``--am`` behavior.
    """

    if network_count < 0:
        raise ValueError("network_count must not be negative")
    raw = tuple(float(value) for value in (values or ()))
    if any(not math.isfinite(value) for value in raw):
        raise ValueError("network strength values must be finite")
    if network_count == 0:
        if raw:
            raise ValueError("network strengths were specified but no networks are loaded")
        return ()

    count = len(raw)
    if count == 2 * network_count:
        return tuple(
            LoraComponentStrengths(raw[index], raw[index], raw[index + 1], 2)
            for index in range(0, count, 2)
        )
    if count == 3 * network_count:
        return tuple(
            LoraComponentStrengths(raw[index], raw[index + 1], raw[index + 2], 3)
            for index in range(0, count, 3)
        )
    if count <= network_count:
        common = list(raw)
        if not common:
            common = [1.0] * network_count
        elif len(common) < network_count:
            fill = common[-1] if repeat_last_legacy_value else 1.0
            common.extend([fill] * (network_count - len(common)))
        return tuple(LoraComponentStrengths(value, value, value, 1) for value in common)

    raise ValueError(
        f"expected at most {network_count} common strengths, "
        f"{2 * network_count} TE/U-Net strengths, or "
        f"{3 * network_count} TE1/TE2/U-Net strengths; got {count}"
    )


def validate_component_strength_compatibility(
    strengths: Sequence[LoraComponentStrengths],
    *,
    network_merge: bool = False,
    network_pre_calc: bool = False,
) -> None:
    """Reject runtime component controls that cannot survive static weights."""

    if not any(value.is_component_split for value in strengths):
        return
    if network_merge:
        raise ValueError("TE/U-Net component strengths cannot be combined with network weight merging")
    if network_pre_calc:
        raise ValueError("TE/U-Net component strengths cannot be combined with --network_pre_calc")


def apply_generation_strengths(network, strengths: LoraComponentStrengths) -> None:
    """Apply resolved strengths without modifying the training LoRA classes."""

    setter = getattr(network, "set_multiplier", None)
    if not callable(setter):
        raise ValueError(f"{type(network).__name__} does not support runtime multiplier changes")
    if not strengths.is_component_split:
        setter(strengths.unet)
        return

    text_encoder_loras = getattr(network, "text_encoder_loras", None)
    unet_loras = getattr(network, "unet_loras", None)
    if text_encoder_loras is None or unet_loras is None:
        raise ValueError(
            f"{type(network).__name__} does not expose Text Encoder and U-Net LoRA modules required for component strengths"
        )

    groups = getattr(network, "_generation_te_groups", None)
    if groups is None:
        groups = ([], [])
        for module in text_encoder_loras:
            name = getattr(module, "lora_name", "")
            if name.startswith("lora_te1_"):
                groups[0].append(module)
            elif name.startswith("lora_te2_"):
                groups[1].append(module)
            else:
                raise ValueError(f"cannot identify SDXL Text Encoder for layer {name!r}")
        network._generation_te_groups = groups
    setter(strengths.unet)
    for group, strength in zip(groups, (strengths.te1, strengths.te2)):
        for module in group:
            module.multiplier = strength


def resolve_prompt_override(values, *, total_slots, dynamic_slots, merge, pre_calc):
    if merge:
        full = resolve_flat_strengths(values, total_slots, repeat_last_legacy_value=True)
        validate_component_strength_compatibility(full, network_merge=True, network_pre_calc=pre_calc)
        if dynamic_slots == 0:
            return ()
    selected = resolve_flat_strengths(values, dynamic_slots, repeat_last_legacy_value=True)
    validate_component_strength_compatibility(selected, network_merge=merge, network_pre_calc=pre_calc)
    return selected
