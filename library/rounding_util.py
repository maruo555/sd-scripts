"""Stochastic fake quantization for an SDXL LoRA output delta."""
from collections.abc import Callable

import torch

# delta, STE output, pre-rounding delta/scale, FP32 scale, signed limit.
QuantObserver = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], None]


CHANNEL_REDUCTION_AXES = {2: (0,), 3: (0, 1), 4: (0, 2, 3)}


def signed_limit(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"delta quantization needs at least 2 bits, got {bits}")
    return (1 << (bits - 1)) - 1


@torch.no_grad()
def compute_scale_bits(delta: torch.Tensor, *, bits: int, range_mul: float) -> torch.Tensor:
    """Channel RMS scale for NC, NLC and NCHW. Other layouts use one tensor RMS."""
    qmax = signed_limit(bits)
    squared = delta.float().square()
    axes = CHANNEL_REDUCTION_AXES.get(delta.ndim)
    mean_square = squared.mean(dim=axes, keepdim=True) if axes is not None else squared.mean()
    return (mean_square + 1e-8).sqrt() * range_mul / qmax


def fake_quantize_levels(
    delta: torch.Tensor,
    *,
    scale: torch.Tensor,
    qmax: int,
    rand: torch.Tensor | None = None,
    observe: QuantObserver | None = None,
) -> torch.Tensor:
    """Round once on a symmetric grid and pass the original gradient through."""
    with torch.no_grad():
        normalized = delta.float() / scale
        if rand is None:
            rand = torch.rand_like(normalized)
        elif rand.shape != delta.shape:
            raise ValueError(f"Rounding random tensor {rand.shape} must match delta {delta.shape}")
        else:
            rand = rand.to(device=delta.device, dtype=torch.float32)
        lower = normalized.floor()
        level = lower + (rand < (normalized - lower))
        rounded = (level.clamp(-qmax, qmax) * scale).to(delta.dtype)

    result = delta + (rounded - delta).detach()
    if observe is not None:
        with torch.no_grad():
            observe(delta.detach(), result.detach(), normalized, scale.detach(), qmax)
    return result


def quantize_delta(
    delta: torch.Tensor,
    *,
    bits: int,
    range_mul: float,
    observe: QuantObserver | None = None,
    rand: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply channel RMS stochastic rounding to the scaled LoRA delta before addition."""
    scale = compute_scale_bits(delta, bits=bits, range_mul=range_mul)
    return fake_quantize_levels(delta, scale=scale, qmax=signed_limit(bits), rand=rand, observe=observe)
