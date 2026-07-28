"""Triton kernels for the rank-4 quantized LoRA-Up fast path.

This module is deliberately import-safe when Triton is not installed.  The
public entry point returns ``None`` whenever an input or device is outside the
validated dispatch set, allowing the caller to use the existing implementation
without consuming another random tensor.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception as e:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
    _TRITON_IMPORT_ERROR = e
else:
    _TRITON_IMPORT_ERROR = None


_VALIDATED_CHANNEL_COUNTS = frozenset({320, 640, 768, 1280, 2560, 3072, 5120, 10240})
_PERFORMANCE_DISPATCH = {
    # RTX 5080 / capability 12.0 / Triton 3.5.1. Each entry passed the
    # production-RNG forward+backward gate (>= 1.05x versus existing A/B).
    ((12, 0), "float16", "float32", "rows_1_128", "none"): _VALIDATED_CHANNEL_COUNTS,
    ((12, 0), "float16", "float32", "rows_129_512", "none"): _VALIDATED_CHANNEL_COUNTS,
    ((12, 0), "float16", "float32", "rows_513_2048", "none"): _VALIDATED_CHANNEL_COUNTS,
    ((12, 0), "float16", "float32", "rows_1_128", "basic"): _VALIDATED_CHANNEL_COUNTS,
    ((12, 0), "float16", "float32", "rows_129_512", "basic"): _VALIDATED_CHANNEL_COUNTS,
    ((12, 0), "float16", "float32", "rows_513_2048", "basic"): _VALIDATED_CHANNEL_COUNTS,
}
_SUPPORTED_CAPABILITIES = frozenset(key[0] for key in _PERFORMANCE_DISPATCH)
_SUPPORTED_CHANNEL_COUNTS = frozenset(
    channel
    for channels in _PERFORMANCE_DISPATCH.values()
    for channel in channels
)
_MAX_ROWS = 2048
_QMAX = 127.0
_RMS_EPS = 1.0e-8
_warned_messages: set[str] = set()
_failed_configs: set[tuple[Any, ...]] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_messages:
        return
    _warned_messages.add(key)
    logger.warning(message)


def is_triton_rank4_quantized_lora_up_available() -> bool:
    """Return whether Triton imported; device eligibility is checked per call."""

    return _TRITON_AVAILABLE


def _row_bucket(row_count: int) -> Optional[str]:
    if 0 < row_count <= 128:
        return "rows_1_128"
    if row_count <= 512:
        return "rows_129_512"
    if row_count <= _MAX_ROWS:
        return "rows_513_2048"
    return None


def _row_launch_config(row_count: int) -> Optional[tuple[int, int, int]]:
    # Keep the scale program near 8K values for the larger buckets to avoid
    # excessive register pressure while still reducing one whole channel.
    if 0 < row_count <= 128:
        return 128, 32, 4
    if row_count <= 512:
        return 512, 16, 8
    if row_count <= _MAX_ROWS:
        return 2048, 4, 8
    return None


def get_triton_rank4_quantized_lora_up_diagnostics(
    device: Optional[torch.device | str | int] = None,
) -> dict[str, Any]:
    """Return import, version, device-dispatch, and failure-cache information."""

    info: dict[str, Any] = {
        "triton_available": _TRITON_AVAILABLE,
        "triton_version": getattr(triton, "__version__", None) if _TRITON_AVAILABLE else None,
        "triton_import_error": None if _TRITON_IMPORT_ERROR is None else repr(_TRITON_IMPORT_ERROR),
        "supported_capabilities": sorted(_SUPPORTED_CAPABILITIES),
        "supported_channel_counts": sorted(_SUPPORTED_CHANNEL_COUNTS),
        "max_rows": _MAX_ROWS,
        "failed_config_count": len(_failed_configs),
        "performance_dispatch": [
            {
                "capability": capability,
                "activation_dtype": activation_dtype,
                "weight_dtype": weight_dtype,
                "row_bucket": row_bucket,
                "stats_mode": stats_mode,
                "channels": sorted(channels),
            }
            for (
                capability,
                activation_dtype,
                weight_dtype,
                row_bucket,
                stats_mode,
            ), channels in sorted(_PERFORMANCE_DISPATCH.items())
        ],
    }
    if device is not None and torch.cuda.is_available():
        try:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                info["device_capability"] = None
                info["device_supported"] = False
            else:
                capability = tuple(torch.cuda.get_device_capability(resolved))
                info["device_capability"] = capability
                info["device_supported"] = capability in _SUPPORTED_CAPABILITIES
        except Exception as e:
            info["device_error"] = repr(e)
            info["device_supported"] = False
    return info


if _TRITON_AVAILABLE:

    @triton.jit
    def _rank4_delta_values(
        z_ptr,
        weight_ptr,
        row_offsets,
        channel_offsets,
        row_count,
        channel_count,
        multiplier,
        lora_scale,
    ):
        mask = (row_offsets < row_count) & (channel_offsets < channel_count)
        z_base = row_offsets * 4
        w_base = channel_offsets * 4
        # Materialize the broadcasted pointer shape. This helper is shared by
        # the 2-D reduction tile and the 1-D output tile.
        z_base = z_base + channel_offsets * 0
        w_base = w_base + row_offsets * 0

        z0 = tl.load(z_ptr + z_base + 0, mask=mask, other=0.0).to(tl.float32)
        z1 = tl.load(z_ptr + z_base + 1, mask=mask, other=0.0).to(tl.float32)
        z2 = tl.load(z_ptr + z_base + 2, mask=mask, other=0.0).to(tl.float32)
        z3 = tl.load(z_ptr + z_base + 3, mask=mask, other=0.0).to(tl.float32)

        # Match CUDA autocast F.linear: the parameter is stored in FP32 but
        # multiplication consumes its FP16 cast. Accumulation stays FP32.
        w0 = tl.load(weight_ptr + w_base + 0, mask=mask, other=0.0).to(tl.float16).to(tl.float32)
        w1 = tl.load(weight_ptr + w_base + 1, mask=mask, other=0.0).to(tl.float16).to(tl.float32)
        w2 = tl.load(weight_ptr + w_base + 2, mask=mask, other=0.0).to(tl.float16).to(tl.float32)
        w3 = tl.load(weight_ptr + w_base + 3, mask=mask, other=0.0).to(tl.float16).to(tl.float32)

        up = z0 * w0
        up += z1 * w1
        up += z2 * w2
        up += z3 * w3
        up = up.to(tl.float16)
        tmp = (up * multiplier).to(tl.float16)
        return (tmp * lora_scale).to(tl.float16)

    @triton.jit
    def _rank4_lora_up_scale_kernel(
        z_ptr,
        weight_ptr,
        scale_ptr,
        row_count,
        channel_count,
        multiplier,
        lora_scale,
        range_mul,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.arange(0, BLOCK_R)[:, None]
        channel_vec = tl.program_id(axis=0) * BLOCK_C + tl.arange(0, BLOCK_C)
        channels = channel_vec[None, :]
        mask = (rows < row_count) & (channels < channel_count)
        delta = _rank4_delta_values(
            z_ptr,
            weight_ptr,
            rows,
            channels,
            row_count,
            channel_count,
            multiplier,
            lora_scale,
        ).to(tl.float32)
        sumsq = tl.sum(tl.where(mask, delta * delta, 0.0), axis=0)
        rms = tl.sqrt(sumsq / row_count + 1.0e-8)
        scale = rms * range_mul / 127.0
        tl.store(scale_ptr + channel_vec, scale, mask=channel_vec < channel_count)

    @triton.jit
    def _rank4_lora_up_quant_kernel(
        z_ptr,
        weight_ptr,
        scale_ptr,
        rand_ptr,
        out_ptr,
        row_count,
        channel_count,
        multiplier,
        lora_scale,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // channel_count
        channels = offsets - rows * channel_count
        delta = _rank4_delta_values(
            z_ptr,
            weight_ptr,
            rows,
            channels,
            row_count,
            channel_count,
            multiplier,
            lora_scale,
        ).to(tl.float32)

        scale = tl.load(scale_ptr + channels, mask=mask, other=1.0).to(tl.float32)
        y = tl.div_rn(delta, scale)
        q_floor = tl.floor(y)
        frac = y - q_floor
        probability = tl.minimum(tl.maximum(frac, 0.0), 1.0)
        random_value = tl.load(rand_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        q = q_floor + (random_value < probability).to(tl.float32)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        out = (q * scale).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _rank4_lora_up_quant_stats_kernel(
        z_ptr,
        weight_ptr,
        scale_ptr,
        rand_ptr,
        out_ptr,
        partial_stats_ptr,
        row_count,
        channel_count,
        multiplier,
        lora_scale,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // channel_count
        channels = offsets - rows * channel_count
        delta = _rank4_delta_values(
            z_ptr,
            weight_ptr,
            rows,
            channels,
            row_count,
            channel_count,
            multiplier,
            lora_scale,
        ).to(tl.float32)

        scale = tl.load(scale_ptr + channels, mask=mask, other=1.0).to(tl.float32)
        y = tl.div_rn(delta, scale)
        q_floor = tl.floor(y)
        frac = y - q_floor
        probability = tl.minimum(tl.maximum(frac, 0.0), 1.0)
        random_value = tl.load(rand_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        q = q_floor + (random_value < probability).to(tl.float32)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        out = (q * scale).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)

        # Match the existing dq_delta stats: x is the FP16-stored delta
        # promoted to FP32, and q is reloaded after the FP16 output store.
        out_stored = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        numel = tl.sum(mask.to(tl.float32), axis=0)
        clip_count = tl.sum(((tl.abs(y) >= 127.0) & mask).to(tl.float32), axis=0)
        sumsq = tl.sum(tl.where(mask, delta * delta, 0.0), axis=0)
        xq_sumsq = tl.sum(tl.where(mask, out_stored * out_stored, 0.0), axis=0)
        xxq_sum = tl.sum(tl.where(mask, delta * out_stored, 0.0), axis=0)

        base = tl.program_id(axis=0) * 5
        tl.store(partial_stats_ptr + base + 0, numel)
        tl.store(partial_stats_ptr + base + 1, clip_count)
        tl.store(partial_stats_ptr + base + 2, sumsq)
        tl.store(partial_stats_ptr + base + 3, xq_sumsq)
        tl.store(partial_stats_ptr + base + 4, xxq_sum)


def _check_inputs(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    *,
    collect_stats: bool,
) -> Optional[tuple[tuple[int, int], tuple[int, int, int], int, int]]:
    if not _TRITON_AVAILABLE:
        return None
    if not (z.is_cuda and weight.is_cuda and rand.is_cuda):
        return None
    if z.device != weight.device or z.device != rand.device:
        return None
    if z.ndim != 3 or z.shape[-1] != 4 or weight.ndim != 2 or weight.shape[1] != 4:
        return None
    if z.dtype != torch.float16 or weight.dtype != torch.float32 or rand.dtype != torch.float32:
        return None
    if not (z.is_contiguous() and weight.is_contiguous() and rand.is_contiguous()):
        return None

    row_count = z.shape[0] * z.shape[1]
    channel_count = weight.shape[0]
    if rand.shape != (*z.shape[:-1], channel_count):
        return None
    row_bucket = _row_bucket(row_count)
    launch = _row_launch_config(row_count)
    if row_bucket is None or launch is None:
        return None

    try:
        capability = tuple(torch.cuda.get_device_capability(z.device))
    except Exception:
        return None
    if capability not in _SUPPORTED_CAPABILITIES:
        _warn_once(
            f"unsupported_capability_{capability}",
            "Triton rank-4 Quantized LoRA-Up has not been validated for CUDA "
            f"capability {capability}; falling back.",
        )
        return None
    dispatch_key = (
        capability,
        "float16",
        "float32",
        row_bucket,
        "basic" if collect_stats else "none",
    )
    if channel_count not in _PERFORMANCE_DISPATCH.get(dispatch_key, ()):
        return None
    return capability, launch, row_count, channel_count


def _launch_rank4_quantized_lora_up(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    *,
    collect_basic_stats: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    checked = _check_inputs(z, weight, rand, collect_stats=collect_basic_stats)
    if checked is None:
        raise RuntimeError("rank-4 Quantized LoRA-Up received ineligible inputs")
    capability, (block_r, block_c, num_warps), row_count, channel_count = checked
    config_key = (
        capability,
        block_r,
        block_c,
        channel_count,
        z.dtype,
        weight.dtype,
        collect_basic_stats,
    )
    if config_key in _failed_configs:
        raise RuntimeError("rank-4 Quantized LoRA-Up configuration previously failed")

    scale = torch.empty((channel_count,), device=z.device, dtype=torch.float32)
    out = torch.empty((*z.shape[:-1], channel_count), device=z.device, dtype=torch.float16)
    n_elements = out.numel()
    _rank4_lora_up_scale_kernel[(triton.cdiv(channel_count, block_c),)](
        z,
        weight,
        scale,
        row_count,
        channel_count,
        float(multiplier),
        float(lora_scale),
        float(range_mul),
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )

    if collect_basic_stats:
        use_large_launch = n_elements >= 65536
        block_size = 1024 if use_large_launch else 256
        quant_num_warps = 2 if use_large_launch else 4
        n_blocks = triton.cdiv(n_elements, block_size)
        partial_stats = torch.empty((n_blocks, 5), device=z.device, dtype=torch.float32)
        _rank4_lora_up_quant_stats_kernel[(n_blocks,)](
            z,
            weight,
            scale,
            rand,
            out,
            partial_stats,
            row_count,
            channel_count,
            float(multiplier),
            float(lora_scale),
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=quant_num_warps,
        )
        return out, partial_stats.sum(dim=0)

    block_size = 256
    _rank4_lora_up_quant_kernel[(triton.cdiv(n_elements, block_size),)](
        z,
        weight,
        scale,
        rand,
        out,
        row_count,
        channel_count,
        float(multiplier),
        float(lora_scale),
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out, None


def _save_backward_context(ctx, z: torch.Tensor, weight: torch.Tensor, multiplier: float, lora_scale: float) -> None:
    ctx.save_for_backward(z, weight)
    ctx.multiplier = float(multiplier)
    ctx.lora_scale = float(lora_scale)


def _rank4_lora_up_backward(ctx, grad_output: torch.Tensor):
    z, weight = ctx.saved_tensors
    # Reverse the two FP16 scalar operations separately. Combining their
    # product changes rounding and does not match native autograd.
    grad_up = (grad_output * ctx.lora_scale).to(torch.float16)
    grad_up = (grad_up * ctx.multiplier).to(torch.float16)
    grad_2d = grad_up.reshape(-1, grad_up.shape[-1])
    z_2d = z.reshape(-1, 4)
    weight_fp16 = weight.to(torch.float16)
    grad_z = torch.mm(grad_2d, weight_fp16).reshape_as(z)
    grad_weight = torch.mm(grad_2d.transpose(0, 1), z_2d).to(torch.float32)
    return grad_z, grad_weight, None, None, None, None


class _Rank4QuantizedLoRAUp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        z: torch.Tensor,
        weight: torch.Tensor,
        rand: torch.Tensor,
        multiplier: float,
        lora_scale: float,
        range_mul: float,
    ) -> torch.Tensor:
        out, _ = _launch_rank4_quantized_lora_up(
            z,
            weight,
            rand,
            multiplier,
            lora_scale,
            range_mul,
            collect_basic_stats=False,
        )
        _save_backward_context(ctx, z, weight, multiplier, lora_scale)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        return _rank4_lora_up_backward(ctx, grad_output)


class _Rank4QuantizedLoRAUpWithStats(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        z: torch.Tensor,
        weight: torch.Tensor,
        rand: torch.Tensor,
        multiplier: float,
        lora_scale: float,
        range_mul: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, packed_stats = _launch_rank4_quantized_lora_up(
            z,
            weight,
            rand,
            multiplier,
            lora_scale,
            range_mul,
            collect_basic_stats=True,
        )
        if packed_stats is None:
            raise RuntimeError("rank-4 Quantized LoRA-Up stats kernel returned no statistics")
        _save_backward_context(ctx, z, weight, multiplier, lora_scale)
        ctx.mark_non_differentiable(packed_stats)
        return out, packed_stats

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor, grad_stats: Optional[torch.Tensor]):
        return _rank4_lora_up_backward(ctx, grad_output)


def triton_rank4_quantized_lora_up(
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    rand: torch.Tensor,
    collect_stats: bool = False,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Run C0 rank-4 LoRA-Up plus stochastic delta fake quantization.

    ``rand`` must be a caller-created, contiguous FP32 CUDA tensor with the
    output shape. It is read-only and is never regenerated here. With
    ``collect_stats=True``, the second result packs ``numel``, ``clip_count``,
    ``sumsq``, ``xq_sumsq``, and ``xxq_sum`` as a non-differentiable FP32
    tensor. Any unsupported input or synchronous Triton failure returns
    ``None``.
    """

    if not _TRITON_AVAILABLE:
        _warn_once(
            "triton_import",
            "Triton rank-4 Quantized LoRA-Up requested, but Triton is unavailable: "
            f"{_TRITON_IMPORT_ERROR}",
        )
        return None
    checked = _check_inputs(z, weight, rand, collect_stats=collect_stats)
    if checked is None:
        return None

    capability, (block_r, block_c, _), _, channel_count = checked
    config_key = (capability, block_r, block_c, channel_count, z.dtype, weight.dtype, collect_stats)
    if config_key in _failed_configs:
        return None

    try:
        function = _Rank4QuantizedLoRAUpWithStats if collect_stats else _Rank4QuantizedLoRAUp
        applied = function.apply(z, weight, rand, float(multiplier), float(lora_scale), float(range_mul))
    except Exception as e:
        _failed_configs.add(config_key)
        _warn_once(
            f"rank4_apply_{config_key}",
            f"Triton rank-4 Quantized LoRA-Up failed; falling back: {e}",
        )
        return None
    if collect_stats:
        return applied
    return applied, None


def triton_rank4_delta_quant(
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    rand: torch.Tensor,
    collect_basic_stats: bool = False,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Compatibility name used by ``networks.lora`` for the C0 route."""

    return triton_rank4_quantized_lora_up(
        z,
        weight,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
        rand=rand,
        collect_stats=collect_basic_stats,
    )
