"""Run the optional Triton RMS and rounding kernels with the same Torch random sample as the reference."""
import torch

from library import triton_quant as kernels
from library.rounding_util import compute_scale_bits, fake_quantize_levels, signed_limit


def quantize_delta_triton(delta, *, bits, range_mul, observe, fused, collect_packed=None, rand=None):
    qmax = signed_limit(bits)
    with torch.no_grad():
        scale = kernels.triton_compute_scale_bits_channel_rms(delta, bits=bits, range_mul=range_mul, eps=1e-8)
        if scale is None:
            scale = compute_scale_bits(delta, bits=bits, range_mul=range_mul)
        if rand is None:
            rand = torch.rand_like(delta, dtype=torch.float32)
        if fused:
            result = kernels.triton_fake_quantize_levels_stoch_with_stats(delta, scale=scale, qmin=-qmax, qmax=qmax, rand=rand)
            if result is not None:
                rounded, packed = result
                if collect_packed is not None:
                    collect_packed(packed)
            else:
                rounded = None
        else:
            rounded = None
        used_fused = rounded is not None
        if rounded is None:
            rounded = kernels.triton_fake_quantize_levels_stoch(delta, scale=scale, qmin=-qmax, qmax=qmax, rand=rand)

    if rounded is None:
        return fake_quantize_levels(delta, scale=scale, qmax=qmax, rand=rand, observe=observe)
    output = delta + (rounded-delta).detach()
    if observe is not None and not used_fused:
        with torch.no_grad():
            observe(delta.detach(), output.detach(), delta.float() / scale, scale, qmax)
    return output
