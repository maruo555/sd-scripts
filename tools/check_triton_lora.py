from __future__ import annotations

import argparse
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import triton
except Exception as e:  # pragma: no cover - standalone CUDA diagnostic
    triton = None
    TRITON_IMPORT_ERROR = e
else:
    TRITON_IMPORT_ERROR = None

from library.rounding_util import compute_scale_bits
from library.triton_quant import (
    triton_compute_scale_bits_channel_rms,
    triton_fake_quantize_levels_stoch,
)
from library import triton_lora


def _resolve_c0() -> Callable[..., object]:
    for name in ("triton_rank4_delta_quant", "triton_rank4_quantized_lora_up"):
        fn = getattr(triton_lora, name, None)
        if fn is not None:
            return fn
    raise RuntimeError("library.triton_lora has no public C0 entry point")


C0 = _resolve_c0()
C0_PARAMETERS = inspect.signature(C0).parameters


@dataclass
class ForwardResult:
    name: str
    shape: tuple[int, int, int]
    output_dtype: str
    up_dtype: str
    first_scalar_dtype: str
    delta_dtype: str
    max_abs: float
    relative_l2: float
    output_mismatch_ratio: float
    integer_mismatch_ratio: float
    rand_unchanged: bool
    rng_state_unchanged: bool
    inputs_unchanged: bool
    output_is_fresh: bool
    finite: bool
    passed: bool


@dataclass
class GradientResult:
    name: str
    grad_z_dtype: str
    grad_weight_dtype: str
    grad_z_equal: bool
    grad_weight_equal: bool
    grad_z_max_abs: float
    grad_weight_max_abs: float
    passed: bool


def _call_c0(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    collect_basic_stats: bool = False,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    kwargs: dict[str, object] = {
        "multiplier": multiplier,
        "lora_scale": lora_scale,
        "range_mul": range_mul,
        "rand": rand,
    }
    if "eps" in C0_PARAMETERS:
        kwargs["eps"] = 1.0e-8
    if "collect_basic_stats" in C0_PARAMETERS:
        kwargs["collect_basic_stats"] = collect_basic_stats
    elif "collect_stats" in C0_PARAMETERS:
        kwargs["collect_stats"] = collect_basic_stats

    value = C0(z, weight, **kwargs)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value, None
    if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], torch.Tensor):
        raise RuntimeError(f"unexpected C0 return type: {type(value)!r}")
    return value


def _native_delta(
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
) -> tuple[torch.Tensor, tuple[torch.dtype, torch.dtype, torch.dtype]]:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        up = F.linear(z, weight)
        first_scalar = up * multiplier
        delta = first_scalar * lora_scale
    return delta, (up.dtype, first_scalar.dtype, delta.dtype)


def _baseline(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.dtype, torch.dtype, torch.dtype]]:
    delta, dtypes = _native_delta(
        z,
        weight,
        multiplier=multiplier,
        lora_scale=lora_scale,
    )
    with torch.no_grad():
        scale = triton_compute_scale_bits_channel_rms(
            delta,
            bits=8,
            range_mul=range_mul,
            eps=1.0e-8,
        )
    if scale is None:
        raise RuntimeError("existing Triton A (channel/RMS scale) returned None")
    raw_quantized = triton_fake_quantize_levels_stoch(
        delta.detach(),
        scale=scale,
        qmin=-127,
        qmax=127,
        rand=rand,
    )
    if raw_quantized is None:
        raise RuntimeError("existing Triton B (stochastic fake quant) returned None")
    quantized = delta + (raw_quantized - delta).detach()
    return quantized, scale, dtypes


def _c0_math_scale(
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
) -> torch.Tensor:
    # Mirrors the documented C0 arithmetic boundaries.  This scale is used
    # only to recover/report integer decisions; the output comparison uses the
    # production A/B baseline above as the oracle.
    z2d = z.detach().reshape(-1, 4).to(torch.float32)
    w = weight.detach().to(torch.float16).to(torch.float32)
    raw = torch.mm(z2d, w.transpose(0, 1)).to(torch.float16)
    raw = (raw * multiplier).to(torch.float16)
    raw = (raw * lora_scale).to(torch.float16)
    raw = raw.reshape(*z.shape[:-1], weight.shape[0])
    return (
        torch.sqrt(torch.mean(raw.to(torch.float32).square(), dim=(0, 1), keepdim=True) + 1.0e-8)
        * range_mul
        / 127.0
    ).to(torch.float32)


def _make_inputs(
    *,
    rows: int,
    channels: int,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    if mode == "zero":
        z = torch.zeros((1, rows, 4), device="cuda", dtype=torch.float16)
        weight = torch.randn(
            (channels, 4), device="cuda", dtype=torch.float32, generator=generator
        ).mul_(0.02)
    elif mode == "extreme":
        z_sign = torch.randint(
            0, 2, (1, rows, 4), device="cuda", dtype=torch.int32, generator=generator
        )
        weight_sign = torch.randint(
            0, 2, (channels, 4), device="cuda", dtype=torch.int32, generator=generator
        )
        z = (z_sign.to(torch.float32).mul_(2).sub_(1).mul_(64)).to(torch.float16)
        weight = weight_sign.to(torch.float32).mul_(2).sub_(1).mul_(128)
    else:
        z = (
            torch.randn(
                (1, rows, 4), device="cuda", dtype=torch.float32, generator=generator
            )
            * 0.08
        ).to(torch.float16)
        weight = (
            torch.randn(
                (channels, 4), device="cuda", dtype=torch.float32, generator=generator
            )
            * 0.04
        )
    rand = torch.rand(
        (1, rows, channels), device="cuda", dtype=torch.float32, generator=generator
    ).contiguous()
    return z.contiguous(), weight.contiguous(), rand


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(denominator.to(torch.float32)).item())
    return float(torch.linalg.vector_norm(numerator.to(torch.float32)).item()) / max(denom, 1.0e-30)


def run_forward_case(
    name: str,
    *,
    rows: int,
    channels: int,
    mode: str,
    seed: int,
) -> tuple[Optional[ForwardResult], Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    multiplier, lora_scale, range_mul = 0.75, 1.25, 3.0
    z, weight, rand = _make_inputs(rows=rows, channels=channels, mode=mode, seed=seed)
    z_before = z.clone()
    weight_before = weight.clone()
    rand_before = rand.clone()
    reference, reference_scale, dtypes = _baseline(
        z,
        weight,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    rng_state_before = torch.cuda.get_rng_state()
    c0 = _call_c0(
        z,
        weight,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    rng_state_unchanged = bool(torch.equal(torch.cuda.get_rng_state(), rng_state_before))
    if c0 is None:
        return None, (z, weight, rand)

    output, packed_stats = c0
    if packed_stats is not None and packed_stats.requires_grad:
        raise AssertionError("C0 packed statistics must be non-differentiable")
    torch.cuda.synchronize()
    diff = output.to(torch.float32) - reference.to(torch.float32)
    output_mismatch = float((output != reference).to(torch.float32).mean().item())
    c0_scale = _c0_math_scale(
        z,
        weight,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    ref_level = torch.round(reference.to(torch.float32) / reference_scale)
    c0_level = torch.round(output.to(torch.float32) / c0_scale)
    integer_mismatch = float((ref_level != c0_level).to(torch.float32).mean().item())
    max_abs = float(diff.abs().max().item())
    relative_l2 = _safe_ratio(diff, reference)
    rand_unchanged = bool(torch.equal(rand, rand_before))
    inputs_unchanged = bool(torch.equal(z, z_before) and torch.equal(weight, weight_before))
    output_is_fresh = all(
        output.data_ptr() != tensor.data_ptr()
        for tensor in (z, weight, rand)
    )
    finite = bool(torch.isfinite(output).all())

    # C0 deliberately uses rank-specialized arithmetic, so exact equality to
    # cuBLAS is not required.  These bounds detect wrong layouts/scales or
    # quantizer levels while tolerating normal reduction-order differences.
    if mode == "zero":
        numerical_ok = max_abs == 0.0 and integer_mismatch == 0.0
    else:
        numerical_ok = relative_l2 <= 0.03 and integer_mismatch <= 0.10
    passed = (
        output.dtype == torch.float16
        and output.shape == reference.shape
        and rand_unchanged
        and rng_state_unchanged
        and inputs_unchanged
        and output_is_fresh
        and finite
        and numerical_ok
    )
    result = ForwardResult(
        name=name,
        shape=tuple(z.shape),
        output_dtype=str(output.dtype).replace("torch.", ""),
        up_dtype=str(dtypes[0]).replace("torch.", ""),
        first_scalar_dtype=str(dtypes[1]).replace("torch.", ""),
        delta_dtype=str(dtypes[2]).replace("torch.", ""),
        max_abs=max_abs,
        relative_l2=relative_l2,
        output_mismatch_ratio=output_mismatch,
        integer_mismatch_ratio=integer_mismatch,
        rand_unchanged=rand_unchanged,
        rng_state_unchanged=rng_state_unchanged,
        inputs_unchanged=inputs_unchanged,
        output_is_fresh=output_is_fresh,
        finite=finite,
        passed=passed,
    )
    return result, (z, weight, rand)


def run_gradient_case(
    name: str,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    seed: int,
) -> GradientResult:
    multiplier, lora_scale, range_mul = 0.75, 1.25, 3.0
    z0, weight0, rand = inputs
    z_ref = z0.detach().clone().requires_grad_(True)
    weight_ref = weight0.detach().clone().requires_grad_(True)
    z_c0 = z0.detach().clone().requires_grad_(True)
    weight_c0 = weight0.detach().clone().requires_grad_(True)

    ref, _, _ = _baseline(
        z_ref,
        weight_ref,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    c0_value = _call_c0(
        z_c0,
        weight_c0,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    if c0_value is None:
        raise RuntimeError(f"C0 became unavailable during gradient case {name}")
    c0 = c0_value[0]

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    grad_output = torch.randn(
        c0.shape, device="cuda", dtype=torch.float16, generator=generator
    ).contiguous()
    grad_z_ref, grad_weight_ref = torch.autograd.grad(
        ref, (z_ref, weight_ref), grad_outputs=grad_output
    )
    grad_z_c0, grad_weight_c0 = torch.autograd.grad(
        c0, (z_c0, weight_c0), grad_outputs=grad_output
    )
    torch.cuda.synchronize()
    z_diff = (grad_z_c0.to(torch.float32) - grad_z_ref.to(torch.float32)).abs()
    weight_diff = (grad_weight_c0 - grad_weight_ref).abs()
    z_equal = bool(torch.equal(grad_z_c0, grad_z_ref))
    weight_equal = bool(torch.equal(grad_weight_c0, grad_weight_ref))
    return GradientResult(
        name=name,
        grad_z_dtype=str(grad_z_c0.dtype).replace("torch.", ""),
        grad_weight_dtype=str(grad_weight_c0.dtype).replace("torch.", ""),
        grad_z_equal=z_equal,
        grad_weight_equal=weight_equal,
        grad_z_max_abs=float(z_diff.max().item()),
        grad_weight_max_abs=float(weight_diff.max().item()),
        passed=(
            z_equal
            and weight_equal
            and grad_z_c0.dtype == torch.float16
            and grad_weight_c0.dtype == torch.float32
        ),
    )


def run_fallback_checks() -> list[tuple[str, bool, bool, bool]]:
    z, weight, rand = _make_inputs(rows=32, channels=640, mode="random", seed=9901)
    cases: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = [
        ("fp16_weight", z, weight.to(torch.float16).contiguous(), rand),
        ("rank5", torch.randn((1, 32, 5), device="cuda", dtype=torch.float16), weight, rand),
        ("two_dimensional_z", z.reshape(-1, 4), weight, rand),
        (
            "too_many_rows",
            torch.randn((1, 2049, 4), device="cuda", dtype=torch.float16),
            weight,
            torch.rand((1, 2049, 640), device="cuda", dtype=torch.float32),
        ),
        (
            "unsupported_channels",
            z,
            torch.randn((7, 4), device="cuda", dtype=torch.float32),
            torch.rand((1, 32, 7), device="cuda", dtype=torch.float32),
        ),
    ]
    cpu_z = torch.randn((1, 8, 4), dtype=torch.float16)
    cpu_weight = torch.randn((640, 4), dtype=torch.float32)
    cpu_rand = torch.rand((1, 8, 640), dtype=torch.float32)
    cases.append(("cpu", cpu_z, cpu_weight, cpu_rand))

    results: list[tuple[str, bool, bool, bool]] = []
    for name, case_z, case_weight, case_rand in cases:
        case_rand = case_rand.contiguous()
        rand_before = case_rand.clone()
        rng_state_before = torch.cuda.get_rng_state() if case_rand.is_cuda else None
        value = _call_c0(
            case_z,
            case_weight,
            case_rand,
            multiplier=0.75,
            lora_scale=1.25,
            range_mul=3.0,
        )
        rng_unchanged = (
            True
            if rng_state_before is None
            else bool(torch.equal(torch.cuda.get_rng_state(), rng_state_before))
        )
        results.append(
            (name, value is None, bool(torch.equal(case_rand, rand_before)), rng_unchanged)
        )
    return results


def run_eps_zero_check() -> tuple[bool, float, float]:
    z, weight, rand = _make_inputs(rows=32, channels=640, mode="zero", seed=9911)
    delta, _ = _native_delta(z, weight, multiplier=0.75, lora_scale=1.25)
    scale = compute_scale_bits(
        delta,
        bits=8,
        granularity="channel",
        stat="rms",
        range_mul=3.0,
        eps=1.0e-8,
        use_triton=False,
    )
    expected = math.sqrt(1.0e-8) * 3.0 / 127.0
    scale_error = float((scale - expected).abs().max().item())
    value = _call_c0(
        z,
        weight,
        rand,
        multiplier=0.75,
        lora_scale=1.25,
        range_mul=3.0,
    )
    c0_max = math.inf if value is None else float(value[0].abs().max().item())
    return value is not None and scale_error <= 1.0e-12 and c0_max == 0.0, scale_error, c0_max


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and fallback checks for rank-4 Quantized LoRA-Up C0"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run two random forward cases plus one gradient case",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2
    if triton is None:
        print(f"ERROR: Triton import failed: {TRITON_IMPORT_ERROR}", file=sys.stderr)
        return 2

    diagnostics = triton_lora.get_triton_rank4_quantized_lora_up_diagnostics("cuda")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__} "
        f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
        f"dispatch_supported={diagnostics.get('device_supported')}"
    )

    cases = [
        ("random_small", 64, 640, "random"),
        ("random_typical", 468, 1280, "random"),
    ]
    if not args.quick:
        cases.extend(
            [
                ("random_wide", 468, 10240, "random"),
                ("random_row_limit", 2048, 640, "random"),
                ("zero", 32, 640, "zero"),
                ("extreme", 64, 640, "extreme"),
            ]
        )

    failures = 0
    successes = 0
    successful_inputs: list[tuple[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    print(
        "forward,name,shape,output_dtype,up_dtype,first_scalar_dtype,delta_dtype,"
        "max_abs,relative_l2,output_mismatch_ratio,integer_mismatch_ratio,"
        "rand_unchanged,rng_state_unchanged,inputs_unchanged,output_is_fresh,finite,passed"
    )
    for index, (name, rows, channels, mode) in enumerate(cases):
        result, inputs = run_forward_case(
            name,
            rows=rows,
            channels=channels,
            mode=mode,
            seed=1200 + index,
        )
        if result is None:
            print(f"forward,{name},C0_UNAVAILABLE")
            continue
        successes += 1
        if inputs is not None:
            successful_inputs.append((name, inputs))
        if not result.passed:
            failures += 1
        print(
            f"forward,{result.name},{result.shape},{result.output_dtype},{result.up_dtype},"
            f"{result.first_scalar_dtype},{result.delta_dtype},{result.max_abs:.9g},"
            f"{result.relative_l2:.9g},{result.output_mismatch_ratio:.9g},"
            f"{result.integer_mismatch_ratio:.9g},{result.rand_unchanged},"
            f"{result.rng_state_unchanged},{result.inputs_unchanged},"
            f"{result.output_is_fresh},{result.finite},{result.passed}"
        )

    print(
        "gradient,name,grad_z_dtype,grad_weight_dtype,grad_z_equal,grad_weight_equal,"
        "grad_z_max_abs,grad_weight_max_abs,passed"
    )
    gradient_cases = successful_inputs[:1] if args.quick else successful_inputs[:2]
    for index, (name, inputs) in enumerate(gradient_cases):
        result = run_gradient_case(name, inputs, seed=2200 + index)
        if not result.passed:
            failures += 1
        print(
            f"gradient,{result.name},{result.grad_z_dtype},{result.grad_weight_dtype},"
            f"{result.grad_z_equal},{result.grad_weight_equal},{result.grad_z_max_abs:.9g},"
            f"{result.grad_weight_max_abs:.9g},{result.passed}"
        )

    print("fallback,name,returned_none,rand_unchanged,rng_state_unchanged,passed")
    for name, returned_none, rand_unchanged, rng_state_unchanged in run_fallback_checks():
        passed = returned_none and rand_unchanged and rng_state_unchanged
        failures += 0 if passed else 1
        print(
            f"fallback,{name},{returned_none},{rand_unchanged},"
            f"{rng_state_unchanged},{passed}"
        )

    eps_ok, scale_error, zero_max = run_eps_zero_check()
    failures += 0 if eps_ok else 1
    print(
        f"eps_zero,eps=1e-8,scale_max_abs_error={scale_error:.9g},"
        f"c0_output_max_abs={zero_max:.9g},passed={eps_ok}"
    )

    # The production C0 route supports packed basic stats and keeps them out
    # of autograd.
    z, weight, rand = _make_inputs(rows=32, channels=640, mode="random", seed=9921)
    stats_value = _call_c0(
        z.requires_grad_(True),
        weight.requires_grad_(True),
        rand,
        multiplier=0.75,
        lora_scale=1.25,
        range_mul=3.0,
        collect_basic_stats=True,
    )
    stats_supported = stats_value is not None and stats_value[1] is not None
    stats_detached = (
        stats_supported
        and not stats_value[1].requires_grad
        and stats_value[1].grad_fn is None
    )
    failures += 0 if (stats_supported and stats_detached) else 1
    print(f"basic_stats,supported={stats_supported},non_differentiable={stats_detached}")

    if successes == 0:
        print("ERROR: C0 did not succeed for any correctness case", file=sys.stderr)
        return 3
    print(f"summary,c0_successes={successes},failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
