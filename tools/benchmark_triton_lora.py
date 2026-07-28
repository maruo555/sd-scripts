from __future__ import annotations

import argparse
import inspect
import math
import statistics
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
except Exception as e:  # pragma: no cover - standalone CUDA benchmark
    triton = None
    TRITON_IMPORT_ERROR = e
else:
    TRITON_IMPORT_ERROR = None

from library.triton_quant import (
    triton_compute_scale_bits_channel_rms,
    triton_fake_quantize_levels_stoch,
    triton_fake_quantize_levels_stoch_with_stats,
)
from library import triton_lora


DEFAULT_ROWS = (64, 256, 468, 1024, 2048)
DEFAULT_CHANNELS = (320, 640, 768, 1280, 2560, 3072, 5120, 10240)
QUICK_CASES = ((64, 640), (468, 1280), (468, 10240))


def _resolve_c0() -> Callable[..., object]:
    for name in ("triton_rank4_delta_quant", "triton_rank4_quantized_lora_up"):
        fn = getattr(triton_lora, name, None)
        if fn is not None:
            return fn
    raise RuntimeError("library.triton_lora has no public C0 entry point")


C0 = _resolve_c0()
C0_PARAMETERS = inspect.signature(C0).parameters


@dataclass
class TimingPair:
    baseline_samples: list[float]
    c0_samples: list[float]

    @property
    def baseline_median(self) -> float:
        return statistics.median(self.baseline_samples)

    @property
    def c0_median(self) -> float:
        return statistics.median(self.c0_samples)

    @property
    def speedup(self) -> float:
        return self.baseline_median / self.c0_median


def parse_case(value: str) -> tuple[int, int]:
    try:
        rows, channels = (int(part.strip()) for part in value.split(","))
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError("case must be ROWS,CHANNELS") from e
    if rows <= 0 or channels <= 0:
        raise argparse.ArgumentTypeError("ROWS and CHANNELS must be positive")
    return rows, channels


def _call_c0(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    collect_basic_stats: bool,
) -> Optional[torch.Tensor]:
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
    return value if isinstance(value, torch.Tensor) else value[0]


def _baseline(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: Optional[torch.Tensor],
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
    collect_basic_stats: bool,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        up = F.linear(z, weight)
        delta = (up * multiplier) * lora_scale
    with torch.no_grad():
        scale = triton_compute_scale_bits_channel_rms(
            delta,
            bits=8,
            range_mul=range_mul,
            eps=1.0e-8,
        )
    if scale is None:
        raise RuntimeError("existing Triton A (channel/RMS scale) returned None")
    if collect_basic_stats:
        fused = triton_fake_quantize_levels_stoch_with_stats(
            delta.detach(),
            scale=scale,
            qmin=-127,
            qmax=127,
            rand=rand,
        )
        if fused is None:
            raise RuntimeError("existing Triton B+stats returned None")
        raw_quantized, _ = fused
    else:
        raw_quantized = triton_fake_quantize_levels_stoch(
            delta.detach(),
            scale=scale,
            qmin=-127,
            qmax=127,
            rand=rand,
        )
        if raw_quantized is None:
            raise RuntimeError("existing Triton B (stochastic fake quant) returned None")
    return delta + (raw_quantized - delta).detach()


def _time_cuda(fn: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _benchmark_pair(
    baseline_fn: Callable[[], object],
    c0_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> TimingPair:
    for _ in range(warmup):
        baseline_fn()
        c0_fn()
    torch.cuda.synchronize()

    baseline_samples: list[float] = []
    c0_samples: list[float] = []
    for repeat in range(repeats):
        # Alternate order to make thermal/clock drift affect both routes.
        if repeat % 2 == 0:
            baseline_samples.append(_time_cuda(baseline_fn, iterations))
            c0_samples.append(_time_cuda(c0_fn, iterations))
        else:
            c0_samples.append(_time_cuda(c0_fn, iterations))
            baseline_samples.append(_time_cuda(baseline_fn, iterations))
    return TimingPair(baseline_samples, c0_samples)


def _make_case(
    rows: int,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1000003 + rows * 101 + channels)
    z = (
        torch.randn((1, rows, 4), device="cuda", dtype=torch.float32, generator=generator)
        * 0.08
    ).to(torch.float16).contiguous().requires_grad_(True)
    weight = (
        torch.randn((channels, 4), device="cuda", dtype=torch.float32, generator=generator)
        * 0.04
    ).contiguous().requires_grad_(True)
    rand = torch.rand(
        (1, rows, channels), device="cuda", dtype=torch.float32, generator=generator
    ).contiguous()
    grad_output = torch.randn(
        (1, rows, channels), device="cuda", dtype=torch.float16, generator=generator
    ).contiguous()
    return z, weight, rand, grad_output


def run_case(
    rows: int,
    channels: int,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    production_rng: bool,
    collect_basic_stats: bool,
) -> Optional[tuple[TimingPair, TimingPair]]:
    multiplier, lora_scale, range_mul = 0.75, 1.25, 3.0
    z, weight, rand, grad_output = _make_case(rows, channels)

    probe = _call_c0(
        z,
        weight,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
        collect_basic_stats=collect_basic_stats,
    )
    if probe is None:
        return None
    del probe

    def baseline_forward():
        return _baseline(
            z,
            weight,
            None if production_rng else rand,
            multiplier=multiplier,
            lora_scale=lora_scale,
            range_mul=range_mul,
            collect_basic_stats=collect_basic_stats,
        )

    def c0_forward():
        c0_rand = (
            torch.rand(
                (1, rows, channels),
                device=z.device,
                dtype=torch.float32,
            ).contiguous()
            if production_rng
            else rand
        )
        value = _call_c0(
            z,
            weight,
            c0_rand,
            multiplier=multiplier,
            lora_scale=lora_scale,
            range_mul=range_mul,
            collect_basic_stats=collect_basic_stats,
        )
        if value is None:
            raise RuntimeError("C0 became unavailable after the benchmark probe")
        return value

    def baseline_forward_backward():
        value = baseline_forward()
        return torch.autograd.grad(
            value,
            (z, weight),
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
        )

    def c0_forward_backward():
        value = c0_forward()
        return torch.autograd.grad(
            value,
            (z, weight),
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
        )

    forward = _benchmark_pair(
        baseline_forward,
        c0_forward,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    forward_backward = _benchmark_pair(
        baseline_forward_backward,
        c0_forward_backward,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    return forward, forward_backward


def run_memory_mode(
    rows: int,
    channels: int,
    mode: str,
    *,
    production_rng: bool,
    collect_basic_stats: bool,
) -> int:
    multiplier, lora_scale, range_mul = 0.75, 1.25, 3.0
    z, weight, rand, grad_output = _make_case(rows, channels)

    def baseline_forward_backward():
        value = _baseline(
            z,
            weight,
            None if production_rng else rand,
            multiplier=multiplier,
            lora_scale=lora_scale,
            range_mul=range_mul,
            collect_basic_stats=collect_basic_stats,
        )
        return torch.autograd.grad(value, (z, weight), grad_outputs=grad_output)

    def c0_forward_backward():
        c0_rand = (
            torch.rand(
                (1, rows, channels),
                device=z.device,
                dtype=torch.float32,
            ).contiguous()
            if production_rng
            else rand
        )
        value = _call_c0(
            z,
            weight,
            c0_rand,
            multiplier=multiplier,
            lora_scale=lora_scale,
            range_mul=range_mul,
            collect_basic_stats=collect_basic_stats,
        )
        if value is None:
            raise RuntimeError("C0 is unavailable for the requested memory case")
        return torch.autograd.grad(value, (z, weight), grad_outputs=grad_output)

    fn = baseline_forward_backward if mode == "baseline" else c0_forward_backward
    # Compile/warm before measuring. Run baseline and C0 in separate process
    # invocations so allocator state is not inherited from the other route.
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    fn()
    torch.cuda.synchronize()
    allocated_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    print(
        f"memory,mode={mode},rows={rows},channels={channels},"
        f"rng_mode={'production' if production_rng else 'fixed'},"
        f"stats_mode={'basic' if collect_basic_stats else 'none'},"
        f"allocated_before_bytes={allocated_before},"
        f"peak_allocated_bytes={allocated_peak},"
        f"peak_allocated_delta_bytes={allocated_peak - allocated_before},"
        f"reserved_before_bytes={reserved_before},"
        f"peak_reserved_bytes={reserved_peak},"
        f"peak_reserved_delta_bytes={reserved_peak - reserved_before}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CUDA-event benchmark: existing Triton A/B LoRA-Up baseline vs "
            "rank-4 Quantized LoRA-Up C0"
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="ROWS,CHANNELS; repeatable (default: the full row/channel grid)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three representative cases and 3/10/3 timing",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("baseline", "c0"),
        help=(
            "Measure one forward+backward route and exit. Invoke baseline and "
            "c0 in separate processes; peak allocated is formal and peak "
            "reserved is reference-only."
        ),
    )
    parser.add_argument(
        "--fixed-rand",
        action="store_true",
        help=(
            "Reuse one pre-generated rand tensor for kernel-only timing. "
            "By default each route includes its production torch.rand allocation."
        ),
    )
    parser.add_argument(
        "--basic-stats",
        action="store_true",
        help=(
            "Benchmark the packed basic-stats variants: existing Triton B+stats "
            "versus C0 collect_basic_stats=True."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2
    if triton is None:
        print(f"ERROR: Triton import failed: {TRITON_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and iterations/repeats must be > 0")

    if args.case:
        cases = args.case
    elif args.quick:
        cases = list(QUICK_CASES)
    else:
        cases = [(rows, channels) for rows in DEFAULT_ROWS for channels in DEFAULT_CHANNELS]

    warmup, iterations, repeats = args.warmup, args.iterations, args.repeats
    if args.quick:
        warmup, iterations, repeats = 3, 10, 3

    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__} "
        f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
        f"warmup={warmup} iterations={iterations} repeats={repeats}"
    )
    if args.memory_mode:
        if len(cases) != 1:
            print("ERROR: --memory-mode requires exactly one --case", file=sys.stderr)
            return 2
        return run_memory_mode(
            *cases[0],
            args.memory_mode,
            production_rng=not args.fixed_rand,
            collect_basic_stats=args.basic_stats,
        )

    print(
        f"rng_mode={'fixed' if args.fixed_rand else 'production'},"
        f"stats_mode={'basic' if args.basic_stats else 'none'}"
    )
    print(
        "rows,channels,numel,operation,baseline_median_ms,c0_median_ms,"
        "speedup,baseline_samples_ms,c0_samples_ms"
    )
    successes = 0
    for rows, channels in cases:
        result = run_case(
            rows,
            channels,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            production_rng=not args.fixed_rand,
            collect_basic_stats=args.basic_stats,
        )
        if result is None:
            print(f"{rows},{channels},{rows * channels},C0_UNAVAILABLE,,,,,")
            continue
        successes += 1
        forward, forward_backward = result
        for name, timing in (
            ("forward", forward),
            ("forward_backward", forward_backward),
        ):
            baseline_samples = "|".join(f"{value:.9g}" for value in timing.baseline_samples)
            c0_samples = "|".join(f"{value:.9g}" for value in timing.c0_samples)
            print(
                f"{rows},{channels},{rows * channels},{name},"
                f"{timing.baseline_median:.9g},{timing.c0_median:.9g},"
                f"{timing.speedup:.9g},{baseline_samples},{c0_samples}"
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if successes == 0:
        print("ERROR: C0 did not succeed for any benchmark case", file=sys.stderr)
        return 3
    print(f"summary,c0_successful_cases={successes},total_cases={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
