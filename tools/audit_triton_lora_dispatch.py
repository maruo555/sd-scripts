from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library import triton_lora
from tools import benchmark_triton_lora as benchmark


TE_ROWS = (77, 154, 231, 308)
TE_CHANNELS = (768, 1280, 3072, 5120)

# Include the lower/upper edges of every current row bucket as well as the
# representative production shapes used by the existing benchmark.
DISPATCH_ROWS = (
    1,
    32,
    64,
    77,
    127,
    128,
    129,
    154,
    256,
    468,
    511,
    512,
    513,
    768,
    1024,
    1536,
    2048,
)
DISPATCH_CHANNELS = tuple(sorted(triton_lora._SUPPORTED_CHANNEL_COUNTS))


@dataclass(frozen=True)
class AuditCase:
    rows: int
    channels: int
    groups: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    case: AuditCase
    stats_mode: str
    available: bool
    forward_baseline_ms: Optional[float] = None
    forward_c0_ms: Optional[float] = None
    forward_speedup: Optional[float] = None
    train_baseline_ms: Optional[float] = None
    train_c0_ms: Optional[float] = None
    train_speedup: Optional[float] = None
    verdict: str = "UNAVAILABLE"


def _add_cases(
    grouped: dict[tuple[int, int], set[str]],
    rows: Iterable[int],
    channels: Iterable[int],
    group: str,
) -> None:
    for row_count in rows:
        for channel_count in channels:
            grouped.setdefault((row_count, channel_count), set()).add(group)


def build_cases(suite: str) -> list[AuditCase]:
    grouped: dict[tuple[int, int], set[str]] = {}
    if suite in ("te", "all"):
        _add_cases(grouped, TE_ROWS, TE_CHANNELS, "te")
    if suite in ("dispatch", "all"):
        _add_cases(grouped, DISPATCH_ROWS, DISPATCH_CHANNELS, "dispatch")
    return [
        AuditCase(rows, channels, tuple(sorted(groups)))
        for (rows, channels), groups in sorted(grouped.items())
    ]


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _verdict(speedup: float, gate: float) -> str:
    if speedup < 1.0:
        return "REGRESSION"
    if speedup < gate:
        return "BELOW_GATE"
    return "PASS"


def run_audit_case(
    case: AuditCase,
    *,
    stats_mode: str,
    warmup: int,
    iterations: int,
    repeats: int,
    gate: float,
) -> AuditResult:
    measured = benchmark.run_case(
        case.rows,
        case.channels,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        production_rng=True,
        collect_basic_stats=stats_mode == "basic",
    )
    if measured is None:
        return AuditResult(case=case, stats_mode=stats_mode, available=False)

    forward, train = measured
    forward_baseline_ms = _median(forward.baseline_samples)
    forward_c0_ms = _median(forward.c0_samples)
    train_baseline_ms = _median(train.baseline_samples)
    train_c0_ms = _median(train.c0_samples)
    train_speedup = train_baseline_ms / train_c0_ms
    return AuditResult(
        case=case,
        stats_mode=stats_mode,
        available=True,
        forward_baseline_ms=forward_baseline_ms,
        forward_c0_ms=forward_c0_ms,
        forward_speedup=forward_baseline_ms / forward_c0_ms,
        train_baseline_ms=train_baseline_ms,
        train_c0_ms=train_c0_ms,
        train_speedup=train_speedup,
        verdict=_verdict(train_speedup, gate),
    )


def _write_csv(path: Path, results: list[AuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "groups",
                "rows",
                "channels",
                "stats_mode",
                "available",
                "forward_baseline_ms",
                "forward_c0_ms",
                "forward_speedup",
                "train_baseline_ms",
                "train_c0_ms",
                "train_speedup",
                "verdict",
            )
        )
        for result in results:
            writer.writerow(
                (
                    "+".join(result.case.groups),
                    result.case.rows,
                    result.case.channels,
                    result.stats_mode,
                    result.available,
                    result.forward_baseline_ms,
                    result.forward_c0_ms,
                    result.forward_speedup,
                    result.train_baseline_ms,
                    result.train_c0_ms,
                    result.train_speedup,
                    result.verdict,
                )
            )


def _print_result(result: AuditResult) -> None:
    group = "+".join(result.case.groups)
    if not result.available:
        print(
            f"{group}: rows={result.case.rows} channels={result.case.channels} "
            f"stats={result.stats_mode} UNAVAILABLE"
        )
        return
    print(
        f"{group}: rows={result.case.rows} channels={result.case.channels} "
        f"stats={result.stats_mode} train_speedup={result.train_speedup:.4f}x "
        f"forward_speedup={result.forward_speedup:.4f}x {result.verdict}"
    )


def _print_summary(results: list[AuditResult], *, gate: float, top: int) -> None:
    available = [result for result in results if result.available]
    unavailable = len(results) - len(available)
    regressions = [result for result in available if result.verdict == "REGRESSION"]
    below_gate = [result for result in available if result.verdict == "BELOW_GATE"]
    passing = [result for result in available if result.verdict == "PASS"]
    print(
        "summary: "
        f"total={len(results)} available={len(available)} unavailable={unavailable} "
        f"pass_at_{gate:.3f}x={len(passing)} below_gate={len(below_gate)} "
        f"regressions={len(regressions)}"
    )
    if not available:
        return

    speedups = [float(result.train_speedup) for result in available]
    print(
        "train_speedup: "
        f"min={min(speedups):.4f}x median={statistics.median(speedups):.4f}x "
        f"max={max(speedups):.4f}x"
    )
    print(f"worst_{min(top, len(available))}:")
    for result in sorted(available, key=lambda item: float(item.train_speedup))[:top]:
        _print_result(result)

    for group in ("te", "dispatch"):
        group_results = [result for result in available if group in result.case.groups]
        if not group_results:
            continue
        group_speedups = [float(result.train_speedup) for result in group_results]
        group_regressions = sum(value < 1.0 for value in group_speedups)
        group_below_gate = sum(1.0 <= value < gate for value in group_speedups)
        print(
            f"group={group}: cases={len(group_results)} "
            f"min={min(group_speedups):.4f}x "
            f"median={statistics.median(group_speedups):.4f}x "
            f"regressions={group_regressions} below_gate={group_below_gate}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the current rank-4 C0 performance dispatch against the existing "
            "Triton A/B route using production-RNG forward+backward timings."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("te", "dispatch", "all"),
        default="all",
        help="TE shapes, current dispatch boundaries, or both (default: all)",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=benchmark.parse_case,
        help="Audit one ROWS,CHANNELS shape; repeatable and overrides --suite",
    )
    parser.add_argument(
        "--stats-mode",
        choices=("none", "basic", "both"),
        default="both",
        help="dq_delta stats path to benchmark (default: both)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 3 warmups, 10 iterations, and 3 repeats for a screening run",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=1.05,
        help="Required training speedup for the current C0 dispatch (default: 1.05)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of slowest cases to print in the summary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every measured case instead of only progress and the summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for all raw median results",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any admitted C0 case is slower than existing A/B",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be >= 0 and iterations/repeats must be positive")
    if args.gate <= 0:
        parser.error("gate must be positive")

    warmup, iterations, repeats = args.warmup, args.iterations, args.repeats
    if args.quick:
        warmup, iterations, repeats = 3, 10, 3

    if args.case:
        cases = [
            AuditCase(rows, channels, ("custom",))
            for rows, channels in sorted(set(args.case))
        ]
    else:
        cases = build_cases(args.suite)
    stats_modes = ("none", "basic") if args.stats_mode == "both" else (args.stats_mode,)
    total = len(cases) * len(stats_modes)
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"triton={getattr(benchmark.triton, '__version__', None)} "
        f"device={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()} "
        f"suite={args.suite} cases={total} warmup={warmup} "
        f"iterations={iterations} repeats={repeats} gate={args.gate:.3f}x"
    )

    results: list[AuditResult] = []
    completed = 0
    for stats_mode in stats_modes:
        for case in cases:
            result = run_audit_case(
                case,
                stats_mode=stats_mode,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                gate=args.gate,
            )
            results.append(result)
            completed += 1
            if args.verbose:
                _print_result(result)
            elif completed == 1 or completed % 10 == 0 or completed == total:
                print(f"progress: {completed}/{total}")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    _print_summary(results, gate=args.gate, top=max(1, args.top))
    if args.output is not None:
        _write_csv(args.output, results)
        print(f"csv={args.output.resolve()}")

    if args.fail_on_regression and any(
        result.available and result.verdict == "REGRESSION" for result in results
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
