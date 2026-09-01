from __future__ import annotations

"""Render a beginner report from an existing practical report artifact."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from dq_profile.v24_beginner_report import render_beginner_report
from dq_profile.v24_practical_report import render_report as render_practical_report


def render_existing_run(
    run_dir: Path,
    output: Path | None = None,
    *,
    refresh_practical_report: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    practical_path = run_dir / "practical_report.json"
    if not practical_path.is_file():
        raise FileNotFoundError(f"practical report was not found: {practical_path}")
    model = json.loads(practical_path.read_text(encoding="utf-8"))
    destination = (output or run_dir / "beginner_report.html").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_beginner_report(model), encoding="utf-8")
    if refresh_practical_report:
        (run_dir / "report.html").write_text(
            render_practical_report(model),
            encoding="utf-8",
        )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate beginner_report.html from an existing DQ Profiler run."
    )
    parser.add_argument("run_dir", type=Path, help="completed profile run directory")
    parser.add_argument("--output", type=Path, help="optional output HTML path")
    parser.add_argument(
        "--refresh-practical-report",
        action="store_true",
        help="also re-render report.html from the same practical_report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        render_existing_run(
            args.run_dir,
            args.output,
            refresh_practical_report=args.refresh_practical_report,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
