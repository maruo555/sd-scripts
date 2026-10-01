from __future__ import annotations

"""Rebuild descriptive gradient curves from saved scalars, never reselect."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from dq_profile.v24_beginner_report import render_beginner_report
from dq_profile.v24_gradient_support import attach_gradient_support, build_gradient_support
from dq_profile.v24_practical_report import render_report


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(run_dir: Path, output_dir: Path, *, gradient_csv: Path | None = None) -> dict:
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    model = _json(run_dir / "practical_report.json")
    if len(model.get("datasets", [])) != 1:
        raise ValueError("A saved single-dataset practical_report.json is required; selection is never rerun")
    summary = _json(run_dir / "summary.json")
    manifest = _json(run_dir / "analysis_manifest.json")
    expected = summary.get("local_gradient_tail_sha256") or (manifest.get("inputs", {}).get("gradient_tail.csv") or {}).get("sha256")
    if not expected:
        expected = (model["datasets"][0].get("gradient_curve_support") or {}).get("provenance", {}).get("input_sha256")
    paths = [gradient_csv.resolve()] if gradient_csv else [p for name in ("raw_gradient_tail.csv", "gradient_tail.csv") if (p := run_dir / name).is_file()]
    if not paths and not gradient_csv:
        # Analysis directories already record their exact source path/hash.
        # Never search sibling runs/edge rounds for an arbitrary CSV.
        recorded = (manifest.get("inputs", {}).get("gradient_tail.csv") or {}).get("path")
        if expected and recorded and Path(recorded).is_file():
            paths = [Path(recorded)]
    hashes = {path: _sha(path) for path in paths if path.is_file()}
    matching = [path for path, digest in hashes.items() if digest == expected] if expected else list(hashes)
    error = None
    path = matching[0] if len({hashes[p] for p in matching}) == 1 else None
    if not hashes:
        error = "no_sample_csv"
    elif expected and not matching:
        error = "raw_csv_hash_mismatch_with_saved_analysis"
    elif path is None:
        error = "ambiguous_raw_csv"
    rows = []
    if path is not None:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    support = build_gradient_support(
        rows, model["datasets"][0]["candidate_cards"], input_error=error,
        provenance={
            "input_path": str(path) if path else None,
            "input_sha256": hashes.get(path), "expected_input_sha256": expected,
            "input_identity": "saved_analysis_hash_verified" if expected and path else "same_directory_legacy_csv" if path and not gradient_csv else "explicit_csv_without_saved_hash" if path else "unavailable",
            "source_report_sha256": _sha(run_dir / "practical_report.json"),
        },
    )
    rebuilt = attach_gradient_support(model, support)
    # Render before writing anything, so unsupported source models do not
    # leave a partial result or alter any original selection/raw artifacts.
    outputs = {
        "gradient_curve_support.json": json.dumps(support, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "practical_report.json": json.dumps(rebuilt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "beginner_report.html": render_beginner_report(rebuilt),
        "report.html": render_report(rebuilt),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    # Preserve the existing technical-report link without generating new
    # selection artifacts. The original bytes are copied verbatim.
    technical = run_dir / "technical_report.html"
    if technical.is_file() and run_dir != output_dir:
        (output_dir / technical.name).write_bytes(technical.read_bytes())
    if manifest:
        # Preserve analysis inputs and selection provenance; only report
        # locations/hashes change when the presentation is regenerated.
        reports = manifest.setdefault("reports", {})
        for key, name in (("primary", "report.html"), ("beginner", "beginner_report.html")):
            reports.setdefault(key, {}).update(
                path=str(output_dir / name), sha256=_sha(output_dir / name)
            )
        if technical.is_file():
            reports.setdefault("technical", {}).update(
                path=str(output_dir / technical.name), sha256=_sha(output_dir / technical.name)
            )
        (output_dir / "analysis_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return support


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="saved analysis or promoted production run")
    parser.add_argument("--output-dir", type=Path, required=True, help="destination for derived JSON and both HTML reports")
    parser.add_argument("--gradient-csv", type=Path, help="exact CSV belonging to this saved analysis (hash checked when recorded)")
    args = parser.parse_args(argv)
    support = rebuild(args.run_dir, args.output_dir, gradient_csv=args.gradient_csv)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "candidates": [{"range_mul": c["range_mul"], "status": c["status"], "d_reasons": c["d_reasons"], "parallel_component_reasons": c["parallel_component_reasons"]} for c in support["candidates"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
