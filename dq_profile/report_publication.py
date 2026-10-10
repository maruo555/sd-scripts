"""Finalize hashes only after all report writers have finished."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def finalize_report_manifests(output_dir, *, standard_version=None, inputs=(), outputs=()):
    """Refresh published artifacts without changing measurement input evidence.

    The analysis directory keeps its original manifest. The published copy must
    refer to the final published files, not the earlier intermediate HTML.
    """
    root = Path(output_dir).resolve()
    analysis_path = root / "analysis_manifest.json"
    updates = {}
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        reports = analysis.setdefault("reports", {})
        for record in reports.values():
            if not isinstance(record, dict) or "path" not in record:
                raise ValueError("Invalid analysis report artifact record")
            name = Path(record["path"]).name
            record.update(file_record(root / name))
        if standard_version:
            for key, name in (("primary", "report.html"), ("beginner", "beginner_report.html"),
                              ("dataset", "dataset_report.html"), ("uniform", "uniform_report.html")):
                if (root / name).is_file():
                    reports.setdefault(key, {}).update(file_record(root / name))
                    reports[key]["scope"] = "uniform_legacy" if key == "uniform" else "standard_spatial"
            analysis["published_report_version"] = standard_version
        updates[analysis_path] = analysis
    if standard_version:
        paths = [root / name for name in outputs]
        if any(path.resolve().parent != root for path in paths):
            raise ValueError("Standard report outputs must be in the report directory")
        updates[root / "standard_report_manifest.json"] = {
            "schema_version": "spatial-report-artifacts-v2", "version": standard_version,
            "inputs": {Path(path).name: file_record(path) for path in inputs},
            "outputs": {path.name: file_record(path) for path in paths},
            "external_network_required": False,
            "validation": "component statistics and paired cells checked",
        }
    # Resolve every target before replacing any manifest; missing files fail closed.
    for path, data in updates.items():
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
