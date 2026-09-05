"""Offline HTML renderer for observed data; no chart libraries or remote assets."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def write_report(path, payload):
    template = Path(__file__).with_name("dataset_report.html").read_text(encoding="utf-8")
    # A caption may contain closing script markup. Escape at the JSON embedding boundary.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    Path(path).write_text(template.replace("__DATASET_PAYLOAD__", data), encoding="utf-8")


def promote_dataset_report(profile, run_dir, selection):
    from dq_profile.dataset_diagnostics import rebuild, write_json
    source, dest = Path(profile) / "data_diagnostics", Path(run_dir) / "data_diagnostics"
    if not (source / "manifest.json").is_file():
        raise RuntimeError("dataset diagnostics requested but worker did not produce its sidecars")
    shutil.copytree(source, dest)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    manifest["local_selection"] = {k: selection.get(k) for k in ("selection_valid", "selection_status", "credible_muls", "edge_unresolved", "hard_unsafe_candidates", "hard_unsafe_reasons", "robustly_dominated_candidates")}
    write_json(dest / "manifest.json", manifest)
    rebuild(dest)
    for name in ("report.html", "beginner_report.html", "technical_report.html"):
        path = Path(run_dir) / name
        if path.is_file():
            html = path.read_text(encoding="utf-8")
            import re
            html, count = re.subn(r"(<body\b[^>]*>)", r'\1<nav style="padding:12px 24px;background:#edf3fb"><a href="data_diagnostics/dataset_report.html">データセット診断 → 画像・フォルダ・タグ・mul比較</a></nav>', html, count=1, flags=re.I)
            if count != 1:
                raise ValueError(f"cannot link dataset report from {name}")
            path.write_text(html, encoding="utf-8")
    return [p.relative_to(run_dir).as_posix() for p in dest.iterdir() if p.is_file()]
