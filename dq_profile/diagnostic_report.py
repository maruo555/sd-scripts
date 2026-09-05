"""Offline HTML renderer for observed data; no chart libraries or remote assets."""
from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path


def report_thumbnails(directory, samples):
    """Cache small previews per measured physical image, separate from metrics.

    A saved preview survives moving the source dataset. It is the source image
    at first preview creation, not the transformed latent used for evaluation.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    cache_path = Path(directory) / "thumbnails.json"
    cached = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache, dict) and cache.get("schema_version") == "dataset-thumbnails-v1" and isinstance(cache.get("images"), dict):
                cached = cache["images"]
        except (OSError, ValueError):
            pass
    measured = {s["image_id"]: s for s in samples if s.get("measured")}
    previews = {}
    for image_id, sample in measured.items():
        source = str(sample["path"])
        previous = cached.get(image_id, {})
        if not isinstance(previous, dict):
            previous = {}
        if (previous.get("status") == "available" and previous.get("source_path") == source
                and str(previous.get("data_url", "")).startswith("data:image/jpeg;base64,")):
            previews[image_id] = previous
            continue
        try:
            with Image.open(source) as original:
                original_size = list(original.size)
                preview = ImageOps.exif_transpose(original)
                preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
                rgba = preview.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                buffer = io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=82, optimize=True)
                previews[image_id] = {"status": "available", "source_path": source,
                    "source_size": original_size, "width": rgb.width, "height": rgb.height,
                    "data_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}
        except FileNotFoundError:
            previews[image_id] = {"status": "missing", "source_path": source}
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            previews[image_id] = {"status": "unreadable", "source_path": source}
    cache_path.write_text(json.dumps({"schema_version": "dataset-thumbnails-v1", "images": previews},
                                    ensure_ascii=False), encoding="utf-8")
    return previews


def write_report(path, payload):
    template = Path(__file__).with_name("dataset_report.html").read_text(encoding="utf-8")
    payload = {**payload, "thumbnails": report_thumbnails(Path(path).parent, payload["samples"])}
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
