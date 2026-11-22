"""Shared validation for a published image comparison. No training, Qt or ledger imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import uuid


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate comparison JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(token):
    raise ValueError(f"Invalid comparison JSON number {token}")


def _parse(data):
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Expected a comparison JSON object")
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {label}")
    return value


def _image_path(folder, condition, raw):
    if not isinstance(raw, str) or "\\" in raw or PureWindowsPath(raw).anchor:
        raise ValueError("Expected a portable relative comparison PNG path")
    if any(segment in {"", ".", ".."} for segment in raw.split("/")):
        raise ValueError(f"Invalid comparison image path: {raw}")
    parts = raw.split("/")
    if len(parts) < 3 or parts[:2] != ["images", condition] or not raw.lower().endswith(".png"):
        raise ValueError(f"Image must belong to its comparison condition: {raw}")
    path = (folder / raw).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"Comparison image escapes its folder: {raw}")
    return path


def _readable_png(path, job):
    from PIL import Image
    try:
        with Image.open(path) as png:
            if png.format != "PNG" or png.size != (job["width"], job["height"]):
                return False
            for tag, expected in (("seed", job["seed"]), ("prompt", job["prompt"]), ("negative-prompt", job["negative"])):
                if tag in png.info and png.info[tag] != str(expected):
                    return False
            expected_tags = job.get("png_parameters", {})
            for tag in ("network-strength-mode", "network-te1-strength", "network-te2-strength",
                        "network-unet-strength", "network-dynamic-slots", "network-merged-slots"):
                if tag in png.info:
                    if not isinstance(expected_tags, dict) or tag not in expected_tags:
                        return False
                    if json.loads(png.info[tag], parse_constant=_nonfinite) != expected_tags[tag]:
                        return False
            png.verify()
        # CRC verification alone does not decode the compressed pixels.
        with Image.open(path) as png:
            png.load()
        return True
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return False


def comparison_jobs(metadata, folder):
    """Validate the job graph and return only jobs whose published PNG is still usable."""
    folder = Path(folder).resolve()
    if metadata.get("schema_version") != 1 or metadata.get("kind") != "lora_image_comparison":
        raise ValueError("Unsupported image comparison metadata")
    for key in ("comparison_id", "attempt_id"):
        uuid.UUID(_identifier(metadata.get(key), key))
    parent = metadata.get("status")
    if parent not in {"done", "partial", "failed", "cancelled", "dry_run"}:
        raise ValueError("Unpublished comparison status")
    parent_rc = metadata.get("returncode")
    if parent_rc is not None and type(parent_rc) is not int:
        raise ValueError("Invalid generator return code")
    if parent == "dry_run" and parent_rc is not None:
        raise ValueError("A dry run must have no generator exit code")
    if parent == "done" and parent_rc != 0:
        raise ValueError("Successful comparison must record generator return code 0")

    prompts, conditions, seeds, jobs = (metadata.get(k) for k in ("prompts", "conditions", "seeds", "jobs"))
    if any(not isinstance(records, list) for records in (prompts, conditions, seeds, jobs)):
        raise ValueError("Missing prompts, seeds, conditions or jobs")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Expected unique integer seeds")
    conditions_by_id = {}
    for condition in conditions:
        cid = _identifier(condition.get("id"), "condition id")
        if cid in conditions_by_id or not isinstance(condition.get("items"), list):
            raise ValueError("Invalid or duplicate condition")
        conditions_by_id[cid] = condition
    job_ids, pairs, images, case_rows, row_cases = set(), set(), set(), {}, {}
    eligible = []
    successes = 0
    for job in jobs:
        jid = _identifier(job.get("job_id"), "job_id")
        case = _identifier(job.get("case_id"), "case_id")
        cid = job.get("condition_id")
        pos = job.get("prompt_index")
        if cid not in conditions_by_id or type(pos) is not int or not 1 <= pos <= len(prompts):
            raise ValueError(f"Unknown prompt row or condition: {jid}")
        if job.get("prompt_id") != prompts[pos - 1].get("id") or job.get("seed") not in seeds:
            raise ValueError(f"Invalid prompt or seed reference: {jid}")
        if type(job.get("seed")) is not int or not isinstance(job.get("prompt"), str) or not isinstance(job.get("negative"), str):
            raise ValueError(f"Missing resolved prompt or seed: {jid}")
        if any(type(job.get(k)) is not int or job[k] <= 0 for k in ("width", "height")):
            raise ValueError(f"Missing resolved image dimensions: {jid}")
        signature = (pos, job["prompt"], job["negative"], job["seed"], job["width"], job["height"])
        if case_rows.setdefault(case, signature) != signature or row_cases.setdefault((pos, job["seed"]), case) != case:
            raise ValueError(f"Inconsistent comparison case: {case}")
        path = _image_path(folder, cid, job.get("image"))
        image_key = str(path).casefold()
        if jid in job_ids or (case, cid) in pairs or image_key in images:
            raise ValueError(f"Duplicate job, candidate or target image: {jid}")
        job_ids.add(jid)
        pairs.add((case, cid))
        images.add(image_key)
        status, rc = job.get("status"), job.get("returncode")
        if status not in {"done", "failed", "cancelled", "dry_run"} or rc is not None and type(rc) is not int:
            raise ValueError(f"Invalid job outcome: {jid}")
        if status == "done" and rc != 0 or status == "dry_run" and rc is not None:
            raise ValueError(f"Job state conflicts with its return code: {jid}")
        if parent == "dry_run" and status != "dry_run":
            raise ValueError("Dry run contains an executed job")
        if status == "done":
            successes += 1
            if parent in {"done","partial"} and _readable_png(path, job):
                eligible.append(job)
    if len(jobs) != len(prompts) * len(seeds) * len(conditions):
        raise ValueError("Comparison jobs do not cover every planned prompt, seed and condition")
    if parent == "done" and (not jobs or successes != len(jobs)):
        raise ValueError("Successful comparison requires all jobs to have succeeded")
    if parent == "partial" and not 0 < successes < len(jobs):
        raise ValueError("Partial comparison requires individually identified successes and failures")
    if parent not in {"done", "partial"} and successes:
        raise ValueError("Non-success comparison contains successful jobs")
    return eligible


def read_comparison(folder):
    folder = Path(folder).resolve()
    def inside(name):
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"Comparison input escapes its folder: {name}")
        return path
    raw = inside("metadata.json").read_bytes()
    metadata = _parse(raw)
    config = inside("config.json").read_bytes()
    _parse(config)
    if metadata.get("config_sha256") != hashlib.sha256(config).hexdigest():
        raise ValueError("Configuration digest differs from the published comparison")
    eligible = comparison_jobs(metadata, folder)
    return metadata, hashlib.sha256(raw).hexdigest(), eligible
