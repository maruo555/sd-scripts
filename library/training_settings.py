"""Two-phase settings snapshot, with no tensor, optimizer-state or training-RNG access."""
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import subprocess

from library.training_run import write_json

logger = logging.getLogger(__name__)
_SECRET = re.compile(r"(?:^|_)(?:api_key|access_token|auth_token|huggingface_token|hf_token|password|secret|authorization)(?:$|_)", re.I)


def snapshot(value, depth=0):
    if depth > 24:
        return {"unrecorded": "nesting_limit"}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"unrecorded": "nonfinite"}
    if isinstance(value, str):
        key, separator, _ = value.partition("=")
        return key + "=[redacted]" if separator and _SECRET.search(key.lstrip("-")) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): "[redacted]" if _SECRET.search(str(k)) else snapshot(v, depth + 1)
                for k, v in value.items() if isinstance(k, (str, int))}
    if isinstance(value, (list, tuple)):
        return [snapshot(v, depth + 1) for v in value]
    return {"unrecorded": "non_json_value"}


def snapshot_metadata(metadata):
    result = snapshot(metadata)
    for key in ("ss_network_args", "ss_datasets", "ss_dataset_dirs", "ss_reg_dataset_dirs", "ss_bucket_info", "ss_tag_frequency"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or raw == "None":
            continue
        try:
            result[key] = json.dumps(snapshot(json.loads(raw)), ensure_ascii=False, allow_nan=False)
        except (ValueError, RecursionError):
            result[key] = "[unrecorded: invalid structured metadata]"
    return result


def optimizer_groups(optimizer, labels=None):
    return [dict(index=i, label=labels[i] if labels and i < len(labels) else f"group {i}",
                 options=snapshot({k: v for k, v in group.items() if k != "params"}))
            for i, group in enumerate(optimizer.param_groups)]


def _read_head(repo):
    """Resolve a loose or packed HEAD without starting Git (including linked worktrees)."""
    git_dir = repo / ".git"
    try:
        if git_dir.is_file():
            marker = git_dir.read_text("utf-8").strip()
            if not marker.startswith("gitdir: "):
                return None
            git_dir = (repo / marker.removeprefix("gitdir: ")).resolve()
        value = (git_dir / "HEAD").read_text("ascii").strip()
        if value.startswith("ref: refs/"):
            ref = value.removeprefix("ref: ")
            if ".." in ref or not re.fullmatch(r"refs/[A-Za-z0-9._/+%-]+", ref):
                return None
            common = git_dir / "commondir"
            common_dir = (git_dir / common.read_text("utf-8").strip()).resolve() if common.is_file() else git_dir
            for directory in (git_dir, common_dir):
                ref_file = directory / ref
                if ref_file.is_file():
                    value = ref_file.read_text("ascii").strip()
                    break
            else:
                with (common_dir / "packed-refs").open(encoding="utf-8") as packed:
                    value = next((line.partition(" ")[0] for line in packed if line.rstrip().endswith(" " + ref)), "")
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    except (OSError, UnicodeError):
        return None


def source_identity(repo):
    repo = Path(repo)
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = _read_head(repo)
    lock = repo / "docs" / "migration" / "baseline" / "windows-cu130.lock.txt"
    return dict(source_sha=sha, dependency_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None)


def save_settings(run, requested, *, resolved=None, provenance=None):
    path = run.directory / "training_settings.json"
    stage = "resolved" if resolved is not None else "requested"
    try:
        data = dict(schema_version=2, run_id=run.run_id, session_id=run.session_id, settings_status=stage,
                    requested=snapshot(requested), resolved=snapshot(resolved), provenance=snapshot(provenance or {}))
        write_json(path, data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        logger.warning("Could not save %s settings for %s (%s)", stage, run.run_id, exc)
        run.set_artifact("training_settings", path, schema_version=2, status="partial" if path.exists() else "missing",
                         failed_stage=stage)
        return False
    run.set_artifact("training_settings", path, schema_version=2, status=stage)
    return True
