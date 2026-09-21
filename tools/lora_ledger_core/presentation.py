"""Dataset labels and chronological display derived from recorded evidence only."""
from __future__ import annotations

import copy
import math
import re
from datetime import datetime, timezone
from pathlib import PureWindowsPath


def _basename(value):
    return PureWindowsPath(value.strip().rstrip("/\\")).name if isinstance(value, str) else ""


def _config_path(source):
    for values in (source.get("values", {}), source.get("requested_args", {}),
                   source.get("resolved", {}).get("args", {})):
        path = values.get("dataset_config")
        if isinstance(path, str) and path.strip().lower().endswith(".toml"):
            return path.strip()
    return None


def automatic_dataset(run):
    sources = run.get("settings_sources", {})
    primary_id = run.get("settings", {}).get("settings_id")
    primary = sources.get(primary_id, {})
    primary_path = _config_path(primary)
    paths = {}
    if primary_path:
        paths[primary_path.casefold().replace("\\", "/")] = (primary_path, primary_id)
    else:
        for key, source in sources.items():
            path = _config_path(source)
            if path:
                paths.setdefault(path.casefold().replace("\\", "/"), (path, key))
    if len(paths) == 1:
        path, key = next(iter(paths.values()))
        return {"display_name": _basename(path), "name_source": "dataset_config",
                "config_path": path, "settings_id": key}
    values = primary.get("values", {})
    dirs = values.get("dataset_dirs", {})
    if not isinstance(dirs, dict):
        dirs = {}
    names = [_basename(key) for key in dirs]
    # An empty name followed by " (2)" etc. is the trainer's collision suffix.
    real_names = [name for name in names if name and not re.fullmatch(r"\(\d+\)", name)]
    if len(paths) > 1:
        return {"display_name": "名称不明（TOMLの記録が複数）", "name_source": "conflicting_config",
                "config_candidates": [path for path, _ in paths.values()]}
    if real_names and len(real_names) == len(names):
        return {"display_name": " / ".join(dict.fromkeys(real_names)), "name_source": "metadata_dirs",
                "subset_count": len(names)}
    datasets = values.get("datasets")
    if isinstance(datasets, dict):
        datasets = [datasets]
    subsets = sum(len(ds.get("subsets", [])) for ds in datasets if isinstance(ds, dict)) if isinstance(datasets, list) else 0
    count = max(len(names), subsets)
    return {"display_name": f"名称不明（{count}サブセット）" if count else "名称不明",
            "name_source": "unknown", "subset_count": count}


def dataset_info(run):
    """Read-time upgrade: old ledgers display correctly without overwriting user data."""
    result = copy.deepcopy(run.get("dataset", {}))
    automatic = automatic_dataset(run)
    manual = result.get("display_name_custom")
    if manual is None:
        manual = result.get("name_source") == "manual" or result.get("identity_status") == "user_asserted"
    result["display_name_custom"] = bool(manual)
    result["automatic"] = automatic
    if manual:
        result["name_source"] = "manual"
    else:
        result["display_name"] = automatic["display_name"]
        result["name_source"] = automatic["name_source"]
    return result


def refresh_dataset(run):
    run["dataset"] = dataset_info(run)


def timestamp(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        datetime.fromtimestamp(number, timezone.utc)
        return number
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def run_time(run):
    identity = run.get("identity", {})
    started = timestamp(identity.get("started_at"))
    if started is None:
        key = identity.get("session_key")
        if isinstance(key, str) and "|" in key:
            started = timestamp(key.split("|", 1)[1])
    if started is not None:
        return {"timestamp": started, "source": "training_started",
                "source_label": "学習開始（メタ情報・設定記録）"}
    choices = []
    for artifact in run.get("artifacts", []):
        ref = artifact.get("ref", {})
        value = ref.get("observed", {}).get("mtime_ns")
        try:
            modified = timestamp(int(value) / 1_000_000_000)
        except (TypeError, ValueError, OverflowError):
            modified = None
        if modified is not None:
            choices.append((ref.get("status") == "exists", artifact.get("role") == "final",
                            modified, artifact))
    if choices:
        _, _, value, artifact = max(choices, key=lambda item: item[:3])
        return {"timestamp": value, "source": "checkpoint_mtime",
                "source_label": "重みの更新日時（学習開始時刻が不明のため代用）",
                "artifact_id": artifact["artifact_id"], "artifact_name": artifact["name"],
                "ref_status": artifact["ref"].get("status")}
    return {"timestamp": None, "source": "unknown", "source_label": "日時不明"}


def time_text(info, iso=False):
    value = info["timestamp"]
    if value is None:
        return "" if iso else "不明"
    moment = datetime.fromtimestamp(value, timezone.utc).astimezone()
    return moment.isoformat(timespec="seconds") if iso else moment.strftime("%Y-%m-%d %H:%M")


def dataset_tooltip(info):
    automatic = info["automatic"]
    source = {"dataset_config": "学習設定のdataset_configから取得",
              "metadata_dirs": "メタ情報のフォルダ名から取得",
              "unknown": "TOML名は記録されていません",
              "conflicting_config": "TOMLの記録が複数あり特定できません"}.get(automatic["name_source"], "")
    parts = ["手動で指定した表示名" if info["display_name_custom"] else source]
    if info["display_name_custom"]:
        parts.append("自動取得: " + automatic["display_name"])
    if automatic.get("config_path"):
        parts.append(automatic["config_path"])
    parts.append("表示名だけでデータセット内容の同一性は確定しません。")
    return "\n".join(parts)
