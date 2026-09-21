"""Conservative discovery and idempotent reconciliation of existing outputs."""
from __future__ import annotations

import copy
import csv
import hashlib
import os
import re
import struct
from pathlib import Path

from tools.lora_training_settings import read_metadata, normalize_metadata, load_training_settings
from library.training_settings import snapshot_metadata
from .storage import (VERSION, LedgerError, Conflict, Cancelled, now, digest, stable_id,
                      atomic_json, read_json, file_stat, same_stat, sha256_file, directory_lock)

from .presentation import refresh_dataset

PARSER_VERSION = 2
LOG_PREFIXES = {
    "gradient_logs+": "gradient", "dq_delta_logs+": "dq", "dq_delta_auto+": "dq_auto",
    "rank_logs+": "rank", "group_loss_logs+": "group_loss", "group_loss_epoch+": "group_epoch",
    "avg_shadow+": "avg", "avg_promote+": "avg",
}
SKIP_DIRS = {".git", "venv", "__pycache__", ".tmp", "cache", "images", "worker"}
STATES = {"new": "新しい学習", "additional": "追加資料あり", "unchanged": "登録済み",
          "changed": "変更あり・要確認", "related": "関連資料のみ", "error": "読込不能",
          "missing": "見つからない", "offline": "参照先にアクセスできない"}
DROP_SETTINGS = {"tag_frequency", "bucket_info", "epoch", "steps", "training_finished_at",
                 "output_name", "session_id", "training_started_at", "training_comment"}


def session_key(metadata):
    session = metadata.get("ss_session_id")
    started = metadata.get("ss_training_started_at")
    return f"{session}|{started}" if session and started else None


def output_base(path):
    name = Path(path).stem
    for prefix in LOG_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    name = re.sub(r"_diagnostic$", "", name)
    return re.sub(r"(?:_final_raw|_raw|_center|-step\d{8,}|-\d{3,})$", "", name)


def role(path):
    stem = Path(path).stem
    for suffix in ("final_raw", "center", "raw"):
        if stem.endswith("_" + suffix):
            return suffix
    if re.search(r"-step\d{8,}$", stem):
        return "step"
    return "epoch" if re.search(r"-\d{3,}$", stem) else "final"


def classify(path):
    name = path.name.lower()
    # Profiler snapshots must never become ordinary training runs.
    if "dq_dataset_profiler" in [p.lower() for p in path.parts]:
        if path.suffix.lower() == ".html" or name in (
                "manifest.json", "report.json", "summary.json", "dataset_diagnostics.json",
                "dataset_report.json", "execution_plan.json"):
            return "profile"
        return None
    if name.endswith(".safetensors"):
        return "checkpoint"
    for prefix, kind in LOG_PREFIXES.items():
        if name.startswith(prefix):
            return kind
    if name.endswith("_diagnostic.json"):
        return "diagnostic"
    if name.endswith("_diagnostic.html"):
        return "diagnostic_html"
    if name in ("requested_args.json", "resolved_config.json"):
        return "settings_payload"
    if name == "manifest.json":
        return "manifest"
    if name == "metadata.json":
        return "comparison"
    if name in ("report.html", "blind_report.html"):
        return "comparison_html"
    return None


def _header_hash(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        size = struct.unpack("<Q", prefix)[0]
        return hashlib.sha256(prefix + stream.read(size)).hexdigest()


def parse_entry(path, kind):
    data = {}
    if kind == "checkpoint":
        metadata = snapshot_metadata(read_metadata(path))
        if not any(k in metadata for k in ("ss_output_name", "ss_session_id", "ss_network_module")):
            raise LedgerError("学習メタ情報なし（必要ならメモだけ登録してください）")
        values = normalize_metadata(metadata)
        data = {"identity": session_key(metadata), "output_name": metadata.get("ss_output_name"),
                "session_id": metadata.get("ss_session_id"), "started_at": metadata.get("ss_training_started_at"),
                "epoch": values.get("epoch", metadata.get("ss_epoch")),
                "step": metadata.get("ss_steps"), "role": role(path),
                "settings": {k: v for k, v in values.items() if k not in DROP_SETTINGS},
                "header_sha256": _header_hash(path)}
    elif kind == "manifest":
        manifest = read_json(path)
        if manifest.get("kind") != "training_settings" or manifest.get("schema_version") != 1:
            return None
        identity = None
        if manifest.get("session_id") and manifest.get("training_started_at"):
            identity = f"{manifest['session_id']}|{manifest['training_started_at']}"
        data = {"identity": identity, "run_id": manifest.get("run_id"),
                "output_name": manifest.get("output_name"), "manifest": manifest}
    elif kind == "settings_payload":
        raw = read_json(path)
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("args"), dict):
            return None
        data = {"run_id": raw.get("run_id"), "stage": path.stem}
    elif kind == "diagnostic":
        raw = read_json(path)
        if not isinstance(raw, dict) or "base_name" not in raw or not any(k in raw for k in ("grad", "dq", "rank")):
            raise LedgerError("診断JSONの構造が不正です")
        settings = raw.get("training_settings") or {}
        data = {"output_name": raw.get("base_name"), "run_id": settings.get("run_id"),
                "generated_at": raw.get("generated_at"),
                "settings": settings.get("values", {}), "settings_status": settings.get("status")}
    elif kind == "comparison":
        raw = read_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("conditions"), list) or not isinstance(raw.get("jobs"), list):
            return None
        data = {"created_at": raw.get("created_at"), "conditions": raw["conditions"],
                "job_count": len(raw["jobs"])}
    elif kind in ("diagnostic_html",):
        data = {"output_name": output_base(path)}
    elif kind in LOG_PREFIXES.values():
        data = {"output_name": output_base(path)}
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            data["header"] = stream.readline().strip()[:5000]
    elif kind == "profile":
        # Large data arrays remain at the source. Discovery only stores a reference.
        data = {"output_name": path.parent.name}
    return data


def _empty_run(run_id, name, identity=None):
    return {"schema_version": VERSION, "kind": "ledger_run", "run_id": run_id, "revision": 0,
            "display_name": name or "名前未設定", "aliases": [], "identity": identity or {},
            "association": "verified" if identity else "candidate",
            "training_status": "unknown", "purpose": "unknown",
            "dataset": {"display_name": "", "identity_status": "unknown"},
            "user": {"experiment_note": ""}, "artifacts": [], "source_refs": [],
            "settings_sources": {}, "settings": {}, "created_at": now()}


def _ref(entry, association="candidate"):
    return {"ref_id": stable_id(entry["root"], entry["relative_path"], digest(entry["observed"])),
            "root": entry["root"], "relative_path": entry["relative_path"], "kind": entry["kind"],
            "observed": entry["observed"], "status": "exists",
            "association": {"status": association, "method": "session" if association == "verified" else "name_candidate"}}


def _append_ref(run, ref):
    for old in run["source_refs"]:
        if old["ref_id"] == ref["ref_id"] or (old["root"] == ref["root"]
                and old["relative_path"] == ref["relative_path"] and same_stat(old["observed"], ref["observed"])):
            return
    run["source_refs"].append(ref)


def _add_alias(run, name):
    if name and name not in run["aliases"]:
        run["aliases"].append(name)


def scan(ledger, progress=None, cancel=None, full=False):
    def emit(message):
        if cancel and cancel():
            raise Cancelled("走査を中止しました")
        if progress:
            progress(message)

    config = ledger.config()
    roots = ledger.locations()
    before_runs = {r["run_id"]: r for r in ledger.list_runs()}
    proposed = copy.deepcopy(before_runs)
    issues = [{"status": "error", "name": message} for message in ledger.errors]
    cache_path = ledger.directory / "cache" / "scan_index.json"
    try:
        old_cache = read_json(cache_path)
        if old_cache.get("parser_version") != PARSER_VERSION:
            old_cache = {}
    except (OSError, ValueError):
        old_cache = {}
    cached = old_cache.get("entries", {})
    shared_settings = old_cache.get("settings", {})
    for item in cached.values():
        key = item.get("data", {}).pop("_settings_key", None)
        if key:
            item["data"]["settings"] = shared_settings[key]
    entries, seen, online_roots = [], set(), set()
    new_cache = {}
    canonical_refs = {}
    for run in before_runs.values():
        for ref in run["source_refs"] + [a["ref"] for a in run["artifacts"]]:
            try:
                canonical_refs.setdefault(os.path.normcase(str(ledger.resolve_ref(ref))),
                                          (ref["root"], ref["relative_path"]))
            except (OSError, ValueError):
                pass
    for source in config["sources"]:
        if not source.get("enabled", True):
            continue
        root_id = source["root"]
        root = Path(roots[root_id]).resolve()
        if not root.is_dir():
            issues.append({"status": "offline", "name": str(root), "root": root_id})
            continue
        online_roots.add(root_id)
        emit(f"参照先を走査: {root}")
        walk_errors = []
        def onerror(exc):
            walk_errors.append({"status": "error", "name": str(exc)})
        for folder, dirs, filenames in os.walk(root, followlinks=False, onerror=onerror):
            parent = Path(folder)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and d not in source.get("exclude", [])
                             and not (parent / d).is_symlink()
                             and not (os.name == "nt" and getattr((parent / d).stat(), "st_file_attributes", 0) & 0x400))
            if not source.get("recursive", True):
                dirs[:] = []
            for name in sorted(filenames):
                path = parent / name
                if path.is_symlink():
                    continue
                key = os.path.normcase(str(path.resolve()))
                if key in seen:
                    continue
                kind = classify(path)
                if not kind:
                    continue
                seen.add(key)
                # Audit retained references even if parsing under a new root fails.
                if key in canonical_refs:
                    online_roots.add(canonical_refs[key][0])
                if len(seen) % 100 == 0:
                    emit(f"ファイル確認: {len(seen):,} 件")
                try:
                    stat = file_stat(path)
                    old = cached.get(key)
                    if old and old["kind"] == kind and same_stat(old["observed"], stat) and not full:
                        entry = copy.deepcopy(old)
                        entry.update(root=root_id, relative_path=path.relative_to(root).as_posix())
                    else:
                        data = parse_entry(path, kind)
                        if data is None:
                            continue
                        observed = dict(stat, fingerprint_status="stat_only")
                        if kind in ("manifest", "settings_payload", "comparison") or (kind == "profile" and path.stat().st_size < 1024 * 1024):
                            observed.update(sha256=sha256_file(path, cancel), fingerprint_status="sha256")
                        if kind == "checkpoint":
                            observed["header_sha256"] = data["header_sha256"]
                        if full:
                            observed.update(sha256=sha256_file(path, cancel), fingerprint_status="sha256")
                        if not same_stat(stat, file_stat(path)):
                            raise Conflict("読込中に変更されました")
                        entry = {"root": root_id, "relative_path": path.relative_to(root).as_posix(),
                                 "path": str(path), "kind": kind, "observed": observed, "data": data}
                    if key in canonical_refs:
                        entry["root"], entry["relative_path"] = canonical_refs[key]
                    entries.append(entry)
                    new_cache[key] = entry
                except Cancelled:
                    raise
                except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                    issues.append({"status": "error", "name": str(path), "message": str(exc)})
            emit(f"走査中: {len(seen):,} ファイル確認済み")
        issues.extend(walk_errors)

    emit("学習と成果物の対応を確認しています")
    current_by_path = {os.path.normcase(e["path"]): e for e in entries}
    # Audit old references without changing their recorded content identity.
    for run in proposed.values():
        for ref in run["source_refs"] + [a["ref"] for a in run["artifacts"]]:
            if ref["root"] not in online_roots:
                continue
            try:
                path = ledger.resolve_ref(ref)
                path_key = os.path.normcase(str(path))
                current = current_by_path.get(path_key)
                if current:
                    status = "exists" if same_stat(ref["observed"], current["observed"]) else "changed"
                    if ref["observed"].get("sha256") and current["observed"].get("sha256"):
                        status = "exists" if ref["observed"]["sha256"] == current["observed"]["sha256"] else "changed"
                elif not path.exists():
                    status = "missing"
                elif path_key in seen:
                    # Discovered but not parsed: presence alone cannot verify the old content.
                    status = "unreadable"
                else:
                    # Excluded paths keep their last known state until explicitly checked.
                    status = ref.get("status", "exists")
                ref["status"] = status
            except (OSError, LedgerError):
                ref["status"] = "unreadable"

    identity_map = {r["identity"]["session_key"]: r["run_id"] for r in proposed.values()
                    if r.get("identity", {}).get("session_key")}
    manifests = {}
    for entry in entries:
        if entry["kind"] != "manifest":
            continue
        data = entry["data"]
        identity = data.get("identity")
        if identity:
            manifests.setdefault(identity, []).append(entry)

    representatives = {}
    for entry in entries:
        if entry["kind"] != "checkpoint":
            continue
        data = entry["data"]
        identity = data["identity"]
        matching = manifests.get(identity, []) if identity else []
        if identity in identity_map:
            run_id = identity_map[identity]
        elif len(matching) == 1 and isinstance(matching[0]["data"].get("run_id"), str) and re.fullmatch(
                r"[A-Za-z0-9_-]{1,100}", matching[0]["data"]["run_id"]):
            run_id = matching[0]["data"]["run_id"]
        else:
            run_id = stable_id(config["ledger_id"], identity or (entry["root"] + ":" + entry["relative_path"]))
        if run_id in proposed and identity and proposed[run_id].get("identity", {}).get("session_key") not in (None, identity):
            issues.append({"status": "error", "name": entry["relative_path"], "message": "manifest run IDのsessionが矛盾"})
            run_id = stable_id(config["ledger_id"], identity)
        if identity:
            identity_map[identity] = run_id
        if run_id not in proposed:
            proposed[run_id] = _empty_run(run_id, output_base(entry["path"]),
                                         {"session_key": identity, "session_id": data.get("session_id"),
                                          "started_at": data.get("started_at")} if identity else {})
        run = proposed[run_id]
        if data.get("started_at") is not None:
            run["identity"].setdefault("started_at", data["started_at"])
        ref = _ref(entry, "verified" if identity else "candidate")
        settings_id = digest(data["settings"])
        run["settings_sources"].setdefault(settings_id,
            {"source": "metadata", "values": data["settings"], "status": "partial"})
        # Existing content at the same reference keeps its artifact ID.
        existing = next((a for a in run["artifacts"] if a["ref"]["root"] == ref["root"]
                         and a["ref"]["relative_path"] == ref["relative_path"]
                         and same_stat(a["ref"]["observed"], ref["observed"])
                         and all(not a["ref"]["observed"].get(k) or not ref["observed"].get(k)
                                 or a["ref"]["observed"][k] == ref["observed"][k]
                                 for k in ("sha256", "header_sha256"))), None)
        if not existing:
            content_hash = ref["observed"].get("sha256")
            existing = next((a for a in run["artifacts"] if content_hash and
                             a["ref"]["observed"].get("sha256") == content_hash), None)
            if existing:
                if existing["ref"].get("status") != "exists":
                    previous = existing.setdefault("previous_refs", [])
                    if not any(r["ref_id"] == existing["ref"]["ref_id"] for r in previous):
                        previous.append(existing["ref"])
                    existing["ref"] = ref
                    existing["name"] = Path(entry["path"]).name
                else:
                    copies = existing.setdefault("additional_refs", [])
                    if not any(r["root"] == ref["root"] and r["relative_path"] == ref["relative_path"]
                               and same_stat(r["observed"], ref["observed"]) for r in copies):
                        copies.append(ref)
        if not existing:
            artifact = {"artifact_id": stable_id(run_id, ref["ref_id"]),
                        "name": Path(entry["path"]).name, "role": data["role"],
                        "role_source": "filename", "epoch": data.get("epoch"), "step": data.get("step"),
                        "settings_id": settings_id, "metadata_output_name": data.get("output_name"),
                        "ref": ref}
            run["artifacts"].append(artifact)
        else:
            existing["ref"]["status"] = "exists"
            if existing.get("role_source") == "filename":
                existing["role"] = role(existing["name"])
        _add_alias(run, data.get("output_name"))
        _add_alias(run, output_base(entry["path"]))
        priority = {"final": 0, "final_raw": 1, "raw": 2, "center": 3, "epoch": 4, "step": 5}[data["role"]]
        if run_id not in representatives or priority < representatives[run_id][0]:
            representatives[run_id] = (priority, entry, settings_id)
        if len(matching) > 1:
            run["association"] = "conflict"
            run.setdefault("issues", [])
            if "同じsessionに複数の設定記録があります" not in run["issues"]:
                run["issues"].append("同じsessionに複数の設定記録があります")

    for run_id, (_, entry, settings_id) in representatives.items():
        run = proposed[run_id]
        if not run.get("user", {}).get("display_name_custom"):
            run["display_name"] = output_base(entry["path"])
        run["settings"] = {"settings_id": settings_id, "source": "metadata", "status": "partial"}
        matching = manifests.get(entry["data"]["identity"], [])
        if len(matching) == 1:
            manifest_entry = matching[0]
            path = Path(manifest_entry["path"])
            try:
                settings = load_training_settings(path.parent, output_base(entry["path"]), entry["path"], path)
                if settings.get("status") in ("recorded", "partial") and settings.get("values") is not None:
                    key = digest(settings)
                    run["settings_sources"][key] = {"source": "record", "values": settings.get("values", {}),
                                                   "status": settings["status"],
                                                   "resolved": settings.get("resolved", {}),
                                                   "requested_args": settings.get("requested_args", {}),
                                                   "source_ref": _ref(manifest_entry, "verified")}
                    run["settings"] = {"settings_id": key, "source": "record", "status": settings["status"]}
                else:
                    run["settings"]["record_issue"] = settings.get("notes", [])
            except (OSError, ValueError) as exc:
                run["settings"]["record_issue"] = [str(exc)]
    # Manifests can also describe runs without a surviving checkpoint.
    record_run_map = {}
    for entry in entries:
        if entry["kind"] != "manifest":
            continue
        data = entry["data"]
        identity = data.get("identity")
        run_id = identity_map.get(identity)
        if not run_id:
            candidate_id = data.get("run_id")
            run_id = candidate_id if isinstance(candidate_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", candidate_id) else stable_id(entry["path"])
            if run_id in proposed and identity and proposed[run_id].get("identity", {}).get("session_key") not in (None, identity):
                issues.append({"status": "error", "name": entry["relative_path"], "message": "設定記録のrun IDが別sessionと重複"})
                run_id = stable_id(config["ledger_id"], identity)
            if run_id not in proposed:
                proposed[run_id] = _empty_run(run_id, data.get("output_name"),
                                             {"session_key": identity} if identity else {})
            if identity:
                identity_map[identity] = run_id
        record_id = data.get("run_id")
        if record_id in record_run_map and record_run_map[record_id] != run_id:
            record_run_map[record_id] = None
        else:
            record_run_map[record_id] = run_id
        run = proposed[run_id]
        _append_ref(run, _ref(entry, "verified" if identity else "candidate"))
        if not run["artifacts"]:
            try:
                path = Path(entry["path"])
                from tools.lora_training_settings import _payload, _validate_manifest_identity, _validate_resolved
                manifest = data["manifest"]
                _validate_manifest_identity(manifest)
                requested = _payload(path.parent, manifest.get("requested_args"), manifest["run_id"])
                settings = {"requested_args": requested["args"], "values": requested["args"], "status": "partial"}
                if manifest.get("settings_status") == "resolved":
                    resolved = _payload(path.parent, manifest.get("resolved_config"), manifest["run_id"])
                    _validate_resolved(resolved)
                    settings.update(values=resolved["args"], resolved=resolved, status="recorded")
                if settings.get("values") is not None and settings.get("status") in ("recorded", "partial"):
                    key = digest(settings)
                    run["settings_sources"][key] = {"source": "record", "values": settings["values"],
                        "resolved": settings.get("resolved", {}), "requested_args": settings.get("requested_args", {}),
                        "status": settings["status"], "source_ref": _ref(entry, "verified" if identity else "candidate")}
                    run["settings"] = {"settings_id": key, "source": "record", "status": settings["status"]}
            except (OSError, ValueError) as exc:
                issues.append({"status": "error", "name": entry["relative_path"], "message": str(exc)})

    names = {}
    for run_id, run in proposed.items():
        for name in set([run["display_name"]] + run["aliases"]):
            if name:
                names.setdefault(name, set()).add(run_id)
    related = []
    for entry in entries:
        kind = entry["kind"]
        if kind in ("checkpoint", "manifest"):
            continue
        name = entry["data"].get("output_name")
        run_ids = names.get(name, set())
        explicit = entry["data"].get("run_id")
        explicit = record_run_map.get(explicit, explicit)
        if explicit in proposed:
            run_ids = {explicit}
        if kind in ("profile", "comparison", "comparison_html"):
            related.append({"status": "related", "name": entry["relative_path"], "entry": entry})
        elif len(run_ids) == 1:
            run_id = next(iter(run_ids))
            _append_ref(proposed[run_id], _ref(entry, "verified" if explicit == run_id else "candidate"))
        elif not run_ids and kind in LOG_PREFIXES.values() or (not run_ids and kind == "diagnostic"):
            run_id = stable_id(config["ledger_id"], entry["root"], str(Path(entry["relative_path"]).parent), name)
            if run_id not in proposed:
                proposed[run_id] = _empty_run(run_id, name)
            _append_ref(proposed[run_id], _ref(entry))
            names.setdefault(name, set()).add(run_id)
        else:
            related.append({"status": "related", "name": entry["relative_path"], "entry": entry})

    rows = []
    for run_id, run in proposed.items():
        refresh_dataset(run)
        old = before_runs.get(run_id)
        if old is None:
            state = "new"
        elif digest(run) == digest(old):
            state = "unchanged"
        elif any(r.get("status") in ("changed", "missing", "unreadable") for r in
                 run["source_refs"] + [a["ref"] for a in run["artifacts"]]):
            state = "changed"
        else:
            state = "additional"
        rows.append({"status": state, "name": run["display_name"], "run_id": run_id,
                     "artifact_count": len(run["artifacts"]), "ref_count": len(run["source_refs"]),
                     "association": run["association"], "run": run,
                     "expected_revision": old["revision"] if old else 0})
    # Avoid caching large training settings per checkpoint by sharing equal settings.
    shared = {}
    packed = copy.deepcopy(new_cache)
    for entry in packed.values():
        values = entry["data"].pop("settings", None)
        if values is not None:
            key = digest(values)
            shared[key] = values
            entry["data"]["_settings_key"] = key
    # Unpack is handled before next scan.
    atomic_json(cache_path, {"parser_version": PARSER_VERSION, "entries": packed, "settings": shared})
    emit(f"走査完了: 学習候補 {len(rows):,} 件")
    return {"config_revision": config["revision"], "locations_hash": digest(roots), "rows": rows,
            "related": related, "issues": issues, "file_count": len(entries), "scanned_at": now(),
            "audited_roots": sorted(online_roots)}


def apply_scan(ledger, result, selected_ids=None, progress=None, cancel=None):
    ids = set(selected_ids) if selected_ids is not None else {
        row["run_id"] for row in result["rows"] if row["status"] in ("new", "additional")}
    applied, errors = [], []
    audited_roots = result.get("audited_roots")
    with directory_lock(ledger.directory):
        ledger._recover()
        if ledger.config()["revision"] != result["config_revision"] or digest(ledger.locations()) != result["locations_hash"]:
            raise Conflict("参照先設定が変わりました。再走査してください。")
        for row in result["rows"]:
            if row["run_id"] not in ids or row["status"] == "unchanged":
                continue
            if cancel and cancel():
                raise Cancelled("反映を中止しました。反映済みの項目は保存されています。")
            run = row["run"]
            try:
                for ref in run["source_refs"] + [a["ref"] for a in run["artifacts"]]:
                    if ref.get("status") != "exists":
                        continue
                    # Offline/disabled roots retain their last known references.
                    if audited_roots is not None and ref["root"] not in audited_roots:
                        continue
                    path = ledger.resolve_ref(ref)
                    if not same_stat(ref["observed"], file_stat(path)):
                        raise Conflict(f"走査後に変更されました: {path.name}")
                ledger._commit(ledger._run_writes(run, row["expected_revision"]))
                applied.append(run["run_id"])
                if progress:
                    progress(f"反映済み: {len(applied)} 件")
            except (OSError, LedgerError) as exc:
                errors.append({"name": row["name"], "message": str(exc)})
    return {"applied": applied, "errors": errors}


def link_reference(ledger, run_id, entry, note="ユーザーが対応を確認"):
    run = ledger.get_run(run_id)
    ref = _ref(entry, "user_asserted")
    ref["association"] = {"status": "user_asserted", "method": "manual_confirmation", "note": note}
    # Upgrade a name-only association rather than adding a second copy.
    old = next((r for r in run["source_refs"] if r["ref_id"] == ref["ref_id"]), None)
    if old:
        old["association"] = ref["association"]
    else:
        _append_ref(run, ref)
    return ledger.save_run(run, run["revision"])


def verify(ledger, full=False, progress=None, cancel=None, run_ids=None):
    results = []
    for run in ledger.list_runs():
        if run_ids and run["run_id"] not in run_ids:
            continue
        changed = False
        for ref in run["source_refs"] + [a["ref"] for a in run["artifacts"]]:
            if cancel and cancel():
                raise Cancelled("確認を中止しました")
            if progress:
                progress(ref["relative_path"])
            try:
                path = ledger.resolve_ref(ref)
                observed = file_stat(path)
                status = "exists" if same_stat(ref["observed"], observed) else "changed"
                if full:
                    value = sha256_file(path, cancel)
                    previous = ref["observed"].get("sha256")
                    if previous:
                        status = "exists" if previous == value else "changed"
                        if status == "exists" and not same_stat(ref["observed"], observed):
                            ref["observed"].update(observed)
                            changed = True
                    elif status == "exists":
                        ref["observed"].update(sha256=value, fingerprint_status="sha256")
                        changed = True
            except Cancelled:
                raise
            except FileNotFoundError:
                status = "missing"
            except (OSError, LedgerError):
                status = "unreadable"
            if ref.get("status") != status:
                ref["status"] = status
                changed = True
            results.append({"run_id": run["run_id"], "path": ref["relative_path"], "status": ref["status"]})
        if changed:
            ledger.save_run(run, run["revision"])
    return results
