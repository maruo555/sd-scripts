"""Register only indexed runs immediately under this ledger's training output directory."""
from collections import Counter
import copy
from pathlib import Path

from library.lora_run_schema import HEADERS
from tools.lora_run_reader import discover_run_indexes, owned_path

from .output_root import output_root
from .presentation import refresh_dataset
from .storage import (Cancelled, Conflict, LedgerError, VERSION, digest, file_stat, now,
                      same_stat, sha256_file, stable_id)


STATES = {"new": "新しい学習", "additional": "更新あり", "unchanged": "登録済み",
          "changed": "要確認", "error": "読込不能", "missing": "見つからない", "offline": "参照先オフライン"}
MOMENT = ("role", "epoch", "attempt_step", "batch_index", "phase")


def _cancel(cancel):
    if cancel and cancel():
        raise Cancelled("処理を中止しました。反映済みのrunは保持されます。")


def _empty_run(reader):
    return {"schema_version": VERSION, "kind": "ledger_run", "run_id": reader.run.run_id,
            "revision": 0, "display_name": reader.run.base_name, "aliases": [], "association": "verified",
            "purpose": "unknown", "training_status": "unknown", "user": {"experiment_note": ""},
            "dataset": {}, "artifacts": [], "settings_sources": {}, "settings": {}, "source_refs": []}


def _ref(root, root_id, path, kind, status="exists", observed=None):
    relative = path.relative_to(root).as_posix()
    if observed is None and status == "exists":
        try:
            observed = file_stat(path)
        except FileNotFoundError:
            status = "missing"
    return {"root": root_id, "relative_path": relative, "kind": kind, "status": status,
            "ref_id": stable_id(root_id, relative, kind), "observed": observed or {},
            "association": {"status": "verified", "method": "run_index"}}


def _weights(reader, root, root_id, old, resolved, full, cancel, issues):
    pending = {item["artifact_id"]: copy.deepcopy(item) for item in old}
    current = []
    seen = set()
    for item in reader.index["weights"]:
        _cancel(cancel)
        moment = tuple(item.get(k) for k in MOMENT)
        relative = item["path"]
        path, status, reason = reader.weight(item, resolved)
        if reason:
            issues.append(f"{relative}: {reason}")
        if path is None:
            continue
        if moment[0] not in {"step", "epoch", "final", "final_raw"} or (moment, relative) in seen:
            raise LedgerError("重みの役割または観測位置が重複・不明です。")
        seen.add((moment, relative))
        ref = _ref(root, root_id, path, "weight", "exists" if status in {"indexed", "session_matched"} else status)
        ref["association"] = {"status": status, "method": "run_index_and_weight_header"}
        same_moment = [a for a in pending.values() if tuple(a.get(k) for k in MOMENT) == moment
                       and a["ref"].get("status") not in {"superseded", "unverified_move"}]
        prior = next((a for a in same_moment if a["ref"]["relative_path"] == ref["relative_path"]), None)
        body_sha = None
        if ref["status"] == "exists":
            if ref["status"] == "exists" and (full or any(a["ref"]["observed"].get("sha256")
                    and (a is not prior or not same_stat(a["ref"]["observed"], ref["observed"])) for a in same_moment)):
                body_sha = sha256_file(path, cancel)
                ref["observed"]["sha256"] = body_sha
        existing = None
        if prior and (ref["status"] in ("pruned", "missing") or
                same_stat(prior["ref"]["observed"], ref["observed"]) and
                (not body_sha or not prior["ref"]["observed"].get("sha256")
                 or body_sha == prior["ref"]["observed"]["sha256"])):
            existing = prior
        if body_sha:
            matches = [a for a in same_moment if a["ref"]["observed"].get("sha256") == body_sha]
            if existing is None and len(matches) == 1:
                existing = matches[0]
        if existing:
            pending.pop(existing["artifact_id"])
            artifact_id = existing["artifact_id"]
            previous = existing["ref"]["observed"]
            if ref["status"] in ("missing", "pruned"):
                ref["observed"] = previous
            elif previous.get("sha256") and not body_sha:
                ref["observed"].update({k: v for k, v in previous.items() if k in ("sha256", "verified_at")})
            if body_sha:
                ref["observed"]["verified_at"] = previous.get("verified_at") or now()
        else:
            artifact_id = stable_id(reader.run.run_id, *moment, ref["relative_path"],
                                    digest(ref["observed"]))
            if body_sha:
                ref["observed"]["verified_at"] = now()
        new = dict(artifact_id=artifact_id, name=path.name, ref=ref, **dict(zip(MOMENT, moment)))
        if existing and existing.get("predecessor_status"):
            new["predecessor_status"] = existing["predecessor_status"]
        if not existing and same_moment:
            moved = [a for a in same_moment if a["ref"]["relative_path"] != ref["relative_path"]]
            if moved:
                new["predecessor_status"] = "unverified_move"
                for other in moved:
                    other["ref"]["status"] = "unverified_move"
            else:
                new["predecessor_status"] = "replacement"
        current.append(new)
    for prior in pending.values():
        if prior["ref"].get("status") not in ("superseded", "unverified_move", "pruned", "missing"):
            prior["ref"]["status"] = "superseded"
        current.append(prior)
    return current


def _propose(reader, root, root_id, previous, full, cancel, issues):
    directory, index = reader.run.directory, reader.index
    identity = {"run_id": reader.run.run_id, "session_id": reader.run.session_id, "started_at": index["started_at"],
                "root": root_id, "relative_path": directory.name}
    if not reader.run.session_id or not identity["started_at"]:
        raise LedgerError("索引にsessionまたは開始日時がありません。")
    if previous:
        old_id = previous.get("identity", {})
        if any(old_id.get(k) != identity[k] for k in ("run_id", "session_id", "started_at", "root")):
            raise Conflict("既登録のrun識別情報と一致しません。")
        previous_path = root / old_id["relative_path"]
        if previous_path != directory and previous_path.exists():
            raise Conflict("旧runフォルダが残っています。")
    run = copy.deepcopy(previous) if previous else _empty_run(reader)
    run.update(identity=identity, indexed_status=index["status"], index_digest=digest(index))
    resolved, settings_status = reader.settings()
    run["settings_status"] = settings_status
    refs = [_ref(root, root_id, directory / "run.json", "run_index")]
    for kind in ("training_settings", *HEADERS):
        if kind not in index["artifacts"]:
            continue
        try:
            path, info = reader.artifact(kind, 2 if kind == "training_settings" else 1)
            if info.get("status") == "disabled":
                continue
            ref = _ref(root, root_id, path, kind)
            ref["schema_version"], ref["indexed_status"] = info["schema_version"], info["status"]
            if kind == "training_settings" and settings_status == "identity_or_version_mismatch":
                ref["status"] = "unverified"
            refs.append(ref)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{kind}: {exc}")
    report_info = index["artifacts"].get("diagnostic_report")
    if report_info:
        from tools.lora_report_publication import read_published_report
        report, reason = read_published_report(reader)
        if report is None:
            issues.append(f"diagnostic_report: {reason}")
        else:
            for key, kind in (("path", "diagnostic"), ("html_path", "diagnostic_html")):
                path = owned_path(directory, report_info[key])
                ref = _ref(root, root_id, path, kind)
                ref.update(schema_version=report_info["schema_version"], generation_id=report_info["generation_id"])
                refs.append(ref)
    run["source_refs"] = refs
    if resolved is None:
        run.update(settings_sources={}, settings={})
    else:
        run["settings_sources"] = {"resolved": {"source": "resolved", "resolved": resolved, "values": resolved.get("args", {})}}
        run["settings"] = {"settings_id": "resolved", "values": resolved.get("args", {})}
    refresh_dataset(run)
    run["artifacts"] = _weights(reader, root, root_id, run["artifacts"], resolved, full, cancel, issues)
    return run


def _indexes(root):
    readers, errors = discover_run_indexes(root)
    counts = Counter(reader.run.run_id for reader in readers)
    return readers, counts, [{"name": item["directory"], "status": "error", "message": item["reason"]} for item in errors]


def scan(ledger, progress=None, cancel=None, full=False, run_ids=None):
    root_id, root = output_root(ledger)
    readers, counts, issues = _indexes(root)
    rows = []
    selected = None if run_ids is None else set(run_ids)
    for reader in readers:
        _cancel(cancel)
        run_id = reader.run.run_id
        if selected is not None and run_id not in selected:
            continue
        if counts[run_id] != 1:
            issues.append({"name": reader.run.directory.name, "status": "changed", "message": "run UUIDが重複しています。"})
            continue
        if progress:
            progress(reader.run.directory.name)
        try:
            path = ledger.directory / "runs" / f"{run_id}.json"
            previous = ledger.get_run(run_id) if path.is_file() else None
            notes = []
            run = _propose(reader, root, root_id, previous, full, cancel, notes)
            issues.extend({"name": run["display_name"], "status": "changed", "message": note} for note in notes)
            rows.append({"run_id": run_id, "name": run["display_name"], "run": run, "association": "verified",
                         "status": ("unchanged" if run == previous else "additional" if previous else "new"),
                         "artifact_count": len(run["artifacts"]), "expected_revision": run["revision"]})
        except Cancelled:
            raise
        except (OSError, LedgerError, ValueError, KeyError, TypeError, AttributeError) as exc:
            issues.append({"name": reader.run.directory.name, "status": "changed", "message": str(exc)})
    for old in ledger.list_runs():
        if ((selected is None or old["run_id"] in selected) and old.get("identity", {}).get("root") == root_id
                and old["run_id"] not in counts):
            issues.append({"name": old["display_name"], "status": "missing", "message": "索引が見つかりません。登録を保持します。"})
    issues.extend({"name": "ledger", "status": "error", "message": message} for message in ledger.errors)
    return {"config_revision": ledger.config()["revision"], "locations_hash": digest(ledger.locations()),
            "rows": rows, "issues": issues, "related": [], "file_count": len(readers), "full": full}


def apply_scan(ledger, result, selected_ids=None, progress=None, cancel=None):
    selected = set(selected_ids) if selected_ids is not None else None
    applied, errors = [], []
    with ledger.write_lock():
        ledger._recover()
        if ledger.config()["revision"] != result["config_revision"] or digest(ledger.locations()) != result["locations_hash"]:
            raise Conflict("参照先が変わりました。再走査してください。")
        root_id, root = output_root(ledger)
        for row in result["rows"]:
            run_id = row["run_id"]
            if row["status"] == "unchanged" or selected is not None and run_id not in selected:
                continue
            if cancel and cancel():
                return {"applied": applied, "errors": errors, "cancelled": True}
            try:
                readers, counts, _ = _indexes(root)
                if counts[run_id] != 1:
                    raise Conflict("索引が移動・重複しています。再走査してください。")
                reader = next(r for r in readers if r.run.run_id == run_id)
                if digest(reader.index) != row["run"]["index_digest"]:
                    raise Conflict("走査後に索引が変更されています。")
                for ref in row["run"]["source_refs"] + [a["ref"] for a in row["run"]["artifacts"]]:
                    if ref["status"] == "exists":
                        path = owned_path(reader.run.directory, (root / ref["relative_path"]).relative_to(reader.run.directory).as_posix())
                        if not same_stat(ref["observed"], file_stat(path)):
                            raise Conflict("走査後に参照先が変更されています。")
                        if result["full"] and ref["observed"].get("sha256") and sha256_file(path, cancel) != ref["observed"]["sha256"]:
                            raise Conflict("走査後に重み内容が変更されています。")
                stored = dict(row["run"])
                stored.setdefault("created_at", now())
                ledger._commit(ledger._run_writes(stored, row["expected_revision"]))
                applied.append(run_id)
                if progress:
                    progress(row["name"])
            except Cancelled:
                return {"applied": applied, "errors": errors, "cancelled": True}
            except (OSError, ValueError, KeyError) as exc:
                errors.append({"name": row["name"], "run_id": run_id, "message": str(exc)})
    return {"applied": applied, "errors": errors, "cancelled": False}


def verify(ledger, full=False, progress=None, cancel=None, run_ids=None):
    root_id, _ = output_root(ledger)
    selected = {r["run_id"] for r in ledger.list_runs() if r.get("identity", {}).get("root") == root_id}
    if run_ids is not None:
        selected.intersection_update(run_ids)
    result = scan(ledger, progress, cancel, full, selected)
    accepted = apply_scan(ledger, result, selected, progress, cancel)
    refs = []
    for row in result["rows"]:
        if selected is not None and row["run_id"] not in selected:
            continue
        for a in row["run"]["artifacts"]:
            refs.append({"run_id": row["run_id"], "artifact_id": a["artifact_id"], "path": a["ref"]["relative_path"],
                         "status": a["ref"]["status"]})
    return {"refs": refs, "issues": result["issues"], **accepted}
