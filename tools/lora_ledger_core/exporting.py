"""Human-readable CSV and reproducible AI bundles; never imports CSV as the ledger."""
from __future__ import annotations
import csv
import json
import os
import tempfile
from pathlib import Path

from .storage import (now, digest, atomic_json, new_id, LedgerError, Cancelled, latest_annotations,
                      same_stat, file_stat, sha256_file)
from .presentation import settings_summary, dataset_info, run_time, time_text
from .features import RunFeatures, DEFINITIONS
from .output_root import output_root
from tools.lora_run_reader import discover_run_indexes
from collections import Counter


def csv_value(value, excel=False):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if excel and isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path, rows, columns=None, excel=False):
    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig" if excel else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or ["status"], lineterminator="\r\n" if excel else "\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key), excel) for key in columns})


def listing_rows(ledger, run_ids=None):
    annotations = latest_annotations(ledger.reviews())
    rows = []
    for run in ledger.list_runs():
        if run_ids is not None and run["run_id"] not in run_ids:
            continue
        dataset = dataset_info(run)
        chronology = run_time(run)
        impressions = [value for key, value in annotations.items() if key[0] == run["run_id"]]
        # One row per annotated target, or an empty row for an unreviewed run.
        for review, annotation in impressions or [(None, {})]:
            artifact_id = annotation.get("artifact_id")
            artifact = next((a for a in run["artifacts"] if a["artifact_id"] == artifact_id), None)
            origin, values = settings_summary(run)
            rows.append({"run_id": run["run_id"], "学習名": run["display_name"],
                         "データセット": dataset["display_name"],
                         "データセット名の出典": dataset["name_source"],
                         "dataset_config": dataset["automatic"].get("config_path"),
                         "学習日時": time_text(chronology, iso=True),
                         "日時の出典": chronology["source_label"],
                         "対象": artifact["name"] if artifact else "学習全体の印象",
                         "artifact_id": artifact_id, "candidate_id": annotation.get("candidate_id"),
                         "お気に入り度": annotation.get("favorite_rating"),
                         "学習の目的": run.get("user", {}).get("experiment_note", ""),
                         "良いところ": annotation.get("good_points", ""),
                         "悪いところ": annotation.get("bad_points", ""),
                         "任意メモ": annotation.get("free_note", ""),
                         "optimizer": values.get("optimizer"), "rank": values.get("network_dim"),
                         "学習率": values.get("learning_rate"), "設定出典": origin,
                         "対応状態": run.get("association"), "旧名": run.get("aliases", []),
                         "review_id": review["review_id"] if review else None,
                         "評価日時": review.get("evaluated_at") if review else None})
    return rows


def evaluation_rows(ledger, run_ids=None, history=False):
    rows = []
    selected = set(run_ids) if run_ids is not None else None
    for review in ledger.reviews(history=history):
        common = {"review_id": review["review_id"], "revision": review["revision"],
                  "evaluated_at": review.get("evaluated_at"), "reviewer": review.get("reviewer"),
                  "use": review.get("use"), "evidence_mode": review.get("evidence_mode"),
                  "selection_mode": review.get("selection_mode"), "blinding": review.get("blinding")}
        for annotation in review.get("annotations", []):
            if selected is None or annotation["run_id"] in selected:
                rows.append({**common, "row_type": "annotation", **annotation})
        candidates = review.get("candidates", [])
        involved = {c.get("run_id") for candidate in candidates for c in candidate.get("components", [])}
        if selected is not None and not involved & selected:
            continue
        # Keep counterpart candidates so the comparison and adoption remain interpretable.
        adoption = review.get("adoption", {})
        for candidate in candidates:
            components = candidate.get("components", [])
            single = components[0] if len(components) == 1 else {}
            candidate_id = candidate["candidate_id"]
            rows.append({**common, "row_type": "candidate", "candidate_id": candidate_id,
                         "candidate_name": candidate.get("name"),
                         "run_id": single.get("run_id"), "artifact_id": single.get("artifact_id"),
                         "strength": single.get("strength"), "lbw": single.get("lbw"),
                         "components": components,
                         "viewed": candidate_id in review["viewed_candidate_ids"] if "viewed_candidate_ids" in review else None,
                         "usability": candidate.get("usability", "unknown"),
                         "usability_reason": candidate.get("usability_reason", ""),
                         "adoption_status": adoption.get("status", "undecided"),
                         "adopted": candidate_id in adoption.get("candidate_ids", [])
                                    if adoption.get("status") in ("selected", "none") else None})
        candidate_map = {c["candidate_id"]: c for c in candidates}
        for pair in review.get("comparisons", []):
            rows.append({**common, "row_type": "comparison", **pair,
                         "left_name": candidate_map.get(pair["left"], {}).get("name"),
                         "right_name": candidate_map.get(pair["right"], {}).get("name")})
    return rows




def export_csv(ledger, destination, kind="listing", run_ids=None, history=False, columns=None):
    if kind not in {"listing", "evaluations"}:
        raise LedgerError("CSVの種類が不明です。")
    rows = listing_rows(ledger, run_ids) if kind == "listing" else evaluation_rows(ledger, run_ids, history)
    if run_ids is not None:
        order = {rid: i for i, rid in enumerate(run_ids)}
        rows.sort(key=lambda r: order.get(r.get("run_id"), len(order)))
    columns = columns or list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    target = Path(destination)
    sidecar = Path(str(target) + ".columns.json")
    if target.exists() or sidecar.exists():
        raise LedgerError("保存済みのCSVまたは列情報があります。新しいファイル名を指定してください。")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".ledger-export-") as temp:
        csv_file, json_file = Path(temp, "export.csv"), Path(temp, "columns.json")
        write_csv(csv_file, rows, columns, excel=True)
        atomic_json(json_file, {"schema_version": 1, "kind": "ledger_csv_snapshot", "created_at": now(),
                    "ledger_id": ledger.config()["ledger_id"], "selection": run_ids, "columns": columns,
                    "csv_sha256": sha256_file(csv_file), "roundtrip": False,
                    "excel_text_escape": "先頭が=+-@の文字列には引用符付与", "history": history, "listing_kind": kind})
        # Install complete files without overwriting output from a concurrent exporter.
        # Windows rename is exclusive, including on drives without hard-link support.
        install = os.rename if os.name == "nt" else os.link
        sidecar_digest = sha256_file(json_file)
        install(json_file, sidecar)
        try:
            install(csv_file, target)
        except BaseException:
            if sidecar.is_file() and sha256_file(sidecar) == sidecar_digest:
                sidecar.unlink()
            raise
    return len(rows)


def _involved(review):
    runs = {a.get("run_id") for a in review.get("annotations", [])}
    for candidate in review.get("candidates", []):
        runs.update(c.get("run_id") for c in candidate.get("components", []))
    return runs - {None}


def _population(ledger, run_ids):
    runs = {r["run_id"]: r for r in ledger.list_runs()}
    selected = set(runs) if run_ids is None else set(run_ids)
    if selected - runs.keys():
        raise LedgerError("選択したrunが台帳にありません。")
    reviews = [r for r in ledger.reviews() if _involved(r) & selected]
    wanted = selected.union(*(_involved(r) for r in reviews))
    return {k:r for k,r in runs.items() if k in wanted}, reviews


def _revisions(runs, reviews):
    return {
        "runs": [{"run_id": r["run_id"], "revision": r["revision"], "hash": digest(r)} for r in runs.values()],
        "reviews": [{"review_id": r["review_id"], "revision": r["revision"], "hash": digest(r)} for r in reviews],
    }


def _unverified_sources(reader, run, root):
    blocked = {}
    refs = {r["kind"]: r for r in run["source_refs"]}
    for kind, item in reader.index["artifacts"].items():
        if item.get("status") == "disabled":
            continue
        if kind == "diagnostic_report":
            pair = [refs.get("diagnostic"), refs.get("diagnostic_html")]
            if any(not r or r.get("generation_id") != item.get("generation_id") for r in pair):
                blocked[kind] = "未登録または変更された診断世代。再走査してください。"
            continue
        ref = refs.get(kind)
        if not ref or ref["status"] not in {"exists", "missing"}:
            blocked[kind] = "未登録または所属未確認の入力。再走査してください。"
            continue
        try:
            path, _ = reader.artifact(kind, 2 if kind == "training_settings" else 1)
            if path != (root / ref["relative_path"]).resolve():
                blocked[kind] = "登録時の入力場所と一致しません。"
            elif path.exists() and (ref["status"] != "exists" or not same_stat(ref["observed"], file_stat(path))):
                blocked[kind] = "登録後に追記・変更された入力。再走査してください。"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blocked[kind] = str(exc)
    return blocked


def export_bundle(ledger, run_ids=None, progress=None, cancel=None):
    if ledger.read_only:
        raise LedgerError("旧台帳への保存は検証コピーを作成してから行ってください。")
    root_id, output_root_path = output_root(ledger)
    # Acquire a consistent group of ledger documents. Training writers use a distinct lock.
    with ledger.write_lock():
        ledger._recover()
        runs, reviews = _population(ledger, run_ids)
        revisions = _revisions(runs, reviews)
    export_id = now().replace(":", "").replace("+", "_") + "_" + new_id()[:8]
    directory = ledger.directory / "derived" / export_id
    directory.mkdir(parents=True)
    start = {"schema_version": 1, "kind": "ledger_export", "export_id": export_id, "created_at": now(),
             "ledger_id": ledger.config()["ledger_id"], "selection": {"run_ids": run_ids},
             "status": "incomplete", **revisions}
    atomic_json(directory / "export_manifest.json", start)
    candidates, comparisons, features, issues = [], [], [], []
    by_run, by_artifact, candidate_map, audit = {}, {}, {}, {}
    selected_artifacts = set()

    def check_cancel():
        if cancel and cancel():
            raise Cancelled("分析用書き出しを中止しました。")
    for review in reviews:
        review_id = review["review_id"]
        for candidate in review.get("candidates", []):
            candidates.append({**candidate, "review_id": review_id})
            candidate_map[(review_id, candidate["candidate_id"])] = candidate
            for c in candidate.get("components", []):
                if c.get("artifact_id") and c.get("run_id"):
                    selected_artifacts.add((c["run_id"], c["artifact_id"]))
        comparisons.extend({**p, "review_id": review_id} for p in review.get("comparisons", []))
    discovered, discovery_errors = discover_run_indexes(output_root_path)
    counts = Counter(r.run.run_id for r in discovered)
    readers = {r.run.run_id: r for r in discovered if counts[r.run.run_id] == 1}
    issues.extend({"source": "run.json", "reason": e["reason"], "folder": e["directory"]} for e in discovery_errors)
    weight_inputs = []
    for run_id, run in runs.items():
        check_cancel()
        if progress:
            progress(run["display_name"])
        identity = run.get("identity", {})
        if identity.get("root") != root_id or identity.get("run_id") != run_id:
            issues.append({"run_id": run_id, "reason": "旧または手動登録のrun。保存済みの評価・exportを参照。"})
            continue
        try:
            relative = identity["relative_path"]
            if Path(relative).name != relative or relative in {".", ".."}:
                raise LedgerError("登録済みrunの相対位置が不正です。")
            directory_path = (output_root_path / relative).resolve()
            if directory_path.parent != output_root_path:
                raise LedgerError("出力フォルダの外にあるrunです。")
            reader = readers.get(run_id)
            if reader is None or reader.run.directory != directory_path:
                raise LedgerError("run索引が移動・重複しています。登録を確認してください。")
            if (reader.run.run_id, reader.run.session_id, reader.index["started_at"]) != (
                    run_id, identity["session_id"], identity["started_at"]):
                raise LedgerError("登録時のrun識別情報と一致しません。")
            data = RunFeatures.load(reader, cancel, _unverified_sources(reader, run, output_root_path))
            by_run[run_id] = data
            audit[run_id] = {"sources": data.sources, "settings_sha256": digest(data.resolved),
                             "expected_rank_sha256": digest(data.expected), "definition_version": DEFINITIONS["version"]}
            issues.extend({"run_id": run_id, "source": kind, "reason": reason} for kind, reason in data.issues)
            for kind, source in data.sources.items():
                if source["status"] not in {"complete","open","disabled"}:
                    issues.append({"run_id": run_id, "source": kind, "reason": source["status"]})
            for row in data.summarize():
                features.append({**row, "run_id": run_id, "subject_id": run_id, "window_id": "observed_run"})
            for artifact in run["artifacts"]:
                aid = artifact["artifact_id"]
                if (run_id, aid) not in selected_artifacts:
                    continue
                if artifact["ref"]["status"] != "exists":
                    issues.append({"run_id": run_id, "artifact_id": aid, "reason": artifact["ref"]["status"]})
                    continue
                rpath = (output_root_path / artifact["ref"]["relative_path"]).resolve()
                moment = {k: artifact.get(k) for k in ("role","epoch","attempt_step","batch_index","phase")}
                indexed = next((i for i in reader.index["weights"] if all(i.get(k) == v for k,v in moment.items())
                                and i.get("path") == rpath.relative_to(directory_path).as_posix()), None)
                if indexed is None or reader.weight(indexed, data.resolved)[1] not in {"indexed","session_matched"}:
                    issues.append({"run_id": run_id, "artifact_id": aid, "reason": "重みの範囲・所属が未確認"})
                    continue
                observed = artifact["ref"]["observed"]
                if not same_stat(observed, file_stat(rpath)) or (observed.get("sha256") and
                        observed["sha256"] != sha256_file(rpath, cancel)):
                    issues.append({"run_id": run_id, "artifact_id": aid, "reason": "登録時の重みと一致しません"})
                    continue
                weight_inputs.append((rpath, dict(observed)))
                moment["epoch_complete"] = indexed.get("epoch_complete")
                by_artifact[(run_id,aid)] = moment
                for row in data.summarize(moment):
                    features.append({**row,"run_id":run_id,"subject_id":aid,"window_id":"checkpoint_local"})
        except Cancelled:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"run_id":run_id,"reason":str(exc)})

    def single(candidate):
        parts = candidate.get("components", []) if candidate else []
        if len(parts) != 1 or parts[0].get("identity_status") == "changed":
            return None
        key = (parts[0].get("run_id"), parts[0].get("artifact_id"))
        return key if key in by_artifact else None

    for pair in comparisons:
        check_cancel()
        left = single(candidate_map.get((pair["review_id"], pair["left"])))
        right = single(candidate_map.get((pair["review_id"], pair["right"])))
        if not left or not right or left[0] == right[0]:
            continue
        for values in by_run[left[0]].compare(by_run[right[0]], by_artifact[left], by_artifact[right]):
            for key, other, row in zip((left,right), (right,left), values):
                features.append({**row, "window_id": "common_interval", "run_id": key[0], "subject_id": key[1],
                                 "counterpart_id": other[1], "review_id": pair["review_id"], "event_id": pair.get("event_id")})
    write_csv(directory / "runs.csv", [{"run_id": r["run_id"], "revision": r["revision"], "name": r["display_name"],
              "purpose": r.get("purpose"), "training_status": r.get("training_status"),
              "dataset": dataset_info(r), "settings": r.get("settings"), "user": r.get("user")} for r in runs.values()])
    write_csv(directory / "candidates.csv", candidates)
    write_csv(directory / "comparisons.csv", comparisons)
    write_csv(directory / "features.csv", features)
    write_csv(directory / "issues.csv", issues)
    with (directory / "reviews.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for review in reviews:
            out.write(json.dumps(review, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_json(directory / "feature_definitions.json", DEFINITIONS)
    (directory / "README.md").write_text(
        "# LoRA研究台帳の分析用出力\n\n件数と欠測は issues.csv、指標の定義は feature_definitions.json を参照してください。"
        "ユーザーの評価は reviews.jsonl にあります。重み時点と比較共通区間を window_id で区別します。\n",
        encoding="utf-8")
    check_cancel()
    for data in by_run.values():
        data.verify_inputs()
    for path, observed in weight_inputs:
        if not same_stat(observed, file_stat(path)) or (observed.get("sha256") and
                sha256_file(path, cancel) != observed["sha256"]):
            raise LedgerError("集計中に評価対象の重みが変更されました。")
    current_indexes, _ = discover_run_indexes(output_root_path)
    final_counts = Counter(r.run.run_id for r in current_indexes)
    if any(final_counts[key] != 1 for key in by_run):
        raise LedgerError("集計中にrun索引が移動・重複しました。")
    with ledger.write_lock():
        if revisions != _revisions(*_population(ledger, run_ids)):
            raise LedgerError("出力中に学習記録または評価の版が変わりました。再実行してください。")
        manifest = {**start, "status": "complete", "sources": audit, "issues_count": len(issues),
                    "definitions_hash": digest(DEFINITIONS), "registered_references": [
                        dict(ref, run_id=r["run_id"]) for r in runs.values()
                        for ref in r["source_refs"] + [a["ref"] for a in r["artifacts"]]]}
        atomic_json(directory / "export_manifest.json", manifest)
    return directory
