"""Human-readable CSV and reproducible AI bundles; never imports CSV as the ledger."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

from .storage import (now, digest, atomic_json, new_id, LedgerError, Cancelled, latest_annotations,
                      same_stat, file_stat, sha256_file)
from .presentation import dataset_info, run_time, time_text
from .features import extract, number, DEFINITIONS, INITIAL_METRICS


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
            source = run["settings_sources"].get(run.get("settings", {}).get("settings_id"), {})
            values = source.get("values", {})
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
                         "学習率": values.get("learning_rate"), "設定出典": source.get("source"),
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
    rows = listing_rows(ledger, run_ids) if kind == "listing" else evaluation_rows(ledger, run_ids, history)
    if run_ids is not None:
        order = {run_id: i for i, run_id in enumerate(run_ids)}
        rows.sort(key=lambda row: order.get(row.get("run_id"), len(order)))
    export_columns = columns or list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    try:
        write_csv(destination, rows, export_columns, excel=True)
    except PermissionError as exc:
        raise LedgerError("CSVを書き込めません。Excelで閉じるか、別名で保存してください。") from exc
    atomic_json(str(destination) + ".columns.json", {
        "schema_version": 1, "kind": "ledger_csv_snapshot", "created_at": now(),
        "ledger_id": ledger.config()["ledger_id"], "selection": run_ids,
        "columns": export_columns,
        "identifiers": "run_id=学習、artifact_id=重み、review_id=評価。値は台帳のID。",
        "datetimes": "タイムゾーン付きISO8601。Excelで自動変換する場合は文字列として読み込む。",
        "excel_text_escape": "先頭が=+-@の文字列には引用符を付与。正本には影響しない。",
        "roundtrip": False})
    return len(rows)


def export_bundle(ledger, run_ids=None, progress=None, cancel=None, include_unverified=False):
    from .features import auto_events
    from .research import read_archive, legacy_features
    from .storage import stable_id
    export_id = now().replace(":", "").replace("+", "_") + "_" + new_id()[:8]
    output = ledger.directory / "derived" / export_id
    output.mkdir(parents=True)
    runs = [r for r in ledger.list_runs() if run_ids is None or r["run_id"] in run_ids]
    selected = {r["run_id"] for r in runs}
    reviews = []
    for review in ledger.reviews():
        involved = {a["run_id"] for a in review.get("annotations", [])}
        involved |= {p.get("run_id") for c in review.get("candidates", []) for p in c.get("components", [])}
        if involved & selected:
            reviews.append(review)
    counterpart_ids = {p.get("run_id") for r in reviews for c in r.get("candidates", []) for p in c.get("components", [])}
    for other in ledger.list_runs():
        if other["run_id"] in counterpart_ids - selected:
            runs.append(other)
    candidates = [dict(c, review_id=r["review_id"]) for r in reviews for c in r.get("candidates", [])]
    comparisons = [dict(p, review_id=r["review_id"], use=r.get("use"),
                        evidence_mode=r.get("evidence_mode"), selection_mode=r.get("selection_mode"))
                   for r in reviews for p in r.get("comparisons", [])]
    features, issues, source_audit = [], [], []
    targets = {r["run_id"]: {r["run_id"]: None} for r in runs}
    run_map = {r["run_id"]: r for r in runs}
    for candidate in candidates:
        if not candidate.get("generation_link_verified"):
            issues.append({"candidate_id": candidate["candidate_id"], "reason": "生成当時の重み内容との一致は未確認"})
        for component in candidate.get("components", []):
            run = run_map.get(component.get("run_id"))
            artifact = next((a for a in run["artifacts"] if a["artifact_id"] == component.get("artifact_id")), None) if run else None
            if not artifact:
                issues.append({"candidate_id": candidate["candidate_id"], "reason": "対象重みの対応未確認"})
                continue
            if artifact["ref"].get("status") != "exists" or component.get("identity_status") == "changed":
                issues.append({"candidate_id": candidate["candidate_id"], "artifact_id": artifact["artifact_id"],
                               "reason": "評価対象の重みは変更または欠落しています"})
            epoch = number(artifact.get("epoch"))
            if epoch is None:
                issues.append({"run_id": run["run_id"], "artifact_id": artifact["artifact_id"], "reason": "checkpoint epoch不明"})
            else:
                targets[run["run_id"]][artifact["artifact_id"]] = epoch
    usable = {}
    raw_kinds = {"gradient", "dq", "dq_auto", "rank", "avg", "group_loss", "group_epoch"}
    extraction_cache = {}
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    import hashlib
    extractor_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in code_files)).hexdigest()

    def calculate(path, kind, cutoff=None, allowed=None, diagnostic=False):
        key = (str(path), kind, cutoff, tuple(allowed) if allowed is not None else None, diagnostic)
        if key not in extraction_cache:
            if kind == "dq_auto":
                value = auto_events(path, cutoff, cancel)
            else:
                value = extract(path, kind, cutoff, cancel, allowed, diagnostic)
            rows, warnings, source_hash = value
            extraction_cache[key] = ([v for v in rows if v["metric_id"] in INITIAL_METRICS], warnings, source_hash)
        import copy
        return copy.deepcopy(extraction_cache[key])

    def decorate(values, run, ref, subject, window, **extra):
        for value in values:
            value.update(run_id=run["run_id"], subject_id=subject, window_id=window,
                         source_ref=ref["ref_id"], source_kind=ref["kind"],
                         association=ref["association"]["status"], **extra)
        return values

    try:
        wanted = set()
        for run in runs:
            usable[run["run_id"]] = []
            for ref in run["source_refs"]:
                if ref["kind"] not in raw_kinds | {"diagnostic"}:
                    continue
                verified = ref.get("association", {}).get("status") in ("verified", "user_asserted")
                if not verified and not include_unverified:
                    issues.append({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": "runとの対応未確認"})
                    continue
                if not verified:
                    issues.append({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": "対応未確認（指定により仮出力）"})
                try:
                    path = ledger.resolve_ref(ref)
                    if ref.get("status") != "exists" or not same_stat(ref["observed"], file_stat(path)):
                        raise LedgerError("元ファイル欠落または変更")
                    usable[run["run_id"]].append((ref, path))
                    wanted.add(os.path.normcase(str(path)))
                except (OSError, ValueError) as exc:
                    issues.append({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": str(exc)})
        archive, archive_audit = {}, {}
        try:
            archive, archive_audit, archive_issues = read_archive(ledger, wanted, cancel, progress)
            issues.extend({"reason": issue} for issue in archive_issues)
        except Cancelled:
            raise
        except (OSError, ValueError, KeyError) as exc:
            issues.append({"reason": "既存研究を参照できません: " + str(exc)})
        for run in runs:
            available_kinds = {ref["kind"] for ref, _ in usable[run["run_id"]]}
            if not usable[run["run_id"]]:
                issues.append({"run_id": run["run_id"], "reason": "利用可能な確認済みの学習ログがありません"})
            for ref, path in usable[run["run_id"]]:
                if cancel and cancel():
                    raise Cancelled("分析出力を中止しました")
                diagnostic = ref["kind"] == "diagnostic"
                if diagnostic and "gradient" in available_kinds:
                    continue
                kind = "gradient" if diagnostic else ref["kind"]
                try:
                    for subject, cutoff in targets[run["run_id"]].items():
                        if progress:
                            progress(f"指標を抽出: {run['display_name']} / {kind}")
                        values, warnings, source_hash = calculate(path, kind, cutoff, diagnostic=diagnostic)
                        expected_hash = ref["observed"].get("sha256")
                        if expected_hash and expected_hash != source_hash:
                            raise LedgerError("元ファイルの内容hashが一致しません")
                        features.extend(decorate(values, run, ref, subject,
                                                 "checkpoint_local" if cutoff is not None else "observed_run"))
                        issues.extend({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": w} for w in warnings)
                        source_audit.append({"ref_id": ref["ref_id"], "root": ref["root"],
                                             "relative_path": ref["relative_path"], "sha256": source_hash,
                                             "cutoff_epoch": cutoff, "diagnostic_fallback": diagnostic})
                        for value in values:
                            if value["status"] not in ("ok", "recorded_events"):
                                issues.append({"run_id": run["run_id"], "subject_id": subject,
                                               "metric_id": value["metric_id"], "reason": value["status"]})
                        if cutoff is None and os.path.normcase(str(path)) in archive:
                            legacy, warnings = legacy_features(archive[os.path.normcase(str(path))], source_hash, kind)
                            features.extend(decorate(legacy, run, ref, subject, "legacy_observed_epochs"))
                            issues.extend({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": w} for w in warnings)
                except Cancelled:
                    raise
                except (OSError, ValueError, KeyError) as exc:
                    issues.append({"run_id": run["run_id"], "ref": ref["ref_id"], "reason": str(exc)})
        # Cross-run pairs additionally receive identically bounded observed-epoch windows.
        # Same-run epoch pairs keep their separate checkpoint-local histories.
        candidate_map = {(c["review_id"], c["candidate_id"]): c for c in candidates}
        completed_groups = set()
        local = list(features)
        for pair in comparisons:
            endpoints = [candidate_map.get((pair["review_id"], pair[side]), {}).get("components", [])
                         for side in ("left", "right")]
            if any(len(parts) != 1 for parts in endpoints):
                pair["feature_comparison_status"] = "compound_or_unresolved_candidate"
                continue
            a, b = (parts[0] for parts in endpoints)
            if not all(p.get("artifact_id") and p.get("run_id") in run_map for p in (a, b)):
                continue
            if a["run_id"] == b["run_id"]:
                pair["feature_comparison_status"] = "same_run_checkpoint_local"
                continue
            group = stable_id(*sorted([a["artifact_id"], b["artifact_id"]]))
            pair["common_window_id"] = group
            if group in completed_groups:
                continue
            completed_groups.add(group)
            left = [f for f in local if f["subject_id"] == a["artifact_id"] and f["window_id"] == "checkpoint_local"]
            right = [f for f in local if f["subject_id"] == b["artifact_id"] and f["window_id"] == "checkpoint_local"]
            requests = {}
            for x in left:
                for y in right:
                    if x["status"] == "metric_version_unrecorded" or y["status"] == "metric_version_unrecorded":
                        continue
                    if (x["metric_id"], x["dimension"], x["definition_version"]) != (
                            y["metric_id"], y["dimension"], y["definition_version"]):
                        continue
                    epochs = tuple(sorted(set(x["epochs"]) & set(y["epochs"])))
                    if not epochs:
                        continue
                    for f, component in ((x, a), (y, b)):
                        key = (component["run_id"], f["source_ref"], component["artifact_id"], epochs)
                        requests.setdefault(key, set()).add((f["metric_id"], f["dimension"]))
            if not requests:
                issues.append({"common_window_id": group, "reason": "共通の定義・観測epochがありません"})
            for (run_id, ref_id, subject, epochs), wanted_metrics in requests.items():
                ref, path = next(item for item in usable[run_id] if item[0]["ref_id"] == ref_id)
                if ref["kind"] == "dq_auto":
                    continue
                diagnostic = ref["kind"] == "diagnostic"
                kind = "gradient" if diagnostic else ref["kind"]
                try:
                    values, warnings, source_hash = calculate(path, kind, max(epochs), epochs, diagnostic)
                    values = [v for v in values if (v["metric_id"], v["dimension"]) in wanted_metrics]
                    features.extend(decorate(values, run_map[run_id], ref, subject, "common_interval", common_window_id=group))
                    issues.extend({"common_window_id": group, "reason": w} for w in warnings)
                except Cancelled:
                    raise
                except (OSError, ValueError) as exc:
                    issues.append({"common_window_id": group, "reason": str(exc)})
        write_csv(output / "runs.csv", [{"run_id": r["run_id"], "name": r["display_name"],
                  "revision": r["revision"], "association": r["association"], "dataset": dataset_info(r),
                  "training_time": time_text(run_time(r), iso=True), "training_time_source": run_time(r)["source"],
                  "purpose": r.get("user", {}).get("experiment_note"), "settings": r["settings"],
                  "settings_sources": r["settings_sources"]} for r in runs])
        write_csv(output / "candidates.csv", candidates)
        write_csv(output / "comparisons.csv", comparisons)
        write_csv(output / "features.csv", features)
        write_csv(output / "issues.csv", issues)
        with (output / "reviews.jsonl").open("w", encoding="utf-8") as stream:
            for review in reviews:
                stream.write(json.dumps(review, ensure_ascii=False, allow_nan=False) + "\n")
        atomic_json(output / "feature_definitions.json", DEFINITIONS)
        manifest = {"schema_version": 1, "kind": "ledger_export", "export_id": export_id,
                    "created_at": now(), "ledger_id": ledger.config()["ledger_id"],
                    "selection": {"run_ids": run_ids, "include_unverified": include_unverified},
                    "runs": [{"run_id": r["run_id"], "revision": r["revision"], "hash": digest(r)} for r in runs],
                    "reviews": [{"review_id": r["review_id"], "revision": r["revision"], "hash": digest(r)} for r in reviews],
                    "sources": source_audit,
                    "registered_references": [dict(ref, run_id=r["run_id"]) for r in runs
                                              for ref in r["source_refs"] + [a["ref"] for a in r["artifacts"]]],
                    "research_archive": archive_audit,
                    "extractor_sha256": extractor_hash, "definitions_hash": digest(DEFINITIONS),
                    "status": "complete", "issues_count": len(issues)}
        atomic_json(output / "export_manifest.json", manifest)
        (output / "README.md").write_text(
            "# LoRA研究台帳・分析用出力\n\n"
            f"学習記録 {len(runs)} 件、評価 {len(reviews)} 件、指標 {len(features)} 行、注意 {len(issues)} 件。\n\n"
            "最初にissues.csvとfeature_definitions.jsonを確認してください。\n"
            "runs.csvには比較相手の学習を含む場合があります。\n"
            "未評価は不合格ではありません。星は主観的な好み、採用は運用判断です。\n"
            "画像数・ログ行数・checkpoint数を独立学習数と数えないでください。\n"
            "observed_runは事後分析用、checkpoint_localは対象checkpointまでです。\n"
            "common_intervalは比較両側の同一定義で共通する観測epoch、legacy_*は定義の異なる既存集計です。\n"
            "DQ autoのepoch不明はcheckpointへ結び付けません。診断JSONのfallbackは勾配配列だけです。\n"
            "診断の推定stepは正確なstepとして使いません。欠測を0や既定値に補完しません。\n"
            "content_verified_at_reviewは評価時点のファイル確認であり、生成当時の重みの証明ではありません。\n"
            "推測、反例、次の検証はresearchフォルダへ保存し、人間評価を変更しないでください。\n"
            "元ログは台帳のlocations.local.jsonでrootを解決して参照します。\n",
            encoding="utf-8")
        return output
    except BaseException:
        atomic_json(output / "export_manifest.json", {"schema_version": 1, "kind": "ledger_export",
                    "export_id": export_id, "status": "incomplete", "created_at": now()})
        raise
