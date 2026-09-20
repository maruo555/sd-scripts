"""Optional read-only adapter for an existing files.csv / epoch_metrics.csv archive.

Legacy aggregates retain distinct metric IDs and their extractor code hash.
They are never substituted for the ledger's median-window definitions.
"""
import csv
import json
import os
from pathlib import Path
from .storage import LedgerError, Cancelled, file_stat, same_stat, sha256_file
from .features import METRICS, number


def read_archive(ledger, wanted_paths, cancel=None, progress=None):
    root_id = ledger.config().get("research_archive_root")
    if not root_id:
        return {}, {}, []
    folder = Path(ledger.locations()[root_id])
    files = folder / "data" / "files.csv"
    metrics = folder / "data" / "epoch_metrics.csv"
    before = {p: file_stat(p) for p in (files, metrics)}
    index = {}
    with files.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = os.path.normcase(str(Path(row["path"]).resolve()))
            if path in wanted_paths and row.get("status") == "ok" and row.get("sha256") and row.get("code_hash"):
                if row.get("segments") == "1":
                    index[row["id"]] = row
    results = {key: [] for key in index}
    with metrics.open(encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i % 10000 == 0:
                if cancel and cancel():
                    raise Cancelled("既存研究の読込を中止しました")
                if progress:
                    progress(f"既存研究の集計を確認: {i:,} 行")
            if row["file_id"] in results:
                results[row["file_id"]].append(row)
    for path in before:
        if not same_stat(before[path], file_stat(path)):
            raise LedgerError("読込中に既存研究のCSVが変更されました")
    audit = {"root": root_id, "files_sha256": sha256_file(files, cancel),
             "epoch_metrics_sha256": sha256_file(metrics, cancel),
             "definition": "legacy per-epoch median; distinct from ledger windows"}
    by_path = {os.path.normcase(str(Path(item["path"]).resolve())): (item, results[key])
               for key, item in index.items()}
    return by_path, audit, []


def legacy_features(record, source_hash, kind, cutoff=None):
    item, rows = record
    if item["sha256"] != source_hash:
        return [], ["既存研究のsource hash不一致"]
    allowed = METRICS.get(kind, {})
    values, issues = [], []
    for row in rows:
        if row["metric"] not in allowed:
            continue
        epoch = number(row["epoch"])
        if epoch is None:
            continue
        # Keep recorded legacy coordinates; never use them for checkpoint features
        # until their origin is independently verified.
        if cutoff is not None:
            continue
        try:
            dimension = json.loads(row["dimension"])
        except (ValueError, TypeError):
            issues.append("既存集計のdimensionが不正")
            continue
        status = "legacy_definition"
        if kind in ("dq", "dq_auto") and (not dimension.get("Scope") or not dimension.get("Target")):
            status = "definition_unconfirmed"
        values.append({"metric_id": "legacy." + row["metric"] + ".epoch_median",
                       "value": number(row["median"]) if status == "legacy_definition" else None,
                       "status": status, "epoch": epoch, "dimension": row["dimension"],
                       "scope": dimension.get("Scope"), "target": dimension.get("Target"),
                       "n_valid": row.get("n"), "n_missing": row.get("missing"),
                       "definition_version": "legacy:" + item["code_hash"],
                       "source_sha256": source_hash, "epoch_origin": "legacy_recorded_unverified"})
    return values, issues
