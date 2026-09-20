"""Small, source-grounded metrics. Reads logs sequentially without importing training."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from .storage import Conflict, Cancelled, LedgerError, file_stat, same_stat, digest, read_json, sha256_file

DEFINITION_VERSION = "ledger-1"
METRICS = {
    "gradient": {"Loss": "loss", "Gradient Norm": "logged_grad_norm", "Scale": "grad_scale"},
    "dq": {"QuantErrRatioEMA": "dq_error_ratio", "QuantErrRMSEMA": "dq_error_rms",
           "RMS": "dq_input_rms", "ClipRateEMA": "dq_clip_rate", "RangeMul": "dq_range_mul"},
    "dq_auto": {"AutoApplied": "dq_auto_applied"},
    "rank": {"RankSatP95": "rank_sat_p95", "RankTop1P95": "rank_top1_p95",
             "RankEnergySum": "rank_energy"},
    "group_loss": {"loss": "group_loss"},
    "group_epoch": {"mean_loss_epoch": "group_epoch_loss"},
}
DEFINITIONS = {
    "version": DEFINITION_VERSION, "epoch": "gradient Epoch + 1; other known logs use recorded epoch",
    "aggregation": "finite observations median within epoch, then equal-epoch median",
    "window": "checkpoint_local: ordered observed epochs through checkpoint; i/(n-1); previous=[.6,.8), tail=[.8,1]",
    "minimum_epochs": 5, "delta": "tail median minus previous median",
    "counts": "recorded observations, never independent runs or optimizer updates",
    "missing": "null, nonfinite and unrecorded are not zero; status and counts retained",
    "metrics": METRICS,
    "rank_per_module": "same-epoch/step/scope snapshot; common module set required; p95 unweighted; energy observed sum",
    "limitations": ["logged grad norm includes scale; not a universal gradient quality metric",
                    "threshold exceedance is not skip rate", "observed windows are not planned training progress"],
}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def extract(path, kind, cutoff=None, cancel=None, allowed_epochs=None, diagnostic=False):
    before = file_stat(path)
    h = hashlib.sha256()
    groups = defaultdict(lambda: defaultdict(list))
    missing = defaultdict(int)
    counts = defaultdict(int)
    issues = []
    last_coordinate = None
    seen_coordinates = set()
    rank_snapshot = {}
    rank_key = None
    rank_modules = {}
    step_ranges = defaultdict(list)
    diagnostic_rows = None
    if diagnostic:
        if kind != "gradient":
            raise LedgerError("診断JSONのこの系統は座標・Scopeの出典確認が必要です")
        raw = read_json(path).get("grad") or {}
        epochs = raw.get("epochs", [])
        columns = {"loss": "Loss", "gradient_norm": "Gradient Norm", "scale": "Scale"}
        if not epochs or any(len(raw.get(k, [])) != len(epochs) for k in columns):
            raise LedgerError("診断JSONの勾配配列が不足しています")
        diagnostic_rows = [{"Epoch": epoch - 1, "Step": i,
                            **{v: raw[k][i] for k, v in columns.items()}}
                           for i, epoch in enumerate(epochs)]


    def add(signature, metric, epoch, value):
        key = (signature, metric)
        value = number(value)
        groups[key][epoch]
        if value is None:
            missing[key] += 1
        else:
            groups[key][epoch].append(value)
            counts[key] += 1

    def flush_rank():
        if not rank_snapshot or rank_key is None:
            return
        epoch, step, scope = rank_key
        modules = frozenset(rank_snapshot)
        if scope in rank_modules and rank_modules[scope] != modules:
            issues.append("rankのmodule集合が変化しました")
        rank_modules[scope] = modules
        values = list(rank_snapshot.values())
        signature = json.dumps({"Scope": scope, "kind": "per_module"}, sort_keys=True)
        add(signature, "rank_sat_p95", epoch, quantile([v[0] for v in values if v[0] is not None], .95))
        add(signature, "rank_top1_p95", epoch, quantile([v[1] for v in values if v[1] is not None], .95))
        energy = [v[2] for v in values if v[2] is not None]
        add(signature, "rank_energy", epoch, sum(energy) if len(energy) == len(values) else None)

    def lines(stream):
        for index, raw in enumerate(stream):
            if cancel and cancel():
                raise Cancelled("集計を中止しました")
            h.update(raw)
            yield raw.decode("utf-8-sig" if index == 0 else "utf-8", errors="strict")

    with open(path, "rb") as stream:
        if diagnostic_rows is not None:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                if cancel and cancel():
                    raise Cancelled("集計を中止しました")
                h.update(block)
            rows = diagnostic_rows
        elif kind == "avg":
            rows = (json.loads(line) for line in lines(stream) if line.strip())
        else:
            rows = csv.DictReader(lines(stream))
        for row in rows:
            epoch = number(row.get("Epoch", row.get("epoch")))
            if epoch is None:
                issues.append("epochが記録されていません")
                continue
            epoch += 1 if kind == "gradient" else 0
            # Still consume all bytes for a full source hash, but never use later data as features.
            if (cutoff is not None and epoch > cutoff) or (allowed_epochs is not None and epoch not in allowed_epochs):
                continue
            step = number(row.get("TrainStep", row.get("Step", row.get("global_step", epoch))))
            coordinate = (epoch, step if step is not None else -1)
            if step is not None:
                limits = step_ranges[epoch]
                if not limits:
                    limits.extend([step, step])
                else:
                    limits[0], limits[1] = min(limits[0], step), max(limits[1], step)
            if last_coordinate is not None and coordinate < last_coordinate:
                issues.append("epoch/stepの巻き戻り")
            if coordinate != last_coordinate:
                seen_coordinates.clear()
            last_coordinate = coordinate
            dimension_keys = ("Scope", "Target", "MetricVersion", "Mode", "Stat", "Granularity", "Bits")
            if kind in ("group_loss", "group_epoch"):
                dimension_keys = ("group",)
            dimension = {k: row.get(k) or "unknown" for k in dimension_keys}
            signature = json.dumps(dimension, sort_keys=True)
            collision_key = (coordinate, signature, row.get("Module"))
            if collision_key in seen_coordinates:
                issues.append("同じ座標に複数の観測があります")
                continue
            seen_coordinates.add(collision_key)
            if kind == "rank" and "Module" in row:
                key = (epoch, step, row.get("Scope", "unknown"))
                if rank_key != key:
                    flush_rank()
                    rank_snapshot = {}
                    rank_key = key
                rank_snapshot[row["Module"]] = (number(row.get("RankSat")), number(row.get("RankTop1")),
                                                number(row.get("RankEnergy")))
                continue
            if kind == "avg":
                dimension = {"mode": row.get("selected_candidate_mode", row.get("avg_mode", "unknown")),
                             "bank_size": row.get("bank_size"), "kind": "proxy"}
                signature = json.dumps(dimension, sort_keys=True)
                raw = number(row.get("raw_proxy_loss"))
                selected = number(row.get("selected_proxy_loss", row.get("center_proxy_loss")))
                add(signature, "avg_proxy_relative_delta", epoch,
                    (selected - raw) / abs(raw) if raw not in (None, 0) and selected is not None else None)
                add(signature, "avg_promote_applied", epoch,
                    int(row["promote_applied"]) if type(row.get("promote_applied")) is bool else None)
            else:
                for column, metric in METRICS.get(kind, {}).items():
                    add(signature, metric, epoch, row.get(column))
    flush_rank()
    if not same_stat(before, file_stat(path)):
        raise Conflict("集計中に元ログが変更されました")
    source_hash = h.hexdigest()
    result = []
    for (signature, metric) in sorted(set(groups) | set(missing)):
        by_epoch = groups[(signature, metric)]
        epochs = sorted(by_epoch)
        previous = [epoch for i, epoch in enumerate(epochs) if len(epochs) > 1 and .6 <= i / (len(epochs) - 1) < .8]
        tail = [epoch for i, epoch in enumerate(epochs) if len(epochs) > 1 and i / (len(epochs) - 1) >= .8]
        def median(es):
            values = [statistics.median(by_epoch[e]) for e in es if by_epoch[e]]
            return statistics.median(values) if values else None
        previous_value, tail_value = median(previous), median(tail)
        status = "ok"
        if issues:
            status = "ambiguous_coordinates_or_coverage"
        elif len(epochs) < 5 or previous_value is None or tail_value is None:
            status = "insufficient_epochs"
        dimension = json.loads(signature)
        if kind in ("dq", "dq_auto") and (dimension.get("Scope") == "unknown" or dimension.get("Target") == "unknown"):
            status = "definition_unconfirmed"
        if status == "ok" and kind == "dq" and dimension.get("MetricVersion") == "unknown":
            status = "metric_version_unrecorded"
        base = {"dimension": signature, "scope": dimension.get("Scope"), "target": dimension.get("Target"),
                "step_ranges": {str(e): step_ranges[e] for e in epochs},
                "step_origin": "diagnostic_array_index" if diagnostic else "recorded",
                "status": status, "n_valid": counts[(signature, metric)],
                "n_missing": missing[(signature, metric)], "source_sha256": source_hash,
                "definition_version": DEFINITION_VERSION, "epochs": epochs,
                "previous_epochs": previous, "tail_epochs": tail}
        # Ambiguous records remain auditable but do not emit apparently usable numbers.
        valid = status in ("ok", "metric_version_unrecorded")
        if metric == "avg_promote_applied":
            result.append({**base, "metric_id": metric + ".recorded_count",
                           "value": sum(sum(v) for v in by_epoch.values()) if not issues and counts[(signature, metric)] else None,
                           "status": ("recorded_events" if counts[(signature, metric)] else "unrecorded") if not issues else status})
            continue
        result.append({**base, "metric_id": metric + ".tail_median", "value": tail_value if valid else None})
        if metric not in ("grad_scale", "logged_grad_norm", "dq_input_rms", "dq_error_rms"):
            result.append({**base, "metric_id": metric + ".tail_delta",
                           "value": tail_value - previous_value if valid else None})
    return result, sorted(set(issues)), source_hash


def auto_events(path, cutoff=None, cancel=None):
    before = file_stat(path)
    groups = {}
    issues = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if cancel and cancel():
                raise Cancelled("集計を中止しました")
            scope, target = row.get("Scope"), row.get("Target")
            if not scope or not target:
                continue
            if cutoff is not None:
                epoch = number(row.get("Epoch"))
                if epoch is None:
                    issues.append("DQ autoにはepochがなくcheckpoint区間へ対応できません")
                    continue
                if epoch > cutoff:
                    continue
            key = json.dumps({k: row.get(k) or "unknown" for k in
                              ("Scope", "Target", "Bits", "ClipRateLowAutoPolicyVersion")}, sort_keys=True)
            group = groups.setdefault(key, {"records": 0, "applied": 0, "missing": 0, "escape_recorded": 0, "escape_valid": 0})
            group["records"] += 1
            applied = number(row.get("AutoApplied"))
            if applied in (0, 1):
                group["applied"] += int(applied)
            else:
                group["missing"] += 1
            decision = row.get("ClipRateLowAutoDecision", "")
            if decision:
                group["escape_valid"] += 1
            if decision == "escape_to_mid":
                group["escape_recorded"] += 1
    source_hash = sha256_file(path, cancel)
    if not same_stat(before, file_stat(path)):
        raise Conflict("集計中に元ログが変更されました")
    rows = []
    for dimension, counts in groups.items():
        for metric in ("records", "applied", "escape_recorded"):
            recorded = metric == "records" or (counts["escape_valid"] > 0 if metric == "escape_recorded"
                                                else counts["records"] > counts["missing"])
            rows.append({"metric_id": "dq_auto." + metric, "value": counts[metric] if recorded else None,
                         "dimension": dimension, "scope": json.loads(dimension)["Scope"],
                         "target": json.loads(dimension)["Target"], "epochs": [],
                         "status": "recorded_events" if recorded else "unrecorded", "n_valid": counts["records"]-counts["missing"],
                         "n_missing": counts["missing"], "source_sha256": source_hash,
                         "definition_version": DEFINITION_VERSION,
                         "coordinate": "recorded_epoch" if cutoff is not None else "all_observed_events"})
    return rows, sorted(set(issues)), source_hash

DEFINITIONS["dq_auto"] = "records: recorded rows with Scope/Target; applied: AutoApplied=1; escape_recorded: exact decision='escape_to_mid', not inferred from permission flags; no checkpoint join without recorded epoch"
DEFINITIONS["avg_promote_applied"] = "sum of recorded boolean promote_applied, not median or all-update rate"
DEFINITIONS["common_interval"] = "intersection of observed epochs per matching dimension and metric, through each compared artifact; separate from checkpoint_local"

INITIAL_METRICS = {
    "loss.tail_median", "loss.tail_delta", "grad_scale.tail_median",
    "dq_error_ratio.tail_median", "dq_error_ratio.tail_delta", "dq_error_rms.tail_median",
    "dq_input_rms.tail_median", "dq_clip_rate.tail_median", "dq_range_mul.tail_delta",
    "dq_auto.records", "dq_auto.applied", "dq_auto.escape_recorded",
    "rank_sat_p95.tail_median", "rank_sat_p95.tail_delta", "rank_top1_p95.tail_median",
    "rank_top1_p95.tail_delta", "rank_energy.tail_median", "rank_energy.tail_delta",
    "avg_proxy_relative_delta.tail_median", "avg_promote_applied.recorded_count",
}
DEFINITIONS["initial_metric_ids"] = sorted(INITIAL_METRICS)

DEFINITIONS["unknown_metric_version"] = "Per-source DQ values remain exploratory; cross-run common_interval requires recorded matching MetricVersion."
