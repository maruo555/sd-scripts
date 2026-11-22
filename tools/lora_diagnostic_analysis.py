"""Numerical interpretation of the six indexed training logs. HTML is produced separately."""
from collections import Counter

from tools.lora_diagnostic_dq import analyze_dq
from tools.lora_diagnostic_numbers import mean_tail, moving_average, safe_float
from tools.lora_diagnostic_rank import analyze_rank
from tools.lora_run_reader import promote_windows
from tools.lora_run_metrics import gradient_fields, promote_fields


def unique_rows(rows, keys):
    counts = Counter(tuple(r.get(k) for k in keys) for r in rows)
    valid = [r for r in rows if all(r.get(k) is not None for k in keys)
             and counts[tuple(r[k] for k in keys)] == 1]
    return sorted(valid, key=lambda r: tuple(r[k] for k in keys)), len(rows) - len(valid)


def analyze_gradient(rows, window):
    rows, ambiguous = unique_rows(rows, ("epoch", "batch_index"))
    raw, disabled, grads, thresholds, scales, markers, cuts = [], [], [], [], [], [], []
    previous = None
    good, exceeded = 0, 0
    for i, row in enumerate(rows):
        coord = (row["epoch"], row["batch_index"])
        if previous is None or coord[0] != previous[0]:
            markers.append({"x": i + 1, "label": f"E{coord[0]}"})
        if previous and (coord[0] > previous[0] + 1 or coord[0] == previous[0] and coord[1] != previous[1] + 1):
            cuts.append(i)
        previous = coord
        value = row["threshold"]
        off = gradient_fields(row)["ThreshOff"]
        disabled.append(None if off is None else int(off))
        threshold, grad = safe_float(value), safe_float(row["grad_norm"])
        if threshold is not None and threshold > 0 and grad is not None:
            good += 1
            exceeded += int(grad > threshold)
        raw.append(safe_float(row["loss"]))
        grads.append(grad)
        thresholds.append(threshold)
        scales.append(safe_float(row["grad_scale"]))
    ma = moving_average(raw, window)
    start, end = mean_tail(ma, .15, False), mean_tail(ma, .15, True)
    known = sum(x is not None for x in disabled)
    return dict(rows=rows, x=list(range(1, len(rows)+1)), markers=markers, cuts=cuts, gradient_norm=grads,
                threshold=thresholds, scale=scales, thresh_off=disabled, loss=raw, loss_ma=ma, ambiguous=ambiguous,
                summary=dict(threshold_valid_count=good, threshold_exceeded_count=exceeded,
                             threshold_exceeded_ratio=exceeded/good if good else None,
                             thresh_off_ratio=sum(x or 0 for x in disabled)/known if known else None,
                             threshold_unknown=len(rows)-known, loss_ma_start=start, loss_ma_end=end,
                             loss_ma_drop_ratio=(start-end)/start if start is not None and start > 0 and end is not None else None))


def analyze_group(rows):
    valid, omitted = unique_rows(rows, ("attempt_step",))
    markers, epochs = [], set()
    for row in valid:
        if row["epoch"] is not None and row["epoch"] not in epochs:
            epochs.add(row["epoch"])
            markers.append({"x": row["attempt_step"], "label": f"E{row['epoch']}"})
    return dict(rows=valid, omitted=omitted, markers=markers)


def analyze_promote(rows, resolved=None):
    output, issues = [], []
    for epoch, decision, window in promote_windows(rows):
        if epoch is None or decision is None:
            issues.append({"epoch": epoch, "reason": "missing or ambiguous decision"})
            continue
        if window is None:
            issues.append({"epoch": epoch, "reason": "incomplete window"})
        derived = promote_fields(decision, window, resolved)
        output.append(dict(decision, window=window, difference_percent=derived["delta_pct"]))
    output.sort(key=lambda r: r["epoch"])
    return dict(rows=output, issues=issues, scored=sum(r["status"] == "scored" for r in output),
                margin_wins=sum(r["margin_ok"] is True for r in output),
                applied=sum(r["promote_applied"] is True for r in output))


def analyze_streams(reader, streams, resolved, window=100, module_param_counts=None):
    result = dict(
        grad=analyze_gradient(streams["gradient"].rows, window),
        dq=analyze_dq(streams["dq"].rows, streams["dq_auto"].rows, resolved),
        rank=analyze_rank(reader, streams["rank"].rows, module_param_counts),
        group_loss=analyze_group(streams["group_loss"].rows),
        promote=analyze_promote(streams["promote"].rows, resolved),
    )
    observed = len(result["grad"]["rows"]) + sum(map(len, result["dq"]["scope_rows"].values()))
    observed += len(result["dq"]["events"]) + len(result["rank"]["rows"]) + len(result["rank"]["incomplete"])
    observed += len(result["group_loss"]["rows"]) + len(result["promote"]["rows"])
    if not observed:
        statuses = ", ".join(f"{k}={v.status} ({len(v.rows)} rows)" for k, v in streams.items())
        raise ValueError(f"No usable diagnostic observations in this run: {statuses}")
    return result
