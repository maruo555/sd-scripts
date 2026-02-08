import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ColorPalette = [
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#d97706",
    "#16a34a",
    "#0891b2",
    "#be185d",
]

DEFAULT_LORA_EPOCH_TREND_MAX_SERIES = 18


def module_label(module: str) -> str:
    mapping = {
        "unet": "UNet",
        "te1": "TE1 (Text Encoder 1)",
        "te2": "TE2 (Text Encoder 2)",
    }
    return mapping.get(module, str(module).upper())


def format_unet_block_label(label: str) -> str:
    if not label:
        return "-"
    if label.startswith("input_blocks_"):
        suffix = label.split("_")[-1]
        return f"Input {suffix}"
    if label.startswith("output_blocks_"):
        suffix = label.split("_")[-1]
        return f"Output {suffix}"
    if label == "middle_block":
        return "Middle Block"
    return label.replace("_", " ").title()


def import_lora_analysis_tools() -> Tuple[Any, Any, Any]:
    try:
        try:
            from tools.analyze_lora_density import (  # type: ignore
                build_checkpoint_history,
                collect_single_analysis,
                find_checkpoint_series,
            )
        except Exception:
            from analyze_lora_density import build_checkpoint_history, collect_single_analysis, find_checkpoint_series  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "analyze_lora_density.py の読み込みに失敗しました。torch/safetensors が必要です。"
        ) from exc
    return collect_single_analysis, find_checkpoint_series, build_checkpoint_history


def color_for_index(index: int, total: int) -> str:
    if total <= 0:
        return "#2563eb"
    hue = (index * 137.508) % 360.0
    saturation = 68.0
    lightness = 45.0 if (index % 2 == 0) else 55.0
    c = (1.0 - abs(2.0 * (lightness / 100.0) - 1.0)) * (saturation / 100.0)
    x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    m = (lightness / 100.0) - c / 2.0
    if hue < 60:
        rp, gp, bp = c, x, 0
    elif hue < 120:
        rp, gp, bp = x, c, 0
    elif hue < 180:
        rp, gp, bp = 0, c, x
    elif hue < 240:
        rp, gp, bp = 0, x, c
    elif hue < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_te_layer_index(label: str, display_label: str) -> Optional[int]:
    patterns = [label, display_label]
    for source in patterns:
        text = (source or "").strip().lower()
        match = re.search(r"layer[_ ](\d+)", text)
        if match:
            return int(match.group(1))
    return None


def lora_trend_sort_key(module: str, entry: Dict[str, Any]) -> Tuple[int, int, str]:
    label = str(entry.get("label") or "")
    display = str(entry.get("display_label") or "")
    if module == "unet":
        if label.startswith("input_blocks_"):
            suffix = label.split("_")[-1]
            idx = int(suffix) if suffix.isdigit() else 999
            return (0, idx, display)
        if label == "middle_block":
            return (1, 0, display)
        if label.startswith("output_blocks_"):
            suffix = label.split("_")[-1]
            idx = int(suffix) if suffix.isdigit() else 999
            return (2, idx, display)
        return (3, 999, display)

    layer_idx = parse_te_layer_index(label, display)
    if layer_idx is not None:
        return (0, layer_idx, display)
    return (1, 999, display)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return list(values)
    result: List[Optional[float]] = []
    queue: List[float] = []
    running_sum = 0.0
    for value in values:
        if value is None:
            result.append(None)
            continue
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def mean_tail(values: List[Optional[float]], ratio: float, take_tail: bool) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    count = max(1, int(len(valid) * ratio))
    target = valid[-count:] if take_tail else valid[:count]
    return float(sum(target) / len(target))


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def infer_steps_per_epoch(rows: List[Dict[str, str]]) -> int:
    if not rows:
        return 1
    step_max_by_epoch: Dict[int, int] = {}
    for row in rows:
        epoch = safe_int(row.get("Epoch"))
        step = safe_int(row.get("Step"))
        if epoch is None or step is None:
            continue
        current = step_max_by_epoch.get(epoch, 0)
        if step > current:
            step_max_by_epoch[epoch] = step
    if not step_max_by_epoch:
        return 1
    candidates = [value + 1 for value in step_max_by_epoch.values() if value >= 0]
    if not candidates:
        return 1
    return int(statistics.median(candidates))


def parse_grad_log(path: str, ma_window: int) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    steps_per_epoch = infer_steps_per_epoch(rows)

    x_values: List[int] = []
    epochs: List[Optional[int]] = []
    gradient_norm: List[Optional[float]] = []
    threshold: List[Optional[float]] = []
    loss: List[Optional[float]] = []
    thresh_off: List[Optional[float]] = []
    scale: List[Optional[float]] = []
    cosine: List[Optional[float]] = []
    first_step_by_epoch: Dict[int, int] = {}

    threshold_valid = 0
    threshold_exceeded = 0
    thresh_off_count = 0
    cosine_valid = 0

    for row in rows:
        epoch_value = safe_int(row.get("Epoch"))
        step_value = safe_int(row.get("Step"))
        if epoch_value is None or step_value is None:
            continue
        global_step = epoch_value * steps_per_epoch + step_value
        x_values.append(global_step)
        epochs.append(epoch_value + 1)
        if epoch_value not in first_step_by_epoch:
            first_step_by_epoch[epoch_value] = global_step

        grad = safe_float(row.get("Gradient Norm"))
        th = safe_float(row.get("Threshold"))
        lo = safe_float(row.get("Loss"))
        toff = safe_float(row.get("ThreshOff"))
        sc = safe_float(row.get("Scale"))
        cs = safe_float(row.get("CosineSim"))

        gradient_norm.append(grad)
        threshold.append(th)
        loss.append(lo)
        thresh_off.append(toff)
        scale.append(sc)
        cosine.append(cs)

        if toff is not None and toff > 0.0:
            thresh_off_count += 1
        if cs is not None:
            cosine_valid += 1
        if grad is not None and th is not None and th > 0:
            threshold_valid += 1
            if grad > th:
                threshold_exceeded += 1

    loss_ma = moving_average(loss, ma_window)
    loss_start = mean_tail(loss_ma, 0.15, take_tail=False)
    loss_end = mean_tail(loss_ma, 0.15, take_tail=True)
    loss_drop = None
    if loss_start is not None and loss_end is not None and loss_start > 0:
        loss_drop = (loss_start - loss_end) / loss_start

    markers = [
        {"x": step_value, "label": f"E{epoch + 1}"}
        for epoch, step_value in sorted(first_step_by_epoch.items(), key=lambda item: item[0])
    ]
    return {
        "path": path,
        "rows": len(x_values),
        "steps_per_epoch": steps_per_epoch,
        "x": x_values,
        "epochs": epochs,
        "markers": markers,
        "gradient_norm": gradient_norm,
        "threshold": threshold,
        "loss": loss,
        "loss_ma": loss_ma,
        "thresh_off": thresh_off,
        "scale": scale,
        "cosine": cosine,
        "summary": {
            "threshold_valid_count": threshold_valid,
            "threshold_exceeded_count": threshold_exceeded,
            "threshold_exceeded_ratio": safe_ratio(float(threshold_exceeded), float(threshold_valid))
            if threshold_valid
            else None,
            "thresh_off_ratio": safe_ratio(float(thresh_off_count), float(len(x_values))) if x_values else None,
            "loss_ma_start": loss_start,
            "loss_ma_end": loss_end,
            "loss_ma_drop_ratio": loss_drop,
            "cosine_valid_ratio": safe_ratio(float(cosine_valid), float(len(x_values))) if x_values else None,
            "max_grad_norm": max((v for v in gradient_norm if v is not None), default=None),
        },
    }


def infer_epoch_from_step(train_step: int, grad_data: Optional[Dict[str, Any]]) -> Optional[int]:
    if not grad_data:
        return None
    steps_per_epoch = grad_data.get("steps_per_epoch")
    if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        return None
    return (train_step // steps_per_epoch) + 1


def parse_dq_logs(
    dq_log_path: str,
    dq_auto_path: Optional[str],
    grad_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = read_csv_rows(dq_log_path)
    parsed_rows: List[Dict[str, Any]] = []
    for row in rows:
        train_step = safe_int(row.get("TrainStep"))
        if train_step is None:
            continue
        epoch = safe_int(row.get("Epoch"))
        if epoch is None:
            epoch = infer_epoch_from_step(train_step, grad_data)

        parsed_rows.append(
            {
                "TrainStep": train_step,
                "Epoch": epoch,
                "Bits": safe_float(row.get("Bits")),
                "RangeMul": safe_float(row.get("RangeMul")),
                "ClipRateRaw": safe_float(row.get("ClipRateRaw")),
                "ClipRateEMA": safe_float(row.get("ClipRateEMA")),
                "QuantErrRatioRaw": safe_float(row.get("QuantErrRatioRaw")),
                "QuantErrRatioEMA": safe_float(row.get("QuantErrRatioEMA")),
                "QuantErrRMSRaw": safe_float(row.get("QuantErrRMSRaw")),
                "QuantErrRMSEMA": safe_float(row.get("QuantErrRMSEMA")),
                "ZeroRate": safe_float(row.get("ZeroRate")),
                "AbsMax": safe_float(row.get("AbsMax")),
                "Range": safe_float(row.get("Range")),
                "RankDim": safe_float(row.get("RankDim")),
                "RankSatWMean": safe_float(row.get("RankSatWMean")),
                "RankSatP50": safe_float(row.get("RankSatP50")),
                "RankSatP95": safe_float(row.get("RankSatP95")),
                "RankSatMax": safe_float(row.get("RankSatMax")),
                "RankTop1P95": safe_float(row.get("RankTop1P95")),
                "RankEnergySum": safe_float(row.get("RankEnergySum")),
                "AutoReason": (row.get("AutoReason") or "").strip(),
            }
        )
    parsed_rows.sort(key=lambda item: item["TrainStep"])

    auto_rows: List[Dict[str, Any]] = []
    if dq_auto_path and os.path.exists(dq_auto_path):
        raw_auto_rows = read_csv_rows(dq_auto_path)
        for row in raw_auto_rows:
            step = safe_int(row.get("TrainStep"))
            if step is None:
                continue
            auto_rows.append(
                {
                    "TrainStep": step,
                    "Bits": safe_float(row.get("Bits")),
                    "ClipRateRaw": safe_float(row.get("ClipRateRaw")),
                    "ClipRateEMA": safe_float(row.get("ClipRateEMA")),
                    "RangeMulBefore": safe_float(row.get("RangeMulBefore")),
                    "RangeMulAfter": safe_float(row.get("RangeMulAfter")),
                    "AutoApplied": safe_int(row.get("AutoApplied")),
                    "WarmupActive": safe_int(row.get("WarmupActive")),
                    "AutoReason": (row.get("AutoReason") or "").strip(),
                    "AutoInitClipTarget": safe_float(row.get("AutoInitClipTarget")),
                }
            )
        auto_rows.sort(key=lambda item: item["TrainStep"])

    markers: List[Dict[str, Any]] = []
    last_epoch: Optional[int] = None
    for item in parsed_rows:
        epoch = item.get("Epoch")
        if epoch is None:
            continue
        if last_epoch != epoch:
            markers.append({"x": item["TrainStep"], "label": f"E{epoch}"})
            last_epoch = epoch

    bits_values = [item["Bits"] for item in parsed_rows if item["Bits"] is not None]
    bit_switches = 0
    if bits_values:
        prev = bits_values[0]
        for current in bits_values[1:]:
            if current != prev:
                bit_switches += 1
            prev = current

    auto_reason_counter = Counter()
    auto_applied_count = 0
    non_warmup_count = 0
    in_band_count = 0
    clip_target_values: List[float] = []
    for row in auto_rows:
        reason = row.get("AutoReason") or ""
        if reason:
            auto_reason_counter[reason] += 1
        if row.get("AutoApplied") == 1:
            auto_applied_count += 1
        warmup = row.get("WarmupActive")
        if warmup == 0:
            non_warmup_count += 1
            if reason == "in_band":
                in_band_count += 1
        target = row.get("AutoInitClipTarget")
        if target is not None:
            clip_target_values.append(target)

    latest = parsed_rows[-1] if parsed_rows else {}
    clip_ema_values = [item["ClipRateEMA"] for item in parsed_rows if item["ClipRateEMA"] is not None]
    clip_ema_cv = None
    if len(clip_ema_values) >= 8:
        tail = clip_ema_values[-max(8, len(clip_ema_values) // 4) :]
        tail_mean = sum(tail) / len(tail)
        if tail_mean > 0:
            variance = sum((x - tail_mean) ** 2 for x in tail) / len(tail)
            clip_ema_cv = math.sqrt(variance) / tail_mean

    summary = {
        "rows": len(parsed_rows),
        "auto_rows": len(auto_rows),
        "bit_switches": bit_switches,
        "bits_unique": sorted(list({int(v) for v in bits_values})) if bits_values else [],
        "auto_applied_count": auto_applied_count,
        "in_band_ratio": safe_ratio(float(in_band_count), float(non_warmup_count)) if non_warmup_count else None,
        "auto_reason_counts": dict(auto_reason_counter),
        "auto_clip_target_median": statistics.median(clip_target_values) if clip_target_values else None,
        "clip_ema_cv": clip_ema_cv,
        "final_clip_rate_ema": latest.get("ClipRateEMA"),
        "final_quant_err_ratio_ema": latest.get("QuantErrRatioEMA"),
        "final_quant_err_rms_ema": latest.get("QuantErrRMSEMA"),
        "final_zero_rate": latest.get("ZeroRate"),
        "final_rank_sat_p95": latest.get("RankSatP95"),
        "final_rank_energy_sum": latest.get("RankEnergySum"),
    }

    return {
        "path": dq_log_path,
        "auto_path": dq_auto_path,
        "rows": parsed_rows,
        "auto_rows": auto_rows,
        "markers": markers,
        "summary": summary,
    }


def parse_group_loss_logs(
    group_step_log_path: Optional[str],
    group_epoch_log_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    step_rows_raw = read_csv_rows(group_step_log_path) if group_step_log_path and os.path.exists(group_step_log_path) else []
    epoch_rows_raw = read_csv_rows(group_epoch_log_path) if group_epoch_log_path and os.path.exists(group_epoch_log_path) else []

    step_rows: List[Dict[str, Any]] = []
    first_step_by_epoch: Dict[int, int] = {}
    group_order: List[str] = []
    group_seen = set()
    group_points: Dict[str, List[Tuple[int, Optional[float]]]] = defaultdict(list)

    for row in step_rows_raw:
        global_step = safe_int(row.get("global_step"))
        if global_step is None:
            continue
        epoch = safe_int(row.get("epoch"))
        group = (row.get("group") or "").strip() or "__ungrouped__"
        ema_loss_group = safe_float(row.get("ema_loss_group"))
        loss = safe_float(row.get("loss"))
        step_rows.append(
            {
                "global_step": global_step,
                "epoch": epoch,
                "group": group,
                "ema_loss_group": ema_loss_group,
                "loss": loss,
            }
        )

        if epoch is not None and epoch not in first_step_by_epoch:
            first_step_by_epoch[epoch] = global_step
        if group not in group_seen:
            group_seen.add(group)
            group_order.append(group)
        group_points[group].append((global_step, ema_loss_group))

    step_rows.sort(key=lambda item: item["global_step"])

    x_values = [row["global_step"] for row in step_rows]
    group_series: List[Dict[str, Any]] = []
    if x_values and group_order:
        color_map = {group: color_for_index(idx, len(group_order)) for idx, group in enumerate(group_order)}
        for group in group_order:
            points = sorted(group_points.get(group, []), key=lambda item: item[0])
            group_series.append(
                {
                    "name": group,
                    "color": color_map[group],
                    "x": [pt[0] for pt in points],
                    "y": [pt[1] for pt in points],
                }
            )

    markers = [{"x": step_value, "label": f"E{epoch}"} for epoch, step_value in sorted(first_step_by_epoch.items())]

    epoch_rows: List[Dict[str, Any]] = []
    for row in epoch_rows_raw:
        epoch = safe_int(row.get("epoch"))
        group = (row.get("group") or "").strip() or "__ungrouped__"
        ema_loss_end = safe_float(row.get("ema_loss_end"))
        count_epoch = safe_int(row.get("count_epoch"))
        mean_loss_epoch = safe_float(row.get("mean_loss_epoch"))
        if epoch is None:
            continue
        epoch_rows.append(
            {
                "epoch": epoch,
                "group": group,
                "ema_loss_end": ema_loss_end,
                "count_epoch": count_epoch,
                "mean_loss_epoch": mean_loss_epoch,
            }
        )

    if not step_rows and not epoch_rows:
        return None

    return {
        "step_path": group_step_log_path,
        "epoch_path": group_epoch_log_path,
        "step_rows": step_rows,
        "epoch_rows": epoch_rows,
        "x": x_values,
        "markers": markers,
        "series": group_series,
        "summary": {
            "step_rows": len(step_rows),
            "epoch_rows": len(epoch_rows),
            "group_count": len(group_order),
            "groups": group_order,
        },
    }


def analyze_lora_checkpoint(model_path: str, bins: int) -> Dict[str, Any]:
    collect_single_analysis, _, _ = import_lora_analysis_tools()
    report = collect_single_analysis(model_path, bins)
    summary = report.get("summary", {}) or {}
    module_summary = report.get("module_summary", []) or []
    unet_block_summary = report.get("unet_block_summary", []) or []
    density_stats = summary.get("density", {}) or {}
    rms_stats = summary.get("rms", {}) or {}
    entropy_stats = summary.get("entropy_norm", {}) or {}
    sparsity_stats = summary.get("sparsity", {}) or {}

    module_density_medians = [
        item.get("density", {}).get("median")
        for item in module_summary
        if isinstance(item.get("density", {}).get("median"), (float, int))
    ]
    module_balance_ratio = None
    if module_density_medians:
        minimum = min(module_density_medians)
        maximum = max(module_density_medians)
        if minimum > 0:
            module_balance_ratio = maximum / minimum

    return {
        "path": model_path,
        "summary": summary,
        "summary_cards": {
            "total_blocks": summary.get("total_blocks"),
            "total_params": summary.get("total_params"),
            "density_median": density_stats.get("median"),
            "rms_median": rms_stats.get("median"),
            "entropy_median": entropy_stats.get("median"),
            "sparsity_median": sparsity_stats.get("median"),
        },
        "module_summary": module_summary,
        "unet_block_summary": unet_block_summary,
        "diagnostic": {
            "module_balance_ratio": module_balance_ratio,
            "unet_block_count": len(unet_block_summary),
            "density_min": density_stats.get("min"),
            "density_max": density_stats.get("max"),
        },
    }


def analyze_lora_epoch_trend(model_path: str, bins: int) -> Dict[str, Any]:
    collect_single_analysis, find_checkpoint_series, build_checkpoint_history = import_lora_analysis_tools()

    checkpoint_series = find_checkpoint_series(model_path)
    series_data: List[Tuple[int, str, bool, Dict[str, Any]]] = []
    total = len(checkpoint_series)
    print(f"[lora_epoch_trend] checkpoint series found: {total}", flush=True)
    for idx, (epoch, path, is_final) in enumerate(checkpoint_series, start=1):
        marker = " [final]" if is_final else ""
        print(
            f"[lora_epoch_trend] analyzing {idx}/{total}: epoch={epoch}{marker} file={os.path.basename(path)}",
            flush=True,
        )
        report = collect_single_analysis(path, bins)
        series_data.append((epoch, path, is_final, report))
    print("[lora_epoch_trend] checkpoint analysis done", flush=True)

    history = build_checkpoint_history(series_data)
    module_series = (history.get("series") or {}) if isinstance(history, dict) else {}

    trend_modules: List[Dict[str, Any]] = []
    for module in ("unet", "te1", "te2"):
        entries = module_series.get(module) or []
        entry_map = {str(item.get("label") or ""): item for item in entries}

        if module == "te2":
            selected: List[Dict[str, Any]] = []
            for layer_idx in range(32):
                key = f"layer_{layer_idx:02d}"
                if key in entry_map:
                    selected.append(entry_map[key])
                else:
                    selected.append(
                        {
                            "label": key,
                            "display_label": f"Layer {layer_idx:02d}",
                            "values": [],
                            "start_density": None,
                            "end_density": None,
                            "delta_density": None,
                        }
                    )
        else:
            sorted_entries = sorted(entries, key=lambda item: lora_trend_sort_key(module, item))
            selected = sorted_entries[: max(1, DEFAULT_LORA_EPOCH_TREND_MAX_SERIES)] if sorted_entries else []

        colors = [color_for_index(i, len(selected)) for i in range(len(selected))]
        series_items: List[Dict[str, Any]] = []
        legend_rows: List[Dict[str, Any]] = []
        for idx, entry in enumerate(selected):
            values = entry.get("values") or []
            x = [point.get("epoch") for point in values]
            y = [point.get("density_mean") for point in values]
            label = entry.get("display_label") or entry.get("label") or f"{module}_{idx + 1}"
            color = colors[idx]
            series_items.append({"name": label, "color": color, "x": x, "y": y})
            legend_rows.append(
                {
                    "color": color,
                    "name": label,
                    "start_density": entry.get("start_density"),
                    "end_density": entry.get("end_density"),
                    "delta_density": entry.get("delta_density"),
                }
            )

        trend_modules.append(
            {
                "module": module,
                "module_label": module_label(module),
                "total_series": len(entries),
                "selected_series": len(selected),
                "series_limit": DEFAULT_LORA_EPOCH_TREND_MAX_SERIES,
                "legend_rows": legend_rows,
                "series": series_items,
            }
        )
        print(
            f"[lora_epoch_trend] prepared module={module} series={len(series_items)} "
            f"(available={len(entries)})",
            flush=True,
        )

    files = history.get("files") if isinstance(history, dict) else []
    print("[lora_epoch_trend] trend payload ready", flush=True)
    return {
        "model_path": model_path,
        "checkpoints": files,
        "modules": trend_modules,
    }


def grade_metric(value: Optional[float], good_max: float, warn_max: float) -> Tuple[str, str]:
    if value is None:
        return "info", "データ不足"
    if value <= good_max:
        return "good", "良好"
    if value <= warn_max:
        return "warn", "注意"
    return "bad", "要改善"


def build_diagnostics(
    grad_data: Optional[Dict[str, Any]],
    dq_data: Optional[Dict[str, Any]],
    lora_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []

    if grad_data:
        grad_summary = grad_data.get("summary", {})
        spike_ratio = grad_summary.get("threshold_exceeded_ratio")
        level, text = grade_metric(spike_ratio, good_max=0.01, warn_max=0.03)
        checks.append(
            {
                "section": "GradNorm",
                "name": "しきい値超過率",
                "value": fmt_percent(spike_ratio),
                "status": level,
                "note": text + " (超過が多いと更新スキップ増加の可能性)",
            }
        )

        thresh_off_ratio = grad_summary.get("thresh_off_ratio")
        level, text = grade_metric(thresh_off_ratio, good_max=0.01, warn_max=0.05)
        checks.append(
            {
                "section": "GradNorm",
                "name": "ThreshOff発生率",
                "value": fmt_percent(thresh_off_ratio),
                "status": level,
                "note": text + " (高い場合はしきい値判定が無効化されがち)",
            }
        )

        loss_drop = grad_summary.get("loss_ma_drop_ratio")
        if loss_drop is None:
            status = "info"
            note = "データ不足"
        elif loss_drop >= 0.10:
            status = "good"
            note = "良好 (Loss移動平均が十分低下)"
        elif loss_drop >= 0.0:
            status = "warn"
            note = "注意 (Loss低下が弱い)"
        else:
            status = "bad"
            note = "要改善 (Loss移動平均が上昇)"
        checks.append(
            {
                "section": "GradNorm",
                "name": "Loss移動平均の低下率",
                "value": fmt_percent(loss_drop),
                "status": status,
                "note": note,
            }
        )

    if dq_data:
        dq_summary = dq_data.get("summary", {})
        in_band_ratio = dq_summary.get("in_band_ratio")
        if in_band_ratio is None:
            status = "info"
            note = "autoログ不足"
        elif in_band_ratio >= 0.70:
            status = "good"
            note = "良好 (auto判定が帯域内で安定)"
        elif in_band_ratio >= 0.40:
            status = "warn"
            note = "注意 (帯域外判定がやや多い)"
        else:
            status = "bad"
            note = "要改善 (帯域外判定が多い)"
        checks.append(
            {
                "section": "DQ",
                "name": "Auto in-band比率",
                "value": fmt_percent(in_band_ratio),
                "status": status,
                "note": note,
            }
        )

        quant_err_ratio = dq_summary.get("final_quant_err_ratio_ema")
        level, text = grade_metric(quant_err_ratio, good_max=0.35, warn_max=0.50)
        checks.append(
            {
                "section": "DQ",
                "name": "最終 QuantErrRatioEMA",
                "value": fmt_float(quant_err_ratio, 4),
                "status": level,
                "note": text + " (高いほど量子化誤差が強い)",
            }
        )

        zero_rate = dq_summary.get("final_zero_rate")
        level, text = grade_metric(zero_rate, good_max=0.05, warn_max=0.10)
        checks.append(
            {
                "section": "DQ",
                "name": "最終 ZeroRate",
                "value": fmt_percent(zero_rate),
                "status": level,
                "note": text + " (高いほど情報が潰れやすい)",
            }
        )

        rank_sat_p95 = dq_summary.get("final_rank_sat_p95")
        level, text = grade_metric(rank_sat_p95, good_max=0.90, warn_max=0.97)
        checks.append(
            {
                "section": "DQ",
                "name": "最終 RankSatP95",
                "value": fmt_float(rank_sat_p95, 4),
                "status": level,
                "note": text + " (高止まりはrank飽和の兆候)",
            }
        )

        clip_cv = dq_summary.get("clip_ema_cv")
        if clip_cv is None:
            status = "info"
            note = "データ不足"
        elif clip_cv <= 0.25:
            status = "good"
            note = "良好 (ClipRateEMA変動が小さい)"
        elif clip_cv <= 0.50:
            status = "warn"
            note = "注意 (ClipRateEMAに揺れがある)"
        else:
            status = "bad"
            note = "要改善 (ClipRateEMAの揺れが大きい)"
        checks.append(
            {
                "section": "DQ",
                "name": "ClipRateEMAの変動係数",
                "value": fmt_float(clip_cv, 4),
                "status": status,
                "note": note,
            }
        )

    if lora_data:
        module_balance = lora_data.get("diagnostic", {}).get("module_balance_ratio")
        if module_balance is None:
            status = "info"
            note = "データ不足"
        elif module_balance <= 2.0:
            status = "good"
            note = "良好 (モジュール間密度の偏りが小さい)"
        elif module_balance <= 3.0:
            status = "warn"
            note = "注意 (モジュール間密度の偏りがやや大きい)"
        else:
            status = "bad"
            note = "要改善 (モジュール間密度の偏りが大きい)"
        checks.append(
            {
                "section": "LoRA",
                "name": "モジュール密度バランス比 (max/min)",
                "value": fmt_float(module_balance, 4),
                "status": status,
                "note": note,
            }
        )

    score = 100
    for check in checks:
        if check["status"] == "warn":
            score -= 10
        elif check["status"] == "bad":
            score -= 20
    score = max(0, score)
    if score >= 80:
        overall_status = "良好"
        overall_class = "good"
    elif score >= 60:
        overall_status = "注意"
        overall_class = "warn"
    else:
        overall_status = "要改善"
        overall_class = "bad"

    return {
        "score": score,
        "overall_status": overall_status,
        "overall_class": overall_class,
        "checks": checks,
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def build_chart_payload(
    grad_data: Optional[Dict[str, Any]],
    dq_data: Optional[Dict[str, Any]],
    group_loss_data: Optional[Dict[str, Any]],
    lora_trend_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"grad": [], "dq": [], "group_loss": [], "lora_trend": []}

    if grad_data:
        x = grad_data.get("x", [])
        markers = grad_data.get("markers", [])
        payload["grad"] = [
            {
                "id": "grad_norm",
                "title": "Gradient Norm / Threshold",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [
                    {"name": "Gradient Norm", "color": ColorPalette[0], "y": grad_data.get("gradient_norm", [])},
                    {"name": "Threshold", "color": ColorPalette[2], "y": grad_data.get("threshold", [])},
                ],
            },
            {
                "id": "loss_ma",
                "title": "Loss (raw / moving average)",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": [
                    {"name": "Loss", "color": "#94a3b8", "y": grad_data.get("loss", [])},
                    {"name": "Loss MA", "color": ColorPalette[1], "y": grad_data.get("loss_ma", [])},
                ],
            },
            {
                "id": "thresh_off",
                "title": "ThreshOff",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_fixed": 0.0,
                "y_max_fixed": 2.0,
                "series": [{"name": "ThreshOff", "color": ColorPalette[3], "y": grad_data.get("thresh_off", [])}],
            },
            {
                "id": "scale",
                "title": "GradScaler Scale",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [{"name": "Scale", "color": ColorPalette[4], "y": grad_data.get("scale", [])}],
            },
            {
                "id": "cosine",
                "title": "Grad Cosine Similarity",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": [{"name": "CosineSim", "color": ColorPalette[5], "y": grad_data.get("cosine", [])}],
            },
        ]

    if dq_data:
        rows = dq_data.get("rows", [])
        x = [item.get("TrainStep") for item in rows]
        markers = dq_data.get("markers", [])

        def series_for(keys: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
            output = []
            for field, label, color in keys:
                output.append({"name": label, "color": color, "y": [item.get(field) for item in rows]})
            return output

        payload["dq"] = [
            {
                "id": "dq_bits",
                "title": "Bits",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_floor": 0.0,
                "series": series_for([("Bits", "Bits", ColorPalette[0])]),
            },
            {
                "id": "dq_range_mul",
                "title": "RangeMul",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": series_for([("RangeMul", "RangeMul", ColorPalette[1])]),
            },
            {
                "id": "dq_clip",
                "title": "ClipRateRaw / ClipRateEMA",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 0.008,
                "y_tick_step": 0.001,
                "y_tick_precision": 3,
                "series": series_for(
                    [("ClipRateRaw", "ClipRateRaw", ColorPalette[2]), ("ClipRateEMA", "ClipRateEMA", ColorPalette[0])]
                ),
            },
            {
                "id": "dq_qerr_ratio",
                "title": "QuantErrRatioRaw / QuantErrRatioEMA",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 0.8,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": series_for(
                    [
                        ("QuantErrRatioRaw", "QuantErrRatioRaw", ColorPalette[2]),
                        ("QuantErrRatioEMA", "QuantErrRatioEMA", ColorPalette[0]),
                    ]
                ),
            },
            {
                "id": "dq_qerr_rms",
                "title": "QuantErrRMSRaw / QuantErrRMSEMA",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_tick_step": 0.05,
                "y_tick_precision": 2,
                "series": series_for(
                    [
                        ("QuantErrRMSRaw", "QuantErrRMSRaw", ColorPalette[2]),
                        ("QuantErrRMSEMA", "QuantErrRMSEMA", ColorPalette[0]),
                    ]
                ),
            },
            {
                "id": "dq_zero_rate",
                "title": "ZeroRate",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_tick_step": 0.02,
                "y_tick_precision": 2,
                "series": series_for([("ZeroRate", "ZeroRate", ColorPalette[3])]),
            },
            {
                "id": "dq_absmax",
                "title": "AbsMax",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 1000.0,
                "y_tick_nice_integer": True,
                "y_min_floor": 0.0,
                "series": series_for([("AbsMax", "AbsMax", ColorPalette[4])]),
            },
            {
                "id": "dq_range",
                "title": "Range",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": series_for([("Range", "Range", ColorPalette[5])]),
            },
            {
                "id": "dq_rank_dim",
                "title": "RankDim",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_floor": 0.0,
                "series": series_for([("RankDim", "RankDim", ColorPalette[6])]),
            },
            {
                "id": "dq_rank_sat",
                "title": "RankSatWMean / P50 / P95 / Max / Top1P95",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 1.1,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "legend_max_rows": 2,
                "series": series_for(
                    [
                        ("RankSatWMean", "RankSatWMean", ColorPalette[0]),
                        ("RankSatP50", "RankSatP50", ColorPalette[1]),
                        ("RankSatP95", "RankSatP95", ColorPalette[2]),
                        ("RankSatMax", "RankSatMax", ColorPalette[3]),
                        ("RankTop1P95", "RankTop1P95", ColorPalette[4]),
                    ]
                ),
            },
            {
                "id": "dq_rank_energy",
                "title": "RankEnergySum",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": series_for([("RankEnergySum", "RankEnergySum", ColorPalette[7])]),
            },
        ]

    if group_loss_data:
        step_x = group_loss_data.get("x", []) or []
        step_markers = group_loss_data.get("markers", []) or []
        step_series = group_loss_data.get("series", []) or []
        if step_x and step_series:
            payload["group_loss"].append(
                {
                    "id": "group_loss_ema_step",
                    "title": "Group Loss EMA (step)",
                    "x_label": "GlobalStep",
                    "markers": step_markers,
                    "x": step_x,
                    "y_min_fixed": 0.0,
                    "legend_max_rows": 4,
                    "series": step_series,
                }
            )

        epoch_rows = group_loss_data.get("epoch_rows", []) or []
        if epoch_rows:
            group_to_points: Dict[str, List[Tuple[int, Optional[float]]]] = defaultdict(list)
            for row in epoch_rows:
                epoch = row.get("epoch")
                group = row.get("group")
                value = row.get("ema_loss_end")
                if isinstance(epoch, int) and isinstance(group, str):
                    group_to_points[group].append((epoch, value))

            group_names = list(group_to_points.keys())
            series = []
            for idx, group in enumerate(group_names):
                points = sorted(group_to_points[group], key=lambda item: item[0])
                series.append(
                    {
                        "name": group,
                        "color": color_for_index(idx, len(group_names)),
                        "x": [pt[0] for pt in points],
                        "y": [pt[1] for pt in points],
                    }
                )

            if series:
                epoch_markers = []
                unique_epochs = sorted(
                    {epoch for point_list in group_to_points.values() for epoch, _ in point_list}
                )
                if unique_epochs:
                    epoch_markers = [{"x": epoch, "label": f"E{epoch}"} for epoch in unique_epochs]
                payload["group_loss"].append(
                    {
                        "id": "group_loss_ema_epoch",
                        "title": "Group Loss EMA (epoch summary)",
                        "x_label": "Epoch",
                        "markers": epoch_markers,
                        "x": unique_epochs,
                        "y_min_fixed": 0.0,
                        "legend_max_rows": 4,
                        "series": series,
                    }
                )

    if lora_trend_data:
        checkpoints = lora_trend_data.get("checkpoints", []) or []
        epoch_markers = []
        for item in checkpoints:
            epoch = item.get("epoch")
            if epoch is None:
                continue
            label = item.get("label") or f"E{epoch}"
            epoch_markers.append({"x": epoch, "label": label})

        module_entries = lora_trend_data.get("modules", []) or []
        trend_charts: List[Dict[str, Any]] = []
        for entry in module_entries:
            module = entry.get("module") or "-"
            series_rows = entry.get("series", []) or []
            chart_series = []
            for row in series_rows:
                chart_series.append(
                    {
                        "name": row.get("name"),
                        "color": row.get("color"),
                        "x": row.get("x", []),
                        "y": row.get("y", []),
                    }
                )
            selected_count = entry.get("selected_series", 0)
            total_count = entry.get("total_series", 0)
            trend_charts.append(
                {
                    "id": f"lora_trend_{module}",
                    "title": f"{entry.get('module_label', module)} 情報密度推移 ({selected_count}/{total_count}系列)",
                    "x_label": "Epoch",
                    "markers": epoch_markers,
                    "x": [item.get("epoch") for item in checkpoints if item.get("epoch") is not None],
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": chart_series,
                    "legend_rows": entry.get("legend_rows", []),
                    "no_inline_legend": True,
                }
            )
        payload["lora_trend"] = trend_charts

    return payload


def render_module_rows(module_items: List[Dict[str, Any]]) -> str:
    if not module_items:
        return "<tr><td colspan='7' class='muted'>データがありません</td></tr>"
    rows: List[str] = []
    for item in module_items:
        density = item.get("density", {}) or {}
        rms = item.get("rms", {}) or {}
        row = (
            f"<tr>"
            f"<td>{module_label(item.get('module', '-'))}</td>"
            f"<td class='num'>{fmt_int(item.get('block_count'))}</td>"
            f"<td class='num'>{fmt_int(item.get('total_params'))}</td>"
            f"<td class='num'>{fmt_float(density.get('mean'))}</td>"
            f"<td class='num'>{fmt_float(density.get('median'))}</td>"
            f"<td class='num'>{fmt_float(rms.get('median'))}</td>"
            f"<td class='num'>{fmt_float((item.get('entropy_norm', {}) or {}).get('median'))}</td>"
            f"</tr>"
        )
        rows.append(row)
    return "".join(rows)


def render_unet_rows(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<tr><td colspan='6' class='muted'>データがありません</td></tr>"
    rows: List[str] = []
    for item in items:
        density = item.get("density", {}) or {}
        rms = item.get("rms", {}) or {}
        row = (
            f"<tr>"
            f"<td>{format_unet_block_label(item.get('label', '-'))}</td>"
            f"<td class='num'>{fmt_int(item.get('block_count'))}</td>"
            f"<td class='num'>{fmt_int(item.get('total_params'))}</td>"
            f"<td class='num'>{fmt_float(density.get('mean'))}</td>"
            f"<td class='num'>{fmt_float(density.get('median'))}</td>"
            f"<td class='num'>{fmt_float(rms.get('median'))}</td>"
            f"</tr>"
        )
        rows.append(row)
    return "".join(rows)


def status_label(status: str) -> str:
    mapping = {"good": "良好", "warn": "注意", "bad": "要改善", "info": "情報"}
    return mapping.get(status, status)


def render_check_rows(checks: List[Dict[str, str]]) -> str:
    if not checks:
        return "<tr><td colspan='5' class='muted'>診断チェックがありません</td></tr>"
    rows: List[str] = []
    for item in checks:
        status = item.get("status", "info")
        rows.append(
            "<tr>"
            f"<td>{item.get('section', '-')}</td>"
            f"<td>{item.get('name', '-')}</td>"
            f"<td class='num'>{item.get('value', '-')}</td>"
            f"<td><span class='badge {status}'>{status_label(status)}</span></td>"
            f"<td>{item.get('note', '-')}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_html(report: Dict[str, Any]) -> str:
    report_json = json.dumps(sanitize_json(report), ensure_ascii=False).replace("</", "<\\/")
    diagnostics = report.get("diagnostics", {})
    lora_data = report.get("lora")
    lora_cards = {}
    module_rows = "<tr><td colspan='7' class='muted'>LoRA解析を実行していません</td></tr>"
    unet_rows = "<tr><td colspan='6' class='muted'>LoRA解析を実行していません</td></tr>"
    if lora_data:
        lora_cards = lora_data.get("summary_cards", {})
        module_rows = render_module_rows(lora_data.get("module_summary", []))
        unet_rows = render_unet_rows(lora_data.get("unet_block_summary", []))
    lora_error = report.get("lora_error")
    lora_error_html = f"<p class='sub' style='color: var(--bad);'>{lora_error}</p>" if lora_error else ""
    lora_trend_error = report.get("lora_trend_error")
    lora_trend_error_html = f"<p class='sub' style='color: var(--bad);'>{lora_trend_error}</p>" if lora_trend_error else ""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LoRA Diagnostic Report</title>
<style>
:root {{
  --bg: #f4f6fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --line: #d7deea;
  --good: #15803d;
  --good-bg: #dcfce7;
  --warn: #b45309;
  --warn-bg: #ffedd5;
  --bad: #b91c1c;
  --bad-bg: #fee2e2;
  --info: #0f766e;
  --info-bg: #ccfbf1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: linear-gradient(160deg, #eef2ff 0%, #f8fafc 45%, #f5f5f4 100%);
  color: var(--text);
  font-family: "Segoe UI", "Noto Sans JP", "Hiragino Kaku Gothic ProN", sans-serif;
}}
.container {{
  max-width: 1520px;
  margin: 0 auto;
  padding: 22px;
}}
h1, h2 {{
  margin: 0;
}}
.sub {{
  color: var(--muted);
  margin-top: 8px;
}}
.panel {{
  margin-top: 16px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
}}
.grid {{
  display: grid;
  gap: 12px;
}}
.grid.cards {{
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
}}
.card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px;
  background: #fbfdff;
}}
.card .title {{
  color: var(--muted);
  font-size: 12px;
}}
.card .value {{
  margin-top: 4px;
  font-size: 24px;
  font-weight: 700;
}}
.card .caption {{
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}}
.overall {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 700;
}}
.score {{
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 15px;
}}
.score.good {{ background: var(--good-bg); color: var(--good); }}
.score.warn {{ background: var(--warn-bg); color: var(--warn); }}
.score.bad {{ background: var(--bad-bg); color: var(--bad); }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: 13px;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 8px 6px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: #334155;
  font-weight: 600;
  background: #f8fafc;
}}
td.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.muted {{ color: var(--muted); }}
.badge {{
  border-radius: 999px;
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;
}}
.badge.good {{ color: var(--good); background: var(--good-bg); }}
.badge.warn {{ color: var(--warn); background: var(--warn-bg); }}
.badge.bad {{ color: var(--bad); background: var(--bad-bg); }}
.badge.info {{ color: var(--info); background: var(--info-bg); }}
.chart-grid {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
}}
.chart-card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  padding: 8px;
}}
.chart-title {{
  font-size: 13px;
  color: #0f172a;
  margin: 2px 2px 8px;
}}
.chart-canvas {{
  width: 100%;
  height: 240px;
  display: block;
}}
.series-legend {{
  margin-top: 8px;
  border-top: 1px solid var(--line);
  padding-top: 8px;
}}
.series-row {{
  display: grid;
  grid-template-columns: 14px 1fr 88px 88px 88px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  padding: 3px 0;
}}
.series-row.header {{
  color: var(--muted);
  font-weight: 600;
  padding-top: 1px;
}}
.series-chip {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid #ffffff;
  box-shadow: 0 0 0 1px #cbd5e1 inset;
}}
.series-row .num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.file-list {{
  line-height: 1.8;
  font-size: 13px;
}}
@media (max-width: 700px) {{
  .container {{ padding: 12px; }}
  .chart-canvas {{ height: 210px; }}
}}
</style>
</head>
<body>
  <div class="container">
    <h1>LoRA Diagnostic Report</h1>
    <p class="sub">{report.get("base_name", "-")} / generated at {report.get("generated_at", "-")}</p>

    <section class="panel">
      <div class="overall">
        総合診断:
        <span class="score {diagnostics.get("overall_class", "warn")}">{diagnostics.get("overall_status", "-")} ({diagnostics.get("score", 0)}点)</span>
      </div>
      <div class="grid cards" style="margin-top: 12px;">
        <div class="card">
          <div class="title">Grad しきい値超過率</div>
          <div class="value">{fmt_percent((((report.get("grad") or {}).get("summary") or {}).get("threshold_exceeded_ratio")), 2)}</div>
          <div class="caption">低いほど更新スキップの偏りが小さい</div>
        </div>
        <div class="card">
          <div class="title">Loss MA 低下率</div>
          <div class="value">{fmt_percent((((report.get("grad") or {}).get("summary") or {}).get("loss_ma_drop_ratio")), 2)}</div>
          <div class="caption">高いほど収束方向</div>
        </div>
        <div class="card">
          <div class="title">DQ Auto in-band比率</div>
          <div class="value">{fmt_percent((((report.get("dq") or {}).get("summary") or {}).get("in_band_ratio")), 2)}</div>
          <div class="caption">高いほど auto 制御が安定</div>
        </div>
        <div class="card">
          <div class="title">最終 QuantErrRatioEMA</div>
          <div class="value">{fmt_float((((report.get("dq") or {}).get("summary") or {}).get("final_quant_err_ratio_ema")), 4)}</div>
          <div class="caption">低いほど量子化誤差が小さい</div>
        </div>
        <div class="card">
          <div class="title">最終 RankSatP95</div>
          <div class="value">{fmt_float((((report.get("dq") or {}).get("summary") or {}).get("final_rank_sat_p95")), 4)}</div>
          <div class="caption">高すぎると rank 飽和の兆候</div>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>カテゴリ</th>
            <th>項目</th>
            <th class="num">値</th>
            <th>判定</th>
            <th>メモ</th>
          </tr>
        </thead>
        <tbody>{render_check_rows(diagnostics.get("checks", []))}</tbody>
      </table>
      <p class="sub">判定はヒューリスティクスです。プロジェクト特性に合わせてしきい値を調整してください。</p>
    </section>

    <section class="panel">
      <h2>GradNorm Dashboard</h2>
      <p class="sub">既存 `make_dashboard.py` 相当の5グラフを統合表示しています。</p>
      <div id="gradCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>DQ Delta Dashboard</h2>
      <p class="sub">X軸は TrainStep。縦線ラベルで Epoch を表示します。</p>
      <div id="dqCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>Group Loss Dashboard</h2>
      <p class="sub">`group_loss_logs+*.csv` を検出した場合に、groupごとのEMA loss推移を表示します。</p>
      <div id="groupLossCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>LoRA Checkpoint Analysis</h2>
      {lora_error_html}
      <div class="grid cards">
        <div class="card">
          <div class="title">総ブロック数</div>
          <div class="value">{fmt_int(lora_cards.get("total_blocks"))}</div>
          <div class="caption">解析対象LoRAブロック数</div>
        </div>
        <div class="card">
          <div class="title">総パラメータ</div>
          <div class="value">{fmt_int(lora_cards.get("total_params"))}</div>
          <div class="caption">up/down重み合算</div>
        </div>
        <div class="card">
          <div class="title">情報密度中央値</div>
          <div class="value">{fmt_float(lora_cards.get("density_median"))}</div>
          <div class="caption">高すぎ/低すぎの偏りを確認</div>
        </div>
        <div class="card">
          <div class="title">RMS中央値</div>
          <div class="value">{fmt_float(lora_cards.get("rms_median"))}</div>
          <div class="caption">更新量の強さの目安</div>
        </div>
        <div class="card">
          <div class="title">Entropy中央値</div>
          <div class="value">{fmt_float(lora_cards.get("entropy_median"))}</div>
          <div class="caption">分布の広がりの目安</div>
        </div>
        <div class="card">
          <div class="title">Sparsity中央値</div>
          <div class="value">{fmt_float(lora_cards.get("sparsity_median"))}</div>
          <div class="caption">高すぎると情報不足の可能性</div>
        </div>
      </div>

      <h3 style="margin-top: 16px;">モジュール別統計</h3>
      <table>
        <thead>
          <tr>
            <th>モジュール</th>
            <th class="num">ブロック数</th>
            <th class="num">総パラメータ</th>
            <th class="num">情報密度平均</th>
            <th class="num">情報密度中央値</th>
            <th class="num">RMS中央値</th>
            <th class="num">Entropy中央値</th>
          </tr>
        </thead>
        <tbody>{module_rows}</tbody>
      </table>

      <h3 style="margin-top: 16px;">UNetブロック別概要</h3>
      <table>
        <thead>
          <tr>
            <th>UNetブロック</th>
            <th class="num">LoRA数</th>
            <th class="num">総パラメータ</th>
            <th class="num">情報密度平均</th>
            <th class="num">情報密度中央値</th>
            <th class="num">RMS中央値</th>
          </tr>
        </thead>
        <tbody>{unet_rows}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>LoRA 情報密度エポック推移</h2>
      <p class="sub">`--lora_epoch_trend` 有効時のみ表示。系列ごとに色を固定し、下の凡例でブロック名との対応を示します。</p>
      {lora_trend_error_html}
      <div id="loraTrendCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>Input Files</h2>
      <div class="file-list">
        <div>Grad Log: <code>{(report.get("grad") or {}).get("path", "-")}</code></div>
        <div>DQ Log: <code>{(report.get("dq") or {}).get("path", "-")}</code></div>
        <div>DQ Auto Log: <code>{(report.get("dq") or {}).get("auto_path", "-")}</code></div>
        <div>Group Loss Step Log: <code>{(report.get("group_loss") or {}).get("step_path", "-")}</code></div>
        <div>Group Loss Epoch Log: <code>{(report.get("group_loss") or {}).get("epoch_path", "-")}</code></div>
        <div>LoRA Final Checkpoint: <code>{(report.get("lora") or {}).get("path", "-")}</code></div>
      </div>
    </section>
  </div>

<script>
const reportData = {report_json};

function withAlpha(hex, alpha) {{
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
}}

function finite(value) {{
  return typeof value === 'number' && Number.isFinite(value);
}}

function reduceMarkers(markers, maxCount = 14) {{
  if (!Array.isArray(markers) || markers.length <= maxCount) return markers || [];
  const step = Math.ceil(markers.length / maxCount);
  const out = [];
  for (let i = 0; i < markers.length; i += step) {{
    out.push(markers[i]);
  }}
  const last = markers[markers.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}}

function decimalsFromStep(step) {{
  if (!finite(step) || step <= 0) return 4;
  const txt = String(step);
  if (!txt.includes('.')) return 0;
  return Math.min(6, txt.split('.')[1].length);
}}

function buildNiceIntegerStep(minVal, maxVal, targetTicks = 6) {{
  const range = Math.max(1e-9, maxVal - minVal);
  const rough = range / Math.max(2, targetTicks);
  const base = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / base;
  let nice = 1;
  if (scaled > 5) {{
    nice = 10;
  }} else if (scaled > 2) {{
    nice = 5;
  }} else if (scaled > 1) {{
    nice = 2;
  }}
  return Math.max(1, Math.round(nice * base));
}}

function buildYAxisTicks(bounds, chart) {{
  let minVal = bounds.yMin;
  let maxVal = bounds.yMax;
  if (finite(chart.y_min_fixed)) {{
    minVal = chart.y_min_fixed;
  }}
  if (finite(chart.y_max_fixed)) {{
    maxVal = chart.y_max_fixed;
  }}
  if (finite(chart.y_min_floor)) {{
    minVal = Math.min(minVal, chart.y_min_floor);
  }}
  if (finite(chart.y_max_ceil)) {{
    maxVal = Math.max(maxVal, chart.y_max_ceil);
  }}
  if (!(maxVal > minVal)) {{
    const pad = Math.abs(maxVal || 1) * 0.05 + 1e-6;
    minVal -= pad;
    maxVal += pad;
  }}

  let step = null;
  let ticks = [];
  if (finite(chart.y_tick_step) && chart.y_tick_step > 0) {{
    step = chart.y_tick_step;
    let start = Math.floor(minVal / step) * step;
    let end = Math.ceil(maxVal / step) * step;
    let count = Math.round((end - start) / step) + 1;
    if (count > 24) {{
      const mul = Math.ceil(count / 24);
      step *= mul;
      start = Math.floor(minVal / step) * step;
      end = Math.ceil(maxVal / step) * step;
      count = Math.round((end - start) / step) + 1;
    }}
    for (let i = 0; i < count; i += 1) {{
      ticks.push(start + i * step);
    }}
    minVal = start;
    maxVal = end;
  }} else if (chart.y_tick_nice_integer) {{
    step = buildNiceIntegerStep(minVal, maxVal, 6);
    const start = Math.floor(minVal / step) * step;
    const end = Math.ceil(maxVal / step) * step;
    const count = Math.round((end - start) / step) + 1;
    for (let i = 0; i < count; i += 1) {{
      ticks.push(start + i * step);
    }}
    minVal = start;
    maxVal = end;
  }} else {{
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i += 1) {{
      ticks.push(minVal + ((maxVal - minVal) * i) / tickCount);
    }}
    step = (maxVal - minVal) / tickCount;
  }}
  return {{ min: minVal, max: maxVal, ticks, step }};
}}

function formatTickValue(value, chart, step) {{
  if (chart.y_tick_integer || chart.y_tick_nice_integer) {{
    return Math.round(value).toString();
  }}
  const precision = finite(chart.y_tick_precision) ? chart.y_tick_precision : decimalsFromStep(step);
  return Number(value).toFixed(Math.max(0, precision));
}}

function layoutInlineLegend(ctx, series, width, marginLeft, marginRight, maxRows) {{
  const rows = Math.max(1, maxRows || 2);
  const rowHeight = 14;
  const entries = [];
  let x = marginLeft;
  let row = 0;
  const rightLimit = Math.max(marginLeft + 20, width - marginRight);
  ctx.font = '11px sans-serif';
  (series || []).forEach((seriesItem) => {{
    const name = seriesItem.name || 'series';
    const itemWidth = 14 + ctx.measureText(name).width + 12;
    if (x + itemWidth > rightLimit && row + 1 < rows) {{
      row += 1;
      x = marginLeft;
    }}
    entries.push({{ series: seriesItem, x, y: 12 + row * rowHeight }});
    x += itemWidth;
  }});
  return {{ entries, height: rowHeight * (row + 1) + 4 }};
}}

function computeBounds(chart) {{
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  let hasData = false;

  (chart.series || []).forEach((series) => {{
    const xVals = Array.isArray(series.x) && series.x.length ? series.x : (Array.isArray(chart.x) ? chart.x : []);
    const yVals = Array.isArray(series.y) ? series.y : [];
    const n = Math.min(xVals.length, yVals.length);
    for (let i = 0; i < n; i += 1) {{
      const x = xVals[i];
      const y = yVals[i];
      if (!finite(x) || !finite(y)) continue;
      hasData = true;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }}
  }});

  if (!hasData) return null;
  if (Math.abs(yMax - yMin) < 1e-12) {{
    const pad = Math.abs(yMax || 1) * 0.05 + 1e-6;
    yMin -= pad;
    yMax += pad;
  }} else {{
    const pad = (yMax - yMin) * 0.10;
    yMin -= pad;
    yMax += pad;
  }}
  if (xMin === xMax) {{
    xMin -= 1;
    xMax += 1;
  }}
  return {{ xMin, xMax, yMin, yMax }};
}}

function drawChart(canvas, chart) {{
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(canvas.clientWidth));
  const height = Math.max(180, Math.floor(canvas.clientHeight));
  const pxW = Math.floor(width * ratio);
  const pxH = Math.floor(height * ratio);
  if (canvas.width !== pxW || canvas.height !== pxH) {{
    canvas.width = pxW;
    canvas.height = pxH;
  }}
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const bounds = computeBounds(chart);
  if (!bounds) {{
    ctx.fillStyle = '#64748b';
    ctx.font = '13px sans-serif';
    ctx.fillText('有効な描画データがありません', 12, 28);
    return;
  }}

  let legendLayout = null;
  let inlineLegendHeight = 0;
  if (!chart.no_inline_legend) {{
    legendLayout = layoutInlineLegend(
      ctx,
      chart.series || [],
      width,
      58,
      14,
      finite(chart.legend_max_rows) ? chart.legend_max_rows : 2
    );
    inlineLegendHeight = legendLayout.height;
  }}

  const margin = {{ top: 18 + inlineLegendHeight, right: 14, bottom: 34, left: 58 }};
  const chartW = Math.max(10, width - margin.left - margin.right);
  const chartH = Math.max(10, height - margin.top - margin.bottom);
  const yAxis = buildYAxisTicks(bounds, chart);
  const yRange = Math.max(1e-12, yAxis.max - yAxis.min);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  yAxis.ticks.forEach((tick) => {{
    const y = margin.top + chartH - ((tick - yAxis.min) / yRange) * chartH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartW, y);
    ctx.stroke();
  }});

  const markers = reduceMarkers(chart.markers, 12);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#cbd5e1';
  markers.forEach((marker) => {{
    if (!finite(marker.x)) return;
    const t = (marker.x - bounds.xMin) / (bounds.xMax - bounds.xMin);
    const x = margin.left + t * chartW;
    if (x < margin.left || x > margin.left + chartW) return;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + chartH);
    ctx.stroke();
    ctx.save();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.fillText(marker.label || '', x + 2, margin.top + 10);
    ctx.restore();
  }});
  ctx.setLineDash([]);

  (chart.series || []).forEach((series) => {{
    const xVals = (Array.isArray(series.x) && series.x.length) ? series.x : (chart.x || []);
    const yVals = series.y || [];
    const n = Math.min(xVals.length, yVals.length);
    ctx.strokeStyle = series.color || '#2563eb';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let moved = false;
    for (let i = 0; i < n; i += 1) {{
      const x = xVals[i];
      const y = yVals[i];
      if (!finite(x) || !finite(y)) {{
        moved = false;
        continue;
      }}
      const px = margin.left + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * chartW;
      const py = margin.top + chartH - ((y - yAxis.min) / yRange) * chartH;
      if (!moved) {{
        ctx.moveTo(px, py);
        moved = true;
      }} else {{
        ctx.lineTo(px, py);
      }}
    }}
    ctx.stroke();
  }});

  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + chartH);
  ctx.lineTo(margin.left + chartW, margin.top + chartH);
  ctx.stroke();

  ctx.fillStyle = '#475569';
  ctx.font = '11px sans-serif';
  yAxis.ticks.forEach((tick) => {{
    const y = margin.top + chartH - ((tick - yAxis.min) / yRange) * chartH;
    ctx.fillText(formatTickValue(tick, chart, yAxis.step), 6, y + 4);
  }});
  ctx.fillText(Math.round(bounds.xMin).toString(), margin.left, margin.top + chartH + 18);
  const xMid = (bounds.xMin + bounds.xMax) / 2;
  const midText = Math.round(xMid).toString();
  const midW = ctx.measureText(midText).width;
  ctx.fillText(midText, margin.left + chartW / 2 - midW / 2, margin.top + chartH + 18);
  const maxText = Math.round(bounds.xMax).toString();
  const maxW = ctx.measureText(maxText).width;
  ctx.fillText(maxText, margin.left + chartW - maxW, margin.top + chartH + 18);

  if (!chart.no_inline_legend && legendLayout) {{
    legendLayout.entries.forEach((entry) => {{
      const series = entry.series || {{}};
      const color = series.color || '#2563eb';
      const name = series.name || 'series';
      ctx.fillStyle = color;
      ctx.fillRect(entry.x, entry.y - 8, 10, 10);
      ctx.fillStyle = '#334155';
      ctx.font = '11px sans-serif';
      ctx.fillText(name, entry.x + 14, entry.y);
    }});
  }}
}}

function appendSeriesLegend(parent, chart) {{
  const rows = Array.isArray(chart.legend_rows) ? chart.legend_rows : [];
  if (!rows.length) return;
  const wrap = document.createElement('div');
  wrap.className = 'series-legend';

  const header = document.createElement('div');
  header.className = 'series-row header';
  header.innerHTML = '<div></div><div>ブロック</div><div class="num">開始密度</div><div class="num">終了密度</div><div class="num">差分</div>';
  wrap.appendChild(header);

  rows.forEach((row) => {{
    const el = document.createElement('div');
    el.className = 'series-row';
    const startVal = finite(row.start_density) ? Number(row.start_density).toFixed(4) : '-';
    const endVal = finite(row.end_density) ? Number(row.end_density).toFixed(4) : '-';
    const deltaVal = finite(row.delta_density) ? Number(row.delta_density).toFixed(4) : '-';
    el.innerHTML =
      `<div class="series-chip" style="background:${{row.color || '#2563eb'}};"></div>` +
      `<div>${{row.name || '-'}}</div>` +
      `<div class="num">${{startVal}}</div>` +
      `<div class="num">${{endVal}}</div>` +
      `<div class="num">${{deltaVal}}</div>`;
    wrap.appendChild(el);
  }});
  parent.appendChild(wrap);
}}

function mountCharts(containerId, charts) {{
  const container = document.getElementById(containerId);
  if (!container || !Array.isArray(charts)) return;
  if (charts.length === 0) {{
    const empty = document.createElement('div');
    empty.className = 'muted';
    empty.textContent = '表示できるデータがありません。';
    container.appendChild(empty);
    return;
  }}
  const states = [];
  charts.forEach((chart, idx) => {{
    const card = document.createElement('div');
    card.className = 'chart-card';
    const title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = chart.title || `Chart ${{idx + 1}}`;
    const canvas = document.createElement('canvas');
    canvas.className = 'chart-canvas';
    card.appendChild(title);
    card.appendChild(canvas);
    appendSeriesLegend(card, chart);
    container.appendChild(card);
    states.push({{ canvas, chart }});
  }});

  const renderAll = () => {{
    states.forEach((state) => drawChart(state.canvas, state.chart));
  }};
  renderAll();
  window.addEventListener('resize', renderAll);
}}

mountCharts('gradCharts', ((reportData.charts || {{}}).grad || []));
mountCharts('dqCharts', ((reportData.charts || {{}}).dq || []));
mountCharts('groupLossCharts', ((reportData.charts || {{}}).group_loss || []));
mountCharts('loraTrendCharts', ((reportData.charts || {{}}).lora_trend || []));
</script>
</body>
</html>
"""


def resolve_paths(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    input_dir = args.input_dir
    base_name = args.base_name

    grad_log = os.path.join(input_dir, f"gradient_logs+{base_name}.txt")
    dq_log = os.path.join(input_dir, f"dq_delta_logs+{base_name}.txt")
    dq_auto_log = os.path.join(input_dir, f"dq_delta_auto+{base_name}.txt")
    group_loss_step_log = os.path.join(input_dir, f"group_loss_logs+{base_name}.csv")
    group_loss_epoch_log = os.path.join(input_dir, f"group_loss_epoch+{base_name}.csv")
    model = os.path.join(input_dir, f"{base_name}.safetensors")

    return {
        "grad_log": grad_log,
        "dq_log": dq_log,
        "dq_auto_log": dq_auto_log,
        "group_loss_step_log": group_loss_step_log,
        "group_loss_epoch_log": group_loss_epoch_log,
        "model": model,
    }


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA学習ログ診断ダッシュボード生成ツール")
    parser.add_argument("--base_name", required=True, help="LoRAのベース名 (例: brak_xl31c_noob075V)")
    parser.add_argument(
        "--input_dir",
        default=".",
        help="ログとモデルがあるディレクトリ（固定名: gradient_logs+/dq_delta_logs+/dq_delta_auto+、任意で group_loss_logs+/group_loss_epoch+）",
    )
    parser.add_argument("--loss_ma_window", type=int, default=100, help="Loss移動平均の窓サイズ")
    parser.add_argument("--lora_bins", type=int, default=128, help="LoRA重み解析のヒストグラムビン数")
    parser.add_argument("--skip_lora_analysis", action="store_true", help="LoRAチェックポイント解析をスキップ")
    parser.add_argument("--lora_epoch_trend", action="store_true", help="情報密度のエポック推移解析を有効化")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="出力ディレクトリ（未指定時は --input_dir/diagnostic_report）",
    )
    parser.add_argument("--output_html", default=None, help="HTML出力先。未指定時は output_dir に自動生成")
    parser.add_argument("--output_json", default=None, help="JSON出力先。未指定時は output_dir に自動生成")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    paths = resolve_paths(args)
    grad_log_path = paths["grad_log"]
    dq_log_path = paths["dq_log"]
    dq_auto_log_path = paths["dq_auto_log"]
    group_loss_step_log_path = paths["group_loss_step_log"]
    group_loss_epoch_log_path = paths["group_loss_epoch_log"]
    model_path = paths["model"]

    if not grad_log_path or not os.path.exists(grad_log_path):
        raise FileNotFoundError(f"grad log が見つかりません: {grad_log_path}")
    if not dq_log_path or not os.path.exists(dq_log_path):
        raise FileNotFoundError(f"dq log が見つかりません: {dq_log_path}")
    if dq_auto_log_path and not os.path.exists(dq_auto_log_path):
        dq_auto_log_path = None
    if group_loss_step_log_path and not os.path.exists(group_loss_step_log_path):
        group_loss_step_log_path = None
    if group_loss_epoch_log_path and not os.path.exists(group_loss_epoch_log_path):
        group_loss_epoch_log_path = None

    grad_data = parse_grad_log(grad_log_path, args.loss_ma_window)
    dq_data = parse_dq_logs(dq_log_path, dq_auto_log_path, grad_data)
    group_loss_data = parse_group_loss_logs(group_loss_step_log_path, group_loss_epoch_log_path)

    lora_data = None
    lora_error = None
    lora_trend_data = None
    lora_trend_error = None
    if not args.skip_lora_analysis:
        if not os.path.exists(model_path):
            lora_error = f"LoRAチェックポイントが見つかりません: {model_path}"
        else:
            try:
                lora_data = analyze_lora_checkpoint(model_path, args.lora_bins)
            except Exception as exc:
                lora_error = f"LoRA解析に失敗しました: {exc}"
            if args.lora_epoch_trend:
                try:
                    print("[lora_epoch_trend] start", flush=True)
                    lora_trend_data = analyze_lora_epoch_trend(
                        model_path=model_path,
                        bins=args.lora_bins,
                    )
                    print("[lora_epoch_trend] complete", flush=True)
                except Exception as exc:
                    lora_trend_error = f"LoRAエポック推移解析に失敗しました: {exc}"
    elif args.lora_epoch_trend:
        lora_trend_error = "--skip_lora_analysis 指定時は --lora_epoch_trend を実行できません。"

    diagnostics = build_diagnostics(grad_data, dq_data, lora_data)
    charts = build_chart_payload(grad_data, dq_data, group_loss_data, lora_trend_data)

    report = {
        "base_name": args.base_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grad": grad_data,
        "dq": dq_data,
        "group_loss": group_loss_data,
        "lora": lora_data,
        "lora_error": lora_error,
        "lora_trend": lora_trend_data,
        "lora_trend_error": lora_trend_error,
        "diagnostics": diagnostics,
        "charts": charts,
    }

    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, "diagnostic_report")
    os.makedirs(output_dir, exist_ok=True)
    html_path = args.output_html or os.path.join(output_dir, f"{args.base_name}_diagnostic.html")
    json_path = args.output_json or os.path.join(output_dir, f"{args.base_name}_diagnostic.json")

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(sanitize_json(report), fp, ensure_ascii=False, indent=2)
    html = build_html(report)
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    print(f"JSON saved: {json_path}")
    print(f"HTML saved: {html_path}")
    if lora_error:
        print(f"[warn] {lora_error}")
    if lora_trend_error:
        print(f"[warn] {lora_trend_error}")


if __name__ == "__main__":
    main()
