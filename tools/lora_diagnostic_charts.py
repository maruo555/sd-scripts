"""Plot data and reference diagnostics for the offline SDXL report."""
from collections import defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.lora_diagnostic_numbers import fmt_float, fmt_percent, safe_float


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
DQ_LOW_QERR_PER_CLIP_THRESHOLD = 130.0


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
    rank_data: Optional[Dict[str, Any]],
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

        if dq_summary.get("low_band_seen"):
            qerr_per_clip_tail = dq_summary.get("tail_mean_qerr_per_clip")
            qerr_per_clip_max = dq_summary.get("max_qerr_per_clip")
            max_bad_streak = dq_summary.get("low_auto_max_bad_streak")
            frozen_bad_count = dq_summary.get("low_auto_frozen_bad_count")
            escaped = dq_summary.get("low_auto_escaped")
            bad_count = dq_summary.get("low_auto_bad_count")
            run_qerr_threshold = dq_summary.get("run_qerr_per_clip_threshold")
            run_threshold_note = ""
            if run_qerr_threshold is not None and abs(run_qerr_threshold - DQ_LOW_QERR_PER_CLIP_THRESHOLD) > 1e-9:
                run_threshold_note = f" run指定閾値は{run_qerr_threshold:g}。"
            if escaped or (frozen_bad_count is not None and frozen_bad_count > 0):
                status = "bad"
                note = "要改善 (low帯では誤差負荷が高い可能性。clip_rate_midまたはclip_rate_highでの再実行を推奨)"
            elif qerr_per_clip_tail is not None and qerr_per_clip_tail >= DQ_LOW_QERR_PER_CLIP_THRESHOLD:
                status = "bad"
                note = "要改善 (QErrPerClipが固定診断閾値130以上で継続)"
            elif (
                (qerr_per_clip_max is not None and qerr_per_clip_max >= DQ_LOW_QERR_PER_CLIP_THRESHOLD)
                or (max_bad_streak is not None and max_bad_streak >= 2)
                or (bad_count is not None and bad_count > 0)
            ):
                status = "warn"
                note = "注意 (low帯で一時的に誤差負荷が高い。mid/highとの比較候補)"
            elif qerr_per_clip_tail is None:
                status = "info"
                note = "データ不足 (QErrPerClip列が無いためlow適性は判定不可)"
            else:
                status = "good"
                note = "良好 (low clip帯でもQErrPerClipは安定)"
            checks.append(
                {
                    "section": "DQ",
                    "name": "clip_rate_low適性",
                    "value": fmt_float(qerr_per_clip_tail, 2),
                    "status": status,
                    "note": note + run_threshold_note,
                }
            )

            if frozen_bad_count is not None and frozen_bad_count > 0:
                checks.append(
                    {
                        "section": "DQ",
                        "name": "low_auto凍結後bad",
                        "value": str(frozen_bad_count),
                        "status": "warn",
                        "note": "freeze_progress後にbadが出ています。この実行ではmidへ逃げませんが、次回はlow以外の比較を推奨します。",
                    }
                )

    if rank_data:
        rank_summary = rank_data.get("summary", {})
        rank_sat_p95 = rank_summary.get("final_rank_sat_p95")
        level, text = grade_metric(rank_sat_p95, good_max=0.90, warn_max=0.97)
        checks.append(
            {
                "section": "Rank",
                "name": "最終 RankSatP95",
                "value": fmt_float(rank_sat_p95, 4),
                "status": level,
                "note": text + " (高止まりはrank飽和の兆候)",
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

    if not checks:
        return {
            "score": None,
            "overall_status": "情報不足",
            "overall_class": "info",
            "checks": [
                {
                    "section": "診断",
                    "name": "スコア算出可否",
                    "value": "-",
                    "status": "info",
                    "note": "Grad/DQ/Rank/LoRA の診断対象データが不足しているため総合スコアを算出できません",
                }
            ],
        }

    if not any(item["status"] in {"good", "warn", "bad"} for item in checks):
        return {"score": None, "overall_status": "情報不足", "overall_class": "info", "checks": checks}
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


def density_summary(report):
    model_path = report["model_path"]
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


def density_trend(history, model_path):
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
    files = history.get("files") if isinstance(history, dict) else []
    return {
        "model_path": model_path,
        "checkpoints": files,
        "modules": trend_modules,
    }


def _component_charts(
    rank_data: Optional[Dict[str, Any]],
    group_loss_data: Optional[Dict[str, Any]],
    lora_trend_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"grad": [], "dq": [], "dq_auto": [], "rank": [], "group_loss": [], "lora_trend": []}

    if rank_data:
        rows = rank_data.get("rows", [])
        x = [item.get("TrainStep") for item in rows]
        markers = rank_data.get("markers", [])
        rank_charts = [
            {
                "id": "rank_dim",
                "title": "RankDim",
                "subtitle": "LoRA rank の設定値",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_floor": 0.0,
                "series": [{"name": "RankDim", "color": ColorPalette[6], "y": [item.get("RankDim") for item in rows]}],
            },
            {
                "id": "rank_sat",
                "title": "RankSatWMean / P50 / P95 / Max / Top1P95",
                "subtitle": "rank の使われ方の広さと偏りの要約",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 1.1,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "legend_max_rows": 2,
                "series": [
                    {"name": "RankSatWMean", "color": ColorPalette[0], "y": [item.get("RankSatWMean") for item in rows]},
                    {"name": "RankSatP50", "color": ColorPalette[1], "y": [item.get("RankSatP50") for item in rows]},
                    {"name": "RankSatP95", "color": ColorPalette[2], "y": [item.get("RankSatP95") for item in rows]},
                    {"name": "RankSatMax", "color": ColorPalette[3], "y": [item.get("RankSatMax") for item in rows]},
                    {"name": "RankTop1P95", "color": ColorPalette[4], "y": [item.get("RankTop1P95") for item in rows]},
                ],
            },
            {
                "id": "rank_energy",
                "title": "RankEnergySum",
                "subtitle": "LoRA 重み量の総量",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [{"name": "RankEnergySum", "color": ColorPalette[7], "y": [item.get("RankEnergySum") for item in rows]}],
            },
        ]

        lr_scope_specs = [
            ("Unet", "UnetLRMin", "UnetLRMax"),
            ("Te1", "Te1LRMin", "Te1LRMax"),
            ("Te2", "Te2LRMin", "Te2LRMax"),
        ]
        lr_series = []
        lr_color_index = 0
        for scope_label, min_key, max_key in lr_scope_specs:
            min_values = [item.get(min_key) for item in rows]
            max_values = [item.get(max_key) for item in rows]
            if not any(value is not None for value in min_values + max_values):
                continue
            if min_values == max_values:
                lr_series.append(
                    {
                        "name": f"{scope_label}LR",
                        "color": color_for_index(lr_color_index, 6),
                        "y": max_values,
                    }
                )
                lr_color_index += 1
            else:
                lr_series.append(
                    {
                        "name": f"{scope_label}LRMin",
                        "color": color_for_index(lr_color_index, 6),
                        "y": min_values,
                    }
                )
                lr_color_index += 1
                lr_series.append(
                    {
                        "name": f"{scope_label}LRMax",
                        "color": color_for_index(lr_color_index, 6),
                        "y": max_values,
                    }
                )
                lr_color_index += 1

        if lr_series:
            rank_charts.append(
                {
                    "id": "rank_lr",
                    "title": "Rank Log LR Snapshot",
                    "subtitle": "学習率の推移",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "legend_max_rows": 2,
                    "series": lr_series,
                }
            )

        grouped = rank_data.get("grouped") or {}
        group_display_names = {
            "path": {
                "down": "Down",
                "mid": "Mid",
                "up": "Up",
                "other": "Other",
            },
            "role": {
                "to_q": "Q",
                "to_k": "K",
                "to_v": "V",
                "to_out": "Out",
                "ff": "FF",
                "resnet": "ResNet",
                "sampler": "Sampler",
                "conv": "Conv",
                "other": "Other",
            },
        }

        def _append_grouped_rank_chart(
            chart_id: str,
            title: str,
            subtitle: str,
            group_kind: str,
            metric_key: str,
            *,
            y_min_fixed: Optional[float] = None,
            y_max_fixed: Optional[float] = None,
            y_tick_step: Optional[float] = None,
            y_tick_precision: Optional[int] = None,
        ):
            group_data = grouped.get(group_kind) or {}
            group_rows = group_data.get("rows", []) or []
            labels = group_data.get("labels", []) or []
            if not group_rows or not labels:
                return

            group_x = [item.get("TrainStep") for item in group_rows]
            series = []
            for idx, label in enumerate(labels):
                y = [((item.get("groups") or {}).get(label) or {}).get(metric_key) for item in group_rows]
                if not any(value is not None for value in y):
                    continue
                series.append(
                    {
                        "name": group_display_names.get(group_kind, {}).get(label, label),
                        "color": color_for_index(idx, max(1, len(labels))),
                        "y": y,
                    }
                )
            if not series:
                return

            chart = {
                "id": chart_id,
                "title": title,
                "subtitle": subtitle,
                "x_label": "TrainStep",
                "markers": markers,
                "x": group_x,
                "legend_max_rows": 2,
                "series": series,
            }
            if y_min_fixed is not None:
                chart["y_min_fixed"] = y_min_fixed
            if y_max_fixed is not None:
                chart["y_max_fixed"] = y_max_fixed
            if y_tick_step is not None:
                chart["y_tick_step"] = y_tick_step
            if y_tick_precision is not None:
                chart["y_tick_precision"] = y_tick_precision
            rank_charts.append(chart)

        _append_grouped_rank_chart(
            "rank_group_path_energy_share",
            "Path Group Energy Share",
            "ブロック位置ごとの重み量シェア（合計1.0）",
            "path",
            "RankEnergyShare",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_path_energy_share_per_param",
            "Path Group Energy Share Per Param",
            "ブロック位置ごとの正規化重み量シェア（合計1.0）",
            "path",
            "RankEnergySharePerParam",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_path_sat",
            "Path Group RankSatWMean",
            "ブロック位置ごとの rank の使われ方の広さ",
            "path",
            "RankSatWMean",
            y_min_fixed=0.0,
            y_max_fixed=1.1,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_energy_share",
            "Role Group Energy Share",
            "役割ごとの重み量シェア（合計1.0）",
            "role",
            "RankEnergyShare",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_energy_share_per_param",
            "Role Group Energy Share Per Param",
            "役割ごとの正規化重み量シェア（合計1.0）",
            "role",
            "RankEnergySharePerParam",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_sat",
            "Role Group RankSatWMean",
            "役割ごとの rank の使われ方の広さ",
            "role",
            "RankSatWMean",
            y_min_fixed=0.0,
            y_max_fixed=1.1,
            y_tick_step=0.1,
            y_tick_precision=1,
        )

        payload["rank"] = rank_charts

    if group_loss_data:
        step_rows = group_loss_data.get("step_rows", []) or []
        group_loss_charts: List[Dict[str, Any]] = []

        if step_rows:
            step_points: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": []})
            for item in step_rows:
                x = item.get("global_step")
                y = item.get("ema_loss_group")
                group = item.get("group") or "other"
                if x is None or y is None:
                    continue
                step_points[group]["x"].append(x)
                step_points[group]["y"].append(y)

            step_groups = sorted(step_points.keys())
            step_series = []
            for idx, group in enumerate(step_groups):
                step_series.append(
                    {
                        "name": group,
                        "color": color_for_index(idx, len(step_groups)),
                        "x": step_points[group]["x"],
                        "y": step_points[group]["y"],
                    }
                )

            group_loss_charts.append(
                {
                    "id": "group_loss_step_ema",
                    "title": "Group Loss EMA (step)",
                    "x_label": "Global Step",
                    "markers": group_loss_data.get("step_markers", []),
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.05,
                    "y_tick_precision": 2,
                    "series": step_series,
                }
            )

        payload["group_loss"] = group_loss_charts

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


def _chart(chart_id, title, axis, rows, x_key, fields, *, condition=None):
    x = [r[x_key] for r in rows]
    series = []
    for i, (field, name) in enumerate(fields):
        values = [safe_float(r.get(field)) for r in rows]
        if any(v is not None for v in values):
            series.append(dict(name=name, color=ColorPalette[i % len(ColorPalette)], y=values))
    return dict(id=chart_id, title=title, x_label=axis, x=x, series=series, comparison_condition=condition or {})


def report_charts(data, trend=None):
    grad, dq, rank, group = (data[k] for k in ("grad", "dq", "rank", "group_loss"))
    groups_for_chart = dict(step_rows=[{"global_step": r["attempt_step"], "group": r["group"],
                                       "ema_loss_group": r["group_ema_loss"]} for r in group["rows"]],
                            step_markers=group["markers"])
    charts = _component_charts(rank if rank["rows"] else None, groups_for_chart, trend)
    points = []
    for i, row in enumerate(grad["rows"]):
        if i in grad["cuts"]:
            points.append({"index": i + .5})
        points.append(dict(index=i + 1, grad_norm=grad["gradient_norm"][i], threshold=grad["threshold"][i],
                           loss=grad["loss"][i], loss_ma=grad["loss_ma"][i], scale=grad["scale"][i],
                           thresh_off=grad["thresh_off"][i]))
    for chart_id, title, fields in [
        ("grad_norm", "Gradient Norm / Threshold", [("grad_norm","Gradient Norm"),("threshold","Threshold")]),
        ("loss_ma", "Loss / Moving Average", [("loss","Loss"),("loss_ma","Loss MA")]),
        ("thresh_off", "閾値の無効化", [("thresh_off","ThreshOff")]),
        ("scale", "GradScaler", [("scale","Scale")]),
    ]:
        chart = _chart(chart_id, title, "Batch observation", points, "index", fields)
        chart["markers"] = grad["markers"]
        if chart["series"]:
            charts["grad"].append(chart)
    groups = {**dq["scope_rows"]}
    if dq["raw_scope"] == "both" and dq["pooled"]:
        groups["both"] = dq["pooled"]
    for scope, rows in groups.items():
        condition = {"scope":scope, "definition_version":dq["definition"], "target":"delta", "source":"observed"}
        for key, title, fields in [
            ("dq_clip","clip率",[("clip_rate_raw","raw")]),
            ("dq_zero_rate","ゼロ率",[("zero_rate","raw")]),
            ("dq_absmax","最大絶対値",[("input_abs_max","abs max")]),
            ("dq_range","量子化範囲",[("quant_range","range")]),
            ("dq_rms","入力RMS",[("input_rms","RMS")]),
            ("dq_qerr_rms","量子化誤差RMS",[("quant_error_rms_raw","raw")]),
            ("dq_qerr_ratio","相対量子化誤差",[("quant_error_ratio_raw","raw")]),
        ]:
            if scope == dq["raw_scope"] and key in {"dq_qerr_rms","dq_qerr_ratio"}:
                fields = fields + [("log_scope_error_rms_ema" if key == "dq_qerr_rms" else "log_scope_error_ratio_ema", "ログ対象EMA")]
            chart = _chart(f"{key}_{scope}",f"{scope} · {title}","Attempt step",rows,"attempt_step",fields,condition=condition)
            if chart["series"]:
                charts["dq"].append(chart)
    common = dq["pooled"]
    for key, field in (("dq_bits","bits"),("dq_range_mul","common_mul_after")):
        chart = _chart(key, field, "Attempt step", common, "attempt_step", [(field,field)],
                       condition={"scope":dq["raw_scope"],"definition_version":dq["definition"]})
        if chart["series"]:
            charts["dq"].append(chart)
    for key, title, fields in [
        ("dq_auto_clip", "制御scopeのclip率と帯", [("clip_rate_raw","raw"),("clip_rate_ema","EMA"),("active_clip_low","下限"),("active_clip_high","上限")]),
        ("dq_auto_mul", "倍率の判定前後", [("common_mul_before","前"),("common_mul_after","後")]),
        ("dq_auto_qerr", "制御scopeの誤差", [("auto_quant_error_ratio_raw","raw"),("auto_quant_error_ratio_ema","EMA")]),
        ("dq_qerr_per_clip", "clip率当たりの誤差", [("qerr_per_clip","誤差 / clip")]),
        ("dq_low_auto_streak", "low_autoの連続数", [("low_auto_bad_streak","連続数")]),
    ]:
        chart = _chart(key,title,"Attempt step",dq["events"],"attempt_step",fields,
                       condition={"scope":dq["auto_scope"],"definition_version":dq["definition"],"source":"controller"})
        if chart["series"]:
            charts["dq_auto"].append(chart)
    chart = _chart("promote_proxy_loss","固定bankのproxy loss","Epoch",data["promote"]["rows"],"epoch",
                   [("raw_proxy_loss","raw"),("selected_proxy_loss","候補")])
    charts["promote"] = [chart] if chart["series"] else []
    for section in ("rank", "group_loss"):
        for chart in charts[section]:
            chart["x_label"] = "Attempt step"
    return charts
