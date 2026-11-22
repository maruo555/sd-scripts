"""Presentation-only filtering and copy for the standalone diagnostic HTML."""

import math
import re
from html import escape
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.lora_diagnostic_numbers import safe_float, sanitize_json, fmt_float, fmt_percent

SECTION_NAMES = {
    "grad": "GradNorm", "dq": "DQ Delta", "dq_auto": "DQ 自動mul調整の監視", "dq_guard": "DQ Guard", "dq_direct": "DQ Direct", "rank": "Rank",
    "group_loss": "Group Loss", "lora_trend": "LoRA 情報密度エポック推移",
}
CONSTANT_CHARTS = {"rank_dim", "dq_rank_dim", "dq_bits", "dq_range_mul", "thresh_off"}
REFERENCE_NAMES = {
    "threshold", "activecliplow", "activecliphigh", "limit", "windowdecisionlimit",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def text(value):
    return "—" if value is None else escape(str(value), quote=True)


def number(value):
    return f"{value:.6g}" if finite(value) else "—"


def is_reference(series):
    name = str(series.get("name", "")).lower()
    return series.get("is_reference", False) or name in REFERENCE_NAMES or name.startswith(("run threshold ", "fixed reference "))


def prepare_display(report):
    """Copy charts without mutating the analysis payload; inspect valid (x, y) pairs."""
    charts, notes = {}, {}
    for section, source_charts in (report.get("charts") or {}).items():
        charts[section], notes[section] = [], []
        for original in source_charts or []:
            chart = dict(original)
            kept, observations, all_observations = [], [], []
            title = chart.get("title") or chart.get("id", "グラフ")
            for original_series in chart.get("series") or []:
                series = dict(original_series)
                xs = series.get("x") or chart.get("x") or []
                ys = series.get("y") or []
                pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
                if not pairs:
                    notes[section].append(f"{title} / {series.get('name', '系列')}：有効値なし")
                    continue
                # Keep the positions of missing values so the renderer never bridges gaps.
                series["y"] = [y if finite(y) else None for y in ys]
                kept.append(series)
                observation = (series, pairs, len(pairs) < max(len(xs), len(ys)))
                all_observations.append(observation)
                if not is_reference(series):
                    observations.append(observation)
            if not observations:
                if kept:
                    notes[section].append(f"{title}：実測値なし（参考線のみ）")
                continue
            chart["series"] = kept
            names = {s.get("name") for s in kept}
            chart["legend_rows"] = [r for r in chart.get("legend_rows", []) if r.get("name") in names]
            # Older trend titles contain a series count that must follow the visible legend.
            chart["title"] = re.sub(r"\(\d+/(\d+)系列\)", lambda m: f"({len(kept)}/{m[1]}系列)", title)
            missing = any(item[2] for item in observations)
            if missing:
                notes[section].append(f"{title}：一部欠測")
            chart["display_values"] = [
                {"name": s.get("name", "系列"), "value": pairs[0][1], "x": pairs[0][0], "missing": gaps}
                for s, pairs, gaps in observations
            ]
            # One sample is an observation, not evidence of a constant time series.
            if all(len(pairs) == 1 for _, pairs, _ in all_observations):
                chart["display_mode"] = "single"
                chart["display_values"] = [
                    {"name": s.get("name", "系列"), "value": pairs[0][1], "x": pairs[0][0], "missing": gaps}
                    for s, pairs, gaps in all_observations
                ]
            elif (
                chart.get("id") in CONSTANT_CHARTS
                and all(len(pairs) >= 2 and all(y == pairs[0][1] for _, y in pairs) for _, pairs, _ in observations)
                and (chart.get("id") != "thresh_off" or all(pairs[0][1] == 0 for _, pairs, _ in observations))
            ):
                chart["display_mode"] = "constant"
            else:
                chart["display_mode"] = "chart"
            chart["display_missing"] = missing
            if chart.get("id") == "dq_qerr_per_clip" and str(chart.get("subtitle", "")).startswith("QErrPerClip ="):
                chart["subtitle"] = "Run threshold：実行時の閾値／Fixed reference：固定参考値（ある場合）"
            if chart.get("id") in {"cosine", "cosine_by_epoch"}:
                original_note = chart.get("subtitle") or ""
                caution = "隣接stepの勾配方向の近さ。上昇だけで過学習は判定できません。"
                chart["subtitle"] = original_note if "過学習" in original_note else " ".join(filter(None, [original_note, caution]))
            charts[section].append(chart)
    return {**report, "charts": charts}, notes


def table(headers, rows, numeric_columns=()):
    if not rows:
        return ""
    table_class = "table-scroll wide-table" if len(headers) >= 4 else "table-scroll"
    return f"<div class='{table_class}'><table><thead><tr>" + "".join(
        f"<th class='num'>{text(h)}</th>" if i in numeric_columns else f"<th>{text(h)}</th>" for i, h in enumerate(headers)
    ) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td class='num'>{cell}</td>" if i in numeric_columns else f"<td>{cell}</td>" for i, cell in enumerate(row)) + "</tr>" for row in rows
    ) + "</tbody></table></div>"


def details(label, body, css="help"):
    return f"<details class='{css}'><summary>{text(label)}</summary>{body}</details>" if body else ""


def panel(title, body, section_id=""):
    return f"<section class='panel' id='{text(section_id)}'><h2>{text(title)}</h2>{body}</section>" if body else ""


def cards(items):
    visible = [item for item in items if finite(item[1])]
    if not visible:
        return ""
    return "<div class='grid cards'>" + "".join(
        f"<div class='card'><div class='title'>{text(label)}</div><div class='value'>{text(fmt(value))}</div>"
        + (f"<div class='caption'>{text(caption)}</div>" if caption else "") + "</div>"
        for label, value, fmt, caption in visible
    ) + "</div>"


def section_help(section, charts):
    if not charts:
        return ""
    ids = {c.get("id", "") for c in charts}
    if section == "rank":
        rows = [
            ("RankSat", "高いほど、設定したrankの成分を広く使用"),
            ("WMean / P50 / P95 / Max", "重み付き平均／中央値／95パーセンタイル／最大値"),
            ("Top1P95", "最大成分への集中度の95パーセンタイル"),
            ("Energy", "LoRA重み量"),
        ]
        if any("energy_share" in i for i in ids):
            rows.append(("Energy Share", "重み量の構成比"))
        if any("per_param" in i for i in ids):
            rows.append(("Energy Share Per Param", "パラメータ数で補正した重み量の構成比"))
        if any("path" in i for i in ids):
            rows.append(("Path", "UNetの位置：Down（入力側）・Mid（中央）・Up（出力側）"))
        if any("role" in i for i in ids):
            rows.append(("Role", "層の役割：Q・K・V・Out（Attention）、FF（特徴変換）、その他"))
        return details("指標と系列の見方", table(["用語", "意味"], [(text(a), text(b)) for a, b in rows])
                       + "<p class='sub'>重み量とrank成分の広がりは一致するとは限りません。</p>")
    if section == "grad":
        return details("指標の見方", "<p>Loss MAは移動平均です。低下率は開始側・終了側の各15%の平均を比較します。"
                       "ScaleはGradScalerの値、ThreshOffはしきい値判定の無効化に関する記録です。</p>")
    if section == "lora_trend":
        return details("指標の見方", "<p>情報密度は、正規化エントロピー・非ゼロ率・正規化RMSの積です。"
                       "RMSの正規化基準は各チェックポイント内の最大値のため、基準自体の変化も推移に含まれます。</p>")
    if section == "dq" and "dq_qerr_per_clip" in ids:
        return details("指標の見方", "<p>QErrPerClip = auto quant-error EMA / max(ClipRateEMA, floor)。"
                       "Run thresholdは実行時の閾値、Fixed referenceは固定参考値です。</p>")
    return ""


def constant_summary(chart):
    items = []
    for item in chart.get("display_values", []):
        qualifier = "有効な記録では一定・一部欠測" if item["missing"] else "記録期間中一定"
        if chart.get("id") == "thresh_off":
            value = "有効な記録では発生なし・一部欠測" if item["missing"] else "発生なし"
        else:
            value = f"{number(item['value'])}（{qualifier}）"
        items.append(f"<span class='constant-value'>{text(item['name'])}：{text(value)}</span>")
    return "".join(items)


def dq_context(data):
    scope, control = data.get("raw_scope"), data.get("auto_scope")
    info = f"<p>ログの範囲 {text(scope)} ／ 自動調整の範囲 {text(control)}</p>"
    if data.get("multiplier_counts"):
        info += table(["開始時の部位別倍率", "対象モジュール数"], [[text(m), text(n)] for m,n in data["multiplier_counts"]])
    stats = data.get("summary", {})
    if data.get("events"):
        info += table(["check数", "帯内率の対象", "帯内率の対象外", "倍率適用"],
                      [[text(stats.get(k)) for k in ("check_count","eligible_band_checks","excluded_band_checks","applied_count")]])
        decisions = [[text(row.get(k)) for k in ("attempt_step", "event_type", "auto_reason",
                      "common_mul_before", "common_mul_after", "low_auto_phase", "low_auto_decision")]
                     for row in data["events"]]
        info += details("倍率の判定履歴", table(["試行", "種別", "理由", "判定前", "判定後", "low_auto 段階", "low_auto 判定"], decisions))
        mode = data.get("controller_preset")
        if mode:
            info += "<p>制御プリセット " + text(mode) + ("。誤差はlow_autoの判定に使用します。" if mode == "clip_rate_low_auto" else "。誤差は参考値です。") + "</p>"
    return info


def render_settings(settings):
    if not settings:
        return ""
    content = "<p>" + text(settings["status"]) + "</p>"
    for key, title in (("requested", "指定値"), ("resolved", "実効設定")):
        if settings.get(key) is not None:
            body = "<pre>" + text(json.dumps(settings[key], ensure_ascii=False, indent=2)) + "</pre>"
            content += details(title, body)
    return panel("学習設定", content)


def render_body(report, notes, heatmaps=""):
    """Render visible sections, leaving numerical diagnostics and raw JSON intact."""
    diagnostics = report.get("diagnostics") or {}
    summary = lambda key: (report.get(key) or {}).get("summary") or {}
    pct = lambda value: f"{value * 100:.2f}%"
    dec = lambda value: f"{value:.4f}"
    score = diagnostics.get("score")
    body = f"<h1>LoRA Diagnostic Report</h1><p class='sub'>{text(report.get('base_name', '—'))} / {text(report.get('generated_at', '—'))}</p>"
    overall = f"<div class='overall'>総合診断：<span class='score {text(diagnostics.get('overall_class', 'info'))}'>"
    overall += text(diagnostics.get("overall_status", "情報不足"))
    overall += f"（{text(score)}点）" if score is not None else ""
    overall += "</span></div>"
    overall += cards([
        ("Grad しきい値超過率", summary("grad").get("threshold_exceeded_ratio"), pct, ""),
        ("Loss MA 低下率", summary("grad").get("loss_ma_drop_ratio"), pct, "開始・終了側の各15%を比較"),
        ("DQ Auto in-band比率", summary("dq").get("in_band_ratio"), pct, ""),
        ("最終 QuantErrRatioEMA", summary("dq").get("final_quant_err_ratio_ema"), dec,
         "ログ対象合算の学習ログ値" if len((report.get("dq") or {}).get("scope_rows") or {}) > 1 else "学習ログの記録値"),
        ("最終 RankSatP95", summary("rank").get("final_rank_sat_p95"), dec, ""),
    ])
    check_rows = []
    missing_checks = []
    labels = {"good": "良好", "warn": "注意", "bad": "要改善", "info": "情報"}
    for check in diagnostics.get("checks") or []:
        status = check.get("status", "info")
        value = check.get("value")
        if status == "info" and value in (None, "-", "—", ""):
            missing_checks.append(f"{check.get('name', '診断')}：{check.get('note', 'データ不足')}")
            continue
        note = re.sub(r"^(良好|注意|要改善)\s*", "", check.get("note") or "")
        if note.startswith("(") and note.endswith(")"):
            note = note[1:-1]
        check_rows.append([text(check.get("section", "")), text(check.get("name", "")), text(value if value is not None else "—"),
                           f"<span class='badge {text(status)}'>{text(labels.get(status, status))}</span>", text(note)])
    overall += table(["カテゴリ", "項目", "値", "判定", "メモ"], check_rows, (2,))
    overall += "<p class='sub'>判定は参考基準に基づく目安です。</p>"
    body += f"<section class='panel'>{overall}</section>"

    if missing_checks:
        body += details("算出できなかった診断項目", "<ul>" + "".join("<li>" + text(v) + "</li>" for v in missing_checks) + "</ul>")
    body += render_settings(report.get("training_settings"))

    errors = [report.get(k) for k in ("lora_error", "lora_trend_error") if report.get(k)]
    if errors:
        body += "<div class='analysis-errors' role='status'><strong>解析に関する確認事項</strong>"
        for error in errors:
            short = str(error).split(":", 1)[0].split("：", 1)[0]
            body += f"<p>{text(short)}</p>"
        body += "</div>"

    descriptions = {
        "grad": "勾配・Lossの推移", "dq": "量子化指標の推移",
        "rank": "重み量とrank成分の広がりを確認します。",
        "group_loss": "グループ別Lossの推移", "lora_trend": "ブロック・層別の情報密度推移",
    }

    def chart_panel(section):
        charts = (report.get("charts") or {}).get(section) or []
        extra = heatmaps if section == "rank" else ""
        warnings = (report.get(section) or {}).get("warnings") or []
        if isinstance(warnings, str):
            warnings = [warnings]
        if not charts and not extra and not warnings:
            return ""
        description = descriptions.get(section)
        content = f"<p class='sub'>{text(description)}</p>" if description else ""
        if section == "dq":
            content += dq_context(report.get("dq") or {})
        if section == "dq_auto":
            monitor = (report.get("dq") or {}).get("auto_monitor") or {}
            content += "".join(f"<p class='sub'>{text(note)}</p>" for note in monitor.get("notes", []))
        if warnings:
            content += "<div class='analysis-errors'><strong>記録された確認事項</strong>" + "".join(f"<p>{text(w)}</p>" for w in warnings) + "</div>"
        if charts:
            axes = {str(c.get("x_label", "")).lower() for c in charts}
            axis = "横軸：epoch" if axes == {"epoch"} or section == "lora_trend" else (
                "横軸：step" if axes <= {"trainstep", "step", "global_step"} else "横軸：step／epoch（各グラフに表示）"
            )
            if any(c.get("markers") for c in charts):
                axis += "／縦線：epoch境界・イベント"
            content += f"<p class='axis-note'>{text(axis)}</p>"
        content += section_help(section, charts)
        # The allocation above is authoritative for spatial mul. Keep the raw
        # RangeMul chart in the disclosure, without a misleading uniform badge.
        spatial_mul = section == "dq" and len((report.get("dq") or {}).get("multiplier_counts") or []) > 1
        constants = "".join(constant_summary(c) for c in charts
                            if c.get("display_mode") == "constant" and not (spatial_mul and c.get("id") == "dq_range_mul"))
        if constants:
            content += f"<div class='constant-values'>{constants}</div>"
        content += f"<div id='charts-{text(section)}' class='chart-grid'></div>" + extra
        return panel(SECTION_NAMES.get(section, section.replace("_", " ").title()), content, f"section-{section}")

    for section in (report.get("charts") or {}):
        if section != "lora_trend":
            body += chart_panel(section)
    if "rank" not in (report.get("charts") or {}) and heatmaps:
        body += chart_panel("rank")

    lora = report.get("lora") or {}
    lc = lora.get("summary_cards") or {}
    lora_body = cards([
        ("総ブロック数", lc.get("total_blocks"), lambda v: f"{v:,.0f}", ""),
        ("総パラメータ", lc.get("total_params"), lambda v: f"{v:,.0f}", "up/down重み合算"),
        ("情報密度中央値", lc.get("density_median"), dec, "重み統計から算出した指標"),
        ("RMS中央値", lc.get("rms_median"), dec, "重みの大きさ"),
        ("Entropy中央値", lc.get("entropy_median"), dec, "重み分布のエントロピー"),
        ("Sparsity中央値", lc.get("sparsity_median"), dec, "ゼロ値の割合"),
    ])
    for key, title, headers in [
        ("module_summary", "モジュール別統計", ["モジュール", "ブロック数", "総パラメータ", "情報密度平均", "情報密度中央値", "RMS中央値", "Entropy中央値"]),
        ("unet_block_summary", "UNetブロック別概要", ["UNetブロック", "LoRA数", "総パラメータ", "情報密度平均", "情報密度中央値", "RMS中央値"]),
    ]:
        rows = []
        for entry in lora.get(key) or []:
            values = [entry.get("block_count"), entry.get("total_params"), (entry.get("density") or {}).get("mean"),
                      (entry.get("density") or {}).get("median"), (entry.get("rms") or {}).get("median")]
            if key == "module_summary":
                values.append((entry.get("entropy_norm") or {}).get("median"))
            if not any(finite(v) for v in values):
                continue
            label = entry.get("module") if key == "module_summary" else entry.get("label")
            if key == "module_summary":
                label = {"unet": "UNet", "te1": "TE1 (Text Encoder 1)", "te2": "TE2 (Text Encoder 2)"}.get(label, label)
            elif label:
                label = re.sub(r"^input_blocks_", "Input ", label)
                label = re.sub(r"^output_blocks_", "Output ", label).replace("middle_block", "Middle Block")
            rows.append([text(label or "—")] + [text(f"{v:,.0f}" if index < 2 else f"{v:.4f}") if finite(v) else "—" for index, v in enumerate(values)])
        if rows:
            lora_body += f"<h3>{text(title)}</h3>" + table(headers, rows, range(1, len(headers)))
    body += panel("LoRA Checkpoint Analysis", lora_body, "section-lora")
    body += chart_panel("lora_trend")

    inputs = report["provenance"]["inputs"]
    status_rows = [[text(k), text(v["status"]), text(v.get("rows")), text(v.get("path") or "—"),
                    text(" / ".join(v.get("issues", [])))] for k, v in inputs.items()]
    body += details("入力ログの状態・観測範囲", table(["入力", "状態", "読取行", "ファイル", "確認事項"], status_rows))
    promote = report.get("promote") or {}
    if promote.get("rows"):
        rows = [[text(r.get(k)) for k in ("epoch","status","window","raw_proxy_loss","selected_proxy_loss",
                 "difference_percent","margin_ok","win_streak","promote_applied","optimizer_reset_applied")] for r in promote["rows"]]
        body += panel("Promoteの判定履歴", table(["Epoch","状態","窓","raw","候補","差 (%)","margin勝ち","連勝","採用","reset"], rows))
    errors = report.get("analysis_issues") or []
    if errors:
        body += details("解析時の確認事項", "<ul>" + "".join("<li>" + text(issue) + "</li>" for issue in errors) + "</ul>")
    return body


def heatmap_display_label(kind: str, label: str) -> str:
    if kind == "path":
        return {"down": "Down", "mid": "Mid", "up": "Up"}.get(label, label)
    if kind == "role":
        return {
            "to_q": "Q",
            "to_k": "K",
            "to_v": "V",
            "to_out": "Out",
            "ff": "FF",
            "other": "Other",
        }.get(label, label)
    return label


def heatmap_cell_style(value: Optional[float], max_value: Optional[float], *, fixed_max: Optional[float] = None) -> str:
    if value is None:
        return "background: #f8fafc; color: #94a3b8;"
    denom = fixed_max if fixed_max is not None else max_value
    if not denom or denom <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / denom))
    hue = 210 - int(22 * ratio)
    sat = 70
    light = 98 - int(34 * ratio)
    text_light = 18 if ratio >= 0.48 else 24
    return f"background: hsl({hue} {sat}% {light}%); color: hsl(217 33% {text_light}%);"


def render_rank_heatmap_card(
    title: str,
    subtitle: str,
    heatmap: Optional[Dict[str, Any]],
    *,
    fixed_max: Optional[float] = None,
) -> str:
    if not heatmap or not (heatmap.get("rows") or []):
        return ""
    role_labels = heatmap.get("role_labels") or []
    rows = heatmap.get("rows") or []
    max_value = heatmap.get("max_value")
    train_step = heatmap.get("TrainStep")
    epoch = heatmap.get("Epoch")
    header_cells = "".join(f"<th class='num'>{heatmap_display_label('role', label)}</th>" for label in role_labels)
    body_rows: List[str] = []
    for row in rows:
        cells_html: List[str] = []
        for cell in row.get("cells") or []:
            value = cell.get("value")
            text = fmt_float(value, 3) if value is not None else "-"
            style = heatmap_cell_style(value, max_value, fixed_max=fixed_max)
            cells_html.append(f"<td class='num heatmap-cell' style='{style}'>{text}</td>")
        body_rows.append(
            "<tr>"
            f"<th>{heatmap_display_label('path', row.get('label', '-'))}</th>"
            f"{''.join(cells_html)}"
            "</tr>"
        )
    caption_bits = []
    if epoch is not None:
        caption_bits.append(f"Epoch {epoch}")
    if train_step is not None:
        caption_bits.append(f"TrainStep {train_step}")
    caption = " / ".join(caption_bits)
    caption_html = f"<p class='sub' style='margin: 0 0 8px 0;'>{caption}</p>" if caption else ""
    return (
        "<div class='heatmap-card'>"
        f"<div class='chart-title'>{title}</div>"
        f"<div class='chart-subtitle'>{subtitle}</div>"
        f"{caption_html}"
        "<table class='heatmap-table'>"
        "<thead><tr><th></th>"
        f"{header_cells}"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )

def build_html(report):
    display_report, display_notes = prepare_display(report)
    report_json = json.dumps(sanitize_json(display_report), ensure_ascii=False).replace("</", "<\\/")
    heatmap_cards = []
    final_heatmaps = (report.get("rank") or {}).get("final_heatmaps") or {}
    for key, title, subtitle, fixed_max in [
        ("energy_share_per_param", "Final Energy Share Per Param Heatmap", "位置×役割ごとの正規化重み量シェア（合計1.0）", None),
        ("rank_sat_wmean", "Final RankSatWMean Heatmap", "位置×役割ごとのrank成分の広がり", 1.0),
    ]:
        heatmap = final_heatmaps.get(key)
        if heatmap and any(
            safe_float(cell.get("value")) is not None
            for row in heatmap.get("rows") or [] for cell in row.get("cells") or []
        ):
            heatmap_cards.append(render_rank_heatmap_card(title, subtitle, heatmap, fixed_max=fixed_max))
    heatmaps = "<div class='heatmap-grid'>" + "".join(heatmap_cards) + "</div>" if heatmap_cards else ""
    body_html = render_body(display_report, display_notes, heatmaps)

    template = (Path(__file__).parent / "lora_report_assets" / "diagnostic.html").read_text(encoding="utf-8")
    identity = "".join(f'<meta name="{k}" content="{escape(str(report["provenance"][k]), quote=True)}">' for k in ("run_id","session_id","generation_id"))
    substitutions = {"IDENTITY": identity, "BODY_HTML": body_html, "REPORT_JSON": report_json}
    return re.sub(r"@@(IDENTITY|BODY_HTML|REPORT_JSON)@@", lambda match: substitutions[match.group(1)], template)
