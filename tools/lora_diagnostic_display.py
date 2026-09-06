"""Presentation-only filtering and copy for the standalone diagnostic HTML."""

import math
import re
from html import escape


SECTION_NAMES = {
    "grad": "GradNorm", "dq": "DQ Delta", "dq_guard": "DQ Guard", "dq_direct": "DQ Direct", "rank": "Rank",
    "group_loss": "Group Loss", "lora_trend": "LoRA 情報密度エポック推移",
}
CONSTANT_CHARTS = {"rank_dim", "dq_rank_dim", "dq_bits", "dq_range_mul", "thresh_off"}
REFERENCE_NAMES = {
    "threshold", "activecliplow", "activecliphigh", "limit", "windowdecisionlimit",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def text(value):
    return escape(str(value), quote=True)


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
        ("最終 QuantErrRatioEMA", summary("dq").get("final_quant_err_ratio_ema"), dec, ""),
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
        constants = "".join(constant_summary(c) for c in charts if c.get("display_mode") == "constant")
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

    status_rows = []
    source_specs = [
        ("grad", "Grad", "path"), ("dq", "DQ", "path"), ("dq", "DQ Auto", "auto_path"),
        ("rank", "Rank", "path"), ("group_loss", "Group Loss step", "step_path"),
        ("group_loss", "Group Loss epoch", "epoch_path"), ("lora", "LoRA", "path"),
        ("lora_trend", "LoRA epoch推移", "model_path"),
    ]
    extra_sections = [k for k in (report.get("charts") or {}) if k not in {"grad", "dq", "rank", "group_loss", "lora_trend"}]
    source_specs += [(k, SECTION_NAMES.get(k, k.replace("_", " ").title()), "path") for k in extra_sections]
    for key, label, field in source_specs:
        data = report.get(key) or {}
        path = data.get(field)
        available = bool(path) if key in ("dq", "group_loss") else bool(data)
        state = "入力あり" if available else "解析結果なし"
        if key == "lora" and report.get("lora_error"):
            state = "確認事項あり"
        status_rows.append([text(label), text(state), f"<code>{text(path or '—')}</code>"])
    status_body = table(["対象", "状況", "入力ファイル"], status_rows)
    omissions = list(dict.fromkeys([n for ns in notes.values() for n in ns] + missing_checks))
    if omissions:
        status_body += "<h3>表示を省略した項目・欠測</h3><ul>" + "".join(f"<li>{text(n)}</li>" for n in omissions) + "</ul>"
    if errors:
        status_body += "<h3>解析の詳細</h3>" + "".join(f"<p>{text(e)}</p>" for e in errors)
    groups = [("grad", "Grad"), ("dq", "DQ"), ("rank", "Rank"),
              ("group_loss", "Group Loss"), ("lora", "LoRA"), ("lora_trend", "epoch推移")]
    groups += [(k, SECTION_NAMES.get(k, k.replace("_", " ").title())) for k in extra_sections]
    present = [name for key, name in groups if report.get(key)]
    absent = [name for key, name in groups if not report.get(key)]
    label = "入力データ・解析状況：" + ("・".join(present) + "あり" if present else "解析結果なし")
    if absent and present:
        label += "／" + "・".join(absent) + "結果なし"
    body += details(label, status_body, "panel input-status")
    return body
