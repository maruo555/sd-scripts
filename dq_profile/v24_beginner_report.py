from __future__ import annotations

"""Beginner-oriented, single-dataset report for SDXL DQ Profiler v2.4.

This renderer consumes the already-built practical report model.  It does not
change candidate selection, recompute measurements, or make a final image
quality/Utility claim.
"""

import html
import json
import math
from typing import Any, Mapping, Sequence

from dq_profile.v24_practical_report import (
    AFFINITY_FIXED_Y_MAX,
    TIMESTEP_BIN_LABELS,
    _candidate_role_matrix,
    _curve_svg,
    _dataset_behavior_table,
    _fmt,
    _mul_list,
    _optional_float,
    _pct,
)


BEGINNER_REPORT_SCHEMA_VERSION = "1.0.0-beta"

# Frozen, anonymized Standard reference cohort.  Only the sorted metric values
# are retained; dataset names and paths are deliberately excluded.  Positions
# are descriptive, not population thresholds or selector evidence.
STANDARD_REFERENCE_COHORT = {
    "version": "standard-reference-cohort-20260901-v1",
    "setting_count": 9,
    "values": {
        "natural_tail": [
            3.3559883169323634,
            3.4388637329410505,
            4.007832695302965,
            4.6616166026748855,
            4.849127871516622,
            5.962866214030089,
            8.220509251499442,
            10.79164303983912,
            10.971045906299404,
        ],
        "minimum_tail": [
            0.4904256093573765,
            0.7004591509672242,
            0.8989471320506843,
            0.9752044732797807,
            1.0035067600763246,
            1.0433186807996206,
            1.125773733891675,
            1.3901657740941091,
            1.4225208351550054,
        ],
        "mul_span": [
            0.4517422077313522,
            0.4925948024863251,
            0.5028121173149882,
            0.5735561549607451,
            0.59463050584933,
            0.6245126369995042,
            0.7469280230193298,
            0.9335902860600387,
            1.2798496383289673,
        ],
        "source_share": [
            0.36854811809072835,
            0.44635793735717644,
            0.526034223421293,
            0.5357525064874519,
            0.5766309459063204,
            0.5912834791099346,
            0.6297516827969767,
            0.7624387022883875,
            0.9327961821889018,
        ],
        "timestep_ratio": [
            1.982947155690137,
            2.292007648321076,
            3.5140447802518278,
            3.62263756327396,
            3.7228274519496978,
            3.8868696653826253,
            4.019653522317842,
            5.40684141363122,
            13.930538497326394,
        ],
    },
    "not_thresholds": True,
    "not_quality_or_utility": True,
}


_MUL_RESPONSE_JA = {
    "flat_within_descriptive_tolerance": "測定範囲ではほぼ平坦です",
    "upper_range_reduces_deformation": "mulを上げるほど変形が減る傾向です",
    "lower_range_reduces_deformation": "mulを下げるほど変形が減る傾向です",
    "interior_valley": "中間のmulに谷がある形です",
    "irregular_or_mixed": "単調ではなく、候補ごとに動きが混在します",
    "insufficient_grid": "測定点が少なく、曲線形状は判定できません",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _require_single_dataset(model: Mapping[str, Any]) -> Mapping[str, Any]:
    datasets = list(model.get("datasets") or [])
    if len(datasets) != 1:
        raise ValueError(
            "beginner report requires exactly one dataset; "
            f"received {len(datasets)}"
        )
    return datasets[0]


def _curve_response(dataset: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    vector = dict(dataset.get("dataset_character_vector") or {})
    channels = dict(vector.get("channels") or {})
    response = dict(channels.get("mul_response") or {})
    label = _MUL_RESPONSE_JA.get(
        str(response.get("label")),
        "曲線形状はまだ分類できません",
    )
    return label, response


def _summary_lines(dataset: Mapping[str, Any]) -> list[str]:
    cards = list(dataset.get("candidate_cards") or [])
    hard_safe = int(dataset.get("hard_safety_pass_count", 0))
    if hard_safe == len(cards) and cards:
        safety = (
            f"試した{len(cards)}候補はすべてHard-safetyを通過し、"
            "重大な数値異常は検出されませんでした。"
        )
    else:
        unsafe = [
            float(card["range_mul"])
            for card in cards
            if not _as_bool(card.get("hard_safety_pass"))
        ]
        safety = (
            "Hard-safetyを通過しなかった候補があります: "
            f"{_mul_list(unsafe)}。該当候補は通常のBody/Tail順位と分けて見ます。"
        )

    curve_label, _ = _curve_response(dataset)
    response = str(dataset.get("absolute_response_label") or "摂動帯は未分類")
    retained = list(dataset.get("fidelity_retained_muls") or [])
    behavior = (
        f"測定結果は「{response}」で、{curve_label}。"
        f"候補内比較で残ったmulは {_mul_list(retained)} です。"
    )

    selection_reason = str(dataset.get("representative_selection_reason") or "")
    if not selection_reason:
        selection_reason = (
            "単一の代表mulを出せるだけの証拠がないため、候補集合として表示します。"
        )
    return [safety, behavior, selection_reason]


def _body_tail_map_svg(cards: Sequence[Mapping[str, Any]]) -> str:
    width, height = 720, 430
    left, right, top, bottom = 76, 30, 34, 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    axis_max = AFFINITY_FIXED_Y_MAX

    def sx(value: float) -> float:
        return left + min(max(value, 0.0), axis_max) / axis_max * plot_w

    def sy(value: float) -> float:
        return top + plot_h - min(max(value, 0.0), axis_max) / axis_max * plot_h

    ticks: list[str] = []
    for value in (0.0, 1.0, 2.0, 3.0, 4.0):
        x = sx(value)
        y = sy(value)
        ticks.extend(
            [
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}" class="map-grid"/>',
                f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" class="map-grid"/>',
                f'<text x="{x:.1f}" y="{height-34}" text-anchor="middle" class="map-label">{value:.1f}</text>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="map-label">{value:.1f}</text>',
            ]
        )

    points: list[str] = []
    for index, card in enumerate(cards):
        body = _optional_float(card.get("body"))
        tail = _optional_float(card.get("tail"))
        if body is None or tail is None:
            continue
        x, y = sx(body), sy(tail)
        color = "#991b1b" if not _as_bool(card.get("hard_safety_pass")) else "#2563eb"
        overflow = body > axis_max or tail > axis_max
        label_y = y - 10 if index % 2 == 0 else y + 18
        points.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" '
            'stroke="white" stroke-width="2"/>'
            f'<text x="{x+9:.1f}" y="{label_y:.1f}" class="map-point">'
            f'{float(card["range_mul"]):.2f}{" ↑" if overflow else ""}</text>'
            f'<title>mul {float(card["range_mul"]):.2f}: Body {body:.3f}, Tail {tail:.3f}</title></g>'
        )

    reference_x, reference_y = sx(1.0), sy(1.0)
    return f"""
<svg class="body-tail-map" viewBox="0 0 {width} {height}" role="img" aria-label="BodyとTailの二次元マップ">
  <style>
    .map-grid{{stroke:#d9e1ec;stroke-width:1}}
    .map-reference{{stroke:#b91c1c;stroke-width:1.8;stroke-dasharray:7 5}}
    .map-label{{font:12px system-ui,sans-serif;fill:#64748b}}
    .map-point{{font:12px system-ui,sans-serif;fill:#26364c;font-weight:800}}
    .map-zone{{font:11px system-ui,sans-serif;fill:#65758c}}
  </style>
  <rect x="{left}" y="{reference_y:.1f}" width="{reference_x-left:.1f}" height="{top+plot_h-reference_y:.1f}" fill="#eaf7f3" opacity=".75"/>
  <rect x="{left}" y="{top}" width="{reference_x-left:.1f}" height="{reference_y-top:.1f}" fill="#fff4e2" opacity=".68"/>
  {''.join(ticks)}
  <line x1="{reference_x:.1f}" x2="{reference_x:.1f}" y1="{top}" y2="{top+plot_h}" class="map-reference"/>
  <line x1="{left}" x2="{left+plot_w}" y1="{reference_y:.1f}" y2="{reference_y:.1f}" class="map-reference"/>
  <text x="{left+12}" y="{top+18}" class="map-zone">Bodyは穏やか／一部条件のTailが大きい</text>
  <text x="{left+12}" y="{top+plot_h-12}" class="map-zone">今回の測定では比較的no_quantに近い</text>
  {''.join(points)}
  <text x="{left+plot_w/2}" y="{height-5}" text-anchor="middle" class="map-label">Body：左ほどno_quantに近い／右ほど変形が強い</text>
  <text x="17" y="{top+plot_h/2}" transform="rotate(-90 17 {top+plot_h/2})" text-anchor="middle" class="map-label">Tail：下ほどno_quantに近い／上ほど難しい帯が強い</text>
</svg>
"""


def _percentile_position(value: float | None, reference: Sequence[float]) -> float | None:
    if value is None or not reference:
        return None
    less = sum(item < value for item in reference)
    equal = sum(math.isclose(item, value, rel_tol=1e-9, abs_tol=1e-12) for item in reference)
    return 100.0 * (less + 0.5 * equal) / len(reference)


def _timestep_ratio(dataset: Mapping[str, Any]) -> float | None:
    baseline = dict(dataset.get("no_quant_baseline_profile") or {})
    values = [
        value
        for row in baseline.get("timestep_rows", [])
        if (value := _optional_float(row.get("grad_norm_rms"))) is not None and value > 0.0
    ]
    if len(values) < 2:
        return None
    return max(values) / min(values)


def _character_metrics(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    vector = dict(dataset.get("dataset_character_vector") or {})
    channels = dict(vector.get("channels") or {})
    acceptance = dict(channels.get("absolute_acceptance") or {})
    response = dict(channels.get("mul_response") or {})
    localization = dict(channels.get("source_localization") or {})
    stability = dict(channels.get("no_quant_stability") or {})
    natural = dict(stability.get("natural_variation") or {})
    values = STANDARD_REFERENCE_COHORT["values"]
    return [
        {
            "key": "natural_tail",
            "label": "no-quant自然変動",
            "value": _optional_float(natural.get("local_tail")),
            "format": "number",
            "meaning": "量子化しなくても短いprobe間でどれだけ勾配が揺れるか。高いほど元の学習信号自体の幅が大きい。",
            "direction": "小さい＝信号が均一 ／ 大きい＝元々揺れやすい",
            "reference": values["natural_tail"],
        },
        {
            "key": "minimum_tail",
            "label": "量子化変形（最小Tail）",
            "value": _optional_float(acceptance.get("minimum_tail")),
            "format": "number",
            "meaning": "試したmulのうち最も小さかったTail。低いほど今回の局所probeではno_quantへ近い。",
            "direction": "小さい＝no_quantに近い ／ 大きい＝最良mulでもTailが残る",
            "reference": values["minimum_tail"],
        },
        {
            "key": "mul_span",
            "label": "mul感度",
            "value": _optional_float(response.get("span")),
            "format": "number",
            "meaning": "mulを変えたときのBody/Tail曲線の振れ幅。高いほどmul選択で動きが変わりやすい。",
            "direction": "小さい＝mulに頑健 ／ 大きい＝mul選びが重要",
            "reference": values["mul_span"],
        },
        {
            "key": "source_share",
            "label": "source集中",
            "value": _optional_float(localization.get("top_source_share")),
            "format": "percent",
            "meaning": "Tail側の負担が最大sourceへ集まる割合。高くても絶対Tailが低ければ直ちに危険とは限らない。",
            "direction": "小さい＝複数sourceへ分散 ／ 大きい＝一部sourceへ集中",
            "reference": values["source_share"],
        },
        {
            "key": "timestep_ratio",
            "label": "timestep偏り",
            "value": _timestep_ratio(dataset),
            "format": "ratio",
            "meaning": "no_quant勾配RMSの最大bin÷最小bin。高いほどtimestep帯による信号差が大きい。",
            "direction": "小さい＝timestep間で均一 ／ 大きい＝特定帯へ偏る",
            "reference": values["timestep_ratio"],
        },
    ]


def _character_profile_html(dataset: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for metric in _character_metrics(dataset):
        value = metric["value"]
        position = _percentile_position(value, metric["reference"])
        if value is None or position is None:
            value_text = "未測定"
            marker = ""
            position_text = "参照位置なし"
        else:
            if metric["format"] == "percent":
                value_text = f"{100.0 * value:.1f}%"
            elif metric["format"] == "ratio":
                value_text = f"{value:.2f}×"
            else:
                value_text = f"{value:.3f}"
            marker = f'<span class="reference-dot" style="left:{position:.2f}%"></span>'
            position_text = f"参照集団内 {position:.0f} / 100位置"
        rows.append(
            f"""
<div class="character-row">
  <div class="character-label"><strong>{html.escape(metric['label'])}</strong><span>{html.escape(metric['meaning'])}</span><span class="direction-note">{html.escape(metric['direction'])}</span></div>
  <div class="character-visual">
    <div class="reference-track"><span class="reference-median"></span>{marker}</div>
    <div class="reference-axis"><span>参照内で小さい</span><span>中央値</span><span>参照内で大きい</span></div>
  </div>
  <div class="character-value"><strong>{html.escape(value_text)}</strong><span>{html.escape(position_text)}</span></div>
</div>
"""
        )
    return "".join(rows)


def _source_focus_html(dataset: Mapping[str, Any]) -> str:
    localization = dict(dataset.get("source_localization") or {})
    profile = dict(localization.get("reference_profile") or {})
    if not profile:
        return '<p class="muted">source集中の追加集計はありません。</p>'
    share = _optional_float(profile.get("top_source_share"))
    loo = _optional_float(profile.get("loo_max_tail_reduction_fraction"))
    share_pct = 100.0 * share if share is not None else 0.0
    loo_pct = 100.0 * loo if loo is not None else 0.0
    return f"""
<div class="source-pair">
  <div>
    <div class="bar-title"><strong>最大sourceが占めるTail超過</strong><span>{_pct(share)}</span></div>
    <div class="simple-bar"><span style="width:{min(max(share_pct, 0.0), 100.0):.2f}%"></span></div>
    <p>最大負担source（{html.escape(str(profile.get('top_source_alias') or '—'))}）がTail超過全体に占める割合です。<span class="direction-note">小さい＝複数sourceへ分散 ／ 大きい＝一部sourceへ集中</span></p>
  </div>
  <div>
    <div class="bar-title"><strong>1 sourceを計算上外したときの最大Tail低下</strong><span>{_pct(loo)}</span></div>
    <div class="simple-bar impact"><span style="width:{min(max(loo_pct, 0.0), 100.0):.2f}%"></span></div>
    <p>最も影響の大きいsource（{html.escape(str(profile.get('loo_most_actionable_source_alias') or '—'))}）を外して再集計した参考値です。<span class="direction-note">小さい＝1 sourceへの依存が弱い ／ 大きい＝特定sourceの影響が大きい</span></p>
  </div>
</div>
<p class="micro">基準mul {_fmt(profile.get('range_mul'), 2)}、実効source数 {_fmt(profile.get('effective_source_count'), 2)} / {int(profile.get('source_group_count') or 0)}。source aliasは画像内容を出さない匿名記号です。高い＝画質不良やsource削除推奨ではありません。</p>
"""


def _heat_color(value: float | None) -> str:
    if value is None:
        return "background:#f1f5f9;color:#64748b"
    if value < 1.0:
        alpha = 0.12 + 0.30 * min(value, 1.0)
        return f"background:rgba(37,99,235,{alpha:.3f});color:#123a78"
    alpha = 0.16 + 0.45 * min((value - 1.0) / 2.0, 1.0)
    return f"background:rgba(217,119,6,{alpha:.3f});color:#6f3600"


def _timestep_heatmap(dataset: Mapping[str, Any]) -> str:
    rows = list(dataset.get("timestep_rows") or [])
    if not rows:
        return '<p class="muted">timestep別の集計はありません。</p>'
    bins = sorted({int(row["timestep_bin"]) for row in rows})
    by_mul: dict[float, dict[int, float | None]] = {}
    for row in rows:
        mul = float(row["range_mul"])
        by_mul.setdefault(mul, {})[int(row["timestep_bin"])] = _optional_float(
            row.get("source_balanced_q95_relative_distance")
        )
    headers = "".join(
        f"<th>bin {index}<span>{html.escape(TIMESTEP_BIN_LABELS.get(index, ('?', '?'))[0])}</span></th>"
        for index in bins
    )
    body: list[str] = []
    for mul in sorted(by_mul):
        cells = "".join(
            f'<td style="{_heat_color(by_mul[mul].get(index))}"><strong>{_fmt(by_mul[mul].get(index))}</strong></td>'
            for index in bins
        )
        body.append(f'<tr><th scope="row">{mul:.2f}</th>{cells}</tr>')
    return (
        '<div class="table-wrap"><table class="heatmap"><thead><tr><th>mul</th>'
        f'{headers}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _uncertainty_table(dataset: Mapping[str, Any]) -> str:
    rows = []
    for card in dataset.get("candidate_cards", []):
        rows.append(
            "<tr>"
            f'<th scope="row">{float(card["range_mul"]):.2f}</th>'
            f'<td>{_fmt(card.get("body"))}<span>CI [{_fmt(card.get("body_ci_low"))}, {_fmt(card.get("body_ci_high"))}]</span></td>'
            f'<td>{_fmt(card.get("tail"))}<span>CI [{_fmt(card.get("tail_ci_low"))}, {_fmt(card.get("tail_ci_high"))}]</span></td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="uncertainty-table"><thead><tr>'
        '<th>mul</th><th>Body</th><th>Tail</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_beginner_report(model: Mapping[str, Any]) -> str:
    """Render a self-contained beginner-oriented report for one dataset."""

    dataset = _require_single_dataset(model)
    cards = list(dataset.get("candidate_cards") or [])
    if not cards:
        raise ValueError("beginner report requires candidate cards")
    summary_items = "".join(
        f'<li><span>{index}</span><p>{html.escape(line)}</p></li>'
        for index, line in enumerate(_summary_lines(dataset), start=1)
    )
    measurement = dict(dataset.get("measurement_quality") or {})
    confidence = dict(dataset.get("local_comparison_confidence") or {})
    maturity = dict(dataset.get("recommendation_maturity") or {})
    image_probed = int(dataset.get("image_count_probed", dataset.get("image_count", 0)))
    image_total = int(dataset.get("image_count_total", dataset.get("image_count", 0)))
    source_probed = int(dataset.get("source_group_count_probed", dataset.get("source_group_count", 0)))
    source_total = int(dataset.get("source_group_count_total", dataset.get("source_group_count", 0)))
    curve = _curve_svg(
        cards,
        fixed_y_max=AFFINITY_FIXED_Y_MAX,
        edge_direction=str(dataset.get("edge_direction") or "resolved"),
    )
    payload = html.escape(
        json.dumps(
            {
                "schema_version": BEGINNER_REPORT_SCHEMA_VERSION,
                "source_schema_version": model.get("schema_version"),
                "dataset_id": dataset.get("dataset_id"),
                "reference_cohort": STANDARD_REFERENCE_COHORT["version"],
                "not_quality_or_utility": True,
            },
            ensure_ascii=False,
        )
    )
    qa_reasons = "".join(
        f"<li>{html.escape(str(reason))}</li>"
        for reason in measurement.get("reasons", [])
    ) or "<li>追加理由なし</li>"
    confidence_reasons = "".join(
        f"<li>{html.escape(str(reason))}</li>"
        for reason in confidence.get("reasons", [])
    ) or "<li>追加理由なし</li>"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDXL DQ Dataset Profiler — 診断レポート（概要）</title>
<style>
:root{{--ink:#172033;--muted:#607086;--line:#d9e1ec;--paper:#f4f7fb;--card:#fff;--blue:#2563eb;--orange:#d97706;--purple:#7c3aed;--teal:#0f766e;--red:#b91c1c;--shadow:0 12px 32px rgba(25,38,70,.09)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Yu Gothic UI",sans-serif}}
header{{background:#13213e;color:white}}.header-inner{{max-width:1180px;margin:auto;padding:13px 22px;display:flex;justify-content:space-between;gap:18px;align-items:center}}.brand{{font-weight:850}}.brand small{{display:block;color:#c8d4ea;font-weight:500}}.header-links{{display:flex;gap:8px;flex-wrap:wrap}}.header-links a{{color:white;text-decoration:none;border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:5px 10px;font-size:12px}}
main{{max-width:1180px;margin:auto;padding:18px 22px 70px}}section{{margin:0 0 24px}}h1{{font-size:clamp(28px,4.5vw,50px);line-height:1.08;margin:5px 0 10px}}h2{{font-size:24px;margin:0 0 6px}}h3{{font-size:17px;margin:0 0 6px}}p{{margin:5px 0}}.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:850;color:#5d6d84}}.section-help,.muted,.micro{{color:var(--muted)}}.micro{{font-size:11px}}
.hero{{background:linear-gradient(135deg,#fff,#eef4ff);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:var(--shadow)}}.hero-top{{display:flex;justify-content:space-between;gap:18px;align-items:start}}.scope-badge{{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;border-radius:999px;padding:6px 11px;font-size:12px;font-weight:800;white-space:nowrap}}.chart-card{{margin-top:14px;background:white;border:1px solid var(--line);border-radius:15px;padding:10px 16px 13px}}.chart-card .chart{{width:100%;height:340px;display:block}}.chart-explain{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}}.chart-explain>div{{background:#f5f7fb;border-radius:9px;padding:9px 10px;font-size:12px}}.chart-explain strong{{display:block}}
.quick-summary{{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(280px,.65fr);gap:14px}}.summary-card,.card{{background:white;border:1px solid var(--line);border-radius:15px;padding:18px;box-shadow:0 5px 18px rgba(25,38,70,.04)}}.summary-list{{list-style:none;padding:0;margin:8px 0;display:grid;gap:10px}}.summary-list li{{display:grid;grid-template-columns:30px 1fr;gap:10px;align-items:start}}.summary-list li>span{{display:grid;place-items:center;width:28px;height:28px;border-radius:50%;background:#dbeafe;color:#17499f;font-weight:900}}.summary-list p{{margin:1px 0}}.qa-kpis{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.qa-kpis>div{{background:#f5f7fb;padding:10px;border-radius:9px}}.qa-kpis span{{display:block;font-size:11px;color:var(--muted)}}.qa-kpis strong{{font-size:18px}}
.warning{{background:#fff4e2;border-left:5px solid var(--orange);padding:13px 15px;border-radius:8px}}.section-title{{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:8px}}.section-title p{{max-width:75ch}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:10px 9px;text-align:left;vertical-align:top}}thead th{{background:#edf2f8;white-space:nowrap}}.mark-cell{{text-align:center!important;vertical-align:middle}}.matrix-mark{{display:inline-grid;place-items:center;min-width:24px;height:24px;padding:0 6px;border-radius:999px;font-weight:900;font-size:12px}}.matrix-mark.pass{{background:#dcfce7;color:#166534}}.matrix-mark.retained{{background:#dbeafe;color:#17499f}}.matrix-mark.attention{{background:#ffedd5;color:#9a4700}}.matrix-mark.danger{{background:#fee2e2;color:#991b1b}}.matrix-mark.representative{{background:#f2ebff;color:#5b21b6}}.matrix-mark.pattern{{background:#e0e7ff;color:#3730a3}}.matrix-mark.off{{color:#a2acba;background:#f3f4f6}}.matrix-legend{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-top:9px;color:var(--muted);font-size:11px}}.matrix-legend>span{{display:inline-flex;align-items:center;gap:5px}}.decision-table th[scope="row"]{{min-width:190px}}.decision-table td:last-child{{min-width:180px}}.role-matrix th:first-child{{min-width:245px}}.role-matrix td{{text-align:center;vertical-align:middle}}.row-help{{display:block;color:var(--muted);font-weight:400;font-size:10px;margin-top:2px}}
.two-column{{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}}.body-tail-map{{width:100%;height:auto;display:block}}.map-notes{{display:grid;gap:9px}}.map-notes>div{{padding:11px;border-radius:9px;background:#f5f7fb}}.map-notes strong{{display:block}}
.character-panel{{background:white;border:1px solid var(--line);border-radius:15px;overflow:hidden}}.character-row{{display:grid;grid-template-columns:minmax(210px,.8fr) minmax(260px,1.45fr) minmax(130px,.45fr);gap:15px;align-items:center;padding:15px 17px;border-bottom:1px solid var(--line)}}.character-row:last-child{{border-bottom:0}}.character-label{{display:grid;gap:2px}}.character-label span,.character-value span{{font-size:11px;color:var(--muted)}}.direction-note{{display:block;color:#315b7d!important;font-weight:750;margin-top:3px}}.character-visual{{min-width:0}}.reference-track{{position:relative;height:13px;border-radius:999px;background:linear-gradient(90deg,#dbeafe,#eef2ff 50%,#ffedd5)}}.reference-median{{position:absolute;left:50%;top:-4px;width:2px;height:21px;background:#64748b}}.reference-dot{{position:absolute;top:50%;width:17px;height:17px;border-radius:50%;background:var(--purple);border:3px solid white;box-shadow:0 0 0 1px #6d28d9;transform:translate(-50%,-50%)}}.reference-axis{{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:4px}}.character-value{{display:grid;text-align:right}}.character-value strong{{font-size:19px}}
.source-pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.source-pair>div{{padding:14px;background:#f8fafc;border:1px solid var(--line);border-radius:11px}}.bar-title{{display:flex;justify-content:space-between;gap:8px}}.simple-bar{{height:13px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin:8px 0}}.simple-bar span{{display:block;height:100%;background:var(--purple);border-radius:999px}}.simple-bar.impact span{{background:var(--teal)}}.source-pair p{{font-size:12px;color:var(--muted)}}
.heatmap th span{{display:block;font-size:10px;color:var(--muted);font-weight:500}}.heatmap td{{text-align:center;min-width:100px}}.uncertainty-table td>span{{display:block;font-size:10px;color:var(--muted)}}details{{background:white;border:1px solid var(--line);border-radius:12px;padding:12px 15px;margin:10px 0}}summary{{cursor:pointer;font-weight:800}}details>div,details>table,details>ul{{margin-top:10px}}.limits{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.limits>div{{padding:16px;border-radius:12px}}.limits .can{{background:#ecfdf5;border:1px solid #a7f3d0}}.limits .cannot{{background:#fff7ed;border:1px solid #fed7aa}}.limits ul{{margin:5px 0;padding-left:20px}}
footer{{font-size:11px;color:var(--muted);padding-top:22px}}code{{font-family:ui-monospace,Consolas,monospace}}a{{color:#1d4ed8}}
@media(max-width:850px){{.quick-summary,.two-column,.source-pair,.limits{{grid-template-columns:1fr}}.chart-explain{{grid-template-columns:1fr}}.character-row{{grid-template-columns:1fr}}.character-value{{text-align:left}}.hero-top{{display:block}}.scope-badge{{display:inline-block;margin-top:8px}}}}
@media(max-width:560px){{main{{padding:12px 9px 50px}}.header-inner{{align-items:flex-start;flex-direction:column}}.hero{{padding:14px}}.chart-card{{padding:5px}}.chart-card .chart{{height:260px}}.qa-kpis{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}header{{background:white;color:var(--ink);border-bottom:1px solid var(--line)}}.brand small{{color:var(--muted)}}.header-links{{display:none}}.card,.summary-card,.hero{{box-shadow:none}}}}
</style>
</head>
<body>
<header><div class="header-inner"><div class="brand">SDXL DQ Dataset Profiler <small>診断レポート（概要）・ Safety/Fidelity ≠ 最終画質Utility</small></div><nav class="header-links"><a href="report.html">詳細レポート</a><a href="technical_report.html">技術レポート</a></nav></div></header>
<main>
<section class="hero">
  <div class="hero-top"><div><div class="eyebrow">Single dataset / fixed-mul local diagnostic</div><h1>{html.escape(str(dataset.get('label') or dataset.get('dataset_id')))}</h1><p>固定mulごとに、量子化した勾配がno_quantからどのように離れるかを可視化します。</p></div><span class="scope-badge">{html.escape(str(dataset.get('execution_mode') or 'unknown').title())} / Local-only</span></div>
  <div class="chart-card">
    {curve}
    <div class="chart-explain">
      <div><strong>Body</strong>全画像・timestepをまとめた、やや厳しめの代表値です。小さいほど幅広い条件でno_quantに近くなります。</div>
      <div><strong>Tail</strong>各timestep帯の代表値のうち最大です。小さいほど厳しい帯でもno_quantに近くなります。</div>
      <div><strong>ヒゲ（淡い縦棒）</strong>画像/sourceを再標本化したときの不確かさです。長いほど細かな順位を決めにくい状態です。</div>
    </div>
  </div>
</section>

<section class="quick-summary">
  <div class="summary-card"><div class="eyebrow">まず読む3行</div><ol class="summary-list">{summary_items}</ol></div>
  <aside class="summary-card"><div class="eyebrow">測定範囲</div><div class="qa-kpis"><div><span>Measurement QA</span><strong>{html.escape(str(measurement.get('level') or '—'))}</strong></div><div><span>比較confidence</span><strong>{html.escape(str(confidence.get('level') or '—'))}</strong></div><div><span>画像</span><strong>{image_probed} / {image_total}</strong></div><div><span>source</span><strong>{source_probed} / {source_total}</strong></div></div><p class="micro">QAは測定手順の正常性、confidenceは候補間比較の確かさです。画質評価ではありません。</p></aside>
</section>

<section class="warning"><strong>大切な読み方:</strong> 基準1.0未満は、今回測った勾配変形が基準帯の内側という意味です。良い画像になる保証でも、量子化なしと同一という意味でもありません。</section>

<section class="card">
  <div class="section-title"><div><h2>試したmulと役割</h2><p class="section-help">✓は通過・保持、橙の「注意」は安全だが相対的に強い摂動、★は数値上の代表です。記号は画質の合否ではありません。</p></div></div>
  {_candidate_role_matrix(dataset)}
</section>

<section class="card">
  <div class="section-title"><div><h2>今回見られた動き</h2><p class="section-help">定型パターンへ○を付けています。「候補内で強い」は同じdataset内の相対比較で、絶対的な危険とは限りません。</p></div></div>
  {_dataset_behavior_table(dataset)}
</section>

<section class="two-column">
  <div class="card"><h2>Body × Tail マップ</h2><p class="section-help">右ほど普段の変形が大きく、上ほど難しい条件の変形が大きい配置です。左下へ近いほど数値上no_quantへ近いだけで、最終画質順位ではありません。</p>{_body_tail_map_svg(cards)}</div>
  <aside class="card map-notes"><h2>4つの見方</h2><div><strong>左下</strong>今回の局所probeではBody/Tailとも比較的穏やかです。</div><div><strong>左上</strong>普段は穏やかでも、一部の難しい条件で大きくずれます。</div><div><strong>右上</strong>普段と難しい条件の両方で変形が大きく見えます。</div><div><strong>基準線付近</strong>不確かさのヒゲも確認し、1.0を少し跨ぐだけで断定しません。</div></aside>
</section>

<section>
  <div class="section-title"><div><h2>データセットの性格カルテ</h2><p class="section-help">5項目を合成せず、別々に表示します。紫の点は匿名化したStandard参照9設定内での位置です。中央線は参照集団の中央値で、理想値や良否の閾値ではありません。</p></div><span class="scope-badge">参照: {html.escape(str(STANDARD_REFERENCE_COHORT['version']))}</span></div>
  <div class="character-panel">{_character_profile_html(dataset)}</div>
</section>

<section class="card">
  <h2>Tailはどの画像グループに偏っているか</h2><p class="section-help">sourceはTOML内の画像グループ（通常はimage_dir／subset）です。Tail基準を超えた変形をsource別の箱に分け、左で最大の箱の割合、右で1箱を計算上外したときのTail低下を見ます。</p>
  {_source_focus_html(dataset)}
</section>

<section class="card">
  <h2>timestep帯ごとの変形</h2><p class="section-help">行はmul、列はtimestep帯（右ほど高ノイズ）です。値が小さいほどno_quantに近く、大きいほどその帯で量子化による変形が強くなります。青は1.0未満、橙は1.0以上の注意帯です。特定の列だけが高ければ影響がその帯へ集中しています。画質判定ではありません。</p>
  {_timestep_heatmap(dataset)}
</section>

<section class="card">
  <h2>測定のばらつきと信頼性</h2><p class="section-help">Body／Tailは上のグラフと同じ値です。CIはsourceを取り直して再集計した推定95%区間で、狭いほど安定し、広いほど細かな候補順位は不確かです。QA通過は測定が有効という意味で、画質保証ではありません。</p>
  {_uncertainty_table(dataset)}
  <details><summary>QAとconfidenceの理由</summary><div class="two-column"><div><h3>Measurement QA</h3><ul>{qa_reasons}</ul></div><div><h3>Local comparison confidence</h3><ul>{confidence_reasons}</ul></div></div><p class="micro">Recommendation maturity: {html.escape(str(maturity.get('level') or '—'))}。Trajectoryと40 epoch画質Utilityは、この概要の選択材料に含めていません。</p></details>
</section>

<section class="limits">
  <div class="can"><h2>この診断で分かること</h2><ul><li>固定mulごとの局所的な勾配変形</li><li>BodyとTailの違い</li><li>候補内で残るmul集合</li><li>source・timestepへの偏り</li><li>測定の不確かさと未解決edge</li></ul></div>
  <div class="cannot"><h2>この診断だけでは分からないこと</h2><ul><li>最終生成画質のbest mul</li><li>量子化がno_quantより有益か</li><li>auto presetの最終軌跡</li><li>測定範囲外のmulの挙動</li><li>長期学習の成功保証</li></ul></div>
</section>

<section class="card"><h2>さらに詳しく見る</h2><p><a href="report.html">詳細レポート</a>では候補表、bootstrap勝率、source LOOなどを確認できます。<a href="technical_report.html">技術レポート</a>には計測契約と技術的成果物への案内があります。</p></section>
<footer>Beginner report schema {BEGINNER_REPORT_SCHEMA_VERSION} ・ source practical schema {html.escape(str(model.get('schema_version') or 'unknown'))} ・ fixed affinity scale 0–{AFFINITY_FIXED_Y_MAX:.1f} ・ no external CDN ・ numerical Safety/Fidelity only.</footer>
</main>
<script id="report-meta" type="application/json">{payload}</script>
</body>
</html>
"""
