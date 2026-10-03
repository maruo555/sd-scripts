"""Compact, display-only image rows from the saved gradient-support namespace."""
from __future__ import annotations

import html
import math
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0 or 10 ** -digits <= abs(value) < 1000:
        return f"{value:.{digits}f}"
    return f"{value:.3g}"


def _available(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "available" and all(
        (value := _number(row.get(key))) is not None and value >= 0
        for key in ("relative_gradient_distance", "grad_norm_noquant", "grad_diff_norm")
    )


def _row_html(row: Mapping[str, Any], alias: str, maximum: float) -> str:
    key = str(row["image_key"])
    filename = key.replace("\\", "/").rsplit("/", 1)[-1]
    label = (f'<span class="ig-image"><span>{alias}</span>'
             f'<span class="ig-filename">{html.escape(filename)}</span></span>')
    count = f'観測 {row.get("observation_count", 0)} / 対応集合 {row.get("expected_observation_count", 0)}'
    if _available(row):
        original, difference, distance = (float(row[field]) for field in (
            "grad_norm_noquant", "grad_diff_norm", "relative_gradient_distance"))
        bars = (f'<span class="ig-pair ig-bars" aria-hidden="true">'
                f'<span class="ig-track"><span class="ig-bar ig-original" style="width:{original / maximum * 100:.9g}%"></span></span>'
                f'<span class="ig-track"><span class="ig-bar ig-difference" style="width:{difference / maximum * 100:.9g}%"></span></span></span>')
        values = f'<span class="ig-pair ig-values" aria-hidden="true"><span>{_fmt(original)}</span><span>{_fmt(difference)}</span></span>'
        d_html = f'<span class="ig-d" aria-hidden="true">{_fmt(distance, 2)}</span>'
        description = f'{alias} {filename}：元勾配 {_fmt(original)}、絶対差 {_fmt(difference)}、最大d {_fmt(distance, 2)}。測定条件を開閉'
        measurement = row.get("measurement") or {}
        conditions = ' / '.join(f'{name} {measurement.get(field, "—")}' for name, field in (
            ("ノイズ帯bin", "timestep_bin"), ("noise_replica", "noise_replica"), ("quant_repeat", "quant_repeat")))
        detail = (f'{html.escape(conditions)}（保存ログの番号） · {count}<br>'
                  f'元勾配 {original:.9g} / 絶対差 {difference:.9g} / d {distance:.9g}<br>'
                  '3つの値はこの同じ測定に対応します。')
    else:
        bars = '<span class="ig-missing">未算出</span>'
        values, d_html = '<span class="ig-values">—</span>', '<span class="ig-d">—</span>'
        description = f'{alias} {filename}：未算出。理由を開閉'
        reasons = row.get("reasons") or ["missing_or_invalid_saved_image_breakdown"]
        detail = f'{count}<br>測定の欠落・不整合などにより最大dを確認できません。<br>' + html.escape(' / '.join(map(str, reasons)))
    return (f'<details class="ig-row"><summary class="ig-columns" aria-label="{html.escape(description, quote=True)}" title="{html.escape(key, quote=True)}">'
            f'{label}{bars}{values}{d_html}</summary><div class="ig-detail">'
            f'<span class="ig-key">{html.escape(key)}</span><br>{detail}</div></details>')


def render_image_breakdown(dataset: Mapping[str, Any]) -> str:
    cards = sorted(dataset.get("candidate_cards") or [], key=lambda card: float(card["range_mul"]))
    if not cards:
        return ""
    breakdowns = [(card.get("gradient_curve_support") or {}).get("image_breakdown") or {} for card in cards]
    if not any(breakdowns):
        return ('<section class="image-gradient-breakdown" data-selector-input="false"><h3>画像ごとのdと、その内訳</h3>'
                '<p>未生成：同じrunのsample CSVからレポートを再生成すると表示できます。</p></section>')
    keys = sorted({str(row["image_key"]) for breakdown in breakdowns for row in breakdown.get("images", [])})
    aliases = {key: f"画像 {index:02d}" for index, key in enumerate(keys, 1)}
    # One scale across images AND Mul panels: changing Mul cannot rescale bars.
    maximum = max((float(row[field]) for breakdown in breakdowns for row in breakdown.get("images", [])
                   if _available(row) for field in ("grad_norm_noquant", "grad_diff_norm")), default=1.) or 1.
    reference = next((float(dataset[field]) for field in (
        "single_representative_mul", "body_representative_mul", "preset_nearest_reference_mul")
        if _number(dataset.get(field)) is not None), None)
    initial = next((i for i, card in enumerate(cards) if reference is not None
                    and math.isclose(float(card["range_mul"]), reference, rel_tol=0, abs_tol=1e-9)), 0)
    options, panels = [], []
    for index, (card, breakdown) in enumerate(zip(cards, breakdowns)):
        unsafe = card.get("hard_safety_pass") is False
        state = " · HardSafety不合格" if unsafe else ""
        options.append(f'<option value="{index}"{" selected" if index == initial else ""}>Mul {float(card["range_mul"]):g}{state}</option>')
        rows = {str(row["image_key"]): row for row in breakdown.get("images", [])}
        available = sum(_available(row) for row in rows.values())
        caption = f'Mul {float(card["range_mul"]):g}{state} · 算出 {available} / {len(keys)}画像'
        row_html = ''.join(_row_html(rows.get(key, {"image_key": key, "reasons": ["no_image_sample_rows"]}), aliases[key], maximum) for key in keys)
        if not keys:
            reasons = (card.get("gradient_curve_support") or {}).get("d_reasons") or ["no_sample_rows"]
            row_html = '<p>画像別の保存測定がありません。' + html.escape(' / '.join(map(str, reasons))) + '</p>'
        panels.append(f'<div class="ig-panel" data-panel="{index}"{" hidden" if index != initial else ""}>'
                      f'<p class="ig-caption">{html.escape(caption)}</p>{row_html}</div>')
    ticks = ''.join(f'<span style="left:{q * 100:g}%">{maximum * q:.3g}</span>' for q in (0, .25, .5, .75, 1))
    total = _number(dataset.get("image_count_total"))
    coverage = f'保存ログの測定画像 {len(keys)}枚'
    if total is not None and total >= len(keys):
        coverage += f' / 全 {int(total)}枚'
    return f'''<section class="image-gradient-breakdown" data-selector-input="false" data-scale-max="{maximum:.17g}">
{_STYLE}
<div class="ig-heading"><h3>画像ごとのdと、その内訳</h3><label class="ig-picker" hidden>表示するMul <select aria-label="画像別の表示Mul">{''.join(options)}</select></label></div>
<p class="ig-description">warmup後の固定モデルで、各画像の診断条件中の最大dを表示します。学習step中の最大値ではありません。</p>
<p class="ig-coverage">{html.escape(coverage)} · 全測定画像を表示。未測定の画像は含みません。</p>
<div class="ig-legend"><span><i class="ig-original" aria-hidden="true"></i>上：元勾配</span><span><i class="ig-difference" aria-hidden="true"></i>下：絶対差</span><span>全画像・全Mulで共通の目盛り</span></div>
<div class="ig-columns ig-list-heading" aria-hidden="true"><span>画像</span><div><span>大きさ</span><div class="ig-scale">{ticks}</div></div><span class="ig-right">保存値</span><span class="ig-right">最大d</span></div>
{''.join(panels)}
<p class="ig-footer">画像キー順 · 行を開くと画像名・測定条件・未算出理由を表示</p>
<details class="ig-method"><summary>集計方法・値の意味</summary>
<p>選択したMulの各画像について、保存されたノイズ帯・ノイズ反復・量子化反復の中でdが最大だった1回を選びます。同率なら保存されたbin・noise_replica・quant_repeatの番号が小さい測定を選びます。元勾配と絶対差も、その同じ測定の値です。別々の最大値や平均値を組み合わせません。</p>
<p>元勾配は量子化OFFの勾配ベクトルの大きさ、絶対差はONとOFFの勾配ベクトルの差の大きさです。dは絶対差を元勾配で割った相対的な値です。二つの勾配の大きさを引いた値ではありません。</p>
<p>最大dはその画像の典型的な反応とは限らず、上のTail（帯別P95の最大）とも異なります。画質の合否や画像削除の必要性を示しません。欠落・不整合がある場合は別の測定で埋めず未算出にします。</p>
</details>
{_SCRIPT}
</section>'''


_STYLE = '''<style>
.image-gradient-breakdown{margin:18px 0 12px;border:1px solid #d6e0ef;border-radius:12px;padding:16px;background:#fff;color:#172338;min-width:0}
.image-gradient-breakdown .ig-heading{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.image-gradient-breakdown h3{font-size:16px;margin:0;color:#172338}
.image-gradient-breakdown .ig-picker{font-size:12px;display:flex;align-items:center;gap:8px;max-width:100%;flex-wrap:wrap}
.image-gradient-breakdown .ig-picker[hidden],.image-gradient-breakdown .ig-panel[hidden]{display:none}
.image-gradient-breakdown select{max-width:100%;font:inherit;color:#172338;background:#fff;border:1px solid #c8d5e8;border-radius:6px;padding:7px}
.image-gradient-breakdown p{font-size:12px;color:#536174;margin:7px 0;overflow-wrap:anywhere}
.image-gradient-breakdown .ig-legend{display:flex;gap:7px 16px;flex-wrap:wrap;font-size:12px;margin:13px 0;color:#536174}
.image-gradient-breakdown .ig-legend>span{display:inline-flex;align-items:center;gap:6px}
.image-gradient-breakdown .ig-legend i{display:inline-block;width:18px;height:6px;border-radius:2px}
.image-gradient-breakdown .ig-original{background:#3467d9}.image-gradient-breakdown .ig-difference{background:#d77b16}
.image-gradient-breakdown .ig-columns{display:grid;grid-template-columns:minmax(95px,20%) minmax(0,1fr) 64px 60px;gap:10px;align-items:center}
.image-gradient-breakdown .ig-list-heading{font-size:11px;color:#536174;padding:0 5px 7px;border-bottom:1px solid #d6e0ef}
.image-gradient-breakdown .ig-right{text-align:right}
.image-gradient-breakdown .ig-scale{height:17px;margin-top:4px;position:relative;font-variant-numeric:tabular-nums}
.image-gradient-breakdown .ig-scale>span{position:absolute;transform:translateX(-50%);white-space:nowrap;font-size:11px}
.image-gradient-breakdown .ig-scale>span:first-child{transform:none}.image-gradient-breakdown .ig-scale>span:last-child{transform:translateX(-100%)}
.image-gradient-breakdown .ig-row{padding:0;margin:0;background:transparent;border:0;border-bottom:1px solid #d6e0ef;border-radius:0}
.image-gradient-breakdown .ig-row>summary{list-style:none;padding:5px;min-height:42px;font-weight:400;cursor:pointer}
.image-gradient-breakdown .ig-row>summary::-webkit-details-marker{display:none}
.image-gradient-breakdown .ig-row>summary:hover{background:#f4f7fb}.image-gradient-breakdown .ig-row[open]{background:#edf3ff}
.image-gradient-breakdown .ig-image{font-size:11px;min-width:0;line-height:15px;display:block}
.image-gradient-breakdown .ig-image>span:first-child:before{content:'▸ ';color:#536174}.image-gradient-breakdown .ig-row[open] .ig-image>span:first-child:before{content:'▾ '}
.image-gradient-breakdown .ig-filename{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
.image-gradient-breakdown .ig-pair{display:grid;grid-template-rows:13px 13px;gap:1px;align-items:center;min-width:0}
.image-gradient-breakdown .ig-track{display:block;height:7px;border-radius:2px;background:#f0f4fa;overflow:hidden}
.image-gradient-breakdown .ig-bar{display:block;height:100%;border-radius:2px}
.image-gradient-breakdown .ig-values{text-align:right;font-size:11px;line-height:13px;font-variant-numeric:tabular-nums}
.image-gradient-breakdown .ig-d{text-align:right;font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.image-gradient-breakdown .ig-detail{padding:5px 12px 12px;font-size:12px;color:#536174;overflow-wrap:anywhere}
.image-gradient-breakdown .ig-missing{font-size:12px;color:#536174}
.image-gradient-breakdown .ig-method{margin:12px 0 0;padding:10px 0 0;border:0;border-top:1px solid #d6e0ef;border-radius:0;background:transparent}
.image-gradient-breakdown .ig-method>summary{padding:0;font-size:12px;min-height:28px}
@media(max-width:650px){.image-gradient-breakdown{padding:12px 8px}.image-gradient-breakdown .ig-columns{grid-template-columns:70px minmax(0,1fr) 52px 42px;gap:6px}.image-gradient-breakdown .ig-scale>span:nth-child(even){display:none}.image-gradient-breakdown .ig-row>summary{min-height:44px;padding:5px 3px}.image-gradient-breakdown .ig-list-heading{padding:0 3px 7px}.image-gradient-breakdown .ig-d{font-size:12px}.image-gradient-breakdown select{font-size:16px}}
@media(max-width:400px){.image-gradient-breakdown .ig-scale>span:nth-child(3){display:none}}
@media(pointer:coarse){.image-gradient-breakdown .ig-row>summary{min-height:44px}}
</style>'''


_SCRIPT = '''<script>(()=>{
const root=document.currentScript.parentElement;
const picker=root.querySelector('.ig-picker');
const select=picker.querySelector('select');
const panels=[...root.querySelectorAll('.ig-panel')];
picker.hidden=false;
select.addEventListener('change',()=>{panels.forEach(panel=>{panel.hidden=panel.dataset.panel!==select.value;});});
})();</script>'''
