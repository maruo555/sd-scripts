from __future__ import annotations

"""Display-only stacked-curve layout and local, accessible inspection state."""

import html
import math
from typing import Any, Callable, Mapping, Sequence


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _value(card: Mapping[str, Any], key: str) -> str:
    value = _finite(card.get(key))
    return str(value) if value is not None else "未算出"


def stacked_curves(
    upper: str, cards: Sequence[Mapping[str, Any]], *,
    sx: Callable[[float], float], upper_sy: Callable[[float], float],
    width: int, height: int, left: int, right: int, top: int, bottom: int,
    x_ticks: Sequence[str], notes: str, overview_html: str = "",
) -> str:
    plot_h = height - top - bottom
    values = [v for c in cards for key in ("parallel_component_p50", "parallel_component_p05") if (v := _finite(c.get(key))) is not None]
    y_min, y_max = min([0.0, *values]), max([1.1, *values])
    extended = y_min < 0 or y_max > 1.1
    if y_min < 0:
        y_min *= 1.08
    if y_max > 1.1:
        y_max *= 1.08

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    labels = []
    table_rows = []
    for card in cards:
        support = card.get("gradient_curve_support") or {}
        state = "HardSafety合格" if card.get("hard_safety_pass") else "Hard unsafe"
        if card.get("relative_status"):
            state += " / " + str(card["relative_status"])
        if card.get("edge_endpoint"):
            state += " / edge"
        metrics = " / ".join(f"{label} {_value(card, key)}" for label, key in (
            ("Body", "body"), ("Tail", "tail"), ("d P50", "d_p50"),
            ("元方向 P50", "parallel_component_p50"), ("元方向 P05", "parallel_component_p05")))
        reasons = sorted(set(support.get("d_reasons", []) + support.get("parallel_component_reasons", [])))
        if not support:
            reasons = ["追加指標未生成：保存済みsample CSVからCPU再生成できます"]
        explanation = " / ".join(reasons)
        counts = f'測定画像数 {support.get("image_count", "—")} / source数 {support.get("source_count", "—")} / 観測 {support.get("observation_count", "—")}'
        label = f'Mul {card["range_mul"]}: {metrics} / {state} / {counts}'
        if explanation:
            label += " / " + explanation
        labels.append(label)
        table_rows.append(
            '<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
                card["range_mul"], *(_value(card, key) for key in ("body", "tail", "d_p50", "parallel_component_p50", "parallel_component_p05")), state, counts, explanation or "算出済み")) + '</tr>')

    def targets(project_y: Callable[[float], float], keys: Sequence[str]) -> str:
        result = []
        for index, card in enumerate(cards):
            x = sx(float(card["range_mul"]))
            rings = ''.join(
                f'<circle cx="{x:.1f}" cy="{project_y(value):.1f}" r="7" fill="none" stroke="#334155" stroke-width="2"/>'
                for key in keys if (value := _finite(card.get(key))) is not None)
            result.append(
                f'<g class="curve-target" data-index="{index}" role="button" tabindex="0" aria-pressed="false" aria-label="{html.escape(labels[index], quote=True)}">'
                f'<g class="curve-highlight" pointer-events="none"><line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}" stroke="#64748b" stroke-dasharray="3 4"/>{rings}</g>'
                f'<rect class="curve-hit" x="{x-12:.1f}" y="{top}" width="24" height="{plot_h}" fill="transparent"><title>{html.escape(labels[index])}</title></rect></g>')
        return ''.join(result)

    upper = upper.replace('</svg>', targets(upper_sy, ("body", "tail", "d_p50")) + '</svg>')
    grid = []
    ticks = sorted({y_min, y_max, 0.0, 1.0, *(y_min + (y_max-y_min)*q for q in (.25, .5, .75))})
    # Keep 0 and 1 explicit; suppress nearby auto ticks to avoid label overlap.
    tick_gap = (y_max - y_min) * .065
    ticks = [v for v in ticks if v in (0., 1.) or all(abs(v-anchor) >= tick_gap for anchor in (0., 1.))]
    for value in ticks:
        y = sy(value)
        grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d7dee9"/><text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="svg-label">{value:.2f}</text>')
    curves = []
    for key, color, dash in (("parallel_component_p50", "#0f766e", ""), ("parallel_component_p05", "#7c3aed", 'stroke-dasharray="6 4"')):
        segments, current = [], []
        for card in cards:
            value = _finite(card.get(key))
            if value is None:
                if current:
                    segments.append(current)
                    current = []
                continue
            x, y = sx(float(card["range_mul"])), sy(value)
            current.append(f"{x:.1f},{y:.1f}")
            curves.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{key}: {value}</title></circle>')
        if current:
            segments.append(current)
        curves.extend(f'<polyline data-metric="{key}" points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" {dash}/>' for points in segments)
    lower = f'''
<h3 class="parallel-title">元の勾配方向への成分 <small>{"軸拡張" if extended else ""}</small></h3>
<svg class="chart parallel-chart" data-y-min="{y_min:.6f}" data-y-max="{y_max:.6f}" viewBox="0 0 {width} {height}" role="group" aria-label="元の勾配方向への成分 P50とP05">
 {''.join(grid)}
 <line x1="{left}" x2="{width-right}" y1="{sy(1):.1f}" y2="{sy(1):.1f}" stroke="#64748b" stroke-dasharray="7 5"/>
 <text x="{width-right-4}" y="{sy(1)-6:.1f}" text-anchor="end" class="svg-label">元方向への成分が同じ大きさ</text>
 {''.join(curves)}{''.join(x_ticks)}
 <text x="{width/2}" y="{height-3}" text-anchor="middle" class="svg-label">range_mul</text>
 <text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" class="svg-label">元方向への成分比</text>
 <g transform="translate({left+8},{height-40})"><circle r="4" fill="#0f766e"/><text x="10" y="4" class="svg-label">中央：P50</text><line x1="120" x2="140" y1="0" y2="0" stroke="#7c3aed" stroke-width="3" stroke-dasharray="6 4"/><text x="148" y="4" class="svg-label">成分が少ない側：P05</text></g>
 {targets(sy, ("parallel_component_p50", "parallel_component_p05"))}
</svg>'''
    return f'''<div class="gradient-curve-stack" data-selector-input="false">
<style>
.gradient-curve-stack .chart{{display:block;width:100%;height:auto;max-height:300px;aspect-ratio:{width}/{height};overflow:visible}}
.gradient-curve-stack .parallel-title{{font-size:16px;margin:6px 0 0 58px}}
.parallel-title small{{font-size:11px;font-weight:400;color:#607086}}
.curve-highlight{{opacity:0}}.curve-target.active .curve-highlight{{opacity:1}}.curve-target{{cursor:pointer;outline:none}}
.curve-target:focus-visible .curve-hit{{stroke:#334155;stroke-width:1}}
.curve-readout{{font-size:12px;min-height:3.3em;background:#f4f7fb;border-radius:6px;padding:6px 10px;overflow-wrap:anywhere}}
.curve-help{{font-size:12px;color:#536174;margin:6px 0}}.curve-values{{overflow:auto}}.curve-values table{{font-size:11px}}
</style>
{upper}{lower}
<p class="curve-help">上：dは小さいほど量子化なしの勾配に近い。下：1は元方向への成分が同じ大きさ。<br>上下を合わせて判断する補助情報です。下段の値は特徴保持率ではなく、どちらも画質の合否を示しません。</p>
<div class="curve-readout" role="status" aria-live="polite">点に触れると数値を表示。クリック・タップ・Enterで固定／解除、Escで解除。Tab・左右キーでも確認できます。</div>
{overview_html}
<details class="curve-values"><summary>追加指標の定義・全候補の数値・未算出理由</summary>
<p>source等重み、source内の保存観測等重みで全binをまとめます。repeatやnoiseを先に平均しません。sourceと元絵の対応は未確認です。旧CSVのsource未記録時は既存Bodyと同じimage_keyを使い、topology未記録時は一致を認証しません。</p>
<p>aは各行のgradient_norm_ratio × gradient_cosineです。P05は5パーセンタイルで、最小値・CI下限・最悪binではありません。a&lt;0は逆向き、a&gt;1は元方向成分の増幅です。aが1に近くても別方向への変形が小さいとは限りません。上下のP50が同じ観測を指すとは限りません。追加指標にCIはなく、selector_input=falseです。</p>
<table><thead><tr><th>Mul</th><th>Body：P95</th><th>Tail</th><th>d P50</th><th>元方向 P50</th><th>元方向 P05</th><th>既存状態</th><th>対象件数</th><th>算出状態・理由</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
{notes}</details>
{_INSPECTION_SCRIPT}
</div>'''


_INSPECTION_SCRIPT = r'''<script>
(() => {
  const root = document.currentScript.parentElement;
  const targets = [...root.querySelectorAll('.curve-target')];
  const readout = root.querySelector('.curve-readout');
  const idle = readout.textContent;
  let pinned = null;
  function show(index) {
    targets.forEach(t => {
      t.classList.toggle('active', t.dataset.index === index);
      t.setAttribute('aria-pressed', String(t.dataset.index === pinned));
    });
    const target = targets.find(t => t.dataset.index === index);
    readout.textContent = target ? (pinned !== null ? '固定中 / ' : '') + target.getAttribute('aria-label') : idle;
  }
  function toggle(target) {
    pinned = pinned === target.dataset.index ? null : target.dataset.index;
    show(pinned);
  }
  targets.forEach(target => {
    target.addEventListener('pointerenter', () => { if (pinned === null) show(target.dataset.index); });
    target.addEventListener('focus', () => { if (pinned === null) show(target.dataset.index); });
    target.addEventListener('pointerleave', () => { if (pinned === null) show(null); });
    target.addEventListener('blur', () => { if (pinned === null) show(null); });
    target.addEventListener('click', () => toggle(target));
    target.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(target); }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        const siblings = [...target.closest('svg').querySelectorAll('.curve-target')];
        const i = siblings.indexOf(target) + (event.key === 'ArrowLeft' ? -1 : 1);
        siblings[(i + siblings.length) % siblings.length].focus();
      }
    });
  });
  root.addEventListener('keydown', event => {
    if (event.key === 'Escape') { pinned = null; show(null); }
  });
})();
</script>'''
