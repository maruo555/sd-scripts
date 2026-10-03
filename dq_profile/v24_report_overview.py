"""Small, display-only summaries of existing diagnostic measurements."""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from dq_profile.dataset_diagnostics import FLOOR, SCHEMA, improvement


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "pass", "passed"}


def attach_report_overview(model: Mapping[str, Any], data_dir: Path, output_dir: Path) -> dict:
    """Read only the sidecar from the active profile/run; never search other runs.

    Keep compact saved values when a report is moved without its sidecars, but
    show a detail link only when its destination exists at generation time.
    """
    result = copy.deepcopy(model)
    if len(result.get("datasets", [])) != 1:
        return result
    dataset = result["datasets"][0]
    path = data_dir / "dataset_summary.json"
    response = dataset.get("warmup_response") or {
        "status": "unavailable", "reason": "学習前後の評価記録がありません。", "selector_input": False,
    }
    if path.is_file():
        response = {"status": "unavailable", "selector_input": False}
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            manifest, values = payload["manifest"], payload["all"]
            if not isinstance(manifest, dict) or not isinstance(values, dict):
                raise ValueError("invalid summary structure")
            response["input_sha256"] = hashlib.sha256(raw).hexdigest()
            if manifest.get("schema_version") != SCHEMA or manifest.get("loss_metric", "raw_mse") != "raw_mse":
                raise ValueError("unsupported measurement contract")
            measured = sorted(float(v) for v in manifest["muls"])
            expected = sorted(float(c["range_mul"]) for c in dataset["candidate_cards"])
            if len(measured) != len(expected) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for a, b in zip(measured, expected)):
                raise ValueError("candidate grid differs from this report")
            pre, post = _number(values.get("loss_pre")), _number(values.get("loss_post"))
            paired, total = int(values["paired_images"]), int(values["inventory_images"])
            measured_images = int(values.get("measured_images", paired))
            if manifest.get("mode") != "warmup":
                response["reason"] = "学習前の評価がないため未算出です。"
            elif pre is None or post is None or pre < 0 or post < 0 or not 0 < paired <= measured_images <= total:
                response["reason"] = "同じ評価入力で比較できる学習前後の値がありません。"
            else:
                response.update(improvement(pre, post), paired_images=paired, inventory_images=total,
                                weighting="image_equal", status="available")
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            response["reason"] = "学習前後の評価記録を確認できないため未算出です。"
    dataset["warmup_response"] = response
    target = data_dir / "dataset_report.html"
    dataset["dataset_diagnostics_href"] = None
    if target.is_file():
        try:
            href = quote(Path(os.path.relpath(target.resolve(), output_dir.resolve())).as_posix(), safe="/")
        except ValueError:  # different Windows drives
            href = target.resolve().as_uri()
        dataset["dataset_diagnostics_href"] = href
    return result


def overview_anchor(dataset: Mapping[str, Any], kind: str) -> str:
    suffix = hashlib.sha256(str(dataset.get("dataset_id", "")).encode()).hexdigest()[:12]
    return f"dq-{kind}-{suffix}"


def _mse(value: float) -> str:
    return f"{value:.4f}" if value >= .001 else f"{value:.3g}"


def render_overview(dataset: Mapping[str, Any]) -> str:
    response = dataset.get("warmup_response") or {}
    pre, post = _number(response.get("loss_pre")), _number(response.get("loss_post"))
    href = dataset.get("dataset_diagnostics_href")
    # Only locally generated relative/file links are supported.
    link = (f'<a href="{html.escape(str(href), quote=True)}">画像ごとの違いを見る →</a>'
            if href and not str(href).lower().startswith(("javascript:", "data:")) else "")
    if response.get("status") == "available" and pre is not None and post is not None:
        rate = improvement(pre, post)["improvement_rel"]
        if pre <= FLOOR or _number(rate) is None:
            main = '<span class="overview-missing">減少率は未算出</span>'
            note = "学習前の誤差がゼロに近いため、割合は表示しません。"
        else:
            word = "誤差減少" if rate > 0 else "誤差増加" if rate < 0 else "変化なし"
            main = f'<strong class="overview-value">{abs(rate) * 100:.1f}%</strong><span>{word}</span>'
            note = "同じ評価入力の平均予測誤差"
        baseline = (f'<div class="overview-main">{main}</div>'
                    f'<div class="overview-before-after"><span><small>学習前</small> <strong>{_mse(pre)}</strong></span>'
                    f'<span aria-hidden="true">→</span><span><small>warmup後</small> <strong>{_mse(post)}</strong></span></div>'
                    f'<p>{note} · 比較画像 {response.get("paired_images", "—")} / {response.get("inventory_images", "—")}枚</p>')
    else:
        baseline = ('<div class="overview-main overview-missing">未算出</div><p>'
                    + html.escape(str(response.get("reason") or "学習前後の評価記録がありません。")) + '</p>')
    cards = list(dataset.get("candidate_cards") or [])
    valid = [c for c in cards if _number(c.get("range_mul")) is not None
             and (tail := _number(c.get("tail"))) is not None and tail >= 0]
    if valid:
        tails = [float(c["tail"]) for c in valid]
        muls = [float(c["range_mul"]) for c in valid]
        quant = (f'<div class="overview-main"><span>Tail</span><strong class="overview-value">{min(tails):.2f}〜{max(tails):.2f}</strong></div>'
                 f'<p>Tailの最小〜最大 · 候補間の範囲（信頼区間ではありません）</p>'
                 f'<p>対象Mul {min(muls):.2f}〜{max(muls):.2f} · {len(valid)} / {len(cards)}候補</p>')
    else:
        quant = '<div class="overview-main overview-missing">未算出</div><p>表示可能なTailの測定値がありません。</p>'
    warnings = []
    unsafe = sum(not _safe(c.get("hard_safety_pass")) for c in cards)
    if unsafe:
        warnings.append(f"HardSafety不合格 {unsafe}候補（数値が有効な候補は範囲に含む）")
    if len(valid) != len(cards):
        warnings.append(f"Tail未算出 {len(cards) - len(valid)}候補")
    if dataset.get("edge_unresolved") or dataset.get("edge_direction", "resolved") not in (None, "resolved"):
        warnings.append("測定範囲の端は未解決")
    if warnings:
        quant += '<p class="overview-note">' + ' / '.join(warnings) + '</p>'
    localization = dataset.get("source_localization") or {}
    reference = localization.get("reference_profile") or {}
    guide, links = [], []
    if localization.get("valid") and reference:
        mul = _number(reference.get("range_mul"))
        alias, share = reference.get("top_source_alias"), _number(reference.get("top_source_share"))
        if mul is not None and alias and share is not None:
            guide.append(f'Mul {mul:.2f}：Tail負担の最大source {alias}（{share * 100:.1f}%）')
        links.append(f'<a href="#{overview_anchor(dataset, "source")}">source別の内訳</a>')
    if dataset.get("timestep_rows"):
        links.append(f'<a href="#{overview_anchor(dataset, "timestep")}">ノイズ帯の内訳</a>')
    if href:
        links.append(link.replace("画像ごとの違いを見る →", "画像別の詳細"))
    guidance = ('<div class="overview-guide"><strong>確認先</strong>'
                + (f'<span>{html.escape(" / ".join(guide))}</span>' if guide else '')
                + ' '.join(links) + ('<span>追加の詳細記録なし</span>' if not links else '') + '</div>')
    return f'''<div class="report-overview" data-selector-input="false">
<style>
.report-overview{{margin:16px 0 12px}}.report-overview .overview-cards{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}}
.report-overview .overview-card{{margin:0;padding:14px 18px;min-height:172px;border:1px solid #d6e0ef;border-radius:12px;background:white;min-width:0}}
.report-overview h3{{font-size:14px;margin:0 0 5px;color:#172338}}.report-overview .overview-main{{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;color:#172338}}
.report-overview .overview-value{{font-size:30px;line-height:1.3;font-weight:800;font-variant-numeric:tabular-nums}}.report-overview .overview-missing{{font-size:20px;font-weight:700}}
.report-overview .overview-before-after{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin:7px 0;color:#172338;font-variant-numeric:tabular-nums}}
.report-overview .overview-before-after strong{{font-size:18px;font-weight:750}}.report-overview .overview-before-after small{{font-size:12px;font-weight:400}}
.report-overview p{{font-size:12px;margin:4px 0;color:#536174}}.report-overview a{{font-size:12px;color:#1d4ed8;display:inline-block}}
.report-overview .overview-note{{color:#855100}}.report-overview .overview-guide{{display:flex;gap:6px 12px;align-items:baseline;flex-wrap:wrap;font-size:12px;margin:10px 3px 0;overflow-wrap:anywhere}}
@media(max-width:650px){{.report-overview .overview-cards{{grid-template-columns:1fr}}.report-overview .overview-card{{padding:14px;min-height:150px}}}}
@media(pointer:coarse){{.report-overview a{{padding:12px 0}}}}
</style>
<div class="overview-cards"><section class="overview-card"><h3>Warmupでの誤差減少（量子化OFF）</h3>{baseline}{link}</section>
<section class="overview-card"><h3>量子化による変化</h3>{quant}</section></div>{guidance}
<script>(()=>{{const root=document.currentScript.parentElement;root.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',()=>{{const target=document.getElementById(a.getAttribute('href').slice(1));if(target&&target.tagName==='DETAILS')target.open=true;}}));}})();</script>
</div>'''
