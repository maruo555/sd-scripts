"""CPU-only DQ log grouping and recorded mul settings for the training report."""
import json
import math
from collections import defaultdict


SCOPE_LABELS = {"unet": "UNet", "te": "TE（TE1＋TE2）", "unknown": "範囲未記録", "both": "UNet＋TE"}
RMS_FIELDS = ("RMS", "QuantErrRMSRaw", "ClipErrRMS", "RoundErrRMS")
RATE_FIELDS = ("ClipRateRaw", "ZeroRate")
RAW_FIELDS = (*RMS_FIELDS, *RATE_FIELDS, "QuantErrRatioRaw", "ClipErrRatio", "RoundErrRatio",
              "ClipShare", "RoundShare", "AbsMax", "Range")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric(value):
    try:
        result = float(value) if not isinstance(value, bool) else None
        return result if finite(result) else None
    except (ValueError, TypeError):
        return None


def recorded_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def common_row(rows):
    """Keep shared controls/EMA once per step, never average conflicting values."""
    return {key: rows[0].get(key) if all(r.get(key) == rows[0].get(key) for r in rows) else None
            for key in set().union(*(r.keys() for r in rows))}


def aggregate_rows(rows, complete=True):
    if len(rows) == 1 and complete:
        return dict(rows[0])
    result = common_row(rows)
    result.update({key: None for key in RAW_FIELDS})
    weights = [r.get("Numel") for r in rows]
    result["Numel"] = None
    if not complete or not all(finite(n) and n > 0 for n in weights):
        return result
    total = sum(weights)
    result["Numel"] = total
    for key in (*RMS_FIELDS, *RATE_FIELDS):
        values = [r.get(key) for r in rows]
        if all(finite(v) and v >= 0 for v in values):
            if key in RMS_FIELDS:
                result[key] = math.sqrt(sum(n * v * v for n, v in zip(weights, values)) / total)
            else:
                result[key] = sum(n * v for n, v in zip(weights, values)) / total
    for error, ratio in (("QuantErrRMSRaw", "QuantErrRatioRaw"), ("ClipErrRMS", "ClipErrRatio"), ("RoundErrRMS", "RoundErrRatio")):
        if finite(result[error]) and finite(result["RMS"]):
            result[ratio] = result[error] / (result["RMS"] + 1e-12)
    for error, share in (("ClipErrRMS", "ClipShare"), ("RoundErrRMS", "RoundShare")):
        if finite(result[error]) and finite(result["QuantErrRMSRaw"]) and result["QuantErrRMSRaw"] > 0:
            result[share] = result[error] ** 2 / result["QuantErrRMSRaw"] ** 2
    if all(finite(r.get("AbsMax")) for r in rows):
        result["AbsMax"] = max(r["AbsMax"] for r in rows)
    # Range is a mean over quantizer scales, not over delta elements. The CSV
    # lacks the number of scales, so it cannot be pooled using Numel.
    return result


def group_dq_rows(rows):
    """Preserve input rows; build one observation per scope/step and pooled step."""
    groups = defaultdict(lambda: defaultdict(list))
    warnings = []
    for row in rows:
        groups[row.get("Scope") or "unknown"][row["TrainStep"]].append(row)
    scopes = {}
    for scope, steps in sorted(groups.items(), key=lambda item: ({"unet": 0, "te": 1}.get(item[0], 2), item[0])):
        scopes[scope] = []
        for step, items in sorted(steps.items()):
            modules = [r.get("Module") for r in items]
            targets = {r.get("Target") for r in items}
            unambiguous = len(items) == 1 or (all(modules) and len(set(modules)) == len(items))
            complete = unambiguous and len(targets) == 1
            if not complete:
                warnings.append(f"{scope} step {step}：同じstepの記録が重複、またはTargetが混在しています。Rawを合算できません。")
            merged = aggregate_rows(items, complete)
            merged.update(Scope=scope, TrainStep=step)
            scopes[scope].append(merged)
    by_step = defaultdict(list)
    for items in scopes.values():
        for row in items:
            by_step[row["TrainStep"]].append(row)
    pooled = []
    for step, items in sorted(by_step.items()):
        complete = len(items) == len(scopes) and len({r.get("Target") for r in items}) == 1
        # A pre-pooled scope must not be counted again beside its components.
        complete = complete and not (len(scopes) > 1 and any(s not in ("unet", "te") for s in scopes))
        merged = aggregate_rows(items, complete)
        if len(items) > 1:
            merged["Scope"] = "both"
        pooled.append(merged)
    if len(scopes) > 1:
        warnings.extend(["一部のstepで範囲が欠ける、重みが未記録、またはTargetが異なるため、合算Rawは欠測として表示します。"]
                        if any(not finite(r.get("QuantErrRatioRaw")) for r in pooled) else [])
        for scope, items in scopes.items():
            available = {r["TrainStep"]: r for r in items}
            scopes[scope] = [available.get(step, {"TrainStep": step, "Scope": scope}) for step in sorted(by_step)]
    return {"scope_rows": scopes, "plot_rows": pooled, "warnings": list(dict.fromkeys(warnings))}


def mul_settings(settings):
    """Use only the run-associated record, never reopen a mutable policy file."""
    settings = settings or {}
    values = settings.get("values") or {}
    metadata = (settings.get("resolved") or {}).get("metadata") or settings.get("raw_metadata") or {}
    policy = metadata.get("ss_dq_mul_policy", values.get("dq_mul_policy"))
    source = "保存されたmul policy"
    if policy is None:
        direct = {key: values.get("dq_delta_range_mul_" + key) for key in ("attn2", "te", "te1", "te2")}
        if values.get("dq_delta_policy_file"):
            return {"rows": [], "note": "policyファイルの指定はありますが、保存された配分がありません。現在のファイルから推定しません。"}
        if not any(v is not None for v in direct.values()):
            return {"rows": [], "note": ""}
        policy = {"base_mul": values.get("dq_delta_range_mul"), "components": {}, "group_overrides": [], "te_quantized": True}
        for key in ("te1", "te2"):
            value = direct[key] if direct[key] is not None else direct["te"]
            if value is not None:
                policy["components"][key] = value
        if direct["attn2"] is not None:
            policy["group_overrides"].append({"group_id": "unet.attn2", "range_mul": direct["attn2"]})
        source = "学習設定記録のCLI指定"
    try:
        policy = json.loads(policy) if isinstance(policy, str) else policy
        base = numeric(policy["base_mul"])
        components = policy.get("components", {})
        te_enabled = policy.get("te_quantized", True)
        if not isinstance(te_enabled, bool):
            raise ValueError("invalid TE quantization flag")
        rows = [{"label": "その他UNet（上書き対象を除く）", "mul": numeric(components.get("unet", base)), "enabled": True}]
        for override in policy.get("group_overrides", []):
            rows.append({"label": override["group_id"], "mul": numeric(override["range_mul"]), "enabled": True})
        for key in ("te1", "te2"):
            rows.append({"label": key.upper(), "mul": numeric(components.get(key, base)), "enabled": te_enabled})
        for module, mul in policy.get("module_overrides", {}).items():
            rows.append({"label": "個別上書き：" + module, "mul": numeric(mul), "enabled": te_enabled or not module.startswith("lora_te")})
        if not finite(base) or base <= 0 or any(not finite(r["mul"]) or r["mul"] <= 0 for r in rows):
            raise ValueError("invalid mul")
        return {"rows": rows, "source": source, "policy": policy, "base_mul": base,
                "scope": values.get("dq_delta_scope"), "fixed": True,
                "note": "量子化に使うmulは上記の部位別配分です。RangeMulログ列の共通設定値は、上書きされた部位のmulを表しません。"}
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"rows": [], "note": "保存されたmul配分を読み取れません。RangeMul列だけでは部位別の値を判断できません。"}


def ema(values, beta):
    state = None
    result = []
    for value in values:
        if not finite(value):
            result.append(None)
            continue
        state = value if state is None else beta * state + (1 - beta) * value
        result.append(state)
    return result


def scope_charts(charts, dq_data):
    """Split raw curves; shared logged EMAs remain exclusively in pooled charts."""
    scopes = dq_data.get("scope_rows") or {}
    if not scopes:
        return charts
    multiple = len(scopes) > 1
    label = "ログ対象合算" if multiple else SCOPE_LABELS.get(next(iter(scopes)), next(iter(scopes)))
    beta = numeric(dq_data.get("ema_beta"))
    beta = beta if beta is not None and 0 <= beta < 1 else None
    raw_charts = {
        "dq_clip": ("ClipRateRaw", "Clip率"),
        "dq_qerr_ratio": ("QuantErrRatioRaw", "相対量子化誤差"),
        "dq_qerr_rms": ("QuantErrRMSRaw", "量子化誤差RMS"),
        "dq_zero_rate": ("ZeroRate", "ゼロ率"),
        "dq_absmax": ("AbsMax", "最大絶対値"),
        "dq_range": ("Range", "量子化範囲"),
    }
    output = []
    for original in charts:
        chart = {**original}
        if chart["id"] == "dq_clip" and (dq_data.get("auto_monitor") or {}).get("charts"):
            chart["series"] = [s for s in chart["series"] if s["name"] not in ("ClipRateEMA", "ActiveClipLow", "ActiveClipHigh")]
        if chart["id"] in raw_charts or chart["id"] in ("dq_error_part_ratio", "dq_error_part_share"):
            chart["title"] = label + " · " + chart["title"]
            chart["dq_scope"] = "combined"
            chart["subtitle"] = ("Raw：要素数と二乗和から再集計／誤差EMA：学習ログの合算値" if multiple
                                 else "Raw：この範囲の記録値／EMA：学習ログの記録値")
            if chart["id"] == "dq_clip":
                chart["subtitle"] = ("Raw：要素数で重み付き集計" if multiple else "Raw：この範囲の記録値") + "／判定EMA・目標帯は下の自動調整監視へ"
        if chart["id"] in ("dq_clip", "dq_qerr_ratio", "dq_absmax", "dq_error_part_share"):
            # Keep historical scale floors while allowing all observations to fit.
            chart["y_max_ceil"] = chart.pop("y_max_fixed", None)
        if chart["id"] == "dq_range_mul" and dq_data.get("mul_settings", {}).get("rows"):
            chart["title"] = "共通設定値のログ（部位別mulではありません）"
            chart["subtitle"] = "部位別の適用値は上の固定配分で確認してください。このログは部位別上書き前の共通設定値です。"
            chart["series"] = [{**s, "name": "共通設定値（RangeMul列）"} for s in chart["series"]]
        output.append(chart)
    if not multiple:
        return output
    for scope, rows in scopes.items():
        for chart_id, (field, title) in raw_charts.items():
            if not any(finite(r.get(field)) for r in rows):
                continue
            values = [r.get(field) for r in rows]
            series = [{"name": field, "color": "#dc2626", "y": values}]
            note = "Raw：範囲別ログ"
            if beta is not None and field in ("ClipRateRaw", "QuantErrRatioRaw", "QuantErrRMSRaw"):
                series.append({"name": "部位EMA（再計算）", "color": "#0f766e", "y": ema(values, beta)})
                note += f"／EMA：記録点から再計算（β={beta:g}）"
            output.append({"id": chart_id + "_" + scope, "title": SCOPE_LABELS.get(scope, scope) + " · " + title,
                           "dq_scope": scope, "subtitle": note, "x_label": "TrainStep",
                           "x": [r["TrainStep"] for r in rows], "series": series,
                           "markers": dq_data.get("markers", []), "y_min_fixed": 0.0})
    # Match each metric's vertical scale so a quieter TE line is not magnified.
    for chart_id in raw_charts:
        peers = [c for c in output if c["id"] == chart_id or c["id"].startswith(chart_id + "_")]
        maximum = max((v for c in peers for s in c["series"] for v in s["y"] if finite(v)), default=0)
        ceiling = max([maximum * 1.05] + [c.get("y_max_ceil") or 0 for c in peers])
        template = next((c for c in peers if c["id"] == chart_id), {})
        for chart in peers:
            chart["y_max_fixed"] = ceiling
            for key in ("y_tick_step", "y_tick_precision"):
                if key in template:
                    chart[key] = template[key]
    return output


def build_auto_monitor(data, settings=None):
    """Keep controller observations separate from display-scope quantization stats."""
    settings = settings or {}
    values = settings.get("values") or {}
    runtime = (settings.get("resolved") or {}).get("runtime") or {}
    low_auto_enabled = recorded_bool(runtime.get("dq_low_auto_enabled"))
    preset = runtime.get("dq_auto_preset") or values.get("dq_delta_auto_preset")
    if low_auto_enabled is None and preset:
        low_auto_enabled = preset == "clip_rate_low_auto"
    declared_scope = runtime.get("dq_auto_scope") or values.get("dq_delta_auto_scope") or values.get("dq_delta_scope")
    auto_rows = data.get("auto_rows") or []
    observed_scopes = {r.get("Scope") for r in auto_rows if r.get("Scope")}
    if len(observed_scopes) == 1:
        declared_scope = next(iter(observed_scopes))
    has_control = any(finite(r.get(k)) for r in auto_rows for k in ("ClipRateRaw", "ClipRateEMA", "QErrPerClip"))
    mode = "recorded" if has_control else "reference" if data.get("auto_enabled") is False else "sampled"
    groups = defaultdict(list)
    if has_control:
        for row in auto_rows:
            groups[row.get("Scope") or declared_scope or "unknown"].append(dict(row))
    elif mode == "reference":
        groups.update({scope: [dict(r) for r in rows] for scope, rows in (data.get("scope_rows") or {}).items()})
    else:
        source = data.get("plot_rows") or []
        if not any(finite(r.get(k)) for r in source for k in ("ClipRateEMA", "QErrPerClip")):
            return {"mode": "unavailable", "charts": [], "notes": ["自動調整の監視ログは未記録です。"]}
        scope = declared_scope or "unknown"
        raw_by_step = {r["TrainStep"]: r for r in (data.get("scope_rows") or {}).get(scope, [])}
        for row in source:
            copied = dict(row)
            if scope != "both":
                copied["ClipRateRaw"] = raw_by_step.get(row["TrainStep"], {}).get("ClipRateRaw")
            elif not {"unet", "te"}.issubset(data.get("scope_rows") or {}):
                # A single logged component cannot reconstruct combined control stats.
                copied["ClipRateRaw"] = None
            groups[scope].append(copied)
    notes = []
    if mode == "reference":
        notes.append("自動mul調整はOFFです。Clip率と保存された参考帯を表示します。帯を外れても、この学習ではmulを変更していません。")
        notes.append("QErrPerClip・判定用の誤差EMA・mul変更履歴は未記録です。自動調整を実行した値としては補いません。")
    else:
        notes.append("判定対象のClip率・EMA・目標帯を表示します。上のUNet/TE表示の切り替えとは独立しています。")
        if low_auto_enabled is False:
            notes.append("この設定のmul判定はClip率を使います。相対量子化誤差は参考表示とし、使用しないlow_autoの閾値は表示しません。")
        else:
            notes.append("QuantErrRatioとQErrPerClipによる退避判定はclip_rate_low_auto用です。通常presetの調整判定はClip率を使います。")
        notes.append("誤差の記録間隔は自動調整の間隔と異なる場合があります。未記録のstepは線で結ばないため、点だけになる場合があります。")
        if mode == "sampled":
            notes.append("auto専用ログがないため、通常ログに残った監視値だけを表示しています。AutoStepすべての履歴ではありません。")
    charts = []
    restored = False
    for scope, source in groups.items():
        # Ambiguous repeated records are gaps, not extra points at the same x.
        steps = defaultdict(list)
        for row in source:
            steps[row["TrainStep"]].append(row)
        rows = [common_row(items) for _, items in sorted(steps.items())]
        label = SCOPE_LABELS.get(scope, scope)
        xs = [r["TrainStep"] for r in rows]
        beta = numeric(data.get("ema_beta"))
        if mode == "reference" and beta is not None and 0 <= beta < 1:
            for row, smoothed in zip(rows, ema([r.get("ClipRateRaw") for r in rows], beta)):
                row["ReferenceClipEMA"] = smoothed
        for row in rows:
            if not finite(row.get("AutoQuantErrRatioEMA")):
                qpc, clip, floor = (row.get(k) for k in ("QErrPerClip", "ClipRateEMA", "QErrPerClipClipFloor"))
                if all(finite(v) for v in (qpc, clip, floor)) and floor > 0:
                    row["AutoQuantErrRatioEMA"] = qpc * max(clip, floor)
                    restored = True

        def chart(suffix, title, fields, subtitle=None):
            series = [{"name": name, "color": color, "is_reference": reference,
                       "y": [r.get(key) for r in rows]}
                      for key, name, color, reference in fields]
            if not any(finite(v) for s in series if not s["is_reference"] for v in s["y"]):
                return
            charts.append({"id": "dq_auto_" + suffix + "_" + scope,
                           "title": label + " · " + title, "x_label": "TrainStep", "x": xs,
                           "y_min_fixed": 0, "series": series, "markers": data.get("markers", []),
                           "subtitle": subtitle or ("自動調整OFF・参考表示" if mode == "reference" else "自動調整で使う範囲の記録")})

        reference = mode == "reference"
        chart("clip", "Clip率と" + ("参考帯" if reference else "目標帯"), [
            ("ClipRateRaw", "ClipRateRaw", "#dc2626", False),
            ("ReferenceClipEMA" if reference else "ClipRateEMA", "参考EMA（再計算）" if reference else "ClipRateEMA（判定値）", "#0f766e", False),
            ("ActiveClipLow", "参考下限（OFF）" if reference else "ActiveClipLow", "#64748b", True),
            ("ActiveClipHigh", "参考上限（OFF）" if reference else "ActiveClipHigh", "#334155", True),
        ])
        if reference:
            continue
        error_reference = low_auto_enabled is False
        error_subtitle = "参考値・mul判定には使いません。未記録のstepは線で結びません。" if error_reference else None
        qerr_fields = [
            ("AutoQuantErrRatioRaw", "AutoQuantErrRatioRaw", "#dc2626", False),
            ("AutoQuantErrRatioEMA", "AutoQuantErrRatioEMA（参考値）" if error_reference else "AutoQuantErrRatioEMA（判定値）", "#0f766e", False),
        ]
        qpc_fields = [
            ("QErrPerClip", "QErrPerClip", "#0f766e", False),
        ]
        if not error_reference:
            qerr_fields.append(("ClipRateLowAutoThresholdQErrRatio", "判定閾値", "#64748b", True))
            qpc_fields.append(("ClipRateLowAutoThresholdQErrPerClip", "判定閾値", "#64748b", True))
        chart("qerr", "相対量子化誤差（参考）" if error_reference else "相対量子化誤差（low_auto用）", qerr_fields, error_subtitle)
        chart("qerr_per_clip", "QErrPerClip（参考）" if error_reference else "QErrPerClip", qpc_fields, error_subtitle)
        chart("mul", "mulの調整履歴", [
            ("RangeMulBefore", "変更前mul", "#94a3b8", False),
            ("RangeMulAfter", "変更後mul", "#2563eb", False),
        ])
        chart("bad", "low_autoの判定・連続回数", [
            ("ClipRateLowAutoBad", "Bad", "#dc2626", False),
            ("ClipRateLowAutoBadStreak", "BadStreak", "#d97706", False),
        ])
    if restored:
        notes.append("旧ログの判定用誤差EMAは、記録されたQErrPerClip × max(ClipRateEMA, floor)から復元しています（丸め差あり）。通常ログの合算EMAは代用していません。")
    return {"mode": mode, "scopes": list(groups), "charts": charts, "notes": notes,
            "path": data.get("auto_path") if has_control else data.get("path")}
