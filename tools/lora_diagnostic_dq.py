"""Interpret the two DQ streams using the scopes and multiplier policy recorded by this run."""
from collections import Counter, defaultdict
import math
import statistics

from library.dq_mul_policy import restore_assignments
from tools.lora_run_metrics import dq_fields
from tools.lora_run_reader import unique_auto_events


def _finite(n):
    return n is not None and math.isfinite(n)


def _pool(rows):
    if len(rows) == 1:
        return dict(rows[0])
    shared = {k: v if all(r.get(k) == v for r in rows[1:]) else None for k, v in rows[0].items()}
    shared["scope"] = "both"
    for name in ("input_rms", "quant_error_rms_raw", "clip_rate_raw", "zero_rate", "quant_error_ratio_raw",
                 "quant_range", "scale_min", "scale_max", "numel", "input_abs_max"):
        shared[name] = None
    if all(_finite(r.get("input_abs_max")) for r in rows):
        shared["input_abs_max"] = max(r["input_abs_max"] for r in rows)
    if any(not r.get("numel") or r["numel"] <= 0 for r in rows):
        return shared
    n = shared["numel"] = sum(r["numel"] for r in rows)
    for key in ("input_rms", "quant_error_rms_raw", "clip_rate_raw", "zero_rate"):
        if all(_finite(r.get(key)) and r[key] >= 0 for r in rows):
            rms = key in ("input_rms", "quant_error_rms_raw")
            mean = sum(r["numel"] * (r[key] ** 2 if rms else r[key]) for r in rows) / n
            shared[key] = math.sqrt(mean) if rms else mean
    if shared["quant_error_rms_raw"] is not None and shared["input_rms"] is not None:
        shared["quant_error_ratio_raw"] = shared["quant_error_rms_raw"] / (shared["input_rms"] + 1e-12)
    return shared


def analyze_dq(rows, auto_rows, settings):
    configs = (settings or {}).get("dq", {})
    scope = configs.get("log_scope")
    expected = {"unet", "te"} if scope == "both" else {scope} if scope in {"unet", "te"} else set()
    events = unique_auto_events(auto_rows)
    issues = [{"attempt_step": step, "reason": "duplicate auto event"} for step, event in events.items() if event is None]
    checks = [r for _, r in sorted(events.items()) if r and r["event_type"] == "check"]
    eligible = [r for r in checks if r["warmup_active"] is False and _finite(r["clip_rate_raw"]) and
                _finite(r["clip_rate_ema"]) and r["auto_reason"]]
    summaries = defaultdict(list)
    for row in rows:
        if any(row.get(k) is None for k in ("epoch", "attempt_step", "scope", "target", "bits")):
            issues.append({"reason": "missing DQ coordinates"})
            continue
        summaries[row["epoch"], row["attempt_step"], row["scope"]].append(row)
    by_step, scopes = defaultdict(dict), defaultdict(list)
    for (epoch, step, component), members in sorted(summaries.items()):
        modules = [r.get("module") for r in members]
        same_type = len({(r["target"], r["bits"]) for r in members}) == 1
        if not same_type or len(modules) > 1 and (None in modules or len(modules) != len(set(modules))):
            issues.append({"epoch": epoch, "attempt_step": step, "scope": component, "reason": "ambiguous DQ observation"})
            continue
        entry = _pool(members)
        entry["scope"] = component
        derived = dq_fields(entry, settings, events)
        entry["controller_decision"] = {key: derived[key] for key in
                                        ("WarmupActive", "WarmupRemain", "AutoApplied", "RangeMulBefore", "RangeMulAfter",
                                         "AutoReason", "ClipRateLowAutoState", "ClipRateLowAutoPhase")}
        event = events.get(entry.get("auto_state_step"))
        if entry.get("auto_state_step") is not None and event is None:
            issues.append({"attempt_step": step, "scope": component, "reason": "auto reference unavailable"})
        elif event:
            entry["referenced_auto_step"] = event["attempt_step"]
            entry["referenced_mul"] = event["common_mul_after"]
        scopes[component].append(entry)
        by_step[epoch, step][component] = entry
    pooled = []
    for (epoch, step), members in sorted(by_step.items()):
        if members.keys() != expected or len({(r["target"], r["bits"]) for r in members.values()}) != 1:
            issues.append({"epoch": epoch, "attempt_step": step, "reason": "incomplete or unknown log scope"})
            continue
        pooled.append(_pool(list(members.values())))
    ema = [r["controller_clip_rate_ema"] for r in pooled if _finite(r.get("controller_clip_rate_ema"))]
    source = "dq"
    if not ema:
        ema = [r["clip_rate_ema"] for r in checks if _finite(r.get("clip_rate_ema"))]
        source = "dq_auto"
    tail = ema[-max(8, len(ema) // 4):] if len(ema) >= 8 else []
    cv = statistics.pstdev(tail) / statistics.mean(tail) if tail and statistics.mean(tail) > 0 else None
    mul_record = configs.get("mul")
    allocation = None
    if mul_record:
        try:
            allocation = sorted(Counter(info["mul"] for info in restore_assignments(mul_record).values()).items())
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"reason": str(exc)})
    inside = sum(r["auto_reason"] == "in_band" for r in eligible)
    return dict(raw_scope=scope, auto_scope=configs.get("auto", {}).get("scope"),
                controller_preset=configs.get("auto", {}).get("preset"), definition=configs.get("metric_definition_version"),
                scope_rows=dict(scopes), pooled=pooled, events=[r for _, r in sorted(events.items()) if r],
                issues=issues, multiplier_counts=allocation,
                summary=dict(check_count=len(checks), applied_count=sum(r["auto_applied"] is True for r in checks),
                             eligible_band_checks=len(eligible), excluded_band_checks=len(checks)-len(eligible),
                             in_band_ratio=inside / len(eligible) if eligible else None, clip_ema_cv=cv,
                             clip_ema_cv_source=source if ema else None))
