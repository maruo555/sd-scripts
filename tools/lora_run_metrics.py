"""Recover the 68 consolidated v1 fields from typed rows and this run's verified resolved settings."""
import math


def gradient_fields(row):
    threshold = row.get("threshold")
    return {"ThreshOff": math.isnan(threshold) if isinstance(threshold, (int, float)) else None}


def group_loss_fields(row):
    return {"count_group": row.get("group_observation_count")}


def _qmax(bits):
    return (1 << (bits - 1)) - 1 if type(bits) is int and bits >= 2 else None


def _dq_settings(resolved):
    dq = (resolved or {}).get("dq") or {}
    auto = dq.get("auto") or {}
    init = auto.get("init") or {}
    low = auto.get("low_auto") or {}
    return auto, low, {
        "Stat": dq.get("stat"), "Granularity": dq.get("granularity"), "Mode": dq.get("rounding_mode"),
        "ClipRateLowAutoMinProgress": low.get("min_progress"), "ClipRateLowAutoFreezeProgress": low.get("freeze_progress"),
        "ClipRateLowAutoThresholdQErrRatio": low.get("qerr_ratio_threshold"),
        "ClipRateLowAutoThresholdQErrPerClip": low.get("qerr_per_clip_threshold"),
        "AutoInitMulApplied": init.get("applied"), "AutoInitMulValue": init.get("mul"),
        "AutoInitClipTarget": init.get("clip_target"),
    }


def dq_auto_fields(row, resolved):
    auto, _, result = _dq_settings(resolved)
    result.update(WarmupRemain=row.get("warmup_remaining_after"), QErrPerClipClipFloor=auto.get("qerr_per_clip_floor"),
                  ClipRateLowAutoCanEscape=row.get("low_auto_can_escape"), Qmax=_qmax(row.get("bits")),
                  RangeMul=row.get("common_mul_after") if row.get("event_type") == "check" else None)
    return result


def dq_fields(row, resolved, events):
    """events is unique_auto_events() of the same run. Missing or ambiguous references do not imply an idle controller."""
    auto, low, result = _dq_settings(resolved)
    ref = row.get("auto_state_step")
    state = events.get(ref) if ref is not None else None
    disabled = auto.get("enabled") is False
    if disabled:
        state = None
    check = bool(state and state.get("event_type") == "check" and state.get("attempt_step") == row.get("attempt_step"))
    known_idle = disabled or bool(state and not check)
    remaining = 0 if disabled else state.get("warmup_remaining_after") if state else None
    qmax = _qmax(row.get("bits"))
    quant_range = row.get("quant_range")
    shape = row.get("input_shape")
    dimensions = None
    if shape == "()":
        dimensions = ()
    elif isinstance(shape, str):
        parts = shape.split("x")
        if parts and all(part.isascii() and part.isdecimal() for part in parts):
            dimensions = tuple(map(int, parts))
    result.update(
        Shape=dimensions, ScaleMin=row.get("scale_min"), ScaleMax=row.get("scale_max"),
        ScaleMean=quant_range / qmax if qmax and quant_range is not None else None, Qmax=qmax,
        ClipRateLowAutoState=state.get("low_auto_state") if state and low.get("enabled") else None,
        ClipRateLowAutoBad=state.get("low_auto_bad") if check else None,
        ClipRateLowAutoBadStreak=state.get("low_auto_bad_streak") if state and low.get("enabled") else None,
        ClipRateLowAutoPhase=state.get("low_auto_phase" if check else "low_auto_phase_after") if state and low.get("enabled") else None,
        AutoApplied=state.get("auto_applied") if check else False if known_idle else None,
        RangeMulBefore=state.get("common_mul_before") if check else row.get("common_mul_after") if known_idle else None,
        RangeMulAfter=row.get("common_mul_after"), WarmupRemain=remaining,
        WarmupActive=state.get("warmup_active") if check else remaining > 0 if remaining is not None else None,
        AutoReason=state.get("auto_reason") if check else ("warmup" if remaining else "in_band") if state and remaining is not None else None,
    )
    return result


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lower = math.floor(pos)
    return ordered[lower] + (ordered[math.ceil(pos)] - ordered[lower]) * (pos - lower)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def rank_energy_total(records):
    values = [row.get("rank_energy") for row in records]
    return sum(values) if values and all(_finite(v) for v in values) else None


def rank_fields(records, completeness_issues):
    """Six legacy diagnostic summaries; records and issues come from rank_observations()."""
    rows = [] if completeness_issues else records
    total = rank_energy_total(rows)
    def values(field):
        all_values = [row for row in rows if _finite(row.get(field))]
        positive = [r for r in all_values if _finite(r.get("rank_energy")) and r["rank_energy"] > 1e-12]
        return [r[field] for r in positive or all_values]
    sat = values("rank_sat")
    pairs = [r["rank_sat"] * r["rank_energy"] for r in rows if _finite(r.get("rank_sat")) and _finite(r.get("rank_energy"))]
    weighted_sat = None
    if total is not None and len(pairs) == len(rows):
        weighted_sat = sum(pairs) / total if total > 1e-12 else 0.0
    return dict(RankSatWMean=weighted_sat,
                RankSatP50=quantile(sat, 0.5), RankSatP95=quantile(sat, 0.95), RankSatMax=max(sat, default=None),
                RankTop1P95=quantile(values("rank_top1"), 0.95), RankEnergySum=total)


def promote_fields(row, window, resolved):
    """The decision and its validated ordered window come from promote_windows()."""
    cfg = (resolved or {}).get("promote") or {}
    epoch_count = ((resolved or {}).get("training") or {}).get("total_epochs")
    status = row.get("status")
    scored = status == "scored"
    mode = row.get("selected_candidate_mode")
    raw = row.get("raw_proxy_loss")
    selected = row.get("selected_proxy_loss") if scored else None
    margin = row.get("margin_ok") if scored else None
    applied = row.get("promote_applied")
    delta = selected - raw if _finite(selected) and _finite(raw) else None
    if status in ("waiting_window", "scored"):
        ready = True
    elif status in ("before_avg_begin", "building_pool"):
        ready = False
    else:
        ready = None
    bins = {band: row.get(f"bank_timestep_{band}_count") for band in ("low", "mid", "high")}
    patience = cfg.get("patience")
    streak = row.get("win_streak")
    return {
        "progress": row["epoch"] / epoch_count if row.get("epoch") and epoch_count else None,
        "construction_pool_target_size": cfg.get("pool_target_size"), "bank_ready": ready,
        "bank_timestep_bins": bins if all(n is not None for n in bins.values()) else None,
        **{"bank_timestep_bins." + band: n for band, n in bins.items()},
        "avg_mode": cfg.get("mode"), "avg_promote_pick": cfg.get("pick"), "avg_window": cfg.get("window_size"),
        "center_proxy_loss": selected, "ema_proxy_loss": selected if mode == "ema" else None,
        "uniform_proxy_loss": selected if mode == "uniform" else None,
        "delta_abs": delta, "delta_pct": 100 * delta / raw if delta is not None and raw != 0 else None,
        "winner": ("center" if margin else "raw") if margin is not None else None,
        "winner_mode": (mode if margin else "raw") if margin is not None else None,
        "virtual_would_promote": (streak >= patience if streak is not None and patience is not None else None) if scored else False,
        "next_epoch_source": ("center" if applied else "raw") if applied is not None else None, "window_epochs": window,
    }
