from __future__ import annotations

"""Descriptive dataset-character channels for the v2.4 Local profiler.

These channels reuse already measured Local gradient rows.  They never vote
in candidate selection and do not predict image quality or training utility.
"""

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dq_profile.v24_acceptance import _weighted_quantile


DESCRIPTIVE_PROFILE_SCHEMA_VERSION = "1.0.0"
DESCRIPTIVE_METRIC_DEFINITION_VERSION = "1.0.0"
SOURCE_TAIL_QUANTILES = (0.85, 0.90, 0.95)
PRIMARY_SOURCE_TAIL_QUANTILE = 0.90


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_mul(row: Mapping[str, Any]) -> float | None:
    value = _optional_finite(row.get("range_mul"))
    if value is None:
        value = _optional_finite(row.get("initial_range_mul"))
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _source_aliases(source_groups: Sequence[str]) -> dict[str, str]:
    return {
        source_group: f"S{index:02d}"
        for index, source_group in enumerate(sorted(set(source_groups)), start=1)
    }


def _source_equal_values(
    rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    for source_group in sorted(rows_by_source):
        finite = [
            number
            for row in rows_by_source[source_group]
            if (number := _optional_finite(row.get(field))) is not None
        ]
        if not finite:
            continue
        per_row = 1.0 / len(finite)
        values.extend(finite)
        weights.extend([per_row] * len(finite))
    return values, weights


def _source_equal_quantile(
    rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
    quantile: float,
) -> float:
    values, weights = _source_equal_values(rows_by_source, field)
    return _weighted_quantile(values, weights, quantile)


def _effective_count(shares: Sequence[float]) -> float | None:
    finite = [float(value) for value in shares if math.isfinite(float(value))]
    denominator = sum(value * value for value in finite)
    return 1.0 / denominator if denominator > 0.0 else None


def _hard_safe_candidates(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in score_rows:
        candidate = str(raw.get("candidate", "")).strip()
        if not candidate or not _as_bool(raw.get("hard_safety_pass", True)):
            continue
        local_tail = _optional_finite(raw.get("local_tail"))
        range_mul = _candidate_mul(raw)
        if local_tail is None or range_mul is None:
            continue
        result[candidate] = {
            "candidate": candidate,
            "range_mul": range_mul,
            "local_body": _optional_finite(raw.get("local_body")),
            "local_tail": local_tail,
            "tail_amplification": _optional_finite(raw.get("tail_amplification")),
            "worst_timestep_bin": (
                int(float(raw["worst_timestep_bin"]))
                if _optional_finite(raw.get("worst_timestep_bin")) is not None
                else None
            ),
        }
    return result


def analyze_source_localization(
    *,
    gradient_tail_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    source_loo_rows: Sequence[Mapping[str, Any]],
    tail_quantiles: Sequence[float] = SOURCE_TAIL_QUANTILES,
    primary_tail_quantile: float = PRIMARY_SOURCE_TAIL_QUANTILE,
) -> dict[str, Any]:
    """Describe where the measured quantization Tail burden is localized.

    The source-balanced threshold is computed independently for each fixed-mul
    candidate.  Per-source burden is the source mean of ``max(d-threshold, 0)``.
    Burden shares are explanatory proxies, not additive image-quality damage.
    """

    quantiles = tuple(sorted({float(value) for value in tail_quantiles}))
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("source localization requires tail quantiles in (0, 1)")
    if not any(
        math.isclose(value, primary_tail_quantile, rel_tol=0.0, abs_tol=1e-12)
        for value in quantiles
    ):
        raise ValueError("primary source tail quantile must be present")

    score_by_candidate = _hard_safe_candidates(score_rows)
    samples_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: set[str] = set()
    for raw in gradient_tail_rows:
        if str(raw.get("record_type", "")) != "sample":
            continue
        candidate = str(raw.get("candidate", "")).strip()
        if candidate not in score_by_candidate:
            continue
        worst_timestep_bin = score_by_candidate[candidate].get(
            "worst_timestep_bin"
        )
        if (
            worst_timestep_bin is not None
            and int(raw.get("timestep_bin", 0)) != int(worst_timestep_bin)
        ):
            continue
        distance = _optional_finite(raw.get("relative_gradient_distance"))
        image_key = str(raw.get("image_key", "")).strip()
        source_group = str(raw.get("source_group", "") or image_key).strip()
        if distance is None or not image_key or not source_group:
            continue
        samples_by_candidate[candidate].append(
            {
                "candidate": candidate,
                "image_key": image_key,
                "source_group": source_group,
                "relative_gradient_distance": distance,
            }
        )
        source_groups.add(source_group)
    if not samples_by_candidate:
        return {
            "schema_version": DESCRIPTIVE_PROFILE_SCHEMA_VERSION,
            "metric_definition_version": DESCRIPTIVE_METRIC_DEFINITION_VERSION,
            "valid": False,
            "invalid_reason": "no_finite_hard_safe_gradient_tail_samples",
            "selector_input": False,
            "not_quality_or_utility": True,
        }

    aliases = _source_aliases(sorted(source_groups))
    loo_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_loo_rows:
        candidate = str(row.get("candidate", "")).strip()
        if candidate in score_by_candidate:
            loo_by_candidate[candidate].append(row)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    candidate_profiles: list[dict[str, Any]] = []
    for candidate in sorted(
        samples_by_candidate,
        key=lambda name: (score_by_candidate[name]["range_mul"], name),
    ):
        rows_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in samples_by_candidate[candidate]:
            rows_by_source[str(row["source_group"])].append(row)
        candidate_threshold_rows: list[dict[str, Any]] = []
        for quantile in quantiles:
            threshold = _source_equal_quantile(
                rows_by_source,
                "relative_gradient_distance",
                quantile,
            )
            burdens: dict[str, float] = {}
            exceedance_rates: dict[str, float] = {}
            for source_group, members in sorted(rows_by_source.items()):
                distances = [
                    float(row["relative_gradient_distance"]) for row in members
                ]
                burdens[source_group] = float(
                    np.mean([max(value - threshold, 0.0) for value in distances])
                )
                exceedance_rates[source_group] = float(
                    np.mean([value > threshold for value in distances])
                )
            total_burden = float(sum(burdens.values()))
            shares = {
                source_group: (
                    burden / total_burden if total_burden > 0.0 else 0.0
                )
                for source_group, burden in burdens.items()
            }
            top_source = min(
                shares,
                key=lambda source_group: (-shares[source_group], source_group),
            )
            effective_count = _effective_count(shares.values())
            active_count = sum(value > 0.0 for value in burdens.values())
            summary_row = {
                "candidate": candidate,
                "range_mul": score_by_candidate[candidate]["range_mul"],
                "timestep_bin": score_by_candidate[candidate].get(
                    "worst_timestep_bin"
                ),
                "tail_quantile": quantile,
                "source_balanced_threshold": threshold,
                "source_group_count": len(rows_by_source),
                "active_source_count": active_count,
                "total_mean_excess": total_burden,
                "top_source_group": top_source,
                "top_source_alias": aliases[top_source],
                "top_source_share": shares[top_source],
                "effective_source_count": effective_count,
                "effective_source_rate": (
                    effective_count / len(rows_by_source)
                    if effective_count is not None and rows_by_source
                    else None
                ),
                "selector_input": False,
                "not_quality_or_utility": True,
            }
            summary_rows.append(summary_row)
            candidate_threshold_rows.append(summary_row)
            for source_group in sorted(rows_by_source):
                detail_rows.append(
                    {
                        "candidate": candidate,
                        "range_mul": score_by_candidate[candidate]["range_mul"],
                        "timestep_bin": score_by_candidate[candidate].get(
                            "worst_timestep_bin"
                        ),
                        "tail_quantile": quantile,
                        "source_balanced_threshold": threshold,
                        "source_group": source_group,
                        "source_alias": aliases[source_group],
                        "sample_count": len(rows_by_source[source_group]),
                        "mean_excess": burdens[source_group],
                        "excess_share": shares[source_group],
                        "exceedance_rate": exceedance_rates[source_group],
                        "selector_input": False,
                        "not_quality_or_utility": True,
                    }
                )

        primary = next(
            row
            for row in candidate_threshold_rows
            if math.isclose(
                float(row["tail_quantile"]),
                primary_tail_quantile,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        top_sources = [str(row["top_source_group"]) for row in candidate_threshold_rows]
        effective_rates = [
            float(row["effective_source_rate"])
            for row in candidate_threshold_rows
            if row.get("effective_source_rate") is not None
        ]
        full_tail = float(score_by_candidate[candidate]["local_tail"])
        loo_candidates: list[tuple[float, str, float]] = []
        for loo_row in loo_by_candidate.get(candidate, []):
            loo_tail = _optional_finite(loo_row.get("local_tail"))
            omitted = str(loo_row.get("omitted_source_group", "")).strip()
            if loo_tail is not None and omitted:
                loo_candidates.append((full_tail - loo_tail, omitted, loo_tail))
        best_loo = (
            min(loo_candidates, key=lambda item: (-item[0], item[1]))
            if loo_candidates
            else None
        )
        candidate_profiles.append(
            {
                **dict(primary),
                "local_body": score_by_candidate[candidate].get("local_body"),
                "local_tail": score_by_candidate[candidate].get("local_tail"),
                "tail_amplification": score_by_candidate[candidate].get(
                    "tail_amplification"
                ),
                "threshold_quantiles": list(quantiles),
                "top_source_stable_across_thresholds": len(set(top_sources)) == 1,
                "top_source_aliases_across_thresholds": [
                    aliases[value] for value in top_sources
                ],
                "effective_source_rate_min": min(effective_rates) if effective_rates else None,
                "effective_source_rate_max": max(effective_rates) if effective_rates else None,
                "effective_source_rate_span": (
                    max(effective_rates) - min(effective_rates)
                    if effective_rates
                    else None
                ),
                "loo_max_tail_reduction": best_loo[0] if best_loo else None,
                "loo_max_tail_reduction_fraction": (
                    best_loo[0] / full_tail
                    if best_loo and full_tail > 0.0
                    else None
                ),
                "loo_most_actionable_source_group": best_loo[1] if best_loo else None,
                "loo_most_actionable_source_alias": (
                    aliases.get(best_loo[1]) if best_loo else None
                ),
                "loo_tail_without_source": best_loo[2] if best_loo else None,
                "loo_positive_reduction_observed": bool(best_loo and best_loo[0] > 0.0),
            }
        )

    reference = min(
        candidate_profiles,
        key=lambda row: (
            score_by_candidate[str(row["candidate"])]["local_tail"],
            float(row["range_mul"]),
        ),
    )
    return {
        "schema_version": DESCRIPTIVE_PROFILE_SCHEMA_VERSION,
        "metric_definition_version": DESCRIPTIVE_METRIC_DEFINITION_VERSION,
        "valid": True,
        "diagnostic_channel": "quantization_tail_source_localization",
        "definition": (
            "per-source mean excess above a source-equal-weighted candidate "
            "distance quantile within that candidate's worst timestep bin"
        ),
        "tail_quantiles": list(quantiles),
        "primary_tail_quantile": primary_tail_quantile,
        "reference_rule": "measured_point_tail_min_candidate_descriptive_only",
        "reference_candidate": str(reference["candidate"]),
        "reference_mul": float(reference["range_mul"]),
        "reference_profile": dict(reference),
        "candidate_profiles": candidate_profiles,
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "source_alias_map": aliases,
        "important_limit": (
            "source burden is a localization proxy, not additive image-quality harm; "
            "a high share with a low absolute Tail is not an alarm"
        ),
        "selector_input": False,
        "not_quality_or_utility": True,
    }


def analyze_no_quant_baseline(
    *,
    gradient_tail_rows: Sequence[Mapping[str, Any]],
    natural_baseline: Mapping[str, Any] | None,
    timestep_bins: int,
) -> dict[str, Any]:
    """Describe the no-quant reference signal already used by Local probes."""

    if timestep_bins <= 0:
        raise ValueError("no-quant baseline requires a positive timestep bin count")
    values_by_probe: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    for raw in gradient_tail_rows:
        if str(raw.get("record_type", "")) != "sample":
            continue
        image_key = str(raw.get("image_key", "")).strip()
        source_group = str(raw.get("source_group", "") or image_key).strip()
        norm = _optional_finite(raw.get("grad_norm_noquant"))
        if not image_key or not source_group or norm is None:
            continue
        key = (
            image_key,
            source_group,
            int(raw.get("timestep_bin", 0)),
            int(raw.get("noise_replica", 0)),
        )
        values_by_probe[key].append(norm)
    if not values_by_probe:
        return {
            "schema_version": DESCRIPTIVE_PROFILE_SCHEMA_VERSION,
            "metric_definition_version": DESCRIPTIVE_METRIC_DEFINITION_VERSION,
            "valid": False,
            "invalid_reason": "no_finite_no_quant_reference_norms",
            "selector_input": False,
            "not_quality_or_utility": True,
        }

    probe_rows: list[dict[str, Any]] = []
    inconsistent_reference_count = 0
    for (image_key, source_group, timestep_bin, noise_replica), norms in sorted(
        values_by_probe.items()
    ):
        median = float(np.median(norms))
        tolerance = max(1e-12, abs(median) * 1e-7)
        spread = max(norms) - min(norms)
        if spread > tolerance:
            inconsistent_reference_count += 1
        probe_rows.append(
            {
                "image_key": image_key,
                "source_group": source_group,
                "timestep_bin": timestep_bin,
                "noise_replica": noise_replica,
                "grad_norm_noquant": median,
                "duplicate_observation_count": len(norms),
                "duplicate_spread": spread,
            }
        )

    rows_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_by_bin: dict[int, dict[str, list[Mapping[str, Any]]]] = {
        index: defaultdict(list) for index in range(timestep_bins)
    }
    for row in probe_rows:
        source_group = str(row["source_group"])
        bin_index = int(row["timestep_bin"])
        rows_by_source[source_group].append(row)
        rows_by_bin[bin_index][source_group].append(row)
    aliases = _source_aliases(sorted(rows_by_source))

    def rms(source_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
        per_source_energy = [
            float(
                np.mean(
                    [float(row["grad_norm_noquant"]) ** 2 for row in members]
                )
            )
            for members in source_rows.values()
            if members
        ]
        return float(math.sqrt(np.mean(per_source_energy)))

    source_energy = {
        source_group: float(
            np.mean(
                [float(row["grad_norm_noquant"]) ** 2 for row in members]
            )
        )
        for source_group, members in rows_by_source.items()
    }
    total_energy = float(sum(source_energy.values()))
    source_shares = {
        source_group: (
            value / total_energy if total_energy > 0.0 else 0.0
        )
        for source_group, value in source_energy.items()
    }
    top_source = min(
        source_shares,
        key=lambda source_group: (-source_shares[source_group], source_group),
    )
    effective_count = _effective_count(source_shares.values())
    source_rows = [
        {
            "source_group": source_group,
            "source_alias": aliases[source_group],
            "probe_count": len(rows_by_source[source_group]),
            "mean_gradient_energy": source_energy[source_group],
            "energy_share": source_shares[source_group],
            "selector_input": False,
            "not_quality_or_utility": True,
        }
        for source_group in sorted(rows_by_source)
    ]
    timestep_rows: list[dict[str, Any]] = []
    for bin_index in range(timestep_bins):
        members = rows_by_bin[bin_index]
        if not members:
            timestep_rows.append(
                {
                    "timestep_bin": bin_index,
                    "source_group_count": 0,
                    "grad_norm_q05": None,
                    "grad_norm_median": None,
                    "grad_norm_q95": None,
                    "grad_norm_rms": None,
                    "selector_input": False,
                    "not_quality_or_utility": True,
                }
            )
            continue
        timestep_rows.append(
            {
                "timestep_bin": bin_index,
                "source_group_count": len(members),
                "grad_norm_q05": _source_equal_quantile(
                    members, "grad_norm_noquant", 0.05
                ),
                "grad_norm_median": _source_equal_quantile(
                    members, "grad_norm_noquant", 0.50
                ),
                "grad_norm_q95": _source_equal_quantile(
                    members, "grad_norm_noquant", 0.95
                ),
                "grad_norm_rms": rms(members),
                "selector_input": False,
                "not_quality_or_utility": True,
            }
        )
    measured_timestep_rows = [
        row for row in timestep_rows if row.get("grad_norm_rms") is not None
    ]
    dominant_bin = min(
        measured_timestep_rows,
        key=lambda row: (-float(row["grad_norm_rms"]), int(row["timestep_bin"])),
    )
    natural = dict(natural_baseline or {})
    natural_valid = bool(natural.get("valid"))
    return {
        "schema_version": DESCRIPTIVE_PROFILE_SCHEMA_VERSION,
        "metric_definition_version": DESCRIPTIVE_METRIC_DEFINITION_VERSION,
        "valid": True,
        "diagnostic_channel": "no_quant_reference_signal_profile",
        "probe_count": len(probe_rows),
        "image_count": len({str(row["image_key"]) for row in probe_rows}),
        "source_group_count": len(rows_by_source),
        "reference_duplicate_consistency_pass": inconsistent_reference_count == 0,
        "inconsistent_reference_probe_count": inconsistent_reference_count,
        "signal_strength": {
            "grad_norm_q05": _source_equal_quantile(
                rows_by_source, "grad_norm_noquant", 0.05
            ),
            "grad_norm_median": _source_equal_quantile(
                rows_by_source, "grad_norm_noquant", 0.50
            ),
            "grad_norm_q95": _source_equal_quantile(
                rows_by_source, "grad_norm_noquant", 0.95
            ),
            "grad_norm_rms": rms(rows_by_source),
            "cross_dataset_comparison_scope": (
                "same canonical model/network/optimizer/precision contract only"
            ),
        },
        "natural_variation": (
            {
                "valid": True,
                "local_body": _optional_finite(natural.get("local_body")),
                "local_tail": _optional_finite(natural.get("local_tail")),
                "tail_amplification": _optional_finite(
                    natural.get("tail_amplification")
                ),
                "worst_timestep_bin": natural.get("worst_timestep_bin"),
            }
            if natural_valid
            else {
                "valid": False,
                "invalid_reason": natural.get(
                    "invalid_reason", "natural_gradient_baseline_unavailable"
                ),
            }
        ),
        "source_load": {
            "top_source_group": top_source,
            "top_source_alias": aliases[top_source],
            "top_source_energy_share": source_shares[top_source],
            "effective_source_count": effective_count,
            "effective_source_rate": (
                effective_count / len(rows_by_source)
                if effective_count is not None and rows_by_source
                else None
            ),
        },
        "dominant_timestep_bin_by_gradient_rms": int(dominant_bin["timestep_bin"]),
        "source_rows": source_rows,
        "timestep_rows": timestep_rows,
        "source_alias_map": aliases,
        "important_limit": (
            "this describes short local no-quant gradient signals; it does not "
            "predict convergence, final quality, optimal rank, LR, or epochs"
        ),
        "selector_input": False,
        "not_quality_or_utility": True,
    }


def build_dataset_character_vector(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    local_phenotype: str,
    source_localization: Mapping[str, Any] | None,
    no_quant_baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build independent descriptive channels without a composite score."""

    candidates = [
        {
            "candidate": str(row.get("candidate", "")),
            "range_mul": _candidate_mul(row),
            "body": _optional_finite(row.get("local_body")),
            "tail": _optional_finite(row.get("local_tail")),
            "amplification": _optional_finite(row.get("tail_amplification")),
        }
        for row in score_rows
        if _as_bool(row.get("hard_safety_pass", True))
    ]
    candidates = [
        row
        for row in candidates
        if row["range_mul"] is not None
        and row["body"] is not None
        and row["tail"] is not None
    ]
    candidates.sort(key=lambda row: (float(row["range_mul"]), str(row["candidate"])))
    alarm_values = [max(float(row["body"]), float(row["tail"])) for row in candidates]
    if len(candidates) < 2:
        mul_response = {
            "label": "insufficient_grid",
            "endpoint_change": None,
            "span": None,
            "heuristic": True,
        }
    else:
        span = max(alarm_values) - min(alarm_values)
        median = float(np.median(alarm_values))
        tolerance = max(0.03, 0.05 * median)
        changes = [
            alarm_values[index + 1] - alarm_values[index]
            for index in range(len(alarm_values) - 1)
        ]
        minimum_index = min(
            range(len(alarm_values)),
            key=lambda index: (alarm_values[index], index),
        )
        endpoint_change = alarm_values[-1] - alarm_values[0]
        if span <= tolerance:
            label = "flat_within_descriptive_tolerance"
        elif all(value <= tolerance for value in changes) and endpoint_change < -tolerance:
            label = "upper_range_reduces_deformation"
        elif all(value >= -tolerance for value in changes) and endpoint_change > tolerance:
            label = "lower_range_reduces_deformation"
        elif (
            0 < minimum_index < len(alarm_values) - 1
            and alarm_values[0] > alarm_values[minimum_index] + tolerance
            and alarm_values[-1] > alarm_values[minimum_index] + tolerance
        ):
            label = "interior_valley"
        else:
            label = "irregular_or_mixed"
        mul_response = {
            "label": label,
            "endpoint_change": endpoint_change,
            "span": span,
            "descriptive_tolerance": tolerance,
            "minimum_observed_mul": float(candidates[minimum_index]["range_mul"]),
            "heuristic": True,
        }

    tail_reference = (
        min(candidates, key=lambda row: (float(row["tail"]), float(row["range_mul"])))
        if candidates
        else None
    )
    source_reference = (
        dict(source_localization.get("reference_profile") or {})
        if source_localization and source_localization.get("valid")
        else None
    )
    no_quant = (
        dict(no_quant_baseline)
        if no_quant_baseline and no_quant_baseline.get("valid")
        else None
    )
    return {
        "schema_version": DESCRIPTIVE_PROFILE_SCHEMA_VERSION,
        "metric_definition_version": DESCRIPTIVE_METRIC_DEFINITION_VERSION,
        "diagnostic_channel": "dataset_character_vector",
        "channels": {
            "absolute_acceptance": {
                "local_phenotype": str(local_phenotype),
                "minimum_body": (
                    min(float(row["body"]) for row in candidates)
                    if candidates
                    else None
                ),
                "minimum_tail": (
                    min(float(row["tail"]) for row in candidates)
                    if candidates
                    else None
                ),
            },
            "mul_response": mul_response,
            "tail_sensitivity": {
                "reference_rule": "measured_point_tail_min_candidate",
                "reference_mul": (
                    float(tail_reference["range_mul"]) if tail_reference else None
                ),
                "tail_amplification": (
                    float(tail_reference["amplification"])
                    if tail_reference and tail_reference["amplification"] is not None
                    else None
                ),
            },
            "source_localization": (
                {
                    "reference_mul": source_reference.get("range_mul"),
                    "top_source_share": source_reference.get("top_source_share"),
                    "effective_source_count": source_reference.get(
                        "effective_source_count"
                    ),
                    "effective_source_rate": source_reference.get(
                        "effective_source_rate"
                    ),
                    "top_source_stable_across_thresholds": source_reference.get(
                        "top_source_stable_across_thresholds"
                    ),
                    "loo_max_tail_reduction_fraction": source_reference.get(
                        "loo_max_tail_reduction_fraction"
                    ),
                }
                if source_reference
                else {"available": False}
            ),
            "no_quant_stability": (
                {
                    "natural_variation": no_quant.get("natural_variation"),
                    "signal_strength": no_quant.get("signal_strength"),
                    "source_load": no_quant.get("source_load"),
                    "dominant_timestep_bin": no_quant.get(
                        "dominant_timestep_bin_by_gradient_rms"
                    ),
                }
                if no_quant
                else {"available": False}
            ),
        },
        "single_composite_score": None,
        "selector_input": False,
        "not_quality_or_utility": True,
        "important_limit": (
            "channels remain separate; this vector is not a quality score, "
            "training-success prediction, or automatic prescription"
        ),
    }
