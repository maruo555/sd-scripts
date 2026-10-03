from __future__ import annotations

"""Read-only, CPU scalar summaries for the two stacked gradient curves.

This module is deliberately downstream of selection. It never repairs raw
observations, drops invalid observations, or imports the training runtime.
"""

from collections import defaultdict
import copy
import math
from typing import Any, Mapping, Sequence

from dq_profile.v24_acceptance import _cluster_quantile


SCHEMA_VERSION = "gradient-curve-support-v1"
METRIC_DEFINITION_VERSION = "1.0.0"
# Same near-zero guard as dataset_diagnostics.gradient_scalars. No replacement
# denominator is used here: a guarded reference makes the metric unavailable.
REFERENCE_NORM_FLOOR = 1e-12
METRICS = ("d_p50", "parallel_component_p50", "parallel_component_p05")
KEY_FIELDS = ("timestep_bin", "noise_replica", "quant_repeat")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any) -> bool | None:
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    return None


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _row_values(row: Mapping[str, Any]) -> tuple[float | None, float | None, list[str], list[str]]:
    common = []
    topology = row.get("gradient_topology_matches")
    if _present(topology) and _flag(topology) is not True:
        common.append("gradient_topology_mismatch_or_invalid")
    g0 = _number(row.get("grad_norm_noquant"))
    gm = _number(row.get("grad_norm_candidate"))
    if g0 is None or g0 < 0:
        common.append("missing_or_invalid_reference_norm")
    elif g0 <= REFERENCE_NORM_FLOOR or _flag(row.get("reference_near_zero")) is True:
        common.append("reference_near_zero")
    if _present(row.get("grad_norm_candidate")) and (gm is None or gm < 0):
        common.append("invalid_candidate_norm")
    if _present(row.get("gradient_invalid_reason")):
        common.append("saved_gradient_invalid_reason:" + str(row["gradient_invalid_reason"]))
    d_reasons, a_reasons = list(common), list(common)
    d = _number(row.get("relative_gradient_distance"))
    if d is None or d < 0:
        d_reasons.append("missing_or_invalid_relative_gradient_distance")
    a = None
    # An explicitly measured zero norm defines a=0 even when cosine is NaN.
    if gm == 0 and not common:
        a = 0.0
    else:
        ratio = _number(row.get("gradient_norm_ratio"))
        if not _present(row.get("gradient_norm_ratio")) and g0 is not None and g0 > REFERENCE_NORM_FLOOR and gm is not None:
            ratio = gm / g0
        cosine = _number(row.get("gradient_cosine"))
        if ratio is None or ratio < 0 or not math.isfinite(ratio):
            a_reasons.append("missing_or_invalid_gradient_norm_ratio")
        if cosine is None:
            a_reasons.append("missing_or_invalid_gradient_cosine")
        if not a_reasons:
            a = ratio * cosine
            if not math.isfinite(a):
                a_reasons.append("nonfinite_parallel_component")
    return d, a, d_reasons, a_reasons


def _image_breakdown(items: Mapping[tuple, Mapping], reference_keys: set[tuple], reasons: set[str]) -> dict:
    """Select one complete saved observation per image, never independent maxima.

    An invalid d anywhere in an image prevents claiming its maximum. A missing
    absolute difference in the chosen observation only disables this new view;
    it must not change the existing curves or choose a different observation.
    """
    by_image: dict[str, list[tuple]] = defaultdict(list)
    expected: dict[str, int] = defaultdict(int)
    for key in reference_keys:
        expected[key[0]] += 1
    for key in sorted(items):
        by_image[key[0]].append(key)
    images = []
    for image in sorted(set(expected) | set(by_image)):
        keys = by_image[image]
        errors = set(reasons)
        observations = []
        for key in keys:
            d, _, invalid, _ = _row_values(items[key]["row"])
            errors.update(invalid)
            if not invalid:
                observations.append((d, key))
        if not keys:
            errors.add("no_image_sample_rows")
        result = {
            "image_key": image, "status": "unavailable",
            "observation_count": len(keys), "expected_observation_count": expected.get(image, 0),
            "relative_gradient_distance": None, "grad_norm_noquant": None, "grad_diff_norm": None,
            "measurement": None,
        }
        if observations and not errors:
            # Keys are sorted; ties select the first (bin, noise, repeat) key.
            d, key = max(observations, key=lambda item: item[0])
            row = items[key]["row"]
            original = _number(row.get("grad_norm_noquant"))
            difference = _number(row.get("grad_diff_norm"))
            if difference is None or difference < 0:
                errors.add("missing_or_invalid_gradient_difference_norm")
            # Saved d and difference are evaluated through two equivalent
            # formulas. Allow cancellation/roundoff near zero, not a different
            # measurement or independently aggregated norm.
            elif not math.isclose(difference / original, d, rel_tol=1e-6, abs_tol=1e-7):
                errors.add("saved_distance_norm_mismatch")
            if not errors:
                result.update(
                    status="available", relative_gradient_distance=d,
                    grad_norm_noquant=original, grad_diff_norm=difference,
                    measurement={field: key[index + 1] for index, field in enumerate(KEY_FIELDS)},
                )
        result["reasons"] = sorted(errors)
        images.append(result)
    return {
        "schema_version": "image-gradient-breakdown-v1", "selector_input": False,
        "selection": "maximum_saved_d_per_image_at_each_mul",
        "tie_break": "ascending_timestep_bin_noise_replica_quant_repeat",
        "available_image_count": sum(row["status"] == "available" for row in images),
        "image_count": len(images), "images": images,
    }


def build_gradient_support(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any] | None = None,
    input_error: str | None = None,
) -> dict[str, Any]:
    """Use the Body source distribution, validating complete candidate support.

    Candidate names and Mul values must come from saved score rows/cards.
    Missing source_group follows Body's existing image_key fallback. Missing
    topology flags in legacy CSVs are not invented or required sidecars.
    """
    ordered = sorted(candidates, key=lambda c: (float(c["range_mul"]), str(c.get("candidate", ""))))
    by_name = {str(c.get("candidate", "")): c for c in ordered}
    observations: dict[str, dict[tuple, dict]] = {name: {} for name in by_name}
    errors: dict[str, set[str]] = {name: set() for name in by_name}
    duplicates = defaultdict(int)
    image_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    contexts: dict[str, set[str | None]] = defaultdict(set)
    for raw in rows:
        if raw.get("record_type") != "sample":
            continue
        name = str(raw.get("candidate", ""))
        if name not in by_name:
            if name != "no_quant":
                for reasons in errors.values():
                    reasons.add("unknown_candidate_in_raw")
            continue
        row = dict(raw)
        # DictReader puts surplus CSV columns under a None key. Keep the
        # observation for coverage checks, but do not aggregate this candidate.
        if any(not isinstance(field, str) for field in row):
            errors[name].add("malformed_sample_columns")
        image = str(row.get("image_key") or "")
        source = str(row.get("source_group") or image)
        if not image:
            errors[name].add("missing_image_key")
            continue
        key_numbers = [_number(row.get(field)) for field in KEY_FIELDS]
        if any(n is None or n < 0 or not n.is_integer() for n in key_numbers):
            errors[name].add("missing_or_invalid_probe_key")
            continue
        key = (image, *(int(n) for n in key_numbers))
        if _present(row.get("range_mul")):
            mul = _number(row["range_mul"])
            if mul is None or not math.isclose(mul, float(by_name[name]["range_mul"]), rel_tol=0, abs_tol=1e-12):
                errors[name].add("candidate_mul_mismatch")
        for field in ("run_id", "snapshot", "snapshot_id", "post_state_hash", "edge_round", "probe_regime"):
            # Entirely unrecorded fields are valid for legacy logs; a mix of
            # recorded and missing values cannot identify one probe context.
            contexts[field].add(str(row[field]) if _present(row.get(field)) else None)
        image_sources[image][source].add(name)
        # Compare all saved columns, so conflicting observations never win by
        # file order. String normalization also handles CSV and in-memory rows.
        signature = tuple(sorted((str(k), str(v)) for k, v in row.items()))
        previous = observations[name].get(key)
        if previous is not None:
            if previous["signature"] == signature:
                duplicates[name] += 1
            else:
                errors[name].add("conflicting_duplicate_probe_key")
            continue
        observations[name][key] = {"row": row, "signature": signature, "source": source}
    if len(by_name) != len(ordered) or "" in by_name:
        input_error = "missing_or_duplicate_candidate_mapping"
    if any(len(values) > 1 for values in contexts.values()):
        input_error = "mixed_run_snapshot_or_probe_context"
    for sources in image_sources.values():
        if len(sources) > 1:
            for names in sources.values():
                for name in names:
                    errors[name].add("image_source_mismatch")
    # Hard-unsafe incomplete probes must not reduce or enlarge the population
    # of the existing Body curves. Use the union of saved safe candidate keys.
    reference_names = [n for n, c in by_name.items() if _flag(c.get("hard_safety_pass")) is True]
    reference_names = reference_names or list(by_name)
    reference_keys = set().union(*(set(observations[n]) for n in reference_names))
    reference_norms: dict[tuple, set[float]] = defaultdict(set)
    for name in reference_names:
        for key, item in observations[name].items():
            norm = _number(item["row"].get("grad_norm_noquant"))
            if norm is not None:
                reference_norms[key].add(norm)
    conflicting_norms = {key for key, values in reference_norms.items() if len(values) > 1}
    output = []
    for card in ordered:
        name = str(card.get("candidate", ""))
        items = observations[name]
        common = set(errors[name])
        if input_error:
            common.add(input_error)
        if not items:
            common.add("no_sample_rows")
        if set(items) != reference_keys:
            common.add("candidate_probe_key_mismatch")
        # Check unsafe candidates against the safe references without letting
        # their norms change the reference population for other candidates.
        if set(items) & conflicting_norms or any(
            (norm := _number(item["row"].get("grad_norm_noquant"))) is not None
            and reference_norms.get(key)
            and norm not in reference_norms[key]
            for key, item in items.items()
        ):
            common.add("reference_norm_mismatch_between_candidates")
        groups: dict[str, list[dict]] = defaultdict(list)
        d_reasons, a_reasons = set(common), set(common)
        d_valid = a_valid = 0
        for key in sorted(items):
            item = items[key]
            d, a, de, ae = _row_values(item["row"])
            d_reasons.update(de)
            a_reasons.update(ae)
            d_valid += not de
            a_valid += not ae
            groups[item["source"]].append({"d": d, "a": a})
        sources = sorted(groups)
        d_p95 = _cluster_quantile(groups, sources, "d", .95) if not d_reasons else None
        saved_body = _number(card.get("body", card.get("local_body")))
        if d_p95 is not None and saved_body is not None and not math.isclose(d_p95, saved_body, rel_tol=1e-10, abs_tol=1e-12):
            d_reasons.add("saved_body_population_mismatch")
            a_reasons.add("saved_body_population_mismatch")
            common.add("saved_body_population_mismatch")
        result = {
            "candidate": name, "range_mul": float(card["range_mul"]),
            "selector_input": False, "metric_definition_version": METRIC_DEFINITION_VERSION,
            "d_p50": _cluster_quantile(groups, sources, "d", .50) if not d_reasons else None,
            "parallel_component_p50": _cluster_quantile(groups, sources, "a", .50) if not a_reasons else None,
            "parallel_component_p05": _cluster_quantile(groups, sources, "a", .05) if not a_reasons else None,
            "d_p95_check": d_p95,
            "d_status": "unavailable" if d_reasons else "available",
            "parallel_component_status": "unavailable" if a_reasons else "available",
            "d_reasons": sorted(d_reasons), "parallel_component_reasons": sorted(a_reasons),
            "observation_count": len(items), "expected_observation_count": len(reference_keys),
            "d_valid_observation_count": d_valid, "parallel_valid_observation_count": a_valid,
            "image_count": len({key[0] for key in items}), "source_count": len(sources),
            "duplicate_row_count": duplicates[name],
            "topology_unrecorded_count": sum(not _present(item["row"].get("gradient_topology_matches")) for item in items.values()),
            "image_breakdown": _image_breakdown(items, reference_keys, common),
        }
        result["status"] = "available" if not d_reasons and not a_reasons else "unavailable" if d_reasons and a_reasons else "partial"
        output.append(result)
    return {
        "schema_version": SCHEMA_VERSION, "metric_definition_version": METRIC_DEFINITION_VERSION,
        "selector_input": False,
        "weighting": "equal_source_group_mass_equal_saved_observation_mass_within_source_all_bins",
        "quantile": "first_value_reaching_q_times_total_weight_no_interpolation",
        "parallel_component_definition": "per_row_gradient_norm_ratio_times_gradient_cosine",
        "source_semantics": "measurement_source_group_not_verified_original_artwork",
        "reference_norm_floor": REFERENCE_NORM_FLOOR,
        "provenance": dict(provenance or {}), "candidates": output,
    }


def attach_gradient_support(model: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    """Attach display-only namespaces without mutating the saved input model."""
    result = copy.deepcopy(model)
    if len(result["datasets"]) != 1:
        raise ValueError("gradient support rebuild requires a single dataset/run")
    dataset = result["datasets"][0]
    dataset["gradient_curve_support"] = copy.deepcopy(support)
    by_name = {row["candidate"]: row for row in support["candidates"]}
    for card in dataset["candidate_cards"]:
        card["gradient_curve_support"] = copy.deepcopy(by_name.get(str(card.get("candidate", "")), {}))
    return result
