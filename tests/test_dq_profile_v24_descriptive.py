from __future__ import annotations

import pytest

from dq_profile.v24_descriptive import (
    analyze_no_quant_baseline,
    analyze_source_localization,
    build_dataset_character_vector,
)


def _score_rows() -> list[dict]:
    return [
        {
            "candidate": "mul_2.700",
            "range_mul": 2.70,
            "hard_safety_pass": True,
            "local_body": 0.80,
            "local_tail": 1.00,
            "tail_amplification": 1.25,
        },
        {
            "candidate": "mul_3.150",
            "range_mul": 3.15,
            "hard_safety_pass": True,
            "local_body": 0.50,
            "local_tail": 0.60,
            "tail_amplification": 1.20,
        },
    ]


def _tail_rows() -> list[dict]:
    rows: list[dict] = []
    candidates = ("mul_2.700", "mul_3.150")
    for candidate in candidates:
        for source_index, source in enumerate(("source-a", "source-b", "source-c")):
            for sample in range(20):
                if candidate == "mul_2.700":
                    distance = 5.0 if source == "source-a" and sample >= 18 else 0.0
                else:
                    distance = 1.0 if sample >= 18 else 0.0
                rows.append(
                    {
                        "record_type": "sample",
                        "candidate": candidate,
                        "image_key": f"{source}-image-{sample:02d}",
                        "source_group": source,
                        "timestep_bin": sample % 4,
                        "noise_replica": 0,
                        "quant_repeat": 0,
                        "relative_gradient_distance": distance,
                        "grad_norm_noquant": float(source_index + 1),
                    }
                )
    return rows


def test_source_localization_separates_concentrated_and_diffuse_tail() -> None:
    result = analyze_source_localization(
        gradient_tail_rows=_tail_rows(),
        score_rows=_score_rows(),
        source_loo_rows=[
            {
                "candidate": "mul_2.700",
                "omitted_source_group": "source-a",
                "local_tail": 0.90,
            },
            {
                "candidate": "mul_2.700",
                "omitted_source_group": "source-b",
                "local_tail": 0.20,
            },
        ],
    )

    profiles = {row["candidate"]: row for row in result["candidate_profiles"]}
    concentrated = profiles["mul_2.700"]
    diffuse = profiles["mul_3.150"]
    assert concentrated["top_source_alias"] == "S01"
    assert concentrated["top_source_share"] == pytest.approx(1.0)
    assert concentrated["effective_source_count"] == pytest.approx(1.0)
    assert diffuse["top_source_share"] == pytest.approx(1.0 / 3.0)
    assert diffuse["effective_source_count"] == pytest.approx(3.0)
    assert concentrated["top_source_stable_across_thresholds"] is True

    # The source with the largest local burden need not be the source whose
    # leave-one-out removal changes the dataset Tail most.
    assert concentrated["loo_most_actionable_source_alias"] == "S02"
    assert concentrated["loo_max_tail_reduction"] == pytest.approx(0.8)
    assert result["selector_input"] is False
    assert result["not_quality_or_utility"] is True


def test_source_localization_uses_each_candidates_worst_timestep_bin() -> None:
    rows: list[dict] = []
    for source in ("source-a", "source-b"):
        for sample in range(20):
            rows.append(
                {
                    "record_type": "sample",
                    "candidate": "mul_3.150",
                    "image_key": f"{source}-tail-{sample}",
                    "source_group": source,
                    "timestep_bin": 3,
                    "noise_replica": 0,
                    "quant_repeat": 0,
                    "relative_gradient_distance": (
                        5.0 if source == "source-a" and sample >= 18 else 0.0
                    ),
                }
            )
            rows.append(
                {
                    "record_type": "sample",
                    "candidate": "mul_3.150",
                    "image_key": f"{source}-body-{sample}",
                    "source_group": source,
                    "timestep_bin": 0,
                    "noise_replica": 0,
                    "quant_repeat": 0,
                    "relative_gradient_distance": (
                        100.0 if source == "source-b" else 0.0
                    ),
                }
            )
    result = analyze_source_localization(
        gradient_tail_rows=rows,
        score_rows=[
            {
                "candidate": "mul_3.150",
                "range_mul": 3.15,
                "hard_safety_pass": True,
                "local_body": 0.5,
                "local_tail": 1.0,
                "tail_amplification": 2.0,
                "worst_timestep_bin": 3,
            }
        ],
        source_loo_rows=[],
    )
    reference = result["reference_profile"]
    assert reference["timestep_bin"] == 3
    assert reference["top_source_alias"] == "S01"
    assert reference["local_tail"] == pytest.approx(1.0)


def test_no_quant_profile_deduplicates_candidate_and_quant_repeat_rows() -> None:
    rows: list[dict] = []
    for candidate in ("mul_2.700", "mul_3.150"):
        for quant_repeat in (0, 1):
            for source, norm in (("source-a", 2.0), ("source-b", 1.0)):
                for timestep_bin in range(4):
                    rows.append(
                        {
                            "record_type": "sample",
                            "candidate": candidate,
                            "image_key": f"{source}-image-{timestep_bin}",
                            "source_group": source,
                            "timestep_bin": timestep_bin,
                            "noise_replica": 0,
                            "quant_repeat": quant_repeat,
                            "grad_norm_noquant": norm,
                        }
                    )
    result = analyze_no_quant_baseline(
        gradient_tail_rows=rows,
        natural_baseline={
            "valid": True,
            "local_body": 0.10,
            "local_tail": 0.20,
            "tail_amplification": 2.0,
            "worst_timestep_bin": 3,
        },
        timestep_bins=4,
    )

    assert result["probe_count"] == 8
    assert result["reference_duplicate_consistency_pass"] is True
    assert result["source_load"]["top_source_alias"] == "S01"
    assert result["source_load"]["top_source_energy_share"] == pytest.approx(0.8)
    assert result["source_load"]["effective_source_count"] == pytest.approx(
        1.0 / (0.8**2 + 0.2**2)
    )
    assert len(result["timestep_rows"]) == 4
    assert result["selector_input"] is False


def test_dataset_character_vector_keeps_channels_separate() -> None:
    source = analyze_source_localization(
        gradient_tail_rows=_tail_rows(),
        score_rows=_score_rows(),
        source_loo_rows=[],
    )
    no_quant = analyze_no_quant_baseline(
        gradient_tail_rows=_tail_rows(),
        natural_baseline={"valid": False, "invalid_reason": "synthetic"},
        timestep_bins=4,
    )
    vector = build_dataset_character_vector(
        score_rows=_score_rows(),
        local_phenotype="selective_window",
        source_localization=source,
        no_quant_baseline=no_quant,
    )

    assert vector["channels"]["mul_response"]["label"] == (
        "upper_range_reduces_deformation"
    )
    assert vector["single_composite_score"] is None
    assert vector["selector_input"] is False
    assert vector["not_quality_or_utility"] is True
