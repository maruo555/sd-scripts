from __future__ import annotations

import json
from pathlib import Path

import pytest

from dq_profile.v24_beginner_report import render_beginner_report
from tools.render_dq_beginner_report import render_existing_run


def _model() -> dict:
    cards = [
        {
            "range_mul": 2.70,
            "body": 0.82,
            "body_ci_low": 0.70,
            "body_ci_high": 0.95,
            "tail": 1.28,
            "tail_ci_low": 1.02,
            "tail_ci_high": 1.60,
            "hard_safety_pass": True,
            "absolute_perturbation": "tail_attention",
            "edge_endpoint": False,
        },
        {
            "range_mul": 3.15,
            "body": 0.69,
            "body_ci_low": 0.61,
            "body_ci_high": 0.78,
            "tail": 0.88,
            "tail_ci_low": 0.72,
            "tail_ci_high": 1.05,
            "hard_safety_pass": True,
            "absolute_perturbation": "low_perturbation",
            "edge_endpoint": False,
        },
        {
            "range_mul": 3.45,
            "body": 0.73,
            "body_ci_low": 0.64,
            "body_ci_high": 0.84,
            "tail": 0.91,
            "tail_ci_low": 0.77,
            "tail_ci_high": 1.08,
            "hard_safety_pass": True,
            "absolute_perturbation": "low_perturbation",
            "edge_endpoint": True,
        },
    ]
    dataset = {
        "dataset_id": "SYN",
        "label": "Example dataset",
        "execution_mode": "standard",
        "candidate_cards": cards,
        "hard_safety_pass_count": 3,
        "fidelity_retained_muls": [3.15, 3.45],
        "relatively_stronger_muls": [2.70],
        "body_representative_mul": 3.15,
        "tail_representative_mul": 3.15,
        "single_representative_mul": 3.15,
        "representative_selection_reason": "BodyとTailの代表が一致しました。",
        "absolute_response": "mixed_absolute_response",
        "absolute_response_label": "mulにより摂動帯が変化",
        "edge_direction": "upper",
        "measurement_quality": {
            "level": "PASS",
            "reasons": ["必須gateを通過しました。"],
        },
        "local_comparison_confidence": {
            "level": "Medium",
            "reasons": ["候補のCIが一部重なります。"],
        },
        "recommendation_maturity": {"level": "Local-only"},
        "image_count": 32,
        "image_count_probed": 32,
        "image_count_total": 40,
        "source_group_count": 6,
        "source_group_count_probed": 6,
        "source_group_count_total": 6,
        "dataset_character_vector": {
            "channels": {
                "no_quant_stability": {
                    "natural_variation": {"local_tail": 4.2}
                },
                "absolute_acceptance": {"minimum_tail": 0.88},
                "mul_response": {
                    "label": "interior_valley",
                    "span": 0.40,
                },
                "source_localization": {"top_source_share": 0.56},
            }
        },
        "no_quant_baseline_profile": {
            "timestep_rows": [
                {"timestep_bin": 0, "grad_norm_rms": 1.0},
                {"timestep_bin": 3, "grad_norm_rms": 3.2},
            ]
        },
        "source_localization": {
            "reference_profile": {
                "range_mul": 3.15,
                "top_source_share": 0.56,
                "top_source_alias": "source-03",
                "loo_max_tail_reduction_fraction": 0.22,
                "loo_most_actionable_source_alias": "source-03",
                "effective_source_count": 4.1,
                "source_group_count": 6,
            }
        },
        "timestep_rows": [
            {
                "range_mul": mul,
                "timestep_bin": bin_index,
                "source_balanced_q95_relative_distance": value,
            }
            for mul, values in {
                2.70: (0.70, 0.84, 1.06, 1.28),
                3.15: (0.62, 0.69, 0.77, 0.88),
                3.45: (0.65, 0.73, 0.82, 0.91),
            }.items()
            for bin_index, value in enumerate(values)
        ],
    }
    return {
        "schema_version": "2.4.3",
        "not_quality_or_utility": True,
        "dataset_count": 1,
        "datasets": [dataset],
    }


def test_beginner_report_leads_with_curve_and_explains_each_channel() -> None:
    rendered = render_beginner_report(_model())
    assert "診断レポート（概要）" in rendered
    assert rendered.index('class="chart-card"') < rendered.index("まず読む3行")
    assert 'data-scale-mode="fixed" data-y-max="4.000000"' in rendered
    assert "<strong>Body</strong>" in rendered
    assert "<strong>Tail</strong>" in rendered
    assert "全画像・timestepをまとめた、やや厳しめの代表値" in rendered
    assert "各timestep帯の代表値のうち最大" in rendered
    assert "ヒゲ（淡い縦棒）" in rendered
    assert "試したmulと役割" in rendered
    assert "橙の「注意」は安全だが相対的に強い摂動" in rendered
    assert 'class="matrix-mark attention"' in rendered
    assert 'class="matrix-mark representative"' in rendered
    assert "今回見られた動き" in rendered
    assert "Body × Tail マップ" in rendered
    assert "Body：左ほどno_quantに近い／右ほど変形が強い" in rendered
    assert "Tail：下ほどno_quantに近い／上ほど難しい帯が強い" in rendered
    assert "データセットの性格カルテ" in rendered
    assert "中央線は参照集団の中央値で、理想値や良否の閾値ではありません" in rendered
    for label in (
        "no-quant自然変動",
        "量子化変形（最小Tail）",
        "mul感度",
        "source集中",
        "timestep偏り",
    ):
        assert label in rendered
    assert "小さい＝信号が均一 ／ 大きい＝元々揺れやすい" in rendered
    assert "小さい＝no_quantに近い ／ 大きい＝最良mulでもTailが残る" in rendered
    assert "小さい＝mulに頑健 ／ 大きい＝mul選びが重要" in rendered
    assert "小さい＝複数sourceへ分散 ／ 大きい＝一部sourceへ集中" in rendered
    assert "小さい＝timestep間で均一 ／ 大きい＝特定帯へ偏る" in rendered
    assert "Tailはどの画像グループに偏っているか" in rendered
    assert "sourceはTOML内の画像グループ" in rendered
    assert "1 sourceを計算上外したときの最大Tail低下" in rendered
    assert "高い＝画質不良やsource削除推奨ではありません" in rendered
    assert "timestep帯ごとの変形" in rendered
    assert "値が小さいほどno_quantに近く、大きいほどその帯で量子化による変形が強くなります" in rendered
    assert "Body／Tailは上のグラフと同じ値です" in rendered
    assert "QA通過は測定が有効という意味で、画質保証ではありません" in rendered
    assert "この診断で分かること" in rendered
    assert "この診断だけでは分からないこと" in rendered
    assert 'href="report.html"' in rendered
    assert 'href="technical_report.html"' in rendered
    assert "最終生成画質のbest mul" in rendered
    assert "固定mulごとの局所的な勾配変形" in rendered


def test_beginner_report_refuses_cross_dataset_input() -> None:
    model = _model()
    model["datasets"] = [model["datasets"][0], dict(model["datasets"][0])]
    with pytest.raises(ValueError, match="exactly one dataset"):
        render_beginner_report(model)


def test_backfill_tool_writes_beginner_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / "practical_report.json").write_text(
        json.dumps(_model(), ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.render_dq_beginner_report.render_practical_report",
        lambda _model: "<html>refreshed practical report</html>",
    )
    output = render_existing_run(run_dir, refresh_practical_report=True)
    assert output == (run_dir / "beginner_report.html").resolve()
    assert output.is_file()
    assert (run_dir / "report.html").is_file()
    rendered = output.read_text(encoding="utf-8")
    assert "Example dataset" in rendered
    assert "Safety/Fidelity ≠ 最終画質Utility" in rendered
    assert "refreshed practical report" in (run_dir / "report.html").read_text(
        encoding="utf-8"
    )
