from __future__ import annotations

import copy
import html
import json
import re

import pytest

from dq_profile.v24_beginner_report import render_beginner_report
from dq_profile.v24_gradient_support import attach_gradient_support, build_gradient_support
from dq_profile.v24_image_breakdown import render_image_breakdown
from dq_profile.v24_practical_report import render_report
from tests.test_dq_profile_gradient_support import _model, _row, _support


def _sample(image="image-a", *, d=2., original=.02, repeat=0, **changes):
    row = _row(image, d=d, grad_diff_norm=d * original)
    row.update(grad_norm_noquant=original, quant_repeat=repeat, **changes)
    return row


def _images(rows, candidates=None):
    return _support(rows, candidates)["candidates"][0]["image_breakdown"]["images"]


def _pop_breakdown(support):
    result = copy.deepcopy(support)
    for candidate in result["candidates"]:
        candidate.pop("image_breakdown")
    return result


def test_maximum_keeps_norms_from_same_observation_not_independent_maxima():
    rows = [_sample(), _sample(d=1., original=.8, repeat=1), _sample("image-b", original=.2)]
    before = copy.deepcopy(rows)
    a, b = _images(rows)
    assert rows == before
    assert a["relative_gradient_distance"] == b["relative_gradient_distance"] == 2.
    assert (a["grad_norm_noquant"], a["grad_diff_norm"]) == (.02, .04)
    assert (b["grad_norm_noquant"], b["grad_diff_norm"]) == (.2, .4)
    assert a["measurement"] == dict(timestep_bin=0, noise_replica=0, quant_repeat=0)
    assert a["observation_count"] == 2


def test_all_bins_replicas_and_repeats_participate_and_ties_are_stable():
    rows = [_sample(d=2. if b == 3 else .1, repeat=q, timestep_bin=b, noise_replica=n)
            for b in range(4) for n in range(2) for q in range(2)]
    result = _images(rows)[0]
    assert result["observation_count"] == result["expected_observation_count"] == 16
    assert result["measurement"] == dict(timestep_bin=3, noise_replica=0, quant_repeat=0)
    assert _images(rows) == _images(list(reversed(rows))) == _images(rows + [rows[-1]])


@pytest.mark.parametrize("change,reason", [
    ({"grad_diff_norm": None}, "missing_or_invalid_gradient_difference_norm"),
    ({"grad_diff_norm": -1.}, "missing_or_invalid_gradient_difference_norm"),
    ({"grad_diff_norm": float("nan")}, "missing_or_invalid_gradient_difference_norm"),
    ({"grad_diff_norm": float("inf")}, "missing_or_invalid_gradient_difference_norm"),
    ({"grad_diff_norm": 2.}, "saved_distance_norm_mismatch"),
])
def test_invalid_selected_difference_only_disables_affected_image_not_existing_curves(change, reason):
    rows = [_sample(), _sample("image-b", original=.2)]
    baseline = _support(rows)
    rows[0].update(change)
    actual = _support(rows)
    assert _pop_breakdown(actual) == _pop_breakdown(baseline)
    a, b = actual["candidates"][0]["image_breakdown"]["images"]
    assert a["status"] == "unavailable" and a["relative_gradient_distance"] is None
    assert reason in a["reasons"]
    assert b["status"] == "available"
    json.dumps(actual, allow_nan=False)


def test_invalid_d_does_not_disappear_from_maximum_and_other_images_remain_visible():
    rows = [_sample(), _sample(d=1., repeat=1), _sample("image-b")]
    rows[1]["relative_gradient_distance"] = None
    a, b = _images(rows)
    assert a["status"] == "unavailable" and "missing_or_invalid_relative_gradient_distance" in a["reasons"]
    assert b["status"] == "available"


def test_nonselected_difference_is_not_required_to_establish_maximum_d():
    rows = [_sample(), _sample(d=1., repeat=1, grad_diff_norm=None)]
    assert _images(rows)[0]["status"] == "available"


@pytest.mark.parametrize("original", [0., 1e-13, -1., float("inf")])
def test_invalid_reference_never_produces_bar_or_fallback_denominator(original):
    result = _images([_sample(original=original)])[0]
    assert result["status"] == "unavailable"
    assert result["grad_norm_noquant"] is result["grad_diff_norm"] is result["measurement"] is None


def test_roundoff_close_to_zero_and_true_zero_difference_remain_available():
    for difference in (0., 1e-8):
        row = _images([_sample(d=0., original=1., grad_diff_norm=difference)])[0]
        assert row["status"] == "available"
        assert row["grad_diff_norm"] == difference


def test_conflicting_duplicate_is_unavailable_in_both_orders():
    rows = [_sample(), _sample(grad_diff_norm=.05)]
    for order in (rows, list(reversed(rows))):
        row = _images(order)[0]
        assert row["status"] == "unavailable"
        assert "conflicting_duplicate_probe_key" in row["reasons"]


def test_missing_image_in_one_candidate_is_not_silently_removed():
    cards = [{"candidate": name, "range_mul": 1.37, "hard_safety_pass": True} for name in ("opaque", "other")]
    rows = [_sample(), _sample("image-b"), _sample(candidate="other")]
    result = _support(rows, cards)["candidates"]
    assert result[0]["image_breakdown"]["available_image_count"] == 2
    assert result[1]["image_breakdown"]["image_count"] == 2
    assert result[1]["image_breakdown"]["available_image_count"] == 0
    assert result[1]["image_breakdown"]["images"][1]["observation_count"] == 0


def _display_model(count=33):
    model = _model()
    rows = []
    for card in model["datasets"][0]["candidate_cards"]:
        for index in range(count):
            rows.append(_sample(f"image-{index:02d}.png", original=1. + index / 100,
                                candidate=card["candidate"], d=card["body"], range_mul=card["range_mul"]))
    support = build_gradient_support(rows, model["datasets"][0]["candidate_cards"])
    model = attach_gradient_support(model, support)
    model["datasets"][0]["image_count_total"] = count + 2
    return model


@pytest.mark.parametrize("render", [render_beginner_report, render_report])
def test_all_images_once_per_mul_below_cards_and_before_existing_details(render):
    model = _display_model()
    before = copy.deepcopy(model)
    markup = render(model)
    assert model == before
    assert markup.count('class="image-gradient-breakdown"') == 1
    assert markup.count('class="ig-row"') == 33 * 3
    assert markup.index('class="report-overview"') < markup.index('class="image-gradient-breakdown"') < markup.index('class="curve-values"')
    assert markup.count('class="ig-panel"') == 3
    assert '保存ログの測定画像 33枚 / 全 35枚' in markup
    assert 'image-32.png' in markup and '全測定画像を表示' in markup
    assert '最大dはその画像の典型的な反応とは限らず' in markup
    assert 'data-selector-input="false"' in markup


def test_norm_scale_shared_across_images_and_all_mul_candidates():
    dataset = _display_model(2)["datasets"][0]
    markup = render_image_breakdown(dataset)
    maximum = float(re.search(r'data-scale-max="([^"]+)"', markup).group(1))
    expected = max(row[key] for card in dataset["candidate_cards"]
                   for row in card["gradient_curve_support"]["image_breakdown"]["images"]
                   for key in ("grad_norm_noquant", "grad_diff_norm"))
    assert maximum == expected
    widths = re.findall(r'ig-bar ig-original" style="width:([\d.e+-]+)%', markup)
    assert widths[:2] == widths[2:4] == widths[4:6]


def test_image_identity_is_escaped_in_label_details_and_attribute():
    dataset = _display_model(1)["datasets"][0]
    key = 'folder/<script>alert("x")</script>&.png'
    for card in dataset["candidate_cards"]:
        card["gradient_curve_support"]["image_breakdown"]["images"][0]["image_key"] = key
    markup = render_image_breakdown(dataset)
    assert key not in markup and '<script>alert(' not in markup
    assert html.escape(key, quote=True) in markup


def test_legacy_model_or_missing_scalars_leaves_existing_report_renderable():
    model = _model()
    assert '未生成：同じrunのsample CSV' in render_beginner_report(model)
    cards = model["datasets"][0]["candidate_cards"]
    rows = [dict(_sample(candidate=c["candidate"], d=c["body"], range_mul=c["range_mul"]), grad_diff_norm=None) for c in cards]
    attached = attach_gradient_support(model, build_gradient_support(rows, cards))
    assert all(c["gradient_curve_support"]["status"] == "available" for c in attached["datasets"][0]["candidate_cards"])
    markup = render_image_breakdown(attached["datasets"][0])
    assert 'ig-bar ig-original' not in markup
    assert markup.count('class="ig-missing"') == 3
    assert 'missing_or_invalid_gradient_difference_norm' in markup


def test_small_nonzero_values_use_scientific_not_false_zero_labels():
    model = _model()
    cards = model["datasets"][0]["candidate_cards"]
    rows = [_sample(candidate=c["candidate"], original=1e-8, d=c["body"], range_mul=c["range_mul"]) for c in cards]
    model = attach_gradient_support(model, build_gradient_support(rows, cards))
    assert '>1e-08<' in render_image_breakdown(model["datasets"][0])
