from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
import re

import numpy as np
import pytest

from dq_profile.v24_acceptance import _cluster_quantile, _cluster_values, _index_rows, _body_tail
from dq_profile.v24_gradient_support import build_gradient_support, attach_gradient_support
from dq_profile.v24_practical_report import _curve_svg
from tests.test_dq_profile_v24_acceptance import _fixture, _analyze
from tests.test_dq_profile_v24_practical_report import _candidate, _detail
from dq_profile.v24_practical_report import build_single_dataset_report_model
from tools.rebuild_dq_gradient_report import rebuild


def _row(image="a", source="A", *, candidate="opaque", ratio=1.0, cosine=1.0, d=0.1, **extra):
    return dict(record_type="sample", candidate=candidate, range_mul=1.37,
                image_key=image, source_group=source, timestep_bin=0, noise_replica=0,
                quant_repeat=0, grad_norm_noquant=1., grad_norm_candidate=ratio,
                gradient_norm_ratio=ratio, gradient_cosine=cosine,
                relative_gradient_distance=d, gradient_topology_matches=True, **extra)


def _support(rows, candidates=None):
    return build_gradient_support(rows, candidates or [{"candidate": "opaque", "range_mul": 1.37, "hard_safety_pass": True}])


def _one(rows):
    return _support(rows)["candidates"][0]


def _model():
    return build_single_dataset_report_model(dataset_id="SYN", candidate_rows=[
        _candidate(2.70, 1.2, 1.3), _candidate(3.15, .7, .8), _candidate(3.45, .8, .9),
    ], detail=_detail())


def test_existing_body_and_selection_unchanged_and_rng_untouched():
    fixture = _fixture()
    result = _analyze(fixture, iterations=12)
    before = json.dumps((fixture, result), sort_keys=True, default=lambda v: v.tolist())
    random_state, numpy_state = random.getstate(), np.random.get_state()
    support = build_gradient_support(fixture[1], result["score_rows"])
    assert json.dumps((fixture, result), sort_keys=True, default=lambda v: v.tolist()) == before
    assert random.getstate() == random_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    for row, score in zip(support["candidates"], result["score_rows"]):
        assert row["status"] == "available"
        assert row["d_p95_check"] == score["local_body"]
        raw = [r for r in fixture[1] if r["candidate"] == row["candidate"]]
        groups, bins = _index_rows(raw, timestep_bins=4)
        assert row["d_p50"] == _cluster_quantile(groups, sorted(groups), "relative_gradient_distance", .50)
        assert _body_tail(groups, bins, sorted(groups), field="relative_gradient_distance", timestep_bins=4)["body"] == row["d_p95_check"]


def test_source_mass_repeat_setting_and_order_do_not_change_quantiles():
    rows = [_row("a", "A", d=0.), _row("b", "A", d=2.), _row("c", "A", d=3.), _row("d", "B", d=10.), _row("e", "C", d=1.)]
    initial = _one(rows)
    assert initial["d_p50"] == 2.  # step quantile, not interpolated median
    expanded = [dict(r, quant_repeat=q, num_repeats=999) for r in rows[:3] for q in range(5)] + rows[3:]
    for candidate_rows in (expanded, list(reversed(expanded)), [dict(r, num_repeats=1) for r in expanded]):
        actual = _one(candidate_rows)
        for key in ("d_p50", "d_p95_check", "parallel_component_p50", "parallel_component_p05"):
            assert actual[key] == initial[key]
    _, weights = _cluster_values({"A": expanded[:15]}, ["A"], "relative_gradient_distance")
    assert sum(weights) == pytest.approx(1.)


def test_exact_cdf_boundary_keeps_existing_floating_point_quantile_contract():
    # Existing np.cumsum can select the next observation at an exact CDF
    # boundary (ten 0.1 weights do not sum to exactly 1). Changing that would
    # change the supplied real-run comparison values, so preserve the helper.
    rows = [_row("a", "A", d=0.), _row("b", "A", d=2.), _row("c", "B", d=10.)]
    expanded = [dict(r, quant_repeat=q) for r in rows[:2] for q in range(5)] + rows[2:]
    for raw in (rows, expanded):
        groups, _ = _index_rows(raw, timestep_bins=1)
        assert _one(raw)["d_p50"] == _cluster_quantile(groups, sorted(groups), "relative_gradient_distance", .5)


@pytest.mark.parametrize("gx,gy,expected", [(1, 0, 1), (.5, 0, .5), (1, 1, 1), (-1, 0, -1), (2, 0, 2), (0, 0, 0)])
def test_direction_vectors_and_norm_fallback(gx, gy, expected):
    norm = math.hypot(gx, gy)
    row = _row(ratio=norm, cosine=gx/norm if norm else float("nan"))
    row.pop("gradient_norm_ratio")
    result = _one([row])
    assert result["parallel_component_p50"] == pytest.approx(expected)
    assert result["parallel_component_p05"] == pytest.approx(expected)


def test_multiply_each_row_before_quantiles_and_pool_bins():
    rows = [_row("a", ratio=2, cosine=.2), _row("b", ratio=.5, cosine=1), _row("c", ratio=1, cosine=.6)]
    rows[1]["timestep_bin"] = 1
    value = _one(rows)
    assert value["parallel_component_p50"] == .5  # not median(ratio)*median(cosine)=.6
    assert value["parallel_component_p05"] == .4


@pytest.mark.parametrize("field,value,reason,both", [
    ("relative_gradient_distance", None, "missing_or_invalid_relative_gradient_distance", False),
    ("gradient_cosine", float("inf"), "missing_or_invalid_gradient_cosine", False),
    ("gradient_norm_ratio", float("nan"), "missing_or_invalid_gradient_norm_ratio", False),
    ("grad_norm_noquant", 1e-13, "reference_near_zero", True),
    ("grad_norm_noquant", 0, "reference_near_zero", True),
    ("grad_norm_noquant", None, "missing_or_invalid_reference_norm", True),
    ("grad_norm_candidate", -1, "invalid_candidate_norm", True),
    ("gradient_topology_matches", "False", "gradient_topology_mismatch_or_invalid", True),
])
def test_invalid_rows_invalidate_whole_candidate_metric(field, value, reason, both):
    rows = [_row("a"), _row("b")]
    rows[1][field] = value
    result = _one(rows)
    reasons = result["d_reasons"] + result["parallel_component_reasons"]
    assert reason in reasons
    if both:
        assert result["d_p50"] is None and result["parallel_component_p50"] is None
    elif field == "relative_gradient_distance":
        assert result["d_p50"] is None and result["parallel_component_p50"] == 1
    else:
        assert result["parallel_component_p50"] is None and result["d_p50"] == .1


def test_exact_duplicate_has_no_vote_conflict_is_not_first_or_last_wins():
    rows = [_row("a", d=.1), _row("b", d=.9)]
    duplicated = _one(rows + [rows[0]] * 10)
    assert duplicated["observation_count"] == 2
    assert duplicated["d_p50"] == _one(rows)["d_p50"]
    assert duplicated["duplicate_row_count"] == 10
    for bad in (dict(rows[0], relative_gradient_distance=2), dict(rows[0], source_group="B")):
        for data in (rows+[bad], [bad]+rows):
            result = _one(data)
            assert result["status"] == "unavailable"
            assert "conflicting_duplicate_probe_key" in result["d_reasons"]


@pytest.mark.parametrize("mutate,reason", [
    (lambda rows: rows.pop(), "candidate_probe_key_mismatch"),
    (lambda rows: rows[-1].update(source_group="different"), "image_source_mismatch"),
    (lambda rows: rows[-1].update(range_mul=999), "candidate_mul_mismatch"),
    (lambda rows: rows[-1].update(grad_norm_noquant=2), "reference_norm_mismatch_between_candidates"),
])
def test_saved_candidate_mapping_and_support(mutate, reason):
    rows = [_row("a"), _row("b"), _row("a", candidate="other"), _row("b", candidate="other")]
    candidates = [{"candidate": n, "range_mul": 1.37, "hard_safety_pass": True} for n in ("opaque", "other")]
    mutate(rows)
    result = _support(rows, candidates)["candidates"]
    assert any(reason in r["d_reasons"] and r["d_p50"] is None for r in result)


@pytest.mark.parametrize("reference_norm", [1., 2.], ids=["matching", "mismatched"])
def test_unsafe_reference_check_does_not_invalidate_other_candidates(reference_norm):
    candidates = [
        {"candidate": "safe", "range_mul": 2., "hard_safety_pass": True},
        {"candidate": "unsafe_valid", "range_mul": 3., "hard_safety_pass": False},
        {"candidate": "unsafe_checked", "range_mul": 4., "hard_safety_pass": False},
    ]
    rows = [dict(_row(candidate=c["candidate"], d=0.), range_mul=c["range_mul"]) for c in candidates]
    baseline = _support(rows, candidates)["candidates"]
    rows[-1].update(grad_norm_noquant=reference_norm, gradient_norm_ratio=1. / reference_norm,
                    relative_gradient_distance=abs(1. - reference_norm) / reference_norm)
    actual = _support(rows, candidates)["candidates"]
    assert actual[:2] == baseline[:2]
    assert all(c["status"] == "available" for c in actual[:2])
    if reference_norm == 1.:
        assert actual == baseline
    else:
        checked = actual[-1]
        assert checked["status"] == "unavailable"
        assert all(checked[key] is None for key in ("d_p50", "parallel_component_p50", "parallel_component_p05"))
        assert checked["d_reasons"] == checked["parallel_component_reasons"] == ["reference_norm_mismatch_between_candidates"]
    assert [c["hard_safety_pass"] for c in candidates] == [True, False, False]


def test_mixed_context_and_saved_body_mismatch_are_unavailable():
    assert _one([_row("a", snapshot="pre"), _row("b", snapshot="post")])["status"] == "unavailable"
    result = _support([_row()], [{"candidate": "opaque", "range_mul": 1.37, "body": 99}])["candidates"][0]
    assert result["d_reasons"] == ["saved_body_population_mismatch"]
    assert result["parallel_component_p50"] is None


def test_legacy_csv_without_sidecars_and_aggregate_only_are_distinct():
    row = _row()
    row.pop("gradient_topology_matches")
    row.pop("source_group")
    assert _one([row])["status"] == "available"
    assert _one([row])["topology_unrecorded_count"] == 1
    assert _one([dict(row, record_type="summary")])["d_reasons"] == ["no_sample_rows"]


def test_support_namespaces_leave_all_existing_model_values_unchanged():
    model = _model()
    before = copy.deepcopy(model)
    support = build_gradient_support([], model["datasets"][0]["candidate_cards"])
    attached = attach_gradient_support(model, support)
    assert model == before
    attached["datasets"][0].pop("gradient_curve_support")
    for card in attached["datasets"][0]["candidate_cards"]:
        card.pop("gradient_curve_support")
    assert attached == before


def test_svg_alignment_gaps_axis_extension_overflow_and_existing_markers():
    cards = _model()["datasets"][0]["candidate_cards"]
    for i, card in enumerate(cards):
        card["gradient_curve_support"] = {"d_p50": None if i == 1 else 5.1, "parallel_component_p50": None if i == 1 else 2., "parallel_component_p05": None if i == 1 else -1.}
    cards[0]["hard_safety_pass"] = False
    markup = _curve_svg(cards, fixed_y_max=4., edge_direction="upper")
    svgs = [ET.fromstring(s) for s in re.findall(r"<svg\b.*?</svg>", markup, flags=re.S)]
    assert len(svgs) == 2
    assert svgs[0].attrib["data-y-max"] == "4.000000"
    assert float(svgs[1].attrib["data-y-min"]) < -1
    assert float(svgs[1].attrib["data-y-max"]) > 2
    positions = [[g.find("./rect").attrib["x"] for g in svg.findall(".//g[@class='curve-target']")] for svg in svgs]
    assert positions[0] == positions[1]
    for svg, metric in ((svgs[0], "d_p50"), (svgs[1], "parallel_component_p50")):
        lines = svg.findall(f".//polyline[@data-metric='{metric}']")
        assert len(lines) == 2 and all(len(line.attrib["points"].split()) == 1 for line in lines)
    assert "軸拡張" in markup and "5.1" in markup and "Hard unsafe" in markup
    assert "stroke-width=\"7\" opacity=\".16\"" in markup  # existing CI
    assert "上側未解決" in markup and "preset線" in markup and "基準 1.0" in markup
    between = markup.split('</svg>', 1)[1].split('<svg', 1)[0]
    assert '<table' not in between and '<details' not in between


def test_rebuild_is_cpu_only_and_preserves_raw_selection_and_hash_inputs(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    model = _model()
    (source / "practical_report.json").write_text(json.dumps(model), encoding="utf-8")
    rows = [_row(candidate=c["candidate"], d=c["body"]) for c in model["datasets"][0]["candidate_cards"]]
    for row, card in zip(rows, model["datasets"][0]["candidate_cards"]):
        row["range_mul"] = card["range_mul"]
    raw = source / "raw_gradient_tail.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    (source / "summary.json").write_text(json.dumps({"local_gradient_tail_sha256": digest}))
    (source / "local_selection.json").write_text('{"unchanged": true}')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    support = rebuild(source, destination)
    assert all(c["status"] == "available" for c in support["candidates"])
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert (destination / "beginner_report.html").is_file()
    # A fresh interpreter forbids torch/training imports and selector calls.
    code = f'''import sys
sys.modules['torch'] = None
from dq_profile import v24_acceptance
v24_acceptance.analyze_local_profile = lambda **kw: (_ for _ in ()).throw(AssertionError('reselection'))
from tools.rebuild_dq_gradient_report import rebuild
from pathlib import Path
rebuild(Path({str(source)!r}), Path({str(destination)!r}))
assert 'dq_profile.trainer_runtime' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    mismatched = rebuild(source, destination)
    assert all("raw_csv_hash_mismatch_with_saved_analysis" in c["d_reasons"] for c in mismatched["candidates"])


def test_rebuild_aggregate_only_preserves_report_and_promote_includes_support(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    (source / "practical_report.json").write_text(json.dumps(_model()), encoding="utf-8")
    support = rebuild(source, destination)
    assert all(c["status"] == "unavailable" for c in support["candidates"])
    assert "元の勾配方向への成分" in (destination / "beginner_report.html").read_text(encoding="utf-8")
    from dq_profile.production_runner import promote_analysis
    for name in ("technical_report.html", "summary.json"):
        (destination / name).write_text("{}")
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    names = promote_analysis(promoted, destination)
    assert "gradient_curve_support.json" in names
    assert (promoted / "gradient_curve_support.json").read_bytes() == (destination / "gradient_curve_support.json").read_bytes()


@pytest.mark.parametrize("in_place", [True, False])
def test_rebuild_refreshes_report_manifest_only(tmp_path, in_place):
    source = tmp_path / "source"
    source.mkdir()
    destination = source if in_place else tmp_path / "out"
    (source / "practical_report.json").write_text(json.dumps(_model()), encoding="utf-8")
    protected = {"summary.json": b"{}", "local_selection.json": b'{"selection": "unchanged"}',
                 "raw_gradient_tail.csv": b"record_type\n"}
    for name, content in protected.items():
        (source / name).write_bytes(content)
    reports = {}
    for key, name in (("primary", "report.html"), ("beginner", "beginner_report.html"), ("technical", "technical_report.html")):
        path = source / name
        path.write_bytes(b"old report\n")
        reports[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "scope": "saved-scope"}
    manifest = {"inputs": {"gradient_tail.csv": {"path": str(source / "raw_gradient_tail.csv"),
                "sha256": hashlib.sha256(protected["raw_gradient_tail.csv"]).hexdigest()}},
                "selection_rule_sha256": "unchanged", "reports": reports}
    (source / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    rebuild(source, destination)
    updated = json.loads((destination / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in updated.items() if k != "reports"} == {k: v for k, v in manifest.items() if k != "reports"}
    for key, record in updated["reports"].items():
        path = Path(record["path"])
        assert path.parent == destination.resolve()
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["scope"] == "saved-scope"
    assert (destination / "technical_report.html").read_bytes() == original["technical_report.html"]
    for name, content in protected.items():
        assert (source / name).read_bytes() == content
    if not in_place:
        assert {p.name: p.read_bytes() for p in source.iterdir()} == original


@pytest.mark.parametrize("body", [3.8, 4.6])
def test_overflow_title_distinguishes_estimate_from_ci(body):
    cards = _model()["datasets"][0]["candidate_cards"]
    cards[0].update(body=body, body_ci_high=5.2)
    markup = _curve_svg(cards, fixed_y_max=4.)
    upper = ET.fromstring(re.search(r"<svg\b.*?</svg>", markup, flags=re.S).group())
    titles = [p.find("title").text for p in upper.findall("./polygon")]
    assert len(titles) == 1
    assert "body CI上限 5.2" in titles[0]
    assert (f"body {body}" in titles[0]) == (body > 4.)


@pytest.mark.parametrize("ragged_csv", [False, True])
def test_normal_analysis_outputs_support_without_changing_selection(tmp_path, monkeypatch, ragged_csv):
    """Run the real analysis entry point on synthetic CPU scalars only."""
    from argparse import Namespace
    from tools import analyze_dq_v24_local as analysis
    profile, output = tmp_path / "profile", tmp_path / "analysis"
    profile.mkdir()
    summary, rows = _fixture()
    (profile / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (profile / "source_manifest.json").write_text(json.dumps({"source_contract": {"sha256": "cpu-fixture"}}))
    (profile / "local_natural_gradient.csv").write_text("image_key\n")
    with (profile / "gradient_tail.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if ragged_csv:
        raw = profile / "gradient_tail.csv"
        lines = raw.read_text(encoding="utf-8").splitlines()
        lines[1] += ","
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(analysis, "parse_args", lambda: Namespace(profile_dir=profile, output_dir=output, dataset_id="SYN", iterations=12, seed=2401))
    assert analysis.main() == 0
    support = json.loads((output / "gradient_curve_support.json").read_text(encoding="utf-8"))
    for candidate in support["candidates"]:
        if ragged_csv and candidate["candidate"] == rows[0]["candidate"]:
            assert candidate["d_p50"] is None and candidate["parallel_component_p50"] is None
            assert "malformed_sample_columns" in candidate["d_reasons"]
            assert candidate["observation_count"] == candidate["expected_observation_count"]
        else:
            assert candidate["status"] == "available"
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
    preserved = {name: (output / name).read_bytes() for name in ("local_selection.json", "local_acceptance.csv", "summary.json", "source_bootstrap.csv")}
    # Simulate unavailable extra metrics. Existing decisions and hash inputs
    # must remain byte-for-byte identical, including in the ordinary entrypoint.
    original_builder = analysis.build_gradient_support
    monkeypatch.setattr(analysis, "build_gradient_support", lambda rows, cards, **kw: original_builder([], cards, **kw))
    assert analysis.main() == 0
    assert {name: (output / name).read_bytes() for name in preserved} == preserved
