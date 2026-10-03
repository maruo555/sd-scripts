from __future__ import annotations

import copy
import hashlib
import json
import re
from urllib.parse import unquote

import pytest

from dq_profile.dataset_diagnostics import SCHEMA
from dq_profile.v24_beginner_report import render_beginner_report
from dq_profile.v24_practical_report import render_report
from dq_profile.v24_report_overview import attach_report_overview, overview_anchor, render_overview
from tests.test_dq_profile_gradient_support import _model
from tools.rebuild_dq_gradient_report import rebuild


def _sidecar(directory, model, *, pre=.3, post=.246):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {"schema_version": SCHEMA, "mode": "warmup", "loss_metric": "raw_mse",
                     "muls": [c["range_mul"] for c in model["datasets"][0]["candidate_cards"]]},
        "all": {"loss_pre": pre, "loss_post": post, "improvement_rel": 999,
                "paired_images": 3, "measured_images": 3, "inventory_images": 4},
    }
    (directory / "dataset_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (directory / "dataset_report.html").write_text("<!doctype html><body>fixture</body>", encoding="utf-8")
    return payload


@pytest.mark.parametrize("pre,post,expected", [(.3, .246, "18.0%"), (.3, .36, "誤差増加"), (.3, .3, "変化なし"), (0., 0., "減少率は未算出")])
def test_learning_response_uses_ratio_of_saved_means_and_keeps_model_unchanged(tmp_path, pre, post, expected):
    model = _model()
    before = copy.deepcopy(model)
    _sidecar(tmp_path / "data_diagnostics", model, pre=pre, post=post)
    updated = attach_report_overview(model, tmp_path / "data_diagnostics", tmp_path)
    assert model == before
    response = updated["datasets"][0]["warmup_response"]
    assert response["status"] == "available" and response["selector_input"] is False
    assert response["improvement_rel"] != 999
    markup = render_overview(updated["datasets"][0])
    assert expected in markup and "3 / 4枚" in markup
    assert 'class="overview-before-after"' in markup
    assert 'href="data_diagnostics/dataset_report.html"' in markup
    for key in ("warmup_response", "dataset_diagnostics_href"):
        updated["datasets"][0].pop(key)
    assert updated == before


@pytest.mark.parametrize("change", ["missing", "invalid_json", "wrong_grid", "local", "missing_pre", "bad_shape", "nonfinite"])
def test_missing_or_unusable_baseline_only_affects_left_card(tmp_path, change):
    model = _model()
    data_dir = tmp_path / "data_diagnostics"
    payload = _sidecar(data_dir, model)
    if change == "wrong_grid":
        payload["manifest"]["muls"] = [9.9]
    if change == "local":
        payload["manifest"]["mode"] = "local"
    if change == "missing_pre":
        payload["all"]["loss_pre"] = None
    if change == "bad_shape":
        payload["manifest"] = []
    if change == "nonfinite":
        payload["all"]["loss_post"] = float("inf")
    path = data_dir / "dataset_summary.json"
    path.write_text("{" if change == "invalid_json" else json.dumps(payload), encoding="utf-8")
    if change == "missing":
        path.unlink()
    updated = attach_report_overview(model, data_dir, tmp_path)
    assert updated["datasets"][0]["warmup_response"]["status"] == "unavailable"
    assert updated["datasets"][0]["candidate_cards"] == model["datasets"][0]["candidate_cards"]
    markup = render_beginner_report(updated)
    assert "0.80〜1.30" in markup and "元の勾配方向への成分" in markup


def test_tail_range_uses_all_displayable_candidates_and_marks_unsafe_and_missing():
    dataset = _model()["datasets"][0]
    dataset["candidate_cards"][0].update(tail=8., hard_safety_pass=False, relative_status="excluded")
    dataset["candidate_cards"][1]["tail"] = None
    dataset["edge_unresolved"] = True
    markup = render_overview(dataset)
    assert "0.90〜8.00" in markup and "2 / 3候補" in markup
    assert "HardSafety不合格 1候補" in markup and "Tail未算出 1候補" in markup
    assert "測定範囲の端は未解決" in markup


@pytest.mark.parametrize("render", [render_beginner_report, render_report])
def test_cards_follow_both_graphs_before_details_and_guides_have_destinations(tmp_path, render):
    model = _model()
    _sidecar(tmp_path / "data_diagnostics", model)
    model = attach_report_overview(model, tmp_path / "data_diagnostics", tmp_path)
    dataset = model["datasets"][0]
    dataset["source_localization"] = {"valid": True, "reference_profile": {
        "range_mul": 3.15, "top_source_alias": '<img src=x onerror="bad">', "top_source_share": .4,
    }}
    dataset["timestep_rows"] = [{"candidate": "mul_3.15", "range_mul": 3.15,
                                 "timestep_bin": 3, "source_balanced_q95_relative_distance": .8}]
    markup = render(model)
    assert markup.count('class="report-overview"') == 1
    stack = markup.split('class="gradient-curve-stack"', 1)[1]
    assert stack.index('class="report-overview"') > [m.start() for m in re.finditer('</svg>', stack)][1]
    assert stack.index('class="report-overview"') < stack.index('class="curve-values"')
    assert '<img src=x onerror="bad">' not in markup
    assert '&lt;img src=x onerror=&quot;bad&quot;&gt;' in markup
    for kind in ("source", "timestep"):
        anchor = overview_anchor(dataset, kind)
        assert f'href="#{anchor}"' in markup and f'id="{anchor}"' in markup


def test_cpu_rebuild_reads_sidecar_preserves_sources_and_resolves_detail_link(tmp_path):
    source, output = tmp_path / "saved run", tmp_path / "new report"
    source.mkdir()
    model = _model()
    (source / "practical_report.json").write_text(json.dumps(model), encoding="utf-8")
    (source / "local_selection.json").write_text('{"unchanged": true}')
    _sidecar(source / "data_diagnostics", model)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    rebuild(source, output)
    updated = json.loads((output / "practical_report.json").read_text(encoding="utf-8"))
    dataset = updated["datasets"][0]
    assert dataset["warmup_response"]["improvement_rel"] == pytest.approx(.18)
    assert (output / unquote(dataset["dataset_diagnostics_href"])).resolve() == source / "data_diagnostics/dataset_report.html"
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    # A moved report retains its compact measured values, without a broken link.
    moved = attach_report_overview(updated, tmp_path / "absent", output)
    assert moved["datasets"][0]["warmup_response"] == dataset["warmup_response"]
    assert moved["datasets"][0]["dataset_diagnostics_href"] is None


def test_production_promotion_refreshes_cards_links_and_only_report_hashes(tmp_path):
    from dq_profile.diagnostic_report import promote_dataset_report
    from tests.test_dq_dataset_diagnostics import write_fixture
    profile, run = tmp_path / "profile", tmp_path / "run"
    run.mkdir()
    payload = write_fixture(profile / "data_diagnostics", 4)
    model = _model()
    cards = model["datasets"][0]["candidate_cards"]
    model["datasets"][0]["candidate_cards"] = [
        {**copy.deepcopy(cards[0]), "range_mul": mul, "candidate": f"mul_{mul:.2f}"}
        for mul in payload["manifest"]["muls"]
    ]
    (run / "practical_report.json").write_text(json.dumps(model), encoding="utf-8")
    (run / "technical_report.html").write_text("<html><body>technical</body></html>")
    manifest = {"inputs": {"unchanged": True}, "selection_rule_sha256": "preserve", "reports": {}}
    (run / "analysis_manifest.json").write_text(json.dumps(manifest))
    selection = {"selection_valid": True, "credible_muls": [3.15]}
    promote_dataset_report(profile, run, selection)
    updated = json.loads((run / "practical_report.json").read_text(encoding="utf-8"))
    assert updated["datasets"][0]["warmup_response"]["status"] == "available"
    assert updated["datasets"][0]["warmup_response"]["loss_pre"] == payload["all"]["loss_pre"]
    assert updated["datasets"][0]["dataset_diagnostics_href"] == "data_diagnostics/dataset_report.html"
    assert selection == {"selection_valid": True, "credible_muls": [3.15]}
    result = json.loads((run / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in result.items() if k != "reports"} == {k: v for k, v in manifest.items() if k != "reports"}
    for name in ("report.html", "beginner_report.html", "technical_report.html"):
        assert 'href="data_diagnostics/dataset_report.html"' in (run / name).read_text(encoding="utf-8")
    for record in result["reports"].values():
        from pathlib import Path
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()


def test_standalone_renderer_refreshes_only_written_report_hashes(tmp_path):
    from tools.render_dq_beginner_report import render_existing_run
    model = _model()
    (tmp_path / "practical_report.json").write_text(json.dumps(model), encoding="utf-8")
    _sidecar(tmp_path / "data_diagnostics", model)
    manifest = {"inputs": {"unchanged": True}, "reports": {"technical": {"sha256": "preserve"}}}
    path = tmp_path / "analysis_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    render_existing_run(tmp_path, refresh_practical_report=True)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["inputs"] == manifest["inputs"]
    assert updated["reports"]["technical"] == manifest["reports"]["technical"]
    for key, name in (("primary", "report.html"), ("beginner", "beginner_report.html")):
        assert updated["reports"][key]["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert "18.0%" in (tmp_path / name).read_text(encoding="utf-8")
