import csv
from collections import Counter
from pathlib import Path

import pytest

from library.lora_run_schema import HEADERS, encode_cell
from library.training_run import TrainingRun
from library.training_settings import save_settings
from tools.lora_run_reader import RunReader
from tools.lora_ledger_core.features import INITIAL_METRICS, RunFeatures
from tools.lora_ledger_core.storage import Conflict


def write(run, kind, rows, **extra):
    path = run.directory / f"{kind}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS[kind])
        for row in rows:
            writer.writerow([encode_cell(k, row.get(k)) for k in HEADERS[kind]])
    run.set_artifact(kind, path.name, schema_version=1, status="complete", row_count=len(rows), **extra)
    return path


def source(root, n=5):
    run = TrainingRun.create(root, "test", 1)
    save_settings(run, {}, resolved={"dq": {"stat": "rms", "granularity": "channel", "rounding_mode": "stoch",
                                          "metric_definition_version": 1, "log_scope": "both",
                                          "auto": {"scope": "unet", "preset": "clip_rate_high"}}})
    gradient, dq, auto, rank, group, promote = [], [], [], [], [], []
    auto.append(dict(event_type="init", attempt_step=0, bits=8, scope="unet", target="delta"))
    for epoch in range(1, n+1):
        step = 10 * epoch
        gradient.append(dict(epoch=epoch, batch_index=0, loss=epoch, grad_scale=32))
        for scope, rms in (("unet", 2), ("te", 1)):
            dq.append(dict(epoch=epoch, attempt_step=step, target="delta", scope=scope, bits=8,
                           input_rms=rms, common_mul_after=epoch, controller_clip_rate_ema=.003,
                           log_scope_error_rms_ema=.1, log_scope_error_ratio_ema=.2))
        auto.append(dict(event_type="check", epoch=epoch, attempt_step=step, bits=8, target="delta", scope="unet",
                         auto_applied=epoch % 2 == 0, low_auto_decision="escape_to_mid" if epoch == 3 else None))
        for name, sat, top, energy in (("a", 0, .4, 0), ("b", .5, .6, 1), ("c", 1, .8, 3)):
            rank.append(dict(epoch=epoch, attempt_step=step, scope="unet", module=name,
                             rank_dim=4, rank_sat=sat, rank_top1=top, rank_energy=energy))
        for i, loss in enumerate((2, 4)):
            group.append(dict(epoch=epoch, attempt_step=step+i, group="a", loss=loss, group_ema_loss=9))
        promote.append(dict(epoch=epoch, row_type="decision", status="scored", selected_candidate_mode="ema",
                            raw_proxy_loss=100, selected_proxy_loss=99, promote_applied=epoch > 1, bank_size=3,
                            window_count=1))
        promote.append(dict(row_type="window_epoch", epoch=epoch, window_position=1, window_epoch=epoch))
    for kind, values in (("gradient", gradient), ("dq", dq), ("dq_auto", auto), ("rank", rank),
                         ("group_loss", group), ("promote", promote)):
        write(run, kind, values, **({"expected_modules_by_scope": {"unet": ["a","b","c"]}} if kind == "rank" else {}))
    return run


def value(rows, metric):
    return next(row for row in rows if row["metric_id"] == metric)


def test_six_streams_twenty_metrics_distinct_dq_scope_and_unweighted_rank(tmp_path, monkeypatch):
    run = source(tmp_path)
    reader = RunReader(run.directory)
    original = reader.stream
    calls = Counter()
    def stream(kind):
        calls[kind] += 1
        return original(kind)
    monkeypatch.setattr(reader, "stream", stream)
    data = RunFeatures.load(reader)
    rows = data.summarize()
    assert {r["metric_id"] for r in rows} == INITIAL_METRICS and len(INITIAL_METRICS) == 20
    assert value(rows, "loss.tail_median")["value"] == 5 and value(rows, "loss.tail_delta")["value"] == 1
    assert value(rows, "rank_sat_p95.tail_median")["value"] == pytest.approx(.95)
    assert value(rows, "rank_top1_p95.tail_median")["value"] == pytest.approx(.78)
    assert value(rows, "rank_energy.tail_median")["value"] == 4
    assert value(rows, "dq_error_rms.tail_median")["n_valid"] == 5
    assert sorted(r["value"] for r in rows if r["metric_id"] == "dq_input_rms.tail_median") == [1,2]
    assert value(rows, "dq_auto.records")["value"] == 5
    assert value(rows, "dq_auto.applied")["value"] == 2
    assert value(rows, "dq_auto.escape_recorded")["value"] == 1
    assert value(data.summarize(internal=True), "group_loss.tail_median")["value"] == 3
    assert value(rows, "avg_proxy_relative_delta.tail_median")["value"] == pytest.approx(-.01)
    for role, phase, epoch, count in (("epoch","before_epoch_decision",2,0),("final_raw","before_epoch_decision",3,1),
                                      ("final","after_epoch_decision",3,2)):
        result = data.summarize({"role": role, "phase": phase, "epoch": epoch})
        assert value(result,"avg_promote_applied.recorded_count")["value"] == count
    assert calls == Counter({kind: 1 for kind in HEADERS})
    data.verify_inputs()


def test_checkpoint_cutoff_partial_and_exact_common_epoch_interval(tmp_path):
    one = source(tmp_path/"one",6)
    two = source(tmp_path/"two",8)
    a = RunFeatures.load(RunReader(one.directory))
    b = RunFeatures.load(RunReader(two.directory))
    pair = next((l,r) for l,r in a.compare(b) if l["metric_id"] == "loss.tail_median")
    assert [(x["value"],x["previous"],x["epochs"]) for x in pair] == [(5.5,4,list(range(1,7)))]*2
    # Relabel two later observations to 999 without allowing them into the common interval.
    for series, values in b.series.items():
        values[:] = [(e, p, 999 if e > 6 else v) for e,p,v in values]
    assert value([l for l,_ in a.compare(b)], "loss.tail_delta")["value"] == 1.5
    a.consume("gradient",[dict(epoch=5,batch_index=1,loss=999,grad_norm=None,grad_scale=1)])
    cutoff = dict(role="step",epoch=5,batch_index=0,attempt_step=50,phase="step_complete")
    assert value(a.summarize(cutoff),"loss.tail_median")["value"] == 5
    missing = value(a.summarize({**cutoff,"batch_index":None}),"loss.tail_median")
    assert missing["value"] is None and missing["epochs"] == [1,2,3,4] and "range_note" in missing
    assert value([l for l,_ in a.compare(b,cutoff,None)],"loss.tail_median")["epochs"] == [1,2,3,4]


def test_missing_epoch_not_removed_and_input_change_is_detected(tmp_path):
    run = source(tmp_path)
    rows = RunReader(run.directory).stream("gradient").rows
    rows[-1]["loss"] = None
    path = write(run,"gradient",rows)
    data = RunFeatures.load(RunReader(run.directory))
    result = value(data.summarize(),"loss.tail_delta")
    assert result["value"] is None and result["tail_epochs"] == [5] and result["n_missing"] == 1
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"32.0",b"31.0"))
    with pytest.raises(Conflict):
        data.verify_inputs()


def test_auto_zero_unmeasured_empty_and_ambiguous_rank(tmp_path):
    run = source(tmp_path)
    auto = RunReader(run.directory).stream("dq_auto").rows
    for r in auto:
        if r["event_type"]=="check":
            r.update(auto_applied=False,low_auto_decision=None)
    write(run,"dq_auto",auto)
    rank_rows = RunReader(run.directory).stream("rank").rows
    write(run,"rank",rank_rows[:-1],expected_modules_by_scope={"unet":["a","b","c"]})
    data = RunFeatures.load(RunReader(run.directory))
    rows = data.summarize()
    assert value(rows,"dq_auto.applied")["value"] == 0
    assert value(rows,"dq_auto.escape_recorded")["value"] is None
    assert value(rows,"rank_sat_p95.tail_median")["value"] is None
    assert any(kind=="rank" for kind,_ in data.issues)
    write(run,"dq_auto",auto[:1])
    rows = RunFeatures.load(RunReader(run.directory)).summarize()
    assert value(rows,"dq_auto.records")["value"] == 0
    assert value(rows,"dq_auto.applied")["value"] is None


def test_published_diagnostic_fallback_requires_missing_csv_matching_pair_and_full_rows(tmp_path):
    import json
    from tools.lora_report_publication import JSON_PATH, HTML_PATH, publish
    from tools.lora_ledger_core.output_root import open_output_ledger
    from tools.lora_ledger_core.scanner import scan, apply_scan
    run = source(tmp_path)
    ledger = open_output_ledger(tmp_path / "ledger", create=True)
    reader = RunReader(run.directory)
    raw = reader.stream("gradient")
    source_info = {"schema_version":1,"path":"gradient.csv","rows":len(raw.rows),"status":"complete",
                   "sha256":raw.sha256,"source_bytes":raw.source_bytes}
    provenance = {"run_id":run.run_id,"session_id":run.session_id,"generation_id":"gen-A",
                  "inputs":{"gradient":source_info}}
    report = {"report_schema_version":1,"provenance":provenance,"grad":{"rows":raw.rows}}
    html = "".join(f'<meta name="{k}" content="{provenance[k]}">' for k in ("run_id","session_id","generation_id"))
    (run.directory/"diagnostic_report").mkdir()
    publish(run,report,html,lambda:None)
    candidate = scan(ledger)
    assert not apply_scan(ledger,candidate)["errors"]
    refs = ledger.get_run(run.run_id)["source_refs"]
    assert {r["kind"] for r in refs if r.get("generation_id") == "gen-A"} == {"diagnostic","diagnostic_html"}
    assert "from_report" not in RunFeatures.load(RunReader(run.directory)).sources["gradient"]
    path = run.directory/"gradient.csv"
    path.write_bytes(path.read_bytes()[:-2])
    assert "from_report" not in RunFeatures.load(RunReader(run.directory)).sources["gradient"]
    path.unlink()
    data = RunFeatures.load(RunReader(run.directory))
    assert data.sources["gradient"]["from_report"] == "gen-A"
    assert value(data.summarize(),"loss.tail_median")["value"] == 5
    data.verify_inputs()
    (run.directory/HTML_PATH).write_text(html.replace("gen-A","gen-B"),encoding="utf-8")
    changed = RunFeatures.load(RunReader(run.directory))
    assert "from_report" not in changed.sources["gradient"]
    assert any(k=="diagnostic_report" for k,reason in changed.issues)
    candidate=scan(ledger)
    assert not apply_scan(ledger,candidate)["errors"]
    assert not any(r["kind"]=="diagnostic" for r in ledger.get_run(run.run_id)["source_refs"])
    with pytest.raises(Conflict):
        data.verify_inputs()


@pytest.mark.parametrize("role", ["final", "final_raw"])
@pytest.mark.parametrize("right_role", ["final", "epoch"])
def test_common_interval_ends_before_unmatched_final_epoch(tmp_path, role, right_role):
    sides = []
    for name, count in [("lhs", 11), ("rhs", 12)]:
        run = source(tmp_path/name, 6)
        write(run, "gradient", [dict(epoch=i//2+1, batch_index=i%2, loss=999 if i==11 else i//2+1)
                                for i in range(count)])
        sides.append(RunFeatures.load(RunReader(run.directory)))
    left = dict(role=role, epoch=6, attempt_step=11, batch_index=0, epoch_complete=False)
    right = dict(role=right_role, epoch=6, attempt_step=12, batch_index=1, epoch_complete=True)
    def compare(a, b):
        return next(pair for pair in sides[0].compare(sides[1], a, b) if pair[0]["metric_id"]=="loss.tail_median")
    for row in compare(left, right):
        assert row["epochs"] == list(range(1, 6)) and row["value"] == 5 and row["n_valid"] == 10
        assert "epoch 6 excluded" in row["range_note"]
    legacy = dict(left, epoch_complete=None, batch_index=None)
    assert all("final position unknown" in row["range_note"] for row in compare(legacy, right))
    matching = dict(left, role="final")
    assert all(row["n_valid"] == 11 and row["value"] == 5.5 for row in compare(left, matching))


@pytest.mark.parametrize("cell,energy", [("", None), ("NaN", None), ("invalid", None), ("0", 0.), ("2", 2.)])
def test_rank_energy_completeness_in_report_heatmap_and_ledger(tmp_path, cell, energy):
    from tools.lora_diagnostic_rank import analyze_rank

    run = source(tmp_path)
    names = ["lora_unet_input_blocks_1_to_q", "lora_unet_output_blocks_1_to_k"]
    rows = [dict(epoch=e, attempt_step=e, scope="unet", module=name, rank_dim=4,
                 rank_energy=1., rank_sat=.8, rank_top1=.7) for e in range(1, 6) for name in names]
    path = write(run, "rank", rows, expected_modules_by_scope={"unet": names})
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        if row["module"] == names[1]:
            row["rank_energy"] = cell
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, HEADERS["rank"])
        writer.writeheader()
        writer.writerows(records)
    reader = RunReader(run.directory)
    report = analyze_rank(reader, reader.stream("rank").rows, dict.fromkeys(names, 4))
    metric = value(RunFeatures.load(reader).summarize(), "rank_energy.tail_median")
    total = None if energy is None else 1 + energy
    share = None if total is None else 1 / total
    assert report["summary"]["final_rank_energy_sum"] == metric["value"] == total
    assert metric["n_missing"] == (5 if energy is None else 0)
    groups = report["grouped"]["path"]["rows"][0]["groups"]
    assert groups["up"]["RankEnergySum"] == energy and groups["down"]["RankEnergyShare"] == share
    heatmap = report["final_heatmaps"]["energy_share_per_param"]
    assert next(c for row in heatmap["rows"] for c in row["cells"]
                if c["role"]=="to_q" and c["path"]=="down")["value"] == share
    assert report["rows"][0]["RankSatP95"] == .8
