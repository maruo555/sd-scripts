import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.lora_ledger import main
from tools.lora_ledger_core.exporting import export_bundle, export_csv
from tools.lora_ledger_core.history_transfer import inspect_saved_export
from tools.lora_ledger_core.output_root import open_output_ledger
from tools.lora_ledger_core.reviews import import_report, pair_cases, preference_pairs, select_cases
from tools.lora_ledger_core.scanner import apply_scan, scan
from tools.lora_ledger_core.storage import Cancelled, LedgerError, read_json
from test_comparison_result import comparison
from test_ledger_features import source
from test_ledger_registration import weight


def bundle_rows(bundle, kind):
    with (bundle / (kind+".csv")).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("partial_final", [False, True])
def test_saved_comparison_review_and_bundle_link_cases_strength_history(tmp_path, partial_final):
    root = tmp_path / "out"
    ledger = open_output_ledger(root / "ledger", create=True)
    first_epoch = 6 if partial_final else 5
    runs = [source(root, first_epoch), source(root, 7)]
    paths = [weight(r,"final.safetensors") for r in runs]
    for run, epoch in zip(runs, (first_epoch, 7)):
        complete = run is runs[1] or not partial_final
        run.update_index(lambda data: data["weights"][0].update(
            epoch=epoch, attempt_step=epoch*10, epoch_complete=complete, batch_index=0))
    assert len(apply_scan(ledger,scan(ledger,full=True))["applied"]) == 2
    folder = tmp_path / "comparison"
    m = comparison(folder)
    for c, path in zip(m["conditions"], paths):
        c["items"] = [{"path":str(path),"module":"networks.lora_lbw","strength":[.25,.5,1],"strength_width":3,
                       "resolved_strength":{"te1":.25,"te2":.5,"unet":1},"lbw":"ALL","resolved_lbw":[1]*12,"slot":0}]
    (folder/"metadata.json").write_text(json.dumps(m),encoding="utf-8")
    review = import_report(ledger,folder)
    assert {c["case_id"] for c in review["cases"]} == {"case-1"}
    assert {c["job_id"] for c in review["cases"]} == {"job-a","job-b"}
    assert all(not c["generation_link_verified"] for c in review["candidates"])
    assert all(c["components"][0]["identity_status"]=="path_match_only" for c in review["candidates"])
    component = review["candidates"][0]["components"][0]
    assert component["resolved_strength"]["te2"] == .5 and component["resolved_lbw"] == [1]*12
    ids=[c["candidate_id"] for c in review["candidates"]]
    assert [c["candidate_id"] for c in import_report(ledger,folder)["candidates"]] == ids
    select_cases(ledger,review,"selected",{"case-1"})
    review["comparisons"]=preference_pairs(review["candidates"],ids,{"overall":"tie"})
    pair_cases(review)
    assert review["comparisons"][0]["case_ids"]==["case-1"]
    saved=ledger.save_review(review)
    bundle=export_bundle(ledger,[runs[0].run_id])
    exported=inspect_saved_export(bundle)
    assert exported.is_available
    assert {r["run_id"] for r in exported.manifest["runs"]}=={r.run_id for r in runs}
    assert exported.manifest["reviews"][0]["review_id"] == saved["review_id"]
    assert len(exported.manifest["sources"])==2
    rows=bundle_rows(bundle,"features")
    filtered = [r for r in rows if r["metric_id"]=="loss.tail_median"]
    assert {r["window_id"] for r in filtered}=={"observed_run","checkpoint_local","common_interval"}
    common = [r for r in filtered if r["window_id"] == "common_interval"]
    assert len(common) == 2 and {float(r["value"]) for r in common} == {5.0}
    if partial_final:
        assert all("epoch 6 excluded" in r["range_note"] for r in common)
    assert json.loads((bundle/"reviews.jsonl").read_text(encoding="utf-8"))["review_id"]==saved["review_id"]
    old=hashlib.sha256((bundle/"export_manifest.json").read_bytes()).hexdigest()
    again=export_bundle(ledger,[runs[0].run_id])
    assert again != bundle and hashlib.sha256((bundle/"export_manifest.json").read_bytes()).hexdigest()==old


def test_csv_no_overwrite_selection_and_partial_bundle(tmp_path,capsys):
    ledger = open_output_ledger(tmp_path/"ledger",create=True)
    run = ledger.create_note_run("=1+2")
    path = tmp_path/"report.csv"
    assert export_csv(ledger,path,run_ids=[run["run_id"]],columns=["run_id","学習名"])==1
    assert path.read_bytes().startswith(b"\xef\xbb\xbf") and "\r\n" in path.read_bytes().decode("utf-8-sig")
    assert "'=1+2" in path.read_text(encoding="utf-8-sig")
    assert read_json(str(path)+".columns.json")["roundtrip"] is False
    before=path.read_bytes()
    with pytest.raises(LedgerError):
        export_csv(ledger,path)
    assert path.read_bytes()==before
    with pytest.raises(Cancelled):
        export_bundle(ledger,cancel=lambda:True)
    exported=next((ledger.directory/"derived").iterdir())
    assert not inspect_saved_export(exported).is_available
    selection=tmp_path/"selection.json"
    selection.write_text(json.dumps({"run_ids":[run["run_id"]]}))
    assert main(["export","--ledger",str(ledger.directory),"--selection",str(selection)])==0
    bundle=Path(capsys.readouterr().out.strip())
    assert inspect_saved_export(bundle).manifest["selection"]["run_ids"]==[run["run_id"]]
    selection.write_text(json.dumps({"run_ids":[],"include_unverified":True}))
    with pytest.raises(SystemExit):
        main(["export","--ledger",str(ledger.directory),"--selection",str(selection)])


def test_changed_registered_stream_only_and_duplicate_run_withhold_features(tmp_path):
    ledger=open_output_ledger(tmp_path/"ledger",create=True)
    run=source(tmp_path)
    apply_scan(ledger,scan(ledger))
    path=run.directory/"gradient.csv"
    path.write_bytes(path.read_bytes()+b"6,0,,1,999,32\r\n")
    exported=export_bundle(ledger,[run.run_id])
    metrics={r["metric_id"] for r in bundle_rows(exported,"features")}
    assert "loss.tail_median" not in metrics and "rank_energy.tail_median" in metrics
    assert any("追記" in r["reason"] for r in bundle_rows(exported,"issues"))
    clone=tmp_path/"duplicate"
    clone.mkdir()
    (clone/"run.json").write_bytes((run.directory/"run.json").read_bytes())
    exported=export_bundle(ledger,[run.run_id])
    assert not bundle_rows(exported,"features")
    assert any("重複" in r["reason"] for r in bundle_rows(exported,"issues"))


def test_native_qt_save_edit_and_reopen(tmp_path,monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM","offscreen")
    from PySide6.QtWidgets import QApplication
    from tools.lora_ledger_core.gui import LedgerWindow
    ledger=open_output_ledger(tmp_path/"ledger",create=True)
    run=ledger.create_note_run("sample")
    app=QApplication.instance() or QApplication([])
    w=LedgerWindow(ledger,tmp_path/"ui.json")
    w.show()
    app.processEvents()
    w.rating.setCurrentIndex(w.rating.findData(4))
    w.good.setPlainText("fixture")
    assert w.save()
    w.close()
    reopened=LedgerWindow(open_output_ledger(ledger.directory),tmp_path/"ui.json")
    reopened.show()
    app.processEvents()
    assert reopened.current_run["run_id"]==run["run_id"] and reopened.rating.currentData()==4
    assert reopened.good.toPlainText()=="fixture"
    reopened.good.setPlainText("revised")
    assert reopened.save()
    assert [r["revision"] for r in ledger.reviews(history=True)] == [1,2]
    reopened.close()
    app.processEvents()
