import json
import os
from pathlib import Path
import shutil
import struct

import pytest

from library.training_run import TrainingRun
from library.training_run_weights import WeightMoment, record_weight
from library.training_settings import save_settings
from tools.lora_ledger import main
from tools.lora_ledger_core.history_transfer import copy_legacy_ledger
from tools.lora_ledger_core.output_root import open_output_ledger
from tools.lora_ledger_core.scanner import apply_scan, scan, verify
from tools.lora_ledger_core.storage import Conflict, Ledger, atomic_json, file_stat, read_json


def weight(run, name, role="final", body=b"a", metadata=None):
    path = run.directory / name
    header = json.dumps({"__metadata__": metadata or {"ss_session_id": run.session_id}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + body)
    record_weight(run, path, WeightMoment(role, 1, 2, None, "after_epoch_decision" if role == "final" else "before_epoch_decision"))
    return path


def apply(ledger, **options):
    result = scan(ledger, **options)
    outcome = apply_scan(ledger, result)
    assert not outcome["errors"]
    return outcome["applied"]


def test_distinct_uuid_one_folder_and_revision_idempotence(tmp_path):
    ledger = open_output_ledger(tmp_path / "ledger", create=True)
    first, second = [TrainingRun.create(tmp_path, "same", 42) for _ in range(2)]
    assert set(apply(ledger)) == {first.run_id, second.run_id}
    assert all(r["revision"] == 1 for r in ledger.list_runs())
    assert not apply(ledger)
    assert save_settings(first, {}, resolved={"args": {"dataset_config": "example.toml"}})
    assert apply(ledger) == [first.run_id]
    assert not apply(ledger)
    entry = ledger.get_run(first.run_id)
    assert entry["revision"] == 2 and entry["dataset"]["display_name"] == "example.toml"
    assert entry["settings_status"] == "resolved" and not entry["artifacts"]
    assert ledger.get_run(second.run_id)["settings_status"] == "unavailable"


def test_preserves_old_source_roots_and_user_edits(tmp_path):
    historical = Ledger.create(tmp_path / "historical")
    old = historical.create_note_run("existing")
    locations = read_json(historical.directory / "locations.local.json")
    locations["roots"]["legacy"] = str(tmp_path / "historical-root")
    atomic_json(historical.directory / "locations.local.json", locations)
    TrainingRun.create(tmp_path / "historical-root", "excluded", 1)
    root = tmp_path / "new"
    copy_legacy_ledger(historical.directory, root / "ledger")
    ledger = open_output_ledger(root / "ledger")
    assert ledger.locations()["legacy"] == locations["roots"]["legacy"]
    assert not scan(ledger)["rows"] and ledger.get_run(old["run_id"]) == old
    run = TrainingRun.create(root, "name", 3)
    assert apply(ledger) == [run.run_id]
    amended = ledger.get_run(run.run_id)
    amended.update(display_name="custom", purpose="manual", training_status="interrupted")
    amended["user"].update(display_name_custom=True, experiment_note="memo")
    ledger.save_run(amended, amended["revision"])
    run.finish("completed", "complete")
    apply(ledger)
    changed = ledger.get_run(run.run_id)
    assert (changed["display_name"], changed["purpose"], changed["training_status"], changed["indexed_status"]) == (
        "custom", "manual", "interrupted", "completed")
    assert changed["user"]["experiment_note"] == "memo"


def test_changed_snapshot_duplicate_uuid_manual_edit_and_retry(tmp_path):
    ledger = open_output_ledger(tmp_path / "ledger", create=True)
    run = TrainingRun.create(tmp_path, "name", 3)
    initial = scan(ledger)
    run.finish("completed", "complete")
    assert apply_scan(ledger, initial)["errors"] and not ledger.list_runs()
    initial = scan(ledger)
    clone = tmp_path / "clone"
    shutil.copytree(run.directory, clone)
    assert not scan(ledger)["rows"]
    assert apply_scan(ledger, initial)["errors"]
    # Remove only the synthetic clone's index to model the duplicate being resolved.
    (clone / "run.json").unlink()
    assert apply(ledger) == [run.run_id]
    run.set_artifact("gradient", "gradient.csv", schema_version=1, status="open")
    (run.directory / "gradient.csv").write_bytes(b"data")
    candidate = scan(ledger)
    amended = ledger.get_run(run.run_id)
    amended["user"]["experiment_note"] = "edited"
    ledger.save_run(amended, amended["revision"])
    assert apply_scan(ledger, candidate)["errors"]
    assert apply(ledger) == [run.run_id]
    assert ledger.get_run(run.run_id)["user"]["experiment_note"] == "edited"


def test_saved_roles_move_verify_and_replaced_same_stat(tmp_path):
    ledger = open_output_ledger(tmp_path / "ledger", create=True)
    first = TrainingRun.create(tmp_path, "name", 3)
    final = weight(first, "final.safetensors", "final")
    raw = weight(first, "raw.safetensors", "final_raw")
    assert final.read_bytes() == raw.read_bytes()
    apply(ledger, full=True)
    artifacts = ledger.get_run(first.run_id)["artifacts"]
    ids = [a["artifact_id"] for a in artifacts]
    assert len(set(ids)) == 2
    moved = tmp_path / "moved"
    first.directory.rename(moved)
    assert apply(ledger) == [first.run_id]
    assert [a["artifact_id"] for a in ledger.get_run(first.run_id)["artifacts"]] == ids
    path = moved / final.name
    oldstat = path.stat()
    path.write_bytes(path.read_bytes()[:-1] + b"b")
    os.utime(path, ns=(oldstat.st_atime_ns, oldstat.st_mtime_ns))
    assert not apply(ledger)
    result = verify(ledger, full=True)
    assert not result["errors"] and result["applied"] == [first.run_id]
    new = ledger.get_run(first.run_id)["artifacts"]
    assert len(new) == 3
    assert next(a for a in new if a["artifact_id"] == ids[0])["ref"]["status"] == "superseded"
    assert next(a for a in new if a["artifact_id"] == ids[1])["ref"]["status"] == "exists"
    assert not apply(ledger, full=True)


def test_moved_without_initial_sha_pruned_missing_header_mismatch(tmp_path):
    ledger = open_output_ledger(tmp_path / "ledger", create=True)
    run = TrainingRun.create(tmp_path, "name", 3)
    weight(run, "epoch.safetensors", "epoch")
    weight(run, "bad.safetensors", metadata={"ss_session_id": "other"})
    apply(ledger)
    before = ledger.get_run(run.run_id)["artifacts"]
    first_id = before[0]["artifact_id"]
    assert before[1]["ref"]["status"] == "unverified"
    moved = tmp_path / "renamed"
    run.directory.rename(moved)
    apply(ledger)
    entries = ledger.get_run(run.run_id)["artifacts"]
    assert next(a for a in entries if a["artifact_id"] == first_id)["ref"]["status"] == "unverified_move"
    assert any(a.get("predecessor_status") == "unverified_move" for a in entries)
    index = read_json(moved / "run.json")
    index["weights"][0]["status"] = "pruned"
    atomic_json(moved / "run.json", index)
    apply(ledger)
    assert any(a["ref"]["status"] == "pruned" for a in ledger.get_run(run.run_id)["artifacts"])
    (moved / "bad.safetensors").unlink()
    apply(ledger)
    assert any(a["ref"]["status"] == "missing" for a in ledger.get_run(run.run_id)["artifacts"])


def test_cli_arm_and_cancel(tmp_path, capsys):
    folder = tmp_path / "ledger"
    assert main(["init", "--ledger", str(folder)]) == 0
    capsys.readouterr()
    runs = [TrainingRun.create(tmp_path, "name", i) for i in (1, 2)]
    assert main(["scan", "--ledger", str(folder)]) == 0
    assert len(json.loads(capsys.readouterr().out)["rows"]) == 2
    assert main(["register", "--ledger", str(folder), "--run-id", runs[0].run_id]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] == [runs[0].run_id]
    ledger = open_output_ledger(folder)
    assert apply_scan(ledger, scan(ledger), cancel=lambda: True)["cancelled"]
    assert len(ledger.list_runs()) == 1
