"""Legacy ledger contracts use generated, non-personal records."""
from pathlib import Path

import pytest

from tools.lora_ledger_core.history_transfer import (
    BUNDLE_FILES, copy_legacy_ledger, inspect_saved_export, list_saved_exports, open_legacy_ledger,
)
from tools.lora_ledger_core.notes import save_note
from tools.lora_ledger_core.reviews import candidate_for, preference_pairs
from tools.lora_ledger_core.storage import (
    Conflict, Ledger, LedgerError, atomic_json, new_review, read_json,
)


def tree_bytes(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_handover_preserves_ids_history_manual_assessments_and_snapshot(tmp_path):
    original = Ledger.create(tmp_path / "origin")
    run = original.create_note_run("sample")
    run["artifacts"] = [{"artifact_id": "a", "name": "epoch-1",
                         "ref": {"root": "offline", "relative_path": "a.safetensors"}}]
    run = original.save_run(run, run["revision"])
    assert not save_note(original, run, None, {"purpose": "quality"})["saved_review"]
    run = original.get_run(run["run_id"])
    left = candidate_for(run, run["artifacts"][0], strength=0.8)
    right = candidate_for(run, run["artifacts"][0], strength=1.0)
    ordered = [left["candidate_id"], right["candidate_id"]]
    review = new_review()
    review.update(
        candidates=[left, right], viewed_candidate_ids=ordered,
        comparisons=preference_pairs([left, right], ordered,
                                     {"overall": ordered[0], "identity": "tie", "naturalness": "undecided"}),
        adoption={"status": "selected", "candidate_ids": ordered[:1]},
        annotations=[{"run_id": run["run_id"], "artifact_id": "a", "favorite_rating": None}],
    )
    first = original.save_review(review)
    second = save_note(original, run, "a", {"favorite_rating": 4}, previous=first, confirm_hash=False)
    third = save_note(original, original.get_run(run["run_id"]), "a", {"favorite_rating": 5},
                      previous=second, reassess=True, confirm_hash=False)
    expected = original.reviews(history=True)
    assert [r["revision"] for r in expected if r["review_id"] == first["review_id"]] == [1, 2]
    assert third["review_id"] != first["review_id"] and third["reassessment_of"] == first["review_id"]
    assert [r["review_id"] for r in original.reviews()] == [third["review_id"]]

    snapshot = original.directory / "exports" / "summary.csv"
    snapshot.write_bytes(b"run_id\r\n")
    atomic_json(str(snapshot) + ".columns.json", {"schema_version": 1, "roundtrip": False})
    (original.directory / "research" / "draft.md").write_bytes(b"fixture")
    (original.directory / "cache" / "retained").mkdir()
    before = tree_bytes(original.directory)
    copy = copy_legacy_ledger(original.directory, tmp_path / "out" / "ledger")
    assert before == tree_bytes(copy.directory) == tree_bytes(original.directory)
    assert (copy.directory / "cache" / "retained").is_dir()
    assert copy.config() == original.config()
    assert copy.list_runs() == original.list_runs()
    assert copy.reviews(history=True) == expected
    assert copy.reviews() == [third]
    stale = copy.get_run(run["run_id"])
    updated = copy.save_run(stale, stale["revision"])
    assert updated["revision"] == stale["revision"] + 1
    with pytest.raises(Conflict):
        copy.save_run(stale, stale["revision"])
    assert read_json(copy.directory / "history/runs" / run["run_id"] / f'{stale["revision"]:06d}.json') == stale
    assert tree_bytes(original.directory) == before


def test_read_only_open_never_replays_or_writes(tmp_path, monkeypatch):
    original = Ledger.create(tmp_path / "origin")
    original.create_note_run("sample")
    before = tree_bytes(original.directory)
    reader = open_legacy_ledger(original.directory)
    assert reader.list_runs() == original.list_runs()
    with pytest.raises(LedgerError, match="読取専用"):
        reader.create_note_run("unwanted")
    assert tree_bytes(original.directory) == before
    config = original.config()
    config["revision"] += 1
    atomic_json(original.directory / ".transaction.json",
                {"schema_version": 1, "kind": "ledger_transaction", "writes": {"ledger.json": config}})
    unresolved = tree_bytes(original.directory)
    with monkeypatch.context() as guard:
        def no_open(*args, **kwargs):
            pytest.fail("A pending legacy journal must be detected before constructing Ledger")
        guard.setattr(Ledger, "__init__", no_open)
        with pytest.raises(LedgerError, match="journal"):
            open_legacy_ledger(original.directory)
    copy = copy_legacy_ledger(original.directory, tmp_path / "out")
    assert copy.config() == config
    assert not (copy.directory / ".transaction.json").exists()
    assert Ledger(copy.directory).config() == config
    assert tree_bytes(original.directory) == unresolved


def test_invalid_journal_never_publishes_partial_copy(tmp_path):
    source = Ledger.create(tmp_path / "source")
    atomic_json(source.directory / ".transaction.json", {
        "schema_version": 1, "kind": "ledger_transaction",
        "writes": {"cache/new.json": {}, "../escaped.json": {}},
    })
    before = tree_bytes(source.directory)
    destination = tmp_path / "destination"
    with pytest.raises(LedgerError):
        copy_legacy_ledger(source.directory, destination)
    assert not destination.exists() and not (tmp_path / "escaped.json").exists()
    assert tree_bytes(source.directory) == before


@pytest.mark.parametrize("destination", ["origin/nested", "origin", ".", "used"])
def test_destination_must_be_new_and_disjoint(tmp_path, destination):
    source = Ledger.create(tmp_path / "origin")
    (tmp_path / "used").mkdir()
    with pytest.raises(LedgerError):
        copy_legacy_ledger(source.directory, tmp_path / destination)


def test_copy_detects_changes_during_read(tmp_path, monkeypatch):
    from tools.lora_ledger_core import history_transfer

    source = Ledger.create(tmp_path / "origin")
    original_copytree = history_transfer.shutil.copytree

    def copy_then_change(src, dst, *args, **kwargs):
        result = original_copytree(src, dst, *args, **kwargs)
        if Path(src) == source.directory:
            (source.directory / "cache" / "concurrent").touch()
        return result

    monkeypatch.setattr(history_transfer.shutil, "copytree", copy_then_change)
    with pytest.raises(Conflict):
        copy_legacy_ledger(source.directory, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


def test_saved_bundle_records_original_status_and_current_missing_files(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    bundle = ledger.directory / "derived" / "bundle"
    bundle.mkdir()
    for name in BUNDLE_FILES:
        (bundle / name).touch()
    manifest = {"schema_version": 1, "kind": "ledger_export", "status": "complete",
                "runs": [{"run_id": "sample", "revision": 4, "hash": "as-saved"}]}
    atomic_json(bundle / "export_manifest.json", manifest)
    assert list_saved_exports(ledger)[0].is_available
    (bundle / "README.md").unlink()
    saved = inspect_saved_export(bundle)
    assert saved.manifest == manifest and saved.missing_files == ("README.md",) and not saved.is_available
    manifest["status"] = "incomplete"
    atomic_json(bundle / "export_manifest.json", manifest)
    (bundle / "README.md").touch()
    saved = inspect_saved_export(bundle)
    assert not saved.is_available and saved.missing_files == ()
    assert saved.manifest["status"] == "incomplete"
    (bundle / "export_manifest.json").unlink()
    saved = list_saved_exports(ledger)[0]
    assert saved.manifest is None and not saved.is_available
    assert saved.missing_files == ("export_manifest.json",)


def test_interrupted_combined_run_review_journal_replays_once_only_on_copy(tmp_path):
    original = Ledger.create(tmp_path / "original")
    first = original.create_note_run("initial")
    next_run = {**first, "display_name": "edited", "revision": 2}
    saved_review = {**new_review(), "schema_version": 1, "kind": "ledger_review", "revision": 1,
                    "annotations": [{"run_id": first["run_id"], "artifact_id": None, "favorite_rating": 4}]}
    review_key = f'reviews/{saved_review["review_id"]}/000001.json'
    writes = {review_key: saved_review, f'runs/{first["run_id"]}.json': next_run,
              f'history/runs/{first["run_id"]}/000001.json': first}
    atomic_json(original.directory / ".transaction.json",
                {"schema_version": 1, "kind": "ledger_transaction", "writes": writes})
    # Simulate the crash after writing the review and before writing run/history.
    atomic_json(original.directory / review_key, saved_review)
    raw = tree_bytes(original.directory)
    copy = copy_legacy_ledger(original.directory, tmp_path / "new")
    assert copy.get_run(first["run_id"]) == next_run
    assert copy.reviews(history=True) == [saved_review]
    assert read_json(copy.directory / "history/runs" / first["run_id"] / "000001.json") == first
    assert Ledger(copy.directory).reviews(history=True) == [saved_review]
    assert tree_bytes(original.directory) == raw
