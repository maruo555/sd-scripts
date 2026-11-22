import json
from pathlib import Path
import uuid

import pytest

from tools.lora_run_reader import discover_run_indexes


def index(directory, run_id=None, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(schema_version=1, kind="training_run", run_id=run_id or str(uuid.uuid4()), session_id="1",
                  base_name="same", output_name_omitted=False, started_at="2026-01-10T12:00:00+09:00",
                  requested_output_name="same", status="completed", artifacts={}, weights=[])
    record.update(fields)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def test_only_direct_training_indexes_and_all_duplicate_ids(tmp_path):
    first = index(tmp_path / "one")
    index(tmp_path / "two", first["run_id"])
    index(tmp_path / "three")
    for name in ("ledger", "cache", "exports", "derived", "diagnostic_report", "density", "sample", ".work"):
        index(tmp_path / name)
    index(tmp_path / "parent" / "nested")
    (tmp_path / "weight.safetensors").touch()
    runs, issues = discover_run_indexes(tmp_path)
    assert not issues
    assert sorted(r.run.directory.name for r in runs) == ["one", "three", "two"]
    assert sum(r.run.run_id == first["run_id"] for r in runs) == 2


def test_malformed_and_unknown_indexes_are_local_issues(tmp_path):
    valid = index(tmp_path / "valid")
    index(tmp_path / "version2", schema_version=2)
    invalid = tmp_path / "broken"
    invalid.mkdir()
    (invalid / "run.json").write_text("{", encoding="utf-8")
    runs, issues = discover_run_indexes(tmp_path)
    assert [r.run.run_id for r in runs] == [valid["run_id"]]
    assert {i["directory"] for i in issues} == {"broken", "version2"}
    with pytest.raises(FileNotFoundError):
        discover_run_indexes(tmp_path / "unknown")


def test_linked_directory_is_reported_without_opening_target(tmp_path):
    original = tmp_path / "actual"
    index(original)
    link = tmp_path / "alias"
    try:
        link.symlink_to(original, target_is_directory=True)
    except OSError:
        # Windows junctions can be created without Developer Mode or administrator rights.
        import os
        import subprocess
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/d", "/c", "mklink", "/j", str(link), str(original)], check=True, capture_output=True)
    runs, issues = discover_run_indexes(tmp_path)
    assert [r.run.directory for r in runs] == [original]
    assert [i["directory"] for i in issues] == ["alias"]
