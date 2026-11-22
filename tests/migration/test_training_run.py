import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from library import training_settings
from library.training_run import TrainingRun, write_json


def index(run):
    return json.loads((run.directory / "run.json").read_text("utf-8"))


def test_atomic_reservation_and_safe_names(tmp_path):
    now = dt.datetime(2026, 11, 21, 12, tzinfo=dt.timezone.utc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: TrainingRun.create(tmp_path, "色見本 test ", 7, now=now), range(4)))
    assert len({r.run_id for r in runs}) == 4
    assert sorted(r.directory.name.split("色見本 test")[1] for r in runs) == ["", "_02", "_03", "_04"]
    for r in runs:
        assert TrainingRun.open(r.directory) == r
        assert index(r)["requested_output_name"] == "色見本 test "
        assert dt.datetime.fromisoformat(index(r)["started_at"]).utcoffset() is not None
    for raw, stem in [(None, "last"), (". .", "last"), ("NUL.txt", "_NUL.txt"), ("x/?", "x__")]:
        r = TrainingRun.create(tmp_path, raw, "x")
        assert r.base_name == stem and r.output_name_omitted == (raw is None)
    import hashlib
    raw = "長" * 500
    assert TrainingRun.create(tmp_path, raw, "x").base_name.endswith(hashlib.sha256(raw.encode()).hexdigest()[:8])


def test_index_updates_and_identity(tmp_path):
    r = TrainingRun.create(tmp_path, "test", 2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: r.set_artifact(str(i), f"{i}.csv", schema_version=1, status="open"), range(12)))
    assert len(index(r)["artifacts"]) == 12
    with pytest.raises(ValueError):
        r.relative_path("../other.csv")
    write_json(r.directory / "run.json", {**index(r), "run_id": "different"})
    with pytest.raises(ValueError):
        r.finish("completed", "complete")


def test_two_phase_and_failure_independence(tmp_path, monkeypatch):
    r = TrainingRun.create(tmp_path, None, 42)
    requested = dict(huggingface_token="PRIVATE", optimizer_args=["api_key=PRIVATE"], settings=object())
    assert training_settings.save_settings(r, requested)
    cfg = r.directory / "training_settings.json"
    assert "PRIVATE" not in cfg.read_text("utf-8")
    assert json.loads(cfg.read_text("utf-8"))["settings_status"] == "requested"
    assert training_settings.save_settings(r, requested, resolved={"training": {"total_epochs": 43}})
    resolved = json.loads(cfg.read_text("utf-8"))
    assert resolved["schema_version"] == 2 and resolved["run_id"] == r.run_id
    assert resolved["session_id"] == "42" and resolved["resolved"]["training"]["total_epochs"] == 43
    def fail(*args):
        raise OSError("disk")
    monkeypatch.setattr(training_settings, "write_json", fail)
    assert not training_settings.save_settings(r, requested, resolved={})
    assert index(r)["artifacts"]["training_settings"]["status"] == "partial"
    assert json.loads(cfg.read_text("utf-8")) == resolved
    r.finish("stopped", "complete")
    assert index(r)["status"] == "stopped"


def test_atomic_json_does_not_replace_good_data_on_error(tmp_path):
    path = tmp_path / "settings.json"
    write_json(path, {"good": True})
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text("utf-8")) == {"good": True}
    assert list(tmp_path.iterdir()) == [path]
