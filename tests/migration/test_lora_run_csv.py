import codecs
import csv
import json
import math

import pytest

from library.lora_run_schema import HEADERS
from library.training_diagnostics import RunCsvWriter
from library.training_run import TrainingRun
from library.training_settings import save_settings
from tools.lora_run_reader import RunReader, owned_path, promote_windows, unique_auto_events


def test_all_streams_types_and_quoting(tmp_path):
    run = TrainingRun.create(tmp_path, None, "a")
    writer = RunCsvWriter(run, HEADERS, expected_modules_by_scope={"unet":["x"]})
    group = dict(epoch=1, attempt_step=1, group='色見本,"中"\r\n続き', loss=0.125, group_observation_count=1,
                 subset_index=-1, timestep=0)
    writer.append("group_loss", [group], attempt_step=1)
    writer.append("gradient", [dict(epoch=1, batch_index=0, loss=-math.inf, threshold=math.nan,
                                     grad_norm=math.inf, grad_scale=None)])
    writer.append("dq_auto", [dict(event_type="init", attempt_step=0, auto_applied=False)])
    assert RunReader(run.directory).stream("group_loss").rows == []
    writer.after_attempt(100)
    assert RunReader(run.directory).stream("group_loss").rows[0]["group"] == group["group"]
    assert writer.close()
    for name, columns in HEADERS.items():
        path = run.directory / f"{name}.csv"
        raw = path.read_bytes()
        assert raw.count(codecs.BOM_UTF8) == 1 and raw.endswith(b"\r\n")
        assert raw[3:].split(b"\r\n")[0].decode() == ",".join(columns)
        assert RunReader(run.directory).stream(name).status == "complete"
    rows = RunReader(run.directory).stream("gradient").rows
    assert len(rows) == 1 and math.isnan(rows[0]["threshold"]) and rows[0]["loss"] == -math.inf
    assert rows[0]["grad_norm"] == math.inf and rows[0]["grad_scale"] is None
    assert RunReader(run.directory).stream("dq_auto").rows[0]["auto_applied"] is False


def test_partial_groups_and_duplicate_module(tmp_path):
    r = TrainingRun.create(tmp_path, "x", 8)
    w = RunCsvWriter(r, ["rank", "promote"], expected_modules_by_scope={"unet":["a", "b"]})
    for step, names in [(100, ["b","a"]), (200, ["a","a"]), (300, ["a"])]:
        w.append("rank", [dict(epoch=1, attempt_step=step, scope="unet", module=name) for name in names])
    w.append("promote", [dict(row_type="window_epoch", epoch=2, window_epoch=2, window_position=2),
                         dict(row_type="decision", epoch=2, window_count=2),
                         dict(row_type="window_epoch", epoch=2, window_epoch=1, window_position=1),
                         dict(row_type="decision", epoch=3, window_count=1)])
    w.close()
    reader = RunReader(r.directory)
    observations = list(reader.rank_observations(reader.stream("rank").rows))
    assert [bool(reasons) for _, _, reasons in observations] == [False,True,True]
    windows = list(promote_windows(reader.stream("promote").rows))
    assert windows[0][2] == [1,2] and windows[1][2] is None
    assert reader.stream("dq").status == "disabled"


@pytest.mark.parametrize("tail", [b'2,"unterminated', b'1,2,0.5,1,1,', b"\xff\xff\r\n"])
def test_truncated_log_preserves_complete_records(tmp_path, tail):
    r = TrainingRun.create(tmp_path, "x", "1")
    w = RunCsvWriter(r, ["gradient"])
    w.append("gradient",[dict(epoch=1, batch_index=0, loss=2)])
    w.close()
    with (r.directory / "gradient.csv").open("ab") as stream:
        stream.write(tail)
    result = RunReader(r.directory).stream("gradient")
    assert result.status == "partial" and len(result.rows) == 1


def test_partial_append_does_not_replay(tmp_path, monkeypatch):
    r = TrainingRun.create(tmp_path, "x", 1)
    w = RunCsvWriter(r, ["group_loss"])
    def partial(name, rows):
        with (r.directory / "group_loss.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(rows[0])
        raise OSError("short write")
    monkeypatch.setattr(w, "_write_rows",partial)
    w.append("group_loss", [dict(epoch=1, attempt_step=1), dict(epoch=1, attempt_step=2)])
    w.end_epoch(2)
    assert not w.close()
    result = RunReader(r.directory).stream("group_loss")
    assert result.status == "partial" and [row["attempt_step"] for row in result.rows] == [1]


def test_settings_identity_and_paths(tmp_path):
    r = TrainingRun.create(tmp_path, "x", "8")
    save_settings(r, {"setting":1})
    assert RunReader(r.directory).settings() == (None, "requested_only")
    save_settings(r, {"setting":1}, resolved={"setting":2})
    assert RunReader(r.directory).settings() == ({"setting":2}, "resolved")
    path = r.directory / "training_settings.json"
    d = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps({**d, "session_id":"other"}), "utf-8")
    assert RunReader(r.directory).settings()[0] is None
    for invalid in ["../foreign", str(tmp_path.resolve()), "C:/foreign", "//server/share/file"]:
        with pytest.raises(ValueError):
            owned_path(r.directory, invalid)
    assert unique_auto_events([{"attempt_step":0}, {"attempt_step":4}, {"attempt_step":4}]) == {0:{"attempt_step":0}, 4:None}
