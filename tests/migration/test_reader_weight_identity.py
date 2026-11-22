import json
from pathlib import Path
import struct
import uuid

import pytest

from tools.lora_run_reader import RunReader


@pytest.fixture
def weight(tmp_path):
    item = dict(role="final", phase="after_epoch_decision", path="last.safetensors", status="saved",
                epoch=1, attempt_step=1, batch_index=None)
    data = dict(schema_version=1, kind="training_run", run_id=str(uuid.uuid4()), session_id="8",
                base_name="last", output_name_omitted=False, started_at="2020-12-09T12:34:56+09:00",
                status="completed", artifacts={}, weights=[item])

    def build(*, metadata=None, body=b"a", **changes):
        data["weights"] = [{**item, **changes}]
        (tmp_path / "run.json").write_text(json.dumps(data), encoding="utf-8")
        header = json.dumps({"__metadata__": metadata or {}}).encode()
        (tmp_path / "last.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + body)
        reader = RunReader(tmp_path)
        return reader, reader.index["weights"][0]
    return build


def test_snapshot_id_and_upstream_clock_are_distinct(weight):
    r, item = weight(metadata={"ss_session_id": "8", "ss_training_started_at": "999"})
    assert r.weight(item, {"metadata": {"ss_training_started_at": 999}})[1] == "session_matched"
    assert r.weight(item, {"metadata": {"ss_training_started_at": 1000}})[1] == "unverified"
    assert r.weight(item)[1] == "session_matched"
    r, item = weight(metadata={"ss_session_id": "7"})
    assert r.weight(item)[1] == "unverified"
    r, item = weight(metadata={})
    assert r.weight(item)[1] == "indexed"


def test_relative_path_and_presence(weight):
    r, item = weight(path="../outside.safetensors")
    assert r.weight(item)[:2] == (None, "unverified")
    assert r.weight({**item, "path": "never-indexed.safetensors"})[1] == "unverified"
    r, item = weight(status="pruned")
    assert r.weight(item)[1] == "pruned"
    r, item = weight()
    path = r.run.directory / item["path"]
    path.unlink()
    assert r.weight(item)[:2] == (path, "missing")
    path.write_bytes(b"broken")
    assert r.weight(item)[1] == "unverified"


def test_role_and_phase_are_part_of_identity(weight):
    r, item = weight(role="final_raw", phase="before_epoch_decision")
    assert r.weight(item)[1] == "indexed"
    r, item = weight(role="step", phase="after_epoch_decision")
    assert r.weight(item)[1] == "unverified"


def test_same_session_but_different_embedded_run_uuid_is_unverified(weight):
    reader, item = weight(metadata={"ss_session_id": "8", "ss_run_id": str(uuid.uuid4())})
    path, status, reason = reader.weight(item)
    assert path.exists() and status == "unverified" and "ss_run_id" in reason
