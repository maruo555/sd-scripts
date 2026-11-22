import hashlib
import json
from types import SimpleNamespace

from safetensors.torch import save_file
import torch
import pytest

from tools.make_lora_diagnostic_report import _lora_param_counts, _settings


def test_settings_payload_and_fingerprint_come_from_one_read(tmp_path):
    path = tmp_path / "training_settings.json"
    run = SimpleNamespace(run_id="mine", session_id="72")
    payload = dict(schema_version=2,run_id=run.run_id, session_id=run.session_id,settings_status="resolved",
                   requested={"v":1},resolved={"v":2})
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    class Reader:
        def artifact(self,kind,version):
            return path, {}
        def settings(self):
            raise AssertionError("must not independently reopen changing settings")
    reader = Reader()
    reader.run = run
    settings, source = _settings(reader)
    assert settings == {"status":"resolved","requested":{"v":1},"resolved":{"v":2}}
    assert source == (path,len(raw),hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(dict(payload,session_id="other")), "utf-8")
    invalid, other = _settings(reader)
    assert invalid["resolved"] is None and invalid["requested"] is None and other[2] != source[2]


def test_parameter_counts_need_both_lora_factors(tmp_path):
    path = tmp_path / "lora.safetensors"
    save_file({"lora_a.lora_up.weight":torch.ones(2,3),"lora_a.lora_down.weight":torch.ones(3,4),
               "lora_b.lora_down.weight":torch.ones(1,3),"lora_b.alpha":torch.tensor(1.)},str(path))
    assert _lora_param_counts(path) == {"lora_a":18}

def test_rank_weight_replaced_while_counting_stops_publication(tmp_path, monkeypatch):
    from tools import make_lora_diagnostic_report as report
    from library.lora_run_schema import HEADERS
    from tools.lora_run_reader import CsvResult

    filename = tmp_path / "a.safetensors"
    filename.write_bytes(b"old")
    class Reader:
        run = SimpleNamespace(run_id="x",session_id="1",directory=tmp_path)
        index = {"artifacts":{}, "weights":[{"path":"a.safetensors","epoch":1,"role":"final","status":"saved"}]}
        def stream(self, kind):
            return CsvResult(kind, "complete", rows=[{}] if kind == "rank" else [])
        def artifact(self, kind, version):
            raise ValueError("optional settings unavailable")
        def weight(self, item, resolved):
            return filename, "session_matched", None
    def replaced(_):
        filename.write_bytes(b"new physical weights")
        return {"lora_a":18}
    monkeypatch.setattr(report, "_lora_param_counts", replaced)
    with pytest.raises(ValueError,match="changed during rank"):
        report._generate(Reader(),report.Options(skip_lora_analysis=True))
