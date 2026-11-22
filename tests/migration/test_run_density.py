import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from library.training_run import TrainingRun
from library.training_run_weights import WeightMoment, publish_weight
from tools import analyze_lora_density as density
from tools.lora_run_reader import RunReader


def tensors(factor=1.0):
    return {"lora_unet_input_blocks_4_proj.lora_down.weight": torch.eye(2) * factor,
            "lora_unet_input_blocks_4_proj.lora_up.weight": torch.eye(2)}


def save(run, name, role, epoch, value):
    phase = "after_epoch_decision" if role == "final" else "step_complete" if role == "step" else "before_epoch_decision"
    moment = WeightMoment(role, epoch, 13 * epoch, 12, phase)
    return Path(publish_weight(run, name, moment, lambda path: save_file(tensors(value), path, metadata={"ss_session_id": run.session_id})))


def test_selected_run_uses_measured_epochs_and_analyzes_once(tmp_path, monkeypatch):
    run = TrainingRun.create(tmp_path, "色見本", "135")
    save(run, "one.safetensors", "epoch", 1, 1)
    save(run, "three_raw.safetensors", "epoch", 3, 2)
    extra = save(run, "step-007.safetensors", "step", 2, 2)
    final = save(run, "trained.safetensors", "final", 3, 4)
    direct = density.collect_single_analysis(str(final), 8)
    reader = RunReader(run.directory)
    calls = []
    original = density.collect_single_analysis
    def counted(path, bins):
        calls.append((Path(path).name, bins))
        return original(path, bins)
    monkeypatch.setattr(density, "collect_single_analysis", counted)
    result, trend, issues = density.analyze_run(reader, 8, trend=True)
    assert not issues
    assert calls == [("one.safetensors", 8), ("trained.safetensors", 8)]
    assert [item["epoch"] for item in trend["files"]] == [1, 3]
    assert result["blocks"] == direct["blocks"]
    report = density.analyze_lora_density(str(extra), 8)
    assert report["checkpoint_history"]["point_count"] == 1 and calls[-1][0] == extra.name
    assert json.loads((run.directory / "density" / "step-007.json").read_text(encoding="utf-8"))["summary"] == report["summary"]
    assert (run.directory / "density" / "step-007.html").is_file()


def test_foreign_weights_and_incomplete_training(tmp_path, monkeypatch):
    run = TrainingRun.create(tmp_path / "root", "partial", "91")
    first = save(run, "partial-000001.safetensors", "epoch", 1, 2)
    result, history, issues = density.analyze_run(RunReader(run.directory), 8, trend=True)
    assert result is None and history["point_count"] == 1 and not issues
    first.unlink()
    result, history, issues = density.analyze_run(RunReader(run.directory), 8, trend=True)
    assert result is None and history["point_count"] == 0 and issues[0]["status"] == "missing"
    foreign = tmp_path / "foreign.safetensors"
    save_file(tensors(), str(foreign))
    monkeypatch.chdir(tmp_path)
    outside = density.analyze_lora_density(str(foreign), 8)
    assert (tmp_path / "analyze_lora_result" / "foreign.html").is_file()
    assert outside["settings"]["bins"] == 8
    unknown = run.directory / "foreign.safetensors"
    save_file(tensors(), str(unknown))
    with pytest.raises(ValueError, match="registered"):
        density.analyze_lora_density(str(unknown), 8)
