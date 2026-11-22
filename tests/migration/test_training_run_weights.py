import argparse
from pathlib import Path
import json

import pytest
from library.training_run import TrainingRun
from library.training_run_weights import WeightMoment, publish_weight, remove_weight, weight_args, sample_args


def weights(run):
    return json.loads((run.directory / "run.json").read_text("utf-8"))["weights"]


def test_atomic_extension_registration_rotation(tmp_path):
    r = TrainingRun.create(tmp_path, "a", 1)
    moment = WeightMoment("step", 1, 2, 1, "step_complete")
    def store(name):
        assert name.endswith(".safetensors")
        Path(name).write_bytes(b"good")
    filename = "a-step00000002.safetensors"
    actual = publish_weight(r, filename, moment, store)
    publish_weight(r, filename, moment, store)
    assert Path(actual).read_bytes() == b"good" and len(weights(r)) == 1
    def fail(name):
        Path(name).write_bytes(b"partial")
        raise OSError("disk full")
    with pytest.raises(OSError):
        publish_weight(r, filename, moment, fail)
    assert Path(actual).read_bytes() == b"good"
    assert not list(r.directory.glob(".weight-*"))
    assert remove_weight(r, filename) == "pruned" and weights(r)[0]["status"] == "pruned"
    publish_weight(r, filename, moment, store)
    Path(actual).unlink()
    assert remove_weight(r, filename) == "missing" and weights(r)[0]["status"] == "missing"


def test_uncertain_index_preserves_published_file(tmp_path, monkeypatch):
    run = TrainingRun.create(tmp_path, None, 8)
    moment = WeightMoment("final", 3, 10, None, "after_epoch_decision")
    def fail(*args):
        raise OSError("index publication failed")
    monkeypatch.setattr(TrainingRun, "update_index", fail)
    with pytest.raises(OSError):
        publish_weight(run, "last.safetensors", moment, lambda path: Path(path).write_bytes(b"final"))
    assert (run.directory / "last.safetensors").read_bytes() == b"final" and not weights(run)


def test_shared_samples_and_omitted_checkpoint_stems(tmp_path):
    from library.checkpoint_io import get_step_ckpt_name, get_epoch_ckpt_name, get_last_ckpt_name
    a = argparse.Namespace(output_dir=str(tmp_path), output_name=None)
    r1, r2 = (TrainingRun.create(tmp_path, a.output_name, 8) for _ in range(2))
    w = weight_args(a, r1)
    assert get_step_ckpt_name(w, ".pt", 2) == "at-step00000002.pt"
    assert get_epoch_ckpt_name(w, ".pt", 1) == "epoch-000001.pt" and get_last_ckpt_name(w, ".pt") == "last.pt"
    assert sample_args(a, r1).output_name != sample_args(a, r2).output_name
    assert sample_args(a, r1).output_dir == str(tmp_path) and sample_args(a, r1).output_name == r1.run_id
    assert w.output_dir == str(r1.directory)
    assert a.output_name is None and weight_args(a, None) is a and sample_args(a, None) is a


@pytest.mark.parametrize("initial_interval", [None, 1])
def test_all_off_epoch_state_retention(tmp_path, initial_interval):
    from types import SimpleNamespace
    from library.checkpoint_io import save_and_remove_state_on_epoch_end
    from library.maruo_training_config import resolve_training_config
    from library.training_run_weights import resolve_epoch_save_args
    from sdxl_train_network import parse_training_args, setup_parser

    argv = ["--network_module", "networks.lora", "--output_dir", str(tmp_path),
            "--save_n_epoch_ratio", "3", "--max_train_epochs", "6", "--save_state",
            "--save_last_n_epochs_state", "1"]
    if initial_interval:
        argv.extend(["--save_every_n_epochs", str(initial_interval)])
    args = parse_training_args(setup_parser(), argv)
    config = resolve_training_config(args)
    assert not config.training_features and not config.observation_features
    run = TrainingRun.create(tmp_path, None, 1)
    saving = resolve_epoch_save_args(args, run, args.max_train_epochs)
    assert args.save_every_n_epochs == saving.save_every_n_epochs == 2
    accelerator = SimpleNamespace(save_state=lambda path: Path(path).mkdir())
    second = Path(save_and_remove_state_on_epoch_end(saving, accelerator, 2))
    assert second.is_dir() and second.parent == run.directory
    fourth = Path(save_and_remove_state_on_epoch_end(saving, accelerator, 4))
    assert fourth.is_dir() and not second.exists()
