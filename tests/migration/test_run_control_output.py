from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import json

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.sdxl_lora_training import SdxlLoraTraining
from library import run_control, sampling, sdxl_train_util
from library.training_run import TrainingRun
from library.training_run_weights import WeightMoment, weight_args
from sdxl_train_network import SdxlNetworkTrainer


def test_controller_metadata_and_shared_sample_paths(tmp_path, monkeypatch):
    args = Namespace(output_dir=str(tmp_path), output_name="色見本", run_control=True, no_metadata=False, huggingface_repo_id=None)
    network_state = {"lora_unet_mock.lora_down.weight":torch.ones(2, 5)}
    network = SimpleNamespace(save_weights=lambda path, dtype, metadata: save_file(network_state, path, metadata=metadata))
    accelerator = SimpleNamespace(print=lambda *args: None)
    shared = tmp_path / "sample"
    shared.mkdir()

    # Replace only the neural image calculation. All caller paths and controller notifications are real.
    def sample(accelerator, actual_args, epoch, step, *_):
        assert Path(actual_args.output_dir) == tmp_path and step == 3
        path = shared / f"{actual_args.output_name}_{step:06d}_00_20261122220000_39.png"
        Image.new("RGB", (2,2), (11,12,13)).save(path)
        sampling.notify_sample_saved(str(path))

    monkeypatch.setattr(sdxl_train_util, "sample_images", sample)
    for _ in range(2):
        run = TrainingRun.create(tmp_path, args.output_name, 7)
        trainer = SdxlNetworkTrainer()
        trainer.training_session = SdxlLoraTraining(args, SimpleNamespace(training_features=set()))
        trainer.training_session.run = run
        trainer._metadata = {"ss_session_id":"7", "ss_training_started_at":"100000.0"}
        trainer._minimum_metadata = {}
        monkeypatch.setattr(trainer, "get_sai_model_spec", lambda args: {})
        target = weight_args(args, run)
        control = run_control.create_run_controller(target)
        trainer.run_controller = control
        control.start({}, max_train_steps=3, num_train_epochs=1)
        with control.watch_samples(step=3):
            trainer.sample_images(accelerator,args,None,3,None,None,None,None,None)
        path = trainer._save_model(args=target, accelerator=accelerator, save_dtype=torch.float32, ckpt_name="色見本.safetensors",
                                   unwrapped_nw=network, steps=3, epoch_no=1, moment=WeightMoment("final",1,3,None,"after_epoch_decision"))
        with safe_open(path, framework="pt") as weights:
            assert weights.metadata()["ss_run_id"] == run.run_id
            assert torch.equal(weights.get_tensor(next(iter(network_state))), next(iter(network_state.values())))
        status = json.loads((Path(control.run_dir) / "status.json").read_text("utf-8"))
        assert Path(status["last_checkpoint"]["path"]) == Path(path)
        image = Path(status["last_sample"]["dir"]) / status["last_sample"]["files"][0]
        assert run.run_id in image.name and image.parent == shared and image.is_file()
        assert Path(control.run_dir).parent == run.directory
        control.finish()
    assert len(set(x.read_bytes() for x in shared.iterdir())) == 1 and len(list(shared.iterdir())) == 2
