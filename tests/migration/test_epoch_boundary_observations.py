"""Exercise CPU trainer boundaries and replay a controlled epoch-end promote decision."""
from argparse import Namespace
from math import isclose
from types import SimpleNamespace as NS

import pytest
from safetensors.torch import load_file, save_file
import torch

from library.sdxl_lora_training import SdxlLoraTraining
from library.avg_training import EpochAveraging
from library.dq_training import DQTraining
from library.grad_norm_guardian import GradNormGuardian
from library.maruo_training_config import resolve_training_config
from library.promote_bank import BankItem
from library.training_run import TrainingRun
from library.training_run_output import RunOutput
from library.training_run_weights import WeightMoment, weight_args
from tools.lora_run_reader import RunReader
from train_network import NetworkTrainer


def _network():
    net = torch.nn.Module()
    adapter = torch.nn.Module()
    adapter.lora_name = "lora_unet_test"
    adapter.scale = adapter.multiplier = 1.0
    adapter.lora_down = torch.nn.Linear(2, 2, bias=False)
    adapter.lora_up = torch.nn.Linear(2, 2, bias=False)
    net.add_module(adapter.lora_name, adapter)
    net.unet_loras = [adapter]
    net.text_encoder_loras = []
    with torch.no_grad():
        adapter.lora_down.weight.copy_(torch.eye(2))
        adapter.lora_up.weight.copy_(torch.eye(2) * 0.25)
    def serialize(path, dtype, metadata):
        save_file(dict(net.state_dict()), path, metadata=metadata)
    net.save_weights = serialize
    return net


@pytest.mark.parametrize("reject", [False, True])
def test_epoch_100_boundary(tmp_path, reject):
    args = Namespace(network_module="networks.lora", optimizer_type="AdamW8bit", output_dir=str(tmp_path),
                     output_name="boundary", max_train_steps=150, grad_norm_mode="stable_no_threshoff",
                     dq_delta_bits=8, dq_delta_log=True, dq_delta_log_every=100, dq_delta_scope="unet",
                     rank_log=True, group_loss_log=True, avg_cp=True, avg_cp_mode="promote", avg_mode="uniform",
                     avg_window=2, avg_begin=0, avg_shadow_bank_size=1, avg_shadow_margin=0, avg_shadow_patience=1,
                     avg_reset_stats=False, max_grad_norm=0, no_metadata=False, huggingface_repo_id=None)
    config = resolve_training_config(args)
    run = TrainingRun.create(tmp_path, args.output_name, 17)
    output = RunOutput(run, vars(args))
    net = _network()
    adapter = net.unet_loras[0]
    opt = torch.optim.SGD(net.parameters(), lr=.25)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    guardian = GradNormGuardian("stable_no_threshoff", net.parameters())
    dq = DQTraining(config, net, total_steps=150, lr_warmup_steps=0, num_processes=1)
    average = EpochAveraging(config.values)
    train = NetworkTrainer()
    session = SdxlLoraTraining(args, config)
    train.training_session = session
    session.run, session.output = run, output
    session.guardian, session.dq, session.averaging = guardian, dq, average
    train.all_reduce_network = lambda *unused: None
    train.get_sai_model_spec = lambda unused: {}
    train._metadata, train._minimum_metadata = {}, {}
    accel = NS(sync_gradients=True, device=torch.device("cpu"), unwrap_model=lambda m: m, scaler=None, print=lambda *unused: None)
    dataset = NS(datasets=[NS(batch_size=1, subsets=[NS(group="swatch")], image_data={})])
    batch = {"loss_weights":torch.ones(1), "subset_indices":[(0,0)], "image_keys":["swatch"]}
    output.configure(args=args, config=config, dq=dq, network=net, optimizer=opt, descriptions=["unet"],
                     dataset=dataset, metadata={}, epochs=3, diffusion_steps=1000, averaging=average)

    item = BankItem.capture({**batch, "class_tokens":["swatch"]}, torch.ones(1), torch.ones(1), torch.tensor([100]), None, None)
    for _ in range(average.bank.pool_limit):
        average.bank.add(item)
    average.finish_epoch(1, 3, net, opt, None, torch.device("cpu"), 1000)
    with torch.no_grad():
        adapter.lora_up.weight.copy_(torch.eye(2) * 2)
    session.before_forward(99)
    adapter.delta_quantizer(torch.ones(2,2))
    adapter.lora_up.weight.grad = torch.eye(2) * (200_001 if reject else 1)
    skipped, loss_value = session.after_backward(torch.tensor(.5), 2, 49, accel)
    assert skipped == reject
    train._step_optimizer(args, accel, net, opt, scheduler, skip_update=skipped)
    session.group_timesteps = torch.tensor([512])
    session.after_sync_step(batch, loss_value, 2, 100, scheduler, skipped)

    diagonal = 2.0 if reject else 1.75
    expected = torch.eye(2) * diagonal
    disk = train._save_model(args=weight_args(args,run), accelerator=accel, save_dtype=torch.float32,
                             ckpt_name="boundary-000002.safetensors", unwrapped_nw=net, steps=100, epoch_no=2,
                             moment=WeightMoment("epoch",2,100,None,"before_epoch_decision"))
    assert torch.equal(load_file(disk)["lora_unet_test.lora_up.weight"], expected)
    before = (dict(output.writer.counts), dict(output.group.history), dq.last_result, torch.get_rng_state().clone())
    def score(_):
        adapter.delta_quantizer(torch.ones(2,2))
        return adapter.lora_up.weight.trace().item()
    session.score = score
    session.after_epoch_save(2, 3, net, opt, accel, 1000, 100)
    assert all(output.writer.counts[key] == count for key, count in before[0].items() if key != "promote")
    assert before[1:3] == (output.group.history, dq.last_result)
    assert torch.equal(before[3], torch.get_rng_state())
    assert torch.equal(adapter.lora_up.weight, (expected + torch.eye(2)*.25) / 2)
    # E3 starts from the adopted weight; the skipped attempt is never replayed.
    session.before_forward(100)
    session.group_timesteps = torch.tensor([0])
    session.after_sync_step(batch, .125, 3, 101, scheduler, False)
    session.finish("completed")
    reader = RunReader(run.directory)
    assert reader.stream("gradient").rows[0]["batch_index"] == 49
    assert [r["attempt_step"] for r in reader.stream("group_loss").rows] == ([101] if reject else [100,101])
    rank = reader.stream("rank").rows
    assert len(rank) == (0 if reject else 1)
    if rank:
        assert isclose(rank[0]["rank_energy"], 2 * diagonal ** 2, rel_tol=2e-5)
    assert [r["attempt_step"] for r in reader.stream("dq").rows] == ([] if reject else [100])
    decision = reader.stream("promote").rows[0]
    assert decision["status"] == "scored"
    assert decision["promote_applied"] and decision["raw_proxy_loss"] == 2 * diagonal


def _tiny_trainer(tmp_path, monkeypatch, mode):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState, PartialState
    from library import dataset, strategy_base
    from sdxl_train_network import SdxlNetworkTrainer, setup_parser as sdxl_parser
    import train_network

    sdxl = mode != "common"
    guardian = mode == "sdxl-guardian"
    accumulation = 2 if mode == "sdxl-observation" else 1

    class TinyDataset(dataset.MinimalDataset):
        def __len__(self):
            return accumulation

        def __getitem__(self, index):
            return {"loss_weights": torch.ones(1), "subset_indices": [(0, 0)]}

    class Model(torch.nn.Sequential):
        @property
        def device(self):
            return next(self.parameters()).device

    # LoRA discovers this target by its class name, as it does on a real UNet.
    class Transformer2DModel(torch.nn.Sequential):
        pass

    trainer = SdxlNetworkTrainer() if sdxl else NetworkTrainer()
    parser = sdxl_parser() if sdxl else train_network.setup_parser()
    args = parser.parse_args([
        "--network_module", "networks.lora", "--network_dim", "1", "--network_train_unet_only",
        "--dataset_class", "test.TinyDataset", "--output_dir", str(tmp_path), "--output_name", "test",
        "--max_train_epochs", "3", "--save_every_n_epochs", "1", "--save_model_as", "safetensors",
        "--max_data_loader_n_workers", "0", "--seed", "12",
        "--gradient_accumulation_steps", str(accumulation),
        *(["--grad_norm_mode", "stable_no_threshoff", "--scale_weight_norms", "100"] if guardian else []),
        *(["--group_loss_log"] if accumulation == 2 else []),
    ])
    # Give this CPU trainer its own Accelerate state even when a previous test initialized CUDA.
    # monkeypatch restores each class's original shared-state descriptor after the run.
    for state in (PartialState, AcceleratorState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    for strategy in (strategy_base.TokenizeStrategy, strategy_base.LatentsCachingStrategy,
                     strategy_base.TextEncodingStrategy, strategy_base.TextEncoderOutputsCachingStrategy):
        monkeypatch.setattr(strategy, "_strategy", None)
    monkeypatch.setattr(dataset, "load_arbitrary_dataset", lambda args: TinyDataset((64, 64), 1))
    monkeypatch.setattr(
        train_network.accelerator_setup, "prepare_accelerator",
        lambda args: Accelerator(cpu=True, gradient_accumulation_steps=args.gradient_accumulation_steps),
    )
    monkeypatch.setattr(trainer, "get_tokenize_strategy", lambda args: NS(tokenizer=None, tokenizer1=None, tokenizer2=None))
    unet = Model(Transformer2DModel(torch.nn.Linear(2, 2)))
    encoders = [Model(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)) for _ in range(2 if sdxl else 1)]
    monkeypatch.setattr(trainer, "load_target_model", lambda *args: ("test", encoders, torch.nn.Identity(), unet))

    batches = []

    def backward(args, accelerator, batch, text_encoders, unet, network, *unused):
        loss = unet(torch.ones(1, 2)).sum()
        accelerator.backward(loss)
        if accumulation == 2:
            trainer.training_session.observe_target(batch, None, torch.tensor([len(batches)]), None, None)
        if len(batches) == 1 and accumulation == 1:
            next(p for p in network.parameters() if p.grad is not None).grad.fill_(300_000)
        batches.append(loss.item())
        return loss

    monkeypatch.setattr(trainer, "forward_backward", backward)
    return trainer, args, batches


@pytest.mark.parametrize("mode", ["common", "sdxl-guardian", "sdxl-observation"])
def test_real_trainer_update_and_save_order(tmp_path, monkeypatch, mode):
    trainer, args, batches = _tiny_trainer(tmp_path, monkeypatch, mode)
    sdxl = mode != "common"
    guardian = mode == "sdxl-guardian"
    epoch_files_at_decision = []
    if sdxl:
        finish_epoch = SdxlLoraTraining.after_epoch_save
        sample_images = trainer.sample_images
        sampled_epochs = set()

        def track_sample(accelerator, args, epoch, *rest):
            sample_images(accelerator, args, epoch, *rest)
            if epoch is not None and epoch > 0:  # Only the end-of-epoch call.
                sampled_epochs.add(epoch)

        monkeypatch.setattr(trainer, "sample_images", track_sample)

        def check_epoch(session, epoch, *args):
            assert epoch in sampled_epochs, f"Epoch {epoch} sample hook must finish before the promote decision"
            epoch_files_at_decision.append((session.run.directory / f"test-{epoch:06}.safetensors").is_file())
            return finish_epoch(session, epoch, *args)

        monkeypatch.setattr(SdxlLoraTraining, "after_epoch_save", check_epoch)

    trainer.train(args)
    assert len(batches) == 3 * args.gradient_accumulation_steps
    if sdxl:
        directory, = tmp_path.iterdir()
        assert epoch_files_at_decision == [True, True, False]
        reader = RunReader(directory)
        if guardian:
            assert len(reader.stream("gradient").rows) == 3
        else:
            rows = reader.stream("group_loss").rows
            assert [row["attempt_step"] for row in rows] == [1, 2, 3]
            assert [row["timestep"] for row in rows] == [1, 3, 5]
    else:
        directory = tmp_path
        assert all(item.is_file() for item in directory.iterdir())
    first, second, last = [load_file(str(directory / name)) for name in
                           ("test-000001.safetensors", "test-000002.safetensors", "test.safetensors")]
    assert all(torch.equal(first[k], second[k]) for k in first) == guardian
    assert any(not torch.equal(last[k], second[k]) for k in last)


@pytest.mark.parametrize("interrupt", ["stop", "failure"])
def test_interrupted_trainer_finalizes_run_before_epoch_decision(tmp_path, monkeypatch, interrupt):
    from library import run_control
    import json

    trainer, args, batches = _tiny_trainer(tmp_path, monkeypatch, "sdxl-guardian")
    args.report_after_training = True

    def interrupt_at_sync(controller, handlers, *, step):
        assert step == 1
        if interrupt == "failure":
            raise RuntimeError("simulated interruption")
        handlers["stop"]({"save": True, "save_state": False})

    monkeypatch.setattr(run_control.RunController, "process_commands", interrupt_at_sync)
    if interrupt == "failure":
        with pytest.raises(RuntimeError, match="simulated interruption"):
            trainer.train(args)
    else:
        trainer.train(args)

    assert len(batches) == 1 and trainer.training_session is None
    directory, = tmp_path.iterdir()
    index = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert index["status"] == ("failed" if interrupt == "failure" else "stopped")
    assert not (directory / "diagnostic_report").exists()
    assert [weight["role"] for weight in index["weights"]] == (["step"] if interrupt == "stop" else [])
    for weight in index["weights"]:
        assert weight["attempt_step"] == 1 and (directory / weight["path"]).is_file()
