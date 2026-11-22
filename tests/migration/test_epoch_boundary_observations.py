"""The hundredth attempt is the last batch of E2; replay both Guardian branches on CPU."""
from argparse import Namespace
import inspect
from math import isclose
from types import SimpleNamespace as NS

import pytest
from safetensors.torch import load_file, save_file
import torch

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


def test_live_trainer_orders_observation_checkpoint_and_decision():
    # This test ties the controlled boundary below to the real trainer's actual branch ordering.
    source = inspect.getsource(NetworkTrainer.train)
    start = source.index("grad_observation = guardian.observe()")
    sequence = ("output.gradient(", "self._step_optimizer(", "apply_max_norm_regularization(",
                "global_step += 1", "output.group_step(", "self.dq_training.finish_attempt(",
                "output.dq_and_rank(", "with run_controller.watch_samples(", "run_controller.process_commands(",
                "if run_controller.stop_requested:", "checkpoint_io.get_epoch_ckpt_name(", "self._save_model(",
                "with run_controller.watch_samples(", "averaging.finish_epoch(", 'output.writer.append("promote"',
                "run_controller.on_epoch_end(")
    for event in sequence:
        start = source.index(event, start) + len(event)


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
    train.training_run = run
    train.all_reduce_network = lambda *unused: None
    train.get_sai_model_spec = lambda unused: {}
    train._metadata, train._minimum_metadata = {}, {}
    accel = NS(sync_gradients=True, print=lambda *unused: None)
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
    dq.begin_attempt(99)
    adapter.delta_quantizer(torch.ones(2,2))
    adapter.lora_up.weight.grad = torch.eye(2) * (200_001 if reject else 1)
    observation = guardian.observe()
    assert observation.skipped == reject
    output.gradient(2, 49, observation, 0.5, None)
    train._step_optimizer(args, accel, net, opt, scheduler, skip_update=observation.skipped)
    output.group_step(batch, torch.tensor([512]), 0.5, 2, 100, reject)
    dq.finish_attempt(100, epoch=2, guardian_skipped=reject)
    output.dq_and_rank(dq, scheduler, 2, 100, reject)

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
    rows = average.finish_epoch(2, 3, net, opt, score, torch.device("cpu"), 1000)
    assert before[:3] == (output.writer.counts, output.group.history, dq.last_result)
    assert torch.equal(before[3], torch.get_rng_state())
    assert rows[0]["promote_applied"] and rows[0]["raw_proxy_loss"] == 2 * diagonal
    output.writer.append("promote", rows)
    assert torch.equal(adapter.lora_up.weight, (expected + torch.eye(2)*.25) / 2)
    # E3 starts from the adopted weight; the skipped attempt is never replayed.
    dq.begin_attempt(100)
    output.group_step(batch, torch.tensor([0]), .125, 3, 101, False)
    dq.finish_attempt(101, epoch=3, guardian_skipped=False)
    output.dq_and_rank(dq, scheduler, 3, 101, False)
    output.finish("completed")
    reader = RunReader(run.directory)
    assert reader.stream("gradient").rows[0]["batch_index"] == 49
    assert [r["attempt_step"] for r in reader.stream("group_loss").rows] == ([101] if reject else [100,101])
    rank = reader.stream("rank").rows
    assert len(rank) == (0 if reject else 1)
    if rank:
        assert isclose(rank[0]["rank_energy"], 2 * diagonal ** 2, rel_tol=2e-5)
    assert [r["attempt_step"] for r in reader.stream("dq").rows] == ([] if reject else [100])
    assert reader.stream("promote").rows[0]["status"] == "scored"
