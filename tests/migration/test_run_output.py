from argparse import Namespace
import json
from types import SimpleNamespace

import pytest
import torch

from library.avg_training import EpochAveraging
from library.dq_training import DQTraining
from library.grad_norm_guardian import GradientObservation
from library.maruo_training_config import resolve_training_config
from library.training_run_output import reserve_run, RunOutput
from tools.lora_run_reader import RunReader


def make_run(root, **changes):
    args = Namespace(output_dir=str(root), output_name="色見本", network_module="networks.lora", max_train_steps=5,
                     optimizer_type="AdamW8bit", **changes)
    config = resolve_training_config(args)
    run = reserve_run(args, SimpleNamespace(is_main_process=True, num_processes=1), 129, 100000.0)
    output = RunOutput(run, vars(args))
    module = torch.nn.Linear(2, 2)
    module.lora_name = "lora_unet_mock"
    net = SimpleNamespace(unet_loras=[module], text_encoder_loras=[])
    opt = torch.optim.SGD(net.unet_loras[0].parameters(), lr=0.03)
    dataset = SimpleNamespace(datasets=[SimpleNamespace(batch_size=1, subsets=[SimpleNamespace(group="色見本")],
                                                        image_data={"a":SimpleNamespace(bucket_reso=(768,1024))})])
    dq = DQTraining(config, net, total_steps=5, lr_warmup_steps=0, num_processes=1) if "dq" in config.training_features else None
    avg = EpochAveraging(config.values) if "averaging" in config.training_features else None
    output.configure(args=args, config=config, dq=dq, network=net, optimizer=opt, descriptions=["unet"],
                     dataset=dataset, metadata={"ss_training_started_at":"100000.0", "ss_session_id":"129"},
                     epochs=2, diffusion_steps=1000, averaging=avg)
    return output, dq


def test_output_settings_log_warmup_and_guardian_skip(tmp_path):
    out, dq = make_run(tmp_path, dq_delta_bits=8, dq_delta_log=True, dq_delta_log_every=1,
                       dq_delta_auto_range_mul=True, dq_delta_auto_preset="clip_rate_high", dq_delta_auto_every=1,
                       grad_norm_mode="stable_no_threshoff", group_loss_log=True, avg_cp=True, avg_cp_mode="promote")
    batch = {"loss_weights": torch.ones(1), "image_keys":["a"], "subset_indices":[(0,0)]}
    out.group.validate_batch(batch)
    for n, skipped in [(1,False), (2,True)]:
        out.gradient(1,n-1, GradientObservation(4.0,3.0,skipped),0.125, SimpleNamespace(get_scale=lambda:32768.0))
        dq.begin_attempt(n-1)
        out.group_step(batch, torch.tensor([0]), 0.125, 1, n, skipped)
        dq.finish_attempt(n, epoch=1, guardian_skipped=skipped)
        out.dq_and_rank(dq, None, 1, n, skipped)
    out.writer.end_epoch(2)
    out.finish("stopped")
    rd = RunReader(out.run.directory)
    assert rd.index["status"] == "stopped" and rd.index["flush_status"] == "complete"
    assert [r["attempt_step"] for r in rd.stream("dq_auto").rows] == [0,1]
    assert [r["attempt_step"] for r in rd.stream("group_loss").rows] == [1]
    assert rd.stream("group_loss").rows[0]["bucket_reso"] == "768x1024"
    assert len(rd.stream("gradient").rows) == 2 and rd.stream("promote").rows == []
    cfg, reason = rd.settings()
    assert cfg["metadata"]["ss_training_started_at"] == "100000.0"
    assert cfg["training"]["total_epochs"] == 2 and cfg["training"]["num_train_timesteps"] == 1000
    assert cfg["dq"]["auto"]["init"] == dq.auto_controller.settings["init"] and cfg["promote"]["pool_target_size"] == 48


def test_off_run_no_training_rng_or_output_collisions(tmp_path):
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    args = Namespace(output_dir=str(tmp_path), output_name=None)
    for _ in range(2):
        r = reserve_run(args, SimpleNamespace(is_main_process=True,num_processes=1), 1, 100000.)
        out = RunOutput(r, vars(args))
        out.finish("failed")
        assert RunReader(r.directory).index["status"] == "failed"
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(list(tmp_path.glob("*last*"))) == 2


def test_zero_observations_and_disabled_streams(tmp_path):
    out, _ = make_run(tmp_path, rank_log=True)
    assert out.finish("completed")
    rd = RunReader(out.run.directory)
    assert rd.index["status"] == "completed"
    assert rd.stream("rank").status == "complete" and rd.stream("rank").rows == []
    assert rd.index["artifacts"]["rank"]["expected_modules_by_scope"] == {"unet":["lora_unet_mock"]}
    assert rd.stream("dq").status == "disabled"
