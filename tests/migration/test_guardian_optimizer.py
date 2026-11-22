"""Update ordering and AMP state at the Guardian decision boundary."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler

from library.grad_norm_guardian import GradNormGuardian
from train_network import NetworkTrainer


def test_skip_changes_only_gradients_and_preserves_update_order():
    trainer = NetworkTrainer()
    events = []
    accel = Mock(sync_gradients=True)
    net = Mock()
    accel.unwrap_model.return_value = net
    trainer.all_reduce_network = lambda *a: events.append("all_reduce")
    accel.clip_grad_norm_.side_effect = lambda *a: events.append("clip")
    net.update_grad_norms.side_effect = lambda: events.append("measure_grad")
    net.update_norms.side_effect = lambda: events.append("measure_weight")
    opt, sched = Mock(), Mock()
    opt.step.side_effect = lambda: events.append("optimizer")
    sched.step.side_effect = lambda: events.append("scheduler")
    opt.zero_grad.side_effect = lambda *, set_to_none: events.append(("zero", set_to_none))
    args = SimpleNamespace(max_grad_norm=1.0)
    trainer._step_optimizer(args, accel, net, opt, sched, skip_update=True)
    assert events == [("zero", True)]
    events.clear()
    trainer._step_optimizer(args, accel, net, opt, sched, skip_update=False)
    assert events == ["all_reduce", "clip", "measure_grad", "measure_weight", "optimizer", "scheduler", ("zero", True)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real fp16 GradScaler required")
def test_guardian_rejection_and_amp_overflow_are_distinct():
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="fp16")
    net = torch.nn.Linear(1, 1, bias=False).cuda()
    param = net.weight
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    prepared, scheduler = accelerator.prepare(optimizer, schedule)
    assert isinstance(prepared, AcceleratedOptimizer)
    assert isinstance(scheduler, AcceleratedScheduler)
    guardian = GradNormGuardian("stable_no_threshoff", net.parameters())
    trainer = NetworkTrainer()
    trainer.all_reduce_network = lambda *args: None
    args = SimpleNamespace(max_grad_norm=0)
    checked = []
    for factor in (1.0, 10.0, float("nan"), 1.0):
        accelerator.backward(param.sum() * factor)
        outcome = guardian.observe()
        before = param.detach().clone()
        scale = accelerator.scaler.get_scale()
        last_epoch = schedule.last_epoch
        trainer._step_optimizer(args, accelerator, net, prepared, scheduler, skip_update=outcome.skipped)
        changed = not torch.equal(param, before)
        if outcome.skipped:
            assert accelerator.scaler.get_scale() == scale
        if torch.isnan(torch.tensor(factor)):
            assert prepared.step_was_skipped
            assert accelerator.scaler.get_scale() < scale
        checked.append((outcome.skipped, changed, schedule.last_epoch - last_epoch, guardian.skipped_steps))
        assert param.grad is None
    assert checked == [(False, True, 1, 0), (True, False, 0, 1), (False, False, 0, 1), (False, True, 1, 1)]
