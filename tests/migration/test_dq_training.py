from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.dq_training import DQTraining
from library.maruo_training_config import resolve_training_config
from library.rounding_util import quantize_delta
from networks.lora import LoRAModule


def configure(network, *, steps=6, warmup=0, processes=1, **options):
    config = resolve_training_config(Namespace(network_module="networks.lora", lr_warmup_steps=warmup, **options))
    return DQTraining(config, network, total_steps=steps, lr_warmup_steps=warmup, num_processes=processes)


def layers(*, device="cpu", dropout=None, rank_dropout=None, module_dropout=None):
    names = ("lora_unet_mid_attn2_to_v", "lora_te1_encoder_layers_11_mlp_fc2", "lora_te2_encoder_layers_31_mlp_fc2")
    adapters = []
    for name in names:
        base = nn.Linear(4, 6, bias=False).to(device)
        adapter = LoRAModule(name, base, lora_dim=2, alpha=1, dropout=dropout,
                             rank_dropout=rank_dropout, module_dropout=module_dropout).to(device)
        nn.init.uniform_(adapter.lora_up.weight, -1.0, 1.0)
        adapter.apply_to()
        adapters.append(adapter)
    return SimpleNamespace(unet_loras=adapters[:1], text_encoder_loras=adapters[1:]), adapters


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("scope", ["both", "unet", "te"])
def test_scope_dropout_rng_gradients_and_fixed_assignments(device, scope):
    torch.manual_seed(4)
    net, modules = layers(device=device, dropout=0.3, rank_dropout=0.2)
    runner = configure(net, dq_delta_bits=8, dq_delta_scope=scope, dq_delta_range_mul=2.7,
                       dq_delta_range_mul_attn2=3.75, dq_delta_range_mul_te=3.75,
                       dq_delta_range_mul_te2=3.45, dq_delta_log=True, dq_delta_log_scope="both", dq_delta_log_every=1)
    loaded_state = [{k: t.clone() for k, t in module.state_dict().items()} for module in modules]
    runner.begin_attempt(0)
    for i, module in enumerate(modules):
        assert set(loaded_state[i]) == {"alpha", "lora_down.weight", "lora_up.weight"}
        module.load_state_dict(loaded_state[i])  # switching to an epoch candidate must preserve the binding
        enabled = scope == "both" or ((scope == "unet") == (i == 0))
        mul = 3.45 if i == 2 else 3.75
        x = torch.randn(2, 3, 4, device=device, requires_grad=True)
        params = (x, module.lora_up.weight, module.lora_down.weight)
        torch.manual_seed(139)
        low = nn.functional.dropout(module.lora_down(x), 0.3)
        mask = (torch.rand(2, 2, device=device) > 0.2).unsqueeze(1)
        delta = module.lora_up(low * mask) * module.multiplier * (module.scale * (1.0 / 0.8))
        ref = module.org_forward(x) + (quantize_delta(delta, bits=8, range_mul=mul) if enabled else delta)
        expected_grads = torch.autograd.grad(ref.sum(), params)
        expected_next = torch.rand(16, device=device)
        torch.manual_seed(139)
        actual = module(x)
        actual_grads = torch.autograd.grad(actual.sum(), params)
        torch.testing.assert_close(ref, actual, atol=0, rtol=0)
        for old, new in zip(expected_grads, actual_grads):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
        torch.testing.assert_close(torch.rand(16, device=device), expected_next, atol=0, rtol=0)
    result = runner.finish_attempt(1, epoch=1, guardian_skipped=False)
    counts = {row["scope"]: row["numel"] for row in result.log_rows}
    assert counts == {"unet": 0 if scope == "te" else 36, "te": 0 if scope == "unet" else 72}
    assert all(row["common_mul_after"] == 2.7 for row in result.log_rows)


def test_schedule_missing_interval_and_guardian_rejection_do_not_advance_averages_or_rng():
    net, (unet, _, _) = layers()
    runner = configure(net, steps=8, dq_delta_bits_sched="0.25:2,0.5:5", dq_delta_log=True,
                       dq_delta_log_every=2, dq_delta_scope="unet")
    x = torch.ones(2, 4)
    raw = unet.org_forward(x) + unet.lora_up(unet.lora_down(x)) * unet.scale
    seen_bits = []
    for g in range(6):
        runner.begin_attempt(g)
        seen_bits.append(runner.active_bits)
        torch.manual_seed(123)
        out = unet(x)
        if g < 2:
            torch.testing.assert_close(out, raw, rtol=0, atol=0)
            actual_next = torch.rand(4)
            torch.manual_seed(123)
            torch.testing.assert_close(actual_next, torch.rand(4), rtol=0, atol=0)
        if g == 3:
            assert runner.totals and runner.error_rms_ema is None
        rng_after_forward = torch.random.get_rng_state()
        result = runner.finish_attempt(g + 1, epoch=1, guardian_skipped=g == 3)
        assert torch.equal(torch.random.get_rng_state(), rng_after_forward)
        assert (result is None) == (g % 2 == 0 or g == 3)
        if g == 1:
            assert result.log_rows[0]["bits"] is None
            assert result.log_rows[0]["numel"] == 0 and result.log_rows[0]["clip_rate_raw"] is None
        if g == 3:
            assert not runner.totals and runner.error_rms_ema is None
        if g == 5:
            assert result.log_rows[0]["numel"] == 12 and result.log_rows[0]["log_scope_error_rms_ema"] is not None
    assert seen_bits == [None, None, 2, 2, 5, 5]
    with pytest.raises(RuntimeError, match="attempt mismatch"):
        runner.finish_attempt(6, epoch=1, guardian_skipped=False)


@pytest.mark.parametrize("warmup,processes", [(2, 1), (0.25, 1), (0.125, 2)])
def test_warmup_boundary_and_evaluation_or_scoring_do_not_write_stats(warmup, processes):
    net, (unet, _, _) = layers()
    runner = configure(net, steps=8, warmup=warmup, processes=processes, dq_delta_bits=8,
                       dq_delta_begin_after_lr_warmup=True, dq_delta_log=True, dq_delta_log_every=1)
    x = torch.ones(2, 4)
    for g in range(3):
        runner.begin_attempt(g)
        unet(x)
        assert (runner.active_bits == 8) == (g >= 2)
        runner.finish_attempt(g + 1, epoch=1, guardian_skipped=False)
    before = runner.last_result
    rng = torch.random.get_rng_state()
    unet.eval()
    unet(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    unet.train()
    with torch.no_grad():
        unet(x)
    assert not torch.equal(rng, torch.random.get_rng_state())  # promote uses active stochastic rounding
    assert runner.last_result is before and not runner.totals and not runner.observers


def test_module_dropout_only_consumes_the_upstream_gate_draw():
    net, (module, _, _) = layers(module_dropout=1.0)
    runner = configure(net, dq_delta_bits=8, dq_delta_log=True, dq_delta_log_every=1)
    x = torch.ones(2, 4)
    runner.begin_attempt(0)
    torch.manual_seed(48)
    torch.rand(1)
    expected_next = torch.rand(4)
    torch.manual_seed(48)
    actual = module(x)
    torch.testing.assert_close(actual, module.org_forward(x), atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(4), expected_next, atol=0, rtol=0)
    assert not runner.totals


def test_same_forward_can_feed_both_scopes_and_auto_uses_the_unet_measurement():
    net, modules = layers()
    runner = configure(net, dq_delta_bits=8, dq_delta_log=True, dq_delta_log_every=2,
                       dq_delta_auto_range_mul=True, dq_delta_auto_every=1, dq_delta_scope="both")
    x = torch.ones(4, 4)
    for g in range(2):
        runner.begin_attempt(g)
        for m in modules:
            m(x)
        result = runner.finish_attempt(g + 1, epoch=1, guardian_skipped=False)
        assert result.auto_metrics["numel"] == 24
        if g == 0:
            assert not result.log_rows and result.auto_metrics["quant_error_ratio_raw"] is None
        else:
            unet, te = result.log_rows
            assert unet["numel"] == 24 and te["numel"] == 48
            assert result.auto_metrics["clip_rate_raw"] == unet["clip_rate_raw"]
            assert result.auto_metrics["quant_error_ratio_raw"] == unet["quant_error_ratio_raw"]
