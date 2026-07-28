from __future__ import annotations

import logging
from typing import Callable

import pytest
import torch

import networks.lora as lora_impl
from library.triton_lora import get_triton_rank4_quantized_lora_up_diagnostics


def _cuda_c0_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    info = get_triton_rank4_quantized_lora_up_diagnostics("cuda")
    return bool(
        info["triton_available"]
        and info.get("device_supported")
        and tuple(info.get("device_capability") or ()) == (12, 0)
    )


requires_c0_cuda = pytest.mark.skipif(
    not _cuda_c0_supported(),
    reason="LoRAModule C0 integration tests require validated CUDA capability (12, 0)",
)


def _make_rank4_module(*, name: str = "lora_unet_test") -> lora_impl.LoRAModule:
    torch.manual_seed(1729)
    original = torch.nn.Linear(8, 640, bias=False)
    module = lora_impl.LoRAModule(
        name,
        original,
        multiplier=0.73,
        lora_dim=4,
        alpha=2,
        delta_q_mode="stoch",
        delta_q_granularity="channel",
        delta_q_stat="rms",
        delta_q_bits=8,
        delta_q_range_mul=3.0,
        delta_q_on_z=False,
        delta_q_use_triton=True,
        delta_q_triton_stats=False,
        delta_q_triton_fused_up_mode="off",
        delta_q_triton_fused_up_scope="unet",
    )
    with torch.no_grad():
        original.weight.zero_()
        module.lora_down.weight.normal_(mean=0.0, std=0.2)
        module.lora_up.weight.normal_(mean=0.0, std=0.2)
    original.requires_grad_(False)
    module.cuda()
    module.apply_to()
    module.train()
    return module


def _run_module(
    module: lora_impl.LoRAModule,
    x: torch.Tensor,
    upstream: torch.Tensor,
    rng_state: torch.Tensor,
    *,
    fused_mode: str,
) -> dict[str, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    module.delta_q_triton_fused_up_mode = fused_mode
    x_run = x.detach().clone().requires_grad_(True)
    torch.cuda.set_rng_state(rng_state)
    with torch.autocast("cuda", dtype=torch.float16):
        output = module(x_run)
        loss = (output * upstream).sum()
    loss.backward()
    torch.cuda.synchronize()
    return {
        "output": output.detach().clone(),
        "down_grad": module.lora_down.weight.grad.detach().clone(),
        "up_grad": module.lora_up.weight.grad.detach().clone(),
        "input_grad": x_run.grad.detach().clone(),
        "rng_state": torch.cuda.get_rng_state().clone(),
    }


def _assert_normal_route_exact(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    for key in ("output", "down_grad", "up_grad", "input_grad", "rng_state"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@requires_c0_cuda
def test_lora_module_c0_matches_existing_triton_route_forward_backward_and_rng():
    module = _make_rank4_module()
    torch.manual_seed(1234)
    x = torch.randn(1, 77, 8, device="cuda", dtype=torch.float16)
    upstream = torch.randn(1, 77, 640, device="cuda", dtype=torch.float16)
    rng_state = torch.cuda.get_rng_state().clone()

    baseline = _run_module(module, x, upstream, rng_state, fused_mode="off")
    c0 = _run_module(module, x, upstream, rng_state, fused_mode="c0")

    forward_rel_l2 = torch.linalg.vector_norm((c0["output"] - baseline["output"]).float())
    forward_rel_l2 /= torch.linalg.vector_norm(baseline["output"].float()).clamp_min(1.0e-12)
    assert forward_rel_l2.item() <= 1.0e-3
    torch.testing.assert_close(c0["output"], baseline["output"], rtol=1.0e-3, atol=5.0e-4)
    torch.testing.assert_close(c0["down_grad"], baseline["down_grad"], rtol=0, atol=0)
    torch.testing.assert_close(c0["up_grad"], baseline["up_grad"], rtol=0, atol=0)
    torch.testing.assert_close(c0["input_grad"], baseline["input_grad"], rtol=0, atol=0)
    torch.testing.assert_close(c0["rng_state"], baseline["rng_state"], rtol=0, atol=0)
    assert module.delta_q_fused_up_ever_attempted
    assert module.delta_q_fused_up_ever_succeeded


@requires_c0_cuda
def test_c0_kernel_fallback_reuses_precreated_rand_and_preserves_output_and_rng(monkeypatch):
    module = _make_rank4_module()
    torch.manual_seed(2345)
    x = torch.randn(1, 77, 8, device="cuda", dtype=torch.float16)
    upstream = torch.randn(1, 77, 640, device="cuda", dtype=torch.float16)
    rng_state = torch.cuda.get_rng_state().clone()

    baseline = _run_module(module, x, upstream, rng_state, fused_mode="off")

    torch.cuda.set_rng_state(rng_state)
    expected_rand = torch.rand((1, 77, 640), device="cuda", dtype=torch.float32)
    expected_rng_state = torch.cuda.get_rng_state().clone()

    captured: dict[str, torch.Tensor] = {}
    original_fake_quantize_levels: Callable = lora_impl.fake_quantize_levels

    def unavailable_kernel(*args, rand: torch.Tensor, **kwargs):
        captured["kernel_rand"] = rand
        return None

    def capture_fallback_rand(*args, **kwargs):
        rand = kwargs.get("rand")
        if rand is not None:
            captured["fallback_rand"] = rand
        return original_fake_quantize_levels(*args, **kwargs)

    monkeypatch.setattr(lora_impl, "triton_rank4_delta_quant", unavailable_kernel)
    monkeypatch.setattr(lora_impl, "fake_quantize_levels", capture_fallback_rand)
    module.delta_q_fused_up_ever_attempted = False
    module.delta_q_fused_up_ever_succeeded = False
    fallback = _run_module(module, x, upstream, rng_state, fused_mode="c0")

    _assert_normal_route_exact(fallback, baseline)
    assert captured["kernel_rand"] is captured["fallback_rand"]
    torch.testing.assert_close(captured["kernel_rand"], expected_rand, rtol=0, atol=0)
    torch.testing.assert_close(fallback["rng_state"], expected_rng_state, rtol=0, atol=0)
    assert module.delta_q_fused_up_ever_attempted
    assert not module.delta_q_fused_up_ever_succeeded
    assert module.delta_q_fused_up_last_fallback_reason == "kernel"


@requires_c0_cuda
@pytest.mark.parametrize("disabled_by", ["scope", "gradient_checkpointing"])
def test_scope_and_gradient_checkpointing_bypass_c0_without_attempt(
    monkeypatch, disabled_by: str
):
    name = "lora_te_test" if disabled_by == "scope" else "lora_unet_test"
    module = _make_rank4_module(name=name)
    if disabled_by == "gradient_checkpointing":
        module.delta_q_gradient_checkpointing = True

    calls = 0

    def unexpected_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("C0 kernel must not be called")

    monkeypatch.setattr(lora_impl, "triton_rank4_delta_quant", unexpected_kernel)
    torch.manual_seed(3456)
    x = torch.randn(1, 77, 8, device="cuda", dtype=torch.float16)
    upstream = torch.randn(1, 77, 640, device="cuda", dtype=torch.float16)
    rng_state = torch.cuda.get_rng_state().clone()

    baseline = _run_module(module, x, upstream, rng_state, fused_mode="off")
    module.delta_q_fused_up_ever_attempted = False
    module.delta_q_fused_up_ever_succeeded = False
    bypassed = _run_module(module, x, upstream, rng_state, fused_mode="c0")

    _assert_normal_route_exact(bypassed, baseline)
    assert calls == 0
    assert not module.delta_q_fused_up_ever_attempted
    assert not module.delta_q_fused_up_ever_succeeded


@requires_c0_cuda
def test_no_grad_forward_bypasses_training_only_c0_without_attempt(monkeypatch):
    module = _make_rank4_module()
    calls = 0

    def unexpected_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("C0 kernel must not run during no-grad sample generation")

    monkeypatch.setattr(lora_impl, "triton_rank4_delta_quant", unexpected_kernel)
    module.delta_q_triton_fused_up_mode = "c0"
    module.delta_q_fused_up_ever_attempted = False
    module.delta_q_fused_up_ever_succeeded = False
    x = torch.randn(1, 77, 8, device="cuda", dtype=torch.float16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = module(x)

    assert output.shape == (1, 77, 640)
    assert calls == 0
    assert not module.delta_q_fused_up_ever_attempted
    assert not module.delta_q_fused_up_ever_succeeded


@requires_c0_cuda
def test_c0_basic_stats_are_added_to_manager_accumulator_and_remain_non_grad(monkeypatch):
    module = _make_rank4_module()
    module.delta_q_triton_stats = True
    module.delta_q_triton_fused_up_mode = "c0"
    manager = lora_impl.DQStatsManager()
    manager.begin_step(
        step_idx=1,
        device=torch.device("cuda"),
        do_log=True,
        do_auto=False,
        collect_full=True,
        collect_zero=False,
        collect_near_zero=False,
        collect_detail=False,
        collect_error_parts=False,
        log_mode="summary",
        log_scope="unet",
        auto_scope="unet",
        target="delta",
    )
    module.dq_stats_manager = manager

    packed_results: list[torch.Tensor] = []
    actual_kernel = lora_impl.triton_rank4_delta_quant

    def capture_stats(*args, **kwargs):
        result = actual_kernel(*args, **kwargs)
        if result is not None and result[1] is not None:
            packed_results.append(result[1])
        return result

    monkeypatch.setattr(lora_impl, "triton_rank4_delta_quant", capture_stats)
    torch.manual_seed(4567)
    x = torch.randn(1, 77, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = module(x)
    torch.cuda.synchronize()

    accumulator = manager.accum["unet"]
    assert isinstance(accumulator, lora_impl.DQStatsAccumulator)
    assert accumulator.basic_stats is not None
    assert packed_results
    assert not packed_results[0].requires_grad
    assert not accumulator.basic_stats.requires_grad
    torch.testing.assert_close(accumulator.basic_stats, packed_results[0].detach(), rtol=0, atol=0)
    assert accumulator.numel.item() == output.numel()
    assert output.requires_grad
    assert module.delta_q_fused_up_ever_succeeded


def test_requested_c0_with_zero_success_logs_end_warning(caplog):
    network = lora_impl.LoRANetwork.__new__(lora_impl.LoRANetwork)
    torch.nn.Module.__init__(network)
    network.text_encoder_loras = []
    network.unet_loras = []
    network.delta_q_triton_fused_up_mode = "c0"
    network.delta_q_triton_fused_up_scope = "unet"
    network.dq_fused_up_diagnostics_manager = None

    with caplog.at_level(logging.WARNING, logger=lora_impl.__name__):
        network.log_delta_triton_fused_up_diagnostics(warn_on_zero=False)
    assert not any(
        "requested but completed with zero successful C0 modules" in record.getMessage()
        for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=lora_impl.__name__):
        summary = network.log_delta_triton_fused_up_diagnostics()

    assert summary["successful_modules"] == 0
    assert any(
        "requested but completed with zero successful C0 modules" in record.getMessage()
        for record in caplog.records
    )
