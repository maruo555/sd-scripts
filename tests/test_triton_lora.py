from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from library.triton_lora import (
    get_triton_rank4_quantized_lora_up_diagnostics,
    triton_rank4_delta_quant,
)


def _cuda_c0_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    info = get_triton_rank4_quantized_lora_up_diagnostics("cuda")
    return bool(info["triton_available"] and info.get("device_supported"))


requires_c0_cuda = pytest.mark.skipif(
    not _cuda_c0_supported(),
    reason="rank-4 Triton LoRA-Up tests require a validated CUDA capability",
)


def _reference(
    z: torch.Tensor,
    weight: torch.Tensor,
    rand: torch.Tensor,
    *,
    multiplier: float,
    lora_scale: float,
    range_mul: float,
) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16):
        up = F.linear(z, weight)
        tmp = up * multiplier
        delta = tmp * lora_scale

    qmax = 127
    with torch.no_grad():
        scale = (
            torch.sqrt(torch.mean(delta.to(torch.float32) ** 2, dim=(0, 1), keepdim=True) + 1.0e-8)
            * range_mul
            / qmax
        )
        y = delta.to(torch.float32) / scale
        q_floor = torch.floor(y)
        probability = (y - q_floor).clamp(0.0, 1.0)
        q = q_floor + (rand < probability).to(torch.float32)
        quantized = (q.clamp(-qmax, qmax) * scale).to(torch.float16)
    return delta + (quantized - delta).detach()


def test_diagnostics_are_import_safe():
    info = get_triton_rank4_quantized_lora_up_diagnostics()
    assert isinstance(info["triton_available"], bool)
    assert info["max_rows"] == 2048
    assert (12, 0) in info["supported_capabilities"]
    assert {320, 640, 768, 1280, 2560, 3072, 5120, 10240}.issubset(info["supported_channel_counts"])


def test_cpu_input_falls_back_without_consuming_rand():
    z = torch.randn(1, 2, 4, dtype=torch.float16)
    weight = torch.randn(640, 4, dtype=torch.float32)
    rand = torch.rand(1, 2, 640, dtype=torch.float32)
    rand_before = rand.clone()
    result = triton_rank4_delta_quant(
        z,
        weight,
        multiplier=0.7,
        lora_scale=0.5,
        range_mul=3.0,
        rand=rand,
    )
    assert result is None
    torch.testing.assert_close(rand, rand_before, rtol=0, atol=0)


@requires_c0_cuda
@pytest.mark.parametrize(
    "mutation",
    ["rank", "z_dtype", "weight_dtype", "rand_dtype", "rand_shape", "rows", "channels"],
)
def test_unsupported_cuda_inputs_fall_back(mutation):
    z = torch.randn(1, 8, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(640, 4, device="cuda", dtype=torch.float32)
    rand = torch.rand(1, 8, 640, device="cuda", dtype=torch.float32)
    if mutation == "rank":
        z = torch.randn(1, 8, 5, device="cuda", dtype=torch.float16)
        weight = torch.randn(640, 5, device="cuda", dtype=torch.float32)
    elif mutation == "z_dtype":
        z = z.to(torch.bfloat16)
    elif mutation == "weight_dtype":
        weight = weight.to(torch.float16)
    elif mutation == "rand_dtype":
        rand = rand.to(torch.float16)
    elif mutation == "rand_shape":
        rand = rand[..., :-1]
    elif mutation == "rows":
        z = torch.randn(1, 2049, 4, device="cuda", dtype=torch.float16)
        rand = torch.rand(1, 2049, 640, device="cuda", dtype=torch.float32)
    elif mutation == "channels":
        weight = torch.randn(64, 4, device="cuda", dtype=torch.float32)
        rand = torch.rand(1, 8, 64, device="cuda", dtype=torch.float32)
    rand_before = rand.clone()
    result = triton_rank4_delta_quant(
        z,
        weight,
        multiplier=0.7,
        lora_scale=0.5,
        range_mul=3.0,
        rand=rand,
    )
    assert result is None
    torch.testing.assert_close(rand, rand_before, rtol=0, atol=0)


@requires_c0_cuda
def test_rank4_c0_matches_native_forward_and_backward():
    torch.manual_seed(1234)
    multiplier = 0.73
    lora_scale = 0.5
    range_mul = 3.0
    z = torch.randn(1, 77, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(640, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    rand = torch.rand(1, 77, 640, device="cuda", dtype=torch.float32)
    rand_before = rand.clone()
    upstream = torch.randn(1, 77, 640, device="cuda", dtype=torch.float16)

    result = triton_rank4_delta_quant(
        z,
        weight,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
        rand=rand,
    )
    assert result is not None
    actual, stats = result
    assert stats is None
    (actual * upstream).sum().backward()
    actual_z_grad = z.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()

    z_ref = z.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected = _reference(
        z_ref,
        weight_ref,
        rand,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
    )
    (expected * upstream).sum().backward()
    torch.cuda.synchronize()

    forward_rel_l2 = torch.linalg.vector_norm((actual - expected).float()) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1.0e-12)
    assert forward_rel_l2.item() <= 1.0e-3
    torch.testing.assert_close(actual_z_grad, z_ref.grad, rtol=1.0e-3, atol=1.0e-3)
    torch.testing.assert_close(actual_weight_grad, weight_ref.grad, rtol=1.0e-3, atol=1.0e-3)
    torch.testing.assert_close(rand, rand_before, rtol=0, atol=0)


@requires_c0_cuda
def test_rank4_c0_basic_stats_match_reference_and_are_non_differentiable():
    torch.manual_seed(4321)
    multiplier = 0.73
    lora_scale = 0.5
    range_mul = 3.0
    z = torch.randn(1, 77, 4, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(640, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    rand = torch.rand(1, 77, 640, device="cuda", dtype=torch.float32)

    result = triton_rank4_delta_quant(
        z,
        weight,
        multiplier=multiplier,
        lora_scale=lora_scale,
        range_mul=range_mul,
        rand=rand,
        collect_basic_stats=True,
    )
    assert result is not None
    actual, packed_stats = result
    assert packed_stats is not None
    assert packed_stats.shape == (5,)
    assert packed_stats.dtype == torch.float32
    assert not packed_stats.requires_grad

    with torch.autocast("cuda", dtype=torch.float16):
        up = F.linear(z.detach(), weight.detach())
        delta = (up * multiplier) * lora_scale
    with torch.no_grad():
        scale = (
            torch.sqrt(torch.mean(delta.float() ** 2, dim=(0, 1), keepdim=True) + 1.0e-8)
            * range_mul
            / 127
        )
        y = delta.float() / scale
        q_floor = torch.floor(y)
        q = q_floor + (rand < (y - q_floor).clamp(0.0, 1.0)).float()
        quantized = (q.clamp(-127, 127) * scale).half()
        expected_stats = torch.stack(
            (
                torch.tensor(float(delta.numel()), device="cuda"),
                (y.abs() >= 127).float().sum(),
                delta.float().square().sum(),
                quantized.float().square().sum(),
                (delta.float() * quantized.float()).sum(),
            )
        )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, quantized, rtol=1.0e-3, atol=5.0e-4)
    torch.testing.assert_close(packed_stats[:2], expected_stats[:2], rtol=0, atol=0)
    torch.testing.assert_close(packed_stats[2:], expected_stats[2:], rtol=1.0e-3, atol=1.0e-2)
    actual.float().sum().backward()
    assert z.grad is not None
    assert weight.grad is not None


@requires_c0_cuda
@pytest.mark.parametrize(
    "rows,channels",
    [
        (77, 768),
        (128, 320),
        (129, 320),
        (512, 640),
        (513, 640),
        (1872, 2560),
        (2048, 320),
    ],
)
def test_rank4_c0_row_buckets(rows, channels):
    z = torch.randn(1, rows, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.float32)
    rand = torch.rand(1, rows, channels, device="cuda", dtype=torch.float32)
    result = triton_rank4_delta_quant(
        z,
        weight,
        multiplier=1.0,
        lora_scale=0.25,
        range_mul=3.0,
        rand=rand,
    )
    assert result is not None
    assert result[0].shape == (1, rows, channels)
    assert result[0].dtype == torch.float16
    assert torch.isfinite(result[0]).all()
