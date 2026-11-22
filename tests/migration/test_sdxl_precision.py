"""Regression tests for the optional SDXL strict fp16 policy (including real CUDA autocast)."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from library import sdxl_original_unet as unet
from library import sdxl_training_options as precision
from sdxl_train_network import SdxlNetworkTrainer


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]


def identical(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind,affine", [("layer", True), ("layer", False), ("spatial", True), ("spatial", False)])
def test_functional_norms_match_fp32_reference_and_gradients(device, kind, affine):
    torch.manual_seed(30)
    if kind == "layer":
        norm = nn.LayerNorm(640, eps=1e-5, elementwise_affine=affine)
        source = torch.randn(2, 5, 640, device=device, dtype=torch.float16)
        func = precision.layer_norm

        def reference(x, norm, w, b):
            return F.layer_norm(x, norm.normalized_shape, w, b, norm.eps)
    else:
        norm = nn.GroupNorm(32, 640, eps=1e-6, affine=affine)
        source = torch.randn(2, 640, 2, 2, device=device, dtype=torch.float16)
        func = precision.spatial_group_norm

        def reference(x, norm, w, b):
            return F.group_norm(x, norm.num_groups, w, b, norm.eps)
    norm = norm.to(device=device, dtype=torch.float16)
    control = copy.deepcopy(norm)
    actual_input = source.detach().requires_grad_(True)
    ref_input = source.detach().requires_grad_(True)
    output = func(actual_input, norm, True)
    ref = reference(ref_input.float(), control, control.weight.float() if affine else None,
                    control.bias.float() if affine else None).to(ref_input.dtype)
    assert output.dtype == source.dtype
    identical(ref, output)
    grad = torch.rand_like(output)
    output.backward(grad)
    ref.backward(grad)
    identical(ref_input.grad, actual_input.grad)
    for a, b in zip(norm.parameters(), control.parameters()):
        identical(a.grad, b.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_fp16_uses_native_norm(device, dtype):
    norm = nn.LayerNorm(16).to(device=device, dtype=dtype)
    x = torch.linspace(-2, 2, 64, device=device, dtype=dtype).view(4, 16)
    with patch.object(norm, "forward", wraps=norm.forward) as native:
        identical(precision.layer_norm(x, norm, strict=True), norm(x))
    assert native.call_count == 2
    spatial = nn.GroupNorm(2, 4).to(device=device, dtype=dtype)
    y = x.reshape(1, 4, 4, 4)
    with patch.object(spatial, "forward", wraps=spatial.forward) as native:
        identical(precision.spatial_group_norm(y, spatial, strict=True), spatial(y))
    assert native.call_count == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("autocast", [True, False])
def test_resnet_group_norm_keeps_its_affine_and_autocast(device, weight_dtype, autocast):
    active = unet.GroupNorm32(32, 640).to(device=device, dtype=weight_dtype)
    active.strict_fp16_norms = True
    control = nn.GroupNorm(32, 640).to(device=device, dtype=weight_dtype)
    control.load_state_dict(active.state_dict())
    source = torch.linspace(-3, 5, 2560, device=device, dtype=torch.float16).reshape(1, 640, 2, 2)
    x, y = source.detach().requires_grad_(True), source.detach().requires_grad_(True)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=autocast):
        try:
            ref = control(y.float()).to(y.dtype)
        except RuntimeError as exc:
            with pytest.raises(type(exc)):
                active(x)
            return
        output = active(x)
    assert output.dtype == x.dtype
    identical(output, ref)
    grad = torch.ones_like(output)
    output.backward(grad)
    ref.backward(grad)
    identical(x.grad, y.grad)
    identical(active.weight.grad, control.weight.grad)
    identical(active.bias.grad, control.bias.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_finite_and_nonfinite_scores(device):
    scores = torch.tensor([[60000., 59968., -60000.], [0., 0., 0.], [float("inf"), -1., 0.],
                           [float("nan"), 1., 2.], [float("-inf"), -1., 0.]], device=device, dtype=torch.float16)
    ref = scores.float().softmax(dim=-1).to(scores.dtype)
    identical(precision.attention_softmax(scores, True), ref)
    identical(precision.attention_softmax(scores, False), scores.softmax(dim=-1))
    assert torch.isnan(ref[2:4]).all() and torch.isfinite(ref[[0, 1, 4]]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sdpa", [False, True])
def test_transformer_dispatch_and_gradients(device, sdpa):
    torch.manual_seed(71)
    block = unet.Transformer2DModel(num_attention_heads=2, attention_head_dim=32, in_channels=64,
                                    cross_attention_dim=64, use_linear_projection=True).to(device=device, dtype=torch.float16)
    block.set_use_sdpa(sdpa)
    unet.SdxlUNet2DConditionModel.set_strict_fp16_norms(block, True)
    x = torch.randn(1, 64, 2, 2, device=device, dtype=torch.float16, requires_grad=True)
    context = torch.randn(1, 5, 64, device=device, dtype=torch.float16, requires_grad=True)
    with patch.object(precision, "spatial_group_norm", wraps=precision.spatial_group_norm) as spatial, \
         patch.object(precision, "layer_norm", wraps=precision.layer_norm) as layer, \
         patch.object(precision, "attention_softmax", wraps=precision.attention_softmax) as softmax:
        block(x, context).sum().backward()
    assert spatial.call_count == 1 and layer.call_count == 3
    assert all(call.args[0].dtype == x.dtype and call.args[-1] for call in [*spatial.call_args_list, *layer.call_args_list])
    assert softmax.call_count == (0 if sdpa else 2)
    assert all(call.args[0].dtype == x.dtype and call.args[-1] for call in softmax.call_args_list)
    assert torch.isfinite(x.grad).all() and torch.isfinite(context.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_large_constant_and_nonfinite_norm_inputs(device):
    values = torch.tensor([[0.] * 8, [65000., -65000.] * 4, [float("nan")] + [1.] * 7,
                           [float("inf")] + [2.] * 7, [-float("inf")] + [3.] * 7], device=device, dtype=torch.float16)
    ln = nn.LayerNorm(8, elementwise_affine=False)
    identical(precision.layer_norm(values, ln, True), F.layer_norm(values.float(), (8,), eps=ln.eps).half())
    gn = nn.GroupNorm(1, 2, affine=False)
    channels = values.reshape(-1, 2, 2, 2)
    identical(precision.spatial_group_norm(channels, gn, True), F.group_norm(channels.float(), 1, eps=gn.eps).half())


@pytest.mark.parametrize("device", DEVICES)
def test_upcast_attention_keeps_score_and_value_dtypes(device):
    attention = unet.CrossAttention(32, heads=2, dim_head=16, upcast_attention=True).to(device=device, dtype=torch.float16)
    attention.strict_fp16_norms = True
    source = torch.randn(1, 4, 32, device=device, dtype=torch.float16)
    with patch.object(precision, "attention_softmax", wraps=precision.attention_softmax) as softmax:
        result = attention(source)
    assert softmax.call_count == 1
    assert softmax.call_args.args[0].dtype == torch.float32
    assert result.dtype == torch.float16


def test_strict_is_per_model_and_preserves_state_keys():
    with torch.device("meta"):
        first, second = unet.SdxlUNet2DConditionModel(), unet.SdxlUNet2DConditionModel()
    keys = tuple(first.state_dict())
    first.set_strict_fp16_norms(True)
    tracked = (unet.GroupNorm32, unet.CrossAttention, unet.BasicTransformerBlock, unet.Transformer2DModel)
    for cls in tracked:
        assert all(m.strict_fp16_norms for m in first.modules() if isinstance(m, cls))
        assert all(not m.strict_fp16_norms for m in second.modules() if isinstance(m, cls))
    assert keys == tuple(first.state_dict()) == tuple(second.state_dict())
    first.set_strict_fp16_norms(False)
    assert all(not m.strict_fp16_norms for m in first.modules() if isinstance(m, tracked))


def test_architecture_metadata():
    trainer = SdxlNetworkTrainer()
    meta = {}
    trainer.update_metadata(meta, SimpleNamespace())
    assert meta == {}
    trainer.maruo_config = SimpleNamespace(values={"fp16_safe_norms_mode": "strict"})
    trainer.update_metadata(meta, SimpleNamespace())
    assert meta == {"ss_fp16_safe_norms_mode": "strict"}
