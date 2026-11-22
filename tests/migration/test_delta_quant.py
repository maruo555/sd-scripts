import math

import pytest
import torch

from library.dq_mul_policy import MulPolicy, restore_assignments, scope_includes
from library.rounding_util import compute_scale_bits, fake_quantize_levels, quantize_delta


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,axes", [
    ((2, 3), (0,)), ((2, 3, 4), (0, 1)), ((2, 3, 4, 2), (0, 2, 3)), ((7,), None), ((1, 2, 1, 3, 2), None),
])
@pytest.mark.parametrize("bits", [2, 8])
def test_channel_rms_grid_and_rng(device, shape, axes, bits):
    delta = torch.linspace(-1.8, 2.5, steps=math.prod(shape),
                           device=device, dtype=torch.float16).reshape(shape).requires_grad_()
    qmax = 2 ** (bits - 1) - 1
    scale = compute_scale_bits(delta, bits=bits, range_mul=0.75)
    squared = delta.detach().float() ** 2
    expected_rms = squared.mean() if axes is None else squared.mean(axes, keepdim=True)
    expected_scale = (expected_rms + 1e-8).sqrt() * 0.75 / qmax
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    assert scale.dtype == torch.float32 and not scale.requires_grad

    torch.manual_seed(42)
    uniforms = torch.rand_like(delta, dtype=torch.float32)
    next_expected = torch.rand(8, device=device)
    normalized = delta.detach().float() / expected_scale
    lower = torch.floor(normalized)
    levels = torch.where(uniforms < normalized - lower, lower + 1, lower)
    rounded = (levels.clamp(-qmax, qmax) * expected_scale).half()
    expected = delta.detach() + (rounded - delta.detach())
    torch.manual_seed(42)
    actual = quantize_delta(delta, bits=bits, range_mul=0.75)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(8, device=device), next_expected, rtol=0, atol=0)
    upstream_gradient = torch.linspace(-1, 1, steps=delta.numel(), device=device, dtype=delta.dtype).reshape(shape)
    actual.backward(upstream_gradient)
    torch.testing.assert_close(delta.grad, upstream_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_counts_use_unrounded_values_and_observation_consumes_no_rng(device):
    delta = torch.tensor([-2.25, -2, -1.75, -0.125, 0, 0.875, 1.75, 2, 2.25], device=device, requires_grad=True)
    uniforms = torch.full_like(delta, 0.5)
    observations = []

    def record(x, out, raw, scale, limit):
        assert not torch.is_grad_enabled() and not x.requires_grad and not out.requires_grad
        observations.append((int((raw.abs() >= limit).sum()), out.clone()))

    args = {"scale": torch.ones((), device=device), "qmax": 2, "rand": uniforms}
    torch.manual_seed(32)
    next_expected = torch.rand(8, device=device)
    torch.manual_seed(32)
    actual = fake_quantize_levels(delta, observe=record, **args)
    assert len(observations) == 1 and observations[0][0] == 4
    assert actual.tolist() == [-2, -2, -2, 0, 0, 1, 2, 2, 2]
    torch.testing.assert_close(actual, observations[0][1], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(8, device=device), next_expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="random tensor"):
        fake_quantize_levels(delta, scale=args["scale"], qmax=2, rand=uniforms[:1])
    with pytest.raises(ValueError, match="at least 2"):
        quantize_delta(delta, bits=1, range_mul=3)


NAMES = [
    "lora_unet_input_blocks_1_0_in_layers_2", "lora_unet_middle_block_0_proj_in",
    "lora_unet_middle_block_1_attn1_to_q", "lora_unet_output_blocks_1_attn2_to_out_0",
    "lora_unet_output_blocks_1_ff_net_0_proj", "lora_te1_layer_11_fc2", "lora_te2_layer_31_fc2",
]


def test_assignments_and_versioned_reconstruction_are_independent_of_scope():
    policy = MulPolicy.from_values({
        "dq_delta_range_mul": 2.7, "dq_delta_range_mul_attn2": 3.75, "dq_delta_range_mul_te": 3.75,
        "dq_delta_range_mul_te2": 3.45, "dq_delta_scope": "unet", "dq_delta_auto_scope": "te",
    })
    assignments = policy.expand(reversed(NAMES))
    assert [assignments[n]["mul"] for n in NAMES] == [2.7, 2.7, 2.7, 3.75, 2.7, 3.75, 3.45]
    assert assignments[NAMES[0]]["group"] == "unet.other"
    assert assignments[NAMES[1]]["group"] == "unet.other_projection"
    assert all(a["enabled"] for a in assignments.values())
    record = policy.record(assignments)
    assert record["module_names"] == sorted(NAMES)
    assert restore_assignments(record) == assignments
    for scope, expected in [("both", {"unet", "te1", "te2"}), ("te", {"te1", "te2"}), ("unet", {"unet"})]:
        assert {a["component"] for a in assignments.values() if scope_includes(scope, a["component"])} == expected
    for key, replacement in [("resolver_version", 2), ("module_names", NAMES[:-1]), ("assignment_digest", "wrong")]:
        with pytest.raises(ValueError):
            restore_assignments({**record, key: replacement})
    with pytest.raises(ValueError, match="Duplicate"):
        policy.expand(NAMES + [NAMES[0]])
    with pytest.raises(ValueError, match="te2"):
        policy.expand(NAMES[:-1])
    with pytest.raises(ValueError, match="Unknown SDXL"):
        policy.expand(NAMES + ["unknown_layer"])
    with pytest.raises(ValueError, match="attn2"):
        MulPolicy(3.0, attn2=3.5).expand([NAMES[0]])
    for bad in (0, -1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive"):
            MulPolicy(3.0, te=bad)
