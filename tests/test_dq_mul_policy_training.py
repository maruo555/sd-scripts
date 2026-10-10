"""Synthetic CPU regression tests; no dataset or pretrained model is loaded."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from library.dq_mul_policy import MulPolicy, module_groups
from networks.lora import LoRAModule, LoRANetwork
from dq_profile.copied_lora import LoRANetwork as OriginalNetwork


NAMES = [
    "lora_te1_text_model_encoder_layers_0_mlp_fc1",
    "lora_te2_text_model_encoder_layers_0_mlp_fc1",
    "lora_unet_input_blocks_1_attn2_to_q",
    "lora_unet_middle_block_1_attn2_to_q",
    "lora_unet_output_blocks_1_attn2_to_q",
    "lora_unet_output_blocks_1_attn2_to_k",
    "lora_unet_output_blocks_1_attn2_to_v",
    "lora_unet_output_blocks_1_attn2_to_out_0",
    "lora_unet_output_blocks_1_ff_net_0_proj",
]


class TinyNetwork(torch.nn.Module):
    set_delta_fake_quant = LoRANetwork.set_delta_fake_quant
    set_delta_quant_enabled = LoRANetwork.set_delta_quant_enabled
    set_delta_mul_policy = LoRANetwork.set_delta_mul_policy
    _apply_delta_mul_policy = LoRANetwork._apply_delta_mul_policy

    def __init__(self, legacy=False):
        super().__init__()
        torch.manual_seed(73)
        self.text_encoder_loras, self.unet_loras, self.originals = [], [], []
        for name in NAMES:
            original = torch.nn.Linear(8, 8, bias=False).requires_grad_(False)
            module = LoRAModule(name, original, lora_dim=2, dropout=.3, rank_dropout=.2)
            module.apply_to()
            with torch.no_grad():
                module.lora_up.weight.normal_(0, .2)
            self.add_module(name, module)
            self.originals.append(original)
            (self.text_encoder_loras if name.startswith("lora_te") else self.unet_loras).append(module)
        if legacy:
            self.set_delta_fake_quant = OriginalNetwork.set_delta_fake_quant.__get__(self)
            self.set_delta_quant_enabled = OriginalNetwork.set_delta_quant_enabled.__get__(self)
        self.set_delta_fake_quant(None, "stoch", granularity="channel", stat="rms", bits=8, range_mul=2.7)

    def forward(self, x):
        return sum(original(x).square().mean() for original in self.originals)


def evaluate(network):
    network.zero_grad(set_to_none=True)
    torch.manual_seed(83)
    loss = network(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 30)
    loss.backward()
    return loss.detach(), {name: p.grad.clone() for name, p in network.named_parameters()}, torch.get_rng_state()


def test_no_policy_preserves_legacy_rng_gradients_and_te_enable():
    old, new = TinyNetwork(True), TinyNetwork()
    for network in (old, new):
        for module in network.text_encoder_loras:
            module.delta_q_enabled = False
        network.set_delta_quant_enabled(True)
        assert all(module.delta_q_enabled for module in network.text_encoder_loras)
    left, right = evaluate(old), evaluate(new)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[2], right[2])
    assert all(torch.equal(left[1][name], right[1][name]) for name in left[1])


def test_explicit_uniform_policy_matches_legacy_exactly():
    old, new = TinyNetwork(True), TinyNetwork()
    new.set_delta_mul_policy(MulPolicy.from_dict({"base_mul": 2.7}))
    left, right = evaluate(old), evaluate(new)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[2], right[2])
    assert all(torch.equal(left[1][name], right[1][name]) for name in left[1])


def test_hierarchy_and_order_independence():
    spec = {"base_mul": 2.7, "components": {"te1": 3.75, "te2": 3.45, "unet": 3.15},
            "group_overrides": [{"group_id": "unet.attn2", "range_mul": 3.45},
                                {"group_id": "unet.attn2.q", "range_mul": 3.75},
                                {"group_id": "unet.attn2.q.output", "range_mul": 4.05}],
            "module_overrides": {NAMES[3]: 2.7}}
    result = MulPolicy.from_dict(spec).expand(NAMES)
    assert [result[name]["mul"] for name in NAMES] == [3.75, 3.45, 3.75, 2.7, 4.05, 3.45, 3.45, 3.45, 3.15]
    spec["group_overrides"].reverse()
    assert MulPolicy.from_dict(spec).expand(list(reversed(NAMES))) == result
    assert module_groups(NAMES[7])[-1] == "unet.attn2.out.output"


def test_policy_survives_warmup_reconfiguration_and_weight_restore():
    network = TinyNetwork()
    before = copy.deepcopy(network.state_dict())
    policy = MulPolicy.from_dict({"base_mul": 2.7, "components": {"te1": 3.75, "te2": 3.75},
                                 "module_overrides": {NAMES[2]: 4.05}})
    record = network.set_delta_mul_policy(policy)
    for enabled in (False, True, False, True):
        network.set_delta_quant_enabled(enabled)
        network.set_delta_fake_quant(None, "stoch", bits=8, range_mul=100)
        network.load_state_dict(before)
        for module in network.unet_loras + network.text_encoder_loras:
            assert module.delta_q_enabled is enabled
            assert module.delta_q_range_mul == record["modules"][module.lora_name]["mul"]
    assert set(before) == set(network.state_dict())
    assert all(parameter.requires_grad for parameter in network.parameters())


def test_te_off_is_explicit_quantization_only_and_stays_off():
    network = TinyNetwork()
    network.set_delta_mul_policy(MulPolicy.from_dict({"base_mul": 3.15, "te_quantized": False}))
    network.set_delta_quant_enabled(True)
    assert all(not m.delta_q_enabled for m in network.text_encoder_loras)
    assert all(m.delta_q_enabled for m in network.unet_loras)
    evaluate(network)
    assert all(p.grad is not None for m in network.text_encoder_loras for p in m.parameters())


@pytest.mark.parametrize("spec", [
    {"base_mul": 3, "module_overrides": {"missing": 4}},
    {"base_mul": 3, "group_overrides": [{"group_id": "unet.attn1.q", "range_mul": 4}]},
    {"base_mul": 3, "group_overrides": [{"group_id": "unet.attn2.q.typo", "range_mul": 4}]},
])
def test_unmatched_override_rejected(spec):
    with pytest.raises(ValueError):
        MulPolicy.from_dict(spec).expand(NAMES)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"base_mul":2.7,"base_mul":3.75}')
    with pytest.raises(ValueError, match="Duplicate"):
        MulPolicy.from_file(path)


def test_round_trip_and_training_option_validation(tmp_path):
    path = tmp_path / "policy.json"
    value = {"base_mul": 2.7, "module_overrides": {NAMES[2]: 3.75}}
    path.write_text(json.dumps(value))
    args = SimpleNamespace(dq_delta_policy_file=str(path), dq_delta_bits=8, dq_delta_stat="rms")
    policy = MulPolicy.from_training_args(args)
    assert MulPolicy.from_dict(policy.to_dict()) == policy
    for field, wrong in (("dq_delta_bits", 0), ("dq_delta_bits_sched", "0:8"), ("dq_delta_stat", "absmax"),
                         ("dq_quantize_z", True), ("dq_delta_auto_range_mul", True)):
        changed = copy.copy(args)
        setattr(changed, field, wrong)
        with pytest.raises(ValueError):
            MulPolicy.from_training_args(changed)
    assert MulPolicy.from_training_args(SimpleNamespace()) is None
