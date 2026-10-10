"""CPU-only coverage of the public CLI and existing LoRA execution path."""
import copy
import json

import pytest
import torch

from library.dq_mul_policy import MulPolicy, fixed_policy_resume_record, validate_fixed_policy_resume
from train_network import setup_parser

from test_dq_mul_policy_training import TinyNetwork, evaluate

torch.set_num_threads(1)

BASE = ["--dq_delta_bits", "8", "--dq_delta_stat", "rms", "--dq_delta_range_mul", "2.70"]
SPATIAL = ["--dq_delta_range_mul_attn2", "3.75", "--dq_delta_range_mul_te", "3.75"]
NAMES = [
    "lora_te1_text_model_encoder_layers_0_mlp_fc1",
    "lora_te2_text_model_encoder_layers_0_mlp_fc1",
    "lora_unet_output_blocks_1_ff_net_0_proj",
    "lora_unet_middle_block_1_attn1_to_q",
    "lora_unet_input_blocks_1_proj_in",
] + [f"lora_unet_{region}_1_attn2_{role}" for region in ("input_blocks", "middle_block", "output_blocks")
     for role in ("to_q", "to_k", "to_v", "to_out_0")]


@pytest.fixture(scope="module")
def parser():
    return setup_parser()


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def policy(parser, extra=()):
    return MulPolicy.from_training_args(parser.parse_args(BASE + list(extra)))


def test_without_overrides_legacy_path_is_unchanged(parser):
    for extra in ([], ["--dq_delta_scope", "unet"], ["--dq_delta_auto_range_mul"], ["--dq_quantize_z"]):
        assert policy(parser, extra) is None
    assert MulPolicy.from_training_args(parser.parse_args([])) is None


@pytest.mark.parametrize("scope", ["unet", "both"])
def test_fixed_reference_covers_all_attn2_regions_and_roles(parser, scope):
    rows = policy(parser, SPATIAL + ["--dq_delta_scope", scope]).expand(NAMES)
    for name, row in rows.items():
        assert row["enabled"] is True
        expected = 3.75 if name.startswith("lora_te") or "_attn2_" in name else 2.7
        assert row["mul"] == expected


@pytest.mark.parametrize("extra, expected", [
    (["--dq_delta_range_mul_attn2", "3.45"], (2.7, 2.7, 3.45)),
    (["--dq_delta_range_mul_te", "4.05"], (4.05, 4.05, 2.7)),
    (["--dq_delta_range_mul_te1", "3.15"], (3.15, 2.7, 2.7)),
    (["--dq_delta_range_mul_te2", "3.45"], (2.7, 3.45, 2.7)),
    (["--dq_delta_range_mul_te", "3.75", "--dq_delta_range_mul_te2", "3.45"], (3.75, 3.45, 2.7)),
    (["--dq_delta_range_mul_te2", "3.45", "--dq_delta_range_mul_te", "3.75"], (3.75, 3.45, 2.7)),
])
def test_partial_overrides_inherit_and_individual_te_wins(parser, extra, expected):
    rows = policy(parser, extra).expand(NAMES)
    assert (rows[NAMES[0]]["mul"], rows[NAMES[1]]["mul"], rows[NAMES[-1]]["mul"]) == expected
    assert rows[NAMES[2]]["mul"] == 2.7


@pytest.mark.parametrize("flag", ["--dq_delta_range_mul", "--dq_delta_range_mul_attn2",
                                  "--dq_delta_range_mul_te", "--dq_delta_range_mul_te1", "--dq_delta_range_mul_te2"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_mul_is_rejected_before_training(parser, flag, value):
    with pytest.raises(ValueError, match="finite positive"):
        policy(parser, SPATIAL + [f"{flag}={value}"])


def test_invalid_common_te_is_not_hidden_by_individual_overrides(parser):
    with pytest.raises(ValueError, match="finite positive"):
        policy(parser, ["--dq_delta_range_mul_te=nan", "--dq_delta_range_mul_te1=3.15", "--dq_delta_range_mul_te2=3.45"])


@pytest.mark.parametrize("extra", [["--dq_delta_bits", "0"], ["--dq_delta_bits_sched", "0:8,0.9:10"],
                                    ["--dq_delta_stat", "absmax"], ["--dq_quantize_z"],
                                    ["--dq_delta_auto_range_mul"]])
def test_incompatible_quantization_modes_are_rejected(parser, extra):
    with pytest.raises(ValueError):
        policy(parser, SPATIAL + extra)


def test_json_and_direct_are_unambiguous_and_old_json_still_works(parser, tmp_path):
    value = {"base_mul": 2.7, "components": {"te1": 3.75, "te2": 3.75},
             "group_overrides": [{"group_id": "unet.attn2", "range_mul": 3.75}]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert policy(parser, ["--dq_delta_policy_file", str(path)]) == policy(parser, SPATIAL)
    with pytest.raises(ValueError, match="cannot be combined"):
        policy(parser, SPATIAL + ["--dq_delta_policy_file", str(tmp_path / "missing.json")])


@pytest.mark.parametrize("flag, names", [("--dq_delta_range_mul_attn2", NAMES[:5]),
                                       ("--dq_delta_range_mul_te", NAMES[2:]),
                                       ("--dq_delta_range_mul_te2", NAMES[:1] + NAMES[2:])])
def test_missing_targets_fail_before_any_training_step(parser, flag, names):
    with pytest.raises(ValueError, match="Unmatched"):
        policy(parser, [flag, "3.75"]).expand(names)


def test_cli_matches_existing_policy_loss_gradients_and_rng(parser):
    direct, original = TinyNetwork(), TinyNetwork()
    direct.set_delta_mul_policy(policy(parser, SPATIAL))
    original.set_delta_mul_policy(MulPolicy.from_dict({"base_mul": 2.7,
        "components": {"te1": 3.75, "te2": 3.75},
        "group_overrides": [{"group_id": "unet.attn2", "range_mul": 3.75}]}))
    left, right = evaluate(direct), evaluate(original)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[2], right[2])
    assert all(torch.equal(left[1][name], right[1][name]) for name in left[1])


def test_explicit_uniform_cli_matches_ordinary_training(parser):
    ordinary, direct = TinyNetwork(True), TinyNetwork()
    direct.set_delta_mul_policy(policy(parser, ["--dq_delta_range_mul_attn2", "2.7", "--dq_delta_range_mul_te", "2.7"]))
    left, right = evaluate(ordinary), evaluate(direct)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[2], right[2])
    assert all(torch.equal(left[1][name], right[1][name]) for name in left[1])


def test_cli_assignment_survives_warmup_setters_and_weight_restore(parser):
    network = TinyNetwork()
    weights = copy.deepcopy(network.state_dict())
    record = network.set_delta_mul_policy(policy(parser, SPATIAL))
    for enabled in (False, True, False, True):
        network.set_delta_quant_enabled(enabled)
        network.set_delta_fake_quant(None, "stoch", bits=8, range_mul=100)
        network.load_state_dict(weights)
        for module in network.text_encoder_loras + network.unet_loras:
            assert module.delta_q_enabled is enabled
            assert module.delta_q_range_mul == record["modules"][module.lora_name]["mul"]


def test_cli_resume_preserves_assignment_and_rejects_changes(parser):
    current = TinyNetwork().set_delta_mul_policy(policy(parser, SPATIAL))
    saved = json.loads(json.dumps({"dq_mul_policy": fixed_policy_resume_record(current)}))
    equal = TinyNetwork().set_delta_mul_policy(policy(parser, SPATIAL))
    validate_fixed_policy_resume(saved, equal)
    changed = TinyNetwork().set_delta_mul_policy(policy(parser, SPATIAL + ["--dq_delta_range_mul_te2", "3.45"]))
    with pytest.raises(ValueError, match="differ"):
        validate_fixed_policy_resume(saved, changed)
    with pytest.raises(ValueError, match="same fixed mul"):
        validate_fixed_policy_resume(saved, None)
    with pytest.raises(ValueError, match="no fixed mul policy"):
        validate_fixed_policy_resume({}, current)


def test_existing_diagnostic_does_not_silently_absorb_training_overrides():
    from dq_profile.production_cli import ProfileCompatibilityError, resolve_training_cli
    with pytest.raises(ProfileCompatibilityError) as error:
        resolve_training_cli(["--pretrained_model_name_or_path=model.safetensors", "--dataset_config=dataset.toml",
                              "--dq_delta_range_mul_attn2=3.75"])
    assert "dq_delta_range_mul_attn2" in str(error.value)


def test_resume_requires_same_declaration_even_when_expanded_values_match(parser, tmp_path):
    original = policy(parser, SPATIAL).to_dict()
    original["components"]["unet"] = original["base_mul"]
    path = tmp_path / "original_policy.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    from_json = TinyNetwork().set_delta_mul_policy(policy(parser, ["--dq_delta_policy_file", str(path)]))
    from_cli = TinyNetwork().set_delta_mul_policy(policy(parser, SPATIAL))
    assert from_json["assignments_sha256"] == from_cli["assignments_sha256"]
    saved = {"dq_mul_policy": fixed_policy_resume_record(from_json)}
    with pytest.raises(ValueError, match="original declaration"):
        validate_fixed_policy_resume(saved, from_cli)
    moved = tmp_path / "moved_policy.json"
    moved.write_bytes(path.read_bytes())
    path.unlink()
    relocated = TinyNetwork().set_delta_mul_policy(policy(parser, ["--dq_delta_policy_file", str(moved)]))
    validate_fixed_policy_resume(saved, relocated)
