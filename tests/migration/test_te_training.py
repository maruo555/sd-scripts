"""SDXL LoRA selection, optimizer grouping and initial training weights."""
import copy

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from library import maruo_training_config
from networks import lora
from sdxl_train_network import SdxlNetworkTrainer, setup_parser


class CLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4, bias=False)
        self.fc2 = nn.Linear(4, 3, bias=False)


class CLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3, 3)


class Transformer2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)


def models():
    encoders = []
    for count in (1, 2):
        encoder = nn.Module()
        encoder.text_model = nn.Module()
        encoder.text_model.encoder = nn.Module()
        layers = nn.ModuleList()
        for _ in range(count):
            layer = nn.Module()
            layer.self_attn = CLIPAttention()
            layer.mlp = CLIPMLP()
            layers.append(layer)
        encoder.text_model.encoder.layers = layers
        encoders.append(encoder)
    unet = nn.Module()
    unet.block = Transformer2DModel()
    return encoders, unet


def trainer_and_args(*options):
    args = setup_parser().parse_args([
        "--network_module", "networks.lora", "--network_dim", "2", "--network_alpha", "1",
        "--learning_rate", "0.00035", *options,
    ])
    trainer = SdxlNetworkTrainer()
    trainer.maruo_config = maruo_training_config.resolve_training_config(args)
    maruo_training_config.require_wired_features(trainer.maruo_config)
    return trainer, args


def create(trainer, args, encoders, unet):
    network = trainer.create_network(args, lora, None, encoders, unet, trainer.parse_network_args(args))
    network.apply_to(encoders, unet, not args.network_train_unet_only, not args.network_train_text_encoder_only)
    return network


def test_fc_selection_order_and_independence():
    encoders, unet = models()
    rng = torch.get_rng_state()
    slim, args = trainer_and_args("--te_mlp_fc_only")
    selected = create(slim, args, *copy.deepcopy((encoders, unet)))
    assert [a.lora_name for a in selected.text_encoder_loras] == [
        f"lora_te{n}_text_model_encoder_layers_{i}_mlp_fc{f}"
        for n, count in [(1, 1), (2, 2)] for i in range(count) for f in (1, 2)
    ]
    plain, args = trainer_and_args()
    standard = create(plain, args, *copy.deepcopy((encoders, unet)))
    assert len(standard.text_encoder_loras) == 9
    assert any("attn" in a.lora_name for a in standard.text_encoder_loras)
    assert [a.lora_name for a in selected.unet_loras] == [a.lora_name for a in standard.unet_loras]
    # Adding an unrelated layer under attention does not consume adapter initialization draws.
    encoders[0].text_model.encoder.layers[0].self_attn.other = nn.Linear(3, 7)
    torch.set_rng_state(rng)
    retry, args = trainer_and_args("--te_mlp_fc_only")
    next_network = create(retry, args, encoders, unet)
    for k, v in selected.state_dict().items():
        assert torch.equal(v, next_network.state_dict()[k])


def test_expected_fc_missing_or_wrong_type_reports_te_and_path():
    encoders, unet = models()
    encoders[1].text_model.encoder.layers[1].mlp.fc2 = nn.Identity()
    trainer, args = trainer_and_args("--te_mlp_fc_only")
    with pytest.raises(ValueError, match=r"TE2.*text_model.encoder.layers.1.mlp.fc2"):
        create(trainer, args, encoders, unet)


@pytest.mark.parametrize(("options", "labels", "rates"), [
    (["--text_encoder_lr", "2e-4", "--text_encoder_lr1", "3e-4", "--text_encoder_lr2", "2e-4"],
     ["textencoder1", "textencoder2", "unet"], [3e-4, 2e-4, 3.5e-4]),
    (["--text_encoder_lr", "2e-4", "2e-4", "--te_mlp_fc_only"],
     ["textencoder1", "textencoder2", "unet"], [2e-4, 2e-4, 3.5e-4]),
    (["--te_mlp_fc_only", "--network_args", "loraplus_lr_ratio=2"],
     ["textencoder1", "textencoder1 plus", "textencoder2", "textencoder2 plus", "unet", "unet plus"], [3.5e-4, 7e-4]*3),
    (["--text_encoder_lr1", "0", "--text_encoder_lr2", "3e-4"], ["textencoder2","unet"], [3e-4,3.5e-4]),
    (["--network_train_unet_only", "--te_mlp_fc_only"], ["unet"], [3.5e-4]),
    (["--network_train_text_encoder_only", "--text_encoder_lr1", "0"], ["textencoder2"], [3.5e-4]),
    (["--text_encoder_lr","2e-4","3e-4"], ["textencoder","unet"],[2e-4,3.5e-4]),
])
def test_lr_groups_use_post_apply_membership(options, labels, rates):
    trainer, args = trainer_and_args(*options)
    encoders, unet = models()
    network = create(trainer, args, encoders, unet)
    groups, descriptions = trainer.prepare_network_optimizer_params(args, network, trainer.get_network_text_encoder_lr(args, network))
    assert descriptions == labels and [group["lr"] for group in groups] == rates
    names = {id(p): k for k, p in network.named_parameters()}
    seen = set()
    for label, group in zip(labels, groups):
        ids = {id(p) for p in group["params"]}
        assert ids and not (ids & seen)
        seen |= ids
        if label.startswith(("textencoder1", "textencoder2")):
            expected = "lora_te" + label[len("textencoder")] + "_"
            assert all(names[i].startswith(expected) for i in ids)
        if label.endswith(" plus"):
            assert all("lora_up" in names[i] for i in ids)
    if args.network_train_unet_only:
        assert not network.text_encoder_loras
    if "textencoder1" not in labels and not args.network_train_unet_only and getattr(args, "text_encoder_lr1", None) == 0:
        te1 = [a for a in network.text_encoder_loras if a.text_encoder_index == 1]
        assert te1 and all(p.requires_grad and id(p) not in seen for a in te1 for p in a.parameters())


def test_te_only_with_all_zero_rates_explains_empty_optimizer():
    trainer, args = trainer_and_args("--network_train_text_encoder_only", "--text_encoder_lr1", "0", "--text_encoder_lr2", "0")
    network = create(trainer, args, *models())
    with pytest.raises(ValueError, match="No trainable LoRA parameter group"):
        trainer.prepare_network_optimizer_params(args, network, trainer.get_network_text_encoder_lr(args, network))


def checkpoint():
    trainer, args = trainer_and_args("--te_mlp_fc_only")
    network = create(trainer, args, *models())
    state = network.state_dict()
    for name in state:
        if name.endswith("lora_up.weight"):
            state[name].fill_(0.5)
    return state


@pytest.mark.parametrize("from_file_rank", [False, True])
def test_initial_weights_single_read_alpha_and_training_dropout(tmp_path, monkeypatch, from_file_rank):
    path = tmp_path/"start.safetensors"
    state = checkpoint()
    base = "lora_te1_text_model_encoder_layers_0_mlp_fc1"
    missing = "lora_te1_text_model_encoder_layers_0_mlp_fc2"
    zero = "lora_unet_block_proj"
    state[base+".alpha"].fill_(0.5)
    del state[missing+".alpha"]
    state[zero+".alpha"].zero_()
    save_file(state, str(path))
    import safetensors.torch
    original = safetensors.torch.load_file
    reads = []
    def read(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(safetensors.torch, "load_file", read)
    switches = ["--dim_from_weights"] if from_file_rank else []
    trainer, args = trainer_and_args(
        "--te_mlp_fc_only", "--network_weights", str(path), "--network_dropout", "0.25",
        "--network_args", "rank_dropout=0.2", "module_dropout=0.1", *switches,
    )
    encoders, unet = models()
    network = create(trainer, args, encoders, unet)
    info = trainer.load_initial_network_weights(args, network)
    assert not (info.missing_keys or info.unexpected_keys)
    assert reads == [str(path)]
    adapters = {a.lora_name:a for a in network.text_encoder_loras + network.unet_loras}
    assert (adapters[base].alpha.item(), adapters[base].scale) == (0.5, 0.25)
    assert (adapters[missing].alpha.item(), adapters[missing].scale) == (2, 1)
    assert (adapters[zero].alpha.item(), adapters[zero].scale) == (0, 1)
    for a in adapters.values():
        assert (a.dropout, a.rank_dropout, a.module_dropout) == (0.25, 0.2, 0.1)
    a = adapters[base]
    data = torch.tensor([[1., 2., 3.], [3., 1., 2.]]).view(1,2,3)
    a.eval()
    torch.testing.assert_close(a(data), a.org_forward(data) + 0.25*a.lora_up(a.lora_down(data)))
    assert trainer._initial_lora_weights is None


@pytest.mark.parametrize("damage", ["shape", "extra", "missing_layer", "missing_up"])
def test_bad_weights_report_specific_keys(tmp_path, damage):
    path = tmp_path/"bad.safetensors"
    state = checkpoint()
    prefix = "lora_te1_text_model_encoder_layers_0_mlp_fc1"
    if damage == "shape":
        state[prefix+".lora_down.weight"] = torch.ones(2,4)
    elif damage == "extra":
        state["lora_te1_attention_extra.lora_down.weight"] = torch.ones(2,4)
    elif damage == "missing_layer":
        state = {k:v for k,v in state.items() if not k.startswith(prefix+".")}
    else:
        del state[prefix+".lora_up.weight"]
    save_file(state, str(path))
    trainer, args = trainer_and_args("--te_mlp_fc_only", "--network_weights", str(path), "--dim_from_weights")
    with pytest.raises((ValueError,RuntimeError), match="lora_te1|TE1"):
        network = create(trainer, args, *models())
        trainer.load_initial_network_weights(args, network)


@pytest.mark.parametrize("only", ["--network_train_unet_only", "--network_train_text_encoder_only"])
def test_explicit_partial_load_ignores_only_other_component(tmp_path, only):
    path = tmp_path/"full.safetensors"
    save_file(checkpoint(), str(path))
    trainer, args = trainer_and_args("--te_mlp_fc_only", "--network_weights", str(path), only, "--dim_from_weights")
    network = create(trainer, args, *models())
    result = trainer.load_initial_network_weights(args, network)
    assert not (result.missing_keys or result.unexpected_keys)
    assert bool(network.unet_loras) == (only == "--network_train_unet_only")


@pytest.mark.parametrize("seed", [0, 3])
def test_training_dropout_matches_legacy_forward_grads_and_next_rng(seed):
    trainer, args = trainer_and_args(
        "--te_mlp_fc_only", "--network_dropout", "0.25", "--network_args", "rank_dropout=0.2", "module_dropout=0.1"
    )
    network = create(trainer, args, *models())
    a = network.text_encoder_loras[0]
    a.lora_up.weight.data.fill_(0.5)
    def forward_and_grads(oracle):
        x = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).requires_grad_()
        torch.manual_seed(seed)
        if oracle:
            output = a.org_forward(x)
            if torch.rand(1).item() >= a.module_dropout:
                reduced = torch.nn.functional.dropout(a.lora_down(x), p=a.dropout)
                mask = torch.rand((x.shape[0], a.lora_dim)).gt(a.rank_dropout).unsqueeze(1)
                delta = a.lora_up(reduced * mask)
                output = output + delta * a.multiplier * (a.scale * (1.0 / (1.0-a.rank_dropout)))
        else:
            output = a(x)
        grads = torch.autograd.grad(output.sum(), (x, a.lora_down.weight, a.lora_up.weight), allow_unused=True)
        return [output.detach(), *grads, torch.rand(4)]

    for actual, expected in zip(forward_and_grads(False), forward_and_grads(True)):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)


def test_common_lr_keeps_stock_checkpoint_metadata_meaning():
    from types import SimpleNamespace
    trainer, args = trainer_and_args("--text_encoder_lr", "2e-4", "--text_encoder_lr1", "3e-4")
    network = create(trainer, args, *models())
    assert trainer.get_network_text_encoder_lr(args, network) == 2e-4
    assert trainer.maruo_config.text_encoder_lrs == (3e-4, 2e-4)
    external_multiple = SimpleNamespace(prepare_optimizer_params_with_multiple_te_lrs=True)
    args.text_encoder_lr = [3e-4, 2e-4]
    assert trainer.get_network_text_encoder_lr(args, external_multiple) == [3e-4, 2e-4]
    args.text_encoder_lr = []
    assert trainer.get_network_text_encoder_lr(args, network) is None
