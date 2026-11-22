from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from library.sdxl_original_unet import SdxlUNet2DConditionModel
from networks import lora
from train_network import NetworkTrainer


class Transformer2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)


class ResnetBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Conv2d(3, 3, 1)
        self.conv = nn.Conv2d(3, 3, 3)


class TinySDXL(SdxlUNet2DConditionModel):
    def __init__(self):
        nn.Module.__init__(self)
        blocks = [nn.Identity() for _ in range(4)] + [nn.Sequential(nn.Identity(), Transformer2DModel())]
        self.input_blocks = nn.ModuleList(blocks)
        self.resnet = ResnetBlock2D()


class CLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)


def text_encoders():
    towers = [nn.Module(), nn.Module()]
    for tower in towers:
        tower.mlp = CLIPMLP()
    return towers


def saved_projection(prefix, alpha=4):
    state = {prefix + ".lora_down.weight": torch.ones(2, 3),
             prefix + ".lora_up.weight": torch.full((3, 2), 0.25)}
    if alpha is not None:
        state[prefix + ".alpha"] = torch.tensor(float(alpha))
    return state


SOURCE = "lora_unet_down_blocks_1_attentions_0_proj"
TARGET = "lora_unet_input_blocks_4_1_proj"


@pytest.mark.parametrize("mode", ["training", "inference", "merge"])
def test_weight_factory_converts_once_and_preserves_application(mode):
    unet = TinySDXL()
    encoders = text_encoders()
    unet.input_blocks[4][1].proj.weight.data.zero_()
    saved = saved_projection(SOURCE)
    network, normalized = lora.create_network_from_weights(
        0.5, None, None, encoders, unet, weights_sd=saved, for_inference=mode != "training",
    )
    assert normalized is saved
    assert set(normalized) == set(saved_projection(TARGET))
    assert type(network) is lora.LoRANetwork
    assert not network.text_encoder_loras
    adapter, = network.unet_loras
    assert type(adapter) is (lora.LoRAModule if mode == "training" else lora.LoRAInfModule)
    assert adapter.lora_name == TARGET and adapter.lora_dim == 2 and adapter.scale == 2
    if mode == "merge":
        network.merge_to(encoders, unet, normalized, torch.float32, "cpu")
    else:
        network.apply_to(encoders, unet, True, True)
        network.load_state_dict(normalized, strict=True)
    # The original projection is zero. 0.5 * (4 / 2) * sum(1, 2, 3) * (2 * 0.25) = 3.
    torch.testing.assert_close(unet.input_blocks[4][1].proj(torch.tensor([1., 2., 3.])), torch.full((3,), 3.))


@pytest.mark.parametrize("suffix", [".safetensors", ".pt"])
def test_training_entry_reads_weight_file_and_uses_rank_if_alpha_absent(tmp_path, suffix):
    encoders, unet = text_encoders(), TinySDXL()
    path = tmp_path / ("rank2" + suffix)
    saved = saved_projection(TARGET, alpha=None)
    (save_file if suffix == ".safetensors" else torch.save)(saved, str(path))
    args = Namespace(dim_from_weights=True, network_weights=str(path))
    network = NetworkTrainer().create_network(args, lora, None, encoders, unet, {})
    assert len(network.unet_loras) == 1 and not network.text_encoder_loras
    adapter, = network.unet_loras
    assert adapter.alpha.item() == adapter.lora_dim == 2 and adapter.scale == 1


@pytest.mark.parametrize("previous", ["weights", "conv"])
def test_default_factory_isolated_across_network_instances(previous, monkeypatch):
    # An upstream class-level += caused both extra adapters and a shifted RNG stream.
    monkeypatch.setattr(lora.LoRANetwork, "UNET_TARGET_REPLACE_MODULE", ["Transformer2DModel"])
    encoders, unet = text_encoders(), TinySDXL()
    args = Namespace(dim_from_weights=False, network_dim=2, network_alpha=1.0, network_dropout=0.0)
    trainer = NetworkTrainer()

    def regular():
        torch.manual_seed(39)
        network = trainer.create_network(args, lora, None, encoders, unet, {})
        adapters = network.text_encoder_loras + network.unet_loras
        return [(a.lora_name, a.lora_down.weight.detach().clone()) for a in adapters], torch.rand(3)

    first, next_draw = regular()
    assert [name for name, _ in first] == ["lora_te1_mlp_fc1", "lora_te2_mlp_fc1", TARGET]
    if previous == "weights":
        lora.create_network_from_weights(1.0, None, None, encoders, unet, weights_sd=saved_projection(TARGET))
    else:
        conv = trainer.create_network(args, lora, None, encoders, unet, {"conv_dim": "2"})
        assert "lora_unet_resnet_conv" in [a.lora_name for a in conv.unet_loras]
    assert lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE == ["Transformer2DModel"]
    second, next_again = regular()
    assert [name for name, _ in first] == [name for name, _ in second]
    assert all(torch.equal(a[1], b[1]) for a, b in zip(first, second))
    assert torch.equal(next_draw, next_again)


def test_shared_weight_factory_uses_selected_network_and_adapter_classes():
    class InferenceNetwork(lora.LoRANetwork):
        pass

    class InferenceAdapter(lora.LoRAInfModule):
        pass

    network, _ = lora._create_network_from_weights(
        1.0, None, text_encoders(), TinySDXL(), saved_projection(TARGET),
        network_class=InferenceNetwork, module_class=InferenceAdapter, block_lr_kwargs={"mid_lr_weight": "0.7"},
    )
    assert type(network) is InferenceNetwork
    assert type(network.unet_loras[0]) is InferenceAdapter
    assert network.block_lr and len(network.block_lr_weight) == 23
    assert network.get_lr_weight(10) == 0.7
