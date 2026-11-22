"""The SDXL diffusion timestep is the only embedding changed by --downscale_freq_shift."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from library import sdxl_original_unet as model
from library.sdxl_train_util import get_size_embeddings
from sdxl_train_network import SdxlNetworkTrainer


def small_unet(device):
    # Construct the real UNet on meta, then replace blocks unrelated to embedding.
    with torch.device("meta"):
        unet = model.SdxlUNet2DConditionModel()
    unet.time_embed = nn.Identity()
    unet.label_emb = nn.Flatten()
    unet.input_blocks = nn.ModuleList([nn.Sequential(nn.Identity())])
    unet.middle_block = nn.Sequential()
    unet.output_blocks = nn.ModuleList([])
    unet.out = nn.Sequential()
    return unet


class LabelProjection(nn.Module):
    def forward(self, value):
        return value[:, :320]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
@pytest.mark.parametrize("delegate", [False, True])
def test_real_forwards_and_independent_instances(device, delegate):
    first, second = small_unet(device), small_unet(device)
    first.timestep_freq_shift = 1
    times = torch.tensor([0, 1.25, 999], device=device)
    sizes = torch.tensor([[768, 640], [896, 768], [1024, 1024]], device=device)
    size_embedding = get_size_embeddings(sizes, sizes * 0, sizes, torch.device(device))
    pool = torch.zeros(3, 1280, device=device)
    y = torch.cat([pool, size_embedding], dim=1).half()
    x = torch.zeros(3, 4, 8, 8, dtype=torch.float16, device=device)
    for unet in (first, second):
        unet.label_emb = LabelProjection()
        if delegate:
            model.InferSdxlUNet2DConditionModel(unet)
    sources = [t.clone() for t in (times, y)]
    for unet, shift in ((first, 1), (second, 0), (first, 1)):
        observed = []
        labels = []
        with unet.time_embed.register_forward_pre_hook(lambda _, args: observed.append(args[0])), \
             unet.label_emb.register_forward_pre_hook(lambda _, args: labels.append(args[0])):
            unet(x, times, y=y)
            unet(x, times[-1], y=y)
        assert torch.equal(observed[0], model.get_timestep_embedding(times, 320, downscale_freq_shift=shift).half())
        assert observed[0].dtype == x.dtype
        assert torch.equal(observed[1], observed[0][-1:].expand(3, -1))
        assert all(torch.equal(label, y) for label in labels)
        assert not unet.state_dict()
    assert first.timestep_freq_shift == 1 and second.timestep_freq_shift == 0
    assert torch.equal(times, sources[0]) and torch.equal(y, sources[1])
    reference = model.get_timestep_embedding(times, 320, downscale_freq_shift=1)
    assert reference[-1, [80, 240]].cpu().tolist() == pytest.approx([-0.961048, -0.276383], abs=1e-5)


@pytest.mark.parametrize("shift", [0, 1])
def test_deep_shrink_uses_original_timestep(shift):
    unet = small_unet("cpu")
    unet.label_emb = LabelProjection()
    unet.timestep_freq_shift = shift
    wrapped = model.InferSdxlUNet2DConditionModel(unet)
    wrapped.set_deep_shrink(0, ds_timesteps_1=650, ds_ratio=0.5)
    seen = []
    with unet.input_blocks[0][0].register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[-2:])):
        for t in (649.5, 650):
            unet(torch.zeros(1, 4, 8, 8), torch.tensor(t), y=torch.zeros(1, 2816))
    assert seen == [(8, 8), (4, 4)]


def test_shift_installed_only_on_returned_model():
    args = SimpleNamespace(mem_eff_attn=False, xformers=False, sdpa=True)
    first, second = small_unet("cpu"), small_unet("cpu")
    trainer = SdxlNetworkTrainer()
    trainer.maruo_config = SimpleNamespace(values={"downscale_freq_shift": True, "fp16_safe_norms_mode": "off"})
    with patch("sdxl_train_network.sdxl_train_util.load_target_model", return_value=(True, None, None, Mock(), first, None, None)), \
         patch("sdxl_train_network.model_io.replace_unet_modules"):
        _, _, _, result = trainer.load_target_model(args, torch.float16, Mock())
    assert result is first and first.timestep_freq_shift == 1 and second.timestep_freq_shift == 0
