from types import SimpleNamespace as NS

import pytest
import torch

from library import lora_rank_stats as rank
from library.group_loss import GroupLossObserver


class Adapter(torch.nn.Module):
    def __init__(self, name, down, up, *, te=None, scale=1.0, multiplier=1.0):
        super().__init__()
        self.lora_name = name
        self.lora_down = torch.nn.Module()
        self.lora_down.register_parameter("weight", torch.nn.Parameter(down))
        self.lora_up = torch.nn.Module()
        self.lora_up.register_parameter("weight", torch.nn.Parameter(up))
        self.text_encoder_index = te
        self.scale = scale
        self.multiplier = multiplier


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_true_rank_uses_post_step_parameters_without_mutation(device, monkeypatch):
    eye = torch.eye(2, device=device)
    one = Adapter("module1", eye.clone(), eye.clone())
    two = Adapter("module2", torch.stack([eye, eye]).reshape(2, 1, 2, 2).half(),
                  eye.reshape(2, 2, 1, 1).half(), scale=0.5)
    singular = Adapter("singular", torch.stack([eye[0], eye[0]]), eye.clone())
    zero = Adapter("zero", eye.clone(), torch.zeros_like(eye))
    te1 = Adapter("te1", eye.clone(), eye.clone(), te=1)
    te2 = Adapter("te2", eye.clone(), eye.clone(), te=2)
    network = NS(unet_loras=[one, two, singular, zero], text_encoder_loras=[te1, te2])
    optimizer = NS(param_groups=[{"params": list(te1.parameters()) + list(te2.parameters())},
                                 {"params": list(one.parameters())}, {"params": list(two.parameters())}])
    observer = rank.RankObserver(network, optimizer)
    calls = []
    def rates():
        calls.append(1)
        return [0.0, .003, .0001]
    scheduler = NS(get_last_lr=rates)
    assert not observer.observe(1, 99, scheduler)
    assert not observer.observe(1, 100, scheduler, guardian_skipped=True)
    assert not calls
    with torch.no_grad():
        one.lora_up.weight.mul_(2)
    one.lora_down.weight.grad = eye.clone()
    before = one.lora_down.weight._version, one.lora_up.weight._version
    cpu_rng = torch.get_rng_state()
    device_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    monkeypatch.setattr(rank, "_RANK_STATS_BATCH_MAX_BYTES", 1)
    rows = observer.observe(2, 200, scheduler)
    assert list(observer.module_names) == [row["module"] for row in rows]
    assert len(calls) == 1
    for module, row in zip(network.unet_loras, rows):
        expected = rank._compute_lora_effective_rank_stats(module)
        assert row["rank_dim"] == expected["r"]
        for field in ("sat", "top1", "energy"):
            assert row["rank_" + field] == pytest.approx(expected[field], rel=2e-5, abs=2e-6)
        assert (row["te1_lr_max"], row["te2_lr_min"], row["unet_lr_min"], row["unet_lr_max"]) == (0, 0, .0001, .003)
    assert rows[0]["rank_energy"] == pytest.approx(8)
    assert rows[0]["rank_top1"] == pytest.approx(.5)
    assert rows[-1]["rank_sat"] == rows[-1]["rank_top1"] == 0
    assert before == (one.lora_down.weight._version, one.lora_up.weight._version)
    assert torch.equal(one.lora_down.weight.grad, eye)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), device_rng)


def test_absent_lr_and_failed_rank_do_not_emit_partial_observations(monkeypatch, caplog):
    a = Adapter("one", torch.eye(2), torch.eye(2))
    observer = rank.RankObserver(NS(unet_loras=[a], text_encoder_loras=[]), NS(param_groups=[]))
    scheduler = NS(get_last_lr=lambda: [])
    assert observer.observe(1, 100, scheduler)[0]["te1_lr_min"] is None
    monkeypatch.setattr(rank, "_compute_lora_effective_rank_stats_batched", lambda modules: [])
    assert observer.observe(2, 200, scheduler) == []
    assert "200" in caplog.text


def make_dataset(*groups):
    return NS(batch_size=1, subsets=[NS(group=g) for g in groups],
              image_data={"same": NS(bucket_reso=(768, 1024))})


def batch(dataset, subset):
    return {"subset_indices": [(dataset, subset)], "image_keys": ["same"], "loss_weights": torch.tensor([.25])}


def test_group_observer_uses_training_identities_and_weighted_loss():
    ds1, ds2 = make_dataset(" A ", "B"), make_dataset("A", " ")
    ds2.image_data["same"].bucket_reso = (1024, 768)
    observer = GroupLossObserver(NS(datasets=[ds1, ds2]))
    a, b, other_a = batch(0, 0), batch(0, 1), batch(1, 0)
    def sample(z, loss, epoch, step, skip=False):
        return observer.observe(z, torch.tensor([0]), loss, epoch, step, skip)
    assert [r["subset_index"] for r in observer.resolved_subsets] == [0, 1, 2, 3]
    assert sample(a, 2., 1, 98)[0]["subset_index"] == 0
    assert sample(b, 10., 1, 99)[0]["group_observation_count"] == 1
    assert not sample(a, 4., 1, 100, True)
    second = sample(other_a, 4., 1, 101)[0]
    assert (second["subset_index"], second["bucket_reso"], second["timestep"]) == (2, "1024x768", 0)
    assert second["group_ema_loss"] == pytest.approx(2.04)
    assert not sample(a, float("nan"), 1, 102)
    weighted = (torch.tensor([4.]) * a["loss_weights"]).mean().item()
    third = sample(a, weighted, 2, 103)[0]
    assert (third["loss"], third["group_observation_count"], third["bucket_reso"]) == (1., 3, "768x1024")
    assert third["group_ema_loss"] == pytest.approx(2.0192)
    assert sample(batch(1, 1), 2., 2, 104)[0]["group"] == "__ungrouped__"


def test_subset_missing_unassigned_zero_and_group_batch_contract():
    observer = GroupLossObserver(NS(datasets=[make_dataset(None)]))
    def read(z, t=None):
        return observer.observe(z, t, 1., 1, 1)[0]
    assert read(batch(0, None))["subset_index"] == -1
    assert read(batch(0, -1))["subset_index"] == -1
    assert read({"latents": torch.ones(1, 4, 8, 8)})["subset_index"] is None
    assert read({"images": torch.ones(1, 3, 8, 8)})["bucket_reso"] is None
    assert read(batch(0, 0))["timestep"] is None
    assert read(batch(0, 0), 0)["timestep"] == 0
    with pytest.raises(ValueError, match="batch size"):
        read({"images": torch.ones(2, 3, 8, 8)})
    with pytest.raises(ValueError, match="infer"):
        read({"not_a_batched_field": None})
    with pytest.raises(ValueError, match="batch size"):
        GroupLossObserver(NS(datasets=[NS(batch_size=2)]))
