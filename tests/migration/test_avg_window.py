import pytest
import torch

from library.avg_ckpt_util import AverageWindow, apply_lora_snapshot, average_state_dicts


def test_pure_average_retains_legacy_common_keys():
    first = {"lora_a": torch.tensor(1.0), "lora_b": torch.tensor(3.0)}
    second = {"lora_a": torch.tensor(3.0), "lora_c": torch.tensor(5.0)}
    result = average_state_dicts([first, second])
    assert list(result) == ["lora_a"] and result["lora_a"].item() == 2


@pytest.mark.parametrize(("mode", "expected"), [("uniform", 3), ("ema", 3.5)])
def test_window_is_independent_from_live_weights_and_eviction(mode, expected):
    network = torch.nn.Module()
    network.register_parameter("lora_linear", torch.nn.Parameter(torch.tensor([1.])))
    network.register_buffer("lora_alpha", torch.tensor(4.))
    network.register_buffer("other", torch.tensor(9.))
    window = AverageWindow(3, mode)
    a, candidate = window.append(2, network)
    assert candidate is None and set(a) == {"lora_linear","lora_alpha"}
    network.lora_linear.data.fill_(3)
    assert window.append(3, network)[1] is None
    network.lora_linear.data.fill_(5)
    raw, candidate = window.append(4, network)
    assert a["lora_linear"].item()==1 and raw["lora_linear"].item()==5
    assert raw["lora_linear"].dtype==torch.float32 and raw["lora_linear"].device.type=="cpu"
    apply_lora_snapshot(network, candidate)
    assert network.lora_linear.item()==expected and raw["lora_linear"].item()==5
    assert network.lora_alpha.item()==4 and network.other.item()==9
    window.append(5, network)
    assert window.epochs==(3,4,5)


def test_window_rejects_different_key_or_shape_before_mutation():
    network = torch.nn.Module()
    network.register_buffer("lora_a", torch.ones(1))
    window = AverageWindow(2,"uniform")
    window.append(1, network)
    for change in ({"lora_other":torch.tensor(1.)},{"lora_a":torch.ones(2)}):
        with pytest.raises(ValueError,match="LoRA window"):
            apply_lora_snapshot(network,change)
        assert torch.equal(network.lora_a,torch.ones(1))
    network.register_buffer("lora_extra",torch.tensor(3.))
    with pytest.raises(ValueError,match="LoRA window keys"):
        window.append(2,network)
    assert window.epochs==(1,)
