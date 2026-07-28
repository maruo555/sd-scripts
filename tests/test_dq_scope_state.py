from types import SimpleNamespace

import pytest
import torch

from networks import lora, lora_lbw


IMPLEMENTATIONS = (lora, lora_lbw)


def _make_modules(implementation):
    te = implementation.LoRAModule("lora_te_test", torch.nn.Linear(4, 4), lora_dim=4)
    unet = implementation.LoRAModule("lora_unet_test", torch.nn.Linear(4, 4), lora_dim=4)
    network = SimpleNamespace(text_encoder_loras=[te], unet_loras=[unet])
    return network, te, unet


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("scope", "te_allowed", "unet_allowed"),
    (("unet", False, True), ("te", True, False), ("both", True, True)),
)
def test_delta_quant_scope_survives_runtime_toggles(implementation, scope, te_allowed, unet_allowed):
    network, te, unet = _make_modules(implementation)

    implementation.LoRANetwork.set_delta_quant_scope(network, scope)
    assert te.delta_q_scope_allowed is te_allowed
    assert unet.delta_q_scope_allowed is unet_allowed
    assert te.delta_q_enabled is te_allowed
    assert unet.delta_q_enabled is unet_allowed

    # Bits schedules and automatic range updates reconfigure quantization at
    # runtime; neither operation may erase the configured module scope.
    implementation.LoRANetwork.set_delta_fake_quant(network, None, bits=6, range_mul=2.5)
    assert te.delta_q_scope_allowed is te_allowed
    assert unet.delta_q_scope_allowed is unet_allowed
    assert te.delta_q_enabled is te_allowed
    assert unet.delta_q_enabled is unet_allowed

    implementation.LoRANetwork.set_delta_quant_enabled(network, False)
    assert te.delta_q_runtime_enabled is False
    assert unet.delta_q_runtime_enabled is False
    assert te.delta_q_enabled is False
    assert unet.delta_q_enabled is False

    implementation.LoRANetwork.set_delta_quant_enabled(network, True)
    assert te.delta_q_runtime_enabled is True
    assert unet.delta_q_runtime_enabled is True
    assert te.delta_q_enabled is te_allowed
    assert unet.delta_q_enabled is unet_allowed


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_legacy_delta_q_enabled_assignment_does_not_override_scope(implementation):
    network, te, _ = _make_modules(implementation)
    implementation.LoRANetwork.set_delta_quant_scope(network, "unet")

    te.delta_q_enabled = True

    assert te.delta_q_runtime_enabled is True
    assert te.delta_q_scope_allowed is False
    assert te.delta_q_enabled is False


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_delta_quant_scope_rejects_invalid_value(implementation):
    network, _, _ = _make_modules(implementation)

    with pytest.raises(ValueError, match="invalid delta quantization scope"):
        implementation.LoRANetwork.set_delta_quant_scope(network, "invalid")


def test_lora_fused_up_runtime_settings_are_propagated():
    network, te, unet = _make_modules(lora)

    lora.LoRANetwork.set_delta_fake_quant(
        network,
        None,
        triton_fused_up_mode="c0",
        triton_fused_up_scope="unet",
        triton_fused_up_diagnostics=True,
    )

    for module in (te, unet):
        assert module.delta_q_triton_fused_up_mode == "c0"
        assert module.delta_q_triton_fused_up_scope == "unet"
        assert module.delta_q_triton_fused_up_diagnostics is True
