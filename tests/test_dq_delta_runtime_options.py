from types import SimpleNamespace

import pytest

from train_network import (
    _dq_delta_is_configured,
    _set_delta_fake_quant_compat,
    _set_delta_quant_scope_compat,
    resolve_dq_delta_runtime_options,
    setup_parser,
)


def _args(**overrides):
    values = {
        "dq_delta_scope": "both",
        "dq_delta_log_scope": None,
        "dq_delta_auto_scope": None,
        "dq_delta_auto_range_mul": False,
        "dq_delta_use_triton": False,
        "dq_delta_triton_stats": False,
        "dq_delta_triton_fused_up_mode": "off",
        "dq_delta_triton_fused_up_scope": "unet",
        "dq_delta_triton_fused_up_diagnostics": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_scope_defaults_inherit_apply_scope():
    options = resolve_dq_delta_runtime_options(_args(dq_delta_scope="te"))

    assert options.apply_scope_requested == "te"
    assert options.apply_scope_resolved == "te"
    assert options.log_scope_requested == "inherit"
    assert options.log_scope_resolved == "te"
    assert options.auto_scope_requested == "inherit"
    assert options.auto_scope_resolved == "te"
    assert options.fused_up_mode_resolved == "off"
    assert options.fused_up_scope_resolved == "none"


def test_auto_scope_must_be_subset_of_apply_scope_when_enabled():
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_dq_delta_runtime_options(
            _args(
                dq_delta_scope="unet",
                dq_delta_auto_scope="both",
                dq_delta_auto_range_mul=True,
            )
        )


def test_auto_scope_subset_is_accepted():
    options = resolve_dq_delta_runtime_options(
        _args(
            dq_delta_scope="both",
            dq_delta_auto_scope="unet",
            dq_delta_auto_range_mul=True,
        )
    )

    assert options.auto_scope_requested == "unet"
    assert options.auto_scope_resolved == "unet"


def test_log_scope_is_intersected_with_apply_scope():
    options = resolve_dq_delta_runtime_options(
        _args(
            dq_delta_scope="unet",
            dq_delta_log_scope="both",
        )
    )

    assert options.log_scope_requested == "both"
    assert options.log_scope_resolved == "unet"
    assert "restricted to the apply scope" in options.log_scope_adjustment_reason


def test_empty_log_scope_intersection_is_safely_resolved_to_none():
    options = resolve_dq_delta_runtime_options(
        _args(
            dq_delta_scope="unet",
            dq_delta_log_scope="te",
        )
    )

    assert options.log_scope_requested == "te"
    assert options.log_scope_resolved == "none"
    assert "log_resolved=none" in options.log_scope_adjustment_reason


def test_explicit_auto_scope_outside_apply_is_rejected_even_while_auto_is_disabled():
    with pytest.raises(ValueError, match="must be a subset"):
        resolve_dq_delta_runtime_options(
            _args(
                dq_delta_scope="unet",
                dq_delta_auto_scope="te",
                dq_delta_auto_range_mul=False,
            )
        )


def test_fused_up_requires_main_triton_switch():
    with pytest.raises(ValueError, match="requires --dq_delta_use_triton"):
        resolve_dq_delta_runtime_options(
            _args(dq_delta_triton_fused_up_mode="c0")
        )


def test_fused_up_scope_is_intersected_with_apply_scope():
    options = resolve_dq_delta_runtime_options(
        _args(
            dq_delta_scope="unet",
            dq_delta_use_triton=True,
            dq_delta_triton_fused_up_mode="c0",
            dq_delta_triton_fused_up_scope="both",
        )
    )

    assert options.fused_up_mode_resolved == "c0"
    assert options.fused_up_scope_resolved == "unet"
    assert options.fused_up_disabled_reason is None


def test_empty_fused_up_intersection_is_a_startup_error():
    with pytest.raises(ValueError, match="must overlap"):
        resolve_dq_delta_runtime_options(
            _args(
                dq_delta_scope="te",
                dq_delta_use_triton=True,
                dq_delta_triton_fused_up_mode="c0",
                dq_delta_triton_fused_up_scope="unet",
            )
        )


def test_runtime_metadata_uses_stable_requested_and_resolved_keys():
    options = resolve_dq_delta_runtime_options(
        _args(
            dq_delta_scope="both",
            dq_delta_log_scope="unet",
            dq_delta_auto_scope="te",
            dq_delta_use_triton=True,
            dq_delta_triton_stats=True,
            dq_delta_triton_fused_up_diagnostics=True,
        )
    )

    assert options.metadata() == {
        "ss_dq_scope_semantics_version": "2",
        "ss_dq_delta_apply_scope_requested": "both",
        "ss_dq_delta_apply_scope_resolved": "both",
        "ss_dq_delta_log_scope_requested": "unet",
        "ss_dq_delta_log_scope_resolved": "unet",
        "ss_dq_delta_auto_scope_requested": "te",
        "ss_dq_delta_auto_scope_resolved": "te",
        "ss_dq_delta_triton_fused_up_mode_requested": "off",
        "ss_dq_delta_triton_fused_up_mode_resolved": "off",
        "ss_dq_delta_triton_fused_up_scope_requested": "unet",
        "ss_dq_delta_triton_fused_up_scope_resolved": "none",
        "ss_dq_delta_triton_fused_up_diagnostics": True,
        "ss_dq_delta_use_triton": True,
        "ss_dq_delta_triton_stats": True,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, False),
        ({"dq_delta_step": 100}, True),
        ({"dq_delta_bits": 8}, True),
        ({"dq_delta_bits_sched": "0.0:6,0.5:8"}, True),
        ({"dq_delta_step": 0, "dq_delta_bits": None, "dq_delta_bits_sched": None}, False),
    ],
)
def test_dq_configured_matches_runtime_activation_inputs(overrides, expected):
    assert _dq_delta_is_configured(_args(**overrides)) is expected


def test_new_cli_defaults_and_values():
    parser = setup_parser()

    defaults = parser.parse_args([])
    assert defaults.dq_delta_auto_scope is None
    assert defaults.dq_delta_triton_fused_up_mode == "off"
    assert defaults.dq_delta_triton_fused_up_scope == "unet"
    assert defaults.dq_delta_triton_fused_up_diagnostics is False

    parsed = parser.parse_args(
        [
            "--dq_delta_auto_scope",
            "te",
            "--dq_delta_triton_fused_up_mode",
            "c0",
            "--dq_delta_triton_fused_up_scope",
            "both",
            "--dq_delta_triton_fused_up_diagnostics",
        ]
    )
    assert parsed.dq_delta_auto_scope == "te"
    assert parsed.dq_delta_triton_fused_up_mode == "c0"
    assert parsed.dq_delta_triton_fused_up_scope == "both"
    assert parsed.dq_delta_triton_fused_up_diagnostics is True


def test_compat_setter_filters_new_triton_kwargs_for_older_networks():
    class OldNetwork:
        def set_delta_fake_quant(self, step, mode, *, bits=None):
            self.call = (step, mode, bits)

    network = OldNetwork()
    _set_delta_fake_quant_compat(
        network,
        None,
        "stoch",
        bits=8,
        use_triton=True,
        triton_stats=True,
        triton_fused_up_mode="c0",
        triton_fused_up_scope="unet",
        triton_fused_up_diagnostics=True,
    )

    assert network.call == (None, "stoch", 8)


def test_scope_compat_uses_native_v2_setter_when_available():
    class Network:
        def set_delta_quant_scope(self, scope):
            self.scope = scope

    network = Network()
    assert _set_delta_quant_scope_compat(network, "te") == "native"
    assert network.scope == "te"


def test_scope_compat_preserves_legacy_network_initial_scope_behavior():
    te_v2 = SimpleNamespace(delta_q_scope_allowed=True)
    te_legacy = SimpleNamespace(delta_q_enabled=True)
    unet_v2 = SimpleNamespace(delta_q_scope_allowed=True)
    unet_legacy = SimpleNamespace(delta_q_enabled=True)
    network = SimpleNamespace(
        text_encoder_loras=[te_v2, te_legacy],
        unet_loras=[unet_v2, unet_legacy],
    )

    assert _set_delta_quant_scope_compat(network, "unet") == "legacy"
    assert te_v2.delta_q_scope_allowed is False
    assert te_legacy.delta_q_enabled is False
    assert unet_v2.delta_q_scope_allowed is True
    assert unet_legacy.delta_q_enabled is True


def test_scope_compat_reapplies_runtime_state_for_legacy_modules():
    te = SimpleNamespace(delta_q_enabled=True)
    unet = SimpleNamespace(delta_q_enabled=True)
    network = SimpleNamespace(text_encoder_loras=[te], unet_loras=[unet])

    # Simulate the old network-wide runtime setter overwriting every module.
    te.delta_q_enabled = False
    unet.delta_q_enabled = False
    assert (
        _set_delta_quant_scope_compat(
            network,
            "unet",
            runtime_enabled=False,
            warn=False,
        )
        == "legacy"
    )
    assert te.delta_q_enabled is False
    assert unet.delta_q_enabled is False

    te.delta_q_enabled = True
    unet.delta_q_enabled = True
    _set_delta_quant_scope_compat(
        network,
        "unet",
        runtime_enabled=True,
        warn=False,
    )
    assert te.delta_q_enabled is False
    assert unet.delta_q_enabled is True
