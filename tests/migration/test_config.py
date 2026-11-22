"""Public SDXL settings, with real upstream CLI/TOML loading and no model or dataset I/O."""

import random
from unittest import mock

import pytest
import toml

from library import maruo_training_config as config
from sdxl_train_network import SdxlNetworkTrainer, parse_training_args, setup_parser


@pytest.fixture(scope="module")
def parser():
    return setup_parser()


def settings(parser, *argv):
    return config.resolve_training_config(parse_training_args(parser, ["--network_module", "networks.lora", *argv]))


def config_file(tmp_path, contents):
    path = tmp_path / "training.toml"
    path.write_text(toml.dumps(contents), encoding="utf-8")
    return str(path)


def test_output_root_preserved_without_creating_run(parser, tmp_path):
    output = tmp_path / "lora_output"
    args = parse_training_args(parser, ["--network_module", "networks.lora", "--output_dir", str(output)])
    resolved = config.resolve_training_config(args)
    assert resolved.output_dir == args.output_dir == str(output)
    assert not output.exists()


def test_all_off_passthrough(parser):
    before = random.getstate()
    args = parse_training_args(parser, ["--network_module", "networks.lora"])
    current = config.resolve_training_config(args)
    assert current.requested == {} and not current.changes_training and not current.observation_features
    assert current.text_encoder_lrs is None and current.bits_schedule == ()
    with mock.patch("train_network.NetworkTrainer.train", return_value=17) as upstream:
        trainer = SdxlNetworkTrainer()
        assert trainer.train(args) == 17
        upstream.assert_called_once_with(args)
    assert random.getstate() == before
    assert not hasattr(args, "dq_delta_mode")


@pytest.mark.parametrize("mode,granularity", [(False, False), (True, False), (False, True), (True, True)])
def test_split_dq_settings(parser, tmp_path, mode, granularity):
    cli = ["--dq_delta_bits", "8"]
    if mode:
        cli.extend(["--dq_delta_mode", "stoch"])
    path = config_file(tmp_path, {"training": {"dq_delta_granularity": "channel"}} if granularity else {})
    current = settings(parser, *cli, "--config_file", path)
    assert current.values["dq_delta_mode"] == "stoch" and current.values["dq_delta_granularity"] == "channel"
    assert ("dq_delta_mode" in current.requested) == mode
    assert ("dq_delta_granularity" in current.requested) == granularity
    assert current.training_features == {"dq"}


@pytest.mark.parametrize("source", ["cli", "toml"])
@pytest.mark.parametrize("dq_mode", ["disabled", "fixed", "automatic"])
def test_dq_configurations(parser, tmp_path, dq_mode, source):
    values = dict(optimizer_type="AdamW8bitFast", text_encoder_lr=0.0002, text_encoder_lr1=0.0003, text_encoder_lr2=0.0002,
                  te_mlp_fc_only=True, downscale_freq_shift=True, grad_norm_mode="stable_no_threshoff",
                  avg_cp=True, avg_cp_mode="promote", avg_mode="ema", avg_reset_stats=False, fp16_safe_norms_mode="strict",
                  avg_save_final_raw=True, rank_log=True, rank_log_mode="per_module", lr_warmup_steps=0.05)
    if dq_mode != "disabled":
        values.update(dq_delta_bits=8, dq_delta_scope="both", dq_delta_log=True, dq_delta_begin_after_lr_warmup=True,
                      dq_delta_use_triton=True, dq_delta_triton_stats=True, dq_delta_mode="stoch", dq_delta_granularity="channel")
    if dq_mode == "fixed":
        values.update(dq_delta_range_mul=2.7, dq_delta_range_mul_attn2=3.75, dq_delta_range_mul_te=3.75)
    if dq_mode == "automatic":
        values.update(dq_delta_range_mul=3.0, dq_delta_auto_range_mul=True, dq_delta_auto_preset="clip_rate_high",
                      dq_delta_auto_init_range_mul_from_band=True, dq_delta_auto_use_raw=True)
    if source == "cli":
        argv = []
        for key, value in values.items():
            if value is False:
                argv.append("--no-" + key)
            elif value is True:
                argv.append("--" + key)
            else:
                argv.extend(["--" + key, str(value)])
    else:
        argv = ["--config_file", config_file(tmp_path, {"train": values})]
    resolved = settings(parser, *argv)
    assert resolved.text_encoder_lrs == (0.0003, 0.0002)
    assert resolved.values["avg_reset_stats"] is False and "avg_reset_stats" in resolved.requested
    assert resolved.requested["avg_cp_mode"] == "promote" and "optimizer_fast" in resolved.training_features
    assert ("dq" in resolved.training_features) == (dq_mode != "disabled")
    if dq_mode == "fixed":
        assert resolved.values["dq_delta_range_mul_attn2"] == resolved.values["dq_delta_range_mul_te"] == 3.75
    if dq_mode == "automatic":
        assert resolved.requested["dq_delta_range_mul"] == 3.0
        assert (resolved.values["dq_delta_scope"], resolved.values["dq_delta_log_scope"], resolved.values["dq_delta_auto_scope"]) == (
            "both", "both", "unet"
        )


def test_cli_zero_and_false_override_toml(parser, tmp_path):
    path = config_file(tmp_path, {"text_encoder_lr2": 0.0002, "train": {"avg_reset_stats": True}})
    current = settings(parser, "--config_file", path, "--text_encoder_lr2", "0", "--no-avg_reset_stats")
    assert current.requested == {"text_encoder_lr2": 0.0, "avg_reset_stats": False}
    assert current.text_encoder_lrs[1] == 0


@pytest.mark.parametrize("key", sorted(config.RETIRED))
def test_retired_rejected_from_cli_and_file(parser, tmp_path, key):
    flag = "--te-lr-after" if key == "te_lr_after" else "--" + key
    with pytest.raises(ValueError, match=key.replace("_", "[-_]")):
        settings(parser, flag)
    for record in ({key: False}, {"train": {key: False}}):
        path = config_file(tmp_path, record)
        with pytest.raises(ValueError, match=key):
            settings(parser, "--config_file", path, "--rank_log_mode", "per_module")


def test_retired_negative_alias(parser):
    with pytest.raises(ValueError, match="avg_promote_pick|avg_shadow_log_jsonl"):
        settings(parser, "--no-avg_shadow_log_jsonl")


@pytest.mark.parametrize("kwargs,key", [
    ({"dq_delta_bits": 1}, "dq_delta_bits"), ({"dq_delta_bits_sched": "0.0:8,0.5:1"}, "0.5:1"),
    ({"dq_delta_bits_sched": "1:nan"}, "1:nan"), ({"dq_delta_bits": True}, "dq_delta_bits"),
    ({"dq_delta_mode": "det"}, "dq_delta_mode"), ({"dq_delta_granularity": "tensor"}, "dq_delta_granularity"),
    ({"dq_delta_stat": "absmax"}, "dq_delta_stat"), ({"dq_delta_log_mode": "per_module"}, "dq_delta_log_mode"),
    ({"dq_delta_log_detail": "full"}, "dq_delta_log_detail"), ({"avg_cp_mode": "shadow"}, "avg_cp_mode"),
    ({"avg_mode": "metric"}, "avg_mode"), ({"rank_log_mode": "summary"}, "rank_log_mode"),
    ({"fp16_safe_norms_mode": "native_accum"}, "fp16_safe_norms_mode"), ({"rank_log": "true"}, "rank_log"),
    ({"dq_delta_bits": 8, "dq_delta_range_mul_attn2": 3.75, "dq_delta_auto_range_mul": True}, "range_mul_attn2"),
    ({"dq_delta_begin": float("nan")}, "dq_delta_begin"), ({"avg_window": 0}, "avg_window")])
def test_invalid_toml_never_loads_model(parser, tmp_path, kwargs, key):
    path = config_file(tmp_path, {"train": kwargs})
    with pytest.raises(ValueError, match=key):
        settings(parser, "--config_file", path)


@pytest.mark.parametrize("source", ["cli", "toml"])
@pytest.mark.parametrize("warmup", [None, 0, -1, float("nan")])
def test_warmup_relative_dq_requires_positive_warmup(parser, tmp_path, source, warmup):
    if source == "cli":
        args = ["--dq_delta_bits", "8", "--dq_delta_begin_after_lr_warmup"]
        if warmup is not None:
            args.extend(["--lr_warmup_steps", str(warmup)])
    else:
        options = {"dq_delta_bits": 8, "dq_delta_begin_after_lr_warmup": True}
        if warmup is not None:
            options["lr_warmup_steps"] = warmup
        args = ["--config_file", config_file(tmp_path, {"train": options})]
    with pytest.raises(ValueError, match="dq_delta_begin_after_lr_warmup.*lr_warmup_steps > 0"):
        settings(parser, *args)


def test_scheduler_and_scope(parser):
    current = settings(parser, "--dq_delta_bits_sched", "0.7:6,0.5:4,", "--dq_delta_scope", "te",
                       "--dq_delta_auto_range_mul", "--dq_delta_begin", "0.7", "--dq_delta_begin_after_lr_warmup",
                       "--lr_warmup_steps", "2")
    assert current.bits_schedule == ((0.5, 4), (0.7, 6)) and current.values["dq_delta_bits"] is None
    assert current.values["dq_delta_auto_preset"] == "default"
    assert current.values["dq_delta_log_scope"] == current.values["dq_delta_auto_scope"] == "te"


def test_te_lrs(parser, tmp_path):
    assert settings(parser, "--text_encoder_lr", "0.0003", "0.0002").text_encoder_lrs is None
    assert settings(parser, "--te_mlp_fc_only", "--text_encoder_lr", "0.0003", "0.0002").text_encoder_lrs == (0.0003, 0.0002)
    assert settings(parser, "--te_mlp_fc_only", "--text_encoder_lr", "--learning_rate", "0.00035").text_encoder_lrs == (0.00035,) * 2
    assert settings(parser, "--text_encoder_lr1", "0.0003", "--text_encoder_lr", "0").text_encoder_lrs == (0.0003, 0)
    with pytest.raises(ValueError, match="text_encoder_lr2"):
        settings(parser, "--text_encoder_lr", "0.0003", "0.0002", "--text_encoder_lr2", "0.0001")
    with pytest.raises(ValueError, match="at most 2"):
        settings(parser, "--te_mlp_fc_only", "--text_encoder_lr", "0.1", "0.2", "0.3")


def test_inactive_defaults_and_observations(parser):
    first = settings(parser, "--avg_cp", "--use_8bit_adam", "--grad_norm_mode", "stable")
    assert first.values["avg_cp_mode"] == "live" and first.values["avg_reset_stats"] is True
    current = settings(parser, "--rank_log_mode", "per_module", "--fp16_safe_norms_mode", "off")
    assert not current.changes_training and not current.observation_features
    observed = settings(parser, "--rank_log", "--group_loss_log", "--resume", "existing-state")
    assert observed.observation_features == {"rank_log", "group_loss_log"}
    assert observed.values["rank_log_mode"] == "per_module" and not observed.changes_training


@pytest.mark.parametrize("option,value", [("resume", "state"), ("initial_epoch", "2"), ("initial_step", "3"),
                                          ("gradient_accumulation_steps", "2")])
def test_custom_start_guard(parser, option, value):
    with pytest.raises(ValueError, match=option):
        settings(parser, "--te_mlp_fc_only", "--" + option, value)
    assert not settings(parser, "--" + option, value).changes_training
    assert settings(parser, "--te_mlp_fc_only", "--initial_step", "0", "--initial_epoch", "3").changes_training


def test_foreign_network_keeps_its_toml(parser, tmp_path):
    path = config_file(tmp_path, {"network_module": "thirdparty.network", "extra": {"rank_log_every": 13, "plugin_local_key": 6}})
    args = parse_training_args(parser, ["--config_file", path])
    assert args.rank_log_every == 13 and args.plugin_local_key == 6
    assert config.resolve_training_config(args) is None
    with pytest.raises(ValueError, match="networks.lora"):
        parse_training_args(parser, ["--config_file", path, "--dq_delta_bits", "8"])


def test_unwired_features_block_standard_entry(parser):
    args = parse_training_args(parser, ["--network_module", "networks.lora", "--rank_log"])
    with mock.patch.object(config, "WIRED_FEATURES", frozenset()), mock.patch("train_network.NetworkTrainer.train") as upstream:
        with pytest.raises(ValueError, match="rank_log"):
            SdxlNetworkTrainer().train(args)
        upstream.assert_not_called()
