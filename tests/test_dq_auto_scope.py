"""Synthetic CPU coverage of independent log and auto-control scopes."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train_network import resolve_dq_delta_auto_scope, setup_parser
from networks.lora import DQStatsManager


@pytest.fixture(scope="module")
def parser():
    return setup_parser()


@pytest.mark.parametrize("scope, expected", [("unet", "unet"), ("both", "unet"), ("te", "te")])
def test_default_auto_scope_and_explicit_legacy_override(parser, scope, expected):
    args = parser.parse_args(["--dq_delta_scope", scope, "--dq_delta_log_scope", "both"])
    assert resolve_dq_delta_auto_scope(args) == expected
    for explicit in ("unet", "te", "both"):
        args.dq_delta_auto_scope = explicit
        assert resolve_dq_delta_auto_scope(args) == explicit


def test_cli_default_and_single_component_training(parser):
    assert resolve_dq_delta_auto_scope(parser.parse_args([])) == "unet"
    assert resolve_dq_delta_auto_scope(parser.parse_args(["--network_train_text_encoder_only"])) == "te"
    assert resolve_dq_delta_auto_scope(parser.parse_args(["--network_train_unet_only"])) == "unet"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dq_delta_auto_scope", "invalid"])
    with pytest.raises(ValueError):
        resolve_dq_delta_auto_scope(SimpleNamespace(dq_delta_auto_scope="invalid"))


def stats(args, do_log):
    manager = DQStatsManager()
    manager.begin_step(step_idx=100, device=torch.device("cpu"), do_log=do_log, do_auto=True,
                       collect_full=True, collect_zero=False, collect_near_zero=False,
                       collect_detail=False, collect_error_parts=False, log_mode="summary",
                       log_scope=args.dq_delta_log_scope or args.dq_delta_scope,
                       auto_scope=resolve_dq_delta_auto_scope(args), target="delta")
    for scope, count in (("unet", 2), ("te", 70)):
        manager.add_stats(scope=scope, module_name=scope, shape="100", numel=torch.tensor(100.),
                          clip_count=torch.tensor(float(count)), zero_count=None, near_zero_count=None,
                          sumsq=torch.tensor(100.), xq_sumsq=torch.tensor(90.), xxq_sum=torch.tensor(94.),
                          absmax=None, scale_min=None, scale_max=None, scale_sum=None, scale_count=None,
                          clip_err_sumsq=None, round_err_sumsq=None)
    return manager.export()


@pytest.mark.parametrize("do_log", [False, True])
def test_te_statistics_do_not_change_unet_control_when_log_scope_expands(parser, do_log):
    legacy = stats(parser.parse_args(["--dq_delta_scope", "unet"]), do_log)
    expanded = stats(parser.parse_args(["--dq_delta_scope", "both"]), do_log)
    assert legacy["auto_scope"] == expanded["auto_scope"] == "unet"
    for record in (legacy, expanded):
        acc = record["accum"][record["auto_scope"]]
        assert (acc.clip_count / acc.numel).item() == pytest.approx(.02)
    assert expanded["accum"]["te"].numel.item() == (100 if do_log else 0)
    assert not torch.cuda.is_initialized()


def test_explicit_combined_controller_collects_te_without_te_log(parser):
    record = stats(parser.parse_args(["--dq_delta_scope", "unet", "--dq_delta_auto_scope", "both"]), False)
    assert record["auto_scope"] == "both"
    accum = record["accum"]
    assert ((accum["unet"].clip_count + accum["te"].clip_count) /
            (accum["unet"].numel + accum["te"].numel)).item() == pytest.approx(.36)


def test_auto_csv_headers_include_control_error_values_in_both_formats():
    # These are closures inside the GPU trainer. Exercise their real pure header
    # builders without creating a trainer, dataset, accelerator or GPU context.
    tree = ast.parse(Path("train_network.py").read_text(encoding="utf-8"))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("_dq_log_header", "_dq_auto_log_header")]
    namespace = {"dq_log_error_parts": False}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<log-header-test>", "exec"), namespace)
    for full in (False, True):
        columns = namespace["_dq_auto_log_header"](full, False).split(",")
        assert columns[-2:] == ["AutoQuantErrRatioRaw", "AutoQuantErrRatioEMA"]
        assert len(columns) == len(set(columns))
