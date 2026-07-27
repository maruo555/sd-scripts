import json
from types import SimpleNamespace

import torch

from tools.make_lora_diagnostic_report import build_chart_payload, parse_grad_log, sanitize_json
from train_network import NetworkTrainer, resolve_grad_norm_settings


def test_all_grad_norm_presets_disable_cosine_logging():
    for mode in ("stable", "stable_no_threshoff", "gamble"):
        settings = resolve_grad_norm_settings(SimpleNamespace(grad_norm_mode=mode))
        assert settings[3] is False


class _SingleProcessAccelerator:
    num_processes = 1

    def reduce(self, tensor, reduction):
        raise AssertionError("single-process training must not call accelerator.reduce")


class _MultiProcessAccelerator:
    num_processes = 2

    def __init__(self):
        self.reduce_calls = 0

    def reduce(self, tensor, reduction):
        assert reduction == "mean"
        self.reduce_calls += 1
        return tensor + 1


def _network_with_grad():
    network = torch.nn.Linear(2, 1, bias=False)
    network.weight.grad = torch.tensor([[2.0, 3.0]])
    return network


def test_all_reduce_network_is_noop_for_single_process():
    network = _network_with_grad()
    original_grad = network.weight.grad

    NetworkTrainer().all_reduce_network(_SingleProcessAccelerator(), network)

    assert network.weight.grad is original_grad
    assert torch.equal(network.weight.grad, torch.tensor([[2.0, 3.0]]))


def test_all_reduce_network_still_reduces_for_multiple_processes():
    network = _network_with_grad()
    accelerator = _MultiProcessAccelerator()

    NetworkTrainer().all_reduce_network(accelerator, network)

    assert accelerator.reduce_calls == 1
    assert torch.equal(network.weight.grad, torch.tensor([[3.0, 4.0]]))


def test_diagnostic_report_accepts_grad_log_without_cosine(tmp_path):
    log_path = tmp_path / "gradient_logs+without_cosine.txt"
    log_path.write_text(
        "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff,Scale\n"
        "0,0,10.0,200000.0,0.5,0,65536\n"
        "0,1,12.0,200000.0,0.4,0,65536\n",
        encoding="utf-8",
    )

    grad_data = parse_grad_log(str(log_path), ma_window=2)
    charts = build_chart_payload(grad_data, None, None, None, None)

    assert grad_data["cosine"] == [None, None]
    assert grad_data["summary"]["cosine_valid_ratio"] == 0.0
    assert next(chart for chart in charts["grad"] if chart["id"] == "cosine")["series"][0]["y"] == [None, None]
    json.dumps(sanitize_json(charts), allow_nan=False)
