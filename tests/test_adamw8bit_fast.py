from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from library.adamw8bit_fast import AdamW8bitFast


def _assert_optimizer_state_equal(stock, fast, stock_params, fast_params) -> None:
    for stock_param, fast_param in zip(stock_params, fast_params):
        stock_state = stock.state[stock_param]
        fast_state = fast.state[fast_param]
        assert stock_state.keys() == fast_state.keys()
        for key in stock_state:
            stock_value = stock_state[key]
            fast_value = fast_state[key]
            if isinstance(stock_value, torch.Tensor):
                assert isinstance(fast_value, torch.Tensor)
                assert stock_value.dtype == fast_value.dtype
                assert torch.equal(stock_value, fast_value), key
            else:
                assert stock_value == fast_value, key


def test_cpu_parameters_use_stock_bitsandbytes_step(monkeypatch):
    param = torch.nn.Parameter(torch.ones(8))
    param.grad = torch.ones_like(param)
    optimizer = AdamW8bitFast([param], lr=1e-3)
    calls = []

    def stock_step(self, closure=None):
        calls.append(closure)
        return "stock-route"

    monkeypatch.setattr(bnb.optim.AdamW8bit, "step", stock_step)

    assert optimizer.step() == "stock-route"
    assert calls == [None]


def test_optimizer_selection_accepts_adamw8bit_fast():
    from library.train_util import get_optimizer

    args = SimpleNamespace(
        optimizer_type="AdamW8bitFast",
        use_8bit_adam=False,
        use_lion_optimizer=False,
        fused_backward_pass=False,
        gradient_accumulation_steps=1,
        optimizer_args=None,
        learning_rate=1e-3,
    )
    param = torch.nn.Parameter(torch.ones(8))

    optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])

    assert optimizer_name == "library.adamw8bit_fast.AdamW8bitFast"
    assert optimizer_args == ""
    assert isinstance(optimizer, AdamW8bitFast)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_step_matches_stock_parameters_and_all_states(dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260801)
    shapes = [(64, 80), (32, 80), (64, 128), (48, 64)]  # both sides of min_8bit_size=4096
    initial = [torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.01 for shape in shapes]
    stock_params = [torch.nn.Parameter(value.clone()) for value in initial]
    fast_params = [torch.nn.Parameter(value.clone()) for value in initial]

    stock_groups = [
        {"params": stock_params[:2], "lr": 3.5e-4},
        {"params": stock_params[2:], "lr": 2.0e-4},
    ]
    fast_groups = [
        {"params": fast_params[:2], "lr": 3.5e-4},
        {"params": fast_params[2:], "lr": 2.0e-4},
    ]
    kwargs = {"lr": 1e-3, "betas": (0.9, 0.995), "weight_decay": 0.01}
    stock = bnb.optim.AdamW8bit(stock_groups, **kwargs)
    fast = AdamW8bitFast(fast_groups, **kwargs)

    for step in range(7):
        for index, (stock_param, fast_param) in enumerate(zip(stock_params, fast_params)):
            if (step + index) % 5 == 0:
                stock_param.grad = None
                fast_param.grad = None
                continue
            grad = torch.randn(
                stock_param.shape,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            stock_param.grad = grad.clone()
            fast_param.grad = grad.clone()

        stock.step()
        fast.step()

    for stock_param, fast_param in zip(stock_params, fast_params):
        assert torch.equal(stock_param, fast_param)
    _assert_optimizer_state_equal(stock, fast, stock_params, fast_params)

    state_dtypes = {
        value.dtype
        for state in fast.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    }
    assert torch.uint8 in state_dtypes
    assert torch.float32 in state_dtypes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_step_synchronizes_once_and_skips_sync_without_gradients(monkeypatch):
    params = [
        torch.nn.Parameter(torch.randn((64, 80), device="cuda", dtype=torch.float16)),
        torch.nn.Parameter(torch.randn((64, 128), device="cuda", dtype=torch.float16)),
    ]
    optimizer = AdamW8bitFast(params, lr=1e-3)
    original_synchronize = torch.cuda.synchronize
    calls = []

    def synchronize_once(device=None):
        calls.append(device)
        return original_synchronize(device=device)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize_once)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()

    assert calls == [params[0].device]

    calls.clear()
    for param in params:
        param.grad = None
    optimizer.step()

    assert calls == []
