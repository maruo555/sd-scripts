from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from library.adamw8bit_fast import AdamW8bitFast


@pytest.fixture
def isolated_global_optim_manager(monkeypatch):
    manager = bnb.optim.GlobalOptimManager.get_instance()
    monkeypatch.setattr(manager, "pid2config", {})
    monkeypatch.setattr(manager, "index2config", {})
    monkeypatch.setattr(manager, "optimizer", None)
    monkeypatch.setattr(manager, "uses_config_override", False)
    monkeypatch.setattr(manager, "module_weight_config_triple", [])
    return manager


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


def test_cpu_parameters_use_stock_bitsandbytes_step_and_log_once(monkeypatch, caplog):
    param = torch.nn.Parameter(torch.ones(8))
    param.grad = torch.ones_like(param)
    optimizer = AdamW8bitFast([param], lr=1e-3)
    calls = []

    def stock_step(self, closure=None):
        calls.append(closure)
        return "stock-route"

    monkeypatch.setattr(bnb.optim.AdamW8bit, "step", stock_step)

    with caplog.at_level(logging.WARNING, logger="library.adamw8bit_fast"):
        assert optimizer.step() == "stock-route"
        assert optimizer.step() == "stock-route"

    assert calls == [None, None]
    messages = [record.getMessage() for record in caplog.records if record.name == "library.adamw8bit_fast"]
    assert len(messages) == 1
    assert "using stock AdamW8bit step" in messages[0]
    assert "parameter is on cpu" in messages[0]
    assert f"bitsandbytes {bnb.__version__}" in messages[0]


def test_optimizer_selection_accepts_adamw8bit_fast():
    from library.optimizer import get_optimizer

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
    kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.995),
        "weight_decay": 0.01,
    }
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


def test_percentile_clipping_uses_stock_step_and_logs_once(monkeypatch, caplog):
    param = torch.nn.Parameter(torch.ones(8))
    optimizer = AdamW8bitFast([param], lr=1e-3)
    optimizer.args.percentile_clipping = 50
    calls = []

    def stock_step(self, closure=None):
        calls.append(closure)
        return "stock-route"

    monkeypatch.setattr(bnb.optim.AdamW8bit, "step", stock_step)

    with caplog.at_level(logging.WARNING, logger="library.adamw8bit_fast"):
        assert optimizer.step() == "stock-route"
        assert not [record for record in caplog.records if record.name == "library.adamw8bit_fast"]
        param.grad = torch.ones_like(param)
        assert optimizer.step() == "stock-route"
        assert optimizer.step() == "stock-route"

    assert calls == [None, None, None]
    messages = [record.getMessage() for record in caplog.records if record.name == "library.adamw8bit_fast"]
    assert len(messages) == 1
    assert "percentile_clipping is enabled" in messages[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("key", "value", "expected_reason"),
    [
        ("percentile_clipping", 50, "parameter override enables percentile_clipping"),
        ("max_unorm", 1.0, "parameter override enables max_unorm"),
    ],
)
def test_unsafe_parameter_overrides_use_stock_step(
    monkeypatch,
    caplog,
    isolated_global_optim_manager,
    key,
    value,
    expected_reason,
):
    param = torch.nn.Parameter(torch.ones(4096, device="cuda"))
    isolated_global_optim_manager.override_config(param, key=key, value=value)
    isolated_global_optim_manager.register_parameters([param])
    optimizer = AdamW8bitFast([param], lr=1e-3)
    param.grad = torch.ones_like(param)
    calls = []

    def stock_step(self, closure=None):
        calls.append(closure)
        return "stock-route"

    monkeypatch.setattr(bnb.optim.AdamW8bit, "step", stock_step)

    with caplog.at_level(logging.WARNING, logger="library.adamw8bit_fast"):
        assert optimizer.step() == "stock-route"

    assert calls == [None]
    messages = [record.getMessage() for record in caplog.records if record.name == "library.adamw8bit_fast"]
    assert len(messages) == 1
    assert expected_reason in messages[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_inactive_unsafe_parameter_override_does_not_disable_fast_path(
    isolated_global_optim_manager,
):
    overridden = torch.nn.Parameter(torch.ones(4096, device="cuda"))
    active = torch.nn.Parameter(torch.ones(4096, device="cuda"))
    isolated_global_optim_manager.override_config(overridden, key="percentile_clipping", value=50)
    isolated_global_optim_manager.register_parameters([overridden, active])
    optimizer = AdamW8bitFast([overridden, active], lr=1e-3)

    active.grad = torch.ones_like(active)
    can_use_fast_path, device, reason = optimizer._fast_path_device()
    assert can_use_fast_path
    assert device == active.device
    assert reason is None

    overridden.grad = torch.ones_like(overridden)
    can_use_fast_path, device, reason = optimizer._fast_path_device()
    assert not can_use_fast_path
    assert device is None
    assert reason == "parameter override enables percentile_clipping"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_module_override_is_applied_before_fast_path_guard(
    monkeypatch,
    isolated_global_optim_manager,
):
    module = torch.nn.Linear(64, 64, bias=False, device="cuda")
    isolated_global_optim_manager.register_module_override(module, "weight", {"max_unorm": 1.0})
    optimizer = AdamW8bitFast(module.parameters(), lr=1e-3)
    module.weight.grad = torch.ones_like(module.weight)
    calls = []

    def stock_step(self, closure=None):
        calls.append(closure)
        return "stock-route"

    monkeypatch.setattr(bnb.optim.AdamW8bit, "step", stock_step)

    assert optimizer.step() == "stock-route"
    assert calls == [None]
    assert isolated_global_optim_manager.index2config[(0, 0)]["max_unorm"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("source_class", "target_class"),
    [
        (bnb.optim.AdamW8bit, AdamW8bitFast),
        (AdamW8bitFast, bnb.optim.AdamW8bit),
    ],
    ids=["stock-to-fast", "fast-to-stock"],
)
def test_state_dict_can_resume_between_stock_and_fast(source_class, target_class):
    generator = torch.Generator(device="cuda").manual_seed(20260802)
    initial = [
        torch.randn((64, 80), device="cuda", dtype=torch.float32, generator=generator) * 0.01,
        torch.randn((32, 80), device="cuda", dtype=torch.float32, generator=generator) * 0.01,
    ]
    source_params = [torch.nn.Parameter(value.clone()) for value in initial]
    source_groups = [
        {"params": source_params[:1], "lr": 3.5e-4},
        {"params": source_params[1:], "lr": 2.0e-4},
    ]
    kwargs = {"lr": 1e-3, "betas": (0.9, 0.995), "weight_decay": 0.01}
    source = source_class(source_groups, **kwargs)

    for _ in range(3):
        for param in source_params:
            param.grad = torch.randn(param.shape, device="cuda", dtype=param.dtype, generator=generator)
        source.step()

    target_params = [torch.nn.Parameter(param.detach().clone()) for param in source_params]
    target_groups = [
        {"params": target_params[:1], "lr": 3.5e-4},
        {"params": target_params[1:], "lr": 2.0e-4},
    ]
    target = target_class(target_groups, **kwargs)
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    _assert_optimizer_state_equal(source, target, source_params, target_params)

    for source_param, target_param in zip(source_params, target_params):
        grad = torch.randn(source_param.shape, device="cuda", dtype=source_param.dtype, generator=generator)
        source_param.grad = grad.clone()
        target_param.grad = grad.clone()

    source.step()
    target.step()

    for source_param, target_param in zip(source_params, target_params):
        assert torch.equal(source_param, target_param)
    _assert_optimizer_state_equal(source, target, source_params, target_params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_path_logs_device_and_bitsandbytes_version_once(caplog):
    param = torch.nn.Parameter(torch.randn((64, 80), device="cuda", dtype=torch.float32))
    optimizer = AdamW8bitFast([param], lr=1e-3)

    with caplog.at_level(logging.INFO, logger="library.adamw8bit_fast"):
        optimizer.step()
        assert not [record for record in caplog.records if record.name == "library.adamw8bit_fast"]
        for _ in range(2):
            param.grad = torch.randn_like(param)
            optimizer.step()

    messages = [record.getMessage() for record in caplog.records if record.name == "library.adamw8bit_fast"]
    assert len(messages) == 1
    assert "fast path enabled on cuda:0" in messages[0]
    assert f"bitsandbytes {bnb.__version__}" in messages[0]


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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA bitsandbytes kernels are required")
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
def test_thresholds_lrs_and_each_update(dtype):
    from bnb_step_oracle import compare_steps

    compare_steps(AdamW8bitFast, {}, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GradScaler")
def test_prepared_guardian_overflow_and_promote():
    from bnb_step_oracle import compare_prepared_boundaries

    compare_prepared_boundaries(AdamW8bitFast, {})

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an active CUDA parameter")
def test_id_only_override_and_failed_update_do_not_repeat(monkeypatch, isolated_global_optim_manager):
    p = torch.nn.Parameter(torch.ones(4, device="cuda"))
    p.grad = torch.ones_like(p)
    optimizer = AdamW8bitFast([p])
    isolated_global_optim_manager.override_config(p, key="max_unorm", value=1.0)
    assert not isolated_global_optim_manager.index2config
    assert optimizer._fast_path_device()[2] == "parameter override enables max_unorm"
    isolated_global_optim_manager.pid2config.clear()
    original = optimizer.update_step
    calls = []
    def fail(*args):
        calls.append(1)
        original(*args)
        raise RuntimeError("failed after update")
    monkeypatch.setattr(optimizer, "update_step", fail)
    with pytest.raises(RuntimeError, match="failed after update"):
        optimizer.step()
    assert calls == [1] and optimizer.state[p]["step"] == 1


@pytest.mark.parametrize("option", ["block_wise", "percentile_clipping"])
def test_removed_stock_controls_are_rejected(option):
    # bitsandbytes 0.50 removed non-blockwise update and percentile clipping.
    # Forwarding unknown arguments keeps errors visible instead of silently changing the requested algorithm.
    p = torch.nn.Parameter(torch.ones(2))
    for cls in (bnb.optim.AdamW8bit, AdamW8bitFast):
        with pytest.raises(TypeError,match=option):
            cls([p], **{option:False if option=="block_wise" else 50})

@pytest.mark.parametrize("extra,expected",[
    ([],torch.optim.AdamW),
    (["--use_8bit_adam"], bnb.optim.AdamW8bit),
    (["--optimizer_type","AdamW8bit"], bnb.optim.AdamW8bit),
    (["--optimizer_type","aDAMw8bITfAST"], AdamW8bitFast),
    (["--optimizer_type","torch.optim.AdamW"],torch.optim.AdamW),
])
def test_sdxl_parser_factory_groups_and_requested_metadata(extra,expected):
    from library.optimizer import get_optimizer
    from sdxl_train_network import parse_training_args,setup_parser

    args=parse_training_args(setup_parser(),["--network_module","networks.lora",*extra,"--optimizer_args",
                                            "betas=(0.85,0.995)","eps=1e-7","amsgrad=False"])
    params=[torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    groups=[{"params":[p],"lr":lr} for p,lr in zip(params,[3e-4,2e-4,3.5e-4])]
    name,original,optim=get_optimizer(args,groups)
    assert type(optim) is expected and name==expected.__module__+"."+expected.__name__
    assert [g["params"][0] is p for g,p in zip(optim.param_groups,params)]==[True]*3
    assert [g["lr"] for g in optim.param_groups]==[3e-4,2e-4,3.5e-4]
    assert all(g["betas"]==(0.85,0.995) and g["eps"]==1e-7 for g in optim.param_groups)
    assert original=="betas=(0.85, 0.995),eps=1e-07,amsgrad=False"


@pytest.mark.parametrize("scenario,expected", [("paged","paged"),("distributed","distributed"),("subclass","subclass")])
def test_non_fast_inputs_delegate_once(monkeypatch,scenario,expected):
    class SpecialParameter(torch.nn.Parameter):
        pass
    p=(SpecialParameter if scenario=="subclass" else torch.nn.Parameter)(torch.ones(2))
    p.grad=torch.ones_like(p)
    opt=AdamW8bitFast([p],is_paged=scenario=="paged")
    if scenario=="distributed":
        monkeypatch.setattr(torch.distributed,"is_initialized",lambda:True)
        monkeypatch.setattr(torch.distributed,"get_world_size",lambda:2)
    calls=[]
    monkeypatch.setattr(bnb.optim.AdamW8bit,"step",lambda self,closure=None:calls.append(True))
    assert expected in opt._fast_path_device()[2]
    opt.step()
    assert calls==[True]


def test_stock_import_failure_only_affects_selected_optimizer(monkeypatch):
    import builtins
    from library.optimizer import get_optimizer
    from sdxl_train_network import setup_parser
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name=="library.adamw8bit_fast":
            raise ImportError("bitsandbytes unavailable")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",blocked)
    args=setup_parser().parse_args([])
    p=torch.nn.Parameter(torch.ones(2))
    assert type(get_optimizer(args,[p])[2]) is torch.optim.AdamW
    args.optimizer_type="AdamW8bitFast"
    with pytest.raises(ImportError,match="AdamW8bitFast requires bitsandbytes"):
        get_optimizer(args,[p])
