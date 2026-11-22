from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from library import avg_training
from library.avg_training import EpochAveraging, save_final_raw
from library.promote_bank import BankItem


def setup_run(*, mode="promote", **overrides):
    values = dict(avg_window=2, avg_mode="uniform", avg_cp_mode=mode, avg_shadow_bank_size=2,
                  avg_begin=0, avg_shadow_margin=0.0, avg_shadow_patience=1,
                  avg_reset_stats=False, avg_save_final_raw=True)
    values.update(overrides)
    controller = EpochAveraging(values)
    net = torch.nn.Module()
    net.register_parameter("lora_w", torch.nn.Parameter(torch.tensor([1.])))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    return controller, net, optimizer


def store(pool, count):
    for i in range(count):
        scalar = torch.ones(1, 1, 1, 1)
        pool.add(BankItem.capture({"loss_weights":torch.ones(1), "class_tokens":["a"], "image_keys":[str(i)]},
                                  scalar, scalar, torch.tensor([i]), None, None))


def finish(run, epoch, total, value, score=None):
    controller, net, optimizer = run
    net.lora_w.data.fill_(value)
    return controller.finish_epoch(epoch, total, net, optimizer, score, torch.device("cpu"), 1000)


@pytest.mark.parametrize("mode", ["promote", "live"])
def test_raw_on_disk_precedes_average_final_raw_and_next_epoch(tmp_path, mode):
    run = setup_run(mode=mode)
    c, model, optimizer = run
    if c.bank:
        store(c.bank, 32)
    observed_rng = []
    def score(items):
        assert model.training and not torch.is_grad_enabled()
        observed_rng.append(torch.rand(3))
        return model.lora_w.item()
    torch.manual_seed(87)
    before = torch.get_rng_state().clone()
    for epoch, weight in enumerate((1,3,5), 1):
        # This is the value H's scheduled epoch saver and sampler see before calling finish_epoch.
        rows = finish(run, epoch, 3, weight, score)
        assert model.lora_w.item() == (1,2,4)[epoch-1]
        if mode == "promote":
            assert rows[0]["status"] == ("waiting_window" if epoch == 1 else "scored")
            assert [r["window_epoch"] for r in rows[1:]] == list(range(max(1,epoch-1),epoch+1))
            assert rows[0]["promote_applied"] == (epoch>1)
        else:
            assert rows == []
    assert torch.equal(torch.get_rng_state(), before)
    if observed_rng:
        assert len(observed_rng)==4 and all(torch.equal(v, observed_rng[0]) for v in observed_rng)
    assert c.final_raw["lora_w"].item()==5
    raw_name = tmp_path/"model_final_raw.safetensors"
    meta={"ss_steps":"39","ss_epoch":"3","ss_session_id":"99"}
    save_final_raw(raw_name,c.final_raw,torch.float16,meta)
    with safe_open(raw_name,framework="pt") as opened:
        assert opened.get_tensor("lora_w").dtype==torch.float16
        assert opened.get_tensor("lora_w").item()==5
        metadata = opened.metadata()
    from library.model_io import precalculate_safetensors_hashes
    hashes = precalculate_safetensors_hashes({"lora_w":torch.tensor([5.],dtype=torch.float16)},metadata)
    assert (metadata["sshs_model_hash"],metadata["sshs_legacy_hash"])==hashes
    assert metadata["ss_avg_shadow_variant"]=="final_raw" and "ss_avg_shadow_variant" not in meta
    assert model.lora_w.item()==4


def test_statuses_margin_equality_patience_and_reset():
    run = setup_run(avg_begin=0.25,avg_shadow_patience=2,avg_shadow_margin=0.25,avg_reset_stats=True)
    c, model, optimizer = run
    state = optimizer.state[model.lora_w]
    state.update(step=torch.tensor(4.),exp_avg=torch.ones(1),exp_avg_sq=torch.ones(1),
                 exp_avg_max=torch.ones(1),state1=torch.ones(1),state2=torch.ones(1))
    def fixed_scores(a,b):
        values=iter([a,b])
        return lambda items:next(values)
    assert finish(run,1,8,1)[0]["status"]=="before_avg_begin"
    store(c.bank,5)
    assert finish(run,2,8,2)[0]["status"]=="building_pool"
    store(c.bank,27)
    for epoch,scores,streak in [(3,(4,3),0),(4,(4,2),1),(5,(4,float("nan")),0),(6,(4,2),1),(7,(4,2),2)]:
        row=finish(run,epoch,8,epoch,fixed_scores(*scores))[0]
        assert row["win_streak"]==streak
        assert row["optimizer_reset_applied"] == row["promote_applied"] == (epoch==7)
        assert model.lora_w.item()==(6.5 if epoch==7 else epoch)
    assert state["step"].item()==4
    assert all(state[k].item()==0 for k in ("exp_avg","exp_avg_sq","exp_avg_max"))
    assert all(state[k].item()==1 for k in ("state1","state2"))
    # State with no step, as in the legacy implementation.
    optimizer.state[object()]={"exp_avg":None}
    avg_training.reset_optimizer_moments(optimizer)
    assert any(entry["step"]==0 for entry in optimizer.state.values())


@pytest.mark.parametrize("failure", ["score", "candidate_load", "restore", "reset"])
def test_failure_propagates_without_publishing_success(monkeypatch, failure):
    run = setup_run(avg_reset_stats=(failure=="reset"))
    c, model, optimizer = run
    store(c.bank,2)  # final epoch alone is allowed to finalize this smaller pool
    assert finish(run,1,2,1)[0]["status"]=="building_pool"
    original=avg_training.apply_lora_snapshot
    def apply(net, state):
        weight=state["lora_w"].item()
        if (failure=="candidate_load" and weight==2) or (failure=="restore" and weight==3):
            raise RuntimeError(failure)
        return original(net,state)
    monkeypatch.setattr(avg_training,"apply_lora_snapshot",apply)
    def score(items):
        torch.rand(4)
        weight=model.lora_w.item()
        if failure=="score" and weight==2:
            raise RuntimeError("score")
        return weight
    if failure=="reset":
        def reset(_):
            raise RuntimeError("reset")
        monkeypatch.setattr(avg_training,"reset_optimizer_moments",reset)
    initial_rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match=failure):
        finish(run,2,2,3,score)
    if failure not in {"restore","reset"}:
        assert model.lora_w.item()==3
    assert torch.equal(initial_rng, torch.get_rng_state())


def test_failed_raw_write_keeps_previous_artifact(tmp_path,monkeypatch):
    import safetensors.torch
    path=tmp_path/"run_final_raw.safetensors"
    path.write_bytes(b"previous")
    def fail(tensors,name,metadata):
        raise OSError("disk full")
    monkeypatch.setattr(safetensors.torch,"save_file",fail)
    with pytest.raises(OSError,match="disk full"):
        save_final_raw(path,{"lora_a":torch.ones(1)},torch.float16,{})
    assert path.read_bytes()==b"previous"
    assert list(tmp_path.iterdir())==[path]
