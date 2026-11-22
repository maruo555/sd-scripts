import math

import pytest

from library.lora_run_schema import HEADERS
from tools.lora_run_reader import rank_observations
from tools.lora_diagnostic_analysis import analyze_gradient, analyze_group, analyze_promote
from tools.lora_diagnostic_rank import analyze_rank
from tools.lora_diagnostic_dq import analyze_dq


def sample(kind, **values):
    row = dict.fromkeys(HEADERS[kind])
    row.update(values)
    return row


def test_threshold_unknown_disabled_infinite_and_unequal_epochs():
    rows = [sample("gradient", epoch=e, batch_index=b, grad_norm=g, threshold=t, loss=lo)
            for (e,b,g,t,lo) in [(1,0,1,float("nan"),0.),(1,1,2,float("inf"),3.),
                                  (1,3,1,None,None),(2,0,1,1.,2.),(2,1,2,1.,4.)]]
    data = analyze_gradient(rows, 2)
    assert data["summary"]["threshold_exceeded_ratio"] == .5
    assert data["summary"]["thresh_off_ratio"] == .25 and data["summary"]["threshold_unknown"] == 1
    assert data["cuts"] == [2] and data["loss_ma"] == [0., 1.5, None, 2.5, 3.]
    assert data["markers"] == [{"x":1,"label":"E1"},{"x":4,"label":"E2"}]


def test_scopes_pool_by_numel_and_use_unique_auto_checks():
    def row(scope, n, x, err, clip):
        return sample("dq", epoch=1, attempt_step=100, bits=8, scope=scope, target="delta", numel=n,
                      input_rms=x, quant_error_rms_raw=err, clip_rate_raw=clip, auto_state_step=50)
    def event(step, clip=0., warm=False, reason="in_band"):
        return sample("dq_auto", event_type="init" if step == 0 else "check", attempt_step=step, clip_rate_raw=clip,
                      clip_rate_ema=.05, warmup_active=warm, auto_reason=reason, auto_applied=step==100)
    samples=[row("unet",9,2,1,.02),row("te",1,1,.1,.01)]
    checks=[event(0),event(50,warm=True),event(75),event(100,reason="high"),event(125,clip=None),event(150,reason="low")]
    settings={"dq":{"log_scope":"both","auto":{"scope":"unet"}}}
    data=analyze_dq(samples,checks,settings)
    obs=data["pooled"][0]
    assert obs["clip_rate_raw"]==pytest.approx(.019)
    assert obs["quant_error_rms_raw"]==pytest.approx(math.sqrt(.901))
    assert obs["quant_error_ratio_raw"]==pytest.approx(math.sqrt(.901)/(math.sqrt(3.7)+1e-12))
    assert data["summary"]["check_count"]==5 and data["summary"]["applied_count"]==1
    assert data["summary"]["in_band_ratio"] == pytest.approx(1/3)
    missing=analyze_dq(samples[:1],[],settings)
    assert not missing["pooled"] and missing["scope_rows"]["unet"][0]["clip_rate_raw"]==.02
    assert any("reference" in i["reason"] for i in missing["issues"])


def test_last_complete_rank_and_active_population():
    class Reader:
        def rank_observations(self, rows):
            return rank_observations(rows, {"unet":["a","b","c"]})
    def row(name, step, sat, top, energy):
        return sample("rank",epoch=step//100, attempt_step=step, scope="unet", module=name, rank_dim=4,
                      rank_sat=sat, rank_top1=top, rank_energy=energy)
    rows=[row("a",100,0,0,0),row("b",200,.1,.2,3),row("c",100,1,.4,3),row("b",100,.5,.8,1)]
    data=analyze_rank(Reader(), rows)
    assert len(data["incomplete"])==1
    last=data["rows"][-1]
    assert [last[k] for k in ("RankSatWMean","RankSatP50","RankSatP95","RankSatMax","RankTop1P95","RankEnergySum")] == pytest.approx([.875,.75,.975,1.,.78,4.])


def test_group_and_promote_decision_units():
    group=[sample("group_loss",attempt_step=s,epoch=1,group="A",loss=l,group_ema_loss=ma)
           for s,l,ma in ((1,2.,2.),(2,4.,2.04))]
    assert [r["group_ema_loss"] for r in analyze_group(group)["rows"]] == [2.,2.04]
    decisions=[sample("promote",row_type="decision",epoch=e,status="scored",raw_proxy_loss=1.,
                      selected_proxy_loss=.99,margin_ok=True,promote_applied=e==2,window_count=1) for e in (1,2)]
    windows=[sample("promote",row_type="window_epoch",epoch=2,window_position=1,window_epoch=2)]
    result=analyze_promote(decisions+windows)
    assert (result["scored"],result["margin_wins"],result["applied"]) == (2,2,1)
    assert [r["difference_percent"] for r in result["rows"]] == pytest.approx([-1.,-1.])
    assert result["rows"][0]["window"] is None and result["rows"][1]["window"] == [2]
