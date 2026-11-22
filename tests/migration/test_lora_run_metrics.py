from pathlib import Path
import math
import json

import pytest
from tools import lora_run_metrics as metrics


def test_approved_68_fields_have_named_reconstruction():
    fixture = Path(__file__).with_name("fixtures") / "csv-contract.json"
    contract = json.loads(fixture.read_text(encoding="utf-8"))
    approved = {stream: set(spec["derived"]) for stream, spec in contract.items()}
    actual = {
        "gradient":metrics.gradient_fields({}), "group_loss":metrics.group_loss_fields({}),
        "dq":metrics.dq_fields({},None,{}), "dq_auto":metrics.dq_auto_fields({},None),
        "rank":metrics.rank_fields([],["unverified"]), "promote":metrics.promote_fields({},None,None),
    }
    assert {stream: set(fields) for stream, fields in actual.items()} == approved


def test_dq_check_and_state_keep_scope_and_missing_distinct():
    cfg = {"dq":{"stat":"rms","rounding_mode":"stoch","granularity":"channel",
                 "auto":{"enabled":True,"scope":"unet","low_auto":{"enabled":True,"min_progress":0.25},
                         "init":{"applied":True,"mul":2.878,"clip_target":0.004},"qerr_per_clip_floor":0.001}}}
    check = dict(event_type="check",attempt_step=10,scope="unet",warmup_active=True,warmup_remaining_after=0,
                 common_mul_before=2.9,common_mul_after=3.0,auto_applied=True,auto_reason="warmup",
                 low_auto_bad=False,low_auto_bad_streak=0,low_auto_state="observe",low_auto_phase="warmup",
                 low_auto_phase_after="pre_min_progress",low_auto_can_escape=False,bits=8)
    row = dict(attempt_step=10,auto_state_step=10,scope="te",bits=8,common_mul_after=3.0,input_shape="2x64x320",quant_range=25.4)
    events = {10:check}
    first = metrics.dq_fields(row,cfg,events)
    assert first["ClipRateLowAutoBad"] is False and first["WarmupActive"] is True and first["ClipRateLowAutoPhase"]=="warmup"
    assert first["RangeMulBefore"]==2.9 and first["ScaleMean"]==pytest.approx(0.2)
    assert first["Shape"]==(2,64,320) and first["Qmax"]==127 and first["AutoInitMulValue"]==2.878
    next_obs = metrics.dq_fields({**row,"attempt_step":11},cfg,events)
    assert next_obs["ClipRateLowAutoBad"] is None and next_obs["WarmupActive"] is False
    assert next_obs["ClipRateLowAutoPhase"]=="pre_min_progress" and next_obs["RangeMulBefore"]==3
    missing = metrics.dq_fields(row,cfg,{10:None})
    assert missing["AutoApplied"] is None and missing["RangeMulBefore"] is None and missing["RangeMulAfter"]==3
    disabled = metrics.dq_fields(row,{"dq":{"auto":{"enabled":False}}},{})
    assert disabled["AutoApplied"] is False and disabled["WarmupRemain"]==0
    auto = metrics.dq_auto_fields(check,cfg)
    assert auto["WarmupRemain"]==0 and auto["RangeMul"]==3 and auto["ClipRateLowAutoCanEscape"] is False
    assert metrics.dq_auto_fields({"event_type":"init"},cfg)["RangeMul"] is None


def test_gradient_group_and_complete_rank_values():
    assert metrics.gradient_fields({"threshold":math.nan})=={"ThreshOff":True}
    for threshold in (0.0, math.inf, -math.inf):
        assert metrics.gradient_fields({"threshold":threshold})["ThreshOff"] is False
    assert metrics.gradient_fields({})["ThreshOff"] is None
    assert metrics.group_loss_fields({"group_observation_count":21})["count_group"]==21
    rows=[dict(rank_sat=a,rank_top1=b,rank_energy=e) for a,b,e in [(0.0,0.0,0.0),(0.5,0.8,1.0),(1.0,0.4,3.0)]]
    result=metrics.rank_fields(rows,[])
    assert tuple(result.values())==pytest.approx((0.875,0.75,0.975,1.0,0.78,4.0))
    assert all(v is None for v in metrics.rank_fields(rows,["missing module"]).values())
    assert metrics.rank_fields([{"rank_sat":0,"rank_top1":0,"rank_energy":0}],[])["RankSatWMean"]==0


def test_promote_derives_from_decision_and_verified_window_only():
    cfg={"promote":{"mode":"ema","pick":"fixed","window_size":4,"pool_target_size":48,"patience":2},"training":{"total_epochs":40}}
    row=dict(epoch=3,status="scored",selected_candidate_mode="ema",raw_proxy_loss=1.0,selected_proxy_loss=0.99,
             margin_ok=True,win_streak=1,promote_applied=False,bank_timestep_low_count=0,bank_timestep_mid_count=2,bank_timestep_high_count=1)
    result=metrics.promote_fields(row,[1,2,3],cfg)
    assert result["delta_abs"]==pytest.approx(-0.01) and result["delta_pct"]==pytest.approx(-1)
    assert result["winner"]=="center" and result["virtual_would_promote"] is False and result["next_epoch_source"]=="raw"
    assert result["window_epochs"]==[1,2,3] and result["bank_timestep_bins"]=={"low":0,"mid":2,"high":1}
    partial=metrics.promote_fields({**row,"raw_proxy_loss":0},None,None)
    assert partial["window_epochs"] is None and partial["delta_pct"] is None and partial["winner"]=="center"
