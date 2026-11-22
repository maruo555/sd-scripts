"""DQ range updates happen after the measured forward of each accepted auto observation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from library.dq_auto import CLIP_BANDS, DQAutoController
from library.dq_training import DQTraining
from library.maruo_training_config import resolve_training_config
from library.rounding_util import quantize_delta


def config(**changes):
    values = dict(network_module="networks.lora", dq_delta_auto_range_mul=True, dq_delta_auto_every=1, dq_delta_bits=8)
    values.update(changes)
    return resolve_training_config(SimpleNamespace(**values))


def controller(preset="clip_rate_high", warmup=False, **changes):
    return DQAutoController(config(dq_delta_auto_preset=preset, dq_delta_auto_warmup=warmup, **changes).values)


def check(auto, clip, error=None, progress=0.3, reset=False):
    return auto.check(epoch=1, step=auto.last_step + 1, progress=progress, bits=8, bits_changed=reset,
                      metrics={"clip_rate_raw": clip, "quant_error_ratio_raw": error})


@pytest.mark.parametrize("preset,band", [
    ("default", "default"), ("clip_rate_high", "high"), ("clip_rate_high_narrow", "high_narrow"),
    ("clip_rate_mid", "mid"), ("clip_rate_low", "low"), ("clip_rate_low_auto", "low"),
])
def test_init_and_all_bands(preset, band):
    auto = controller(preset, warmup=True, dq_delta_auto_init_range_mul_from_band=True)
    assert auto.band == band and auto.warmup_remaining == 40
    low, high = CLIP_BANDS[band]
    target = (low + high) / 2
    assert auto.settings["init"] == {
        "applied": True, "clip_target": target,
        "mul": 2**0.5 * torch.erfinv(torch.tensor(1 - target, dtype=torch.float32)).item(),
    }
    assert auto.init_row["scope"] == "unet" and auto.init_row["event_type"] == "init"
    assert auto.init_row["warmup_remaining_after"] == 40 and auto.init_row["active_clip_band"] == band
    assert auto.init_row["bits"] is None and auto.init_row["epoch"] is None
    if band == "high":
        assert auto.mul == pytest.approx(2.87816, abs=1e-5)


@pytest.mark.parametrize("raw,want,reason", [(0.006, 3.03, "clip_high"), (0.002, 2.985, "clip_low"),
                                            (0.003, 3., "in_band"), (0.005, 3., "in_band")])
def test_raw_and_ema_thresholds(raw, want, reason):
    auto = controller(dq_delta_auto_use_raw=True)
    row = check(auto, raw)
    assert row["auto_reason"] == reason and row["common_mul_before"] == 3.
    assert row["common_mul_after"] == pytest.approx(want)
    assert row["auto_applied"] == (want != 3.)
    assert row["low_auto_bad"] is None and row["clip_rate_ema"] == raw


def test_clamp_disagreement_and_ema_only():
    auto = controller(dq_delta_auto_use_raw=True, dq_delta_range_mul=6.)
    first = check(auto, 0.01)
    second = check(auto, 0.001)
    assert first["auto_reason"] == "clip_high" and not first["auto_applied"]
    assert second["auto_reason"] == "in_band" and not second["auto_applied"]
    auto = controller(dq_delta_auto_use_raw=False)
    check(auto, 0.01)
    assert check(auto, 0.001)["auto_reason"] == "clip_high"


def test_warmup_including_last_observation_and_missing_reset():
    auto = controller(warmup=True)
    seq = [check(auto, 0.004), check(auto, 0.004), check(auto, None), check(auto, 0.004)]
    assert [row["warmup_remaining_after"] for row in seq] == [39, 38, 38, 0]
    assert all(row["warmup_active"] and not row["auto_applied"] for row in seq)
    assert not check(auto, 0.05)["warmup_active"]
    before = auto.mul
    missing = check(auto, None, reset=True)
    assert missing["clip_rate_raw"] is None and missing["warmup_remaining_after"] == 0 and auto.mul == before
    reset = check(auto, 0, reset=True)
    assert reset["warmup_active"] and reset["warmup_remaining_after"] == 39
    assert reset["clip_rate_ema"] == 0 and auto.mul == before


def test_low_state_escape_same_check_freeze_and_later_bits():
    auto = controller("clip_rate_low_auto")
    assert check(auto, 0.0015, 0.5, progress=0.24)["low_auto_reason"] == "min_progress"
    assert check(auto, 0.0015, 0.5, progress=0.25)["low_auto_bad_streak"] == 1
    check(auto, 0.0015, 0.5, progress=0.26)
    escaped = check(auto, 0.0015, 0.5, progress=0.27)
    assert (escaped["active_clip_band"], escaped["low_auto_decision"], escaped["auto_reason"]) == ("mid", "escape_to_mid", "clip_low")
    assert not escaped["low_auto_can_escape"] and escaped["qerr_per_clip"] == pytest.approx(0.5 / 0.0015)
    locked = check(auto, 0.003, 0.1, progress=0.55, reset=True)
    assert locked["low_auto_state"] == "mid_lock" and locked["low_auto_bad_streak"] == 0
    assert locked["auto_quant_error_ratio_ema"] == 0.1 and locked["active_clip_band"] == "mid"
    fresh = controller("clip_rate_low_auto")
    assert check(fresh, 0, progress=0.25)["low_auto_reason"] == "insufficient_qerr_stats"
    frozen = check(fresh, 0, 1.0, progress=0.55)
    assert frozen["low_auto_state"] == "frozen" and frozen["low_auto_bad_streak"] == 1
    assert frozen["auto_reason"] == "clip_low" and frozen["qerr_per_clip"] == 1000


def test_low_warmup_phase_is_not_persisted():
    auto = controller("clip_rate_low_auto", warmup=True, dq_delta_auto_warmup_updates=1)
    warm = check(auto, 0, 0.5)
    assert warm["low_auto_phase"] == "warmup" and warm["low_auto_phase_after"] == "pre_min_progress"
    active = check(auto, 0, 0.5)
    assert active["low_auto_phase"] == active["low_auto_phase_after"] == "active"


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scope,raw", [("unet", 2/1000), ("both", 4/1100)])
def test_weighted_scopes_logging_and_skipped_attempts(device, scope, raw):
    cfg = config(dq_delta_auto_scope=scope, dq_delta_auto_warmup=False, dq_delta_log=True, dq_delta_log_every=2,
                 dq_delta_bits_sched="0.5:4")
    mod = SimpleNamespace(lora_name="lora_unet_in_attn2_to_q")
    net = SimpleNamespace(unet_loras=[mod], text_encoder_loras=[])
    dq = DQTraining(cfg, net, total_steps=4, lr_warmup_steps=0, num_processes=1)

    def packed(scope, n, clips):
        nums = [n, clips] + ([1, 1, 1] if dq.full_stats else [])
        dq.add_packed(scope, torch.tensor(nums, dtype=torch.float32, device=device))

    dq.begin_attempt(0)
    packed("unet", 1000, 2)
    assert dq.finish_attempt(1, epoch=1, guardian_skipped=True) is None
    assert dq.auto_controller.last_step == 0 and dq.common_mul == 3.
    dq.begin_attempt(1)
    packed("unet", 1000, 2)
    packed("te", 100, 2)
    # AMP-only overflow reaches this same accepted-observation path.
    with patch.object(dq, "_metrics", wraps=dq._metrics) as reads:
        second = dq.finish_attempt(2, epoch=1, guardian_skipped=False)
    assert len(reads.call_args_list) == 3  # combined log summary plus two components
    assert second.auto_row["clip_rate_raw"] == pytest.approx(raw)
    assert second.auto_row["bits"] == 8 and second.auto_row["epoch"] == 1
    assert {r["auto_state_step"] for r in second.log_rows} == {2}
    assert all(r["common_mul_after"] == second.auto_row["common_mul_after"] for r in second.log_rows)
    dq.begin_attempt(2)  # bits change; no data: check is recorded, reset still pending
    third = dq.finish_attempt(3, epoch=2, guardian_skipped=False)
    assert third.auto_row["bits"] == 4 and third.auto_row["clip_rate_raw"] is None
    assert dq.bits_changed_since_auto and dq.auto_controller.clip_rate_ema == pytest.approx(raw)
    dq.begin_attempt(3)
    packed("unet", 20, 2)
    fourth = dq.finish_attempt(4, epoch=2, guardian_skipped=False)
    assert fourth.auto_row["clip_rate_ema"] == pytest.approx(0.1) and not dq.bits_changed_since_auto


@pytest.mark.parametrize("device", DEVICES)
def test_forward_sees_updated_mul_only_after_check(device):
    cfg = config(dq_delta_auto_warmup=False, dq_delta_auto_preset="clip_rate_high")
    mod = SimpleNamespace(lora_name="lora_unet_in_attn2_to_q")
    dq = DQTraining(cfg, SimpleNamespace(unet_loras=[mod], text_encoder_loras=[]), total_steps=4,
                    lr_warmup_steps=0, num_processes=1)
    delta = torch.tensor([1., 1., 1., 1., 1., 9., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                         dtype=torch.float16, device=device).reshape(1, 16, 1)
    def compare_forward(mul):
        torch.manual_seed(39)
        with torch.random.fork_rng(devices=[delta.device] if delta.is_cuda else []):
            expected = quantize_delta(delta, bits=8, range_mul=mul)
        torch.testing.assert_close(mod.delta_quantizer(delta), expected, rtol=0, atol=0)
    dq.begin_attempt(0)
    compare_forward(3.)
    first = dq.finish_attempt(1, epoch=1, guardian_skipped=False)
    assert first.auto_row["auto_reason"] == "clip_high" and not first.log_rows
    assert first.auto_row["common_mul_before"] == 3. and first.auto_row["common_mul_after"] == pytest.approx(3.03)
    # Sample/proxy forward shares the new numerical setting, but may not add training statistics.
    compare_forward(3.03)
    assert not dq.totals and dq.auto_controller.last_step == 1
    dq.begin_attempt(1)
    compare_forward(3.03)
    second = dq.finish_attempt(2, epoch=1, guardian_skipped=False)
    assert second.auto_row["common_mul_before"] == first.auto_row["common_mul_after"]
