import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from library.grad_norm_guardian import GradNormGuardian, INITIAL_THRESHOLD, WINDOW


def observe_values(mode, values, device="cpu", dtype=torch.float32):
    param = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
    guardian = GradNormGuardian(mode, [param])
    rows = []
    for value in values:
        param.grad = torch.full_like(param, value)
        rows.append(guardian.observe())
    return guardian, rows


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode,expected", [
    ("stable", [False, False, True]),
    ("stable_no_threshoff", [False, False, True]),
    ("gamble", [False, True, True]),
])
def test_before_window_and_nonfinite(mode, expected, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    guardian, rows = observe_values(mode, [INITIAL_THRESHOLD, float("nan"), float("inf")], device)
    assert [r.skipped for r in rows] == expected
    assert all(r.threshold == INITIAL_THRESHOLD for r in rows)
    assert guardian.skipped_steps == sum(expected)
    assert len(guardian.history) == (3 if mode == "stable" else 1)


def test_threshold_uses_current_observation_and_population_std():
    values = [1.] * (WINDOW - 1) + [3.]
    _, rows = observe_values("stable_no_threshoff", values)
    assert rows[-2].threshold == INITIAL_THRESHOLD
    assert rows[-1].threshold == pytest.approx(np.mean(values) + 2.5 * np.std(values))
    assert rows[-1].skipped


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_stable_disables_poisoned_threshold_for_exactly_one_window(invalid):
    _, rows = observe_values("stable", [invalid] + [2.] * WINDOW)
    assert math.isnan(rows[WINDOW - 1].threshold)
    assert rows[WINDOW].threshold == 2
    assert not rows[WINDOW].skipped


def test_cap_gamble_and_empty_gradients():
    for mode, expected in [("stable_no_threshoff", INITIAL_THRESHOLD), ("gamble", INITIAL_THRESHOLD * 2)]:
        _, rows = observe_values(mode, [INITIAL_THRESHOLD * 2] * WINDOW)
        assert rows[-1].threshold == expected
        assert rows[-1].skipped == (mode != "gamble")
    assert GradNormGuardian("stable", []).observe().norm == 0


def test_foreach_fallback_is_per_run_and_only_for_unsupported_operations():
    p = torch.nn.Parameter(torch.zeros(2))
    q = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.tensor([3., 4.])
    guarded = GradNormGuardian("stable", [p, q])
    with patch("torch.nn.utils.get_total_norm", side_effect=RuntimeError("not supported")) as fn:
        assert guarded.observe().norm == 5
        q.grad = torch.tensor([12.])
        assert guarded.observe().norm == 13
        assert fn.call_count == 1
    with patch("torch.nn.utils.get_total_norm", side_effect=RuntimeError("unrelated fault")):
        with pytest.raises(RuntimeError, match="unrelated"):
            GradNormGuardian("stable", [p]).observe()


def test_half_precision_uses_original_grad_dtype_for_squares():
    _, rows = observe_values("stable_no_threshoff", [256], dtype=torch.float16)
    assert math.isinf(rows[0].norm)
