"""Gradient-norm decisions for SDXL LoRA, before AMP unscale and clipping."""
from collections import deque
from dataclasses import dataclass
import logging
import math

import numpy as np
import torch


logger = logging.getLogger(__name__)
WINDOW = 200
INITIAL_THRESHOLD = 200_000.0


@dataclass(frozen=True)
class GuardianPolicy:
    cap: float | None
    keep_nonfinite: bool
    reject_nonfinite: bool


PRESETS = {
    "stable": GuardianPolicy(INITIAL_THRESHOLD, True, False),
    "stable_no_threshoff": GuardianPolicy(INITIAL_THRESHOLD, False, False),
    "gamble": GuardianPolicy(None, False, True),
}


@dataclass(frozen=True)
class GradientObservation:
    norm: float
    threshold: float
    skipped: bool


def _legacy_norm(grads):
    if not grads:
        return torch.tensor(0.0)
    squared = torch.tensor(0.0, device=grads[0].device)
    for grad in grads:
        squared += (grad * grad).sum()
    return squared.sqrt()


def _supports_foreach(grads):
    first = grads[0]
    return (
        first.dtype == torch.float32 and first.layout == torch.strided and first.device.type in ("cuda", "cpu")
        and all(g.dtype == first.dtype and g.layout == first.layout and g.device == first.device for g in grads[1:])
    )


class GradNormGuardian:
    def __init__(self, mode, parameters):
        self.policy = PRESETS[mode]
        self.parameters = tuple(parameters)
        self.history = deque(maxlen=WINDOW)
        self.skipped_steps = 0
        self._foreach_available = True

    def _norm(self, grads):
        if not grads or not self._foreach_available or not _supports_foreach(grads):
            return _legacy_norm(grads)
        try:
            total_norm = getattr(torch.nn.utils, "get_total_norm", None)
            if total_norm is not None:
                norm = total_norm(grads, norm_type=2.0, error_if_nonfinite=False, foreach=True)
            else:
                foreach_norm = getattr(torch, "_foreach_norm", None)
                if foreach_norm is None:
                    self._foreach_available = False
                    return _legacy_norm(grads)
                norm = torch.linalg.vector_norm(torch.stack(foreach_norm(grads, 2.0)), 2.0)
            # Match the finite/underflow/overflow classification of the legacy sum of squares.
            return (norm * norm).sqrt()
        except (TypeError, NotImplementedError):
            self._foreach_available = False
        except RuntimeError as error:
            if not any(s in str(error).lower() for s in ("not implemented", "not supported", "unsupported",
                                                         "can't use the foreach", "cannot use the foreach")):
                raise
            self._foreach_available = False
        logger.info("GradNorm Guardian: using the legacy sum of squares for this run")
        return _legacy_norm(grads)

    @torch.no_grad()
    def observe(self):
        grads = [p.grad.detach() for p in self.parameters if p.grad is not None]
        norm = self._norm(grads).item()
        finite = math.isfinite(norm)
        if finite or self.policy.keep_nonfinite:
            self.history.append(norm)
        threshold = INITIAL_THRESHOLD
        if len(self.history) == WINDOW:
            with np.errstate(invalid="ignore"):
                threshold = float(np.mean(self.history) + 2.5 * np.std(self.history))
            if self.policy.cap is not None and threshold > self.policy.cap:
                threshold = self.policy.cap
        skipped = (not finite and self.policy.reject_nonfinite) or norm > threshold
        if skipped:
            self.skipped_steps += 1
        return GradientObservation(norm, threshold, skipped)
