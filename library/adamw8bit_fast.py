from __future__ import annotations

from typing import Callable, Optional

import torch

try:
    import bitsandbytes as bnb
except ImportError as exc:  # pragma: no cover - handled by optimizer selection
    raise ImportError("AdamW8bitFast requires bitsandbytes") from exc


class AdamW8bitFast(bnb.optim.AdamW8bit):
    """AdamW8bit with one CUDA synchronization per optimizer step.

    bitsandbytes updates parameters one at a time and synchronizes the whole
    device after every parameter. LoRA commonly has many small parameter
    tensors, so those synchronizations dominate the optimizer step. CUDA
    stream ordering already keeps the queued updates in order; synchronizing
    once after the final update preserves the synchronous ``step()`` contract.

    The fast path is intentionally limited to ordinary parameters on one CUDA
    device. Paged, distributed, tensor-subclass, sparse, multi-device, CPU and
    closure-based calls use the unmodified bitsandbytes implementation.
    """

    def _fast_path_device(self) -> tuple[bool, Optional[torch.device]]:
        if self.is_paged:
            return False, None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                return False, None

        active_device: Optional[torch.device] = None
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if type(param) is not torch.nn.Parameter or type(grad) is not torch.Tensor:
                    return False, None
                if param.device.type != "cuda" or grad.device != param.device:
                    return False, None
                if param.layout != torch.strided or grad.layout != torch.strided or grad.is_sparse:
                    return False, None
                if active_device is None:
                    active_device = param.device
                elif param.device != active_device:
                    return False, None

        return True, active_device

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        # A closure may replace gradients, so let bitsandbytes retain complete
        # control of closure evaluation and route selection in that uncommon
        # case.
        if closure is not None:
            return super().step(closure)

        can_use_fast_path, device = self._fast_path_device()
        if not can_use_fast_path:
            return super().step()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        updated = False
        for group_index, group in enumerate(self.param_groups):
            for param_index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    self.init_state(group, param, group_index, param_index)

                self.prefetch_state(param)
                self.update_step(group, param, group_index, param_index)
                updated = True

        if updated:
            # Stock bitsandbytes synchronizes after every parameter. One final
            # device sync still surfaces asynchronous CUDA errors before step()
            # returns without introducing per-parameter CPU/GPU round trips.
            torch.cuda.synchronize(device=device)

        return None
