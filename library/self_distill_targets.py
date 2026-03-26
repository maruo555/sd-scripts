from typing import Dict, List, Optional

import torch

from library import self_distill_cache


def sample_target_step_indices(
    num_inference_steps: int,
    num_target_timesteps: int,
    sampling_mode: str,
    custom_timesteps: Optional[List[int]] = None,
) -> List[int]:
    if num_target_timesteps <= 0:
        raise ValueError("num_target_timesteps must be positive.")

    max_index = num_inference_steps - 1
    if max_index < 0:
        raise ValueError("num_inference_steps must be positive.")

    if sampling_mode == "custom":
        if not custom_timesteps:
            raise ValueError("custom timestep sampling requires custom_timesteps.")
        result = sorted({int(v) for v in custom_timesteps if 0 <= int(v) <= max_index})
        if not result:
            raise ValueError("custom timesteps are empty after bounds check.")
        return result[:num_target_timesteps]

    if num_target_timesteps >= num_inference_steps:
        return list(range(num_inference_steps))

    if sampling_mode == "late_bias":
        start = max(0, int(num_inference_steps * 0.35))
        positions = torch.linspace(start, max_index, steps=num_target_timesteps)
    else:
        positions = torch.linspace(0, max_index, steps=num_target_timesteps)
    return sorted({int(round(pos.item())) for pos in positions})


def build_teacher_rollout_targets(
    unet,
    scheduler,
    conditioning: Dict[str, torch.Tensor],
    initial_latents: torch.Tensor,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    target_type: str,
    capture_step_indices: List[int],
):
    capture_set = set(capture_step_indices)
    latents = initial_latents
    captures = []

    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    for step_index, timestep in enumerate(scheduler.timesteps):
        guided = self_distill_cache.unet_predict_cfg(unet, scheduler, conditioning, latents, timestep, height, width, guidance_scale)
        if step_index in capture_set:
            target = self_distill_cache.prediction_to_target(guided, target_type, scheduler, latents, timestep)
            captures.append(
                {
                    "step_index": step_index,
                    "timestep": int(timestep.item()),
                    "x_t": latents.detach().clone(),
                    "target": target.detach().clone(),
                }
            )
        latents = scheduler.step(guided, timestep, latents).prev_sample

    captures.sort(key=lambda item: item["step_index"])
    return latents, captures
