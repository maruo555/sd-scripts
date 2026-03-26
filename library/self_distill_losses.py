from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from library import lbw_profile, self_distill_cache


def _gaussian_residual(x: torch.Tensor, kernel: int) -> torch.Tensor:
    blurred = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x - blurred


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def _dog(x: torch.Tensor) -> torch.Tensor:
    low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    high = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    return low - high


def high_pass(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode == "dog":
        return _dog(x)
    if mode == "laplacian":
        return _laplacian(x)
    if mode == "gaussian_residual":
        return _gaussian_residual(x, kernel=5)
    raise ValueError(f"Unsupported high-pass mode: {mode}")


def low_pass(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode == "avg":
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    if mode == "gaussian":
        return F.avg_pool2d(x, kernel_size=9, stride=1, padding=4)
    if mode == "identity":
        return x
    raise ValueError(f"Unsupported low-pass mode: {mode}")


def _variant_weight(args, variant_type: str) -> float:
    weights = self_distill_cache.parse_mapping_arg(getattr(args, "per_variant_loss_weight", None))
    return float(weights.get(variant_type, 1.0))


def _anchor_group_weight(args, parameter_name: str) -> float:
    weights = self_distill_cache.parse_mapping_arg(getattr(args, "per_block_anchor_weight", None))
    lora_name = parameter_name.rsplit(".", 2)[0]
    group = lbw_profile.group_for_lora_name(lora_name)
    return float(weights.get(group, weights.get("default", 1.0)))


def compute_weight_anchor_loss(network, anchor_targets: Optional[Dict[str, torch.Tensor]], args) -> Optional[torch.Tensor]:
    if not getattr(args, "use_weight_anchor_loss", True):
        return None
    if anchor_targets is None:
        return None

    base_weight = float(getattr(args, "weight_anchor_loss_weight", 0.0))
    if base_weight <= 0:
        return None

    total = None
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue
        target = anchor_targets.get(name)
        if target is None:
            continue
        loss = F.mse_loss(param, target.to(device=param.device, dtype=param.dtype))
        loss = loss * base_weight * _anchor_group_weight(args, name)
        total = loss if total is None else total + loss
    return total


def compute_self_distill_loss(
    student_target: torch.Tensor,
    teacher_target: torch.Tensor,
    base_target: torch.Tensor,
    variant_type: str,
    loss_role: str,
    args,
    network=None,
    anchor_targets: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, torch.Tensor] = {}
    variant_weight = _variant_weight(args, variant_type)

    student_delta = student_target - base_target
    teacher_delta = teacher_target - base_target
    is_keep = loss_role == "keep"
    is_suppress = loss_role in {"off", "suppress"}

    if getattr(args, "use_keep_delta_loss", True) and is_keep:
        losses["keep_delta_loss"] = F.mse_loss(student_delta, teacher_delta) * float(getattr(args, "keep_delta_loss_weight", 1.0))

    if getattr(args, "use_suppress_to_base_loss", True) and is_suppress:
        losses["suppress_to_base_loss"] = F.mse_loss(student_target, base_target) * float(
            getattr(args, "suppress_to_base_loss_weight", 1.0)
        )

    if getattr(args, "use_coarse_preservation_loss", False):
        if is_keep:
            coarse_target_mode = getattr(args, "coarse_target_mode", "teacher")
            coarse_target = teacher_target if coarse_target_mode == "teacher" else base_target
        else:
            coarse_target = base_target
        losses["coarse_preservation_loss"] = (
            F.mse_loss(low_pass(student_target, args.low_pass_mode), low_pass(coarse_target, args.low_pass_mode))
            * float(getattr(args, "coarse_preservation_loss_weight", 0.0))
        )

    if getattr(args, "use_high_pass_delta_loss", False) and is_keep:
        losses["high_pass_delta_loss"] = (
            F.mse_loss(high_pass(student_delta, args.high_pass_mode), high_pass(teacher_delta, args.high_pass_mode))
            * float(getattr(args, "high_pass_delta_loss_weight", 0.0))
        )

    if getattr(args, "use_low_pass_delta_loss", False) and is_keep:
        losses["low_pass_delta_loss"] = (
            F.mse_loss(low_pass(student_delta, args.low_pass_mode), low_pass(teacher_delta, args.low_pass_mode))
            * float(getattr(args, "low_pass_delta_loss_weight", 0.0))
        )

    if getattr(args, "use_sparse_loss", False):
        losses["sparse_loss"] = student_delta.abs().mean() * float(getattr(args, "sparse_loss_weight", 0.0))

    anchor_loss = compute_weight_anchor_loss(network, anchor_targets, args)
    if anchor_loss is not None:
        losses["weight_anchor_loss"] = anchor_loss

    if not losses:
        losses["fallback_loss"] = F.mse_loss(student_target, teacher_target if is_keep else base_target)

    total = sum(losses.values()) * variant_weight
    scalar_logs = {name: float(value.detach().item()) for name, value in losses.items()}
    scalar_logs["variant_weight"] = variant_weight
    scalar_logs["loss"] = float(total.detach().item())
    return total, scalar_logs
