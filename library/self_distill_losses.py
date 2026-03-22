from typing import Dict, Tuple

import torch
import torch.nn.functional as F


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


def _variant_is_positive(variant_type: str) -> bool:
    return variant_type in {"strong", "weak", "frontier", "support_only"}


def compute_self_distill_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    base_latent: torch.Tensor,
    variant_type: str,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, torch.Tensor] = {}
    student_delta = student_latent - base_latent
    teacher_delta = teacher_latent - base_latent
    positive_variant = _variant_is_positive(variant_type)

    if getattr(args, "positive_high_pass_delta_weight", 0.0) > 0 and positive_variant:
        student_hp = high_pass(student_delta, args.high_pass_mode)
        teacher_hp = high_pass(teacher_delta, args.high_pass_mode)
        losses["positive_high_pass_delta"] = F.mse_loss(student_hp, teacher_hp) * args.positive_high_pass_delta_weight

    if getattr(args, "coarse_preservation_weight", 0.0) > 0:
        target = teacher_latent if positive_variant else base_latent
        losses["coarse_preservation"] = (
            F.mse_loss(low_pass(student_latent, args.low_pass_mode), low_pass(target, args.low_pass_mode))
            * args.coarse_preservation_weight
        )

    if getattr(args, "off_loss_weight", 0.0) > 0 and variant_type == "off":
        losses["off_loss"] = F.mse_loss(student_latent, base_latent) * args.off_loss_weight

    if getattr(args, "anchor_loss_weight", 0.0) > 0:
        target = teacher_latent if positive_variant else base_latent
        losses["anchor_loss"] = F.mse_loss(student_latent, target) * args.anchor_loss_weight

    if getattr(args, "sparse_loss_weight", 0.0) > 0:
        losses["sparse_loss"] = student_delta.abs().mean() * args.sparse_loss_weight

    if not losses:
        losses["anchor_loss"] = F.mse_loss(student_latent, teacher_latent if positive_variant else base_latent)

    total = sum(losses.values())
    scalar_logs = {name: float(value.detach().item()) for name, value in losses.items()}
    scalar_logs["loss"] = float(total.detach().item())
    return total, scalar_logs
