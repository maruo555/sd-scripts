from pathlib import Path
import argparse
import json
import logging
import math
import re
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.utils import add_logging_arguments, setup_logging


logger = logging.getLogger(__name__)


@dataclass
class TensorAnalysis:
    name: str
    kind: str
    shape: List[int]
    dtype: str
    num_params: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    abs_mean: float
    max_abs: float
    l1_norm: float
    l2_norm: float
    rms: float
    sparsity: float
    entropy: float
    entropy_norm: float
    skewness: float
    kurtosis: float
    svd_rows: Optional[int] = None
    svd_cols: Optional[int] = None
    svd_top_energy_ratio: Optional[float] = None
    svd_top4_energy_ratio: Optional[float] = None
    svd_rank90: Optional[int] = None
    svd_effective_rank: Optional[float] = None


@dataclass
class BlockReport:
    name: str
    module: str
    alpha: Optional[float]
    total_params: int
    density_score: float
    density_components: Dict[str, float]
    tensors: List[TensorAnalysis]
    aggregated: Dict[str, Any]


def infer_module_name(layer_name: str) -> str:
    base = layer_name
    if base.startswith("lora_"):
        base = base[len("lora_") :]
    parts = base.split("_")
    if not parts:
        return "unknown"
    prefix = parts[0]
    if prefix in {"unet", "te1", "te2"}:
        return prefix
    if prefix == "textencoder":
        return "te1"
    if prefix == "clip":
        return "te2"
    return prefix


def infer_unet_block_label(layer_name: str) -> str:
    if not layer_name.startswith("lora_unet_"):
        return "unet"
    suffix = layer_name[len("lora_unet_") :]
    tokens = suffix.split("_")
    if len(tokens) >= 3 and tokens[0] in {"input", "output"} and tokens[1] == "blocks":
        return f"{tokens[0]}_blocks_{tokens[2]}"
    if len(tokens) >= 2 and tokens[0] == "middle" and tokens[1] == "block":
        return "middle_block"
    return tokens[0]


def _strip_text_encoder_prefix(layer_name: str, module: str) -> str:
    prefixes = [f"lora_{module}_"]
    if module == "te1":
        prefixes.extend(
            [
                "lora_te_",
                "lora_text_encoder_",
                "lora_textencoder_",
            ]
        )
    elif module == "te2":
        prefixes.extend(
            [
                "lora_clip_",
            ]
        )
    for prefix in prefixes:
        if layer_name.startswith(prefix):
            return layer_name[len(prefix) :]
    return layer_name


def infer_text_encoder_block_label(layer_name: str, module: str) -> str:
    base = _strip_text_encoder_prefix(layer_name, module)
    normalized = base.replace(".", "_")
    lower = normalized.lower()

    layer_index: Optional[int] = None
    for marker in ("encoder_layers_", "layers_", "layer_"):
        if marker in lower:
            after = lower.split(marker, 1)[1]
            digits: List[str] = []
            for ch in after:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                layer_index = int("".join(digits))
                break
    if layer_index is not None:
        return f"layer_{layer_index:02d}"

    special_map = {
        "final_layer_norm": "final_layer_norm",
        "finallayernorm": "final_layer_norm",
        "layer_norm": "layer_norm",
        "text_projection": "text_projection",
        "textprojection": "text_projection",
        "positional_embedding": "positional_embedding",
        "position_embedding": "positional_embedding",
        "token_embedding": "token_embedding",
        "embeddings": "embeddings",
    }
    for key, label in special_map.items():
        if key in lower:
            return label

    tokens = [token for token in lower.split("_") if token]
    if tokens:
        return f"other_{tokens[0]}"
    return "other"


def text_encoder_block_sort_key(label: str) -> Tuple[int, Any]:
    match = re.match(r"layer_(\d+)", label)
    if match:
        return (0, int(match.group(1)))
    return (1, label)


def module_label(module: str) -> str:
    lookup = {
        "unet": "UNet",
        "te1": "TE1 (Text Encoder 1)",
        "te2": "TE2 (Text Encoder 2)",
    }
    return lookup.get(module, module.upper())


def format_unet_block_label(label: str) -> str:
    if not label:
        return "-"
    if label.startswith("input_blocks_"):
        suffix = label.split("_")[-1]
        return f"Input {suffix}"
    if label.startswith("output_blocks_"):
        suffix = label.split("_")[-1]
        return f"Output {suffix}"
    if label == "middle_block":
        return "Middle Block"
    return label.replace("_", " ").title()


def format_text_encoder_block_label(label: str) -> str:
    if not label:
        return "-"
    if label.startswith("layer_"):
        index_part = label.split("_", 1)[1] if "_" in label else ""
        if index_part.isdigit():
            return f"Layer {int(index_part):02d}"
        if index_part:
            return f"Layer {index_part}"
    if label.startswith("other_"):
        tail = label[len("other_") :]
        if tail:
            return "その他: " + tail.replace("_", " ").title()
        return "その他"
    mapping = {
        "final_layer_norm": "Final Layer Norm",
        "layer_norm": "Layer Norm",
        "text_projection": "Text Projection",
        "positional_embedding": "Positional Embedding",
        "token_embedding": "Token Embedding",
        "embeddings": "Embeddings",
    }
    lowered = label.lower()
    for key, value in mapping.items():
        if lowered == key:
            return value
    return label.replace("_", " ").title()


def unet_block_sort_key(label: str) -> Tuple[int, Any]:
    if label.startswith("input_blocks_"):
        suffix = label.split("_")[-1]
        try:
            index = int(suffix)
        except ValueError:
            index = float("inf")
        return (0, index)
    if label == "middle_block":
        return (1, 0)
    if label.startswith("output_blocks_"):
        suffix = label.split("_")[-1]
        try:
            index = int(suffix)
        except ValueError:
            index = float("inf")
        return (2, index)
    return (3, label)


def format_epoch_label(epoch: int, is_final: bool) -> str:
    if is_final:
        return "Epoch 最終"
    return f"Epoch {epoch:02d}"


def find_checkpoint_series(model_path: str) -> List[Tuple[int, str, bool]]:
    directory = os.path.dirname(model_path) or "."
    filename = os.path.basename(model_path)
    base_name, ext = os.path.splitext(filename)
    pattern = re.compile(rf"^{re.escape(base_name)}-(\d+){re.escape(ext)}$")

    checkpoints: List[Tuple[int, str, bool]] = []
    try:
        for entry in os.listdir(directory):
            match = pattern.match(entry)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, os.path.join(directory, entry), False))
    except FileNotFoundError:
        logger.warning("Directory not found for model path: %s", directory)

    checkpoints.sort(key=lambda item: item[0])
    if checkpoints:
        final_epoch = checkpoints[-1][0] + 1
    else:
        final_epoch = 1
    checkpoints.append((final_epoch, model_path, True))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def compute_distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "q1": None, "q3": None}
    sorted_values = sorted(values)
    total = sum(values)
    count = len(values)
    return {
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(total / count),
        "median": _quantile(sorted_values, 0.5),
        "q1": _quantile(sorted_values, 0.25),
        "q3": _quantile(sorted_values, 0.75),
    }


def summarize_block(report: BlockReport) -> Dict[str, Any]:
    aggregated = report.aggregated
    components = report.density_components
    top_energy_values = [t.svd_top_energy_ratio for t in report.tensors if t.svd_top_energy_ratio is not None]
    effective_rank_values = [t.svd_effective_rank for t in report.tensors if t.svd_effective_rank is not None]

    def average(values: List[Optional[float]]) -> Optional[float]:
        usable = [v for v in values if v is not None]
        if not usable:
            return None
        return float(sum(usable) / len(usable))

    return {
        "name": report.name,
        "module": report.module,
        "alpha": report.alpha,
        "total_params": report.total_params,
        "density_score": report.density_score,
        "rms": aggregated.get("rms"),
        "entropy_norm": aggregated.get("entropy_norm"),
        "non_zero_ratio": components.get("non_zero_ratio"),
        "sparsity": aggregated.get("sparsity"),
        "mean": aggregated.get("mean"),
        "std": aggregated.get("std"),
        "max_abs": aggregated.get("max_abs"),
        "l1_norm": aggregated.get("l1_norm"),
        "l2_norm": aggregated.get("l2_norm"),
        "skewness": aggregated.get("skewness"),
        "kurtosis": aggregated.get("kurtosis"),
        "svd_top_energy_ratio_max": max(top_energy_values) if top_energy_values else None,
        "svd_top_energy_ratio_mean": average(top_energy_values),
        "svd_effective_rank_mean": average(effective_rank_values),
        "tensor_count": len(report.tensors),
    }


def load_lora_state(path: str) -> Dict[str, torch.Tensor]:
    logger.info("Loading LoRA: %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def group_lora_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    layers: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            layers.setdefault(base, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            base = key[: -len(".lora_up.weight")]
            layers.setdefault(base, {})["up"] = tensor
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            layers.setdefault(base, {})["alpha"] = tensor
        else:
            layers.setdefault(key, {})[os.path.split(key)[-1]] = tensor
    return layers


def compute_histogram_entropy(values: torch.Tensor, bins: int) -> Tuple[float, float]:
    finite_mask = torch.isfinite(values)
    if not bool(finite_mask.all()):
        values = values[finite_mask]
    if values.numel() == 0:
        return 0.0, 0.0

    min_val = float(values.min().item())
    max_val = float(values.max().item())
    if math.isclose(min_val, max_val):
        adjustment = 1e-6 if min_val >= 0 else -1e-6
        max_val = min_val + adjustment

    hist = torch.histc(values, bins=bins, min=min_val, max=max_val)
    counts = hist.float()

    total = float(sum(counts))
    if total <= 0:
        entropy = 0.0
        entropy_norm = 0.0
    else:
        probabilities = counts / total
        non_zero = probabilities > 0
        entropy = float(-(probabilities[non_zero] * torch.log(probabilities[non_zero])).sum().item())
        max_entropy = math.log(bins)
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(entropy), float(entropy_norm)


def compute_svd_metrics(tensor: torch.Tensor) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[int], Optional[float]]:
    if tensor.ndim == 0:
        return None, None, None, None, None, None

    mat = tensor.detach().cpu().float()
    if mat.ndim == 1:
        mat = mat.unsqueeze(0)
    elif mat.ndim > 2:
        mat = mat.reshape(mat.shape[0], -1)

    rows, cols = mat.shape
    if rows == 0 or cols == 0:
        return rows, cols, None, None, None, None

    try:
        singular_values = torch.linalg.svdvals(mat)
    except RuntimeError as exc:
        logger.debug("SVD failed for shape %s: %s", list(mat.shape), exc)
        return rows, cols, None, None, None, None

    if singular_values.numel() == 0:
        return rows, cols, None, None, None, None

    energy = singular_values.pow(2)
    energy_sum = energy.sum().item()
    if energy_sum == 0.0:
        return rows, cols, None, None, None, 0.0

    energy_prob = energy / energy_sum
    cumulative = torch.cumsum(energy_prob, dim=0)
    rank90_tensor = torch.nonzero(cumulative >= 0.9, as_tuple=False)
    rank90 = int(rank90_tensor[0].item() + 1) if rank90_tensor.numel() > 0 else singular_values.numel()

    top_energy_ratio = energy_prob[0].item()
    top4_energy_ratio = energy_prob[:4].sum().item() if energy_prob.numel() >= 4 else energy_prob.sum().item()

    effective_rank = float(torch.exp(-(energy_prob * torch.log(energy_prob)).sum()).item())

    return rows, cols, float(top_energy_ratio), float(top4_energy_ratio), rank90, effective_rank


def compute_tensor_analysis(
    name: str,
    kind: str,
    tensor: torch.Tensor,
    bins: int,
) -> TensorAnalysis:
    tensor_cpu = tensor.detach().cpu()
    shape = list(tensor_cpu.shape)
    num_params = int(torch.numel(tensor_cpu))

    if num_params == 0:
        mean = std = median = min_val = max_val = abs_mean = max_abs = 0.0
        l1_norm = l2_norm = rms = 0.0
        sparsity = 0.0
        skewness = kurtosis = 0.0
        entropy = entropy_norm = 0.0
    else:
        values = tensor_cpu.flatten().float()
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        median = float(values.median().item())
        min_val = float(values.min().item())
        max_val = float(values.max().item())
        abs_mean = float(values.abs().mean().item())
        max_abs = float(values.abs().max().item())
        l1_norm = float(values.abs().sum().item())
        l2_norm = float(torch.linalg.vector_norm(values).item())
        rms = float(torch.sqrt((values.pow(2).mean())).item())
        zero_ratio = float((values == 0).sum().item()) / float(num_params)
        sparsity = zero_ratio

        centered = values - mean
        if std > 0:
            skewness = float((centered.pow(3).mean() / (std ** 3)).item())
            kurtosis = float((centered.pow(4).mean() / (std ** 4)).item())
        else:
            skewness = 0.0
            kurtosis = 0.0

        entropy, entropy_norm = compute_histogram_entropy(values, bins)

    svd_rows, svd_cols, top_energy_ratio, top4_energy_ratio, rank90, eff_rank = compute_svd_metrics(tensor_cpu)

    return TensorAnalysis(
        name=name,
        kind=kind,
        shape=shape,
        dtype=str(tensor_cpu.dtype),
        num_params=num_params,
        mean=mean,
        std=std,
        median=median,
        min=min_val,
        max=max_val,
        abs_mean=abs_mean,
        max_abs=max_abs,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        rms=rms,
        sparsity=sparsity,
        entropy=entropy,
        entropy_norm=entropy_norm,
        skewness=skewness,
        kurtosis=kurtosis,
        svd_rows=svd_rows,
        svd_cols=svd_cols,
        svd_top_energy_ratio=top_energy_ratio,
        svd_top4_energy_ratio=top4_energy_ratio,
        svd_rank90=rank90,
        svd_effective_rank=eff_rank,
    )


def build_block_reports(
    layers: Dict[str, Dict[str, torch.Tensor]],
    bins: int,
) -> Tuple[List[BlockReport], Dict[str, Any]]:
    block_reports: List[BlockReport] = []
    aggregated_rms_values: List[float] = []

    for name in sorted(layers.keys()):
        layer = layers[name]
        tensors: List[TensorAnalysis] = []
        total_params = 0

        if "down" in layer:
            analysis = compute_tensor_analysis(name, "lora_down", layer["down"], bins)
            tensors.append(analysis)
            total_params += analysis.num_params
        if "up" in layer:
            analysis = compute_tensor_analysis(name, "lora_up", layer["up"], bins)
            tensors.append(analysis)
            total_params += analysis.num_params

        flattened_parts: List[torch.Tensor] = []
        for tensor_key in ("down", "up"):
            if tensor_key in layer:
                flattened_parts.append(layer[tensor_key].detach().cpu().float().flatten())

        aggregated_stats: Dict[str, Any] = {}
        if flattened_parts:
            combined = torch.cat(flattened_parts)
            combined_analysis = compute_tensor_analysis(name, "combined", combined, bins)
            aggregated_stats = {
                "num_params": combined_analysis.num_params,
                "mean": combined_analysis.mean,
                "std": combined_analysis.std,
                "median": combined_analysis.median,
                "min": combined_analysis.min,
                "max": combined_analysis.max,
                "abs_mean": combined_analysis.abs_mean,
                "max_abs": combined_analysis.max_abs,
                "l1_norm": combined_analysis.l1_norm,
                "l2_norm": combined_analysis.l2_norm,
                "rms": combined_analysis.rms,
                "sparsity": combined_analysis.sparsity,
                "entropy": combined_analysis.entropy,
                "entropy_norm": combined_analysis.entropy_norm,
                "skewness": combined_analysis.skewness,
                "kurtosis": combined_analysis.kurtosis,
            }
            aggregated_rms_values.append(combined_analysis.rms)
        else:
            aggregated_stats = {
                "num_params": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "abs_mean": 0.0,
                "max_abs": 0.0,
                "l1_norm": 0.0,
                "l2_norm": 0.0,
                "rms": 0.0,
                "sparsity": 0.0,
                "entropy": 0.0,
                "entropy_norm": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }

        alpha_tensor = layer.get("alpha")
        alpha_value = None
        if isinstance(alpha_tensor, torch.Tensor):
            if alpha_tensor.numel() == 1:
                alpha_value = float(alpha_tensor.item())
            else:
                alpha_value = float(alpha_tensor.detach().cpu().float().mean().item())

        density_components = {
            "entropy_norm": aggregated_stats["entropy_norm"],
            "non_zero_ratio": 1.0 - aggregated_stats["sparsity"],
            "rms": aggregated_stats["rms"],
        }

        block_reports.append(
            BlockReport(
                name=name,
                module=infer_module_name(name),
                alpha=alpha_value,
                total_params=total_params,
                density_score=0.0,
                density_components=density_components,
                tensors=tensors,
                aggregated=aggregated_stats,
            )
        )

    max_rms = max(aggregated_rms_values) if aggregated_rms_values else 0.0
    for report in block_reports:
        components = report.density_components
        entropy_term = components["entropy_norm"]
        sparsity_term = components["non_zero_ratio"]
        if max_rms > 0:
            rms_term = components["rms"] / max_rms
        else:
            rms_term = 0.0
        report.density_score = entropy_term * sparsity_term * rms_term
        report.density_components["rms_norm"] = rms_term

    global_summary = {
        "total_blocks": len(block_reports),
        "total_params": sum(r.total_params for r in block_reports),
        "max_rms": max_rms,
    }

    return block_reports, global_summary






def collect_single_analysis(model_path: str, bins: int) -> Dict[str, Any]:
    state_dict = load_lora_state(model_path)
    layers = group_lora_layers(state_dict)
    blocks, base_summary = build_block_reports(layers, bins)

    block_snapshots = [summarize_block(block) for block in blocks]

    density_values = [snap["density_score"] for snap in block_snapshots]
    rms_values = [snap["rms"] for snap in block_snapshots if snap["rms"] is not None]
    entropy_values = [snap["entropy_norm"] for snap in block_snapshots if snap["entropy_norm"] is not None]
    sparsity_values = [snap["sparsity"] for snap in block_snapshots if snap["sparsity"] is not None]

    summary = {
        "total_blocks": base_summary.get("total_blocks"),
        "total_params": base_summary.get("total_params"),
        "max_rms": base_summary.get("max_rms"),
        "density": compute_distribution(density_values),
        "rms": compute_distribution(rms_values),
        "entropy_norm": compute_distribution(entropy_values),
        "sparsity": compute_distribution(sparsity_values),
    }

    module_groups: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot in block_snapshots:
        module_groups.setdefault(snapshot["module"], []).append(snapshot)

    module_summary = []
    module_density_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for module in sorted(module_groups.keys()):
        group = module_groups[module]
        density_values_group = [item["density_score"] for item in group]
        rms_values_group = [item["rms"] for item in group if item["rms"] is not None]
        entropy_values_group = [item["entropy_norm"] for item in group if item["entropy_norm"] is not None]

        density_stats_group = compute_distribution(density_values_group)
        module_density_bounds[module] = (density_stats_group["min"], density_stats_group["max"])

        module_summary.append(
            {
                "module": module,
                "block_count": len(group),
                "total_params": sum(item["total_params"] for item in group),
                "density": density_stats_group,
                "rms": compute_distribution(rms_values_group),
                "entropy_norm": compute_distribution(entropy_values_group),
            }
        )

    for snapshot in block_snapshots:
        module = snapshot["module"]
        low, high = module_density_bounds.get(module, (None, None))
        norm = None
        if low is not None and high is not None:
            if math.isclose(high, low):
                norm = 0.5
            else:
                norm = (snapshot["density_score"] - low) / (high - low)
        snapshot["module_density_norm"] = None if norm is None else max(0.0, min(1.0, norm))

    top_n_global = min(12, len(block_snapshots))
    top_blocks = sorted(block_snapshots, key=lambda item: item["density_score"], reverse=True)[:top_n_global]
    bottom_blocks = sorted(block_snapshots, key=lambda item: item["density_score"])[:top_n_global]

    module_rank_limit = 8
    module_rankings = []
    for module in sorted(module_groups.keys()):
        group = module_groups[module]
        top_group = sorted(group, key=lambda item: item["density_score"], reverse=True)[: module_rank_limit]
        bottom_group = sorted(group, key=lambda item: item["density_score"])[: module_rank_limit]
        module_rankings.append(
            {
                "module": module,
                "top": top_group,
                "bottom": bottom_group,
                "top_count": len(top_group),
                "bottom_count": len(bottom_group),
            }
        )

    te_block_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for snapshot in block_snapshots:
        module = snapshot["module"]
        if module not in {"te1", "te2"}:
            continue
        label = infer_text_encoder_block_label(snapshot["name"], module)
        te_block_groups.setdefault(module, {}).setdefault(label, []).append(snapshot)

    te_block_summary: Dict[str, List[Dict[str, Any]]] = {}
    for module, groups in te_block_groups.items():
        summary_rows: List[Dict[str, Any]] = []
        for label in sorted(groups.keys(), key=text_encoder_block_sort_key):
            group = groups[label]
            density_values_group = [item["density_score"] for item in group]
            rms_values_group = [item["rms"] for item in group if item["rms"] is not None]
            summary_rows.append(
                {
                    "module": module,
                    "label": label,
                    "block_count": len(group),
                    "total_params": sum(item["total_params"] for item in group),
                    "density": compute_distribution(density_values_group),
                    "rms": compute_distribution(rms_values_group),
                }
            )
        te_block_summary[module] = summary_rows

    unet_block_groups: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot in block_snapshots:
        if snapshot["module"] != "unet":
            continue
        block_label = infer_unet_block_label(snapshot["name"])
        unet_block_groups.setdefault(block_label, []).append(snapshot)

    unet_block_summary = []
    for block_label in sorted(unet_block_groups.keys(), key=unet_block_sort_key):
        group = unet_block_groups[block_label]
        density_stats_group = compute_distribution([item["density_score"] for item in group])
        unet_block_summary.append(
            {
                "label": block_label,
                "block_count": len(group),
                "total_params": sum(item["total_params"] for item in group),
                "density": density_stats_group,
                "rms": compute_distribution([item["rms"] for item in group if item["rms"] is not None]),
            }
        )

    sparse_threshold = 0.99
    sparse_blocks = [
        item for item in block_snapshots if item["non_zero_ratio"] is not None and item["non_zero_ratio"] < sparse_threshold
    ]
    sparse_blocks = sorted(sparse_blocks, key=lambda item: item["non_zero_ratio"])

    report_data = {
        "model_path": model_path,
        "settings": {
            "bins": bins,
            "top_n": top_n_global,
            "module_top_n": module_rank_limit,
            "sparse_threshold": sparse_threshold,
        },
        "summary": summary,
        "module_summary": module_summary,
        "module_rankings": module_rankings,
        "unet_block_summary": unet_block_summary,
        "te1_block_summary": te_block_summary.get("te1", []),
        "te2_block_summary": te_block_summary.get("te2", []),
        "top_blocks": top_blocks,
        "bottom_blocks": bottom_blocks,
        "sparse_blocks": sparse_blocks,
        "blocks": block_snapshots,
    }

    return report_data


def build_checkpoint_history(
    series_data: List[Tuple[int, str, bool, Dict[str, Any]]]
) -> Dict[str, Any]:
    if len(series_data) <= 1:
        return {}

    files: List[Dict[str, Any]] = []
    module_series_map: Dict[str, Dict[str, Dict[str, Any]]] = {"unet": {}, "te1": {}, "te2": {}}

    for epoch, path, is_final, report in series_data:
        files.append(
            {
                "epoch": epoch,
                "label": format_epoch_label(epoch, is_final),
                "path": path,
                "filename": os.path.basename(path),
                "is_final": is_final,
            }
        )

        summary_sources = {
            "unet": (
                report.get("unet_block_summary", []),
                unet_block_sort_key,
                format_unet_block_label,
            ),
            "te1": (
                report.get("te1_block_summary", []),
                text_encoder_block_sort_key,
                format_text_encoder_block_label,
            ),
            "te2": (
                report.get("te2_block_summary", []),
                text_encoder_block_sort_key,
                format_text_encoder_block_label,
            ),
        }

        for module, (summary_entries, _sort_key, label_formatter) in summary_sources.items():
            module_records = module_series_map[module]
            for entry in summary_entries:
                label = entry.get("label")
                if not label:
                    continue
                density_stats = entry.get("density", {}) or {}
                record = module_records.setdefault(
                    label,
                    {
                        "label": label,
                        "display_label": label_formatter(label),
                        "values": [],
                    },
                )
                record["values"].append(
                    {
                        "epoch": epoch,
                        "density_mean": density_stats.get("mean"),
                        "density_median": density_stats.get("median"),
                        "block_count": entry.get("block_count"),
                        "total_params": entry.get("total_params"),
                    }
                )

    files.sort(key=lambda item: item["epoch"])

    series_output: Dict[str, List[Dict[str, Any]]] = {}
    for module, records in module_series_map.items():
        if module == "unet":
            sorted_entries = sorted(records.values(), key=lambda item: unet_block_sort_key(item["label"]))
        else:
            sorted_entries = sorted(records.values(), key=lambda item: text_encoder_block_sort_key(item["label"]))

        processed_entries: List[Dict[str, Any]] = []
        for info in sorted_entries:
            info["values"].sort(key=lambda val: val["epoch"])
            start_density = info["values"][0]["density_mean"] if info["values"] else None
            end_density = info["values"][-1]["density_mean"] if info["values"] else None
            delta_density = None
            if start_density is not None and end_density is not None:
                delta_density = end_density - start_density
            info["start_density"] = start_density
            info["end_density"] = end_density
            info["delta_density"] = delta_density
            processed_entries.append(info)

        series_output[module] = processed_entries

    return {
        "files": files,
        "series": series_output,
        "module_labels": {
            "unet": module_label("unet"),
            "te1": module_label("te1"),
            "te2": module_label("te2"),
        },
    }


def _registered_series(reader, selected=None, *, trend=True):
    """Return approved epoch/terminal paths and explicit holes, independent of file naming."""
    resolved, _ = reader.settings()
    finals = [item for item in reader.index["weights"] if item["role"] == "final" and item["status"] == "saved"]
    if len(finals) > 1:
        raise ValueError("Multiple final weights in the run index")
    if selected is not None:
        chosen = [item for item in reader.index["weights"]
                  if item["status"] == "saved" and item["path"] == selected.name]
        if len(chosen) != 1:
            raise ValueError(f"Expected one registered weight: {selected}")
        current = chosen[0]
        if current["role"] != "final":
            path, status, reason = reader.weight(current, resolved)
            if status not in {"indexed", "session_matched"}:
                raise ValueError(f"Cannot analyze {selected}: {reason or status}")
            return [(current["epoch"], str(path), True)], []
    points, issues = {}, []
    # A published final supersedes the pre-promotion epoch snapshot at the same epoch.
    chosen_final = finals[0] if finals else None
    for item in reader.index["weights"]:
        role, epoch = item["role"], item["epoch"]
        if role not in ("epoch", "final") or not trend and role != "final":
            continue
        if chosen_final and role == "epoch" and epoch == chosen_final["epoch"]:
            continue
        path, status, reason = reader.weight(item, resolved)
        if status not in {"indexed", "session_matched"}:
            issues.append({"epoch": epoch, "path": item["path"], "status": status, "reason": reason or status})
            continue
        if epoch in points:
            raise ValueError(f"Duplicate density checkpoint at epoch {epoch}")
        points[epoch] = (epoch, str(path), role == "final")
    return [points[e] for e in sorted(points)], issues


def analyze_series(points, bins, *, partial=False):
    """Analyze each weight once. Only the final retains full module details across the series."""
    if bins < 2:
        raise ValueError("--bins requires at least two histogram bins")
    series, issues = [], []
    final = None
    for epoch, path, is_final in points:
        logger.info("Analyzing epoch %d: %s", epoch, path)
        try:
            data = collect_single_analysis(path, bins)
        except (OSError, ValueError, RuntimeError) as exc:
            if not partial:
                raise
            issues.append({"epoch": epoch, "path": path, "reason": str(exc)})
            continue
        if is_final:
            final = data
        summaries = {key: data[key] for key in ("unet_block_summary", "te1_block_summary", "te2_block_summary")}
        series.append((epoch, path, is_final, summaries))
    history = build_checkpoint_history(series)
    history["point_count"] = len(series)
    if len(series) == 1:
        e, path, is_final, _ = series[0]
        history["files"] = [{"epoch": e, "path": path, "filename": Path(path).name, "is_final": is_final,
                             "label": format_epoch_label(e, is_final)}]
    return final, history, issues


def analyze_run(reader, bins, *, trend):
    points, issues = _registered_series(reader, trend=trend)
    final, history, failed = analyze_series(points, bins, partial=True)
    return final, history, issues + failed


def analyze_lora_density(model_path: str, bins: int) -> Dict[str, Any]:
    model = Path(model_path).resolve(strict=True)
    index = model.parent / "run.json"
    if index.exists():
        from tools.lora_run_reader import RunReader
        reader = RunReader(model.parent)
        points, issues = _registered_series(reader, selected=model)
        output_dir = reader.run.directory / "density"
    else:
        points, issues = find_checkpoint_series(str(model)), []
        output_dir = Path.cwd() / "analyze_lora_result"
    final, history, _ = analyze_series(points, bins)
    if final is None:
        raise ValueError(f"Selected weight is unavailable: {model}; {issues}")
    final.update(model_path=str(model), generated_at=datetime.now().isoformat(timespec="seconds"),
                 checkpoint_history=history, missing_checkpoints=issues)
    output_dir.mkdir(parents=True, exist_ok=True)
    from tools.lora_density_display import build_html_report
    from library.training_run import write_json
    json_path, html_path = (output_dir / (model.stem + suffix) for suffix in (".json", ".html"))
    write_json(json_path, final)
    html_path.write_text(build_html_report(final), encoding="utf-8")
    logger.info("Saved density analysis: %s, %s", json_path, html_path)
    return final


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRAブロック密度分析レポート生成ツール")
    parser.add_argument("model", type=str, help="解析対象のLoRAファイル（.safetensors など）")
    parser.add_argument("--bins", type=int, default=128, help="ヒストグラムのビン数")
    add_logging_arguments(parser)
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    setup_logging(args, reset=True)
    analyze_lora_density(
        model_path=args.model,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
