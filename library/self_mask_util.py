import fnmatch
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from library.sdxl_original_unet import Transformer2DModel
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenSpec:
    pattern: str
    weight: float


class SelfMaskManager:
    """Controls self-mask generation from cross-attention maps during SDXL training."""

    def __init__(
        self,
        args,
        accelerator,
        tokenizers: Sequence,
        *,
        is_sdxl: bool,
    ) -> None:
        if not is_sdxl:
            raise ValueError("SelfMaskManager is only supported for SDXL training")

        if not getattr(args, "self_mask_tokens", None):
            raise ValueError("--self_mask_enable requires --self_mask_tokens")

        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device
        self.tokenizers = tokenizers
        self.tokenizer_main = tokenizers[0]

        self.token_specs = self._parse_token_specs(args.self_mask_tokens)
        self.ignore_patterns = self._parse_ignore_patterns(args.self_mask_ignore_tokens)
        self.layer_filters = self._parse_layer_filters(args.self_mask_layers)
        self.smooth_config = self._parse_smooth_config(args.self_mask_smooth)

        self.mode = args.self_mask_mode
        self.threshold = args.self_mask_thresh
        self.foreground_weight = args.self_mask_fg
        self.background_weight = args.self_mask_bg
        self.other_scale = getattr(args, "self_mask_other_scale", 1.0) or 1.0
        self.warmup_fraction = args.self_mask_warmup or 0.0
        self.conf_min_sep = args.self_mask_conf_min_sep
        self.conf_cov_range = self._parse_cov_range(args.self_mask_conf_cov_range)
        self.log_interval = args.self_mask_log or 0
        self.log_mask_dir = getattr(args, "self_mask_log_mask_dir", None)
        self.log_mask_interval = getattr(args, "self_mask_log_mask_interval", 0)
        self.ema_decay = getattr(args, "self_mask_ema_decay", None)

        self.collecting = False
        self.active_variant = "main"
        self.records: Dict[str, List[Dict]] = {"main": [], "contrast": []}
        self.record_buffer: List[Dict] = []
        self.transformer_shapes: Dict[str, Tuple[int, int, int]] = {}

        self.char_indices: List[List[Dict]] = []
        self.other_indices: List[Optional[torch.Tensor]] = []
        self.context_length: int = 0
        self.batch_size: int = 0
        self.latent_hw: Tuple[int, int] = (0, 0)

        self.latest_metrics: Dict[str, float] = {}
        self.mask_enabled: bool = False
        self.last_mask: Optional[torch.Tensor] = None
        self.mask_ema: Optional[torch.Tensor] = None
        self.fallback_mask: Optional[torch.Tensor] = None

        self.special_token_ids = self._collect_special_token_ids(tokenizers)

        if self.background_weight <= 0:
            raise ValueError("--self_mask_bg must be > 0 to keep background gradients")

    # region public API -------------------------------------------------
    def register_unet(self, unet) -> None:
        handles = []
        for name, module in unet.named_modules():
            if isinstance(module, Transformer2DModel):
                if not self._layer_selected(name):
                    continue

                handles.append(module.register_forward_pre_hook(self._make_transformer_pre_hook(name)))

                for idx, block in enumerate(module.transformer_blocks):
                    attn = getattr(block, "attn2", None)
                    if attn is None:
                        continue
                    self._patch_cross_attention(attn, f"{name}.block{idx}", module_name=name)

        self._handles = handles  # for completeness

    def begin_step(
        self,
        *,
        batch: Dict,
        text_conds,
        latent_hw: Tuple[int, int],
        global_step: int,
        max_train_steps: int,
    ) -> None:
        self.batch_size = batch["input_ids"].shape[0]
        self.context_length = batch["input_ids"].shape[-1]
        self.latent_hw = latent_hw
        self.global_step = global_step
        self.max_train_steps = max_train_steps
        self.active_variant = "main"
        self.records = {"main": [], "contrast": []}
        self.record_buffer = []
        self.collecting = False
        self.transformer_shapes.clear()
        self.fallback_mask = None

        with torch.no_grad():
            self._prepare_token_indices(batch)

        self.text_conds = text_conds
        self.collecting = True

    def after_main_forward(self) -> None:
        self.collecting = False
        self.records["main"] = list(self.record_buffer)
        self.record_buffer = []

    def requires_contrast(self) -> bool:
        return self.mode == "contrast" and any(spec for spec_list in self.char_indices for spec in spec_list)

    def begin_contrast_pass(self):
        if not self.requires_contrast():
            return None

        hs1, hs2, pool = self.text_conds
        hs1_contrast = hs1.clone()
        hs2_contrast = hs2.clone()

        for b_idx, spec_list in enumerate(self.char_indices):
            index_list = [spec["indices"] for spec in spec_list if spec["indices"].numel() > 0]
            if not index_list:
                continue
            idx = torch.unique(torch.cat(index_list))
            hs1_contrast[b_idx, idx, :] = 0
            hs2_contrast[b_idx, idx, :] = 0

        self.collecting = True
        self.active_variant = "contrast"
        self.record_buffer = []
        return (hs1_contrast, hs2_contrast, self.text_conds[2])

    def after_contrast_forward(self) -> None:
        self.collecting = False
        self.records["contrast"] = list(self.record_buffer)
        self.record_buffer = []
        self.active_variant = "main"

    def finalize_step(self) -> Dict:
        result = {
            "mask": None,
            "enabled": False,
            "metrics": {},
        }

        if not self.records["main"]:
            self.latest_metrics = {"reason": "no_records"}
            return result

        char_main, other_main = self._aggregate_variant(self.records["main"])
        char_contrast, other_contrast = (None, None)
        if self.mode == "contrast" and self.records["contrast"]:
            char_contrast, other_contrast = self._aggregate_variant(self.records["contrast"])

        if char_main is None:
            self.latest_metrics = {"reason": "no_char"}
            return result

        mask_prob, metrics = self._build_mask(char_main, other_main, char_contrast, other_contrast)

        if mask_prob is None:
            self.latest_metrics = metrics
            return result

        mask_binary = self._apply_threshold(mask_prob)
        mask_binary = self._apply_morphology(mask_binary)

        enabled = self._should_enable_mask(metrics, mask_binary)
        if not enabled:
            metrics["enabled"] = False
            self.latest_metrics = metrics
            result["metrics"] = metrics
            result["mask"] = None
            return result

        if self.fallback_mask is not None:
            mask_binary = self.fallback_mask.to(mask_binary.device)
            self.fallback_mask = None

        mask_binary = mask_binary.to(self.device)
        self._update_mask_ema(mask_binary)

        self.last_mask = mask_binary
        self.mask_enabled = True

        metrics["enabled"] = True
        result["mask"] = mask_binary
        result["enabled"] = True
        result["metrics"] = metrics
        self.latest_metrics = metrics

        self._maybe_log(metrics)
        self._maybe_save_debug(mask_binary)

        return result

    def apply_loss_weight(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return loss
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        weight = self.background_weight + (self.foreground_weight - self.background_weight) * mask
        return loss * weight.to(loss.dtype)

    # endregion --------------------------------------------------------

    # region hooks -----------------------------------------------------
    def _make_transformer_pre_hook(self, module_name: str):
        def _hook(module, inputs):
            if not self.collecting:
                return
            hidden_states = inputs[0]
            if hidden_states is None or hidden_states.dim() != 4:
                return
            _, _, height, width = hidden_states.shape
            self.transformer_shapes[module_name] = (self.batch_size, height, width)

        return _hook

    def _patch_cross_attention(self, attn_module, name: str, module_name: str) -> None:
        orig_attention = attn_module._attention
        heads = attn_module.heads

        def _patched_attention(self_module, query, key, value):
            if not self.collecting:
                return orig_attention(query, key, value)

            q = query
            k = key
            if self_module.upcast_attention:
                q = q.float()
                k = k.float()

            attention_scores = torch.baddbmm(
                torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
                q,
                k.transpose(-1, -2),
                beta=0,
                alpha=self_module.scale,
            )

            if attention_scores.dtype == torch.float16 and getattr(self.args, "fp16_safe_norms", False):
                attention_probs = attention_scores.float().softmax(dim=-1).to(dtype=attention_scores.dtype)
            else:
                attention_probs = attention_scores.softmax(dim=-1)

            attention_probs = attention_probs.to(value.dtype)

            self._record_attention(module_name, name, attention_probs, heads)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self_module.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        attn_module._attention = _patched_attention.__get__(attn_module, attn_module.__class__)

    def _record_attention(self, module_name: str, name: str, attention_probs: torch.Tensor, heads: int) -> None:
        if not self.char_indices:
            return

        batch = attention_probs.shape[0] // heads
        query_len = attention_probs.shape[1]
        key_len = attention_probs.shape[2]

        attn = attention_probs.view(batch, heads, query_len, key_len).mean(dim=1)

        char_map = torch.zeros(batch, query_len, device=attn.device)
        other_map = torch.zeros(batch, query_len, device=attn.device)

        for b_idx in range(batch):
            specs = self.char_indices[b_idx]
            if specs:
                for spec in specs:
                    indices = spec["indices"]
                    if indices is None or indices.numel() == 0:
                        continue
                    sel = attn[b_idx].index_select(-1, indices)
                    char_map[b_idx] += spec["weight"] * sel.mean(dim=-1)

            other_idx = self.other_indices[b_idx]
            if other_idx is not None and other_idx.numel() > 0:
                other_sel = attn[b_idx].index_select(-1, other_idx)
                other_map[b_idx] = other_sel.mean(dim=-1)

        shape = self.transformer_shapes.get(module_name)
        if shape is None:
            return

        record = {
            "module": module_name,
            "name": name,
            "char": char_map.detach().to(torch.float32),
            "other": other_map.detach().to(torch.float32),
            "shape": shape,
        }
        self.record_buffer.append(record)

    # endregion --------------------------------------------------------

    # region aggregation ------------------------------------------------
    def _aggregate_variant(self, records: List[Dict]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not records:
            return None, None

        char_total = None
        other_total = None
        count = 0

        target_hw = self.latent_hw

        for record in records:
            batch, height, width = record["shape"]
            char_map = record["char"].view(batch, height, width)
            other_map = record["other"].view(batch, height, width)

            if (height, width) != target_hw:
                char_map = F.interpolate(char_map.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)
                other_map = F.interpolate(other_map.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False).squeeze(1)

            if char_total is None:
                char_total = char_map
                other_total = other_map
            else:
                char_total = char_total + char_map
                other_total = other_total + other_map
            count += 1

        if count == 0:
            return None, None

        char_total = char_total / count
        other_total = other_total / count

        return char_total, other_total

    def _build_mask(
        self,
        char_main: torch.Tensor,
        other_main: torch.Tensor,
        char_contrast: Optional[torch.Tensor],
        other_contrast: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        metrics: Dict[str, float] = {}

        char_mean = char_main.mean().item()
        other_mean = other_main.mean().item() if other_main is not None else 0.0
        sep = (char_mean - other_mean) / (other_mean + 1e-6)
        metrics["sep"] = sep
        metrics["char_mean"] = char_mean
        metrics["other_mean"] = other_mean

        if self.mode == "ratio":
            denom = char_main + self.other_scale * other_main + 1e-6
            mask_prob = torch.clamp(char_main / denom, min=0.0, max=1.0)
        elif self.mode == "diff":
            mask_prob = char_main - self.other_scale * other_main
            mask_prob = torch.clamp(mask_prob, min=0.0)
            max_val = mask_prob.amax(dim=(1, 2), keepdim=True) + 1e-6
            mask_prob = mask_prob / max_val
        elif self.mode == "contrast":
            if char_contrast is None:
                metrics["reason"] = "missing_contrast"
                return None, metrics
            delta = torch.clamp(char_main - char_contrast, min=0.0)
            denom = delta + self.other_scale * (other_main if other_main is not None else 0) + 1e-6
            mask_prob = torch.clamp(delta / denom, min=0.0, max=1.0)
        else:
            metrics["reason"] = "invalid_mode"
            return None, metrics

        metrics["raw_mean"] = mask_prob.mean().item()
        return mask_prob, metrics

    def _apply_threshold(self, mask_prob: torch.Tensor) -> torch.Tensor:
        thresh = self.threshold
        mask_prob = torch.clamp(mask_prob, 0.0, 1.0)

        if thresh is None or thresh <= 0:
            return mask_prob

        if thresh <= 1:
            return (mask_prob >= thresh).float()

        ratio = min(max(thresh, 1.0), 100.0) / 100.0
        flat = mask_prob.view(mask_prob.shape[0], -1)
        total = flat.shape[1]
        k = max(int(total * ratio), 1)
        if k > total:
            k = total
        thresh_values = []
        for b_idx in range(flat.shape[0]):
            topk = torch.topk(flat[b_idx], k, sorted=True).values
            thresh_values.append(topk[-1])
        threshold_tensor = torch.stack(thresh_values).view(-1, 1, 1)
        return (mask_prob >= threshold_tensor).float()

    def _apply_morphology(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.smooth_config:
            return mask

        out = mask
        if "blur" in self.smooth_config and out.dtype in (torch.float16, torch.float32, torch.float64):
            radius = max(int(self.smooth_config["blur"]), 0)
            if radius > 0:
                kernel = self._gaussian_kernel(radius, out.device, out.dtype)
                out = F.conv2d(out.unsqueeze(1), kernel, padding=radius, groups=1).squeeze(1)
        if "open" in self.smooth_config:
            iterations = max(int(self.smooth_config["open"]), 0)
            if iterations > 0:
                out = self._morph_open(out, iterations)
        if "close" in self.smooth_config:
            iterations = max(int(self.smooth_config["close"]), 0)
            if iterations > 0:
                out = self._morph_close(out, iterations)
        return torch.clamp(out, 0.0, 1.0)

    # endregion --------------------------------------------------------

    # region enable / logging -----------------------------------------
    def _should_enable_mask(self, metrics: Dict[str, float], mask: torch.Tensor) -> bool:
        coverage = mask.mean().item()
        metrics["coverage"] = coverage

        progress = self.global_step / max(self.max_train_steps, 1)
        if progress < self.warmup_fraction:
            metrics["reason"] = "warmup"
            return False

        if self.conf_min_sep is not None and metrics.get("sep", 0.0) < self.conf_min_sep:
            metrics["reason"] = "low_sep"
            return self._fallback_with_ema(metrics)

        if self.conf_cov_range is not None:
            cov_min, cov_max = self.conf_cov_range
            if coverage < cov_min or coverage > cov_max:
                metrics["reason"] = "cov_out_of_range"
                return self._fallback_with_ema(metrics)

        metrics["reason"] = "ok"
        return True

    def _fallback_with_ema(self, metrics: Dict[str, float]) -> bool:
        if self.ema_decay is None or self.mask_ema is None:
            return False
        metrics["fallback"] = "ema"
        self.fallback_mask = self.mask_ema.clone()
        return True

    def _update_mask_ema(self, mask: torch.Tensor) -> None:
        if self.ema_decay is None:
            return
        if self.mask_ema is None:
            self.mask_ema = mask.detach()
        else:
            self.mask_ema = self.mask_ema * self.ema_decay + mask.detach() * (1.0 - self.ema_decay)

    def _maybe_log(self, metrics: Dict[str, float]) -> None:
        if self.log_interval <= 0:
            return
        if self.global_step % self.log_interval != 0:
            return
        if not self.accelerator.is_main_process:
            return
        logger.info(
            "self-mask step=%s sep=%.5f coverage=%.5f fg=%.3f bg=%.3f reason=%s",
            self.global_step,
            metrics.get("sep", 0.0),
            metrics.get("coverage", 0.0),
            self.foreground_weight,
            self.background_weight,
            metrics.get("reason", ""),
        )

    def _maybe_save_debug(self, mask: torch.Tensor) -> None:
        if not self.log_mask_dir or self.log_mask_interval <= 0:
            return
        if self.global_step % self.log_mask_interval != 0:
            return
        if not self.accelerator.is_main_process:
            return

        os.makedirs(self.log_mask_dir, exist_ok=True)
        img = mask[0].detach().cpu().clamp(0, 1)
        img = (img * 255).to(torch.uint8)
        path = os.path.join(self.log_mask_dir, f"mask_step_{self.global_step:08d}.png")
        try:
            from PIL import Image

            Image.fromarray(img.numpy()).save(path)
        except Exception as exc:
            logger.warning("failed to save self-mask debug image: %s", exc)

    # endregion --------------------------------------------------------

    # region preparation -----------------------------------------------
    def _prepare_token_indices(self, batch: Dict) -> None:
        input_ids = batch["input_ids"]
        captions = batch.get("captions", [""] * self.batch_size)

        char_indices: List[List[Dict]] = []
        other_indices: List[Optional[torch.Tensor]] = []

        for b_idx in range(self.batch_size):
            seq = input_ids[b_idx].tolist()
            tag_map = self._build_tag_position_map(captions[b_idx], seq)

            spec_entries = []
            collected_indices = set()

            for spec in self.token_specs:
                indices = self._match_pattern_positions(spec.pattern, tag_map, seq)
                if indices:
                    tensor_idx = torch.tensor(sorted(indices), device=self.device, dtype=torch.long)
                    collected_indices.update(indices)
                else:
                    tensor_idx = torch.empty(0, dtype=torch.long, device=self.device)
                spec_entries.append({"indices": tensor_idx, "weight": spec.weight})

            ignore_indices = set()
            for pattern in self.ignore_patterns:
                ignore_indices.update(self._match_pattern_positions(pattern, tag_map, seq))

            valid_positions = [
                idx
                for idx in range(self.context_length)
                if seq[idx] not in self.special_token_ids and idx < self.context_length
            ]

            other = sorted(set(valid_positions) - collected_indices - ignore_indices)
            other_tensor = torch.tensor(other, device=self.device, dtype=torch.long) if other else None

            char_indices.append(spec_entries)
            other_indices.append(other_tensor)

        self.char_indices = char_indices
        self.other_indices = other_indices

    def _build_tag_position_map(self, caption: str, seq: List[int]) -> Dict[str, List[int]]:
        tag_map: Dict[str, List[int]] = {}
        for raw_tag in caption.split(","):
            tag = raw_tag.strip()
            if not tag:
                continue
            simplified = tag.lower()
            seq_ids = self._encode_text(tag)
            if not seq_ids:
                continue
            positions = self._find_sequence_positions(seq, seq_ids)
            if positions:
                tag_map[simplified] = positions
        return tag_map

    def _match_pattern_positions(self, pattern: str, tag_map: Dict[str, List[int]], seq: List[int]) -> List[int]:
        matches: List[int] = []
        lower_pattern = pattern.lower()
        for tag, positions in tag_map.items():
            if fnmatch.fnmatchcase(tag, lower_pattern):
                matches.extend(positions)

        if matches:
            return matches

        seq_ids = self._encode_text(pattern)
        if seq_ids:
            matches.extend(self._find_sequence_positions(seq, seq_ids))

        return matches

    def _find_sequence_positions(self, seq: List[int], pattern_ids: List[int]) -> List[int]:
        if not pattern_ids:
            return []
        positions: List[int] = []
        limit = len(seq) - len(pattern_ids) + 1
        for start in range(limit):
            if seq[start : start + len(pattern_ids)] == pattern_ids:
                positions.extend(range(start, start + len(pattern_ids)))
        return positions

    def _encode_text(self, text: str) -> List[int]:
        encoded = self.tokenizer_main(text, add_special_tokens=False)["input_ids"]
        return [idx for idx in encoded if idx not in self.special_token_ids]

    # endregion --------------------------------------------------------

    # region helpers ----------------------------------------------------
    def _parse_token_specs(self, raw: str) -> List[TokenSpec]:
        specs: List[TokenSpec] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                pattern, weight = item.rsplit(":", 1)
                try:
                    weight_val = float(weight)
                except ValueError:
                    weight_val = 1.0
            else:
                pattern = item
                weight_val = 1.0
            specs.append(TokenSpec(pattern=pattern.strip(), weight=weight_val))
        return specs

    def _parse_ignore_patterns(self, raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _parse_layer_filters(self, raw: Optional[str]) -> List[str]:
        if not raw:
            return ["mid"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _parse_smooth_config(self, raw: Optional[str]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if not raw:
            return result
        for item in raw.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            try:
                num = float(value)
            except ValueError:
                continue
            result[key] = num
        return result

    def _parse_cov_range(self, raw: Optional[str]) -> Optional[Tuple[float, float]]:
        if not raw:
            return None
        parts = raw.split(",")
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _collect_special_token_ids(self, tokenizers: Sequence) -> set:
        ids = set()
        for tok in tokenizers:
            for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
                value = getattr(tok, attr, None)
                if value is not None:
                    ids.add(int(value))
        return ids

    def _layer_selected(self, module_name: str) -> bool:
        if not self.layer_filters:
            return False

        for token in self.layer_filters:
            if token == "mid" and module_name.startswith("middle_block"):
                return True
            if token == "down" and module_name.startswith("input_blocks"):
                return True
            if token == "up" and module_name.startswith("output_blocks"):
                return True
            if token in module_name:
                return True
        return False

    def _gaussian_kernel(self, radius: int, device, dtype) -> torch.Tensor:
        size = radius * 2 + 1
        if size <= 1:
            return torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
        coords = torch.arange(size, device=device, dtype=dtype) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * max(radius, 1) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, size, size)

    def _morph_open(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        result = mask
        for _ in range(iterations):
            result = -F.max_pool2d(-result.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        for _ in range(iterations):
            result = F.max_pool2d(result.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        return result

    def _morph_close(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        result = mask
        for _ in range(iterations):
            result = F.max_pool2d(result.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        for _ in range(iterations):
            result = -F.max_pool2d(-result.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        return result

    # endregion --------------------------------------------------------
