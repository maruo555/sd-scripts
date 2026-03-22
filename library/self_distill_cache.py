import copy
import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from library import sdxl_train_util, train_util
from library.sdxl_lpw_stable_diffusion import get_weighted_text_embeddings


PROMPT_BANK_VERSION = 1
CACHE_MANIFEST_VERSION = 1


@dataclass
class CacheRecord:
    record_id: str
    prompt_text: str
    negative_prompt: str
    variant_type: str
    seed: int
    generation_settings: Dict[str, Any]
    tensors_path: str
    preview_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prompt_text": self.prompt_text,
            "negative_prompt": self.negative_prompt,
            "variant_type": self.variant_type,
            "seed": self.seed,
            "generation_settings": self.generation_settings,
            "tensors_path": self.tensors_path,
            "preview_image_path": self.preview_image_path,
        }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_optional_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            return json.load(f)
        if ext == ".toml":
            import toml

            return toml.load(f)
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to read YAML config files.") from exc
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config format: {path}")


def save_prompt_bank(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_prompt_bank(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != PROMPT_BANK_VERSION:
        raise ValueError(f"Unsupported prompt bank version: {payload.get('version')}")
    return payload


def save_manifest(path: str, entries: List[CacheRecord]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    header = {"version": CACHE_MANIFEST_VERSION}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Manifest is empty: {path}")
    header = json.loads(lines[0])
    if header.get("version") != CACHE_MANIFEST_VERSION:
        raise ValueError(f"Unsupported cache manifest version: {header.get('version')}")
    return [json.loads(line) for line in lines[1:]]


def save_tensor_bundle(path: str, tensors: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    arrays = {}
    for key, value in tensors.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().to("cpu").numpy()
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_tensor_bundle(path: str) -> Dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        result = {}
        for key in data.files:
            result[key] = torch.from_numpy(np.array(data[key]))
        return result


def parse_network_args(network_args: Optional[List[str]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not network_args:
        return parsed

    for item in network_args:
        key, value = item.split("=", 1)
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def generation_settings_from_prompt_record(record: Dict[str, Any], fallback_resolution: int) -> Dict[str, Any]:
    settings = copy.deepcopy(record.get("generation_settings", {}))
    settings.setdefault("width", record.get("width", fallback_resolution))
    settings.setdefault("height", record.get("height", fallback_resolution))
    settings.setdefault("sample_steps", record.get("sample_steps", 20))
    settings.setdefault("scale", record.get("scale", 7.5))
    settings.setdefault("sample_sampler", record.get("sample_sampler", "euler_a"))
    settings.setdefault("negative_prompt", record.get("negative_prompt", ""))
    return settings


def make_initial_latents(
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    scheduler,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    channels = getattr(scheduler, "init_noise_sigma", None)
    _ = channels
    latent_height = height // 8
    latent_width = width // 8
    latents = torch.randn((1, 4, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    return latents


def _make_pipe_stub(tokenizer, text_encoder, device):
    return SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder, device=device)


def build_prompt_conditioning(
    tokenizers,
    text_encoders,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_embeddings_multiples: int = 3,
    clip_skip: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    prompt_embeds = []
    negative_prompt_embeds = []
    pooled = None
    negative_pooled = None

    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
        stub = _make_pipe_stub(tokenizer, text_encoder, device)
        text_encoder = text_encoder.to(device)
        text_encoder.eval()

        emb, pool, uncond_emb, uncond_pool = get_weighted_text_embeddings(
            stub,
            prompt=prompt,
            uncond_prompt=negative_prompt,
            max_embeddings_multiples=max_embeddings_multiples,
            no_boseos_middle=False,
            clip_skip=clip_skip,
            is_sdxl_text_encoder2=idx == 1,
        )
        prompt_embeds.append(emb)
        negative_prompt_embeds.append(uncond_emb)
        if pool is not None:
            pooled = pool
        if uncond_pool is not None:
            negative_pooled = uncond_pool

    prompt_embeds = torch.cat(prompt_embeds, dim=2).to(dtype)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=2).to(dtype)
    pooled = pooled.to(dtype)
    negative_pooled = negative_pooled.to(dtype)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": negative_pooled,
    }


def scheduler_from_settings(sample_sampler: str, v_parameterization: bool):
    return train_util.get_my_scheduler(sample_sampler=sample_sampler, v_parameterization=v_parameterization)


def run_sdxl_rollout(
    unet,
    scheduler,
    conditioning: Dict[str, torch.Tensor],
    initial_latents: torch.Tensor,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    device = initial_latents.device
    dtype = initial_latents.dtype
    latents = initial_latents

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    orig_size = torch.tensor([[height, width]], device=device, dtype=dtype)
    crop_size = torch.zeros_like(orig_size)
    target_size = orig_size
    size_embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device).to(dtype)

    cond_prompt = conditioning["prompt_embeds"].to(device=device, dtype=dtype)
    uncond_prompt = conditioning["negative_prompt_embeds"].to(device=device, dtype=dtype)
    cond_pool = conditioning["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    uncond_pool = conditioning["negative_pooled_prompt_embeds"].to(device=device, dtype=dtype)

    text_embedding = torch.cat([uncond_prompt, cond_prompt], dim=0)
    vector_embedding = torch.cat(
        [torch.cat([uncond_pool, size_embs], dim=1), torch.cat([cond_pool, size_embs], dim=1)], dim=0
    ).to(dtype)

    for timestep in timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = unet(latent_model_input, timestep, text_embedding, vector_embedding)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(guided, timestep, latents).prev_sample

    return latents


def metadata_for_lora(args, network_args: Dict[str, Any]) -> Dict[str, Any]:
    metadata = train_util.build_minimum_network_metadata(
        v2=False,
        base_model="sdxl_base_v1-0",
        network_module=args.network_module,
        network_dim=str(args.network_dim if args.network_dim is not None else "from_weights"),
        network_alpha=str(args.network_alpha),
        network_args=network_args,
    )
    return {str(key): str(value) for key, value in metadata.items()}


def default_generation_name(entry: Dict[str, Any], index: int) -> str:
    variant = entry.get("variant_type", "sample")
    seed = entry.get("seed", 0)
    return f"{index:05d}_{variant}_{seed}"
