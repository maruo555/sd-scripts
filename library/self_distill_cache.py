import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from library import sdxl_train_util, train_util
from library.sdxl_lpw_stable_diffusion import get_weighted_text_embeddings


PROMPT_BANK_VERSION = 2
CACHE_MANIFEST_VERSION = 2


@dataclass
class CacheManifestHeader:
    version: int
    base_model_identifier: str
    base_model_hash: str
    teacher_lora_identifier: str
    teacher_lora_hash: str
    teacher_te_included: bool
    export_te_mode: str
    lbw_profile_hash: str
    prediction_type: str
    resolution: int
    scheduler: str
    xt_source_mode: str
    timestep_sampling_mode: str
    prompt_bank_hash: str
    cache_schema_version: int
    num_target_timesteps: int
    prompt_bank_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model_identifier": self.base_model_identifier,
            "base_model_hash": self.base_model_hash,
            "teacher_lora_identifier": self.teacher_lora_identifier,
            "teacher_lora_hash": self.teacher_lora_hash,
            "teacher_te_included": self.teacher_te_included,
            "export_te_mode": self.export_te_mode,
            "lbw_profile_hash": self.lbw_profile_hash,
            "prediction_type": self.prediction_type,
            "resolution": self.resolution,
            "scheduler": self.scheduler,
            "xt_source_mode": self.xt_source_mode,
            "timestep_sampling_mode": self.timestep_sampling_mode,
            "prompt_bank_hash": self.prompt_bank_hash,
            "cache_schema_version": self.cache_schema_version,
            "num_target_timesteps": self.num_target_timesteps,
            "prompt_bank_path": self.prompt_bank_path,
        }


@dataclass
class CacheRecord:
    record_id: str
    prompt_text: str
    negative_prompt: str
    variant_type: str
    split: str
    seed: int
    conditioning_source: str
    loss_role: str
    generation_settings: Dict[str, Any]
    tensors_path: str
    template: str
    template_index: int
    preview_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prompt_text": self.prompt_text,
            "negative_prompt": self.negative_prompt,
            "variant_type": self.variant_type,
            "split": self.split,
            "seed": self.seed,
            "conditioning_source": self.conditioning_source,
            "loss_role": self.loss_role,
            "generation_settings": self.generation_settings,
            "tensors_path": self.tensors_path,
            "template": self.template,
            "template_index": self.template_index,
            "preview_image_path": self.preview_image_path,
        }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_path(path: Optional[str]) -> str:
    if path is None:
        return ""
    return os.path.abspath(os.path.expanduser(path))


def file_sha256(path: Optional[str]) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def parse_mapping_arg(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)
    raise TypeError(f"Unsupported mapping type: {type(value)}")


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


def save_manifest(path: str, header: CacheManifestHeader, entries: List[CacheRecord]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": header.to_dict()}, ensure_ascii=False) + "\n")
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def load_manifest_with_header(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Manifest is empty: {path}")
    first_line = json.loads(lines[0])
    header = first_line.get("header")
    if not header:
        raise ValueError(f"Manifest header is missing: {path}")
    if header.get("version") != CACHE_MANIFEST_VERSION:
        raise ValueError(f"Unsupported cache manifest version: {header.get('version')}")
    entries = [json.loads(line) for line in lines[1:]]
    return header, entries


def load_manifest(path: str) -> List[Dict[str, Any]]:
    _, entries = load_manifest_with_header(path)
    return entries


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


def load_tensor_bundle(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result: Dict[str, Any] = {}
        for key in data.files:
            value = np.array(data[key])
            if value.dtype.kind in {"U", "S", "O"}:
                result[key] = value.tolist()
                continue
            result[key] = torch.from_numpy(value)
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
    settings.setdefault("prediction_target", record.get("prediction_target", "eps"))
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
    latent_height = height // 8
    latent_width = width // 8
    latents = torch.randn((1, 4, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
    return latents * scheduler.init_noise_sigma


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

    return {
        "prompt_embeds": torch.cat(prompt_embeds, dim=2).to(dtype),
        "negative_prompt_embeds": torch.cat(negative_prompt_embeds, dim=2).to(dtype),
        "pooled_prompt_embeds": pooled.to(dtype),
        "negative_pooled_prompt_embeds": negative_pooled.to(dtype),
    }


def conditioning_to_cache(bundle: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}_prompt_embeds": bundle["prompt_embeds"],
        f"{prefix}_negative_prompt_embeds": bundle["negative_prompt_embeds"],
        f"{prefix}_pooled_prompt_embeds": bundle["pooled_prompt_embeds"],
        f"{prefix}_negative_pooled_prompt_embeds": bundle["negative_pooled_prompt_embeds"],
    }


def cached_conditioning_from_bundle(bundle: Dict[str, Any], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        "prompt_embeds": bundle[f"{prefix}_prompt_embeds"],
        "negative_prompt_embeds": bundle[f"{prefix}_negative_prompt_embeds"],
        "pooled_prompt_embeds": bundle[f"{prefix}_pooled_prompt_embeds"],
        "negative_pooled_prompt_embeds": bundle[f"{prefix}_negative_pooled_prompt_embeds"],
    }


def select_cached_conditioning(bundle: Dict[str, Any], source: str) -> Dict[str, torch.Tensor]:
    prefix = "teacher" if source == "teacher" else "base"
    return cached_conditioning_from_bundle(bundle, prefix)


def resolve_prediction_type(args) -> str:
    prediction_target = getattr(args, "prediction_target", None)
    if prediction_target in {"eps", "v"}:
        return "epsilon" if prediction_target == "eps" else "v_prediction"
    return "v_prediction" if getattr(args, "v_parameterization", False) else "epsilon"


def scheduler_from_settings(sample_sampler: str, v_parameterization: bool = False, prediction_type: Optional[str] = None):
    if prediction_type is not None:
        v_parameterization = prediction_type == "v_prediction"
    return train_util.get_my_scheduler(sample_sampler=sample_sampler, v_parameterization=v_parameterization)


def prediction_to_target(
    model_prediction: torch.Tensor,
    target_type: str,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if target_type == "eps" and prediction_type == "epsilon":
        return model_prediction
    if target_type == "v" and prediction_type == "v_prediction":
        return model_prediction

    timestep = timestep.to(device=latents.device)
    if timestep.ndim == 0:
        timestep = timestep[None]
    alphas = scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)[timestep.long()]
    while alphas.ndim < latents.ndim:
        alphas = alphas.unsqueeze(-1)
    sqrt_alpha = alphas.sqrt()
    sqrt_one_minus_alpha = (1.0 - alphas).sqrt()

    if prediction_type == "epsilon":
        epsilon = model_prediction
    elif prediction_type == "v_prediction":
        epsilon = sqrt_alpha * model_prediction + sqrt_one_minus_alpha * latents
    else:
        raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")

    if target_type == "eps":
        return epsilon

    x0 = (latents - sqrt_one_minus_alpha * epsilon) / sqrt_alpha.clamp_min(1e-6)
    return sqrt_alpha * epsilon - sqrt_one_minus_alpha * x0


def _prepare_embeddings(conditioning: Dict[str, torch.Tensor], height: int, width: int, device, dtype):
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
    return text_embedding, vector_embedding


def unet_predict_cfg(
    unet,
    scheduler,
    conditioning: Dict[str, torch.Tensor],
    latents: torch.Tensor,
    timestep: torch.Tensor,
    height: int,
    width: int,
    guidance_scale: float,
) -> torch.Tensor:
    device = latents.device
    dtype = latents.dtype
    text_embedding, vector_embedding = _prepare_embeddings(conditioning, height, width, device, dtype)
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    model_output = unet(latent_model_input, timestep, text_embedding, vector_embedding)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample
    noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


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
    latents = initial_latents
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    for timestep in scheduler.timesteps:
        guided = unet_predict_cfg(unet, scheduler, conditioning, latents, timestep, height, width, guidance_scale)
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
    metadata["ss_self_distill_version"] = "v2"
    metadata["ss_export_te_mode"] = str(getattr(args, "export_te_mode", "preserve"))
    return {str(key): str(value) for key, value in metadata.items()}


def default_generation_name(entry: Dict[str, Any], index: int) -> str:
    variant = entry.get("variant_type", "sample")
    seed = entry.get("seed", 0)
    return f"{index:05d}_{variant}_{seed}"


def build_manifest_header(args, teacher_te_included: bool, prompt_bank_path: str) -> CacheManifestHeader:
    prediction_type = resolve_prediction_type(args)
    return CacheManifestHeader(
        version=CACHE_MANIFEST_VERSION,
        base_model_identifier=normalize_path(args.pretrained_model_name_or_path),
        base_model_hash=file_sha256(args.pretrained_model_name_or_path),
        teacher_lora_identifier=normalize_path(args.teacher_lora_weights),
        teacher_lora_hash=file_sha256(args.teacher_lora_weights),
        teacher_te_included=teacher_te_included,
        export_te_mode=getattr(args, "export_te_mode", "preserve"),
        lbw_profile_hash=file_sha256(getattr(args, "lbw_profile", None)),
        prediction_type=prediction_type,
        resolution=int(args.resolution),
        scheduler=str(getattr(args, "sample_sampler", "euler_a")),
        xt_source_mode=str(getattr(args, "xt_source_mode", "teacher_rollout")),
        timestep_sampling_mode=str(getattr(args, "timestep_sampling_mode", "uniform")),
        prompt_bank_hash=file_sha256(prompt_bank_path),
        cache_schema_version=CACHE_MANIFEST_VERSION,
        num_target_timesteps=int(getattr(args, "num_target_timesteps", 2)),
        prompt_bank_path=normalize_path(prompt_bank_path),
    )


def validate_manifest_header(header: Dict[str, Any], args, prompt_bank_path: Optional[str] = None) -> None:
    expected_prediction = resolve_prediction_type(args)
    checks = {
        "base_model_identifier": normalize_path(args.pretrained_model_name_or_path),
        "base_model_hash": file_sha256(args.pretrained_model_name_or_path),
        "lbw_profile_hash": file_sha256(getattr(args, "lbw_profile", None)),
        "prediction_type": expected_prediction,
        "resolution": int(args.resolution),
        "scheduler": str(getattr(args, "sample_sampler", header.get("scheduler", ""))),
        "xt_source_mode": str(getattr(args, "xt_source_mode", "teacher_rollout")),
        "timestep_sampling_mode": str(getattr(args, "timestep_sampling_mode", header.get("timestep_sampling_mode", ""))),
        "cache_schema_version": CACHE_MANIFEST_VERSION,
    }
    teacher_lora_path = getattr(args, "teacher_lora_weights", None)
    if teacher_lora_path:
        checks["teacher_lora_identifier"] = normalize_path(teacher_lora_path)
        checks["teacher_lora_hash"] = file_sha256(teacher_lora_path)
    if prompt_bank_path is not None:
        checks["prompt_bank_hash"] = file_sha256(prompt_bank_path)
    for key, expected in checks.items():
        actual = header.get(key)
        if actual != expected:
            raise ValueError(f"Strict manifest check failed for {key}: expected={expected!r} actual={actual!r}")


def validate_resume_manifest_header(existing_header: Dict[str, Any], new_header: Dict[str, Any]) -> None:
    keys = [
        "base_model_identifier",
        "base_model_hash",
        "teacher_lora_identifier",
        "teacher_lora_hash",
        "teacher_te_included",
        "export_te_mode",
        "lbw_profile_hash",
        "prediction_type",
        "resolution",
        "scheduler",
        "xt_source_mode",
        "timestep_sampling_mode",
        "prompt_bank_hash",
        "cache_schema_version",
        "num_target_timesteps",
    ]
    for key in keys:
        if existing_header.get(key) != new_header.get(key):
            raise ValueError(
                f"--resume_cache cannot reuse cache built with different {key}: "
                f"existing={existing_header.get(key)!r} new={new_header.get(key)!r}"
            )


def resolve_attention_backend(args) -> Tuple[bool, bool]:
    backend = getattr(args, "attention_backend", None)
    if backend is None:
        return bool(getattr(args, "xformers", False)), bool(getattr(args, "sdpa", False))
    backend = backend.lower()
    if backend == "xformers":
        return True, False
    if backend == "sdpa":
        return False, True
    if backend == "auto":
        if importlib.util.find_spec("xformers") is not None:
            return True, False
        return False, True
    return False, False


def apply_attention_backend(unet, args) -> None:
    use_xformers, use_sdpa = resolve_attention_backend(args)
    train_util.replace_unet_modules(unet, False, use_xformers, use_sdpa)
