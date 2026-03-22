import argparse
import importlib
import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import lbw_profile, self_distill_cache, sdxl_model_util, sdxl_train_util, train_util
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _prepare_teacher_network(args, text_encoders, vae, unet):
    network_module = importlib.import_module(args.network_module)
    network, weights_sd = network_module.create_network_from_weights(
        args.teacher_lora_multiplier,
        args.teacher_lora_weights,
        vae,
        text_encoders,
        unet,
        for_inference=True,
    )
    profile = lbw_profile.load_profile(args.lbw_profile)
    weights_sd = lbw_profile.scale_lora_state_dict(weights_sd, profile)
    return network, weights_sd


def _teacher_uses_text_encoder(network) -> bool:
    return len(getattr(network, "text_encoder_loras", [])) > 0


def _save_record(
    output_dir: str,
    record: Dict,
    settings: Dict,
    conditioning: Dict[str, torch.Tensor],
    initial_latents: torch.Tensor,
    base_latents: torch.Tensor,
    teacher_latents: torch.Tensor,
    preview_image_path: str = None,
) -> self_distill_cache.CacheRecord:
    record_name = self_distill_cache.default_generation_name(record, int(record["template_index"]))
    tensors_path = os.path.join(output_dir, "tensors", record_name + ".npz")
    bundle = {
        "initial_noise_latent": initial_latents,
        "base_final_latent": base_latents,
        "teacher_final_latent": teacher_latents,
    }
    if conditioning is not None:
        bundle.update(conditioning)
    self_distill_cache.save_tensor_bundle(tensors_path, bundle)
    return self_distill_cache.CacheRecord(
        record_id=record["record_id"],
        prompt_text=record["prompt_text"],
        negative_prompt=record.get("negative_prompt", ""),
        variant_type=record["variant_type"],
        seed=int(record["seed"]),
        generation_settings=settings,
        tensors_path=tensors_path,
        preview_image_path=preview_image_path,
    )


def build_cache(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    prompt_bank = self_distill_cache.load_prompt_bank(args.prompt_bank)
    records = prompt_bank["records"]

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, _ = train_util.prepare_dtype(args)
    tokenizers = sdxl_train_util.load_tokenizers(args)
    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    text_encoders = [text_encoder1, text_encoder2]

    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device if args.cache_prompt_embeddings else "cpu", dtype=torch.float32)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

    teacher_network, teacher_weights_sd = _prepare_teacher_network(args, text_encoders, vae, unet)
    if _teacher_uses_text_encoder(teacher_network) and not args.cache_prompt_embeddings:
        raise ValueError(
            "teacher LoRA includes Text Encoder weights, so --cache_prompt_embeddings is required for self-distill cache."
        )

    manifest_entries: List[self_distill_cache.CacheRecord] = []
    for index, record in enumerate(records):
        settings = self_distill_cache.generation_settings_from_prompt_record(record, args.resolution)
        scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], args.v_parameterization)
        base_conditioning = self_distill_cache.build_prompt_conditioning(
            tokenizers,
            text_encoders,
            record["prompt_text"],
            record.get("negative_prompt", ""),
            accelerator.device,
            weight_dtype,
            max_embeddings_multiples=args.max_embeddings_multiples,
            clip_skip=args.clip_skip,
        )
        initial_latents = self_distill_cache.make_initial_latents(
            record["seed"],
            settings["height"],
            settings["width"],
            accelerator.device,
            weight_dtype,
            scheduler,
        )
        with torch.no_grad():
            base_latents = self_distill_cache.run_sdxl_rollout(
                unet,
                scheduler,
                base_conditioning,
                initial_latents.clone(),
                settings["height"],
                settings["width"],
                settings["sample_steps"],
                settings["scale"],
            )

        preview_path = None
        if args.save_previews and index == 0:
            preview_path = os.path.join(args.output_dir, "preview", "base_preview.png")
            self_distill_cache.ensure_dir(os.path.dirname(preview_path))
            from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline

            pipe = SdxlStableDiffusionLongPromptWeightingPipeline(
                vae=vae,
                text_encoder=text_encoders,
                tokenizer=tokenizers,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                clip_skip=args.clip_skip or 1,
            )
            pipe.to(accelerator.device)
            pipe.latents_to_image(base_latents.to(vae.dtype))[0].save(preview_path)
            del pipe
            clean_memory_on_device(accelerator.device)

        teacher_network.merge_to(text_encoders, unet, teacher_weights_sd, weight_dtype, accelerator.device)
        teacher_conditioning = self_distill_cache.build_prompt_conditioning(
            tokenizers,
            text_encoders,
            record["prompt_text"],
            record.get("negative_prompt", ""),
            accelerator.device,
            weight_dtype,
            max_embeddings_multiples=args.max_embeddings_multiples,
            clip_skip=args.clip_skip,
        )
        with torch.no_grad():
            teacher_scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], args.v_parameterization)
            teacher_latents = self_distill_cache.run_sdxl_rollout(
                unet,
                teacher_scheduler,
                teacher_conditioning,
                initial_latents.clone(),
                settings["height"],
                settings["width"],
                settings["sample_steps"],
                settings["scale"],
            )
        # reload base model for the next sample because merge_to is destructive.
        _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
            args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
        )
        text_encoders = [text_encoder1, text_encoder2]
        for text_encoder in text_encoders:
            text_encoder.to(accelerator.device if args.cache_prompt_embeddings else "cpu", dtype=torch.float32)
            text_encoder.eval()
            text_encoder.requires_grad_(False)
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        unet.eval()
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        teacher_network, teacher_weights_sd = _prepare_teacher_network(args, text_encoders, vae, unet)

        conditioning = teacher_conditioning if args.cache_prompt_embeddings else None

        entry = _save_record(args.output_dir, record, settings, conditioning, initial_latents, base_latents, teacher_latents, preview_path)
        manifest_entries.append(entry)
        clean_memory_on_device(accelerator.device)

    self_distill_cache.save_manifest(os.path.join(args.output_dir, "manifest.jsonl"), manifest_entries)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, False)
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--prompt_bank", type=str, required=True)
    parser.add_argument("--teacher_lora_weights", type=str, required=True)
    parser.add_argument("--teacher_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--max_embeddings_multiples", type=int, default=3)
    parser.add_argument("--cache_prompt_embeddings", action="store_true")
    parser.add_argument("--save_previews", action="store_true")
    parser.add_argument("--lbw_profile", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    build_cache(args)
