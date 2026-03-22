import argparse
import importlib
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from PIL import Image, ImageDraw

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import lbw_profile, self_distill_cache, sdxl_model_util, sdxl_train_util, train_util
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _load_model_bundle(args, accelerator, weight_dtype):
    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    tokenizers = sdxl_train_util.load_tokenizers(args)
    text_encoders = [text_encoder1, text_encoder2]
    for te in text_encoders:
        te.to(accelerator.device, dtype=torch.float32)
        te.eval()
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.eval()
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
    return tokenizers, text_encoders, vae, unet


def _load_eval_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload["records"]


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def _grid(images: List[List[Image.Image]], labels: List[str]) -> Image.Image:
    cell_w = images[0][0].width
    cell_h = images[0][0].height
    header_h = 40
    grid = Image.new("RGB", (cell_w * len(labels), header_h + cell_h * len(images)), color=(32, 32, 32))
    draw = ImageDraw.Draw(grid)
    for col, label in enumerate(labels):
        draw.text((col * cell_w + 10, 10), label, fill=(255, 255, 255))
    for row, row_images in enumerate(images):
        for col, image in enumerate(row_images):
            grid.paste(image, (col * cell_w, header_h + row * cell_h))
    return grid


def evaluate(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, _ = train_util.prepare_dtype(args)
    records = _load_eval_prompts(args.eval_prompts)
    self_distill_cache.ensure_dir(args.output_dir)

    def generate_for_mode(mode_name: str, lora_path: str = None, multiplier: float = 1.0, profile_path: str = None):
        tokenizers, text_encoders, vae, unet = _load_model_bundle(args, accelerator, weight_dtype)
        if lora_path is not None:
            network_module = importlib.import_module(args.network_module)
            network, weights_sd = network_module.create_network_from_weights(multiplier, lora_path, vae, text_encoders, unet, for_inference=True)
            weights_sd = lbw_profile.scale_lora_state_dict(weights_sd, lbw_profile.load_profile(profile_path))
            network.merge_to(text_encoders, unet, weights_sd, weight_dtype, accelerator.device)
        pipe = SdxlStableDiffusionLongPromptWeightingPipeline(
            vae=vae,
            text_encoder=text_encoders,
            tokenizer=tokenizers,
            unet=unet,
            scheduler=self_distill_cache.scheduler_from_settings(args.sample_sampler, args.v_parameterization),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            clip_skip=args.clip_skip or 1,
        )
        pipe.to(accelerator.device)

        outputs = []
        with torch.no_grad():
            for record in records:
                settings = self_distill_cache.generation_settings_from_prompt_record(record, args.resolution)
                conditioning = self_distill_cache.build_prompt_conditioning(
                    tokenizers,
                    text_encoders,
                    record["prompt_text"],
                    record.get("negative_prompt", ""),
                    accelerator.device,
                    weight_dtype,
                    max_embeddings_multiples=args.max_embeddings_multiples,
                    clip_skip=args.clip_skip,
                )
                scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], args.v_parameterization)
                initial = self_distill_cache.make_initial_latents(
                    record["seed"],
                    settings["height"],
                    settings["width"],
                    accelerator.device,
                    weight_dtype,
                    scheduler,
                )
                latents = self_distill_cache.run_sdxl_rollout(
                    unet,
                    scheduler,
                    conditioning,
                    initial,
                    settings["height"],
                    settings["width"],
                    settings["sample_steps"],
                    settings["scale"],
                )
                image = pipe.latents_to_image(latents.to(vae.dtype))[0]
                outputs.append((record, latents.detach().to("cpu"), image))
        clean_memory_on_device(accelerator.device)
        return outputs

    base_outputs = generate_for_mode("base")
    teacher_outputs = generate_for_mode("teacher", args.teacher_lora_weights, args.teacher_lora_multiplier, args.lbw_profile)
    student_outputs = generate_for_mode("student", args.student_lora_weights, 1.0, args.student_lbw_profile)

    images = []
    positive_scores = []
    off_scores = []
    drift_scores = []
    per_prompt = []
    for (record, base_latent, base_img), (_, teacher_latent, teacher_img), (_, student_latent, student_img) in zip(
        base_outputs, teacher_outputs, student_outputs
    ):
        images.append([base_img, teacher_img, student_img])
        base_teacher = _mse(base_latent, teacher_latent)
        student_teacher = _mse(student_latent, teacher_latent)
        student_base = _mse(student_latent, base_latent)
        metric = {
            "record_id": record["record_id"],
            "variant_type": record["variant_type"],
            "base_teacher_mse": base_teacher,
            "student_teacher_mse": student_teacher,
            "student_base_mse": student_base,
        }
        per_prompt.append(metric)
        if record["variant_type"] == "off":
            off_scores.append(student_base / (base_teacher + 1e-6))
        else:
            positive_scores.append(max(0.0, 1.0 - student_teacher / (base_teacher + 1e-6)))
        drift_scores.append(student_teacher)

    metrics = {
        "retain_proxy": float(sum(positive_scores) / max(len(positive_scores), 1)),
        "leakage_proxy": float(sum(off_scores) / max(len(off_scores), 1)),
        "drift_proxy": float(sum(drift_scores) / max(len(drift_scores), 1)),
        "per_prompt": per_prompt,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    grid = _grid(images, ["base", "teacher", "student"])
    grid.save(os.path.join(args.output_dir, "preview_grid.png"))


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, False)
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--eval_prompts", type=str, required=True)
    parser.add_argument("--teacher_lora_weights", type=str, required=True)
    parser.add_argument("--teacher_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--student_lora_weights", type=str, required=True)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--lbw_profile", type=str, default=None)
    parser.add_argument("--student_lbw_profile", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--max_embeddings_multiples", type=int, default=3)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    evaluate(args)
