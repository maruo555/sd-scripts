import argparse
import importlib
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import lbw_profile, self_distill_cache, self_distill_targets, sdxl_model_util, sdxl_train_util, train_util
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


def _load_model_bundle_to_cpu(args, accelerator, weight_dtype, tokenizers=None):
    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    if tokenizers is None:
        tokenizers = sdxl_train_util.load_tokenizers(args)
    text_encoders = [text_encoder1, text_encoder2]
    for te in text_encoders:
        te.to("cpu", dtype=torch.float32)
        te.eval()
        te.requires_grad_(False)
    vae.to("cpu", dtype=weight_dtype)
    unet.to("cpu", dtype=weight_dtype)
    unet.eval()
    self_distill_cache.apply_attention_backend(unet, args)
    return {
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "vae": vae,
        "unet": unet,
    }


def _move_bundle(bundle: Dict, device: torch.device, weight_dtype: torch.dtype, cache_conditioning: bool) -> None:
    te_device = device if cache_conditioning else torch.device("cpu")
    for te in bundle["text_encoders"]:
        te.to(te_device, dtype=torch.float32)
        te.eval()
    bundle["unet"].to(device, dtype=weight_dtype)
    bundle["unet"].eval()


def _offload_bundle(bundle: Dict, weight_dtype: torch.dtype) -> None:
    for te in bundle["text_encoders"]:
        te.to("cpu", dtype=torch.float32)
        te.eval()
    bundle["unet"].to("cpu", dtype=weight_dtype)
    bundle["unet"].eval()


def _build_teacher_bundle(args, accelerator, weight_dtype, tokenizers):
    bundle = _load_model_bundle_to_cpu(args, accelerator, weight_dtype, tokenizers=tokenizers)
    teacher_network, teacher_weights_sd = _prepare_teacher_network(args, bundle["text_encoders"], bundle["vae"], bundle["unet"])
    teacher_te_included = _teacher_uses_text_encoder(teacher_network)
    teacher_network.merge_to(bundle["text_encoders"], bundle["unet"], teacher_weights_sd, weight_dtype, torch.device("cpu"))
    del teacher_network
    del teacher_weights_sd
    return bundle, teacher_te_included


def _conditioning_for_prompt(args, tokenizers, text_encoders, prompt_text: str, negative_prompt: str, device, weight_dtype):
    te_device = device if args.cache_conditioning else torch.device("cpu")
    return self_distill_cache.build_prompt_conditioning(
        tokenizers,
        text_encoders,
        prompt_text,
        negative_prompt,
        device,
        weight_dtype,
        max_embeddings_multiples=args.max_embeddings_multiples,
        clip_skip=args.clip_skip,
        text_encoder_device=te_device,
    )


def _save_record(
    output_dir: str,
    record: Dict,
    settings: Dict,
    bundle: Dict[str, torch.Tensor],
    preview_image_path: Optional[str] = None,
) -> self_distill_cache.CacheRecord:
    tensors_path = os.path.join(output_dir, "tensors", record["record_id"] + ".npz")
    self_distill_cache.save_tensor_bundle(tensors_path, bundle)
    return self_distill_cache.CacheRecord(
        record_id=record["record_id"],
        prompt_text=record["prompt_text"],
        negative_prompt=record.get("negative_prompt", ""),
        variant_type=record["variant_type"],
        split=record.get("split", "train"),
        seed=int(record["seed"]),
        conditioning_source=record.get("conditioning_source", "teacher"),
        loss_role=record.get("loss_role", "keep"),
        generation_settings=settings,
        tensors_path=tensors_path,
        template=record.get("template", ""),
        template_index=int(record.get("template_index", -1)),
        preview_image_path=preview_image_path,
    )


def _load_resume_entries(manifest_path: str) -> Dict[str, Dict]:
    if not os.path.exists(manifest_path):
        return {"header": None, "entries": {}}
    header, entries = self_distill_cache.load_manifest_with_header(manifest_path)
    return {"header": header, "entries": {entry["record_id"]: entry for entry in entries}}


def build_cache(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    prompt_bank = self_distill_cache.load_prompt_bank(args.prompt_bank)
    records = prompt_bank["records"]

    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, _ = train_util.prepare_dtype(args)
    tokenizers = sdxl_train_util.load_tokenizers(args)
    base_bundle = _load_model_bundle_to_cpu(args, accelerator, weight_dtype, tokenizers=tokenizers)
    teacher_bundle, teacher_te_included = _build_teacher_bundle(args, accelerator, weight_dtype, tokenizers=tokenizers)

    self_distill_cache.ensure_dir(args.output_dir)
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    resume_state = _load_resume_entries(manifest_path) if args.resume_cache else {"header": None, "entries": {}}
    header = self_distill_cache.build_manifest_header(args, teacher_te_included, args.prompt_bank)
    if args.resume_cache and resume_state["header"] is not None:
        self_distill_cache.validate_resume_manifest_header(resume_state["header"], header.to_dict())
    manifest_entries: List[self_distill_cache.CacheRecord] = []

    if args.dry_run:
        logger.info("dry_run enabled: validating cache build prerequisites only.")
        logger.info("teacher_te_included=%s", teacher_te_included)
        return

    for record in records:
        record_id = record["record_id"]
        tensors_path = os.path.join(args.output_dir, "tensors", record_id + ".npz")
        if args.resume_cache and os.path.exists(tensors_path) and record_id in resume_state["entries"]:
            entry = resume_state["entries"][record_id]
            manifest_entries.append(self_distill_cache.CacheRecord(**entry))
            continue

        settings = self_distill_cache.generation_settings_from_prompt_record(record, args.resolution)
        prediction_type = header.prediction_type
        target_type = settings.get("prediction_target", args.prediction_target)

        _move_bundle(base_bundle, accelerator.device, weight_dtype, args.cache_conditioning)
        base_conditioning = _conditioning_for_prompt(
            args,
            base_bundle["tokenizers"],
            base_bundle["text_encoders"],
            record["prompt_text"],
            record.get("negative_prompt", ""),
            accelerator.device,
            weight_dtype,
        )

        initial_latents = self_distill_cache.make_initial_latents(
            record["seed"],
            settings["height"],
            settings["width"],
            accelerator.device,
            weight_dtype,
            self_distill_cache.scheduler_from_settings(settings["sample_sampler"], prediction_type=prediction_type),
        )
        _offload_bundle(base_bundle, weight_dtype)

        _move_bundle(teacher_bundle, accelerator.device, weight_dtype, args.cache_conditioning)
        teacher_conditioning = _conditioning_for_prompt(
            args,
            teacher_bundle["tokenizers"],
            teacher_bundle["text_encoders"],
            record["prompt_text"],
            record.get("negative_prompt", ""),
            accelerator.device,
            weight_dtype,
        )
        chosen_conditioning = teacher_conditioning if record.get("conditioning_source", "teacher") == "teacher" else base_conditioning
        teacher_scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], prediction_type=prediction_type)
        capture_indices = self_distill_targets.sample_target_step_indices(
            int(settings["sample_steps"]),
            int(args.num_target_timesteps),
            args.timestep_sampling_mode,
            args.custom_timestep_list,
        )
        with torch.no_grad():
            teacher_final_latent, captures = self_distill_targets.build_teacher_rollout_targets(
                teacher_bundle["unet"],
                teacher_scheduler,
                chosen_conditioning,
                initial_latents.clone(),
                int(settings["height"]),
                int(settings["width"]),
                int(settings["sample_steps"]),
                float(settings["scale"]),
                target_type,
                capture_indices,
            )
        _offload_bundle(teacher_bundle, weight_dtype)

        _move_bundle(base_bundle, accelerator.device, weight_dtype, args.cache_conditioning)
        base_conditioning_reloaded = _conditioning_for_prompt(
            args,
            base_bundle["tokenizers"],
            base_bundle["text_encoders"],
            record["prompt_text"],
            record.get("negative_prompt", ""),
            accelerator.device,
            weight_dtype,
        )
        chosen_base_conditioning = teacher_conditioning if record.get("conditioning_source", "teacher") == "teacher" else base_conditioning_reloaded
        base_scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], prediction_type=prediction_type)

        x_t_list = []
        timestep_list = []
        teacher_target_list = []
        base_target_list = []
        with torch.no_grad():
            for capture in captures:
                timestep_tensor = torch.tensor(capture["timestep"], device=accelerator.device)
                base_pred = self_distill_cache.unet_predict_cfg(
                    base_bundle["unet"],
                    base_scheduler,
                    chosen_base_conditioning,
                    capture["x_t"].to(accelerator.device, dtype=weight_dtype),
                    timestep_tensor,
                    int(settings["height"]),
                    int(settings["width"]),
                    float(settings["scale"]),
                )
                base_target = self_distill_cache.prediction_to_target(
                    base_pred,
                    target_type,
                    base_scheduler,
                    capture["x_t"].to(accelerator.device, dtype=weight_dtype),
                    timestep_tensor,
                )
                x_t_list.append(capture["x_t"].to("cpu"))
                timestep_list.append(capture["timestep"])
                teacher_target_list.append(capture["target"].to("cpu"))
                base_target_list.append(base_target.detach().to("cpu"))
        _offload_bundle(base_bundle, weight_dtype)

        bundle = {
            "target_timesteps": torch.tensor(timestep_list, dtype=torch.long),
            "x_t": torch.stack(x_t_list, dim=0),
            "teacher_target": torch.stack(teacher_target_list, dim=0),
            "base_target": torch.stack(base_target_list, dim=0),
            "initial_noise_latent": initial_latents.to("cpu"),
            "teacher_final_latent": teacher_final_latent.detach().to("cpu"),
        }
        bundle.update(self_distill_cache.conditioning_to_cache(base_conditioning_reloaded, "base"))
        bundle.update(self_distill_cache.conditioning_to_cache(teacher_conditioning, "teacher"))

        entry = _save_record(args.output_dir, record, settings, bundle)
        manifest_entries.append(entry)
        clean_memory_on_device(accelerator.device)

    self_distill_cache.save_manifest(manifest_path, header, manifest_entries)


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
    parser.add_argument("--max_embeddings_multiples", type=int, default=3)
    parser.add_argument("--cache_conditioning", action="store_true", dest="cache_conditioning")
    parser.add_argument("--no_cache_conditioning", action="store_false", dest="cache_conditioning")
    parser.set_defaults(cache_conditioning=True)
    parser.add_argument("--lbw_profile", type=str, default=None)
    parser.add_argument("--num_target_timesteps", type=int, default=2)
    parser.add_argument("--timestep_sampling_mode", type=str, choices=["uniform", "late_bias", "custom"], default="uniform")
    parser.add_argument("--custom_timestep_list", type=int, nargs="*", default=None)
    parser.add_argument("--prediction_target", type=str, choices=["eps", "v"], default="eps")
    parser.add_argument("--xt_source_mode", type=str, choices=["teacher_rollout"], default="teacher_rollout")
    parser.add_argument("--export_te_mode", type=str, choices=["preserve", "drop"], default="preserve")
    parser.add_argument("--attention_backend", type=str, choices=["auto", "sdpa", "xformers"], default="auto")
    parser.add_argument("--resume_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    build_cache(args)
