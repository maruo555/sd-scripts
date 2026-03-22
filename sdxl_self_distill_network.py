import argparse
import importlib
import json
import math
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import lbw_profile, self_distill_cache, self_distill_dataset, self_distill_losses, sdxl_model_util, sdxl_train_util, train_util
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _freeze_text_encoder_lora(network) -> None:
    for lora in getattr(network, "text_encoder_loras", []):
        lora.requires_grad_(False)
        for param in lora.parameters():
            param.requires_grad = False

    for lora in getattr(network, "unet_loras", []):
        lora.requires_grad_(True)
        for param in lora.parameters():
            param.requires_grad = True


def _create_network(args, vae, text_encoders, unet):
    network_module = importlib.import_module(args.network_module)
    net_kwargs = self_distill_cache.parse_network_args(args.network_args)
    init_weights = args.student_init_weights or args.network_weights
    if args.dim_from_weights:
        if init_weights is None:
            raise ValueError("--dim_from_weights requires --student_init_weights or --network_weights.")
        network, _ = network_module.create_network_from_weights(1.0, init_weights, vae, text_encoders, unet, **net_kwargs)
    else:
        if "dropout" not in net_kwargs:
            net_kwargs["dropout"] = args.network_dropout
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            text_encoders,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
    # Keep TE LoRA modules attached so fixed TE weights are preserved in the final student safetensors.
    # They are frozen below and excluded from optimizer parameter groups.
    network.apply_to(text_encoders, unet, True, True)
    if init_weights is not None:
        logger.info("load student init weights: %s", init_weights)
        network.load_weights(init_weights)
    lbw_profile.apply_profile_to_network(network, lbw_profile.load_profile(args.lbw_profile))
    _freeze_text_encoder_lora(network)
    return network, net_kwargs


def _prepare_conditioning(args, item, tokenizers, text_encoders, device, weight_dtype):
    if "prompt_embeds" in item:
        return {
            "prompt_embeds": item["prompt_embeds"].to(device=device, dtype=weight_dtype),
            "negative_prompt_embeds": item["negative_prompt_embeds"].to(device=device, dtype=weight_dtype),
            "pooled_prompt_embeds": item["pooled_prompt_embeds"].to(device=device, dtype=weight_dtype),
            "negative_pooled_prompt_embeds": item["negative_pooled_prompt_embeds"].to(device=device, dtype=weight_dtype),
        }
    return self_distill_cache.build_prompt_conditioning(
        tokenizers,
        text_encoders,
        item["prompt_text"],
        item.get("negative_prompt", ""),
        device,
        weight_dtype,
        max_embeddings_multiples=args.max_embeddings_multiples,
        clip_skip=args.clip_skip,
    )


def _save_checkpoint(args, accelerator, network, network_args, global_step):
    model = accelerator.unwrap_model(network)
    model_name = train_util.default_if_none(args.output_name, "self_distill")
    file_path = os.path.join(args.output_dir, f"{model_name}-step{global_step:06d}.safetensors")
    metadata = None if args.no_metadata else self_distill_cache.metadata_for_lora(args, network_args)
    _, save_dtype = train_util.prepare_dtype(args)
    model.save_weights(file_path, save_dtype, metadata)
    return file_path


def train(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    self_distill_cache.ensure_dir(args.output_dir)

    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, _ = train_util.prepare_dtype(args)
    train_util.verify_command_line_training_args(args)

    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    text_encoders = [text_encoder1, text_encoder2]
    tokenizers = sdxl_train_util.load_tokenizers(args)

    for text_encoder in text_encoders:
        text_encoder.to("cpu", dtype=torch.float32)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vae.to("cpu")
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.train()
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

    network, network_args = _create_network(args, vae, text_encoders, unet)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()

    params, _ = network.prepare_optimizer_params(None, args.unet_lr or args.learning_rate, args.learning_rate, active_text_encoder_indices=[])
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, params)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    dataset = self_distill_dataset.SelfDistillDataset(args.cache_manifest, require_prompt_embeddings=args.require_cached_prompt_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=min(args.max_data_loader_n_workers, 2),
        collate_fn=self_distill_dataset.collate_single,
    )

    network, optimizer, dataloader, lr_scheduler = accelerator.prepare(network, optimizer, dataloader, lr_scheduler)

    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="steps")
    for epoch in range(10**6):
        for item in dataloader:
            settings = item["generation_settings"]
            if args.dry_run:
                _prepare_conditioning(args, item, tokenizers, text_encoders, accelerator.device, weight_dtype)
                logger.info("dry_run checked record: %s", item["record_id"])
                return

            with accelerator.accumulate(network):
                conditioning = _prepare_conditioning(args, item, tokenizers, text_encoders, accelerator.device, weight_dtype)
                initial_latents = item["initial_noise_latent"].to(accelerator.device, dtype=weight_dtype)
                teacher_latents = item["teacher_final_latent"].to(accelerator.device, dtype=weight_dtype)
                base_latents = item["base_final_latent"].to(accelerator.device, dtype=weight_dtype)

                scheduler = self_distill_cache.scheduler_from_settings(settings["sample_sampler"], args.v_parameterization)
                with accelerator.autocast():
                    student_latents = self_distill_cache.run_sdxl_rollout(
                        unet,
                        scheduler,
                        conditioning,
                        initial_latents.clone(),
                        int(settings["height"]),
                        int(settings["width"]),
                        int(settings["sample_steps"]),
                        float(settings["scale"]),
                    )
                    loss, loss_logs = self_distill_losses.compute_self_distill_loss(
                        student_latents,
                        teacher_latents,
                        base_latents,
                        item["variant_type"],
                        args,
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss_logs['loss']:.4f}", variant=item["variant_type"])
                if accelerator.is_main_process and args.logging_dir:
                    accelerator.log(loss_logs, step=global_step)
                if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                    _save_checkpoint(args, accelerator, network, network_args, global_step)
                if global_step >= args.max_train_steps:
                    break
            clean_memory_on_device(accelerator.device)

        if global_step >= args.max_train_steps:
            break

    final_path = _save_checkpoint(args, accelerator, network, network_args, global_step)
    logger.info("saved final self-distill checkpoint to %s", final_path)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, False)
    train_util.add_optimizer_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--cache_manifest", type=str, required=True)
    parser.add_argument("--student_init_weights", type=str, default=None)
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--network_dim", type=int, default=8)
    parser.add_argument("--network_alpha", type=float, default=1.0)
    parser.add_argument("--network_dropout", type=float, default=None)
    parser.add_argument("--network_args", type=str, nargs="*", default=None)
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--unet_lr", type=float, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--max_embeddings_multiples", type=int, default=3)
    parser.add_argument("--require_cached_prompt_embeddings", action="store_true")
    parser.add_argument("--lbw_profile", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--positive_high_pass_delta_weight", type=float, default=1.0)
    parser.add_argument("--coarse_preservation_weight", type=float, default=0.25)
    parser.add_argument("--off_loss_weight", type=float, default=1.0)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.1)
    parser.add_argument("--sparse_loss_weight", type=float, default=0.0)
    parser.add_argument("--high_pass_mode", type=str, default="dog", choices=["dog", "laplacian", "gaussian_residual"])
    parser.add_argument("--low_pass_mode", type=str, default="avg", choices=["avg", "gaussian", "identity"])
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)
