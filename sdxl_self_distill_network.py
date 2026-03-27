import argparse
import importlib
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    lbw_profile,
    self_distill_cache,
    self_distill_dataset,
    self_distill_losses,
    self_distill_sampler,
    sdxl_model_util,
    sdxl_train_util,
    train_util,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _weights_include_text_encoder_lora(weights_path: Optional[str]) -> bool:
    if not weights_path:
        return False
    if os.path.splitext(weights_path)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("lora_te"):
                    return True
        return False

    weights_sd = torch.load(weights_path, map_location="cpu")
    return any(key.startswith("lora_te") for key in weights_sd.keys())


def _load_weights_sd(weights_path: str) -> Dict[str, torch.Tensor]:
    if os.path.splitext(weights_path)[1] == ".safetensors":
        from safetensors.torch import load_file

        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")


def _infer_network_shape_from_weights(weights_path: str) -> Dict[str, Optional[float]]:
    weights_sd = _load_weights_sd(weights_path)
    base_dims = set()
    base_alphas = set()
    conv_dims = set()
    conv_alphas = set()

    for key, value in weights_sd.items():
        if not key.endswith("lora_down.weight"):
            continue
        lora_name = key.rsplit(".", 2)[0]
        alpha_key = f"{lora_name}.alpha"
        alpha_value = weights_sd.get(alpha_key, value.shape[0])
        if value.ndim == 4 and tuple(value.shape[-2:]) != (1, 1):
            conv_dims.add(int(value.shape[0]))
            conv_alphas.add(float(alpha_value))
        else:
            base_dims.add(int(value.shape[0]))
            base_alphas.add(float(alpha_value))

    if len(base_dims) != 1 or len(base_alphas) != 1:
        raise ValueError(f"Could not infer a single base LoRA rank/alpha from {weights_path}.")
    if len(conv_dims) > 1 or len(conv_alphas) > 1:
        raise ValueError(f"Could not infer a single Conv2d(3x3) LoRA rank/alpha from {weights_path}.")

    conv_dim = next(iter(conv_dims)) if conv_dims else None
    conv_alpha = next(iter(conv_alphas)) if conv_alphas else None
    return {
        "network_dim": next(iter(base_dims)),
        "network_alpha": next(iter(base_alphas)),
        "conv_lora_dim": conv_dim,
        "conv_alpha": conv_alpha,
    }


def _freeze_text_encoder_lora(network) -> None:
    for lora in getattr(network, "text_encoder_loras", []):
        lora.requires_grad_(False)
        for param in lora.parameters():
            param.requires_grad = False

    for lora in getattr(network, "unet_loras", []):
        lora.requires_grad_(True)
        for param in lora.parameters():
            param.requires_grad = True


def _create_network(args, vae, text_encoders, unet, require_text_encoder_lora: bool = False):
    network_module = importlib.import_module(args.network_module)
    net_kwargs = self_distill_cache.parse_network_args(args.network_args)
    init_weights = args.student_init_weights or args.network_weights
    attach_text_encoder = require_text_encoder_lora or _weights_include_text_encoder_lora(init_weights)
    can_create_from_weights = args.dim_from_weights and not (attach_text_encoder and init_weights and not _weights_include_text_encoder_lora(init_weights))
    if can_create_from_weights:
        if init_weights is None:
            raise ValueError("--dim_from_weights requires --student_init_weights or --network_weights.")
        network, _ = network_module.create_network_from_weights(1.0, init_weights, vae, text_encoders, unet, **net_kwargs)
    else:
        inferred_shape = None
        if args.dim_from_weights:
            if init_weights is None:
                raise ValueError("--dim_from_weights requires --student_init_weights or --network_weights.")
            inferred_shape = _infer_network_shape_from_weights(init_weights)
        if "dropout" not in net_kwargs:
            net_kwargs["dropout"] = args.network_dropout
        network_dim = inferred_shape["network_dim"] if inferred_shape is not None else args.network_dim
        network_alpha = inferred_shape["network_alpha"] if inferred_shape is not None else args.network_alpha
        conv_lora_dim = inferred_shape["conv_lora_dim"] if inferred_shape is not None else net_kwargs.get("conv_lora_dim")
        conv_alpha = inferred_shape["conv_alpha"] if inferred_shape is not None else net_kwargs.get("conv_alpha")
        if inferred_shape is not None:
            if conv_lora_dim is not None:
                net_kwargs["conv_lora_dim"] = conv_lora_dim
            if conv_alpha is not None:
                net_kwargs["conv_alpha"] = conv_alpha
        network = network_module.create_network(
            1.0,
            network_dim,
            network_alpha,
            vae,
            text_encoders,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
    network.apply_to(text_encoders, unet, attach_text_encoder, True)
    if init_weights is not None:
        logger.info("load student init weights: %s", init_weights)
        network.load_weights(init_weights)
    lbw_profile.apply_profile_to_network(network, lbw_profile.load_profile(args.lbw_profile))
    _freeze_text_encoder_lora(network)
    return network, net_kwargs


def _expand_optimizer_preset(args) -> None:
    preset = getattr(args, "optimizer_preset", None)
    if not preset:
        return
    preset = preset.lower()
    if preset == "adamw8bit":
        args.optimizer_type = "AdamW8bit"
    elif preset == "adafactor_fixedlr":
        args.optimizer_type = "Adafactor"
        args.optimizer_args = ["relative_step=False", "scale_parameter=False", "warmup_init=False"]
    elif preset == "adagrad8bit":
        args.optimizer_type = "bitsandbytes.optim.Adagrad8bit"
    elif preset == "rmsprop8bit":
        args.optimizer_type = "bitsandbytes.optim.RMSprop8bit"
    else:
        raise ValueError(f"Unsupported optimizer_preset: {args.optimizer_preset}")


def _select_step_tensors(item, device, dtype):
    num_steps = item["target_timesteps"].shape[0]
    index = torch.randint(low=0, high=num_steps, size=(1,)).item()
    timestep = item["target_timesteps"][index].to(device=device)
    x_t = item["x_t"][index].to(device=device, dtype=dtype)
    teacher_target = item["teacher_target"][index].to(device=device, dtype=dtype)
    base_target = item["base_target"][index].to(device=device, dtype=dtype)
    return index, timestep, x_t, teacher_target, base_target


def _save_state_dict(file_path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]]) -> None:
    if os.path.splitext(file_path)[1] == ".safetensors":
        from safetensors.torch import save_file

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(state_dict, file_path, metadata)
        return
    torch.save(state_dict, file_path)


def _save_checkpoint(args, accelerator, network, network_args, global_step):
    model = accelerator.unwrap_model(network)
    model_name = train_util.default_if_none(args.output_name, "self_distill")
    file_path = os.path.join(args.output_dir, f"{model_name}-step{global_step:06d}.safetensors")
    metadata = None if args.no_metadata else self_distill_cache.metadata_for_lora(args, network_args)
    _, save_dtype = train_util.prepare_dtype(args)

    if args.export_te_mode == "preserve":
        model.save_weights(file_path, save_dtype, metadata)
        return file_path

    state_dict = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("lora_te"):
            continue
        filtered_state_dict[key] = value.detach().clone().to("cpu").to(save_dtype)
    _save_state_dict(file_path, filtered_state_dict, metadata)
    return file_path


def train(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    self_distill_cache.ensure_dir(args.output_dir)
    _expand_optimizer_preset(args)
    if args.network_train_text_encoder_only:
        raise ValueError("self-distill v2 currently supports U-Net-only training; --network_train_text_encoder_only is not supported.")

    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, _ = train_util.prepare_dtype(args)
    train_util.verify_command_line_training_args(args)

    header, _ = self_distill_cache.load_manifest_with_header(args.cache_manifest)
    self_distill_cache.validate_manifest_header(header, args)
    if bool(header["teacher_te_included"]) and args.export_te_mode == "drop":
        raise ValueError(
            "export_te_mode=drop is not supported when the cache was built from a teacher that includes Text Encoder LoRA."
        )

    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    text_encoders = [text_encoder1, text_encoder2]
    for text_encoder in text_encoders:
        text_encoder.to("cpu", dtype=torch.float32)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vae.to("cpu")
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.train()
    self_distill_cache.apply_attention_backend(unet, args)

    network, network_args = _create_network(args, vae, text_encoders, unet, require_text_encoder_lora=bool(header["teacher_te_included"]))
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()

    params, _ = network.prepare_optimizer_params(None, args.unet_lr or args.learning_rate, args.learning_rate, active_text_encoder_indices=[])
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, params)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    dataset = self_distill_dataset.SelfDistillDataset(args.cache_manifest, split="train", require_teacher_conditioning=bool(header["teacher_te_included"]))
    required_samples = max(
        len(dataset),
        int(args.max_train_steps)
        * max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        * max(1, int(accelerator.num_processes)),
    )
    sampler = self_distill_sampler.VariantQuotaSampler(
        dataset.entries,
        variant_quota=self_distill_sampler.quota_from_args(args),
        num_samples=required_samples,
        seed=args.seed or 0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=min(args.max_data_loader_n_workers, 2),
        collate_fn=self_distill_dataset.collate_single,
    )

    network, optimizer, dataloader, lr_scheduler = accelerator.prepare(network, optimizer, dataloader, lr_scheduler)
    if accelerator.is_main_process and args.logging_dir:
        tracker_name = args.log_tracker_name or "self_distill_v2"
        accelerator.init_trackers(tracker_name, config=vars(args))

    anchor_targets = None
    if getattr(args, "use_weight_anchor_loss", True):
        unwrapped = accelerator.unwrap_model(network)
        anchor_targets = {
            name: param.detach().clone().to("cpu", dtype=torch.float32)
            for name, param in unwrapped.named_parameters()
            if param.requires_grad
        }

    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="steps")
    for item in dataloader:
        if args.dry_run:
            _, timestep, x_t, teacher_target, base_target = _select_step_tensors(item, accelerator.device, weight_dtype)
            conditioning = self_distill_cache.select_cached_conditioning(item, item["conditioning_source"])
            student_pred = self_distill_cache.unet_predict_cfg(
                unet,
                self_distill_cache.scheduler_from_settings(
                    item["generation_settings"]["sample_sampler"], prediction_type=self_distill_cache.resolve_prediction_type(args)
                ),
                conditioning,
                x_t,
                timestep,
                int(item["generation_settings"]["height"]),
                int(item["generation_settings"]["width"]),
                float(item["generation_settings"]["scale"]),
            )
            _ = self_distill_cache.prediction_to_target(
                student_pred,
                args.prediction_target,
                self_distill_cache.scheduler_from_settings(
                    item["generation_settings"]["sample_sampler"], prediction_type=self_distill_cache.resolve_prediction_type(args)
                ),
                x_t,
                timestep,
            )
            logger.info("dry_run checked record: %s", item["record_id"])
            return

        with accelerator.accumulate(network):
            _, timestep, x_t, teacher_target, base_target = _select_step_tensors(item, accelerator.device, weight_dtype)
            conditioning = self_distill_cache.select_cached_conditioning(item, item["conditioning_source"])
            scheduler = self_distill_cache.scheduler_from_settings(
                item["generation_settings"]["sample_sampler"], prediction_type=self_distill_cache.resolve_prediction_type(args)
            )

            with accelerator.autocast():
                student_pred = self_distill_cache.unet_predict_cfg(
                    unet,
                    scheduler,
                    conditioning,
                    x_t,
                    timestep,
                    int(item["generation_settings"]["height"]),
                    int(item["generation_settings"]["width"]),
                    float(item["generation_settings"]["scale"]),
                )
                student_target = self_distill_cache.prediction_to_target(student_pred, args.prediction_target, scheduler, x_t, timestep)
                loss, loss_logs = self_distill_losses.compute_self_distill_loss(
                    student_target,
                    teacher_target,
                    base_target,
                    item["variant_type"],
                    item["loss_role"],
                    args,
                    network=accelerator.unwrap_model(network),
                    anchor_targets=anchor_targets,
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
    parser.add_argument("--teacher_lora_weights", type=str, default=None)
    parser.add_argument("--student_init_weights", type=str, default=None)
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--network_module", type=str, default="networks.lora")
    parser.add_argument("--network_dim", type=int, default=8)
    parser.add_argument("--network_alpha", type=float, default=1.0)
    parser.add_argument("--network_dropout", type=float, default=None)
    parser.add_argument("--network_args", type=str, nargs="*", default=None)
    parser.add_argument("--network_train_unet_only", action="store_true")
    parser.add_argument("--network_train_text_encoder_only", action="store_true")
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--unet_lr", type=float, default=None)
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--lbw_profile", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prediction_target", type=str, choices=["eps", "v"], default="eps")
    parser.add_argument("--xt_source_mode", type=str, choices=["teacher_rollout"], default="teacher_rollout")
    parser.add_argument("--export_te_mode", type=str, choices=["preserve", "drop"], default="preserve")
    parser.add_argument("--optimizer_preset", type=str, default="adamw8bit")
    parser.add_argument("--attention_backend", type=str, choices=["auto", "sdpa", "xformers"], default="auto")
    parser.add_argument("--timestep_sampling_mode", type=str, choices=["uniform", "late_bias", "custom"], default="uniform")
    parser.add_argument("--variant_quota", type=str, default="")
    parser.add_argument("--use_keep_delta_loss", action="store_true", default=True)
    parser.add_argument("--use_suppress_to_base_loss", action="store_true", default=True)
    parser.add_argument("--use_weight_anchor_loss", action="store_true", default=True)
    parser.add_argument("--use_coarse_preservation_loss", action="store_true")
    parser.add_argument("--use_high_pass_delta_loss", action="store_true")
    parser.add_argument("--use_low_pass_delta_loss", action="store_true")
    parser.add_argument("--use_sparse_loss", action="store_true")
    parser.add_argument("--keep_delta_loss_weight", type=float, default=1.0)
    parser.add_argument("--suppress_to_base_loss_weight", type=float, default=1.0)
    parser.add_argument("--weight_anchor_loss_weight", type=float, default=0.05)
    parser.add_argument("--coarse_preservation_loss_weight", type=float, default=0.0)
    parser.add_argument("--high_pass_delta_loss_weight", type=float, default=0.0)
    parser.add_argument("--low_pass_delta_loss_weight", type=float, default=0.0)
    parser.add_argument("--sparse_loss_weight", type=float, default=0.0)
    parser.add_argument("--high_pass_mode", type=str, default="dog", choices=["dog", "laplacian", "gaussian_residual"])
    parser.add_argument("--low_pass_mode", type=str, default="avg", choices=["avg", "gaussian", "identity"])
    parser.add_argument("--coarse_target_mode", type=str, default="teacher", choices=["base", "teacher"])
    parser.add_argument("--per_variant_loss_weight", type=str, default="")
    parser.add_argument("--per_block_anchor_weight", type=str, default="")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train(args)
