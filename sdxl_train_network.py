import argparse

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from library import sdxl_model_util, sdxl_train_util, train_util
import train_network
import library.maruo_global_config as maruoCfg
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        if args.token_gate:
            assert (
                args.token_gate_tokens and args.token_gate_tokens.strip()
            ), "--token_gate_tokens is required when --token_gate is enabled / --token_gate使用時は--token_gate_tokensを指定してください"
            tokens = [tok.strip() for tok in args.token_gate_tokens.split(",") if tok.strip()]
            assert (
                len(tokens) > 0
            ), "token list must contain at least one entry / ゲート対象トークンが1つ以上必要です"
            assert (
                not args.cache_text_encoder_outputs
            ), "token gate cannot be combined with cached text encoder outputs / トークンゲートはテキストエンコーダー出力キャッシュと併用できません"
            assert (
                0.0 <= args.token_drop_prob < 1.0
            ), "token_drop_prob must be in [0,1) / token_drop_probは0以上1未満で指定してください"
            assert 0.0 <= args.anchor_ratio <= 1.0, "anchor_ratio must be in [0,1] / anchor_ratioは0から1の範囲で指定してください"
            assert 0.0 <= args.neg_gate_ratio <= 1.0, "neg_gate_ratio must be in [0,1] / neg_gate_ratioは0から1の範囲で指定してください"
            assert (
                args.anchor_ratio + args.neg_gate_ratio <= 1.0 + 1e-6
            ), "anchor_ratio + neg_gate_ratio must not exceed 1 / anchor_ratioとneg_gate_ratioの合計は1を超えられません"
            if args.neg_gate_ratio > 0:
                assert (
                    args.neg_gate_bit
                ), "neg_gate_ratio requires --neg_gate_bit / neg_gate_ratioを使う場合は--neg_gate_bitを有効にしてください"
            assert (
                args.token_gate_l1 >= 0.0
            ), "token_gate_l1 must be non-negative / token_gate_l1は0以上にしてください"
            args.token_gate_tokens_list = tokens
        else:
            if args.anchor_ratio > 0 or args.neg_gate_ratio > 0 or args.token_drop_prob > 0:
                logger.warning(
                    "token gate is disabled; anchor_ratio, neg_gate_ratio and token_drop_prob are ignored / トークンゲート無効時はanchor_ratio, neg_gate_ratio, token_drop_probは無視されます"
                )
            args.anchor_ratio = 0.0
            args.neg_gate_ratio = 0.0
            args.token_drop_prob = 0.0
            args.token_gate_tokens_list = []

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument(
        "--downscale_freq_shift",
        action="store_true",
        help="sdxl_original_unet.py の get_timestep_embedding() で downscale_freq_shift=1 にする（通常は0)",
    )
    parser.add_argument(
        "--te_mlp_fc_only",
        action="store_true",
        help="Enable TE-MLP-FC-only training",
    )
    parser.add_argument(
        "--fp16_safe_norms",
        action="store_true",
        help="Compute reduction ops (LayerNorm/GroupNorm/Softmax) in fp32 while keeping weights/other ops in fp16.",
    )
    # presence-based token gate options
    parser.add_argument(
        "--token_gate",
        action="store_true",
        help="Enable token-based gating for LoRA (SDXL presence T-LoRA mode)",
    )
    parser.add_argument(
        "--token_gate_tokens",
        type=str,
        default=None,
        help="Comma separated list of character tokens used for gating",
    )
    parser.add_argument(
        "--token_gate_scope",
        type=str,
        default="cross_attn",
        choices=["cross_attn"],
        help="Scope of modules affected by the gate",
    )
    parser.add_argument(
        "--token_gate_dim",
        type=str,
        default="head",
        choices=["head"],
        help="Granularity of gating (currently head only)",
    )
    parser.add_argument(
        "--token_gate_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight applied to gate parameters",
    )
    parser.add_argument(
        "--token_drop_prob",
        type=float,
        default=0.0,
        help="Probability to drop character tokens from captions during positive steps",
    )
    parser.add_argument(
        "--anchor_ratio",
        type=float,
        default=0.0,
        help="Ratio of training steps run as anchor distillation",
    )
    parser.add_argument(
        "--neg_gate_bit",
        action="store_true",
        help="Enable the negative gate control bit",
    )
    parser.add_argument(
        "--neg_gate_ratio",
        type=float,
        default=0.0,
        help="Ratio of training steps where the negative gate bit is activated",
    )
    parser.add_argument(
        "--anchor_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the anchor/negative distillation loss",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    # map CLI options to global config
    maruoCfg.downscale_freq_shift = bool(getattr(args, "downscale_freq_shift", False))
    maruoCfg.te_mlp_fc_only = bool(getattr(args, "te_mlp_fc_only", False))
    maruoCfg.fp16_safe_norms = bool(getattr(args, "fp16_safe_norms", False))
    logger.info(
        f"fp16_safe_norms is {'enabled' if maruoCfg.fp16_safe_norms else 'disabled'}"
    )

    trainer = SdxlNetworkTrainer()
    trainer.train(args)
