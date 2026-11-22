"""SDXL networks.lora settings shared by the training entry point and migration tools.

Only explicit custom CLI/TOML values are put in the argparse namespace. This
lets the resolver distinguish omitted values from explicit false and zero.
"""

import argparse
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    value_type: type
    default: object
    description: str
    choices: tuple = ()
    negatable: bool = False


SCOPES = ("unet", "te", "both")
AUTO_PRESETS = ("default", "clip_rate_high", "clip_rate_high_narrow", "clip_rate_mid", "clip_rate_low", "clip_rate_low_auto")
OPTIONS = {
    "text_encoder_lr1": Option(float, None, "SDXL TE1 base learning rate"),
    "text_encoder_lr2": Option(float, None, "SDXL TE2 base learning rate"),
    "te_mlp_fc_only": Option(bool, False, "Train TE MLP fc1/fc2 LoRA layers"),
    "downscale_freq_shift": Option(bool, False, "Use SDXL diffusion timestep frequency shift 1"),
    "fp16_safe_norms_mode": Option(str, "off", "SDXL UNet normalization policy", ("off", "strict")),
    "grad_norm_mode": Option(str, None, "Guardian gradient update policy", ("stable", "stable_no_threshoff", "gamble")),
    "avg_cp": Option(bool, False, "Enable epoch checkpoint averaging"),
    "avg_cp_mode": Option(str, "live", "Epoch averaging application mode", ("live", "promote")),
    "avg_window": Option(int, 4, "Number of checkpoints to average"),
    "avg_begin": Option(float, 0.6, "Epoch fraction at which averaging begins"),
    "avg_mode": Option(str, "ema", "Checkpoint averaging weights", ("ema", "uniform")),
    "avg_shadow_bank_size": Option(int, 12, "Number of fixed proxy scoring batches"),
    "avg_shadow_margin": Option(float, 0.003, "Required relative loss improvement for promote"),
    "avg_shadow_patience": Option(int, 2, "Required winning streak for promote"),
    "avg_reset_stats": Option(bool, True, "Reset supported optimizer moments after averaging", negatable=True),
    "avg_save_final_raw": Option(bool, False, "Also save raw final weights", negatable=True),
    "rank_log": Option(bool, False, "Record LoRA rank statistics"),
    "rank_log_mode": Option(str, "per_module", "Rank observation granularity", ("per_module",)),
    "group_loss_log": Option(bool, False, "Record training sample and group losses"),
    "dq_delta_bits": Option(int, None, "Delta quantization bits"),
    "dq_delta_mode": Option(str, "stoch", "Delta rounding mode", ("stoch",)),
    "dq_delta_granularity": Option(str, "channel", "Delta quantization granularity", ("channel",)),
    "dq_delta_stat": Option(str, "rms", "Delta scale statistic", ("rms",)),
    "dq_delta_range_mul": Option(float, 3.0, "Base delta range multiplier"),
    "dq_delta_scope": Option(str, "both", "Quantized components", SCOPES),
    "dq_delta_begin_after_lr_warmup": Option(bool, False, "Activate quantization after LR warmup"),
    "dq_delta_begin": Option(float, 0.0, "Scheduled attempt fraction for quantization start"),
    "dq_delta_range_mul_attn2": Option(float, None, "UNet attn2 range multiplier"),
    "dq_delta_range_mul_te": Option(float, None, "Common text encoder range multiplier"),
    "dq_delta_range_mul_te1": Option(float, None, "TE1 range multiplier override"),
    "dq_delta_range_mul_te2": Option(float, None, "TE2 range multiplier override"),
    "dq_delta_bits_sched": Option(str, None, "Comma-separated progress:bits schedule"),
    "dq_delta_use_triton": Option(bool, False, "Use eligible Triton quantization kernels"),
    "dq_delta_triton_stats": Option(bool, False, "Use eligible fused Triton statistics"),
    "dq_delta_log": Option(bool, False, "Record delta statistics"),
    "dq_delta_log_every": Option(int, 100, "Delta statistics observation interval"),
    "dq_delta_log_scope": Option(str, None, "Components observed by the delta log", SCOPES),
    "dq_delta_log_mode": Option(str, "summary", "Delta statistics aggregation", ("summary",)),
    "dq_delta_log_detail": Option(str, "basic", "Delta statistics detail level", ("basic",)),
    "dq_delta_auto_range_mul": Option(bool, False, "Adapt the quantization range multiplier"),
    "dq_delta_auto_scope": Option(str, None, "Components used to decide multiplier changes", SCOPES),
    "dq_delta_auto_preset": Option(str, None, "Automatic multiplier target band", AUTO_PRESETS),
    "dq_delta_auto_every": Option(int, 50, "Automatic multiplier observation interval"),
    "dq_delta_auto_use_raw": Option(bool, False, "Include raw clip rate in decisions"),
    "dq_delta_auto_init_range_mul_from_band": Option(bool, False, "Initialize multiplier from the target band"),
    "dq_delta_auto_warmup": Option(bool, True, "Warm up the automatic controller", negatable=True),
    "dq_delta_auto_warmup_updates": Option(int, 0, "Number of auto updates in warmup (0 selects automatically)"),
    "dq_delta_qerr_per_clip_floor": Option(float, 0.001, "Floor for the quantization error per clip denominator"),
    "report_after_training": Option(bool, False, "Build the diagnostic report after training"),
}


def add_training_arguments(parser):
    group = parser.add_argument_group("SDXL LoRA extensions")
    for key, option in OPTIONS.items():
        kwargs = {"default": argparse.SUPPRESS, "help": option.description}
        if option.value_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction if option.negatable else "store_true"
        else:
            kwargs["type"] = option.value_type
            if option.choices:
                kwargs["choices"] = option.choices
        group.add_argument("--" + key, **kwargs)


# These keys are known old options. Other tools and dataset TOMLs have their own schemas.
_RETIRED_GROUPS = {
    "use common/individual TE learning rates; both encoders are selected by default": (
        "network_te_train_targets", "te1_lr_warmup_steps", "te2_lr_warmup_steps", "te1_freeze_at", "te2_freeze_at", "te_lr_after",
    ),
    "use --fp16_safe_norms_mode strict": ("fp16_safe_norms",),
    "choose a --grad_norm_mode preset": (
        "skip_grad_norm", "skip_grad_norm_max", "grad_norm_log", "nan_to_window", "inf_to_window", "skip_nan_immediate",
        "skip_inf_immediate", "grad_cosine_log",
    ),
    "promote scores the configured averaging method; results are saved in the run": (
        "avg_promote_pick", "avg_shadow_log_jsonl", "avg_save_last_candidates",
    ),
    "rank statistics use a fixed interval of 100 attempts": ("rank_log_every",),
    "use --output_dir; each log is stored in its run folder": ("rank_log_file", "dq_delta_log_file", "dq_delta_auto_log_file"),
    "use --group_loss_log; sample logging and flushing are managed per run": (
        "group_loss_ema_beta", "group_loss_log_every_n_steps", "group_loss_epoch_summary",
    ),
    "use the upstream noise offset controls": (
        "noise_offset_random_min_ratio", "noise_offset_random_max_ratio", "noise_offset_random_min_ratio_sched",
        "noise_offset_random_max_ratio_sched",
    ),
    "use delta bits and direct range multiplier options": ("dq_delta_policy_file", "dq_delta_step", "dq_quantize_z"),
    "use the summary/basic delta log": ("dq_delta_log_extra",),
    "choose --dq_delta_auto_preset; controller constants and log format are internal": (
        "dq_delta_auto_clip_low", "dq_delta_auto_clip_high", "dq_delta_auto_mul_up", "dq_delta_auto_mul_down", "dq_delta_auto_min",
        "dq_delta_auto_max", "dq_delta_auto_ema", "dq_delta_auto_log_format", "dq_delta_clip_rate_low_auto_min_progress",
        "dq_delta_clip_rate_low_auto_bad_streak", "dq_delta_clip_rate_low_auto_freeze_progress", "dq_delta_clip_rate_low_auto_qerr_ratio",
        "dq_delta_clip_rate_low_auto_qerr_per_clip",
    ),
    "post-update LoRA weight rounding is outside this migration": ("round_lora_step", "round_lora_mode", "round_lora_every", "round_lora_begin"),
}
RETIRED = {key: reason for reason, keys in _RETIRED_GROUPS.items() for key in keys}

# Add a feature only after its training entry point has been connected and verified.
WIRED_FEATURES = frozenset({"dq_triton", "optimizer_bnb", "optimizer_fast", "averaging", "te_lr", "te_mlp_fc_only", "downscale_freq_shift", "strict_norms", "dq", "dq_auto", "guardian", "rank_log", "group_loss_log", "dq_log", "report_after_training"})


@dataclass
class TrainingConfig:
    output_dir: str | None
    requested: dict
    values: dict
    training_features: frozenset
    observation_features: frozenset
    text_encoder_lrs: tuple | None
    bits_schedule: tuple

    @property
    def changes_training(self):
        return bool(self.training_features)


def reject_retired(name, source):
    key = name.removeprefix("--").removeprefix("no-").replace("-", "_")
    if key in RETIRED:
        raise ValueError(f"{source}: {name} is retired; {RETIRED[key]}")


def _value(name, value, option):
    kind = option.value_type
    if kind is float and type(value) in (float, int):
        value = float(value)
    if type(value) is not kind:
        raise ValueError(f"--{name}: expected {kind.__name__}, got {value!r}")
    if kind is float and not math.isfinite(value):
        raise ValueError(f"--{name}: expected a finite number, got {value!r}")
    if option.choices and value not in option.choices:
        raise ValueError(f"--{name}={value!r}: choose from {', '.join(option.choices)}")
    return value


def _bits_schedule(spec):
    if not spec:
        return ()
    result = []
    for segment in spec.split(","):
        if not segment.strip():
            continue
        try:
            fraction, number = segment.strip().split(":")
            progress, bits = float(fraction), int(number)
            if not 0 <= progress <= 1 or bits < 2:
                raise ValueError
        except ValueError:
            raise ValueError(f"--dq_delta_bits_sched: invalid segment {segment!r}, expected progress 0..1 and bits >= 2") from None
        result.append((progress, bits))
    return tuple(sorted(result, key=lambda entry: entry[0]))


def _te_lrs(args, values):
    if not (values["te_mlp_fc_only"] or values["text_encoder_lr1"] is not None or values["text_encoder_lr2"] is not None):
        return None
    common = getattr(args, "text_encoder_lr", None)
    rates = common if isinstance(common, (list, tuple)) else ([] if common is None else [common])
    if len(rates) > 2:
        raise ValueError(f"--text_encoder_lr: expected at most 2 values, got {rates!r}")
    checked = [_value("text_encoder_lr", lr, OPTIONS["text_encoder_lr1"]) for lr in rates]
    base = getattr(args, "learning_rate", None) if not checked else checked[0]
    pair = list(checked) if len(checked) == 2 else [base, base]
    for i, key in enumerate(("text_encoder_lr1", "text_encoder_lr2")):
        override = values[key]
        if override is not None:
            if len(checked) == 2 and override != pair[i]:
                raise ValueError(f"--text_encoder_lr {pair} conflicts with --{key}={override}")
            pair[i] = override
    if any(rate is not None and rate < 0 for rate in pair):
        raise ValueError(f"text encoder learning rates must be nonnegative, got {pair}")
    return tuple(pair)


def _validate_ranges(v):
    for key in ("avg_begin", "dq_delta_begin"):
        if not 0 <= v[key] <= 1:
            raise ValueError(f"--{key}: expected a fraction in [0,1], got {v[key]}")
    for key in ("avg_window", "avg_shadow_bank_size", "avg_shadow_patience", "dq_delta_log_every", "dq_delta_auto_every",
                "dq_delta_qerr_per_clip_floor", "dq_delta_range_mul", "dq_delta_range_mul_attn2", "dq_delta_range_mul_te",
                "dq_delta_range_mul_te1", "dq_delta_range_mul_te2"):
        if v[key] is not None and v[key] <= 0:
            raise ValueError(f"--{key}: expected a positive value, got {v[key]}")
    for key in ("avg_shadow_margin", "dq_delta_auto_warmup_updates"):
        if v[key] < 0:
            raise ValueError(f"--{key}: expected a nonnegative value, got {v[key]}")
    if v["dq_delta_bits"] is not None and v["dq_delta_bits"] < 2:
        raise ValueError(f"--dq_delta_bits: expected at least 2, got {v['dq_delta_bits']}")


def _guard_training(args, features):
    if not features:
        return
    for key in ("deepspeed", "resume", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk", "train_inpainting"):
        if getattr(args, key, None):
            raise ValueError(f"--{key} is unsupported with: {', '.join(sorted(features))}")
    initial_step = getattr(args, "initial_step", None)
    if initial_step is not None and initial_step != 0:
        raise ValueError(f"--initial_step={initial_step}: custom training must begin at step 0")
    if initial_step is None and getattr(args, "initial_epoch", None) not in (None, 1):
        raise ValueError(f"--initial_epoch={args.initial_epoch}: custom training must begin at epoch 1")
    if getattr(args, "gradient_accumulation_steps", 1) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("custom training requires one process and --gradient_accumulation_steps 1")


def resolve_training_config(args):
    """Resolve the SDXL entry point's namespace without mutating its values or using RNG."""
    if getattr(args, "network_module", None) != "networks.lora":
        return None
    for key in vars(args):
        reject_retired(key, getattr(args, "config_file", None) or "training arguments")
    requested = {key: getattr(args, key) for key in OPTIONS if hasattr(args, key)}
    values = {key: _value(key, requested[key], opt) if key in requested else opt.default for key, opt in OPTIONS.items()}
    _validate_ranges(values)
    if values["dq_delta_begin_after_lr_warmup"]:
        warmup = getattr(args, "lr_warmup_steps", None)
        if type(warmup) not in (int, float) or not 0 < warmup < math.inf:
            raise ValueError("--dq_delta_begin_after_lr_warmup requires a finite --lr_warmup_steps > 0")
    schedule = _bits_schedule(values["dq_delta_bits_sched"])
    has_dq = values["dq_delta_bits"] is not None or bool(schedule)
    overrides = [k for k in OPTIONS if k.startswith("dq_delta_range_mul_") and values[k] is not None]
    if overrides and (not has_dq or schedule or values["dq_delta_auto_range_mul"]):
        raise ValueError(f"--{overrides[0]} requires fixed --dq_delta_bits without a schedule or automatic multiplier")
    if values["dq_delta_auto_range_mul"] and not has_dq:
        raise ValueError("--dq_delta_auto_range_mul requires --dq_delta_bits or --dq_delta_bits_sched")
    if values["dq_delta_log_scope"] is None:
        values["dq_delta_log_scope"] = values["dq_delta_scope"]
    if values["dq_delta_auto_scope"] is None:
        te_only = getattr(args, "network_train_text_encoder_only", False)
        unet_only = getattr(args, "network_train_unet_only", False)
        values["dq_delta_auto_scope"] = "te" if te_only or (not unet_only and values["dq_delta_scope"] == "te") else "unet"
    if values["dq_delta_auto_range_mul"] and values["dq_delta_auto_preset"] is None:
        values["dq_delta_auto_preset"] = "default"

    te_lrs = _te_lrs(args, values)
    opt_name = (getattr(args, "optimizer_type", "") or "").lower()
    if not opt_name and getattr(args, "use_8bit_adam", False):
        opt_name = "adamw8bit"
    if values["avg_cp"] and opt_name not in ("adamw8bit", "adamw8bitfast", "adamwbnb"):
        raise ValueError("--avg_cp requires AdamW8bit, AdamW8bitFast or AdamWBnb")
    training = {
        "te_lr": values["text_encoder_lr1"] is not None or values["text_encoder_lr2"] is not None,
        "te_mlp_fc_only": values["te_mlp_fc_only"], "downscale_freq_shift": values["downscale_freq_shift"],
        "strict_norms": values["fp16_safe_norms_mode"] == "strict", "guardian": values["grad_norm_mode"] is not None,
        "averaging": values["avg_cp"], "dq": has_dq, "dq_auto": values["dq_delta_auto_range_mul"],
        "dq_triton": has_dq and values["dq_delta_use_triton"], "optimizer_fast": opt_name == "adamw8bitfast",
        "optimizer_bnb": opt_name == "adamwbnb",
    }
    observations = {"rank_log": values["rank_log"], "group_loss_log": values["group_loss_log"],
                    "dq_log": values["dq_delta_log"], "report_after_training": values["report_after_training"]}
    active = frozenset(key for key, enabled in training.items() if enabled)
    _guard_training(args, active)
    return TrainingConfig(
        output_dir=getattr(args, "output_dir", None),
        requested=requested,
        values=values,
        training_features=active,
        observation_features=frozenset(key for key, enabled in observations.items() if enabled),
        text_encoder_lrs=te_lrs,
        bits_schedule=schedule,
    )


def require_wired_features(config):
    if config is None:
        return
    missing = (config.training_features | config.observation_features) - WIRED_FEATURES
    if missing:
        options = ", ".join("--" + key for key in sorted(config.requested))
        raise ValueError(f"SDXL LoRA features not yet connected in this revision: {', '.join(sorted(missing))}. Requested: {options}")
