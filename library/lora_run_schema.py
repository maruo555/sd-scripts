"""SDXL LoRA run CSV v1. Both the trainer and offline readers use this definition."""
import math
import re

VERSION = 1
HEADERS = {
    "gradient": "epoch,batch_index,grad_norm,threshold,loss,grad_scale",
    "dq": ("epoch,attempt_step,scope,target,bits,common_mul_after,module,input_rms,input_abs_max,quant_range,"
           "clip_rate_raw,controller_clip_rate_ema,zero_rate,quant_error_rms_raw,log_scope_error_rms_ema,"
           "quant_error_ratio_raw,log_scope_error_ratio_ema,active_clip_band,active_clip_low,active_clip_high,"
           "train_progress,numel,input_shape,scale_min,scale_max,auto_state_step"),
    "dq_auto": ("event_type,epoch,attempt_step,scope,target,bits,clip_rate_raw,clip_rate_ema,common_mul_before,"
                "common_mul_after,auto_applied,warmup_active,warmup_remaining_after,auto_reason,qerr_per_clip,"
                "active_clip_band,active_clip_low,active_clip_high,low_auto_state,low_auto_decision,low_auto_reason,"
                "low_auto_bad,low_auto_bad_streak,train_progress,low_auto_phase,low_auto_can_escape,"
                "auto_quant_error_ratio_raw,auto_quant_error_ratio_ema,low_auto_phase_after"),
    "rank": ("epoch,attempt_step,scope,module,rank_dim,rank_sat,rank_top1,rank_energy,unet_lr_min,unet_lr_max,"
             "te1_lr_min,te1_lr_max,te2_lr_min,te2_lr_max"),
    "promote": ("row_type,epoch,status,selected_candidate_mode,raw_proxy_loss,selected_proxy_loss,margin_ok,"
                "win_streak,promote_applied,optimizer_reset_applied,construction_pool_size,bank_size,"
                "bank_unique_class_tokens,bank_max_per_class_tokens,bank_unique_image_keys,bank_max_per_image_key,"
                "bank_timestep_low_count,bank_timestep_mid_count,bank_timestep_high_count,window_count,"
                "window_position,window_epoch"),
    "group_loss": "epoch,attempt_step,group,subset_index,timestep,bucket_reso,loss,group_ema_loss,group_observation_count",
}
HEADERS = {kind: tuple(header.split(",")) for kind, header in HEADERS.items()}
_INT = set(("epoch batch_index attempt_step bits numel auto_state_step warmup_remaining_after low_auto_bad_streak "
            "rank_dim win_streak construction_pool_size bank_size bank_unique_class_tokens bank_max_per_class_tokens "
            "bank_unique_image_keys bank_max_per_image_key bank_timestep_low_count bank_timestep_mid_count "
            "bank_timestep_high_count window_count window_position window_epoch subset_index timestep group_observation_count").split())
_STR = set(("scope target module active_clip_band input_shape event_type auto_reason low_auto_state low_auto_decision "
            "low_auto_reason low_auto_phase low_auto_phase_after row_type status selected_candidate_mode group bucket_reso").split())
_BOOL = set(("auto_applied warmup_active low_auto_bad low_auto_can_escape margin_ok promote_applied optimizer_reset_applied").split())
_OTHER_TYPES = _BOOL | _STR | _INT
_SPECIAL = {"NaN": math.nan, "+Inf": math.inf, "-Inf": -math.inf}
_INTEGER = re.compile(r"-?[0-9]+\Z")


def encode_cell(name, value):
    if value is None:
        return ""
    if name in _STR and isinstance(value, str):
        return value
    if name in _BOOL and isinstance(value, bool):
        return "true" if value else "false"
    if name in _INT and type(value) is int:
        return str(value)
    if name not in _OTHER_TYPES and type(value) in (int, float):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "+Inf" if number > 0 else "-Inf"
        return repr(number)
    raise ValueError(f"{name} has invalid value {value!r}")


def decode_cell(name, value):
    if value == "":
        return None
    if name in _STR:
        return value
    if name in _BOOL:
        if value in ("true", "false"):
            return value == "true"
    elif name in _INT:
        if _INTEGER.fullmatch(value):
            return int(value)
    else:
        if value in _SPECIAL:
            return _SPECIAL[value]
        number = float(value)
        if math.isfinite(number):
            return number
    raise ValueError(f"Invalid {name}: {value!r}")
