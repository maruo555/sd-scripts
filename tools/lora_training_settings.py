"""Read settings sidecars or legacy checkpoint headers without loading tensors."""
import json
import math
from pathlib import Path
import struct

MAX_JSON = 16 * 1024 * 1024
# Conservative conversion of known scalar/structured metadata. Unknown keys stay text.
BOOL_KEYS = set("gradient_checkpointing full_fp16 v2 cache_latents lowram zero_terminal_snr debiased_estimation scale_v_pred_loss_like_noise_pred".split())
NUMBER_KEYS = set("learning_rate text_encoder_lr unet_lr network_dim network_alpha network_dropout seed num_epochs max_train_steps gradient_accumulation_steps lr_warmup_steps clip_skip max_token_length max_grad_norm min_snr_gamma noise_offset multires_noise_iterations multires_noise_discount adaptive_noise_scale prior_loss_weight num_train_images num_reg_images num_batches_per_epoch batch_size epoch steps".split())
JSON_KEYS = set("datasets dataset_dirs reg_dataset_dirs bucket_info network_args tag_frequency".split())
ARTIFACT_KEYS = set("ss_epoch ss_steps ss_training_finished_at sshs_model_hash sshs_legacy_hash".split())


def _reject_constant(value):
    raise ValueError("nonfinite JSON")


def _loads(data):
    return json.loads(data, parse_constant=_reject_constant)


def _read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError("JSON exceeds 16 MiB")
    value = _loads(data)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("safetensors header required")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated header length")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > MAX_JSON or size > path.stat().st_size - 8:
            raise ValueError("invalid header length")
        header = _loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("invalid header object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("invalid metadata object")
    return metadata


def normalize_metadata(metadata):
    values = {}
    for full_key, raw in metadata.items():
        if not full_key.startswith("ss_") or full_key in ARTIFACT_KEYS:
            continue
        key = full_key[3:]
        value = raw
        if raw == "None":
            value = None
        elif key in BOOL_KEYS and raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        elif key in NUMBER_KEYS:
            try:
                parsed = _loads(raw)
                if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) and math.isfinite(parsed):
                    value = parsed
            except (ValueError, OverflowError):
                pass
        elif key in JSON_KEYS:
            try:
                value = _loads(raw)
            except ValueError:
                pass
        values[key] = value
    return values


def _payload(directory, relative, run_id):
    if not isinstance(relative, str):
        raise ValueError("missing settings path")
    target = (directory / relative).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError("settings path leaves run directory")
    data = _read_json(target)
    if data.get("schema_version") != 1 or data.get("run_id") != run_id:
        raise ValueError("settings version or run ID mismatch")
    if not isinstance(data.get("args"), dict):
        raise ValueError("settings args must be an object")
    return data


def load_training_settings(input_dir, base_name, model_path, manifest_path=None):
    """Select a whole source. Never fill sidecar gaps from checkpoint metadata."""
    result = dict(schema_version=1, status="unavailable", source="none", values={}, notes=[],
                  checkpoint={"path": str(model_path), "status": "unavailable", "metadata": {}})
    metadata = {}
    try:
        metadata = read_metadata(model_path)
        result["checkpoint"].update(status="read", metadata={k: v for k, v in metadata.items() if k in ARTIFACT_KEYS})
    except (OSError, ValueError, OverflowError) as exc:
        result["checkpoint"]["error"] = type(exc).__name__
    identity = (metadata.get("ss_session_id"), metadata.get("ss_training_started_at"))
    has_identity = all(identity)
    names = {base_name, metadata.get("ss_output_name")}
    candidates, unreadable = [], []
    if manifest_path:
        paths = [Path(manifest_path)]
    else:
        try:
            paths = sorted((Path(input_dir) / "run_records").glob("*/manifest.json"))
        except OSError:
            paths = []
            unreadable.append("run_records")
    for path in paths:
        try:
            manifest = _read_json(path)
            if manifest_path or manifest.get("output_name") in names or (
                has_identity and (manifest.get("session_id"), manifest.get("training_started_at")) == identity
            ):
                candidates.append((path, manifest))
        except (OSError, ValueError) as exc:
            unreadable.append(str(path))
    if has_identity and not manifest_path:
        matched = [(p, m) for p, m in candidates if (m.get("session_id"), m.get("training_started_at")) == identity]
        if matched:
            candidates = matched
    if not candidates:
        if unreadable or manifest_path:
            result.update(status="error", source="record")
            result["notes"].append("設定記録を確認できません。メタデータへの自動切替は行いません。")
            result["unreadable_records"] = unreadable
        elif metadata:
            result.update(status="partial", source="metadata", source_path=str(model_path),
                          values=normalize_metadata(metadata), raw_metadata=metadata)
            result["notes"].append("過去のメタデータから取得。未記録の設定は不明です。")
        else:
            result["notes"].append("学習設定の記録がありません。")
        return result
    result["source"] = "record"
    if len(candidates) != 1:
        result.update(status="ambiguous", candidates=[str(p) for p, _ in candidates])
        result["notes"].append("同名の設定記録が複数あり特定できません。--training_settings でmanifest.jsonを指定してください。")
        return result
    path, manifest = candidates[0]
    result.update(source_path=str(path), run_id=manifest.get("run_id"))
    record_identity = (manifest.get("session_id"), manifest.get("training_started_at"))
    if has_identity and record_identity != identity:
        result["status"] = "mismatch"
        result["notes"].append("設定記録とチェックポイントのsession ID・開始時刻が一致しません。")
        return result
    result["association"] = "session_metadata" if has_identity else "explicit" if manifest_path else "unique_output_name"
    if not has_identity:
        result["notes"].append("チェックポイントとの対応は未検証です（session情報なし）。")
    try:
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "training_settings" or not isinstance(manifest.get("run_id"), str):
            raise ValueError("unsupported manifest")
        requested = _payload(path.parent, manifest.get("requested_args"), manifest["run_id"])
        result["requested_args"] = requested["args"]
        if manifest.get("settings_status") == "requested":
            result.update(status="partial", values=requested["args"], value_stage="requested")
            result["notes"].append("引数のみの記録です。初期化後の確定値はありません。")
        elif manifest.get("settings_status") == "resolved":
            resolved = _payload(path.parent, manifest.get("resolved_config"), manifest["run_id"])
            result.update(status="recorded", values=resolved["args"], value_stage="initialized", resolved=resolved)
        else:
            raise ValueError("unsupported settings status")
    except (OSError, ValueError) as exc:
        result.update(status="error", values={}, error=type(exc).__name__)
        result["notes"].append("設定記録が不完全または未対応です。メタデータでは補完しません。")
    return result
