"""Small, best-effort settings snapshots. No tensor access or training RNG use."""
import json
import logging
import math
import os
from pathlib import Path
import re
import uuid

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
_SECRET = re.compile(r"(?:^|_)(?:api_key|access_token|auth_token|huggingface_token|hf_token|password|secret|authorization)(?:$|_)", re.I)


def snapshot(value, depth=0):
    """Copy JSON-shaped config only; never repr()/item() arbitrary objects."""
    if depth > 24:
        return {"unrecorded": "nesting_limit"}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"unrecorded": "nonfinite"}
    if isinstance(value, str):
        # Some CLI collections contain key=value strings rather than dictionaries.
        key, separator, _ = value.partition("=")
        return key + "=[redacted]" if separator and _SECRET.search(key.lstrip("-")) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): "[redacted]" if _SECRET.search(str(k)) else snapshot(v, depth + 1)
                for k, v in value.items() if isinstance(k, (str, int))}
    if isinstance(value, (list, tuple)):
        return [snapshot(v, depth + 1) for v in value]
    return {"unrecorded": "non_json_value"}


def snapshot_metadata(metadata):
    """Redact known structured metadata while retaining its string-based format."""
    result = snapshot(metadata)
    for key in ("ss_network_args", "ss_datasets", "ss_dataset_dirs", "ss_reg_dataset_dirs", "ss_bucket_info", "ss_tag_frequency"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or raw == "None":
            continue
        try:
            parsed = json.loads(raw)
            cleaned = snapshot(parsed)
            if cleaned != parsed:
                result[key] = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)
        except (ValueError, RecursionError):
            # Do not copy a malformed structured field whose secrets cannot be checked.
            result[key] = "[unrecorded: invalid structured metadata]"
    return result


def dataset_batch_settings(datasets, num_processes, accumulation_steps):
    """Record configured dataset batches, not the CLI default or observed tail batches."""
    try:
        return [dict(dataset_index=i, batch_size_per_device=dataset.batch_size,
                     nominal_effective_batch_size=dataset.batch_size * num_processes * accumulation_steps)
                for i, dataset in enumerate(datasets)]
    except Exception as exc:
        logger.warning("Dataset batch settings unavailable (%s).", type(exc).__name__)
        return {"unrecorded": "dataset_batch_sizes"}


def optimizer_groups(optimizer, descriptions=None):
    """Capture group options, excluding parameters and optimizer state."""
    try:
        return [dict(index=i, label=(descriptions[i] if descriptions and i < len(descriptions) else f"group {i}"),
                     options=snapshot({k: v for k, v in group.items() if k != "params"}))
                for i, group in enumerate(optimizer.param_groups)]
    except Exception as exc:
        logger.warning("Optimizer settings unavailable (%s).", type(exc).__name__)
        return {"unrecorded": "optimizer_groups"}



def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def start_record(output_dir, output_name, session_id, started_at, requested):
    """Called on the main process only, after accelerator initialization."""
    try:
        run_id = str(uuid.uuid4())
        directory = Path(output_dir) / "run_records" / run_id
        manifest = dict(schema_version=SCHEMA_VERSION, kind="training_settings", run_id=run_id,
                        output_name=output_name, session_id=str(session_id), training_started_at=str(started_at),
                        settings_status="requested", requested_args="inputs/requested_args.json",
                        resolved_config="inputs/resolved_config.json")
        _write(directory / "manifest.json", manifest)
        _write(directory / manifest["requested_args"], dict(schema_version=SCHEMA_VERSION, run_id=run_id, args=requested))
        logger.info("training settings: %s", directory)
        return directory, manifest
    except Exception as exc:
        logger.warning("Training settings could not be saved (%s); training continues.", type(exc).__name__)
        return None


def finish_record(record, args, optimizer, descriptions, created_groups, metadata, runtime):
    """Mark settings resolved, NOT training completed. Failure must not stop training."""
    if record is None:
        return
    try:
        directory, manifest = record
        resolved = dict(schema_version=SCHEMA_VERSION, run_id=manifest["run_id"],
                        args=snapshot(vars(args)), runtime=snapshot(runtime),
                        optimizer_groups_created=created_groups,
                        optimizer_groups_at_start=optimizer_groups(optimizer, descriptions),
                        metadata=snapshot_metadata(metadata))
        _write(directory / manifest["resolved_config"], resolved)
        _write(directory / "manifest.json", {**manifest, "settings_status": "resolved"})
    except Exception as exc:
        logger.warning("Resolved training settings could not be saved (%s); training continues.", type(exc).__name__)
