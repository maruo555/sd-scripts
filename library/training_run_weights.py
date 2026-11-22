"""Publish retained LoRA files and label the exact training state they represent."""
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class WeightMoment:
    role: str
    epoch: int
    attempt_step: int
    batch_index: int | None
    phase: str
    epoch_complete: bool | None = None


def weight_args(args, run):
    if run is None:
        return args
    local = copy.copy(args)
    local.output_dir = str(run.directory)
    local.output_name = None if run.output_name_omitted else run.base_name
    return local


def resolve_epoch_save_args(args, run, num_train_epochs):
    """Resolve the cadence before redirecting checkpoint writes to the run."""
    if args.save_n_epoch_ratio is not None and args.save_n_epoch_ratio > 0:
        args.save_every_n_epochs = max(1, num_train_epochs // args.save_n_epoch_ratio)
    return weight_args(args, run)


def sample_args(args, run):
    if run is None:
        return args
    local = copy.copy(args)
    local.output_dir = str(run.output_root)
    local.output_name = run.run_id if run.output_name_omitted else f"{run.base_name}_{run.run_id}"
    return local


def record_weight(run, path, moment):
    relative = run.relative_path(path)
    key = asdict(moment)
    def register(index):
        for item in index["weights"]:
            if item["path"] == relative and all(item.get(k) == v for k, v in key.items()):
                item["status"] = "saved"
                break
        else:
            index["weights"].append({**key, "path": relative, "status": "saved"})
    run.update_index(register)


def publish_weight(run, name, moment, serialize):
    if Path(name).name != name:
        raise ValueError(f"Expected a checkpoint filename: {name}")
    relative = run.relative_path(name)
    path = run.directory / relative
    with tempfile.NamedTemporaryFile(prefix=".weight-", suffix=path.suffix, dir=run.directory, delete=False) as stream:
        temp = Path(stream.name)
    try:
        serialize(str(temp))
        with temp.open("rb+") as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                raise ValueError(f"Empty serialized weight: {name}")
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    record_weight(run, path, moment)
    return str(path)


def remove_weight(run, name):
    if Path(name).name != name:
        raise ValueError(f"Expected a checkpoint filename: {name}")
    relative = run.relative_path(name)
    path = run.directory / relative
    try:
        path.unlink()
        status = "pruned"
    except FileNotFoundError:
        status = "missing"
    def update(index):
        for item in index["weights"]:
            if item["path"] == relative and item["status"] == "saved":
                item["status"] = status
    run.update_index(update)
    return status
