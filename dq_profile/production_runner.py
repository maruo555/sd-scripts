from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dq_profile.production_cli import ResolvedProfileRequest
from dq_profile.protocol import (
    inspect_dataset_config,
    loader_visible_image_files,
    resolve_dataset_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
PROFILE_SCRIPT = REPO_ROOT / "sdxl_dq_dataset_profile.py"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "lora_output" / "dq_dataset_profiler"
RUN_SCHEMA_VERSION = "1.1-beta"
EXECUTION_PLAN_SCHEMA_VERSION = "1.0-beta"
INVALID_FILENAME = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def console_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe)
    sys.stdout.flush()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _windows_ansi_encoding() -> str:
    # Accelerate relays worker output through a text-mode subprocess pipe. On
    # Windows, native helpers can still emit the active ANSI code page even
    # when Python UTF-8 mode is enabled. Keep every Python layer and the outer
    # reader on that code page so one native line cannot terminate Accelerate's
    # background reader thread.
    import ctypes

    code_page = int(ctypes.windll.kernel32.GetACP())
    return f"cp{code_page}" if code_page > 0 else "mbcs"


def subprocess_text_environment(
    base_environment: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
) -> tuple[dict[str, str], str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        encoding = _windows_ansi_encoding()
        environment["PYTHONUTF8"] = "0"
        environment["PYTHONIOENCODING"] = f"{encoding}:replace"
    else:
        encoding = "utf-8"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8:replace"
    return environment, encoding


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def git_tracked_state() -> dict[str, Any]:
    status = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    diff = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "-c",
            "diff.algorithm=myers",
            "diff",
            "--binary",
            "--full-index",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--no-indent-heuristic",
            "HEAD",
            "--",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    combined = hashlib.sha256()
    combined.update(b"status\0")
    combined.update(status)
    combined.update(b"\0diff\0")
    combined.update(diff)
    return {
        "contract": "tracked-status-and-head-binary-diff-v1",
        "dirty": bool(status),
        "untracked_files": "excluded",
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "state_sha256": combined.hexdigest(),
    }


def sanitize_profile_name(value: str) -> str:
    name = INVALID_FILENAME.sub("_", str(value).strip()).strip(" .")
    name = re.sub(r"\s+", "_", name)
    if not name or name in {".", ".."}:
        name = "dq_profile"
    if name.split(".", 1)[0].upper() in WINDOWS_RESERVED:
        name = f"dq_{name}"
    return name[:80].rstrip(" .") or "dq_profile"


def source_dirs_from_dataset_config(
    dataset_config: Path,
    *,
    train_batch_size: int = 1,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    min_bucket_reso: int = 384,
    max_bucket_reso: int = 1024,
    bucket_reso_steps: int = 64,
    minimum_source_groups: int = 1,
) -> tuple[Path, ...]:
    layout = resolve_dataset_layout(
        dataset_config,
        train_batch_size=train_batch_size,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        bucket_reso_steps=bucket_reso_steps,
        dataset_repeats=1,
        cache_info=False,
        minimum_source_groups=minimum_source_groups,
        require_resolution=True,
    )
    return tuple(subset.image_dir for subset in layout.subsets)


def _inventory(
    source_dir: Path,
    *,
    visible_images: Sequence[Path] | None = None,
) -> list[dict[str, Any]]:
    visible_images = (
        loader_visible_image_files(source_dir)
        if visible_images is None
        else tuple(visible_images)
    )
    visible_keys = {
        os.path.normcase(os.path.normpath(str(path)))
        for path in visible_images
    }
    rows: list[dict[str, Any]] = []
    # DreamBoothDataset uses non-recursive glob_images(image_dir, "*"). Keep
    # the source inventory and preflight image count on exactly that topology.
    for path in sorted(source_dir.iterdir(), key=lambda item: (str(item).casefold(), str(item))):
        if not path.is_file():
            continue
        rows.append(
            {
                "name": str(path.relative_to(source_dir)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "is_image": os.path.normcase(os.path.normpath(str(path))) in visible_keys,
            }
        )
    return rows


def build_source_map(source_dirs: Sequence[Path], *, dataset_key: str) -> tuple[list[dict[str, Any]], int]:
    payload: list[dict[str, Any]] = []
    for index, source_dir in enumerate(source_dirs, start=1):
        visible_images = loader_visible_image_files(source_dir)
        files = _inventory(source_dir, visible_images=visible_images)
        source_image_count = len(visible_images)
        if source_image_count == 0:
            nested_image_count = sum(
                len(loader_visible_image_files(path))
                for path in source_dir.rglob("*")
                if path.is_dir()
            )
            if nested_image_count:
                raise ValueError(
                    "canonical-v1 follows the training loader's non-recursive image discovery; "
                    f"{source_dir} has {nested_image_count} image files only in descendant directories. "
                    "Move them to the image_dir root or declare each child directory as a separate subset."
                )
            raise ValueError(f"dataset image_dir has no loader-visible image files: {source_dir}")
        payload.append(
            {
                "pattern": str(source_dir.resolve()) + os.sep,
                "source_group": f"{dataset_key}-source-{index:02d}",
                "match": "prefix",
                "directory": str(source_dir.resolve()),
                "files": files,
                "image_count": source_image_count,
            }
        )
    image_count = sum(int(row["image_count"]) for row in payload)
    if image_count < 8:
        raise ValueError(
            "canonical-v1 requires at least 8 non-recursive loader-visible images; "
            f"found {image_count}"
        )
    return payload, image_count


def _apply_source_probe_selection(
    source_map_payload: Sequence[Mapping[str, Any]],
    *,
    probe_budget: int,
) -> list[dict[str, Any]]:
    """Annotate which source groups must be represented in the local probe.

    The complete source inventory remains in the source map and therefore in
    the source contract. When there are more independent image_dir groups than
    probe images, only the probe requirement is reduced. Evenly spaced indices
    cover the full deterministic TOML/source order instead of taking a biased
    prefix.
    """

    total = len(source_map_payload)
    budget = int(probe_budget)
    if total <= 0:
        raise ValueError("source probe selection requires at least one source group")
    if budget <= 0:
        raise ValueError("source probe selection requires a positive probe budget")
    selected_count = min(total, budget)
    if selected_count == total:
        selected_indices = list(range(total))
        policy = "all_source_groups"
    elif selected_count == 1:
        selected_indices = [0]
        policy = "deterministic_evenly_spaced_source_groups_v1"
    else:
        denominator = selected_count - 1
        selected_indices = [
            (rank * (total - 1) + denominator // 2) // denominator
            for rank in range(selected_count)
        ]
        policy = "deterministic_evenly_spaced_source_groups_v1"
    if len(set(selected_indices)) != selected_count:
        raise RuntimeError("deterministic source probe selection produced duplicate indices")
    rank_by_index = {index: rank for rank, index in enumerate(selected_indices)}
    return [
        {
            **dict(row),
            "probe_selected": index in rank_by_index,
            "probe_selection_rank": rank_by_index.get(index),
            "probe_selection_policy": policy,
        }
        for index, row in enumerate(source_map_payload)
    ]


def _source_probe_selection_summary(
    source_map_payload: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(source_map_payload)
    selected = sum(bool(row.get("probe_selected", True)) for row in source_map_payload)
    policy = next(
        (
            str(row.get("probe_selection_policy"))
            for row in source_map_payload
            if row.get("probe_selection_policy")
        ),
        "all_source_groups" if selected == total else "custom_partial_source_groups",
    )
    return {
        "source_group_count_total": total,
        "source_group_count_probed": selected,
        "source_group_count_omitted": total - selected,
        "source_group_coverage_complete": selected == total,
        "source_group_coverage_fraction": float(selected / total) if total else 1.0,
        "source_group_selection_policy": policy,
    }


def validate_source_map_inventory(source_map: Path) -> None:
    """Refuse to mix worker stages that observed different dataset bytes."""

    payload = json.loads(source_map.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"source group map must contain a non-empty list: {source_map}")
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"source group map row {index} is not an object: {source_map}")
        source_dir = Path(str(row.get("directory", ""))).expanduser().resolve()
        if not source_dir.is_dir():
            raise RuntimeError(
                "dataset source inventory changed after preflight: "
                f"source directory is unavailable: {source_dir}"
            )
        visible_images = loader_visible_image_files(source_dir)
        current = {
            "files": _inventory(source_dir, visible_images=visible_images),
            "image_count": len(visible_images),
        }
        expected = {
            "files": row.get("files"),
            "image_count": row.get("image_count"),
        }
        if current != expected:
            source_group = row.get("source_group", f"row-{index}")
            raise RuntimeError(
                "dataset source inventory changed after preflight; refusing to mix "
                "worker stages with stale provenance: "
                f"source_group={source_group!r}, "
                f"expected_sha256={canonical_sha256(expected)}, "
                f"current_sha256={canonical_sha256(current)}"
            )


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def validate_output_base(
    output_base: Path,
    *,
    source_dirs: Sequence[Path],
    normal_output_dir: Path | None,
) -> Path:
    output_base = output_base.expanduser().resolve()
    if output_base.parent == output_base or output_base == PROJECT_ROOT.resolve():
        raise ValueError(f"diagnostic output base is too broad: {output_base}")
    forbidden = (REPO_ROOT, REPO_ROOT / ".git", REPO_ROOT / "venv")
    if any(_is_within(output_base, item) for item in forbidden):
        raise ValueError(f"diagnostic output must stay outside the repository/.git/venv: {output_base}")
    for source_dir in source_dirs:
        if _is_within(output_base, source_dir) or _is_within(source_dir, output_base):
            raise ValueError(f"diagnostic output must not overlap dataset image_dir: {source_dir}")
    if normal_output_dir is not None and output_base == normal_output_dir.resolve():
        raise ValueError(
            "diagnostic output base cannot be the exact normal checkpoint output directory; "
            "use its dq_dataset_profiler child or another directory"
        )
    output_base.mkdir(parents=True, exist_ok=True)
    probe = output_base / ".dq_profile_write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_base


def _model_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"pretrained model path was not found: {path}")
    path = path.resolve()
    stat = path.stat()
    common = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file():
        return {
            **common,
            "kind": "file",
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        model_files = [item for item in path.rglob("*") if item.is_file()]
        model_files.sort(
            key=lambda item: (
                item.relative_to(path).as_posix().casefold(),
                item.relative_to(path).as_posix(),
            )
        )
        inventory = []
        for item in model_files:
            item_stat = item.stat()
            inventory.append(
                {
                    "name": item.relative_to(path).as_posix(),
                    "size": item_stat.st_size,
                    "sha256": sha256_file(item),
                }
            )
        if not inventory:
            raise ValueError(f"pretrained model directory contains no files: {path}")
        return {
            **common,
            "kind": "directory",
            "file_count": len(inventory),
            "total_file_size": sum(int(row["size"]) for row in inventory),
            "inventory_sha256": canonical_sha256(inventory),
        }
    raise ValueError(f"pretrained model path is neither a file nor a directory: {path}")


def validate_model_identity(model_path: Path, protocol_fingerprint: Path) -> None:
    payload = read_json(protocol_fingerprint)
    expected = payload.get("model")
    if not isinstance(expected, Mapping):
        raise ValueError(
            f"protocol fingerprint has no model identity: {protocol_fingerprint}"
        )
    current = _model_identity(model_path)
    if current != expected:
        raise RuntimeError(
            "pretrained model changed after protocol fingerprinting; refusing to mix "
            "worker stages with stale provenance: "
            f"expected_sha256={canonical_sha256(expected)}, "
            f"current_sha256={canonical_sha256(current)}"
        )


def build_protocol_fingerprint(
    request: ResolvedProfileRequest,
    *,
    source_map_payload: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    code_files = (
        Path(__file__).resolve(),
        REPO_ROOT / "dq_profile" / "production_cli.py",
        REPO_ROOT / "dq_profile" / "production_preset.py",
        PROFILE_SCRIPT,
        REPO_ROOT / "tools" / "analyze_dq_v24_local.py",
        REPO_ROOT / "tools" / "check_dq_calibration_gate.py",
    )
    tracked_state = git_tracked_state()
    training_contract = request.preset.contract()
    local_contract = request.local_measurement.contract()
    execution_contract = request.execution_mode.contract()
    model_identity = _model_identity(request.model_path)
    dataset_contract = {
        "path": str(request.dataset_config),
        "sha256": sha256_file(request.dataset_config),
        "source_map_sha256": canonical_sha256(source_map_payload),
    }
    shared_local_contract = {
        "training": training_contract,
        "local_measurement": local_contract,
        "model": model_identity,
        "dataset": dataset_contract,
        "profile_seed": int(request.preset.expected_explicit["seed"]),
        "scope": "local_body_tail_only",
    }
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "git_head": git_head(),
        "git_tracked_dirty": tracked_state["dirty"],
        "git_tracked_state": tracked_state,
        "preset": training_contract,
        "local_measurement_contract": local_contract,
        "execution_mode": execution_contract,
        "qa_depth": request.execution_mode.qa_depth,
        "internal_profile_level": request.execution_mode.internal_profile_level,
        "model": model_identity,
        "dataset_config": dataset_contract,
        "source_map_sha256": dataset_contract["source_map_sha256"],
        "contract_hashes": {
            "training_contract_sha256": canonical_sha256(training_contract),
            "dataset_contract_sha256": canonical_sha256(dataset_contract),
            "local_measurement_contract_sha256": canonical_sha256(local_contract),
            "execution_qa_contract_sha256": canonical_sha256(execution_contract),
            "local_probe_contract_sha256": canonical_sha256(shared_local_contract),
            "shared_local_contract_sha256": canonical_sha256(shared_local_contract),
        },
        "code": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in code_files
        },
        "scope": "local_body_tail_only",
        "trajectory": "research_only_not_run",
        "not_quality_or_utility": True,
    }
    fingerprint_sha256 = canonical_sha256(payload)
    return {
        **payload,
        "fingerprint_sha256": fingerprint_sha256,
        "full_protocol_fingerprint_sha256": fingerprint_sha256,
    }


def _local_probe_counts(
    *,
    probe_images: int,
    timestep_bins: int,
    mul_count: int,
    no_quant_noise_replicas: int,
    candidate_noise_replicas: int,
    stochastic_quant_repeats: int,
) -> dict[str, int]:
    points = int(probe_images) * int(timestep_bins)
    no_quant = points * int(no_quant_noise_replicas)
    per_candidate = (
        points
        * int(candidate_noise_replicas)
        * int(stochastic_quant_repeats)
    )
    candidates = int(mul_count) * per_candidate
    return {
        "probe_images": int(probe_images),
        "timestep_bins": int(timestep_bins),
        "mul_count": int(mul_count),
        "no_quant_probes": no_quant,
        "candidate_probes_per_mul": per_candidate,
        "candidate_probes": candidates,
        "total_local_probes": no_quant + candidates,
    }


def build_execution_plan(
    request: ResolvedProfileRequest,
    *,
    image_count: int,
    probe_budget: int,
    source_group_count: int | None = None,
    probed_source_group_count: int | None = None,
    source_group_selection_policy: str = "all_source_groups",
) -> dict[str, Any]:
    """Describe deterministic work volume before any GPU worker starts.

    The timing section is deliberately secondary.  It extrapolates one known
    RTX 5080 run and is not a promise: bucket shape, cache state, storage,
    driver/library versions, and the warmup boundary can all change runtime.
    """

    preset = request.preset
    local = request.local_measurement
    execution = request.execution_mode
    expected = preset.expected_explicit
    preflight_summary = inspect_dataset_config(
        request.dataset_config,
        max_train_epochs=int(expected["max_train_epochs"]),
        max_train_steps=None,
        lr_warmup_steps=float(expected["lr_warmup_steps"]),
        branch_steps_override=int(local.sweep_steps),
        max_images=int(local.max_images),
        timestep_bins=int(local.timestep_bins),
        stochastic_repeats=int(local.stochastic_repeats),
        train_batch_size=int(expected["train_batch_size"]),
        enable_bucket=bool(expected["enable_bucket"]),
        bucket_no_upscale=bool(expected["bucket_no_upscale"]),
        min_bucket_reso=int(expected["min_bucket_reso"]),
        max_bucket_reso=int(expected["max_bucket_reso"]),
        bucket_reso_steps=int(expected["bucket_reso_steps"]),
        dataset_repeats=1,
        cache_info=False,
        minimum_source_groups=int(local.minimum_source_groups),
        require_resolution=True,
    )
    if int(preflight_summary.unique_images) != int(image_count):
        raise RuntimeError(
            "dataset image inventory disagrees between source-map and execution-plan "
            f"preflight: source_map={image_count}, protocol={preflight_summary.unique_images}"
        )

    core_counts = _local_probe_counts(
        probe_images=probe_budget,
        timestep_bins=local.timestep_bins,
        mul_count=len(execution.core_grid),
        no_quant_noise_replicas=local.no_quant_noise_replicas,
        candidate_noise_replicas=local.candidate_noise_replicas,
        stochastic_quant_repeats=local.stochastic_quant_repeats,
    )
    minimum_processes, maximum_processes = execution.gpu_process_count_range
    warmup = int(preflight_summary.dq_begin_step)

    # Historical calibration from one 13-image/420-warmup RTX 5080 run.
    # Keep these constants in the artifact so the estimate is auditable.
    reference_warmup_seconds = 247.0 * warmup / 420.0
    reference_prefix_update_seconds = 1033.0 / 512.0
    reference_local_probe_seconds = 1115.0 / 1196.0

    def estimated_local_seconds(mul_count: int) -> float:
        counts = _local_probe_counts(
            probe_images=probe_budget,
            timestep_bins=local.timestep_bins,
            mul_count=mul_count,
            no_quant_noise_replicas=local.no_quant_noise_replicas,
            candidate_noise_replicas=local.candidate_noise_replicas,
            stochastic_quant_repeats=local.stochastic_quant_repeats,
        )
        return (
            reference_warmup_seconds
            + counts["total_local_probes"] * reference_local_probe_seconds
        )

    base_seconds = (
        execution.standalone_snapshot_count * reference_warmup_seconds
        + reference_warmup_seconds
        + execution.prefix_branch_updates * reference_prefix_update_seconds
        + estimated_local_seconds(len(execution.core_grid))
    )
    maximum_seconds = base_seconds
    projected_edge_mul_counts: list[int] = []
    for round_index in range(1, execution.max_edge_extension_rounds + 1):
        # Strict historically adds one adjacent edge point per round.  The
        # actual conditional plan is reported separately and can stop early.
        mul_count = len(execution.core_grid) + round_index
        projected_edge_mul_counts.append(mul_count)
        maximum_seconds += estimated_local_seconds(mul_count)

    estimate_minutes = {
        "minimum": round(base_seconds / 60.0, 1),
        "maximum_if_all_edge_rounds_run": round(maximum_seconds / 60.0, 1),
    }
    exact_process_count = (
        minimum_processes if minimum_processes == maximum_processes else None
    )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "execution_mode": execution.name,
        "qa_depth": execution.qa_depth,
        "internal_profile_level": execution.internal_profile_level,
        "measurement_contract": local.name,
        "work_volume": {
            "gpu_process_count": exact_process_count,
            "gpu_process_count_min": minimum_processes,
            "gpu_process_count_max": maximum_processes,
            "warmup_boundary_updates": warmup,
            "warmup_executions": exact_process_count,
            "warmup_executions_min": minimum_processes,
            "warmup_executions_max": maximum_processes,
            "total_warmup_updates": (
                warmup * exact_process_count
                if exact_process_count is not None
                else None
            ),
            "total_warmup_updates_min": warmup * minimum_processes,
            "total_warmup_updates_max": warmup * maximum_processes,
            "standalone_snapshot_count": execution.standalone_snapshot_count,
            "snapshot_replica_count": execution.standalone_snapshot_count,
            "prefix": {
                "candidates": ["no_quant", f"mul_{execution.prefix_anchor_mul:.2f}"],
                "short_run_a_updates": execution.prefix_short_steps,
                "short_run_b_updates": execution.prefix_short_steps,
                "long_run_updates": execution.prefix_long_steps,
                "checkpoints": list(execution.prefix_checkpoints),
                "total_branch_updates": execution.prefix_branch_updates,
            },
            "local": {
                **core_counts,
                "fixed_grid": list(execution.core_grid),
                "edge_extension": execution.edge_policy,
                "maximum_edge_extension_rounds": execution.max_edge_extension_rounds,
                "projected_mul_counts_if_each_edge_round_adds_one": projected_edge_mul_counts,
            },
            "dataset": {
                "unique_images": int(preflight_summary.unique_images),
                "repeat_weighted_samples": int(preflight_summary.repeat_weighted_samples),
                "steps_per_epoch": int(preflight_summary.steps_per_epoch),
                "normal_training_steps": int(preflight_summary.normal_training_steps),
                "probe_images": int(probe_budget),
                "source_group_count_total": (
                    None if source_group_count is None else int(source_group_count)
                ),
                "source_group_count_probed": (
                    None
                    if probed_source_group_count is None
                    else int(probed_source_group_count)
                ),
                "source_group_coverage_complete": (
                    None
                    if source_group_count is None or probed_source_group_count is None
                    else int(source_group_count) == int(probed_source_group_count)
                ),
                "source_group_selection_policy": source_group_selection_policy,
            },
        },
        "reference_time_estimate": {
            "minutes": estimate_minutes,
            "is_guarantee": False,
            "basis": "single historical RTX 5080 run with 13 probe images and warmup boundary 420",
            "calibration": {
                "warmup_seconds_at_420_updates": 247.0,
                "prefix_seconds_per_branch_update": reference_prefix_update_seconds,
                "local_seconds_per_probe": reference_local_probe_seconds,
            },
            "important_variability": [
                "bucket resolutions and aspect ratios",
                "latent cache state and storage speed",
                "GPU, driver, CUDA, PyTorch, bitsandbytes, and Triton versions",
                "dataset repeat weighting, which changes the warmup boundary",
                "conditional edge stages in strict mode",
            ],
        },
        "scope": "numerical Local Body/Tail Safety/Fidelity; not final image quality or Utility",
    }
    return {**payload, "execution_plan_sha256": canonical_sha256(payload)}


def allocate_run_directory(output_base: Path, profile_name: str, fingerprint: str) -> Path:
    profile_root = output_base / sanitize_profile_name(profile_name)
    profile_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{fingerprint[:10]}"
    for index in range(1, 100):
        suffix = "" if index == 1 else f"_{index:02d}"
        candidate = profile_root / f"{stem}{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a unique diagnostic run directory under {profile_root}")


def is_complete(path: Path) -> bool:
    status_path = path / "status.json"
    return status_path.is_file() and read_json(status_path).get("status") == "complete"


class Launcher:
    def __init__(self, run_dir: Path, *, dry_run: bool = False) -> None:
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.log_path = run_dir / "run.log"

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        console_write(line + "\n")
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def run(self, command: Sequence[str], *, label: str) -> None:
        rendered = subprocess.list2cmdline([str(item) for item in command])
        self.log(f"RUN {label}: {rendered}")
        if self.dry_run:
            return
        environment, stream_encoding = subprocess_text_environment()
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=stream_encoding,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log_stream:
            for line in process.stdout:
                console_write(line)
                log_stream.write(line)
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"{label} failed with exit code {return_code}")
        self.log(f"DONE {label}")


@dataclass(frozen=True)
class ProductionRunOptions:
    output_base: Path = DEFAULT_OUTPUT_BASE
    profile_name: str | None = None
    preflight_only: bool = False
    dry_run: bool = False
    open_report: bool = False


@dataclass(frozen=True)
class ProductionRunResult:
    run_dir: Path
    status: str
    report: Path | None


def stage_names() -> dict[str, str]:
    return {
        "snapshot_a": "00a_snapshot_a",
        "snapshot_b": "00b_snapshot_b",
        "snapshot_parity": "00c_snapshot_parity.json",
        "prefix": "00d_prefix",
        "prefix_snapshot_parity": "00e_prefix_snapshot_parity.json",
        "prefix_source_gate": "00f_prefix_source_gate.json",
        "core_raw": "01_core_local_raw",
        "core_analysis": "02_core_local_analysis",
    }


def edge_names(round_index: int) -> dict[str, str]:
    return {
        "raw": f"{2 + round_index * 3:02d}_edge{round_index}_local_raw",
        "analysis": f"{3 + round_index * 3:02d}_edge{round_index}_local_analysis",
        "parity": f"{4 + round_index * 3:02d}_edge{round_index}_parity.json",
    }


def accelerate_prefix(preset: Any) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes=1",
        "--num_machines=1",
        "--num_cpu_threads_per_process",
        str(preset.num_cpu_threads_per_process),
        str(PROFILE_SCRIPT),
    ]


def profile_command(
    request: ResolvedProfileRequest,
    *,
    run_dir: Path,
    source_map: Path,
    name: str,
    protocol: str,
    range_muls: Sequence[float],
    max_images: int,
    gate: Path | None = None,
    snapshot_only: bool = False,
) -> list[str]:
    preset = request.preset
    local = request.local_measurement
    execution = request.execution_mode
    command = [
        *accelerate_prefix(preset),
        f"--pretrained_model_name_or_path={request.model_path}",
        f"--dataset_config={request.dataset_config}",
        f"--output_dir={run_dir / '_internal_training_output'}",
        f"--output_name={request.output_name}",
        *preset.training_tokens,
        (
            f"--training_comment=DQ Profiler Beta {preset.name} "
            f"{execution.name} Local-only"
        ),
        f"--dq_profile_output_dir={run_dir}",
        f"--dq_profile_name={name}",
        f"--dq_profile_level={execution.internal_profile_level}",
        f"--dq_profile_protocol={protocol}",
        f"--dq_profile_execution_mode={execution.name}",
        f"--dq_profile_qa_depth={execution.qa_depth}",
        f"--dq_profile_measurement_contract={local.name}",
        f"--dq_profile_sampling_depth={local.sampling_depth}",
        f"--dq_profile_confidence_ceiling={local.confidence_ceiling}",
        "--dq_profile_prefix_kernel_mode=deterministic",
        f"--dq_profile_prefix_short_steps={execution.prefix_short_steps}",
        f"--dq_profile_prefix_long_steps={execution.prefix_long_steps}",
        f"--dq_profile_max_images={max_images}",
        f"--dq_profile_timestep_bins={local.timestep_bins}",
        f"--dq_profile_stochastic_repeats={local.stochastic_repeats}",
        "--dq_profile_range_muls=" + ",".join(f"{value:.2f}" for value in range_muls),
        f"--dq_profile_sweep_steps={local.sweep_steps}",
        f"--dq_profile_branch_repeats={local.branch_repeats}",
        "--dq_profile_guardian_ablation=common_only",
        f"--dq_profile_sketch_width={local.sketch_width}",
        f"--dq_profile_sketch_seeds={local.sketch_seeds}",
        f"--dq_profile_source_group_map={source_map}",
    ]
    if gate is not None:
        command.append(f"--dq_profile_prefix_gate_file={gate}")
    if snapshot_only:
        command.append("--dq_profile_snapshot_only")
    return command


def run_profile(
    launcher: Launcher,
    request: ResolvedProfileRequest,
    *,
    source_map: Path,
    name: str,
    protocol: str,
    range_muls: Sequence[float],
    max_images: int,
    gate: Path | None = None,
    snapshot_only: bool = False,
) -> Path:
    validate_model_identity(
        request.model_path,
        launcher.run_dir / "protocol_fingerprint.json",
    )
    # The subprocess reads the original image/caption/cache paths, not copies.
    # Re-hash them immediately before every worker so a long staged run cannot
    # silently combine bytes that differ from source_group_map.json.
    validate_source_map_inventory(source_map)
    output = launcher.run_dir / name
    if output.exists():
        raise RuntimeError(f"stage output already exists and will not be overwritten: {output}")
    launcher.run(
        profile_command(
            request,
            run_dir=launcher.run_dir,
            source_map=source_map,
            name=name,
            protocol=protocol,
            range_muls=range_muls,
            max_images=max_images,
            gate=gate,
            snapshot_only=snapshot_only,
        ),
        label=f"{protocol} / {name}",
    )
    if not launcher.dry_run and not is_complete(output):
        raise RuntimeError(f"profile returned success but stage status is not complete: {output}")
    return output


def run_snapshot_parity(launcher: Launcher, left: Path, right: Path, output_name: str) -> Path:
    output = launcher.run_dir / output_name
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.check_dq_snapshot_parity",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output-json",
            str(output),
        ],
        label=f"snapshot parity {left.name} vs {right.name}",
    )
    if not launcher.dry_run and read_json(output).get("passed") is not True:
        raise RuntimeError(f"snapshot parity failed: {output}")
    return output


def run_local_analysis(
    launcher: Launcher,
    *,
    request: ResolvedProfileRequest,
    profile: Path,
    output_name: str,
    dataset_id: str,
) -> Path:
    output = launcher.run_dir / output_name
    if output.exists():
        raise RuntimeError(f"analysis output already exists and will not be overwritten: {output}")
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.analyze_dq_v24_local",
            "--profile-dir",
            str(profile),
            "--output-dir",
            str(output),
            "--dataset-id",
            dataset_id,
            "--iterations",
            str(request.local_measurement.bootstrap_iterations),
            "--seed",
            str(request.local_measurement.bootstrap_seed),
        ],
        label=f"Local Body/Tail analysis / {output_name}",
    )
    if not launcher.dry_run and not is_complete(output):
        raise RuntimeError(f"Local analysis did not complete: {output}")
    return output


def update_status(run_dir: Path, *, status: str, current_stage: str, **extra: Any) -> None:
    existing = read_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    write_json(
        run_dir / "status.json",
        {
            **existing,
            "schema_version": RUN_SCHEMA_VERSION,
            "status": status,
            "current_stage": current_stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "not_quality_or_utility": True,
            **extra,
        },
    )


def run_local_pipeline(
    launcher: Launcher,
    request: ResolvedProfileRequest,
    *,
    source_map: Path,
    probe_budget: int,
    dataset_id: str,
) -> tuple[Path, Path, tuple[float, ...], int]:
    names = stage_names()
    execution = request.execution_mode
    update_status(launcher.run_dir, status="running", current_stage="snapshot_a")
    snapshot_a = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["snapshot_a"],
        protocol="v2-prefix-smoke",
        range_muls=(execution.prefix_anchor_mul,),
        max_images=execution.snapshot_a_max_images,
        snapshot_only=True,
    )
    snapshot_reference = snapshot_a
    if execution.standalone_snapshot_count >= 2:
        update_status(launcher.run_dir, status="running", current_stage="snapshot_b")
        snapshot_b = run_profile(
            launcher,
            request,
            source_map=source_map,
            name=names["snapshot_b"],
            protocol="v2-prefix-smoke",
            range_muls=(execution.prefix_anchor_mul,),
            max_images=probe_budget,
            snapshot_only=True,
        )
        run_snapshot_parity(
            launcher, snapshot_a, snapshot_b, names["snapshot_parity"]
        )
        snapshot_reference = snapshot_b

    update_status(launcher.run_dir, status="running", current_stage="prefix_gate")
    prefix = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["prefix"],
        protocol="v2-prefix-smoke",
        range_muls=(execution.prefix_anchor_mul,),
        max_images=probe_budget,
    )
    run_snapshot_parity(
        launcher,
        snapshot_reference,
        prefix,
        names["prefix_snapshot_parity"],
    )
    gate = prefix / "calibration_gate.json"
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.check_dq_calibration_gate",
            "--gate",
            str(gate),
            "--dataset-config",
            str(request.dataset_config),
            "--source-group-map",
            str(source_map),
            "--repo-root",
            str(REPO_ROOT),
            "--output-json",
            str(launcher.run_dir / names["prefix_source_gate"]),
        ],
        label="prefix/source contract gate",
    )

    update_status(launcher.run_dir, status="running", current_stage="core_local")
    active_grid = tuple(float(value) for value in execution.core_grid)
    active_profile = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["core_raw"],
        protocol="v24-acceptance-local",
        range_muls=active_grid,
        max_images=probe_budget,
        gate=gate,
    )
    active_analysis = run_local_analysis(
        launcher,
        request=request,
        profile=active_profile,
        output_name=names["core_analysis"],
        dataset_id=dataset_id,
    )
    if launcher.dry_run:
        return active_profile, active_analysis, active_grid, 0

    selection = read_json(active_analysis / "local_selection.json")
    completed_edge_rounds = 0
    for round_index in range(1, execution.max_edge_extension_rounds + 1):
        if selection.get("selection_valid") is not True:
            break
        additions = tuple(float(value) for value in selection.get("edge_extension_recommended", ()))
        expanded_grid = tuple(sorted(set((*active_grid, *additions))))
        if not additions or expanded_grid == active_grid:
            break
        round_names = edge_names(round_index)
        update_status(
            launcher.run_dir,
            status="running",
            current_stage=f"edge_extension_{round_index}",
            measured_grid=list(expanded_grid),
        )
        edge_profile = run_profile(
            launcher,
            request,
            source_map=source_map,
            name=round_names["raw"],
            protocol="v24-acceptance-local",
            range_muls=expanded_grid,
            max_images=probe_budget,
            gate=gate,
        )
        edge_analysis = run_local_analysis(
            launcher,
            request=request,
            profile=edge_profile,
            output_name=round_names["analysis"],
            dataset_id=dataset_id,
        )
        launcher.run(
            [
                sys.executable,
                "-m",
                "tools.check_dq_v24_local_extension_parity",
                "--core-profile-dir",
                str(active_profile),
                "--extension-profile-dir",
                str(edge_profile),
                "--common-muls",
                ",".join(f"{value:.2f}" for value in active_grid),
                "--output-json",
                str(launcher.run_dir / round_names["parity"]),
            ],
            label=f"edge extension {round_index} shared-probe parity",
        )
        active_profile = edge_profile
        active_analysis = edge_analysis
        active_grid = expanded_grid
        completed_edge_rounds = round_index
        selection = read_json(active_analysis / "local_selection.json")
    return active_profile, active_analysis, active_grid, completed_edge_rounds


PROMOTED_ANALYSIS_FILES = (
    "report.html",
    "technical_report.html",
    "summary.json",
    "practical_report.json",
    "report_contract.json",
    "analysis_manifest.json",
    "local_selection.json",
    "local_acceptance.csv",
    "local_timestep.csv",
    "source_bootstrap.csv",
    "bootstrap_regret.csv",
    "robust_dominance.csv",
    "source_loo.csv",
    "natural_gradient_baseline.json",
    "source_localization.json",
    "source_localization.csv",
    "source_localization_detail.csv",
    "no_quant_baseline_profile.json",
    "no_quant_source_load.csv",
    "no_quant_timestep_profile.csv",
    "dataset_character_vector.json",
    "acceptance_contract.json",
)


def promote_analysis(run_dir: Path, analysis_dir: Path) -> list[str]:
    promoted: list[str] = []
    for name in PROMOTED_ANALYSIS_FILES:
        source = analysis_dir / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
            promoted.append(name)
    for required in ("report.html", "technical_report.html", "summary.json", "practical_report.json"):
        if required not in promoted:
            raise FileNotFoundError(f"required product artifact was not generated: {analysis_dir / required}")
    return promoted


def promote_profile_provenance(run_dir: Path, profile_dir: Path) -> list[str]:
    mapping = {
        "source_manifest.json": "source_manifest.json",
        "candidate_definitions.json": "candidate_definitions.json",
        "probe_manifest.json": "probe_manifest.json",
        "calibration_gate.json": "profile_calibration_gate.json",
        "gradient_tail.csv": "raw_gradient_tail.csv",
        "local_natural_gradient.csv": "raw_local_natural_gradient.csv",
    }
    promoted: list[str] = []
    for source_name, target_name in mapping.items():
        source = profile_dir / source_name
        if source.is_file():
            shutil.copy2(source, run_dir / target_name)
            promoted.append(target_name)
    if "source_manifest.json" not in promoted:
        raise FileNotFoundError(f"active profile has no source_manifest.json: {profile_dir}")
    return promoted


def preflight(request: ResolvedProfileRequest, source_dirs: Sequence[Path]) -> None:
    required_files = (
        PROFILE_SCRIPT,
        REPO_ROOT / "tools" / "analyze_dq_v24_local.py",
        REPO_ROOT / "tools" / "check_dq_calibration_gate.py",
        REPO_ROOT / "tools" / "check_dq_snapshot_parity.py",
        REPO_ROOT / "tools" / "check_dq_v24_local_extension_parity.py",
        request.dataset_config,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DQ profile preflight is missing required files: {missing}")
    if not request.model_path.exists():
        raise FileNotFoundError(f"pretrained model path was not found: {request.model_path}")
    if len(source_dirs) < request.local_measurement.minimum_source_groups:
        raise ValueError(
            "canonical diagnostic source bootstrap requires at least "
            f"{request.local_measurement.minimum_source_groups} active image_dir groups; "
            f"found {len(source_dirs)}"
        )
    if importlib.util.find_spec("accelerate.commands.launch") is None:
        raise ModuleNotFoundError(
            f"accelerate is not available in the selected Python runtime: {sys.executable}"
        )


def _write_initial_artifacts(
    run_dir: Path,
    request: ResolvedProfileRequest,
    *,
    fingerprint: Mapping[str, Any],
    source_map_payload: Sequence[Mapping[str, Any]],
    image_count: int,
    probe_budget: int,
    execution_plan: Mapping[str, Any],
) -> None:
    source_probe = _source_probe_selection_summary(source_map_payload)
    write_json(run_dir / "resolved_args.json", request.provenance())
    write_json(run_dir / "protocol_fingerprint.json", fingerprint)
    write_json(run_dir / "execution_plan.json", execution_plan)
    write_json(run_dir / "source_group_map.json", source_map_payload)
    shutil.copy2(request.dataset_config, run_dir / "dataset_config_snapshot.toml")
    write_json(
        run_dir / "status.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "preparing",
            "current_stage": "preflight",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "preset": request.preset.name,
            "execution_mode": request.execution_mode.name,
            "qa_depth": request.execution_mode.qa_depth,
            "internal_profile_level": request.execution_mode.internal_profile_level,
            "local_measurement_contract": request.local_measurement.name,
            "sampling_depth": request.local_measurement.sampling_depth,
            "confidence_ceiling": request.local_measurement.confidence_ceiling,
            "profile_scope": "local_body_tail_only",
            "image_count": image_count,
            "probe_budget": probe_budget,
            "source_group_count": source_probe["source_group_count_total"],
            **source_probe,
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "execution_plan_sha256": execution_plan["execution_plan_sha256"],
            "not_quality_or_utility": True,
            "normal_training_sources_modified": False,
        },
    )


def run_profile_request(
    request: ResolvedProfileRequest,
    options: ProductionRunOptions,
) -> ProductionRunResult:
    source_dirs = source_dirs_from_dataset_config(
        request.dataset_config,
        train_batch_size=int(request.preset.expected_explicit["train_batch_size"]),
        enable_bucket=bool(request.preset.expected_explicit["enable_bucket"]),
        bucket_no_upscale=bool(
            request.preset.expected_explicit["bucket_no_upscale"]
        ),
        min_bucket_reso=int(request.preset.expected_explicit["min_bucket_reso"]),
        max_bucket_reso=int(request.preset.expected_explicit["max_bucket_reso"]),
        bucket_reso_steps=int(request.preset.expected_explicit["bucket_reso_steps"]),
        minimum_source_groups=request.local_measurement.minimum_source_groups,
    )
    preflight(request, source_dirs)
    output_base = validate_output_base(
        options.output_base,
        source_dirs=source_dirs,
        normal_output_dir=request.normal_output_dir,
    )
    source_map_payload, image_count = build_source_map(
        source_dirs,
        dataset_key=sanitize_profile_name(options.profile_name or request.output_name).casefold(),
    )
    probe_budget = min(image_count, request.local_measurement.max_images)
    source_map_payload = _apply_source_probe_selection(
        source_map_payload,
        probe_budget=probe_budget,
    )
    source_probe = _source_probe_selection_summary(source_map_payload)
    fingerprint = build_protocol_fingerprint(request, source_map_payload=source_map_payload)
    execution_plan = build_execution_plan(
        request,
        image_count=image_count,
        probe_budget=probe_budget,
        source_group_count=source_probe["source_group_count_total"],
        probed_source_group_count=source_probe["source_group_count_probed"],
        source_group_selection_policy=source_probe[
            "source_group_selection_policy"
        ],
    )
    run_dir = allocate_run_directory(
        output_base,
        options.profile_name or request.output_name,
        str(fingerprint["fingerprint_sha256"]),
    )
    _write_initial_artifacts(
        run_dir,
        request,
        fingerprint=fingerprint,
        source_map_payload=source_map_payload,
        image_count=image_count,
        probe_budget=probe_budget,
        execution_plan=execution_plan,
    )
    launcher = Launcher(run_dir, dry_run=options.dry_run)
    launcher.log(f"DQ Profiler Beta run directory: {run_dir}")
    launcher.log("Scope: Local Body/Tail numerical Safety/Fidelity only; not quality or Utility")
    work = execution_plan["work_volume"]
    timing = execution_plan["reference_time_estimate"]["minutes"]
    process_text = (
        str(work["gpu_process_count"])
        if work["gpu_process_count"] is not None
        else f"{work['gpu_process_count_min']}-{work['gpu_process_count_max']}"
    )
    time_text = (
        f"{timing['minimum']:.1f} min"
        if timing["minimum"] == timing["maximum_if_all_edge_rounds_run"]
        else (
            f"{timing['minimum']:.1f}-"
            f"{timing['maximum_if_all_edge_rounds_run']:.1f} min"
        )
    )
    launcher.log(
        "Execution plan: "
        f"mode={request.execution_mode.name}, GPU processes={process_text}, "
        f"warmup boundary={work['warmup_boundary_updates']} updates, "
        f"prefix branch updates={work['prefix']['total_branch_updates']}, "
        f"Local probes={work['local']['total_local_probes']}"
    )
    launcher.log(
        "Source coverage: "
        f"{source_probe['source_group_count_probed']}/"
        f"{source_probe['source_group_count_total']} groups in the probe "
        f"({source_probe['source_group_selection_policy']}); "
        "the full inventory remains in the source contract"
    )
    launcher.log(
        "Reference time estimate (not guaranteed): "
        f"{time_text}; see execution_plan.json for assumptions"
    )
    try:
        if options.preflight_only:
            update_status(
                run_dir,
                status="preflight_complete",
                current_stage="none",
                execution_started=False,
                report=None,
            )
            launcher.log("Preflight completed; GPU execution was not started")
            return ProductionRunResult(run_dir, "preflight_complete", None)
        if options.dry_run:
            names = stage_names()
            gate = run_dir / names["prefix"] / "calibration_gate.json"
            command = profile_command(
                request,
                run_dir=run_dir,
                source_map=run_dir / "source_group_map.json",
                name=names["core_raw"],
                protocol="v24-acceptance-local",
                range_muls=request.execution_mode.core_grid,
                max_images=probe_budget,
                gate=gate,
            )
            write_json(
                run_dir / "dry_run_command.json",
                {
                    "stage": names["core_raw"],
                    "argv": command,
                    "requires_prefix_gate": str(gate),
                    "note": (
                        "This is the resolved Core worker command. The normal runner first "
                        "creates and validates the referenced prefix gate."
                    ),
                },
            )
            update_status(
                run_dir,
                status="dry_run_complete",
                current_stage="none",
                execution_started=False,
                report=None,
            )
            launcher.log("Dry run completed; no GPU process was started")
            return ProductionRunResult(run_dir, "dry_run_complete", None)

        active_profile, active_analysis, measured_grid, edge_rounds = run_local_pipeline(
            launcher,
            request,
            source_map=run_dir / "source_group_map.json",
            probe_budget=probe_budget,
            dataset_id=sanitize_profile_name(options.profile_name or request.output_name),
        )
        promoted = promote_analysis(run_dir, active_analysis)
        promoted.extend(promote_profile_provenance(run_dir, active_profile))
        selection = read_json(run_dir / "local_selection.json")
        update_status(
            run_dir,
            status="complete",
            current_stage="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            report="report.html",
            technical_report="technical_report.html",
            active_profile=str(active_profile),
            active_analysis=str(active_analysis),
            measured_grid=list(measured_grid),
            execution_mode=request.execution_mode.name,
            qa_depth=request.execution_mode.qa_depth,
            edge_policy=request.execution_mode.edge_policy,
            edge_extension_rounds=edge_rounds,
            edge_unresolved=bool(selection.get("edge_unresolved")),
            selection_valid=bool(selection.get("selection_valid")),
            retained_candidates=selection.get("credible_muls", []),
            promoted_artifacts=promoted,
            trajectory_status="research_only_not_run",
        )
        report = run_dir / "report.html"
        launcher.log(f"Primary report: {report}")
        launcher.log(f"Technical report: {run_dir / 'technical_report.html'}")
        if options.open_report and os.name == "nt":
            os.startfile(report)  # type: ignore[attr-defined]
        return ProductionRunResult(run_dir, "complete", report)
    except Exception as error:
        update_status(
            run_dir,
            status="failed",
            current_stage="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=repr(error),
            automatic_retry=False,
            existing_outputs_overwritten=False,
        )
        launcher.log(f"FAILED: {error!r}")
        raise
