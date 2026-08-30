from __future__ import annotations

import hashlib
import math
import os
import stat
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python 3.10 used by this project
    tomllib = None

import toml

from dq_profile import RUNTIME_PROTOCOL_VERSION as PROTOCOL_VERSION


AUTO_BANDS: dict[str, tuple[float, float]] = {
    "clip_rate_high": (0.003, 0.005),
    "clip_rate_low": (0.0005, 0.0022),
}
DEFAULT_V2_RANGE_MULS: tuple[float, ...] = (2.70, 2.85, 3.00, 3.15, 3.30, 3.45)
STATELESS_RNG_DEFINITION_VERSION = "sdxl-dq-profile-v1"


@dataclass(frozen=True)
class CandidateDefinition:
    name: str
    quantized: bool
    clip_low: Optional[float]
    clip_high: Optional[float]
    initial_range_mul: Optional[float]
    auto_enabled: bool
    mechanism: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSubsetSummary:
    image_dir: str
    repeats: int
    image_count: int
    weighted_count: int


@dataclass(frozen=True)
class ResolvedDatasetSettings:
    dataset_index: int
    batch_size: int
    enable_bucket: bool
    bucket_no_upscale: bool
    min_bucket_reso: int
    max_bucket_reso: int


@dataclass(frozen=True)
class ResolvedDatasetSubset:
    dataset_index: int
    subset_index: int
    image_dir: Path
    repeats: int
    image_files: tuple[Path, ...]


@dataclass(frozen=True)
class ResolvedDatasetLayout:
    dataset_config: Path
    settings: tuple[ResolvedDatasetSettings, ...]
    subsets: tuple[ResolvedDatasetSubset, ...]


@dataclass(frozen=True)
class PreflightSummary:
    dataset_config: str
    dataset_batch_sizes: tuple[int, ...]
    subsets: tuple[DatasetSubsetSummary, ...]
    unique_images: int
    repeat_weighted_samples: int
    steps_per_epoch: int
    normal_training_steps: int
    dq_begin_step: int
    branch_steps: int
    probe_images: int
    probe_points_per_replica: int
    stochastic_repeats: int
    standard_probe_replicas: int
    full_probe_replicas: int
    full_budget_steps: int
    full_budget_core_exceeded: bool
    estimated_standard_steps: int
    estimated_full_steps: int
    estimated_standard_epochs: float
    estimated_full_epochs: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subsets"] = [asdict(item) for item in self.subsets]
        return payload


def canonical_json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_seed(
    protocol_seed: int,
    *,
    phase: str,
    probe_or_step: str | int,
    module_name: str,
    invocation: int,
    repeat: int,
) -> int:
    """Return a stateless unsigned 64-bit seed.

    Candidate identity is intentionally absent so high/low candidates use
    common random numbers.
    """

    fields = (
        # Keep the v1 stateless stream stable when report/schema versions
        # evolve. Candidate identity remains intentionally absent.
        STATELESS_RNG_DEFINITION_VERSION,
        str(int(protocol_seed)),
        str(phase),
        str(probe_or_step),
        str(module_name),
        str(int(invocation)),
        str(int(repeat)),
    )
    digest = hashlib.blake2b("\x1f".join(fields).encode("utf-8"), digest_size=8, person=b"dqprofv1").digest()
    return int.from_bytes(digest, "little", signed=False)


def initial_range_mul(clip_low: float, clip_high: float, minimum: float = 1.0, maximum: float = 6.0) -> float:
    clip_target = (float(clip_low) + float(clip_high)) / 2.0
    if not 0.0 < clip_target < 1.0:
        raise ValueError(f"clip target must be in (0, 1), got {clip_target}")
    value = statistics.NormalDist().inv_cdf(1.0 - clip_target / 2.0)
    return max(float(minimum), min(float(maximum), float(value)))


def default_candidates() -> tuple[CandidateDefinition, ...]:
    candidates = [CandidateDefinition("no_quant", False, None, None, None, False)]
    for name in ("clip_rate_high", "clip_rate_low"):
        low, high = AUTO_BANDS[name]
        candidates.append(CandidateDefinition(name, True, low, high, initial_range_mul(low, high), True))
    return tuple(candidates)


def parse_range_muls(
    value: str | Sequence[float],
    *,
    minimum_count: int = 3,
    maximum_count: int | None = None,
) -> tuple[float, ...]:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if maximum_count is not None and maximum_count < minimum_count:
        raise ValueError("maximum_count must be >= minimum_count")
    raw = value.split(",") if isinstance(value, str) else value
    parsed: list[float] = []
    for item in raw:
        number = float(str(item).strip())
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"range_mul must be finite and positive, got {item!r}")
        if not any(abs(number - existing) <= 1e-12 for existing in parsed):
            parsed.append(number)
    if len(parsed) < minimum_count:
        raise ValueError(
            "fixed range sweep requires at least "
            f"{minimum_count} distinct range_mul values"
        )
    if maximum_count is not None and len(parsed) > maximum_count:
        raise ValueError(
            "fixed range sweep accepts at most "
            f"{maximum_count} distinct range_mul values"
        )
    return tuple(sorted(parsed))


def parse_mechanism_muls(value: Optional[str | Sequence[float]]) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    parsed: list[float] = []
    for item in raw:
        number = float(str(item).strip())
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"mechanism range_mul must be finite and positive, got {item!r}")
        if not any(abs(number - existing) <= 1e-12 for existing in parsed):
            parsed.append(number)
    if not parsed:
        raise ValueError("at least one mechanism range_mul is required")
    return tuple(sorted(parsed))


def fixed_range_candidates(
    range_muls: Sequence[float],
    *,
    minimum_count: int = 3,
    maximum_count: int | None = None,
) -> tuple[CandidateDefinition, ...]:
    candidates = [CandidateDefinition("no_quant", False, None, None, None, False)]
    for value in parse_range_muls(
        tuple(range_muls),
        minimum_count=minimum_count,
        maximum_count=maximum_count,
    ):
        candidates.append(
            CandidateDefinition(
                name=f"mul_{value:.3f}",
                quantized=True,
                clip_low=None,
                clip_high=None,
                initial_range_mul=float(value),
                auto_enabled=False,
                mechanism="full",
            )
        )
    return tuple(candidates)


def mechanism_candidates(range_mul: float) -> tuple[CandidateDefinition, ...]:
    value = float(range_mul)
    return (
        CandidateDefinition("no_quant", False, None, None, None, False),
        CandidateDefinition(f"mul_{value:.3f}__full", True, None, None, value, False, "full"),
        CandidateDefinition(f"mul_{value:.3f}__clip_only", True, None, None, value, False, "clip_only"),
        CandidateDefinition(f"mul_{value:.3f}__round_only", True, None, None, value, False, "round_only"),
    )


def calculate_dq_begin_step(lr_warmup_steps: int | float, max_train_steps: int, num_processes: int = 1) -> int:
    """Mirror the copied trainer's dq_delta_begin_after_lr_warmup rule."""

    if isinstance(lr_warmup_steps, float):
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when lr_warmup_steps is a float")
        value = int(lr_warmup_steps * max_train_steps * num_processes)
    else:
        value = int(lr_warmup_steps)
    return max(0, value)


def resolve_branch_steps(total_steps: int, override: Optional[int]) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("dq_profile_branch_steps must be positive")
        return int(override)
    return max(64, min(256, int(math.ceil(total_steps * 0.025))))


def _load_toml(path: Path) -> Mapping[str, Any]:
    if tomllib is not None:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return toml.load(stream)


def training_image_extensions() -> frozenset[str]:
    # Match the loader's runtime extension set, including optional AVIF/JXL
    # plugins when they are installed.
    from library.train_util import IMAGE_EXTENSIONS as LOADER_IMAGE_EXTENSIONS

    return frozenset(str(extension) for extension in LOADER_IMAGE_EXTENSIONS)


def loader_visible_image_files(directory: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Return exactly the files DreamBoothDataset's non-recursive glob sees."""

    from library.train_util import glob_images

    return tuple(Path(value) for value in glob_images(str(directory), "*"))


def _first_symlink_or_reparse_component(path: Path) -> Optional[Path]:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                return current
            attributes = int(getattr(current.lstat(), "st_file_attributes", 0))
        except OSError:
            continue
        if attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            return current
    return None


def _fallback_value(key: str, sources: Sequence[Mapping[str, Any]], default: Any) -> Any:
    for source in sources:
        value = source.get(key)
        if value is not None:
            return value
    return default


def _bool_setting(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _int_setting(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(value)


def resolve_dataset_layout(
    dataset_config: str | os.PathLike[str],
    *,
    train_batch_size: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    min_bucket_reso: int = 256,
    max_bucket_reso: int = 1024,
    dataset_repeats: int = 1,
    cache_info: bool = False,
    minimum_source_groups: int = 1,
) -> ResolvedDatasetLayout:
    path = Path(dataset_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset config was not found: {path}")
    config = _load_toml(path)
    datasets = config.get("datasets")
    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise ValueError(f"no [[datasets]] sections were found in {path}")
    general = config.get("general", {})
    if not isinstance(general, Mapping):
        raise ValueError("dataset [general] must be a table")

    settings: list[ResolvedDatasetSettings] = []
    resolved_subsets: list[ResolvedDatasetSubset] = []
    source_dirs: list[Path] = []
    seen: set[str] = set()
    for dataset_index, raw_dataset in enumerate(datasets):
        if not isinstance(raw_dataset, Mapping):
            raise ValueError(f"datasets[{dataset_index}] must be a table")
        dataset = raw_dataset
        batch_size = _int_setting(
            _fallback_value("batch_size", (dataset, general), train_batch_size),
            label=f"datasets[{dataset_index}].batch_size",
        )
        bucket_enabled = _bool_setting(
            _fallback_value("enable_bucket", (dataset, general), enable_bucket),
            label=f"datasets[{dataset_index}].enable_bucket",
        )
        effective_bucket_no_upscale = _bool_setting(
            _fallback_value(
                "bucket_no_upscale",
                (dataset, general),
                bucket_no_upscale,
            ),
            label=f"datasets[{dataset_index}].bucket_no_upscale",
        )
        effective_min_bucket = _int_setting(
            _fallback_value("min_bucket_reso", (dataset, general), min_bucket_reso),
            label=f"datasets[{dataset_index}].min_bucket_reso",
        )
        effective_max_bucket = _int_setting(
            _fallback_value("max_bucket_reso", (dataset, general), max_bucket_reso),
            label=f"datasets[{dataset_index}].max_bucket_reso",
        )
        mismatches: list[str] = []
        if batch_size != int(train_batch_size):
            mismatches.append(f"batch_size={batch_size} (required {int(train_batch_size)})")
        if bucket_enabled is not bool(enable_bucket):
            mismatches.append(f"enable_bucket={bucket_enabled} (required {bool(enable_bucket)})")
        if effective_bucket_no_upscale is not bool(bucket_no_upscale):
            mismatches.append(
                "bucket_no_upscale="
                f"{effective_bucket_no_upscale} (required {bool(bucket_no_upscale)})"
            )
        if bucket_enabled:
            if effective_min_bucket != int(min_bucket_reso):
                mismatches.append(
                    f"min_bucket_reso={effective_min_bucket} (required {int(min_bucket_reso)})"
                )
            if effective_max_bucket != int(max_bucket_reso):
                mismatches.append(
                    f"max_bucket_reso={effective_max_bucket} (required {int(max_bucket_reso)})"
                )
        if mismatches:
            raise ValueError(
                "dataset TOML overrides canonical effective settings in "
                f"datasets[{dataset_index}]: {', '.join(mismatches)}"
            )
        settings.append(
            ResolvedDatasetSettings(
                dataset_index=dataset_index,
                batch_size=batch_size,
                enable_bucket=bucket_enabled,
                bucket_no_upscale=effective_bucket_no_upscale,
                min_bucket_reso=effective_min_bucket,
                max_bucket_reso=effective_max_bucket,
            )
        )

        subsets = dataset.get("subsets")
        if not isinstance(subsets, Sequence) or isinstance(subsets, (str, bytes)) or not subsets:
            raise ValueError(f"datasets[{dataset_index}] has no [[datasets.subsets]] entries")
        for subset_index, raw_subset in enumerate(subsets):
            if not isinstance(raw_subset, Mapping):
                raise ValueError(f"datasets[{dataset_index}].subsets[{subset_index}] must be a table")
            subset = raw_subset
            if subset.get("metadata_file") is not None:
                raise ValueError("canonical diagnostic supports DreamBooth image_dir subsets only")
            effective_color_aug = _bool_setting(
                _fallback_value("color_aug", (subset, dataset, general), False),
                label=(
                    f"datasets[{dataset_index}].subsets[{subset_index}].color_aug"
                ),
            )
            effective_random_crop = _bool_setting(
                _fallback_value("random_crop", (subset, dataset, general), False),
                label=(
                    f"datasets[{dataset_index}].subsets[{subset_index}].random_crop"
                ),
            )
            if effective_color_aug or effective_random_crop:
                raise ValueError(
                    "canonical diagnostic enables cache_latents and therefore requires "
                    "color_aug=false and random_crop=false after dataset fallback: "
                    f"datasets[{dataset_index}].subsets[{subset_index}] has "
                    f"color_aug={effective_color_aug}, random_crop={effective_random_crop}"
                )
            raw_dir = subset.get("image_dir")
            if raw_dir is None:
                raise ValueError(
                    f"datasets[{dataset_index}].subsets[{subset_index}] has no image_dir"
                )
            raw_image_dir = Path(str(raw_dir))
            if not raw_image_dir.is_absolute():
                raise ValueError(
                    "canonical diagnostic requires absolute image_dir paths because the training "
                    f"loader resolves relative paths from its process cwd: {raw_dir!r}"
                )
            linked_component = _first_symlink_or_reparse_component(raw_image_dir)
            if linked_component is not None:
                raise ValueError(
                    "canonical diagnostic rejects image_dir paths containing symlink or reparse "
                    "components because source-group prefixes must match worker-visible image keys: "
                    f"{linked_component}"
                )
            image_dir = raw_image_dir.resolve()
            if not image_dir.is_dir():
                raise FileNotFoundError(f"dataset image_dir was not found: {image_dir}")

            repeats = _int_setting(
                _fallback_value("num_repeats", (subset, dataset, general), dataset_repeats),
                label=f"datasets[{dataset_index}].subsets[{subset_index}].num_repeats",
            )
            if repeats < 1:
                raise ValueError(
                    "canonical diagnostic rejects subsets ignored by the training loader: "
                    f"datasets[{dataset_index}].subsets[{subset_index}].num_repeats={repeats}"
                )
            use_cache_info = _bool_setting(
                _fallback_value("cache_info", (subset, dataset, general), cache_info),
                label=f"datasets[{dataset_index}].subsets[{subset_index}].cache_info",
            )
            if use_cache_info:
                raise ValueError(
                    "canonical diagnostic does not support cache_info=true because preflight must "
                    "derive the same immutable image inventory as every worker"
                )

            folded = str(image_dir).casefold()
            if folded in seen:
                raise ValueError(f"duplicate image_dir is ambiguous for source bootstrap: {image_dir}")
            for existing in source_dirs:
                if existing in image_dir.parents or image_dir in existing.parents:
                    raise ValueError(f"nested image_dir values are ambiguous: {existing} and {image_dir}")
            seen.add(folded)
            source_dirs.append(image_dir)

            image_files = tuple(
                sorted(
                    (item.resolve() for item in loader_visible_image_files(image_dir)),
                    key=lambda item: (str(item).casefold(), str(item)),
                )
            )
            if not image_files:
                nested_count = sum(
                    len(loader_visible_image_files(item))
                    for item in image_dir.rglob("*")
                    if item.is_dir()
                )
                if nested_count:
                    raise ValueError(
                        "training image discovery is non-recursive; "
                        f"{image_dir} has {nested_count} supported images only below child directories"
                    )
                raise ValueError(f"dataset image_dir has no loader-visible image files: {image_dir}")
            resolved_subsets.append(
                ResolvedDatasetSubset(
                    dataset_index=dataset_index,
                    subset_index=subset_index,
                    image_dir=image_dir,
                    repeats=repeats,
                    image_files=image_files,
                )
            )

    required_groups = max(1, int(minimum_source_groups))
    if len(resolved_subsets) < required_groups:
        raise ValueError(
            "canonical diagnostic source bootstrap requires at least "
            f"{required_groups} active image_dir groups; found {len(resolved_subsets)}"
        )
    return ResolvedDatasetLayout(path, tuple(settings), tuple(resolved_subsets))


def inspect_dataset_config(
    dataset_config: str | os.PathLike[str],
    *,
    max_train_epochs: Optional[int],
    max_train_steps: Optional[int],
    lr_warmup_steps: int | float,
    branch_steps_override: Optional[int],
    max_images: int,
    timestep_bins: int,
    stochastic_repeats: int,
    train_batch_size: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    min_bucket_reso: int = 256,
    max_bucket_reso: int = 1024,
    dataset_repeats: int = 1,
    cache_info: bool = False,
    minimum_source_groups: int = 1,
) -> PreflightSummary:
    layout = resolve_dataset_layout(
        dataset_config,
        train_batch_size=train_batch_size,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        dataset_repeats=dataset_repeats,
        cache_info=cache_info,
        minimum_source_groups=minimum_source_groups,
    )
    path = layout.dataset_config

    dataset_batch_sizes = [setting.batch_size for setting in layout.settings]
    subset_summaries: list[DatasetSubsetSummary] = []
    unique_paths: set[str] = set()
    weighted_samples = 0
    for subset in layout.subsets:
        count = len(subset.image_files)
        unique_paths.update(os.path.normcase(str(item)) for item in subset.image_files)
        weighted = count * subset.repeats
        weighted_samples += weighted
        subset_summaries.append(
            DatasetSubsetSummary(str(subset.image_dir), subset.repeats, count, weighted)
        )
    if weighted_samples <= 0:
        raise ValueError(f"no training images were found through {path}")
    steps_per_epoch = weighted_samples
    if max_train_epochs is not None:
        normal_steps = int(max_train_epochs) * steps_per_epoch
    elif max_train_steps is not None and int(max_train_steps) > 0:
        normal_steps = int(max_train_steps)
    else:
        raise ValueError("max_train_epochs or positive max_train_steps is required")
    begin_step = calculate_dq_begin_step(lr_warmup_steps, normal_steps)
    branch_steps = resolve_branch_steps(normal_steps, branch_steps_override)
    probe_images = min(max(1, int(max_images)), max(1, len(unique_paths)))
    repeats = max(1, int(stochastic_repeats))
    probe_points = probe_images * max(1, int(timestep_bins))
    probe_backward_per_replica = probe_points * (1 + 2 * repeats)
    core_steps = begin_step + branch_steps * 3
    standard_steps = core_steps + probe_backward_per_replica
    requested_full_budget = min(standard_steps * 2, int(normal_steps * 0.75))
    full_budget_core_exceeded = requested_full_budget < standard_steps
    full_budget = max(standard_steps, requested_full_budget)
    full_probe_replicas = max(1, (full_budget - core_steps) // max(probe_backward_per_replica, 1))
    full_steps = core_steps + full_probe_replicas * probe_backward_per_replica
    return PreflightSummary(
        dataset_config=str(path),
        dataset_batch_sizes=tuple(dataset_batch_sizes),
        subsets=tuple(subset_summaries),
        unique_images=len(unique_paths),
        repeat_weighted_samples=weighted_samples,
        steps_per_epoch=steps_per_epoch,
        normal_training_steps=normal_steps,
        dq_begin_step=begin_step,
        branch_steps=branch_steps,
        probe_images=probe_images,
        probe_points_per_replica=probe_points,
        stochastic_repeats=repeats,
        standard_probe_replicas=1,
        full_probe_replicas=full_probe_replicas,
        full_budget_steps=full_budget,
        full_budget_core_exceeded=full_budget_core_exceeded,
        estimated_standard_steps=standard_steps,
        estimated_full_steps=full_steps,
        estimated_standard_epochs=standard_steps / steps_per_epoch,
        estimated_full_epochs=full_steps / steps_per_epoch,
    )


class AutoRangeController:
    """Candidate-local implementation of the production high/low controller."""

    def __init__(
        self,
        candidate: CandidateDefinition,
        *,
        every: int = 50,
        ema: float = 0.95,
        mul_up: float = 1.01,
        mul_down: float = 0.995,
        minimum: float = 1.0,
        maximum: float = 6.0,
        warmup: bool = True,
        warmup_updates: int = 0,
        use_raw: bool = False,
    ) -> None:
        if not candidate.quantized or candidate.clip_low is None or candidate.clip_high is None:
            raise ValueError("AutoRangeController requires a quantized candidate")
        self.candidate = candidate
        self.every = max(1, int(every))
        self.ema_decay = float(ema)
        self.mul_up = float(mul_up)
        self.mul_down = float(mul_down)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.use_raw = bool(use_raw)
        self.range_mul = float(candidate.initial_range_mul)
        self.ema_value: Optional[float] = None
        self.observation_count = 0
        self.post_warmup_observation_count = 0
        self.inband_streak = 0
        if warmup and 0.0 < self.ema_decay < 1.0:
            self.warmup_updates = int(warmup_updates) if int(warmup_updates) > 0 else int(math.ceil(2.0 / (1.0 - self.ema_decay)))
        else:
            self.warmup_updates = 0
        self.warmup_remaining = self.warmup_updates
        self.rows: list[dict[str, Any]] = []

    @property
    def warmup_completed(self) -> bool:
        return self.warmup_remaining <= 0

    def observe(self, step: int, clip_rate: Optional[float]) -> dict[str, Any]:
        before = self.range_mul
        reason = "not_observed"
        applied = False
        raw = None if clip_rate is None else float(clip_rate)
        warmup_active = self.warmup_remaining > 0
        if raw is not None and math.isfinite(raw):
            self.observation_count += 1
            self.ema_value = raw if self.ema_value is None else self.ema_value * self.ema_decay + raw * (1.0 - self.ema_decay)
            ema_value = self.ema_value
            low = float(self.candidate.clip_low)
            high = float(self.candidate.clip_high)
            if warmup_active:
                self.inband_streak = self.inband_streak + 1 if low <= ema_value <= high else 0
                self.warmup_remaining = max(0, self.warmup_remaining - 1)
                if self.inband_streak >= 3:
                    self.warmup_remaining = 0
                reason = "warmup"
            else:
                self.post_warmup_observation_count += 1
                high_hit = ema_value > high and (not self.use_raw or raw > high)
                low_hit = ema_value < low and (not self.use_raw or raw < low)
                if high_hit:
                    self.range_mul *= self.mul_up
                    reason = "clip_high"
                elif low_hit:
                    self.range_mul *= self.mul_down
                    reason = "clip_low"
                else:
                    reason = "in_band"
                self.range_mul = max(self.minimum, min(self.maximum, self.range_mul))
                applied = self.range_mul != before
        row = {
            "step": int(step),
            "candidate": self.candidate.name,
            "clip_rate_raw": raw,
            "clip_rate_ema": self.ema_value,
            "range_mul_before": before,
            "range_mul_after": self.range_mul,
            "auto_applied": applied,
            "auto_reason": reason,
            "warmup_active": warmup_active,
            "warmup_remaining": self.warmup_remaining,
        }
        self.rows.append(row)
        return row

    def validity(self) -> dict[str, Any]:
        valid = self.warmup_completed and self.post_warmup_observation_count >= 3
        if not self.warmup_completed:
            reason = "auto_warmup_not_completed"
        elif self.post_warmup_observation_count < 3:
            reason = "fewer_than_3_post_warmup_observations"
        else:
            reason = None
        return {
            "auto_observation_count": self.observation_count,
            "auto_post_warmup_observation_count": self.post_warmup_observation_count,
            "auto_warmup_completed": self.warmup_completed,
            "auto_trajectory_metrics_valid": valid,
            "auto_invalid_reason": reason,
        }
