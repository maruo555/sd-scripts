from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import dq_profile.production_runner as production_runner
import dq_profile.__main__ as production_main
from dq_profile.production_cli import ProfileCompatibilityError, resolve_training_cli
from dq_profile.production_runner import (
    DEFAULT_OUTPUT_BASE,
    Launcher,
    ProductionRunOptions,
    _model_identity,
    allocate_run_directory,
    build_execution_plan,
    build_protocol_fingerprint,
    build_source_map,
    git_tracked_state,
    profile_command,
    promote_analysis,
    promote_profile_provenance,
    run_local_pipeline,
    sanitize_profile_name,
    source_dirs_from_dataset_config,
    subprocess_text_environment,
    validate_model_identity,
    validate_source_map_inventory,
    validate_output_base,
)


MODEL = r"D:\models\model.safetensors"
DATASET = r"D:\data set\日本語\dataset.toml"


def minimal_cli(*extra: str) -> list[str]:
    return [
        f"--pretrained_model_name_or_path={MODEL}",
        f"--dataset_config={DATASET}",
        *extra,
    ]


def test_minimal_cli_uses_versioned_preset_and_japanese_paths() -> None:
    request = resolve_training_cli(minimal_cli())
    assert request.preset.name == "canonical-v1"
    assert request.local_measurement.name == "local-body-tail-v1"
    assert request.execution_mode.name == "standard"
    assert request.dataset_config.name == "dataset.toml"
    assert request.output_name == "dataset"
    assert {row["action"] for row in request.dispositions} == {"consumed"}


def test_standard_mode_keeps_the_shared_local_ruler_and_shortens_only_qa() -> None:
    strict = resolve_training_cli(minimal_cli(), execution_mode_name="strict")
    standard = resolve_training_cli(minimal_cli(), execution_mode_name="standard")
    assert standard.preset.contract() == strict.preset.contract()
    assert standard.local_measurement.contract() == strict.local_measurement.contract()
    assert standard.execution_mode.core_grid == (2.70, 3.15, 3.45, 3.75, 4.05)
    assert standard.execution_mode.prefix_checkpoints == (0, 1, 4, 8)
    assert standard.execution_mode.prefix_branch_updates == 64
    assert standard.execution_mode.max_edge_extension_rounds == 0
    assert strict.execution_mode.prefix_checkpoints == (0, 1, 32, 64)
    assert strict.execution_mode.prefix_branch_updates == 512
    assert strict.execution_mode.max_edge_extension_rounds == 2


def test_quick_mode_uses_a_separate_reduced_sampling_contract() -> None:
    standard = resolve_training_cli(minimal_cli(), execution_mode_name="standard")
    quick = resolve_training_cli(minimal_cli(), execution_mode_name="quick")
    assert quick.preset.contract() == standard.preset.contract()
    assert quick.local_measurement.name == "local-body-tail-quick-v1"
    assert quick.local_measurement.metric_definition_version == "2.4.0"
    assert quick.local_measurement.max_images == 16
    assert quick.local_measurement.timestep_bins == 4
    assert quick.local_measurement.no_quant_noise_replicas == 3
    assert quick.local_measurement.candidate_noise_replicas == 2
    assert quick.local_measurement.stochastic_quant_repeats == 2
    assert quick.local_measurement.sampling_depth == "reduced_16_image"
    assert quick.execution_mode.core_grid == standard.execution_mode.core_grid
    assert quick.execution_mode.prefix_checkpoints == (0, 1, 4, 8)
    assert quick.execution_mode.standalone_snapshot_count == 1
    assert quick.execution_mode.gpu_process_count_range == (3, 3)


def test_cross_mode_fingerprints_split_shared_local_from_execution_qa(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[general]\nresolution=1024\n", encoding="utf-8")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    argv = [
        f"--pretrained_model_name_or_path={model}",
        f"--dataset_config={dataset}",
    ]
    strict = build_protocol_fingerprint(
        resolve_training_cli(argv, execution_mode_name="strict"),
        source_map_payload=({"source_group": "source-01"},),
    )
    standard = build_protocol_fingerprint(
        resolve_training_cli(argv, execution_mode_name="standard"),
        source_map_payload=({"source_group": "source-01"},),
    )
    strict_hashes = strict["contract_hashes"]
    standard_hashes = standard["contract_hashes"]
    assert strict_hashes["training_contract_sha256"] == standard_hashes["training_contract_sha256"]
    assert strict_hashes["dataset_contract_sha256"] == standard_hashes["dataset_contract_sha256"]
    assert strict_hashes["local_probe_contract_sha256"] == standard_hashes["local_probe_contract_sha256"]
    assert strict_hashes["execution_qa_contract_sha256"] != standard_hashes["execution_qa_contract_sha256"]
    assert strict["full_protocol_fingerprint_sha256"] != standard["full_protocol_fingerprint_sha256"]

    quick = build_protocol_fingerprint(
        resolve_training_cli(argv, execution_mode_name="quick"),
        source_map_payload=({"source_group": "source-01"},),
    )
    quick_hashes = quick["contract_hashes"]
    assert quick_hashes["training_contract_sha256"] == standard_hashes["training_contract_sha256"]
    assert quick_hashes["dataset_contract_sha256"] == standard_hashes["dataset_contract_sha256"]
    assert quick_hashes["local_measurement_contract_sha256"] != standard_hashes["local_measurement_contract_sha256"]
    assert quick_hashes["local_probe_contract_sha256"] != standard_hashes["local_probe_contract_sha256"]
    assert quick_hashes["execution_qa_contract_sha256"] != standard_hashes["execution_qa_contract_sha256"]


def test_full_legacy_style_cli_is_explicit_about_overrides() -> None:
    request = resolve_training_cli(
        minimal_cli(
            "--output_dir=D:\\lora_output",
            "--output_name=sample_r4",
            "--optimizer_type=AdamW8bitFast",
            "--network_module=networks.lora",
            "--network_dim=4",
            "--network_args",
            "rank_dropout=0.2",
            "--mixed_precision=fp16",
            "--fp16_safe_norms_mode=strict",
            "--network_dropout=0.3",
            "--max_data_loader_n_workers=1",
            "--dq_delta_auto_range_mul",
            "--dq_delta_auto_preset=clip_rate_low_auto",
        )
    )
    actions = {(row["destination"], row["action"]) for row in request.dispositions}
    assert ("optimizer_type", "matched_preset") in actions
    assert ("max_data_loader_n_workers", "overridden_with_reason") in actions
    assert ("dq_delta_auto_preset", "overridden_with_reason") in actions
    assert request.output_name == "sample_r4"


def test_canonical_long_training_command_can_be_reused() -> None:
    request = resolve_training_cli(
        minimal_cli(
            "--prior_loss_weight=1.0", "--output_dir=..\\lora_output",
            "--output_name=long_command", "--learning_rate=3.5e-4",
            "--max_train_epochs=40", "--optimizer_type=AdamW8bitFast", "--sdpa",
            "--mixed_precision=fp16", "--save_precision=fp16", "--seed=39",
            "--save_model_as=safetensors", "--save_every_n_epochs=1",
            "--max_data_loader_n_workers=1", "--network_module=networks.lora",
            "--network_dim=4", "--network_args", "rank_dropout=0.2",
            "--enable_bucket", "--min_bucket_reso=384", "--max_bucket_reso=1024",
            "--noise_offset=0.15", "--adaptive_noise_scale=0.1", "--network_dropout=0.3",
            "--cache_latents", "--text_encoder_lr=2e-4", "--downscale_freq_shift",
            "--te_mlp_fc_only", "--grad_norm_mode=stable_no_threshoff", "--avg_cp",
            "--avg_cp_mode=promote", "--avg_window=4", "--avg_begin=0.6",
            "--avg_mode=ema", "--avg_shadow_bank_size=12", "--no-avg_reset_stats",
            "--avg_save_final_raw", "--fp16_safe_norms", "--dq_delta_bits=8",
            "--dq_delta_granularity=channel", "--dq_delta_stat=rms",
            "--dq_delta_range_mul=3.0", "--dq_delta_mode=stoch",
            "--dq_delta_begin_after_lr_warmup", "--dq_delta_scope=unet", "--dq_delta_log",
            "--dq_delta_log_detail=basic", "--dq_delta_auto_range_mul",
            "--dq_delta_auto_preset=clip_rate_low_auto",
            "--dq_delta_auto_init_range_mul_from_band", "--dq_delta_auto_use_raw",
            "--dq_delta_use_triton", "--dq_delta_triton_stats",
            "--text_encoder_lr1=3e-4", "--text_encoder_lr2=2e-4",
            "--lr_scheduler=constant_with_warmup", "--lr_warmup_steps=0.05",
            "--rank_log", "--rank_log_mode=per_module",
        )
    )
    assert request.output_name == "long_command"
    assert any(
        row["destination"] == "dq_delta_range_mul" and row["action"] == "overridden_with_reason"
        for row in request.dispositions
    )


@pytest.mark.parametrize(
    ("option", "required"),
    (
        ("--optimizer_type=AdamW8bit", "AdamW8bitFast"),
        ("--network_dim=8", "network_dim=4"),
        ("--bucket_reso_steps=32", "bucket_reso_steps=64"),
    ),
)
def test_conflicting_canonical_option_is_rejected(option: str, required: str) -> None:
    with pytest.raises(ProfileCompatibilityError, match=required):
        resolve_training_cli(minimal_cli(option))


def test_resume_and_unknown_options_are_rejected() -> None:
    with pytest.raises(ProfileCompatibilityError, match="resume is unsupported"):
        resolve_training_cli(minimal_cli("--resume=D:\\checkpoint"))
    with pytest.raises(ProfileCompatibilityError, match="unknown to the SDXL training parser"):
        resolve_training_cli(minimal_cli("--not_a_real_option=1"))


@pytest.mark.parametrize(
    "unknown_tokens",
    (("--api-token=do-not-print",), ("--api-token", "do-not-print")),
)
def test_unknown_sensitive_option_value_is_redacted(unknown_tokens: tuple[str, ...]) -> None:
    with pytest.raises(ProfileCompatibilityError) as captured:
        resolve_training_cli(minimal_cli(*unknown_tokens))
    message = str(captured.value)
    assert "do-not-print" not in message
    assert "<redacted>" in message


def test_fp16_safe_norms_alias_is_accepted() -> None:
    request = resolve_training_cli(minimal_cli("--fp16_safe_norms"))
    assert any(row["destination"] == "fp16_safe_norms" for row in request.dispositions)


def test_bucket_no_upscale_is_rejected_by_canonical_preset() -> None:
    with pytest.raises(ProfileCompatibilityError, match="bucket_no_upscale"):
        resolve_training_cli(minimal_cli("--bucket_no_upscale"))


def test_canonical_bucket_resolution_steps_are_accepted() -> None:
    request = resolve_training_cli(minimal_cli("--bucket_reso_steps=64"))
    assert any(
        row["destination"] == "bucket_reso_steps"
        and row["action"] == "matched_preset"
        for row in request.dispositions
    )


def test_source_dirs_follow_toml_order_and_reject_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "画像 A"
    second = tmp_path / "画像 B"
    first.mkdir()
    second.mkdir()
    (first / "first.png").write_bytes(b"image")
    (second / "second.png").write_bytes(b"image")
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[general]\n"
        "resolution=1024\n"
        "[[datasets]]\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(second))}\n",
        encoding="utf-8",
    )
    assert source_dirs_from_dataset_config(config) == (first.resolve(), second.resolve())
    config.write_text(
        "[general]\n"
        "resolution=1024\n"
        "[[datasets]]\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate image_dir"):
        source_dirs_from_dataset_config(config)


def test_source_dirs_reject_too_few_active_source_groups(tmp_path: Path) -> None:
    source_dirs = []
    for index in range(3):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        (source / "image.png").write_bytes(b"image")
        source_dirs.append(source)
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[general]\n"
        "resolution=1024\n"
        "[[datasets]]\n"
        + "".join(
            "[[datasets.subsets]]\n" + f"image_dir = {json.dumps(str(source))}\n"
            for source in source_dirs
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 4 active image_dir groups"):
        source_dirs_from_dataset_config(config, minimum_source_groups=4)


def test_source_dirs_reject_missing_effective_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "image.png").write_bytes(b"image")
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[[datasets]]\n"
        "[[datasets.subsets]]\n"
        f"image_dir = {json.dumps(str(source))}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resolution is required"):
        source_dirs_from_dataset_config(config)


def test_source_map_rejects_images_visible_only_recursively(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    nested = source / "nested"
    nested.mkdir(parents=True)
    for index in range(8):
        (nested / f"image_{index:02d}.png").write_bytes(b"image")
    with pytest.raises(ValueError, match="non-recursive image discovery"):
        build_source_map((source,), dataset_key="example")


def test_source_map_counts_only_loader_visible_root_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    nested = source / "nested"
    nested.mkdir(parents=True)
    for index in range(8):
        (source / f"image_{index:02d}.png").write_bytes(b"image")
    (source / "caption.txt").write_text("caption", encoding="utf-8")
    (nested / "hidden.png").write_bytes(b"image")
    payload, image_count = build_source_map((source,), dataset_key="example")
    assert image_count == 8
    assert payload[0]["image_count"] == 8
    assert {row["name"] for row in payload[0]["files"]} == {
        "caption.txt",
        *(f"image_{index:02d}.png" for index in range(8)),
    }


def test_source_inventory_revalidation_detects_same_size_caption_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    for index in range(8):
        (source / f"image_{index:02d}.png").write_bytes(b"image")
    caption = source / "caption.txt"
    caption.write_text("caption-a", encoding="utf-8")
    payload, _ = build_source_map((source,), dataset_key="example")
    source_map = tmp_path / "source_group_map.json"
    source_map.write_text(json.dumps(payload), encoding="utf-8")

    validate_source_map_inventory(source_map)
    caption.write_text("caption-b", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        validate_source_map_inventory(source_map)


def test_source_probe_capacity_rejects_partial_group_coverage() -> None:
    with pytest.raises(ValueError, match="source_groups=33 exceeds probe_budget=32"):
        production_runner._validate_source_probe_capacity(
            source_group_count=33,
            probe_budget=32,
        )
    production_runner._validate_source_probe_capacity(
        source_group_count=32,
        probe_budget=32,
    )


def test_output_base_policy_and_unique_run_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(production_runner, "REPO_ROOT", repo)
    source = tmp_path / "dataset"
    source.mkdir()
    base = tmp_path / "reports"
    assert validate_output_base(base, source_dirs=(source,), normal_output_dir=None) == base.resolve()
    with pytest.raises(ValueError, match="overlap dataset"):
        validate_output_base(source / "report", source_dirs=(source,), normal_output_dir=None)
    first = allocate_run_directory(base, "日本語 profile", "a" * 64)
    second = allocate_run_directory(base, "日本語 profile", "a" * 64)
    assert first != second
    assert first.parent.name == "日本語_profile"


def test_default_output_location_is_project_lora_output() -> None:
    assert DEFAULT_OUTPUT_BASE.parts[-2:] == ("lora_output", "dq_dataset_profiler")


def test_model_file_identity_hashes_contents_even_when_metadata_is_reused(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model-a")
    original_stat = model.stat()
    first = _model_identity(model)
    model.write_bytes(b"model-b")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = _model_identity(model)
    assert first["size"] == second["size"]
    assert first["mtime_ns"] == second["mtime_ns"]
    assert first["sha256"] != second["sha256"]


def test_model_directory_identity_hashes_nested_file_contents(tmp_path: Path) -> None:
    model = tmp_path / "model"
    weights = model / "unet" / "weights.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights-a")
    first = _model_identity(model)
    weights.write_bytes(b"weights-b")
    second = _model_identity(model)
    assert first["file_count"] == second["file_count"] == 1
    assert first["total_file_size"] == second["total_file_size"]
    assert first["inventory_sha256"] != second["inventory_sha256"]


def test_model_identity_revalidation_detects_same_metadata_change(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model-a")
    original_stat = model.stat()
    fingerprint = tmp_path / "protocol_fingerprint.json"
    fingerprint.write_text(
        json.dumps({"model": _model_identity(model)}),
        encoding="utf-8",
    )
    validate_model_identity(model, fingerprint)

    model.write_bytes(b"model-b")
    os.utime(model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="model changed"):
        validate_model_identity(model, fingerprint)


def test_git_tracked_state_hashes_the_complete_dirty_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    dependency = repo / "dq_profile" / "omitted_dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", str(dependency)], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=DQ Profile Test",
            "-c",
            "user.email=dq-profile-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(production_runner, "REPO_ROOT", repo)

    clean = git_tracked_state()
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    first_dirty = git_tracked_state()
    dependency.write_text("VALUE = 3\n", encoding="utf-8")
    second_dirty = git_tracked_state()

    assert clean["dirty"] is False
    assert first_dirty["dirty"] is True
    assert second_dirty["dirty"] is True
    assert clean["state_sha256"] != first_dirty["state_sha256"]
    assert first_dirty["diff_sha256"] != second_dirty["diff_sha256"]
    assert first_dirty["state_sha256"] != second_dirty["state_sha256"]


def test_windows_subprocess_text_environment_overrides_utf8_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(production_runner, "_windows_ansi_encoding", lambda: "cp932")
    environment, encoding = subprocess_text_environment(
        {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        platform_name="nt",
    )
    assert encoding == "cp932"
    assert environment["PYTHONUTF8"] == "0"
    assert environment["PYTHONIOENCODING"] == "cp932:replace"


def test_launcher_uses_matched_subprocess_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        stdout = iter(["worker output\n"])

        @staticmethod
        def wait() -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        production_runner,
        "subprocess_text_environment",
        lambda: ({"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp932:replace"}, "cp932"),
    )
    monkeypatch.setattr(production_runner.subprocess, "Popen", fake_popen)
    Launcher(tmp_path).run(["python", "worker.py"], label="encoding test")
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["encoding"] == "cp932"
    assert kwargs["errors"] == "replace"
    assert kwargs["env"] == {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp932:replace"}


def test_profile_command_uses_request_paths_and_python_module(tmp_path: Path) -> None:
    request = resolve_training_cli(minimal_cli("--output_name=portable"))
    command = profile_command(
        request,
        run_dir=tmp_path,
        source_map=tmp_path / "source.json",
        name="01_core",
        protocol="v24-acceptance-local",
        range_muls=(2.70, 3.15, 3.45),
        max_images=16,
    )
    assert command[1:3] == ["-m", "accelerate.commands.launch"]
    assert "--num_processes=1" in command
    assert "--num_machines=1" in command
    assert "--num_cpu_threads_per_process" in command
    from accelerate.commands.launch import launch_command_parser

    launch_args = launch_command_parser().parse_args(command[3:])
    assert launch_args.num_processes == 1
    assert launch_args.num_machines == 1
    assert f"--pretrained_model_name_or_path={request.model_path}" in command
    assert f"--dataset_config={request.dataset_config}" in command
    assert "--dq_profile_protocol=v24-acceptance-local" in command


def test_standard_profile_command_carries_distinct_execution_and_internal_levels(
    tmp_path: Path,
) -> None:
    request = resolve_training_cli(
        minimal_cli("--output_name=standard"),
        execution_mode_name="standard",
    )
    command = profile_command(
        request,
        run_dir=tmp_path,
        source_map=tmp_path / "source.json",
        name="01_core",
        protocol="v24-acceptance-local",
        range_muls=request.execution_mode.core_grid,
        max_images=16,
    )
    assert "--dq_profile_execution_mode=standard" in command
    assert "--dq_profile_qa_depth=standard_smoke" in command
    assert "--dq_profile_level=standard" in command
    assert "--dq_profile_prefix_short_steps=8" in command
    assert "--dq_profile_prefix_long_steps=16" in command
    assert "--dq_profile_range_muls=2.70,3.15,3.45,3.75,4.05" in command


def test_quick_profile_command_carries_reduced_sampling_provenance(
    tmp_path: Path,
) -> None:
    request = resolve_training_cli(
        minimal_cli("--output_name=quick"),
        execution_mode_name="quick",
    )
    command = profile_command(
        request,
        run_dir=tmp_path,
        source_map=tmp_path / "source.json",
        name="01_core",
        protocol="v24-acceptance-local",
        range_muls=request.execution_mode.core_grid,
        max_images=request.local_measurement.max_images,
    )
    assert "--dq_profile_execution_mode=quick" in command
    assert "--dq_profile_qa_depth=quick_smoke" in command
    assert "--dq_profile_measurement_contract=local-body-tail-quick-v1" in command
    assert "--dq_profile_sampling_depth=reduced_16_image" in command
    assert "--dq_profile_confidence_ceiling=reduced_descriptive" in command
    assert "--dq_profile_max_images=16" in command


def test_dry_run_serializes_the_required_prefix_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dirs = []
    for source_index in range(4):
        source = tmp_path / f"source-{source_index}"
        source.mkdir()
        for image_index in range(2):
            (source / f"image-{image_index}.png").write_bytes(b"image")
        source_dirs.append(source)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        "[general]\n"
        "resolution=1024\n"
        "[[datasets]]\n"
        + "".join(
            "[[datasets.subsets]]\n" + f"image_dir = {json.dumps(str(source))}\n"
            for source in source_dirs
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    request = resolve_training_cli(
        [
            f"--pretrained_model_name_or_path={model}",
            f"--dataset_config={dataset}",
            "--output_name=dry-run",
        ],
        execution_mode_name="standard",
    )
    monkeypatch.setattr(production_runner, "preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        production_runner,
        "build_protocol_fingerprint",
        lambda *_args, **_kwargs: {"fingerprint_sha256": "a" * 64},
    )
    output_base = tmp_path / "reports"
    output_base.mkdir()
    monkeypatch.setattr(
        production_runner,
        "validate_output_base",
        lambda value, **_kwargs: value.resolve(),
    )
    result = production_runner.run_profile_request(
        request,
        production_runner.ProductionRunOptions(
            output_base=output_base,
            dry_run=True,
        ),
    )
    payload = json.loads((result.run_dir / "dry_run_command.json").read_text(encoding="utf-8"))
    expected_gate = (
        result.run_dir
        / production_runner.stage_names()["prefix"]
        / "calibration_gate.json"
    )
    assert payload["requires_prefix_gate"] == str(expected_gate)
    assert f"--dq_profile_prefix_gate_file={expected_gate}" in payload["argv"]
    plan = json.loads((result.run_dir / "execution_plan.json").read_text(encoding="utf-8"))
    assert plan["execution_mode"] == "standard"
    assert plan["qa_depth"] == "standard_smoke"
    assert plan["work_volume"]["gpu_process_count"] == 4
    assert plan["work_volume"]["prefix"]["total_branch_updates"] == 64
    assert plan["work_volume"]["local"]["fixed_grid"] == [2.7, 3.15, 3.45, 3.75, 4.05]
    assert plan["work_volume"]["local"]["total_local_probes"] == 736
    assert plan["reference_time_estimate"]["is_guarantee"] is False


def test_execution_plan_matches_the_known_13_image_standard_work_volume(
    tmp_path: Path,
) -> None:
    source_dirs = []
    for source_index, image_total in enumerate((4, 3, 3, 3)):
        source = tmp_path / f"source-{source_index}"
        source.mkdir()
        for image_index in range(image_total):
            (source / f"image-{image_index}.png").write_bytes(b"image")
        source_dirs.append(source)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        "[general]\n"
        "resolution=1024\n"
        "[[datasets]]\n"
        + "".join(
            "[[datasets.subsets]]\n"
            + f"image_dir = {json.dumps(str(source))}\n"
            + "num_repeats = 40\n"
            for source in source_dirs
        ),
        encoding="utf-8",
    )
    request = resolve_training_cli(
        [
            f"--pretrained_model_name_or_path={tmp_path / 'model.safetensors'}",
            f"--dataset_config={dataset}",
        ],
        execution_mode_name="standard",
    )
    plan = build_execution_plan(request, image_count=13, probe_budget=13)
    work = plan["work_volume"]
    assert work["warmup_boundary_updates"] == 1040
    assert work["total_warmup_updates"] == 4160
    assert work["prefix"]["checkpoints"] == [0, 1, 4, 8]
    assert work["prefix"]["total_branch_updates"] == 64
    assert work["local"]["no_quant_probes"] == 156
    assert work["local"]["candidate_probes"] == 1040
    assert work["local"]["total_local_probes"] == 1196
    assert plan["reference_time_estimate"]["minutes"]["minimum"] > 0

    quick_request = resolve_training_cli(
        [
            f"--pretrained_model_name_or_path={tmp_path / 'model.safetensors'}",
            f"--dataset_config={dataset}",
        ],
        execution_mode_name="quick",
    )
    quick_plan = build_execution_plan(
        quick_request,
        image_count=13,
        probe_budget=13,
    )
    quick_work = quick_plan["work_volume"]
    assert quick_plan["measurement_contract"] == "local-body-tail-quick-v1"
    assert quick_work["gpu_process_count"] == 3
    assert quick_work["standalone_snapshot_count"] == 1
    assert quick_work["total_warmup_updates"] == 3120
    assert quick_work["prefix"]["total_branch_updates"] == 64


def test_quick_pipeline_skips_the_redundant_second_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = resolve_training_cli(minimal_cli(), execution_mode_name="quick")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    profile_calls: list[tuple[str, bool]] = []
    parity_calls: list[tuple[str, str, str]] = []

    def fake_run_profile(_launcher, _request, **kwargs):
        name = str(kwargs["name"])
        profile_calls.append((name, bool(kwargs.get("snapshot_only"))))
        output = run_dir / name
        output.mkdir()
        if name == production_runner.stage_names()["prefix"]:
            (output / "calibration_gate.json").write_text("{}", encoding="utf-8")
        return output

    def fake_snapshot_parity(_launcher, left, right, output_name):
        parity_calls.append((left.name, right.name, output_name))
        return run_dir / output_name

    def fake_analysis(_launcher, **kwargs):
        output = run_dir / str(kwargs["output_name"])
        output.mkdir()
        (output / "local_selection.json").write_text(
            json.dumps({"selection_valid": True}),
            encoding="utf-8",
        )
        return output

    class FakeLauncher:
        dry_run = False

        def __init__(self) -> None:
            self.run_dir = run_dir

        def run(self, _command, *, label: str) -> None:
            assert label == "prefix/source contract gate"

    monkeypatch.setattr(production_runner, "run_profile", fake_run_profile)
    monkeypatch.setattr(production_runner, "run_snapshot_parity", fake_snapshot_parity)
    monkeypatch.setattr(production_runner, "run_local_analysis", fake_analysis)
    run_local_pipeline(
        FakeLauncher(),
        request,
        source_map=tmp_path / "source-map.json",
        probe_budget=16,
        dataset_id="quick",
    )
    names = production_runner.stage_names()
    assert profile_calls == [
        (names["snapshot_a"], True),
        (names["prefix"], False),
        (names["core_raw"], False),
    ]
    assert parity_calls == [
        (names["snapshot_a"], names["prefix"], names["prefix_snapshot_parity"])
    ]
    assert not (run_dir / names["snapshot_b"]).exists()
    assert not (run_dir / names["snapshot_parity"]).exists()


def test_product_artifacts_are_promoted_to_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis = tmp_path / "analysis"
    profile = tmp_path / "profile"
    run_dir.mkdir()
    analysis.mkdir()
    profile.mkdir()
    for name in ("report.html", "technical_report.html", "summary.json", "practical_report.json"):
        (analysis / name).write_text(name, encoding="utf-8")
    (profile / "source_manifest.json").write_text("{}", encoding="utf-8")
    (profile / "candidate_definitions.json").write_text("{}", encoding="utf-8")
    promoted = promote_analysis(run_dir, analysis)
    promoted.extend(promote_profile_provenance(run_dir, profile))
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "source_manifest.json").is_file()
    assert "candidate_definitions.json" in promoted


def test_direct_module_entry_preserves_training_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_profile_mode(training_argv: list[str], **kwargs: object) -> int:
        captured["training_argv"] = training_argv
        captured["kwargs"] = kwargs
        return 17

    monkeypatch.setattr(production_main, "run_profile_mode", fake_run_profile_mode)
    training = [
        "--dataset_config=D:\\space path\\日本語.toml",
        "--pretrained_model_name_or_path=D:\\models\\model.safetensors",
        "--network_args",
        "rank_dropout=0.2",
    ]
    result = production_main.main(["--dq-profile-name=test", *training])
    assert result == 17
    assert captured["training_argv"] == training
    assert captured["kwargs"] == {
        "preset_name": "canonical-v1",
        "execution_mode_name": "standard",
        "output_base": DEFAULT_OUTPUT_BASE,
        "profile_name": "test",
        "preflight_only": False,
        "dry_run": False,
        "open_report": False,
    }


def test_direct_module_entry_selects_standard_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_profile_mode(training_argv: list[str], **kwargs: object) -> int:
        captured["training_argv"] = training_argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(production_main, "run_profile_mode", fake_run_profile_mode)
    training = [
        f"--dataset_config={DATASET}",
        f"--pretrained_model_name_or_path={MODEL}",
    ]
    assert production_main.main(["--dq-profile-mode=standard", *training]) == 0
    assert captured["training_argv"] == training
    assert captured["kwargs"]["execution_mode_name"] == "standard"


def test_direct_module_entry_selects_quick_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_profile_mode(training_argv: list[str], **kwargs: object) -> int:
        captured["training_argv"] = training_argv
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(production_main, "run_profile_mode", fake_run_profile_mode)
    training = [
        f"--dataset_config={DATASET}",
        f"--pretrained_model_name_or_path={MODEL}",
    ]
    assert production_main.main(["--dq-profile-mode=quick", *training]) == 0
    assert captured["training_argv"] == training
    assert captured["kwargs"]["execution_mode_name"] == "quick"


def test_profile_name_sanitization_is_windows_safe() -> None:
    assert sanitize_profile_name("  A/B: test  ") == "A_B__test"
    assert sanitize_profile_name("CON") == "dq_CON"
    assert sanitize_profile_name("CON.txt") == "dq_CON.txt"
    assert sanitize_profile_name("LPT1.run") == "dq_LPT1.run"
    assert sanitize_profile_name("COM10.txt") == "COM10.txt"
