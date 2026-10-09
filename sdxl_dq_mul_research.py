"""Run an explicitly approved private fixed-mul research plan.

The workspace contains local approval/input records and is never a dataset
discovery mechanism. Ordinary training and the canonical profiler are separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dq_profile.research_budget import ResearchBudget, ResearchStop, file_hash, proposal_hash, read_json, write_json


REPOSITORY = Path(__file__).resolve().parent


def training_tokens(flags):
    tokens = []
    for key, value in flags.items():
        if isinstance(value, bool):
            if value:
                tokens.append("--" + key)
            elif key == "avg_reset_stats":
                tokens.append("--no-avg_reset_stats")
        elif isinstance(value, list):
            tokens.extend(["--" + key, *map(str, value)])
        elif value is not None:
            tokens.append(f"--{key}={value}")
    return tokens


def prepare(workspace):
    budget = ResearchBudget(workspace, require_finalized=False)
    if (budget.root / "usage.json").exists():
        raise ResearchStop("An execution contract cannot be replaced after jobs start")
    receipt = read_json(budget.root / "fixture_results.json")
    if receipt.get("status") != "pass":
        raise ResearchStop("Implementation fixtures have not passed")
    budget.validate_inputs(full_model_hash=True)
    from dq_profile.production_runner import build_source_map
    source_map, count = build_source_map([Path(s["path"]) for s in budget.manifest["sources"]], dataset_key=budget.manifest["dataset_id"])
    write_json(budget.root / "source_group_map.json", source_map)
    names = subprocess.check_output(["git", "ls-files", "--", "*.py", "dq_profile/copied_sources.json"], cwd=REPOSITORY, text=True).splitlines()
    names += ["library/dq_mul_policy.py", "dq_profile/research_budget.py", "dq_profile/mul_research.py", "sdxl_dq_mul_research.py"]
    code = {name: file_hash(REPOSITORY / name) for name in sorted(set(names))}
    contract = {"repository": str(REPOSITORY), "code_files": code, "input_manifest_sha256": proposal_hash(budget.manifest), "fixture_receipt_sha256": file_hash(budget.root / "fixture_results.json"), "source_map_sha256": file_hash(budget.root / "source_group_map.json"), "baseline_sha256": file_hash(budget.root / "baseline_contract.private.json"), "approved_limits": budget.approval["approved_limits"], "approved_stages": budget.approval["approved_stages"], "approved_images": count, "created_at": time.time()}
    write_json(budget.root / "execution_contract.json", contract)
    approval = budget.approval
    approval.update(contract_finalized=True, real_data_execution_allowed=True, execution_contract_sha256=proposal_hash(contract))
    write_json(budget.root / "approval.json", approval)
    print("Execution contract finalized; no real-data job started.")


def worker(workspace, mode, name, gate=None, *, runtime_class=None, guard_warmup=False):
    budget = ResearchBudget(workspace)
    active = budget.usage().get("active_job")
    if not active or active["name"] != name or not (budget.root / "runner.lock").is_file():
        raise ResearchStop("Workers must be started by the budgeted runner")
    budget.validate_inputs(full_model_hash=True)
    budget.check(storage=True)
    import torch
    torch.set_num_threads(8)
    import sdxl_dq_dataset_profile as standard
    # Import the same research modules for both processes. Their file hashes
    # also belong to the explicit source manifest used by the prefix gate.
    from dq_profile.mul_research import MulResearchRuntime
    baseline = read_json(budget.root / "baseline_contract.private.json")
    flags = dict(baseline["user_training_flags"])
    flags["training_comment"] = "fixed-mul research"
    images = sum(len(s["image_files"]) for s in budget.manifest["sources"])
    tokens = training_tokens(flags) + [
        f"--output_dir={budget.root / 'runs'}", f"--output_name={name}",
        f"--dq_profile_output_dir={budget.root / 'runs'}", f"--dq_profile_name={name}",
        "--dq_profile_level=standard", "--dq_profile_execution_mode=standard", "--dq_profile_qa_depth=standard_smoke",
        "--dq_profile_measurement_contract=local-body-tail-v2", "--dq_profile_prefix_kernel_mode=deterministic",
        "--dq_profile_prefix_short_steps=8", "--dq_profile_prefix_long_steps=16", "--dq_profile_data_diagnostics=off",
        f"--dq_profile_source_group_map={budget.root / 'source_group_map.json'}",
        f"--dq_profile_max_images={8 if mode == 'prefix' else images}",
        "--dq_profile_timestep_bins=4", "--dq_profile_stochastic_repeats=2",
        "--dq_profile_range_muls=" + ("2.70,3.15,3.45,3.75,4.05" if runtime_class is not None else "2.70,3.15,3.75"),
        "--dq_profile_protocol=" + ("v2-prefix-smoke" if mode == "prefix" else "v24-acceptance-local"),
    ]
    if gate:
        tokens.append(f"--dq_profile_prefix_gate_file={gate}")
    parser = standard.setup_parser()
    args = parser.parse_args(tokens)
    standard.train_util.verify_command_line_training_args(args)
    standard._configure_prefix_kernel_policy(args)
    standard._validate_and_isolate(args)
    standard._preflight(args)
    standard.configure_sdxl_globals(args)
    if args.cache_text_encoder_outputs or getattr(args, "cache_text_encoder_outputs_to_disk", False):
        raise ResearchStop("Trainable TE embeddings cannot be cached")
    if getattr(args, "dq_delta_auto_range_mul", False) or args.dq_quantize_z or args.dq_delta_stat != "rms" or args.dq_delta_granularity != "channel" or args.dq_delta_mode != "stoch":
        raise ResearchStop("Research policy requires static channel/RMS stochastic delta quantization")
    args._dq_profile_budget_callback = budget.consume_forward_backward
    args._dq_research_budget = budget
    if guard_warmup:
        from dq_profile.full_training import check_backward, check_updated_state
        args._dq_profile_after_backward = check_backward
        args._dq_profile_after_update = check_updated_state
    if mode == "measure":
        args._dq_profile_runtime_class = runtime_class or MulResearchRuntime
    artifacts = standard.ProfileArtifacts(args.dq_profile_run_dir)
    artifacts.initialize()
    artifacts.ensure_known_result()
    args.dq_profile_execution_log_path = str(artifacts.root / "execution.log")
    trainer = standard.SdxlNetworkTrainer()
    try:
        trainer.train(args)
    except BaseException as exc:
        artifacts.mark_failed(exc)
        raise
    finally:
        handler = getattr(trainer, "_dq_profile_execution_log_handler", None)
        if handler is not None:
            standard.logging.getLogger().removeHandler(handler)
            handler.close()


def _finalize_research_status(result_path):
    status_path = result_path.parent / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    status.update(status="complete", protocol="dq-mul-research-v1", research_summary_sha256=file_hash(result_path))
    write_json(status_path, status)


def launch(workspace):
    budget = ResearchBudget(workspace)
    budget.validate_inputs(full_model_hash=True)
    budget.check(storage=True)
    lock = budget.root / "runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, stream)
    jobs_path = budget.root / "jobs.json"
    jobs = read_json(jobs_path) if jobs_path.exists() else {"prefix": [], "measure": []}
    gate = None
    try:
        for mode in ("prefix", "measure"):
            successful = [item for item in jobs[mode] if item.get("status") == "complete"]
            if successful:
                job = successful[-1]
                if file_hash(job["result_file"]) != job["result_sha256"]:
                    raise ResearchStop("Completed artifact checksum changed")
                if mode == "prefix":
                    gate = job["result_file"]
                else:
                    _finalize_research_status(Path(job["result_file"]))
                continue
            if len(jobs[mode]) >= 2:
                raise ResearchStop("Job attempt limit reached")
            attempt = len(jobs[mode]) + 1
            name = f"{mode}_{attempt:02d}"
            # The direct profiler writes calibration_gate.json. The profile_
            # prefix belongs only to the separate production export bundle.
            result_path = budget.root / "runs" / name / ("calibration_gate.json" if mode == "prefix" else "research_summary.json")
            budget.check(reserve_seconds=300, storage=True)
            usage_path = budget.root / "usage.json"
            usage = read_json(usage_path) if usage_path.exists() else {"gpu_job_seconds": 0., "pilot_gpu_job_seconds": 0., "first_started_at": None, "active_job": None}
            if usage["active_job"]:
                raise ResearchStop("An unfinished job must be reconciled before restart")
            now = time.time()
            usage["first_started_at"] = usage["first_started_at"] or now
            usage["active_job"] = {"name": name, "stage": "P1", "started_at": now}
            write_json(usage_path, usage)
            job = {"name": name, "attempt": attempt, "status": "running", "started_at": now, "result_file": str(result_path)}
            jobs[mode].append(job)
            write_json(jobs_path, jobs)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--workspace", str(budget.root), "--worker", mode, "--name", name]
            if gate:
                command += ["--gate", gate]
            log_path = budget.root / f"{name}.log"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                job["pid"] = process.pid
                write_json(jobs_path, jobs)
                try:
                    while process.poll() is None:
                        active = read_json(usage_path).get("active_job")
                        budget.stage = active["stage"] if active else "P1"
                        budget.check(storage=True)
                        time.sleep(2)
                except BaseException:
                    process.terminate()
                    process.wait(timeout=30)
                    raise
                finally:
                    usage = read_json(usage_path)
                    active = usage["active_job"]
                    if active:
                        seconds = max(0, time.time() - active["started_at"])
                        usage["gpu_job_seconds"] += seconds
                        if active["stage"] == "P1":
                            usage["pilot_gpu_job_seconds"] += seconds
                    usage["active_job"] = None
                    write_json(usage_path, usage)
                    job.update(exit_code=process.returncode, completed_at=time.time(), status="failed")
                    write_json(jobs_path, jobs)
            if process.returncode != 0 or not result_path.is_file():
                raise ResearchStop(f"{name} failed; inspect its private log")
            result = read_json(result_path)
            if mode == "prefix" and (result.get("passed") is not True or result.get("gate") not in {"pass_exact", "pass_numeric"}):
                raise ResearchStop("Prefix comparison did not pass")
            if mode == "measure" and result.get("status") != "complete":
                raise ResearchStop("Research output is incomplete")
            if mode == "measure":
                _finalize_research_status(result_path)
            job.update(status="complete", result_sha256=file_hash(result_path))
            write_json(jobs_path, jobs)
            if mode == "prefix":
                gate = str(result_path)
            print(f"RESEARCH_JOB_COMPLETE {name}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", choices=("prefix", "measure"))
    parser.add_argument("--name")
    parser.add_argument("--gate")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.workspace)
    elif args.worker:
        if not args.name:
            parser.error("--worker requires --name")
        worker(args.workspace, args.worker, args.name, args.gate)
    else:
        launch(args.workspace)


if __name__ == "__main__":
    main()
