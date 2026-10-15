"""Budgeted hierarchical diagnosis followed by independently initialized full runs."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import psutil

from dq_profile.research_budget import ResearchBudget, ResearchStop, file_hash, proposal_hash, read_json, write_json

REPOSITORY = Path(__file__).resolve().parent


def gpu_idle(*, allowed_pids=(), desktop_baseline=()):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    desktop = set()
    for pid in pids:
        try:
            process = psutil.Process(pid)
            for record in desktop_baseline:
                if process.name().casefold() != record["name"].casefold():
                    continue
                exact_instance = pid == record["pid"] and process.create_time() == record["created_at"]
                # A reviewed desktop application may restart during a long run.
                # Require its full executable path; a matching name alone is
                # insufficient. Python/CUDA workers are never baseline entries.
                same_executable = bool(record.get("exe")) and process.exe().casefold() == record["exe"].casefold()
                if exact_instance or same_executable:
                    desktop.add(pid)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    other = pids - set(allowed_pids) - desktop
    if other:
        raise ResearchStop("Another CUDA process is active: " + ",".join(map(str, sorted(other))))


def job_processes(process):
    try:
        parent = psutil.Process(process.pid)
        return [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def terminate_job(process, known_members=()):
    # Windows venv launchers can start a second python.exe. Terminating only
    # the launcher would leave the GPU worker alive and invalidate accounting.
    members = list({(item.pid, item.create_time()): item for item in [*known_members, *job_processes(process)] if item.is_running()}.values())
    for member in reversed(members):
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(members, timeout=20)
    for member in alive:
        member.kill()
    process.wait(timeout=10)


def prepare(workspace, *, revision=False):
    budget = ResearchBudget(workspace, require_finalized=False)
    revision_number = 1
    supersedes = None
    if revision:
        previous = read_json(budget.root / "execution_contract.json")
        revision_number = int(previous.get("revision", 1)) + 1
        review = read_json(budget.root / "repair_review.json")
        archive = budget.root / "revisions" / f"revision_{revision_number - 1:02d}"
        if (budget.root / "runner.lock").exists() or read_json(budget.root / "usage.json")["active_job"]:
            raise ResearchStop("Cannot revise a contract while a job is active")
        if any(job["mode"] == "train" for job in read_json(budget.root / "jobs.json")):
            raise ResearchStop("This repair workflow is limited to pre-training validation")
        if revision_number > 1 + budget.limits["same_job_transient_retries"]:
            raise ResearchStop("The approved attempt allowance is exhausted")
        if review.get("previous_contract_sha256") != proposal_hash(previous) or not review.get("reason"):
            raise ResearchStop("Repair review does not identify the previous contract and reason")
        for name in ("execution_contract.json", "approval.json", "usage.json", "counters.json", "jobs.json", "state.json"):
            if file_hash(archive / name) != file_hash(budget.root / name):
                raise ResearchStop("Previous contract or cumulative usage was not preserved: " + name)
        if file_hash(archive / "frozen_sources.zip") != file_hash(budget.root / "frozen_sources.zip"):
            raise ResearchStop("Original frozen sources were not preserved")
        supersedes = {"contract_sha256": proposal_hash(previous), "archive": str(archive),
                      "repair_review_sha256": file_hash(budget.root / "repair_review.json")}
    elif (budget.root / "usage.json").exists():
        raise ResearchStop("Cannot replace the contract after execution begins")
    if read_json(budget.root / "fixture_results.json").get("status") != "pass":
        raise ResearchStop("Implementation fixtures have not passed")
    from tools.check_dq_profile_copy_drift import validate_copy_manifest
    validate_copy_manifest(REPOSITORY)
    budget.validate_inputs(full_model_hash=True, check_code=not revision)
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py", "dq_profile/copied_sources.json"], cwd=REPOSITORY, text=True).splitlines()
    code = {name: file_hash(REPOSITORY / name) for name in sorted(set(names))}
    prior = Path(budget.proposal["prior_evidence_root"]) / "runs/measure_01"
    prior_names = ["snapshot_parity.json", "P5_dropout_confirmation/per_image.jsonl", "P5_dropout_confirmation/gradient_tail.jsonl", "P5_dropout_confirmation/result.json"]
    contract = {"repository": str(REPOSITORY), "code_files": code,
        "input_manifest_sha256": proposal_hash(budget.manifest), "source_map_sha256": file_hash(budget.root / "source_group_map.json"),
        "baseline_sha256": file_hash(budget.root / "baseline_contract.private.json"),
        "fixture_receipt_sha256": file_hash(budget.root / "fixture_results.json"),
        "approved_limits": budget.approval["approved_limits"], "approved_stages": budget.approval["approved_stages"],
        "approved_images": sum(len(source["image_files"]) for source in budget.manifest["sources"]),
        "prior_evidence_files": {str(prior / name): file_hash(prior / name) for name in prior_names},
        "support_files": {str(budget.root / "gpu_desktop_baseline.json"): file_hash(budget.root / "gpu_desktop_baseline.json")},
        "reserved_full_training_calls": 6 * (13600 + 1000),
        "training_reserve_note": "six scheduled runs plus a conservative allowance for forward-only avg shadow scoring",
        "revision": revision_number, "supersedes": supersedes,
        "created_at": time.time()}
    if supersedes:
        previous_archive = Path(supersedes["archive"])
        for name in ("archive_manifest.json", "execution_contract.json", "frozen_sources.zip", "fixture_results.json"):
            path = previous_archive / name
            contract["support_files"][str(path)] = file_hash(path)
    source_archive = "frozen_sources.zip" if revision_number == 1 else f"frozen_sources_revision_{revision_number}.zip"
    with zipfile.ZipFile(budget.root / source_archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            archive.write(REPOSITORY / name, name)
    write_json(budget.root / "execution_contract.json", contract)
    approval = budget.approval
    approval.update(contract_finalized=True, real_data_execution_allowed=True, execution_contract_sha256=proposal_hash(contract))
    write_json(budget.root / "approval.json", approval)
    print("Continuation contract finalized; no real-data job started.", flush=True)


def train_worker(budget, name, condition):
    import torch
    torch.set_num_threads(8)
    from dq_profile.full_training import BudgetTrainingObserver
    from sdxl_dq_mul_research import training_tokens
    import sdxl_train_network as ordinary
    selection = read_json(budget.root / "training_selection.json")
    item = next(row for row in selection["candidates"] if row["id"] == condition)
    output = budget.root / "training" / name
    output.mkdir(parents=True, exist_ok=False)
    policy_path = output / "policy.json"
    write_json(policy_path, item["policy"])
    flags = dict(read_json(budget.root / "baseline_contract.private.json")["user_training_flags"])
    flags.update(output_dir=str(output), output_name="mul_" + condition, dq_delta_policy_file=str(policy_path),
                 dq_delta_range_mul=item["policy"]["base_mul"], training_comment="fixed-mul full research")
    parser = ordinary.setup_parser()
    args = parser.parse_args(training_tokens(flags))
    ordinary.train_util.verify_command_line_training_args(args)
    ordinary.maruoCfg.downscale_freq_shift = bool(args.downscale_freq_shift)
    ordinary.maruoCfg.te_mlp_fc_only = bool(args.te_mlp_fc_only)
    mode = ordinary.resolve_fp16_safe_norms_mode(args)
    ordinary.maruoCfg.fp16_safe_norms_mode = mode
    ordinary.maruoCfg.fp16_safe_norms = mode != "off"
    args.fp16_safe_norms_mode_resolved = mode
    for field in ("sample_at_first", "sample_every_n_epochs", "sample_every_n_steps", "huggingface_repo_id", "resume", "network_weights"):
        if getattr(args, field, None):
            raise ResearchStop("Full research must not sample, upload, resume, or load prior LoRA weights")
    write_json(output / "launch_flags.private.json", flags)
    handling = read_json(budget.root / "execution_contract.json").get("training_gradient_handling", "strict")
    if handling != budget.approval.get("training_gradient_handling", "strict"):
        raise ResearchStop("Training gradient handling differs from approval")
    observer = BudgetTrainingObserver(budget, output, item, selection["assignment_hashes"][condition], gradient_handling=handling)
    trainer = ordinary.SdxlNetworkTrainer()
    trainer.training_observer = observer
    trainer.train(args)
    observer.finish()


def worker(workspace, mode, name, *, gate=None, condition=None):
    # Import identical research modules for both diagnostic workers so their
    # independently computed source contracts include the same Python files.
    from dq_profile.mul_hierarchical import HierarchicalMulRuntime
    from dq_profile.full_training import BudgetTrainingObserver
    from sdxl_dq_mul_research import worker as diagnostic_worker
    budget = ResearchBudget(workspace)
    active = budget.usage().get("active_job")
    if not active or active["name"] != name or not (budget.root / "runner.lock").is_file():
        raise ResearchStop("Worker has no matching budgeted supervisor")
    budget.stage = "P6" if mode == "train" else "P1"
    budget.validate_inputs(full_model_hash=True)
    budget.check(storage=True)
    try:
        # Cache encoding is forward-only and precedes trainer hooks. Reserve
        # one call per approved image conservatively, even if encoding batches
        # or a failure would reduce the number actually executed.
        for _ in range(sum(len(source["image_files"]) for source in budget.manifest["sources"])):
            budget.consume_forward_backward("latent_cache_image_upper_bound")
        if mode == "train":
            train_worker(budget, name, condition)
        else:
            diagnostic_worker(workspace, mode, name, gate, runtime_class=HierarchicalMulRuntime, guard_warmup=True)
    except BaseException as exc:
        write_json(budget.root / (name + "_failure.json"), {"type": type(exc).__name__, "reason": str(exc), "guard_stop": isinstance(exc, ResearchStop), "at": time.time()})
        raise


def job_attempt_limit(budget, contract, mode, key):
    extra = contract.get("additional_training_attempts", {})
    if extra:
        if extra != budget.approval.get("additional_training_attempts") or not budget.approval.get("recovery_approval_reference"):
            raise ResearchStop("Additional training attempts lack matching approval")
        if any(type(count) is not int or count <= 0 for count in extra.values()):
            raise ResearchStop("Invalid additional training attempt allowance")
    return 1 + budget.limits["same_job_transient_retries"] + (extra.get(key, 0) if mode == "train" else 0)


def run_job(budget, jobs, mode, *, gate=None, definition=None):
    key = definition["id"] if definition else mode
    contract = read_json(budget.root / "execution_contract.json")
    revision = int(contract.get("revision", 1))
    previous_attempts = [job for job in jobs if job["key"] == key]
    matching = [job for job in previous_attempts if job.get("contract_revision", 1) == revision]
    completed = [job for job in matching if job["status"] == "complete"]
    if completed:
        job = completed[-1]
        if file_hash(job["result_file"]) != job["result_sha256"]:
            raise ResearchStop("Completed job artifact changed")
        return job
    if matching:
        raise ResearchStop("A previous attempt needs failure review before any retry: " + key)
    if len(previous_attempts) >= job_attempt_limit(budget, contract, mode, key):
        raise ResearchStop("The approved attempt allowance is exhausted: " + key)
    desktop = read_json(budget.root / "gpu_desktop_baseline.json")["processes"]
    gpu_idle(allowed_pids=[os.getpid()], desktop_baseline=desktop)
    budget.stage = "P6" if mode == "train" else "P1"
    budget.validate_inputs()
    budget.check(reserve_seconds=300, storage=True)
    name = ("train_" + key if mode == "train" else mode) + f"_{len(previous_attempts) + 1:02d}"
    result_path = (budget.root / "training" / name / "training_complete.json") if mode == "train" else (budget.root / "runs" / name / ("calibration_gate.json" if mode == "prefix" else "research_summary.json"))
    usage_path = budget.root / "usage.json"
    usage = read_json(usage_path) if usage_path.exists() else {"gpu_job_seconds": 0., "pilot_gpu_job_seconds": 0., "first_started_at": None, "active_job": None}
    if usage["active_job"]:
        raise ResearchStop("An unreconciled GPU job exists")
    now = time.time()
    usage["first_started_at"] = usage["first_started_at"] or now
    usage["active_job"] = {"name": name, "stage": budget.stage, "started_at": now}
    write_json(usage_path, usage)
    job = {"key": key, "name": name, "mode": mode, "contract_revision": revision, "status": "running", "started_at": now, "result_file": str(result_path)}
    jobs.append(job)
    write_json(budget.root / "jobs.json", jobs)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--workspace", str(budget.root), "--worker", mode, "--name", name]
    if gate:
        command += ["--gate", str(gate)]
    if definition:
        command += ["--condition", definition["id"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    process = None
    known_members = {}
    try:
        with (budget.root / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job["pid"] = process.pid
            write_json(budget.root / "jobs.json", jobs)
            check_number = 0
            while process.poll() is None:
                for member in job_processes(process):
                    known_members[(member.pid, member.create_time())] = member
                current = read_json(usage_path).get("active_job")
                budget.stage = current["stage"] if current else budget.stage
                budget.check(storage=True)
                if check_number % 5 == 0:
                    gpu_idle(allowed_pids=[os.getpid(), *(item.pid for item in known_members.values() if item.is_running())], desktop_baseline=desktop)
                check_number += 1
                time.sleep(2)
        if process.returncode != 0 or not result_path.is_file():
            raise ResearchStop("GPU job failed; inspect its private log: " + name)
        result = read_json(result_path)
        if mode == "prefix":
            valid = result.get("passed") is True and result.get("gate") in {"pass_exact", "pass_numeric"}
        else:
            valid = result.get("status") == "complete"
        if not valid:
            raise ResearchStop("GPU job did not pass its completion gate: " + name)
        if mode == "measure":
            status_path = result_path.parent / "status.json"
            status = read_json(status_path) if status_path.exists() else {}
            status.update(status="complete", protocol=result["protocol"], research_summary_sha256=file_hash(result_path))
            write_json(status_path, status)
        job.update(status="complete", result_sha256=file_hash(result_path))
        print("CONTINUATION_JOB_COMPLETE " + name, flush=True)
    finally:
        if process is not None and (process.poll() is None or any(item.is_running() for item in known_members.values())):
            terminate_job(process, known_members.values())
        usage = read_json(usage_path)
        active = usage["active_job"]
        if active:
            elapsed = max(0., time.time() - active["started_at"])
            usage["gpu_job_seconds"] += elapsed
            if active["stage"] == "P1":
                usage["pilot_gpu_job_seconds"] += elapsed
        usage["active_job"] = None
        write_json(usage_path, usage)
        job.update(exit_code=None if process is None else process.returncode, finished_at=time.time())
        if job["status"] != "complete":
            job["status"] = "failed"
        write_json(budget.root / "jobs.json", jobs)
    return job


def checkpoint_catalog(root, jobs):
    rows = []
    for job in jobs:
        if job["mode"] == "train" and job["status"] == "complete":
            rows.extend(read_json(Path(job["result_file"]).parent / "checkpoint_index.json")["rows"])
    write_json(root / "checkpoint_catalog.json", {"rows": rows, "not_image_quality": True})
    cells = []
    for row in rows:
        values = [row["condition"], row["epoch"], row["step"], row["variant"], row["provenance"], row["path"]]
        cells.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in values) + "</tr>")
    document = """<!doctype html><html lang="ja"><meta charset="utf-8"><title>Checkpoint catalog</title>
<style>body{font-family:system-ui;margin:24px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}td:last-child{word-break:break-all}input{padding:8px;width:28em}</style>
<h1>LoRAチェックポイント一覧</h1><p>診断から画質順位は決めていません。raw/平均化と学習中のpromoteの履歴は各条件のログを参照してください。</p>
<input placeholder="条件・epoch・ファイル名で絞り込み" oninput="document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))">
<table><thead><tr><th>条件</th><th>epoch</th><th>step</th><th>種別</th><th>保存履歴</th><th>パス</th></tr></thead><tbody>""" + "".join(cells) + "</tbody></table></html>"
    (root / "checkpoint_catalog.html").write_text(document, encoding="utf-8")


def launch(workspace):
    budget = ResearchBudget(workspace)
    budget.validate_inputs(full_model_hash=True)
    lock = budget.root / "runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, stream)
    jobs = read_json(budget.root / "jobs.json") if (budget.root / "jobs.json").exists() else []
    try:
        prefix = run_job(budget, jobs, "prefix")
        measured = run_job(budget, jobs, "measure", gate=prefix["result_file"])
        result = read_json(measured["result_file"])
        selection = {"candidates": result["full_training_candidates"], "assignment_hashes": result["candidate_assignment_hashes"],
                     "diagnostic_result_sha256": measured["result_sha256"], "selection_reasons": result["selection_reasons"]}
        if not 5 <= len(selection["candidates"]) <= 6:
            raise ResearchStop("Training selection has an unexpected number of conditions")
        selection_path = budget.root / "training_selection.json"
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ResearchStop("Frozen training selection changed")
        write_json(selection_path, selection)
        budget.stage = "P6"
        for definition in selection["candidates"]:
            remaining = [item for item in selection["candidates"] if not any(job["key"] == item["id"] and job["status"] == "complete" for job in jobs)]
            completed_seconds = [job["finished_at"] - job["started_at"] for job in jobs if job["mode"] == "train" and job["status"] == "complete"]
            per_run_seconds = max([3 * 3600, *(seconds * 1.2 for seconds in completed_seconds)])
            budget.check(reserve_fb=len(remaining) * (13600 + 1000), reserve_seconds=len(remaining) * per_run_seconds, storage=True)
            run_job(budget, jobs, "train", definition=definition)
            checkpoint_catalog(budget.root, jobs)
        write_json(budget.root / "state.json", {"status": "execution_complete_pending_final_audit", "completed_training_conditions": len(selection["candidates"]), "updated_at": time.time()})
    except BaseException as exc:
        write_json(budget.root / "state.json", {"status": "stopped", "reason": str(exc), "updated_at": time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-revision", action="store_true", help="Freeze a reviewed pre-training repair with preserved evidence and cumulative usage")
    parser.add_argument("--worker", choices=("prefix", "measure", "train"))
    parser.add_argument("--name")
    parser.add_argument("--gate")
    parser.add_argument("--condition")
    args = parser.parse_args()
    if args.prepare or args.prepare_revision:
        prepare(args.workspace, revision=args.prepare_revision)
    elif args.worker:
        if not args.name or (args.worker == "train" and not args.condition):
            parser.error("Worker name and training condition are required")
        worker(args.workspace, args.worker, args.name, gate=args.gate, condition=args.condition)
    else:
        launch(args.workspace)


if __name__ == "__main__":
    main()
