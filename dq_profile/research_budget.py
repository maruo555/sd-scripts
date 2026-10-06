"""Private approval, immutable inputs and cumulative budgets for research jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


class ResearchStop(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def proposal_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class ResearchBudget:
    def __init__(self, workspace, *, require_finalized=True):
        self.root = Path(workspace).resolve()
        self.approval = read_json(self.root / "approval.json")
        self.proposal = read_json(self.root / "proposal.private.json")
        if self.approval.get("status") != "approved" or not self.approval.get("user_approval_reference"):
            raise ResearchStop("No explicit user approval")
        if self.approval.get("proposal_sha256") != proposal_hash(self.proposal):
            raise ResearchStop("Proposal changed after approval")
        if require_finalized and not self.approval.get("contract_finalized"):
            raise ResearchStop("Execution contract is not finalized")
        self.limits = self.approval["approved_limits"]
        self.manifest = read_json(self.root / "input_manifest.private.json")
        if self.manifest["dataset_id"] != self.proposal["dataset_id"]:
            raise ResearchStop("Dataset is outside the approved scope")
        self.counter_path = self.root / "counters.json"
        self.stage = "P1"
        self._checks = 0

    def validate_inputs(self, *, full_model_hash=False):
        if file_hash(self.manifest["dataset_toml"]) != self.manifest["dataset_toml_sha256"]:
            raise ResearchStop("Dataset TOML changed")
        for source in self.manifest["sources"]:
            expected = {str(Path(row["path"]).resolve()) for row in source["image_files"]}
            actual = {str(p.resolve()) for p in Path(source["path"]).iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}}
            if actual != expected:
                raise ResearchStop("Image inventory changed")
            for row in source["image_files"]:
                if file_hash(row["path"]) != row["sha256"]:
                    raise ResearchStop("An input image changed")
                for extension, field in ((".caption", "caption_exists"), (".txt", "txt_exists")):
                    if Path(row["path"]).with_suffix(extension).exists() != row[field]:
                        raise ResearchStop("Caption inventory changed")
                    if row[field]:
                        raise ResearchStop("This manifest requires caption content hashes before use")
        model = self.manifest["model"]
        stat = Path(model["path"]).stat()
        if stat.st_size != model["bytes"] or stat.st_mtime_ns != model["mtime_ns"]:
            raise ResearchStop("Base model identity changed")
        if not model.get("sha256") or (full_model_hash and file_hash(model["path"]) != model["sha256"]):
            raise ResearchStop("Base model hash is unverified or changed")
        contract_path = self.root / "execution_contract.json"
        if contract_path.exists():
            contract = read_json(contract_path)
            if self.approval.get("execution_contract_sha256") != proposal_hash(contract):
                raise ResearchStop("Execution contract changed")
            if proposal_hash(self.manifest) != contract["input_manifest_sha256"]:
                raise ResearchStop("Input contract changed")
            if file_hash(self.root / "source_group_map.json") != contract["source_map_sha256"]:
                raise ResearchStop("Source map changed")
            if file_hash(self.root / "baseline_contract.private.json") != contract["baseline_sha256"]:
                raise ResearchStop("Training baseline changed")
            if self.approval["approved_limits"] != contract["approved_limits"] or self.approval["approved_stages"] != contract["approved_stages"]:
                raise ResearchStop("Approval scope changed")
            for name, expected in contract["code_files"].items():
                if file_hash(Path(contract["repository"]) / name) != expected:
                    raise ResearchStop(f"Code changed during the experiment: {name}")

    def usage(self):
        path = self.root / "usage.json"
        value = read_json(path) if path.exists() else {"gpu_job_seconds": 0, "pilot_gpu_job_seconds": 0, "first_started_at": None, "active_job": None}
        active = value.get("active_job")
        if active:
            elapsed = max(0, time.time() - active["started_at"])
            value["gpu_job_seconds"] += elapsed
            if active["stage"] == "P1":
                value["pilot_gpu_job_seconds"] += elapsed
        return value

    def check(self, *, reserve_seconds=0, reserve_fb=0, storage=False):
        if self.stage not in self.approval["approved_stages"]:
            raise ResearchStop(f"Stage is not approved: {self.stage}")
        usage = self.usage()
        limit = self.limits
        if usage["gpu_job_seconds"] + reserve_seconds >= limit["all_gpu_job_seconds"]:
            raise ResearchStop("GPU job time budget exhausted")
        if self.stage == "P1" and usage["pilot_gpu_job_seconds"] + reserve_seconds >= limit["pilot_gpu_job_seconds"]:
            raise ResearchStop("Pilot time budget exhausted")
        if usage["first_started_at"] and time.time() - usage["first_started_at"] + reserve_seconds >= limit["elapsed_seconds_from_first_real_data_job"]:
            raise ResearchStop("Elapsed-time budget exhausted")
        counters = read_json(self.counter_path) if self.counter_path.exists() else {"forward_backward_calls": 0}
        if counters["forward_backward_calls"] + reserve_fb > limit["real_data_forward_backward_calls_including_warmup_prefix_retries"]:
            raise ResearchStop("Forward/backward budget exhausted")
        if storage:
            if shutil.disk_usage(self.root).free < limit["minimum_free_bytes_on_D"]:
                raise ResearchStop("Minimum free disk space reached")
            size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if size >= limit["storage_increment_bytes"]:
                raise ResearchStop("Research storage budget exhausted")

    def consume_forward_backward(self, kind="diagnostic"):
        self._checks += 1
        self.check(reserve_fb=1, storage=self._checks % 25 == 1)
        counters = read_json(self.counter_path) if self.counter_path.exists() else {"forward_backward_calls": 0, "by_kind": {}, "by_stage": {}}
        counters["forward_backward_calls"] += 1
        counters["by_kind"][kind] = counters["by_kind"].get(kind, 0) + 1
        counters["by_stage"][self.stage] = counters["by_stage"].get(self.stage, 0) + 1
        counters["updated_at"] = time.time()
        write_json(self.counter_path, counters)

    def set_stage(self, stage):
        self.stage = stage
        usage_path = self.root / "usage.json"
        usage = read_json(usage_path)
        active = usage.get("active_job")
        if active:
            elapsed = max(0, time.time() - active["started_at"])
            usage["gpu_job_seconds"] += elapsed
            if active["stage"] == "P1":
                usage["pilot_gpu_job_seconds"] += elapsed
            active.update(stage=stage, started_at=time.time())
            write_json(usage_path, usage)
        self.check(storage=True)
        self.validate_inputs()
