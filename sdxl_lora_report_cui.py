#!/usr/bin/env python
"""Build an SDXL comparison and publish its validated, per-image outcome."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from tools.lora_report_plan import make_attempt, order_jobs_for_generation
from tools.lora_report_result import comparison_jobs
from tools.lora_report_queue import atomic_json, process_identity


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2),encoding="utf-8")


def worker_result(directory,config,jobs,attempt_id,dry_run,cancel_file=None):
    worker_dir=directory/"worker"
    worker_dir.mkdir()
    ordered=order_jobs_for_generation(jobs,config["_comparison_plan"]["conditions"])
    plan={"cancel_file": str(cancel_file) if cancel_file else None, "attempt_id":attempt_id,"sdxl_gen_img":config["sdxl_gen_img"],"conditions":config["_comparison_plan"]["conditions"],
          "work_outdir":str(worker_dir/"images"),"jobs":[dict(j,job_index=j["display_order"]-1) for j in ordered]}
    plan_path,result_path=worker_dir/"worker_job.json",worker_dir/"worker_result.json"
    write_json(plan_path,plan)
    argv=[sys.executable,str(Path(__file__).with_name("sdxl_lora_report_worker.py")),"--job-json",str(plan_path),"--result-json",str(result_path)]
    if dry_run:
        argv.append("--dry-run")
    runner=subprocess.run(argv,cwd=Path(__file__).resolve().parent)
    try:
        response=json.loads(result_path.read_text(encoding="utf-8"))
        if response.get("attempt_id")!=attempt_id:
            raise ValueError("worker result belongs to another attempt")
        ordered_result=response["results"]
        by_id={x["job_id"]:x for x in ordered_result}
        if len(by_id)!=len(ordered_result) or set(by_id)!={j["job_id"] for j in jobs}:
            raise ValueError("duplicate, foreign or missing job IDs in worker result")
        if response["status"] in {"done","dry_run"} and runner.returncode!=0:
            raise ValueError("worker process failed after writing its result")
        for j in jobs:
            item=by_id[j["job_id"]]
            j.update(status=item["status"],returncode=item.get("returncode"),error=item.get("error"))
        return {k:response.get(k) for k in ("status","returncode","command","slots","error")}
    except (ValueError,OSError,KeyError,TypeError) as exc:
        for job in jobs:
            job.update(status="failed",returncode=None,error=str(exc))
        return {"status":"failed","returncode":None,"error":str(exc)}


def run(args):
    config_path=Path(args.config).resolve()
    original=config_path.read_bytes()
    attempt_id=str(uuid.UUID(args.attempt_id)) if args.attempt_id else str(uuid.uuid4())
    directory,config,jobs=make_attempt(json.loads(original.decode("utf-8-sig")),config_path.parent,attempt_id)
    frozen=config["_comparison_plan"]
    # Validate restored queue data and its relative targets before giving the worker any paths.
    preview=dict(kind="lora_image_comparison",schema_version=1,comparison_id=frozen["comparison_id"],attempt_id=attempt_id,
                 status="dry_run",returncode=None,prompts=frozen["prompts"],seeds=frozen["seeds"],conditions=frozen["conditions"],
                 jobs=[dict(j,status="dry_run",returncode=None) for j in jobs])
    comparison_jobs(preview,directory)
    directory.mkdir(parents=True,exist_ok=False)
    if getattr(args, "receipt", None):
        atomic_json(args.receipt, {"comparison_id": frozen["comparison_id"], "attempt_id": attempt_id,
                                   "directory": str(directory), "process": process_identity(os.getpid())})
    (directory/"config.json").write_bytes(original)
    (directory/"prompts.txt").write_bytes(base64.b64decode(frozen["prompt_file_base64"],validate=True))
    write_json(directory/"prompts.parsed.json",{"source":config["prompt_file"], "prompts":frozen["prompts"]})
    print(f"Comparison {frozen['comparison_id']} attempt {attempt_id}: {directory}",flush=True)
    worker=worker_result(directory,config,jobs,attempt_id,args.dry_run,getattr(args,"cancel_file",None))
    published=[{k:v for k,v in j.items() if k!="target_path"} for j in jobs]
    metadata=dict(kind="lora_image_comparison",schema_version=1,comparison_id=frozen["comparison_id"],attempt_id=attempt_id,
                  status=worker["status"],returncode=worker["returncode"],config_sha256=hashlib.sha256(original).hexdigest(),
                  prompts=frozen["prompts"],conditions=frozen["conditions"],seeds=frozen["seeds"],jobs=published,
                  generator={"entry":"sdxl_gen_img.py","command":worker.get("command"),"slots":worker.get("slots")})
    meta_bytes=(json.dumps(metadata,ensure_ascii=False,allow_nan=False,indent=2)+"\n").encode()
    eligible=comparison_jobs(metadata,directory)
    if len(eligible)!=sum(j["status"]=="done" for j in jobs):
        raise ValueError("an accepted PNG changed before publication")
    # The published JSON is the completion marker for the GUI, renderer and research ledger.
    from tools.lora_report_views import render_comparison

    render_comparison(directory,meta_bytes)
    temp=directory/"metadata.json.tmp"
    temp.write_bytes(meta_bytes)
    os.replace(temp,directory/"metadata.json")
    print(f"Metadata: {directory/'metadata.json'}",flush=True)
    print(f"Report: {directory/'report.html'}",flush=True)
    return 0 if worker["status"] in {"done","dry_run"} else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--dry-run",action="store_true",help="Build the exact generation plan without launching the generator")
    parser.add_argument("--attempt-id",help=argparse.SUPPRESS)
    parser.add_argument("--receipt",help=argparse.SUPPRESS)
    parser.add_argument("--cancel-file",help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:
        return run(args)
    except (ValueError,OSError) as exc:
        print(f"Comparison failed: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
