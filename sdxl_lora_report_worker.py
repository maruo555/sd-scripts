#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

from library.generation_lora_strengths import format_strength_value
from tools.lora_report_plan import build_lora_slots, slot_values
from tools.lora_report_queue import atomic_json, process_identity, run_generator


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_LBW = "ALL"
BLOCKED_COMMON_ARGS = {
    "--network_merge","--network_merge_n_models","--outdir","--from_file","--network_module","--network_weights",
    "--network_mul","--network_lbw","--network_args","--use_original_file_name","--sequential_file_name",
    "--images_per_prompt","--interactive","--n_iter","--prompt","--seed","--W","--H","--ckpt",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_arg(command: list[str], option: str, value):
    if value is not None:
        command.extend([option, str(value)])


def list_images(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def prompt_line(job: dict, slots: list[dict]) -> str:
    text = job["prompt"]
    if job.get("negative"):
        text += f" --n {job['negative']}"
    text += f" --d {job['seed']}"
    if job.get("width"):
        text += f" --w {job['width']}"
    if job.get("height"):
        text += f" --h {job['height']}"

    if slots:
        multipliers = slot_values(job["condition_items"],slots)
        text += " --am " + ",".join(format_strength_value(v) for v in multipliers)

    return text


def build_command(script_dir: Path, job_plan: dict, prompt_file: Path, outdir: Path, slots: list[dict]) -> list[str]:
    gen_config = job_plan["sdxl_gen_img"]
    command = [sys.executable, str(script_dir / "sdxl_gen_img.py")]

    append_arg(command, "--ckpt", gen_config.get("ckpt"))
    append_arg(command, "--vae", gen_config.get("vae"))
    command.extend(["--outdir", str(outdir)])
    append_arg(command, "--W", gen_config.get("width"))
    append_arg(command, "--H", gen_config.get("height"))
    append_arg(command, "--steps", gen_config.get("steps"))
    append_arg(command, "--sampler", gen_config.get("sampler"))
    append_arg(command, "--scale", gen_config.get("scale"))
    append_arg(command, "--batch_size", gen_config.get("batch_size", 1))
    append_arg(command, "--images_per_prompt", gen_config.get("images_per_prompt", 1))

    common_args = gen_config.get("common_args", [])
    if not isinstance(common_args, list):
        raise ValueError("sdxl_gen_img.common_args must be a list")
    for arg in common_args:
        option = str(arg).split("=", 1)[0]
        if option.startswith('--') and any(managed.startswith(option) for managed in BLOCKED_COMMON_ARGS):
            raise ValueError(
                f"{option} is managed by the comparison worker and cannot be overridden"
            )
    command.extend(str(arg) for arg in common_args)

    if slots:
        command.append("--network_module")
        command.extend(slot["module"] for slot in slots)
        command.append("--network_weights")
        command.extend(slot["path"] for slot in slots)
        command.append("--network_mul")
        command.extend("1.0" for _ in slots)
        if any(slot.get("lbw") is not None for slot in slots):
            command.append("--network_lbw")
            command.extend(str(slot.get("lbw") or DEFAULT_LBW) for slot in slots)

    command.extend(["--from_file", str(prompt_file), "--sequential_file_name"])
    return command


def move_outputs(outdir, jobs, slots, *, highres, generator_code):
    from PIL import Image
    expected = {f"im_{'1' if highres else ''}{j['execution_order']:06d}.png":j for j in jobs}
    allowed_intermediates = {f"im_0{j['execution_order']:06d}.png" for j in jobs} if highres else set()
    extras = {p.name for p in list_images(outdir)} - expected.keys() - allowed_intermediates
    results = []
    for filename,job in expected.items():
        result = {"job_id":job["job_id"],"job_index":job["job_index"], "status":"failed","returncode":generator_code or 1}
        source = outdir/filename
        try:
            if extras:
                raise ValueError(f"unexpected image files: {sorted(extras)}")
            with Image.open(source) as png:
                if png.format!="PNG" or png.size!=(job["width"],job["height"]):
                    raise ValueError(f"{filename}: expected PNG {job['width']}x{job['height']}")
                for k,v in (("prompt",job["prompt"]),("negative-prompt",job["negative"]),("seed",str(job["seed"]))):
                    if k in png.info and png.info[k]!=v:
                        raise ValueError(f"{filename} mismatched {k}")
                for tag,expected_value in job["png_parameters"].items():
                    if tag in png.info and json.loads(png.info[tag])!=expected_value:
                        raise ValueError(f"{filename}: mismatched {tag}")
                png.verify()
            target = Path(job["target_path"])
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
            result.update(status="done",returncode=0)
        except (ValueError,OSError,SyntaxError) as error:
            result["error"]=str(error)
        results.append(result)
    return results


def run_worker(plan,dry_run):
    jobs=plan["jobs"]
    slots=build_lora_slots(plan["conditions"])
    config=plan["sdxl_gen_img"]
    if int(config.get("images_per_prompt",1)) != 1:
        raise ValueError("comparison worker requires images_per_prompt=1")
    outdir=Path(plan["work_outdir"])
    outdir.mkdir(parents=True,exist_ok=False)
    prompt_file=outdir.parent/"generation-prompts.txt"
    prompt_file.write_text("".join(prompt_line(j,slots)+"\n" for j in jobs),encoding="utf-8")
    command=build_command(Path(__file__).resolve().parent,plan,prompt_file,outdir,slots)
    atomic_json(outdir.parent/"worker-process.json", process_identity(os.getpid()))
    cancel_file = plan.get("cancel_file")
    cancelled_before_launch = bool(cancel_file and Path(cancel_file).is_file())
    if dry_run and not cancelled_before_launch:
        return {"status":"dry_run","returncode":None,"command":command,"slots":slots,
                "results":[{"job_index":j["job_index"],"job_id":j["job_id"],"status":"dry_run","returncode":None} for j in jobs]}
    returncode, cancelled = run_generator(command, Path(__file__).resolve().parent, outdir.parent, cancel_file)
    highres=any(arg.split("=",1)[0]=="--highres_fix_scale" for arg in command)
    results=move_outputs(outdir,jobs,slots,highres=highres,generator_code=returncode)
    successful=sum(x["status"]=="done" for x in results)
    if cancelled:
        for item in results:
            if item["status"] != "done":
                item.update(status="cancelled", returncode=returncode, error="Cancelled by the user")
        # An interrupted generator may have finished individual, independently verified outputs.
        # If all files appeared but the generator was killed, there is no verified successful exit.
        if successful == len(jobs):
            for item in results:
                item.update(status="cancelled", returncode=returncode, error="Generator cancelled before a successful exit")
            successful = 0
        return {"status": "partial" if successful else "cancelled", "returncode": returncode,
                "command": command, "slots": slots, "results": results}
    if returncode and successful==len(jobs):
        for item in results:
            item.update(status="failed",returncode=returncode,error="generator failed after writing all images")
        successful=0
    status="done" if successful==len(jobs) and not returncode else "partial" if successful else "failed"
    return {"status":status,"returncode":returncode,"command":command,"slots":slots,"results":results}


def main():
    parser = argparse.ArgumentParser(description="Generate all LoRA report conditions in one sdxl_gen_img.py process.")
    parser.add_argument("--job-json", required=True, help="Worker job JSON path.")
    parser.add_argument("--result-json", required=True, help="Worker result JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Do not run image generation.")
    args = parser.parse_args()

    job_plan_path = Path(args.job_json).resolve()
    result_path = Path(args.result_json).resolve()
    job_plan = load_json(job_plan_path)
    try:
        result = run_worker(job_plan, args.dry_run)
    except Exception as exc:
        result = {
            "status": "failed",
            "returncode": 1,
            "command": [],
            "slots": [],
            "error": str(exc),
            "results": [
                {
                    "job_index": job["job_index"], "job_id":job["job_id"],
                    "status": "failed",
                    "output": job["target_path"],
                    "error": str(exc),
                }
                for job in job_plan.get("jobs", [])
            ],
        }
    result["attempt_id"] = job_plan["attempt_id"]
    write_json(result_path, result)
    raise SystemExit(0 if result["status"] in {"done","dry_run"} else result.get("returncode") or 1)


if __name__ == "__main__":
    main()
