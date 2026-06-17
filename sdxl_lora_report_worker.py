#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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


def lora_slot_key(item: dict) -> tuple[str, str, str | None]:
    return (item["module"], item["path"], item.get("lbw"))


def build_lora_slots(conditions: list[dict]) -> list[dict]:
    slots = []
    slot_by_key = {}
    for condition in conditions:
        for item in condition["items"]:
            key = lora_slot_key(item)
            if key in slot_by_key:
                continue
            if item.get("lbw") is None:
                raise ValueError("LoRA report worker requires lbw for every LoRA item")
            slot_by_key[key] = len(slots)
            slots.append(dict(item))
    return slots


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
        multipliers = [0.0] * len(slots)
        slot_by_key = {lora_slot_key(slot): index for index, slot in enumerate(slots)}
        for item in job["condition_items"]:
            multipliers[slot_by_key[lora_slot_key(item)]] = item["strength"]
        text += " --am " + ",".join(str(v) for v in multipliers)

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
    command.extend(str(arg) for arg in common_args)

    if slots:
        command.append("--network_module")
        command.extend(slot["module"] for slot in slots)
        command.append("--network_weights")
        command.extend(slot["path"] for slot in slots)
        command.append("--network_mul")
        command.extend("1.0" for _ in slots)
        command.append("--network_lbw")
        command.extend(str(slot["lbw"]) for slot in slots)

    command.extend(["--from_file", str(prompt_file), "--sequential_file_name"])
    return command


def move_outputs(outdir: Path, before: set[Path], jobs: list[dict]) -> list[dict]:
    after = list_images(outdir)
    created = sorted(after - before, key=lambda p: p.name)
    if len(created) < len(jobs):
        return [
            {
                "job_index": job["job_index"],
                "status": "missing_image",
                "output": job["target_path"],
                "error": f"expected {len(jobs)} images but found {len(created)}",
            }
            for job in jobs
        ]

    results = []
    for image, job in zip(created, jobs):
        target = Path(job["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(image), str(target))
        results.append({"job_index": job["job_index"], "status": "done", "output": str(target), "error": None})
    return results


def run_worker(job_plan: dict, dry_run: bool) -> dict:
    script_dir = Path(__file__).resolve().parent
    gen_config = job_plan["sdxl_gen_img"]
    if int(gen_config.get("images_per_prompt", 1)) != 1:
        raise ValueError("LoRA report worker currently requires images_per_prompt=1")

    outdir = Path(job_plan["work_outdir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = job_plan["jobs"]
    conditions = job_plan["conditions"]
    slots = build_lora_slots(conditions)

    with tempfile.TemporaryDirectory(prefix="lora_report_worker_") as tmpdir:
        prompt_file = Path(tmpdir) / "prompts.txt"
        with prompt_file.open("w", encoding="utf-8") as f:
            for job in jobs:
                f.write(prompt_line(job, slots) + "\n")

        command = build_command(script_dir, job_plan, prompt_file, outdir, slots)
        if dry_run:
            return {
                "status": "dry_run",
                "returncode": 0,
                "command": command,
                "slots": slots,
                "results": [
                    {"job_index": job["job_index"], "status": "dry_run", "output": job["target_path"], "error": None}
                    for job in jobs
                ],
            }

        before = list_images(outdir)
        result = subprocess.run(command, cwd=script_dir)
        if result.returncode != 0:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "command": command,
                "slots": slots,
                "results": [
                    {"job_index": job["job_index"], "status": "failed", "output": job["target_path"], "error": "worker command failed"}
                    for job in jobs
                ],
            }

    results = move_outputs(outdir, before, jobs)
    status = "done" if all(r["status"] == "done" for r in results) else "missing_image"
    return {"status": status, "returncode": 0, "command": command, "slots": slots, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Generate all LoRA report conditions in one sdxl_gen_img.py process.")
    parser.add_argument("--job-json", required=True, help="Worker job JSON path.")
    parser.add_argument("--result-json", required=True, help="Worker result JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Do not run image generation.")
    args = parser.parse_args()

    job_plan_path = Path(args.job_json).resolve()
    result_path = Path(args.result_json).resolve()
    result = run_worker(load_json(job_plan_path), args.dry_run)
    write_json(result_path, result)
    raise SystemExit(int(result.get("returncode", 1)))


if __name__ == "__main__":
    main()
