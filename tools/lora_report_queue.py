"""Track comparison results and control running generator processes."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import time

import psutil

from tools.lora_report_result import read_comparison


def atomic_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as output:
            json.dump(data,output,ensure_ascii=False,allow_nan=False,indent=2)
            output.write("\n")
        os.replace(temp,path)
    finally:
        Path(temp).unlink(missing_ok=True)


def process_identity(pid):
    proc=psutil.Process(pid)
    return {"pid":pid,"created":proc.create_time()}


def is_running(record):
    try:
        p=psutil.Process(record["pid"])
        return p.create_time()==record["created"] and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except (KeyError,TypeError,psutil.Error):
        return False


def find_result(directory, comparison_id, attempt_id):
    folder = Path(directory)
    if (folder/"metadata.json").is_file():
        meta,_,_=read_comparison(folder)
        if (meta["comparison_id"],meta["attempt_id"])!=(comparison_id,attempt_id):
            raise ValueError("published result does not match the running comparison")
        reports = (folder/"report.html", folder/"blind_report.html", folder/"blind_report_detailed.html")
        if not all(path.is_file() for path in reports):
            raise ValueError("published comparison is missing a rendered report")
        return meta["status"],reports[0]
    for file in (folder/"worker/generator-process.json", folder/"worker/worker-process.json"):
        if file.is_file() and is_running(json.loads(file.read_text(encoding="utf-8"))):
            return "running",None
    return None


def run_generator(command,cwd,control_dir,cancel_file):
    if cancel_file and Path(cancel_file).exists():
        return None,True
    with subprocess.Popen(command,cwd=cwd) as proc:
        try:
            identity = process_identity(proc.pid)
        except psutil.NoSuchProcess:
            return proc.wait(), False
        atomic_json(control_dir/"generator-process.json", identity)
        while proc.poll() is None:
            if cancel_file and Path(cancel_file).exists():
                try:
                    parent=psutil.Process(proc.pid)
                    children=parent.children(recursive=True)
                    for p in [*reversed(children),parent]:
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    _,alive=psutil.wait_procs(children+[parent],timeout=15)
                    if alive:
                        raise RuntimeError("comparison generator or descendant is still running")
                except psutil.NoSuchProcess:
                    pass
                return proc.wait(),True
            time.sleep(0.1)
        return proc.returncode,False
