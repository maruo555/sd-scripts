"""Qt's real QProcess must carry the same frozen comparison through retries and cancellation."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import sdxl_lora_report_gui as gui
from tools.lora_report_plan import make_attempt
from tools.lora_report_queue import find_result, is_running


def wait_for(app, predicate):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.025)
    raise AssertionError("Timed out waiting for the comparison process")


def prepare_scripts(root):
    """Run real GUI/CUI/worker in isolation; only the GPU generator is a sleeping test process."""
    repo = Path(gui.__file__).parent
    names = ["sdxl_lora_report_cui.py", "sdxl_lora_report_worker.py",
             "tools/lora_report_plan.py", "tools/lora_report_result.py",
             "tools/lora_report_views.py", "tools/lora_report_queue.py",
             "library/generation_lora_strengths.py", "library/generation_dynamic_prompt.py",
             "library/generation_lbw.py", "lora_report_templates/blind_report_detailed.html"]
    for name in names:
        target = root / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(repo / name, target)
    for package in ("library", "tools"):
        (root / package / "__init__.py").touch()
    (root / "sdxl_gen_img.py").write_text(
        "import json, pathlib, subprocess, sys, time, psutil\n"
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pathlib.Path('descendant.json').write_text(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))\n"
        "p.wait()\n", encoding="utf-8")


@pytest.fixture
def comparison_gui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    sandbox = tmp_path / "comparison scripts"
    prepare_scripts(sandbox)
    for key, value in {"SCRIPT_DIR": sandbox, "GUI_CONFIG_PATH": sandbox / ".tmp/last.json",
                       "QUEUE_DIR": sandbox / ".tmp/queue", "QUEUE_STATE_PATH": sandbox / ".tmp/queue/items.json"}.items():
        monkeypatch.setattr(gui, key, value)
    return app, sandbox


def test_gui_preferences_and_single_run(tmp_path, comparison_gui):
    app, _ = comparison_gui
    gen = {"width": 512, "height": 768, "steps": 11, "scale": 5.25, "batch_size": 2, "sampler": "euler"}
    model, output = str(tmp_path / "model.ckpt"), str(tmp_path / "out")
    expected = {"model": model, "output_root": output, "generation": gen,
                "common_args": {"precision": "fp16", "attention": "xformers", "extra_args": "--clip_skip 2"}}
    gui.GUI_CONFIG_PATH.parent.mkdir(parents=True)
    # Startup reads preferences without rewriting the file.
    saved = json.dumps(expected, indent=4).encode()
    gui.GUI_CONFIG_PATH.write_bytes(saved)
    w = gui.MainWindow()
    assert gui.GUI_CONFIG_PATH.read_bytes() == saved
    assert not w.prompt_edit.text() and not w.conditions and not w.baseline_check.isChecked()
    assert w.run_name_edit.text() == "lora_report" and w.seed_values_edit.text() == "12345"

    prompt = tmp_path / "new.txt"
    prompt.write_text("a paper lantern\n", encoding="utf-8")
    w.prompt_edit.setText(str(prompt))
    w.baseline_check.setChecked(True)
    w.dry_run_check.setChecked(True)
    w.run_report()
    wait_for(app, lambda: w.process is None)
    assert w.current_report and w.current_report.is_file()
    assert not list(gui.QUEUE_DIR.glob("*"))
    job = json.loads(w.current_report.with_name("config.json").read_bytes())
    assert job["prompt_file"] == str(prompt) and job["_comparison_plan"]["prompts"]
    assert json.loads(gui.GUI_CONFIG_PATH.read_bytes()) == expected

    w.prompt_edit.clear()
    w.output_edit.setText(str(tmp_path / "next"))
    w.extra_args_edit.setText("--clip_skip   3")
    w.close()
    fresh = gui.MainWindow()
    assert fresh.model_edit.text() == model and fresh.output_edit.text() == str(tmp_path / "next")
    assert fresh.generation_settings() == gen
    assert fresh.precision_combo.currentText() == "fp16" and fresh.attention_combo.currentText() == "xformers"
    assert fresh.extra_args_edit.text() == "--clip_skip   3"
    assert not fresh.prompt_edit.text() and not fresh.baseline_check.isChecked() and not fresh.dry_run_check.isChecked()
    fresh.close()


def test_stop_before_worker_starts(tmp_path, comparison_gui):
    app, _ = comparison_gui
    w = gui.MainWindow()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("paper lantern", encoding="utf-8")
    w.model_edit.setText(str(tmp_path / "model.safetensors"))
    w.prompt_edit.setText(str(prompt))
    w.output_edit.setText(str(tmp_path / "out"))
    w.baseline_check.setChecked(True)
    w.run_report()
    w.stop_report()
    wait_for(app, lambda: w.process is None)
    assert w.current_report and w.current_report.is_file()
    assert json.loads(w.current_report.with_name("metadata.json").read_bytes())["status"] == "cancelled"
    assert not list(gui.QUEUE_DIR.glob("*"))
    w.close()


def test_gui_process_freeze_cancel_reconcile_and_retry(tmp_path, comparison_gui):
    import subprocess
    app, sandbox = comparison_gui
    prompt = tmp_path / "two rows.txt"
    prompt.write_text("same prompt\nsame prompt\n", encoding="utf-8")
    w = gui.MainWindow()
    w.model_edit.setText(str(tmp_path / "model.safetensors"))
    w.prompt_edit.setText(str(prompt))
    w.output_edit.setText(str(tmp_path / "shared output"))
    w.baseline_check.setChecked(True)
    w.random_count_spin.setValue(1)
    w.add_current_settings_to_queue()
    w.dry_run_check.setChecked(True)
    w.add_current_settings_to_queue()
    first, second = w.queue
    frozen = first["config"]["_comparison_plan"]
    assert len(frozen["prompts"]) == 2
    _, _, original_jobs = make_attempt(first["config"], tmp_path, str(gui.uuid.uuid4()))
    prompt.unlink()
    w.run_queue()
    w.stop_after_current()
    descendant = sandbox / "descendant.json"
    wait_for(app, descendant.is_file)
    child = json.loads(descendant.read_text())
    assert is_running(child)
    assert set(gui.QUEUE_DIR.iterdir()) == {gui.QUEUE_STATE_PATH}
    # Another GUI recovering the durable state must recognize the same live execution.
    recovered = gui.MainWindow()
    assert recovered.queue[0]["status"] == "Running"
    recovered.run_queue()
    assert recovered.process is None
    recovered.reconcile_timer.stop()
    w.cancel_running()
    assert w.cancel_path(first["attempt"]).is_file()
    wait_for(app, lambda: w.process is None)
    assert [i["status"] for i in w.queue] == ["Canceled", "Waiting"]
    assert set(gui.QUEUE_DIR.iterdir()) == {gui.QUEUE_STATE_PATH}
    assert not is_running(child)
    original_report = Path(first["report_path"])
    published = json.loads(original_report.with_name("metadata.json").read_bytes())
    assert published["status"] == "cancelled" and {j["status"] for j in published["jobs"]} == {"cancelled"}
    # Simulate a GUI crash immediately before persisting the final queue status.
    first["status"] = "Running"
    w.reconcile_queue()
    assert first["status"] == "Canceled"
    with pytest.raises(ValueError, match="does not match"):
        find_result(first["attempt"]["directory"], second["config"]["_comparison_plan"]["comparison_id"], first["attempt"]["id"])
    w.queue_tree.setCurrentItem(w.queue_tree.topLevelItem(0))
    w.retry_selected()
    first["options"]["dry_run"] = True
    w.run_queue()
    wait_for(app, lambda: w.process is None and second["status"] == "Dry run")
    assert first["status"] == "Dry run"
    retry = json.loads(Path(first["report_path"]).with_name("metadata.json").read_bytes())
    assert retry["attempt_id"] != published["attempt_id"] and retry["comparison_id"] == published["comparison_id"]
    assert [(j["job_id"],j["display_order"],j["execution_order"]) for j in original_jobs] == [(j["job_id"],j["display_order"],j["execution_order"]) for j in retry["jobs"]]
    assert retry["seeds"] == frozen["seeds"] and Path(first["report_path"]) != original_report
    assert original_report.is_file()
    assert set(gui.QUEUE_DIR.iterdir()) == {gui.QUEUE_STATE_PATH}
    assert not list(tmp_path.rglob("receipt.json"))
    # The published config alone must reproduce the jobs and generator options via the file-based CLI.
    direct_config_path = original_report.with_name("config.json")
    cp = subprocess.run([sys.executable, str(sandbox/"sdxl_lora_report_cui.py"), "--config", str(direct_config_path), "--dry-run"],
                        cwd=tmp_path, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    direct_dir = next(p for p in (tmp_path/"shared output").iterdir() if p not in {
        original_report.parent, Path(first["report_path"]).parent, Path(second["report_path"]).parent})
    direct = json.loads((direct_dir/"metadata.json").read_bytes())
    assert direct["jobs"] == retry["jobs"]
    def options(meta):
        args = meta["generator"]["command"][:]
        for flag in ("--outdir","--from_file"):
            args[args.index(flag)+1] = "<attempt>"
        return args
    assert options(direct) == options(retry)
    w.close()
    recovered.close()
