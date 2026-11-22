"""Launch the optional post-training diagnostic in its own CPU process."""
import logging
from pathlib import Path
import subprocess
import sys
import time


logger = logging.getLogger(__name__)


def report_after_training(run):
    from tools.lora_report_publication import HTML_PATH, record_failure

    script = Path(__file__).resolve().parents[1] / "tools" / "make_lora_diagnostic_report.py"
    command = [sys.executable, str(script), "--run_dir", str(run.directory), "--lora_epoch_trend"]
    started = time.perf_counter()
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        # The child may have failed before importing its run reader or writing its reason.
        logger.error("Diagnostic failed (%s). Retry with %s", exc, subprocess.list2cmdline(command))
        try:
            import json
            index = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
            if not index["artifacts"].get("diagnostic_report", {}).get("last_error"):
                record_failure(run, exc)
        except (OSError, ValueError, KeyError):
            logger.exception("Could not record diagnostic failure in %s", run.directory)
        return None
    path = run.directory / HTML_PATH
    logger.info("Diagnostic report (%.2f s): %s", time.perf_counter() - started, path)
    return path
