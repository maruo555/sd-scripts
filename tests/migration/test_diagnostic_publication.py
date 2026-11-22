import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from library.training_run import TrainingRun
from library.training_run_report import report_after_training
from tools import lora_report_publication as publication
from tools.lora_run_reader import RunReader


def make_report(run, generation):
    ids = dict(run_id=run.run_id, session_id=run.session_id, generation_id=generation)
    doc = dict(report_schema_version=1, provenance=ids)
    html = "<html><head>" + "".join(f'<meta name="{k}" content="{v}">' for k, v in ids.items()) + "</head></html>"
    return doc, html


def install(run, generation, verify=lambda: None):
    with publication.report_lock(run.directory):
        return publication.publish(run, *make_report(run, generation), verify)


@pytest.mark.parametrize("failure", ["source", "second_replace", "register"])
def test_partial_publication_is_never_accepted_as_valid(tmp_path, monkeypatch, failure):
    run = TrainingRun.create(tmp_path, "色見本", 12)
    run.finish("completed", "complete")
    install(run, "previous")
    assert publication.read_published_report(RunReader(run.directory))[0]["provenance"]["generation_id"] == "previous"
    def fail(*args, **kwargs):
        raise OSError("injected")
    verify = fail if failure == "source" else lambda: None
    real_replace = publication.os.replace
    if failure == "second_replace":
        def replace(src, dest):
            if Path(dest).name == "report.html":
                fail()
            real_replace(src, dest)
        monkeypatch.setattr(publication.os, "replace", replace)
    elif failure == "register":
        monkeypatch.setattr(TrainingRun, "set_artifact", fail)
    with pytest.raises(OSError):
        install(run, "next", verify)
    publication.record_failure(run, "injected")
    reader = RunReader(run.directory)
    report, error = publication.read_published_report(reader)
    if failure == "source":
        assert report["provenance"]["generation_id"] == "previous"
    else:
        assert report is None and "identity" in error
    assert reader.index["status"] == "completed"
    assert reader.index["artifacts"]["diagnostic_report"]["last_error"] == "injected"
    monkeypatch.undo()
    install(run, "recovered")
    latest = RunReader(run.directory)
    assert publication.read_published_report(latest)[0]["provenance"]["generation_id"] == "recovered"
    assert latest.index["artifacts"]["diagnostic_report"]["last_error"] is None


def test_auto_report_uses_exact_directory_and_does_not_change_training_status(tmp_path):
    run = TrainingRun.create(tmp_path, "same", 2)
    run.finish("completed", "complete")
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(2, "report")) as child:
        assert report_after_training(run) is None
    argv = child.call_args.args[0]
    assert argv[-3:] == ["--run_dir", str(run.directory), "--lora_epoch_trend"]
    assert child.call_args.kwargs["check"]
    index = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    assert (index["status"], index["flush_status"]) == ("completed", "complete")
    assert index["artifacts"]["diagnostic_report"]["last_error"]
    with patch("subprocess.run"):
        assert report_after_training(run) == run.directory / publication.HTML_PATH


def test_no_observations_report_reason_and_no_generated_files(tmp_path):
    from tools.make_lora_diagnostic_report import generate_report, Options
    run = TrainingRun.create(tmp_path, "empty", 3)
    run.finish("stopped", "complete")
    with pytest.raises(ValueError, match="No observations"):
        generate_report(run, Options())
    reader = RunReader(run.directory)
    assert reader.index["status"] == "stopped"
    assert reader.index["artifacts"]["diagnostic_report"]["status"] == "failed"
    assert publication.read_published_report(reader)[0] is None
    assert not (run.directory / publication.HTML_PATH).exists()
