"""Compare manual and post-training diagnostic generation on an actual completed run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from library.training_run_report import report_after_training
from tools.lora_report_publication import read_published_report
from tools.lora_run_reader import RunReader


def snapshot(directory):
    reader = RunReader(directory)
    doc, error = read_published_report(reader)
    assert doc is not None and error is None, error
    assert reader.index["status"] == "completed"
    for kind in ("gradient", "dq", "dq_auto", "rank", "promote", "group_loss"):
        stream = reader.stream(kind)
        entry = doc["provenance"]["inputs"][kind]
        assert entry["rows"] == len(stream.rows) and entry["sha256"] == stream.sha256
    assert doc["provenance"]["options"]["lora_epoch_trend"]
    assert doc["lora"] and doc["lora_trend"]["point_count"] >= 1
    epoch_count = doc["lora_trend"]["point_count"]
    del doc["generated_at"]
    del doc["provenance"]["generation_id"]
    return doc, epoch_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    run = RunReader(args.run_dir).run
    auto = report_after_training(run)
    assert auto and auto.is_file()
    first, epochs = snapshot(run.directory)
    subprocess.run([sys.executable, str(REPO / "tools/make_lora_diagnostic_report.py"),
                    "--run_dir", str(run.directory), "--lora_epoch_trend"], check=True)
    second, _ = snapshot(run.directory)
    assert first == second, [k for k in first if first[k] != second[k]]
    print(json.dumps(dict(run_id=run.run_id, epochs=epochs, same_analysis=True,
                          charts={k: len(v) for k, v in first["charts"].items()})))


if __name__ == "__main__":
    main()
