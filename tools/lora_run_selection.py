"""Resolve a diagnostic request to one indexed training run before writing any output."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from tools.lora_run_reader import RunReader, discover_run_indexes


class RunSelectionError(ValueError):
    pass


def _matches(reader, name):
    return name in (reader.index.get("requested_output_name"), reader.run.base_name)


def select_run(*, base_name=None, input_dir=None, run_dir=None, options=(), terminal=None, output=None):
    terminal = sys.stdin if terminal is None else terminal
    output = sys.stderr if output is None else output
    explicit = None if run_dir is None else RunReader(run_dir)
    if explicit and input_dir is not None and explicit.run.output_root != Path(input_dir).resolve():
        raise RunSelectionError("--input_dir does not own the selected --run_dir")
    if explicit and base_name is not None and not _matches(explicit, base_name):
        raise RunSelectionError("--base_name conflicts with the selected run")
    if explicit is None and not base_name:
        raise RunSelectionError("Specify --run_dir or --base_name")
    root = explicit.run.output_root if explicit else Path(input_dir or ".")
    readers, issues = discover_run_indexes(root)
    for issue in issues:
        print(f"{issue['directory']}: {issue['reason']}", file=output)
    counts = Counter(reader.run.run_id for reader in readers)
    if explicit:
        matches = [r for r in readers if r.run.directory == explicit.run.directory]
    else:
        matches = [r for r in readers if _matches(r, base_name)]
    if not matches:
        raise RunSelectionError(f"No indexed run for {base_name or run_dir!r} under {root}")
    for reader in matches:
        if counts[reader.run.run_id] != 1:
            raise RunSelectionError(f"Duplicate run UUID {reader.run.run_id} under {root}")
    if len(matches) == 1:
        return matches[0]
    matches.sort(key=lambda r: (datetime.fromisoformat(r.index["started_at"]), r.run.directory.name), reverse=True)
    for number, reader in enumerate(matches, 1):
        print(f"{number}. {reader.index['started_at']}  {reader.index['status']}  {reader.run.run_id}", file=output)
        command = [sys.executable, str(Path(__file__).with_name("make_lora_diagnostic_report.py")),
                   "--run_dir", str(reader.run.directory), *options]
        print("   " + subprocess.list2cmdline(command), file=output)
    if not terminal.isatty():
        raise RunSelectionError("Multiple runs match. Use one of the --run_dir commands above.")
    print("番号を入力してください（空欄で中止） ", file=output, end="", flush=True)
    answer = terminal.readline().strip()
    if not answer.isdecimal() or not 1 <= int(answer) <= len(matches):
        raise RunSelectionError("Run selection cancelled")
    return matches[int(answer) - 1]
