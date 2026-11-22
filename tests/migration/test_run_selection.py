import io
import shutil

import pytest

from library.training_run import TrainingRun
from tools.lora_run_selection import RunSelectionError, select_run


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_matching_names_and_multiple_choices(tmp_path):
    a = TrainingRun.create(tmp_path, "同名 run", "a")
    assert select_run(input_dir=tmp_path, base_name="同名 run").run == a
    assert select_run(run_dir=a.directory).run == a
    b = TrainingRun.create(tmp_path, "同名 run", "b")
    output = io.StringIO()
    with pytest.raises(RunSelectionError, match="Multiple"):
        select_run(input_dir=tmp_path, base_name="同名 run", options=["--lora_epoch_trend"],
                   terminal=io.StringIO(), output=output)
    assert output.getvalue().count("--run_dir") == output.getvalue().count("--lora_epoch_trend") == 2
    assert select_run(input_dir=tmp_path, base_name="同名 run", terminal=Terminal("1\n")).run == b
    with pytest.raises(RunSelectionError, match="cancelled"):
        select_run(input_dir=tmp_path, base_name="同名 run", terminal=Terminal("\n"))
    with pytest.raises(RunSelectionError, match="conflicts"):
        select_run(run_dir=a.directory, base_name="other")
    with pytest.raises(RunSelectionError, match="own"):
        select_run(run_dir=a.directory, input_dir=a.directory)
    with pytest.raises(RunSelectionError, match="No indexed"):
        select_run(input_dir=tmp_path, base_name="同名")
    shutil.copytree(a.directory, tmp_path / "copy")
    with pytest.raises(RunSelectionError, match="Duplicate"):
        select_run(run_dir=a.directory)


def test_omitted_output_name_and_no_recursive_scan(tmp_path):
    a = TrainingRun.create(tmp_path, None, "a")
    TrainingRun.create(tmp_path / "unrelated", None, "nested")
    assert select_run(input_dir=tmp_path, base_name="last").run == a
