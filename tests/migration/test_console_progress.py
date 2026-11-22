"""Acceptance tests for the SDXL console clock, independent of wall time."""
import io
from unittest.mock import patch

import pytest
from tqdm import tqdm

from library.logging_util import start_step_progress


def test_first_batch_restarts_displayed_rate():
    with patch("tqdm.std.time", return_value=110.0) as clock:
        with tqdm(total=8, file=io.StringIO(), mininterval=0, smoothing=0) as bar:
            clock.return_value = 126.84
            with patch("library.logging_util.logger.info") as log:
                start_step_progress(bar, 16.84, main_process=True)
            fmt, *values = log.call_args.args
            assert "学習開始前の初期化時間: 16.84 秒" in fmt % tuple(values)
            clock.return_value += 0.5
            bar.update()
            assert bar.n == 1 and bar.total == 8
            assert bar.format_dict["elapsed"] == pytest.approx(0.5)
            assert "2.00it/s" in str(bar)


def test_disabled_worker_bar():
    with tqdm(total=8, disable=True) as bar:
        with patch("library.logging_util.logger.info") as log:
            start_step_progress(bar, 16.84, main_process=False)
        log.assert_not_called()
        assert bar.n == 0
