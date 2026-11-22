"""Isolate SDXL training random state from optional command logging and image generation."""
from contextlib import contextmanager
import random

from library import run_control


@contextmanager
def preserve_python_rng():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


class TrainingRunController(run_control.RunController):
    def process_commands(self, handlers, *, step, only=None):
        # Rich's command receipt message can consume Python RNG even before the sample handler starts.
        with preserve_python_rng():
            return super().process_commands(handlers, step=step, only=only)


def create_run_controller(args, is_main_process, config):
    if is_main_process and getattr(args, "run_control", False) and config is not None and config.changes_training:
        return TrainingRunController(run_control.get_run_dir(args))
    return run_control.create_run_controller(args, is_main_process)
