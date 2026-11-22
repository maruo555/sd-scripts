from argparse import Namespace
import random
from types import SimpleNamespace

import pytest

from library import sdxl_train_util
from library.training_run import TrainingRun
from sdxl_train_network import SdxlNetworkTrainer


@pytest.mark.parametrize("custom,schedule,raises,restores", [(True,1,False,True), (True,1,True,True), (False,1,False,False)])
def test_custom_sample_rng_boundary(tmp_path, monkeypatch, custom, schedule, raises, restores):
    args=Namespace(output_dir=str(tmp_path), output_name=None, sample_at_first=False, sample_every_n_steps=schedule,
                   sample_every_n_epochs=None)
    trainer=SdxlNetworkTrainer()
    trainer.training_run=TrainingRun.create(tmp_path,None,12)
    trainer.maruo_config=SimpleNamespace(changes_training=custom)
    def generate(*args):
        random.random()
        if raises:
            raise ValueError("sample failed")
    monkeypatch.setattr(sdxl_train_util,"sample_images",generate)
    original=random.getstate()
    try:
        if raises:
            with pytest.raises(ValueError):
                trainer.sample_images(None,args,None,1,None,None,None,None,None)
        else:
            trainer.sample_images(None,args,None,1,None,None,None,None,None)
        assert (random.getstate()==original) == restores
    finally:
        random.setstate(original)
