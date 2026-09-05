import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dq_profile.dataset_diagnostics import summarize_samples
from dq_profile.dataset_uncertainty import source_intervals
from dq_profile.diagnostic_eval import build_inventory, scalar_tree
from dq_profile.production_cli import resolve_training_cli
from dq_profile.production_runner import profile_command
from test_dq_dataset_diagnostics import fixture_rows


def test_real_loader_shape_and_caption_precedence_and_multiple_contexts():
    info=NS(absolute_path='D:/fixture/a.png',bucket_reso=np.array([720,640]),caption='from_txt',num_repeats=20,is_reg=False)
    subset=NS(image_dir='D:/fixture',subset_index=0,class_tokens='fallback_tag',group='explicit')
    one=NS(width=720,height=720,image_data={'D:/fixture/a.png':info},image_to_subset={'D:/fixture/a.png':subset})
    second=NS(width=1024,height=1024,image_data=one.image_data,image_to_subset={'D:/fixture/a.png':NS(**{**vars(subset),'subset_index':1})})
    rows=build_inventory(NS(datasets=[one,second]))
    assert rows[0]['image_id']==rows[1]['image_id']
    assert rows[0]['sample_id']!=rows[1]['sample_id']
    assert rows[0]['resolution']==[720,720] and rows[1]['resolution']==[1024,1024]
    assert rows[0]['tags']==['from_txt'] and 'fallback_tag' not in rows[0]['tags']
    json.dumps(scalar_tree(rows),allow_nan=False)


def test_bootstrap_uses_same_draw_for_pre_post_and_is_deterministic():
    inv,refs,qs,ms=fixture_rows(4)
    for row in refs:
        row['raw_mse']=(int(row['sample_id'][1:])+1)*(1 if row['snapshot']=='pre' else .5)
    ss=summarize_samples(inv,refs,qs,ms)
    a=source_intervals(ss,ms,'test',iterations=200)
    b=source_intervals(ss,ms,'test',iterations=200)
    assert a==b
    assert a['baseline']['improvement_rel']['low']==.5
    assert a['baseline']['improvement_rel']['high']==.5
    assert a['quant'][0]['d_q95']['status']=='available'
    assert source_intervals(ss[:3],ms,'test')['status']=='insufficient_source_clusters'


def test_worker_options_only_on_local_not_prefix_or_standalone_snapshot(tmp_path):
    request=resolve_training_cli(['--pretrained_model_name_or_path=D:/model.safetensors','--dataset_config=D:/dataset.toml'])
    request=replace(request,data_diagnostics='warmup',group_map=tmp_path/'groups.json')
    args=dict(request=request,run_dir=tmp_path,source_map=tmp_path/'source.json',name='stage',range_muls=request.execution_mode.core_grid,max_images=request.local_measurement.max_images)
    local=profile_command(**args,protocol='v24-acceptance-local')
    snapshot=profile_command(**args,protocol='v24-acceptance-local',snapshot_only=True)
    prefix=profile_command(**args,protocol='v2-prefix-smoke')
    assert '--dq_profile_max_images=52' in local
    assert '--dq_profile_data_diagnostics=warmup' in local
    assert f'--dq_profile_group_map={request.group_map}' in local
    assert not any('--dq_profile_data_diagnostics=' in a for a in snapshot+prefix)


def test_promote_creates_links_and_keeps_selection_unchanged(tmp_path):
    from dq_profile.diagnostic_report import promote_dataset_report
    from test_dq_dataset_diagnostics import write_fixture
    profile=tmp_path/'worker';run=tmp_path/'run';run.mkdir()
    write_fixture(profile/'data_diagnostics',4)
    selection=dict(selection_valid=False,credible_muls=[],edge_unresolved=True)
    before=copy.deepcopy(selection)
    for name in ('report.html','beginner_report.html','technical_report.html'):
        (run/name).write_text('<html><body>existing report</body></html>',encoding='utf-8')
    artifacts=promote_dataset_report(profile,run,selection)
    assert 'data_diagnostics/dataset_report.html' in artifacts
    assert selection==before
    assert 'data_diagnostics/dataset_report.html' in (run/'report.html').read_text(encoding='utf-8')
    summary=json.loads((run/'data_diagnostics/dataset_summary.json').read_text(encoding='utf-8'))
    assert summary['manifest']['local_selection']['selection_valid'] is False
    assert summary['manifest']['selector_input'] is False


@pytest.mark.parametrize("mode", ["off", "local", "warmup"])
def test_trainer_initializes_diagnostics_after_dataset_local_is_deleted(mode, monkeypatch):
    """Exercise the trainer's real startup block with a prepared DataLoader.

    Model loading and unrelated profile setup are omitted, but the deleted-local
    lifetime, Accelerate loader and DatasetDiagnostics constructor are real.
    """
    import ast
    import torch
    from accelerate import Accelerator
    from dq_profile import trainer_runtime

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            key = "D:/fixture/startup.png"
            info = NS(absolute_path=key, bucket_reso=(720, 720), caption="from_caption",
                      num_repeats=20, is_reg=False)
            subset = NS(image_dir="D:/fixture", subset_index=0, class_tokens="fallback")
            self.datasets = [NS(width=720, height=720, image_data={key: info},
                                image_to_subset={key: subset})]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise AssertionError("diagnostic initialization must not consume training batches")

    accelerator = Accelerator(cpu=True)
    dataset = Dataset()
    loader = accelerator.prepare_data_loader(torch.utils.data.DataLoader(dataset, batch_size=1))
    network = torch.nn.Linear(2, 2)
    base = torch.nn.Linear(2, 2).requires_grad_(False)
    trainer = NS(_dataset_diagnostics="stale")
    args = NS(dq_profile_enabled=True, dq_profile_data_diagnostics=mode,
              dq_profile_protocol="v24-acceptance-local", gradient_accumulation_steps=1)
    runtime_calls = []
    monkeypatch.setattr(trainer_runtime, "DiagnosticProfileRuntime",
                        lambda **kw: runtime_calls.append(kw))

    source = Path(__file__).parents[1] / "dq_profile" / "copied_train_network.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NetworkTrainer")
    train = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    deleted = next(n for n in train.body if isinstance(n, ast.Delete)
                   and any(isinstance(t, ast.Name) and t.id == "train_dataset_group" for t in n.targets))
    start = next(i for i, n in enumerate(train.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == "_dataset_diagnostics" for t in n.targets))
    end = next(i for i in range(start, len(train.body)) if isinstance(train.body[i], ast.For))
    assert deleted.lineno < train.body[start].lineno
    wrapper = ast.parse("def initialize(self, args):\n    train_dataset_group = train_dataloader.dataset\n").body[0]
    wrapper.body += [copy.deepcopy(deleted), *copy.deepcopy(train.body[start:end])]
    namespace = dict(train_dataloader=loader, accelerator=accelerator, network=network,
                     unet=base, text_encoders=[])
    code = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)
    namespace["initialize"](trainer, args)

    assert runtime_calls == [dict(args=args, trainer=trainer)]
    observer = trainer._dataset_diagnostics
    if mode == "off":
        assert observer is None
        return
    assert len(observer.inventory) == 1
    assert observer.inventory[0]["caption"] == "from_caption"
    assert observer.inventory[0]["presented_count"] == 0
    if mode == "warmup":
        saved = observer.initial["network"]["weight"].clone()
        with torch.no_grad():
            network.weight.add_(1)
        assert torch.equal(observer.initial["network"]["weight"], saved)
        assert not torch.equal(network.weight, saved)
    else:
        assert observer.initial is None
