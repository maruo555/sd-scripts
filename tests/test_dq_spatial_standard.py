"""Standard diagnostic coverage on a tiny synthetic CPU model."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from PIL import Image

import numpy as np
import pytest
import torch

from dq_profile.spatial import (UNIFORM_MULS, STANDARD_VERSION, VIEWS, LEAVES,
    build_conditions, component_moments, metrics_from_moments, summarize)
from dq_profile.spatial_report import build_data, write_standard_report
from dq_profile.spatial_runtime import SpatialDiagnosticRuntime
from dq_profile.snapshot import TrainingSnapshot
from dq_profile.replay import ReplayBatch
from dq_profile.metrics import ExactGradient
from dq_profile.v2_calibration import fingerprint_tree
from library.dq_mul_policy import MulPolicy

from dq_spatial_test_support import TinyDiagnosticNetwork


@pytest.mark.parametrize('body,count', [(None,6),(2.7,6),(3.15,7),(3.45,8),(3.75,7),(4.05,8)])
def test_conditions_and_aliases(body,count):
    rows, skipped=build_conditions(body)
    assert len(rows)==count
    assert len({tuple(r['mul']) for r in rows})==count
    assert [r['mul'][0] for r in rows[:5]]==list(UNIFORM_MULS)
    assert next(r for r in rows if r['id']=='attn2_te_reference')['mul']==[2.7,3.75,3.75,3.75]
    if body==3.75:
        assert rows[5]['aliases'][0]['id']=='attn2_te_body_l270'
    assert len(build_conditions(body,spatial=False)[0])==5


def test_component_math_matches_flat_vectors():
    names=['lora_te1_x','lora_te2_x','lora_unet_x_attn2_to_k','lora_unet_x_ff_net']
    left={k:torch.tensor([i+1.,i+2.]) for i,k in enumerate(names)}
    right={k:torch.tensor([i+1.5,i-1.]) for i,k in enumerate(names)}
    stats=metrics_from_moments(component_moments(ExactGradient(left,1),ExactGradient(right,1)))
    a=torch.cat(list(left.values())).double();b=torch.cat(list(right.values())).double()
    assert stats['all']['distance']==pytest.approx(float((b-a).norm()/a.norm()))
    assert stats['all']['parallel']==pytest.approx(float(torch.dot(a,b)/a.square().sum()))
    assert sum(stats[k]['difference_sq'] for k in LEAVES)==stats['all']['difference_sq']
    zero={k:{'reference_sq':0.,'quantized_sq':1.,'dot':0.,'difference_sq':1.} for k in LEAVES}
    assert metrics_from_moments(zero)['all']['distance'] is None
    with pytest.raises(ValueError,match='topology'):
        component_moments(ExactGradient(left,1),ExactGradient({},1))


def test_incomplete_bins_rejected():
    with pytest.raises(ValueError,match='timestep'):
        summarize([dict(source_group='s',timestep_bin=0,value=1.,parallel=1.)])


TRAIN=['--pretrained_model_name_or_path=synthetic.safetensors','--dataset_config=synthetic.toml']


@pytest.mark.parametrize('mode',['standard','strict'])
@pytest.mark.parametrize('flags',[[],['--dq-profile-dropout-on'],['--dq-profile-uniform-only']])
def test_public_standard(monkeypatch,tmp_path,mode,flags):
    import dq_profile.__main__ as cli
    import dq_profile.production_entry as entry
    from dq_profile.production_runner import profile_command
    seen=[]
    monkeypatch.setattr(entry,'run_profile_request',lambda request,options:(seen.append(request) or NS(status='test',run_dir=tmp_path,report=None)))
    assert cli.main(TRAIN+['--dq-profile-mode='+mode]+flags)==0
    r=seen[0]
    assert r.preset.name=='canonical-v2' and r.te_quantized
    assert r.execution_mode.core_grid==UNIFORM_MULS and r.execution_mode.max_edge_extension_rounds==0
    assert r.data_diagnostics=='warmup'
    argv=profile_command(r,protocol='v24-acceptance-local',run_dir=tmp_path,source_map=tmp_path/'map.json',name='local',range_muls=UNIFORM_MULS,max_images=8)
    assert '--dq_profile_standard_version=2' in argv
    assert '--dq_profile_te_quantized' in argv
    assert ('--dq_profile_dropout_on' in argv)==('--dq-profile-dropout-on' in flags)
    assert r.fixed_policy_contract()['standard_version']==STANDARD_VERSION


def test_scope_legacy_and_rejections(monkeypatch,tmp_path):
    import dq_profile.__main__ as cli
    import dq_profile.production_entry as entry
    seen=[]
    monkeypatch.setattr(entry,'run_profile_request',lambda request,options:(seen.append(request) or NS(status='test',run_dir=tmp_path,report=None)))
    with pytest.raises(ValueError,match='fixes'):
        cli.main(TRAIN+['--dq-profile-no-te-quantized'])
    assert not seen
    cli.main(TRAIN+['--dq-profile-preset=canonical-v1','--dq-profile-no-te-quantized'])
    assert not seen[-1].te_quantized and seen[-1].data_diagnostics=='off'
    cli.main(TRAIN+['--dq_delta_scope=unet'])
    assert seen[-1].preset.expected_explicit['dq_delta_scope']=='both'


def run_cpu_runtime(root,dropout=True,uniform=False):
    root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(817)
    net=TinyDiagnosticNetwork()
    class Trainer:
        def _set_network_multiplier_from_batch(self,*a):pass
        def _get_text_conds_for_batch(self,*a,**kw):return ()
        def _compute_batch_loss(self,args,acc,batch,sched,unet,conds,noisy,ts,target,huber,dtype,**kw):
            return (unet(noisy)-target).square().mean()
    trainer=Trainer()
    optimizer=torch.optim.SGD(net.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1)
    args=NS(seed=39,dq_profile_seed=39,dq_profile_protocol='v24-acceptance-local',
        dq_profile_range_muls_resolved=UNIFORM_MULS,dq_profile_run_dir=str(root),
        dq_delta_granularity='channel',dq_delta_stat='rms',dq_delta_bits=8,dq_delta_range_mul=3.15,
        dq_quantize_z=False,dq_delta_use_triton=False,dq_delta_step=None,dq_delta_mode='stoch',dq_delta_scope='unet',
        dq_profile_stochastic_repeats=2,gradient_checkpointing=False,dq_profile_timestep_bins=4,
        dq_profile_max_images=8,dq_profile_safety_no_quant_noise_replicas_resolved=3,
        dq_profile_safety_candidate_noise_replicas_resolved=2,dq_profile_safety_quant_repeats_resolved=2,
        dq_profile_dropout_on=dropout,dq_profile_uniform_only=uniform,
        dq_profile_policy_grid_resolved={f'{v:.12g}':MulPolicy.from_dict({'base_mul':v}).to_dict() for v in UNIFORM_MULS})
    runtime=SpatialDiagnosticRuntime(args=args,trainer=trainer)
    selected=[ReplayBatch(index=i,source_epoch=0,source_step=i,global_step=10,
        batch={'image_keys':[str((root/f'image_{i}.png').resolve())]},latents=torch.ones(8,8),noise=torch.ones(8,8),
        noisy_latents=torch.arange(64,dtype=torch.float32).reshape(8,8)/30+i*.03,
        timesteps=torch.tensor([0]),target=torch.full((8,8),.5+i*.01),model_seed=39) for i in range(8)]
    runtime._select_probe_items=lambda sequence:(selected,NS(resolve=lambda k:Path(k).stem,manifest=lambda:{}))
    runtime._fixed_timestep_item=lambda source,bin_index,probe_replica,**kw:replace(source,
        timesteps=torch.tensor([bin_index*200+probe_replica]),noisy_latents=source.noisy_latents+.01*probe_replica+.02*bin_index)
    accelerator=NS(unwrap_model=lambda n:n,backward=lambda loss:loss.backward(),device=torch.device('cpu'),scaler=None)
    snapshot=TrainingSnapshot.capture(network=net,optimizer=optimizer,scheduler=scheduler,scaler=None,trainer=trainer,guardian=None,global_step=10,epoch=0,data_step=10)
    before=fingerprint_tree(net.state_dict())
    ctx=dict(sequence=None,snapshot=snapshot,accelerator=accelerator,network=net,optimizer=optimizer,lr_scheduler=scheduler,
        grad_norm_guardian=None,unet=net,text_encoders=[torch.nn.Identity(),torch.nn.Identity()],tokenizers=[],
        train_unet=True,train_text_encoder=True,training_model=net,on_step_start=lambda *a:None,weight_dtype=torch.float32,noise_scheduler=None)
    runtime._run_tail_probes(**ctx)
    assert fingerprint_tree(net.state_dict())==before
    assert scheduler.last_epoch==0 and not optimizer.state
    contract=json.loads((root/'spatial_contract.json').read_text(encoding='utf-8'))
    assert len(runtime.spatial_rows)==len(contract['conditions'])*8*4*2*2*(2 if dropout else 1)
    return runtime,contract


def test_real_cpu_forward_backward_report_and_pairing(monkeypatch, tmp_path):
    ROOT = tmp_path
    runtime,contract=run_cpu_runtime(ROOT/'runtime')
    rows=runtime.spatial_rows
    assert {r['regime'] for r in rows}=={'off','on'}
    for reg in ('off','on'):
        assert len({r['dropout_mask_digest'] for r in rows if r['regime']==reg})>0
    off=next(r for r in rows if r['regime']=='off');on=next(r for r in rows if r['regime']=='on')
    assert off['reference_hash']!=on['reference_hash']
    data,public=build_data(contract,rows,iterations=40)
    d=data['datasets'][0]
    for view in VIEWS:
        for cid in (c['id'] for c in contract['conditions']):
            assert np.isfinite(d['regimes']['off']['scores'][view][cid]['parallel_p50'])
    serialized=json.dumps(public)
    assert str(ROOT) not in serialized and 'image_0.png' not in serialized
    with pytest.raises(ValueError,match='duplicate'):
        build_data(contract,rows+[rows[0]],iterations=0)
    broken=copy.deepcopy(rows);broken[1]['reference_hash']='corrupt'
    with pytest.raises(ValueError,match='mismatched'):
        build_data(contract,broken,iterations=0)
    # Supply matched synthetic initial/post MSE records separately for the UI fixture.
    inventory=[];refs=[]
    for i,path in enumerate(sorted({r['image_key'] for r in rows})):
        Image.new('RGB',(48,48),(20+i*20,70,140)).save(path)
        inventory.append(dict(path=path,image_id=f'i{i}',sample_id=f's{i}',name=f'CPU入力{i}',tags=['CPU検証'],
            caption='<script>window.invalidCaptionExecuted=true</script> synthetic caption',
            dataset_index=0,subset_index=i,num_repeats=3,is_reg=False,presented_count=9,updated_count=8,skipped_count=1))
        for b in range(4):
            for snap,mse in [('pre',.8+i*.01),('post',.4+i*.01)]:
                refs.append(dict(sample_id=f's{i}',bin=b,snapshot=snap,eval_input_id=f'{i}:{b}',raw_mse=mse))
    folder=ROOT/'runtime'/'data_diagnostics';folder.mkdir(exist_ok=True)
    for name,content in [('inventory',inventory),('reference_probes',refs)]:
        (folder/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in content),encoding='utf-8')
    # Exercise the independent production analysis, not a mock Body result.
    summary={'schema_version':'2.1.0','profile':{'protocol':'v24-acceptance-local','timestep_bins':4},
             'candidates':[c.to_dict()|{'candidate':c.name} for c in runtime.candidates]}
    runtime.artifacts.write_json('summary.json',summary)
    runtime.artifacts.write_json('source_manifest.json',{'source_contract':{'sha256':'cpu-fixture'}})
    runtime.artifacts.write_csv('gradient_tail.csv',runtime._tail_probe_result['gradient_tail_rows'])
    runtime.artifacts.write_csv('local_natural_gradient.csv',runtime._tail_probe_result['local_natural_gradient_rows'])
    import tools.analyze_dq_v24_local as analysis
    monkeypatch.setattr(analysis,'parse_args',lambda:NS(profile_dir=ROOT/'runtime',output_dir=ROOT/'analysis',dataset_id='cpu_fixture',iterations=2000,seed=2401))
    assert analysis.main()==0
    from dq_profile.production_runner import promote_analysis
    (ROOT/'demo').mkdir()
    promote_analysis(ROOT/'demo',ROOT/'analysis')
    analysis_before=(ROOT/'analysis'/'analysis_manifest.json').read_bytes()
    write_standard_report(ROOT/'runtime',ROOT/'demo',iterations=40)
    from test_dq_spatial_report_contract import assert_artifact_records
    for filename,sections in [('analysis_manifest.json',['inputs','reports']),('standard_report_manifest.json',['inputs','outputs'])]:
        manifest=json.loads((ROOT/'demo'/filename).read_text(encoding='utf-8'))
        for section in sections:assert_artifact_records(manifest[section])
    assert (ROOT/'analysis'/'analysis_manifest.json').read_bytes()==analysis_before
    shared=json.loads((ROOT/'demo'/'ai_summary.json').read_text(encoding='utf-8'))
    assert shared['warmup']['available']
    assert shared['measurement_contract']['protocol_seed']==contract['protocol_seed']
    assert len(shared['measurement_contract']['source_map_sha256'])==64
    assert public[0]['pairing']['reference_hash']==rows[0]['reference_hash']
    assert str(ROOT) not in json.dumps(shared) and 'image_0.png' not in json.dumps(shared)


@pytest.mark.parametrize('preset',['canonical-v1','canonical-v2'])
@pytest.mark.parametrize('option',[['--network_args','rank_dropout=0.4'],['--fp16_safe_norms_mode=off']])
def test_compatibility_error_identifies_the_selected_preset(preset,option):
    from dq_profile.production_cli import ProfileCompatibilityError,resolve_training_cli
    with pytest.raises(ProfileCompatibilityError) as error:
        resolve_training_cli(TRAIN+option,preset_name=preset)
    assert preset in str(error.value)


def test_public_dry_run_counts_without_gpu(monkeypatch,tmp_path):
    import dq_profile.__main__ as cli
    import dq_profile.production_runner as runner
    sources=[]
    for i in range(4):
        s=tmp_path/f'source_{i}';s.mkdir();sources.append(s)
        for j in range(2):(s/f'{j}.png').write_bytes(b'placeholder-no-image-processing')
    dataset=tmp_path/'dataset.toml'
    dataset.write_text('[general]\nresolution=1024\n[[datasets]]\n'+''.join('[[datasets.subsets]]\nimage_dir='+json.dumps(str(s))+'\n' for s in sources))
    model=tmp_path/'model.safetensors';model.write_bytes(b'placeholder-no-model-loading')
    monkeypatch.setattr(runner,'preflight',lambda *a:None)
    monkeypatch.setattr(runner,'validate_output_base',lambda p,**kw:p.resolve())
    assert cli.main(['--pretrained_model_name_or_path='+str(model),'--dataset_config='+str(dataset),
        '--dq-profile-output-dir='+str(tmp_path/'reports'),'--dq-profile-dry-run','--dq-profile-dropout-on'])==0
    plan=json.loads(next((tmp_path/'reports').rglob('execution_plan.json')).read_text(encoding='utf-8'))
    # Every quantized probe has a backward; no-quant references and warmup forwards are explicit.
    text=json.dumps(plan)
    assert 'maximum_unique_conditions' in text and 'warmup_initial_forward_only' in text
    def find(d,k):
        if isinstance(d,dict):
            if k in d:return d[k]
            return next((x for v in d.values() if (x:=find(v,k)) is not None),None)
    assert find(plan,'maximum_unique_conditions')==8
    assert find(plan,'minimum_total_probes_with_additions')==8*4*(3+20)+8*4*2*((1+2)+(1+6*2))
    assert find(plan,'maximum_total_probes_with_additions')==8*4*(3+20)+8*4*2*((1+6)+(1+8*2))


def test_uniform_only_runtime_has_no_additional_probes(tmp_path):
    ROOT = tmp_path
    runtime,contract=run_cpu_runtime(ROOT/'uniform_runtime',dropout=False,uniform=True)
    assert len(contract['conditions'])==5
    assert contract['additional_reference_probes']==0
    assert contract['local_forward_backward_calls']==8*4*(3+20)
    assert {r['regime'] for r in runtime.spatial_rows}=={'off'}
