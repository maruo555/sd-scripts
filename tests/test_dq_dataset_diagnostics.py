from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from dq_profile.dataset_diagnostics import (SCHEMA, aggregate, belongs, gradient_metrics,
    improvement, load_group_map, mean, rebuild, summarize_samples, tags, weighted_quantile, write_json)


def fixture_rows(count=8):
    inventory, refs, quant = [], [], []
    muls = [2.70, 3.15, 3.45, 3.75, 4.05]
    for i in range(count):
        sid = f"s{i}"
        inventory.append(dict(sample_id=sid, image_id=f"i{i}", path=f"D:/fixture/f{i%4}/image{i}.png",
            name=f"画像 {i}.png", folder_id=f"f{i%4}", folder_path=f"D:/fixture/f{i%4}",
            folder_name=f"フォルダ {i%4}", caption="character_a,character_b,2girls" if i%3==0 else "character_a" if i%3==1 else "character_b",
            tags=["character_a", "character_b", "2girls"] if i%3==0 else ["character_a"] if i%3==1 else ["character_b"],
            subset_index=i%4, subset_group=None, resolution=[720,720], source_group_id=f"source{i%4}",
            presented_count=20, updated_count=19, skipped_count=1, first_seen_step=0, last_seen_step=70))
        for b in range(4):
            for r in range(3):
                post=(i+1)*.01+b*.001+r*.002
                key=dict(sample_id=sid,image_id=f"i{i}",eval_input_id=f"{sid}:{b}:{r}",bin=b,noise=r)
                refs.extend([{**key,"snapshot":"pre","raw_mse":post*1.2}, {**key,"snapshot":"post","raw_mse":post}])
                if r<2:
                    for mul in muls:
                        for q in range(2):
                            quant.append({**key,"mul":mul,"quant_repeat":q,"raw_mse":post+.0001*mul*(i+1),
                                "d":.01*mul*(i+1)+q*.001,"cosine":.99,"norm_ratio":1.01})
    return inventory, refs, quant, muls


def test_paired_reference_uses_two_noises_and_counts_q_once():
    inv, refs, qs, ms = fixture_rows(1)
    for row in refs:
        row["raw_mse"] = [1, 3, 100][row["noise"]]
    for row in qs:
        row["raw_mse"] = [1.1, 3.1][row["noise"]]
    ss = summarize_samples(inv,refs,qs,ms)
    a=aggregate(ss,ms)
    assert a["quant"][0]["matched"] == 2
    assert a["quant"][0]["delta"] == pytest.approx(.1)
    assert a["quant"][0]["relative"] == pytest.approx(.05)
    assert a["loss_post_available"] == pytest.approx(104/3)


def test_image_equal_and_ratio_of_means_not_mean_ratios():
    inv, refs, qs, ms=fixture_rows(3)
    inv[1]["image_id"] = inv[0]["image_id"]
    for row in refs:
        row["raw_mse"] = [1,1,100][int(row["sample_id"][1:])] if row["snapshot"]=="pre" else [.4,.6,90][int(row["sample_id"][1:])]
    a=aggregate(summarize_samples(inv,refs,qs,ms),ms)
    assert a["loss_pre"] == 50.5
    assert a["loss_post"] == 45.25
    assert a["improvement_rel"] == pytest.approx(5.25/50.5)


def test_missing_cell_does_not_change_bin_weights_or_bridge_mul_curve():
    inv,refs,qs,ms=fixture_rows(2)
    qs.pop(0)
    ss=summarize_samples(inv,refs,qs,ms)
    assert ss[0]["bins"][0]["quant"][0]["d"] is None
    a=aggregate(ss,ms,common_muls=True)
    assert all(q["images"]==1 and q["samples"]==1 for q in a["quant"])
    assert aggregate(ss,ms)["quant"][1]["images"]==2


def test_conflicting_duplicate_ref_and_topology_invalid():
    inv,refs,qs,ms=fixture_rows(1)
    with pytest.raises(ValueError,match="conflicting duplicate"):
        summarize_samples(inv,refs+[{**refs[0],"raw_mse":999}],qs,ms)
    assert len(summarize_samples(inv,refs+refs,qs+qs,ms))==1
    metrics=gradient_metrics(dict(reference_norm=1,candidate_norm=1,difference_norm=.2,cosine=.98,topology_matches=False))
    assert metrics["d"] is None and metrics["grad_diff_norm"] is None


@pytest.mark.parametrize("value",[float('nan'),float('inf'),None])
def test_nonfinite_is_missing(value):
    assert mean([1,value]) is None
    assert improvement(value,1)["improvement_rel"] is None


def test_floor_and_weighted_quantile_and_exact_tag_matching():
    g=gradient_metrics(dict(reference_norm=0,candidate_norm=1,difference_norm=1,cosine=0,topology_matches=True))
    assert g["d"] is None and g["cosine"] is None and g["norm_ratio"] is None
    assert g["grad_diff_norm"]==1 and g["symmetric_d"]==2
    assert improvement(0,1)["improvement_rel"] is None
    assert weighted_quantile([.1,.2,5],[.49,.49,.02])==.2
    assert weighted_quantile([0,1],[.95-1e-16,.05+1e-16])==1
    assert tags(" e\u0301 ,é, anna,Anna ")==["Anna","anna","é"]
    assert not belongs({"tags":["joanna"]},{"tags_any":["anna"]})
    assert not belongs({"tags":["anna"]},{"tags_all":[]})
    assert belongs({"tags":["anna","beth"]},{"tags_all":["anna","beth"]})


def test_group_map_matches_approved_format_and_rejects_chains(tmp_path):
    p=tmp_path/'groups.json'
    write_json(p,dict(schema_version="dataset-groups-v1",aliases={"a(work)":"a"},groups=[dict(id="a",label="A",tags_any=["a"],image_paths=[],subset_groups=[],tags_all=[])]))
    groups=load_group_map(p)
    assert belongs({"tags":["a(work)"]},groups["groups"][0],groups["aliases"])
    write_json(p,dict(aliases={"a":"b","b":"c"},groups=[]))
    with pytest.raises(ValueError,match="one-hop"):
        load_group_map(p)


@pytest.mark.parametrize('aliases', [
    {' e\u0301 ': ' character_a '},
    {'é': 'character_a', ' e\u0301 ': ' character_a '},
])
def test_alias_normalization_preserves_membership(tmp_path, aliases):
    path = tmp_path / 'groups.json'
    write_json(path, {'aliases': aliases, 'groups': [{'id': 'a', 'tags_any': ['character_a']}]})
    group_map = load_group_map(path)
    assert group_map['aliases'] == {'é': 'character_a'}
    assert belongs({'tags': tags('é')}, group_map['groups'][0], group_map['aliases'])


@pytest.mark.parametrize('aliases', [
    {' ': 'a'}, {'a': ' '},
    {'é': 'a', 'e\u0301': 'b'},
    {'x': ' e\u0301 ', 'é': 'y'},
    {'é': ' e\u0301 '},
])
def test_alias_normalization_rejects_empty_conflicts_and_chains(tmp_path, aliases):
    path = tmp_path / 'groups.json'
    write_json(path, {'aliases': aliases, 'groups': []})
    with pytest.raises(ValueError):
        load_group_map(path)


def test_alias_value_unicode_normalizes_before_group_membership(tmp_path):
    path = tmp_path / 'groups.json'
    write_json(path, {'aliases': {'character_a': ' e\u0301 '}, 'groups': [{'id': 'a', 'tags_any': ['é']}]})
    group_map = load_group_map(path)
    assert group_map['aliases'] == {'character_a': 'é'}
    assert belongs({'tags': ['character_a']}, group_map['groups'][0], group_map['aliases'])


def test_rebuild_normalizes_legacy_manifest_aliases(tmp_path):
    write_fixture(tmp_path, 4)
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    manifest['group_map'] = {'aliases': {' character_a ': ' e\u0301 '},
                             'groups': [{'id': 'unicode', 'tags_any': [' e\u0301 ']}]}
    write_json(tmp_path / 'manifest.json', manifest)
    payload = rebuild(tmp_path)
    assert payload['manifest']['group_map']['aliases'] == {'character_a': 'é'}
    assert payload['manifest']['group_map']['groups'][0]['tags_any'] == ['é']
    expected = [s for s in payload['samples'] if 'character_a' in s['tags']]
    assert len(expected) == 3
    assert all(s['group_memberships'][0]['id'] == 'unicode' for s in expected)


def write_fixture(directory, count=8):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    inv,refs,qs,ms=fixture_rows(count)
    write_json(directory/'manifest.json',dict(schema_version=SCHEMA,mode="warmup",selector_input=False,measurement_contract="synthetic-test-only",max_images=52,bins=4,muls=ms,group_map=dict(schema_version="dataset-groups-v1",aliases={},groups=[dict(id=t,label=t,tags_any=[t]) for t in ("character_a","character_b")]),state_restoration=dict(status="test_fixture"),numerical_mode_parity="synthetic_fixture",ci_status="not_computed"))
    for name,rows in (("inventory",inv),("reference_probes",refs),("quant_probes",qs)):
        (directory/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    return rebuild(directory)


def test_rebuild_cpu_exports_missing_safe_json_and_escaped_html(tmp_path):
    data=write_fixture(tmp_path)
    assert data["all"]["measured_images"]==8
    from dq_profile.diagnostic_report import write_report
    data['samples'][0]['caption']='</script><img src=x onerror=alert(1)>'
    write_report(tmp_path/'dataset_report.html',data)
    html=(tmp_path/'dataset_report.html').read_text(encoding='utf-8')
    assert '</script><img src=x' not in html
    assert '\\u003c/script' in html
    assert (tmp_path/'group_quant.csv').is_file()


def toy_context(device, mode='warmup', use_lora=False):
    from dq_profile.diagnostic_eval import DatasetDiagnostics
    from dq_profile.protocol import CandidateDefinition
    from dq_profile.quant_context import ProfileQuantContext
    from dq_profile.trainer_runtime import DiagnosticProfileRuntime
    from dq_profile.replay import ReplayBatch
    torch.manual_seed(818)
    network=torch.nn.ModuleDict({'te':torch.nn.Linear(2,2,bias=False),'unet':torch.nn.Linear(2,2,bias=False)}).to(device)
    if use_lora:
        from dq_profile.copied_lora import LoRAModule, LoRANetwork, DQStatsManager
        class ToyLoRA(torch.nn.ModuleDict):
            set_delta_fake_quant = LoRANetwork.set_delta_fake_quant
            set_delta_quant_enabled = LoRANetwork.set_delta_quant_enabled
            set_dq_profile_context = LoRANetwork.set_dq_profile_context
            set_dq_stats_state = LoRANetwork.set_dq_stats_state
            export_dq_stats = LoRANetwork.export_dq_stats
            discard_dq_stats_step = LoRANetwork.discard_dq_stats_step
        modules = {}
        for scope in ('te','unet'):
            base = torch.nn.Linear(2,2,bias=False).to(device).requires_grad_(False)
            lora = LoRAModule('lora_'+scope,base,lora_dim=2,alpha=2,delta_q_bits=4,delta_q_mode='stoch')
            lora.apply_to();lora.to(device)
            torch.nn.init.normal_(lora.lora_up.weight,std=.1)
            modules[scope]=lora
        network=ToyLoRA(modules)
        network.text_encoder_loras=[network['te']];network.unet_loras=[network['unet']]
        network.dq_stats_manager=DQStatsManager()
        for lora in modules.values():lora.dq_stats_manager=network.dq_stats_manager
    network['unet'].register_buffer('marker' ,torch.tensor([1.],device=device))
    if not use_lora:network.dq_stats_manager=NS(active=False,counts=torch.tensor([2.],device=device))
    network.delta_q_enabled=False
    trainer=NS(_te_frozen_param_ids=set())
    args=NS(dq_profile_data_diagnostics=mode,dq_profile_protocol='v24-acceptance-local',gradient_accumulation_steps=1,gradient_checkpointing=False,dq_quantize_z=False,dq_delta_granularity='tensor',dq_delta_stat='rms',dq_delta_bits=4,dq_delta_range_mul=3.15,dq_delta_use_triton=False,dq_delta_step=None,dq_delta_mode='stoch')
    info=NS(absolute_path='D:/fixture/a.png',bucket_reso=(720,720),caption='a,other',num_repeats=20,is_reg=False)
    subset=NS(image_dir='D:/fixture',subset_index=0,class_tokens='wrong',group=None)
    ds=NS(image_data={'D:/fixture/a.png':info},image_to_subset={'D:/fixture/a.png':subset},resolution=[720,720])
    observer=DatasetDiagnostics(args,ds,network,network['unet'],[network['te']],trainer)
    trainer._dataset_diagnostics=observer
    backward_calls=[]
    scaler=torch.amp.GradScaler('cuda') if device=='cuda' else None
    accelerator=NS(device=torch.device(device),scaler=scaler,unwrap_model=lambda n:n,backward=lambda loss:(backward_calls.append(True),(scaler.scale(loss) if scaler is not None else loss).backward()))
    trainer._set_network_multiplier_from_batch=lambda *a:None
    trainer._get_text_conds_for_batch=lambda args,acc,batch,toks,tes,dtype,**kw:[tes[0](batch['input_ids'].float())]
    def loss_fn(args,acc,batch,sched,unet,conds,noisy,ts,target,huber,dtype,**kw):
        prediction=unet(noisy+conds[0]);observer.tap(prediction,target)
        return ((prediction-target)**2).mean()*3
    trainer._compute_batch_loss=loss_fn
    runtime=object.__new__(DiagnosticProfileRuntime)
    runtime.trainer=trainer;runtime.args=args;runtime._stats_sequence=1000000;runtime.protocol_seed=39;runtime.quant_context=ProfileQuantContext(39)
    candidate=CandidateDefinition(name='no_quant',quantized=False,initial_range_mul=None,clip_low=None,clip_high=None,auto_enabled=False)
    runtime.candidates=(candidate,)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1)
    for _ in range(2):
        optimizer.zero_grad();network['unet'](network['te'](torch.ones(1,2,device=device))).square().mean().backward();optimizer.step();scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    replay=ReplayBatch(0,0,0,2,dict(image_keys=['D:/fixture/a.png'],subset_indices=[0],input_ids=torch.ones(1,2),captions=['a,other']),latents=torch.ones(1,2),noise=torch.ones(1,2),noisy_latents=torch.ones(1,2),timesteps=torch.tensor([125]),target=torch.zeros(1,2))
    context=dict(accelerator=accelerator,network=network,optimizer=optimizer,lr_scheduler=scheduler,grad_norm_guardian=None,unet=network['unet'],text_encoders=[network['te']],tokenizers=[],train_unet=True,train_text_encoder=True,training_model=network,on_step_start=lambda *a:None,weight_dtype=torch.float32,noise_scheduler=None)
    call=dict(replay=replay,candidate=candidate,range_mul=None,phase='v2_tail_probe',probe_or_step='toy',repeat=0,dropout_enabled=False,shadow=False,update=False,do_auto_observation=False,absolute_step=2,epoch=0,**context)
    return runtime,observer,replay,call,context,backward_calls


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('fail',[False,True])
def test_initial_forward_restores_state_and_te_and_exception(device,fail):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dq_profile.snapshot import TrainingSnapshot
    from dq_profile.v2_calibration import fingerprint_tree
    runtime,observer,replay,call,context,backward=toy_context(device,use_lora=True)
    network=context['network']
    post,_,_=runtime._run_pass(**call)
    observer.record(replay,dict(source_group='source',timestep_bin=0,noise_replica=0,timestep=125),post,model_seed_id='toy')
    args=dict(network=network,optimizer=context['optimizer'],scheduler=context['lr_scheduler'],scaler=context['accelerator'].scaler,trainer=runtime.trainer,guardian=None,global_step=2,epoch=0,data_step=2)
    snapshot=TrainingSnapshot.capture(**args)
    before=fingerprint_tree(vars(snapshot))
    saved_stats=runtime._stats_sequence
    backward.clear()
    original=runtime._run_pass
    def forward(**kw):
        assert kw['diagnostic_forward_only'] is True
        assert fingerprint_tree(network.state_dict())==observer.initial_hash
        if fail:
            random.random();np.random.rand();torch.rand(2,device=device)
            network.dq_stats_manager.active=True
            network['unet'].marker.add_(5)
            raise RuntimeError('test injected failure')
        return original(**kw)
    runtime._run_pass=forward
    if fail:
        with pytest.raises(RuntimeError,match='test injected failure'):observer.evaluate_initial(runtime,snapshot,**context)
    else:
        observer.evaluate_initial(runtime,snapshot,**context)
        assert observer.references[-1]['snapshot']=='pre'
        assert observer.references[-1]['raw_mse'] != observer.references[0]['raw_mse']
    assert not backward
    assert fingerprint_tree(vars(TrainingSnapshot.capture(**args)))==before
    assert runtime._stats_sequence==saved_stats
    assert network.dq_stats_manager.active is False
    assert network['unet'].marker.item()==1
    assert observer.restoration['status']=='passed'


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('quantized',[False,True])
def test_actual_run_pass_observer_keeps_original_rows_and_gradients(device,quantized):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dq_profile.replay import seed_step_rng
    from dq_profile.v2_calibration import fingerprint_tree
    runtime,observer,replay,call,context,_=toy_context(device,'local',use_lora=True)
    if quantized:
        from dq_profile.protocol import CandidateDefinition
        call['candidate']=CandidateDefinition('mul_3.15',True,None,None,3.15,False)
        call['range_mul']=3.15
    runtime.trainer._dataset_diagnostics=None
    seed_step_rng(39,'toy')
    off,goff,_=runtime._run_pass(**call)
    runtime.trainer._dataset_diagnostics=observer
    seed_step_rng(39,'toy')
    local,glocal,_=runtime._run_pass(**call)
    assert off==local
    assert fingerprint_tree(goff.values)==fingerprint_tree(glocal.values)
    assert observer.raw_mse==pytest.approx(local['loss']/3)
