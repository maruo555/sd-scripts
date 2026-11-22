from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from library.promote_bank import BankItem, PromoteBank
from sdxl_train_network import SdxlNetworkTrainer


def item(index, cls, key, ids=None, weighted=None):
    batch = {"loss_weights": torch.ones(1), "class_tokens":[cls], "image_keys":[key],
             "input_ids_list": ids if ids is not None else [torch.tensor([[[2,1,5]]]),torch.tensor([[[2,1,5]]])],
             "original_sizes_hw":torch.tensor([[48,48]]), "crop_top_lefts":torch.tensor([[0,0]]),"target_sizes_hw":torch.tensor([[48,48]])}
    return BankItem.capture(batch,torch.ones(1,1,2,2,dtype=torch.float32),torch.ones(1,1,2,2)*0.5,torch.tensor([index]),None,None,weighted)


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent/"fixtures"/"promote-bank.json").read_text())["cases"],
                         ids=lambda case:case["case"])
def test_bank_selection_matches_legacy(case):
    bank = PromoteBank(case["bank_size"])
    for i,(cls,key) in enumerate(case["rows"]):
        bank.add(item(i,cls,key))
    assert bank.finalize(final_epoch=True)
    selected = [int(it.timesteps.item()) for it in bank.items]
    assert selected == case["selected"]
    assert not bank.wants_item() and not bank.pool
    assert bank.construction_size == len(case["rows"])
    assert bank.finalize(final_epoch=False)
    assert [int(it.timesteps.item()) for it in bank.items] == selected


def test_incomplete_pool_and_independent_materialization():
    bank = PromoteBank(2)
    ids = [torch.tensor([[[2,1,5]]]), torch.tensor([[[2,1,5]]])]
    tokens = (ids,[torch.ones_like(v,dtype=torch.float32) for v in ids])
    one = item(100,"a","file",ids,tokens)
    ids[0].zero_()
    assert one.batch["input_ids_list"][0][0,0,0].item()==2
    assert one.noisy_latents.dtype == one.target.dtype == torch.float16
    assert one.timesteps.dtype == torch.int64
    first = one.materialize(torch.device("cpu"),torch.float32)
    first[0]["input_ids_list"][0].zero_()
    first[-1][0][1].zero_()
    second=one.materialize(torch.device("cpu"),torch.float32)
    assert second[0]["input_ids_list"][0][0,0,0].item()==2
    assert second[-1][0][1][0,0,0].item()==2
    bank.add(one)
    assert not bank.finalize(final_epoch=True)
    bank.add(item(899,None,None))
    assert not bank.finalize(final_epoch=False)
    assert bank.finalize(final_epoch=True)
    assert bank.summary(1000)=={
        "bank_unique_class_tokens":1,"bank_max_per_class_tokens":1,"bank_unique_image_keys":1,"bank_max_per_image_key":1,
        "bank_timestep_low_count":1,"bank_timestep_mid_count":0,"bank_timestep_high_count":1}


def test_score_reencodes_both_saved_id_streams_without_recording(monkeypatch):
    trainer=SdxlNetworkTrainer()
    records=[]
    monkeypatch.setattr(trainer.run_controller,"record_sample_losses",lambda *args: records.append(args))
    cond_traces=[]
    def call_unet(args,acc,unet,noisy,t,conds,batch,dtype):
        assert not torch.is_grad_enabled()
        cond_traces.append([val.clone() for val in conds])
        return noisy + sum(x.mean() for x in conds)
    monkeypatch.setattr(trainer,"call_unet",call_unet)
    monkeypatch.setattr(trainer,"post_process_loss",lambda loss,*args:loss)
    accelerator=SimpleNamespace(device=torch.device("cpu"),autocast=nullcontext,unwrap_model=lambda model:model)
    model1,model2=torch.tensor(1.),torch.tensor(2.)
    strategy=SimpleNamespace()
    def encode(_, models, ids):
        assert not torch.is_grad_enabled()
        assert models[0] is model1 and models[1] is model2 and models[2] is model2
        return [ids[0].float()*model1, ids[1].float()*model2, ids[1][:,0].float()*model2]
    strategy.encode_tokens = encode
    def weighted(tokenizer,models,ids,weights):
        return encode(tokenizer,models,[values*amt for values,amt in zip(ids,weights)])
    strategy.encode_tokens_with_weights = weighted
    args=SimpleNamespace(full_fp16=False,weighted_captions=False,loss_type="l2",masked_loss=False)
    it=item(15,"a","path")
    params=(args,accelerator,None,[model1,model2],None,torch.float32,strategy,None)
    raw=trainer.score_promote_bank([it],*params)
    model2.add_(1)
    candidate=trainer.score_promote_bank([it],*params)
    assert raw!=candidate and not records
    assert torch.equal(cond_traces[0][0],cond_traces[1][0])
    assert all(not torch.equal(cond_traces[0][i],cond_traces[1][i]) for i in (1,2))
    # Weighted captions reuse the precise tokenized ID/weight arrays, without a second tokenization.
    ids=it.batch["input_ids_list"]
    weighted_item=item(15,"a","path",weighted=(ids,[torch.ones_like(x) for x in ids]))
    args.weighted_captions=True
    assert trainer.score_promote_bank([weighted_item],*params)==candidate and not records



@pytest.mark.parametrize("weighted", [False, True])
def test_bank_collection_keeps_training_loss_grad_and_rng(monkeypatch, weighted):
    import train_network

    args = SimpleNamespace(gradient_checkpointing=False, v_parameterization=False, full_fp16=False,
                           weighted_captions=weighted, loss_type="l2", masked_loss=False,
                           min_snr_gamma=None, scale_v_pred_loss_like_noise_pred=False, v_pred_like_loss=False,
                           debiased_estimation_loss=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext, unwrap_model=lambda m:m,
                                  backward=lambda loss:loss.backward())
    def noise_sampler(args, scheduler, latents):
        noise = torch.rand_like(latents)
        return noise, latents+noise, torch.tensor([20])
    monkeypatch.setattr(train_network.loss_util, "get_noise_noisy_latents_and_timesteps",noise_sampler)
    def run(collect):
        torch.manual_seed(7)
        trainer=SdxlNetworkTrainer()
        trainer.maruo_config = SimpleNamespace(observation_features={"group_loss_log"})
        bank=PromoteBank(1)
        trainer._promote_bank=bank if collect else None
        monkeypatch.setattr(trainer,"shift_scale_latents",lambda args,x:x)
        w=torch.nn.Parameter(torch.tensor(0.25))
        v=torch.nn.Parameter(torch.tensor(0.5))
        observed=[]
        monkeypatch.setattr(trainer.run_controller,"record_sample_losses",lambda batch,t,l:observed.append(l.detach().clone()))
        def denoise(args,acc,unet,x,t,conditions,batch,dtype):
            return x*w + sum(z.mean() for z in conditions)
        monkeypatch.setattr(trainer,"call_unet",denoise)
        ids=[torch.tensor([[[1,2,3]]]),torch.tensor([[[4,5,6]]])]
        token_calls=[]
        def tokenize(captions):
            token_calls.append(captions)
            return ids, [torch.ones_like(x) for x in ids]
        def encode(tokenizer,models,ids):
            return [torch.nn.functional.dropout(ids[0].float()*v,0.2),ids[1]*v,(ids[1]*v).mean().reshape(1)]
        strategy=SimpleNamespace(encode_tokens=encode,encode_tokens_with_weights=lambda t,m,ids,weights:encode(t,m,ids))
        tok=SimpleNamespace(tokenize_with_weights=tokenize)
        batch=item(0,"a","a.png",ids).batch
        batch.update(latents=torch.ones(1,1,2,2),captions=["a"],loss_weights=torch.tensor([0.75]))
        loss=trainer.forward_backward(args,accelerator,batch,[w,v],None,None,None,None,None,
                                      torch.float32,strategy,tok,True,True)
        assert trainer.group_loss_timesteps.tolist() == [20]
        assert len(observed)==1
        assert len(token_calls)==int(weighted)
        if collect:
            assert len(bank.pool)==1 and trainer._promote_capture is None
            assert (bank.pool[0].weighted_tokens is not None) == weighted
        return loss.detach(), w.grad, v.grad, torch.rand(5), observed[0]

    for reference,tracked in zip(run(False),run(True)):
        assert torch.equal(reference,tracked)


def test_score_uses_upstream_sdxl_layer_taps_eos_and_first_chunk():
    from library.strategy_sdxl import SdxlTextEncodingStrategy

    class TinyCLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Parameter(torch.zeros(12))
            self.text_projection = torch.nn.Linear(1, 1, bias=False)
            self.text_projection.weight.data.fill_(2)
            self.device = torch.device("cpu")

        def forward(self, ids, **kwargs):
            x = (ids + ids[:, :1]).float().unsqueeze(-1)
            hidden = [x]
            for layer in self.layers:
                x = x + layer
                hidden.append(x)
            return {"hidden_states": hidden, "last_hidden_state": x}

    trainer = SdxlNetworkTrainer()
    trainer.post_process_loss = lambda loss, *args: loss
    conditions_seen = []
    def denoise(args, acc, net, x, times, cond, batch, dtype):
        conditions_seen.append([v.clone() for v in cond])
        return x + sum(v.mean() for v in cond)
    trainer.call_unet = denoise
    te1, te2 = TinyCLIP(), TinyCLIP()
    ids = torch.zeros(1, 2, 77, dtype=torch.long)
    ids[0, :, 0] = torch.tensor([4, 9])
    ids[0, :, 5] = 2
    entry = item(15, "a", "image", [ids.clone(), ids.clone()])
    tokenizer = SimpleNamespace(model_max_length=77, eos_token_id=2)
    args = SimpleNamespace(full_fp16=False, weighted_captions=False, loss_type="l2", masked_loss=False)
    acc = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext, unwrap_model=lambda m:m)
    def score():
        return trainer.score_promote_bank([entry], args, acc, None, [te1, te2], None, torch.float32,
                                          SdxlTextEncodingStrategy(), SimpleNamespace(tokenizer1=tokenizer, tokenizer2=tokenizer))
    initial = score()
    assert conditions_seen[-1][2].item() == 2 * (4 + 2)
    with torch.no_grad():
        te1.layers[-1].fill_(3)
    assert score() == initial
    with torch.no_grad():
        te1.layers[-2].fill_(3)
    assert score() != initial
    first_changed = conditions_seen[-1]
    with torch.no_grad():
        te2.layers[-1].fill_(3)
    score()
    second_changed = conditions_seen[-1]
    assert torch.equal(first_changed[0], second_changed[0])
    assert torch.equal(first_changed[1], second_changed[1])
    assert second_changed[2].item() == first_changed[2].item() + 6
