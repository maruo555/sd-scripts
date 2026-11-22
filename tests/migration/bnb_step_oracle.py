"""Shared per-update oracle for synchronized-once and high-threshold bitsandbytes optimizers."""
import copy
from types import SimpleNamespace

import bitsandbytes as bnb
import torch


def assert_pair(reference, actual):
    assert len(reference.param_groups)==len(actual.param_groups)
    for ref_group, group in zip(reference.param_groups, actual.param_groups):
        assert {k:v for k,v in ref_group.items() if k!="params"} == {k:v for k,v in group.items() if k!="params"}
        for ref, param in zip(ref_group["params"], group["params"]):
            assert ref.shape==param.shape and ref.dtype==param.dtype and ref.device==param.device
            assert torch.equal(ref, param)
            assert (ref in reference.state)==(param in actual.state)
            if ref not in reference.state:
                continue
            expected, state=reference.state[ref],actual.state[param]
            assert expected.keys()==state.keys()
            for key,a in expected.items():
                b=state[key]
                if isinstance(a,torch.Tensor):
                    assert (a.dtype,a.device,a.shape)==(b.dtype,b.device,b.shape), key
                    assert torch.equal(a,b),key
                else:
                    assert a==b,key


def compare_steps(cls,options,dtype):
    shapes=[(4095,), (4096,), (4097,), (4,768), (1280,4), (320,4)]
    gen=torch.Generator(device="cuda").manual_seed(33)
    initial=[torch.randn(s,generator=gen,device="cuda",dtype=dtype)*0.03 for s in shapes]
    def create(kind,weights):
        params=[torch.nn.Parameter(v.clone()) for v in weights]
        groups=[{"params":params[:2],"lr":3e-4},{"params":params[2:4],"lr":2e-4},{"params":params[4:],"lr":3.5e-4}]
        return kind(groups,**options),params
    stock,ref_params=create(bnb.optim.AdamW8bit,initial)
    custom,params=create(cls,initial)
    for step in range(7):
        for group_index,(g1,g2) in enumerate(zip(stock.param_groups,custom.param_groups)):
            g1["lr"]=g2["lr"]=(0.,3e-4,2e-4,3.5e-4)[0 if step==0 else group_index+1]
        for i,(ref,param) in enumerate(zip(ref_params,params)):
            grad=torch.randn(param.shape,device="cuda",dtype=dtype,generator=gen)
            if step==4:
                grad.zero_()
            ref.grad=param.grad=None
            if (i==4 and step<2) or (i==5 and step==5):
                continue
            ref.grad,param.grad=grad,grad.clone()
        def closure():
            return torch.tensor(19.0,device="cuda")
        if step==3:
            assert custom.step(closure).item()==stock.step(closure).item()==19
        else:
            stock.step()
            custom.step()
        assert_pair(stock,custom)
        for ref in ref_params:
            if ref in stock.state:
                state=stock.state[ref]
                assert state["state1"].dtype==state["state2"].dtype==(torch.float32 if ref.numel()<stock.args.min_8bit_size else torch.uint8)
        if step==4:
            # Load each implementation's prior history into its counterpart; keep parameter identities and group order.
            a,b=copy.deepcopy(stock.state_dict()),copy.deepcopy(custom.state_dict())
            stock.load_state_dict(b)
            custom.load_state_dict(a)
            assert_pair(stock,custom)


def compare_prepared_boundaries(cls,options):
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer
    from accelerate.scheduler import AcceleratedScheduler
    from library.avg_ckpt_util import apply_lora_snapshot
    from library.avg_training import reset_optimizer_moments
    from library.grad_norm_guardian import GradNormGuardian
    from train_network import NetworkTrainer

    accelerator=Accelerator(mixed_precision="fp16")
    root=torch.ones(2, device="cuda")
    net=torch.nn.Module()
    net.register_parameter("lora_w",torch.nn.Parameter(root.clone()))
    ref=torch.nn.Parameter(root.clone())
    stock=bnb.optim.AdamW8bit([ref], lr=0.01, **options)
    custom=cls([net.lora_w], lr=0.01, **options)
    scheduler=torch.optim.lr_scheduler.StepLR(custom, 1, gamma=0.95)
    reference_scheduler=torch.optim.lr_scheduler.StepLR(stock, 1, gamma=0.95)
    prepared,wrapped_scheduler=accelerator.prepare(custom,scheduler)
    assert isinstance(prepared, AcceleratedOptimizer) and isinstance(wrapped_scheduler, AcceleratedScheduler)
    trainer=NetworkTrainer()
    trainer.all_reduce_network=lambda *args:None
    guardian=GradNormGuardian("stable_no_threshoff",net.parameters())
    for i,factor in enumerate((1., 10., float("nan"), 1., -0.5, 0.25, -0.25)):
        scale=accelerator.scaler.get_scale()
        # Real accepted attempts, Guardian rejection and independent AMP overflow.
        accelerator.backward(net.lora_w.sum()*factor)
        decision=guardian.observe()
        ref.grad=net.lora_w.grad.detach().clone()/scale
        old_step=scheduler.last_epoch
        trainer._step_optimizer(SimpleNamespace(max_grad_norm=0),accelerator,net,prepared,wrapped_scheduler,
                                skip_update=decision.skipped)
        if i in (1,2):
            assert scheduler.last_epoch==old_step and decision.skipped==(i==1)
            assert (accelerator.scaler.get_scale()==scale)==(i==1)
        else:
            assert not decision.skipped and not prepared.step_was_skipped
            stock.step()
            reference_scheduler.step()
        stock.zero_grad(set_to_none=True)
        assert_pair(stock, custom)
        assert scheduler.last_epoch==reference_scheduler.last_epoch
        # Apply candidate values twice (with / without reset), as the average controller does.
        if i in (3,4):
            shifted=net.lora_w.detach().cpu().clone()-0.1
            original_param=net.lora_w
            apply_lora_snapshot(net,{"lora_w":shifted})
            with torch.no_grad():
                ref.copy_(shifted)
            if i==4:
                reset_optimizer_moments(prepared)
            assert original_param is custom.param_groups[0]["params"][0] and original_param is net.lora_w
            assert_pair(stock,custom)
