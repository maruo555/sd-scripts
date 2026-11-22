from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from library import dq_triton
from library.dq_training import DQTraining
from library.maruo_training_config import resolve_training_config
from library.rounding_util import compute_scale_bits, fake_quantize_levels


@pytest.mark.parametrize("has_scale,has_fused,has_regular,fused,expected",
                         [(False,True,True,True,"sf"),(True,False,True,True,"sfr"),
                          (True,False,False,True,"sfr"),(True,False,True,False,"sr"),
                          (True,False,False,False,"sr")])
def test_each_fallback_reuses_one_draw(monkeypatch,has_scale,has_fused,has_regular,fused,expected):
    routes=[]
    rand_seen=[]
    packed_seen=[]
    def scale(x,*,bits,range_mul,eps):
        routes.append("s")
        return compute_scale_bits(x,bits=bits,range_mul=range_mul) if has_scale else None
    def quant(x,kargs):
        rand_seen.append(kargs["rand"])
        return fake_quantize_levels(x,scale=kargs["scale"],qmax=kargs["qmax"],rand=kargs["rand"]).detach()
    def regular(x,**kw):
        routes.append("r")
        y=quant(x,kw)
        return y if has_regular else None
    def fused_kernel(x,**kw):
        routes.append("f")
        y=quant(x,kw)
        return (y,torch.tensor([1,2,3,4,5])) if has_fused else None
    monkeypatch.setattr(dq_triton.kernels,"triton_compute_scale_bits_channel_rms",scale)
    monkeypatch.setattr(dq_triton.kernels,"triton_fake_quantize_levels_stoch",regular)
    monkeypatch.setattr(dq_triton.kernels,"triton_fake_quantize_levels_stoch_with_stats",fused_kernel)
    x=torch.linspace(-1,1,24).reshape(2,3,4).requires_grad_()
    state=torch.get_rng_state()
    scale_ref=compute_scale_bits(x,bits=8,range_mul=2.7)
    ref=fake_quantize_levels(x,scale=scale_ref,qmax=127)
    next_ref=torch.rand(3)
    torch.set_rng_state(state)
    observed=[]
    y=dq_triton.quantize_delta_triton(x,bits=8,range_mul=2.7,observe=lambda *args:observed.append(args),
                                      fused=fused,collect_packed=packed_seen.append)
    y.sum().backward()
    assert torch.equal(y,ref) and torch.equal(torch.rand(3),next_ref) and torch.equal(x.grad,torch.ones_like(x))
    assert "".join(routes)==expected
    assert all(t.data_ptr()==rand_seen[0].data_ptr() for t in rand_seen)
    successful_fused=fused and has_fused
    assert len(observed)==int(not successful_fused) and len(packed_seen)==int(successful_fused)


@pytest.mark.parametrize("skip", [False,True])
def test_fused_route_for_sample_and_promote_keeps_only_previous_route(skip):
    network=SimpleNamespace(unet_loras=[SimpleNamespace(lora_name="lora_unet_mid_attn2_to_v")],
                            text_encoder_loras=[SimpleNamespace(lora_name="lora_te1_encoder_layers_11_mlp_fc2")])
    cfg=resolve_training_config(Namespace(network_module="networks.lora",dq_delta_bits=8,dq_delta_scope="both",
          dq_delta_log=True,dq_delta_log_every=2,dq_delta_log_scope="both",dq_delta_triton_stats=True,dq_delta_use_triton=True,
          dq_delta_auto_range_mul=True,dq_delta_auto_preset="clip_rate_high",dq_delta_auto_every=1))
    controller=DQTraining(cfg,network,total_steps=4,lr_warmup_steps=0,num_processes=1)
    routes=[]
    def fake_kernel(x,*,bits,range_mul,fused,observe,collect_packed):
        routes.append((bits,fused,observe is not None,collect_packed is not None))
        if collect_packed:
            collect_packed(torch.tensor([10.,1.,1.,1.,1.]))
        elif observe:
            observe(x,x,x.float(),torch.ones(1),127)
        return x
    controller.triton_quantizer=fake_kernel
    def exercise():
        x=torch.ones(2,3,4)
        for mod in network.unet_loras+network.text_encoder_loras:
            mod.delta_quantizer(x)
    exercise()
    assert routes==[(8,False,False,False)]*2 and controller.attempt is None
    routes.clear()
    controller.begin_attempt(0)
    exercise()
    assert routes==[(8,False,True,False),(8,False,False,False)]
    controller.finish_attempt(1,epoch=1,guardian_skipped=False)
    controller.begin_attempt(1)
    routes.clear()
    exercise()
    assert routes==[(8,True,True,True)]*2
    last=controller.finish_attempt(2,epoch=1,guardian_skipped=skip)
    mul=controller.common_mul
    routes.clear()
    exercise()
    assert routes==[(8,not skip,False,False)]*2
    assert not controller.totals and controller.last_result is last and controller.common_mul==mul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Triton kernels required")
@pytest.mark.parametrize("shape", [(48,10),(2,77,128),(1,8,16,32)])
def test_real_scale_rounding_and_packed_statistics(shape):
    kernels=dq_triton.kernels
    if not kernels.is_triton_available():
        pytest.skip("Triton unavailable")
    gen=torch.Generator(device="cuda").manual_seed(511)
    x=torch.randn(shape,device="cuda",dtype=torch.float16,generator=gen)
    rnd=torch.rand(shape,device="cuda",generator=gen)
    scale=kernels.triton_compute_scale_bits_channel_rms(x,bits=8,range_mul=2.7,eps=1e-8)
    assert scale is not None
    torch.testing.assert_close(scale,compute_scale_bits(x,bits=8,range_mul=2.7),rtol=5e-6,atol=1e-8)
    ref=fake_quantize_levels(x,scale=scale,qmax=127,rand=rnd)
    regular=kernels.triton_fake_quantize_levels_stoch(x,scale=scale,qmin=-127,qmax=127,rand=rnd)
    packed=kernels.triton_fake_quantize_levels_stoch_with_stats(x,scale=scale,qmin=-127,qmax=127,rand=rnd)
    assert regular is not None and packed is not None
    rounded,sums=packed
    # Triton returns the stored rounded value. All dispatch routes apply the STE expression in Torch.
    assert torch.equal(regular,rounded)
    assert torch.equal(x+(rounded-x).detach(),ref)
    xd,yd=x.float().flatten(),rounded.float().flatten()
    expect=torch.stack([xd.new_tensor(x.numel()),(x.float().div(scale).abs()>=127).sum().float(),
                        xd.dot(xd),yd.dot(yd),xd.dot(yd)])
    assert torch.equal(sums[:2],expect[:2])
    torch.testing.assert_close(sums[2:],expect[2:],rtol=1e-5,atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA RNG and Triton")
@pytest.mark.parametrize("guardian_skipped",[False,True])
@pytest.mark.parametrize("fallback",[False,True])
def test_promote_pair_after_observation_restores_cuda_rng_and_keeps_training_state(monkeypatch,guardian_skipped,fallback):
    from library.avg_training import score_pair

    if not dq_triton.kernels.is_triton_available():
        pytest.skip("real Triton kernels unavailable")
    adapter=SimpleNamespace(lora_name="lora_unet_mid_attn2_to_v")
    network=SimpleNamespace(unet_loras=[adapter],text_encoder_loras=[])
    config=resolve_training_config(Namespace(network_module="networks.lora",dq_delta_bits=8,dq_delta_log=True,
                   dq_delta_log_every=1,dq_delta_auto_range_mul=True,dq_delta_auto_every=1,dq_delta_auto_preset="clip_rate_high",
                   dq_delta_triton_stats=True,dq_delta_use_triton=True))
    ctrl=DQTraining(config,network,total_steps=4,lr_warmup_steps=0,num_processes=1)
    model=torch.nn.Module()
    model.register_parameter("lora_w",torch.nn.Parameter(torch.tensor(1.,device="cuda")))
    x=torch.arange(72,device="cuda",dtype=torch.float16).reshape(2,3,12)/8
    ctrl.begin_attempt(0)
    adapter.delta_quantizer(x)
    last=ctrl.finish_attempt(1,epoch=1,guardian_skipped=guardian_skipped)
    state=(ctrl.common_mul,ctrl.active_bits,ctrl.auto_controller.last_step,ctrl.error_rms_ema)
    traces=[]
    for name in ("triton_fake_quantize_levels_stoch","triton_fake_quantize_levels_stoch_with_stats"):
        kernel=getattr(dq_triton.kernels,name)
        def wrapper(*args,_name=name,_kernel=kernel,**kwargs):
            traces.append(_name)
            return None if fallback else _kernel(*args,**kwargs)
        monkeypatch.setattr(dq_triton.kernels,name,wrapper)
    rng=torch.cuda.get_rng_state().clone()
    cpu=torch.get_rng_state().clone()
    raw={"lora_w":torch.tensor(1.)}
    candidate={"lora_w":torch.tensor(1.5)}
    sample=SimpleNamespace(items=[x])
    inputs=[]
    def score(items):
        y=adapter.delta_quantizer(items[0]*model.lora_w)
        inputs.append(y.clone())
        return y.float().square().mean().item()
    a,b=score_pair(model,raw,candidate,score,sample,torch.device("cuda",0))
    assert a<b and model.lora_w.item()==1
    assert torch.equal(torch.cuda.get_rng_state(),rng) and torch.equal(torch.get_rng_state(),cpu)
    assert state==(ctrl.common_mul,ctrl.active_bits,ctrl.auto_controller.last_step,ctrl.error_rms_ema)
    assert ctrl.last_result is last and not ctrl.totals
    want="triton_fake_quantize_levels_stoch"
    expected=[want] if guardian_skipped else [want+"_with_stats",want] if fallback else [want+"_with_stats"]
    assert traces==expected*2
    # Repeat the independently controlled pair using the very same scale and random draws as the kernel.
    for v,recorded in zip((1.,1.5),inputs):
        torch.cuda.set_rng_state(rng)
        value=x*v
        scale=dq_triton.kernels.triton_compute_scale_bits_channel_rms(value,bits=8,range_mul=ctrl.common_mul,eps=1e-8)
        oracle=fake_quantize_levels(value,scale=scale,qmax=127)
        assert torch.equal(oracle,recorded)
    torch.cuda.set_rng_state(rng)
