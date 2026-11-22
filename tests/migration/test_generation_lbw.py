import pytest
import torch

from library.generation_lora_strengths import (
    component_strengths_from_spec, apply_generation_strengths, format_strength_value, parse_prompt_strengths, resolve_prompt_override,
)
from networks import lora, lora_lbw
from types import SimpleNamespace


@pytest.mark.parametrize("spec", ["NONE", "xlMlT1", "1,0.5,0,1,-1e-13,0,1,0,1,0,1,0"])
def test_lbw_presets_mapping_and_reset(spec):
    values=lora_lbw.parse_sdxl_lbw_weights(spec)
    names=["lora_te1_text_model", "lora_te2_text_model"]+[
        "lora_unet_input_blocks_"+str(i)+"_layer" for i in (4,5,7,8)]+[
        "lora_unet_middle_block_2"]+["lora_unet_output_blocks_"+str(i)+"_layer" for i in range(6)]+["lora_unet_input_blocks_3_other"]
    assert len(values)==12
    for n,index in zip(names,[0,0,*range(1,12),None]):
        base=torch.nn.Linear(3,2,bias=False)
        module=lora_lbw.LoRAInfModule(n,base,multiplier=0.75,lora_dim=2,alpha=1)
        module.lora_up.weight.data.fill_(1)
        module.lora_down.weight.data.fill_(2)
        module.set_block_weights(values)
        expected=values[index] if index is not None else 1
        assert module.block_index==index and module.lbw_multiplier==expected
        # Explicit multiplier for get_weight(), normal merge() and dynamic forward use the same alpha/rank.
        w=module.get_weight(-0.5)
        torch.testing.assert_close(w,torch.full((2,3),-0.5*expected*2))
        module.set_block_weights(None)
        assert module.scale==0.5 and module.lbw_multiplier==1


def test_prompt_split_exact_value_and_full_merge():
    for v in [-1e-13,-0.01234567890123456,0.,1.,1e20]:
        encoded=format_strength_value(v)
        assert parse_prompt_strengths(encoded)==(v,)
    for text in ["1.2x","0.8,,1","inf","-nan"]:
        with pytest.raises(ValueError):
            parse_prompt_strengths(text)
    for count,left,values in [(2,1,[0.5,0.8]),(2,0,[0.1]*6),(1,1,[1,1,1])]:
        with pytest.raises(ValueError):
            resolve_prompt_override(values,total_slots=count,dynamic_slots=left,merge=True,pre_calc=False)
    assert resolve_prompt_override([0.5],total_slots=2,dynamic_slots=0,merge=True,pre_calc=False)==()
    for text in ["1,2","0,0,0,0,0,0,0,0,0,0,0,nan","x"]:
        with pytest.raises(ValueError):
            lora_lbw.parse_sdxl_lbw_weights(text)


@pytest.mark.parametrize("device", ["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable"))])
@pytest.mark.parametrize("slot_weights",[(0.6,0.),(0.,0.),(0.5,0.7)])
@pytest.mark.parametrize("layer", ["attn2_to_q","attn2_to_out"])
def test_last_regional_slot_keeps_query_and_masked_output(layer,slot_weights,device):
    def render(cls):
        base=torch.nn.Linear(3,4,bias=False,device=device)
        base.weight.data.copy_(torch.arange(12,device=device).reshape(4,3)*0.2)
        calls=[]
        parent=base.forward
        def forward(x):
            calls.append(x.shape)
            return parent(x)
        base.forward=forward
        adapters=[]
        shared={}
        for i,weight in enumerate(slot_weights):
            adapter=cls("lora_unet_middle_block_1_"+layer,base,1.,2,2)
            adapter.to(device)
            adapter.lora_up.weight.data.fill_(0.3+i)
            adapter.lora_down.weight.data.fill_(0.2+i)
            adapter.multiplier=weight
            adapter.apply_to()
            mask=torch.tensor([[1.,0.],[0.,1.]]) if i==0 else torch.tensor([[0.,1.],[1.,0.]])
            adapter.set_network(SimpleNamespace(batch_size=1, num_sub_prompts=2, sub_prompt_index=i, is_last_network=i==1,
                                 shared=shared,mask_dic={4:mask.to(device)}))
            adapters.append(adapter)
        n=2 if layer=="attn2_to_q" else 3
        x=torch.arange(n*4*3,dtype=torch.float32,device=device).reshape(n,4,3)*0.1
        y=base(x)
        assert not shared and len(calls)==1
        return y
    class AlwaysCompute(lora.LoRAInfModule):
        def _regional_delta(self, x):
            return self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
    assert torch.equal(render(lora_lbw.LoRAInfModule),render(AlwaysCompute))


def test_ordinary_zero_bypasses_low_rank_and_repeated_lbw_is_independent():
    base=torch.nn.Linear(4,3,bias=False)
    mod=lora_lbw.LoRAInfModule("lora_te2_fc",base,1.,2,2)
    mod.apply_to()
    x=torch.ones(1,4)
    calls=[]
    mod.lora_down.register_forward_hook(lambda *args:calls.append(1))
    mod.set_block_weights(lora_lbw.parse_sdxl_lbw_weights("NONE"))
    original=mod.org_forward(x)
    assert torch.equal(base(x),original) and not calls
    mod.set_block_weights(lora_lbw.parse_sdxl_lbw_weights("ALL"))
    mod.multiplier=1e-13
    base(x)
    assert calls==[1]


@pytest.mark.parametrize("name", ["lora_te1_mlp_fc1","lora_te2_mlp_fc1","lora_unet_input_blocks_4_1_proj"])
def test_dynamic_merge_and_precalc_match_saved_projection(name):
    from test_lora_factory import TinySDXL,text_encoders,saved_projection

    x=torch.ones(1,3)
    results=[]
    for mode in ("dynamic","merge","precalc"):
        base=TinySDXL()
        tes=text_encoders()
        source=tes[0].mlp.fc1 if "te1" in name else tes[1].mlp.fc1 if "te2" in name else base.input_blocks[4][1].proj
        source.weight.data.zero_()
        if source.bias is not None:
            source.bias.data.zero_()
        state=saved_projection(name,alpha=None)
        net,normalized=lora_lbw.create_network_from_weights(0.5,None,None,tes,base,weights_sd=state,for_inference=True)
        net.set_lbw_weights([0.8]*12)
        if mode=="merge":
            net.merge_to(tes,base,normalized,torch.float32,"cpu")
        else:
            net.apply_to(tes,base)
            info=net.load_state_dict(normalized,strict=False)
            assert list(info.missing_keys)==[name+".alpha"] and not info.unexpected_keys
            apply_generation_strengths(net,component_strengths_from_spec([0.5,0.5,0.5]))
            if mode=="precalc":
                net.backup_weights()
                net.restore_weights()
                net.pre_calculation()
        results.append(source(x))
    for value in results:
        torch.testing.assert_close(value,torch.full((1,3),0.6))

@pytest.mark.parametrize("kind",["linear","conv1","conv3"])
def test_known_delta_merge_precalc_and_restore(kind):
    generator=torch.Generator().manual_seed(123)
    x=torch.randn((2,3) if kind=="linear" else (1,3,7,7),generator=generator)
    for mode in ("dynamic","merge","precalc"):
        base=torch.nn.Linear(3,4,bias=False) if kind=="linear" else torch.nn.Conv2d(3,4,1 if kind=="conv1" else 3,
                                                                                 padding=0 if kind=="conv1" else 1,bias=False)
        base.weight.data.zero_()
        adapter=lora_lbw.LoRAInfModule("lora_unet_output_blocks_0_layer",base,0.5,2,1)
        adapter.lora_up.weight.data.fill_(0.25)
        adapter.lora_down.weight.data.fill_(0.5)
        adapter.set_block_weights(lora_lbw.parse_sdxl_lbw_weights("XLMLT1"))
        reference=adapter.lora_up(adapter.lora_down(x)).detach() * (0.5 * 0.6 * 0.5)
        state={"lora_up.weight":adapter.lora_up.weight.detach(),"lora_down.weight":adapter.lora_down.weight.detach()}
        if mode=="merge":
            adapter.merge_to(state,torch.float32,"cpu")
            actual=base(x)
        else:
            adapter.apply_to()
            if mode=="precalc":
                net=SimpleNamespace(text_encoder_loras=[],unet_loras=[adapter])
                lora.LoRANetwork.backup_weights(net)
                lora.LoRANetwork.pre_calculation(net)
                torch.testing.assert_close(base(x),reference)
                lora.LoRANetwork.restore_weights(net)
                assert torch.equal(base.weight,torch.zeros_like(base.weight))
                continue
            actual=base(x)
        torch.testing.assert_close(actual,reference,rtol=2e-6,atol=2e-6)


def test_unit_delta_for_each_component_and_sdxl_block():
    components=["lora_te1_mlp_fc1","lora_te2_mlp_fc1","lora_unet_output_blocks_0_attn",
                "lora_unet_output_blocks_4_attn","lora_unet_other"]
    mods=[]
    for name in components:
        src=torch.nn.Linear(1,1,bias=False)
        src.weight.data.zero_()
        adapter=lora_lbw.LoRAInfModule(name,src,1.,2,1)
        adapter.lora_up.weight.data.fill_(0.5)
        adapter.lora_down.weight.data.fill_(1.)
        adapter.set_block_weights(lora_lbw.parse_sdxl_lbw_weights("XLMLT1"))
        adapter.apply_to()
        mods.append(adapter)
    net=SimpleNamespace(text_encoder_loras=mods[:2],unet_loras=mods[2:])
    net.set_multiplier=lambda value:lora.LoRANetwork.set_multiplier(net,value)
    apply_generation_strengths(net,component_strengths_from_spec([0.8,0.25,0.5]))
    actual=torch.stack([m(torch.ones(1,1)).reshape(()) for m in mods])
    torch.testing.assert_close(actual,torch.tensor([0.4,0.125,0.15,0.,0.25]))
