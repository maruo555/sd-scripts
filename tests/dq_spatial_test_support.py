"""Small CPU-only LoRA fixture for diagnostic regressions."""
import torch
from dq_profile.copied_lora import LoRAModule, LoRANetwork


class TinyDiagnosticNetwork(torch.nn.Module):
    set_delta_fake_quant = LoRANetwork.set_delta_fake_quant
    set_delta_quant_enabled = LoRANetwork.set_delta_quant_enabled
    set_dq_profile_context = LoRANetwork.set_dq_profile_context

    def __init__(self):
        super().__init__()
        names = ['lora_te1_text_model_encoder_layers_0_mlp_fc1', 'lora_te2_text_model_encoder_layers_0_mlp_fc1', 'lora_unet_input_blocks_0_attn1_to_q', 'lora_unet_input_blocks_0_attn2_to_k', 'lora_unet_middle_block_0_ff_net_0_proj', 'lora_unet_output_blocks_0_proj_out']
        self.text_encoder_loras, self.unet_loras = [], []
        self.originals = []
        for name in names:
            original = torch.nn.Linear(8, 8, bias=False)
            original.requires_grad_(False)
            module = LoRAModule(name, original, lora_dim=2, dropout=.3, rank_dropout=.2, delta_q_bits=8, delta_q_mode='stoch', delta_q_granularity='channel', delta_q_stat='rms', delta_q_range_mul=3.15)
            module.apply_to()
            with torch.no_grad():
                module.lora_up.weight.normal_(0, .3)
            self.add_module(name, module)
            self.originals.append(original)
            (self.text_encoder_loras if name.startswith('lora_te') else self.unet_loras).append(module)
        self.set_delta_fake_quant(None, 'stoch', granularity='channel', stat='rms', bits=8, range_mul=3.15)

    def forward(self, x):
        for original in self.originals:
            x = torch.tanh(original(x))
        return x
