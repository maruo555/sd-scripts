"""SDXL generation-only LoRA block weights on top of the standard inference adapter."""

from networks import lora


from library.generation_lbw import parse_sdxl_lbw_weights

BLOCKS = {
    "lora_unet_input_blocks_": {4:1, 5:2, 7:3, 8:4},
    "lora_unet_output_blocks_": {i:6+i for i in range(6)},
}


def get_sdxl_lbw_index(name):
    if name.startswith(("lora_te1_", "lora_te2_", "lora_te_")):
        return 0
    if name.startswith("lora_unet_middle_block_"):
        return 5
    for prefix, lookup in BLOCKS.items():
        if name.startswith(prefix):
            block = name[len(prefix):].partition("_")[0]
            return lookup.get(int(block)) if block.isdigit() else None
    return None


class LoRAInfModule(lora.LoRAInfModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_scale = self.scale
        self.block_index = get_sdxl_lbw_index(self.lora_name)
        self.lbw_multiplier = 1.0

    def set_block_weights(self, weights):
        self.lbw_multiplier = weights[self.block_index] if weights is not None and self.block_index is not None else 1.0
        self.scale = self._alpha_scale * self.lbw_multiplier


class LoRANetwork(lora.LoRANetwork):
    def set_lbw_weights(self, weights):
        resolved = parse_sdxl_lbw_weights(weights)
        for adapter in self.text_encoder_loras + self.unet_loras:
            adapter.set_block_weights(resolved)


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    if not for_inference:
        raise ValueError("networks.lora_lbw is for inference; train with networks.lora")
    return lora._create_network_from_weights(
        multiplier, file, text_encoder, unet, weights_sd,
        network_class=LoRANetwork, module_class=LoRAInfModule, block_lr_kwargs=kwargs,
    )
