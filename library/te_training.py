"""Target selection and checkpoint compatibility for custom SDXL LoRA training."""

import torch


def mlp_fc_targets(text_encoders):
    if not isinstance(text_encoders, (list, tuple)) or len(text_encoders) != 2:
        raise ValueError("--te_mlp_fc_only requires both SDXL text encoders")
    targets = []
    for index, encoder in enumerate(text_encoders, 1):
        path = "text_model.encoder.layers"
        layers = getattr(getattr(getattr(encoder, "text_model", None), "encoder", None), "layers", None)
        if not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)) or not len(layers):
            raise ValueError(f"TE{index}: expected encoder blocks at {path}")
        selected = []
        for block_index, layer in enumerate(layers):
            for name in ("fc1", "fc2"):
                full_path = f"{path}.{block_index}.mlp.{name}"
                linear = getattr(getattr(layer, "mlp", None), name, None)
                if not isinstance(linear, torch.nn.Linear):
                    raise ValueError(f"TE{index}: expected a Linear at {full_path}")
                selected.append((full_path, linear))
        targets.append(selected)
    return targets



def load_initial_weights(network, weights, args):
    """Load a normalized training checkpoint after apply_to, including old files that omit alpha."""
    if args.network_train_unet_only:
        weights = {k: v for k, v in weights.items() if not k.startswith(("lora_te1_", "lora_te2_"))}
    if args.network_train_text_encoder_only:
        weights = {k: v for k, v in weights.items() if not k.startswith("lora_unet_")}
    adapters = network.text_encoder_loras + network.unet_loras
    for adapter in adapters:
        name = adapter.lora_name
        key = name + ".alpha"
        down = weights.get(name + ".lora_down.weight")
        if key not in weights and down is not None and down.ndim:
            weights[key] = torch.tensor(float(down.shape[0]))
    info = network.load_state_dict(weights, strict=True)
    for adapter in adapters:
        # Legacy checkpoints preserve a zero buffer, with effective alpha equal to rank.
        alpha = adapter.alpha.item()
        adapter.scale = (alpha if alpha != 0 else adapter.lora_dim) / adapter.lora_dim
    return info
