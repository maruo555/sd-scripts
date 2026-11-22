"""Load saved SDXL adapters into their original model on CPU and check every parameter."""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def verify(repo, model, paths):
    sys.path.insert(0, str(repo.resolve()))
    from library import sdxl_model_util
    from networks import lora

    for module in (sdxl_model_util, lora):
        if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
            raise ValueError(f"Module does not belong to the specified checkout: {module.__file__}")
    te1, te2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, str(model), "cpu", torch.float16
    )
    encoders = [te1, te2]
    network, initial = lora.create_network_from_weights(1.0, str(paths[0]), vae, encoders, unet)
    network.apply_to(encoders, unet, True, True)
    del initial
    results = []
    for path in paths:
        saved = load_file(str(path), device="cpu")
        info = network.load_weights(str(path))
        if info.missing_keys or info.unexpected_keys:
            raise ValueError(f"{path} has incompatible keys: {info}")
        actual = network.state_dict()
        if set(saved) != set(actual):
            raise ValueError(f"Saved keys differ from loaded keys: {path}")
        for name, value in saved.items():
            got = actual[name].detach().cpu()
            if got.shape != value.shape or not torch.equal(got, value.to(dtype=got.dtype)):
                raise ValueError(f"Restored parameter differs: {path}, {name}")
        for adapter in network.text_encoder_loras + network.unet_loras:
            name = adapter.lora_name
            alpha = saved[name + ".alpha"].item()
            rank = saved[name + ".lora_down.weight"].shape[0]
            if float(adapter.scale) != alpha / rank:
                raise ValueError(f"Restored scale differs: {path}, {name}")
        results.append({
            "file": str(path), "tensors": len(saved),
            "te": len(network.text_encoder_loras), "unet": len(network.unet_loras),
            "missing": [], "unexpected": []
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("weights", nargs="+", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a new report filename.")
    torch.set_num_threads(2)
    with torch.no_grad():
        results = verify(args.repo, args.model, args.weights)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
