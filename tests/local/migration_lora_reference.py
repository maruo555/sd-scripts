"""Record or compare the upstream LoRA step with all custom features disabled.

Run record, then compare on the same device and dependency lock. References
are local to the machine and never overwritten.
"""
import argparse
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library.optimizer import get_optimizer, get_scheduler_fix
from networks.lora import LoRAModule
from train_network import setup_parser


def observe(device):
    torch.manual_seed(39)
    cuda = device == "cuda"
    optimizer_type = "AdamW8bit" if cuda else "AdamW"
    args = setup_parser().parse_args([
        "--optimizer_type", optimizer_type, "--learning_rate", "0.00035",
        "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "1",
        "--max_train_steps", "3",
    ])
    # Rank 4 gives 4096 values per matrix, enough for both 8-bit states on CUDA.
    base = torch.nn.Linear(1024, 1024, bias=False).requires_grad_(False).to(device)
    adapter = LoRAModule("lora_unet_probe", base, dropout=0.3, rank_dropout=0.2)
    adapter.apply_to()
    adapter.to(device)
    _, _, optimizer = get_optimizer(args, list(adapter.parameters()))
    scheduler = get_scheduler_fix(args, optimizer, 1)
    result = {}

    def keep(key, value):
        value = torch.as_tensor(value).detach().cpu().clone().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite reference at {key}")
        result[key] = value

    for step in range(3):
        prefix = f"step{step}/"
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 2, 1024, device=device, requires_grad=True)
        with torch.autocast(device, dtype=torch.float16, enabled=cuda):
            prediction = base(x)
        loss = prediction.float().square().mean()
        loss.backward()
        keep(prefix + "prediction", prediction)
        keep(prefix + "loss", loss)
        keep(prefix + "input_gradient", x.grad)
        keep(prefix + "lr_used", scheduler.get_last_lr())
        for name, param in adapter.named_parameters():
            keep(prefix + name + "/gradient", param.grad)
        optimizer.step()
        scheduler.step()
        keep(prefix + "next_lr", scheduler.get_last_lr())
        keep(prefix + "scheduler_epoch", scheduler.last_epoch)
        for name, param in adapter.named_parameters():
            keep(prefix + name + "/weight", param)
            for key, value in optimizer.state[param].items():
                keep(prefix + name + "/state/" + key, value)
    keep("next_rng_cpu", torch.rand(16))
    if cuda:
        keep("next_rng_cuda", torch.rand(16, device=device))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "compare"])
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "record" and args.reference.exists():
        parser.error(f"Reference already exists: {args.reference}")
    actual = observe(args.device)
    if args.action == "record":
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        save_file(actual, str(args.reference), metadata={"device": args.device})
    else:
        expected = load_file(str(args.reference))
        if actual.keys() != expected.keys():
            raise ValueError(f"Reference keys changed: {actual.keys() ^ expected.keys()}")
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=lambda msg: f"{key}: {msg}")
    print(f"{args.action} {len(actual)} fields on {args.device}: {args.reference}")


if __name__ == "__main__":
    main()
