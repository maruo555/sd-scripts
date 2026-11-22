"""Four-attempt real SDXL regression with a forced Guardian rejection and AMP overflow."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sdxl_train_network import SdxlNetworkTrainer, parse_training_args, setup_parser


class ProbeTrainer(SdxlNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.events = []

    def forward_backward(self, *args, **kwargs):
        loss = super().forward_backward(*args, **kwargs)
        # H calls forward_backward(args, accelerator, batch, text_encoders, unet, network, ...).
        network = args[5]
        with torch.no_grad():
            gradient = next(p.grad for p in network.parameters() if p.grad is not None)
            if len(self.events) == 1:
                gradient.view(-1)[0] = 200_001
            elif len(self.events) == 2:
                gradient.view(-1)[0] = float("nan")
        return loss

    def _step_optimizer(self, args, accelerator, network, optimizer, lr_scheduler, *, skip_update):
        scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)
        before_epoch = scheduler.last_epoch
        scale = accelerator.scaler.get_scale()
        super()._step_optimizer(args, accelerator, network, optimizer, lr_scheduler, skip_update=skip_update)
        self.events.append({
            "guardian": skip_update, "amp": None if skip_update else optimizer.step_was_skipped,
            "lr_ticks": scheduler.last_epoch - before_epoch,
            "scale_before": scale, "scale_after": accelerator.scaler.get_scale(),
        })


if __name__ == "__main__":
    args = parse_training_args(setup_parser())
    trainer = ProbeTrainer()
    trainer.train(args)
    expected = [(False, False, 1), (True, None, 0), (False, True, 0), (False, False, 1)]
    observed = [(e["guardian"], e["amp"], e["lr_ticks"]) for e in trainer.events]
    out = Path(args.output_dir).parent / "guardian-probe.json"
    out.write_text(json.dumps(trainer.events, indent=2), encoding="utf-8")
    assert observed == expected, trainer.events
