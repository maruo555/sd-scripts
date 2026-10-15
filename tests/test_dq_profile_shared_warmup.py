"""Synthetic CPU coverage of shared warmup; no models/datasets or GPU required."""
from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dq_profile.production_cli import resolve_training_cli
from dq_profile.production_runner import profile_command
from dq_profile.replay import ReplayBatch, ReplaySequence
from dq_profile.shared_warmup import run_shared_prefix, snapshot_fingerprints
from dq_profile.snapshot import TrainingSnapshot, capture_rng_state
from dq_profile.trainer_runtime import DiagnosticProfileRuntime
from dq_profile.v2_calibration import fingerprint_tree
from sdxl_dq_dataset_profile import _preflight, _validate_and_isolate, setup_parser


def worker_args(tmp_path, *, preset="canonical-v2", mode="standard", quant_repeats=None):
    from PIL import Image
    sources = []
    for i in range(4):
        source = tmp_path / f"group-{i}"
        source.mkdir()
        Image.new("RGB", (32, 32), (i, i, i)).save(source / "sample.png")
        sources.append(source)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        "[general]\nresolution=512\n[[datasets]]\n" + "".join(
            "[[datasets.subsets]]\nimage_dir=" + json.dumps(str(source)) + "\n" for source in sources
        ), encoding="utf-8",
    )
    source_map = tmp_path / "sources.json"
    source_map.write_text(json.dumps([
        {"pattern": str(source), "source_group": f"group-{i}", "match": "prefix"}
        for i, source in enumerate(sources)
    ]), encoding="utf-8")
    request = resolve_training_cli([
        f"--pretrained_model_name_or_path={tmp_path / 'synthetic.safetensors'}",
        f"--dataset_config={dataset}",
    ], preset_name=preset, execution_mode_name=mode, quant_repeats=quant_repeats)
    request = replace(request, te_quantized=(preset == "canonical-v2"))
    command = profile_command(
        request, run_dir=tmp_path / "run", source_map=source_map, name="local",
        protocol="v24-acceptance-local", range_muls=request.execution_mode.core_grid, max_images=4,
    )
    index = next(i for i, token in enumerate(command) if token.endswith("sdxl_dq_dataset_profile.py"))
    return setup_parser().parse_args(command[index + 1:])


@pytest.mark.parametrize("preset", ["canonical-v1", "canonical-v2"])
@pytest.mark.parametrize("quant_repeats", [None, 2])
def test_standard_entry_preserves_local_budget_and_requires_internal_qa(tmp_path, preset, quant_repeats):
    args = worker_args(tmp_path, preset=preset, quant_repeats=quant_repeats)
    _validate_and_isolate(args)
    payload = _preflight(args)
    assert args.dq_profile_shared_warmup
    assert args.dq_profile_prefix_gate_file is None
    assert args.dq_profile_shared_source_contract_sha256
    assert payload["v2"]["cost_components"]["shared_prefix_branch_steps"] == 64
    assert payload["v2"]["cost_components"]["warmup_steps"] == payload["dq_begin_step"]
    assert args.dq_profile_range_muls_resolved == (2.70, 3.15, 3.45, 3.75, 4.05)
    assert args.dq_profile_capture_steps == 128
    repeats = quant_repeats if quant_repeats is not None else (1 if preset == "canonical-v2" else 2)
    assert args.dq_profile_safety_quant_repeats_resolved == repeats
    assert payload["v2"]["cost_components"]["candidate_local_steps"] == 4 * 4 * 2 * 5 * repeats
    assert payload["v2"]["cost_components"]["no_quant_structural_steps"] == 4 * 4 * 3


@pytest.mark.parametrize("preset,repeats", [("canonical-v1", 2), ("canonical-v2", 1)])
def test_worker_default_rounding_count_matches_public_preset(tmp_path, preset, repeats):
    args = worker_args(tmp_path, preset=preset)
    args.dq_profile_quant_repeats = None
    _validate_and_isolate(args)
    assert args.dq_profile_safety_quant_repeats_resolved == repeats


@pytest.mark.parametrize("change", ["strict", "external_gate", "wrong_prefix", "formal", "no_shared"])
def test_shared_entry_cannot_bypass_the_prefix_gate(tmp_path, change):
    args = worker_args(tmp_path)
    if change == "strict":
        args.dq_profile_execution_mode = "strict"
    elif change == "external_gate":
        args.dq_profile_prefix_gate_file = "not-a-gate.json"
    elif change == "wrong_prefix":
        args.dq_profile_prefix_short_steps = 7
    elif change == "formal":
        args.dq_profile_protocol = "v24-acceptance-formal"
        args.dq_profile_data_diagnostics = "off"
        args.dq_profile_quant_repeats = None
    else:
        args.dq_profile_shared_warmup = False
    with pytest.raises(ValueError, match="shared warmup|requires --dq_profile_prefix_gate_file"):
        _validate_and_isolate(args)


class TinyNetwork(torch.nn.Module):
    # The runtime's isolation guard remains active in this synthetic test.
    __module__ = "dq_profile.copied_lora"

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.delta_q_enabled = False
        self.delta_q_range_mul = 2.7

    def set_delta_fake_quant(self, step, mode, **kwargs):
        self.delta_q_range_mul = kwargs["range_mul"]

    def set_delta_quant_enabled(self, enabled):
        self.delta_q_enabled = enabled

    def get_trainable_params(self):
        return list(self.parameters())

    def forward(self, value):
        output = self.linear(self.dropout(value))
        if self.delta_q_enabled:
            output = output + (torch.round(output * 16) / 16 - output).detach()
        return output


class TinyAccelerator:
    num_processes = 1
    sync_gradients = True
    optimizer_step_was_skipped = False
    device = torch.device("cpu")
    scaler = None

    def __init__(self):
        self.ended = 0

    def unwrap_model(self, network):
        return network

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, norm):
        return torch.nn.utils.clip_grad_norm_(params, norm)

    def end_training(self):
        self.ended += 1


class TinyTrainer:
    def __init__(self, network):
        self.network = network
        self._dataset_diagnostics = SimpleNamespace(rows=["warmup"], initial={"preserved": True})
        self.calls = 0

    def _set_network_multiplier_from_batch(self, *args):
        pass

    def _apply_te_freeze_if_ready(self, *args):
        pass

    def _get_text_conds_for_batch(self, *args, **kwargs):
        return []

    def _compute_batch_loss(self, args, accelerator, batch, noise_scheduler, unet, text_conditions,
                            noisy_latents, timesteps, target, huber_c, weight_dtype, **kwargs):
        assert self._dataset_diagnostics is None  # Prefix must not contaminate A->B/observer data.
        self.calls += 1
        unet.counter.add_(1)
        return (self.network(noisy_latents) - target).square().mean()

    def all_reduce_network(self, *args):
        pass

    def _apply_te_lr_after_if_ready(self, *args):
        pass


def boundary_fixture(tmp_path, monkeypatch):
    args = worker_args(tmp_path)
    args.dq_profile_dropout_on = True
    _validate_and_isolate(args)
    _preflight(args)
    args.max_train_steps = 100
    network = TinyNetwork()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)
    # Give the shared boundary nonempty optimizer history.
    network(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    network.eval()
    accelerator = TinyAccelerator()
    unet = torch.nn.Identity()
    unet.register_buffer("counter", torch.tensor(5))
    unet.eval()
    encoder = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    trainer = TinyTrainer(network)
    runtime = DiagnosticProfileRuntime(args=args, trainer=trainer)
    runtime.artifacts.initialize()
    snapshot = TrainingSnapshot.capture(
        network=network, optimizer=optimizer, scheduler=scheduler, scaler=None,
        trainer=trainer, guardian=None, global_step=10, epoch=2, data_step=0,
    )
    sequence = ReplaySequence()
    for i in range(128):
        tensor = torch.tensor([[0.1, 0.2]])
        sequence.append(ReplayBatch(
            index=i, source_epoch=2, source_step=i, global_step=10+i,
            batch={"image_keys": [f"synthetic-{i}"]}, latents=tensor,
            noise=tensor, noisy_latents=tensor, target=tensor, timesteps=torch.tensor([50]), model_seed=i,
        ))
    sequence.seal()
    noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def synthetic_noise(args, scheduler, latents, **kwargs):
        noise = torch.randn_like(latents)
        return noise, latents + noise, torch.tensor([50]), None

    monkeypatch.setattr("dq_profile.trainer_runtime.train_util.get_noise_noisy_latents_and_timesteps", synthetic_noise)
    context = dict(
        accelerator=accelerator, network=network, optimizer=optimizer, lr_scheduler=scheduler,
        grad_norm_guardian=None, global_step=10, epoch=2, data_step=0,
        first_batch=sequence[0].batch, epoch_iterator=iter(()), train_dataloader=[],
        current_epoch=SimpleNamespace(value=3), current_step=SimpleNamespace(value=10), num_train_epochs=40,
        vae=None, vae_dtype=torch.float32, weight_dtype=torch.float32, noise_scheduler=noise_scheduler,
        unet=unet, text_encoders=[encoder], tokenizers=[], train_unet=True, train_text_encoder=True,
        training_model=network, on_step_start=lambda *args: None, dq_delta_begin_step=10,
    )
    return runtime, snapshot, sequence, context


def test_real_cpu_prefix_reuses_boundary_without_dataloader_or_second_warmup(tmp_path, monkeypatch):
    runtime, snapshot, sequence, context = boundary_fixture(tmp_path, monkeypatch)
    observer = runtime.trainer._dataset_diagnostics
    network = context["network"]
    rng_before = fingerprint_tree(capture_rng_state())
    state_before = snapshot_fingerprints(snapshot)

    def local_gradients():
        # Compare fixed-input OFF/ON probes under multiple masks; a clone keeps
        # this independent check out of the real shared boundary/observer.
        values = []
        with torch.random.fork_rng(devices=[]):
            model = copy.deepcopy(network)
            for enabled in (False, True):
                model.train(enabled)
                for seed in (11, 23, 37):
                    torch.manual_seed(seed)
                    model.zero_grad(set_to_none=True)
                    model(torch.ones(1, 2)).square().sum().backward()
                    values.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        return values

    gradients_before = local_gradients()

    def forbidden(*args, **kwargs):
        pytest.fail("shared Prefix must not capture/materialize the data again")

    monkeypatch.setattr(DiagnosticProfileRuntime, "_capture_batches", forbidden)
    monkeypatch.setattr(DiagnosticProfileRuntime, "_materialize_replay", forbidden)
    record = run_shared_prefix(runtime=runtime, snapshot=snapshot, sequence=sequence, boundary_context=context)
    assert record["passed"]
    assert record["snapshot_fingerprints"] == record["restored_snapshot_fingerprints"] == state_before
    assert not record["independent_warmup_parity_performed"]
    assert runtime.trainer.calls == 64
    assert context["accelerator"].ended == 0
    assert context["unet"].counter.item() == 5
    assert not context["unet"].training and not network.training
    assert network.delta_q_enabled is False and network.delta_q_range_mul == 2.7
    assert runtime.trainer._dataset_diagnostics is observer
    assert observer.rows == ["warmup"]
    assert fingerprint_tree(capture_rng_state()) == rng_before
    assert len(sequence) == 128
    prefix_dir = runtime.artifacts.root / "shared_prefix"
    gate = json.loads((prefix_dir / "calibration_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] and gate["prefix_contract"]["branch_updates"] == 64
    manifest = json.loads((prefix_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_fingerprints"] == state_before
    assert manifest["branch_batches"] == sequence.manifest()[:16]
    assert runtime.args.dq_profile_dropout_on is True
    assert runtime.args.dq_profile_standard_version == 2
    assert all(torch.equal(left, right) for left, right in zip(gradients_before, local_gradients()))


@pytest.mark.parametrize("failure", ["exception", "bad_gate", "changed_source", "mutated_replay"])
def test_prefix_failure_restores_state_and_blocks_local(tmp_path, monkeypatch, failure):
    runtime, snapshot, sequence, context = boundary_fixture(tmp_path, monkeypatch)
    observer = runtime.trainer._dataset_diagnostics
    rng_before = fingerprint_tree(capture_rng_state())
    original = DiagnosticProfileRuntime.run_from_boundary

    def failing_prefix(self, **kwargs):
        if failure == "exception":
            with torch.no_grad():
                context["network"].linear.weight.add_(9)
            context["unet"].train()
            context["unet"].counter.add_(2)
            random.random()
            np.random.rand()
            torch.rand(3)
            raise RuntimeError("synthetic Prefix failure")
        result = original(self, **kwargs)
        if failure == "bad_gate":
            path = self.artifacts.root / "calibration_gate.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["passed"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
        elif failure == "changed_source":
            runtime.args.dq_profile_shared_source_contract_sha256 = "changed"
        else:
            kwargs["_prepared_boundary"][1][0].noise.add_(1)
        return result

    monkeypatch.setattr(DiagnosticProfileRuntime, "run_from_boundary", failing_prefix)
    with pytest.raises(RuntimeError, match="Prefix|source contract|replay changed"):
        run_shared_prefix(runtime=runtime, snapshot=snapshot, sequence=sequence, boundary_context=context)
    assert not (runtime.artifacts.root / "shared_warmup.json").exists()
    assert runtime.trainer._dataset_diagnostics is observer
    assert fingerprint_tree(context["network"].state_dict()) == fingerprint_tree(snapshot.network_state)
    assert context["unet"].counter.item() == 5 and not context["unet"].training
    assert fingerprint_tree(capture_rng_state()) == rng_before
    assert context["accelerator"].ended == 0
    torch.testing.assert_close(sequence[0].noise, torch.tensor([[0.1, 0.2]]))
