"""Failed standard probes retain evidence without changing measurement policy."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from dq_profile.metrics import ExactGradient
from dq_profile.protocol import CandidateDefinition
from dq_profile.spatial import UNIFORM_MULS, VIEWS, FAILURE_POLICY
from dq_profile.spatial_report import build_data
from dq_profile.spatial_runtime import SpatialDiagnosticRuntime
from dq_profile.trainer_runtime import DiagnosticProfileRuntime
from library.dq_mul_policy import MulPolicy


@pytest.mark.parametrize("failure", ["loss", "gradient", "mse", "safety"])
@pytest.mark.parametrize("quantized", [False, True])
def test_invalid_probe_reports_context_and_preserves_failure(monkeypatch, tmp_path, failure, quantized):
    args = NS(dq_profile_run_dir=str(tmp_path))
    trainer = NS(_dataset_diagnostics=NS(raw_mse=.3))
    runtime = SpatialDiagnosticRuntime(args=args, trainer=trainer)
    runtime.spatial_collect = True
    runtime.spatial_regime = "on"
    runtime.quant_context.grad_scale = 65536.
    policy = MulPolicy.from_dict({"base_mul": 2.7, "components": {"te1": 4.05, "te2": 4.05},
        "group_overrides": [{"group_id": "unet.attn2", "range_mul": 4.05}]})
    runtime.spatial_policies["mixed_test"] = policy
    candidate = CandidateDefinition("mixed_test" if quantized else "no_quant", quantized, None, None, 2.7 if quantized else None, False)
    row = {"loss": .2, "forced_safety_abort": False, "noise_digest": "noise", "gradient_hash": "gradient",
        "dropout_mask_digest": "mask", "replay_digest": "replay", "timestep_digest": "time", "rng_digest_before": "rng"}
    gradient = ExactGradient({"lora_te1_test.weight": torch.tensor([1., 2.])}, 5.)
    runtime.spatial_source_map = NS(resolve=lambda key: "synthetic_group")
    runtime.spatial_reference = ("tail:3:2:1:synthetic.png", dict(row), gradient, .3)
    if failure == "loss":
        row["loss"] = float("nan")
    elif failure == "gradient":
        gradient = ExactGradient({"lora_te1_test.weight": torch.tensor([float("inf"), float("nan")])}, float("nan"))
    elif failure == "mse":
        trainer._dataset_diagnostics.raw_mse = float("inf")
    else:
        row.update(forced_safety_abort=True, invalid_reason="synthetic_safety_failure")
    monkeypatch.setattr(DiagnosticProfileRuntime, "_run_pass", lambda *a, **kw: (row, gradient, []))
    context = dict(phase="v2_tail_probe", candidate=candidate, range_mul=candidate.initial_range_mul,
        probe_or_step="tail:3:2:1:synthetic.png", repeat=1,
        replay=NS(image_keys=["synthetic.png"], timesteps=torch.tensor([625]), digest="input"))
    if quantized:
        result = runtime._run_pass(**context)
        assert result[0]["forced_safety_abort"]
        assert len(runtime.spatial_rows) == 1
        assert runtime.spatial_rows[0]["components"] is None
        assert runtime.spatial_rows[0]["measurement_status"] == "invalid"
    else:
        with pytest.raises(FloatingPointError, match="dropout=on, image_index=3, timestep_bin=2"):
            runtime._run_pass(**context)
        assert not runtime.spatial_rows
    text = (tmp_path / "spatial_failure.json").read_text(encoding="utf-8")
    details = json.loads(text, parse_constant=lambda value: pytest.fail(f"Non-JSON value: {value}"))
    assert details["candidate"] == candidate.name and details["quantized"] == quantized
    assert details["image_key"] == "synthetic.png" and details["timesteps"] == [625]
    assert details["noise_replica"] == details["quant_repeat"] == 1
    assert details["grad_scale"] == 65536. and details["not_for_report"]
    assert details["policy"] == (policy.to_dict() if quantized else None)
    assert not details["automatic_retry"]
    assert details["action"] == ("exclude_condition" if quantized else "stop_invalid_reference")
    if failure == "gradient":
        assert details["gradient_norm"] is None
        assert details["nonfinite_gradient_tensors"] == [{"parameter": "lora_te1_test.weight",
            "elements": 2, "nan_elements": 1, "inf_elements": 1}]
    elif failure == "loss":
        assert details["loss"] is None and details["nonfinite_values"]["loss"] == "nan"
    # A failure to save diagnostics must not replace the original numerical error.
    def deny_write(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(runtime.artifacts, "write_json", deny_write)
    runtime.spatial_failure_examples.clear()
    with pytest.raises(OSError if quantized else FloatingPointError):
        runtime._run_pass(**context)


@pytest.mark.parametrize("bad_regime", ["off", "on"])
def test_one_failed_spatial_probe_excludes_only_that_condition_and_regime(monkeypatch, tmp_path, bad_regime):
    from test_dq_spatial_standard import run_cpu_runtime

    original = DiagnosticProfileRuntime._run_pass
    seen = []
    injected = []
    def fail_first_extra(self, **kwargs):
        result = original(self, **kwargs)
        seen.append(kwargs["candidate"].name)
        if kwargs["candidate"].name == "attn2_te_reference" and self.spatial_regime == bad_regime and not injected:
            injected.append(True)
            row, gradient, shadows = result
            return row, ExactGradient(gradient.values, float("inf")), shadows
        return result
    monkeypatch.setattr(DiagnosticProfileRuntime, "_run_pass", fail_first_extra)
    runtime, contract = run_cpu_runtime(tmp_path)
    cid = "attn2_te_reference"
    other = "on" if bad_regime == "off" else "off"
    assert contract["failure_policy"] == FAILURE_POLICY
    assert contract["condition_status"][bad_regime][cid] == {"status": "unmeasurable", "observations": 128,
        "invalid_probe_count": 1, "reasons": ["nonfinite_gradient_norm"]}
    assert contract["condition_status"][other][cid]["status"] == "available"
    assert seen.count(cid) == 256  # Complete planned cells; no retry or altered sample population.
    assert not (tmp_path / "spatial_observations.partial.jsonl").exists()
    failure = json.loads((tmp_path / "spatial_failure.json").read_text(encoding="utf-8"))
    assert failure["regime"] == bad_regime and failure["candidate"] == cid
    assert not runtime.spatial_collect
    data, public = build_data(contract, runtime.spatial_rows, iterations=0)
    dataset = data["datasets"][0]
    for part in VIEWS:
        assert dataset["regimes"][bad_regime]["scores"][part][cid]["body"] is None
        assert dataset["regimes"][other]["scores"][part][cid]["body"] is not None
    # Healthy observations from the failed condition are retained only as audit evidence.
    excluded = [r for r in public if r["regime"] == bad_regime and r["candidate"] == cid]
    assert len(excluded) == 128 and not any(r["eligible_for_comparison"] for r in excluded)
    assert sum(r["measurement_status"] == "invalid" for r in excluded) == 1
    for sample in dataset["samples"]:
        assert all(b["q"][cid]["delta"] is None and b["q"][cid]["observations"] == 0 for b in sample["bins"][bad_regime])


@pytest.mark.parametrize("failure", ["loss", "gradient", "mse"])
def test_uniform_failure_cannot_become_body_reference(monkeypatch, tmp_path, failure):
    from test_dq_spatial_standard import run_cpu_runtime
    original = DiagnosticProfileRuntime._run_pass
    injected = []
    def inject(self, **kwargs):
        result = original(self, **kwargs)
        if failure == "mse":
            self.trainer._dataset_diagnostics.raw_mse = .3
        if kwargs["candidate"].name == "mul_2.700" and not injected:
            injected.append(True)
            row, gradient, shadows = result
            if failure == "gradient":
                gradient = ExactGradient({name: torch.full_like(value, float("nan")) for name, value in gradient.values.items()}, float("nan"))
            elif failure == "loss":
                row = {**row, "loss": float("nan")}
            else:
                self.trainer._dataset_diagnostics.raw_mse = float("nan")
            return row, gradient, shadows
        return result
    # Observer recording is irrelevant here; exercise the raw-MSE-only failure path too.
    if failure == "mse":
        init = SpatialDiagnosticRuntime.__init__
        def with_observer(self, **kwargs):
            init(self, **kwargs)
            self.trainer._dataset_diagnostics = NS(raw_mse=.3, record=lambda *a, **k: None,
                evaluate_initial=lambda *a, **k: None)
        monkeypatch.setattr(SpatialDiagnosticRuntime, "__init__", with_observer)
    monkeypatch.setattr(DiagnosticProfileRuntime, "_run_pass", inject)
    runtime, contract = run_cpu_runtime(tmp_path, dropout=False, uniform=True)
    assert contract["body_mul"] != 2.7
    assert contract["condition_status"]["off"]["mul_2.700"]["status"] == "unmeasurable"
    data, _ = build_data(contract, runtime.spatial_rows, iterations=0)
    assert data["datasets"][0]["regimes"]["off"]["scores"]["all"]["mul_2.700"]["body"] is None


def test_invalid_reference_still_stops_and_keeps_partial_evidence(monkeypatch, tmp_path):
    from test_dq_spatial_standard import run_cpu_runtime
    from dq_profile.v2_calibration import fingerprint_tree
    original = DiagnosticProfileRuntime._run_pass
    state = {}
    def inject(self, **kwargs):
        state.setdefault("before", fingerprint_tree(kwargs["network"].state_dict()))
        state.update(network=kwargs["network"], runtime=self)
        row, gradient, shadows = original(self, **kwargs)
        if kwargs["candidate"].name == "no_quant" and self.spatial_rows:
            row = {**row, "loss": float("nan")}
        return row, gradient, shadows
    monkeypatch.setattr(DiagnosticProfileRuntime, "_run_pass", inject)
    with pytest.raises(FloatingPointError, match="candidate=no_quant"):
        run_cpu_runtime(tmp_path)
    progress = json.loads((tmp_path / "spatial_progress.partial.json").read_text(encoding="utf-8"))
    assert progress["not_for_report"] and progress["completed_quantized_probes"] == 10
    assert not (tmp_path / "spatial_contract.json").exists()
    assert fingerprint_tree(state["network"].state_dict()) == state["before"]
    assert not state["runtime"].spatial_collect


@pytest.mark.parametrize("error", ["pairing", "topology", "all_conditions"])
def test_structural_or_total_failure_still_stops(monkeypatch, tmp_path, error):
    from test_dq_spatial_standard import run_cpu_runtime
    original = DiagnosticProfileRuntime._run_pass
    def inject(self, **kwargs):
        row, gradient, shadows = original(self, **kwargs)
        if kwargs["candidate"].quantized:
            row = {**row, "loss": float("nan")}
            if error == "pairing":
                row["noise_digest"] = "mismatched"
            elif error == "topology":
                gradient = ExactGradient({}, 0.)
        return row, gradient, shadows
    monkeypatch.setattr(DiagnosticProfileRuntime, "_run_pass", inject)
    with pytest.raises(ValueError, match={"pairing": "mismatch", "topology": "topology", "all_conditions": "no finite hard-safety candidate"}[error]):
        run_cpu_runtime(tmp_path, dropout=False, uniform=True)
    assert (tmp_path / "spatial_progress.partial.json").exists()
    assert not (tmp_path / "spatial_contract.json").exists()


def invalid_fixture(root):
    from copy import deepcopy
    from test_dq_spatial_report_contract import synthetic_observations
    from dq_profile.spatial import condition_statuses
    contract, rows = synthetic_observations(root)
    contract["dropout_on"] = True
    rows += [{**deepcopy(r), "regime": "on"} for r in rows]
    before, _ = build_data(contract, rows, iterations=0)
    bad = next(r for r in rows if r["candidate"] == "mul_3.450" and r["regime"] == "on")
    bad.update(components=None, quantized_loss=None, quantized_mse=None,
        measurement_status="invalid", invalid_reasons=["nonfinite_loss", "nonfinite_raw_mse"])
    contract["failure_policy"] = FAILURE_POLICY
    contract["condition_status"] = condition_statuses(contract["conditions"], ["off", "on"], rows)
    return contract, rows, before


def test_report_excludes_whole_condition_without_changing_healthy_scores(tmp_path):
    contract, rows, before = invalid_fixture(tmp_path)
    data, public = build_data(contract, rows, iterations=0)
    actual, expected = data["datasets"][0], before["datasets"][0]
    assert actual["warmup"] == expected["warmup"]
    for regime in ("off", "on"):
        for condition in contract["conditions"]:
            cid = condition["id"]
            for view in VIEWS:
                score = actual["regimes"][regime]["scores"][view][cid]
                if (regime, cid) == ("on", "mul_3.450"):
                    assert score["body"] is None and score["observations"] == 0
                else:
                    assert score == expected["regimes"][regime]["scores"][view][cid]
    assert sum(not r["eligible_for_comparison"] for r in public) == 8
    assert actual["loss_contract"]["selected_kind"] == "raw_mse"
    assert actual["loss_contract"]["raw_mse_coverage"]["observations"] == len(rows) - 8
    serialized = json.dumps(public, allow_nan=False)
    assert str(tmp_path) not in serialized and "image_0.png" not in serialized


@pytest.mark.parametrize("corruption,match", [
    ("missing_cell", "same measured cells"), ("pairing", "mismatched reference"),
    ("wrong_status", "status disagrees"), ("missing_policy", "explicit condition exclusion"),
    ("unmarked_nan", "Nonfinite"), ("unknown_reason", "supported numerical failure"),
    ("unmarked_component_nan", "Nonfinite"),
])
def test_invalid_probe_does_not_bypass_report_validation(tmp_path, corruption, match):
    contract, rows, _ = invalid_fixture(tmp_path)
    bad = next(r for r in rows if r.get("measurement_status") == "invalid")
    if corruption == "missing_cell":
        rows.remove(bad)
    elif corruption == "pairing":
        bad["reference_hash"] = "not-matched"
    elif corruption == "wrong_status":
        contract["condition_status"]["on"]["mul_3.450"]["invalid_probe_count"] = 2
    elif corruption == "missing_policy":
        contract.pop("failure_policy")
    elif corruption == "unmarked_nan":
        rows[0]["quantized_loss"] = float("nan")
    elif corruption == "unmarked_component_nan":
        rows[0]["components"]["te1"]["dot"] = float("nan")
    else:
        bad["invalid_reasons"] = ["unrelated_runtime_error"]
    with pytest.raises((ValueError, FloatingPointError), match=match):
        build_data(contract, rows, iterations=0)


def test_invalid_condition_publishes_explicit_ai_status_and_verified_html(tmp_path):
    from test_dq_spatial_report_contract import write_fixture, assert_artifact_records
    from dq_profile.spatial_report import write_standard_report
    contract, rows, _ = invalid_fixture(tmp_path)
    profile, output = tmp_path / "profile", tmp_path / "output"
    write_fixture(profile, contract, rows)
    legacy = output / "data_diagnostics"
    legacy.mkdir(parents=True)
    (legacy / "dataset_report.html").write_text("legacy partial aggregation", encoding="utf-8")
    write_standard_report(profile, output, iterations=0)
    summary = json.loads((output / "ai_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_policy"] == FAILURE_POLICY
    assert summary["metrics"]["on"]["condition_status"]["mul_3.450"]["invalid_probe_count"] == 1
    data = json.loads((output / "data.js").read_text(encoding="utf-8").removeprefix("window.DQ_DATA=").strip().removesuffix(";"))
    assert data["datasets"][0]["legacy_detail_url"] is None
    for page in ("beginner_report.html", "dataset_report.html"):
        text = (output / page).read_text(encoding="utf-8")
        assert 'id="condition-status"' in text and 'id="measurement-warnings"' in text
    manifest = json.loads((output / "standard_report_manifest.json").read_text(encoding="utf-8"))
    assert_artifact_records(manifest["outputs"])
