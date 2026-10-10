"""Offline report regression coverage using synthetic observations and images."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from dq_profile.spatial import LEAVES, STANDARD_VERSION, build_conditions
from dq_profile.spatial_report import build_data, write_standard_report
from dq_profile.report_publication import file_record, finalize_report_manifests


def synthetic_observations(root):
    conditions, skipped = build_conditions(2.7)
    contract = {"version": STANDARD_VERSION, "conditions": conditions, "skipped": skipped,
                "dropout_on": False, "body_mul": 2.7, "bins": 4, "candidate_noises": 1,
                "quant_repeats": 1, "state_sha256": "a" * 64, "step": 10,
                "source_map": {"source_group_count_total": 2}}
    rows = []
    for ci, condition in enumerate(conditions):
        for image in range(2):
            for bin_index in range(4):
                delta = .02 * (ci + bin_index + 1)
                raw = .4 + .1 * image + .01 * bin_index
                components = {part: {"reference_sq": 2., "quantized_sq": 2 * (1 + delta)**2,
                    "dot": 2 * (1 + delta), "difference_sq": 2 * delta**2} for part in LEAVES}
                rows.append({"candidate": condition["id"], "regime": "off",
                    "image_key": str((root / f"image_{image}.png").resolve()), "image_index": image,
                    "source_group": f"group_{image}", "timestep_bin": bin_index,
                    "noise_replica": 0, "quant_repeat": 0, "components": components,
                    "input_id": f"input_{image}_{bin_index}", "noise_digest": "b" * 64,
                    "dropout_mask_digest": "c" * 64, "reference_hash": "d" * 64,
                    "quantized_hash": "e" * 64, "reference_loss": raw * 2,
                    "quantized_loss": (raw + delta) * 2, "reference_mse": raw,
                    "quantized_mse": raw + delta})
    return contract, rows


def write_fixture(root, contract, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / "spatial_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (root / "spatial_observations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.mark.parametrize("coverage, kind, reason", [
    ("all", "raw_mse", "all_raw_mse_available"),
    ("none", "objective_loss", "raw_mse_not_recorded"),
    ("partial", "objective_loss", "raw_mse_partially_missing"),
])
def test_shared_metric_types_are_unambiguous(tmp_path, coverage, kind, reason):
    contract, rows = synthetic_observations(tmp_path)
    if coverage == "none":
        for row in rows:
            row.pop("reference_mse")
            row.pop("quantized_mse")
    elif coverage == "partial":
        rows[0]["quantized_mse"] = None
    profile, output = tmp_path / "profile", tmp_path / "output"
    write_fixture(profile, contract, rows)
    write_standard_report(profile, output, iterations=0)
    summary = json.loads((output / "ai_summary.json").read_text(encoding="utf-8"))
    observations = json.loads((output / "observations.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == observations["schema_version"] == "spatial-shared-v2"
    spec = summary["loss_contract"]
    assert spec == observations["loss_contract"]
    assert spec["selected_kind"] == kind and spec["selection_reason"] == reason
    for source, exported in zip(rows, observations["rows"]):
        assert exported["loss_kind"] == kind
        assert exported["objective_loss"] == {"reference": source["reference_loss"], "quantized": source["quantized_loss"]}
        assert exported["raw_mse"]["reference"] == source.get("reference_mse")
        assert exported["raw_mse"]["quantized"] == source.get("quantized_mse")
        assert exported["reference_loss"] == exported[kind]["reference"]
        assert exported["quantized_loss"] == exported[kind]["quantized"]
    expected_missing = {"all": None, "none": "missing_reference_and_quantized", "partial": "missing_quantized"}
    assert observations["rows"][0]["raw_mse"]["missing_reason"] == expected_missing[coverage]
    assert summary["warmup"]["loss_kind"] == "raw_mse"


def test_nonfinite_metrics_fail_instead_of_changing_metric_kind(tmp_path):
    contract, rows = synthetic_observations(tmp_path)
    rows[0]["quantized_mse"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        build_data(contract, rows, iterations=0)


def assert_artifact_records(records):
    for record in records.values():
        target = Path(record["path"])
        assert target.is_file()
        assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_manifests_describe_final_files_not_overwritten_uniform_html(tmp_path):
    profile, stage, output = (tmp_path / name for name in ("profile", "analysis", "published"))
    contract, rows = synthetic_observations(tmp_path)
    write_fixture(profile, contract, rows)
    stage.mkdir(); output.mkdir()
    reports = {}
    for key, name in (("primary", "report.html"), ("beginner", "beginner_report.html"),
                      ("technical", "technical_report.html"), ("contract", "report_contract.json")):
        (stage / name).write_text("uniform intermediate", encoding="utf-8")
        reports[key] = file_record(stage / name)
        (output / name).write_bytes((stage / name).read_bytes())
    original = {"inputs": {"observations": file_record(profile / "spatial_observations.jsonl")}, "reports": reports}
    for directory in (stage, output):
        (directory / "analysis_manifest.json").write_text(json.dumps(original), encoding="utf-8")
    source_manifest = (stage / "analysis_manifest.json").read_bytes()
    write_standard_report(profile, output, iterations=0)
    published = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    standard = json.loads((output / "standard_report_manifest.json").read_text(encoding="utf-8"))
    for records in (published["inputs"], published["reports"], standard["inputs"], standard["outputs"]):
        assert_artifact_records(records)
    assert published["inputs"] == original["inputs"]
    assert published["reports"]["primary"]["sha256"] != original["reports"]["primary"]["sha256"]
    assert (stage / "analysis_manifest.json").read_bytes() == source_manifest
    assert (output / "uniform_report.html").read_text(encoding="utf-8") == "uniform intermediate"
    # Rebuilding the new report must preserve the original uniform copy.
    write_standard_report(profile, output, iterations=0)
    assert (output / "uniform_report.html").read_text(encoding="utf-8") == "uniform intermediate"


def test_publication_fails_closed_for_missing_output_and_preserves_manifest(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("old", encoding="utf-8")
    manifest = tmp_path / "analysis_manifest.json"
    manifest.write_text(json.dumps({"reports": {"primary": file_record(report)}}), encoding="utf-8")
    before = manifest.read_bytes()
    report.write_text("new", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        finalize_report_manifests(tmp_path, standard_version=STANDARD_VERSION, outputs=["missing.html"])
    assert manifest.read_bytes() == before
    assert not (tmp_path / "standard_report_manifest.json").exists()
    finalize_report_manifests(tmp_path)
    assert_artifact_records(json.loads(manifest.read_text(encoding="utf-8"))["reports"])


def test_image_context_and_cache_survive_source_loss_without_shared_private_data(tmp_path):
    contract, rows = synthetic_observations(tmp_path)
    profile, output = tmp_path / "profile", tmp_path / "published"
    write_fixture(profile, contract, rows)
    folder = profile / "data_diagnostics"
    folder.mkdir()
    caption = '<script>window.invalidCaptionExecuted=true</script> synthetic caption'
    inventory = []
    for i in range(2):
        path = tmp_path / f"image_{i}.png"
        Image.new("RGB", (32, 32), (20 + i * 70, 40, 80)).save(path)
        inventory.append({"image_id": f"img{i}", "sample_id": f"sample{i}", "path": str(path),
            "caption": caption, "dataset_index": 0, "subset_index": i, "num_repeats": 3,
            "presented_count": 9, "updated_count": 8, "skipped_count": 1})
    (folder / "inventory.jsonl").write_text("\n".join(map(json.dumps, inventory)), encoding="utf-8")
    write_standard_report(profile, output, iterations=0)
    first = json.loads((output / "data.js").read_text(encoding="utf-8").removeprefix("window.DQ_DATA=").strip().removesuffix(";"))
    for i in range(2):
        (tmp_path / f"image_{i}.png").unlink()
    write_standard_report(profile, output, iterations=0)
    encoded = (output / "data.js").read_text(encoding="utf-8")
    second = json.loads(encoded.removeprefix("window.DQ_DATA=").strip().removesuffix(";"))
    for a, b in zip(first["datasets"][0]["samples"], second["datasets"][0]["samples"]):
        assert a["image_uri"] == b["image_uri"]
        assert b["image_uri"].startswith("data:image/jpeg;base64,")
        assert b["preview_kind"] == "cached_thumbnail"
        assert b["contexts"][0]["caption"] == caption
        assert b["contexts"][0]["skipped_count"] == 1
    assert "<script>" not in encoded
    for name in ("ai_summary.json", "observations.json"):
        text = (output / name).read_text(encoding="utf-8")
        for private in (caption, str(tmp_path), "image_0.png", "data:image/jpeg;base64,"):
            assert private not in text


def test_ambiguous_training_contexts_do_not_invent_warmup_pair(tmp_path):
    contract, rows = synthetic_observations(tmp_path)
    inventory = [{"image_id": "same", "sample_id": str(i), "path": rows[0]["image_key"],
                  "caption": f"caption {i}", "subset_index": i} for i in range(2)]
    references = [{"sample_id": str(i), "bin": b, "snapshot": snap, "eval_input_id": f"{i}-{b}", "raw_mse": mse}
                  for i in range(2) for b in range(4) for snap, mse in (("pre", 2.), ("post", 1.))]
    data, _ = build_data(contract, rows, inventory=inventory, references=references, iterations=0)
    sample = data["datasets"][0]["samples"][0]
    assert sample["inventory_match"] == "multiple_contexts" and len(sample["contexts"]) == 2
    assert all(row["pre"] is None and row["post"] is None for row in sample["bins"]["off"])
