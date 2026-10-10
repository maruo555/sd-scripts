"""Offline standard reports built exclusively from measured component observations."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from dq_profile.spatial import STANDARD_VERSION, VIEWS, LEAVES, metrics_from_moments, balanced_mean, summarize
from dq_profile.spatial_report_data import SHARED_DATA_VERSION, loss_contract, public_loss_values, local_image_context


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()] if Path(path).exists() else []


def mean(values):
    return sum(values) / len(values) if values and all(v is not None and math.isfinite(v) for v in values) else None


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def build_data(contract, rows, *, inventory=(), references=(), thumbnails=None, iterations=2000):
    if contract["version"] != STANDARD_VERSION or not rows:
        raise ValueError("Missing or unsupported standard diagnostic observations")
    candidates = contract["conditions"]
    ids = [c["id"] for c in candidates]
    regimes = ["off", "on"] if contract["dropout_on"] else ["off"]
    if len(ids) != len(set(ids)) or len(ids) > 8:
        raise ValueError("Duplicate or excessive standard diagnostic conditions")
    grouped = defaultdict(list)
    identity_sets = {}
    for row in rows:
        if row["candidate"] not in ids or row["regime"] not in regimes:
            raise ValueError("Unregistered diagnostic condition or dropout regime")
        grouped[row["regime"], row["candidate"]].append(row)
    for regime in regimes:
        for cid in ids:
            values = grouped[regime, cid]
            keys = {(r["image_key"], r["image_index"], r["timestep_bin"], r["noise_replica"], r["quant_repeat"]) for r in values}
            if not values or len(keys) != len(values):
                raise ValueError("Missing or duplicate diagnostic observations")
            identity_sets[regime, cid] = keys
    expected = identity_sets["off", ids[0]]
    if any(keys != expected for keys in identity_sets.values()):
        raise ValueError("Conditions/regimes do not share the same measured cells")
    images = sorted({r["image_key"] for r in rows})
    expected_per_image = contract["bins"] * contract["candidate_noises"] * contract["quant_repeats"]
    if len(expected) != len(images) * expected_per_image:
        raise ValueError("Incomplete standard diagnostic cell coverage")
    for image in images:
        cells = {(r[2], r[3], r[4]) for r in expected if r[0] == image}
        required = {(b, n, q) for b in range(contract["bins"]) for n in range(contract["candidate_noises"])
                    for q in range(contract["quant_repeats"])}
        if cells != required:
            raise ValueError("Invalid per-image diagnostic cell coverage")
    # Verify reference identity inside each regime independently; dropout may change it across regimes.
    paired = {}
    for row in rows:
        key = row["regime"], row["image_key"], row["timestep_bin"], row["noise_replica"]
        identity = tuple(row[k] for k in ("input_id", "noise_digest", "dropout_mask_digest", "reference_hash", "reference_loss")) + (row.get("reference_mse"),)
        if paired.setdefault(key, identity) != identity:
            raise ValueError("Candidate observations have mismatched reference inputs or gradients")
    image_ids = {key: f"I{i+1:03}" for i, key in enumerate(images)}
    source_ids = {key: f"S{i+1:03}" for i, key in enumerate(sorted({r["source_group"] for r in rows}))}
    source_by_image = {}
    for row in rows:
        if source_by_image.setdefault(row["image_key"], row["source_group"]) != row["source_group"]:
            raise ValueError("Image source grouping changed between conditions")
    loss_spec = loss_contract(rows)
    has_mse = loss_spec["selected_kind"] == "raw_mse"
    loss_fields = ("reference_mse", "quantized_mse") if has_mse else ("reference_loss", "quantized_loss")
    public_rows = []
    by_image = defaultdict(list)
    summaries = {}
    for regime in regimes:
        scores = {view: {} for view in VIEWS}
        contributions = {}
        for cid in ids:
            transformed = []
            for row in grouped[regime, cid]:
                metrics = metrics_from_moments(row["components"])
                entry = {**row, "metrics": metrics}
                by_image[regime, row["image_key"], row["timestep_bin"], cid].append(entry)
                transformed.append(entry)
                public_rows.append({"sample": image_ids[row["image_key"]], "source": source_ids[row["source_group"]],
                    "candidate": cid, "regime": regime, "bin": row["timestep_bin"], "noise": row["noise_replica"],
                    "repeat": row["quant_repeat"], "components": row["components"], "metrics": metrics,
                    "pairing": {key: row[key] for key in ("input_id", "noise_digest", "dropout_mask_digest", "reference_hash", "quantized_hash")},
                    **public_loss_values(row, loss_spec["selected_kind"])})
            for view in VIEWS:
                scores[view][cid] = summarize([{ "source_group": r["source_group"], "timestep_bin": r["timestep_bin"],
                    "value": r["metrics"][view]["distance"], "parallel": r["metrics"][view]["parallel"]} for r in transformed],
                    bins=contract["bins"], iterations=iterations if view == "all" else 0)
            contributions[cid] = {part: balanced_mean([{ "source_group": r["source_group"],
                "value": r["components"][part]["difference_sq"] / r["metrics"]["all"]["reference_sq"]
                         if r["metrics"]["all"]["reference_sq"] > 1e-30 else None} for r in transformed], "value") for part in LEAVES}
        summaries[regime] = {"scores": scores, "contributions": contributions, "state_sha256": contract["state_sha256"]}
    inventory_by_path = defaultdict(list)
    for item in inventory:
        inventory_by_path[str(Path(item["path"]).resolve()).casefold()].append(item)
    reference_index = defaultdict(list)
    for row in references:
        reference_index[row["sample_id"], row["bin"], row["snapshot"]].append(row)
    samples = []
    for image_path in images:
        path = Path(image_path)
        matches = inventory_by_path.get(str(path.resolve()).casefold(), [])
        item = matches[0] if len(matches) == 1 else {}
        cells = {}
        for regime in regimes:
            cells[regime] = []
            for bin_index in range(contract["bins"]):
                pre = reference_index[item.get("sample_id"), bin_index, "pre"]
                post = reference_index[item.get("sample_id"), bin_index, "post"]
                pre_ids = {r["eval_input_id"] for r in pre}
                paired = bool(pre_ids) and len(pre_ids) == len(pre) == len(post) and pre_ids == {r["eval_input_id"] for r in post}
                q = {}
                for cid in ids:
                    measured = by_image[regime, image_path, bin_index, cid]
                    ref = mean([r[loss_fields[0]] for r in measured])
                    quant = mean([r[loss_fields[1]] for r in measured])
                    q[cid] = {"d": {view: mean([r["metrics"][view]["distance"] for r in measured]) for view in VIEWS},
                        "parallel_by_part": {view: mean([r["metrics"][view]["parallel"] for r in measured]) for view in VIEWS},
                        "parallel": mean([r["metrics"]["all"]["parallel"] for r in measured]),
                        "reference_loss": ref, "quantized_loss": quant,
                        "delta": mean([r[loss_fields[1]] - r[loss_fields[0]] for r in measured]), "observations": len(measured)}
                cells[regime].append({"bin": bin_index, "pre": mean([r["raw_mse"] for r in pre]) if paired else None,
                    "post": mean([r["raw_mse"] for r in post]) if paired else None, "q": q})
        samples.append({"id": image_ids[image_path], "name": item.get("name", path.name),
            "folder": item.get("folder_name", path.parent.name), "source": source_ids[source_by_image[image_path]],
            "tags": item.get("tags", []), "bins": cells,
            **local_image_context(path, matches, thumbnails or {})})
    pre = mean([mean([b["pre"] for b in s["bins"]["off"]]) for s in samples])
    post = mean([mean([b["post"] for b in s["bins"]["off"]]) for s in samples])
    warmup = {"loss_kind": "raw_mse", "available": pre is not None and post is not None and pre > 0, "before": pre, "after": post,
              "reduction_pct": 100 * (pre-post)/pre if pre is not None and pre > 0 and post is not None else None,
              "reason": "対応する学習前後のraw MSEが未記録です。warmup診断を指定すると計測されます。"}
    dataset = {"id": "dataset", "label": "このデータセット", "short_label": "診断結果", "views": VIEWS,
        "candidates": candidates, "regimes": summaries, "samples": samples, "body_mul": contract["body_mul"],
        "body_note": "一律5点・dropout OFFの既存Body代表選出規則。追加配分や部位選択で選び直しません。",
        "quant_scope": "UNet＋TE1＋TE2", "scope_note": "同じsnapshot・入力・noise・timestepで配分を比較。dropout ONは追加確認です。",
        "loss_label": "raw MSE" if has_mse else "学習目的関数のloss",
        "loss_contract": loss_spec,
        "loss_note": "予測誤差はモデル全体の値です。部位ごとへ分解しません。" if has_mse else
            ("raw MSEが一部未記録のため、全条件を学習目的関数のlossに統一して表示します。" if loss_spec["selection_reason"] == "raw_mse_partially_missing" else
             "raw MSE未記録のため学習目的関数のlossを表示します。") + "raw MSEとは異なります。",
        "warmup": warmup, "skipped": contract["skipped"], "meta": {"images": len(images), "groups": len(source_ids), "timestep_bins": contract["bins"],
            "noise_replicas": contract["candidate_noises"], "quant_repeats": contract["quant_repeats"], "step": contract["step"],
            "total_groups": contract["source_map"].get("source_group_count_total", len(source_ids)),
            "quantization": contract.get("quantization"), "training_dropout": contract.get("training_dropout")}}
    return {"version": STANDARD_VERSION, "datasets": [dataset]}, public_rows


def write_standard_report(profile_dir, output_dir, *, iterations=2000):
    from plotly.offline import get_plotlyjs
    profile_dir, output_dir = Path(profile_dir), Path(output_dir)
    contract = json.loads((profile_dir / "spatial_contract.json").read_text(encoding="utf-8"))
    observations = read_lines(profile_dir / "spatial_observations.jsonl")
    data_dir = profile_dir / "data_diagnostics"
    inventory = read_lines(data_dir / "inventory.jsonl")
    from dq_profile.diagnostic_report import report_thumbnails
    preview_dir = output_dir / "data_diagnostics"
    preview_dir.mkdir(parents=True, exist_ok=True)
    old_cache = data_dir / "thumbnails.json"
    if old_cache.is_file() and not (preview_dir / "thumbnails.json").exists():
        shutil.copyfile(old_cache, preview_dir / "thumbnails.json")
    measured_paths = {str(Path(row["image_key"]).resolve()).casefold() for row in observations}
    thumbnail_samples = [{**item, "measured": str(Path(item["path"]).resolve()).casefold() in measured_paths}
                         for item in inventory if item.get("image_id")]
    thumbnails = report_thumbnails(preview_dir, thumbnail_samples) if thumbnail_samples else {}
    data, public = build_data(contract, observations, inventory=inventory,
        references=read_lines(data_dir / "reference_probes.jsonl"), thumbnails=thumbnails, iterations=iterations)
    data["datasets"][0]["legacy_detail_url"] = "data_diagnostics/dataset_report.html" if (preview_dir / "dataset_report.html").is_file() else None
    practical = output_dir / "practical_report.json"
    if practical.exists():
        original_body = json.loads(practical.read_text(encoding="utf-8"))["datasets"][0]["body_representative_mul"]
        if original_body != contract["body_mul"]:
            raise ValueError("Standard Body reference disagrees with the independent uniform report")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name("spatial_assets")
    for name in ("beginner_report.html", "dataset_report.html", "report.js", "report.css"):
        shutil.copyfile(assets / name, output_dir / name)
    (output_dir / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (output_dir / "data.js").write_text("window.DQ_DATA=" + encoded + ";\n", encoding="utf-8")
    dataset = data["datasets"][0]
    measurement_contract = {key: contract.get(key) for key in ("state_sha256", "step", "protocol_seed",
        "probe_address_version", "quantization", "training_dropout", "dropout_on", "bins",
        "candidate_noises", "quant_repeats", "local_forward_backward_calls", "additional_reference_probes")}
    measurement_contract["source_map_sha256"] = hashlib.sha256(
        json.dumps(contract["source_map"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    fingerprint_path = output_dir / "protocol_fingerprint.json"
    measurement_contract["protocol_fingerprint_sha256"] = (
        hashlib.sha256(fingerprint_path.read_bytes()).hexdigest() if fingerprint_path.is_file() else None)
    write_json(output_dir / "ai_summary.json", {"schema_version": SHARED_DATA_VERSION, "version": STANDARD_VERSION, "conditions": contract["conditions"],
        "body_mul": contract["body_mul"], "skipped": contract["skipped"], "measurement": dataset["meta"],
        "measurement_contract": measurement_contract,
        "loss_contract": dataset["loss_contract"],
        "quant_scope": dataset["quant_scope"], "metrics": dataset["regimes"], "warmup": dataset["warmup"],
        "weighting": {"gradient": "source_equal_then_observation_equal", "warmup_and_image_loss": "image_equal"},
        "not_image_quality": True})
    write_json(output_dir / "observations.json", {"schema_version": SHARED_DATA_VERSION, "version": STANDARD_VERSION,
        "loss_contract": dataset["loss_contract"], "rows": public})
    if (output_dir / "report.html").exists() and not (output_dir / "uniform_report.html").exists():
        shutil.copyfile(output_dir / "report.html", output_dir / "uniform_report.html")
    shutil.copyfile(output_dir / "beginner_report.html", output_dir / "report.html")
    outputs = ["report.html", "beginner_report.html", "dataset_report.html", "report.js", "report.css", "data.js", "plotly.min.js", "ai_summary.json", "observations.json"]
    from dq_profile.report_publication import finalize_report_manifests
    finalize_report_manifests(output_dir, standard_version=STANDARD_VERSION,
        inputs=[profile_dir / name for name in ("spatial_contract.json", "spatial_observations.jsonl")], outputs=outputs)
    return outputs + ["standard_report_manifest.json"]
