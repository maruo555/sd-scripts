"""Explicit metric types and local-only image context for spatial reports."""
from __future__ import annotations

import math
from pathlib import Path

SHARED_DATA_VERSION = "spatial-shared-v2"
CONTEXT_FIELDS = (
    "sample_id", "dataset_index", "subset_index", "subset_group", "resolution",
    "bucket_resolution", "num_repeats", "is_reg", "caption", "caption_separator",
    "class_tokens", "presented_count", "updated_count", "skipped_count",
    "first_seen_step", "last_seen_step",
)


def loss_contract(rows):
    coverage = {"observations": len(rows), "complete_pairs": 0,
                "missing_reference": 0, "missing_quantized": 0}
    for row in rows:
        for key in ("reference_loss", "quantized_loss", "reference_mse", "quantized_mse"):
            value = row.get(key)
            if value is None and key.endswith("_mse"):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or missing diagnostic metric: {key}")
        coverage["missing_reference"] += row.get("reference_mse") is None
        coverage["missing_quantized"] += row.get("quantized_mse") is None
        coverage["complete_pairs"] += row.get("reference_mse") is not None and row.get("quantized_mse") is not None
    complete = coverage["complete_pairs"] == len(rows)
    absent = coverage["missing_reference"] == coverage["missing_quantized"] == len(rows)
    return {
        "selected_kind": "raw_mse" if complete else "objective_loss",
        "selection_reason": "all_raw_mse_available" if complete else "raw_mse_not_recorded" if absent else "raw_mse_partially_missing",
        "raw_mse_coverage": coverage,
        "definitions": {
            "raw_mse": "Unweighted mean squared prediction error recorded by the diagnostic observer; whole model, not a component loss.",
            "objective_loss": "The scalar training objective used for backward; may include training weights and transformations.",
        },
        "compatibility_fields": {"reference_loss": "selected_kind reference value", "quantized_loss": "selected_kind quantized value"},
        "selection_scope": "One kind for all candidates and regimes; never mix MSE and objective loss in a comparison.",
    }


def public_loss_values(row, kind):
    raw = {"reference": row.get("reference_mse"), "quantized": row.get("quantized_mse")}
    missing = [name for name, value in raw.items() if value is None]
    raw["missing_reason"] = "missing_" + "_and_".join(missing) if missing else None
    objective = {"reference": row["reference_loss"], "quantized": row["quantized_loss"]}
    selected = raw if kind == "raw_mse" else objective
    return {"loss_kind": kind, "raw_mse": raw, "objective_loss": objective,
            "reference_loss": selected["reference"], "quantized_loss": selected["quantized"]}


def local_image_context(path, inventory_matches, thumbnails):
    """Do not choose one of several training contexts for the same physical image."""
    contexts = [{key: item.get(key) for key in CONTEXT_FIELDS} for item in inventory_matches]
    image_ids = {item.get("image_id") for item in inventory_matches} - {None}
    preview = thumbnails.get(next(iter(image_ids)), {}) if len(image_ids) == 1 else {}
    available = preview.get("status") == "available"
    uri = preview.get("data_url", "")
    available = available and uri.startswith("data:image/jpeg;base64,")
    match = "unique" if len(contexts) == 1 else "multiple_contexts" if contexts else "not_recorded"
    return {"contexts": contexts, "inventory_match": match,
            "image_uri": uri if available else Path(path).resolve().as_uri(),
            "preview_kind": "cached_thumbnail" if available else "original_file",
            "preview_status": "available" if available else preview.get("status", "not_cached")}
