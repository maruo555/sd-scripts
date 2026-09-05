"""Descriptive source-block intervals conditional on this one training run."""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np

from dq_profile.dataset_diagnostics import FLOOR, SCHEMA, finite, mean


def source_intervals(samples, muls, group_id, iterations=2000):
    sources = sorted({s.get("source_group_id") for s in samples if s.get("source_group_id")})
    seed = int.from_bytes(hashlib.sha256(f"{SCHEMA}:{group_id}".encode()).digest()[:8], "little")
    result = {"method": "source_block_image_equal", "iterations": iterations, "seed": seed,
              "source_count": len(sources), "status": "available" if len(sources) >= 4 else "insufficient_source_clusters",
              "baseline": {}, "quant": [], "selector_input": False}
    if len(sources) < 4:
        return result
    index = {s: i for i, s in enumerate(sources)}
    draws = np.random.default_rng(seed).integers(0, len(sources), size=(iterations, len(sources)))
    counts = np.stack([np.bincount(row, minlength=len(sources)) for row in draws]).astype(float)

    def weights(valid):
        per_image = Counter(s["image_id"] for s in valid)
        return np.asarray([1 / (len(per_image) * per_image[s["image_id"]]) for s in valid])

    def reduce(valid, vals):
        if len({s.get("source_group_id") for s in valid}) < 4:
            return np.full(iterations, np.nan)
        v = np.asarray(vals)
        w = weights(valid)
        sampled = counts[:, [index[s["source_group_id"]] for s in valid]] * w
        den = sampled.sum(axis=1)
        return np.divide(sampled @ v, den, out=np.full(iterations, np.nan), where=den > 0)

    def quantile_draws(valid, mi, bin_index=None):
        if len({s.get("source_group_id") for s in valid}) < 4:
            return np.full(iterations, np.nan)
        values, ws, src = [], [], []
        for sample, sw in zip(valid, weights(valid)):
            selected_bins = sample["bins"] if bin_index is None else [sample["bins"][bin_index]]
            for b in selected_bins:
                ds = b["quant"][mi]["d_values"]
                values.extend(ds)
                ws.extend([sw / (len(selected_bins) * len(ds))] * len(ds))
                src.extend([index[sample["source_group_id"]]] * len(ds))
        order = np.argsort(values, kind="stable")
        values = np.asarray(values)[order]
        sampled = counts[:, np.asarray(src)[order]] * np.asarray(ws)[order]
        cumul = np.cumsum(sampled, axis=1)
        total = cumul[:, -1]
        positions = np.argmax(cumul >= .95 * total[:, None], axis=1)
        return np.where(total > 0, values[positions], np.nan)

    def interval(values):
        values = np.asarray(values)
        valid = values[np.isfinite(values)]
        return {"low": float(np.quantile(valid, .025, method="linear")) if len(valid) else None,
                "high": float(np.quantile(valid, .975, method="linear")) if len(valid) else None,
                "valid_draws": len(valid), "status": "available" if len(valid) else "insufficient_source_clusters_or_invalid_denominator"}

    def divide(a, b):
        return np.divide(a, b, out=np.full(iterations, np.nan), where=b > FLOOR)

    base = [s for s in samples if s.get("source_group_id") and all(finite(b["loss_pre"]) and finite(b["loss_post"]) for b in s["bins"])]
    pre = reduce(base, [mean(b["loss_pre"] for b in s["bins"]) for s in base])
    post = reduce(base, [mean(b["loss_post"] for b in s["bins"]) for s in base])
    result["baseline"] = {key: interval(v) for key, v in (("loss_pre", pre), ("loss_post", post), ("improvement_abs", pre-post), ("improvement_rel", divide(pre-post, pre)))}
    for mi, mul in enumerate(muls):
        valid = [s for s in samples if s.get("source_group_id") and all(finite(b["quant"][mi]["d"]) and finite(b["quant"][mi]["delta"]) for b in s["bins"])]
        d = reduce(valid, [mean(b["quant"][mi]["d"] for b in s["bins"]) for s in valid])
        delta = reduce(valid, [mean(b["quant"][mi]["delta"] for b in s["bins"]) for s in valid])
        matched = reduce(valid, [mean(b["quant"][mi]["matched"] for b in s["bins"]) for s in valid])
        result["quant"].append({"mul": mul, "d": interval(d), "delta": interval(delta), "relative": interval(divide(delta, matched)), "d_q95": interval(quantile_draws(valid, mi)), "d_tail_q95": interval(np.max(np.stack([quantile_draws(valid, mi, b) for b in range(len(samples[0]["bins"]))]), axis=0))})
    return result
