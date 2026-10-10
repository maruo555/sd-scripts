"""Versioned standard diagnostic conditions and component statistics (CPU)."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from library.dq_mul_policy import MulPolicy, module_group
from dq_profile.v24_acceptance import _cluster_quantile, _index_rows, _body_tail

STANDARD_VERSION = "attn2-te-standard-v1"
UNIFORM_MULS = (2.70, 3.15, 3.45, 3.75, 4.05)
VIEWS = {"all": "全体", "te": "TE全体", "te1": "TE1", "te2": "TE2",
         "unet": "UNet全体", "attn2": "UNetのattn2", "unet_rest": "attn2以外のUNet"}
LEAVES = ("te1", "te2", "attn2", "unet_rest")
MEMBERS = {"all": LEAVES, "te": ("te1", "te2"), "te1": ("te1",), "te2": ("te2",),
           "unet": ("attn2", "unet_rest"), "attn2": ("attn2",), "unet_rest": ("unet_rest",)}


def condition(cid, low, high=None, *, kind="uniform", label=None):
    high = low if high is None else high
    policy = MulPolicy.from_dict({"base_mul": low, "components": {"te1": high, "te2": high},
        "group_overrides": [{"group_id": "unet.attn2", "range_mul": high}]})
    return {"id": cid, "kind": kind, "family": "uniform" if kind == "uniform" else "spatial",
            "label": label or f"一律{low:.2f}", "short": f"一律{low:.2f}" if kind == "uniform" else f"{low:.2f}/{high:.2f}",
            "mul": [low, high, high, high], "policy": policy.to_dict(), "aliases": [],
            "symbol": {"uniform": "circle", "adaptive": "diamond", "reference": "star"}[kind],
            "assignment_label": f"その他UNet {low:.2f} / attn2 {high:.2f} / TE1・TE2 {high:.2f}",
            "training_cli": f"--dq_delta_scope both --dq_delta_range_mul {low:.2f} --dq_delta_range_mul_attn2 {high:.2f} --dq_delta_range_mul_te {high:.2f}"}


def build_conditions(body_mul=None, *, spatial=True):
    rows = [condition(f"mul_{mul:.3f}", mul) for mul in UNIFORM_MULS]
    skipped = []
    if not spatial:
        return rows, [{"reason": "spatial_comparison_disabled"}]
    if body_mul is not None and float(body_mul) not in UNIFORM_MULS:
        raise ValueError("Body reference must come from the OFF uniform five-point grid")
    proposed = [condition("attn2_te_reference", 2.7, 3.75, kind="reference", label="attn2・TE高mul型［固定基準］")]
    if body_mul is None:
        skipped.append({"reason": "body_representative_unavailable"})
    else:
        for low in (2.7, 3.15):
            label = f"attn2・TE高mul型［Body基準・低側{low:.2f}］"
            if low > body_mul:
                skipped.append({"label": label, "reason": "low_exceeds_body", "low": low, "high": body_mul})
            else:
                proposed.append(condition(f"attn2_te_body_l{int(round(low * 100))}", low, body_mul, kind="adaptive", label=label))
    for item in proposed:
        previous = next((r for r in rows if r["mul"] == item["mul"]), None)
        if previous is not None:
            previous["aliases"].append({"id": item["id"], "label": item["label"], "reused_from": previous["id"]})
        else:
            rows.append(item)
    return rows, skipped


def body_from_analysis(result):
    """Same Body representative eligibility as the practical report, independent of spatial rows."""
    selection = result["selection"]
    name = selection.get("point_body_min_candidate")
    credible = selection.get("credible_candidates", [])
    row = next((r for r in result["score_rows"] if r["candidate"] == name), None)
    if row and name in credible and row["hard_safety_pass"] and not row["robustly_dominated"]:
        return float(row["range_mul"])
    return None


def component_moments(reference, candidate):
    """Keep additive squared norms and inner products; never infer causal attribution."""
    import torch
    if set(reference.values) != set(candidate.values):
        raise ValueError("Component gradient topology differs from the reference")
    result = {part: dict(reference_sq=0.0, quantized_sq=0.0, dot=0.0, difference_sq=0.0) for part in LEAVES}
    for name, value in reference.values.items():
        other = candidate.values[name]
        if value.shape != other.shape:
            raise ValueError("Component gradient tensor shape mismatch")
        group = module_group(name.split(".", 1)[0])
        part = group if group in ("te1", "te2") else "attn2" if group == "unet.attn2" else "unet_rest"
        left, right = value.double(), other.double()
        values = (float(torch.sum(left * left)), float(torch.sum(right * right)),
                  float(torch.sum(left * right)), float(torch.sum((right - left) ** 2)))
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite component gradient measurement")
        for key, number in zip(result[part], values):
            result[part][key] += number
    return result


def metrics_from_moments(moments):
    result = {}
    for view, parts in MEMBERS.items():
        sums = {k: sum(moments[p][k] for p in parts) for k in ("reference_sq", "quantized_sq", "dot", "difference_sq")}
        den = sums["reference_sq"]
        result[view] = {**sums, "distance": math.sqrt(sums["difference_sq"] / den) if den > 1e-30 else None,
                        "parallel": sums["dot"] / den if den > 1e-30 else None,
                        "missing_reason": None if den > 1e-30 else "reference_gradient_too_small"}
    return result


def balanced_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None or not math.isfinite(value):
            return None
        groups[row["source_group"]].append(value)
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def summarize(rows, *, bins=4, iterations=0, seed=2401):
    if not rows or any(r.get("value") is None or r.get("parallel") is None for r in rows):
        return {"missing_reason": "missing_or_degenerate_component_observation"}
    groups, by_bin = _index_rows(rows, timestep_bins=bins)
    sources = sorted(groups)
    if any(set(by_bin[b]) != set(sources) for b in range(bins)):
        raise ValueError("Incomplete timestep coverage in diagnostic observations")
    q = lambda p, field="value": _cluster_quantile(groups, sources, field, p)
    result = {f"p{int(p*100):02d}": q(p) for p in (.05, .1, .25, .5, .75, .9, .95)}
    result.update(body=q(.95), tail=max(_cluster_quantile(by_bin[b], sources, "value", .95) for b in range(bins)),
                  parallel_p50=q(.5, "parallel"), parallel_p05=q(.05, "parallel"),
                  mean_sq_d=balanced_mean([{**r, "sq": r["value"] ** 2} for r in rows], "sq"),
                  observations=len(rows), sources=len(sources))
    if iterations and len(sources) >= 2:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(iterations):
            selected = [sources[int(i)] for i in rng.integers(0, len(sources), len(sources))]
            values = _body_tail(groups, by_bin, selected, field="value", timestep_bins=bins)
            draws.append((values["body"], values["tail"]))
        result["overall_ci"] = {name: np.quantile(np.asarray(draws)[:, i], [.025, .975]).tolist() for i, name in enumerate(("body", "tail"))}
    return result
