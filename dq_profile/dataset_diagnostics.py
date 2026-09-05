"""CPU-only dataset diagnostics. These values are NEVER selector inputs.

Raw observations are the authority; reports can be rebuilt without torch/GPU.
Missing or incomplete paired cells stay missing instead of changing denominators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "dataset-diagnostics-v1.2"
FLOOR = 1e-12


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values and all(finite(x) for x in values) else None


def ratio(a, b):
    return a / b if finite(a) and finite(b) and b > FLOOR else None


def identity(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def normalized_path(value):
    return os.path.normcase(os.path.realpath(os.fspath(value)))


def tags(caption):
    return sorted(set(unicodedata.normalize("NFC", x.strip()) for x in (caption or "").split(",") if x.strip()))


def improvement(pre, post):
    delta = pre - post if finite(pre) and finite(post) else None
    return {"loss_pre": pre, "loss_post": post, "improvement_abs": delta,
            "improvement_rel": ratio(delta, pre)}


def gradient_metrics(comparison):
    g0, gm, gd = (comparison.get(k) for k in ("reference_norm", "candidate_norm", "difference_norm"))
    valid = comparison.get("topology_matches") is True and all(finite(x) and x >= 0 for x in (g0, gm, gd))
    reason = None if valid else "gradient_topology_or_nonfinite"
    return {"grad_norm_noquant": g0 if finite(g0) else None,
            "grad_norm_quant": gm if finite(gm) else None,
            "grad_diff_norm": gd if valid else None,
            "d": ratio(gd, g0) if valid else None,
            "norm_ratio": ratio(gm, g0) if valid else None,
            "cosine": comparison.get("cosine") if valid and g0 > FLOOR and gm > FLOOR and finite(comparison.get("cosine")) else None,
            "symmetric_d": ratio(2 * gd, g0 + gm) if valid else None,
            "reference_near_zero": finite(g0) and 0 <= g0 <= FLOOR,
            "gradient_topology_matches": comparison.get("topology_matches") is True,
            "gradient_invalid_reason": reason or ("reference_near_zero" if g0 <= FLOOR else None)}


def weighted_quantile(values, weights, p=.95):
    values, weights = list(values), list(weights)
    if len(values) != len(weights) or not finite(p) or not 0 <= p <= 1:
        raise ValueError("invalid weighted quantile arguments")
    pairs = list(zip(values, weights))
    if not pairs or any(not finite(x) or not finite(w) or w < 0 for x, w in pairs):
        return None
    pairs.sort(key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    acc = 0
    for value, weight in pairs:
        acc += weight
        if acc >= p * total:
            return value
    return pairs[-1][0]


def load_group_map(path=None):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {"schema_version": "dataset-groups-v1", "aliases": {}, "groups": []}
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise ValueError("group-map must contain a groups array")
    aliases = data.setdefault("aliases", {})
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v or v in aliases for k, v in aliases.items()):
        raise ValueError("aliases require nonempty strings and one-hop mappings without chains/cycles")
    seen = {"__ungrouped__", "__unknown__", "__all__"}
    for group in data["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str) or not group["id"] or group["id"] in seen:
            raise ValueError("group-map needs unique nonempty string ids")
        seen.add(group["id"])
        if set(group) - {"id", "label", "kind", "tags_any", "tags_all", "image_paths", "subset_groups"}:
            raise ValueError("unknown group-map field")
        for key in ("tags_any", "tags_all", "image_paths", "subset_groups"):
            if key in group and (not isinstance(group[key], list) or any(not isinstance(x, str) or not x for x in group[key])):
                raise ValueError(f"invalid group-map {key}")
        if not any(group.get(key) for key in ("tags_any", "tags_all", "image_paths", "subset_groups")):
            raise ValueError("group-map group has no membership rule")
        for key in ("tags_any", "tags_all"):
            group[key] = sorted({aliases.get(unicodedata.normalize("NFC", v.strip()), unicodedata.normalize("NFC", v.strip())) for v in group.get(key, [])})
        group.setdefault("label", group["id"])
        if not isinstance(group["label"], str) or group.get("kind", "character") not in {"character", "subset", "semantic"}:
            raise ValueError("invalid group label/kind")
        for i, value in enumerate(group.get("image_paths", [])):
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = Path(path).resolve().parent / resolved
            if not resolved.is_file():
                raise ValueError(f"explicit group image does not exist: {resolved}")
            group["image_paths"][i] = normalized_path(resolved)
    return data


def belongs(sample, group, aliases=None):
    aliases = aliases or {}
    st = {aliases.get(t, t) for t in sample.get("tags", [])}
    # Alternatives are OR; a tags_all clause itself is an intersection.
    return bool(st.intersection(group.get("tags_any", [])) or
                (group.get("tags_all") and st.issuperset(group["tags_all"])) or
                sample.get("path") in group.get("image_paths", []) or
                sample.get("subset_group") in group.get("subset_groups", []))


def unique_rows(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row.get(k) for k in fields)
        if key in result and result[key] != row:
            raise ValueError(f"conflicting duplicate observation: {key}")
        result[key] = row
    return list(result.values())


def summarize_samples(inventory, refs, quant, muls, bins=4, noises=3, candidate_noises=2, repeats=2):
    refs = unique_rows(refs, ("sample_id", "eval_input_id", "snapshot"))
    quant = unique_rows(quant, ("sample_id", "eval_input_id", "mul", "quant_repeat"))
    ref_index = {}
    by_quant = defaultdict(list)
    for row in refs:
        key = (row["sample_id"], row["bin"], row["noise"], row["snapshot"])
        if key in ref_index and ref_index[key] != row:
            raise ValueError("multiple evaluation inputs in a sample cell")
        ref_index[key] = row
    for row in quant:
        by_quant[(row["sample_id"], row["mul"], row["bin"], row["noise"], row["quant_repeat"])].append(row)
    result = []
    for sample in inventory:
        sid = sample["sample_id"]
        bin_values = []
        for b in range(bins):
            pre = [ref_index.get((sid, b, r, "pre"), {}) for r in range(noises)]
            post = [ref_index.get((sid, b, r, "post"), {}) for r in range(noises)]
            paired = all(x.get("eval_input_id") and x.get("eval_input_id") == y.get("eval_input_id") for x, y in zip(pre, post))
            lpost = mean(x.get("raw_mse") for x in post)
            lpre = mean(x.get("raw_mse") for x in pre) if paired else None
            bv = {"bin": b, **improvement(lpre, lpost), "quant": []}
            for mul in muls:
                cells = [by_quant.get((sid, mul, b, r, q), []) for r in range(candidate_noises) for q in range(repeats)]
                rows = [cell[0] for cell in cells if len(cell) == 1]
                valid = len(rows) == candidate_noises * repeats and all(
                    x.get("eval_input_id") == post[x["noise"]].get("eval_input_id") and finite(x.get("raw_mse")) and finite(post[x["noise"]].get("raw_mse")) for x in rows)
                delta = mean(x["raw_mse"] - post[x["noise"]]["raw_mse"] for x in rows) if valid else None
                matched = mean(post[r].get("raw_mse") for r in range(candidate_noises)) if valid else None
                ds = [x.get("d") for x in rows]
                dv = mean(ds) if valid else None
                bv["quant"].append({"mul": mul, "delta": delta, "matched": matched,
                    "relative": ratio(delta, matched), "d": dv,
                    "d_values": ds if valid and dv is not None else [],
                    "cosine": mean(x.get("cosine") for x in rows) if valid else None,
                    "norm_ratio_values": [x.get("norm_ratio") for x in rows] if valid and all(finite(x.get("norm_ratio")) for x in rows) else [],
                    "norm_ratio": weighted_quantile([x.get("norm_ratio") for x in rows], [1]*len(rows), .5) if valid and all(finite(x.get("norm_ratio")) for x in rows) else None,
                    **{k: mean(x.get(k) for x in rows) if valid else None for k in ("grad_norm_noquant", "grad_diff_norm", "clip_rate", "quant_error_ratio", "clip_error_rms", "round_error_rms")},
                    "observed": len(rows), "expected": candidate_noises * repeats,
                    "invalid_reason": None if dv is not None and delta is not None else "missing_nonfinite_or_invalid_gradient"})
            bin_values.append(bv)
        by_noise = []
        for r in range(noises):
            pairs = [(ref_index.get((sid, b, r, "pre"), {}), ref_index.get((sid, b, r, "post"), {})) for b in range(bins)]
            by_noise.append(mean(x["raw_mse"] - y["raw_mse"] for x, y in pairs) if all(x.get("eval_input_id") == y.get("eval_input_id") and finite(x.get("raw_mse")) and finite(y.get("raw_mse")) for x,y in pairs) else None)
        result.append({**sample, "improvement_by_noise": by_noise, "bins": bin_values, "measured": any(x["loss_post"] is not None for x in bin_values),
                       "baseline_complete": all(x["loss_pre"] is not None and x["loss_post"] is not None for x in bin_values)})
    return result


def image_equal_weights(samples):
    counts = Counter(s["image_id"] for s in samples)
    return [1 / (len(counts) * counts[s["image_id"]]) for s in samples]


def aggregate(samples, muls, bin_index=None, common_muls=False):
    """Equal image weight after membership filtering, then equal sample/bin weight."""
    def bins(s):
        return s["bins"] if bin_index is None else [s["bins"][bin_index]]
    def image_mean(items, getter):
        per_image = defaultdict(list)
        for s in items:
            per_image[s["image_id"]].append(getter(s))
        return mean(mean(v) for v in per_image.values())
    baseline = [s for s in samples if all(finite(b["loss_pre"]) and finite(b["loss_post"]) for b in bins(s))]
    post_only = [s for s in samples if all(finite(b["loss_post"]) for b in bins(s))]
    base = improvement(image_mean(baseline, lambda s: mean(b["loss_pre"] for b in bins(s))),
                       image_mean(baseline, lambda s: mean(b["loss_post"] for b in bins(s))))
    base["loss_post_available"] = image_mean(post_only, lambda s: mean(b["loss_post"] for b in bins(s)))
    curves = []
    for mi, mul in enumerate(muls):
        required = range(len(muls)) if common_muls else [mi]
        valid = [s for s in samples if all(finite(b["quant"][j]["d"]) and finite(b["quant"][j]["delta"]) for b in bins(s) for j in required)]
        values, weights, norm_values, bin_rows = [], [], [], defaultdict(lambda: ([], []))
        count_by_image = defaultdict(int)
        for s in valid:
            count_by_image[s["image_id"]] += 1
        for s, sample_weight in zip(valid, image_equal_weights(valid)):
            bs = bins(s)
            for b in bs:
                ds = b["quant"][mi]["d_values"]
                weight = sample_weight / (len(bs) * len(ds))
                values.extend(ds); weights.extend([weight]*len(ds))
                norm_values.extend(b["quant"][mi]["norm_ratio_values"])
                bin_rows[b["bin"]][0].extend(ds); bin_rows[b["bin"]][1].extend([weight]*len(ds))
        def metric(key):
            return image_mean(valid, lambda s: mean(b["quant"][mi][key] for b in bins(s)))
        delta, matched = metric("delta"), metric("matched")
        curves.append({"mul": mul, "d": metric("d"), "d_q95": weighted_quantile(values, weights),
                       "d_tail_q95": max((weighted_quantile(v, w) for v, w in bin_rows.values()), default=None),
                       "delta": delta, "matched": matched, "relative": ratio(delta, matched),
                       "images": len(count_by_image), "samples": len(valid),
                       "source_count": len({s.get("source_group_id") for s in valid if s.get("source_group_id")}),
                       "d_max_bin_mean": max((sum(vv*ww for vv,ww in zip(v,w))/sum(w) for v,w in bin_rows.values()), default=None),
                       "cosine_mean": metric("cosine"),
                       "norm_ratio_median": weighted_quantile(norm_values, weights, .5) if len(norm_values) == len(weights) else None,
                       **{k + "_mean": metric(k) for k in ("grad_norm_noquant", "grad_diff_norm", "clip_rate", "quant_error_ratio", "clip_error_rms", "round_error_rms")}})
    return {**base, "quant": curves, "inventory_images": len({s["image_id"] for s in samples}),
            "measured_images": len({s["image_id"] for s in post_only}), "paired_images": len({s["image_id"] for s in baseline}),
            "samples": len(samples), "baseline_valid_samples": len(baseline), "source_count": len({s.get("source_group_id") for s in post_only if s.get("source_group_id")}), "weighting": "image_equal", "selector_input": False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            # Text cells cannot execute spreadsheet formulae when exported.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in row.items()})


def rebuild(directory, group_map=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported dataset diagnostics schema")
    def read(name):
        return [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    inventory, refs, quant = read("inventory.jsonl"), read("reference_probes.jsonl"), read("quant_probes.jsonl")
    groups = load_group_map(group_map) if group_map else manifest["group_map"]
    # Explicit subset groups remain available without user TOML changes.
    for name in sorted({s.get("subset_group") for s in inventory if s.get("subset_group")}):
        found = next((g for g in groups["groups"] if g["id"] == name), None)
        if found is None:
            groups["groups"].append({"id": name, "label": name, "kind": "subset", "subset_groups": [name]})
        else:
            found["subset_groups"] = sorted(set(found.get("subset_groups", []) + [name]))
    samples = summarize_samples(inventory, refs, quant, manifest["muls"], manifest["bins"])
    eval_file = directory / "evaluation_inputs.jsonl"
    evaluations = read("evaluation_inputs.jsonl") if eval_file.is_file() else []
    for sample in samples:
        sample["evaluation_inputs"] = [row for row in evaluations if row["sample_id"] == sample["sample_id"]]
        sample["group_memberships"] = [{"id": g["id"], "label": g.get("label", g["id"]), "selector_input": False} for g in groups["groups"] if belongs(sample, g, groups.get("aliases"))]
    payload = {"manifest": {**manifest, "group_map": groups}, "samples": samples,
               "all": aggregate(samples, manifest["muls"])}
    write_json(directory / "dataset_summary.json", payload)
    write_csv(directory / "sample_baseline.csv", ({**{k: v for k, v in s.items() if k != "bins"}, **{k: v for k, v in aggregate([s], manifest["muls"]).items() if k != "quant"}} for s in samples))
    write_csv(directory / "sample_quant.csv", ({"sample_id": s["sample_id"], "image_id": s["image_id"], **q} for s in samples for q in aggregate([s], manifest["muls"])["quant"]))
    entities = [("all", "all", samples)]
    entities += [("folder", key, [s for s in samples if s["folder_id"] == key]) for key in sorted({s["folder_id"] for s in samples})]
    entities += [("character", g["id"], [s for s in samples if belongs(s, g, groups.get("aliases"))]) for g in groups["groups"]]
    from dq_profile.dataset_uncertainty import source_intervals
    summaries = [(kind, key, aggregate(items, manifest["muls"])) for kind, key, items in entities]
    intervals = [{"kind": kind, "id": key, **source_intervals(items, manifest["muls"], f"{kind}:{key}")} for kind, key, items in entities]
    payload["intervals"] = intervals
    payload["manifest"]["grouping_hash"] = hashlib.sha256(json.dumps(groups, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload["manifest"]["ci_status"] = "source_block_reference_intervals"
    write_json(directory / "dataset_summary.json", payload)
    write_json(directory / "source_intervals.json", intervals)
    write_json(directory / "manifest.json", payload["manifest"])
    ci_index = {(v["kind"], v["id"]): v for v in intervals}
    def ci_fields(kind, key, mi=None):
        ci = ci_index[(kind, key)]
        metrics = ci["baseline"] if mi is None else (ci["quant"][mi] if ci["quant"] else {})
        return {"ci_status": ci["status"], "ci_source_count": ci["source_count"], **{f"{metric}_ci_{edge}": value.get(edge) for metric, value in metrics.items() if isinstance(value, dict) for edge in ("low", "high", "valid_draws")}}
    write_csv(directory / "group_baseline.csv", ({"group_kind": k, "group_id": i, **ci_fields(k, i), **{key: value for key, value in s.items() if key != "quant"}} for k, i, s in summaries))
    write_csv(directory / "group_quant.csv", ({"group_kind": k, "group_id": i, **q, **ci_fields(k, i, mi), "weighting": "image_equal", "selector_input": False} for k, i, s in summaries for mi, q in enumerate(s["quant"])))
    from dq_profile.diagnostic_report import write_report
    write_report(directory / "dataset_report.html", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Rebuild dataset diagnostics from raw sidecars, without GPU")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--group-map", type=Path)
    args = parser.parse_args()
    rebuild(args.input_dir, args.group_map)


if __name__ == "__main__":
    main()
