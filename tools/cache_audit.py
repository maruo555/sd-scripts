import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from library import self_distill_cache


def _mean(values):
    return float(sum(values) / max(len(values), 1))


def audit_cache(args):
    header, entries = self_distill_cache.load_manifest_with_header(args.cache_manifest)
    rows = []
    by_variant = defaultdict(list)
    by_seed = defaultdict(list)
    conditioning_source_deltas = defaultdict(list)

    for entry in entries:
        bundle = self_distill_cache.load_tensor_bundle(entry["tensors_path"])
        deltas = (bundle["teacher_target"] - bundle["base_target"]).float().view(bundle["teacher_target"].shape[0], -1).norm(dim=1)
        delta_values = [float(v.item()) for v in deltas]
        row = {
            "record_id": entry["record_id"],
            "variant_type": entry["variant_type"],
            "conditioning_source": entry.get("conditioning_source", "teacher"),
            "seed": int(entry["seed"]),
            "delta_norm_mean": _mean(delta_values),
            "delta_norm_min": min(delta_values),
            "delta_norm_max": max(delta_values),
        }
        rows.append(row)
        by_variant[entry["variant_type"]].extend(delta_values)
        by_seed[int(entry["seed"])].append(row["delta_norm_mean"])
        conditioning_source_deltas[entry.get("conditioning_source", "teacher")].append(row["delta_norm_mean"])

    summary = {
        "manifest_header": header,
        "num_records": len(entries),
        "delta_norm_global_mean": _mean([row["delta_norm_mean"] for row in rows]),
        "per_variant": {
            variant: {
                "count": len(values),
                "delta_norm_mean": _mean(values),
                "delta_norm_min": min(values),
                "delta_norm_max": max(values),
            }
            for variant, values in sorted(by_variant.items())
        },
        "seed_stability": {
            str(seed): {
                "count": len(values),
                "mean_delta_norm": _mean(values),
            }
            for seed, values in sorted(by_seed.items())
        },
        "weak_keep_records": sum(
            1
            for row in rows
            if row["variant_type"].startswith("keep") and row["delta_norm_mean"] < args.keep_delta_floor
        ),
        "strong_suppress_records": sum(
            1
            for row in rows
            if ("suppress" in row["variant_type"] or row["variant_type"] == "off_null")
            and row["delta_norm_mean"] > args.suppress_delta_ceiling
        ),
        "conditioning_source_mean": {source: _mean(values) for source, values in sorted(conditioning_source_deltas.items())},
    }

    self_distill_cache.ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, "cache_audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.output_csv and rows:
        with open(os.path.join(args.output_dir, "cache_audit_rows.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        f"records={summary['num_records']}",
        f"global_delta_norm_mean={summary['delta_norm_global_mean']:.6f}",
        f"weak_keep_records={summary['weak_keep_records']}",
        f"strong_suppress_records={summary['strong_suppress_records']}",
    ]
    with open(os.path.join(args.output_dir, "cache_audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_csv", action="store_true")
    parser.add_argument("--keep_delta_floor", type=float, default=0.02)
    parser.add_argument("--suppress_delta_ceiling", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    audit_cache(args)
