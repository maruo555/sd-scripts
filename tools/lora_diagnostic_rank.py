"""Effective-rank summaries and heatmaps. Only complete module sets enter the statistics."""
import hashlib
import json
import statistics
from typing import Any, Dict, List, Optional, Tuple

from library.lora_run_schema import HEADERS
from tools.lora_run_metrics import rank_fields
from tools.lora_diagnostic_numbers import safe_int, safe_float


def _summarize(rows, names, module_param_counts):
    rank_lr_keys = [
        "UnetLRMin",
        "UnetLRMax",
        "Te1LRMin",
        "Te1LRMax",
        "Te2LRMin",
        "Te2LRMax",
    ]
    empty_grouped = {
        "path": {"labels": [], "rows": []},
        "role": {"labels": [], "rows": []},
    }
    empty_final_heatmaps = {}
    if not rows:
        return {
            "path": "rank.csv",
            "rows": [],
            "markers": [],
            "grouped": empty_grouped,
            "final_heatmaps": empty_final_heatmaps,
            "summary": {
                "rows": 0,
                "final_rank_sat_p95": None,
                "final_rank_energy_sum": None,
            },
        }

    def _classify_path_group(module_name: str) -> str:
        lowered = (module_name or "").lower()
        if "down_blocks_" in lowered or "input_blocks_" in lowered:
            return "down"
        if "mid_block_" in lowered or "middle_block_" in lowered:
            return "mid"
        if "up_blocks_" in lowered or "output_blocks_" in lowered:
            return "up"
        return "other"

    def _classify_role_group(module_name: str) -> str:
        lowered = (module_name or "").lower()
        if "_to_q" in lowered:
            return "to_q"
        if "_to_k" in lowered:
            return "to_k"
        if "_to_v" in lowered:
            return "to_v"
        if "_to_out" in lowered:
            return "to_out"
        if "ff_net" in lowered or "_ff_" in lowered:
            return "ff"
        if "resnets_" in lowered:
            return "resnet"
        if "upsamplers_" in lowered or "downsamplers_" in lowered:
            return "sampler"
        if "conv" in lowered:
            return "conv"
        return "other"

    def _collapse_heatmap_role_label(role_label: str) -> str:
        if role_label in {"to_q", "to_k", "to_v", "to_out", "ff"}:
            return role_label
        return "other"

    def _order_group_labels(group_name: str, labels: List[str]) -> List[str]:
        preferred_orders = {
            "path": ["down", "mid", "up", "other"],
            "role": ["to_q", "to_k", "to_v", "to_out", "ff", "resnet", "sampler", "conv", "other"],
        }
        ordered = [label for label in preferred_orders.get(group_name, []) if label in labels]
        ordered.extend(sorted(label for label in labels if label not in ordered))
        return ordered

    def _finalize_grouped_rows(group_name: str, group_buckets: Dict[Tuple[int, str], Dict[str, Any]], eps: float):
        grouped_rows: List[Dict[str, Any]] = []
        labels = set()
        for bucket in group_buckets.values():
            groups_out: Dict[str, Dict[str, Any]] = {}
            total_energy = sum(info["energy_sum"] for info in bucket["groups"].values())
            shares_complete = not any(info["missing_params"] for info in bucket["groups"].values())
            total_energy_per_param = 0.0
            for info in bucket["groups"].values():
                param_count = info.get("param_count_sum")
                if not info["missing_params"] and param_count and param_count > 0:
                    total_energy_per_param += info["energy_sum"] / param_count
            for label, info in bucket["groups"].items():
                labels.add(label)
                sat_wmean = None
                if info["sat_weighted_energy"] > eps:
                    sat_wmean = info["sat_weighted_sum"] / info["sat_weighted_energy"]
                elif info["sat_values"]:
                    sat_wmean = statistics.mean(info["sat_values"])
                energy_share = None
                if total_energy > eps:
                    energy_share = info["energy_sum"] / total_energy
                energy_per_param = None
                energy_share_per_param = None
                param_count = info.get("param_count_sum")
                if not info["missing_params"] and param_count and param_count > 0:
                    energy_per_param = info["energy_sum"] / param_count
                    if shares_complete and total_energy_per_param > eps:
                        energy_share_per_param = energy_per_param / total_energy_per_param
                groups_out[label] = {
                    "RankEnergySum": info["energy_sum"],
                    "RankEnergyShare": energy_share,
                    "RankEnergyPerParam": energy_per_param,
                    "RankEnergySharePerParam": energy_share_per_param,
                    "RankSatWMean": sat_wmean,
                    "ParamCount": param_count,
                }

            row_out = {
                "TrainStep": bucket["TrainStep"],
                "Epoch": bucket.get("Epoch"),
                "Scope": bucket["Scope"],
                "groups": groups_out,
            }
            for lr_key in rank_lr_keys:
                row_out[lr_key] = bucket.get(lr_key)
            grouped_rows.append(row_out)

        grouped_rows.sort(key=lambda item: item["TrainStep"])
        return {
            "labels": _order_group_labels(group_name, list(labels)),
            "rows": grouped_rows,
        }

    parsed_rows: List[Dict[str, Any]] = []
    grouped = empty_grouped
    final_heatmaps = empty_final_heatmaps

    buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
    eps = 1e-12
    grouped_buckets = {
        "path": {},
        "role": {},
    }
    final_step_modules: List[Dict[str, Any]] = []
    last_step = max(row["attempt_step"] for row in rows)
    for record in rows:
        row = {label: record[key] for key, label in names}
        train_step = safe_int(row.get("TrainStep"))
        if train_step is None:
            continue
        epoch = safe_int(row.get("Epoch"))
        scope = (row.get("Scope") or "").strip()
        key = (train_step, scope)
        if key not in buckets:
            buckets[key] = {
                "TrainStep": train_step,
                "Epoch": epoch,
                "Scope": scope,
                "rank_dims": [],
                "records": [],
            }
            for lr_key in rank_lr_keys:
                buckets[key][lr_key] = None
        bucket = buckets[key]
        if bucket.get("Epoch") is None and epoch is not None:
            bucket["Epoch"] = epoch

        lr_values = {lr_key: safe_float(row.get(lr_key)) for lr_key in rank_lr_keys}
        for lr_key, lr_value in lr_values.items():
            if bucket.get(lr_key) is None and lr_value is not None:
                bucket[lr_key] = lr_value

        rank_dim = safe_float(row.get("RankDim"))
        sat = safe_float(row.get("RankSat"))
        top1 = safe_float(row.get("RankTop1"))
        energy = safe_float(row.get("RankEnergy"))
        module_name = (row.get("Module") or "").strip()
        module_param_count = module_param_counts.get(module_name) if module_param_counts is not None else None

        if rank_dim is not None:
            bucket["rank_dims"].append(rank_dim)
        bucket["records"].append(record)

        group_names = {
            "path": _classify_path_group(module_name),
            "role": _classify_role_group(module_name),
        }
        for grouping_name, label in group_names.items():
            grouping_buckets = grouped_buckets[grouping_name]
            if key not in grouping_buckets:
                grouping_buckets[key] = {
                    "TrainStep": train_step,
                    "Epoch": epoch,
                    "Scope": scope,
                    "groups": {},
                }
                for lr_key in rank_lr_keys:
                    grouping_buckets[key][lr_key] = None
            grouped_bucket = grouping_buckets[key]
            if grouped_bucket.get("Epoch") is None and epoch is not None:
                grouped_bucket["Epoch"] = epoch
            for lr_key, lr_value in lr_values.items():
                if grouped_bucket.get(lr_key) is None and lr_value is not None:
                    grouped_bucket[lr_key] = lr_value
            if label not in grouped_bucket["groups"]:
                grouped_bucket["groups"][label] = {
                    "energy_sum": 0.0,
                    "sat_weighted_sum": 0.0,
                    "sat_weighted_energy": 0.0,
                    "sat_values": [],
                    "param_count_sum": 0, "missing_params": False,
                }
            info = grouped_bucket["groups"][label]
            if energy is not None:
                info["energy_sum"] += energy
            if module_param_count is not None and module_param_count > 0:
                info["param_count_sum"] += int(module_param_count)
            else:
                info["missing_params"] = True
            if sat is not None:
                if energy is not None and energy > eps:
                    info["sat_weighted_sum"] += sat * energy
                    info["sat_weighted_energy"] += energy
                else:
                    info["sat_values"].append(sat)

        if train_step == last_step:
            final_step_modules.append(
                {
                    "TrainStep": train_step,
                    "Epoch": epoch,
                    "Scope": scope,
                    "Module": module_name,
                    "PathGroup": group_names["path"],
                    "RoleGroup": _collapse_heatmap_role_label(group_names["role"]),
                    "RankSat": sat,
                    "RankEnergy": energy,
                    "ParamCount": module_param_count,
                }
            )

    for bucket in buckets.values():
        rank_dim = None
        if bucket["rank_dims"]:
            rank_dim_set = set(bucket["rank_dims"])
            if len(rank_dim_set) == 1:
                rank_dim = next(iter(rank_dim_set))

        parsed_rows.append(
            {
                "TrainStep": bucket["TrainStep"],
                "Epoch": bucket.get("Epoch"),
                "Scope": bucket["Scope"],
                "RankDim": rank_dim,
                **rank_fields(bucket["records"], ()),
            }
        )
        for lr_key in rank_lr_keys:
            parsed_rows[-1][lr_key] = bucket.get(lr_key)

    grouped = {
        "path": _finalize_grouped_rows("path", grouped_buckets["path"], eps),
        "role": _finalize_grouped_rows("role", grouped_buckets["role"], eps),
    }

    if final_step_modules:
        final_step = max(item["TrainStep"] for item in final_step_modules)
        final_modules = [item for item in final_step_modules if item["TrainStep"] == final_step]
        heatmap_path_labels = ["down", "mid", "up"]
        heatmap_role_labels = ["to_q", "to_k", "to_v", "to_out", "ff", "other"]
        cell_buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for path_label in heatmap_path_labels:
            for role_label in heatmap_role_labels:
                cell_buckets[(path_label, role_label)] = {
                    "energy_sum": 0.0,
                    "sat_weighted_sum": 0.0,
                    "sat_weighted_energy": 0.0,
                    "sat_values": [],
                    "param_count_sum": 0, "missing_params": False,
                }

        final_epoch = None
        final_scope = None
        for item in final_modules:
            path_label = item.get("PathGroup")
            role_label = item.get("RoleGroup")
            if path_label not in heatmap_path_labels or role_label not in heatmap_role_labels:
                continue
            if final_epoch is None and item.get("Epoch") is not None:
                final_epoch = item.get("Epoch")
            if final_scope is None and item.get("Scope"):
                final_scope = item.get("Scope")
            info = cell_buckets[(path_label, role_label)]
            energy = item.get("RankEnergy")
            sat = item.get("RankSat")
            param_count = item.get("ParamCount")
            if energy is not None:
                info["energy_sum"] += energy
            if param_count is not None and param_count > 0:
                info["param_count_sum"] += int(param_count)
            else:
                info["missing_params"] = True
            if sat is not None:
                if energy is not None and energy > eps:
                    info["sat_weighted_sum"] += sat * energy
                    info["sat_weighted_energy"] += energy
                else:
                    info["sat_values"].append(sat)

        shares_complete = not any(info["missing_params"] for info in cell_buckets.values())
        total_energy_per_param = 0.0
        for info in cell_buckets.values():
            if not info["missing_params"] and info["param_count_sum"] > 0:
                total_energy_per_param += info["energy_sum"] / info["param_count_sum"]

        def _build_heatmap(metric_key: str) -> Dict[str, Any]:
            rows_out = []
            max_value = 0.0
            for path_label in heatmap_path_labels:
                cells = []
                for role_label in heatmap_role_labels:
                    info = cell_buckets[(path_label, role_label)]
                    value = None
                    if metric_key == "RankEnergySharePerParam":
                        if not info["missing_params"] and info["param_count_sum"] > 0:
                            energy_per_param = info["energy_sum"] / info["param_count_sum"]
                            if shares_complete and total_energy_per_param > eps:
                                value = energy_per_param / total_energy_per_param
                    elif metric_key == "RankSatWMean":
                        if info["sat_weighted_energy"] > eps:
                            value = info["sat_weighted_sum"] / info["sat_weighted_energy"]
                        elif info["sat_values"]:
                            value = statistics.mean(info["sat_values"])
                    if value is not None:
                        max_value = max(max_value, value)
                    cells.append(
                        {
                            "path": path_label,
                            "role": role_label,
                            "value": value,
                            "RankEnergySum": info["energy_sum"],
                            "ParamCount": None if info["missing_params"] else info["param_count_sum"] or None,
                        }
                    )
                rows_out.append({"label": path_label, "cells": cells})
            return {
                "rows": rows_out,
                "path_labels": heatmap_path_labels,
                "role_labels": heatmap_role_labels,
                "max_value": max_value if max_value > 0 else None,
                "TrainStep": final_step,
                "Epoch": final_epoch,
                "Scope": final_scope,
            }

        final_heatmaps = {
            "energy_share_per_param": _build_heatmap("RankEnergySharePerParam"),
            "rank_sat_wmean": _build_heatmap("RankSatWMean"),
        }
    parsed_rows.sort(key=lambda item: item["TrainStep"])

    markers: List[Dict[str, Any]] = []
    last_epoch: Optional[int] = None
    for item in parsed_rows:
        epoch = item.get("Epoch")
        if epoch is None:
            continue
        if last_epoch != epoch:
            markers.append({"x": item["TrainStep"], "label": f"E{epoch}"})
            last_epoch = epoch

    latest = parsed_rows[-1] if parsed_rows else {}
    summary = {
        "rows": len(parsed_rows),
        "final_rank_sat_p95": latest.get("RankSatP95"),
        "final_rank_energy_sum": latest.get("RankEnergySum"),
    }

    return {
        "path": "rank.csv",
        "rows": parsed_rows,
        "markers": markers,
        "grouped": grouped,
        "final_heatmaps": final_heatmaps,
        "summary": summary,
    }


def analyze_rank(reader, rows, module_param_counts=None):
    labels = ("Epoch", "TrainStep", "Scope", "Module", "RankDim", "RankSat", "RankTop1", "RankEnergy",
              "UnetLRMin", "UnetLRMax", "Te1LRMin", "Te1LRMax", "Te2LRMin", "Te2LRMax")
    names = tuple(zip(HEADERS["rank"], labels))
    valid, omissions = [], []
    observed_dims = {}
    inconsistent_dims = False
    for coordinate, records, reasons in reader.rank_observations(rows):
        if reasons:
            omissions.append({"coordinate": coordinate, "reasons": reasons, "module_count": len(records)})
        else:
            valid.extend(records)
            scope_dims = observed_dims.setdefault(coordinate[2], {})
            for row in records:
                name, dim = row["module"], row["rank_dim"]
                if type(dim) is not int or dim <= 0 or scope_dims.setdefault(name, dim) != dim:
                    inconsistent_dims = True
    output = _summarize(valid, names, module_param_counts)
    output["incomplete"] = omissions
    expected = getattr(reader, "index", {}).get("artifacts", {}).get("rank", {}).get("expected_modules_by_scope", {})
    contract_complete = expected and set(expected) == set(observed_dims) and not inconsistent_dims
    digest = None
    if contract_complete:
        canonical = json.dumps(observed_dims, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
    output["expected_modules_sha256"] = digest
    output["missing_param_count_modules"] = sorted({r["module"] for r in valid
                                                   if not (module_param_counts or {}).get(r["module"])})
    return output
