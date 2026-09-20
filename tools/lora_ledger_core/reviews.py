"""Evaluation helpers; ratings never imply pairwise wins or adoption."""
from __future__ import annotations
import copy
import itertools
import os
from pathlib import Path
from .storage import (LedgerError, Cancelled, new_id, stable_id, new_review, read_json,
                      file_stat, same_stat, sha256_file)


def candidate_for(run, artifact, strength=None, lbw=None):
    return {"candidate_id": new_id(), "name": f"{run['display_name']} / {artifact['name']}",
            "components": [{"run_id": run["run_id"], "artifact_id": artifact["artifact_id"],
                            "strength": strength, "lbw": lbw}],
            "generation_link_verified": False}


def fingerprint_review(ledger, review, cancel=None, progress=None):
    review = copy.deepcopy(review)
    for candidate in review.get("candidates", []):
        if ("viewed_candidate_ids" in review and candidate["candidate_id"] not in review["viewed_candidate_ids"]
                and candidate.get("usability", "unknown") == "unknown"
                and candidate["candidate_id"] not in review.get("adoption", {}).get("candidate_ids", [])):
            continue
        for component in candidate.get("components", []):
            if cancel and cancel():
                raise Cancelled("保存を中止しました")
            if not component.get("run_id") or not component.get("artifact_id"):
                component["identity_status"] = "unresolved"
                continue
            run = ledger.get_run(component["run_id"])
            artifact = next(a for a in run["artifacts"] if a["artifact_id"] == component["artifact_id"])
            ref = artifact["ref"]
            try:
                path = ledger.resolve_ref(ref)
                if progress:
                    progress(f"重みの同一性を確認: {path.name}")
                if not same_stat(ref["observed"], file_stat(path)):
                    component["identity_status"] = "changed"
                    continue
                value = sha256_file(path, cancel)
                expected = ref["observed"].get("sha256") or component.get("content_sha256")
                if expected and expected != value:
                    component["identity_status"] = "changed"
                    continue
                component.update(content_sha256=value, identity_status="content_verified_at_review")
            except Cancelled:
                raise
            except (OSError, LedgerError):
                component["identity_status"] = "unverified"
    return review


def preference_pairs(candidates, viewed_ids, choices, reasons=None):
    """choices[axis]: candidate ID, 'tie', 'undecided', or None. One event per axis."""
    valid = {c["candidate_id"] for c in candidates}
    viewed = [c for c in viewed_ids if c in valid]
    pairs = []
    for axis, choice in choices.items():
        event = new_id()
        if not choice:
            continue
        if choice not in ("tie", "undecided") and choice not in viewed:
            raise LedgerError("第一候補は閲覧済みの候補から選んでください")
        if choice in ("tie", "undecided"):
            targets = [(a, b, choice) for a, b in itertools.combinations(viewed, 2)]
        else:
            targets = [(choice, other, "left") for other in viewed if other != choice]
        for left, right, result in targets:
            pairs.append({"axis": axis, "left": left, "right": right, "result": result,
                          "event_id": event, "origin": "overall_choice",
                          "reason": (reasons or {}).get(axis, "")})
    return pairs


def import_report(ledger, folder):
    folder = Path(folder).resolve()
    metadata_path = folder / "metadata.json"
    config_path = folder / "config.json"
    metadata = read_json(metadata_path)
    if not isinstance(metadata.get("conditions"), list) or not isinstance(metadata.get("jobs"), list):
        raise LedgerError("画像比較のmetadata.jsonを選んでください")
    config = read_json(config_path) if config_path.exists() else {}
    hashes = {"metadata_sha256": sha256_file(metadata_path),
              "config_sha256": sha256_file(config_path) if config_path.exists() else None}
    artifacts = {}
    for run in ledger.list_runs():
        for artifact in run["artifacts"]:
            try:
                key = os.path.normcase(str(ledger.resolve_ref(artifact["ref"])))
                artifacts.setdefault(key, []).append((run, artifact))
            except LedgerError:
                pass
    review = new_review()
    review["generation"] = config.get("sdxl_gen_img", config)
    review["report"] = {"path": str(folder), **hashes}
    review["selection_mode"] = "all_cases"
    review["cases"] = []
    condition_ids = {}
    for condition in metadata["conditions"]:
        candidate_id = stable_id(hashes["metadata_sha256"], condition["id"])
        condition_ids[condition["id"]] = candidate_id
        components = []
        for item in condition.get("items", []):
            path = Path(item.get("path", ""))
            if not path.is_absolute():
                path = folder / path
            matches = artifacts.get(os.path.normcase(str(path.resolve())), [])
            component = {"source_path": str(path), "strength": item.get("strength"),
                         "lbw": item.get("lbw"), "module": item.get("module"),
                         "identity_status": "unresolved"}
            if len(matches) == 1:
                run, artifact = matches[0]
                component.update(run_id=run["run_id"], artifact_id=artifact["artifact_id"],
                                 identity_status="path_match_only")
            components.append(component)
        review["candidates"].append({"candidate_id": candidate_id, "name": condition.get("name", condition["id"]),
                                      "components": components, "generation_link_verified": False})
    for index, job in enumerate(metadata["jobs"]):
        candidate_id = condition_ids.get(job.get("condition_id"))
        if not candidate_id or job.get("status") not in ("done", "success") or job.get("returncode") not in (0, None):
            continue
        image = (folder / job.get("image", "")).resolve()
        if not image.is_relative_to(folder) or not image.is_file():
            continue
        # Prompt contents as well as seed/dimensions define a paired generation case.
        key = stable_id(job.get("prompt_id"), job.get("prompt"), job.get("negative"),
                        job.get("seed"), job.get("width"), job.get("height"))
        review["cases"].append({"case_id": key, "candidate_id": candidate_id,
                                "prompt_id": job.get("prompt_id"), "prompt": job.get("prompt"),
                                "negative": job.get("negative"), "seed": job.get("seed"),
                                "width": job.get("width"), "height": job.get("height"),
                                "image": image.relative_to(folder).as_posix(), "job_index": index})
    return review


def pair_cases(review):
    if "cases" not in review:
        return
    cases = {}
    for case in review["cases"]:
        cases.setdefault(case["candidate_id"], set()).add(case["case_id"])
    for pair in review.get("comparisons", []):
        common = sorted(cases.get(pair["left"], set()) & cases.get(pair["right"], set()))
        pair["case_ids"] = common
        pair["case_status"] = "common_successful_cases" if common else "no_common_cases"
