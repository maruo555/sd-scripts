"""Save the five optional fields without manufacturing empty evaluation events."""
from __future__ import annotations
import copy
from .storage import LedgerError, new_review, new_id, now, latest_annotations
from .presentation import refresh_dataset
from .reviews import candidate_for, fingerprint_review


def save_note(ledger, run, artifact_id, fields, previous=None, reassess=False,
              confirm_hash=True, progress=None, cancel=None):
    updated = copy.deepcopy(run)
    updated.setdefault("user", {})["experiment_note"] = fields.get("experiment_note", "")
    name = fields.get("display_name", run["display_name"]).strip()
    if not name:
        raise LedgerError("表示名を入力してください")
    if name != run["display_name"]:
        updated["display_name"] = name
        updated["user"]["display_name_custom"] = True
    refresh_dataset(updated)
    dataset_name = fields.get("dataset_name", updated["dataset"].get("display_name", "")).strip()
    manual = fields.get("dataset_name_manual")
    if manual is None:
        manual = (updated["dataset"]["display_name_custom"] or
                  ("dataset_name" in fields and dataset_name != updated["dataset"]["display_name"]))
    updated["dataset"]["display_name_custom"] = bool(manual)
    if manual:
        updated["dataset"].update(display_name=dataset_name, name_source="manual")
    else:
        updated["dataset"]["display_name"] = updated["dataset"]["automatic"]["display_name"]
        updated["dataset"]["name_source"] = updated["dataset"]["automatic"]["name_source"]
    if "purpose" in fields:
        updated["purpose"] = fields["purpose"]
    if "training_status" in fields and fields["training_status"] != run.get("training_status"):
        updated["training_status"] = fields["training_status"]
        updated["training_status_evidence"] = {"source": "user_asserted", "recorded_at": now()}
    run_update = (updated, run["revision"]) if updated != run else None
    annotation = {"run_id": run["run_id"], "artifact_id": artifact_id, "candidate_id": None,
                  "favorite_rating": fields.get("favorite_rating"),
                  **{k: fields.get(k, "").strip() for k in ("good_points", "bad_points", "free_note")}}
    has_note = annotation["favorite_rating"] is not None or any(annotation[k] for k in
                                                               ("good_points", "bad_points", "free_note"))
    if not has_note and previous is None:
        if run_update:
            ledger.save_run(*run_update)
        return {"saved_review": False}
    review = copy.deepcopy(previous) if previous else new_review()
    if reassess and previous:
        review.update(review_id=new_id(), revision=0, reassessment_of=previous["review_id"], evaluated_at=now())
    review["annotations"] = [annotation]
    review["evidence_mode"] = fields.get("evidence_mode", "current")
    review["evaluation_time_status"] = "recorded_now" if review["evidence_mode"] == "current" else "original_date_unknown"
    review["use"] = fields.get("use", ledger.config()["default_use"])
    if artifact_id and not review["candidates"]:
        artifact = next(a for a in run["artifacts"] if a["artifact_id"] == artifact_id)
        review["candidates"] = [candidate_for(run, artifact)]
    if confirm_hash and artifact_id:
        review = fingerprint_review(ledger, review, cancel, progress)
    if cancel and cancel():
        from .storage import Cancelled
        raise Cancelled("保存を中止しました")
    return ledger.save_review(review, review["revision"], run_update)
