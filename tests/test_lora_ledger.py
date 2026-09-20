"""Behavioral tests for the ledger's identity, non-destructive writes and exports."""
import copy
import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.lora_ledger_core.storage import (Ledger, Conflict, LedgerError, new_id, new_review,
                                            atomic_json, read_json, digest)
from tools.lora_ledger_core.scanner import scan, apply_scan, link_reference, verify
from tools.lora_ledger_core.reviews import candidate_for, preference_pairs, import_report
from tools.lora_ledger_core.exporting import export_csv, export_bundle
from tools.lora_ledger_core.features import extract


def checkpoint(path, session="1", name="old_name", epoch="10", lr="0.0001", payload=b"tensor"):
    metadata = {"ss_session_id": session, "ss_training_started_at": "100." + session,
                "ss_output_name": name, "ss_epoch": epoch, "ss_steps": "100",
                "ss_learning_rate": lr, "ss_network_dim": "4", "ss_network_alpha": "1",
                "ss_full_fp16": "False", "ss_seed": "0", "ss_unet_lr": "None",
                "ss_dataset_dirs": '{"dataset_A":{"n_repeats":20}}'}
    header = json.dumps({"__metadata__": metadata}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "outputs"
        self.source.mkdir()
        self.ledger = Ledger.create(self.root / "ledger")
        self.add_source(self.source)

    def add_source(self, path):
        config, roots = self.ledger.config(), self.ledger.locations()
        root_id = new_id()
        roots[root_id] = str(path)
        sources = config["sources"] + [{"id": new_id(), "root": root_id, "name": path.name,
                                       "enabled": True, "recursive": True}]
        self.ledger.set_sources(sources, roots, config["revision"])

    def register(self):
        result = scan(self.ledger)
        applied = apply_scan(self.ledger, result)
        self.assertEqual(applied["errors"], [])
        return self.ledger.list_runs()

    def test_renamed_checkpoint_uses_own_metadata_and_current_name(self):
        checkpoint(self.source / "corrected_name.safetensors")
        runs = self.register()
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["display_name"], "corrected_name")
        self.assertIn("old_name", run["aliases"])
        values = run["settings_sources"][run["settings"]["settings_id"]]["values"]
        self.assertEqual(values["learning_rate"], .0001)
        self.assertIsNone(values["unet_lr"])
        self.assertFalse(values["full_fp16"])
        self.assertEqual(values["seed"], 0)
        self.assertNotIn("avg_mode", values)

    def test_duplicate_scan_preserves_reviews_and_does_not_increment_revision(self):
        checkpoint(self.source / "new.safetensors")
        run = self.register()[0]
        run["user"]["experiment_note"] = "実験機能は不採用、LoRAは好き"
        run = self.ledger.save_run(run, run["revision"])
        review = new_review()
        review["annotations"] = [{"run_id": run["run_id"], "favorite_rating": 4, "good_points": "良い"}]
        self.ledger.save_review(review)
        before = digest(self.ledger.list_runs())
        result = scan(self.ledger)
        self.assertEqual([r["status"] for r in result["rows"]], ["unchanged"])
        self.assertEqual(apply_scan(self.ledger, result)["applied"], [])
        self.assertEqual(before, digest(self.ledger.list_runs()))
        self.assertEqual(self.ledger.reviews()[0]["annotations"][0]["favorite_rating"], 4)

    def test_new_checkpoint_and_report_added_to_same_run(self):
        checkpoint(self.source / "name.safetensors")
        run = self.register()[0]
        checkpoint(self.source / "name-000005.safetensors", epoch="5")
        (self.source / "rank_logs+name.txt").write_text("Epoch,TrainStep,Scope,RankSatP95\n1,1,unet,.5\n")
        result = scan(self.ledger)
        self.assertEqual(result["rows"][0]["status"], "additional")
        apply_scan(self.ledger, result)
        self.assertEqual(len(self.ledger.list_runs()), 1)
        self.assertEqual(len(self.ledger.get_run(run["run_id"])["artifacts"]), 2)

    def test_step_checkpoints_and_old_scan_cache_do_not_replace_final(self):
        checkpoint(self.source / "a-step00000100.safetensors", name="a", lr="0.0001")
        checkpoint(self.source / "a.safetensors", name="a", lr="0.0002")
        checkpoint(self.source / "b-step00000100.safetensors", name="b", session="2")
        a, b = sorted(self.register(), key=lambda r: r["display_name"])
        self.assertEqual([a["display_name"], b["display_name"]], ["a", "b"])
        self.assertEqual(b["artifacts"][0]["role"], "step")
        artifacts = {v["name"]: v for v in a["artifacts"]}
        step = artifacts["a-step00000100.safetensors"]
        self.assertEqual(step["role"], "step")
        self.assertEqual(artifacts["a.safetensors"]["role"], "final")
        self.assertEqual(a["settings_sources"][a["settings"]["settings_id"]]["values"]["learning_rate"], .0002)
        # Simulate a registered ledger and cache written before step recognition.
        a["display_name"] = "a-step00000100"
        step["role"] = "final"
        a["settings"]["settings_id"] = step["settings_id"]
        self.ledger.save_run(a, a["revision"])
        review = new_review()
        review["annotations"] = [{"run_id": a["run_id"], "artifact_id": step["artifact_id"], "favorite_rating": 4}]
        self.ledger.save_review(review)
        cache_path = self.ledger.directory / "cache/scan_index.json"
        cache = read_json(cache_path)
        cache["parser_version"] = 1
        for entry in cache["entries"].values():
            if "-step" in entry["relative_path"]:
                entry["data"]["role"] = "final"
        atomic_json(cache_path, cache)
        result = scan(self.ledger)
        self.assertEqual(apply_scan(self.ledger, result)["errors"], [])
        updated = self.ledger.get_run(a["run_id"])
        self.assertEqual(updated["display_name"], "a")
        restored = next(v for v in updated["artifacts"] if v["artifact_id"] == step["artifact_id"])
        self.assertEqual(restored["role"], "step")
        self.assertEqual(updated["settings_sources"][updated["settings"]["settings_id"]]["values"]["learning_rate"], .0002)
        self.assertEqual(self.ledger.reviews()[0]["annotations"][0]["artifact_id"], step["artifact_id"])
        self.assertTrue(all(r["status"] == "unchanged" for r in scan(self.ledger)["rows"]))

    def test_same_metadata_name_different_sessions_never_merge(self):
        checkpoint(self.source / "correct_A.safetensors", session="1")
        checkpoint(self.source / "correct_B.safetensors", session="2")
        (self.source / "gradient_logs+old_name.txt").write_text("Epoch,Step,Loss\n0,0,.1\n")
        result = scan(self.ledger)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(len(result["related"]), 1)

    def test_unrelated_same_name_manifest_does_not_block_metadata(self):
        checkpoint(self.source / "renamed.safetensors")
        directory = self.source / "run_records" / new_id()
        directory.mkdir(parents=True)
        atomic_json(directory / "manifest.json", {"schema_version": 1, "kind": "training_settings",
                    "run_id": directory.name, "output_name": "old_name", "session_id": "2",
                    "training_started_at": "100.2", "settings_status": "requested",
                    "requested_args": "inputs/requested_args.json", "resolved_config": "inputs/resolved_config.json"})
        runs = self.register()
        run = next(r for r in runs if r["display_name"] == "renamed")
        self.assertEqual(run["settings"]["source"], "metadata")

    def test_overwrite_creates_new_artifact_and_preserves_old_review(self):
        path = self.source / "name.safetensors"
        checkpoint(path)
        run = self.register()[0]
        old_id = run["artifacts"][0]["artifact_id"]
        review = new_review()
        review["candidates"] = [candidate_for(run, run["artifacts"][0], .8)]
        review["annotations"] = [{"run_id": run["run_id"], "artifact_id": old_id, "favorite_rating": 5}]
        self.ledger.save_review(review)
        checkpoint(path, payload=b"replacement tensor contents")
        result = scan(self.ledger)
        self.assertEqual(result["rows"][0]["status"], "changed")
        apply_scan(self.ledger, result, [run["run_id"]])
        updated = self.ledger.get_run(run["run_id"])
        self.assertEqual(len(updated["artifacts"]), 2)
        self.assertEqual(updated["artifacts"][0]["ref"]["status"], "changed")
        self.assertEqual(self.ledger.reviews()[0]["annotations"][0]["artifact_id"], old_id)

    def test_overlap_sources_and_profiler_snapshots(self):
        child = self.source / "sub"
        child.mkdir()
        checkpoint(child / "run.safetensors")
        checkpoint(self.source / "dq_dataset_profiler" / "trial" / "snapshot.safetensors")
        self.add_source(child)
        result = scan(self.ledger)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["artifact_count"], 1)

    def test_change_between_scan_and_apply_is_rejected(self):
        path = self.source / "run.safetensors"
        checkpoint(path)
        result = scan(self.ledger)
        checkpoint(path, payload=b"changed contents")
        applied = apply_scan(self.ledger, result)
        self.assertEqual(applied["applied"], [])
        self.assertEqual(len(applied["errors"]), 1)

    def test_revision_conflict_and_old_history(self):
        run = self.ledger.create_note_run("note")
        other = copy.deepcopy(run)
        run["user"]["experiment_note"] = "first"
        self.ledger.save_run(run, run["revision"])
        with self.assertRaises(Conflict):
            self.ledger.save_run(other, other["revision"])
        self.assertTrue((self.ledger.directory / "history" / "runs" / run["run_id"] / "000001.json").exists())

    def test_review_correction_keeps_history_and_null_rating(self):
        run = self.ledger.create_note_run("note")
        review = new_review()
        review["annotations"] = [{"run_id": run["run_id"], "favorite_rating": 1}]
        saved = self.ledger.save_review(review)
        saved["annotations"][0]["favorite_rating"] = None
        self.ledger.save_review(saved, saved["revision"])
        self.assertEqual(len(self.ledger.reviews()), 1)
        self.assertEqual(len(self.ledger.reviews(history=True)), 2)
        self.assertIsNone(self.ledger.reviews()[0]["annotations"][0]["favorite_rating"])

    def test_winner_only_creates_observed_pairs(self):
        candidates = [{"candidate_id": v} for v in "ABC"]
        pairs = preference_pairs(candidates, ["A", "B"], {"overall": "B"})
        self.assertEqual([(p["left"], p["right"]) for p in pairs], [("B", "A")])
        self.assertEqual(preference_pairs(candidates, ["A", "B", "C"], {}), [])
        ties = preference_pairs(candidates, ["A", "B"], {"overall": "tie"})
        self.assertEqual(ties[0]["result"], "tie")

    def test_source_move_and_path_escape(self):
        checkpoint(self.source / "name.safetensors")
        run = self.register()[0]
        moved = self.root / "moved"
        self.source.rename(moved)
        config, roots = self.ledger.config(), self.ledger.locations()
        roots[config["sources"][0]["root"]] = str(moved)
        self.ledger.set_sources(config["sources"], roots, config["revision"])
        self.assertTrue(self.ledger.resolve_ref(run["artifacts"][0]["ref"]).exists())
        with self.assertRaises(LedgerError):
            self.ledger.resolve_ref({"root": config["sources"][0]["root"], "relative_path": "../escape"})

    def test_csv_is_snapshot_and_escapes_formula_text(self):
        run = self.ledger.create_note_run("note")
        run["user"]["experiment_note"] = "=1+1\n日本語"
        self.ledger.save_run(run, run["revision"])
        path = self.root / "list.csv"
        export_csv(self.ledger, path)
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["学習の目的"], "'=1+1\n日本語")
        path.write_text("changed in Excel")
        self.assertEqual(self.ledger.get_run(run["run_id"])["user"]["experiment_note"], "=1+1\n日本語")

    def test_evaluation_csv_keeps_usability_adoption_and_history(self):
        checkpoint(self.source / "a.safetensors")
        run = self.register()[0]
        review = new_review()
        candidate = candidate_for(run, run["artifacts"][0], strength=0.8, lbw="preset")
        candidate.update(usability="usable", usability_reason="usable independently")
        review["candidates"] = [candidate]
        review["adoption"] = {"status": "none", "candidate_ids": []}
        first = self.ledger.save_review(review)
        path = self.root / "details.csv"
        self.assertEqual(export_csv(self.ledger, path, "evaluations"), 1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["row_type"], "candidate")
        self.assertEqual(row["usability"], "usable")
        self.assertEqual(row["usability_reason"], "usable independently")
        self.assertEqual(row["adoption_status"], "none")
        self.assertEqual(row["adopted"], "False")
        self.assertEqual(row["run_id"], run["run_id"])
        self.assertEqual(row["artifact_id"], run["artifacts"][0]["artifact_id"])
        self.assertEqual(row["strength"], "0.8")
        self.assertEqual(row["lbw"], "preset")
        first["adoption"] = {"status": "selected", "candidate_ids": [candidate["candidate_id"]]}
        self.ledger.save_review(first, first["revision"])
        self.assertEqual(export_csv(self.ledger, path, "evaluations"), 1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["revision"], "2")
        self.assertEqual(row["adopted"], "True")
        self.assertEqual(export_csv(self.ledger, path, "evaluations", history=True), 2)

    def test_evaluation_csv_keeps_comparison_targets_and_all_column_definitions(self):
        checkpoint(self.source / "a.safetensors", name="a")
        checkpoint(self.source / "b.safetensors", name="b", session="2")
        a, b = sorted(self.register(), key=lambda r: r["display_name"])
        review = new_review()
        left, right = [candidate_for(r, r["artifacts"][0], strength=0.7) for r in (a, b)]
        right["components"].append({**left["components"][0], "strength": 0.3})
        right.update(usability="conditional", usability_reason="combined LoRAs")
        review["annotations"] = [{"run_id": a["run_id"], "favorite_rating": 4}]
        review["candidates"] = [left, right]
        review["viewed_candidate_ids"] = [c["candidate_id"] for c in review["candidates"]]
        review["comparisons"] = preference_pairs(review["candidates"], review["viewed_candidate_ids"],
                                                 {"overall": right["candidate_id"]})
        self.ledger.save_review(review)
        unrelated = self.ledger.create_note_run("unrelated")
        other = new_review()
        other["annotations"] = [{"run_id": unrelated["run_id"], "favorite_rating": 1}]
        self.ledger.save_review(other)
        path = self.root / "details.csv"
        export_csv(self.ledger, path, "evaluations", [a["run_id"]])
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(read_json(str(path) + ".columns.json")["columns"], reader.fieldnames)
        self.assertEqual({r["review_id"] for r in rows}, {review["review_id"]})
        self.assertEqual({r["row_type"] for r in rows}, {"annotation", "candidate", "comparison"})
        candidates = {r["candidate_id"]: r for r in rows if r["row_type"] == "candidate"}
        self.assertEqual(candidates[left["candidate_id"]]["artifact_id"], a["artifacts"][0]["artifact_id"])
        compound = candidates[right["candidate_id"]]
        self.assertEqual(compound["artifact_id"], "")
        self.assertEqual(json.loads(compound["components"]), right["components"])
        self.assertEqual(compound["adopted"], "")
        pair = next(r for r in rows if r["row_type"] == "comparison")
        self.assertEqual(pair["left_name"], right["name"])
        self.assertEqual(pair["right_name"], left["name"])

    def test_feature_cutoff_and_scope_separation(self):
        path = self.source / "dq.txt"
        text = "Epoch,TrainStep,Scope,Target,MetricVersion,Mode,Stat,Granularity,Bits,QuantErrRatioEMA,QuantErrRMSEMA,RMS,ClipRateEMA,RangeMul\n"
        for epoch in range(1, 11):
            for target in ("delta", "guard"):
                value = epoch if epoch <= 5 else 100000
                text += f"{epoch},{epoch},unet,{target},1,stoch,rms,channel,8,{value},.1,1,.02,2\n"
        path.write_text(text)
        features, issues, _ = extract(path, "dq", cutoff=5)
        self.assertFalse(issues)
        ratio = [r for r in features if r["metric_id"] == "dq_error_ratio.tail_median"]
        self.assertEqual(len(ratio), 2)
        self.assertEqual({r["value"] for r in ratio}, {5})
        self.assertTrue(all(max(r["epochs"]) <= 5 for r in ratio))

    def test_transaction_recovery_and_bad_record_isolation(self):
        run = self.ledger.create_note_run("note")
        updated = copy.deepcopy(run)
        updated["display_name"] = "recovered"
        atomic_json(self.ledger.directory / ".transaction.json",
                    {"schema_version": 1, "kind": "ledger_transaction",
                     "writes": {f"runs/{run['run_id']}.json": updated}})
        reopened = Ledger(self.ledger.directory)
        self.assertEqual(reopened.get_run(run["run_id"])["display_name"], "recovered")
        (reopened.directory / "runs" / "broken.json").write_text("{")
        self.assertEqual(len(reopened.list_runs()), 1)
        self.assertEqual(len(reopened.errors), 1)

    def test_export_omits_unconfirmed_log_features_preserves_notes(self):
        checkpoint(self.source / "name.safetensors")
        (self.source / "gradient_logs+name.txt").write_text("Epoch,Step,Loss\n0,0,.1\n")
        run = self.register()[0]
        before = digest(self.ledger.list_runs())
        output = export_bundle(self.ledger)
        manifest = read_json(output / "export_manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertGreater(manifest["issues_count"], 0)
        self.assertEqual(before, digest(self.ledger.list_runs()))


if __name__ == "__main__":
    unittest.main()
