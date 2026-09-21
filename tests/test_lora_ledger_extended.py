"""Acceptance coverage for optional notes, provenance and real comparison windows."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path
import test_lora_ledger as fixtures
checkpoint = fixtures.checkpoint
from tools.lora_ledger import main as ledger_cli
from tools.lora_ledger_core.storage import new_review, read_json, atomic_json, sha256_file, Cancelled
from tools.lora_ledger_core.scanner import scan, apply_scan, verify
from tools.lora_ledger_core.reviews import candidate_for, preference_pairs, import_report, fingerprint_review
from tools.lora_ledger_core.notes import save_note
from tools.lora_ledger_core.features import extract, auto_events
from tools.lora_ledger_core.exporting import export_bundle

# Reuse setup helpers without inheriting the base module's test methods.
class ExtendedTests(unittest.TestCase):
    setUp = fixtures.LedgerTests.setUp
    add_source = fixtures.LedgerTests.add_source
    register = fixtures.LedgerTests.register

    def test_empty_and_purpose_only_do_not_create_review(self):
        checkpoint(self.source / "a.safetensors")
        run = self.register()[0]
        save_note(self.ledger, run, None, {})
        self.assertEqual(self.ledger.reviews(), [])
        save_note(self.ledger, run, None, {"experiment_note": "機能は不採用"})
        self.assertEqual(self.ledger.reviews(), [])
        self.assertEqual(self.ledger.get_run(run["run_id"])["user"]["experiment_note"], "機能は不採用")

    def test_notes_stay_at_checkpoint_and_reassessment_is_one_vote(self):
        checkpoint(self.source / "a-000010.safetensors")
        checkpoint(self.source / "a.safetensors", epoch="20")
        run = self.register()[0]
        artifact = run["artifacts"][0]
        first = save_note(self.ledger, run, artifact["artifact_id"],
                          {"favorite_rating": 5}, confirm_hash=False)
        self.assertEqual(first["annotations"][0]["artifact_id"], artifact["artifact_id"])
        self.assertEqual(first["comparisons"], [])
        second = save_note(self.ledger, run, artifact["artifact_id"],
                           {"favorite_rating": 3}, first, reassess=True, confirm_hash=False)
        self.assertEqual(second["reassessment_of"], first["review_id"])
        self.assertEqual(len(self.ledger.reviews()), 1)
        self.assertEqual(len(self.ledger.reviews(history=True)), 2)

    def test_full_verify_does_not_duplicate_sources_on_rescan(self):
        checkpoint(self.source / "a.safetensors", name="a")
        (self.source / "gradient_logs+a.txt").write_text("Epoch,Step,Loss\n0,0,1\n")
        run = self.register()[0]
        verify(self.ledger, full=True)
        again = scan(self.ledger)
        self.assertEqual(again["rows"][0]["status"], "unchanged")
        self.assertEqual(len(again["rows"][0]["run"]["source_refs"]), 1)

    def test_cli_restores_removed_source_without_changing_existing_refs(self):
        checkpoint(self.source / "a.safetensors", name="a")
        run = self.register()[0]
        original = run["artifacts"][0]
        config, roots = self.ledger.config(), self.ledger.locations()
        self.ledger.set_sources([], {}, config["revision"])
        checkpoint(self.source / "a-000020.safetensors", name="a", epoch="20")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ledger_cli(["register", "--ledger", str(self.ledger.directory),
                        "--source", str(self.source), "--source", str(self.source)])
        self.assertEqual(len(self.ledger.config()["sources"]), 1)
        self.assertEqual(self.ledger.config()["sources"][0]["root"], original["ref"]["root"])
        self.assertEqual(self.ledger.locations(), roots)
        updated = self.ledger.get_run(run["run_id"])
        self.assertEqual(len(updated["artifacts"]), 2)
        restored = next(a for a in updated["artifacts"] if a["artifact_id"] == original["artifact_id"])
        self.assertEqual(restored["ref"], original["ref"])

    def test_cli_reenables_disabled_source_and_preserves_options(self):
        config, roots = self.ledger.config(), self.ledger.locations()
        source = config["sources"][0]
        source.update(enabled=False, recursive=False, exclude=["ignored"])
        self.ledger.set_sources([source], roots, config["revision"])
        checkpoint(self.source / "a.safetensors", name="a")
        self.assertEqual(scan(self.ledger)["rows"], [])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ledger_cli(["register", "--ledger", str(self.ledger.directory),
                        "--source", str(self.source)])
            ledger_cli(["register", "--ledger", str(self.ledger.directory),
                        "--source", str(self.source)])
        source["enabled"] = True
        self.assertEqual(self.ledger.config()["sources"], [source])
        self.assertEqual(self.ledger.locations(), roots)
        self.assertEqual(len(self.ledger.list_runs()), 1)
        self.assertEqual(len(self.ledger.list_runs()[0]["artifacts"]), 1)

    def test_verify_unavailable_refs_only_saves_status_transitions(self):
        checkpoint(self.source / "a.safetensors", name="a")
        run = self.register()[0]
        history = self.ledger.directory / "history/runs" / run["run_id"]
        for error, status in [(FileNotFoundError, "missing"), (PermissionError, "unreadable")]:
            with self.subTest(status=status):
                before = self.ledger.get_run(run["run_id"])
                with patch("tools.lora_ledger_core.scanner.file_stat", side_effect=error):
                    self.assertEqual(verify(self.ledger)[0]["status"], status)
                    changed = self.ledger.get_run(run["run_id"])
                    self.assertEqual(changed["revision"], before["revision"] + 1)
                    snapshots = sorted(history.glob("*.json"))
                    self.assertEqual(verify(self.ledger)[0]["status"], status)
                    self.assertEqual(verify(self.ledger, full=True)[0]["status"], status)
                self.assertEqual(self.ledger.get_run(run["run_id"]), changed)
                self.assertEqual(sorted(history.glob("*.json")), snapshots)
        self.assertEqual(verify(self.ledger)[0]["status"], "exists")
        recovered = self.ledger.get_run(run["run_id"])
        self.assertEqual(recovered["revision"], changed["revision"] + 1)
        verify(self.ledger)
        self.assertEqual(self.ledger.get_run(run["run_id"]), recovered)

    def test_scan_packs_settings_and_reuses_header_cache(self):
        checkpoint(self.source / "a.safetensors")
        self.register()
        cache = read_json(self.ledger.directory / "cache/scan_index.json")
        self.assertTrue(cache["settings"])
        with patch("tools.lora_ledger_core.scanner.parse_entry", side_effect=AssertionError("reparsed")):
            self.assertEqual(scan(self.ledger)["rows"][0]["status"], "unchanged")

    def test_offline_root_keeps_review_and_does_not_mark_all_missing(self):
        checkpoint(self.source / "a.safetensors")
        run = self.register()[0]
        save_note(self.ledger, run, None, {"favorite_rating": 4}, confirm_hash=False)
        self.source.rename(self.root / "offline")
        result = scan(self.ledger)
        self.assertTrue(any(i["status"] == "offline" for i in result["issues"]))
        self.assertEqual(result["rows"][0]["status"], "unchanged")
        self.assertEqual(len(self.ledger.reviews()), 1)

    def test_offline_log_root_does_not_block_online_checkpoint_registration(self):
        checkpoint(self.source / "a.safetensors", name="a")
        logs = self.root / "separate_logs"
        logs.mkdir()
        (logs / "gradient_logs+a.txt").write_text("Epoch,Step,Loss\n0,0,1\n")
        self.add_source(logs)
        run = self.register()[0]
        save_note(self.ledger, run, None, {"favorite_rating": 4}, confirm_hash=False)
        previous = self.ledger.get_run(run["run_id"])
        offline = self.root / "offline_logs"
        self.assertTrue(logs.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(offline.resolve().is_relative_to(self.root.resolve()))
        logs.rename(offline)
        checkpoint(self.source / "a-000020.safetensors", name="a", epoch="20")
        result = scan(self.ledger)
        self.assertEqual(result["rows"][0]["status"], "additional")
        self.assertTrue(any(i["status"] == "offline" for i in result["issues"]))
        applied = apply_scan(self.ledger, result)
        self.assertEqual(applied["errors"], [])
        updated = self.ledger.get_run(run["run_id"])
        self.assertEqual(len(updated["artifacts"]), 2)
        self.assertEqual(updated["source_refs"], previous["source_refs"])
        self.assertEqual(self.ledger.reviews()[0]["annotations"][0]["favorite_rating"], 4)
        # A file that was online during scan must still be checked at apply time.
        path = self.source / "a-000030.safetensors"
        checkpoint(path, name="a", epoch="30")
        result = scan(self.ledger)
        checkpoint(path, name="a", epoch="30", payload=b"changed after scan")
        self.assertTrue(apply_scan(self.ledger, result)["errors"])
        self.assertEqual(self.ledger.get_run(run["run_id"]), updated)

    def test_unresolved_report_and_failed_job_are_preserved(self):
        folder = self.root / "report"
        folder.mkdir()
        (folder / "good.png").write_bytes(b"image")
        atomic_json(folder / "metadata.json", {
            "conditions": [{"id": "A", "name": "A", "items": [{"path": "lost.safetensors", "strength": 1}]},
                           {"id": "B", "name": "B", "items": []}],
            "jobs": [{"condition_id": "A", "prompt_id": "p1", "status": "done", "returncode": 0, "image": "good.png"},
                     {"condition_id": "B", "prompt_id": "p1", "status": "failed", "returncode": 1, "image": "good.png"}]})
        review = import_report(self.ledger, folder)
        self.assertEqual(len(review["cases"]), 1)
        review["candidates"][0]["usability"] = "usable"
        saved = self.ledger.save_review(review)
        self.assertEqual(saved["candidates"][0]["components"][0]["identity_status"], "unresolved")

    def test_hash_check_detects_same_stat_replacement(self):
        path = self.source / "a.safetensors"
        checkpoint(path, payload=b"first!")
        run = self.register()[0]
        review = new_review()
        review["candidates"] = [candidate_for(run, run["artifacts"][0])]
        first = fingerprint_review(self.ledger, review)
        st = path.stat()
        path.write_bytes(path.read_bytes()[:-6] + b"second")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        second = fingerprint_review(self.ledger, first)
        self.assertEqual(second["candidates"][0]["components"][0]["identity_status"], "changed")

    def test_rank_scopes_do_not_share_module_set(self):
        path = self.source / "rank_logs+a.txt"
        rows = ["Epoch,TrainStep,Scope,Module,RankSat,RankTop1,RankEnergy"]
        for epoch in range(1, 7):
            rows += [f"{epoch},{epoch},unet,u,0.5,0.8,1", f"{epoch},{epoch},te,t,0.2,0.3,1"]
        path.write_text("\n".join(rows))
        values, issues, _ = extract(path, "rank")
        self.assertFalse(issues)
        self.assertTrue(all(v["status"] == "ok" for v in values))
        self.assertEqual({v["scope"] for v in values}, {"unet", "te"})

    def test_missing_epochs_are_not_removed_from_window(self):
        path = self.source / "gradient_logs+a.txt"
        path.write_text("Epoch,Step,Loss\n0,0,1\n1,1,2\n2,2,3\n3,3,4\n4,4,nan\n")
        values, _, _ = extract(path, "gradient")
        loss = next(v for v in values if v["metric_id"] == "loss.tail_median")
        self.assertEqual(loss["tail_epochs"], [5.0])
        self.assertIsNone(loss["value"])
        self.assertEqual(loss["n_missing"], 1)

    def test_auto_events_no_invented_checkpoint_epoch(self):
        path = self.source / "dq_delta_auto+a.txt"
        path.write_text("TrainStep,Scope,Target,AutoApplied,ClipRateLowAutoDecision\n1,unet,delta,1,escape_to_mid\n2,unet,delta,0,hold\n")
        values, _, _ = auto_events(path)
        self.assertEqual({v["metric_id"]: v["value"] for v in values},
                         {"dq_auto.records": 2, "dq_auto.applied": 1, "dq_auto.escape_recorded": 1})
        values, issues, _ = auto_events(path, cutoff=5)
        self.assertEqual(values, [])
        self.assertTrue(issues)

    def test_comparison_common_interval_excludes_later_epochs(self):
        for i, name in enumerate(("a", "b")):
            checkpoint(self.source / (name + ".safetensors"), session=str(i + 1), name=name, epoch="8" if i else "6")
            rows = ["Epoch,Step,Loss,Gradient Norm,Scale"] + [
                f"{e-1},{e},{e if e <= 6 else 999},1,1" for e in range(1, 9)]
            (self.source / ("gradient_logs+" + name + ".txt")).write_text("\n".join(rows))
        runs = self.register()
        for run in runs:
            run["source_refs"][0]["association"]["status"] = "user_asserted"
            self.ledger.save_run(run, run["revision"])
        review = new_review()
        review["candidates"] = [candidate_for(run, run["artifacts"][0]) for run in runs]
        ids = [c["candidate_id"] for c in review["candidates"]]
        review["comparisons"] = preference_pairs(review["candidates"], ids, {"overall": ids[0]})
        self.ledger.save_review(review)
        before = [read_json(p) for p in (self.ledger.directory / "runs").glob("*.json")]
        output = export_bundle(self.ledger)
        with (output / "features.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        common = [r for r in rows if r["window_id"] == "common_interval" and r["metric_id"] == "loss.tail_median"]
        self.assertEqual(len(common), 2)
        self.assertEqual({float(r["value"]) for r in common}, {5.5})
        self.assertTrue(all(max(json.loads(r["epochs"])) == 6 for r in common))
        self.assertEqual(before, [read_json(p) for p in (self.ledger.directory / "runs").glob("*.json")])

    def test_diagnostic_gradient_fallback_is_explicit(self):
        path = self.source / "a_diagnostic.json"
        atomic_json(path, {"grad": {"epochs": [1, 2, 3, 4, 5, 6],
                                  "loss": [1, 2, 3, 4, 5, 6], "gradient_norm": [1]*6, "scale": [1]*6}})
        values, _, _ = extract(path, "gradient", cutoff=5, diagnostic=True)
        self.assertTrue(all(v["step_origin"] == "diagnostic_array_index" for v in values))
        self.assertTrue(all(max(v["epochs"]) == 5 for v in values))

    def test_cancel_leaves_saved_data_intact(self):
        checkpoint(self.source / "a.safetensors")
        run = self.register()[0]
        before = self.ledger.get_run(run["run_id"])
        with self.assertRaises(Cancelled):
            verify(self.ledger, True, cancel=lambda: True)
        self.assertEqual(before, self.ledger.get_run(run["run_id"]))

    def test_legacy_archive_hash_gate_and_metric_namespace(self):
        from tools.lora_ledger_core.research import read_archive, legacy_features
        path = self.source / "gradient_logs+a.txt"
        path.write_text("Epoch,Step,Loss\n0,1,1\n")
        folder = self.root / "research/data"
        folder.mkdir(parents=True)
        with (folder / "files.csv").open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=["id","path","status","sha256","code_hash","segments"])
            w.writeheader();w.writerow({"id":"1","path":str(path),"status":"ok","sha256":sha256_file(path),"code_hash":"abc","segments":"1"})
        (folder / "epoch_metrics.csv").write_text('file_id,segment,epoch,dimension,metric,n,missing,median\n1,0,0,{},Loss,1,0,1\n')
        self.ledger.set_research_archive(folder.parent)
        archive, audit, _ = read_archive(self.ledger, {os.path.normcase(str(path))})
        record = archive[os.path.normcase(str(path))]
        values, issues = legacy_features(record, sha256_file(path), "gradient")
        self.assertEqual(values[0]["metric_id"], "legacy.Loss.epoch_median")
        self.assertTrue(audit["files_sha256"])
        values, issues = legacy_features(record, "wrong", "gradient")
        self.assertEqual(values, [])
        self.assertTrue(issues)


    def test_switch_from_parent_source_to_child_preserves_ids(self):
        nested = self.source / "nested"
        checkpoint(nested / "a.safetensors")
        self.add_source(nested)
        run = self.register()[0]
        config = self.ledger.config()
        config["sources"][0]["enabled"] = False
        self.ledger.set_sources(config["sources"], self.ledger.locations(), config["revision"])
        result = scan(self.ledger)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["status"], "unchanged")
        self.assertEqual(len(result["rows"][0]["run"]["artifacts"]), 1)

    def test_settings_only_manifest_retains_values_without_completion(self):
        from library.training_settings import start_record
        start_record(self.source, "settings_only", "99", "123", {"learning_rate": 0.0003, "seed": 0})
        run = self.register()[0]
        self.assertEqual(run["training_status"], "unknown")
        self.assertEqual(run["artifacts"], [])
        values = run["settings_sources"][run["settings"]["settings_id"]]["values"]
        self.assertEqual(values["learning_rate"], 0.0003)
        self.assertTrue(any(ref["kind"] == "settings_payload" and ref["observed"].get("sha256")
                            for ref in run["source_refs"]))

    def test_settings_only_run_refreshes_when_resolved_record_arrives(self):
        from types import SimpleNamespace
        from library.training_settings import start_record, finish_record
        record = start_record(self.source, "pending", "99", "123",
                              {"max_train_steps": None, "dataset_config": None})
        first = self.register()[0]
        old_settings_id = first["settings"]["settings_id"]
        save_note(self.ledger, first, None,
                  {"dataset_name": "my dataset", "dataset_name_manual": True, "favorite_rating": 4},
                  confirm_hash=False)
        reviews_before = self.ledger.reviews()
        finish_record(record, SimpleNamespace(max_train_steps=1200, dataset_config="resolved.toml"),
                      SimpleNamespace(param_groups=[]), [], [], {}, {})
        result = scan(self.ledger)
        self.assertEqual(result["rows"][0]["status"], "changed")
        self.assertEqual(self.ledger.get_run(first["run_id"])["settings"]["status"], "partial")
        applied = apply_scan(self.ledger, result, [first["run_id"]])
        self.assertEqual(applied["errors"], [])
        self.assertEqual(applied["applied"], [first["run_id"]])
        updated = self.ledger.get_run(first["run_id"])
        source = updated["settings_sources"][updated["settings"]["settings_id"]]
        self.assertEqual(updated["settings"]["status"], "recorded")
        self.assertEqual(source["values"]["max_train_steps"], 1200)
        self.assertEqual(source["values"]["dataset_config"], "resolved.toml")
        self.assertEqual(source["source_ref"]["observed"]["sha256"], sha256_file(record[0] / "manifest.json"))
        self.assertEqual(updated["settings_sources"][old_settings_id], first["settings_sources"][old_settings_id])
        self.assertEqual(updated["dataset"]["display_name"], "my dataset")
        self.assertEqual(updated["dataset"]["automatic"]["display_name"], "resolved.toml")
        self.assertEqual(updated["training_status"], "unknown")
        self.assertEqual(self.ledger.reviews(), reviews_before)
        self.assertEqual(scan(self.ledger)["rows"][0]["status"], "unchanged")

    def test_dq_unknown_version_retains_values_with_restriction(self):
        path = self.source / "dq.txt"
        path.write_text("Epoch,TrainStep,Scope,Target,QuantErrRatioEMA\n" +
                        "".join(f"{e},{e},unet,delta,0.1\n" for e in range(1,7)))
        values, _, _ = extract(path,"dq")
        value = next(v for v in values if v["metric_id"] == "dq_error_ratio.tail_median")
        self.assertEqual(value["status"], "metric_version_unrecorded")
        self.assertEqual(value["value"], 0.1)


    def test_full_hash_copy_scan_is_idempotent_and_rename_keeps_target(self):
        import shutil
        path = self.source / "a.safetensors"
        checkpoint(path)
        run = self.register()[0]
        verify(self.ledger, full=True)
        shutil.copyfile(path, self.source / "copy.safetensors")
        result = scan(self.ledger, full=True)
        apply_scan(self.ledger, result)
        again = scan(self.ledger, full=True)
        self.assertEqual(again["rows"][0]["status"], "unchanged")
        self.assertEqual(len(again["rows"][0]["run"]["artifacts"]), 1)
        path.unlink()
        result = scan(self.ledger, full=True)
        apply_scan(self.ledger, result, [run["run_id"]])
        updated = self.ledger.get_run(run["run_id"])
        self.assertEqual(updated["artifacts"][0]["artifact_id"], run["artifacts"][0]["artifact_id"])
        self.assertEqual(updated["artifacts"][0]["name"], "copy.safetensors")


    def test_unrecorded_auto_fields_are_not_zero(self):
        path = self.source / "auto.txt"
        path.write_text("TrainStep,Scope,Target\n1,unet,delta\n")
        values, _, _ = auto_events(path)
        indexed = {v["metric_id"]: v for v in values}
        self.assertEqual(indexed["dq_auto.records"]["value"],1)
        self.assertIsNone(indexed["dq_auto.applied"]["value"])
        self.assertIsNone(indexed["dq_auto.escape_recorded"]["value"])
