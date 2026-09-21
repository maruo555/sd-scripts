"""Evidence-based dataset names, manual overrides, and chronological sorting."""
import copy
import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import test_lora_ledger as fixtures
from tools.lora_ledger_core.presentation import automatic_dataset, dataset_info, run_time
from tools.lora_ledger_core.notes import save_note
from tools.lora_ledger_core.storage import atomic_json, read_json
from tools.lora_ledger_core.scanner import scan, apply_scan
from tools.lora_ledger_core.exporting import listing_rows
from library.training_settings import start_record


class PresentationTests(unittest.TestCase):
    setUp = fixtures.LedgerTests.setUp
    add_source = fixtures.LedgerTests.add_source
    register = fixtures.LedgerTests.register

    def test_recorded_toml_name_wins_and_preserves_path(self):
        fixtures.checkpoint(self.source / "a.safetensors")
        start_record(self.source, "old_name", "1", "100.1",
                     {"dataset_config": r"D:\datasets\人物\character.toml", "config_file": "train.toml"})
        run = self.register()[0]
        info = dataset_info(run)
        self.assertEqual(info["display_name"], "character.toml")
        self.assertEqual(info["name_source"], "dataset_config")
        self.assertEqual(info["automatic"]["config_path"], r"D:\datasets\人物\character.toml")

    def test_empty_name_collision_suffixes_are_not_dataset_names(self):
        run = {"settings": {"settings_id": "s"}, "settings_sources": {
            "s": {"values": {"dataset_dirs": {"": {}, " (2)": {}, " (3)": {}}}}}}
        self.assertEqual(automatic_dataset(run)["display_name"], "名称不明（3サブセット）")

    def test_training_config_is_not_mislabelled_as_dataset_config(self):
        run = {"settings": {"settings_id": "s"}, "settings_sources": {
            "s": {"values": {"config_file": "training.toml"}}}}
        self.assertEqual(automatic_dataset(run)["display_name"], "名称不明")

    def test_manual_name_survives_rescan_and_can_return_to_auto(self):
        fixtures.checkpoint(self.source / "a.safetensors")
        record = start_record(self.source, "old_name", "1", "100.1",
                              {"dataset_config": "dataset_a.toml"})
        run = self.register()[0]
        save_note(self.ledger, run, None, {"dataset_name": "私の名前", "dataset_name_manual": True})
        args = record[0] / "inputs/requested_args.json"
        contents = read_json(args)
        contents["args"]["dataset_config"] = "dataset_b.toml"
        atomic_json(args, contents)
        result = scan(self.ledger)
        apply_scan(self.ledger, result, [run["run_id"]])
        updated = self.ledger.get_run(run["run_id"])
        self.assertEqual(dataset_info(updated)["display_name"], "私の名前")
        self.assertEqual(dataset_info(updated)["automatic"]["display_name"], "dataset_b.toml")
        self.assertEqual(scan(self.ledger)["rows"][0]["status"], "unchanged")
        save_note(self.ledger, updated, None, {"dataset_name_manual": False})
        self.assertEqual(dataset_info(self.ledger.get_run(run["run_id"]))["display_name"], "dataset_b.toml")
        self.assertFalse(self.ledger.reviews())

    def test_legacy_manual_names_and_explicit_blank_are_preserved(self):
        run = {"dataset": {"display_name": "昔の手入力", "identity_status": "user_asserted"}}
        self.assertEqual(dataset_info(run)["display_name"], "昔の手入力")
        run["dataset"].update(display_name="", display_name_custom=True)
        self.assertEqual(dataset_info(run)["display_name"], "")

    def test_datetime_uses_start_before_weight_mtime_and_does_not_use_registration(self):
        run = {"identity": {"started_at": "1700000000.5"}, "created_at": "2026-09-12T00:00:00Z",
               "artifacts": [{"artifact_id": "a", "name": "a.safetensors", "role": "final",
                              "ref": {"status": "exists", "observed": {"mtime_ns": "1800000000000000000"}}}]}
        self.assertEqual(run_time(run)["timestamp"], 1700000000.5)
        self.assertEqual(run_time(run)["source"], "training_started")
        run["identity"] = {}
        self.assertEqual(run_time(run)["source"], "checkpoint_mtime")
        self.assertEqual(run_time(run)["timestamp"], 1800000000)
        run["artifacts"] = []
        self.assertIsNone(run_time(run)["timestamp"])

    def test_settings_only_session_time_and_bad_metadata_time(self):
        run = {"identity": {"session_key": "7|1700000000.125"}}
        self.assertEqual(run_time(run)["timestamp"], 1700000000.125)
        self.assertIsNone(run_time({"identity": {"started_at": "nan"}})["timestamp"])

    def test_csv_contains_dataset_and_datetime_provenance(self):
        fixtures.checkpoint(self.source / "a.safetensors")
        self.register()
        row = listing_rows(self.ledger)[0]
        self.assertTrue(row["学習日時"])
        self.assertIn("学習開始", row["日時の出典"])
        self.assertEqual(row["データセット名の出典"], "metadata_dirs")

    def test_gui_sort_keeps_unknown_last_in_both_orders(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from tools.lora_ledger_core.gui import RunModel, RunFilter
        app = QApplication.instance() or QApplication([])
        base = self.ledger.create_note_run("base")
        records = []
        for name, value in (("later", "1700000000"), ("unknown", None), ("earlier", "1600000000")):
            run = copy.deepcopy(base)
            run.update(run_id=name, display_name=name, identity={"started_at": value})
            records.append(run)
        model = RunModel()
        model.replace(records, [])
        proxy = RunFilter()
        proxy.setSourceModel(model)
        proxy.setSortRole(RunModel.SORT_ROLE)
        proxy.sort(RunModel.TIME_COLUMN, Qt.SortOrder.AscendingOrder)
        self.assertEqual([proxy.data(proxy.index(i, 0)) for i in range(3)], ["earlier", "later", "unknown"])
        proxy.sort(RunModel.TIME_COLUMN, Qt.SortOrder.DescendingOrder)
        self.assertEqual([proxy.data(proxy.index(i, 0)) for i in range(3)], ["later", "earlier", "unknown"])
