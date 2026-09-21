"""Native Qt interaction tests with an offscreen platform, no production data writes."""
import os
import copy
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import patch
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog, QLineEdit
from PySide6.QtCore import QTimer
import test_lora_ledger as fixtures
checkpoint = fixtures.checkpoint
from tools.lora_ledger_core.gui import LedgerWindow
from tools.lora_ledger_core.dialogs import ComparisonDialog, TaskDialog
from tools.lora_ledger_core.storage import new_review, sha256_file, atomic_json
from tools.lora_ledger_core.reviews import candidate_for, preference_pairs, fingerprint_review, import_report


class GuiTests(unittest.TestCase):
    setUp_base = fixtures.LedgerTests.setUp
    add_source = fixtures.LedgerTests.add_source
    register = fixtures.LedgerTests.register

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.setUp_base()
        checkpoint(self.source / "a-000010.safetensors", name="a", epoch="10")
        checkpoint(self.source / "a.safetensors", name="a", epoch="20")
        checkpoint(self.source / "b.safetensors", session="2", name="b")
        self.runs = self.register()
        self.window = LedgerWindow(self.ledger, self.root / "local-state.json")
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.dirty = False
        self.window.close()
        self.app.processEvents()

    def test_save_and_switch_target_never_copy_stars(self):
        w = self.window
        w.rating.setCurrentIndex(w.rating.findData(4))
        self.assertTrue(w.save())
        self.assertEqual(w.rating.currentData(), 4)
        w.target.setCurrentIndex(1)
        self.assertIsNone(w.rating.currentData())
        w.good.setPlainText("破綻が少ない")
        self.assertTrue(w.save())
        w.target.setCurrentIndex(0)
        self.assertEqual(w.rating.currentData(), 4)
        self.assertEqual(w.good.toPlainText(), "")

    def test_unsaved_cancel_and_save_on_row_change(self):
        w = self.window
        original = w.current_run["run_id"]
        w.purpose.setPlainText("試験")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            w.table.setCurrentIndex(w.proxy.index(1, 0))
        self.assertEqual(w.current_run["run_id"], original)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Save):
            w.table.setCurrentIndex(w.proxy.index(1, 0))
        self.assertNotEqual(w.current_run["run_id"], original)
        self.assertEqual(self.ledger.get_run(original)["user"]["experiment_note"], "試験")
        self.assertEqual(self.ledger.reviews(), [])

    def test_star_filter_uses_all_target_ratings(self):
        w = self.window
        w.rating.setCurrentIndex(w.rating.findData(5))
        w.save()
        w.target.setCurrentIndex(1)
        w.rating.setCurrentIndex(w.rating.findData(2))
        w.save()
        w.star_filter.setCurrentIndex(w.star_filter.findData(2))
        self.assertEqual(w.proxy.rowCount(), 1)
        self.assertIn("複数評価", w.proxy.data(w.proxy.index(0, w.model.HEADERS.index("お気に入り"))))

    def test_comparison_edit_preserves_inputs_and_saves_revision(self):
        candidates = [candidate_for(r, r["artifacts"][0]) for r in self.runs]
        review = new_review()
        review["candidates"] = candidates
        ids = [c["candidate_id"] for c in candidates]
        review["viewed_candidate_ids"] = ids
        review["comparisons"] = preference_pairs(candidates, ids, {"overall": ids[0]})
        saved = self.ledger.save_review(review)
        dialog = ComparisonDialog(self.window, self.ledger, saved)
        self.assertEqual(dialog.choices["overall"].currentData(), ids[0])
        dialog.table.item(0, 4).setText("十分使える")
        dialog.remember()
        dialog.populate()
        self.assertEqual(dialog.table.item(0, 4).text(), "十分使える")
        dialog.hash_check.setChecked(False)
        dialog.save()
        self.assertEqual(self.ledger.reviews()[0]["revision"], 2)
        self.assertEqual(len(self.ledger.reviews(history=True)), 2)
        self.window.reload()
        self.assertTrue(all(row["reviewed"] for row in self.window.model.rows))
        dialog.close()

    def test_rebinding_candidate_refreshes_hash_and_preserves_history(self):
        a, b = sorted(self.runs, key=lambda run: run["display_name"])
        original = a["artifacts"][0]
        for confirm_hash, target_run, target in ((False, a, a["artifacts"][1]), (True, b, b["artifacts"][0])):
            with self.subTest(confirm_hash=confirm_hash):
                review = new_review()
                candidate = candidate_for(a, original, strength=0.8)
                candidate["usability"] = "usable"
                review["candidates"] = [candidate]
                saved = self.ledger.save_review(fingerprint_review(self.ledger, review))
                old_hash = saved["candidates"][0]["components"][0]["content_sha256"]
                dialog = ComparisonDialog(self.window, self.ledger, saved)
                self.addCleanup(dialog.close)
                dialog.table.setCurrentCell(0, 0)
                with patch("tools.lora_ledger_core.dialogs.choose_artifact", return_value=(a, original)):
                    dialog.resolve_candidate()
                self.assertEqual(dialog.review["candidates"][0]["components"][0]["content_sha256"], old_hash)
                with patch("tools.lora_ledger_core.dialogs.choose_artifact", return_value=(target_run, target)):
                    dialog.resolve_candidate()
                self.assertNotIn("content_sha256", dialog.review["candidates"][0]["components"][0])
                dialog.hash_check.setChecked(confirm_hash)
                dialog.save()
                updated = next(r for r in self.ledger.reviews() if r["review_id"] == saved["review_id"])
                component = updated["candidates"][0]["components"][0]
                self.assertEqual(updated["revision"], 2)
                self.assertEqual(component["artifact_id"], target["artifact_id"])
                self.assertEqual(component["strength"], 0.8)
                if confirm_hash:
                    self.assertEqual(component["identity_status"], "content_verified_at_review")
                    self.assertEqual(component["content_sha256"], sha256_file(self.ledger.resolve_ref(target["ref"])))
                else:
                    self.assertNotIn("content_sha256", component)
                    self.assertEqual(component["identity_status"], "user_asserted")
                previous = next(r for r in self.ledger.reviews(history=True)
                                if r["review_id"] == saved["review_id"] and r["revision"] == 1)
                self.assertEqual(previous["candidates"][0]["components"][0]["content_sha256"], old_hash)
                self.assertEqual(previous["candidates"][0]["components"][0]["artifact_id"], original["artifact_id"])
                dialog.close()

    def test_partial_comparison_edit_keeps_cases_and_allows_explicit_clear(self):
        review = new_review()
        review["candidates"] = [candidate_for(r, r["artifacts"][0]) for r in self.runs]
        ids = [c["candidate_id"] for c in review["candidates"]]
        review["viewed_candidate_ids"] = ids
        review["comparisons"] = preference_pairs(review["candidates"], ids, {"overall": ids[0]})
        review["cases"] = [{"candidate_id": c, "case_id": prompt, "prompt_id": prompt}
                           for c in ids for prompt in ("p1", "p2")]
        dialog = ComparisonDialog(self.window, self.ledger, review)
        dialog.selection.setCurrentIndex(dialog.selection.findData("selected"))
        dialog.case_filter.setText("p1")
        dialog.hash_check.setChecked(False)
        dialog.save()
        first = self.ledger.reviews()[0]
        self.assertEqual(len(first["cases"]), len(ids))
        self.assertEqual(first["comparisons"][0]["case_ids"], ["p1"])
        dialog.close()

        dialog = ComparisonDialog(self.window, self.ledger, first)
        self.assertEqual(dialog.case_filter.text(), "p1")
        dialog.reasons["overall"].setText("reason correction only")
        dialog.hash_check.setChecked(False)
        dialog.save()
        corrected = self.ledger.reviews()[0]
        self.assertEqual(corrected["revision"], 2)
        self.assertEqual(corrected["cases"], first["cases"])
        self.assertEqual(corrected["comparisons"][0]["case_ids"], ["p1"])
        dialog.close()

        dialog = ComparisonDialog(self.window, self.ledger, corrected)
        dialog.case_filter.clear()
        dialog.hash_check.setChecked(False)
        dialog.save()
        cleared = self.ledger.reviews()[0]
        self.assertNotIn("cases", cleared)
        self.assertNotIn("case_ids", cleared["comparisons"][0])
        self.assertTrue(cleared["case_selection_note"])
        self.assertEqual(self.ledger.reviews(history=True)[0]["cases"], first["cases"])
        dialog.close()

    def test_partial_comparison_can_expand_change_and_restore_all_cases(self):
        review = new_review()
        review["candidates"] = [candidate_for(r, r["artifacts"][0]) for r in self.runs]
        ids = [c["candidate_id"] for c in review["candidates"]]
        review["viewed_candidate_ids"] = ids
        review["comparisons"] = preference_pairs(review["candidates"], ids, {"overall": ids[0]})
        all_cases = [{"candidate_id": c, "case_id": p, "prompt_id": p}
                     for c in ids for p in ("p1", "p2")]
        review["cases"] = copy.deepcopy(all_cases)
        steps = [("selected", "p1", {"p1"}), ("selected", "p1, p2", {"p1", "p2"}),
                 ("selected", "p2", {"p2"}), ("all_cases", "p2", {"p1", "p2"}),
                 ("selected", "", None), ("all_cases", "", {"p1", "p2"})]
        for mode, prompt_text, expected in steps:
            dialog = ComparisonDialog(self.window, self.ledger, review)
            dialog.selection.setCurrentIndex(dialog.selection.findData(mode))
            dialog.case_filter.setText(prompt_text)
            dialog.hash_check.setChecked(False)
            dialog.save()
            dialog.close()
            review = self.ledger.reviews()[0]
            self.assertEqual(review["available_cases"], all_cases)
            if expected is None:
                self.assertNotIn("cases", review)
                self.assertNotIn("case_ids", review["comparisons"][0])
            else:
                self.assertEqual({c["prompt_id"] for c in review["cases"]}, expected)
                self.assertEqual(set(review["comparisons"][0]["case_ids"]), expected)
        dialog = ComparisonDialog(self.window, self.ledger, review)
        dialog.selection.setCurrentIndex(dialog.selection.findData("selected"))
        dialog.case_filter.setText("p1, typo")
        with patch("tools.lora_ledger_core.dialogs.show_error") as error:
            dialog.save()
        error.assert_called_once()
        self.assertEqual(self.ledger.reviews()[0], review)
        self.assertEqual(len(self.ledger.reviews(history=True)), len(steps))
        dialog.close()

    def test_legacy_partial_comparison_restores_only_the_original_report(self):
        folder = self.root / "comparison_report"
        folder.mkdir()
        (folder / "image.png").write_bytes(b"image")
        conditions = [{"id": str(i), "name": str(i), "items": [{"path": str(
            self.ledger.resolve_ref(r["artifacts"][0]["ref"]))}]} for i, r in enumerate(self.runs)]
        metadata = {"conditions": conditions, "jobs": [
            {"condition_id": c["id"], "prompt_id": p, "status": "done", "returncode": 0,
             "image": "image.png"} for c in conditions for p in ("p1", "p2")]}
        atomic_json(folder / "metadata.json", metadata)
        review = import_report(self.ledger, folder)
        ids = [c["candidate_id"] for c in review["candidates"]]
        review["viewed_candidate_ids"] = ids
        review["comparisons"] = preference_pairs(review["candidates"], ids, {"overall": ids[0]})
        review["selection_mode"] = "selected"
        review["cases"] = [c for c in review["cases"] if c["prompt_id"] == "p1"]
        original = self.ledger.save_review(review)
        dialog = ComparisonDialog(self.window, self.ledger, original)
        dialog.case_filter.setText("p1, p2")
        dialog.hash_check.setChecked(False)
        dialog.save()
        dialog.close()
        updated = self.ledger.reviews()[0]
        self.assertEqual({c["prompt_id"] for c in updated["cases"]}, {"p1", "p2"})
        self.assertEqual(len(updated["available_cases"]), len(ids) * 2)
        self.assertEqual(self.ledger.reviews(history=True)[0], original)
        metadata["created_at"] = "changed"
        atomic_json(folder / "metadata.json", metadata)
        dialog = ComparisonDialog(self.window, self.ledger, original)
        dialog.case_filter.setText("p1, p2")
        with patch("tools.lora_ledger_core.dialogs.show_error") as error:
            dialog.save()
        error.assert_called_once()
        self.assertIn("変更", str(error.call_args.args[1]))
        self.assertEqual(self.ledger.reviews()[0], updated)
        dialog.close()

    def test_strength_editor_preserves_split_strengths_and_validates_input(self):
        run = self.runs[0]
        for initial, text, expected in [(1.0, None, 1.0), ([.5, .9], None, [.5, .9]),
                ([.5, .7, 1.0], None, [.5, .7, 1.0]), (1.0, ".4, .8, 1", [.4, .8, 1.0]),
                (1.0, "", None), (1.0, "nan", 1.0), (1.0, "1, 2, 3, 4", 1.0)]:
            with self.subTest(initial=initial, text=text):
                review = new_review()
                candidate = candidate_for(run, run["artifacts"][0], initial, [.1, .2])
                candidate["usability"] = "usable"
                review["candidates"] = [candidate]
                saved = self.ledger.save_review(review)
                dialog = ComparisonDialog(self.window, self.ledger, saved)
                dialog.table.setCurrentCell(0, 2)
                def accept(editor):
                    if text is not None:
                        editor.findChildren(QLineEdit)[0].setText(text)
                    return QDialog.DialogCode.Accepted
                with patch("tools.lora_ledger_core.dialogs.QDialog.exec", new=accept), \
                     patch("tools.lora_ledger_core.dialogs.show_error") as error:
                    dialog.edit_strengths()
                self.assertEqual(error.call_count, int(text in ("nan", "1, 2, 3, 4")))
                dialog.hash_check.setChecked(False)
                dialog.save()
                updated = next(r for r in self.ledger.reviews() if r["review_id"] == saved["review_id"])
                component = updated["candidates"][0]["components"][0]
                self.assertEqual(component["strength"], expected)
                self.assertEqual(component["lbw"], [.1, .2])
                dialog.close()

    def test_worker_can_cancel_without_blocking_gui(self):
        def operation(progress, cancel):
            import time
            while not cancel():
                time.sleep(.01)
            return "cancelled"
        dialog = TaskDialog(self.window, "test", operation)
        QTimer.singleShot(30, dialog.reject)
        dialog.exec()
        dialog.worker.wait()
        self.assertEqual(dialog.value, "cancelled")


    def test_discard_on_target_switch_restores_run_fields(self):
        w = self.window
        original = w.name.text()
        w.name.setText("discard me")
        w.purpose.setPlainText("discard purpose")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
            w.target.setCurrentIndex(1)
        self.assertEqual(w.name.text(), original)
        self.assertEqual(w.purpose.toPlainText(), "")
        self.assertEqual(w.target.currentIndex(), 1)
        self.assertFalse(w.dirty)

    def test_typing_dataset_name_locks_it_and_auto_button_restores(self):
        from PySide6.QtTest import QTest
        w = self.window
        auto_name = w.dataset.text()
        w.dataset.selectAll()
        QTest.keyClicks(w.dataset, "my_dataset.toml")
        self.assertTrue(w.dataset_manual.isChecked())
        self.assertTrue(w.save())
        self.assertEqual(w.dataset.text(), "my_dataset.toml")
        self.assertTrue(w.dataset_manual.isChecked())
        w.dataset_manual.setChecked(False)
        self.assertEqual(w.dataset.text(), auto_name)
        self.assertTrue(w.save())
        self.assertFalse(w.dataset_manual.isChecked())

    def test_sorting_keeps_unsaved_input_on_selected_run(self):
        w = self.window
        run_id = w.current_run["run_id"]
        w.good.setPlainText("still editing")
        w.sort_order.setCurrentIndex(2)
        self.assertEqual(w.sort_order.currentText(), "日時（新しい順）")
        self.assertEqual(w.current_run["run_id"], run_id)
        self.assertEqual(w.good.toPlainText(), "still editing")
        self.assertTrue(w.dirty)
