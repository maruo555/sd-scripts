"""Native Qt dialogs and cancellable background operations."""
from __future__ import annotations
import copy
import json
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QLineEdit, QFileDialog, QDialogButtonBox, QInputDialog, QComboBox, QTabWidget,
    QPlainTextEdit, QFormLayout, QListWidget, QListWidgetItem, QAbstractItemView)

from .storage import LedgerError, Cancelled, new_id, new_review, file_stat, now
from .scanner import STATES, apply_scan, link_reference
from library.generation_lora_strengths import format_strength_spec, serialize_strength_spec
from .reviews import candidate_for, import_report, fingerprint_review, preference_pairs, pair_cases, select_cases


class Worker(QThread):
    progress = Signal(str)
    completed = Signal(object, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel_event = threading.Event()

    def run(self):
        try:
            value = self.fn(self.progress.emit, self.cancel_event.is_set)
            self.completed.emit(value, None)
        except Exception as exc:
            self.completed.emit(None, exc)


class TaskDialog(QDialog):
    def __init__(self, parent, title, fn):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(540)
        self.value, self.error = None, None
        self.finished_work = False
        layout = QVBoxLayout(self)
        self.label = QLabel(title)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        bar = QProgressBar()
        bar.setRange(0, 0)
        layout.addWidget(bar)
        self.cancel_button = QPushButton("中止")
        layout.addWidget(self.cancel_button)
        self.worker = Worker(fn, self)
        self.worker.progress.connect(self.label.setText)
        self.worker.completed.connect(self.complete)
        self.cancel_button.clicked.connect(self.reject)
        self.worker.start()

    def reject(self):
        if self.finished_work:
            super().reject()
        else:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.label.setText("中止しています。保存済みの内容は保持されます。")

    def complete(self, value, error):
        self.value, self.error = value, error
        self.finished_work = True
        self.accept()


def task(parent, title, fn):
    dialog = TaskDialog(parent, title, fn)
    dialog.exec()
    dialog.worker.wait()
    if dialog.error:
        if isinstance(dialog.error, Cancelled):
            QMessageBox.information(parent, "中止", str(dialog.error))
        else:
            QMessageBox.warning(parent, title, str(dialog.error))
        return None
    return dialog.value


def show_error(parent, exc):
    QMessageBox.warning(parent, "操作を完了できません", str(exc))


def button(text, callback):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    return widget


def choose_artifact(parent, ledger):
    runs = [r for r in ledger.list_runs() if r["artifacts"]]
    if not runs:
        QMessageBox.information(parent, "候補なし", "先に学習結果を登録してください。")
        return None
    labels = [r["display_name"] + "  [" + r["run_id"][:8] + "]" for r in runs]
    label, ok = QInputDialog.getItem(parent, "学習を選ぶ", "学習", labels, editable=False)
    if not ok:
        return None
    run = runs[labels.index(label)]
    artifacts = run["artifacts"]
    names = [a["name"] + ("  [変更・欠落あり]" if a["ref"].get("status") != "exists" else "")
             + "  [" + a["artifact_id"][:8] + "]" for a in artifacts]
    name, ok = QInputDialog.getItem(parent, "重みを選ぶ", "評価したcheckpoint", names, editable=False)
    return (run, artifacts[names.index(name)]) if ok else None


class SourcesDialog(QDialog):
    def __init__(self, parent, ledger):
        super().__init__(parent)
        self.ledger = ledger
        self.config = ledger.config()
        self.roots = ledger.locations()
        self.sources = copy.deepcopy(self.config["sources"])
        self.setWindowTitle("ログ・レポートの参照先")
        self.resize(1000, 440)
        layout = QVBoxLayout(self)
        info = QLabel("学習出力の親フォルダを追加すると、設定記録・診断レポートも探索します。\n"
                      "参照先を外しても、登録済みの評価や元ファイルは削除しません。")
        layout.addWidget(info)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["有効", "名前", "フォルダ", "子フォルダも探索", "除外フォルダ（;区切り）"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        actions.addWidget(button("フォルダを追加", self.add))
        actions.addWidget(button("選択した場所を変更", self.change))
        actions.addWidget(button("選択を探索対象から外す", self.remove))
        actions.addStretch()
        layout.addLayout(actions)
        research = QHBoxLayout()
        research.addWidget(QLabel("既存研究（任意）"))
        self.research_path = QLineEdit(self.roots.get(self.config.get("research_archive_root"), ""))
        self.research_path.setPlaceholderText("data/files.csv と data/epoch_metrics.csv があるフォルダ")
        research.addWidget(self.research_path, 1)
        research.addWidget(button("選択", self.choose_research))
        layout.addLayout(research)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.populate()

    def populate(self):
        self.table.setRowCount(len(self.sources))
        for i, source in enumerate(self.sources):
            enabled = QCheckBox()
            enabled.setChecked(source.get("enabled", True))
            self.table.setCellWidget(i, 0, enabled)
            self.table.setItem(i, 1, QTableWidgetItem(source["name"]))
            item = QTableWidgetItem(self.roots[source["root"]])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, item)
            recursive = QCheckBox()
            recursive.setChecked(source.get("recursive", True))
            self.table.setCellWidget(i, 3, recursive)
            self.table.setItem(i, 4, QTableWidgetItem(";".join(source.get("exclude", []))))

    def choose_research(self):
        path = QFileDialog.getExistingDirectory(self, "既存研究のフォルダ")
        if path:
            self.research_path.setText(path)

    def collect(self):
        for i, source in enumerate(self.sources):
            source["enabled"] = self.table.cellWidget(i, 0).isChecked()
            source["recursive"] = self.table.cellWidget(i, 3).isChecked()
            source["name"] = self.table.item(i, 1).text()
            source["exclude"] = [s.strip() for s in self.table.item(i, 4).text().split(";") if s.strip()]

    def add(self):
        path = QFileDialog.getExistingDirectory(self, "学習出力・レポートのフォルダ")
        if not path:
            return
        self.collect()
        root_id = new_id()
        self.roots[root_id] = path
        self.sources.append({"id": new_id(), "root": root_id, "name": Path(path).name,
                             "enabled": True, "recursive": True, "exclude": []})
        self.populate()

    def change(self):
        row = self.table.currentRow()
        if row < 0:
            return
        path = QFileDialog.getExistingDirectory(self, "移動後のフォルダ")
        if path:
            self.collect()
            self.roots[self.sources[row]["root"]] = path
            self.populate()

    def remove(self):
        row = self.table.currentRow()
        if row >= 0:
            self.collect()
            self.sources.pop(row)
            self.populate()

    def save(self):
        try:
            self.collect()
            archive = self.research_path.text().strip()
            if archive and (not (Path(archive) / "data/files.csv").is_file() or
                            not (Path(archive) / "data/epoch_metrics.csv").is_file()):
                raise LedgerError("既存研究のdata/files.csvとdata/epoch_metrics.csvが見つかりません")
            self.ledger.set_sources(self.sources, self.roots, self.config["revision"])
            self.config = self.ledger.config()
            self.ledger.set_research_archive(archive or None)
            self.accept()
        except (LedgerError, OSError) as exc:
            show_error(self, exc)


class ScanDialog(QDialog):
    def __init__(self, parent, ledger, result):
        super().__init__(parent)
        self.ledger, self.result = ledger, result
        self.setWindowTitle("再走査の結果")
        self.resize(1050, 650)
        layout = QVBoxLayout(self)
        counts = {state: sum(r["status"] == state for r in result["rows"]) for state in STATES}
        layout.addWidget(QLabel(f"新規 {counts['new']} 件 / 追加 {counts['additional']} 件 / "
                               f"要確認 {counts['changed']} 件 / 登録済み {counts['unchanged']} 件\n"
                               f"確認ファイル {result['file_count']:,} 件。重みの本数と学習回数は別です。"))
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["反映", "状態", "学習名", "重み", "対応の確かさ"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.display_rows = sorted(result["rows"], key=lambda r: (r["status"] == "unchanged", r["name"]))
        self.table.setRowCount(len(self.display_rows))
        for i, row in enumerate(self.display_rows):
            check = QCheckBox()
            check.setChecked(row["status"] in ("new", "additional"))
            check.setEnabled(row["status"] != "unchanged")
            self.table.setCellWidget(i, 0, check)
            for col, text in enumerate([STATES[row["status"]], row["name"], str(row["artifact_count"]),
                                       "session等で確認" if row["association"] == "verified" else "未確認（仮登録）"], 1):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, col, item)
        tabs.addTab(self.table, "学習の差分")
        self.related = QListWidget()
        for row in result["related"]:
            self.related.addItem(row["name"])
        related_page = QDialog()
        related_layout = QVBoxLayout(related_page)
        related_layout.addWidget(self.related)
        related_layout.addWidget(button("選択した資料を登録済みの学習に結び付ける", self.link))
        tabs.addTab(related_page, f"関連資料 {len(result['related'])}")
        issues = QPlainTextEdit()
        issues.setReadOnly(True)
        issues.setPlainText("\n".join(f"{STATES.get(r['status'], r['status'])}: {r['name']}\n{r.get('message', '')}"
                                      for r in result["issues"]) or "読込エラーはありません。")
        tabs.addTab(issues, f"確認事項 {len(result['issues'])}")
        actions = QHBoxLayout()
        actions.addWidget(button("変更なしを表示／非表示", self.toggle_unchanged))
        actions.addStretch()
        self.apply_button = button("選択した新規・資料追加を反映", self.apply)
        actions.addWidget(self.apply_button)
        actions.addWidget(button("閉じる", self.accept))
        layout.addLayout(actions)
        self.hide_unchanged = False
        self.toggle_unchanged()

    def toggle_unchanged(self):
        self.hide_unchanged = not self.hide_unchanged
        for i, row in enumerate(self.display_rows):
            self.table.setRowHidden(i, self.hide_unchanged and row["status"] == "unchanged")

    def apply(self):
        ids = [row["run_id"] for i, row in enumerate(self.display_rows) if self.table.cellWidget(i, 0).isChecked()]
        if not ids:
            return
        result = task(self, "台帳へ反映", lambda progress, cancel: apply_scan(
            self.ledger, self.result, ids, progress, cancel))
        if result is None:
            return
        for i, row in enumerate(self.display_rows):
            if row["run_id"] in result["applied"]:
                self.table.cellWidget(i, 0).setChecked(False)
                self.table.cellWidget(i, 0).setEnabled(False)
                self.table.item(i, 1).setText("反映済み")
                row["status"] = "unchanged"
        errors = "\n".join(f"{r['name']}: {r['message']}" for r in result["errors"])
        QMessageBox.information(self, "反映結果", f"{len(result['applied'])} 件を反映しました。\n{errors}")

    def link(self):
        index = self.related.currentRow()
        runs = self.ledger.list_runs()
        if index < 0 or not runs:
            return
        labels = [r["display_name"] + " [" + r["run_id"][:8] + "]" for r in runs]
        label, ok = QInputDialog.getItem(self, "資料の対応", "どの学習の資料ですか", labels, editable=False)
        if not ok:
            return
        try:
            link_reference(self.ledger, runs[labels.index(label)]["run_id"], self.result["related"][index]["entry"])
            self.related.item(index).setText(self.result["related"][index]["name"] + "  [関連付け済み]")
        except (LedgerError, OSError) as exc:
            show_error(self, exc)


class ComparisonDialog(QDialog):
    def __init__(self, parent, ledger, existing=None):
        super().__init__(parent)
        self.ledger = ledger
        self.review = copy.deepcopy(existing) if existing else new_review()
        self.review.setdefault("use", ledger.config()["default_use"])
        self.setWindowTitle("比較の判断を記録")
        self.resize(1150, 760)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("実際に見た候補にチェックを入れて判断を残します。"
                               "第一候補以外の順位は自動で付けません。"))
        actions = QHBoxLayout()
        actions.addWidget(button("比較レポートから取り込む", self.load_report))
        actions.addWidget(button("重みから候補を追加", self.add_candidate))
        actions.addWidget(button("選択候補の重みを対応付け", self.resolve_candidate))
        actions.addWidget(button("選択候補を外す", self.remove_candidate))
        actions.addWidget(button("レポートを開く", self.open_report))
        layout.addLayout(actions)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["閲覧済み", "候補", "強度 / LBW", "単独の実用性", "理由", "採用", "対応"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        layout.addWidget(button("選択候補の強度・LBWを編集", self.edit_strengths))
        self.table.cellDoubleClicked.connect(lambda row, column: self.edit_strengths() if column == 2 else None)
        form = QFormLayout()
        self.use = QLineEdit(self.review["use"])
        form.addRow("用途", self.use)
        self.choices, self.reasons = {}, {}
        for axis, label in (("overall", "総合の好み"), ("identity", "本人らしさ"), ("naturalness", "顔・人体の自然さ")):
            row = QHBoxLayout()
            choice, reason = QComboBox(), QLineEdit()
            reason.setPlaceholderText("理由（任意）")
            self.choices[axis], self.reasons[axis] = choice, reason
            row.addWidget(choice)
            row.addWidget(reason)
            form.addRow(label, row)
        self.adoption = QComboBox()
        for label, value in (("未決定", "undecided"), ("選択した候補を採用", "selected"), ("今回は採用なし", "none")):
            self.adoption.addItem(label, value)
        form.addRow("実際の採用", self.adoption)
        context = QHBoxLayout()
        self.evidence = QComboBox()
        for label, value in (("今回見直した", "current"), ("当時の評価を登録", "imported"), ("記憶から登録", "memory")):
            self.evidence.addItem(label, value)
        self.selection = QComboBox()
        for label, value in (("閲覧範囲は未記録", "unknown"), ("指定した全ケース", "all_cases"), ("選別した一部", "selected")):
            self.selection.addItem(label, value)
        self.blind = QComboBox()
        for label, value in (("名前の表示は未記録", "unknown"), ("名前を伏せて評価", "blind"), ("名前を見て評価", "named")):
            self.blind.addItem(label, value)
        context.addWidget(self.evidence)
        context.addWidget(self.selection)
        context.addWidget(self.blind)
        form.addRow("評価の背景", context)
        self.case_filter = QLineEdit()
        self.case_filter.setPlaceholderText("一部の場合：prompt IDをカンマ区切り。未記録なら空欄")
        form.addRow("対象prompt", self.case_filter)
        self.generation = QPlainTextEdit()
        self.generation.setMaximumHeight(75)
        self.generation.setPlaceholderText('生成条件JSON（任意）。例: {"seed": 123, "sampler": "euler_a"}')
        form.addRow("生成条件", self.generation)
        layout.addLayout(form)
        self.case_label = QLabel("")
        layout.addWidget(self.case_label)
        self.hash_check = QCheckBox("保存時に、対象の重みの内容を確認する")
        self.hash_check.setChecked(True)
        layout.addWidget(self.hash_check)
        self.reassess = QCheckBox("訂正ではなく、新たに見直した評価として保存")
        self.reassess.setVisible(existing is not None)
        layout.addWidget(self.reassess)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.populate()
        if existing:
            for combo, key in ((self.evidence, "evidence_mode"), (self.selection, "selection_mode"),
                               (self.blind, "blinding")):
                combo.setCurrentIndex(max(0, combo.findData(existing.get(key))))
            self.adoption.setCurrentIndex(max(0, self.adoption.findData(existing.get("adoption", {}).get("status"))))
            self.generation.setPlainText(json.dumps(existing.get("generation", {}), ensure_ascii=False))
            if existing.get("selection_mode") == "selected":
                prompt_ids = dict.fromkeys(str(case["prompt_id"]) for case in existing.get("cases", [])
                                           if case.get("prompt_id") is not None)
                self.case_filter.setText(", ".join(prompt_ids))
            for axis, combo in self.choices.items():
                pairs = [p for p in existing.get("comparisons", []) if p["axis"] == axis]
                if pairs:
                    first = pairs[0]
                    value = first[first["result"]] if first["result"] in ("left", "right") else first["result"]
                    combo.setCurrentIndex(max(0, combo.findData(value)))
                    self.reasons[axis].setText(first.get("reason", ""))
        self.original_case_filter = self.case_filter.text()

    def remember(self):
        for i, candidate in enumerate(self.review["candidates"]):
            if i >= self.table.rowCount():
                break
            candidate["_ui"] = {"viewed": self.table.cellWidget(i, 0).isChecked(),
                                "usability": self.table.cellWidget(i, 3).currentData(),
                                "reason": self.table.item(i, 4).text(),
                                "adopted": self.table.cellWidget(i, 5).isChecked()}

    def populate(self):
        self.table.setRowCount(len(self.review["candidates"]))
        available_cases = self.review.get("available_cases", self.review.get("cases"))
        available = {c["candidate_id"] for c in available_cases or []}
        for i, candidate in enumerate(self.review["candidates"]):
            state = candidate.get("_ui", {})
            check = QCheckBox()
            check.setChecked(state.get("viewed", candidate["candidate_id"] in self.review.get("viewed_candidate_ids", [])))
            check.setEnabled(available_cases is None or candidate["candidate_id"] in available)
            self.table.setCellWidget(i, 0, check)
            item = QTableWidgetItem(candidate["name"])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, item)
            strengths = ["強度 " + (str(c["strength"]) if c.get("strength") is not None else "不明") +
                         (" / LBW指定" if c.get("lbw") is not None else "") for c in candidate["components"]]
            strength_item = QTableWidgetItem(" + ".join(strengths) or "LoRAなし")
            strength_item.setFlags(strength_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            strength_item.setToolTip("ダブルクリックで強度とLBWを編集")
            self.table.setItem(i, 2, strength_item)
            usable = QComboBox()
            for label, value in (("未判断", "unknown"), ("使える", "usable"), ("条件付き", "conditional"), ("使いにくい", "difficult")):
                usable.addItem(label, value)
            usable.setCurrentIndex(max(0, usable.findData(state.get("usability", candidate.get("usability", "unknown")))))
            self.table.setCellWidget(i, 3, usable)
            self.table.setItem(i, 4, QTableWidgetItem(state.get("reason", candidate.get("usability_reason", ""))))
            adopted = QCheckBox()
            adopted.setChecked(state.get("adopted", candidate["candidate_id"] in
                                         self.review.get("adoption", {}).get("candidate_ids", [])))
            self.table.setCellWidget(i, 5, adopted)
            unresolved = any(not c.get("artifact_id") for c in candidate["components"])
            item = QTableWidgetItem("未対応あり" if unresolved else "生成当時のhash未確認")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 6, item)
        for combo in self.choices.values():
            selected = combo.currentData()
            combo.clear()
            combo.addItem("未評価", None)
            combo.addItem("同程度", "tie")
            combo.addItem("判断保留", "undecided")
            for candidate in self.review["candidates"]:
                combo.addItem(candidate["name"], candidate["candidate_id"])
            index = combo.findData(selected)
            combo.setCurrentIndex(max(0, index))
        count = len(available_cases or [])
        self.case_label.setText(f"取り込んだ生成job: {count} 件（閲覧対象は対象promptで指定）" if available_cases is not None else
                                "生成レポート未指定。条件が分からない欄は不明のまま保存できます。")

    def edit_strengths(self):
        index = self.table.currentRow()
        if index < 0:
            return
        self.remember()
        candidate = self.review["candidates"][index]
        dialog = QDialog(self)
        dialog.setWindowTitle("強度・LBW")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        entries = []
        for i, component in enumerate(candidate["components"]):
            name = Path(component.get("source_path", "")).name or f"LoRA {i + 1}"
            strength = QLineEdit("" if component.get("strength") is None else format_strength_spec(component["strength"]))
            strength.setPlaceholderText("共通 / TE, UNet / TE1, TE2, UNet。不明なら空欄")
            lbw_value = component.get("lbw")
            lbw = QLineEdit("" if lbw_value is None else json.dumps(lbw_value) if isinstance(lbw_value, list) else str(lbw_value))
            lbw.setPlaceholderText("不明・未指定なら空欄。既存プリセット名や重み列")
            form.addRow(name + " の強度", strength)
            form.addRow("LBW", lbw)
            entries.append((component, strength, lbw, lbw.text()))
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            try:
                updates = []
                for component, strength, lbw, original_lbw_text in entries:
                    value = serialize_strength_spec(strength.text()) if strength.text().strip() else None
                    lbw_value = (component.get("lbw") if lbw.text() == original_lbw_text
                                 else lbw.text().strip() or None)
                    updates.append((component, value, lbw_value))
                for component, value, lbw in updates:
                    component.update(strength=value, lbw=lbw)
                self.populate()
            except ValueError as exc:
                show_error(self, exc)

    def add_candidate(self):
        selected = choose_artifact(self, self.ledger)
        if selected:
            self.remember()
            self.review["candidates"].append(candidate_for(*selected))
            self.populate()

    def remove_candidate(self):
        index = self.table.currentRow()
        if index >= 0:
            self.remember()
            self.review["candidates"].pop(index)
            self.populate()

    def resolve_candidate(self):
        index = self.table.currentRow()
        if index < 0:
            return
        self.remember()
        candidate = self.review["candidates"][index]
        for component in candidate["components"]:
            selected = choose_artifact(self, self.ledger)
            if selected is None:
                return
            run, artifact = selected
            if (component.get("run_id"), component.get("artifact_id")) != (run["run_id"], artifact["artifact_id"]):
                component.pop("content_sha256", None)
                candidate["generation_link_verified"] = False
            component.update(run_id=run["run_id"], artifact_id=artifact["artifact_id"],
                             identity_status="user_asserted")
        self.populate()

    def load_report(self):
        folder = QFileDialog.getExistingDirectory(self, "metadata.jsonがある画像比較フォルダ")
        if not folder:
            return
        result = task(self, "比較条件を読み込む", lambda progress, cancel: import_report(self.ledger, folder))
        if result:
            self.review = result
            self.generation.setPlainText(json.dumps(result.get("generation", {}), ensure_ascii=False))
            self.selection.setCurrentIndex(self.selection.findData("all_cases"))
            self.case_filter.clear()
            self.populate()

    def open_report(self):
        folder = self.review.get("report", {}).get("path")
        if folder:
            path = Path(folder) / "report.html"
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def save(self):
        try:
            review = copy.deepcopy(self.review)
            if not review["candidates"]:
                raise LedgerError("比較する候補を追加してください")
            viewed = []
            for i, candidate in enumerate(review["candidates"]):
                candidate.pop("_ui", None)
                if self.table.cellWidget(i, 0).isChecked():
                    viewed.append(candidate["candidate_id"])
                usable = self.table.cellWidget(i, 3).currentData()
                candidate["usability"] = usable
                candidate["usability_reason"] = self.table.item(i, 4).text()
            review["viewed_candidate_ids"] = viewed
            review["comparisons"] = preference_pairs(review["candidates"], viewed,
                {k: c.currentData() for k, c in self.choices.items()},
                {k: c.text() for k, c in self.reasons.items()})
            status = self.adoption.currentData()
            adopted = [c["candidate_id"] for i, c in enumerate(review["candidates"])
                       if self.table.cellWidget(i, 5).isChecked()] if status == "selected" else []
            if status == "selected" and not adopted:
                raise LedgerError("採用する候補の「採用」にチェックしてください")
            review["adoption"] = {"status": status, "candidate_ids": adopted}
            review.update(use=self.use.text(), evidence_mode=self.evidence.currentData(),
                          blinding=self.blind.currentData(),
                          evaluation_time_status="recorded_now" if self.evidence.currentData() == "current" else "original_date_unknown")
            if self.generation.toPlainText().strip():
                review["generation"] = json.loads(self.generation.toPlainText())
                if not isinstance(review["generation"], dict):
                    raise LedgerError("生成条件はJSONオブジェクトで入力してください")
            if not review["comparisons"] and status == "undecided" and all(
                    c.get("usability") == "unknown" and not c.get("usability_reason") for c in review["candidates"]):
                raise LedgerError("好み・実用性・採用のいずれかの判断を入力してください")
            selection_mode = self.selection.currentData()
            selection_changed = (selection_mode != self.review.get("selection_mode") or
                                 self.case_filter.text() != self.original_case_filter)
            if selection_changed:
                ids = {s.strip() for s in self.case_filter.text().split(",") if s.strip()}
                select_cases(self.ledger, review, selection_mode, ids)
            pair_cases(review)
            if any(not p["case_ids"] for p in review["comparisons"] if "case_ids" in p):
                raise LedgerError("閲覧対象に共通する生成ケースがない候補対があります。候補・promptを確認してください")
            confirm_hash = self.hash_check.isChecked()
            if self.reassess.isChecked() and review["revision"]:
                review.update(reassessment_of=review["review_id"], review_id=new_id(), revision=0, evaluated_at=now())
            def save(progress, cancel):
                value = fingerprint_review(self.ledger, review, cancel, progress) if confirm_hash else review
                return self.ledger.save_review(value, review["revision"])
            result = task(self, "比較評価を保存", save)
            if result:
                self.accept()
        except (LedgerError, ValueError, TypeError) as exc:
            show_error(self, exc)
