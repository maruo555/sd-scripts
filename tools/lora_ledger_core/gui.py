"""PySide6 desktop front end for the lightweight research ledger."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QUrl
from PySide6.QtGui import QAction, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPlainTextEdit, QSplitter, QTableView, QHeaderView,
    QAbstractItemView, QTabWidget, QFormLayout, QCheckBox, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QInputDialog, QTableWidget, QTableWidgetItem, QScrollArea)

from .storage import (Ledger, LedgerError, REPO, read_json, atomic_json, latest_annotations,
                      new_id)
from .scanner import scan, verify
from .notes import save_note
from .presentation import dataset_info, dataset_tooltip, automatic_dataset, run_time, time_text
from .exporting import export_csv, export_bundle, listing_rows
from .dialogs import (button, task, show_error, SourcesDialog, ScanDialog, ComparisonDialog)


STATUS_NAMES = {"exists": "参照可能", "missing": "見つからない", "changed": "変更あり",
                "unreadable": "読込不能"}
ASSOCIATIONS = {"verified": "IDで確認", "user_asserted": "人が確認", "candidate": "対応未確認"}


class RunModel(QAbstractTableModel):
    HEADERS = ["学習名", "学習日時", "データセット", "お気に入り", "重み数", "学習の目的"]
    TIME_COLUMN = 1
    SORT_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self):
        super().__init__()
        self.rows = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row["values"][index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return row["search"]
        if role == self.SORT_ROLE:
            if index.column() == self.TIME_COLUMN:
                return row["chronology"]["timestamp"]
            if index.column() == 4:
                return len(row["run"]["artifacts"])
            return row["values"][index.column()].casefold()
        if role == Qt.ItemDataRole.ToolTipRole:
            if index.column() == self.TIME_COLUMN:
                info = row["chronology"]
                return info["source_label"] + "\n" + time_text(info, iso=True) + (
                    "\n" + info["artifact_name"] if info.get("artifact_name") else "")
            if index.column() == 2:
                return dataset_tooltip(row["dataset_info"])
            return row["run"]["display_name"] + "\n" + row["run"]["run_id"]

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]

    def replace(self, runs, reviews):
        annotations = latest_annotations(reviews)
        usability = {}
        comparison_runs = set()
        for review in reviews:
            for candidate in review.get("candidates", []):
                for component in candidate.get("components", []):
                    usability.setdefault(component.get("run_id"), set()).add(candidate.get("usability", "unknown"))
                    if review.get("comparisons") or candidate.get("usability", "unknown") != "unknown" or review.get("adoption", {}).get("status") not in (None, "undecided"):
                        comparison_runs.add(component.get("run_id"))
        self.beginResetModel()
        self.rows = []
        for run in sorted(runs, key=lambda r: r["display_name"].casefold()):
            notes = [a for key, (_, a) in annotations.items() if key[0] == run["run_id"]]
            ratings = sorted({a["favorite_rating"] for a in notes if a.get("favorite_rating") is not None})
            stars = "比較評価あり" if run["run_id"] in comparison_runs else "未評価"
            if notes:
                if len(notes) == 1:
                    label = "重み" if notes[0].get("artifact_id") else "全体"
                    stars = label + " " + ("★" * ratings[0] if ratings else "所見あり")
                else:
                    stars = f"複数評価 ({len(notes)})"
            dataset = dataset_info(run)
            chronology = run_time(run)
            values = [run["display_name"], time_text(chronology), dataset["display_name"], stars,
                      str(len(run["artifacts"])), run.get("user", {}).get("experiment_note", "")]
            self.rows.append({"run": run, "reviewed": bool(notes) or run["run_id"] in comparison_runs, "ratings": ratings, "usability": usability.get(run["run_id"], set()),
                              "values": values, "dataset_info": dataset, "chronology": chronology,
                              "search": json.dumps([run, notes, dataset, values], ensure_ascii=False).casefold()})
        self.endResetModel()


class RunFilter(QSortFilterProxyModel):
    query = ""
    mode = "all"
    rating = None
    dataset = ""
    purpose = None
    usability = None

    def lessThan(self, left, right):
        if left.column() == RunModel.TIME_COLUMN:
            a = self.sourceModel().data(left, RunModel.SORT_ROLE)
            b = self.sourceModel().data(right, RunModel.SORT_ROLE)
            if a is None or b is None:
                if a is None and b is None:
                    return left.row() < right.row()
                # Qt reverses the comparator for descending order; keep unknown last.
                return (a is None) == (self.sortOrder() == Qt.SortOrder.DescendingOrder)
            if a != b:
                return a < b
            return left.row() < right.row()
        return super().lessThan(left, right)

    def filterAcceptsRow(self, row, parent):
        value = self.sourceModel().rows[row]
        if self.purpose and value["run"].get("purpose") != self.purpose:
            return False
        if self.usability and self.usability not in value["usability"]:
            return False
        if self.rating is not None and self.rating not in value["ratings"]:
            return False
        if self.dataset and self.dataset.casefold() not in value["dataset_info"]["display_name"].casefold():
            return False
        if self.query and any(word not in value["search"] for word in self.query.casefold().split()):
            return False
        if self.mode == "unreviewed" and value["reviewed"]:
            return False
        if self.mode == "reviewed" and not value["reviewed"]:
            return False
        if self.mode == "issues":
            run = value["run"]
            refs = run["source_refs"] + [a["ref"] for a in run["artifacts"]]
            return run["association"] != "verified" or any(
                r.get("status") != "exists" or r.get("association", {}).get("status") == "candidate" for r in refs)
        return True


class LedgerWindow(QMainWindow):
    def __init__(self, ledger=None, state_path=None):
        super().__init__()
        self.state_path = Path(state_path) if state_path else REPO / ".local" / "lora_ledger_gui.json"
        self.ledger = None
        self.current_run = None
        self.current_review = None
        self.loading = True
        self.dirty = False
        self.annotations = {}
        self.resize(1500, 900)
        self.setMinimumSize(1020, 720)
        self.setWindowTitle("LoRA 研究台帳")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        for label, callback in (("新規台帳", self.create_ledger), ("台帳を開く", self.open_ledger)):
            top.addWidget(button(label, callback))
        self.ledger_path = QLabel("最初に、リポジトリ外へ台帳を作成するか、既存の台帳を開いてください。")
        self.ledger_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.ledger_path, 1)
        layout.addLayout(top)
        actions = QHBoxLayout()
        self.operation_buttons = []
        for label, callback in (("参照先の設定", self.sources), ("再走査", self.rescan),
                                ("比較の判断を記録", self.compare), ("CSVを書き出す", self.csv),
                                ("AI相談用データ", self.ai_export)):
            widget = button(label, callback)
            self.operation_buttons.append(widget)
            actions.addWidget(widget)
        actions.addStretch()
        layout.addLayout(actions)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("学習名・旧名・設定・メモを検索")
        self.filter = QComboBox()
        for label, value in (("すべて", "all"), ("未評価", "unreviewed"), ("評価あり", "reviewed"),
                             ("対応・参照の確認が必要", "issues")):
            self.filter.addItem(label, value)
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.filter)
        left_layout.addLayout(search_row)
        filters = QHBoxLayout()
        self.dataset_filter = QLineEdit()
        self.dataset_filter.setPlaceholderText("データセットで絞込み")
        self.star_filter = QComboBox()
        self.star_filter.addItem("星：すべて", None)
        for rating in range(1, 6):
            self.star_filter.addItem("★" * rating + " の評価あり", rating)
        filters.addWidget(self.dataset_filter, 1)
        filters.addWidget(self.star_filter)
        left_layout.addLayout(filters)
        detail_filters = QHBoxLayout()
        self.purpose_filter = QComboBox()
        self.usability_filter = QComboBox()
        for label, value in (("学習区分：すべて", None), ("通常学習", "normal"), ("試験", "experiment"),
                             ("診断", "diagnostic"), ("区分不明", "unknown")):
            self.purpose_filter.addItem(label, value)
        for label, value in (("実用性：すべて", None), ("使える評価あり", "usable"),
                             ("条件付きの評価あり", "conditional"), ("使いにくい評価あり", "difficult")):
            self.usability_filter.addItem(label, value)
        detail_filters.addWidget(self.purpose_filter)
        detail_filters.addWidget(self.usability_filter)
        left_layout.addLayout(detail_filters)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("並び順"))
        self.sort_order = QComboBox()
        self.sort_order.setMinimumWidth(170)
        for label, value in (("学習名（A→Z）", (0, 0)), ("学習名（Z→A）", (0, 1)),
                             ("日時（新しい順）", (RunModel.TIME_COLUMN, 1)),
                             ("日時（古い順）", (RunModel.TIME_COLUMN, 0))):
            self.sort_order.addItem(label, value)
        sort_row.addWidget(self.sort_order)
        sort_hint = QLabel("開始日時を優先。不明なら重みの更新日時")
        sort_hint.setWordWrap(True)
        sort_row.addWidget(sort_hint, 1)
        left_layout.addLayout(sort_row)
        self.model = RunModel()
        self.proxy = RunFilter()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(RunModel.SORT_ROLE)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setColumnWidth(0, 245)
        self.table.setColumnWidth(1, 155)
        self.table.setColumnWidth(2, 145)
        self.table.setColumnWidth(3, 105)
        self.table.setColumnWidth(4, 55)
        self.table.setColumnWidth(5, 120)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.sort_order.currentIndexChanged.connect(self.sort_changed)
        self.table.horizontalHeader().sortIndicatorChanged.connect(self.sort_header_changed)
        left_layout.addWidget(self.table, 1)
        self.count = QLabel("登録 0 件")
        left_layout.addWidget(self.count)
        extra = QHBoxLayout()
        extra.addWidget(button("元資料がない学習をメモ登録", self.note_only))
        extra.addWidget(button("台帳フォルダを開く", self.open_folder))
        left_layout.addLayout(extra)
        splitter.addWidget(left)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([850, 650])
        page = QWidget()
        form_layout = QVBoxLayout(page)
        form = QFormLayout()
        self.name = QLineEdit()
        self.dataset = QLineEdit()
        self.dataset.setPlaceholderText("TOML名や分かりやすい名前を手入力できます")
        self.dataset_manual = QCheckBox("手動名を優先（再走査でも保持）")
        self.dataset_source = QLabel("")
        self.dataset_source.setWordWrap(True)
        self.target = QComboBox()
        self.target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target.setMinimumContentsLength(25)
        self.rating = QComboBox()
        self.rating.addItem("未評価（空欄）", None)
        for i in range(1, 6):
            self.rating.addItem("★" * i, i)
        self.purpose = QPlainTextEdit()
        self.good = QPlainTextEdit()
        self.bad = QPlainTextEdit()
        self.free = QPlainTextEdit()
        for widget in (self.purpose, self.good, self.bad, self.free):
            widget.setMaximumHeight(90)
        self.purpose.setPlaceholderText("例：clip_rate_highを試す／この実験の機能は正式採用しなかった")
        self.good.setPlaceholderText("例：破綻が少ない")
        self.bad.setPlaceholderText("例：キャラの再現性が低い")
        self.free.setPlaceholderText("例：Aと比べるとこちらが好み。理由など自由に記録")
        form.addRow("表示名", self.name)
        form.addRow("データセット", self.dataset)
        dataset_options = QHBoxLayout()
        dataset_options.addWidget(self.dataset_manual)
        dataset_options.addWidget(button("自動取得に戻す", lambda: self.dataset_manual.setChecked(False)))
        form.addRow("", dataset_options)
        form.addRow("", self.dataset_source)
        form.addRow("評価する対象", self.target)
        self.target_note = QLabel("対象が不明なら「学習全体の印象」で記録できます。")
        self.target_note.setWordWrap(True)
        form.addRow(self.target_note)
        form.addRow("お気に入り度", self.rating)
        form.addRow("学習の目的・経緯", self.purpose)
        form.addRow("良いところ", self.good)
        form.addRow("悪いところ", self.bad)
        form.addRow("任意メモ", self.free)
        self.evidence = QComboBox()
        for label, value in (("今回見直した", "current"), ("当時の評価を登録", "imported"), ("記憶から登録", "memory")):
            self.evidence.addItem(label, value)
        form.addRow("評価の背景", self.evidence)
        form_layout.addLayout(form)
        self.hash_check = QCheckBox("重みの内容を確認して保存（対象checkpointのみ）")
        self.hash_check.setChecked(True)
        form_layout.addWidget(self.hash_check)
        self.saved = QLabel("5項目はすべて任意です。星や不採用から勝敗は作りません。")
        self.saved.setWordWrap(True)
        form_layout.addWidget(self.saved)
        save_row = QHBoxLayout()
        self.save_button = button("保存  Ctrl+S", self.save)
        self.reassess_button = button("新しく見直した評価として保存", lambda: self.save(reassess=True))
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.reassess_button)
        form_layout.addLayout(save_row)
        form_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "入力")
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.addWidget(QLabel("元ファイルの設定を表示します。未記録の値は不明のままです。"))
        classifications = QFormLayout()
        self.purpose_kind, self.training_status = QComboBox(), QComboBox()
        for label, value in (("不明", "unknown"), ("通常学習", "normal"), ("試験", "experiment"), ("診断", "diagnostic")):
            self.purpose_kind.addItem(label, value)
        for label, value in (("不明", "unknown"), ("完了", "completed"), ("途中終了", "interrupted"), ("失敗", "failed")):
            self.training_status.addItem(label, value)
        classifications.addRow("学習区分（人が指定）", self.purpose_kind)
        classifications.addRow("終了状態（人が指定）", self.training_status)
        settings_layout.addLayout(classifications)
        self.setting_values = QTableWidget(0, 2)
        self.setting_values.setHorizontalHeaderLabels(["項目", "値・出典"])
        self.setting_values.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.setting_values.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        settings_layout.addWidget(self.setting_values, 1)
        self.settings = QPlainTextEdit()
        self.settings.setReadOnly(True)
        self.settings.setVisible(False)
        settings_layout.addWidget(button("詳しい設定と出典を表示／非表示",
                                         lambda: self.settings.setVisible(not self.settings.isVisible())))
        settings_layout.addWidget(self.settings, 1)
        settings_layout.addWidget(QLabel("区分・終了状態の変更は「保存」またはCtrl+Sで保存します。"))
        self.tabs.addTab(settings_page, "設定・出典")
        refs_page = QWidget()
        refs_layout = QVBoxLayout(refs_page)
        self.refs_table = QTableWidget(0, 4)
        self.refs_table.setHorizontalHeaderLabels(["資料", "種類", "状態", "学習との対応"])
        self.refs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.refs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.refs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        refs_layout.addWidget(self.refs_table)
        refs_actions = QHBoxLayout()
        refs_actions.addWidget(button("資料を開く", self.open_ref))
        refs_actions.addWidget(button("選択資料はこの学習のもの", self.confirm_ref))
        refs_layout.addLayout(refs_actions)
        refs_layout.addWidget(button("この学習のファイルを内容まで再確認", self.verify_files))
        self.tabs.addTab(refs_page, "元ログ・レポート")
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        history_layout.addWidget(self.history)
        history_layout.addWidget(button("比較評価を選んで訂正・見直し", self.edit_comparison))
        self.tabs.addTab(history_page, "評価の履歴")
        self.dataset.textEdited.connect(self.dataset_edited)
        self.dataset_manual.toggled.connect(self.dataset_mode_changed)
        self.search.textChanged.connect(self.apply_filter)
        self.filter.currentIndexChanged.connect(self.apply_filter)
        self.star_filter.currentIndexChanged.connect(self.apply_filter)
        self.dataset_filter.textChanged.connect(self.apply_filter)
        self.purpose_filter.currentIndexChanged.connect(self.apply_filter)
        self.usability_filter.currentIndexChanged.connect(self.apply_filter)
        self.purpose_kind.currentIndexChanged.connect(self.mark_dirty)
        self.training_status.currentIndexChanged.connect(self.mark_dirty)
        self.table.selectionModel().currentRowChanged.connect(self.select_row)
        self.target.currentIndexChanged.connect(self.select_target)
        for widget in (self.name, self.dataset):
            widget.textChanged.connect(self.mark_dirty)
        for widget in (self.purpose, self.good, self.bad, self.free):
            widget.textChanged.connect(self.mark_dirty)
        self.rating.currentIndexChanged.connect(self.mark_dirty)
        self.evidence.currentIndexChanged.connect(self.mark_dirty)
        shortcut = QAction("保存", self)
        shortcut.setShortcut(QKeySequence.StandardKey.Save)
        shortcut.triggered.connect(self.save)
        self.addAction(shortcut)
        self.loading = False
        self.set_enabled(False)
        if ledger:
            self.activate(ledger)

    def sort_changed(self, *_):
        value = self.sort_order.currentData()
        if value is None:
            return
        loading = self.loading
        self.loading = True
        self.table.sortByColumn(value[0], Qt.SortOrder(value[1]))
        self.loading = loading
        if not loading and self.proxy.rowCount():
            if self.dirty:
                self.table.scrollTo(self.table.currentIndex())
            else:
                self.table.setCurrentIndex(self.proxy.index(0, 0))

    def sort_header_changed(self, column, order):
        self.sort_order.blockSignals(True)
        index = next((i for i in range(self.sort_order.count())
                      if tuple(self.sort_order.itemData(i)) == (column, order.value)), -1)
        self.sort_order.setCurrentIndex(index)
        self.sort_order.blockSignals(False)

    def dataset_edited(self, *_):
        if not self.loading:
            self.dataset_manual.setChecked(True)

    def dataset_mode_changed(self, manual):
        if self.loading or not self.current_run:
            return
        if not manual:
            self.dataset.setText(automatic_dataset(self.current_run)["display_name"])
        self.dataset_source.setText("手動入力した名前を保存します。" if manual else
                                    "記録から自動取得。TOML名がなければフォルダ名または名称不明。")
        self.mark_dirty()

    def set_enabled(self, enabled):
        self.tabs.setEnabled(enabled)
        for widget in self.operation_buttons:
            widget.setEnabled(self.ledger is not None)

    def mark_dirty(self, *_):
        if not self.loading and self.current_run:
            self.dirty = True
            self.saved.setText("未保存の変更があります。")

    def guard(self):
        if not self.dirty:
            return True
        choice = QMessageBox.question(self, "入力を保存", "未保存の入力があります。",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            return self.save()
        self.dirty = False
        if self.current_run:
            self.reload(self.current_run["run_id"], self.active_target)
        return True

    def activate(self, ledger):
        self.ledger = ledger
        self.ledger_path.setText(str(ledger.directory))
        self.ledger_path.setToolTip(str(ledger.directory))
        try:
            atomic_json(self.state_path, {"last_ledger": str(ledger.directory)})
        except OSError as exc:
            self.statusBar().showMessage(f"起動設定の保存に失敗: {exc}")
        self.reload()

    def reload(self, run_id=None, artifact_id=None):
        if not self.ledger:
            return
        self.loading = True
        runs = self.ledger.list_runs()
        reviews = self.ledger.reviews()
        self.annotations = latest_annotations(reviews)
        self.model.replace(runs, reviews)
        self.current_run = None
        self.current_review = None
        self.dirty = False
        self.set_enabled(False)
        self.loading = False
        self.count.setText(f"表示 {self.proxy.rowCount()} / 登録 {len(runs)} 件")
        target = next((i for i, r in enumerate(self.model.rows) if r["run"]["run_id"] == run_id), None)
        if target is not None:
            index = self.proxy.mapFromSource(self.model.index(target, 0))
            if index.isValid():
                self.table.setCurrentIndex(index)
                index = self.target.findData(artifact_id)
                if index >= 0:
                    self.target.setCurrentIndex(index)
        if self.current_run is None and self.proxy.rowCount():
            self.table.setCurrentIndex(self.proxy.index(0, 0))
        if not self.current_run:
            self.loading = True
            for widget in (self.name, self.dataset, self.purpose, self.good, self.bad, self.free):
                widget.clear()
            self.target.clear()
            self.loading = False
        errors = self.ledger.errors
        self.statusBar().showMessage(f"読み込みに確認事項 {len(errors)} 件: " + " / ".join(errors[:3])
                                    if errors else "準備できました")

    def apply_filter(self, *_):
        if self.loading:
            return
        if not self.guard():
            return
        if hasattr(self.proxy, "beginFilterChange"):
            self.proxy.beginFilterChange()
        self.proxy.query = self.search.text()
        self.proxy.mode = self.filter.currentData()
        self.proxy.rating = self.star_filter.currentData()
        self.proxy.dataset = self.dataset_filter.text()
        self.proxy.purpose = self.purpose_filter.currentData()
        self.proxy.usability = self.usability_filter.currentData()
        if hasattr(self.proxy, "endFilterChange"):
            self.proxy.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.proxy.invalidateFilter()
        self.count.setText(f"表示 {self.proxy.rowCount()} / 登録 {self.model.rowCount()} 件")
        if not self.table.currentIndex().isValid():
            if self.proxy.rowCount():
                self.table.setCurrentIndex(self.proxy.index(0, 0))
            else:
                self.current_run = None
                self.current_review = None
                self.set_enabled(False)

    def select_row(self, current, previous):
        if self.loading or not current.isValid():
            return
        selected_run_id = self.model.rows[self.proxy.mapToSource(current).row()]["run"]["run_id"]
        if not self.guard():
            self.loading = True
            self.table.setCurrentIndex(previous)
            self.loading = False
            return
        row = next(i for i, value in enumerate(self.model.rows) if value["run"]["run_id"] == selected_run_id)
        self.current_run = copy.deepcopy(self.model.rows[row]["run"])
        self.loading = True
        self.table.setCurrentIndex(self.proxy.mapFromSource(self.model.index(row, 0)))
        run = self.current_run
        self.name.setText(run["display_name"])
        dataset = dataset_info(run)
        self.dataset.setText(dataset["display_name"])
        self.dataset_manual.setChecked(dataset["display_name_custom"])
        self.dataset.setToolTip(dataset_tooltip(dataset))
        self.dataset_source.setText(dataset_tooltip(dataset).split("\n")[0])
        self.purpose.setPlainText(run.get("user", {}).get("experiment_note", ""))
        self.target.clear()
        self.target.addItem("学習全体の印象（checkpointを特定しない）", None)
        for artifact in sorted(run["artifacts"], key=lambda a: (a.get("role") != "final", a["name"])):
            self.target.addItem(artifact["name"] + f"  [epoch {artifact.get('epoch') or '不明'}]", artifact["artifact_id"])
        self.active_target = None
        self.purpose_kind.setCurrentIndex(max(0, self.purpose_kind.findData(run.get("purpose", "unknown"))))
        self.training_status.setCurrentIndex(max(0, self.training_status.findData(run.get("training_status", "unknown"))))
        source = run["settings_sources"].get(run.get("settings", {}).get("settings_id"), {})
        values = source.get("values", {})
        summary = [("設定の出典", {"metadata": "checkpointのメタ情報（部分記録）", "record": "学習設定ファイル"}.get(source.get("source"), "記録なし")),
                   ("メタ情報の旧名", " / ".join(run.get("aliases", []))),
                   ("対応", ASSOCIATIONS.get(run["association"], run["association"]))]
        for key, label in (("learning_rate", "学習率"), ("network_dim", "rank"), ("network_alpha", "alpha"),
                           ("optimizer", "optimizer"), ("optimizer_type", "optimizer_type"),
                           ("seed", "seed"), ("sd_model_name", "base model"),
                           ("network_module", "network module"), ("network_args", "network args")):
            value = values.get(key)
            summary.append((label, "不明・未記録" if value is None else
                            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)))
        self.setting_values.setRowCount(len(summary))
        for i, (label, value) in enumerate(summary):
            self.setting_values.setItem(i, 0, QTableWidgetItem(label))
            self.setting_values.setItem(i, 1, QTableWidgetItem(value))
        self.settings.setPlainText("現在のファイル名を表示名に使い、メタ情報の旧名も残しています。\n"
            "記録されていない設定は補完しません。\n\n" + json.dumps(
                {k: run.get(k) for k in ("identity", "aliases", "dataset", "settings", "settings_sources")},
                ensure_ascii=False, indent=2))
        self.display_refs = run["source_refs"] + [a["ref"] for a in run["artifacts"]]
        self.refs_table.setRowCount(len(self.display_refs))
        for i, ref in enumerate(self.display_refs):
            for j, value in enumerate((ref["relative_path"], ref["kind"],
                                       STATUS_NAMES.get(ref["status"], ref["status"]),
                                       ASSOCIATIONS.get(ref["association"]["status"], ref["association"]["status"]))):
                self.refs_table.setItem(i, j, QTableWidgetItem(value))
        histories = []
        for review in self.ledger.reviews(history=True):
            involved = {a["run_id"] for a in review.get("annotations", [])}
            involved |= {p.get("run_id") for c in review.get("candidates", []) for p in c.get("components", [])}
            if run["run_id"] in involved:
                histories.append(review)
        self.history.setPlainText(json.dumps(histories, ensure_ascii=False, indent=2) if histories else "まだ評価はありません。")
        self.loading = False
        self.set_enabled(True)
        self.load_note()

    def select_target(self, *_):
        if self.loading or not self.current_run:
            return
        requested = self.target.currentData()
        if not self.guard():
            self.loading = True
            self.target.setCurrentIndex(max(0, self.target.findData(self.active_target)))
            self.loading = False
            return
        self.active_target = requested
        self.loading = True
        self.target.setCurrentIndex(max(0, self.target.findData(requested)))
        self.loading = False
        self.load_note()

    def load_note(self):
        self.loading = True
        key = (self.current_run["run_id"], self.active_target, None)
        self.current_review, annotation = self.annotations.get(key, (None, {}))
        self.rating.setCurrentIndex(max(0, self.rating.findData(annotation.get("favorite_rating"))))
        self.good.setPlainText(annotation.get("good_points", ""))
        self.bad.setPlainText(annotation.get("bad_points", ""))
        self.free.setPlainText(annotation.get("free_note", ""))
        self.evidence.setCurrentIndex(max(0, self.evidence.findData((self.current_review or {}).get("evidence_mode", "current"))))
        self.hash_check.setEnabled(self.active_target is not None)
        self.reassess_button.setEnabled(self.current_review is not None)
        self.saved.setText(f"保存済み：版 {self.current_review['revision']}。訂正は旧版を残します。"
                           if self.current_review else "未評価。入力した欄だけ保存できます。")
        self.dirty = False
        self.loading = False

    def save(self, checked=False, reassess=False):
        if not self.current_run:
            return True
        fields = {"display_name": self.name.text(), "dataset_name": self.dataset.text(),
                  "dataset_name_manual": self.dataset_manual.isChecked(),
                  "experiment_note": self.purpose.toPlainText(), "favorite_rating": self.rating.currentData(),
                  "good_points": self.good.toPlainText(), "bad_points": self.bad.toPlainText(),
                  "free_note": self.free.toPlainText(), "evidence_mode": self.evidence.currentData(),
                  "purpose": self.purpose_kind.currentData(), "training_status": self.training_status.currentData()}
        run, artifact_id = copy.deepcopy(self.current_run), self.active_target
        previous, confirm = copy.deepcopy(self.current_review), self.hash_check.isChecked()
        result = task(self, "入力を保存", lambda progress, cancel: save_note(
            self.ledger, run, artifact_id, fields, previous, reassess, confirm, progress, cancel))
        if result is None:
            return False
        self.dirty = False
        self.reload(run["run_id"], artifact_id)
        self.statusBar().showMessage("保存しました", 8000)
        return True

    def create_ledger(self):
        if not self.guard():
            return
        path = QFileDialog.getExistingDirectory(self, "新しい台帳用の空フォルダ（リポジトリ外）", str(REPO.parent))
        if not path:
            return
        try:
            self.activate(Ledger.create(path))
            self.sources()
        except (OSError, LedgerError) as exc:
            show_error(self, exc)

    def open_ledger(self):
        if not self.guard():
            return
        path = QFileDialog.getExistingDirectory(self, "ledger.jsonがある台帳フォルダ", str(REPO.parent))
        if path:
            try:
                self.activate(Ledger(path))
            except (OSError, ValueError) as exc:
                show_error(self, exc)

    def sources(self):
        if self.ledger and self.guard():
            SourcesDialog(self, self.ledger).exec()

    def rescan(self):
        if not self.ledger or not self.guard():
            return
        if not self.ledger.config()["sources"]:
            self.sources()
            if not self.ledger.config()["sources"]:
                return
        current_id = self.current_run["run_id"] if self.current_run else None
        result = task(self, "参照先を再走査", lambda progress, cancel: scan(self.ledger, progress, cancel))
        if result is not None:
            ScanDialog(self, self.ledger, result).exec()
        self.reload(current_id)

    def compare(self):
        if self.ledger and self.guard():
            ComparisonDialog(self, self.ledger).exec()
            self.reload(self.current_run["run_id"] if self.current_run else None)

    def edit_comparison(self):
        if not self.ledger or not self.guard():
            return
        reviews = [r for r in self.ledger.reviews() if r.get("comparisons") or
                   (r.get("candidates") and not r.get("annotations"))]
        if not reviews:
            QMessageBox.information(self, "比較評価", "保存された比較評価はありません。")
            return
        labels = [r.get("evaluated_at", "") + "  " + " / ".join(c["name"] for c in r["candidates"])
                  + " [" + r["review_id"][:8] + "]" for r in reviews]
        selected, ok = QInputDialog.getItem(self, "比較評価を選ぶ", "評価", labels, editable=False)
        if ok:
            ComparisonDialog(self, self.ledger, reviews[labels.index(selected)]).exec()
            self.reload(self.current_run["run_id"] if self.current_run else None)

    def filtered_ids(self):
        return [self.model.rows[self.proxy.mapToSource(self.proxy.index(i, 0)).row()]["run"]["run_id"]
                for i in range(self.proxy.rowCount())]

    def csv(self):
        if not self.ledger or not self.guard():
            return
        kind, ok = QInputDialog.getItem(self, "CSV書き出し", "表示中の学習について書き出す内容",
                                        ["一覧・最新の所見", "評価の詳細", "評価の全訂正履歴"], editable=False)
        if not ok:
            return
        columns = None
        if kind == "一覧・最新の所見":
            rows = listing_rows(self.ledger, self.filtered_ids())
            if rows:
                dialog = QDialog(self)
                dialog.setWindowTitle("CSVに出す列")
                form = QVBoxLayout(dialog)
                checks = []
                for column in rows[0]:
                    check = QCheckBox(column)
                    check.setChecked(True)
                    checks.append(check)
                    form.addWidget(check)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                form.addWidget(buttons)
                if not dialog.exec():
                    return
                columns = [check.text() for check in checks if check.isChecked()]
                if not columns:
                    return
        path, _ = QFileDialog.getSaveFileName(self, "Excelで開くCSV", str(self.ledger.directory / "exports" / "ledger.csv"), "CSV (*.csv)")
        if path:
            ids = self.filtered_ids()
            result = task(self, "CSVを書き出す", lambda progress, cancel: export_csv(
                self.ledger, path, "listing" if kind == "一覧・最新の所見" else "evaluations",
                ids, history=kind == "評価の全訂正履歴", columns=columns))
            if result is not None:
                QMessageBox.information(self, "書き出しました", f"{result} 行\n{path}\nCSVの編集は台帳へ反映されません。\n数式として解釈される文字列の先頭に引用符を付けています。")

    def ai_export(self):
        if not self.ledger or not self.guard():
            return
        ids = self.filtered_ids()
        if not ids:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("AI相談用データ")
        layout = QVBoxLayout(dialog)
        scope = QComboBox()
        scope.addItem(f"一覧に表示中の学習（{len(ids)} 件）", "filtered")
        if self.current_run:
            scope.addItem("選択中の学習だけ", "current")
        layout.addWidget(scope)
        include = QCheckBox("対応未確認のログも、仮データと明示して含める")
        layout.addWidget(include)
        info = QLabel("既定では、学習との対応を確認したログから20種類の指標を抽出します。\n"
                      "未確認の資料と欠測の理由はissues.csvに残ります。")
        info.setWordWrap(True)
        layout.addWidget(info)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if not dialog.exec():
            return
        if scope.currentData() == "current":
            ids = [self.current_run["run_id"]]
        allow_unverified = include.isChecked()
        result = task(self, "AI相談用データを書き出す", lambda progress, cancel:
                      export_bundle(self.ledger, ids, progress, cancel, allow_unverified))
        if result:
            QMessageBox.information(self, "分析用データを書き出しました",
                f"{result}\nREADME.mdとissues.csvを先に確認してください。")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(result)))

    def note_only(self):
        if not self.ledger or not self.guard():
            return
        name, ok = QInputDialog.getText(self, "メモだけ登録", "学習名（元ファイルが失われた場合など）")
        if ok and name.strip():
            try:
                run = self.ledger.create_note_run(name)
                self.reload(run["run_id"])
            except (LedgerError, OSError) as exc:
                show_error(self, exc)

    def open_folder(self):
        if self.ledger:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.ledger.directory)))

    def open_ref(self):
        row = self.refs_table.currentRow()
        if row >= 0:
            try:
                path = self.ledger.resolve_ref(self.display_refs[row])
                if not path.exists():
                    raise LedgerError("元ファイルが見つかりません")
                # Weight binaries open their containing directory.
                if path.suffix.lower() == ".safetensors":
                    path = path.parent
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except (LedgerError, OSError) as exc:
                show_error(self, exc)

    def confirm_ref(self):
        row = self.refs_table.currentRow()
        if row < 0 or not self.guard():
            return
        selected_rows = {index.row() for index in self.refs_table.selectionModel().selectedRows()} or {row}
        ref_ids = {self.display_refs[i]["ref_id"] for i in selected_rows}
        try:
            run = self.ledger.get_run(self.current_run["run_id"])
            for ref in run["source_refs"] + [a["ref"] for a in run["artifacts"]]:
                if ref["ref_id"] in ref_ids:
                    ref["association"] = {"status": "user_asserted", "method": "manual_confirmation",
                                          "note": "GUIで資料と学習の対応を確認"}
            self.ledger.save_run(run, run["revision"])
            self.reload(run["run_id"], self.active_target)
        except (LedgerError, OSError) as exc:
            show_error(self, exc)

    def verify_files(self):
        if self.current_run and self.guard():
            run_id = self.current_run["run_id"]
            task(self, "元ファイルの内容を確認", lambda progress, cancel:
                 verify(self.ledger, True, progress, cancel, [run_id]))
            self.reload(run_id)

    def closeEvent(self, event):
        event.accept() if self.guard() else event.ignore()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="LoRA研究台帳")
    parser.add_argument("--ledger", help="既存の台帳フォルダ")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("LoRA 研究台帳")
    window = LedgerWindow()
    path = args.ledger
    if not path and window.state_path.exists():
        try:
            path = read_json(window.state_path).get("last_ledger")
        except (OSError, ValueError):
            pass
    if path:
        try:
            window.activate(Ledger(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(window, "台帳を開けません", f"{path}\n{exc}\n台帳の場所を選び直してください。")
    window.show()
    return app.exec()
