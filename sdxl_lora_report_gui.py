#!/usr/bin/env python
import html
import json
import random
import re
import uuid

import psutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLockFile, QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tools.lora_report_plan import attempt_directory, freeze_comparison, sanitize_id
from tools.lora_report_queue import atomic_json, find_result, is_running, process_identity

from library.generation_lora_strengths import (
    component_strengths_from_spec,
    format_strength_spec,
    normalize_strength_spec,
    serialize_strength_spec,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GUI_CONFIG_PATH = SCRIPT_DIR / ".tmp" / "lora_report_gui_last.json"
QUEUE_DIR = SCRIPT_DIR / ".tmp" / "queue"
QUEUE_STATE_PATH = QUEUE_DIR / "queue_state.json"
DEFAULT_LBW_PRESETS = ["ALL", "XLMIDD", "XLMLT1"]


class StrengthEditorWidget(QWidget):
    """Meaning-aware editor that serializes to the CLI's 1/2/3-value format."""

    MODE_LABELS = ["Common", "TE / U-Net", "TE1 / TE2 / U-Net"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODE_LABELS)
        layout.addWidget(self.mode_combo)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.common_spin = self._strength_spin()
        self.te_spin = self._strength_spin()
        self.pair_unet_spin = self._strength_spin()
        self.te1_spin = self._strength_spin()
        self.te2_spin = self._strength_spin()
        self.split_unet_spin = self._strength_spin()

        self.stack.addWidget(self._form_page([("Common", self.common_spin)]))
        self.stack.addWidget(self._form_page([("TE", self.te_spin), ("U-Net", self.pair_unet_spin)]))
        self.stack.addWidget(
            self._form_page(
                [("TE1", self.te1_spin), ("TE2", self.te2_spin), ("U-Net", self.split_unet_spin)]
            )
        )

        self._last_mode = 0
        self.set_strength_spec((0.8,))
        self.mode_combo.currentIndexChanged.connect(self._change_mode)

    @staticmethod
    def _strength_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-sys.float_info.max, sys.float_info.max)
        spin.setSingleStep(0.05)
        spin.setDecimals(12)
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return spin

    @staticmethod
    def _form_page(rows) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        for label, widget in rows:
            form.addRow(label, widget)
        return page

    def _spec_for_mode(self, mode: int) -> tuple[float, ...]:
        if mode == 0:
            return (self.common_spin.value(),)
        if mode == 1:
            return (self.te_spin.value(), self.pair_unet_spin.value())
        return (self.te1_spin.value(), self.te2_spin.value(), self.split_unet_spin.value())

    def _change_mode(self, mode: int):
        previous = component_strengths_from_spec(self._spec_for_mode(self._last_mode))
        if mode == 0:
            self.common_spin.setValue(previous.te1)
        elif mode == 1:
            self.te_spin.setValue(previous.te1)
            self.pair_unet_spin.setValue(previous.unet)
        else:
            self.te1_spin.setValue(previous.te1)
            self.te2_spin.setValue(previous.te2)
            self.split_unet_spin.setValue(previous.unet)
        self.stack.setCurrentIndex(mode)
        self._last_mode = mode

    def set_strength_spec(self, value):
        values = normalize_strength_spec(value)
        strengths = component_strengths_from_spec(values)
        self.common_spin.setValue(strengths.te1)
        self.te_spin.setValue(strengths.te1)
        self.pair_unet_spin.setValue(strengths.unet)
        self.te1_spin.setValue(strengths.te1)
        self.te2_spin.setValue(strengths.te2)
        self.split_unet_spin.setValue(strengths.unet)
        mode = len(values) - 1
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(mode)
        self.mode_combo.blockSignals(False)
        self.stack.setCurrentIndex(mode)
        self._last_mode = mode

    def strength_spec(self) -> tuple[float, ...]:
        return normalize_strength_spec(self._spec_for_mode(self.mode_combo.currentIndex()))


class StrengthDialog(QDialog):
    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LoRA component strengths")
        layout = QVBoxLayout(self)
        hint = QLabel("Choose how this LoRA is divided. Values are final inference strengths, not extra scales.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.editor = StrengthEditorWidget()
        self.editor.set_strength_spec(value)
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def strength_spec(self) -> tuple[float, ...]:
        return self.editor.strength_spec()


def path_to_name(path: str) -> str:
    return Path(path).stem


def absolutize_gui_path(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return str(p.resolve())


@dataclass
class LoraAsset:
    asset_id: str
    name: str
    path: str
    strength: tuple[float, ...] = (0.8,)
    lbw: str = "XLMLT1"


@dataclass
class ConditionItem:
    asset_id: str
    name: str
    path: str
    strength: tuple[float, ...] = (0.8,)
    lbw: str = "XLMLT1"


@dataclass
class LoraCondition:
    condition_id: str
    name: str
    items: list[ConditionItem] = field(default_factory=list)


class LoraListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.window().add_lora_paths(urls_to_paths(event.mimeData().urls()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PathDropLineEdit(QLineEdit):
    def __init__(self, kind: str, extensions: set[str] | None = None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.extensions = {ext.lower() for ext in extensions or set()}
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if self._path_from_event(event) is not None:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._path_from_event(event) is not None:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        path = self._path_from_event(event)
        if path is None:
            self.window().log(f"Drop ignored for {self.kind}: expected {self._description()}.")
            super().dropEvent(event)
            return
        self.setText(str(path))
        event.acceptProposedAction()

    def _path_from_event(self, event) -> Path | None:
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if self.kind == "folder" and path.is_dir():
                return path.resolve()
            if self.kind == "file" and path.is_file():
                if not self.extensions or path.suffix.lower() in self.extensions:
                    return path.resolve()
        return None

    def _description(self) -> str:
        if self.kind == "folder":
            return "a folder"
        if self.extensions:
            return " / ".join(sorted(self.extensions))
        return "a file"


class ConditionTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setHeaderLabels(["Condition / LoRA", "Strength(s)", "LBW", "Path"])
        self.setColumnWidth(0, 260)
        self.setColumnWidth(1, 180)
        self.setColumnWidth(2, 120)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.window().add_lora_paths(urls_to_paths(event.mimeData().urls()))
            self.window().add_selected_assets_to_selected_condition()
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


def urls_to_paths(urls: list[QUrl]) -> list[str]:
    paths = []
    for url in urls:
        if url.isLocalFile():
            path = url.toLocalFile()
            if Path(path).suffix.lower() == ".safetensors":
                paths.append(path)
    return paths


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDXL LoRA Report GUI")
        self.resize(1320, 860)
        self.assets: list[LoraAsset] = []
        self.conditions: list[LoraCondition] = []
        self.queue: list[dict] = []
        self.running_queue_index: int | None = None
        self.stop_after_current_requested = False
        self.process: QProcess | None = None
        self.process_mode: str | None = None
        self.current_report: Path | None = None
        self._updating_tree = False
        self._build_ui()
        self._set_defaults()
        self.restore_last_settings()
        self.active_attempt = None
        self.load_queue()
        self.reconcile_timer = QTimer(self)
        self.reconcile_timer.setInterval(750)
        self.reconcile_timer.timeout.connect(self.reconcile_queue)
        self.reconcile_timer.start()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        top = QGroupBox("Inputs")
        main.addWidget(top)
        top_grid = QGridLayout(top)
        self.model_edit = PathDropLineEdit("file", {".safetensors", ".ckpt"})
        self.output_edit = PathDropLineEdit("folder")
        self.prompt_edit = PathDropLineEdit("file", {".txt", ".tsv"})
        self.run_name_edit = QLineEdit()
        top_grid.addWidget(QLabel("Model"), 0, 0)
        top_grid.addWidget(self.model_edit, 0, 1)
        top_grid.addWidget(self._browse_button(self.model_edit, "Model", "Model (*.safetensors *.ckpt);;All files (*)"), 0, 2)
        top_grid.addWidget(QLabel("Output root"), 1, 0)
        top_grid.addWidget(self.output_edit, 1, 1)
        top_grid.addWidget(self._folder_button(self.output_edit, "Output root"), 1, 2)
        top_grid.addWidget(QLabel("Prompt file"), 2, 0)
        top_grid.addWidget(self.prompt_edit, 2, 1)
        top_grid.addWidget(self._browse_button(self.prompt_edit, "Prompt file", "Prompt files (*.txt *.tsv);;All files (*)"), 2, 2)
        top_grid.addWidget(QLabel("Run name"), 3, 0)
        top_grid.addWidget(self.run_name_edit, 3, 1, 1, 2)

        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter, 1)
        splitter.addWidget(self._build_asset_panel())
        splitter.addWidget(self._build_condition_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setSizes([310, 610, 360])

        bottom = QHBoxLayout()
        main.addLayout(bottom)
        self.dry_run_check = QCheckBox("Dry run")
        self.run_button = QPushButton("Run report")
        self.add_queue_button = QPushButton("Add to queue")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.open_report_button = QPushButton("Open report")
        self.open_report_button.setEnabled(False)
        bottom.addWidget(self.dry_run_check)
        bottom.addStretch(1)
        bottom.addWidget(self.run_button)
        bottom.addWidget(self.add_queue_button)
        bottom.addWidget(self.stop_button)
        bottom.addWidget(self.open_report_button)

        queue_box = QGroupBox("Queue")
        queue_layout = QVBoxLayout(queue_box)
        queue_controls = QHBoxLayout()
        self.run_queue_button = QPushButton("Run queue")
        self.stop_after_current_button = QPushButton("Stop after current")
        self.cancel_running_button = QPushButton("Cancel running")
        self.remove_queue_button = QPushButton("Remove selected")
        self.retry_button = QPushButton("Retry selected")
        self.retry_button.clicked.connect(self.retry_selected)
        self.clear_done_button = QPushButton("Clear done")
        self.load_queue_button = QPushButton("Load selected into setup")
        self.open_queue_report_button = QPushButton("Open selected report")
        self.stop_after_current_button.setEnabled(False)
        self.cancel_running_button.setEnabled(False)
        queue_controls.addWidget(self.run_queue_button)
        queue_controls.addWidget(self.stop_after_current_button)
        queue_controls.addWidget(self.cancel_running_button)
        queue_controls.addWidget(self.retry_button)
        queue_controls.addWidget(self.remove_queue_button)
        queue_controls.addWidget(self.clear_done_button)
        queue_controls.addWidget(self.load_queue_button)
        queue_controls.addStretch(1)
        queue_controls.addWidget(self.open_queue_report_button)
        queue_layout.addLayout(queue_controls)
        self.queue_tree = QTreeWidget()
        self.queue_tree.setHeaderLabels(["Status", "Run name", "Conditions", "Seeds", "Jobs", "Report"])
        self.queue_tree.setAlternatingRowColors(True)
        self.queue_tree.setMinimumHeight(150)
        self.queue_tree.setColumnWidth(0, 95)
        self.queue_tree.setColumnWidth(1, 220)
        self.queue_tree.setColumnWidth(5, 360)
        queue_layout.addWidget(self.queue_tree)
        main.addWidget(queue_box)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        main.addWidget(self.log_edit)

        self.run_button.clicked.connect(self.run_report)
        self.add_queue_button.clicked.connect(self.add_current_settings_to_queue)
        self.run_queue_button.clicked.connect(self.run_queue)
        self.stop_after_current_button.clicked.connect(self.stop_after_current)
        self.cancel_running_button.clicked.connect(self.cancel_running)
        self.remove_queue_button.clicked.connect(self.remove_selected_queue_items)
        self.clear_done_button.clicked.connect(self.clear_done_queue_items)
        self.load_queue_button.clicked.connect(self.load_selected_queue_into_setup)
        self.stop_button.clicked.connect(self.stop_report)
        self.open_report_button.clicked.connect(self.open_report)
        self.open_queue_report_button.clicked.connect(self.open_selected_queue_report)

    def _build_asset_panel(self) -> QWidget:
        panel = QGroupBox("LoRA assets")
        layout = QVBoxLayout(panel)
        hint = QLabel("Drop .safetensors files here.")
        hint.setAlignment(Qt.AlignCenter)
        self.asset_list = LoraListWidget()
        layout.addWidget(hint)
        layout.addWidget(self.asset_list, 1)

        controls = QHBoxLayout()
        add_button = QPushButton("Add files")
        remove_button = QPushButton("Remove")
        make_conditions_button = QPushButton("Make single conditions")
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        layout.addLayout(controls)
        layout.addWidget(make_conditions_button)

        defaults = QFormLayout()
        self.default_strength_editor = StrengthEditorWidget()
        self.default_lbw_combo = QComboBox()
        self.default_lbw_combo.setEditable(True)
        self.default_lbw_combo.addItems(DEFAULT_LBW_PRESETS)
        defaults.addRow("Default strengths", self.default_strength_editor)
        defaults.addRow("Default LBW", self.default_lbw_combo)
        layout.addLayout(defaults)

        add_button.clicked.connect(self.browse_loras)
        remove_button.clicked.connect(self.remove_selected_assets)
        make_conditions_button.clicked.connect(self.make_single_conditions)
        return panel

    def _build_condition_panel(self) -> QWidget:
        panel = QGroupBox("Comparison conditions")
        layout = QVBoxLayout(panel)
        strength_hint = QLabel(
            "Strength(s): 1 value = common, 2 = TE / U-Net, 3 = TE1 / TE2 / U-Net (comma-separated)"
        )
        strength_hint.setWordWrap(True)
        layout.addWidget(strength_hint)
        self.condition_tree = ConditionTreeWidget()
        # Double-click is reserved for the structured strength dialog.  F2
        # remains available for direct comma-separated cell editing.
        self.condition_tree.setEditTriggers(QAbstractItemView.EditKeyPressed)
        layout.addWidget(self.condition_tree, 1)

        controls = QGridLayout()
        add_condition_button = QPushButton("Add condition")
        add_selected_button = QPushButton("Add selected LoRA")
        duplicate_button = QPushButton("Duplicate condition")
        remove_button = QPushButton("Remove")
        move_up_button = QPushButton("Move up")
        move_down_button = QPushButton("Move down")
        edit_strength_button = QPushButton("Edit strengths")
        clear_all_button = QPushButton("Clear all")
        controls.addWidget(add_condition_button, 0, 0)
        controls.addWidget(add_selected_button, 0, 1)
        controls.addWidget(duplicate_button, 1, 0)
        controls.addWidget(remove_button, 1, 1)
        controls.addWidget(move_up_button, 2, 0)
        controls.addWidget(move_down_button, 2, 1)
        controls.addWidget(edit_strength_button, 3, 0, 1, 2)
        controls.addWidget(clear_all_button, 4, 0, 1, 2)
        layout.addLayout(controls)

        self.condition_tree.itemChanged.connect(self.on_condition_item_changed)
        self.condition_tree.itemDoubleClicked.connect(self.on_condition_item_double_clicked)
        add_condition_button.clicked.connect(self.add_empty_condition)
        add_selected_button.clicked.connect(self.add_selected_assets_to_selected_condition)
        duplicate_button.clicked.connect(self.duplicate_selected_condition)
        remove_button.clicked.connect(self.remove_selected_condition_items)
        move_up_button.clicked.connect(lambda: self.move_selected_condition(-1))
        move_down_button.clicked.connect(lambda: self.move_selected_condition(1))
        edit_strength_button.clicked.connect(self.edit_selected_condition_strengths)
        clear_all_button.clicked.connect(self.clear_all_conditions)
        return panel

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        gen = QGroupBox("Generation")
        layout.addWidget(gen)
        form = QFormLayout(gen)
        self.width_spin = self._int_spin(64, 4096, 1024, 64)
        self.height_spin = self._int_spin(64, 4096, 1024, 64)
        self.steps_spin = self._int_spin(1, 200, 20, 1)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 30.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setValue(7.0)
        self.scale_spin.setDecimals(2)
        self.batch_spin = self._int_spin(1, 16, 1, 1)
        self.sampler_combo = QComboBox()
        self.sampler_combo.setEditable(True)
        self.sampler_combo.addItems(["euler_a", "euler", "ddim", "dpm_2", "dpm_2_a", "dpmsolver++"])
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Steps", self.steps_spin)
        form.addRow("Sampler", self.sampler_combo)
        form.addRow("Scale", self.scale_spin)
        form.addRow("Batch size (1 for strict compare)", self.batch_spin)

        common = QGroupBox("Common args")
        layout.addWidget(common)
        common_layout = QFormLayout(common)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp32 / none", "bf16", "fp16"])
        self.attention_combo = QComboBox()
        self.attention_combo.addItems(["none", "sdpa", "xformers"])
        self.extra_args_edit = QLineEdit()
        common_layout.addRow("Precision", self.precision_combo)
        common_layout.addRow("Attention", self.attention_combo)
        common_layout.addRow("Extra args", self.extra_args_edit)

        seeds = QGroupBox("Seeds")
        layout.addWidget(seeds)
        seed_layout = QFormLayout(seeds)
        self.seed_values_edit = QLineEdit()
        self.random_count_spin = self._int_spin(0, 1000, 0, 1)
        self.random_seed_button = QPushButton("Generate random seeds")
        seed_layout.addRow("Seed values", self.seed_values_edit)
        seed_layout.addRow("Random count", self.random_count_spin)
        seed_layout.addRow(self.random_seed_button)

        options = QGroupBox("Options")
        layout.addWidget(options)
        options_layout = QFormLayout(options)
        self.baseline_check = QCheckBox("Include baseline")
        options_layout.addRow(self.baseline_check)

        layout.addStretch(1)
        self.random_seed_button.clicked.connect(self.generate_random_seed_values)
        return panel

    def _browse_button(self, edit: QLineEdit, title: str, file_filter: str) -> QPushButton:
        button = QPushButton("Browse")

        def browse():
            path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        return button

    def _folder_button(self, edit: QLineEdit, title: str) -> QPushButton:
        button = QPushButton("Browse")

        def browse():
            path = QFileDialog.getExistingDirectory(self, title)
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        return button

    def _int_spin(self, minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def _set_defaults(self):
        self.output_edit.setText(str((SCRIPT_DIR / ".." / "lora_reports").resolve()))
        self.run_name_edit.setText("lora_report")
        self.default_strength_editor.set_strength_spec((0.8,))
        self.default_lbw_combo.setCurrentText("XLMLT1")
        self.seed_values_edit.setText("12345")
        self.precision_combo.setCurrentText("bf16")
        self.attention_combo.setCurrentText("sdpa")

    def generation_settings(self):
        return {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "steps": self.steps_spin.value(),
            "sampler": self.sampler_combo.currentText().strip(),
            "scale": self.scale_spin.value(),
            "batch_size": self.batch_spin.value(),
        }

    def save_last_settings(self):
        settings = {
            "model": self.model_edit.text(),
            "output_root": self.output_edit.text(),
            "generation": self.generation_settings(),
            "common_args": {
                "precision": self.precision_combo.currentText(),
                "attention": self.attention_combo.currentText(),
                "extra_args": self.extra_args_edit.text(),
            },
        }
        try:
            atomic_json(GUI_CONFIG_PATH, settings)
        except OSError as exc:
            self.log(f"Could not save GUI settings: {exc}")

    def restore_last_settings(self):
        if not GUI_CONFIG_PATH.exists():
            return
        try:
            with GUI_CONFIG_PATH.open("r", encoding="utf-8") as f:
                settings = json.load(f)
            if not isinstance(settings, dict):
                raise ValueError("Expected a settings object")
            if "output_root" in settings:
                self.output_edit.setText(str(settings["output_root"]))
            self.model_edit.setText(str(settings.get("model", "")))
            self.apply_generation_settings(settings.get("generation", {}))
            common = settings.get("common_args", {})
            if "precision" in common:
                self.precision_combo.setCurrentText(common["precision"])
            if "attention" in common:
                self.attention_combo.setCurrentText(common["attention"])
            self.extra_args_edit.setText(common.get("extra_args", ""))
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            self.log(f"Could not restore GUI settings: {exc}")
            return
        self.log(f"Restored GUI settings from {GUI_CONFIG_PATH}")

    def apply_generation_settings(self, gen_config):
        self._set_spin_value(self.width_spin, gen_config.get("width"))
        self._set_spin_value(self.height_spin, gen_config.get("height"))
        self._set_spin_value(self.steps_spin, gen_config.get("steps"))
        self._set_spin_value(self.scale_spin, gen_config.get("scale"))
        self._set_spin_value(self.batch_spin, gen_config.get("batch_size"))

        sampler = gen_config.get("sampler")
        if sampler:
            self.sampler_combo.setCurrentText(str(sampler))

    def _set_spin_value(self, spin, value):
        if value is None:
            return
        try:
            spin.setValue(float(value) if isinstance(spin, QDoubleSpinBox) else int(value))
        except (TypeError, ValueError):
            return

    def restore_common_args(self, common_args):
        if not isinstance(common_args, list):
            return
        args = [str(arg) for arg in common_args]
        if "--fp16" in args:
            self.precision_combo.setCurrentText("fp16")
        elif "--bf16" in args:
            self.precision_combo.setCurrentText("bf16")
        else:
            self.precision_combo.setCurrentText("fp32 / none")

        if "--xformers" in args:
            self.attention_combo.setCurrentText("xformers")
        elif "--sdpa" in args:
            self.attention_combo.setCurrentText("sdpa")
        else:
            self.attention_combo.setCurrentText("none")

        known = {"--sdpa", "--xformers", "--bf16", "--fp16"}
        self.extra_args_edit.setText(" ".join(arg for arg in args if arg not in known))

    def browse_loras(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add LoRA files", "", "LoRA (*.safetensors);;All files (*)")
        self.add_lora_paths(paths)

    def add_lora_paths(self, paths: list[str]):
        existing = {Path(asset.path).resolve() for asset in self.assets}
        added_items = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists() or path.suffix.lower() != ".safetensors":
                continue
            resolved = path.resolve()
            if resolved in existing:
                continue
            asset_id = self.unique_asset_id(path.stem)
            asset = LoraAsset(
                asset_id=asset_id,
                name=path_to_name(str(path)),
                path=str(resolved),
                strength=self.default_strength_editor.strength_spec(),
                lbw=self.default_lbw_combo.currentText().strip() or "ALL",
            )
            self.assets.append(asset)
            item = QListWidgetItem(f"{asset.name}\n{asset.path}")
            item.setData(Qt.UserRole, asset.asset_id)
            self.asset_list.addItem(item)
            item.setSelected(True)
            added_items.append(item)
            existing.add(resolved)
        for row in range(self.asset_list.count()):
            item = self.asset_list.item(row)
            if item not in added_items:
                item.setSelected(False)
        if paths and not added_items:
            self.log("No new .safetensors files were added.")

    def unique_asset_id(self, name: str, current: str | None = None, used_ids: set[str] | None = None) -> str:
        base = sanitize_id(name, f"lora_{len(self.assets) + 1:02d}")
        asset_id = base
        used = set(used_ids) if used_ids is not None else {asset.asset_id for asset in self.assets if asset.asset_id != current}
        counter = 2
        while asset_id in used:
            asset_id = f"{base}_{counter}"
            counter += 1
        return asset_id

    def remove_selected_assets(self):
        selected_ids = {item.data(Qt.UserRole) for item in self.asset_list.selectedItems()}
        if not selected_ids:
            return
        used = {item.asset_id for condition in self.conditions for item in condition.items}
        blocked = selected_ids & used
        if blocked:
            QMessageBox.warning(self, "LoRA is in use", "Remove it from conditions before deleting the asset.")
            return
        self.assets = [asset for asset in self.assets if asset.asset_id not in selected_ids]
        self.refresh_asset_list()

    def refresh_asset_list(self):
        selected_ids = {item.data(Qt.UserRole) for item in self.asset_list.selectedItems()}
        self.asset_list.clear()
        for asset in self.assets:
            item = QListWidgetItem(f"{asset.name}\n{asset.path}")
            item.setData(Qt.UserRole, asset.asset_id)
            self.asset_list.addItem(item)
            item.setSelected(asset.asset_id in selected_ids)

    def make_single_conditions(self):
        for asset in self.selected_assets():
            condition_id = self.unique_condition_id(asset.name)
            self.conditions.append(
                LoraCondition(
                    condition_id=condition_id,
                    name=asset.name,
                    items=[self.condition_item_from_asset(asset)],
                )
            )
        self.refresh_condition_tree()

    def add_empty_condition(self):
        number = len(self.conditions) + 1
        self.conditions.append(LoraCondition(self.unique_condition_id(f"condition_{number:02d}"), f"condition_{number:02d}"))
        self.refresh_condition_tree()

    def duplicate_selected_condition(self):
        condition = self.selected_condition()
        if condition is None:
            return
        new_name = f"{condition.name}_copy"
        self.conditions.append(
            LoraCondition(
                condition_id=self.unique_condition_id(new_name),
                name=new_name,
                items=[ConditionItem(**vars(item)) for item in condition.items],
            )
        )
        self.refresh_condition_tree()

    def add_selected_assets_to_selected_condition(self):
        assets = self.selected_assets()
        if not assets:
            return
        condition = self.selected_condition()
        if condition is None:
            names = "+".join(asset.name for asset in assets)
            condition = LoraCondition(self.unique_condition_id(names), names)
            self.conditions.append(condition)
        for asset in assets:
            condition.items.append(self.condition_item_from_asset(asset))
        if len(condition.items) > 1 and condition.name.startswith("condition_"):
            condition.name = "+".join(item.name for item in condition.items)
            condition.condition_id = self.unique_condition_id(condition.name, current=condition.condition_id)
        self.refresh_condition_tree()

    def remove_selected_condition_items(self):
        item = self.condition_tree.currentItem()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            condition_id = item.data(0, Qt.UserRole)
            self.conditions = [condition for condition in self.conditions if condition.condition_id != condition_id]
        else:
            condition_id = parent.data(0, Qt.UserRole)
            item_index = item.data(0, Qt.UserRole)
            for condition in self.conditions:
                if condition.condition_id == condition_id and 0 <= item_index < len(condition.items):
                    condition.items.pop(item_index)
                    break
        self.refresh_condition_tree()

    def clear_all_conditions(self):
        if not self.conditions:
            return
        answer = QMessageBox.question(
            self,
            "Clear comparison conditions",
            "Clear all registered comparison conditions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.conditions.clear()
        self.refresh_condition_tree()

    def selected_assets(self) -> list[LoraAsset]:
        ids = [item.data(Qt.UserRole) for item in self.asset_list.selectedItems()]
        by_id = {asset.asset_id: asset for asset in self.assets}
        return [by_id[asset_id] for asset_id in ids if asset_id in by_id]

    def selected_condition(self) -> LoraCondition | None:
        item = self.condition_tree.currentItem()
        if item is None:
            return None
        if item.parent() is not None:
            item = item.parent()
        condition_id = item.data(0, Qt.UserRole)
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        return None

    def selected_condition_index(self) -> int | None:
        item = self.condition_tree.currentItem()
        if item is None:
            return None
        if item.parent() is not None:
            item = item.parent()
        condition_id = item.data(0, Qt.UserRole)
        for index, condition in enumerate(self.conditions):
            if condition.condition_id == condition_id:
                return index
        return None

    def move_selected_condition(self, offset: int):
        index = self.selected_condition_index()
        if index is None:
            return
        new_index = index + offset
        if new_index < 0 or new_index >= len(self.conditions):
            return
        self.conditions[index], self.conditions[new_index] = self.conditions[new_index], self.conditions[index]
        moved_id = self.conditions[new_index].condition_id
        self.refresh_condition_tree()
        self.select_condition_by_id(moved_id)

    def select_condition_by_id(self, condition_id: str):
        for index in range(self.condition_tree.topLevelItemCount()):
            item = self.condition_tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == condition_id:
                self.condition_tree.setCurrentItem(item)
                return

    def condition_item_from_asset(self, asset: LoraAsset) -> ConditionItem:
        # Defaults belong to condition creation, not asset registration.  This
        # makes changing the controls immediately before "Make single
        # conditions" or "Add selected LoRA" behave as the UI promises.
        return ConditionItem(
            asset.asset_id,
            asset.name,
            asset.path,
            self.default_strength_editor.strength_spec(),
            self.default_lbw_combo.currentText().strip() or "ALL",
        )

    def rebuild_assets_from_conditions(self):
        by_path: dict[str, LoraAsset] = {}
        used_asset_ids: set[str] = set()
        for condition in self.conditions:
            for item in condition.items:
                key = str(Path(item.path).resolve())
                if key not in by_path:
                    asset_id = self.unique_asset_id(Path(item.path).stem, used_ids=used_asset_ids)
                    used_asset_ids.add(asset_id)
                    by_path[key] = LoraAsset(
                        asset_id=asset_id,
                        name=item.name or path_to_name(item.path),
                        path=key,
                        strength=item.strength,
                        lbw=item.lbw,
                    )
                item.asset_id = by_path[key].asset_id
        self.assets = list(by_path.values())
        self.refresh_asset_list()

    def unique_condition_id(self, name: str, current: str | None = None) -> str:
        base = sanitize_id(name, f"condition_{len(self.conditions) + 1:02d}")
        condition_id = base
        used = {condition.condition_id for condition in self.conditions if condition.condition_id != current}
        counter = 2
        while condition_id in used:
            condition_id = f"{base}_{counter}"
            counter += 1
        return condition_id

    def refresh_condition_tree(self):
        self._updating_tree = True
        self.condition_tree.clear()
        for condition in self.conditions:
            top = QTreeWidgetItem([condition.name, "", "", ""])
            top.setData(0, Qt.UserRole, condition.condition_id)
            top.setFlags(top.flags() | Qt.ItemIsEditable)
            self.condition_tree.addTopLevelItem(top)
            for index, item in enumerate(condition.items):
                child = QTreeWidgetItem([item.name, format_strength_spec(item.strength), item.lbw, item.path])
                child.setData(0, Qt.UserRole, index)
                child.setFlags(child.flags() | Qt.ItemIsEditable)
                top.addChild(child)
            top.setExpanded(True)
        self._updating_tree = False

    def on_condition_item_changed(self, tree_item: QTreeWidgetItem, column: int):
        if self._updating_tree:
            return
        parent = tree_item.parent()
        if parent is None:
            condition_id = tree_item.data(0, Qt.UserRole)
            for condition in self.conditions:
                if condition.condition_id == condition_id:
                    condition.name = tree_item.text(0).strip() or condition.condition_id
                    condition.condition_id = self.unique_condition_id(condition.name, current=condition.condition_id)
                    tree_item.setData(0, Qt.UserRole, condition.condition_id)
                    break
            return

        condition_id = parent.data(0, Qt.UserRole)
        item_index = tree_item.data(0, Qt.UserRole)
        for condition in self.conditions:
            if condition.condition_id == condition_id and 0 <= item_index < len(condition.items):
                condition_item = condition.items[item_index]
                condition_item.name = tree_item.text(0).strip() or path_to_name(condition_item.path)
                try:
                    condition_item.strength = normalize_strength_spec(tree_item.text(1).strip())
                    tree_item.setText(1, format_strength_spec(condition_item.strength))
                except ValueError as exc:
                    tree_item.setText(1, format_strength_spec(condition_item.strength))
                    self.log(f"Invalid strength setting: {exc}")
                condition_item.lbw = tree_item.text(2).strip() or "ALL"
                condition_item.path = tree_item.text(3).strip() or condition_item.path
                break

    def on_condition_item_double_clicked(self, tree_item: QTreeWidgetItem, column: int):
        if tree_item.parent() is not None and column == 1:
            self.edit_selected_condition_strengths()

    def edit_selected_condition_strengths(self):
        tree_item = self.condition_tree.currentItem()
        if tree_item is None or tree_item.parent() is None:
            QMessageBox.information(self, "Select a LoRA", "Select a LoRA row inside a comparison condition first.")
            return

        parent = tree_item.parent()
        condition_id = parent.data(0, Qt.UserRole)
        item_index = tree_item.data(0, Qt.UserRole)
        condition_item = None
        for condition in self.conditions:
            if condition.condition_id == condition_id and 0 <= item_index < len(condition.items):
                condition_item = condition.items[item_index]
                break
        if condition_item is None:
            return

        dialog = StrengthDialog(condition_item.strength, self)
        if dialog.exec() != QDialog.Accepted:
            return
        condition_item.strength = dialog.strength_spec()
        tree_item.setText(1, format_strength_spec(condition_item.strength))

    def generate_random_seed_values(self):
        count = self.random_count_spin.value()
        if count <= 0:
            return
        seeds = [str(random.randint(0, 2**32 - 1)) for _ in range(count)]
        self.seed_values_edit.setText(", ".join(seeds))
        self.random_count_spin.setValue(0)

    def parse_seed_values(self) -> list[int]:
        text = self.seed_values_edit.text().strip()
        if not text:
            return []
        values = []
        for part in re.split(r"[\s,]+", text):
            if part:
                values.append(int(part))
        return values

    def common_args(self) -> list[str]:
        args = []
        precision = self.precision_combo.currentText()
        attention = self.attention_combo.currentText()
        if precision == "bf16":
            args.append("--bf16")
        elif precision == "fp16":
            args.append("--fp16")
        if attention == "sdpa":
            args.append("--sdpa")
        elif attention == "xformers":
            args.append("--xformers")
        args.extend(arg for arg in self.extra_args_edit.text().split() if arg)
        return args

    def build_config(self) -> dict:
        model = self.model_edit.text().strip()
        output_root = self.output_edit.text().strip()
        prompt_file = self.prompt_edit.text().strip()
        run_name = self.run_name_edit.text().strip() or "lora_report"
        if not model:
            raise ValueError("Model is required.")
        if not output_root:
            raise ValueError("Output root is required.")
        if not prompt_file:
            raise ValueError("Prompt file is required.")
        if not self.conditions and not self.baseline_check.isChecked():
            raise ValueError("Add at least one condition or enable baseline.")

        model = absolutize_gui_path(model)
        output_root = absolutize_gui_path(output_root)
        prompt_file = absolutize_gui_path(prompt_file)

        seeds = self.parse_seed_values()
        random_count = self.random_count_spin.value()
        if not seeds and random_count <= 0:
            raise ValueError("At least one seed is required.")

        loras = []
        for index, condition in enumerate(self.conditions, 1):
            if not condition.items:
                continue
            condition_id = sanitize_id(condition.condition_id or condition.name, f"condition_{index:02d}")
            entry = {"id": condition_id, "name": condition.name}
            if len(condition.items) == 1:
                item = condition.items[0]
                entry.update(
                    {
                        "path": absolutize_gui_path(item.path),
                        "strength": serialize_strength_spec(item.strength),
                        "lbw": item.lbw,
                    }
                )
            else:
                entry["items"] = [
                    {
                        "name": item.name,
                        "path": absolutize_gui_path(item.path),
                        "strength": serialize_strength_spec(item.strength),
                        "lbw": item.lbw,
                    }
                    for item in condition.items
                ]
            loras.append(entry)

        return {
            "output_root": output_root,
            "run_name": run_name,
            "prompt_file": prompt_file,
            "sdxl_gen_img": {
                "ckpt": model,
                **self.generation_settings(),
                "images_per_prompt": 1,
                "common_args": self.common_args(),
            },
            "seeds": {"values": seeds, "random_count": random_count},
            "include_baseline": self.baseline_check.isChecked(),
            "loras": loras,
        }

    def load_queue(self):
        if QUEUE_STATE_PATH.exists():
            data = json.loads(QUEUE_STATE_PATH.read_text(encoding="utf-8"))
            self.queue = data["items"]
            for item in self.queue:
                if "_comparison_plan" not in item["config"]:
                    item["status"] = "Needs setup"
        self.reconcile_queue()
        self.refresh_queue_tree()
        self.update_queue_buttons()

    def save_queue(self):
        atomic_json(QUEUE_STATE_PATH, {"items": self.queue})

    def reconcile_queue(self):
        resolved = []
        for index, item in enumerate(self.queue):
            if item["status"] != "Running" or (self.process is not None and self.running_queue_index == index):
                continue
            if self.resolve_attempt(item, orphan=True):
                resolved.append(item)
        if resolved:
            self.save_queue()
            for item in resolved:
                if item.get("attempt"):
                    self.clear_cancel_request(item["attempt"])
            self.refresh_queue_tree()
            self.update_queue_buttons()

    def resolve_attempt(self, item, *, orphan=False):
        attempt = item.get("attempt")
        if not attempt:
            item["status"] = "Waiting" if orphan else "Failed"
            return True
        try:
            result = find_result(attempt["directory"], attempt["comparison_id"], attempt["id"])
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self.log(f"Could not verify comparison: {exc}")
            item["status"] = "Failed"
            return True
        if (result and result[0] == "running") or (not result and is_running(attempt.get("process", {}))):
            return False
        if result:
            outcome, path = result
            item["status"] = {"done": "Done", "dry_run": "Dry run", "partial": "Partial",
                              "cancelled": "Canceled", "failed": "Failed"}[outcome]
            item["report_path"] = str(path)
        elif self.cancel_path(attempt).is_file():
            item["status"] = "Canceled"
        else:
            item["status"] = "Waiting" if orphan else "Failed"
        item["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return True

    def retry_selected(self):
        for index in self.selected_queue_indexes():
            item = self.queue[index]
            if item["status"] in {"Failed", "Canceled", "Partial"}:
                item["status"] = "Waiting"
        self.save_queue()
        self.refresh_queue_tree()

    def add_current_settings_to_queue(self):
        try:
            config = freeze_comparison(self.build_config(), SCRIPT_DIR)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        item = {
            "id": str(uuid.uuid4()),
            "status": "Waiting",
            "config": config,
            "summary": self.queue_summary(config),
            "options": {
                "dry_run": self.dry_run_check.isChecked(),
            },
            "report_path": "",
            "exit_code": None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": "",
            "finished_at": "",
        }
        self.queue.append(item)
        self.save_queue()
        self.refresh_queue_tree()
        self.save_last_settings()
        self.log(f"Queued: {config.get('run_name', 'lora_report')}")

    def queue_summary(self, config: dict) -> dict:
        plan = config["_comparison_plan"]
        conditions, seeds, prompts = (len(plan[key]) for key in ("conditions", "seeds", "prompts"))
        return {"run_name": config.get("run_name", "lora_report"), "conditions": conditions,
                "seeds": seeds, "prompts": prompts, "jobs": conditions * seeds * prompts}

    def refresh_queue_tree(self):
        self.queue_tree.clear()
        for index, item in enumerate(self.queue):
            summary = item.get("summary", {})
            tree_item = QTreeWidgetItem(
                [
                    item.get("status", ""),
                    str(summary.get("run_name", "")),
                    str(summary.get("conditions", "")),
                    str(summary.get("seeds", "")),
                    str(summary.get("jobs", "")),
                    item.get("report_path", ""),
                ]
            )
            tree_item.setData(0, Qt.UserRole, index)
            self.queue_tree.addTopLevelItem(tree_item)

    def selected_queue_indexes(self) -> list[int]:
        indexes = []
        for item in self.queue_tree.selectedItems():
            index = item.data(0, Qt.UserRole)
            if isinstance(index, int):
                indexes.append(index)
        return sorted(set(indexes), reverse=True)

    def run_queue(self):
        if self.process is not None or any(item["status"] == "Running" for item in self.queue):
            return
        self.stop_after_current_requested = False
        self.start_next_queue_item()

    def start_next_queue_item(self):
        for index, item in enumerate(self.queue):
            if item["status"] != "Waiting":
                continue
            try:
                item["attempt"] = self.new_attempt(item["config"])
            except (KeyError, ValueError) as exc:
                item["status"] = "Failed"
                item["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.log(f"Invalid queued comparison: {exc}")
                self.save_queue()
                self.refresh_queue_tree()
                continue
            self.running_queue_index = index
            break
        else:
            self.running_queue_index = None
            self.update_queue_buttons()
            self.log("Queue finished.")
            return

        item["status"] = "Running"
        item["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        item["finished_at"] = ""
        item["exit_code"] = None
        item["report_path"] = ""
        self.save_queue()
        self.refresh_queue_tree()
        options = item.get("options", {})
        self.start_process_for_config(
            item["config"],
            "queue",
            dry_run=bool(options.get("dry_run")), attempt=item["attempt"],
        )

    @staticmethod
    def new_attempt(config):
        attempt_id = str(uuid.uuid4())
        started_at = datetime.now()
        return {
            "id": attempt_id,
            "comparison_id": config["_comparison_plan"]["comparison_id"],
            "started_at": started_at.isoformat(),
            "directory": str(attempt_directory(config, attempt_id, started_at)),
        }

    @staticmethod
    def cancel_path(attempt):
        return QUEUE_DIR / f"cancel-{uuid.UUID(attempt['id'])}.json"

    def clear_cancel_request(self, attempt):
        try:
            self.cancel_path(attempt).unlink(missing_ok=True)
        except OSError as exc:
            self.log(f"Could not clear cancellation request: {exc}")

    def stop_after_current(self):
        self.stop_after_current_requested = True
        self.log("Queue will stop after the current job.")
        self.update_queue_buttons()

    def cancel_running(self):
        for item in self.queue:
            if item["status"] == "Running" and item.get("attempt"):
                atomic_json(self.cancel_path(item["attempt"]), {"attempt_id": item["attempt"]["id"]})
                self.log("Cancellation requested. Waiting for the worker and generator to stop.")

    def remove_selected_queue_items(self):
        if self.process_mode == "queue" and self.running_queue_index is not None:
            QMessageBox.warning(self, "Queue is running", "Remove queue items after the current queue run has stopped.")
            return
        for index in self.selected_queue_indexes():
            if 0 <= index < len(self.queue):
                self.queue.pop(index)
        self.save_queue()
        self.refresh_queue_tree()

    def clear_done_queue_items(self):
        if self.process_mode == "queue" and self.running_queue_index is not None:
            QMessageBox.warning(self, "Queue is running", "Clear done items after the current queue run has stopped.")
            return
        self.queue = [item for item in self.queue if item.get("status") not in {"Done", "Dry run"}]
        self.save_queue()
        self.refresh_queue_tree()

    def load_selected_queue_into_setup(self):
        indexes = self.selected_queue_indexes()
        if not indexes:
            return
        config = self.queue[min(indexes)].get("config")
        if not isinstance(config, dict):
            return
        try:
            self.apply_config_to_setup(config)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load queue item", str(exc))
            return
        self.log(f"Loaded queued settings: {config.get('run_name', 'lora_report')}")

    def apply_config_to_setup(self, config: dict):
        self.output_edit.setText(str(config.get("output_root") or ""))
        self.run_name_edit.setText(str(config.get("run_name") or "lora_report"))
        self.prompt_edit.setText(str(config.get("prompt_file") or ""))
        self.baseline_check.setChecked(bool(config.get("include_baseline", False)))

        gen_config = config.get("sdxl_gen_img", {})
        self.model_edit.setText(str(gen_config.get("ckpt") or ""))
        self.apply_generation_settings(gen_config)
        self.restore_common_args(gen_config.get("common_args", []))

        seed_config = config.get("seeds", {})
        self.seed_values_edit.setText(", ".join(str(seed) for seed in seed_config.get("values", [])))
        self.random_count_spin.setValue(int(seed_config.get("random_count", 0) or 0))

        self.conditions = self.conditions_from_config(config)
        self.rebuild_assets_from_conditions()
        self.refresh_condition_tree()

    def conditions_from_config(self, config: dict) -> list[LoraCondition]:
        conditions = []
        for index, raw in enumerate(config.get("loras", []), 1):
            condition_id = sanitize_id(raw.get("id") or raw.get("name"), f"condition_{index:02d}")
            raw_items = raw.get("items") or [raw]
            items = []
            for item_index, raw_item in enumerate(raw_items, 1):
                path = str(Path(raw_item.get("path", "")).resolve()) if raw_item.get("path") else ""
                name = raw_item.get("name") or path_to_name(path) or f"lora_{item_index:02d}"
                items.append(
                    ConditionItem(
                        asset_id="",
                        name=name,
                        path=path,
                        strength=normalize_strength_spec(raw_item.get("strength", 1.0)),
                        lbw=str(raw_item.get("lbw") or "ALL"),
                    )
                )
            conditions.append(LoraCondition(condition_id=condition_id, name=raw.get("name") or condition_id, items=items))
        return conditions

    def open_selected_queue_report(self):
        indexes = self.selected_queue_indexes()
        if not indexes:
            return
        item = self.queue[min(indexes)]
        report_path = item.get("report_path")
        if report_path and Path(report_path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))

    def update_queue_buttons(self):
        queue_running = self.process is not None and self.process_mode == "queue"
        idle = self.process is None and not any(item["status"] == "Running" for item in self.queue)
        self.run_queue_button.setEnabled(idle)
        self.stop_after_current_button.setEnabled(queue_running and not self.stop_after_current_requested)
        self.cancel_running_button.setEnabled(any(item["status"] == "Running" for item in self.queue))
        self.add_queue_button.setEnabled(self.process is None)
        self.remove_queue_button.setEnabled(idle)
        self.retry_button.setEnabled(idle)
        self.clear_done_button.setEnabled(idle)
        self.run_button.setEnabled(idle)

    def run_report(self):
        if self.process is not None:
            return
        try:
            config = freeze_comparison(self.build_config(), SCRIPT_DIR)
            attempt = self.new_attempt(config)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self.current_report = None
        self.open_report_button.setEnabled(False)
        self.log_edit.clear()
        self.save_last_settings()
        self.start_process_for_config(
            config,
            "single",
            dry_run=self.dry_run_check.isChecked(), attempt=attempt,
        )

    def start_process_for_config(self, config: dict, mode: str, dry_run: bool, attempt: dict):
        args = [str(SCRIPT_DIR / "sdxl_lora_report_cui.py"), "--config", "-"]
        self.active_attempt = attempt
        args.extend(["--attempt-id", attempt["id"], "--started-at", attempt["started_at"],
                     "--cancel-file", str(self.cancel_path(attempt))])
        if dry_run:
            args.append("--dry-run")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(SCRIPT_DIR))
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.started.connect(lambda: self.send_running_config(config))
        self.process.errorOccurred.connect(self.process_error)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process_mode = mode
        self.current_report = None
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_queue_buttons()
        self.log(f"Comparison output: {attempt['directory']}")
        self.process.start()

    def read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self.log(text.rstrip())

    def send_running_config(self, config):
        pid = int(self.process.processId())
        if pid:
            try:
                self.active_attempt["process"] = process_identity(pid)
            except (OSError, psutil.Error):
                pass
        try:
            if self.process_mode == "queue":
                self.save_queue()
            payload = json.dumps(config, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.process.write(payload)
        except (OSError, ValueError) as exc:
            self.log(f"Could not submit comparison: {exc}")
        finally:
            self.process.closeWriteChannel()

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.log(self.process.errorString())
            self.process_finished(-1, None)

    def process_finished(self, exit_code: int, _exit_status):
        mode = self.process_mode
        if self.process is None:
            return
        item = self.queue[self.running_queue_index] if mode == "queue" else {"attempt": self.active_attempt}
        item["exit_code"] = exit_code
        # The CUI's exit status alone cannot establish that the corresponding reports were published.
        item["status"] = "Running"
        self.resolve_attempt(item)
        self.current_report = Path(item["report_path"]) if item.get("report_path") else None
        self.log(f"Process exited {exit_code}: {item['status']}")
        self.process.deleteLater()
        self.process = None
        self.active_attempt = None
        self.process_mode = None
        self.running_queue_index = None
        self.stop_button.setEnabled(False)
        self.open_report_button.setEnabled(self.current_report is not None)
        if mode == "queue":
            self.save_queue()
            self.refresh_queue_tree()
        if item["status"] != "Running":
            self.clear_cancel_request(item["attempt"])
        self.update_queue_buttons()
        if mode == "queue" and not self.stop_after_current_requested and item["status"] != "Running":
            self.start_next_queue_item()

    def stop_report(self):
        if self.active_attempt:
            atomic_json(self.cancel_path(self.active_attempt), {"attempt_id": self.active_attempt["id"]})
            self.log("Cancellation requested. Waiting for the worker and generator to stop.")

    def closeEvent(self, event):
        if self.process is not None:
            QMessageBox.information(self, "Comparison running", "Stop the current comparison or wait for it to finish before closing.")
            event.ignore()
            return
        self.save_last_settings()
        super().closeEvent(event)

    def open_report(self):
        if self.current_report and self.current_report.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_report)))

    def log(self, message: str):
        if not message:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_edit.append(html.escape(f"[{timestamp}] {message}"))


def main():
    app = QApplication(sys.argv)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(QUEUE_DIR / "gui.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None, "SDXL LoRA Report GUI", "The comparison queue is already open.")
        return
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
