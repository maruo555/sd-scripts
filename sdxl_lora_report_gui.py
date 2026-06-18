#!/usr/bin/env python
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GUI_CONFIG_PATH = SCRIPT_DIR / ".tmp" / "lora_report_gui_last.json"
DEFAULT_LBW_PRESETS = ["ALL", "XLMIDD", "XLMLT1"]


def sanitize_id(value: str, fallback: str) -> str:
    value = (value or "").strip() or fallback
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.ASCII).strip("._-")
    return value or fallback


def path_to_name(path: str) -> str:
    return Path(path).stem


@dataclass
class LoraAsset:
    asset_id: str
    name: str
    path: str
    strength: float = 0.8
    lbw: str = "XLMLT1"


@dataclass
class ConditionItem:
    asset_id: str
    name: str
    path: str
    strength: float = 0.8
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
        self.setHeaderLabels(["Condition / LoRA", "Strength", "LBW", "Path"])
        self.setColumnWidth(0, 260)
        self.setColumnWidth(1, 90)
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
        self.process: QProcess | None = None
        self.current_report: Path | None = None
        self._updating_tree = False
        self._build_ui()
        self._set_defaults()
        self.restore_last_generation_settings()

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
        self.skip_existing_check = QCheckBox("Skip existing")
        self.skip_existing_check.setChecked(True)
        self.run_button = QPushButton("Run report")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.open_report_button = QPushButton("Open report")
        self.open_report_button.setEnabled(False)
        bottom.addWidget(self.dry_run_check)
        bottom.addWidget(self.skip_existing_check)
        bottom.addStretch(1)
        bottom.addWidget(self.run_button)
        bottom.addWidget(self.stop_button)
        bottom.addWidget(self.open_report_button)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(150)
        main.addWidget(self.log_edit)

        self.run_button.clicked.connect(self.run_report)
        self.stop_button.clicked.connect(self.stop_report)
        self.open_report_button.clicked.connect(self.open_report)

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
        self.default_strength_spin = QDoubleSpinBox()
        self.default_strength_spin.setRange(-10.0, 10.0)
        self.default_strength_spin.setSingleStep(0.05)
        self.default_strength_spin.setDecimals(3)
        self.default_lbw_combo = QComboBox()
        self.default_lbw_combo.setEditable(True)
        self.default_lbw_combo.addItems(DEFAULT_LBW_PRESETS)
        defaults.addRow("Default strength", self.default_strength_spin)
        defaults.addRow("Default LBW", self.default_lbw_combo)
        layout.addLayout(defaults)

        add_button.clicked.connect(self.browse_loras)
        remove_button.clicked.connect(self.remove_selected_assets)
        make_conditions_button.clicked.connect(self.make_single_conditions)
        return panel

    def _build_condition_panel(self) -> QWidget:
        panel = QGroupBox("Comparison conditions")
        layout = QVBoxLayout(panel)
        self.condition_tree = ConditionTreeWidget()
        layout.addWidget(self.condition_tree, 1)

        controls = QGridLayout()
        add_condition_button = QPushButton("Add condition")
        add_selected_button = QPushButton("Add selected LoRA")
        duplicate_button = QPushButton("Duplicate condition")
        remove_button = QPushButton("Remove")
        controls.addWidget(add_condition_button, 0, 0)
        controls.addWidget(add_selected_button, 0, 1)
        controls.addWidget(duplicate_button, 1, 0)
        controls.addWidget(remove_button, 1, 1)
        layout.addLayout(controls)

        self.condition_tree.itemChanged.connect(self.on_condition_item_changed)
        add_condition_button.clicked.connect(self.add_empty_condition)
        add_selected_button.clicked.connect(self.add_selected_assets_to_selected_condition)
        duplicate_button.clicked.connect(self.duplicate_selected_condition)
        remove_button.clicked.connect(self.remove_selected_condition_items)
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
        self.images_spin = self._int_spin(1, 16, 1, 1)
        self.sampler_combo = QComboBox()
        self.sampler_combo.setEditable(True)
        self.sampler_combo.addItems(["euler_a", "euler", "ddim", "dpm_2", "dpm_2_a", "dpmsolver++"])
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Steps", self.steps_spin)
        form.addRow("Sampler", self.sampler_combo)
        form.addRow("Scale", self.scale_spin)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Images / prompt", self.images_spin)

        common = QGroupBox("Common args")
        layout.addWidget(common)
        common_layout = QVBoxLayout(common)
        self.sdpa_check = QCheckBox("--sdpa")
        self.bf16_check = QCheckBox("--bf16")
        self.xformers_check = QCheckBox("--xformers")
        self.extra_args_edit = QLineEdit()
        common_layout.addWidget(self.sdpa_check)
        common_layout.addWidget(self.bf16_check)
        common_layout.addWidget(self.xformers_check)
        common_layout.addWidget(QLabel("Extra args, separated by spaces"))
        common_layout.addWidget(self.extra_args_edit)

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
        self.default_strength_spin.setValue(0.8)
        self.default_lbw_combo.setCurrentText("XLMLT1")
        self.seed_values_edit.setText("12345")
        self.sdpa_check.setChecked(True)
        self.bf16_check.setChecked(True)

    def restore_last_generation_settings(self):
        if not GUI_CONFIG_PATH.exists():
            return
        try:
            with GUI_CONFIG_PATH.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            self.log(f"Could not restore last settings: {exc}")
            return

        gen_config = config.get("sdxl_gen_img", {})
        model = gen_config.get("ckpt")
        if model:
            self.model_edit.setText(str(model))

        self._set_spin_value(self.width_spin, gen_config.get("width"))
        self._set_spin_value(self.height_spin, gen_config.get("height"))
        self._set_spin_value(self.steps_spin, gen_config.get("steps"))
        self._set_spin_value(self.scale_spin, gen_config.get("scale"))
        self._set_spin_value(self.batch_spin, gen_config.get("batch_size"))
        self._set_spin_value(self.images_spin, gen_config.get("images_per_prompt"))

        sampler = gen_config.get("sampler")
        if sampler:
            self.sampler_combo.setCurrentText(str(sampler))

        self.restore_common_args(gen_config.get("common_args", []))
        self.log(f"Restored generation settings from {GUI_CONFIG_PATH}")

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
        self.sdpa_check.setChecked("--sdpa" in args)
        self.bf16_check.setChecked("--bf16" in args)
        self.xformers_check.setChecked("--xformers" in args)
        known = {"--sdpa", "--bf16", "--xformers"}
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
            asset_id = sanitize_id(path.stem, f"lora_{len(self.assets) + 1:02d}")
            base_id = asset_id
            counter = 2
            while any(asset.asset_id == asset_id for asset in self.assets):
                asset_id = f"{base_id}_{counter}"
                counter += 1
            asset = LoraAsset(
                asset_id=asset_id,
                name=path_to_name(str(path)),
                path=str(resolved),
                strength=float(self.default_strength_spin.value()),
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

    def condition_item_from_asset(self, asset: LoraAsset) -> ConditionItem:
        return ConditionItem(asset.asset_id, asset.name, asset.path, asset.strength, asset.lbw)

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
                child = QTreeWidgetItem([item.name, str(item.strength), item.lbw, item.path])
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
                    condition_item.strength = float(tree_item.text(1).strip())
                except ValueError:
                    tree_item.setText(1, str(condition_item.strength))
                condition_item.lbw = tree_item.text(2).strip() or "ALL"
                condition_item.path = tree_item.text(3).strip() or condition_item.path
                break

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
        if self.sdpa_check.isChecked():
            args.append("--sdpa")
        if self.bf16_check.isChecked():
            args.append("--bf16")
        if self.xformers_check.isChecked():
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
                        "path": item.path,
                        "strength": item.strength,
                        "lbw": item.lbw,
                    }
                )
            else:
                entry["items"] = [
                    {
                        "name": item.name,
                        "path": item.path,
                        "strength": item.strength,
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
                "width": self.width_spin.value(),
                "height": self.height_spin.value(),
                "steps": self.steps_spin.value(),
                "sampler": self.sampler_combo.currentText().strip(),
                "scale": self.scale_spin.value(),
                "batch_size": self.batch_spin.value(),
                "images_per_prompt": self.images_spin.value(),
                "common_args": self.common_args(),
            },
            "seeds": {"values": seeds, "random_count": random_count},
            "include_baseline": self.baseline_check.isChecked(),
            "loras": loras,
        }

    def write_gui_config(self, config: dict) -> Path:
        GUI_CONFIG_PATH.parent.mkdir(exist_ok=True)
        with GUI_CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return GUI_CONFIG_PATH

    def run_report(self):
        if self.process is not None:
            return
        try:
            config = self.build_config()
            config_path = self.write_gui_config(config)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return

        self.current_report = None
        self.open_report_button.setEnabled(False)
        self.log_edit.clear()
        self.log(f"Config: {config_path}")

        args = [str(SCRIPT_DIR / "sdxl_lora_report_cui.py"), "--config", str(config_path)]
        if self.dry_run_check.isChecked():
            args.append("--dry-run")
        if self.skip_existing_check.isChecked():
            args.append("--skip-existing")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(SCRIPT_DIR))
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start()

    def read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self.log(text.rstrip())
        for line in text.splitlines():
            if line.startswith("Report:"):
                self.current_report = Path(line.partition(":")[2].strip())

    def process_finished(self, exit_code: int, _exit_status):
        self.log(f"Process finished with exit code {exit_code}")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.open_report_button.setEnabled(self.current_report is not None and self.current_report.exists())
        self.process = None

    def stop_report(self):
        if self.process is not None:
            self.process.kill()

    def open_report(self):
        if self.current_report and self.current_report.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_report)))

    def log(self, message: str):
        if not message:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {message}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
