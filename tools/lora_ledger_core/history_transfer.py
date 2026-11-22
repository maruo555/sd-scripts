"""Verified handover of version-1 ledgers and inspection of saved analysis bundles."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .storage import Conflict, Ledger, LedgerError, read_json, sha256_file, validate_version


BUNDLE_FILES = (
    "runs.csv", "candidates.csv", "comparisons.csv", "features.csv", "issues.csv",
    "reviews.jsonl", "feature_definitions.json", "export_manifest.json", "README.md",
)


def _walk_error(error):
    raise error


def _require_regular_entry(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise LedgerError(f"台帳外の参照をコピーできません: {path.name}")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise LedgerError(f"通常のファイルまたはフォルダが必要です: {path.name}")
    return info


def _inventory(directory):
    inventory = {}
    _require_regular_entry(directory)
    for parent, dirs, files in os.walk(directory, onerror=_walk_error, followlinks=False):
        for name in [*dirs, *files]:
            entry = Path(parent, name)
            info = _require_regular_entry(entry)
            relative = entry.relative_to(directory).as_posix()
            inventory[relative] = (None if stat.S_ISDIR(info.st_mode)
                                   else (info.st_size, sha256_file(entry)))
    return inventory


def open_legacy_ledger(directory):
    """Open a stopped historical ledger without creating locks or recovering its journal."""
    if (Path(directory) / ".transaction.json").exists():
        raise LedgerError("旧台帳に未処理のjournalがあります。回復は検証コピーで行ってください。")
    return Ledger(directory, read_only=True)


def copy_legacy_ledger(source, destination):
    """Writers/exports must be stopped. Recover pending transactions only on the verified copy."""
    source = Path(source).absolute()
    _require_regular_entry(source)
    source = source.resolve()
    destination = Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise LedgerError("旧台帳と移植先には重ならないフォルダを指定してください。")
    if destination.exists():
        raise LedgerError("移植先はまだ存在しないフォルダを指定してください。")
    if not (source / "ledger.json").is_file():
        raise LedgerError("移植元にledger.jsonがありません。")

    before = _inventory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".ledger-copy-") as temp:
        staged = Path(temp, "verified")
        # Preserve links as links during the copy. The inventory rejects any link,
        # including one introduced in the interval after the first inventory.
        shutil.copytree(source, staged, symlinks=True)
        if _inventory(staged) != before or _inventory(source) != before:
            raise Conflict("コピー中に旧台帳が変化しました。旧台帳の保存を停止して再試行してください。")
        copied = Ledger(staged)
        copied.locations()
        if destination.exists():
            raise Conflict("コピー中に別の台帳が移植先に作成されました。")
        staged.rename(destination)
    return Ledger(destination)


@dataclass(frozen=True)
class SavedExport:
    directory: Path
    manifest: dict | None
    missing_files: tuple[str, ...]

    @property
    def is_available(self):
        return self.manifest is not None and self.manifest["status"] == "complete" and not self.missing_files


def inspect_saved_export(directory):
    """Keep the as-saved manifest and current file presence separate. No recomputation."""
    directory = Path(directory)
    manifest_path = directory / "export_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        validate_version(manifest, "ledger_export")
        if manifest.get("status") not in ("complete", "incomplete"):
            raise LedgerError("export_manifest.jsonのstatusが不正です。")
    return SavedExport(directory, manifest,
                       tuple(name for name in BUNDLE_FILES if not (directory / name).is_file()))


def list_saved_exports(ledger):
    return [inspect_saved_export(path) for path in sorted((ledger.directory / "derived").glob("*"))
            if path.is_dir()]
