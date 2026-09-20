"""Versioned JSON storage, transactional writes, and portable source locations."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1
REPO = Path(__file__).resolve().parents[2]
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class LedgerError(ValueError):
    pass


class Conflict(LedgerError):
    pass


class Cancelled(LedgerError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id():
    return str(uuid.uuid4())


def stable_id(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "\0".join(str(p) for p in parts)))


def checked_id(value):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise LedgerError("不正な識別IDです")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path, limit=128 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise LedgerError(f"JSONが大きすぎます: {path.name}")
    def reject(value):
        raise LedgerError(f"JSONの非有限値: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def file_stat(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": str(stat.st_mtime_ns)}


def same_stat(left, right):
    return all(left.get(k) == right.get(k) for k in ("size", "mtime_ns"))


def sha256_file(path, cancel=None):
    before = file_stat(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if cancel and cancel():
                raise Cancelled("中止しました")
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    if not same_stat(before, file_stat(path)):
        raise Conflict(f"読込中に変更されました: {path}")
    return h.hexdigest()


def validate_version(data, kind):
    if not isinstance(data, dict) or data.get("schema_version") != VERSION or data.get("kind") != kind:
        raise LedgerError(f"未対応または不正な形式です: {kind}")


@contextmanager
def directory_lock(directory):
    """Kernel locks release on crashes; thread lock also serializes local writers."""
    directory = Path(directory)
    key = os.path.normcase(str(directory.resolve()))
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        with (directory / ".ledger.lock").open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise Conflict("別の処理が台帳へ保存しています。再試行してください。") from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Ledger:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.errors = []
        if not (self.directory / "ledger.json").is_file():
            raise LedgerError("台帳がありません。「新規台帳」で作成してください。")
        if (self.directory / ".transaction.json").exists():
            with directory_lock(self.directory):
                self._recover()
        self.config()

    @classmethod
    def create(cls, directory, allow_repo=False):
        directory = Path(directory).resolve()
        if not allow_repo and directory.is_relative_to(REPO):
            raise LedgerError("台帳はsd-scriptsリポジトリの外に作成してください。")
        if directory.exists() and any(directory.iterdir()):
            raise LedgerError("新規台帳には空のフォルダを指定してください。")
        directory.mkdir(parents=True, exist_ok=True)
        for child in ("runs", "reviews", "history/runs", "research", "cache", "exports", "derived"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "locations.local.json",
                    {"schema_version": VERSION, "kind": "locations", "roots": {}})
        atomic_json(directory / "ledger.json",
                    {"schema_version": VERSION, "kind": "lora_ledger", "ledger_id": new_id(),
                     "revision": 1, "created_at": now(), "sources": [],
                     "rubric_version": 1, "default_use": "普段のキャラクター生成"})
        return cls(directory)

    def config(self):
        data = read_json(self.directory / "ledger.json")
        validate_version(data, "lora_ledger")
        checked_id(data["ledger_id"])
        if not isinstance(data.get("sources"), list):
            raise LedgerError("参照先設定が不正です")
        return data

    def locations(self):
        data = read_json(self.directory / "locations.local.json")
        validate_version(data, "locations")
        if not isinstance(data.get("roots"), dict):
            raise LedgerError("参照先の場所が不正です")
        return data["roots"]

    def resolve_ref(self, ref):
        roots = self.locations()
        if ref.get("root") not in roots:
            raise LedgerError("参照先の場所が未設定です")
        root = Path(roots[ref["root"]]).resolve()
        relative = ref.get("relative_path", "")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise LedgerError("不正な相対パスです")
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise LedgerError("参照先の外へ出るパスです")
        return result

    def _target(self, relative):
        result = (self.directory / relative).resolve()
        if not result.is_relative_to(self.directory) or result == self.directory:
            raise LedgerError("台帳の外へ書き込むことはできません")
        return result

    def _recover(self):
        journal = self.directory / ".transaction.json"
        if not journal.exists():
            return
        data = read_json(journal)
        validate_version(data, "ledger_transaction")
        for relative, value in data["writes"].items():
            atomic_json(self._target(relative), value)
        journal.unlink()

    def _commit(self, writes):
        for relative in writes:
            self._target(relative)
        atomic_json(self.directory / ".transaction.json",
                    {"schema_version": VERSION, "kind": "ledger_transaction", "writes": writes})
        self._recover()

    def set_sources(self, sources, roots, expected_revision):
        # Keeping removed roots preserves existing references.
        with directory_lock(self.directory):
            self._recover()
            config = self.config()
            if config["revision"] != expected_revision:
                raise Conflict("参照先設定が別の操作で変更されました。開き直してください。")
            all_roots = {**self.locations(), **roots}
            normalized = []
            for source in sources:
                item = copy.deepcopy(source)
                checked_id(item["id"])
                root_id = checked_id(item["root"])
                if root_id not in all_roots:
                    raise LedgerError("参照先rootがありません")
                root = Path(all_roots[root_id]).resolve()
                if root == self.directory or self.directory.is_relative_to(root) or root.is_relative_to(self.directory):
                    raise LedgerError("台帳と探索フォルダは重ならない場所にしてください。")
                if root == REPO or root.is_relative_to(REPO):
                    raise LedgerError("学習出力フォルダを指定してください（コードのリポジトリ外）。")
                item.setdefault("enabled", True)
                item.setdefault("recursive", True)
                item.setdefault("exclude", [])
                normalized.append(item)
                all_roots[root_id] = str(root)
            config.update(sources=normalized, revision=config["revision"] + 1)
            self._commit({"locations.local.json": {"schema_version": VERSION, "kind": "locations",
                                                   "roots": all_roots}, "ledger.json": config})

    def set_research_archive(self, folder):
        with directory_lock(self.directory):
            self._recover()
            config = self.config()
            locations = self.locations()
            if folder:
                path = Path(folder).resolve()
                if not (path / "data/files.csv").is_file() or not (path / "data/epoch_metrics.csv").is_file():
                    raise LedgerError("data/files.csv と data/epoch_metrics.csv がある研究フォルダを指定してください")
                key = config.get("research_archive_root") or new_id()
                locations[key] = str(path)
                config["research_archive_root"] = key
            else:
                config.pop("research_archive_root", None)
            config["revision"] += 1
            self._commit({"ledger.json": config, "locations.local.json": {
                "schema_version": VERSION, "kind": "locations", "roots": locations}})

    def list_runs(self):
        self.errors = []
        runs = []
        for path in sorted((self.directory / "runs").glob("*.json")):
            try:
                run = read_json(path)
                validate_version(run, "ledger_run")
                if checked_id(run["run_id"]) != path.stem:
                    raise LedgerError("run IDとファイル名が一致しません")
                runs.append(run)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.errors.append(f"{path.name}: {exc}")
        return runs

    def get_run(self, run_id):
        path = self.directory / "runs" / (checked_id(run_id) + ".json")
        run = read_json(path)
        validate_version(run, "ledger_run")
        return run

    def _run_writes(self, run, expected_revision):
        run_id = checked_id(run["run_id"])
        path = self.directory / "runs" / (run_id + ".json")
        previous = read_json(path) if path.exists() else None
        actual = previous["revision"] if previous else 0
        if actual != expected_revision:
            raise Conflict("登録情報が別の操作で変更されました。再読込してください。")
        run = copy.deepcopy(run)
        run.update(schema_version=VERSION, kind="ledger_run", revision=actual + 1, updated_at=now())
        result = {f"runs/{run_id}.json": run}
        if previous:
            result[f"history/runs/{run_id}/{actual:06d}.json"] = previous
        return result

    def save_run(self, run, expected_revision):
        with directory_lock(self.directory):
            self._recover()
            self._commit(self._run_writes(run, expected_revision))
        return self.get_run(run["run_id"])

    def create_note_run(self, name):
        run_id = new_id()
        run = {"schema_version": VERSION, "kind": "ledger_run", "run_id": run_id,
               "revision": 0, "display_name": name.strip() or "名前未設定",
               "identity": {}, "association": "user_asserted", "aliases": [],
               "training_status": "unknown", "purpose": "unknown",
               "dataset": {"display_name": "", "identity_status": "unknown"},
               "user": {"experiment_note": ""}, "artifacts": [], "source_refs": [],
               "settings_sources": {}, "settings": {}, "created_at": now()}
        return self.save_run(run, 0)

    def reviews(self, history=False):
        records = []
        for directory in sorted((self.directory / "reviews").iterdir()):
            if not directory.is_dir():
                continue
            revisions = []
            for path in sorted(directory.glob("[0-9]*.json")):
                try:
                    item = read_json(path)
                    validate_version(item, "ledger_review")
                    if item["review_id"] != directory.name:
                        raise LedgerError("評価IDが一致しません")
                    revisions.append(item)
                except (ValueError, OSError, KeyError) as exc:
                    self.errors.append(f"評価 {path.name}: {exc}")
            if revisions:
                records.extend(revisions if history else revisions[-1:])
        if not history:
            superseded = {r.get("reassessment_of") for r in records if r.get("reassessment_of")}
            records = [r for r in records if r["review_id"] not in superseded]
        return sorted(records, key=lambda r: (r.get("evaluated_at", ""), r["review_id"]))

    def save_review(self, review, expected_revision=0, run_update=None):
        review = copy.deepcopy(review)
        review_id = checked_id(review.setdefault("review_id", new_id()))
        candidates = {c["candidate_id"] for c in review.get("candidates", [])}
        if len(candidates) != len(review.get("candidates", [])):
            raise LedgerError("比較候補のIDが重複しています")
        for annotation in review.get("annotations", []):
            rating = annotation.get("favorite_rating")
            if rating is not None and (type(rating) is not int or not 1 <= rating <= 5):
                raise LedgerError("お気に入り度は1〜5、または未評価です")
            if annotation.get("candidate_id") and annotation["candidate_id"] not in candidates:
                raise LedgerError("所見の比較対象がありません")
            run = self.get_run(annotation["run_id"])
            if annotation.get("artifact_id") and annotation["artifact_id"] not in {
                    a["artifact_id"] for a in run["artifacts"]}:
                raise LedgerError("所見の対象checkpointがありません")
        for pair in review.get("comparisons", []):
            if pair.get("left") not in candidates or pair.get("right") not in candidates or pair["left"] == pair["right"]:
                raise LedgerError("比較対象が不正です")
            if pair.get("result") not in ("left", "right", "tie", "undecided"):
                raise LedgerError("比較結果が不正です")
            if pair.get("axis") not in ("overall", "identity", "naturalness"):
                raise LedgerError("評価軸が不正です")
        adoption = review.get("adoption", {})
        if any(c not in candidates for c in adoption.get("candidate_ids", [])):
            raise LedgerError("採用対象が不正です")
        for candidate in review.get("candidates", []):
            for component in candidate.get("components", []):
                if not component.get("run_id") and not component.get("artifact_id"):
                    component["identity_status"] = "unresolved"
                    continue
                if not component.get("run_id") or not component.get("artifact_id"):
                    raise LedgerError("重みの対応には学習とcheckpointの両方が必要です")
                run = self.get_run(component["run_id"])
                if component["artifact_id"] not in {a["artifact_id"] for a in run["artifacts"]}:
                    raise LedgerError("対象の重みが登録されていません")
        with directory_lock(self.directory):
            self._recover()
            directory = self.directory / "reviews" / review_id
            existing = sorted(directory.glob("[0-9]*.json")) if directory.exists() else []
            actual = int(existing[-1].stem) if existing else 0
            if actual != expected_revision:
                raise Conflict("評価が別の操作で変更されました。開き直してください。")
            review.update(schema_version=VERSION, kind="ledger_review",
                          revision=actual + 1, saved_at=now())
            review.setdefault("evaluated_at", now())
            review.setdefault("reviewer", "自分")
            review.setdefault("use", self.config()["default_use"])
            review.setdefault("rubric_version", self.config()["rubric_version"])
            writes = {f"reviews/{review_id}/{actual + 1:06d}.json": review}
            if run_update:
                writes.update(self._run_writes(*run_update))
            self._commit(writes)
        return review


def new_review():
    return {"review_id": new_id(), "revision": 0, "reviewer": "自分", "evaluated_at": now(),
            "evidence_mode": "current", "selection_mode": "unknown", "blinding": "unknown",
            "annotations": [], "candidates": [], "comparisons": [],
            "adoption": {"status": "undecided", "candidate_ids": []}, "evidence_refs": []}


def annotation_key(annotation):
    return (annotation.get("run_id"), annotation.get("artifact_id"), annotation.get("candidate_id"))


def latest_annotations(reviews):
    result = {}
    for review in reviews:
        for annotation in review.get("annotations", []):
            result[annotation_key(annotation)] = (review, annotation)
    return result
