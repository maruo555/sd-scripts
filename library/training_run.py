"""Reservation, atomic JSON publication and ownership of one standard SDXL LoRA run."""
import contextlib
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid


_INDEX_LOCK = threading.RLock()
_RESERVED = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)


def _units(text):
    return len(str(text).encode("utf-16-le")) // 2


def _safe_stem(output_name, root, timestamp):
    raw = str(output_name) if output_name is not None else ""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", raw).rstrip(" .") or "last"
    if _RESERVED.match(stem):
        stem = "_" + stem
    # Reserve length for collision counters, publication temporary names, long step
    # numbers and the upstream sample name. Count Windows UTF-16 code units.
    budget = min((245 - _units(root / (timestamp + "_999999") / "-step0000000000-final_raw.safetensors")) // 2,
                 245 - _units(root / "sample" / ("0" * 36 + "_0000000000_00000000_00000000_00000000000000000000.png")),
                 245 - _units(root / (timestamp + "_999999") / "diagnostic_report" / "lora_diagnostic_report.html"))
    if budget < 10:
        raise ValueError(f"Output root leaves insufficient space for run artifacts: {root}")
    if _units(stem) > budget:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        clipped = []
        used = 0
        for char in stem:
            size = _units(char)
            if used + size > budget - 9:
                break
            clipped.append(char)
            used += size
        stem = "".join(clipped) + "_" + digest
    return stem


def write_json(path, data):
    """Install only a completely serialized and flushed JSON document."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".writing-",
                                     suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _index_lock(directory):
    # Unlike an exclusive-create lockfile this OS lock is released after a crash.
    with _INDEX_LOCK, (directory / ".run-index.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class TrainingRun:
    directory: Path
    run_id: str
    session_id: str
    base_name: str
    output_name_omitted: bool

    @property
    def output_root(self):
        return self.directory.parent

    @classmethod
    def create(cls, output_root, output_name, session_id, *, now=None):
        root = Path(output_root).resolve()
        started = (now or dt.datetime.now().astimezone()).astimezone()
        timestamp = started.strftime("%Y%m%d-%H%M%S")
        base_name = _safe_stem(output_name, root, timestamp)
        root.mkdir(parents=True, exist_ok=True)
        counter = 1
        while True:
            suffix = "" if counter == 1 else f"_{counter:02d}"
            directory = root / f"{timestamp}_{base_name}{suffix}"
            try:
                directory.mkdir()
                break
            except FileExistsError:
                counter += 1
        run = cls(directory, str(uuid.uuid4()), str(session_id), base_name, output_name is None)
        write_json(directory / "run.json", dict(schema_version=1, kind="training_run", run_id=run.run_id,
                   session_id=run.session_id, started_at=started.isoformat(), folder=directory.name,
                   requested_output_name=output_name, output_name_omitted=run.output_name_omitted,
                   base_name=base_name, status="running", flush_status="pending", artifacts={}, weights=[]))
        return run

    @classmethod
    def open(cls, directory):
        directory = Path(directory).resolve()
        with (directory / "run.json").open(encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("schema_version") != 1 or data.get("kind") != "training_run":
            raise ValueError(f"Unsupported run index in {directory}")
        uuid.UUID(data["run_id"])
        return cls(directory, data["run_id"], data["session_id"], data["base_name"], data["output_name_omitted"])

    def relative_path(self, path):
        path = Path(path)
        resolved = (self.directory / path).resolve()
        if resolved == self.directory or not resolved.is_relative_to(self.directory):
            raise ValueError(f"Artifact outside this run: {path}")
        return resolved.relative_to(self.directory).as_posix()

    def update_index(self, change):
        path = self.directory / "run.json"
        with _index_lock(self.directory):
            with path.open(encoding="utf-8") as stream:
                index = json.load(stream)
            if (index.get("run_id"), index.get("session_id")) != (self.run_id, self.session_id):
                raise ValueError(f"Run identity changed: {self.directory}")
            change(index)
            write_json(path, index)

    def set_artifact(self, kind, path, *, schema_version, status, **details):
        relative = self.relative_path(path)
        def update(index):
            artifact = index["artifacts"].setdefault(kind, {})
            artifact.update(kind=kind, path=relative, schema_version=schema_version, status=status, **details)
        self.update_index(update)

    def finish(self, status, flush_status):
        if status not in {"completed", "stopped", "failed"} or flush_status not in {"complete", "partial"}:
            raise ValueError(f"Unknown run outcome: {status}, {flush_status}")
        self.update_index(lambda index: index.update(status=status, flush_status=flush_status,
                          finished_at=dt.datetime.now().astimezone().isoformat()))
