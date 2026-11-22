"""Offline-safe validation and atomic publication of a run's diagnostic JSON/HTML pair."""
from contextlib import contextmanager
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile
import time

from tools.lora_run_reader import owned_path

KIND = "diagnostic_report"
VERSION = 1
JSON_PATH = "diagnostic_report/report.json"
HTML_PATH = "diagnostic_report/report.html"


@contextmanager
def report_lock(directory):
    """Serialize same-run generators without holding the trainer's short index lock."""
    file = owned_path(directory, "diagnostic_report/.generation.lock")
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            import errno
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(.15)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class _Identity(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name") in ("run_id", "session_id", "generation_id"):
            key = attrs["name"]
            if key in self.values:
                raise ValueError("Duplicate HTML identity")
            self.values[key] = attrs.get("content")


def _read_pair(json_path, html_path, expected):
    with json_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("report_schema_version") != VERSION:
        raise ValueError("Unknown diagnostic report version")
    if {k: report["provenance"].get(k) for k in expected} != expected:
        raise ValueError("Diagnostic JSON identity mismatch")
    identity = _Identity()
    with html_path.open(encoding="utf-8") as stream:
        identity.feed(stream.read(8192))
    if identity.values != expected:
        raise ValueError("Diagnostic HTML identity mismatch")
    return report


def read_published_report(reader):
    try:
        index = reader.index["artifacts"][KIND]
        if (index.get("kind"), index.get("status"), index.get("schema_version")) != (KIND, "complete", VERSION):
            raise ValueError("Unpublished or unknown diagnostic report")
        if (index.get("path"), index.get("html_path")) != (JSON_PATH, HTML_PATH):
            raise ValueError("Unexpected diagnostic paths")
        identity = dict(run_id=reader.run.run_id, session_id=reader.run.session_id, generation_id=index["generation_id"])
        return _read_pair(owned_path(reader.run.directory, JSON_PATH),
                          owned_path(reader.run.directory, HTML_PATH), identity), None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, str(exc)


def hash_prefix(path, length=None):
    digest = sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining is None or remaining > 0:
            block = stream.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
        if remaining:
            raise ValueError(f"Input truncated during report generation: {path.name}")
    return digest.hexdigest()


def publish(run, report, html, verify_inputs):
    output = owned_path(run.directory, JSON_PATH)
    destination = owned_path(run.directory, HTML_PATH)
    paths = []
    identity = {k: report["provenance"][k] for k in ("run_id","session_id","generation_id")}
    if (identity["run_id"], identity["session_id"]) != (run.run_id, run.session_id):
        raise ValueError("Report belongs to a different run")
    try:
        for payload in (report, html):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                                             prefix=".publishing-", delete=False) as stream:
                paths.append(Path(stream.name))
                if isinstance(payload, dict):
                    json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                else:
                    stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        _read_pair(paths[0], paths[1], identity)
        verify_inputs()
        os.replace(paths[0], output)
        os.replace(paths[1], destination)
        run.set_artifact(KIND, output, schema_version=VERSION, status="complete",
                         html_path=HTML_PATH, generation_id=identity["generation_id"], last_error=None)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
    return destination


def record_failure(run, reason):
    def record(index):
        artifact = index["artifacts"].setdefault(KIND, dict(kind=KIND, path=JSON_PATH, html_path=HTML_PATH,
                                                            status="failed", schema_version=VERSION))
        artifact["last_error"] = str(reason)
    run.update_index(record)
