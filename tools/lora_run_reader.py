"""Read indexed SDXL run data without importing the training, GUI or tensor stack."""
from collections import defaultdict
import csv
import hashlib
from dataclasses import dataclass, field
import json
import struct
from pathlib import Path, PureWindowsPath

from library.lora_run_schema import HEADERS, VERSION, decode_cell
from library.training_run import TrainingRun


@dataclass
class CsvResult:
    kind: str
    status: str
    rows: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    invalid_cells: int = 0
    source_bytes: int = 0
    sha256: str | None = None


class _Lines:
    """Decode one physical line at a time so corruption cannot discard an earlier record."""
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.byte_count = 0
        self.first = True
        self.terminated = False

    def __iter__(self):
        return self

    def __next__(self):
        raw = next(self.stream)
        self.sha.update(raw)
        self.byte_count += len(raw)
        self.terminated = raw.endswith(b"\r\n")
        if self.first:
            self.first = False
            if not raw.startswith(b"\xef\xbb\xbf"):
                raise ValueError("missing UTF-8 BOM")
            raw = raw[3:]
        return raw.decode("utf-8")


def read_csv(path, kind):
    result = CsvResult(kind, "complete")
    columns = HEADERS[kind]
    try:
        with Path(path).open("rb") as stream:
            lines = _Lines(stream)
            reader = csv.reader(lines, strict=True)
            if tuple(next(reader, ())) != columns or not lines.terminated:
                raise ValueError("incomplete or unsupported CSV header")
            result.source_bytes, result.sha256 = lines.byte_count, lines.sha.hexdigest()
            for cells in reader:
                if not lines.terminated:
                    raise ValueError(f"record at line {reader.line_num} missing CRLF")
                # A CSV record may span several physical CRLF lines. Snapshot only a parsed record.
                result.source_bytes, result.sha256 = lines.byte_count, lines.sha.hexdigest()
                if len(cells) != len(columns):
                    result.issues.append(f"line {reader.line_num}: wrong column count")
                    continue
                data = {}
                for column, value in zip(columns, cells):
                    try:
                        data[column] = decode_cell(column, value)
                    except (ValueError, OverflowError):
                        data[column] = None
                        result.invalid_cells += 1
                        if result.invalid_cells <= 10:
                            result.issues.append(f"line {reader.line_num}: invalid {column}")
                result.rows.append(data)
    except FileNotFoundError:
        result.status = "missing"
        return result
    except (OSError, ValueError, csv.Error) as exc:
        result.issues.append(str(exc))
    if result.issues:
        result.status = "partial"
    return result


def owned_path(directory, relative):
    if not isinstance(relative, str) or not relative or PureWindowsPath(relative).anchor or Path(relative).is_absolute():
        raise ValueError(f"Invalid relative path {relative!r}")
    base = Path(directory).resolve()
    path = (base / relative).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f"Path escapes run: {relative}")
    return path


def _weight_metadata(path):
    if path.suffix.lower() != ".safetensors":
        return {}
    with path.open("rb") as stream:
        size_raw = stream.read(8)
        if len(size_raw) != 8:
            raise ValueError("truncated safetensors header")
        size = struct.unpack("<Q", size_raw)[0]
        if size < 2 or size > min(path.stat().st_size - 8, 100_000_000):
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(size))
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("invalid safetensors metadata")
    return metadata


class RunReader:
    def __init__(self, directory):
        self.run = TrainingRun.open(directory)
        with (self.run.directory / "run.json").open(encoding="utf-8") as stream:
            self.index = json.load(stream)
        if (self.index.get("run_id"), self.index.get("session_id")) != (self.run.run_id, self.run.session_id):
            raise ValueError("Run identity changed while opening")

    def artifact(self, kind, version):
        item = self.index["artifacts"].get(kind)
        if not item:
            raise ValueError(f"Unindexed artifact: {kind}")
        if item.get("kind") != kind or item.get("schema_version") != version:
            raise ValueError(f"Unknown {kind} index entry")
        return owned_path(self.run.directory, item["path"]), item

    def stream(self, kind):
        try:
            path, info = self.artifact(kind, VERSION)
            if info["status"] == "disabled":
                return CsvResult(kind, "disabled")
            result = read_csv(path, kind)
            if result.status == "complete":
                if info["status"] in ("open", "partial"):
                    result.status = info["status"]
                if info["status"] == "complete" and len(result.rows) != info.get("row_count"):
                    result.status = "partial"
                    result.issues.append("closed stream row count differs from index")
            return result
        except (KeyError, TypeError, ValueError) as exc:
            return CsvResult(kind, "unverified", issues=[str(exc)])

    def weight(self, item, resolved=None):
        """Resolve a registered checkpoint; verify identity using metadata alone."""
        path = None
        try:
            if not isinstance(item, dict) or item not in self.index["weights"]:
                raise ValueError("Weight not present in run index")
            path = owned_path(self.run.directory, item["path"])
            phases = {"step": "step_complete", "epoch": "before_epoch_decision",
                      "final_raw": "before_epoch_decision", "final": "after_epoch_decision"}
            if item.get("role") not in phases or item.get("phase") != phases[item["role"]]:
                raise ValueError("Unknown checkpoint role or capture phase")
            if item["status"] in {"pruned", "missing"}:
                return path, item["status"], None
            if item["status"] != "saved":
                raise ValueError("Unknown indexed checkpoint state")
            if not path.is_file():
                return path, "missing", "Indexed checkpoint is missing"
            metadata = _weight_metadata(path)
            settings_metadata = (resolved or {}).get("metadata", {})
            if not isinstance(settings_metadata, dict):
                settings_metadata = {}
            # ss_training_started_at is set by the upstream trainer after run reservation.
            # The ISO timestamp of the run folder is independent of this weight metadata.
            for key, expected in (("ss_run_id", self.run.run_id), ("ss_session_id", self.run.session_id),
                                  ("ss_training_started_at", settings_metadata.get("ss_training_started_at"))):
                if key in metadata and expected is not None and metadata[key] != str(expected):
                    raise ValueError(f"Checkpoint metadata conflicts with this run: {key}")
            status = "session_matched" if "ss_session_id" in metadata else "indexed"
            return path, status, None
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return path, "unverified", str(exc)

    def settings(self):
        try:
            path, _ = self.artifact("training_settings", 2)
            with path.open(encoding="utf-8") as stream:
                data = json.load(stream)
            if (data.get("schema_version"), data.get("run_id"), data.get("session_id")) != (2, self.run.run_id, self.run.session_id):
                return None, "identity_or_version_mismatch"
            if data.get("settings_status") != "resolved" or not isinstance(data.get("resolved"), dict):
                return None, "requested_only"
            return data["resolved"], "resolved"
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None, "unavailable"

    def rank_observations(self, rows):
        expected = self.index["artifacts"].get("rank", {}).get("expected_modules_by_scope", {})
        return rank_observations(rows, expected)


def rank_observations(rows, expected_by_scope):
    """Yield coordinates, module records, and reasons excluding an observation from totals."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["epoch"], row["attempt_step"], row["scope"]].append(row)
    for key, records in groups.items():
        expected = expected_by_scope.get(key[2])
        reasons = []
        if None in key:
            reasons.append("missing observation coordinate")
        names = [row["module"] for row in records]
        if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
            reasons.append("unknown expected module set")
        elif set(names) != set(expected):
            reasons.append("missing or unexpected module")
        if len(names) != len(set(names)):
            reasons.append("duplicate module")
        lr_fields = HEADERS["rank"][-6:]
        if any(any(row[k] != records[0][k] for k in lr_fields) for row in records[1:]):
            reasons.append("inconsistent learning rates")
        yield key, records, reasons


def promote_windows(rows):
    by_epoch = defaultdict(list)
    for row in rows:
        by_epoch[row["epoch"]].append(row)
    for epoch, records in by_epoch.items():
        decisions = [r for r in records if r["row_type"] == "decision"]
        members = [r for r in records if r["row_type"] == "window_epoch"]
        decision = decisions[0] if len(decisions) == 1 else None
        positions = [r["window_position"] for r in members]
        count = decision["window_count"] if decision else None
        valid = (epoch is not None and isinstance(count, int) and count >= 0 and len(records) == 1 + len(members)
                 and len(members) == count and set(positions) == set(range(1, count + 1))
                 and all(r["window_epoch"] is not None for r in members))
        window = [r["window_epoch"] for r in sorted(members, key=lambda r: r["window_position"])] if valid else None
        yield epoch, decision, window


def unique_auto_events(rows):
    """Explicit, unambiguous references only; step zero denotes the initialization event."""
    result = {}
    for row in rows:
        step = row["attempt_step"]
        if step is not None:
            result[step] = row if step not in result else None
    return result


def discover_run_indexes(output_root):
    """Return indexed runs immediately under output_root and per-directory discovery issues.

    Duplicate UUIDs are retained so a caller can stop association or publish a precise diagnostic.
    The caller decides how to present an offline root; enumeration errors propagate.
    """
    import stat
    root = Path(output_root).resolve()
    runs, issues = [], []
    skip_names = {"ledger", "cache", "exports", "derived", "diagnostic_report", "density", "sample"}
    for directory in sorted(root.iterdir()):
        if directory.name.startswith(".") or directory.name.casefold() in skip_names:
            continue
        try:
            for entry in (directory, directory / "run.json"):
                info = entry.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ValueError("Training run or run.json is a link")
                if entry == directory and not stat.S_ISDIR(info.st_mode):
                    break
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Expected a regular run.json")
                runs.append(RunReader(directory))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append({"directory": directory.name, "reason": str(exc)})
    return runs, issues
