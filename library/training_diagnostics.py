"""Main-process CSV buffers for a single reserved run."""
import csv
import logging

from library.lora_run_schema import HEADERS, VERSION, encode_cell

logger = logging.getLogger(__name__)


class RunCsvWriter:
    def __init__(self, run, enabled, *, expected_modules_by_scope=None):
        self.run = run
        self.enabled = frozenset(enabled)
        self.pending = {kind: [] for kind in self.enabled}
        self.counts = {kind: 0 for kind in self.enabled}
        self.failed = set()
        self.closed = False
        self.group_flush_step = 0
        unknown = self.enabled - HEADERS.keys()
        if unknown:
            raise ValueError(f"Unknown CSV streams: {unknown}")
        for kind, header in HEADERS.items():
            path = run.directory / f"{kind}.csv"
            details = {}
            if kind == "rank" and kind in self.enabled:
                expected = expected_modules_by_scope or {}
                if any(len(names) != len(set(names)) for names in expected.values()):
                    raise ValueError("Expected rank module names must be unique within each scope")
                details["expected_modules_by_scope"] = {scope: sorted(names) for scope, names in expected.items()}
            state = "disabled"
            if kind in self.enabled:
                with path.open("x", encoding="utf-8-sig", newline="") as stream:
                    csv.writer(stream).writerow(header)
                state = "open"
            run.set_artifact(kind, path, schema_version=VERSION, status=state, row_count=0, **details)

    def append(self, kind, rows, *, attempt_step=None):
        if self.closed:
            raise RuntimeError("The run CSV writer is closed")
        if kind not in self.enabled or kind in self.failed:
            return
        header = HEADERS[kind]
        encoded = []
        for row in rows:
            if row.keys() - header:
                raise ValueError(f"Unexpected columns for {kind}: {row.keys() - header}")
            encoded.append(tuple(encode_cell(name, row.get(name)) for name in header))
        self.pending[kind].extend(encoded)
        if kind == "gradient":
            if encoded and int(encoded[-1][1]) % 100 == 0:
                self.flush(kind)
        elif kind == "group_loss":
            if attempt_step is not None:
                self.after_attempt(attempt_step)
        else:
            self.flush(kind)

    def after_attempt(self, step):
        if step - self.group_flush_step >= 100:
            self.flush("group_loss")
            self.group_flush_step = step

    def end_epoch(self, step):
        self.flush("group_loss")
        self.group_flush_step = step

    def _write_rows(self, kind, rows):
        path = self.run.directory / f"{kind}.csv"
        with path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def flush(self, kind=None):
        kinds = (kind,) if kind else self.enabled
        for name in kinds:
            if not self.pending.get(name) or name in self.failed:
                continue
            # Ownership changes before I/O. A short write cannot safely be retried.
            batch, self.pending[name] = self.pending[name], []
            try:
                self._write_rows(name, batch)
            except OSError:
                self.failed.add(name)
                logger.exception("CSV append failed for %s; completed records remain readable", name)
                self.run.set_artifact(name, f"{name}.csv", schema_version=VERSION,
                                      status="partial", confirmed_rows=self.counts[name])
                continue
            self.counts[name] += len(batch)

    def close(self, *, interrupted=False):
        if not self.closed:
            self.flush()
            for kind in self.enabled - self.failed:
                self.run.set_artifact(kind, f"{kind}.csv", schema_version=VERSION,
                                      status="partial" if interrupted else "complete", row_count=self.counts[kind])
            self.closed = True
        return not self.failed and not interrupted
