"""Research metrics calculated from indexed, typed run observations once per export."""
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from statistics import median

from library.lora_run_schema import HEADERS
from tools.lora_run_reader import promote_windows, unique_auto_events
from tools.lora_report_publication import hash_prefix, read_published_report

from .storage import Cancelled, Conflict, digest


DEFINITION_VERSION = "ledger-2"
INITIAL_METRICS = {
    "loss.tail_median", "loss.tail_delta", "grad_scale.tail_median",
    "dq_error_ratio.tail_median", "dq_error_ratio.tail_delta", "dq_error_rms.tail_median",
    "dq_input_rms.tail_median", "dq_clip_rate.tail_median", "dq_range_mul.tail_delta",
    "dq_auto.records", "dq_auto.applied", "dq_auto.escape_recorded",
    "rank_sat_p95.tail_median", "rank_sat_p95.tail_delta", "rank_top1_p95.tail_median",
    "rank_top1_p95.tail_delta", "rank_energy.tail_median", "rank_energy.tail_delta",
    "avg_proxy_relative_delta.tail_median", "avg_promote_applied.recorded_count",
}
DEFINITIONS = {
    "version": DEFINITION_VERSION, "initial_metric_ids": sorted(INITIAL_METRICS),
    "epoch": "one-based recorded epoch, gradient uses within-epoch batch_index",
    "window": "median within epoch, then equal-epoch median; previous=[0.6,0.8), tail=[0.8,1]",
    "minimum_epochs": 5, "tail_delta": "tail minus previous; neither may be missing",
    "rank": "complete module set. P95 of finite module observations, unweighted; energy requires all finite",
    "auto": "unique measured checks; decision, application and escape have independent missingness",
    "promote": "one decision per epoch; score (selected-raw)/abs(raw); application is independent",
    "null": "not observed, not applicable, insufficient or conflicting; never imputed as zero",
}
COUNTS = {"dq_auto.records", "dq_auto.applied", "dq_auto.escape_recorded", "avg_promote_applied.recorded_count"}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def quantile(values, q):
    values = sorted(value for value in values if number(value) is not None)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def _consistent(rows, name):
    values = [number(r.get(name)) for r in rows]
    return values[0] if values and all(v == values[0] for v in values) else None


def _dimension(**values):
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _complete_dimension(dimension):
    return all(v is not None for v in json.loads(dimension).values())


@dataclass(frozen=True)
class Series:
    stream: str
    metric: str
    dimension: str


class RunFeatures:
    def __init__(self):
        self.series = defaultdict(list)
        self.sources = {}
        self.issues = []
        self.reader = None
        self.resolved = {}
        self.expected = {}
        self.auto_coverage = False
        self.settings_verified = True

    def add(self, kind, metric, dimension, epoch, point, value):
        if type(epoch) is int and epoch >= 1:
            self.series[Series(kind, metric, dimension)].append((epoch, point, number(value)))

    @classmethod
    def load(cls, reader, cancel=None, blocked=None):
        self = cls()
        self.reader = reader
        blocked = blocked or {}
        self.settings_verified = "training_settings" not in blocked
        self.resolved, _ = reader.settings() if self.settings_verified else (None, "unverified")
        self.issues.extend(blocked.items())
        self.resolved = self.resolved or {}
        self.expected = reader.index["artifacts"].get("rank", {}).get("expected_modules_by_scope", {})
        for kind in HEADERS:
            if cancel and cancel():
                raise Cancelled("集計を中止しました")
            info = reader.index["artifacts"].get(kind) or {}
            if kind in blocked:
                self.sources[kind] = dict(path=info.get("path"), status="unverified", sha256=None, source_bytes=0)
                continue
            stream = reader.stream(kind)
            self.sources[kind] = dict(path=info.get("path"), schema_version=info.get("schema_version"),
                                      status=stream.status, rows=len(stream.rows), source_bytes=stream.source_bytes,
                                      sha256=stream.sha256)
            if stream.issues:
                self.issues.extend((kind, reason) for reason in stream.issues)
            if kind == "gradient" and stream.status == "missing" and "diagnostic_report" not in blocked:
                self._diagnostic_gradient(reader)
            else:
                self.consume(kind, stream.rows)
        return self

    def _diagnostic_gradient(self, reader):
        if "diagnostic_report" not in reader.index["artifacts"]:
            return
        report, reason = read_published_report(reader)
        if report is None:
            self.issues.append(("diagnostic_report", reason))
            return
        source = report.get("provenance", {}).get("inputs", {}).get("gradient", {})
        rows = report.get("grad", {}).get("rows")
        info = reader.index["artifacts"]["gradient"]
        if (source.get("status") not in ("complete", "open") or source.get("schema_version") != 1
                or source.get("path") != info.get("path") or not isinstance(rows, list)
                or source.get("rows") != len(rows) or not isinstance(source.get("source_bytes"), int)
                or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64
                or any(not isinstance(row, dict) or not set(HEADERS["gradient"]).issubset(row)
                       or type(row["epoch"]) is not int or type(row["batch_index"]) is not int for row in rows)):
            self.issues.append(("diagnostic_report", "missing or insufficient raw gradient provenance"))
            return
        self.sources["gradient"] = {**source, "from_report": reader.index["artifacts"]["diagnostic_report"]["generation_id"]}
        self.consume("gradient", rows)

    def verify_inputs(self):
        """An active stream may append, but none of the bytes used by an export may change."""
        from tools.lora_run_reader import RunReader
        reader = RunReader(self.reader.run.directory)
        if (reader.run.run_id, reader.run.session_id) != (self.reader.run.run_id, self.reader.run.session_id):
            raise Conflict("集計中にrunの所属が変更されました。")
        resolved, _ = reader.settings()
        if self.settings_verified and digest(resolved or {}) != digest(self.resolved):
            raise Conflict("集計中に設定が変更されました。")
        if (reader.index["artifacts"].get("rank", {}).get("expected_modules_by_scope", {}) != self.expected):
            raise Conflict("集計中に期待するrankモジュールが変更されました。")
        for kind, source in self.sources.items():
            generation = source.get("from_report")
            if generation:
                report, _ = read_published_report(reader)
                if not report or reader.index["artifacts"].get("diagnostic_report", {}).get("generation_id") != generation:
                    raise Conflict("集計中に診断の公開世代が変更されました。")
                continue
            if not source["sha256"]:
                continue
            path, info = reader.artifact(kind, 1)
            if (info["path"] != source["path"] or hash_prefix(path, source["source_bytes"]) != source["sha256"] or
                    source["status"] == "complete" and path.stat().st_size != source["source_bytes"]):
                raise Conflict(f"集計中に入力が変更されました: {kind}")

    def consume(self, kind, rows):
        plain = _dimension()
        if kind in ("gradient", "group_loss"):
            by_key = defaultdict(list)
            for r in rows:
                key = ((r["epoch"], r["batch_index"]) if kind == "gradient" else
                       (r["epoch"], r["attempt_step"], r["group"]))
                by_key[key].append(r)
            for key, records in by_key.items():
                epoch, point = key[:2]
                if point is None or len(records) != 1:
                    self.issues.append((kind, "missing or repeated coordinate"))
                values = records[0] if point is not None and len(records) == 1 else {}
                if kind == "gradient":
                    for col in ("loss", "grad_scale", "grad_norm"):
                        self.add(kind, col, plain, epoch, point, values.get(col))
                else:
                    self.add(kind, "group_loss", _dimension(group=key[2]), epoch, point, values.get("loss"))
        elif kind == "dq":
            self._dq(rows)
        elif kind == "dq_auto":
            self._auto(rows)
        elif kind == "promote":
            for epoch, d, window in promote_windows(rows):
                if not d:
                    self.issues.append((kind, f"missing or ambiguous decision: {epoch}"))
                    continue
                if window is None:
                    self.issues.append((kind, f"incomplete window detail: {epoch}"))
                dim = _dimension(mode=d["selected_candidate_mode"], bank_size=d["bank_size"], kind="proxy")
                raw, selected = number(d["raw_proxy_loss"]), number(d["selected_proxy_loss"])
                relative = ((selected - raw) / abs(raw) if d["status"] == "scored" and
                            raw not in (None, 0) and selected is not None else None)
                self.add(kind, "avg_proxy_relative_delta", dim, epoch, None, relative)
                applied = d["promote_applied"]
                self.add(kind, "avg_promote_applied.recorded_count", dim, epoch, None,
                         int(applied) if type(applied) is bool else None)
        elif kind == "rank":
            for (epoch, step, scope), records, reasons in self.reader.rank_observations(rows):
                if reasons:
                    self.issues.extend((kind, str(reason)) for reason in reasons)
                    continue
                dims = {r["module"]: r["rank_dim"] for r in records}
                dim = _dimension(scope=scope, module_ranks_sha256=digest(dims) if all(type(x) is int for x in dims.values()) else None)
                for column in ("rank_sat", "rank_top1"):
                    self.add(kind, column + "_p95", dim, epoch, step, quantile((r[column] for r in records), .95))
                energies = [number(r["rank_energy"]) for r in records]
                self.add(kind, "rank_energy", dim, epoch, step, sum(energies) if None not in energies else None)

    def _dq(self, rows):
        config = self.resolved.get("dq", {})
        auto = config.get("auto", {})
        groups = defaultdict(list)
        for row in rows:
            groups[row["epoch"], row["attempt_step"], row["target"], row["bits"]].append(row)
        for (epoch, step, target, bits), batch in groups.items():
            signature = dict(target=target, bits=bits, **{key: config.get(key) for key in
                             ("granularity", "stat", "rounding_mode", "metric_definition_version")})
            scopes = defaultdict(list)
            for row in batch:
                scopes[row["scope"]].append(row)
            for scope, parts in scopes.items():
                self.add("dq", "dq_input_rms", _dimension(scope=scope, **signature), epoch, step,
                         parts[0]["input_rms"] if len(parts) == 1 else None)
            for metric, column, scope in (("dq_error_ratio", "log_scope_error_ratio_ema", config.get("log_scope")),
                                         ("dq_error_rms", "log_scope_error_rms_ema", config.get("log_scope")),
                                         ("dq_clip_rate", "controller_clip_rate_ema", auto.get("scope")),
                                         ("dq_range_mul", "common_mul_after", "common")):
                extra = {"preset": auto.get("preset")} if metric == "dq_clip_rate" else {}
                self.add("dq", metric, _dimension(scope=scope, **signature, **extra), epoch, step,
                         _consistent(batch, column) if all(len(v) == 1 for v in scopes.values()) else None)

    def _auto(self, rows):
        config = self.resolved.get("dq", {})
        policy = config.get("auto", {})
        events = unique_auto_events(rows)
        init = events.get(0)
        checks = [r for step, r in events.items() if r is not None and r["event_type"] == "check" and step > 0]
        self.auto_coverage = bool(init and init["event_type"] == "init" and all(r is not None for r in events.values()))
        if self.auto_coverage and not checks:
            dim = _dimension(scope=init["scope"], target=init["target"], bits=init["bits"], preset=policy.get("preset"),
                             version=config.get("metric_definition_version"))
            for metric in ("dq_auto.records", "dq_auto.applied", "dq_auto.escape_recorded"):
                self.series[Series("dq_auto", metric, dim)] = []
        for step, row in events.items():
            if not row:
                self.issues.append(("dq_auto", f"duplicate check: {step}"))
                continue
            if row["event_type"] != "check" or step <= 0:
                continue
            dim = _dimension(scope=row["scope"], target=row["target"], bits=row["bits"], preset=policy.get("preset"),
                             version=config.get("metric_definition_version"))
            self.add("dq_auto", "dq_auto.records", dim, row["epoch"], step, 1)
            applied = row["auto_applied"]
            self.add("dq_auto", "dq_auto.applied", dim, row["epoch"], step, int(applied) if type(applied) is bool else None)
            decision = row["low_auto_decision"]
            escape = int(decision == "escape_to_mid") if isinstance(decision, str) and decision else None
            self.add("dq_auto", "dq_auto.escape_recorded", dim, row["epoch"], step, escape)

    def groups(self, series, cutoff=None):
        result = defaultdict(list)
        for epoch, point, value in self.series[series]:
            if cutoff is not None:
                last = cutoff.get("epoch")
                if type(last) is not int or epoch > last:
                    continue
                if epoch == last:
                    if series.stream == "promote" and cutoff.get("phase") != "after_epoch_decision":
                        continue
                    if cutoff.get("role") == "step":
                        limit = cutoff.get("batch_index" if series.stream == "gradient" else "attempt_step")
                        if type(limit) is not int or type(point) is not int or point > limit:
                            continue
            result[epoch].append(value)
        return result

    def summarize(self, cutoff=None, internal=False):
        rows = []
        for series in self.series:
            groups = self.groups(series, cutoff)
            rows.extend(self._summaries(series, groups, internal))
        for row in rows:
            row["cutoff"] = {k: cutoff.get(k) for k in ("role", "epoch", "attempt_step", "batch_index", "phase")} if cutoff else None
            if cutoff and cutoff.get("role") == "step":
                clock = "batch_index" if row["source_kind"] == "gradient" else "attempt_step"
                if cutoff.get(clock) is None:
                    row["range_note"] = f"missing checkpoint {clock}; through prior complete epoch"
        return rows

    def _summaries(self, series, groups, internal=False):
        epochs = sorted(groups)
        finite = {e: [v for v in values if v is not None] for e, values in groups.items()}
        valid = sum(len(items) for items in finite.values())
        missing = sum(len(v) for v in groups.values()) - valid
        n = len(epochs)
        prev = [e for i, e in enumerate(epochs) if n >= 5 and .6 <= i / (n - 1) < .8]
        tail = [e for i, e in enumerate(epochs) if n >= 5 and i / (n - 1) >= .8]
        def window(subset):
            values = [median(finite[e]) for e in subset if finite[e]]
            return median(values) if values else None
        before, after = window(prev), window(tail)
        base = dict(dimension=series.dimension, epochs=epochs, n_valid=valid, n_missing=missing, previous_epochs=prev,
                    tail_epochs=tail, previous=before, source_kind=series.stream, definition_version=DEFINITION_VERSION)
        if series.metric in COUNTS:
            observed = bool(valid) or (series.metric == "dq_auto.records" and self.auto_coverage)
            return [{**base, "metric_id": series.metric, "value": sum(sum(v) for v in finite.values()) if observed else None,
                     "status": "recorded_events" if observed else "unrecorded"}]
        ok = before is not None and after is not None
        base["status"] = "ok" if ok else "insufficient_epochs"
        rows = [{**base, "metric_id": f"{series.metric}.tail_median", "value": after if ok else None},
                {**base, "metric_id": f"{series.metric}.tail_delta", "value": after - before if ok else None}]
        return [r for r in rows if internal or r["metric_id"] in INITIAL_METRICS]

    def compare(self, other, left_cutoff=None, right_cutoff=None):
        """Match comparable dimensions and the intersection of their measured epochs."""
        pairs = []
        for series in self.series.keys() & other.series.keys():
            if series.stream == "dq_auto" or not _complete_dimension(series.dimension):
                continue
            left = self.groups(series, left_cutoff)
            right = other.groups(series, right_cutoff)
            allowed = left.keys() & right.keys()
            for a, b in ((left_cutoff, right_cutoff), (right_cutoff, left_cutoff)):
                if a and a.get("role") == "step" and a.get("epoch") in allowed:
                    key = "batch_index" if series.stream == "gradient" else "attempt_step"
                    epoch = a["epoch"]
                    if (not b or b.get("role") != "step" or epoch != b.get("epoch") or type(a.get(key)) is not int
                            or a.get(key) != b.get(key)):
                        allowed = allowed - {epoch}
                        continue
                    def bounds(owner):
                        points = [p for e, p, _ in owner.series[series] if e == epoch and type(p) is int and p <= a[key]]
                        return (min(points), max(points)) if points else None
                    if bounds(self) is None or bounds(self) != bounds(other):
                        allowed = allowed - {epoch}
            a = self._summaries(series, {e: left[e] for e in allowed})
            b = other._summaries(series, {e: right[e] for e in allowed})
            pairs.extend(zip(a, b))
        return pairs
