"""Generate the offline diagnostic report for one indexed SDXL LoRA training run."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.lora_run_schema import HEADERS
from tools.lora_diagnostic_analysis import analyze_streams
from tools.lora_diagnostic_charts import build_diagnostics, density_summary, density_trend, report_charts
from tools.lora_diagnostic_display import build_html
from tools.lora_diagnostic_numbers import sanitize_json
from tools.lora_report_publication import (JSON_PATH, HTML_PATH, VERSION, hash_prefix, publish,
                                            record_failure, report_lock)
from tools.lora_run_reader import RunReader, owned_path
from tools.lora_run_selection import select_run


@dataclass(frozen=True)
class Options:
    loss_ma_window: int = 100
    lora_bins: int = 128
    skip_lora_analysis: bool = False
    lora_epoch_trend: bool = False


def _lora_param_counts(path):
    if path.suffix.lower() != ".safetensors":
        from tools.analyze_lora_density import group_lora_layers, load_lora_state
        return {k: parts["down"].numel() + parts["up"].numel()
                for k, parts in group_lora_layers(load_lora_state(str(path))).items() if "down" in parts and "up" in parts}
    from safetensors import safe_open
    sizes = {}
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            for side in ("up", "down"):
                suffix = f".lora_{side}.weight"
                if key.endswith(suffix):
                    sizes.setdefault(key[:-len(suffix)], {})[side] = math.prod(handle.get_slice(key).get_shape())
    return {k: parts["up"] + parts["down"] for k, parts in sizes.items() if "up" in parts and "down" in parts}


def _settings(reader):
    data = {"requested": None, "resolved": None, "status": "unavailable"}
    source = None
    try:
        path, _ = reader.artifact("training_settings", 2)
        raw = path.read_bytes()
        source = (path, len(raw), hashlib.sha256(raw).hexdigest())
        payload = json.loads(raw)
        if (payload.get("run_id"), payload.get("session_id"), payload.get("schema_version")) != (reader.run.run_id, reader.run.session_id, 2):
            data["status"] = "identity_or_version_mismatch"
        else:
            data["requested"] = payload.get("requested")
            if payload.get("settings_status") == "resolved" and isinstance(payload.get("resolved"), dict):
                data.update(status="resolved", resolved=payload["resolved"])
            else:
                data["status"] = "requested_only"
    except (KeyError, OSError, TypeError, ValueError, AttributeError):
        pass
    return data, source


def _signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns, stat.st_size



def _generate(reader, options):
    run = reader.run
    streams = {kind: reader.stream(kind) for kind in HEADERS}
    if not any(stream.rows for stream in streams.values()):
        raise ValueError("No observations: " + ", ".join(f"{kind}={stream.status}" for kind, stream in streams.items()))
    settings, settings_source = _settings(reader)
    resolved = settings["resolved"]
    verified_inputs = []
    inputs = {}
    for kind, stream in streams.items():
        record = reader.index["artifacts"].get(kind, {})
        inputs[kind] = dict(path=record.get("path"), schema_version=record.get("schema_version"), status=stream.status,
                            rows=len(stream.rows), issues=stream.issues, source_bytes=stream.source_bytes, sha256=stream.sha256)
        if stream.sha256:
            path, _ = reader.artifact(kind, 1)
            verified_inputs.append((path, stream.source_bytes, stream.sha256))
    settings_entry = reader.index["artifacts"].get("training_settings", {})
    inputs["training_settings"] = dict(path=settings_entry.get("path"), schema_version=settings_entry.get("schema_version"),
                                        status=settings["status"], source_bytes=settings_source[1] if settings_source else None,
                                        sha256=settings_source[2] if settings_source else None)
    if settings_source:
        verified_inputs.append(settings_source)
    issues = []
    weights = [r for r in reader.index["weights"] if r.get("role") in {"final", "epoch"} and r.get("status") == "saved"]
    final = None
    for item in sorted(weights, key=lambda r: (r.get("role")=="final", r.get("epoch") or 0), reverse=True):
        path, status, reason = reader.weight(item, resolved)
        if status in {"indexed", "session_matched"}:
            final = path
            break
        issues.append(f"{item.get('path')}: {reason or status}")
    param_counts = None
    weight_sources = {}
    if streams["rank"].rows and final:
        try:
            signature = _signature(final)
            param_counts = _lora_param_counts(final)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(f"rank parameter counts: {exc}")
        else:
            if _signature(final) != signature:
                raise ValueError(f"Checkpoint changed during rank parameter count: {final.name}")
            weight_sources[final] = signature
    data = analyze_streams(reader, streams, resolved, options.loss_ma_window, param_counts)
    inputs["rank"]["expected_modules_sha256"] = data["rank"]["expected_modules_sha256"]
    missing_counts = data["rank"]["missing_param_count_modules"]
    if missing_counts:
        issues.append(f"rankの正規化に必要な重みのパラメーター数がないモジュール: {len(missing_counts)}件")
    data["lora"] = data["lora_trend"] = None
    if not options.skip_lora_analysis:
        try:
            from tools.analyze_lora_density import _registered_series, analyze_series
            points, omitted = _registered_series(reader, trend=options.lora_epoch_trend)
            for _, source, _ in points:
                path = Path(source)
                weight_sources.setdefault(path, _signature(path))
            weight_data, history, failed = analyze_series(points, options.lora_bins, partial=True)
            if weight_data:
                data["lora"] = density_summary(weight_data)
            else:
                issues.append("通常finalの密度は未取得")
            if options.lora_epoch_trend:
                data["lora_trend"] = density_trend(history, str(final or ""))
                data["lora_trend"]["point_count"] = history["point_count"]
                if history["point_count"] < 2:
                    issues.append(f"密度推移は {history['point_count']} 点")
            issues.extend(f"{r.get('path')}: {r['reason']}" for r in omitted + failed)
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append(f"density: {exc}")
    elif options.lora_epoch_trend:
        issues.append("skip_lora_analysis により単体・epoch推移とも省略")
    dq = data["dq"]
    stats = dq["summary"]
    if dq["pooled"]:
        last = dq["pooled"][-1]
        stats.update(final_quant_err_ratio_ema=last["log_scope_error_ratio_ema"], final_zero_rate=last["zero_rate"])
    data["diagnostics"] = build_diagnostics(data["grad"] if data["grad"]["rows"] else None,
                                            dq if dq["scope_rows"] or dq["events"] else None,
                                            data["rank"] if data["rank"]["rows"] else None, data["lora"])
    data["charts"] = report_charts(data, data["lora_trend"])
    if data["rank"]["incomplete"]:
        issues.append(f"不完全なrank観測 {len(data['rank']['incomplete'])}件")
    issues.extend(i["reason"] for k in ("dq", "promote") for i in data[k]["issues"])
    generation = str(uuid.uuid4())
    data.update(report_schema_version=VERSION, base_name=run.base_name,
                generated_at=datetime.now().astimezone().isoformat(), training_settings=settings,
                analysis_issues=list(dict.fromkeys(issues)),
                provenance=dict(run_id=run.run_id, session_id=run.session_id, generation_id=generation,
                                started_at=reader.index["started_at"], training_started_at=(resolved or {}).get("metadata", {}).get("ss_training_started_at"),
                                output_name=reader.index["requested_output_name"], run_status=reader.index["status"],
                                association="run_index", inputs=inputs, options=asdict(options)))
    def verify_inputs():
        latest = RunReader(run.directory)
        if latest.run.run_id != run.run_id or latest.run.session_id != run.session_id:
            raise ValueError("Run identity changed during diagnosis")
        for kind in [*HEADERS, "training_settings"]:
            old = reader.index["artifacts"].get(kind, {})
            new = latest.index["artifacts"].get(kind, {})
            if any(old.get(k) != new.get(k) for k in ("path", "schema_version", "expected_modules_by_scope")):
                raise ValueError(f"Input reference changed: {kind}")
        for path, size, digest in verified_inputs:
            if hash_prefix(path, size) != digest or settings_source and path == settings_source[0] and path.stat().st_size != size:
                raise ValueError(f"Input changed: {path.name}")
        for path, signature in weight_sources.items():
            stat = path.stat()
            if (stat.st_ino, stat.st_mtime_ns, stat.st_size) != signature:
                raise ValueError(f"Checkpoint changed: {path.name}")
    report = sanitize_json(data)
    return publish(run, report, build_html(report), verify_inputs)


def generate_report(run, options):
    with report_lock(run.directory):
        try:
            return _generate(RunReader(run.directory), options)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            record_failure(run, exc)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--base_name")
    parser.add_argument("--loss_ma_window", type=int, default=100)
    parser.add_argument("--lora_bins", type=int, default=128)
    parser.add_argument("--skip_lora_analysis", action="store_true")
    parser.add_argument("--lora_epoch_trend", action="store_true")
    for field in ("training_settings", "output_dir", "output_html", "output_json"):
        parser.add_argument("--"+field, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    options = Options(args.loss_ma_window, args.lora_bins, args.skip_lora_analysis, args.lora_epoch_trend)
    flags = ["--loss_ma_window", str(options.loss_ma_window), "--lora_bins", str(options.lora_bins)]
    flags.extend("--"+name for name in ("skip_lora_analysis", "lora_epoch_trend") if getattr(options, name))
    try:
        reader = select_run(input_dir=args.input_dir, base_name=args.base_name, run_dir=args.run_dir, options=flags)
        paths = dict(output_dir=Path(JSON_PATH).parent, output_json=JSON_PATH, output_html=HTML_PATH)
        if args.training_settings is not None:
            paths["training_settings"] = reader.index["artifacts"]["training_settings"]["path"]
        for key, relative in paths.items():
            actual = getattr(args, key)
            expected = owned_path(reader.run.directory, str(relative))
            if actual is not None and actual.resolve() != expected:
                raise ValueError(f"--{key} must refer to {expected}")
        report = generate_report(reader.run, options)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
