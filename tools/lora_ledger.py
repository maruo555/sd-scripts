#!/usr/bin/env python
"""Command-line entry point for the same storage and scanner used by the GUI."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.lora_ledger_core.storage import Ledger, new_id, LedgerError, read_json
from tools.lora_ledger_core.scanner import scan, apply_scan, verify
from tools.lora_ledger_core.exporting import export_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="LoRA研究台帳（元ログは読み取り専用）")
    parser.add_argument("command", choices=["init", "scan", "register", "verify", "export"])
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--source", action="append", help="追加する探索フォルダ（複数指定可）")
    parser.add_argument("--full", action="store_true", help="全内容hash検証（重みの全読込を含む）")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--selection", help="export用JSON: run_ids, include_unverified")
    parser.add_argument("--include-unverified", action="store_true")
    args = parser.parse_args(argv)
    if args.selection:
        selection = read_json(args.selection)
        if not isinstance(selection, dict) or not isinstance(selection.get("run_ids", []), list):
            parser.error("--selectionはrun_ids配列を持つJSONオブジェクトにしてください")
        args.run_id = args.run_id if args.run_id is not None else selection.get("run_ids")
        args.include_unverified = args.include_unverified or selection.get("include_unverified") is True
    ledger = Ledger.create(args.ledger) if args.command == "init" else Ledger(args.ledger)
    if args.source:
        config, roots = ledger.config(), ledger.locations()
        sources = config["sources"]
        for value in args.source:
            path = str(Path(value).resolve())
            if path in roots.values():
                continue
            root_id = new_id()
            roots[root_id] = path
            sources.append({"id": new_id(), "root": root_id, "name": Path(path).name,
                            "enabled": True, "recursive": True, "exclude": []})
        ledger.set_sources(sources, roots, config["revision"])
    if args.command == "init":
        print(ledger.directory)
    elif args.command in ("scan", "register"):
        result = scan(ledger, progress=lambda value: print(value, file=sys.stderr), full=args.full)
        summary = [{"run_id": r["run_id"], "name": r["name"], "status": r["status"],
                    "artifacts": r["artifact_count"]} for r in result["rows"]]
        if args.command == "register":
            summary = apply_scan(ledger, result, args.run_id)
        print(json.dumps({"result": summary, "issues": result["issues"],
                          "related": len(result["related"])}, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify(ledger, args.full, run_ids=args.run_id), ensure_ascii=False, indent=2))
    else:
        print(export_bundle(ledger, args.run_id, include_unverified=args.include_unverified))


if __name__ == "__main__":
    try:
        main()
    except (LedgerError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
