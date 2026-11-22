#!/usr/bin/env python
"""CLI entry point shared with the research ledger GUI."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.lora_ledger_core.output_root import open_output_ledger
from tools.lora_ledger_core.scanner import apply_scan, scan, verify
from tools.lora_ledger_core.storage import LedgerError, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="LoRA研究台帳")
    parser.add_argument("command", choices=("init", "scan", "register", "verify", "export"))
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--full", action="store_true", help="重みの全内容SHAを確認")
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--selection", help="export用JSON（run_ids配列）")
    args = parser.parse_args(argv)
    if args.selection:
        if args.command != "export":
            parser.error("--selectionはexport専用です。")
        selection = read_json(args.selection)
        if (not isinstance(selection, dict) or set(selection) - {"run_ids"} or
                not isinstance(selection.get("run_ids"), list) or
                not all(isinstance(rid, str) for rid in selection["run_ids"])):
            parser.error("--selectionにはrun_ids配列のみを指定してください。")
        if args.run_id is None:
            args.run_id = selection["run_ids"]
    ledger = open_output_ledger(args.ledger, create=args.command == "init")
    if args.command == "init":
        print(ledger.directory)
        return 0
    if args.command == "export":
        from tools.lora_ledger_core.exporting import export_bundle
        print(export_bundle(ledger, args.run_id))
        return 0
    if args.command == "verify":
        result = verify(ledger, full=args.full, run_ids=args.run_id)
    else:
        result = scan(ledger, full=args.full, run_ids=args.run_id)
        if args.command == "register":
            result = {**apply_scan(ledger, result, args.run_id), "issues": result["issues"]}
        else:
            result = {"issues": result["issues"],
                      "rows": [{k: r[k] for k in ("run_id", "name", "status", "artifact_count")}
                               for r in result["rows"] if args.run_id is None or r["run_id"] in args.run_id]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors") or result.get("cancelled") else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (LedgerError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
