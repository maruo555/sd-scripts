"""Bind the one nonrecursive training output root of a local ledger."""
from pathlib import Path

from .history_transfer import open_legacy_ledger
from .storage import Ledger, LedgerError, VERSION, new_id


def open_output_ledger(directory, *, create=False):
    directory = Path(directory).resolve()
    if directory.name.casefold() != "ledger":
        if create:
            raise LedgerError("新規台帳は対象の出力フォルダ直下の ledger に作成してください。")
        return open_legacy_ledger(directory)
    ledger = Ledger.create(directory) if create else Ledger(directory)
    with ledger.write_lock():
        ledger._recover()
        config, locations = ledger.config(), ledger.locations()
        root_id = config.get("training_output_root")
        if root_id and root_id in locations and Path(locations[root_id]).resolve() == directory.parent:
            return ledger
        if not root_id:
            root_id = new_id()
            config["training_output_root"] = root_id
        locations[root_id] = str(directory.parent)
        config["revision"] += 1
        ledger._commit({"ledger.json": config, "locations.local.json":
                        {"schema_version": VERSION, "kind": "locations", "roots": locations}})
    return ledger


def output_root(ledger):
    root_id = ledger.config().get("training_output_root")
    roots = ledger.locations()
    if (ledger.read_only or not root_id or root_id not in roots or
            Path(roots[root_id]).resolve() != ledger.directory.parent):
        raise LedgerError("新runの探索は選択した出力フォルダ直下の台帳から開始してください。")
    return root_id, ledger.directory.parent
