"""Keep the published CSV schema and the legacy field decisions consistent."""

import json
from pathlib import Path

from library.lora_run_schema import HEADERS


def test_all_field_decisions_and_headers():
    fixture = Path(__file__).with_name("fixtures") / "csv-contract.json"
    contract = json.loads(fixture.read_text(encoding="utf-8"))
    assert set(HEADERS) == set(contract)

    counts = {"retained": 0, "derived": 0, "retired": 0}
    for stream, spec in contract.items():
        columns = HEADERS[stream]
        assert tuple(spec["columns"]) == columns, stream
        assert len(columns) == len(set(columns)), stream
        assert set(spec["retained"].values()) <= set(columns), stream

        old_names = [name for category in counts for name in spec[category]]
        assert len(old_names) == len(set(old_names)), stream
        output_keys = {name.replace("_", "").casefold() for name in columns}
        for name in spec["retired"]:
            assert name.replace("_", "").casefold() not in output_keys, (stream, name)
        for category in counts:
            counts[category] += len(spec[category])

    assert counts == {"retained": 90, "derived": 68, "retired": 33}
