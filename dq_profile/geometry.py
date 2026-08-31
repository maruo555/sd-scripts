from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _normalized(value: str) -> str:
    return str(value).strip().replace("\\", "/").casefold()


@dataclass(frozen=True)
class SourceRule:
    pattern: str
    source_group: str
    match: str = "exact"
    probe_selected: bool = True
    probe_selection_rank: Optional[int] = None


class SourceGroupMap:
    """Resolve post-collation image keys to independent source groups.

    JSON accepts either ``{"image key": "source"}`` or a list of records.
    CSV records use ``image_key`` (or ``pattern``), ``source_group`` and an
    optional ``match`` column (``exact`` or ``prefix``).
    """

    def __init__(self, rules: Iterable[SourceRule] = ()) -> None:
        self.rules = tuple(rules)

    @classmethod
    def load(cls, path: Optional[str | Path]) -> "SourceGroupMap":
        if path in (None, ""):
            return cls()
        source = Path(path).expanduser().resolve()
        if source.suffix.casefold() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            if isinstance(payload, Mapping):
                rules = [SourceRule(str(key), str(value), "exact") for key, value in payload.items()]
            elif isinstance(payload, list):
                rules = [cls._record_to_rule(record) for record in payload]
            else:
                raise ValueError("source group JSON must be an object or a list of records")
        else:
            with source.open("r", encoding="utf-8-sig", newline="") as stream:
                rules = [cls._record_to_rule(record) for record in csv.DictReader(stream)]
        return cls(rules)

    @staticmethod
    def _record_to_rule(record: Mapping[str, Any]) -> SourceRule:
        pattern = record.get("image_key", record.get("pattern", record.get("source", "")))
        group = record.get("source_group", record.get("group", ""))
        match = str(record.get("match", "exact") or "exact").casefold()
        if not str(pattern).strip() or not str(group).strip():
            raise ValueError(f"source group record requires image_key/pattern and source_group: {record!r}")
        if match not in {"exact", "prefix"}:
            raise ValueError(f"source group match must be exact or prefix, got {match!r}")
        selected_value = record.get("probe_selected", True)
        if isinstance(selected_value, str):
            probe_selected = selected_value.strip().casefold() not in {
                "",
                "0",
                "false",
                "no",
                "off",
            }
        else:
            probe_selected = bool(selected_value)
        rank_value = record.get("probe_selection_rank")
        probe_selection_rank = (
            None if rank_value in (None, "") else int(rank_value)
        )
        return SourceRule(
            str(pattern),
            str(group),
            match,
            probe_selected,
            probe_selection_rank,
        )

    def resolve(self, image_key: str) -> str:
        normalized_key = _normalized(image_key)
        exact = [rule for rule in self.rules if rule.match == "exact" and _normalized(rule.pattern) == normalized_key]
        if exact:
            return exact[0].source_group
        prefixes = [
            rule
            for rule in self.rules
            if rule.match == "prefix" and normalized_key.startswith(_normalized(rule.pattern))
        ]
        if prefixes:
            return max(prefixes, key=lambda rule: len(_normalized(rule.pattern))).source_group
        return str(image_key)

    def manifest(self) -> dict[str, Any]:
        all_groups = list(dict.fromkeys(str(rule.source_group) for rule in self.rules))
        selected_groups = list(
            dict.fromkeys(
                str(rule.source_group)
                for rule in self.rules
                if rule.probe_selected
            )
        )
        total = len(all_groups)
        selected = len(selected_groups)
        return {
            "provided": bool(self.rules),
            "rule_count": len(self.rules),
            "fallback": "image_key_is_source_group",
            "source_group_count_total": total,
            "source_group_count_probed": selected,
            "source_group_coverage_complete": selected == total,
            "source_group_coverage_fraction": (
                float(selected / total) if total else 1.0
            ),
            "source_group_selection_policy": (
                "all_source_groups"
                if selected == total
                else "deterministic_evenly_spaced_source_groups_v1"
            ),
            "probed_source_groups": selected_groups,
            "rules": [rule.__dict__ for rule in self.rules],
        }


def captions_from_batches(batches: Iterable[Mapping[str, Any]]) -> list[str]:
    captions: list[str] = []
    for batch in batches:
        raw = batch.get("captions")
        if raw is None:
            continue
        if isinstance(raw, str):
            captions.append(raw)
        elif isinstance(raw, (list, tuple)):
            captions.extend(str(item) for item in raw)
    return captions
