from __future__ import annotations

import json

import pytest

from dq_profile.geometry import SourceGroupMap, SourceRule


@pytest.mark.parametrize("match", ["exact", "prefix"])
@pytest.mark.parametrize("repeated_side", ["key", "rule", "both"])
def test_source_group_map_matches_repeated_windows_separators(tmp_path, match, repeated_side):
    directory = "D:/train_data/日本語/source/"
    pattern = directory + "image.png" if match == "exact" else directory
    image_key = directory + "image.png"
    if repeated_side in {"rule", "both"}:
        pattern = pattern.replace("/", "\\\\")
    if repeated_side in {"key", "both"}:
        image_key = image_key.replace("/", "\\\\")
    source = tmp_path / "sources.json"
    source.write_text(
        json.dumps([{"pattern": pattern, "source_group": "source", "match": match}]),
        encoding="utf-8",
    )

    mapping = SourceGroupMap.load(source)

    assert mapping.resolve(image_key) == "source"
    # Keep raw provenance and unmatched keys unchanged.
    assert mapping.manifest()["rules"][0]["pattern"] == pattern
    unmatched = r"D:\\train_data\日本語\other\image.png"
    assert mapping.resolve(unmatched) == unmatched


def test_source_group_map_preserves_directory_boundaries_and_precedence():
    mapping = SourceGroupMap([
        SourceRule("D:/images/", "all", "prefix"),
        SourceRule(r"D:\\images\\a\\", "a", "prefix"),
        SourceRule("D:/images/a/nested/", "nested", "prefix"),
        SourceRule("D:/images/a/nested/exact.png", "exact"),
    ])

    assert mapping.resolve(r"D:\\images\a\image.png") == "a"
    assert mapping.resolve(r"D:\\images\ab\image.png") == "all"
    assert mapping.resolve(r"D:\\images\a\nested\image.png") == "nested"
    assert mapping.resolve(r"D:\\images\a\nested\exact.png") == "exact"


@pytest.mark.parametrize("match", ["exact", "prefix"])
def test_source_group_map_preserves_unc_root(match):
    directory = "//server/share/images/"
    pattern = directory + "image.png" if match == "exact" else directory
    mapping = SourceGroupMap([SourceRule(pattern, "remote", match)])

    assert mapping.resolve(r"\\server\share\\images\\image.png") == "remote"
    local_key = "/server/share/images/image.png"
    assert mapping.resolve(local_key) == local_key
