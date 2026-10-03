from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.check_dq_profile_copy_drift import (
    CopyDriftError,
    normalized_sha256,
    validate_copy_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalized_hash(data: bytes) -> str:
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def test_repository_copied_sources_manifest_is_current() -> None:
    result = validate_copy_manifest(REPO_ROOT)
    assert result["status"] == "pass"
    assert len(result["files"]) == 2


@pytest.mark.parametrize("changed_file", ["train_network.py", "dq_profile/copied_train_network.py"])
def test_snapshot_verification_without_git_history_still_detects_drift(tmp_path, changed_file):
    # Isolate this fixture from any parent checkout's historical Git objects.
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    manifest = json.loads((REPO_ROOT / "dq_profile/copied_sources.json").read_text(encoding="utf-8"))
    for copied, record in manifest["files"].items():
        for name in (copied, record["source"]):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, target)
    manifest_path = tmp_path / "dq_profile/copied_sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_copy_manifest(tmp_path)
    assert result["verification_mode"] == "normalized_sha256"
    changed = tmp_path / changed_file
    changed.write_bytes(changed.read_bytes() + b"\n# unreviewed change\n")
    with pytest.raises(CopyDriftError, match="drifted"):
        validate_copy_manifest(tmp_path)


def test_legacy_manifest_keeps_git_provenance_checks(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    manifest = tmp_path / "copied_sources.json"
    manifest.write_text(json.dumps({
        "schema_version": "1.0.0", "source_commit": "0" * 40,
        "files": {"copied.py": {"source": "source.py"}},
    }), encoding="utf-8")
    with pytest.raises(CopyDriftError, match="git rev-parse"):
        validate_copy_manifest(tmp_path, manifest)


def test_copy_drift_check_is_line_ending_independent_and_detects_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    copied = tmp_path / "copied.py"
    source.write_bytes(b"value = 1\r\n")
    copied.write_bytes(b"value = 2\n")
    manifest = tmp_path / "copied_sources.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source_commit": "test-only",
                "files": {
                    "copied.py": {
                        "source": "source.py",
                        "source_normalized_sha256": _normalized_hash(b"value = 1\n"),
                        "copied_normalized_sha256": _normalized_hash(b"value = 2\r\n"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = validate_copy_manifest(tmp_path, manifest, verify_git=False)
    assert result["status"] == "pass"
    assert normalized_sha256(source) == _normalized_hash(b"value = 1\n")

    source.write_text("value = 3\n", encoding="utf-8")
    with pytest.raises(CopyDriftError, match="source drifted"):
        validate_copy_manifest(tmp_path, manifest, verify_git=False)
