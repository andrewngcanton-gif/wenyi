from __future__ import annotations

from pathlib import Path

from tests.helpers import minimal_manifest
from wenyi_eval.controls import run_negative_controls


def test_negative_controls_detect_corrupt_identity_receipts_and_unsafe_snapshots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root)

    controls = run_negative_controls(root, manifest)

    assert controls["passed"] is True
    by_name = {item["name"]: item for item in controls["controls"]}
    assert set(by_name) == {
        "unsafe_snapshot_body_target",
        "corrupted_code_identity",
        "duplicate_terminal_receipt",
        "missing_vs_empty_preserved",
    }
    assert all(item["passed"] for item in controls["controls"])
    assert by_name["unsafe_snapshot_body_target"]["detail"].startswith("UnsafeSnapshotError:")
    assert by_name["corrupted_code_identity"]["detail"].startswith("EvalError:")
    assert by_name["duplicate_terminal_receipt"]["detail"].startswith("OwnershipError:")
    assert "missing and empty remained distinct" in by_name["missing_vs_empty_preserved"]["detail"]
    assert (root / "_controls" / "controls.json").is_file()
