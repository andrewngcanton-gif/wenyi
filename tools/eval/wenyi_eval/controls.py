"""Negative controls proving the harness rejects unsafe or ambiguous evidence."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from .common import EvalError, write_json_atomic
from .extract import extract_group
from .manifest import validate_manifest
from .run_variants import TerminalStore
from .snapshot import create_snapshot

_SOURCE_SHA = "a" * 64


def _record(name: str, expected: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "expected": expected, "passed": passed, "detail": detail}


def _control_root(root: Path, name: str) -> Path:
    path = root / "_controls" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_control(name: str, expected: str, action: Callable[[], None]) -> dict[str, Any]:
    try:
        action()
    except Exception as error:
        passed = any(cls.__name__ == expected for cls in type(error).__mro__)
        return _record(name, expected, passed, f"{type(error).__name__}: {error}")
    return _record(name, expected, False, "control did not raise")


def _run_success_control(name: str, action: Callable[[], None]) -> dict[str, Any]:
    try:
        action()
    except Exception as error:
        return _record(name, "preserved statuses", False, f"{type(error).__name__}: {error}")
    return _record(name, "preserved statuses", True, "missing and empty remained distinct")


def unsafe_snapshot_control(root: Path) -> dict[str, Any]:
    from wenyi_core.models.document import Chapter, Segment
    from wenyi_core.storage.file import FileStorage

    def action() -> None:
        control = _control_root(root, "unsafe-snapshot")
        run = control / "run"
        store = FileStorage(str(run), create=True)
        try:
            store.save_chapter(
                Chapter(
                    index=0,
                    title="Leaked",
                    segments=[Segment(index=0, source="source", target="body target")],
                )
            )
            store.save_manifest(
                {
                    "title": "Leaked",
                    "fmt": "text",
                    "source_sha256": _SOURCE_SHA,
                    "source_lang": "en",
                    "target_lang": "zh",
                    "meta": {},
                    "initialized": True,
                    "chapters": [{"index": 0, "title": "Leaked", "status": "pending"}],
                }
            )
        finally:
            store.close()
        create_snapshot(
            root=control,
            run_dir=run,
            expected_source_sha256=_SOURCE_SHA,
            expected_target_lang="zh",
        )

    return _run_control("unsafe_snapshot_body_target", "UnsafeSnapshotError", action)


def corrupted_identity_control(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    def action() -> None:
        corrupted = copy.deepcopy(manifest)
        corrupted["groups"]["A"]["expected_head_sha"] = "0" * 40
        validate_manifest(corrupted, repo_root=Path(manifest["repo"]["root"]))

    return _run_control("corrupted_code_identity", "EvalError", action)


def duplicate_receipt_control(root: Path) -> dict[str, Any]:
    def action() -> None:
        control = _control_root(root, "duplicate-receipt")
        store = TerminalStore(control)
        row = {
            "protocol_version": 1,
            "group": "control",
            "stage": "duplicate",
            "status": "succeeded",
        }
        store.write(row)
        store.write(row)

    return _run_control("duplicate_terminal_receipt", "OwnershipError", action)


def missing_empty_control(root: Path) -> dict[str, Any]:
    from wenyi_core.models.document import Chapter, Segment
    from wenyi_core.storage.file import FileStorage

    def action() -> None:
        control = _control_root(root, "missing-vs-empty")
        state = control / "state"
        run = state / "probe" / "targets" / "zh"
        store = FileStorage(str(run), create=True)
        try:
            store.save_chapter(
                Chapter(
                    index=0,
                    title="Probe",
                    segments=[
                        Segment(index=0, source="missing source", target=None),
                        Segment(index=1, source="empty source", target=""),
                    ],
                )
            )
            store.save_manifest(
                {
                    "title": "Probe",
                    "fmt": "text",
                    "source_sha256": _SOURCE_SHA,
                    "source_lang": "en",
                    "target_lang": "zh",
                    "meta": {},
                    "chapters": [{"index": 0, "title": "Probe", "status": "done"}],
                }
            )
        finally:
            store.close()
        expected = {
            "source_sha256": _SOURCE_SHA,
            "chapters": [
                {
                    "index": 0,
                    "title": "Probe",
                    "segments": [
                        {"index": 0, "source": "missing source", "kind": "text", "id": "m0"},
                        {"index": 1, "source": "empty source", "kind": "text", "id": "m1"},
                    ],
                }
            ],
        }
        result = extract_group(group="control", state_dir=state, expected=expected)
        statuses = [row["status"] for row in result["chapters"][0]["segments"]]
        if statuses != ["missing", "empty"]:
            raise EvalError(
                f"missing/empty statuses were {statuses}, expected ['missing', 'empty']"
            )

    return _run_success_control("missing_vs_empty_preserved", action)


def run_negative_controls(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    results = [
        unsafe_snapshot_control(root),
        corrupted_identity_control(root, manifest),
        duplicate_receipt_control(root),
        missing_empty_control(root),
    ]
    payload = {
        "protocol_version": 1,
        "controls": results,
        "passed": all(item["passed"] for item in results),
    }
    write_json_atomic(root / "_controls" / "controls.json", payload)
    return payload
