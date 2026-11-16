"""Read translated state through FileStorage and preserve missing or stale identities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import (
    CorruptIdentityError,
    PathGuard,
    hash_tree,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .manifest import SUPPORTED_GROUPS
from .run_variants import find_run_dir

EXTRACTION_PROTOCOL = 1
_FAILURE_EVENTS = re.compile(r"(?i)(fail|retry|error|pause|recover|timeout|deferred)")


def segment_id(source_sha256: str, chapter: int, segment: int) -> str:
    return f"{source_sha256[:16]}:{chapter}:{segment}"


def expected_inventory(snapshot_state: Path) -> dict[str, Any]:
    """Read the immutable prepared chapters used to define required segment keys."""
    from wenyi_core.storage.file import FileStorage

    before = hash_tree(snapshot_state)
    store = FileStorage(str(snapshot_state), create=False)
    try:
        manifest = store.load_manifest()
        chapters = []
        for entry in manifest["chapters"]:
            chapter = store.load_chapter(entry["index"])
            chapters.append(
                {
                    "index": chapter.index,
                    "title": chapter.title,
                    "status": entry.get("status"),
                    "segments": [
                        {
                            "index": segment.index,
                            "source": segment.source,
                            "kind": segment.kind,
                            "id": segment_id(
                                manifest["source_sha256"], chapter.index, segment.index
                            ),
                        }
                        for segment in chapter.segments
                    ],
                }
            )
    finally:
        store.close()
    after = hash_tree(snapshot_state)
    if before != after:
        raise CorruptIdentityError("Reading the preparation snapshot changed it")
    return {
        "source_sha256": manifest["source_sha256"],
        "title": manifest.get("title"),
        "source_lang": manifest.get("source_lang"),
        "target_lang": manifest.get("target_lang"),
        "chapters": chapters,
        "tree_sha256": before,
    }


def _review_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "missing", "issues": None, "reviewed_segments": None}
    if not isinstance(value, dict):
        return {"status": "invalid", "issues": None, "reviewed_segments": None}
    result = value.get("result") if isinstance(value.get("result"), dict) else value
    issues = result.get("issues")
    reviewed = result.get("reviewed_segments")
    return {
        "status": "available",
        "issues": len(issues) if isinstance(issues, list) else None,
        "reviewed_segments": reviewed if isinstance(reviewed, int) else None,
        "complete": result.get("complete") if isinstance(result.get("complete"), bool) else None,
    }


def _reflection_status(run_dir: Path | None) -> dict[str, Any]:
    candidates = (
        [
            path
            for name in ("reflections", "reflection")
            for path in [run_dir / name]
            if path.exists()
        ]
        if run_dir is not None
        else []
    )
    return {
        "status": "unsupported",
        "reason": (
            "WP2 reflection artifacts and the reflection evaluator are not available in this "
            "code source; reflection quality is not reported as clean."
        ),
        "paths_present": [path.name for path in candidates],
    }


def _event_counts(review_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in review_events:
        name = row.get("event")
        if isinstance(name, str) and _FAILURE_EVENTS.search(name):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _empty_counts(expected: dict[str, Any]) -> dict[str, int]:
    return {
        "expected": sum(
            len(chapter.get("segments", [])) for chapter in expected.get("chapters", [])
        ),
        "translated": 0,
        "empty": 0,
        "missing": 0,
        "stale": 0,
        "error": 0,
    }


def _expected_rows(expected: dict[str, Any], *, status: str, reason: str) -> list[dict[str, Any]]:
    """Preserve one explicit record per expected segment for a whole-group failure."""
    chapters: list[dict[str, Any]] = []
    for chapter in expected.get("chapters", []):
        chapters.append(
            {
                "index": chapter["index"],
                "title": chapter.get("title"),
                "status": "unknown",
                "chapter_file": "unavailable",
                "load_error": reason,
                "reason": reason,
                "segments": [
                    {
                        "id": segment["id"],
                        "segment": segment["index"],
                        "kind": segment["kind"],
                        "source": segment["source"],
                        "target": None,
                        "target_before_polish": None,
                        "status": status,
                        "reason": reason,
                    }
                    for segment in chapter.get("segments", [])
                ],
            }
        )
    return chapters


def _group_failure(
    *,
    group: str,
    expected: dict[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    counts = _empty_counts(expected)
    counts[status if status in counts else "error"] = counts["expected"]
    return {
        "status": status,
        "group": group,
        "run_dir": None,
        "source_sha256": expected.get("source_sha256"),
        "reason": reason,
        "chapters": _expected_rows(expected, status=status, reason=reason),
        "coverage": counts,
        "usage": None,
        "review": _review_summary(None),
        "events": {"total": 0, "failure_or_retry_counts": {}},
        "reflection": _reflection_status(None),
        "read_only_verified": True,
        "state_tree_sha256": None,
        "extracted_at": utc_now(),
    }


def extract_group(
    *,
    group: str,
    state_dir: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Extract one variant without writing to its state.

    Missing, stale and unreadable state keeps every expected segment in the
    coverage denominator. Stale text is never reported as a current target.
    """
    from wenyi_core.storage.file import FileStorage

    state_dir = Path(state_dir)
    if not state_dir.is_dir():
        return _group_failure(
            group=group,
            expected=expected,
            status="missing",
            reason="group state directory does not exist",
        )
    before = hash_tree(state_dir)
    try:
        run_dir = find_run_dir(state_dir)
    except Exception as error:
        return _group_failure(
            group=group,
            expected=expected,
            status="missing",
            reason=f"no prepared run state: {type(error).__name__}: {error}",
        )
    try:
        store = FileStorage(str(run_dir), create=False)
        try:
            manifest = store.load_manifest()
        finally:
            store.close()
    except Exception as error:
        return _group_failure(
            group=group,
            expected=expected,
            status="error",
            reason=f"run manifest is unreadable: {type(error).__name__}: {error}",
        )
    if manifest.get("source_sha256") != expected["source_sha256"]:
        result = _group_failure(
            group=group,
            expected=expected,
            status="stale",
            reason="source identity differs from the prepared snapshot",
        )
        result["run_dir"] = str(run_dir)
        result["source_sha256"] = manifest.get("source_sha256")
        result["state_tree_sha256"] = before
        return result
    store = FileStorage(str(run_dir), create=False)
    try:
        entries = {entry["index"]: entry for entry in manifest.get("chapters", [])}
        usage = store.load_usage()
        review = store.load_latest_review_result()
        events = store.list_events(limit=0)
        chapters: list[dict[str, Any]] = []
        counts = {
            "expected": 0,
            "translated": 0,
            "empty": 0,
            "missing": 0,
            "stale": 0,
            "error": 0,
        }
        for expected_chapter in expected["chapters"]:
            count = expected_chapter["index"]
            expected_segments = expected_chapter["segments"]
            counts["expected"] += len(expected_segments)
            entry = entries.get(count)
            chapter_exists = entry is not None
            loaded = None
            load_error = None
            if chapter_exists:
                try:
                    loaded = store.load_chapter(count)
                except (OSError, ValueError, KeyError) as error:
                    load_error = str(error)
            actual = {segment.index: segment for segment in loaded.segments} if loaded else {}
            rows = []
            expected_indices = {item["index"] for item in expected_segments}
            for expected_segment in expected_segments:
                index = expected_segment["index"]
                segment = actual.get(index)
                reason = None
                if load_error:
                    status = "error"
                    target = None
                    before_polish = None
                    reason = f"chapter file is unreadable: {load_error}"
                elif segment is None:
                    status = "missing"
                    target = None
                    before_polish = None
                elif segment.source != expected_segment["source"]:
                    status = "stale"
                    target = None
                    before_polish = None
                    reason = "segment source differs from the prepared snapshot"
                elif segment.target is None:
                    status = "missing"
                    target = None
                    before_polish = segment.target_before_polish
                elif segment.target == "":
                    status = "empty"
                    target = ""
                    before_polish = segment.target_before_polish
                else:
                    status = "translated"
                    target = segment.target
                    before_polish = segment.target_before_polish
                counts[status] += 1
                rows.append(
                    {
                        "id": expected_segment["id"],
                        "segment": index,
                        "kind": expected_segment["kind"],
                        "source": expected_segment["source"],
                        "target": target,
                        "target_before_polish": before_polish,
                        "status": status,
                        "reason": reason,
                    }
                )
            if not load_error:
                for index in sorted(set(actual) - expected_indices):
                    counts["stale"] += 1
                    segment = actual[index]
                    rows.append(
                        {
                            "id": segment_id(expected["source_sha256"], count, index),
                            "segment": index,
                            "kind": segment.kind,
                            "source": segment.source,
                            "target": None,
                            "target_before_polish": None,
                            "status": "stale",
                            "reason": "segment is not present in the prepared snapshot",
                        }
                    )
            rows.sort(key=lambda row: row["segment"])
            chapters.append(
                {
                    "index": count,
                    "title": loaded.title if loaded else expected_chapter["title"],
                    "status": entry.get("status") if entry else "missing",
                    "chapter_file": "loaded"
                    if loaded
                    else ("missing" if not load_error else "invalid"),
                    "load_error": load_error,
                    "reason": f"chapter file is unreadable: {load_error}" if load_error else None,
                    "segments": rows,
                }
            )
        review_summary = _review_summary(review)
        event_counts = _event_counts(events)
    finally:
        store.close()
    after = hash_tree(state_dir)
    if before != after:
        raise CorruptIdentityError(f"Extraction modified group {group} state")
    status = "ok"
    if counts["error"]:
        status = "error"
    elif counts["missing"] and not counts["translated"] and not counts["empty"]:
        status = "missing"
    elif counts["stale"] and not counts["translated"] and not counts["empty"]:
        status = "stale"
    return {
        "status": status,
        "group": group,
        "run_dir": str(run_dir),
        "source_sha256": manifest.get("source_sha256"),
        "chapters": chapters,
        "coverage": counts,
        "usage": usage,
        "review": review_summary,
        "events": {
            "total": len(events),
            "failure_or_retry_counts": event_counts,
        },
        "reflection": _reflection_status(run_dir),
        "read_only_verified": True,
        "state_tree_sha256": before,
        "extracted_at": utc_now(),
    }


def extract_all(
    *,
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    guard = PathGuard(root)
    snapshot_state = guard.child("_prepared", "state")
    expected = expected_inventory(snapshot_state)
    groups: dict[str, Any] = {}
    for group in SUPPORTED_GROUPS:
        plan = manifest["groups"][group]
        try:
            groups[group] = extract_group(
                group=group,
                state_dir=Path(plan["state_dir"]),
                expected=expected,
            )
        except Exception as error:
            # Unexpected harness faults still keep the prepared denominator so a
            # lost group can never improve a coverage ratio.
            groups[group] = _group_failure(
                group=group,
                expected=expected,
                status="error",
                reason=f"{type(error).__name__}: {error}",
            )
    result = {
        "protocol_version": EXTRACTION_PROTOCOL,
        "eval_id": manifest["eval_id"],
        "book": manifest["book"],
        "extracted_at": utc_now(),
        "expected": expected,
        "groups": groups,
    }
    write_json_atomic(root / "extraction.json", result)
    return result


def load_extraction(root: Path) -> dict[str, Any]:
    return read_json(root / "extraction.json")


def source_state_hashes(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Hash each source file in the fixture directory without writing to state."""
    guard = PathGuard(root)
    source = guard.ensure(manifest["source"]["path"], label="source")
    fixture_dir = source.parent
    return {
        path.name: sha256_file(path) for path in sorted(fixture_dir.iterdir()) if path.is_file()
    }
