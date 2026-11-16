"""Read-side evidence for WP2 reflection artifacts.

Artifacts are diagnostic records, never a quality verdict. This module reads them
through public storage APIs, rebuilds the expected batch identity from the
current post-alignment text and the group configuration, and delegates the
strict comparison to the product's ``reflection_artifact_status``. Missing,
stale, invalid or ambiguous records stay in the denominator and are never
presented as clean or as current adoption.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

_ARTIFACT_KEY = re.compile(r"^reflection/chapter-(\d+)/batch-(\d+)-(\d+)\.json$")
_BATCH_EVENT_STAGES = ("written", "write_failed", "not_written")
_SEVERITY = {"ambiguous": 0, "invalid": 1, "stale": 2, "current": 3, "missing": 4}


def _policy_or_none(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    reflection = plan.get("reflection") or {}
    if not reflection.get("enabled"):
        return None
    return reflection


def expected_reflection_evidence(
    config_path: str,
    *,
    source_sha256: str,
    source_lang: str,
) -> dict[str, Any] | None:
    """Rebuild the policy and configured route evidence for a reflective group."""
    from .snapshot import expected_reflection_identity

    try:
        return expected_reflection_identity(
            config_path, source_sha256=source_sha256, source_lang=source_lang
        )
    except Exception:  # noqa: BLE001 - A configuration that cannot be resolved is reported.
        return None


def _empty_summary(
    *,
    enabled: bool,
    expected_segments: int,
    reason: str,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reflection = _policy_or_none(plan) or {}
    return {
        "status": "missing" if enabled else "not_enabled",
        "enabled": enabled,
        "reason": reason,
        "denominator": expected_segments,
        "clean_action": reflection.get("clean_action"),
        "min_severity": reflection.get("min_severity"),
        "route_separation": (reflection.get("route_separation") or {"status": "not_applicable"}),
        "configured_routes": reflection.get("configured_routes"),
        "expected_policy": None,
        "artifacts": {
            "total": 0,
            "current": 0,
            "stale": 0,
            "invalid": 0,
            "ambiguous": 0,
            "unreadable": 0,
            "write_failed_events": 0,
        },
        "batches": [],
        "segments": {
            "expected": expected_segments,
            "current": 0,
            "missing": expected_segments if enabled else 0,
            "stale": 0,
            "invalid": 0,
            "ambiguous": 0,
            "not_enabled": 0 if enabled else expected_segments,
        },
        "adoption_counts": {},
        "critique_counts": {},
        "failure_counts": {},
        "observed_routes": {},
        "observed_status": "unknown",
        "event_artifact_status": {stage: 0 for stage in _BATCH_EVENT_STAGES},
    }


def _expected_window(
    chapter, start: int, count: int
) -> tuple[list[Any], list[int], list[str]] | None:
    """Return the contiguous current batch window or None when the key is outside it."""
    segments = chapter.text_segments
    if count <= 0 or start < 0 or start + count > len(segments):
        return None
    window = list(segments[start : start + count])
    return window, [segment.index for segment in window], [segment.source for segment in window]


def _record_adoption(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "critique": record.get("critique"),
        "adoption": record.get("adoption"),
        "adoption_verified": record.get("adoption_verified"),
        "failure": record.get("failure"),
        "critique_attempts": record.get("critique_attempts"),
        "revision_attempts": record.get("revision_attempts"),
    }


def _observed_routes(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        routes = record.get("routes")
        calls = routes.get("observed") if isinstance(routes, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            operation = str(call.get("operation") or "unknown")
            model = str(call.get("model") or "unknown")
            counts.setdefault(operation, {})
            counts[operation][model] = counts[operation].get(model, 0) + 1
    return counts


def extract_reflection(
    *,
    group: str,
    plan: dict[str, Any] | None,
    store,
    run_dir: Path,
    source_sha256: str,
    source_language: str,
    target_language: str,
    chapters: dict[int, Any],
    expected: dict[str, Any],
    events: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    """Return the group reflection summary and a per-segment evidence map."""
    expected_segments = sum(len(chapter.get("segments", [])) for chapter in expected["chapters"])
    reflection = _policy_or_none(plan)
    if reflection is None:
        summary = _empty_summary(
            enabled=False,
            expected_segments=expected_segments,
            reason="reflection is not enabled for this group",
            plan=plan,
        )
        return summary, {}

    config_path = str(plan.get("config") or "")
    evidence = (
        expected_reflection_evidence(
            config_path, source_sha256=source_sha256, source_lang=source_language
        )
        if config_path
        else None
    )
    summary = _empty_summary(
        enabled=True,
        expected_segments=expected_segments,
        reason="no reflection artifact matched the current post-alignment text",
        plan=plan,
    )
    summary["expected_policy"] = (evidence or {}).get("policy")
    if evidence and evidence.get("configured_routes"):
        summary["configured_routes"] = evidence["configured_routes"]

    for event in events:
        if event.get("event") != "batch_translated":
            continue
        payload = event.get("reflection")
        if not isinstance(payload, dict):
            continue
        stage = payload.get("artifact_status")
        if stage in summary["event_artifact_status"]:
            summary["event_artifact_status"][stage] += 1
    summary["artifacts"]["write_failed_events"] = summary["event_artifact_status"]["write_failed"]

    keys = [
        key
        for key in store.list_artifacts("reflection/")
        if key.startswith("reflection/") and key.endswith(".json")
    ]
    summary["artifacts"]["total"] = len(keys)
    if evidence is None:
        summary["status"] = "invalid"
        summary["reason"] = (
            "reflection policy and configured route evidence could not be reconstructed; "
            "no artifact is accepted as current"
        )
        summary["segments"]["invalid"] = expected_segments
        summary["segments"]["missing"] = 0
        return summary, _all_segments(expected, "invalid", summary)
    if not keys:
        _finish_segments(summary)
        return summary, _all_segments(expected, "missing", summary)

    from wenyi_core.pipeline.translate.reflection_records import reflection_artifact_status

    parsed: list[dict[str, Any]] = []
    invalid_batches: list[dict[str, Any]] = []
    for key in keys:
        match = _ARTIFACT_KEY.match(key)
        if not match:
            invalid_batches.append(
                {
                    "key": key,
                    "chapter": None,
                    "start_index": None,
                    "count": None,
                    "status": "invalid",
                    "reason": "artifact key is not a reflection batch key",
                }
            )
            continue
        chapter_index, start, count = (int(value) for value in match.groups())
        parsed.append(
            {
                "key": key,
                "chapter": chapter_index,
                "start_index": start,
                "count": count,
            }
        )

    prepared_chapters = {
        chapter["index"]: [
            segment for segment in chapter.get("segments", []) if segment["source"].strip()
        ]
        for chapter in expected["chapters"]
    }
    # A current-state window must also match the immutable prepared inventory.
    # Matching a self-consistent changed state and its artifact is insufficient.
    windows: dict[int, list[dict[str, Any]]] = {}
    for item in parsed:
        chapter = chapters.get(item["chapter"])
        if chapter is None:
            item["status"] = "stale"
            item["reason"] = "chapter is not present in the extracted state"
            continue
        window = _expected_window(chapter, item["start_index"], item["count"])
        if window is None:
            item["status"] = "invalid"
            item["reason"] = "batch window is outside the current chapter"
            continue
        item["window"] = window
        prepared = prepared_chapters.get(item["chapter"], [])[
            item["start_index"] : item["start_index"] + item["count"]
        ]
        actual_identity = [(segment.index, segment.source, segment.kind) for segment in window[0]]
        prepared_identity = [
            (segment["index"], segment["source"], segment["kind"]) for segment in prepared
        ]
        if len(prepared) != item["count"] or actual_identity != prepared_identity:
            item["status"] = "stale"
            item["reason"] = "batch window differs from the prepared segment inventory"
            continue
        windows.setdefault(item["chapter"], []).append(item)

    for items in windows.values():
        ambiguous: set[str] = set()
        for position, item in enumerate(items):
            indices = tuple(item["window"][1])
            for other in items[position + 1 :]:
                other_indices = tuple(other["window"][1])
                if indices == other_indices:
                    ambiguous.update({item["key"], other["key"]})
                elif set(indices) & set(other_indices):
                    ambiguous.update({item["key"], other["key"]})
        for item in items:
            if item["key"] in ambiguous:
                item["status"] = "ambiguous"
                item["reason"] = "duplicate or overlapping artifacts describe the batch window"

    batches: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for item in parsed:
        record: dict[str, Any] | None = None
        if item.get("status") in {"ambiguous", "stale", "invalid"}:
            item.setdefault("reason", "artifact is not current")
        else:
            try:
                record = store.read_artifact(item["key"])
            except Exception as error:  # noqa: BLE001 - Corrupt diagnostics stay in the denominator.
                item["status"] = "invalid"
                item["reason"] = f"artifact is unreadable: {type(error).__name__}"
            else:
                if record is None:
                    item["status"] = "invalid"
                    item["reason"] = "artifact disappeared while reading"
        if item.get("status") not in {"ambiguous", "stale", "invalid"} and isinstance(record, dict):
            chapter = chapters[item["chapter"]]
            window, indices, sources = item["window"]
            expected_identity = {
                "chapter": item["chapter"],
                "start_index": item["start_index"],
                "segment_indices": indices,
                "source_sha256": source_sha256,
                "source_language": source_language,
                "target_language": target_language,
                "batch_source_sha256": _content_hash(sources),
                "policy": (evidence or {}).get("policy") or {},
                "configured_route_fingerprints": (evidence or {}).get(
                    "configured_route_fingerprints"
                ),
            }
            status = reflection_artifact_status(record, expected_identity, window)
            item["status"] = "invalid" if status.get("reason") == "schema" else status["status"]
            item["reason"] = status.get("reason")
            if item["status"] == "current":
                records.append(record)
                item["evidence"] = _record_adoption(record)
                item["observed_status"] = (record.get("routes") or {}).get(
                    "observed_status", "unknown"
                )
                batches.append(
                    {
                        "key": item["key"],
                        "chapter": item["chapter"],
                        "start_index": item["start_index"],
                        "count": item["count"],
                        "status": "current",
                        "reason": None,
                        **_record_adoption(record),
                        "observed_status": item["observed_status"],
                        "observed_calls": len((record.get("routes") or {}).get("observed") or []),
                    }
                )
                continue
        batches.append(
            {
                "key": item["key"],
                "chapter": item["chapter"],
                "start_index": item["start_index"],
                "count": item["count"],
                "status": item.get("status", "invalid"),
                "reason": item.get("reason"),
                "critique": None,
                "adoption": None,
                "adoption_verified": None,
                "failure": None,
            }
        )

    for item in invalid_batches:
        batches.append(
            {
                **item,
                "critique": None,
                "adoption": None,
                "adoption_verified": None,
                "failure": None,
            }
        )

    for item in [*parsed, *invalid_batches]:
        status = item.get("status", "invalid")
        key = (
            "ambiguous"
            if status == "ambiguous"
            else "invalid"
            if status in {"invalid", "unreadable"}
            else "stale"
            if status == "stale"
            else "current"
        )
        summary["artifacts"][key] += 1

    summary["batches"] = sorted(
        batches,
        key=lambda row: (
            row["chapter"] if isinstance(row["chapter"], int) else -1,
            row["start_index"] if isinstance(row["start_index"], int) else -1,
            row["key"],
        ),
    )
    summary["adoption_counts"] = _counts(records, "adoption")
    summary["critique_counts"] = _counts(records, "critique")
    summary["failure_counts"] = _counts(records, "failure")
    summary["observed_routes"] = _observed_routes(records)
    observed_states = {
        (record.get("routes") or {}).get("observed_status", "unknown") for record in records
    }
    summary["observed_status"] = (
        "recorded" if observed_states == {"recorded"} and records else "unknown"
    )

    segment_map = _assign_segments(expected, parsed)
    for item in parsed:
        if item.get("status") == "current":
            for index in item["window"][1]:
                segment_evidence = segment_map.get((item["chapter"], index))
                if segment_evidence is None:
                    continue
                segment_evidence.update(
                    {
                        "artifact_key": item["key"],
                        "critique": item["evidence"]["critique"],
                        "adoption": item["evidence"]["adoption"],
                        "failure": item["evidence"]["failure"],
                        "adoption_verified": item["evidence"]["adoption_verified"],
                    }
                )
    _finish_segments(summary, segment_map)
    summary["status"] = _group_status(summary)
    if summary["status"] != "current":
        summary["reason"] = _group_reason(summary)
    else:
        summary["reason"] = None
    return summary, segment_map


def _content_hash(value: Any) -> str:
    from wenyi_core.i18n.policy.models import content_hash

    return content_hash(value)


def _counts(records: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _all_segments(expected: dict[str, Any], status: str, summary: dict[str, Any]) -> dict:
    segment_map: dict[tuple[int, int], dict[str, Any]] = {}
    for chapter in expected["chapters"]:
        for segment in chapter.get("segments", []):
            segment_map[(chapter["index"], segment["index"])] = {
                "status": status,
                "reason": summary.get("reason"),
                "artifact_key": None,
                "critique": None,
                "adoption": None,
                "failure": None,
                "adoption_verified": None,
            }
    return segment_map


def _assign_segments(
    expected: dict[str, Any],
    parsed: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    segment_map: dict[tuple[int, int], dict[str, Any]] = {}
    for chapter in expected["chapters"]:
        for segment in chapter.get("segments", []):
            segment_map[(chapter["index"], segment["index"])] = {
                "status": "missing",
                "reason": "no reflection artifact applies to this segment",
                "artifact_key": None,
                "critique": None,
                "adoption": None,
                "failure": None,
                "adoption_verified": None,
            }
    for item in parsed:
        if "window" not in item:
            continue
        status = item.get("status", "invalid")
        if status not in _SEVERITY:
            status = "invalid"
        for index in item["window"][1]:
            current = segment_map.get((item["chapter"], index))
            if current is None:
                continue
            if _SEVERITY[status] < _SEVERITY[current["status"]]:
                current["status"] = status
                current["reason"] = item.get("reason")
                current["artifact_key"] = item["key"]
    return segment_map


def _finish_segments(summary: dict[str, Any], segment_map: dict | None = None) -> None:
    counts = {
        "expected": summary["denominator"],
        "current": 0,
        "missing": 0,
        "stale": 0,
        "invalid": 0,
        "ambiguous": 0,
        "not_enabled": 0,
    }
    if segment_map is not None:
        for value in segment_map.values():
            status = value.get("status", "missing")
            if status not in counts:
                status = "invalid"
            counts[status] += 1
    elif not summary["enabled"]:
        counts["not_enabled"] = summary["denominator"]
    else:
        counts["missing"] = summary["denominator"]
    summary["segments"] = counts


def _group_status(summary: dict[str, Any]) -> str:
    segments = summary["segments"]
    if segments["expected"] == 0:
        return "current" if summary["artifacts"]["current"] else "missing"
    if segments["current"] == segments["expected"]:
        return "current"
    for status in ("ambiguous", "invalid", "stale"):
        if segments[status]:
            return status
    if segments["missing"] == segments["expected"]:
        return "missing"
    return "partial"


def _group_reason(summary: dict[str, Any]) -> str:
    segments = summary["segments"]
    parts = [
        f"{status}={segments[status]}"
        for status in ("missing", "stale", "invalid", "ambiguous")
        if segments[status]
    ]
    if summary["artifacts"]["write_failed_events"]:
        parts.append(f"write_failed_events={summary['artifacts']['write_failed_events']}")
    if summary["artifacts"]["current"] == 0:
        parts.append("no current artifact")
    return "reflection coverage: " + ", ".join(parts or ["no current artifact"])


__all__ = ["extract_reflection", "expected_reflection_evidence"]
