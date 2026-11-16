"""Diagnostic reflection records: construction, keys and read-side validation.

These records are evidence about what a batch did, never a gate: they do not participate in
product skipping, resume or export. A missing or stale record is reported as such and never
as a clean batch. Summaries are hashes with lengths; only bounded issue evidence carries text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...i18n.policy.models import content_hash
from ...models.document import Segment
from .translation_batch import ReflectionOutcome

SCHEMA_VERSION = 1
CODE_IDENTITY = "reflective-polish/1"
TARGET_STAGE = "post_alignment"
TARGET_BEFORE_POLISH_STAGE = "post_alignment_before_polish"
EXECUTOR_STAGE = "executor_output"
EXECUTOR_BEFORE_POLISH_STAGE = "executor_before_polish"


def reflection_artifact_key(chapter: int, start_index: int, count: int) -> str:
    """Keep the diagnostic location readable without creating stage checkpoints."""
    return f"reflection/chapter-{chapter:04d}/batch-{start_index:06d}-{count:04d}.json"


def _text_digest(value: str | None) -> dict[str, Any]:
    """Distinguish an absent target from an intentionally empty translation."""
    if value is None:
        return {"state": "none", "sha256": None, "length": 0}
    if value == "":
        return {"state": "empty", "sha256": None, "length": 0}
    return {"state": "text", "sha256": content_hash(value), "length": len(value)}


def _segment_digests(
    values: Sequence[str | None], segment_indices: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {"segment_index": index, **_text_digest(value)}
        for index, value in zip(segment_indices, values)
    ]


def _issue_records(issues: Sequence[Any], segment_indices: Sequence[int]) -> list[dict[str, Any]]:
    records = []
    for issue in issues:
        record = dict(issue.to_record())
        position = issue.index
        record["segment_index"] = (
            segment_indices[position] if 0 <= position < len(segment_indices) else None
        )
        records.append(record)
    return records


def build_reflection_record(
    *,
    chapter: int,
    start_index: int,
    segment_indices: Sequence[int],
    sources: Sequence[str],
    source_sha256: str,
    source_language: str,
    target_language: str,
    outcome: ReflectionOutcome,
    raw_drafts: Sequence[str] | None,
    positions: Sequence[int] | None,
    executor_targets: Sequence[str | None],
    executor_before_polish: Sequence[str | None],
    formal_segments: Sequence[Segment],
    policy: dict[str, Any],
    configured_routes: dict[str, Any],
    observed_calls: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the bounded diagnostic record for one saved batch."""
    aligned = (
        len(segment_indices)
        == len(sources)
        == len(executor_targets)
        == len(executor_before_polish)
        == len(formal_segments)
    )
    if not aligned or tuple(segment.index for segment in formal_segments) != tuple(segment_indices):
        raise ValueError("Reflection record inputs are not aligned to the batch")
    return {
        "schema": SCHEMA_VERSION,
        "code_identity": CODE_IDENTITY,
        "chapter": chapter,
        "start_index": start_index,
        "segment_indices": list(segment_indices),
        "source_sha256": source_sha256,
        "source_language": source_language,
        "target_language": target_language,
        "batch_source_sha256": content_hash(list(sources)),
        "batch_source_chars": sum(len(source) for source in sources),
        "critique": outcome.critique,
        "adoption": outcome.adoption,
        "adoption_verified": outcome.adoption_verified,
        "failure": outcome.failure,
        "positions": list(positions) if positions is not None else None,
        "critique_attempts": outcome.critique_attempts,
        "revision_attempts": outcome.revision_attempts,
        "ordinary_polish_attempted": outcome.ordinary_polish_attempted,
        "standalone_polish_attempted": outcome.standalone_polish_attempted,
        "issues": _issue_records(outcome.issues, segment_indices),
        "gated": _issue_records(outcome.gated, segment_indices),
        "drafts": {
            "raw_sha256": (
                [content_hash(draft) for draft in raw_drafts] if raw_drafts is not None else None
            ),
            "raw_lengths": [len(draft) for draft in raw_drafts] if raw_drafts else None,
            "formal": _segment_digests(
                [segment.target for segment in formal_segments], segment_indices
            ),
        },
        "targets": {
            "stage": TARGET_STAGE,
            "segments": _segment_digests(
                [segment.target for segment in formal_segments], segment_indices
            ),
        },
        "targets_before_polish": {
            "stage": TARGET_BEFORE_POLISH_STAGE,
            "segments": _segment_digests(
                [segment.target_before_polish for segment in formal_segments], segment_indices
            ),
        },
        "executor": {
            "targets": {
                "stage": EXECUTOR_STAGE,
                "segments": _segment_digests(executor_targets, segment_indices),
            },
            "before_polish": {
                "stage": EXECUTOR_BEFORE_POLISH_STAGE,
                "segments": _segment_digests(executor_before_polish, segment_indices),
            },
        },
        "policy": dict(policy),
        "routes": {
            "configured": dict(configured_routes),
            "observed": [dict(call) for call in observed_calls],
            "observed_status": "recorded" if observed_calls else "unknown",
        },
    }


def _identity_mismatch(record: dict[str, Any], expected: dict[str, Any]) -> str | None:
    for key in (
        "chapter",
        "start_index",
        "source_sha256",
        "source_language",
        "target_language",
        "batch_source_sha256",
    ):
        if key not in record or key not in expected or record[key] != expected[key]:
            return f"identity:{key}"
        if key in {"chapter", "start_index"}:
            if type(record[key]) is not int or type(expected[key]) is not int or record[key] < 0:
                return f"identity:{key}"
        elif not isinstance(record[key], str) or not record[key].strip():
            return f"identity:{key}"
    stored_indices = record.get("segment_indices")
    expected_indices = expected.get("segment_indices")
    if not isinstance(stored_indices, list) or not isinstance(expected_indices, (list, tuple)):
        return "identity:segment_indices"
    if (
        not stored_indices
        or any(type(index) is not int or index < 0 for index in stored_indices)
        or any(type(index) is not int or index < 0 for index in expected_indices)
        or len(set(stored_indices)) != len(stored_indices)
        or stored_indices != list(expected_indices)
    ):
        return "identity:segment_indices"
    if (
        record.get("code_identity") != CODE_IDENTITY
        or expected.get("code_identity", CODE_IDENTITY) != CODE_IDENTITY
    ):
        return "identity:code_identity"
    return None


def _policy_mismatch(record: dict[str, Any], expected: dict[str, Any]) -> str | None:
    stored = record.get("policy")
    if not isinstance(stored, dict):
        return "policy:missing"
    for key, value in expected.get("policy", {}).items():
        if stored.get(key) != value:
            return f"policy:{key}"
    return None


def _route_mismatch(record: dict[str, Any], expected: dict[str, Any]) -> str | None:
    fingerprints = expected.get("configured_route_fingerprints")
    if fingerprints is None:
        return None
    routes = record.get("routes")
    stored = routes.get("configured") if isinstance(routes, dict) else None
    if not isinstance(stored, dict):
        return "routes:missing"
    for operation, fingerprint in fingerprints.items():
        entry = stored.get(operation)
        if not isinstance(entry, dict) or entry.get("inference_fingerprint") != fingerprint:
            return f"routes:{operation}"
    return None


def reflection_artifact_status(
    record: dict[str, Any] | None,
    expected_identity: dict[str, Any],
    current_segments: Sequence[Segment],
) -> dict[str, Any]:
    """Return ``current``/``stale``/``missing`` for a record against current formal text.

    ``current`` only means the record applies to this batch; it never means the critique
    was clean. A missing record is never treated as clean or as a successful critique.
    """
    if record is None:
        return {"status": "missing", "reason": "not_found"}
    if (
        not isinstance(record, dict)
        or type(record.get("schema")) is not int
        or record["schema"] != SCHEMA_VERSION
    ):
        return {"status": "stale", "reason": "schema"}
    reason = _identity_mismatch(record, expected_identity)
    if reason is not None:
        return {"status": "stale", "reason": reason}
    reason = _policy_mismatch(record, expected_identity)
    if reason is not None:
        return {"status": "stale", "reason": reason}
    reason = _route_mismatch(record, expected_identity)
    if reason is not None:
        return {"status": "stale", "reason": reason}

    targets = record.get("targets")
    if not isinstance(targets, dict) or targets.get("stage") != TARGET_STAGE:
        return {"status": "stale", "reason": "targets:stage"}
    rows = targets.get("segments")
    if not isinstance(rows, list):
        return {"status": "stale", "reason": "targets:stage"}
    segment_indices = record.get("segment_indices")
    if not isinstance(segment_indices, list) or len(rows) != len(segment_indices):
        return {"status": "stale", "reason": "targets:coverage"}
    by_index = {segment.index: segment for segment in current_segments}
    for index, row in zip(segment_indices, rows):
        if not isinstance(row, dict):
            return {"status": "stale", "reason": "targets:row"}
        if type(row.get("segment_index")) is not int or row["segment_index"] != index:
            return {"status": "stale", "reason": "targets:coverage"}
        segment = by_index.get(row.get("segment_index"))
        if segment is None:
            return {"status": "stale", "reason": "targets:segment_missing"}
        if _text_digest(segment.target) != {
            "state": row.get("state"),
            "sha256": row.get("sha256"),
            "length": row.get("length"),
        }:
            return {"status": "stale", "reason": "targets:post_alignment"}

    before = record.get("targets_before_polish")
    if not isinstance(before, dict) or before.get("stage") != TARGET_BEFORE_POLISH_STAGE:
        return {"status": "stale", "reason": "targets_before_polish:stage"}
    before_rows = before.get("segments")
    if not isinstance(before_rows, list) or len(before_rows) != len(segment_indices):
        return {"status": "stale", "reason": "targets_before_polish:coverage"}
    for index, row in zip(segment_indices, before_rows):
        if not isinstance(row, dict):
            return {"status": "stale", "reason": "targets_before_polish:row"}
        if type(row.get("segment_index")) is not int or row["segment_index"] != index:
            return {"status": "stale", "reason": "targets_before_polish:coverage"}
        segment = by_index.get(row.get("segment_index"))
        if segment is None:
            return {"status": "stale", "reason": "targets_before_polish:segment_missing"}
        if _text_digest(segment.target_before_polish) != {
            "state": row.get("state"),
            "sha256": row.get("sha256"),
            "length": row.get("length"),
        }:
            return {"status": "stale", "reason": "targets_before_polish:digest"}
    return {"status": "current", "reason": None}


__all__ = [
    "CODE_IDENTITY",
    "SCHEMA_VERSION",
    "build_reflection_record",
    "reflection_artifact_key",
    "reflection_artifact_status",
]
