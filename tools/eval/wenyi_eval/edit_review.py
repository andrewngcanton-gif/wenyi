"""Blind before/after edit packet for the human degradation-rate gate.

Only segments a candidate actually edited relative to its own pre-polish draft
enter the packet. The reviewer sees opaque edit ids and the before/after texts,
never the variant name. The private key keeps the group, segment identity and
text hashes; changed outputs invalidate a submitted rating.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import PathGuard, append_jsonl, read_json, utc_now, write_json_atomic
from .pairwise import (
    RatingError,
    _current_targets,
    _read_csv,
    _side_hash,
    _target_index,
    _verify_text,
    cluster_for,
    protocol_comparisons_for,
)

EDIT_COLUMNS = (
    "edit_id",
    "source_text",
    "before_text",
    "after_text",
    "worsened",
    "severity",
    "notes",
)
EDIT_KEY_COLUMNS = (
    "edit_id",
    "group",
    "comparison",
    "segment_id",
    "cluster",
    "source_sha256",
    "source_text_sha256",
    "before_sha256",
    "after_sha256",
)
WORSENED = {"yes", "no", "skip"}
SEVERITY = {"low", "medium", "high"}


def _edit_id(seed: int, group: str, segment_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:edit:{group}:{segment_id}".encode("utf-8")).hexdigest()
    return f"edit-{digest[:20]}"


def export_edit_review(
    *,
    root: Path,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a blind before/after packet for every candidate in the comparisons."""
    guard = PathGuard(root)
    directory = guard.child("samples")
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(manifest["llm"]["seed"]) + 2
    comparisons = protocol_comparisons_for(protocol, extraction.get("groups", {}))
    candidates = {
        item["candidate"]: item["id"]
        for item in comparisons
        if item["candidate"] in extraction["groups"]
    }
    rows: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for group, comparison_id in sorted(candidates.items()):
        data = extraction["groups"].get(group) or {}
        edited = 0
        considered = 0
        for chapter in data.get("chapters", []):
            for segment in chapter.get("segments", []):
                considered += 1
                target = segment.get("target")
                before = segment.get("target_before_polish")
                if (
                    segment.get("status") != "translated"
                    or not isinstance(target, str)
                    or not isinstance(before, str)
                    or target == before
                ):
                    continue
                edited += 1
                edit_id = _edit_id(seed, group, segment["id"])
                source_text = segment.get("source") or ""
                rows.append(
                    {
                        "edit_id": edit_id,
                        "source_text": source_text,
                        "before_text": before,
                        "after_text": target,
                        "worsened": "",
                        "severity": "",
                        "notes": "",
                    }
                )
                keys.append(
                    {
                        "edit_id": edit_id,
                        "group": group,
                        "comparison": comparison_id,
                        "segment_id": segment["id"],
                        "cluster": cluster_for(
                            protocol=protocol,
                            book=manifest.get("book", ""),
                            chapter_index=chapter.get("index"),
                            segment_index=segment.get("segment"),
                        ),
                        "source_sha256": manifest["source"]["sha256"],
                        "source_text_sha256": _side_hash(source_text),
                        "before_sha256": _side_hash(before),
                        "after_sha256": _side_hash(target),
                    }
                )
        counts[group] = {"considered": considered, "edited": edited}
    with open(directory / "edit_review.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(directory / "edit_key.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EDIT_KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(keys)
    preregistration = {
        "protocol_version": 1,
        "created_at": utc_now(),
        "seed": seed,
        "source_sha256": manifest["source"]["sha256"],
        "groups": counts,
        "denominator": len(rows),
        "worsened_allowed": sorted(WORSENED),
        "severity_allowed": sorted(SEVERITY),
        "blinding": "anonymous_edit_ids_not_double_blind",
        "protocol_content_sha256": None,
        "reason_unrated": (
            "No human edit ratings have been imported; the degradation rate is unknown."
        ),
    }
    if protocol is not None:
        from .protocol import content_hash

        preregistration["protocol_content_sha256"] = content_hash(protocol)
    write_json_atomic(directory / "edit_preregistration.json", preregistration)
    return preregistration


def _severity(value: str | None, *, line: int) -> str | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text not in SEVERITY:
        raise RatingError(f"edit ratings line {line}: severity must be low, medium, high or blank")
    return text


def _prepare_edit_import(
    root: Path,
    ratings_path: Path,
    *,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the whole edit packet before publishing any rating."""
    directory = Path(root) / "samples"
    preregistration_path = directory / "edit_preregistration.json"
    if not preregistration_path.is_file():
        raise RatingError("Edit-review preregistration is missing")
    preregistration = read_json(preregistration_path)
    key_rows = _read_csv(directory / "edit_key.csv", ("edit_id", "group"), "edit key")
    key = {row["edit_id"]: row for row in key_rows}
    if len(key) != preregistration.get("denominator"):
        raise RatingError("Edit-review denominator disagrees with the exported packet")
    exported = {
        row["edit_id"]: row
        for row in _read_csv(directory / "edit_review.csv", ("edit_id",), "edit review")
    }
    targets = _target_index(extraction) if extraction is not None else _current_targets(root)
    expected_source_sha = (
        (extraction.get("expected") or {}).get("source_sha256")
        if isinstance(extraction, dict)
        else None
    )
    rows = _read_csv(ratings_path, ("edit_id", "worsened"), "edit ratings")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        edit_id = (row.get("edit_id") or "").strip()
        if not edit_id:
            raise RatingError(f"edit ratings line {line_number}: edit_id is blank")
        if edit_id in seen:
            raise RatingError(f"edit ratings line {line_number}: duplicate edit_id {edit_id}")
        seen.add(edit_id)
        if edit_id not in key:
            raise RatingError(f"edit ratings line {line_number}: unknown edit_id {edit_id}")
        key_row = key[edit_id]
        exported_row = exported.get(edit_id, {})
        if (
            expected_source_sha
            and key_row.get("source_sha256")
            and key_row["source_sha256"] != expected_source_sha
        ):
            raise RatingError(
                f"edit ratings line {line_number}: the source identity changed; re-export "
                "before rating"
            )
        _verify_text(
            line=line_number,
            label="source",
            text=row.get("source_text") or exported_row.get("source_text"),
            expected=key_row.get("source_text_sha256"),
        )
        _verify_text(
            line=line_number,
            label="before",
            text=row.get("before_text") or exported_row.get("before_text"),
            expected=key_row.get("before_sha256"),
        )
        _verify_text(
            line=line_number,
            label="after",
            text=row.get("after_text") or exported_row.get("after_text"),
            expected=key_row.get("after_sha256"),
        )
        if targets is not None:
            segment = targets.get(f"{key_row['group']}|{key_row['segment_id']}")
            if segment is None:
                raise RatingError(
                    f"edit ratings line {line_number}: current extraction lost this segment"
                )
            if segment.get("status") != "translated":
                raise RatingError(
                    f"edit ratings line {line_number}: current segment is not translated"
                )
            if _side_hash(segment["target"]) != key_row.get("after_sha256"):
                raise RatingError(
                    f"edit ratings line {line_number}: current target differs from the packet"
                )
            before = segment.get("target_before_polish")
            if not isinstance(before, str) or _side_hash(before) != key_row.get("before_sha256"):
                raise RatingError(
                    f"edit ratings line {line_number}: pre-polish draft differs from the packet"
                )
        worsened = (row.get("worsened") or "").strip().lower()
        if worsened and worsened not in WORSENED:
            raise RatingError(
                f"edit ratings line {line_number}: worsened must be yes, no, skip or blank"
            )
        normalized.append(
            {
                "edit_id": edit_id,
                "group": key_row["group"],
                "cluster": key_row.get("cluster") or "chapter:unknown",
                "source_sha256": key_row.get("source_sha256"),
                "source_text_sha256": key_row.get("source_text_sha256"),
                "before_sha256": key_row.get("before_sha256"),
                "after_sha256": key_row.get("after_sha256"),
                "worsened": worsened or None,
                "severity": _severity(row.get("severity"), line=line_number),
                "notes": (row.get("notes") or "").strip(),
            }
        )
    payload_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalized
    ).encode("utf-8")
    from .common import sha256_bytes, sha256_file

    assessed = [row for row in normalized if row["worsened"] in {"yes", "no"}]
    summary = {
        "protocol_version": 2,
        "imported_at": utc_now(),
        "input_path": str(ratings_path),
        "denominator": preregistration["denominator"],
        "key_sha256": sha256_file(directory / "edit_key.csv"),
        "preregistration_sha256": sha256_file(preregistration_path),
        "ratings_sha256": sha256_bytes(payload_bytes),
        "protocol_content_sha256": preregistration.get("protocol_content_sha256"),
        "submitted_rows": len(normalized),
        "assessed_rows": len(assessed),
        "skipped": sum(1 for row in normalized if row["worsened"] == "skip"),
        "blank": sum(1 for row in normalized if row["worsened"] is None),
        "worsened_counts": {
            value: sum(1 for row in assessed if row["worsened"] == value)
            for value in sorted({"yes", "no"})
        },
        "outputs_verified": targets is not None,
        "blinding": "anonymous_edit_ids_not_double_blind",
        "source": "caller-supplied CSV; human identity is reported, not verified",
    }
    return {
        "kind": "edit",
        "root": Path(root),
        "rows": normalized,
        "summary": summary,
        "ratings_path": directory / "edit_ratings.jsonl",
        "summary_path": directory / "edit_import.json",
    }


def publish_edit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    append_jsonl(payload["ratings_path"], payload["rows"])
    write_json_atomic(payload["summary_path"], payload["summary"])
    return payload["summary"]


def import_edit_ratings(
    root: Path,
    ratings_path: Path,
    *,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _prepare_edit_import(root, ratings_path, extraction=extraction)
    return publish_edit_payload(payload)
