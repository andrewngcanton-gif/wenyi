"""Anonymous pairwise export and strict rating import.

The ratings CSV contains no group labels. A separate key file maps each pair.
This is anonymous to the rater, but it is not described as double blind because
the harness cannot verify how ratings were produced.

Annotations are bound to the exported source and translation hashes. A rating
whose pair id, text hashes or current output no longer match is rejected
atomically instead of being silently applied to changed translations.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .common import EvalError, PathGuard, append_jsonl, read_json, utc_now, write_json_atomic

DEFAULT_COMPARISONS = (
    {"id": "A_vs_Aprime", "baseline": "A", "candidate": "A_prime"},
    {"id": "A_vs_B", "baseline": "A", "candidate": "B"},
    {"id": "Aprime_vs_B", "baseline": "A_prime", "candidate": "B"},
)
REFLECTIVE_COMPARISONS = (
    {"id": "Aprime_vs_C", "baseline": "A_prime", "candidate": "C"},
    {"id": "Aprime_vs_C_keep", "baseline": "A_prime", "candidate": "C_keep"},
    {"id": "Aprime_vs_D", "baseline": "A_prime", "candidate": "D"},
)
ALL_COMPARISONS = DEFAULT_COMPARISONS + REFLECTIVE_COMPARISONS
RATINGS_COLUMNS = (
    "pair_id",
    "source_text",
    "source_context_before",
    "source_context_after",
    "left_text",
    "right_text",
    "preference",
    "omission",
    "addition",
    "left_mqm_errors",
    "right_mqm_errors",
    "left_severe",
    "right_severe",
    "notes",
)
IMPORT_COLUMNS = (
    "pair_id",
    "preference",
    "omission",
    "addition",
    "notes",
)
PREFERENCES = {"left", "right", "tie", "skip"}
ANOMALY = {"none", "left", "right", "both"}
SEVERE = {"yes", "no"}
KEY_COLUMNS = (
    "pair_id",
    "comparison",
    "role",
    "left_group",
    "right_group",
    "left_sha256",
    "right_sha256",
    "segment_id",
    "cluster",
    "source_sha256",
    "source_text_sha256",
    "holdout",
)


class RatingError(EvalError):
    """Submitted ratings are invalid; no partial import is published."""


def _target_index(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for group in extraction.get("groups", {}):
        data = extraction["groups"].get(group, {})
        for chapter in data.get("chapters", []):
            for segment in chapter.get("segments", []):
                index[f"{group}|{segment['id']}"] = segment
    return index


def comparisons_for(groups: Iterable[str]) -> tuple[dict[str, str], ...]:
    """Select preregistered comparisons whose two groups are both present."""
    available = set(groups)
    return tuple(
        comparison
        for comparison in ALL_COMPARISONS
        if comparison["baseline"] in available and comparison["candidate"] in available
    )


def protocol_comparisons_for(
    protocol: dict[str, Any] | None, groups: Iterable[str]
) -> tuple[dict[str, str], ...]:
    """Use the frozen comparison list when registered, else the legacy default."""
    if not protocol:
        return comparisons_for(groups)
    available = set(groups)
    selected = []
    for item in protocol.get("comparisons", []):
        if item["baseline"] in available and item["candidate"] in available:
            selected.append(
                {
                    "id": item["id"],
                    "baseline": item["baseline"],
                    "candidate": item["candidate"],
                    "role": item.get("role"),
                }
            )
    return tuple(selected)


def _pair_id(seed: int, comparison_id: str, segment_id: str) -> str:
    """Return an opaque identifier so blind packets never carry variant labels."""
    digest = hashlib.sha256(f"{seed}:{comparison_id}:{segment_id}".encode("utf-8")).hexdigest()
    return f"pair-{digest[:20]}"


def _side_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _orientation(pair_id: str, seed: int) -> bool:
    """Return True when the candidate is shown on the left; deterministic per id."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()[:8], "big")
    return bool(value & 1)


def cluster_for(
    *,
    protocol: dict[str, Any] | None,
    book: str,
    chapter_index: int | None,
    segment_index: int | None,
) -> str:
    """Assign a pre-declared cluster identity; never inferred from results."""
    sampling = (protocol or {}).get("sampling") or {}
    unit = sampling.get("clusters") or "chapter"
    if unit == "book":
        return f"book:{book}"
    if unit == "block":
        size = int(sampling.get("block_size") or 1)
        chapter = "unknown" if chapter_index is None else chapter_index
        block = 0 if segment_index is None else segment_index // size
        return f"block:{chapter}:{block}"
    return f"chapter:{'unknown' if chapter_index is None else chapter_index}"


def _strata_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_pairwise(
    *,
    root: Path,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guard = PathGuard(root)
    directory = guard.child("samples")
    strata_path = directory / "strata.jsonl"
    if not strata_path.is_file():
        raise EvalError("Stratified sample is missing; run sample generation first")
    strata = _strata_rows(strata_path)
    holdout_path = directory / "holdout_strata.jsonl"
    holdout_strata = _strata_rows(holdout_path) if holdout_path.is_file() else []
    targets = _target_index(extraction)
    seed = int(manifest["llm"]["seed"]) + 1
    sections = {
        **{row["id"]: row["source_text"] for row in strata},
        **{row["id"]: row["source_text"] for row in holdout_strata},
    }
    groups = manifest.get("groups") or extraction.get("groups") or {}
    comparisons = protocol_comparisons_for(protocol, groups)
    pairs: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    cluster_counts: dict[str, int] = {}
    holdout_counts: dict[str, int] = {}
    observed = [(row, False) for row in strata] + [(row, True) for row in holdout_strata]
    for row, holdout in observed:
        cluster = cluster_for(
            protocol=protocol,
            book=manifest.get("book", ""),
            chapter_index=row.get("chapter_index"),
            segment_index=row.get("segment_index"),
        )
        for comparison in comparisons:
            baseline = targets.get(f"{comparison['baseline']}|{row['id']}")
            candidate = targets.get(f"{comparison['candidate']}|{row['id']}")
            pair_id = _pair_id(seed, comparison["id"], row["id"])
            if (
                not baseline
                or not candidate
                or baseline.get("status") != "translated"
                or candidate.get("status") != "translated"
            ):
                missing.append(
                    {
                        "pair_id": pair_id,
                        "comparison": comparison["id"],
                        "holdout": holdout,
                        "baseline_status": baseline.get("status") if baseline else "missing",
                        "candidate_status": candidate.get("status") if candidate else "missing",
                    }
                )
                continue
            candidate_left = _orientation(pair_id, seed)
            left = candidate["target"] if candidate_left else baseline["target"]
            right = baseline["target"] if candidate_left else candidate["target"]
            left_group = comparison["candidate"] if candidate_left else comparison["baseline"]
            right_group = comparison["baseline"] if candidate_left else comparison["candidate"]
            if not isinstance(left, str) or not isinstance(right, str):
                missing.append(
                    {
                        "pair_id": pair_id,
                        "comparison": comparison["id"],
                        "holdout": holdout,
                        "baseline_status": "invalid",
                        "candidate_status": "invalid",
                    }
                )
                continue
            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_text": sections[row["id"]],
                    "source_context_before": row["source_context_before"],
                    "source_context_after": row["source_context_after"],
                    "left_text": left,
                    "right_text": right,
                    "preference": "",
                    "omission": "",
                    "addition": "",
                    "left_mqm_errors": "",
                    "right_mqm_errors": "",
                    "left_severe": "",
                    "right_severe": "",
                    "notes": "",
                }
            )
            keys.append(
                {
                    "pair_id": pair_id,
                    "comparison": comparison["id"],
                    "role": comparison.get("role") or "legacy",
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_sha256": _side_hash(left),
                    "right_sha256": _side_hash(right),
                    "segment_id": row["id"],
                    "cluster": cluster,
                    "source_sha256": manifest["source"]["sha256"],
                    "source_text_sha256": _side_hash(sections[row["id"]]),
                    "holdout": holdout,
                }
            )
            if holdout:
                holdout_counts[comparison["id"]] = holdout_counts.get(comparison["id"], 0) + 1
            else:
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    ratings_path = directory / "pairwise.csv"
    with open(ratings_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATINGS_COLUMNS)
        writer.writeheader()
        writer.writerows(pairs)
    key_path = directory / "pairwise_key.csv"
    with open(key_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(keys)
    if missing:
        append_jsonl(directory / "pairwise_missing.jsonl", missing)
    main_pairs = sum(1 for key in keys if not key["holdout"])
    holdout_pairs = len(pairs) - main_pairs
    preregistration = {
        "protocol_version": 2,
        "created_at": utc_now(),
        "seed": seed,
        "source_sha256": manifest["source"]["sha256"],
        "comparisons": [
            {
                "id": item["id"],
                "baseline": item["baseline"],
                "candidate": item["candidate"],
                "role": item.get("role") or "legacy",
            }
            for item in comparisons
        ],
        "sampled_segments": len(strata),
        "expected_pairs": len(pairs),
        "missing_pairs": len(missing),
        "ratings_allowed": sorted(PREFERENCES),
        "omission_allowed": sorted(ANOMALY),
        "addition_allowed": sorted(ANOMALY),
        "severe_allowed": sorted(SEVERE),
        "mqm_scale": "nonnegative error count per side; blank means not assessed",
        "denominator": main_pairs,
        "holdout_denominator": holdout_pairs,
        "available_pairs": main_pairs + holdout_pairs,
        "comparison_denominators": {
            item["id"]: {"main": len(strata), "holdout": len(holdout_strata)}
            for item in comparisons
        },
        "holdout_pairs": holdout_counts,
        "human_ratings_received": 0,
        "blinding": "anonymous_side_labels_not_double_blind",
        "blind_packet_labels": "opaque_pair_ids_absent_from_key",
        "protocol_content_sha256": None,
        "cluster_unit": ((protocol or {}).get("sampling") or {}).get("clusters"),
        "clusters": cluster_counts,
        "reason_unrated": "No human ratings have been imported; all pairwise cells remain missing.",
    }
    if protocol is not None:
        from .protocol import content_hash

        preregistration["protocol_content_sha256"] = content_hash(protocol)
    write_json_atomic(directory / "pairwise_preregistration.json", preregistration)
    return preregistration


def _read_csv(path: Path, required: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise RatingError(f"{label} file is missing: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RatingError(f"{label} file has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise RatingError(f"{label} file lacks columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _key_index(directory: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(directory / "pairwise_key.csv", ("pair_id",), "pairwise key")
    return {row["pair_id"]: row for row in rows}


def _optional_count(value: str | None, *, line: int, field: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        raise RatingError(f"ratings line {line}: {field} must be a nonnegative integer") from None
    if parsed < 0:
        raise RatingError(f"ratings line {line}: {field} must be a nonnegative integer")
    return parsed


def _optional_flag(value: str | None, *, line: int, field: str) -> str | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text not in SEVERE:
        raise RatingError(f"ratings line {line}: {field} must be yes, no or blank")
    return text


def _verify_text(
    *,
    line: int,
    label: str,
    text: str | None,
    expected: str | None,
) -> None:
    if expected and text is not None and text != "":
        if _side_hash(text) != expected:
            raise RatingError(
                f"ratings line {line}: {label} text no longer matches the exported hash"
            )


def _current_targets(root: Path) -> dict[str, dict[str, Any]] | None:
    path = Path(root) / "extraction.json"
    if not path.is_file():
        return None
    try:
        extraction = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return _target_index(extraction)


def verify_output_identity(
    *,
    key_row: dict[str, str],
    targets: dict[str, dict[str, Any]] | None,
) -> str | None:
    """Return a stale reason when current outputs no longer match the blind packet."""
    if targets is None:
        return None
    segment_id = key_row.get("segment_id")
    if not segment_id:
        return None
    left_group = key_row.get("left_group", "")
    right_group = key_row.get("right_group", "")
    left = targets.get(f"{left_group}|{segment_id}")
    right = targets.get(f"{right_group}|{segment_id}")
    if left is None or right is None:
        return "current extraction no longer contains this pair"
    if left.get("status") != "translated" or right.get("status") != "translated":
        return "current extraction no longer has both translated sides"
    if _side_hash(left["target"]) != key_row.get("left_sha256"):
        return "current left translation differs from the exported text"
    if _side_hash(right["target"]) != key_row.get("right_sha256"):
        return "current right translation differs from the exported text"
    return None


def _prepare_pairwise_import(
    root: Path,
    ratings_path: Path,
    *,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize submitted ratings; nothing is published here."""
    directory = Path(root) / "samples"
    preregistration_path = directory / "pairwise_preregistration.json"
    if not preregistration_path.is_file():
        raise RatingError("Pairwise preregistration is missing")
    preregistration = read_json(preregistration_path)
    rows = _read_csv(ratings_path, IMPORT_COLUMNS, "ratings")
    allowed = _key_index(directory)
    exported_rows = {
        row["pair_id"]: row
        for row in _read_csv(directory / "pairwise.csv", ("pair_id",), "pairwise export")
    }
    if len(exported_rows) != (
        preregistration.get("denominator", 0) + preregistration.get("holdout_denominator", 0)
    ):
        raise RatingError("Preregistered denominator disagrees with exported pairs")
    targets = _target_index(extraction) if extraction is not None else _current_targets(root)
    expected_source_sha = (
        (extraction.get("expected") or {}).get("source_sha256")
        if isinstance(extraction, dict)
        else None
    )
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    verified = 0
    for line_number, row in enumerate(rows, start=2):
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            raise RatingError(f"ratings line {line_number}: pair_id is blank")
        if pair_id in seen:
            raise RatingError(f"ratings line {line_number}: duplicate pair_id {pair_id}")
        seen.add(pair_id)
        if pair_id not in allowed:
            raise RatingError(f"ratings line {line_number}: unknown pair_id {pair_id}")
        key_row = allowed[pair_id]
        exported = exported_rows.get(pair_id, {})
        if (
            expected_source_sha
            and key_row.get("source_sha256")
            and key_row["source_sha256"] != expected_source_sha
        ):
            raise RatingError(
                f"ratings line {line_number}: the source identity changed; re-export before rating"
            )
        _verify_text(
            line=line_number,
            label="source",
            text=row.get("source_text") or exported.get("source_text"),
            expected=key_row.get("source_text_sha256"),
        )
        _verify_text(
            line=line_number,
            label="left",
            text=row.get("left_text") or exported.get("left_text"),
            expected=key_row.get("left_sha256"),
        )
        _verify_text(
            line=line_number,
            label="right",
            text=row.get("right_text") or exported.get("right_text"),
            expected=key_row.get("right_sha256"),
        )
        stale = verify_output_identity(key_row=key_row, targets=targets)
        if stale:
            raise RatingError(f"ratings line {line_number}: {stale}; re-export before rating")
        if key_row.get("left_sha256") and key_row.get("right_sha256"):
            verified += 1
        values = {
            "preference": (row.get("preference") or "").strip().lower(),
            "omission": (row.get("omission") or "").strip().lower(),
            "addition": (row.get("addition") or "").strip().lower(),
            "notes": (row.get("notes") or "").strip(),
            "left_mqm_errors": _optional_count(
                row.get("left_mqm_errors"), line=line_number, field="left_mqm_errors"
            ),
            "right_mqm_errors": _optional_count(
                row.get("right_mqm_errors"), line=line_number, field="right_mqm_errors"
            ),
            "left_severe": _optional_flag(
                row.get("left_severe"), line=line_number, field="left_severe"
            ),
            "right_severe": _optional_flag(
                row.get("right_severe"), line=line_number, field="right_severe"
            ),
        }
        if values["preference"] and values["preference"] not in PREFERENCES:
            raise RatingError(
                f"ratings line {line_number}: invalid preference {values['preference']!r}"
            )
        for field in ("omission", "addition"):
            if values[field] and values[field] not in ANOMALY:
                raise RatingError(
                    f"ratings line {line_number}: invalid {field} value {values[field]!r}"
                )
        normalized.append(
            {
                "pair_id": pair_id,
                "left_sha256": key_row.get("left_sha256"),
                "right_sha256": key_row.get("right_sha256"),
                "source_sha256": key_row.get("source_sha256"),
                "source_text_sha256": key_row.get("source_text_sha256"),
                **values,
            }
        )
    rated = [row for row in normalized if row["preference"]]
    missing = sorted(set(allowed) - {row["pair_id"] for row in normalized})
    unrated_blank = [row["pair_id"] for row in normalized if not row["preference"]]
    payload_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalized
    ).encode("utf-8")
    from .common import sha256_bytes, sha256_file

    summary = {
        "protocol_version": 3,
        "imported_at": utc_now(),
        "input_path": str(ratings_path),
        "denominator": preregistration["denominator"]
        + preregistration.get("holdout_denominator", 0),
        "main_denominator": preregistration["denominator"],
        "holdout_denominator": preregistration.get("holdout_denominator", 0),
        "submitted_rows": len(normalized),
        "key_sha256": sha256_file(directory / "pairwise_key.csv"),
        "preregistration_sha256": sha256_file(preregistration_path),
        "ratings_sha256": sha256_bytes(payload_bytes),
        "protocol_content_sha256": preregistration.get("protocol_content_sha256"),
        "rated_rows": len(rated),
        "blank_ratings": len(unrated_blank),
        "missing_ids": len(missing),
        "missing_pair_ids": missing,
        "preference_counts": {
            value: sum(1 for row in rated if row["preference"] == value)
            for value in sorted(PREFERENCES)
        },
        "omission_counts": {
            value: sum(1 for row in normalized if row["omission"] == value)
            for value in sorted(ANOMALY)
        },
        "addition_counts": {
            value: sum(1 for row in normalized if row["addition"] == value)
            for value in sorted(ANOMALY)
        },
        "mqm_assessed": sum(
            1
            for row in normalized
            if row["left_mqm_errors"] is not None and row["right_mqm_errors"] is not None
        ),
        "severe_assessed": sum(
            1
            for row in normalized
            if row["left_severe"] is not None and row["right_severe"] is not None
        ),
        "text_hash_verified": verified,
        "outputs_verified": targets is not None,
        "blinding": "anonymous_side_labels_not_double_blind",
        "source": "caller-supplied CSV; human identity and blinding are not verified",
    }
    return {
        "kind": "pairwise",
        "root": Path(root),
        "rows": normalized,
        "summary": summary,
        "ratings_path": directory / "pairwise_ratings.jsonl",
        "summary_path": directory / "pairwise_import.json",
    }


def publish_pairwise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    append_jsonl(payload["ratings_path"], payload["rows"])
    write_json_atomic(payload["summary_path"], payload["summary"])
    return payload["summary"]


def import_pairwise_ratings(
    root: Path,
    ratings_path: Path,
    *,
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _prepare_pairwise_import(root, ratings_path, extraction=extraction)
    return publish_pairwise_payload(payload)
