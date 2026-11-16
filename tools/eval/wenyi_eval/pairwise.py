"""Anonymous pairwise export and strict rating import.

The ratings CSV contains no group labels. A separate key file maps each pair.
This is anonymous to the rater, but it is not described as double blind because
the harness cannot verify how ratings were produced.
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
KEY_COLUMNS = (
    "pair_id",
    "comparison",
    "left_group",
    "right_group",
    "left_sha256",
    "right_sha256",
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


def export_pairwise(
    *, root: Path, manifest: dict[str, Any], extraction: dict[str, Any]
) -> dict[str, Any]:
    guard = PathGuard(root)
    directory = guard.child("samples")
    strata_path = directory / "strata.jsonl"
    if not strata_path.is_file():
        raise EvalError("Stratified sample is missing; run sample generation first")
    strata = []
    with open(strata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                strata.append(json.loads(line))
    targets = _target_index(extraction)
    seed = int(manifest["llm"]["seed"]) + 1
    sections = {**{row["id"]: row["source_text"] for row in strata}}
    groups = manifest.get("groups") or extraction.get("groups") or {}
    comparisons = comparisons_for(groups)
    pairs: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in strata:
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
                    "notes": "",
                }
            )
            keys.append(
                {
                    "pair_id": pair_id,
                    "comparison": comparison["id"],
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_sha256": _side_hash(left),
                    "right_sha256": _side_hash(right),
                }
            )
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
    preregistration = {
        "protocol_version": 1,
        "created_at": utc_now(),
        "seed": seed,
        "source_sha256": manifest["source"]["sha256"],
        "comparisons": list(comparisons),
        "sampled_segments": len(strata),
        "expected_pairs": len(pairs),
        "missing_pairs": len(missing),
        "ratings_allowed": sorted(PREFERENCES),
        "omission_allowed": sorted(ANOMALY),
        "addition_allowed": sorted(ANOMALY),
        "denominator": len(pairs),
        "human_ratings_received": 0,
        "blinding": "anonymous_side_labels_not_double_blind",
        "blind_packet_labels": "opaque_pair_ids_absent_from_key",
        "reason_unrated": "No human ratings have been imported; all pairwise cells remain missing.",
    }
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


def import_pairwise_ratings(root: Path, ratings_path: Path) -> dict[str, Any]:
    """Validate and normalize submitted ratings; invalid input publishes nothing."""
    directory = Path(root) / "samples"
    preregistration_path = directory / "pairwise_preregistration.json"
    if not preregistration_path.is_file():
        raise RatingError("Pairwise preregistration is missing")
    preregistration = read_json(preregistration_path)
    rows = _read_csv(ratings_path, IMPORT_COLUMNS, "ratings")
    allowed_ids = set()
    with open(directory / "pairwise.csv", "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            allowed_ids.add(row["pair_id"])
    if len(allowed_ids) != preregistration.get("denominator"):
        raise RatingError("Preregistered denominator disagrees with exported pairs")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        pair_id = (row.get("pair_id") or "").strip()
        if not pair_id:
            raise RatingError(f"ratings line {line_number}: pair_id is blank")
        if pair_id in seen:
            raise RatingError(f"ratings line {line_number}: duplicate pair_id {pair_id}")
        seen.add(pair_id)
        if pair_id not in allowed_ids:
            raise RatingError(f"ratings line {line_number}: unknown pair_id {pair_id}")
        values = {
            "preference": (row.get("preference") or "").strip().lower(),
            "omission": (row.get("omission") or "").strip().lower(),
            "addition": (row.get("addition") or "").strip().lower(),
            "notes": (row.get("notes") or "").strip(),
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
        normalized.append({"pair_id": pair_id, **values})
    rated = [row for row in normalized if row["preference"]]
    missing = sorted(allowed_ids - {row["pair_id"] for row in normalized})
    unrated_blank = [row["pair_id"] for row in normalized if not row["preference"]]
    normalized_path = directory / "pairwise_ratings.jsonl"
    append_jsonl(normalized_path, normalized)
    summary = {
        "protocol_version": 1,
        "imported_at": utc_now(),
        "input_path": str(ratings_path),
        "denominator": preregistration["denominator"],
        "submitted_rows": len(normalized),
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
        "blinding": "anonymous_side_labels_not_double_blind",
        "source": "caller-supplied CSV; human identity and blinding are not verified",
    }
    write_json_atomic(directory / "pairwise_import.json", summary)
    return summary
