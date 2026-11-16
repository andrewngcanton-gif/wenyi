"""Seeded continuous-chapter and stratified human-review suggestions.

This module never marks a row as reviewed. It produces suggestions that a human
reviewer may later accept, edit or reject; the provenance and review state are
explicit in every row.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from .common import EvalError, PathGuard, append_jsonl, read_jsonl, utc_now, write_json_atomic

MIN_PARAGRAPHS = 200
MAX_PARAGRAPHS = 300
STRATA = ("narration", "dialogue", "interior", "verse")


class SampleSizeError(EvalError):
    """The source cannot supply the required human-review subset."""

    def __init__(self, available: int):
        super().__init__(
            f"Only {available} eligible paragraphs are available; "
            f"the human-review subset requires {MIN_PARAGRAPHS}-{MAX_PARAGRAPHS}."
        )
        self.available = available


def classify_stratum(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(('"', "“", "'")) or stripped.count('"') >= 2:
        return "dialogue"
    lowered = stripped.lower()
    if any(word in lowered for word in ("wondered", "remembered", "thought", "felt")):
        return "interior"
    if stripped.count(";") >= 2 or stripped.endswith((";", ",")):
        return "verse"
    return "narration"


def _paragraphs(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chapter in chapters:
        for segment in chapter["segments"]:
            if segment["kind"] == "heading" or not segment["source"].strip():
                continue
            rows.append(
                {
                    "id": segment["id"],
                    "chapter": chapter["index"],
                    "segment": segment["index"],
                    "source": segment["source"],
                    "stratum": classify_stratum(segment["source"]),
                }
            )
    return rows


def select_continuous_window(
    chapters: list[dict[str, Any]], *, seed: int, minimum: int = MIN_PARAGRAPHS
) -> tuple[list[int], list[int]]:
    """Choose a seeded contiguous chapter window with at least ``minimum`` paragraphs."""
    counts = [
        sum(
            1
            for segment in chapter["segments"]
            if segment["kind"] != "heading" and segment["source"].strip()
        )
        for chapter in chapters
    ]
    windows: list[tuple[int, int, int]] = []
    for start in range(len(chapters)):
        total = 0
        for end in range(start, len(chapters)):
            total += counts[end]
            if total >= minimum:
                windows.append((start, end + 1, total))
                break
    if not windows:
        raise SampleSizeError(sum(counts))
    preferred = [window for window in windows if window[2] <= MAX_PARAGRAPHS]
    choices = preferred or windows
    start, end, total = random.Random(seed).choice(choices)
    return list(range(start, end)), [start, end, total]


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    size: int,
    minimum: int = MIN_PARAGRAPHS,
    maximum: int = MAX_PARAGRAPHS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Round-robin strata after a seeded shuffle, then restore source order."""
    if not minimum <= size <= maximum:
        raise SampleSizeError(len(rows))
    if len(rows) < size:
        raise SampleSizeError(len(rows))
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for row in rows:
        buckets.setdefault(row["stratum"], []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    quotas = dict.fromkeys(buckets, 0)
    while len(selected) < size:
        progressed = False
        for stratum in STRATA:
            bucket = buckets.get(stratum, [])
            if quotas[stratum] < len(bucket) and len(selected) < size:
                selected.append(bucket[quotas[stratum]])
                quotas[stratum] += 1
                progressed = True
        if not progressed:
            # Any extra strata introduced by future fixtures use source order.
            leftovers = [
                row
                for name, bucket in buckets.items()
                if name not in STRATA
                for row in bucket[quotas.get(name, 0) :]
            ]
            if not leftovers:
                raise SampleSizeError(len(rows))
            quotas.setdefault("other", 0)
            selected.extend(leftovers[: size - len(selected)])
            break
    selected.sort(key=lambda row: (row["chapter"], row["segment"]))
    return selected, {stratum: count for stratum, count in quotas.items() if count}


def _enrich_rows(
    rows: list[dict[str, Any]],
    chapters: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    fixture: dict[str, Any],
    holdout: bool,
) -> list[dict[str, Any]]:
    by_index = {chapter["index"]: chapter for chapter in chapters}
    enriched = []
    for row in rows:
        chapter = by_index[row["chapter"]]
        position = next(
            index
            for index, segment in enumerate(chapter["segments"])
            if segment["index"] == row["segment"]
        )
        before = chapter["segments"][position - 1]["source"] if position > 0 else ""
        after = (
            chapter["segments"][position + 1]["source"]
            if position + 1 < len(chapter["segments"])
            else ""
        )
        enriched.append(
            {
                "id": row["id"],
                "source_sha256": source["sha256"],
                "chapter_index": row["chapter"],
                "chapter_title": chapter["title"],
                "segment_index": row["segment"],
                "stratum": row["stratum"],
                "source_text": row["source"],
                "source_context_before": before,
                "source_context_after": after,
                "license": fixture["license"],
                "provenance": fixture["provenance"],
                "synthetic": bool(fixture.get("synthetic")),
                "source_language": source["language"]["source"],
                "target_language": source["language"]["target"],
                "holdout": holdout,
                "reviewed": False,
                "review_status": (
                    "declared_holdout_not_reviewed" if holdout else "suggested_not_reviewed"
                ),
            }
        )
    return enriched


def build_samples(
    *,
    root: Path,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    limits: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guard = PathGuard(root)
    directory = guard.child("samples")
    directory.mkdir(parents=True, exist_ok=True)
    minimum = int((limits or {}).get("min") or MIN_PARAGRAPHS)
    maximum = int((limits or {}).get("max") or MAX_PARAGRAPHS)
    if minimum < 1 or maximum < minimum:
        raise EvalError("Human sampling limits must satisfy 1 <= min <= max")
    seed = int(manifest["llm"]["seed"])
    chapters = extraction["expected"]["chapters"]
    holdout_chapters = sorted(
        {
            int(value)
            for value in (((protocol or {}).get("sampling") or {}).get("holdout") or {}).get(
                "chapters", []
            )
        }
    )
    existing = {chapter["index"] for chapter in chapters}
    holdout_found = [chapter for chapter in chapters if chapter["index"] in holdout_chapters]
    main_chapters = [chapter for chapter in chapters if chapter["index"] not in holdout_chapters]
    if not main_chapters:
        raise SampleSizeError(0)
    window, window_detail = select_continuous_window(main_chapters, seed=seed, minimum=minimum)
    selected_chapters = [main_chapters[index] for index in window]
    rows = _paragraphs(selected_chapters)
    if len(rows) < minimum:
        raise SampleSizeError(len(rows))
    size = min(maximum, max(minimum, round(len(rows) * 0.8)))
    sample, quotas = stratified_sample(rows, seed=seed, size=size, minimum=minimum, maximum=maximum)
    source = manifest["source"]
    fixture = manifest["fixture"]
    enriched = _enrich_rows(
        sample, selected_chapters, source=source, fixture=fixture, holdout=False
    )
    holdout_rows = _paragraphs(holdout_found)
    holdout_enriched = _enrich_rows(
        holdout_rows, holdout_found, source=source, fixture=fixture, holdout=True
    )
    append_jsonl(directory / "strata.jsonl", enriched)
    append_jsonl(directory / "holdout_strata.jsonl", holdout_enriched)
    selection = {
        "protocol_version": 2,
        "created_at": utc_now(),
        "seed": seed,
        "chapter_window": window_detail,
        "eligible_paragraphs": len(rows),
        "target_size": size,
        "selected": len(enriched),
        "strata_quotas": quotas,
        "holdout_declared": holdout_chapters,
        "holdout_chapters": [chapter["index"] for chapter in holdout_found],
        "holdout_missing_chapters": sorted(set(holdout_chapters) - existing),
        "holdout_selected": len(holdout_enriched),
        "human_review_executed": False,
        "review_status": "suggested_not_reviewed",
        "reason": (
            "These rows are suggestions only. No human reviewer has rated them, so they must "
            "not be counted as human quality evidence."
        ),
        "license": fixture["license"],
        "provenance": fixture["provenance"],
        "source_sha256": source["sha256"],
        "groups": list(manifest.get("groups") or {}),
    }
    write_json_atomic(directory / "selection.json", selection)
    return selection


def read_strata(root: Path) -> list[dict[str, Any]]:
    path = Path(root) / "samples" / "strata.jsonl"
    return read_jsonl(path)


def write_empty_ratings_template(root: Path) -> Path:
    """Write a ratings CSV with every row blank; blanks stay missing on import."""
    rows = read_strata(root)
    path = Path(root) / "samples" / "ratings_template.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "rating", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "rating": "", "notes": ""})
    return path
