from __future__ import annotations

import csv
from pathlib import Path

import pytest

from wenyi_eval.common import sha256_file
from wenyi_eval.fixture import (
    CHAPTER_TITLES,
    chapter_paragraphs,
    fixture_metadata,
    write_fixture,
)
from wenyi_eval.sample import (
    MAX_PARAGRAPHS,
    MIN_PARAGRAPHS,
    SampleSizeError,
    build_samples,
    classify_stratum,
    read_strata,
    select_continuous_window,
    stratified_sample,
    write_empty_ratings_template,
)


def _chapters(count: int = 12, paragraphs: int = 20) -> list[dict]:
    texts = (
        '"The tide is turning," she said.',
        "Mira wondered about the old path.",
        "a; b; c",
        "The lantern glowed.",
    )
    chapters = []
    for chapter_index in range(count):
        segments = [{"index": 0, "kind": "heading", "source": f"Chapter {chapter_index}"}]
        for position in range(1, paragraphs + 1):
            source = texts[(position - 1) % len(texts)]
            segments.append(
                {
                    "index": position,
                    "kind": "text",
                    "source": source,
                    "id": f"src:{chapter_index}:{position}",
                }
            )
        chapters.append(
            {"index": chapter_index, "title": f"Chapter {chapter_index}", "segments": segments}
        )
    return chapters


def test_classify_stratum_covers_all_human_review_strata() -> None:
    assert classify_stratum('"The tide is turning," she said.') == "dialogue"
    assert classify_stratum("Mira wondered about the old path.") == "interior"
    assert classify_stratum("a; b; c") == "verse"
    assert classify_stratum("The lantern glowed.") == "narration"


def test_continuous_window_is_seeded_contiguous_and_at_least_minimum() -> None:
    chapters = _chapters()

    first, detail = select_continuous_window(chapters, seed=41)
    second, other_detail = select_continuous_window(chapters, seed=41)

    assert first == second
    assert detail == other_detail
    assert detail[2] >= MIN_PARAGRAPHS
    assert first == list(range(detail[0], detail[1]))
    assert first == list(range(first[0], first[0] + len(first)))


def test_stratified_sample_is_seeded_and_ordered_by_source_position() -> None:
    rows = []
    for chapter in _chapters():
        for segment in chapter["segments"][1:]:
            rows.append(
                {
                    "id": segment["id"],
                    "chapter": chapter["index"],
                    "segment": segment["index"],
                    "source": segment["source"],
                    "stratum": classify_stratum(segment["source"]),
                }
            )

    first, quotas = stratified_sample(rows, seed=7, size=200)
    second, second_quotas = stratified_sample(rows, seed=7, size=200)

    assert first == second
    assert quotas == second_quotas
    assert len(first) == 200
    assert len({row["id"] for row in first}) == 200
    assert first == sorted(first, key=lambda row: (row["chapter"], row["segment"]))
    assert set(quotas) == {"dialogue", "interior", "narration", "verse"}
    assert all(count == 50 for count in quotas.values())


def test_sample_requires_at_least_the_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SampleSizeError):
        stratified_sample([], seed=1, size=MIN_PARAGRAPHS - 1)
    with pytest.raises(SampleSizeError):
        stratified_sample([], seed=1, size=MAX_PARAGRAPHS + 1)


def test_build_samples_marks_rows_as_unreviewed_suggestions(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    source = root / "source" / "fixture.en.md"
    source.parent.mkdir(parents=True)
    source.write_text("placeholder\n", encoding="utf-8")
    source_sha = sha256_file(source)
    chapters = []
    for chapter_index, title in enumerate(CHAPTER_TITLES):
        segments = [{"index": 0, "kind": "heading", "source": title}]
        for position, paragraph in enumerate(chapter_paragraphs(chapter_index), start=1):
            segments.append(
                {
                    "index": position,
                    "kind": "text",
                    "source": paragraph,
                    "id": f"{source_sha[:16]}:{chapter_index}:{position}",
                }
            )
        chapters.append({"index": chapter_index, "title": title, "segments": segments})
    manifest = {
        "source": {
            "sha256": source_sha,
            "language": {"source": "en", "target": "zh"},
        },
        "fixture": fixture_metadata(),
        "llm": {"seed": 2026},
    }
    extraction = {"expected": {"chapters": chapters}}

    selection = build_samples(root=root, manifest=manifest, extraction=extraction)
    rows = read_strata(root)

    assert MIN_PARAGRAPHS <= selection["selected"] <= MAX_PARAGRAPHS
    assert selection["human_review_executed"] is False
    assert selection["review_status"] == "suggested_not_reviewed"
    assert selection["source_sha256"] == source_sha
    assert selection["license"] == "CC0-1.0"
    assert len(rows) == selection["selected"]
    assert all(row["reviewed"] is False for row in rows)
    assert all(row["review_status"] == "suggested_not_reviewed" for row in rows)
    assert all(row["license"] == "CC0-1.0" for row in rows)
    assert all(row["source_sha256"] == source_sha for row in rows)
    assert all(row["provenance"] for row in rows)
    assert len(selection["chapter_window"]) == 3

    template = write_empty_ratings_template(root)
    with open(template, "r", encoding="utf-8", newline="") as handle:
        ratings = list(csv.DictReader(handle))
    assert len(ratings) == len(rows)
    assert all(row["rating"] == "" for row in ratings)


def test_fixture_writer_is_license_clear_and_deterministic(tmp_path: Path) -> None:
    first_source, first_metadata = write_fixture(tmp_path / "first")
    second_source, _ = write_fixture(tmp_path / "second")

    assert first_source.read_bytes() == second_source.read_bytes()
    metadata = fixture_metadata()
    assert metadata["license"] == "CC0-1.0"
    assert metadata["synthetic"] is True
    assert metadata["chapters"] == len(CHAPTER_TITLES)
    assert metadata["paragraphs"] == len(CHAPTER_TITLES) * 24
    assert first_metadata.is_file()
