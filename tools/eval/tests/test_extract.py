from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import chapter_with, write_run
from wenyi_eval.common import hash_tree
from wenyi_eval.extract import (
    _review_summary,
    expected_inventory,
    extract_group,
    segment_id,
)


def _expected() -> dict:
    return {
        "source_sha256": "a" * 64,
        "chapters": [
            {
                "index": 0,
                "title": "Chapter",
                "status": "done",
                "segments": [
                    {"index": 0, "source": "alpha", "kind": "text", "id": "id:0"},
                    {"index": 1, "source": "beta", "kind": "text", "id": "id:1"},
                    {"index": 2, "source": "gamma", "kind": "text", "id": "id:2"},
                    {"index": 3, "source": "expected delta", "kind": "text", "id": "id:3"},
                ],
            }
        ],
    }


def _run_with_mixed_targets(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    run = state / "book.en" / "targets" / "zh"
    chapter = chapter_with(
        [
            ("alpha", "译文甲"),
            ("beta", ""),
            ("gamma", None),
            ("actual delta", "旧译文"),
            ("extra", "多余"),
        ]
    )
    write_run(run, chapter=chapter, status="done")
    (run / "events.jsonl").write_text(
        json.dumps({"event": "translation.retry"})
        + "\n"
        + json.dumps({"event": "chapter.completed"})
        + "\n",
        encoding="utf-8",
    )
    return state


def test_extract_preserves_translated_empty_missing_and_stale_rows(tmp_path: Path) -> None:
    state = _run_with_mixed_targets(tmp_path)
    before = hash_tree(state)

    result = extract_group(group="A", state_dir=state, expected=_expected())

    assert result["status"] == "ok"
    assert result["coverage"] == {
        "expected": 4,
        "translated": 1,
        "empty": 1,
        "missing": 1,
        "stale": 2,
        "error": 0,
    }
    rows = result["chapters"][0]["segments"]
    assert [row["status"] for row in rows] == [
        "translated",
        "empty",
        "missing",
        "stale",
        "stale",
    ]
    assert rows[1]["target"] == ""
    assert rows[2]["target"] is None
    assert rows[3]["target"] is None
    assert rows[3]["reason"] == "segment source differs from the prepared snapshot"
    assert rows[4]["segment"] == 4
    assert rows[4]["target"] is None
    assert result["review"]["status"] == "missing"
    assert result["events"]["failure_or_retry_counts"] == {"translation.retry": 1}
    assert result["reflection"]["status"] == "unsupported"
    assert result["read_only_verified"] is True
    assert hash_tree(state) == before


def test_extract_keeps_missing_chapter_files_in_the_denominator(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run = state / "book.en" / "targets" / "zh"
    write_run(run, chapter=chapter_with([("alpha", "甲")]), status="done")
    expected = _expected()
    expected["chapters"][0]["segments"] = expected["chapters"][0]["segments"][:1]
    expected["chapters"].append(
        {
            "index": 1,
            "title": "Missing",
            "status": "pending",
            "segments": [{"index": 0, "source": "lost", "kind": "text", "id": "id:1:0"}],
        }
    )

    result = extract_group(group="A", state_dir=state, expected=expected)

    assert result["chapters"][1]["chapter_file"] == "missing"
    assert result["chapters"][1]["segments"][0]["status"] == "missing"
    assert result["coverage"] == {
        "expected": 2,
        "translated": 1,
        "empty": 0,
        "missing": 1,
        "stale": 0,
        "error": 0,
    }


def test_extract_keeps_a_stale_source_in_the_denominator_without_reusing_text(
    tmp_path: Path,
) -> None:
    state = _run_with_mixed_targets(tmp_path)
    expected = _expected()
    expected["source_sha256"] = "c" * 64

    result = extract_group(group="A", state_dir=state, expected=expected)

    assert result["status"] == "stale"
    assert result["source_sha256"] == "a" * 64
    assert result["coverage"] == {
        "expected": 4,
        "translated": 0,
        "empty": 0,
        "missing": 0,
        "stale": 4,
        "error": 0,
    }
    rows = [row for chapter in result["chapters"] for row in chapter["segments"]]
    assert len(rows) == 4
    assert all(row["status"] == "stale" for row in rows)
    assert all(row["target"] is None for row in rows)
    assert all(
        row["reason"] == "source identity differs from the prepared snapshot" for row in rows
    )


def test_extract_keeps_an_absent_group_in_the_denominator(tmp_path: Path) -> None:
    result = extract_group(group="A", state_dir=tmp_path / "absent", expected=_expected())

    assert result["status"] == "missing"
    assert result["coverage"] == {
        "expected": 4,
        "translated": 0,
        "empty": 0,
        "missing": 4,
        "stale": 0,
        "error": 0,
    }
    assert len(result["chapters"][0]["segments"]) == 4


def test_expected_inventory_reads_chapters_without_changing_the_tree(tmp_path: Path) -> None:
    state = tmp_path / "prepared"
    run = state / "book.en" / "targets" / "zh"
    write_run(run, chapter=chapter_with([("alpha", None)]))
    before = hash_tree(run)

    expected = expected_inventory(run)

    assert expected["chapters"][0]["segments"][0]["id"] == segment_id("a" * 64, 0, 0)
    assert hash_tree(run) == before


def test_extract_all_keeps_lost_source_mismatch_and_corrupt_groups_in_the_denominator(
    tmp_path: Path,
) -> None:
    from tests.helpers import minimal_manifest
    from wenyi_eval.extract import extract_all

    root = tmp_path / "eval"
    source = root / "source" / "book.en.md"
    source.parent.mkdir(parents=True)
    source.write_text("Synthetic source.\n", encoding="utf-8")
    prepared = root / "_prepared" / "state"
    write_run(prepared, chapter=chapter_with([("alpha", None), ("beta", None)]), status="done")
    manifest = minimal_manifest(root, source=source)
    manifest["eval_id"] = "extract-all"
    manifest["book"] = "book.en"
    for group in ("A", "A2", "A_prime", "A_prime2", "B"):
        manifest["groups"][group]["state_dir"] = str(root / group / "state")

    good = root / "A" / "state" / "book.en" / "targets" / "zh"
    write_run(good, chapter=chapter_with([("alpha", "甲"), ("beta", "乙")]), status="done")

    stale = root / "A2" / "state" / "book.en" / "targets" / "zh"
    write_run(
        stale,
        chapter=chapter_with([("alpha", "甲")]),
        source_sha256="c" * 64,
        status="done",
    )

    corrupt = root / "A_prime" / "state" / "book.en" / "targets" / "zh"
    write_run(corrupt, chapter=chapter_with([("alpha", "甲")]), status="done")
    (corrupt / "chapters" / "ch0.json").write_text("{not json", encoding="utf-8")

    result = extract_all(root=root, manifest=manifest)

    assert result["groups"]["A"]["coverage"]["expected"] == 2
    assert result["groups"]["A"]["coverage"]["translated"] == 2
    assert result["groups"]["A2"]["status"] == "stale"
    assert result["groups"]["A2"]["coverage"]["stale"] == 2
    assert result["groups"]["A_prime"]["status"] == "error"
    assert result["groups"]["A_prime"]["coverage"]["error"] == 2
    assert result["groups"]["A_prime2"]["status"] == "missing"
    assert result["groups"]["A_prime2"]["coverage"]["missing"] == 2
    assert result["groups"]["B"]["status"] == "missing"
    for group in ("A", "A2", "A_prime", "A_prime2", "B"):
        assert result["groups"][group]["coverage"]["expected"] == 2


def test_review_summary_distinguishes_missing_invalid_and_available() -> None:
    assert _review_summary(None)["status"] == "missing"
    assert _review_summary([])["status"] == "invalid"
    available = _review_summary({"issues": [{"id": 1}], "reviewed_segments": 4, "complete": True})
    assert available == {
        "status": "available",
        "issues": 1,
        "reviewed_segments": 4,
        "complete": True,
    }
