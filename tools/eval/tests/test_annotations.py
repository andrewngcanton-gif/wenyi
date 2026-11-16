from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wenyi_eval.common import append_jsonl
from wenyi_eval.edit_review import export_edit_review, import_edit_ratings
from wenyi_eval.pairwise import (
    RatingError,
    export_pairwise,
    import_pairwise_ratings,
)


def _strata(root: Path) -> None:
    append_jsonl(
        root / "samples" / "strata.jsonl",
        [
            {
                "id": "seg:0",
                "chapter_index": 0,
                "segment_index": 0,
                "source_text": "first source",
                "source_context_before": "",
                "source_context_after": "second source",
            },
            {
                "id": "seg:1",
                "chapter_index": 0,
                "segment_index": 1,
                "source_text": "second source",
                "source_context_before": "first source",
                "source_context_after": "",
            },
        ],
    )


def _extraction(*, changed: bool = False, before_polish: bool = True) -> dict:
    groups = {}
    for group in ("A_prime", "C"):
        groups[group] = {
            "chapters": [
                {
                    "index": 0,
                    "segments": [
                        {
                            "id": "seg:0",
                            "segment": 0,
                            "source": "first source",
                            "target": "changed" if changed and group == "C" else f"{group}-zero",
                            "target_before_polish": (
                                f"{group}-draft-zero" if before_polish else None
                            ),
                            "status": "translated",
                        },
                        {
                            "id": "seg:1",
                            "segment": 1,
                            "source": "second source",
                            "target": f"{group}-one",
                            "target_before_polish": (
                                f"{group}-draft-one" if before_polish else None
                            ),
                            "status": "translated",
                        },
                    ],
                }
            ]
        }
    return {"groups": groups}


def _manifest() -> dict:
    return {
        "source": {"sha256": "a" * 64},
        "llm": {"seed": 11},
        "book": "book",
        "groups": {"A_prime": {}, "C": {}},
    }


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_export_carries_hash_bound_mqm_and_severe_columns(tmp_path: Path) -> None:
    _strata(tmp_path)
    before = _extraction()
    export_pairwise(root=tmp_path, manifest=_manifest(), extraction=before)
    key = _read(tmp_path / "samples" / "pairwise_key.csv")
    ratings = _read(tmp_path / "samples" / "pairwise.csv")

    assert all(column in ratings[0] for column in ("left_mqm_errors", "right_severe"))
    assert key[0]["segment_id"] == "seg:0"
    assert key[0]["cluster"] == "chapter:0"
    assert key[0]["source_text_sha256"]
    assert key[0]["left_sha256"] and key[0]["right_sha256"]
    assert "Aprime_vs_C" not in json.dumps(ratings)
    assert all("group" not in column.lower() for column in ratings[0])
    assert "comparison" not in ratings[0]


def test_import_accepts_mqm_and_severe_and_preserves_blanks(tmp_path: Path) -> None:
    _strata(tmp_path)
    extraction = _extraction()
    export_pairwise(root=tmp_path, manifest=_manifest(), extraction=extraction)
    ratings = _read(tmp_path / "samples" / "pairwise.csv")
    ratings[0].update(
        {
            "preference": "left",
            "omission": "none",
            "addition": "none",
            "left_mqm_errors": "2",
            "right_mqm_errors": "0",
            "left_severe": "no",
            "right_severe": "no",
        }
    )
    _write(tmp_path / "samples" / "pairwise.csv", ratings, list(ratings[0]))

    summary = import_pairwise_ratings(
        tmp_path, tmp_path / "samples" / "pairwise.csv", extraction=extraction
    )

    assert summary["rated_rows"] == 1
    assert summary["mqm_assessed"] == 1
    assert summary["severe_assessed"] == 1
    assert summary["blank_ratings"] == 1
    assert summary["outputs_verified"] is True
    stored = _read_jsonl(tmp_path / "samples" / "pairwise_ratings.jsonl")
    assert stored[0]["left_mqm_errors"] == 2
    assert stored[1]["left_mqm_errors"] is None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_changed_output_invalidates_ratings(tmp_path: Path) -> None:
    _strata(tmp_path)
    export_pairwise(root=tmp_path, manifest=_manifest(), extraction=_extraction())

    with pytest.raises(RatingError, match="differs from the exported text"):
        import_pairwise_ratings(
            tmp_path,
            tmp_path / "samples" / "pairwise.csv",
            extraction=_extraction(changed=True),
        )


def test_edited_text_in_the_submitted_csv_is_rejected(tmp_path: Path) -> None:
    _strata(tmp_path)
    export_pairwise(root=tmp_path, manifest=_manifest(), extraction=_extraction())
    ratings = _read(tmp_path / "samples" / "pairwise.csv")
    ratings[0]["left_text"] = "tampered"
    _write(tmp_path / "samples" / "pairwise.csv", ratings, list(ratings[0]))

    with pytest.raises(RatingError, match="left text no longer matches"):
        import_pairwise_ratings(tmp_path, tmp_path / "samples" / "pairwise.csv")


def test_invalid_mqm_or_severe_is_rejected_atomically(tmp_path: Path) -> None:
    _strata(tmp_path)
    export_pairwise(root=tmp_path, manifest=_manifest(), extraction=_extraction())
    ratings = _read(tmp_path / "samples" / "pairwise.csv")
    ratings[0]["left_mqm_errors"] = "-1"
    ratings[0]["left_severe"] = "maybe"
    _write(tmp_path / "samples" / "pairwise.csv", ratings, list(ratings[0]))

    with pytest.raises(RatingError):
        import_pairwise_ratings(tmp_path, tmp_path / "samples" / "pairwise.csv")
    assert not (tmp_path / "samples" / "pairwise_import.json").exists()


def test_edit_packet_counts_only_actual_edits(tmp_path: Path) -> None:
    _strata(tmp_path)
    extraction = _extraction()
    extraction["groups"]["A_prime"]["chapters"][0]["segments"][0]["target"] = "same"
    extraction["groups"]["A_prime"]["chapters"][0]["segments"][0]["target_before_polish"] = "same"
    preregistration = export_edit_review(root=tmp_path, manifest=_manifest(), extraction=extraction)
    rows = _read(tmp_path / "samples" / "edit_review.csv")
    key = _read(tmp_path / "samples" / "edit_key.csv")

    assert preregistration["denominator"] == 2
    assert preregistration["groups"]["C"]["edited"] == 2
    assert len(rows) == 2
    assert all("C" not in row["edit_id"] for row in rows)
    assert key[0]["group"] == "C"
    assert all(row["worsened"] == "" for row in rows)


def test_edit_import_binds_before_and_after_hashes(tmp_path: Path) -> None:
    _strata(tmp_path)
    extraction = _extraction()
    export_edit_review(root=tmp_path, manifest=_manifest(), extraction=extraction)
    rows = _read(tmp_path / "samples" / "edit_review.csv")
    rows[0]["worsened"] = "yes"
    rows[0]["severity"] = "medium"
    rows[1]["worsened"] = "no"
    _write(tmp_path / "samples" / "edit_review.csv", rows, list(rows[0]))

    summary = import_edit_ratings(
        tmp_path, tmp_path / "samples" / "edit_review.csv", extraction=extraction
    )

    assert summary["assessed_rows"] == 2
    assert summary["worsened_counts"] == {"no": 1, "yes": 1}

    stale = _extraction()
    stale["groups"]["C"]["chapters"][0]["segments"][0]["target"] = "new output"
    with pytest.raises(RatingError, match="current target differs"):
        import_edit_ratings(tmp_path, tmp_path / "samples" / "edit_review.csv", extraction=stale)


def test_edit_import_rejects_unknown_ids_and_values(tmp_path: Path) -> None:
    _strata(tmp_path)
    extraction = _extraction()
    export_edit_review(root=tmp_path, manifest=_manifest(), extraction=extraction)
    rows = _read(tmp_path / "samples" / "edit_review.csv")
    rows[0]["edit_id"] = "unknown"
    _write(tmp_path / "samples" / "edit_review.csv", rows, list(rows[0]))
    with pytest.raises(RatingError, match="unknown edit_id"):
        import_edit_ratings(
            tmp_path, tmp_path / "samples" / "edit_review.csv", extraction=extraction
        )

    export_edit_review(root=tmp_path, manifest=_manifest(), extraction=extraction)
    rows = _read(tmp_path / "samples" / "edit_review.csv")
    rows[0]["worsened"] = "maybe"
    _write(tmp_path / "samples" / "edit_review.csv", rows, list(rows[0]))
    with pytest.raises(RatingError, match="worsened"):
        import_edit_ratings(
            tmp_path, tmp_path / "samples" / "edit_review.csv", extraction=extraction
        )
