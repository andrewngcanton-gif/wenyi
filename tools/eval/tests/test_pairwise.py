from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wenyi_eval.common import append_jsonl, write_json_atomic
from wenyi_eval.pairwise import (
    KEY_COLUMNS,
    RATINGS_COLUMNS,
    RatingError,
    export_pairwise,
    import_pairwise_ratings,
)


def _manifest() -> dict:
    return {"source": {"sha256": "a" * 64}, "llm": {"seed": 11}}


def _strata(root: Path) -> None:
    rows = [
        {
            "id": "seg:0",
            "source_text": "first source",
            "source_context_before": "",
            "source_context_after": "second source",
        },
        {
            "id": "seg:1",
            "source_text": "second source",
            "source_context_before": "first source",
            "source_context_after": "",
        },
    ]
    append_jsonl(root / "samples" / "strata.jsonl", rows)


def _extraction(*, missing_b: bool = False) -> dict:
    groups = {}
    for group in ("A", "A2", "A_prime", "A_prime2", "B"):
        segments = [
            {"id": "seg:0", "target": f"{group}-zero", "status": "translated"},
            {
                "id": "seg:1",
                "target": None if (missing_b and group == "B") else f"{group}-one",
                "status": "missing" if (missing_b and group == "B") else "translated",
            },
        ]
        groups[group] = {"chapters": [{"segments": segments}]}
    return {"groups": groups}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_import(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "preference", "omission", "addition", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_export_keeps_ratings_anonymous_and_key_separate(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _strata(root)

    preregistration = export_pairwise(root=root, manifest=_manifest(), extraction=_extraction())

    ratings = _read_csv(root / "samples" / "pairwise.csv")
    key = _read_csv(root / "samples" / "pairwise_key.csv")
    assert list(ratings[0]) == list(RATINGS_COLUMNS)
    assert list(key[0]) == list(KEY_COLUMNS)
    assert all("group" not in column.lower() for column in ratings[0])
    assert len(ratings) == 6
    assert len(key) == 6
    assert preregistration["expected_pairs"] == 6
    assert preregistration["denominator"] == 6
    assert preregistration["human_ratings_received"] == 0
    assert preregistration["blinding"] == "anonymous_side_labels_not_double_blind"
    assert {row["pair_id"] for row in ratings} == {row["pair_id"] for row in key}
    for row in ratings:
        assert row["preference"] == ""
        assert row["omission"] == ""
        assert row["addition"] == ""


def test_export_preserves_missing_pairs_instead_of_dropping_them(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _strata(root)

    preregistration = export_pairwise(
        root=root, manifest=_manifest(), extraction=_extraction(missing_b=True)
    )

    key = {row["pair_id"]: row for row in _read_csv(root / "samples" / "pairwise_key.csv")}
    assert preregistration["expected_pairs"] == 4
    assert preregistration["missing_pairs"] == 2
    assert preregistration["denominator"] == 4
    missing = [
        json.loads(line)
        for line in (root / "samples" / "pairwise_missing.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["comparison"] for row in missing} == {"A_vs_B", "Aprime_vs_B"}
    assert all(row["candidate_status"] == "missing" for row in missing)
    assert all(row["pair_id"] not in key for row in missing)


def test_blank_ratings_stay_missing_and_keep_the_denominator(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _strata(root)
    preregistration = export_pairwise(root=root, manifest=_manifest(), extraction=_extraction())
    ratings_path = root / "samples" / "pairwise.csv"

    summary = import_pairwise_ratings(root, ratings_path)

    assert summary["denominator"] == preregistration["denominator"] == 6
    assert summary["rated_rows"] == 0
    assert summary["blank_ratings"] == 6
    assert summary["missing_ids"] == 0
    assert summary["missing_pair_ids"] == []
    normalized = [
        json.loads(line)
        for line in (root / "samples" / "pairwise_ratings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["preference"] == "" for row in normalized)
    assert all(row["omission"] == "" for row in normalized)


def test_valid_ratings_are_normalized_without_imputing_blanks(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _strata(root)
    export_pairwise(root=root, manifest=_manifest(), extraction=_extraction())
    pair_ids = [row["pair_id"] for row in _read_csv(root / "samples" / "pairwise.csv")]

    _write_import(
        root / "samples" / "submitted.csv",
        [
            {
                "pair_id": pair_ids[0],
                "preference": "LEFT",
                "omission": "none",
                "addition": "both",
                "notes": "checked",
            },
            {
                "pair_id": pair_ids[1],
                "preference": "",
                "omission": "",
                "addition": "",
                "notes": "not reviewed",
            },
        ],
    )

    summary = import_pairwise_ratings(root, root / "samples" / "submitted.csv")

    assert summary["submitted_rows"] == 2
    assert summary["rated_rows"] == 1
    assert summary["blank_ratings"] == 1
    assert summary["missing_ids"] == 4
    assert summary["preference_counts"]["left"] == 1
    assert summary["addition_counts"]["both"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [{"pair_id": "unknown", "preference": "left", "omission": "", "addition": "", "notes": ""}],
        [
            {
                "pair_id": "PAIR:0",
                "preference": "maybe",
                "omission": "",
                "addition": "",
                "notes": "",
            }
        ],
        [
            {
                "pair_id": "PAIR:0",
                "preference": "left",
                "omission": "sometimes",
                "addition": "",
                "notes": "",
            }
        ],
        [
            {
                "pair_id": "PAIR:0",
                "preference": "left",
                "omission": "",
                "addition": "",
                "notes": "",
            },
            {
                "pair_id": "PAIR:0",
                "preference": "right",
                "omission": "",
                "addition": "",
                "notes": "",
            },
        ],
    ],
)
def test_import_rejects_unknown_invalid_or_duplicate_ratings(
    tmp_path: Path, rows: list[dict[str, str]]
) -> None:
    root = tmp_path / "eval"
    _strata(root)
    export_pairwise(root=root, manifest=_manifest(), extraction=_extraction())
    pair_ids = [row["pair_id"] for row in _read_csv(root / "samples" / "pairwise.csv")]
    resolved = [
        {**row, "pair_id": pair_ids[0] if row["pair_id"] == "PAIR:0" else row["pair_id"]}
        for row in rows
    ]
    _write_import(root / "samples" / "bad.csv", resolved)

    with pytest.raises(RatingError):
        import_pairwise_ratings(root, root / "samples" / "bad.csv")


def test_import_rejects_a_denominator_that_disagrees_with_exported_pairs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    _strata(root)
    export_pairwise(root=root, manifest=_manifest(), extraction=_extraction())
    preregistration_path = root / "samples" / "pairwise_preregistration.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    preregistration["denominator"] = 5
    write_json_atomic(preregistration_path, preregistration)

    with pytest.raises(RatingError, match="denominator"):
        import_pairwise_ratings(root, root / "samples" / "pairwise.csv")


def test_reflective_comparisons_use_opaque_blind_ids(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    _strata(root)
    extraction = {
        "groups": {
            "A_prime": {
                "chapters": [
                    {
                        "segments": [
                            {"id": "seg:0", "target": "prime-zero", "status": "translated"},
                            {"id": "seg:1", "target": "prime-one", "status": "translated"},
                        ]
                    }
                ]
            },
            "C": {
                "chapters": [
                    {
                        "segments": [
                            {"id": "seg:0", "target": "c-zero", "status": "translated"},
                            {"id": "seg:1", "target": "c-one", "status": "translated"},
                        ]
                    }
                ]
            },
        }
    }
    manifest = {
        "source": {"sha256": "a" * 64},
        "llm": {"seed": 5},
        "groups": {"A_prime": {}, "C": {}},
    }

    preregistration = export_pairwise(root=root, manifest=manifest, extraction=extraction)

    assert [item["id"] for item in preregistration["comparisons"]] == ["Aprime_vs_C"]
    ratings = _read_csv(root / "samples" / "pairwise.csv")
    key = _read_csv(root / "samples" / "pairwise_key.csv")
    assert len(ratings) == 2
    assert all(row["pair_id"].startswith("pair-") for row in ratings)
    assert "Aprime_vs_C" not in json.dumps(ratings)
    assert {row["comparison"] for row in key} == {"Aprime_vs_C"}
    assert {row["left_group"] for row in key} | {row["right_group"] for row in key} == {
        "A_prime",
        "C",
    }
