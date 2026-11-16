from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.helpers import minimal_manifest
from wenyi_eval.common import append_jsonl, write_json_atomic
from wenyi_eval.manifest import SUPPORTED_GROUPS
from wenyi_eval.report import write_report


def _extraction() -> dict:
    return {
        "groups": {
            group: {
                "status": "ok",
                "coverage": {
                    "expected": 2,
                    "translated": 2,
                    "empty": 0,
                    "missing": 0,
                    "stale": 0,
                },
                "review": {"status": "available", "issues": 0, "reviewed_segments": 2},
                "reflection": {"status": "unsupported"},
            }
            for group in SUPPORTED_GROUPS
        }
    }


def _execution(*, failed_group: str | None = None) -> dict:
    terminals = {"preparation.prepare": {"status": "succeeded"}}
    for group in SUPPORTED_GROUPS:
        for stage in ("translate", "status", "inspect"):
            status = "failed" if failed_group == group and stage == "translate" else "succeeded"
            terminal = {"status": status, "receipt_validation": {"status": "valid"}}
            if stage != "translate":
                terminal.pop("receipt_validation")
            terminals[f"{group}.{stage}"] = terminal
    return {
        "terminals": terminals,
        "source_unchanged": True,
        "preparation_unchanged": True,
        "groups": {group: {"durations": {"translate": 1.0}} for group in SUPPORTED_GROUPS},
    }


def _cost() -> dict:
    groups = {
        "preparation": {
            "tokens": {"calls": 1},
            "cost": {"status": "unknown", "amount": None},
        }
    }
    for group in SUPPORTED_GROUPS:
        groups[group] = {
            "tokens": {"calls": 2},
            "cost": {"status": "unknown", "amount": None},
        }
    return {
        "price_status": "unknown",
        "price_date": None,
        "groups": groups,
        "aggregate_cost": {"status": "unknown", "amount": None},
        "aggregate_tokens": {"calls": 11},
    }


def _report(root: Path, manifest: dict, execution: dict | None = None) -> dict:
    return write_report(
        root=root,
        manifest=manifest,
        extraction=_extraction(),
        execution=execution or _execution(),
        cost=_cost(),
    )


def _write_human_evidence(root: Path, preference: str, *, preferred: str = "candidate") -> None:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        samples / "pairwise_preregistration.json",
        {
            "comparisons": [{"id": "A_vs_Aprime", "baseline": "A", "candidate": "A_prime"}],
            "decision_rule": {
                "comparison": "A_vs_Aprime",
                "preferred": preferred,
                "min_rated": 1,
                "min_share": 0.5,
            },
        },
    )
    write_json_atomic(
        samples / "pairwise_import.json",
        {"denominator": 1, "rated_rows": 1},
    )
    with open(samples / "pairwise_key.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "comparison",
                "left_group",
                "right_group",
                "left_sha256",
                "right_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "p1",
                "comparison": "A_vs_Aprime",
                "left_group": "A_prime",
                "right_group": "A",
                "left_sha256": "0",
                "right_sha256": "1",
            }
        )
    append_jsonl(
        samples / "pairwise_ratings.jsonl",
        [
            {
                "pair_id": "p1",
                "preference": preference,
                "omission": "none",
                "addition": "none",
                "notes": "",
            }
        ],
    )


def test_diagnostic_report_is_inconclusive_and_publishes_all_summaries(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root, diagnostic=True)

    report = _report(root, manifest)

    assert report["execution_ok"] is True
    assert report["conclusion"] == "inconclusive"
    assert "Diagnostic offline run" in report["decision_reason"]
    assert (root / "report" / "report.json").is_file()
    assert (root / "report" / "group_summary.csv").is_file()
    assert (root / "report" / "human_summary.csv").is_file()
    markdown = (root / "report" / "summary.md").read_text(encoding="utf-8")
    assert "**Conclusion: inconclusive**" in markdown
    assert "Missing and stale keys are never silently dropped." in markdown
    assert "Unknown prices are reported as unknown, never zero." in markdown
    assert "A `current` artifact only means the record applies to the current post-alignment" in (
        markdown
    )
    assert "it is never a clean quality verdict" in markdown
    assert "not described as double blind" in markdown
    assert "unknown" in markdown


def test_execution_failure_is_reported_as_failed_even_for_a_diagnostic_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root, diagnostic=True)

    report = _report(root, manifest, execution=_execution(failed_group="A2"))

    assert report["execution_ok"] is False
    assert report["conclusion"] == "failed"
    assert any("A2.translate" in reason for reason in report["execution_reasons"])


@pytest.mark.parametrize(
    ("preference", "preferred", "expected"),
    [
        ("left", "candidate", "passed"),
        ("right", "candidate", "failed"),
        ("left", "baseline", "failed"),
    ],
)
def test_preregistered_human_rule_can_pass_or_fail(
    tmp_path: Path, preference: str, preferred: str, expected: str
) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root)
    _write_human_evidence(root, preference, preferred=preferred)

    report = _report(root, manifest)

    assert report["conclusion"] == expected
    assert "Preregistered rule met" in report["decision_reason"]


def test_missing_human_ratings_or_decision_rule_stay_inconclusive(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root)

    report = _report(root, manifest)
    assert report["conclusion"] == "inconclusive"
    assert "No pairwise ratings" in report["decision_reason"]

    _write_human_evidence(root, "left")
    (root / "samples" / "pairwise_preregistration.json").write_text(
        json.dumps(
            {"comparisons": [{"id": "A_vs_Aprime", "baseline": "A", "candidate": "A_prime"}]}
        ),
        encoding="utf-8",
    )
    report = _report(root, manifest)
    assert report["conclusion"] == "inconclusive"
    assert "No preregistered statistical decision rule" in report["decision_reason"]


def test_missing_or_stale_reflection_is_reported_and_never_clean(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root, diagnostic=True, groups=("C",))
    extraction = {
        "groups": {
            "C": {
                "status": "ok",
                "coverage": {
                    "expected": 2,
                    "translated": 2,
                    "empty": 0,
                    "missing": 0,
                    "stale": 0,
                    "error": 0,
                },
                "review": {"status": "available", "issues": 0, "reviewed_segments": 2},
                "reflection": {
                    "enabled": True,
                    "status": "stale",
                    "artifacts": {
                        "current": 0,
                        "stale": 1,
                        "invalid": 0,
                        "ambiguous": 0,
                        "write_failed_events": 0,
                    },
                    "segments": {
                        "expected": 2,
                        "current": 0,
                        "missing": 1,
                        "stale": 1,
                        "invalid": 0,
                        "ambiguous": 0,
                        "not_enabled": 0,
                    },
                    "adoption_counts": {},
                    "critique_counts": {},
                    "failure_counts": {},
                },
            }
        }
    }
    terminals = {"preparation.prepare": {"status": "succeeded"}}
    for stage in ("translate", "status", "inspect"):
        terminal = {"status": "succeeded", "receipt_validation": {"status": "valid"}}
        terminals[f"C.{stage}"] = terminal
    execution = {
        "terminals": terminals,
        "source_unchanged": True,
        "preparation_unchanged": True,
        "groups": {"C": {"durations": {"translate": 1.0}}},
    }
    cost = {
        "price_status": "unknown",
        "price_date": None,
        "groups": {
            "preparation": {"tokens": {"calls": 1}, "cost": {"status": "unknown"}},
            "C": {"tokens": {"calls": 2}, "cost": {"status": "unknown"}},
        },
        "aggregate_cost": {"status": "unknown", "amount": None},
        "aggregate_tokens": {"calls": 3},
    }

    report = write_report(
        root=root,
        manifest=manifest,
        extraction=extraction,
        execution=execution,
        cost=cost,
    )

    assert report["execution_ok"] is False
    assert report["conclusion"] == "failed"
    assert any(
        "without a current reflection artifact" in reason for reason in report["execution_reasons"]
    )
    markdown = (root / "report" / "summary.md").read_text(encoding="utf-8")
    assert "### Reflection evidence" in markdown
    assert "never counted as adoption" in markdown
    assert report["reflection"]["C"]["status"] == "stale"
