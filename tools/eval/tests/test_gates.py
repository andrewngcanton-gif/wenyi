from __future__ import annotations

from pathlib import Path

from tests.helpers import minimal_manifest
from wenyi_eval.gates import FAILED, INCONCLUSIVE, PASSED, evaluate_gates
from wenyi_eval.protocol import draft_protocol


def _protocol(**overrides):
    data = draft_protocol()
    data["status"] = "registered"
    data["scope"] = "release"
    data["comparisons"] = [
        {
            "id": "Aprime_vs_C",
            "baseline": "A_prime",
            "candidate": "C",
            "role": "confirmatory",
        }
    ]
    data["sampling"].update(
        {
            "seed": 1234,
            "min_paragraphs": 200,
            "max_paragraphs": 300,
            "holdout": {"chapters": list(range(280, 300)), "verified": True},
        }
    )
    data["statistics"].update(
        {
            "resamples": 300,
            "seed": 77,
            "min_clusters": 10,
            "min_assessable_fraction": 0.95,
            "multiple_comparisons": "holm",
        }
    )
    data["decision"]["preference"] = {"lower_bound": 0.5, "mqm_max_mean_increase": 0.0}
    data.update(overrides)
    return data


def _manifest(root: Path) -> dict:
    manifest = minimal_manifest(root, groups=("A_prime", "C"))
    manifest["diagnostic"] = False
    manifest["fixture"]["synthetic"] = False
    manifest["llm"]["external_provider"] = True
    manifest["llm"]["seed"] = 1234
    manifest["sampling"] = {"min": 200, "max": 300}
    manifest["protocol"] = {"status": "valid", "content_sha256": "frozen"}
    for plan in manifest["groups"].values():
        plan["binding_status"] = "clean"
        plan["expected_head_sha"] = plan["git"]["head_sha"]
    return manifest


def _extraction() -> dict:
    groups = {}
    for group in ("A_prime", "C"):
        groups[group] = {
            "status": "ok",
            "coverage": {
                "expected": 300,
                "translated": 300,
                "empty": 0,
                "missing": 0,
                "stale": 0,
                "error": 0,
            },
            "review": {"status": "available", "issues": 0, "reviewed_segments": 300},
            "reflection": {"status": "not_enabled", "enabled": False},
            "chapters": [
                {
                    "index": chapter,
                    "segments": [
                        {
                            "id": f"src:{chapter}:0",
                            "source": f"source-{chapter}",
                            "target": f"{group}-{chapter}",
                            "target_before_polish": f"draft-{chapter}",
                            "status": "translated",
                            "segment": 0,
                        }
                    ],
                }
                for chapter in range(300)
            ],
        }
    return {
        "groups": groups,
        "expected": {
            "source_sha256": "a" * 64,
            "chapters": [{"index": n, "segments": [{"id": f"src:{n}:0"}]} for n in range(300)],
        },
    }


def _cost(*, baseline: float = 100.0, candidate: float = 140.0, known: bool = True) -> dict:
    rows = []
    for group, amount in (("A_prime", baseline), ("C", candidate)):
        rows.append(
            {
                "group": group,
                "category": "body",
                "operation": "subtotal",
                "cost_status": "known" if known else "unknown",
                "cost": amount if known else "unknown",
                "currency": "USD",
            }
        )
        rows.append(
            {
                "group": group,
                "category": "review",
                "operation": "subtotal",
                "cost_status": "known",
                "cost": 999.0,
                "currency": "USD",
            }
        )
    return {
        "price_status": "known" if known else "unknown",
        "price_date": "2026-11-16",
        "rows": rows,
        "groups": {},
        "aggregate_cost": {"status": "known", "amount": baseline + candidate},
        "aggregate_tokens": {"calls": 1},
    }


def _evidence(*, clusters: int = 300, preference: str = "right") -> dict:
    from wenyi_eval.pairwise import _side_hash

    key = {}
    ratings = []
    for chapter in range(clusters):
        pair_id = f"pair-{chapter}"
        key[pair_id] = {
            "pair_id": pair_id,
            "comparison": "Aprime_vs_C",
            "left_group": "A_prime",
            "right_group": "C",
            "left_sha256": _side_hash(f"A_prime-{chapter}"),
            "right_sha256": _side_hash(f"C-{chapter}"),
            "source_sha256": "a" * 64,
            "source_text_sha256": _side_hash(f"source-{chapter}"),
            "segment_id": f"src:{chapter}:0",
            "cluster": f"chapter:{chapter}",
            "holdout": str(chapter >= 280).lower(),
        }
        ratings.append(
            {
                "pair_id": pair_id,
                "preference": "left" if preference == "right" and chapter % 5 == 0 else preference,
                "omission": "both" if chapter % 20 == 0 else "none",
                "addition": "both" if chapter % 20 == 0 else "none",
                "left_mqm_errors": 1,
                "right_mqm_errors": 0,
                "left_severe": "no",
                "right_severe": "no",
                "notes": "",
                **{
                    field: key[pair_id][field]
                    for field in (
                        "left_sha256",
                        "right_sha256",
                        "source_sha256",
                        "source_text_sha256",
                    )
                },
            }
        )
    edit_ratings = [
        {
            "edit_id": f"edit-{chapter}",
            "group": "C",
            "cluster": f"chapter:{chapter}",
            "worsened": "yes" if chapter == 0 else "no",
            "severity": None,
            "notes": "",
            "source_sha256": "a" * 64,
            "source_text_sha256": _side_hash(f"source-{chapter}"),
            "before_sha256": _side_hash(f"draft-{chapter}"),
            "after_sha256": _side_hash(f"C-{chapter}"),
        }
        for chapter in range(clusters)
    ]
    return {
        "pairwise": {
            "preregistration": {"denominator": clusters},
            "import": {"rated_rows": clusters},
            "ratings": ratings,
            "key": key,
        },
        "edits": {
            "preregistration": {"groups": {"C": {"edited": clusters}}},
            "import": {"assessed_rows": clusters},
            "ratings": edit_ratings,
            "key": {
                row["edit_id"]: {
                    "edit_id": row["edit_id"],
                    "group": "C",
                    "cluster": row["cluster"],
                    "segment_id": f"src:{n}:0",
                    **{
                        field: row[field]
                        for field in (
                            "source_sha256",
                            "source_text_sha256",
                            "before_sha256",
                            "after_sha256",
                        )
                    },
                }
                for n, row in enumerate(edit_ratings)
            },
        },
    }


def _evaluate(
    root: Path,
    *,
    protocol: dict | None = None,
    evidence: dict | None = None,
    cost: dict | None = None,
    execution_ok: bool = True,
    protocol_late: bool = False,
    protocol_status: str = "valid",
) -> dict:
    data = protocol or _protocol()
    state = {"status": protocol_status, "protocol": data, "content_sha256": "frozen"}
    if protocol_status == "missing":
        state = {"status": "missing", "protocol": None, "content_sha256": None}
    return evaluate_gates(
        manifest=_manifest(root),
        extraction=_extraction(),
        cost=cost or _cost(),
        execution_ok=execution_ok,
        execution_reasons=[] if execution_ok else ["C.translate failed"],
        protocol_state=state,
        evidence=evidence or _evidence(),
        protocol_late=protocol_late,
        selection={
            "holdout_declared": data["sampling"]["holdout"]["chapters"],
            "holdout_chapters": list(range(280, 300)),
            "holdout_missing_chapters": [],
            "holdout_selected": 20,
        },
        annotation_binding={
            "status": "valid",
            "reason": "Validated at the file boundary in separate import tests.",
        },
    )


def test_every_required_gate_passes_on_complete_frozen_evidence(tmp_path: Path) -> None:
    result = _evaluate(tmp_path)

    assert result["quality_conclusion"] == PASSED, {
        name: gate["reason"] for name, gate in result["gates"].items() if gate["status"] != PASSED
    }
    assert result["empirical_evidence"] is True
    assert result["failed_required"] == []
    assert result["inconclusive_required"] == []
    assert 0.5 < result["gates"]["Aprime_vs_C.preference"]["details"]["lower"] < 1.0
    assert result["gates"]["Aprime_vs_C.cost"]["details"]["ratio"] == 1.4


def test_same_preference_estimate_with_too_few_clusters_is_inconclusive(tmp_path: Path) -> None:
    few = _evaluate(tmp_path, evidence=_evidence(clusters=5))
    many = _evaluate(tmp_path, evidence=_evidence(clusters=300))

    assert few["gates"]["Aprime_vs_C.preference"]["status"] == INCONCLUSIVE
    assert "independent clusters" in few["gates"]["Aprime_vs_C.preference"]["reason"]
    assert many["gates"]["Aprime_vs_C.preference"]["status"] == PASSED


def test_severe_new_mistranslation_blocks_with_failed(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["pairwise"]["ratings"][0]["right_severe"] = "yes"
    evidence["pairwise"]["ratings"][0]["left_severe"] = "no"

    result = _evaluate(tmp_path, evidence=evidence)

    assert result["gates"]["Aprime_vs_C.severe"]["status"] == FAILED
    assert result["quality_conclusion"] == FAILED


def test_severe_candidate_error_blocks_even_when_baseline_flag_is_unknown(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    evidence["pairwise"]["ratings"][0]["right_severe"] = "yes"
    evidence["pairwise"]["ratings"][0]["left_severe"] = ""

    result = _evaluate(tmp_path, evidence=evidence)

    assert result["gates"]["Aprime_vs_C.severe"]["status"] == FAILED
    assert result["quality_conclusion"] == FAILED


def test_mqm_deterioration_blocks_preference_and_overall(tmp_path: Path) -> None:
    evidence = _evidence()
    for row in evidence["pairwise"]["ratings"]:
        row["right_mqm_errors"] = 3

    result = _evaluate(tmp_path, evidence=evidence)

    assert result["gates"]["Aprime_vs_C.mqm"]["status"] == FAILED
    assert result["gates"]["Aprime_vs_C.preference"]["status"] == INCONCLUSIVE
    assert result["quality_conclusion"] == FAILED


def test_zero_edits_is_not_a_zero_of_zero_pass(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["edits"]["preregistration"]["groups"]["C"]["edited"] = 0
    evidence["edits"]["ratings"] = []

    result = _evaluate(tmp_path, evidence=evidence)

    assert result["gates"]["Aprime_vs_C.degradation"]["status"] == INCONCLUSIVE
    assert "0/0" in result["gates"]["Aprime_vs_C.degradation"]["reason"]


def test_fifty_clean_edits_cannot_establish_five_percent(tmp_path: Path) -> None:
    evidence = _evidence(clusters=50)
    for row in evidence["edits"]["ratings"]:
        row["worsened"] = "no"
    result = _evaluate(tmp_path, evidence=evidence)

    gate = result["gates"]["Aprime_vs_C.degradation"]
    assert gate["status"] == INCONCLUSIVE
    assert gate["details"]["upper"] is None


def test_exact_cost_boundary_passes_and_overrun_fails(tmp_path: Path) -> None:
    boundary = _evaluate(tmp_path, cost=_cost(candidate=140.0))
    over = _evaluate(tmp_path, cost=_cost(candidate=140.01))

    assert boundary["gates"]["Aprime_vs_C.cost"]["status"] == PASSED
    assert over["gates"]["Aprime_vs_C.cost"]["status"] == FAILED


def test_unknown_or_zero_baseline_cost_is_inconclusive(tmp_path: Path) -> None:
    unknown = _evaluate(tmp_path, cost=_cost(known=False))
    zero = _evaluate(tmp_path, cost=_cost(baseline=0.0))

    assert unknown["gates"]["Aprime_vs_C.cost"]["status"] == INCONCLUSIVE
    assert zero["gates"]["Aprime_vs_C.cost"]["status"] == INCONCLUSIVE
    assert "zero or unknown" in zero["gates"]["Aprime_vs_C.cost"]["reason"]


def test_pilot_scope_and_unverified_holdout_block_release_pass(tmp_path: Path) -> None:
    pilot = _protocol(scope="pilot")
    unverified = _protocol()
    unverified["sampling"]["holdout"]["verified"] = False

    assert _evaluate(tmp_path, protocol=pilot)["gates"]["holdout"]["status"] == INCONCLUSIVE
    assert _evaluate(tmp_path, protocol=unverified)["gates"]["holdout"]["status"] == INCONCLUSIVE


def test_multiple_confirmatory_comparisons_need_a_declared_correction(tmp_path: Path) -> None:
    data = _protocol()
    data["comparisons"].append(
        {
            "id": "Aprime_vs_D",
            "baseline": "A_prime",
            "candidate": "D",
            "role": "confirmatory",
        }
    )
    data["statistics"]["multiple_comparisons"] = "none"

    result = _evaluate(tmp_path, protocol=data)

    assert result["gates"]["multiple_comparisons"]["status"] == INCONCLUSIVE
    assert result["quality_conclusion"] == INCONCLUSIVE


def test_late_protocol_binding_and_missing_protocol_cannot_pass(tmp_path: Path) -> None:
    late = _evaluate(tmp_path, protocol_late=True)
    missing = _evaluate(tmp_path, protocol_status="missing")

    assert late["gates"]["protocol"]["status"] == FAILED
    assert late["quality_conclusion"] == FAILED
    assert missing["gates"]["protocol"]["status"] == INCONCLUSIVE
    assert missing["quality_conclusion"] == INCONCLUSIVE


def test_missing_cluster_assignment_is_inconclusive_not_paragraph_iid(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["pairwise"]["key"]["pair-0"]["cluster"] = ""

    result = _evaluate(tmp_path, evidence=evidence)

    gate = result["gates"]["Aprime_vs_C.preference"]
    assert gate["status"] == INCONCLUSIVE
    assert "cluster" in gate["reason"]
    assert result["quality_conclusion"] == INCONCLUSIVE


def test_execution_failure_is_failed_and_missing_protocol_is_not_a_pass(tmp_path: Path) -> None:
    failed = _evaluate(tmp_path, execution_ok=False)

    assert failed["gates"]["completeness"]["status"] == FAILED
    assert failed["quality_conclusion"] == FAILED


def _persist_evidence(root: Path, evidence: dict, protocol_hash: str) -> None:
    import csv

    from wenyi_eval.annotation_import import publish_imports
    from wenyi_eval.common import sha256_bytes, sha256_file, write_json_atomic

    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    evidence["pairwise"]["preregistration"].update(
        {
            "protocol_content_sha256": protocol_hash,
            "comparisons": [{"id": "Aprime_vs_C", "baseline": "A_prime", "candidate": "C"}],
        }
    )
    evidence["edits"]["preregistration"]["protocol_content_sha256"] = protocol_hash
    import json

    payloads = []
    for kind, section in (("pairwise", "pairwise"), ("edit", "edits")):
        data = evidence[section]
        keys = list(data["key"].values())
        with open(samples / f"{kind}_key.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(keys[0]))
            writer.writeheader()
            writer.writerows(keys)
        prereg_path = samples / f"{kind}_preregistration.json"
        write_json_atomic(prereg_path, data["preregistration"])
        ratings_bytes = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in data["ratings"]
        ).encode("utf-8")
        summary = {
            **data["import"],
            "key_sha256": sha256_file(samples / f"{kind}_key.csv"),
            "preregistration_sha256": sha256_file(prereg_path),
            "ratings_sha256": sha256_bytes(ratings_bytes),
        }
        payloads.append(
            {
                "ratings_path": samples / f"{kind}_ratings.jsonl",
                "summary_path": samples / f"{kind}_import.json",
                "rows": data["ratings"],
                "summary": summary,
            }
        )
    publish_imports(root, payloads)
    write_json_atomic(
        samples / "selection.json",
        {
            "holdout_declared": list(range(280, 300)),
            "holdout_chapters": list(range(280, 300)),
            "holdout_missing_chapters": [],
            "holdout_selected": 20,
        },
    )


def _execution() -> dict:
    terminals = {"preparation.prepare": {"status": "succeeded"}}
    for group in ("A_prime", "C"):
        for stage in ("translate", "status", "inspect"):
            terminal = {"status": "succeeded"}
            if stage == "translate":
                terminal["receipt_validation"] = {"status": "valid"}
            terminals[f"{group}.{stage}"] = terminal
    return {
        "terminals": terminals,
        "source_unchanged": True,
        "preparation_unchanged": True,
        "groups": {group: {"durations": {"translate": 1.0}} for group in ("A_prime", "C")},
    }


def _freeze(root: Path, data: dict) -> dict:
    import json

    from wenyi_eval.protocol import freeze_protocol

    path = root.parent / "registered-protocol.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return freeze_protocol(root, path)


def test_report_exposes_gate_by_gate_results_and_can_pass(tmp_path: Path) -> None:
    from wenyi_eval.report import write_report

    root = tmp_path / "eval"
    manifest = _manifest(root)
    manifest["created_at"] = "2099-01-01T00:00:00+00:00"
    binding = _freeze(root, _protocol())
    manifest["protocol"] = {
        "status": "valid",
        "content_sha256": binding["content_sha256"],
        "frozen_at": binding["frozen_at"],
    }
    _persist_evidence(root, _evidence(), binding["content_sha256"])

    report = write_report(
        root=root,
        manifest=manifest,
        extraction=_extraction(),
        execution=_execution(),
        cost=_cost(),
    )

    assert report["conclusion"] == "passed"
    assert report["conclusion_scope"] == "quality"
    assert report["quality_conclusion"] == "passed"
    assert report["empirical_evidence"] is True
    assert report["gates"]["Aprime_vs_C.preference"]["status"] == PASSED
    markdown = (root / "report" / "summary.md").read_text(encoding="utf-8")
    assert "## Quality gates" in markdown
    assert "Aprime_vs_C.preference" in markdown
    assert "Empirical quality: passed" in markdown
    human_csv = (root / "report" / "human_summary.csv").read_text(encoding="utf-8")
    assert "Aprime_vs_C.cost" in human_csv


def test_report_marks_a_late_protocol_binding_as_failed(tmp_path: Path) -> None:
    import os

    from wenyi_eval.report import write_report

    root = tmp_path / "eval"
    manifest = _manifest(root)
    manifest["created_at"] = "2099-01-01T00:00:00+00:00"
    binding = _freeze(root, _protocol())
    manifest["protocol"] = {
        "status": "valid",
        "content_sha256": binding["content_sha256"],
        "frozen_at": binding["frozen_at"],
    }
    execution_path = root / "records" / "execution.json"
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    execution_path.write_text("{}", encoding="utf-8")
    binding_path = root / "experiment_protocol.frozen.json"
    os.utime(
        binding_path, (execution_path.stat().st_atime + 60, execution_path.stat().st_mtime + 60)
    )
    _persist_evidence(root, _evidence(), binding["content_sha256"])

    report = write_report(
        root=root,
        manifest=manifest,
        extraction=_extraction(),
        execution=_execution(),
        cost=_cost(),
    )

    assert report["protocol"]["late_binding"] is True
    assert report["conclusion"] == "failed"
    assert report["gates"]["protocol"]["status"] == FAILED


def test_unavailable_pairs_remain_in_each_comparison_split_denominator(tmp_path: Path) -> None:
    from wenyi_eval.gates import _comparison_observations

    evidence = _evidence()["pairwise"]
    evidence["preregistration"]["comparison_denominators"] = {
        "Aprime_vs_C": {"main": 300, "holdout": 25},
        "Aprime_vs_D": {"main": 500, "holdout": 80},
    }
    comparison = _protocol()["comparisons"][0]
    assert _comparison_observations(comparison=comparison, evidence=evidence)["expected"] == 300
    assert (
        _comparison_observations(comparison=comparison, evidence=evidence, holdout=True)["expected"]
        == 25
    )


def test_legacy_unavailable_pairs_without_split_counts_cannot_pass(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["pairwise"]["preregistration"]["missing_pairs"] = 1
    result = _evaluate(tmp_path, evidence=evidence)
    assert result["gates"]["evidence"]["status"] == INCONCLUSIVE
    assert result["quality_conclusion"] != PASSED


def test_holdout_flags_must_match_actual_chapters(tmp_path: Path) -> None:
    for identity, value in (("pair-0", "true"), ("pair-280", "false")):
        evidence = _evidence()
        evidence["pairwise"]["key"][identity]["holdout"] = value
        result = _evaluate(tmp_path, evidence=evidence)
        assert result["gates"]["holdout"]["status"] == FAILED


def test_unsubmitted_pair_rows_do_not_shrink_quality_denominators(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["pairwise"]["ratings"] = evidence["pairwise"]["ratings"][:30]
    result = _evaluate(tmp_path, evidence=evidence)
    gate = result["gates"]["Aprime_vs_C.preference"]
    assert gate["status"] == INCONCLUSIVE
    assert gate["details"]["expected_pairs"] == 280
    assert gate["details"]["unsubmitted_pairs"] == 250


def test_skipped_preferences_do_not_count_as_assessable(tmp_path: Path) -> None:
    evidence = _evidence()
    for row in evidence["pairwise"]["ratings"][30:]:
        row["preference"] = "skip"
    result = _evaluate(tmp_path, evidence=evidence)
    assert result["gates"]["Aprime_vs_C.preference"]["status"] == INCONCLUSIVE
    assert result["gates"]["holdout"]["status"] == INCONCLUSIVE


def test_one_holdout_cluster_is_inconclusive_without_raising(tmp_path: Path) -> None:
    data = _protocol()
    data["sampling"]["holdout"]["chapters"] = [299]
    evidence = _evidence()
    for row in evidence["pairwise"]["key"].values():
        row["holdout"] = str(row["segment_id"] == "src:299:0").lower()
    from wenyi_eval.gates import _holdout_gate

    result = _holdout_gate(
        protocol=data,
        extraction=_extraction(),
        selection={"holdout_declared": [299], "holdout_chapters": [299], "holdout_selected": 1},
        evidence=evidence,
        alpha=0.05,
    )
    assert result["status"] == INCONCLUSIVE
    assert "at least two" in result["reason"]


def test_report_rejects_corrupt_import_summary_without_raising(tmp_path: Path) -> None:
    from wenyi_eval.report import write_report

    root = tmp_path / "eval"
    manifest = _manifest(root)
    manifest["created_at"] = "2099-01-01T00:00:00+00:00"
    binding = _freeze(root, _protocol())
    manifest["protocol"] = {
        "status": "valid",
        "content_sha256": binding["content_sha256"],
        "frozen_at": binding["frozen_at"],
    }
    _persist_evidence(root, _evidence(), binding["content_sha256"])
    (root / "samples" / "pairwise_import.json").write_text("[", encoding="utf-8")
    report = write_report(
        root=root, manifest=manifest, extraction=_extraction(), execution=_execution(), cost=_cost()
    )
    assert report["gates"]["annotations"]["status"] == FAILED
    assert report["quality_conclusion"] == FAILED


def test_stronger_comparison_cannot_mask_an_inconclusive_holdout(tmp_path: Path) -> None:
    from wenyi_eval.gates import _holdout_gate

    protocol = _protocol()
    protocol["comparisons"].append(
        {"id": "Aprime_vs_D", "baseline": "A_prime", "candidate": "D", "role": "confirmatory"}
    )
    evidence = _evidence()
    for chapter in range(300):
        identity = f"D-pair-{chapter}"
        evidence["pairwise"]["key"][identity] = {
            **evidence["pairwise"]["key"][f"pair-{chapter}"],
            "pair_id": identity,
            "comparison": "Aprime_vs_D",
            "right_group": "D",
        }
        evidence["pairwise"]["ratings"].append(
            {
                **evidence["pairwise"]["ratings"][chapter],
                "pair_id": identity,
                "preference": "left" if chapter % 5 < 2 else "right",
            }
        )
    result = _holdout_gate(
        protocol=protocol,
        extraction=_extraction(),
        selection={
            "holdout_declared": list(range(280, 300)),
            "holdout_chapters": list(range(280, 300)),
            "holdout_selected": 20,
        },
        evidence=evidence,
        alpha=0.025,
    )
    assert result["status"] == INCONCLUSIVE
    assert result["details"]["comparisons"]["Aprime_vs_C"]["status"] == PASSED
    assert result["details"]["comparisons"]["Aprime_vs_D"]["status"] == INCONCLUSIVE
