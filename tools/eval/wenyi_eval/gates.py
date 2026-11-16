"""Gate-by-gate evaluation of the frozen HLD 9.4 thresholds.

A workflow that ran correctly is not empirical quality. Every required gate must
pass on frozen code, a valid pre-registration and caller-reported human
annotations before the overall quality conclusion can be ``passed``. Missing
evidence is ``inconclusive``; a demonstrated violation or tampered protocol is
``failed``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .common import read_json, read_jsonl
from .protocol import confirmatory_comparisons, correction_alpha
from .stats import (
    StatsError,
    paired_mean_difference,
    paired_rate_difference,
    paired_worsened_rate,
    preference_interval,
)

PASSED = "passed"
FAILED = "failed"
INCONCLUSIVE = "inconclusive"

METRICS = ("preference", "mqm", "omission", "addition", "severe", "degradation", "cost")


def _gate(
    status: str,
    reason: str,
    *,
    required: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "required": required,
        "details": details or {},
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _cluster_fallback(segment_id: str | None) -> str:
    parts = (segment_id or "").split(":")
    if len(parts) >= 3:
        return f"chapter:{parts[-2]}"
    return "chapter:unknown"


def load_human_evidence(root: Path) -> dict[str, Any]:
    """Load pairwise and edit annotations without inventing any missing row."""
    samples = Path(root) / "samples"
    pairwise_key = {
        row["pair_id"]: row for row in _csv_rows(samples / "pairwise_key.csv") if row.get("pair_id")
    }
    for row in pairwise_key.values():
        row.setdefault("cluster", "")
    edit_key = {
        row["edit_id"]: row for row in _csv_rows(samples / "edit_key.csv") if row.get("edit_id")
    }
    return {
        "pairwise": {
            "preregistration": _optional_json(samples / "pairwise_preregistration.json"),
            "import": _optional_json(samples / "pairwise_import.json"),
            "ratings": _optional_jsonl(samples / "pairwise_ratings.jsonl"),
            "key": pairwise_key,
        },
        "edits": {
            "preregistration": _optional_json(samples / "edit_preregistration.json"),
            "import": _optional_json(samples / "edit_import.json"),
            "ratings": _optional_jsonl(samples / "edit_ratings.jsonl"),
            "key": edit_key,
        },
    }


def _side(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    return text or None


def _side_flag(row: dict[str, Any], base: str, side: str) -> Any:
    return row.get(f"{side}_{base}")


def _comparison_observations(
    *,
    comparison: dict[str, Any],
    evidence: dict[str, Any],
    holdout: bool = False,
) -> dict[str, Any]:
    candidate = comparison["candidate"]
    key_rows = evidence["key"]
    expected = {
        pair_id: row
        for pair_id, row in key_rows.items()
        if row.get("comparison") == comparison["id"]
        and (str(row.get("holdout", "")).lower() == "true") == holdout
    }
    preference: list[dict[str, Any]] = []
    omission: list[dict[str, Any]] = []
    addition: list[dict[str, Any]] = []
    mqm: list[dict[str, Any]] = []
    severe: list[dict[str, Any]] = []
    for row in evidence["ratings"]:
        pair_id = row.get("pair_id")
        key_row = expected.get(pair_id)
        if key_row is None:
            continue
        if key_row.get("left_group") == candidate:
            candidate_side, baseline_side = "left", "right"
        else:
            candidate_side, baseline_side = "right", "left"
        cluster = key_row.get("cluster") or _cluster_fallback(key_row.get("segment_id"))
        choice = _side(row.get("preference"))
        preference.append(
            {
                "cluster": cluster,
                "decisive": choice in {"left", "right"},
                "candidate_win": choice == candidate_side,
                "tie": choice == "tie",
                "skipped": choice == "skip",
                "missing": choice is None,
            }
        )
        for label, target in (("omission", omission), ("addition", addition)):
            value = _side(row.get(label))
            if value is None:
                continue
            if value not in {"none", "left", "right", "both"}:
                continue
            target.append(
                {
                    "cluster": cluster,
                    "candidate": 1.0 if value in {candidate_side, "both"} else 0.0,
                    "baseline": 1.0 if value in {baseline_side, "both"} else 0.0,
                }
            )
        candidate_mqm = _side_flag(row, "mqm_errors", candidate_side)
        baseline_mqm = _side_flag(row, "mqm_errors", baseline_side)
        if isinstance(candidate_mqm, int) and isinstance(baseline_mqm, int):
            mqm.append({"cluster": cluster, "candidate": candidate_mqm, "baseline": baseline_mqm})
        candidate_severe = _side_flag(row, "severe", candidate_side)
        baseline_severe = _side_flag(row, "severe", baseline_side)
        if candidate_severe in {"yes", "no"}:
            severe.append(
                {
                    "cluster": cluster,
                    "candidate": candidate_severe,
                    "baseline": baseline_severe if baseline_severe in {"yes", "no"} else None,
                }
            )
    preregistration = evidence.get("preregistration") or {}
    counts = (preregistration.get("comparison_denominators") or {}).get(comparison["id"]) or {}
    declared_count = counts.get("holdout" if holdout else "main")
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        declared_count = 0 if holdout else preregistration.get("sampled_segments", 0)
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        declared_count = 0
    expected_count = max(len(expected), declared_count)
    return {
        "preference": preference,
        "omission": omission,
        "addition": addition,
        "mqm": mqm,
        "severe": severe,
        "expected": expected_count,
    }


def _preference_gate(
    *,
    observations: list[dict[str, Any]],
    expected: int,
    protocol: dict[str, Any],
    alpha: float,
    mqm_gate: dict[str, Any],
) -> dict[str, Any]:
    statistics = protocol["statistics"]
    threshold = float(protocol["decision"]["preference"]["lower_bound"])
    details = {
        "expected_pairs": expected,
        "available_pairs": len(observations),
        "unsubmitted_pairs": max(0, expected - len(observations)),
    }
    assessed = sum(1 for row in observations if row["decisive"] or row.get("tie"))
    coverage = (assessed / expected) if expected > 0 else 0.0
    details["assessed_preferences"] = assessed
    details["coverage"] = coverage
    if expected <= 0:
        return _gate(
            INCONCLUSIVE,
            "The frozen packet has no expected pairwise denominator.",
            details=details,
        )
    if coverage < statistics["min_assessable_fraction"]:
        return _gate(
            INCONCLUSIVE,
            f"Only {assessed} of {expected} expected pairs have a preference assessment; "
            f"the preregistered {statistics['min_assessable_fraction']:.0%} coverage is not met.",
            details=details,
        )
    try:
        result = preference_interval(
            observations,
            seed=statistics["seed"],
            resamples=statistics["resamples"],
            confidence=statistics["confidence"],
            alpha=alpha,
        )
    except StatsError as error:
        return _gate(INCONCLUSIVE, f"Preference statistics unavailable: {error}", details=details)
    details.update(result)
    details["missing_ratings"] = max(
        0,
        expected
        - (
            (result.get("decisive") or 0) + (result.get("ties") or 0) + (result.get("skipped") or 0)
        ),
    )
    if result["clusters"] < statistics["min_clusters"]:
        return _gate(
            INCONCLUSIVE,
            f"Only {result['clusters']} independent clusters; "
            f"{statistics['min_clusters']} required.",
            details=details,
        )
    if not result.get("available"):
        return _gate(
            INCONCLUSIVE,
            result.get("reason") or "Preference interval is unavailable.",
            details=details,
        )
    if result["lower"] > threshold:
        if mqm_gate["status"] != PASSED:
            return _gate(
                INCONCLUSIVE,
                "Preference interval clears the threshold, but the preregistered human MQM "
                f"non-deterioration rule is {mqm_gate['status']}: {mqm_gate['reason']}",
                details=details,
            )
        return _gate(
            PASSED,
            f"Candidate non-tie win share {result['point']:.3f}; cluster-paired two-sided "
            f"{statistics['confidence']:.0%} lower bound {result['lower']:.3f} > {threshold:.2f}; "
            "human MQM non-deterioration satisfied.",
            details=details,
        )
    if result["upper"] < threshold:
        return _gate(
            FAILED,
            f"Candidate win share upper bound {result['upper']:.3f} is below {threshold:.2f}.",
            details=details,
        )
    return _gate(
        INCONCLUSIVE,
        f"Candidate win share {result['point']:.3f}; interval "
        f"[{result['lower']:.3f}, {result['upper']:.3f}] does not clear {threshold:.2f}.",
        details=details,
    )


def _mqm_gate(
    *,
    observations: list[dict[str, Any]],
    rated: int,
    expected: int,
    protocol: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    statistics = protocol["statistics"]
    maximum = float(protocol["decision"]["preference"]["mqm_max_mean_increase"])
    details = {
        "expected_pairs": expected,
        "available_pairs": len(observations),
        "rated_pairs": rated,
        "assessed_pairs": len(observations),
        "missing_annotations": max(0, expected - len(observations)),
    }
    if not observations:
        return _gate(
            INCONCLUSIVE,
            "No pair has human MQM errors for both sides; non-deterioration is unknown.",
            details=details,
        )
    coverage = (len(observations) / expected) if expected > 0 else 0.0
    details["coverage"] = coverage
    if coverage < statistics["min_assessable_fraction"]:
        return _gate(
            INCONCLUSIVE,
            f"Only {len(observations)} of {expected} expected pairs have MQM evidence; "
            f"{statistics['min_assessable_fraction']:.0%} are required.",
            details=details,
        )
    try:
        result = paired_mean_difference(
            observations,
            seed=statistics["seed"],
            resamples=statistics["resamples"],
            confidence=statistics["confidence"],
            alpha=alpha,
        )
    except StatsError as error:
        return _gate(INCONCLUSIVE, f"MQM statistics unavailable: {error}", details=details)
    details.update(result)
    if result["clusters"] < statistics["min_clusters"] or not result.get("available"):
        return _gate(
            INCONCLUSIVE,
            result.get("reason") or "Insufficient independent clusters for MQM.",
            details=details,
        )
    if result["upper"] <= maximum:
        return _gate(
            PASSED,
            f"Human MQM mean increase upper bound {result['upper']:.3f} <= {maximum:.3f}.",
            details=details,
        )
    if result["point"] > maximum:
        return _gate(
            FAILED,
            f"Human MQM mean increase {result['point']:.3f} exceeds {maximum:.3f}.",
            details=details,
        )
    return _gate(
        INCONCLUSIVE,
        f"Human MQM mean increase {result['point']:.3f}; upper bound {result['upper']:.3f} "
        f"does not establish <= {maximum:.3f}.",
        details=details,
    )


def _defect_gate(
    *,
    label: str,
    observations: list[dict[str, Any]],
    rated: int,
    expected: int,
    protocol: dict[str, Any],
    alpha: float,
    threshold: float,
) -> dict[str, Any]:
    statistics = protocol["statistics"]
    details = {
        "expected_pairs": expected,
        "available_pairs": len(observations),
        "rated_pairs": rated,
        "assessed_pairs": len(observations),
        "missing_annotations": max(0, expected - len(observations)),
    }
    if not observations:
        return _gate(
            INCONCLUSIVE,
            f"No pair has a human {label} assessment on both sides.",
            details=details,
        )
    coverage = (len(observations) / expected) if expected > 0 else 0.0
    details["coverage"] = coverage
    if coverage < statistics["min_assessable_fraction"]:
        return _gate(
            INCONCLUSIVE,
            f"Only {len(observations)} of {expected} expected pairs assess {label}; the "
            f"preregistered {statistics['min_assessable_fraction']:.0%} coverage is not met.",
            details=details,
        )
    try:
        result = paired_rate_difference(
            observations,
            seed=statistics["seed"],
            resamples=statistics["resamples"],
            confidence=statistics["confidence"],
            alpha=alpha,
        )
    except StatsError as error:
        return _gate(INCONCLUSIVE, f"{label} statistics unavailable: {error}", details=details)
    details.update(result)
    if result["clusters"] < statistics["min_clusters"] or not result.get("available"):
        return _gate(
            INCONCLUSIVE,
            result.get("reason") or "Insufficient independent clusters.",
            details=details,
        )
    if result["upper"] <= threshold:
        return _gate(
            PASSED,
            f"Candidate-minus-baseline {label} rate upper bound {result['upper']:.4f} <= "
            f"{threshold:.4f} over {result['assessable']} paired segments.",
            details=details,
        )
    if result["point"] > threshold:
        return _gate(
            FAILED,
            f"Candidate-minus-baseline {label} rate {result['point']:.4f} exceeds {threshold:.4f}.",
            details=details,
        )
    return _gate(
        INCONCLUSIVE,
        f"{label} difference {result['point']:.4f}; one-sided upper bound "
        f"{result['upper']:.4f} does not establish <= {threshold:.4f}.",
        details=details,
    )


def _severe_gate(
    *,
    observations: list[dict[str, Any]],
    rated: int,
    expected: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    details = {
        "expected_pairs": expected,
        "available_pairs": len(observations),
        "rated_pairs": rated,
        "assessed_pairs": len(observations),
    }
    fully_assessed = sum(1 for row in observations if row["baseline"] in {"yes", "no"})
    details["fully_assessed_pairs"] = fully_assessed
    details["missing_annotations"] = max(0, expected - fully_assessed)
    if not observations:
        return _gate(
            INCONCLUSIVE,
            "No rated pair records severe-mistranslation evidence for the candidate.",
            details=details,
        )
    introduced = [
        row for row in observations if row["candidate"] == "yes" and row["baseline"] != "yes"
    ]
    details["newly_introduced"] = len(introduced)
    details["baseline_severe"] = sum(1 for row in observations if row["baseline"] == "yes")
    if introduced:
        return _gate(
            FAILED,
            f"{len(introduced)} human-confirmed severe newly introduced mistranslation(s) "
            "block a quality pass.",
            details=details,
        )
    minimum = protocol["statistics"]["min_assessable_fraction"]
    coverage = (fully_assessed / expected) if expected > 0 else 0.0
    details["coverage"] = coverage
    if coverage < minimum:
        return _gate(
            INCONCLUSIVE,
            f"Only {fully_assessed} of {expected} expected pairs assess severe errors on both "
            f"sides; {minimum:.0%} coverage is required.",
            details=details,
        )
    return _gate(
        PASSED,
        f"No human-confirmed severe newly introduced mistranslation across "
        f"{fully_assessed} fully assessed pairs.",
        details=details,
    )


def _degradation_gate(
    *,
    candidate: str,
    evidence: dict[str, Any],
    protocol: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    statistics = protocol["statistics"]
    maximum = float(protocol["decision"]["degradation_max"])
    preregistration = evidence.get("preregistration") or {}
    group_counts = (preregistration.get("groups") or {}).get(candidate) or {}
    denominator = group_counts.get("edited")
    rows = [
        {
            "cluster": row.get("cluster") or "chapter:unknown",
            "worsened": 1 if row.get("worsened") == "yes" else 0,
        }
        for row in evidence.get("ratings", [])
        if row.get("group") == candidate and row.get("worsened") in {"yes", "no"}
    ]
    unit_prefix = f"{protocol['sampling']['clusters']}:"
    if any(
        not row["cluster"].startswith(unit_prefix) or row["cluster"].endswith(":unknown")
        for row in rows
    ):
        return _gate(
            INCONCLUSIVE,
            "The edit packet has an invalid cluster assignment for the frozen design.",
            details={"edited_denominator": denominator},
        )
    details = {
        "edited_denominator": denominator,
        "assessed_edits": len(rows),
        "missing_edits": (
            None if not isinstance(denominator, int) else max(0, denominator - len(rows))
        ),
    }
    if not isinstance(denominator, int):
        return _gate(
            INCONCLUSIVE,
            "No preregistered edited-segment denominator exists for this candidate.",
            details=details,
        )
    if denominator == 0:
        return _gate(
            INCONCLUSIVE,
            "The candidate edited no segments relative to its own pre-polish draft; "
            "degradation cannot be 0/0 pass.",
            details=details,
        )
    if not rows:
        return _gate(
            INCONCLUSIVE,
            "No assessable edited samples; degradation is unknown, not a pass.",
            details=details,
        )
    if len(rows) / denominator < statistics["min_assessable_fraction"]:
        return _gate(
            INCONCLUSIVE,
            f"Only {len(rows)} of {denominator} edited segments are assessed; "
            f"{statistics['min_assessable_fraction']:.0%} coverage is required.",
            details=details,
        )
    try:
        result = paired_worsened_rate(
            rows,
            seed=statistics["seed"],
            resamples=statistics["resamples"],
            confidence=statistics["confidence"],
            alpha=alpha,
        )
    except StatsError as error:
        return _gate(INCONCLUSIVE, f"Degradation statistics unavailable: {error}", details=details)
    details.update(result)
    if result["clusters"] < statistics["min_clusters"] or not result.get("available"):
        return _gate(
            INCONCLUSIVE,
            result.get("reason") or "Insufficient independent clusters for degradation.",
            details=details,
        )
    if result["upper"] <= maximum:
        return _gate(
            PASSED,
            f"Human-confirmed worsening upper bound {result['upper']:.4f} <= {maximum:.2f} "
            f"over {result['assessable']} edited segments.",
            details=details,
        )
    if result["point"] > maximum:
        return _gate(
            FAILED,
            f"Human-confirmed worsening rate {result['point']:.4f} exceeds {maximum:.2f}.",
            details=details,
        )
    return _gate(
        INCONCLUSIVE,
        f"Worsening rate {result['point']:.4f}; one-sided upper bound {result['upper']:.4f} "
        f"does not establish <= {maximum:.2f}.",
        details=details,
    )


def _body_subtotal(cost: dict[str, Any], group: str) -> dict[str, Any] | None:
    for row in cost.get("rows", []):
        if (
            row.get("group") == group
            and row.get("category") == "body"
            and row.get("operation") == "subtotal"
        ):
            return row
    return None


def _coverage_complete(extraction: dict[str, Any], group: str) -> bool:
    data = extraction.get("groups", {}).get(group) or {}
    if data.get("status") != "ok":
        return False
    coverage = data.get("coverage") or {}
    expected = coverage.get("expected")
    if not isinstance(expected, int) or expected <= 0:
        return False
    if any(coverage.get(key) for key in ("missing", "stale", "error")):
        return False
    return coverage.get("translated", 0) + coverage.get("empty", 0) == expected


def _cost_gate(
    *,
    candidate: str,
    baseline: str,
    protocol: dict[str, Any],
    cost: dict[str, Any],
    extraction: dict[str, Any],
) -> dict[str, Any]:
    maximum = float(protocol["decision"]["cost_max_ratio"])
    details = {
        "baseline_group": baseline,
        "candidate_group": candidate,
        "price_status": cost.get("price_status"),
        "price_date": cost.get("price_date"),
    }
    if cost.get("price_status") != "known":
        return _gate(
            INCONCLUSIVE,
            "Provider prices are not known; token counts are never priced as money.",
            details=details,
        )
    baseline_row = _body_subtotal(cost, baseline)
    candidate_row = _body_subtotal(cost, candidate)
    details["baseline_body"] = baseline_row
    details["candidate_body"] = candidate_row
    if baseline_row is None or candidate_row is None:
        return _gate(
            INCONCLUSIVE,
            "A body-cost subtotal is missing; preparation and Review are not substituted.",
            details=details,
        )
    if baseline_row.get("cost_status") != "known" or candidate_row.get("cost_status") != "known":
        return _gate(INCONCLUSIVE, "Body cost is unknown for at least one side.", details=details)
    if baseline_row.get("currency") != candidate_row.get("currency"):
        return _gate(INCONCLUSIVE, "Body costs use different currencies.", details=details)
    baseline_amount = baseline_row.get("cost")
    candidate_amount = candidate_row.get("cost")
    if (
        isinstance(baseline_amount, bool)
        or not isinstance(baseline_amount, (int, float))
        or baseline_amount <= 0
    ):
        return _gate(
            INCONCLUSIVE,
            "Baseline body cost is zero or unknown; a ratio cannot pass.",
            details=details,
        )
    if (
        isinstance(candidate_amount, bool)
        or not isinstance(candidate_amount, (int, float))
        or candidate_amount <= 0
    ):
        return _gate(INCONCLUSIVE, "Candidate body cost is zero or unknown.", details=details)
    if not (_coverage_complete(extraction, baseline) and _coverage_complete(extraction, candidate)):
        return _gate(
            INCONCLUSIVE,
            "Workload is incomplete; cost is not compared on the same complete workload.",
            details=details,
        )
    baseline_expected = (extraction["groups"][baseline].get("coverage") or {}).get("expected")
    candidate_expected = (extraction["groups"][candidate].get("coverage") or {}).get("expected")
    if baseline_expected != candidate_expected:
        return _gate(
            INCONCLUSIVE,
            "The two groups do not cover the same expected segment workload.",
            details=details,
        )
    ratio = float(candidate_amount) / float(baseline_amount)
    details.update(
        {
            "ratio": ratio,
            "threshold": maximum,
            "currency": baseline_row.get("currency"),
        }
    )
    if ratio <= maximum:
        return _gate(
            PASSED,
            f"Candidate/baseline body cost ratio {ratio:.4f} <= {maximum:.2f}.",
            details=details,
        )
    return _gate(
        FAILED,
        f"Candidate/baseline body cost ratio {ratio:.4f} exceeds {maximum:.2f}.",
        details=details,
    )


def _protocol_gate(
    *,
    protocol_state: dict[str, Any],
    manifest: dict[str, Any],
    protocol_late: bool,
    protocol_reasons: list[str],
) -> dict[str, Any]:
    details = {
        "status": protocol_state.get("status"),
        "content_sha256": protocol_state.get("content_sha256"),
        "late_binding": protocol_late,
        "reasons": protocol_reasons,
    }
    if protocol_late:
        return _gate(
            FAILED,
            "The protocol binding was written after the evaluation outputs; late "
            "pre-registration is invalid.",
            details=details,
        )
    if protocol_state.get("status") == "missing":
        return _gate(INCONCLUSIVE, protocol_state.get("reason"), details=details)
    if protocol_state.get("status") != "valid":
        return _gate(FAILED, protocol_state.get("reason"), details=details)
    if protocol_reasons:
        return _gate(FAILED, "; ".join(protocol_reasons), details=details)
    return _gate(PASSED, protocol_state.get("reason"), details=details)


def _evidence_gate(
    *,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "diagnostic": manifest.get("diagnostic"),
        "synthetic_fixture": (manifest.get("fixture") or {}).get("synthetic"),
        "external_provider": (manifest.get("llm") or {}).get("external_provider"),
        "bindings": {
            group: (plan or {}).get("binding_status")
            for group, plan in (manifest.get("groups") or {}).items()
        },
    }
    reasons = []
    if manifest.get("diagnostic"):
        reasons.append("the run is marked diagnostic")
    if (manifest.get("fixture") or {}).get("synthetic"):
        reasons.append("the fixture is synthetic")
    if not (manifest.get("llm") or {}).get("external_provider"):
        reasons.append("the external model boundary was replaced")
    for group, plan in (manifest.get("groups") or {}).items():
        if (plan or {}).get("binding_status") != "clean":
            reasons.append(f"{group} code binding is {(plan or {}).get('binding_status')}")
        expected = (plan or {}).get("expected_head_sha")
        if expected == "unfrozen":
            reasons.append(f"{group} code SHA is unfrozen")
    required_groups = {item["baseline"] for item in confirmatory_comparisons(protocol)} | {
        item["candidate"] for item in confirmatory_comparisons(protocol)
    }
    missing = required_groups - set(manifest.get("groups") or {})
    if missing:
        reasons.append(f"protocol groups are not present: {sorted(missing)}")
    preregistration = evidence["pairwise"].get("preregistration") or {}
    counts = preregistration.get("comparison_denominators") or {}
    if preregistration.get("missing_pairs") and any(
        not isinstance(counts.get(item["id"]), dict)
        or any(not isinstance(counts[item["id"]].get(split), int) for split in ("main", "holdout"))
        for item in confirmatory_comparisons(protocol)
    ):
        reasons.append("unavailable pairs have no planned comparison/holdout denominators")
    details["reasons"] = reasons
    if reasons:
        return _gate(
            INCONCLUSIVE,
            "Frozen, non-synthetic evidence is not available: " + "; ".join(reasons),
            details=details,
        )
    return _gate(
        PASSED,
        "Frozen non-diagnostic code and real provider outputs support gate evaluation; "
        "human origin is caller-reported, not identity-verified.",
        details=details,
    )


def _completeness_gate(*, execution_ok: bool, execution_reasons: list[str]) -> dict[str, Any]:
    details = {"execution_reasons": execution_reasons}
    if not execution_ok:
        return _gate(
            FAILED,
            "Execution, coverage or identity checks failed: " + "; ".join(execution_reasons),
            details=details,
        )
    return _gate(
        PASSED,
        "Preparation, translation, coverage, receipts and source identity checks passed.",
        details=details,
    )


def _holdout_gate(
    *,
    protocol: dict[str, Any],
    extraction: dict[str, Any],
    selection: dict[str, Any] | None,
    evidence: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    holdout = protocol["sampling"]["holdout"]
    declared = list(holdout.get("chapters") or [])
    details = {
        "scope": protocol.get("scope"),
        "declared_chapters": declared,
        "declared_verified_boolean": holdout.get("verified"),
        "selection": {
            key: (selection or {}).get(key)
            for key in (
                "holdout_declared",
                "holdout_chapters",
                "holdout_missing_chapters",
                "holdout_selected",
            )
        },
    }
    if protocol.get("scope") != "release":
        return _gate(
            INCONCLUSIVE,
            "The protocol scope is a pilot; a 200-300 paragraph pilot is not a general "
            "release-quality result.",
            details=details,
        )
    if not declared:
        return _gate(
            INCONCLUSIVE,
            "No final holdout chapters were declared before results.",
            details=details,
        )
    if not holdout.get("verified"):
        return _gate(
            INCONCLUSIVE, "The caller has not declared the holdout verified.", details=details
        )
    expected_chapters = {
        chapter.get("index")
        for chapter in ((extraction.get("expected") or {}).get("chapters") or [])
    }
    if not expected_chapters or not set(declared) <= expected_chapters:
        return _gate(
            INCONCLUSIVE,
            "Declared holdout chapters do not exist in the prepared source inventory.",
            details=details,
        )
    if selection is None:
        return _gate(
            INCONCLUSIVE,
            "No sampling selection records the pre-result holdout split.",
            details=details,
        )
    if list(selection.get("holdout_declared") or []) != declared:
        return _gate(
            INCONCLUSIVE,
            "The sampled holdout split does not match the frozen protocol.",
            details=details,
        )
    if selection.get("holdout_missing_chapters"):
        return _gate(
            INCONCLUSIVE,
            "Some declared holdout chapters have no eligible observations.",
            details=details,
        )
    if set(selection.get("holdout_chapters") or []) != set(declared):
        return _gate(
            INCONCLUSIVE, "Selected holdout chapters differ from the protocol.", details=details
        )
    chapters_by_segment = {
        segment["id"]: chapter["index"]
        for chapter in (extraction.get("expected") or {}).get("chapters", [])
        for segment in chapter.get("segments", [])
    }
    for row in evidence["pairwise"]["key"].values():
        is_holdout = str(row.get("holdout", "")).lower() == "true"
        chapter = chapters_by_segment.get(row.get("segment_id"))
        if chapter is None or (chapter in declared) != is_holdout:
            return _gate(
                FAILED,
                "Pairwise holdout labels do not match actual source chapters.",
                details=details,
            )
    if not selection.get("holdout_selected"):
        return _gate(
            INCONCLUSIVE,
            "The declared holdout has no sampled observations to verify.",
            details=details,
        )

    comparisons = confirmatory_comparisons(protocol)
    if len(comparisons) > 1:
        results = {
            comparison["id"]: _holdout_gate(
                protocol={**protocol, "comparisons": [comparison]},
                extraction=extraction,
                selection=selection,
                evidence=evidence,
                alpha=alpha,
            )
            for comparison in comparisons
        }
        details["comparisons"] = results
        status = (
            FAILED
            if any(item["status"] == FAILED for item in results.values())
            else INCONCLUSIVE
            if any(item["status"] != PASSED for item in results.values())
            else PASSED
        )
        return _gate(
            status,
            "Every confirmatory comparison must pass its own holdout checks.",
            details=details,
        )

    holdout_expected = 0
    holdout_preference: list[dict[str, Any]] = []
    holdout_mqm: list[dict[str, Any]] = []
    holdout_severe: list[dict[str, Any]] = []
    for comparison in confirmatory_comparisons(protocol):
        observed = _comparison_observations(
            comparison=comparison, evidence=evidence["pairwise"], holdout=True
        )
        holdout_expected += observed["expected"]
        holdout_preference.extend(observed["preference"])
        holdout_mqm.extend(observed["mqm"])
        holdout_severe.extend(observed["severe"])
    details["holdout_expected_pairs"] = holdout_expected
    if holdout_expected <= 0:
        return _gate(
            INCONCLUSIVE,
            "No holdout-specific paired observations or ratings exist; a caller boolean is "
            "not verification.",
            details=details,
        )
    assessed = sum(row["decisive"] or row.get("tie", False) for row in holdout_preference)
    if assessed / holdout_expected < protocol["statistics"]["min_assessable_fraction"]:
        return _gate(
            INCONCLUSIVE,
            "Holdout preference coverage is below its planned denominator.",
            details=details,
        )
    try:
        preference = preference_interval(
            holdout_preference,
            seed=protocol["statistics"]["seed"],
            resamples=protocol["statistics"]["resamples"],
            confidence=protocol["statistics"]["confidence"],
            alpha=alpha,
        )
    except StatsError as error:
        return _gate(INCONCLUSIVE, f"Holdout statistics unavailable: {error}", details=details)
    details["preference"] = preference
    severe = _severe_gate(
        observations=holdout_severe,
        rated=len(holdout_preference),
        expected=holdout_expected,
        protocol=protocol,
    )
    details["severe"] = severe
    if severe["status"] == FAILED:
        return _gate(
            FAILED, "Holdout severe-error gate failed: " + severe["reason"], details=details
        )
    mqm = _mqm_gate(
        observations=holdout_mqm,
        rated=len(holdout_preference),
        expected=holdout_expected,
        protocol=protocol,
        alpha=alpha,
    )
    details["mqm"] = mqm
    if mqm["status"] == FAILED:
        return _gate(FAILED, "Holdout MQM gate failed: " + mqm["reason"], details=details)
    if preference.get("clusters", 0) < protocol["statistics"]["min_clusters"] or not preference.get(
        "available"
    ):
        return _gate(
            INCONCLUSIVE,
            preference.get("reason") or "Holdout preference evidence is insufficient.",
            details=details,
        )
    threshold = float(protocol["decision"]["preference"]["lower_bound"])
    if preference["lower"] <= threshold:
        return _gate(
            INCONCLUSIVE,
            f"Holdout preference lower bound {preference['lower']:.3f} does not clear "
            f"{threshold:.2f}.",
            details=details,
        )
    if severe["status"] != PASSED or mqm["status"] != PASSED:
        return _gate(
            INCONCLUSIVE,
            "Holdout preference clears the threshold, but holdout MQM/severe evidence is "
            f"{mqm['status']}/{severe['status']}.",
            details=details,
        )
    return _gate(
        PASSED,
        f"Holdout ratings pass preference, MQM and severe checks over "
        f"{holdout_expected} expected pairs.",
        details=details,
    )


def _multiple_comparison_gate(
    *, protocol: dict[str, Any], alpha: float, corrected: float
) -> dict[str, Any]:
    confirmatory = confirmatory_comparisons(protocol)
    method = protocol["statistics"]["multiple_comparisons"]
    details = {
        "confirmatory": len(confirmatory),
        "method": method,
        "alpha": alpha,
        "corrected_alpha": corrected,
    }
    if len(confirmatory) > 1 and corrected >= alpha:
        return _gate(
            INCONCLUSIVE,
            "Multiple confirmatory comparisons lack a declared correction or pre-specified family.",
            details=details,
        )
    return _gate(
        PASSED,
        f"{len(confirmatory)} confirmatory comparison(s); correction method {method} gives "
        f"alpha {corrected:.4f}.",
        details=details,
    )


def evaluate_gates(
    *,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    cost: dict[str, Any],
    execution_ok: bool,
    execution_reasons: list[str],
    protocol_state: dict[str, Any],
    evidence: dict[str, Any],
    selection: dict[str, Any] | None = None,
    annotation_binding: dict[str, Any] | None = None,
    protocol_late: bool = False,
    protocol_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate every HLD 9.4 gate and return the tri-state quality conclusion."""
    reasons = list(protocol_reasons or [])
    protocol = protocol_state.get("protocol") if protocol_state.get("status") == "valid" else {}
    protocol = protocol or {}
    gates: dict[str, Any] = {}
    gates["protocol"] = _protocol_gate(
        protocol_state=protocol_state,
        manifest=manifest,
        protocol_late=protocol_late,
        protocol_reasons=reasons,
    )
    gates["evidence"] = (
        _evidence_gate(manifest=manifest, protocol=protocol, evidence=evidence)
        if protocol
        else _gate(
            INCONCLUSIVE,
            "No frozen protocol declares the evidence this run must provide.",
            details={},
        )
    )
    gates["completeness"] = _completeness_gate(
        execution_ok=execution_ok, execution_reasons=execution_reasons
    )
    annotation_binding = annotation_binding or {
        "status": "missing",
        "reason": "No verified annotation generation was supplied.",
    }
    gates["annotations"] = _gate(
        PASSED
        if annotation_binding["status"] == "valid"
        else FAILED
        if annotation_binding["status"] == "invalid"
        else INCONCLUSIVE,
        annotation_binding.get("reason", "Annotation identity is incomplete."),
        details=annotation_binding,
    )
    if not protocol:
        required = {name: gate for name, gate in gates.items() if gate.get("required", True)}
        failed = sorted(name for name, gate in required.items() if gate["status"] == FAILED)
        inconclusive = sorted(
            name for name, gate in required.items() if gate["status"] == INCONCLUSIVE
        )
        return {
            "gates": gates,
            "quality_conclusion": FAILED if failed else INCONCLUSIVE,
            "failed_required": failed,
            "inconclusive_required": inconclusive,
            "empirical_evidence": False,
            "protocol_status": protocol_state.get("status"),
        }

    alpha = 1.0 - float(protocol["statistics"]["confidence"])
    corrected = correction_alpha(protocol, alpha=alpha)
    gates["multiple_comparisons"] = _multiple_comparison_gate(
        protocol=protocol, alpha=alpha, corrected=corrected
    )
    gates["holdout"] = _holdout_gate(
        protocol=protocol,
        extraction=extraction,
        selection=selection,
        evidence=evidence,
        alpha=corrected,
    )

    for comparison in protocol["comparisons"]:
        required = comparison.get("role") == "confirmatory"
        prefix = comparison["id"]
        unit_prefix = f"{protocol['sampling']['clusters']}:"
        invalid_clusters = [
            pair_id
            for pair_id, row in evidence["pairwise"]["key"].items()
            if row.get("comparison") == prefix
            and (
                not (row.get("cluster") or "").strip()
                or not (row.get("cluster") or "").startswith(unit_prefix)
                or (row.get("cluster") or "").endswith(":unknown")
            )
        ]
        if invalid_clusters:
            invalid = _gate(
                INCONCLUSIVE,
                f"{len(invalid_clusters)} pair(s) have no valid cluster assignment; the "
                "pre-registered cluster design cannot be evaluated.",
                required=required,
                details={"invalid_pair_ids": sorted(invalid_clusters)[:20]},
            )
            for metric in METRICS:
                gates[f"{prefix}.{metric}"] = dict(invalid)
            continue
        observations = _comparison_observations(
            comparison=comparison,
            evidence=evidence["pairwise"],
        )
        rated = len(
            [
                row
                for row in evidence["pairwise"]["ratings"]
                if row.get("preference") in {"left", "right", "tie", "skip"}
                and (evidence["pairwise"]["key"].get(row.get("pair_id")) or {}).get("comparison")
                == prefix
            ]
        )
        mqm = _mqm_gate(
            observations=observations["mqm"],
            rated=rated,
            expected=observations["expected"],
            protocol=protocol,
            alpha=corrected,
        )
        gates[f"{prefix}.mqm"] = {**mqm, "required": required}
        gates[f"{prefix}.preference"] = {
            **_preference_gate(
                observations=observations["preference"],
                expected=observations["expected"],
                protocol=protocol,
                alpha=corrected,
                mqm_gate=mqm,
            ),
            "required": required,
        }
        gates[f"{prefix}.omission"] = {
            **_defect_gate(
                label="omission",
                observations=observations["omission"],
                rated=rated,
                expected=observations["expected"],
                protocol=protocol,
                alpha=corrected,
                threshold=float(protocol["decision"]["omission_max_delta"]),
            ),
            "required": required,
        }
        gates[f"{prefix}.addition"] = {
            **_defect_gate(
                label="addition",
                observations=observations["addition"],
                rated=rated,
                expected=observations["expected"],
                protocol=protocol,
                alpha=corrected,
                threshold=float(protocol["decision"]["addition_max_delta"]),
            ),
            "required": required,
        }
        gates[f"{prefix}.severe"] = {
            **_severe_gate(
                observations=observations["severe"],
                rated=rated,
                expected=observations["expected"],
                protocol=protocol,
            ),
            "required": required,
        }
        gates[f"{prefix}.degradation"] = {
            **_degradation_gate(
                candidate=comparison["candidate"],
                evidence=evidence["edits"],
                protocol=protocol,
                alpha=corrected,
            ),
            "required": required,
        }
        gates[f"{prefix}.cost"] = {
            **_cost_gate(
                candidate=comparison["candidate"],
                baseline=comparison["baseline"],
                protocol=protocol,
                cost=cost,
                extraction=extraction,
            ),
            "required": required,
        }

    required_gates = {name: gate for name, gate in gates.items() if gate.get("required", True)}
    failed = sorted(name for name, gate in required_gates.items() if gate["status"] == FAILED)
    inconclusive = sorted(
        name for name, gate in required_gates.items() if gate["status"] == INCONCLUSIVE
    )
    if failed:
        conclusion = FAILED
    elif inconclusive:
        conclusion = INCONCLUSIVE
    else:
        conclusion = PASSED
    return {
        "gates": gates,
        "quality_conclusion": conclusion,
        "failed_required": failed,
        "inconclusive_required": inconclusive,
        "empirical_evidence": gates["evidence"]["status"] == PASSED,
        "protocol_status": protocol_state.get("status"),
        "alpha": alpha,
        "corrected_alpha": corrected,
    }
