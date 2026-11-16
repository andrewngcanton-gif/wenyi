from __future__ import annotations

import math

import pytest

from wenyi_eval.stats import (
    StatsError,
    paired_rate_difference,
    paired_worsened_rate,
    preference_interval,
    validate_clusters,
)


def test_finite_all_win_sample_does_not_claim_population_certainty() -> None:
    rows = [
        {"cluster": f"chapter:{chapter}", "decisive": True, "candidate_win": True}
        for chapter in range(30)
        for _ in range(4)
    ]

    result = preference_interval(rows, seed=7, resamples=500)

    assert result["available"] is False
    assert result["point"] == 1.0
    assert result["lower"] is None
    assert result["upper"] is None
    assert result["clusters"] == 30


def test_ties_skips_and_missing_ratings_stay_out_of_decisive_denominators() -> None:
    rows = []
    for chapter in range(20):
        rows.append({"cluster": f"chapter:{chapter}", "decisive": True, "candidate_win": True})
        rows.append({"cluster": f"chapter:{chapter}", "decisive": False, "tie": True})
        rows.append({"cluster": f"chapter:{chapter}", "decisive": False, "skipped": True})
        rows.append({"cluster": f"chapter:{chapter}", "decisive": False, "missing": True})

    result = preference_interval(rows, seed=3, resamples=400)

    assert result["decisive"] == 20
    assert result["wins"] == 20
    assert result["ties"] == 20
    assert result["skipped"] == 20
    assert result["missing"] == 20
    assert result["point"] == 1.0


def test_zero_candidate_defects_cannot_prove_a_pooled_rate_bound() -> None:
    few = [
        {"cluster": f"chapter:{chapter}", "candidate": 0.0, "baseline": 0.0}
        for chapter in range(30)
    ]
    many = [
        {"cluster": f"chapter:{chapter}", "candidate": 0.0, "baseline": 0.0}
        for chapter in range(300)
    ]

    few_result = paired_rate_difference(few, seed=1, resamples=300)
    many_result = paired_rate_difference(many, seed=1, resamples=300)

    assert few_result["point"] == many_result["point"] == 0.0
    assert few_result["available"] is many_result["available"] is False
    assert few_result["upper"] is many_result["upper"] is None
    assert "pooled-segment" in few_result["reason"]


def test_unequal_cluster_zero_events_and_all_losses_remain_unavailable() -> None:
    defects = [
        {"cluster": "chapter:0", "candidate": 0.0, "baseline": 0.0},
        *[{"cluster": "chapter:1", "candidate": 0.0, "baseline": 0.0} for _ in range(1000)],
    ]
    result = paired_rate_difference(defects, seed=7, resamples=100)
    assert result["available"] is False
    assert result["upper"] is None
    losses = [
        {"cluster": f"chapter:{n}", "decisive": True, "candidate_win": False} for n in range(2)
    ]
    result = preference_interval(losses, seed=7, resamples=100)
    assert result["point"] == 0
    assert result["available"] is False
    assert result["lower"] is result["upper"] is None


def test_paired_difference_point_is_hand_computable() -> None:
    rows = []
    for chapter in range(10):
        for _ in range(10):
            rows.append({"cluster": f"chapter:{chapter}", "candidate": 1.0, "baseline": 0.0})
        for _ in range(90):
            rows.append({"cluster": f"chapter:{chapter}", "candidate": 0.0, "baseline": 0.0})

    result = paired_rate_difference(rows, seed=5, resamples=400)

    assert result["point"] == pytest.approx(0.1)
    assert result["candidate_events"] == 100
    assert result["baseline_events"] == 0
    assert result["assessable"] == 1000
    assert result["upper"] >= result["point"]


def test_resamples_without_decisive_pairs_are_reported_not_hidden() -> None:
    rows = [
        {"cluster": f"chapter:{chapter}", "decisive": False, "missing": True}
        for chapter in range(20)
    ]
    rows[0] = {"cluster": "chapter:0", "decisive": True, "candidate_win": True}
    rows[1] = {"cluster": "chapter:1", "decisive": True, "candidate_win": False}

    result = preference_interval(rows, seed=11, resamples=400)

    assert result["available"] is False
    assert result["no_decisive_resamples"] > 0
    assert "decisive" in result["reason"]


def test_degradation_with_zero_errors_in_50_edits_cannot_pass_five_percent() -> None:
    rows = [{"cluster": f"chapter:{chapter}", "worsened": 0} for chapter in range(50)]

    result = paired_worsened_rate(rows, seed=2, resamples=400)

    assert result["point"] == 0.0
    assert result["available"] is False
    assert result["upper"] is None


def test_no_assessable_edits_is_undefined_not_zero_of_zero() -> None:
    result = paired_worsened_rate([], seed=2, resamples=100)

    assert result["available"] is False
    assert result["point"] is None
    assert "inconclusive" in result["reason"]


def test_cluster_validation_rejects_missing_identities_and_single_cluster() -> None:
    with pytest.raises(StatsError, match="no cluster"):
        validate_clusters([{"cluster": ""}])
    with pytest.raises(StatsError, match="at least two"):
        validate_clusters([{"cluster": "chapter:0"}, {"cluster": "chapter:0"}])


def test_cluster_count_controls_precision_not_correlated_paragraph_count() -> None:
    few = []
    for chapter in range(2):
        for _ in range(100):
            few.append(
                {
                    "cluster": f"chapter:{chapter}",
                    "decisive": True,
                    "candidate_win": chapter == 0,
                }
            )
    many = []
    for chapter in range(20):
        for _ in range(10):
            many.append(
                {
                    "cluster": f"chapter:{chapter}",
                    "decisive": True,
                    "candidate_win": chapter < 10,
                }
            )

    few_result = preference_interval(few, seed=13, resamples=500)
    many_result = preference_interval(many, seed=13, resamples=500)

    assert few_result["point"] == many_result["point"] == 0.5
    assert few_result["lower"] == 0.0
    assert few_result["upper"] == 1.0
    assert many_result["lower"] > 0.0
    assert many_result["upper"] < 1.0


def test_seeded_resampling_is_deterministic_and_not_brittle() -> None:
    rows = [
        {
            "cluster": f"chapter:{chapter}",
            "decisive": True,
            "candidate_win": chapter % 3 != 0,
        }
        for chapter in range(40)
    ]

    first = preference_interval(rows, seed=99, resamples=500)
    second = preference_interval(rows, seed=99, resamples=500)

    assert first == second
    assert math.isclose(first["point"], 26 / 40, rel_tol=0, abs_tol=1e-12)
    assert first["lower"] < first["point"] < first["upper"]
