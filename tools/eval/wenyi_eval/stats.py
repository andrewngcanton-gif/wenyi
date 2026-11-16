"""Reproducible cluster-paired resampling for the formal quality gates.

Paragraphs inside one book/chapter/block are correlated, so an i.i.d. paragraph
bootstrap would overstate precision. Every observation here carries a cluster
identity chosen by the frozen protocol; resampling happens over clusters only.

Degenerate and sparse samples do not become certainty. An all-win or all-loss
preference sample, and a zero-event defect/degradation sample, return
``available=False``: a percentile bootstrap or a cluster-level binomial bound
cannot justify a finite pooled-segment interval in those cases. Callers must
report inconclusive rather than pass a threshold.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable


class StatsError(ValueError):
    """A cluster assignment or observation is invalid for the declared design."""


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise StatsError("Cannot take a quantile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def validate_clusters(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return the ordered cluster ids; missing or blank identities are rejected."""
    seen: list[str] = []
    known: set[str] = set()
    for index, row in enumerate(rows):
        cluster = row.get("cluster")
        if not isinstance(cluster, str) or not cluster.strip():
            raise StatsError(f"Observation {index} has no cluster assignment")
        if cluster not in known:
            known.add(cluster)
            seen.append(cluster)
    if len(seen) < 2:
        raise StatsError("Cluster-paired statistics require at least two independent clusters")
    return seen


def _bootstrap_units(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> tuple[list[str], list[list[dict[str, Any]]], int]:
    clusters = validate_clusters(rows)
    buckets: dict[str, list[dict[str, Any]]] = {cluster: [] for cluster in clusters}
    for row in rows:
        buckets[row["cluster"]].append(row)
    rng = random.Random(seed)
    samples: list[list[dict[str, Any]]] = []
    empty = 0
    for _ in range(resamples):
        chosen = [rng.choice(clusters) for _ in clusters]
        pooled: list[dict[str, Any]] = []
        for cluster in chosen:
            pooled.extend(buckets[cluster])
        if not pooled:
            empty += 1
            continue
        samples.append(pooled)
    return clusters, samples, empty


def _empty_limit(resamples: int, empty: int) -> str | None:
    if resamples <= 0:
        raise StatsError("resamples must be positive")
    if empty * 20 > resamples:
        return (
            f"{empty} of {resamples} cluster resamples contained no usable observation; "
            "the interval cannot be estimated honestly"
        )
    return None


def _base_result(
    *,
    clusters: list[str],
    resamples: int,
    seed: int,
    confidence: float,
    method: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "clusters": len(clusters),
        "cluster_ids": clusters,
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "no_decisive_resamples": 0,
        "sparse_treatment": None,
    }


def preference_interval(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-paired non-tie candidate win share with a two-sided interval.

    Each row is ``{"cluster": str, "decisive": bool, "candidate_win": bool}``.
    Ties, skips and missing ratings are non-decisive and stay in the reported
    denominators; they are never imputed as wins or losses.
    """
    rows = list(rows)
    if not rows:
        result = _base_result(
            clusters=[],
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            method="cluster_paired_bootstrap_two_sided",
        )
        result.update(
            {
                "available": False,
                "point": None,
                "lower": None,
                "upper": None,
                "reason": "No paired ratings were supplied.",
            }
        )
        return result
    clusters = validate_clusters(rows)
    decisive_rows = [row for row in rows if row["decisive"]]
    wins = sum(1 for row in decisive_rows if row["candidate_win"])
    result = _base_result(
        clusters=clusters,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        method="cluster_paired_bootstrap_two_sided",
    )
    result["observations"] = len(rows)
    result["decisive"] = len(decisive_rows)
    result["wins"] = wins
    result["losses"] = len(decisive_rows) - wins
    result["ties"] = sum(1 for row in rows if row.get("tie"))
    result["missing"] = sum(1 for row in rows if row.get("missing"))
    result["skipped"] = sum(1 for row in rows if row.get("skipped"))
    if not decisive_rows:
        result.update({"available": False, "point": None, "lower": None, "upper": None})
        result["reason"] = "No decisive paired ratings; the win share is undefined."
        return result
    point = wins / len(decisive_rows)
    if wins in {0, len(decisive_rows)}:
        result.update({"available": False, "point": point, "lower": None, "upper": None})
        result["reason"] = (
            "All decisive pairs favour one side; a finite-sample cluster interval cannot "
            "establish an exact 0% or 100% population win share."
        )
        return result
    _, samples, empty = _bootstrap_units(rows, seed=seed, resamples=resamples)
    limit = _empty_limit(resamples, empty)
    shares: list[float] = []
    no_decisive = 0
    for sample in samples:
        sample_wins = sum(1 for row in sample if row["decisive"] and row["candidate_win"])
        sample_decisive = sum(1 for row in sample if row["decisive"])
        if sample_decisive == 0:
            no_decisive += 1
            continue
        shares.append(sample_wins / sample_decisive)
    result["no_decisive_resamples"] = no_decisive
    if limit is not None or no_decisive * 20 > resamples or not shares:
        result.update({"available": False, "point": point, "lower": None, "upper": None})
        result["reason"] = limit or (
            "Too many cluster resamples had no decisive pairs; the interval is unavailable."
        )
        return result
    lower = _quantile(shares, alpha / 2)
    upper = _quantile(shares, 1 - alpha / 2)
    result.update(
        {
            "available": True,
            "point": point,
            "lower": lower,
            "upper": upper,
            "interval_quantiles": [alpha / 2, 1 - alpha / 2],
        }
    )
    return result


def _paired_rows(rows: Iterable[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    usable = []
    for row in rows:
        candidate = row.get("candidate")
        baseline = row.get("baseline")
        if candidate is None or baseline is None:
            continue
        if isinstance(candidate, bool) or isinstance(baseline, bool):
            raise StatsError(f"{label} values must be numeric/count values, not booleans")
        if not math.isfinite(float(candidate)) or not math.isfinite(float(baseline)):
            raise StatsError(f"{label} values must be finite")
        usable.append(row)
    return usable


def paired_rate_difference(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-paired pooled rate difference ``candidate - baseline``.

    Rows are confirmed-defect indicators (0/1) over assessable paired segments.
    The returned one-sided upper bound is the gate input. With zero candidate
    events the pooled segment estimand has no justified finite cluster interval
    under unequal cluster weights, so the result is explicitly unavailable
    instead of claiming an exact zero-event bound.
    """
    rows = _paired_rows(rows, label="paired rate")
    if not rows:
        result = _base_result(
            clusters=[],
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            method="cluster_paired_bootstrap_one_sided",
        )
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "assessable": 0,
                "reason": "No assessable paired segments; the difference is undefined.",
            }
        )
        return result
    clusters = validate_clusters(rows)
    result = _base_result(
        clusters=clusters,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        method="cluster_paired_bootstrap_one_sided",
    )
    candidate_total = sum(float(row["candidate"]) for row in rows)
    baseline_total = sum(float(row["baseline"]) for row in rows)
    result.update(
        {
            "assessable": len(rows),
            "candidate_events": candidate_total,
            "baseline_events": baseline_total,
            "paired_denominator": len(rows),
        }
    )
    if not rows:
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "reason": "No assessable paired segments; the difference is undefined.",
            }
        )
        return result
    point = (candidate_total - baseline_total) / len(rows)
    if candidate_total == 0:
        result.update(
            {
                "available": False,
                "point": point,
                "upper": None,
                "sparse_treatment": "zero candidate defects: unavailable",
                "reason": (
                    "Zero candidate defects do not justify a finite pooled-segment upper "
                    "bound when clusters have unequal weights; the result is inconclusive."
                ),
            }
        )
        return result
    _, samples, empty = _bootstrap_units(rows, seed=seed, resamples=resamples)
    limit = _empty_limit(resamples, empty)
    differences: list[float] = []
    for sample in samples:
        sample_candidate = sum(float(row["candidate"]) for row in sample)
        sample_baseline = sum(float(row["baseline"]) for row in sample)
        differences.append((sample_candidate - sample_baseline) / len(sample))
    if limit is not None or not differences:
        result.update({"available": False, "point": point, "upper": None})
        result["reason"] = limit or "No usable cluster resamples; the upper bound is unavailable."
        return result
    upper = _quantile(differences, 1 - alpha)
    result.update({"available": True, "point": point, "upper": upper})
    return result


def paired_worsened_rate(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided cluster-paired bound for the human-confirmed degradation rate.

    Rows are ``{"cluster": str, "worsened": 0|1}`` over segments that this
    candidate actually edited relative to its own pre-polish draft.
    """
    rows = [row for row in rows if row.get("worsened") is not None]
    if not rows:
        result = _base_result(
            clusters=[],
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            method="cluster_paired_bootstrap_one_sided",
        )
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "assessable": 0,
                "reason": "No assessable edited samples; degradation is inconclusive, not 0/0.",
            }
        )
        return result
    clusters = validate_clusters(rows)
    result = _base_result(
        clusters=clusters,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        method="cluster_paired_bootstrap_one_sided",
    )
    events = sum(int(row["worsened"]) for row in rows)
    result.update({"assessable": len(rows), "events": events, "paired_denominator": len(rows)})
    if not rows:
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "reason": "No assessable edited samples; degradation is inconclusive, not 0/0.",
            }
        )
        return result
    point = events / len(rows)
    if events == 0:
        result.update(
            {
                "available": False,
                "point": point,
                "upper": None,
                "sparse_treatment": "zero confirmed degradations: unavailable",
                "reason": (
                    "Zero confirmed degradations do not justify a finite pooled-segment "
                    "upper bound under clustered weights; degradation is inconclusive."
                ),
            }
        )
        return result
    _, samples, empty = _bootstrap_units(rows, seed=seed, resamples=resamples)
    limit = _empty_limit(resamples, empty)
    rates: list[float] = []
    for sample in samples:
        sample_events = sum(int(row["worsened"]) for row in sample)
        rates.append(sample_events / len(sample))
    if limit is not None or not rates:
        result.update({"available": False, "point": point, "upper": None})
        result["reason"] = limit or "No usable cluster resamples; the upper bound is unavailable."
        return result
    upper = _quantile(rates, 1 - alpha)
    result.update({"available": True, "point": point, "upper": upper})
    return result


def paired_mean_difference(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-paired one-sided upper bound for ``candidate - baseline`` means."""
    rows = _paired_rows(rows, label="paired mean")
    if not rows:
        result = _base_result(
            clusters=[],
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            method="cluster_paired_bootstrap_one_sided",
        )
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "assessable": 0,
                "reason": "No assessable paired annotations.",
            }
        )
        return result
    clusters = validate_clusters(rows)
    result = _base_result(
        clusters=clusters,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        method="cluster_paired_bootstrap_one_sided",
    )
    if not rows:
        result.update(
            {
                "available": False,
                "point": None,
                "upper": None,
                "assessable": 0,
                "reason": "No assessable paired annotations.",
            }
        )
        return result
    differences = [float(row["candidate"]) - float(row["baseline"]) for row in rows]
    point = sum(differences) / len(differences)
    _, samples, empty = _bootstrap_units(rows, seed=seed, resamples=resamples)
    limit = _empty_limit(resamples, empty)
    means: list[float] = []
    for sample in samples:
        means.append(
            sum(float(row["candidate"]) - float(row["baseline"]) for row in sample) / len(sample)
        )
    if limit is not None or not means:
        result.update(
            {
                "available": False,
                "point": point,
                "upper": None,
                "assessable": len(rows),
                "reason": limit or "No usable cluster resamples.",
            }
        )
        return result
    result.update(
        {
            "available": True,
            "point": point,
            "upper": _quantile(means, 1 - alpha),
            "assessable": len(rows),
        }
    )
    return result
