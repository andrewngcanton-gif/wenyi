"""Optional formal experiment pre-registration frozen before any output exists.

A protocol is a small JSON document that fixes comparisons, sampling clusters,
holdout, seed, resampling method and decision thresholds. ``plan``/``run`` freeze
it into a binding record; later ``export``/``import``/``report`` verify the
binding instead of trusting a file that may have been edited afterwards.

A missing protocol keeps the older diagnostic workflow. A protocol edited after
the evaluation outputs were produced is invalid pre-registration: its gates can
never pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import EvalError, PathGuard, read_json, utc_now, write_json_atomic

PROTOCOL_VERSION = 2
BINDING_NAME = "experiment_protocol.frozen.json"
DRAFT_STATUS = "DRAFT"
REGISTERED_STATUS = "registered"

COMPARISON_ROLES = {"confirmatory", "exploratory"}
CLUSTER_UNITS = {"chapter", "book", "block"}
STATISTICS_METHODS = {"cluster_paired_bootstrap"}
MULTIPLE_COMPARISON_METHODS = {"none", "single", "holm", "bonferroni"}
SCOPES = {"pilot", "release"}
MAIN_BASELINE = "A_prime"
MAIN_CANDIDATES = {"C", "C_keep", "D"}


class ProtocolError(EvalError):
    """The protocol cannot be frozen or does not describe a usable experiment."""


def canonical_bytes(value: Any) -> bytes:
    """Stable bytes for hashing a protocol independent of key or file order."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _number(value: Any, *, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a number")
    if not low <= float(value) <= high:
        raise ProtocolError(f"{label} must be between {low} and {high}")
    return float(value)


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def validate_protocol(data: Any, *, require_registered: bool = True) -> dict[str, Any]:
    """Validate the complete protocol; thresholds are never inferred later."""
    if not isinstance(data, dict):
        raise ProtocolError("Protocol must be a JSON object")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"Protocol must declare protocol_version {PROTOCOL_VERSION}")
    status = data.get("status")
    if require_registered and status != REGISTERED_STATUS:
        raise ProtocolError(
            "Only a registered protocol can be frozen; the draft template is not a registration"
        )
    if status not in {REGISTERED_STATUS, DRAFT_STATUS}:
        raise ProtocolError("Protocol status must be 'registered' or 'DRAFT'")
    if not isinstance(data.get("experiment_id"), str) or not data["experiment_id"].strip():
        raise ProtocolError("Protocol requires a non-empty experiment_id")
    if data.get("scope") not in SCOPES:
        raise ProtocolError("Protocol scope must be 'pilot' or 'release'")

    comparisons = data.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ProtocolError("Protocol requires at least one comparison")
    seen_ids: set[str] = set()
    confirmatory: list[dict[str, Any]] = []
    for index, item in enumerate(comparisons):
        if not isinstance(item, dict):
            raise ProtocolError(f"comparisons[{index}] must be an object")
        comparison_id = item.get("id")
        if not isinstance(comparison_id, str) or not comparison_id.strip():
            raise ProtocolError(f"comparisons[{index}] requires an id")
        if comparison_id in seen_ids:
            raise ProtocolError(f"Duplicate comparison id: {comparison_id}")
        seen_ids.add(comparison_id)
        baseline = item.get("baseline")
        candidate = item.get("candidate")
        if not isinstance(baseline, str) or not isinstance(candidate, str):
            raise ProtocolError(f"{comparison_id} requires baseline and candidate groups")
        if baseline == candidate:
            raise ProtocolError(f"{comparison_id} compares a group with itself")
        role = item.get("role")
        if role not in COMPARISON_ROLES:
            raise ProtocolError(f"{comparison_id} role must be confirmatory or exploratory")
        if role == "confirmatory":
            if baseline != MAIN_BASELINE:
                raise ProtocolError(
                    f"{comparison_id}: a confirmatory comparison must use {MAIN_BASELINE} "
                    "as the baseline"
                )
            if candidate not in MAIN_CANDIDATES:
                raise ProtocolError(
                    f"{comparison_id}: a confirmatory candidate must be one of "
                    f"{sorted(MAIN_CANDIDATES)}"
                )
            confirmatory.append(item)
    if not confirmatory:
        raise ProtocolError("Protocol requires at least one confirmatory comparison")

    sampling = data.get("sampling")
    if not isinstance(sampling, dict):
        raise ProtocolError("Protocol requires sampling")
    if sampling.get("clusters") not in CLUSTER_UNITS:
        raise ProtocolError("sampling.clusters must be chapter, book or block")
    if sampling.get("clusters") == "block":
        _integer(sampling.get("block_size"), label="sampling.block_size", minimum=1)
    _integer(sampling.get("seed"), label="sampling.seed", minimum=0)
    _integer(sampling.get("min_paragraphs"), label="sampling.min_paragraphs", minimum=1)
    _integer(sampling.get("max_paragraphs"), label="sampling.max_paragraphs", minimum=1)
    if sampling["max_paragraphs"] < sampling["min_paragraphs"]:
        raise ProtocolError("sampling.max_paragraphs must be >= sampling.min_paragraphs")
    holdout = sampling.get("holdout")
    if not isinstance(holdout, dict):
        raise ProtocolError("sampling.holdout must declare chapters and verification state")
    chapters = holdout.get("chapters")
    if not isinstance(chapters, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in chapters
    ):
        raise ProtocolError(
            "sampling.holdout.chapters must be a list of non-negative chapter indices"
        )
    if len(set(chapters)) != len(chapters):
        raise ProtocolError("sampling.holdout.chapters must not contain duplicates")
    if not isinstance(holdout.get("verified"), bool):
        raise ProtocolError("sampling.holdout.verified must be true or false")

    statistics = data.get("statistics")
    if not isinstance(statistics, dict):
        raise ProtocolError("Protocol requires statistics")
    if statistics.get("method") not in STATISTICS_METHODS:
        raise ProtocolError("statistics.method must be cluster_paired_bootstrap")
    _integer(statistics.get("resamples"), label="statistics.resamples", minimum=100)
    _number(statistics.get("confidence"), label="statistics.confidence", low=0.5, high=0.999)
    _integer(statistics.get("seed"), label="statistics.seed", minimum=0)
    _integer(statistics.get("min_clusters"), label="statistics.min_clusters", minimum=2)
    _number(
        statistics.get("min_assessable_fraction"),
        label="statistics.min_assessable_fraction",
        low=0.0,
        high=1.0,
    )

    decision = data.get("decision")
    if not isinstance(decision, dict):
        raise ProtocolError("Protocol requires decision thresholds")
    preference = decision.get("preference")
    if not isinstance(preference, dict):
        raise ProtocolError("decision.preference must be declared")
    _number(preference.get("lower_bound"), label="decision.preference.lower_bound", low=0, high=1)
    _number(
        preference.get("mqm_max_mean_increase"),
        label="decision.preference.mqm_max_mean_increase",
        low=0,
        high=1000,
    )
    _number(decision.get("omission_max_delta"), label="decision.omission_max_delta", low=0, high=1)
    _number(decision.get("addition_max_delta"), label="decision.addition_max_delta", low=0, high=1)
    _number(decision.get("degradation_max"), label="decision.degradation_max", low=0, high=1)
    _number(decision.get("cost_max_ratio"), label="decision.cost_max_ratio", low=0, high=100)
    if decision.get("severe_new_mistranslation") != "failed":
        raise ProtocolError("decision.severe_new_mistranslation must be 'failed'")

    method = statistics.get("multiple_comparisons")
    if method not in MULTIPLE_COMPARISON_METHODS:
        raise ProtocolError(
            "statistics.multiple_comparisons must be none, single, holm or bonferroni"
        )
    if len(confirmatory) > 1 and method in {"none", "single"}:
        raise ProtocolError(
            "Multiple confirmatory comparisons require a declared correction "
            "(holm or bonferroni) or a single pre-specified family"
        )
    return data


def load_protocol(
    path: str | os.PathLike[str], *, require_registered: bool = True
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ProtocolError(f"Protocol file does not exist: {source}")
    try:
        data = read_json(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Protocol is not readable JSON: {error}") from error
    return validate_protocol(data, require_registered=require_registered)


def freeze_protocol(root: Path, protocol_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Write an immutable binding before any manifest or translation output."""
    guard = PathGuard(root)
    source = Path(protocol_path).resolve()
    if not source.is_file():
        raise ProtocolError(f"Protocol file does not exist: {source}")
    data = load_protocol(source)
    digest = content_hash(data)
    binding_path = guard.child(BINDING_NAME)
    if binding_path.is_file():
        existing = read_json(binding_path)
        if existing.get("content_sha256") != digest:
            raise ProtocolError(
                "A different protocol is already frozen for this evaluation; "
                "pre-registration is never overwritten"
            )
        return existing
    stat = source.stat()
    binding = {
        "protocol_version": PROTOCOL_VERSION,
        "content_sha256": digest,
        "frozen_at": utc_now(),
        "bound_stage": "plan",
        "source_path": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "protocol": data,
    }
    write_json_atomic(binding_path, binding)
    return binding


def protocol_groups(data: dict[str, Any]) -> set[str]:
    groups: set[str] = set()
    for item in data.get("comparisons", []):
        groups.update({item["baseline"], item["candidate"]})
    return groups


def confirmatory_comparisons(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in data.get("comparisons", []) if item.get("role") == "confirmatory"]


def correction_alpha(data: dict[str, Any], alpha: float = 0.05) -> float:
    """Conservative family-wise alpha; Holm's first step equals alpha/k."""
    count = len(confirmatory_comparisons(data))
    method = (data.get("statistics") or {}).get("multiple_comparisons")
    if count <= 1 or method == "none":
        return alpha
    return alpha / count


def verify_binding(root: Path) -> dict[str, Any]:
    """Return a tri-state protocol identity; never repairs or rewrites it."""
    guard = PathGuard(root)
    binding_path = guard.child(BINDING_NAME)
    if not binding_path.is_file():
        return {
            "status": "missing",
            "reason": (
                "No frozen experiment protocol exists. Legacy or diagnostic runs cannot "
                "support a formal quality conclusion."
            ),
            "content_sha256": None,
            "protocol": None,
        }
    try:
        binding = read_json(binding_path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "reason": f"Frozen protocol binding is unreadable: {type(error).__name__}",
            "content_sha256": None,
            "protocol": None,
        }
    if not isinstance(binding, dict):
        return {
            "status": "invalid",
            "reason": "Frozen protocol binding must be a JSON object, not "
            f"{type(binding).__name__}.",
            "content_sha256": None,
            "protocol": None,
        }
    data = binding.get("protocol")
    digest = binding.get("content_sha256")
    try:
        validate_protocol(data)
    except ProtocolError as error:
        return {
            "status": "invalid",
            "reason": f"Frozen protocol content is invalid: {error}",
            "content_sha256": digest,
            "protocol": data if isinstance(data, dict) else None,
        }
    if content_hash(data) != digest:
        return {
            "status": "invalid",
            "reason": "Frozen protocol binding content does not match its recorded hash.",
            "content_sha256": digest,
            "protocol": data,
        }
    source_path = binding.get("source_path")
    if isinstance(source_path, str) and Path(source_path).is_file():
        source = Path(source_path)
        try:
            current = content_hash(load_protocol(source))
        except ProtocolError:
            current = None
        if current != digest:
            return {
                "status": "invalid",
                "reason": "The protocol source file no longer matches the frozen content hash.",
                "content_sha256": digest,
                "protocol": data,
            }
        if source.stat().st_size != binding.get("source_size"):
            return {
                "status": "invalid",
                "reason": "The protocol source file changed size after it was frozen.",
                "content_sha256": digest,
                "protocol": data,
            }
        if source.stat().st_mtime_ns != binding.get("source_mtime_ns"):
            return {
                "status": "invalid",
                "reason": (
                    "The protocol source file was modified after it was frozen; edited files "
                    "are not valid pre-registration."
                ),
                "content_sha256": digest,
                "protocol": data,
            }
    return {
        "status": "valid",
        "reason": (
            "Protocol content, comparisons, sampling, statistics and thresholds were frozen "
            "before execution."
        ),
        "content_sha256": digest,
        "frozen_at": binding.get("frozen_at"),
        "source_path": source_path,
        "protocol": data,
    }


def _draft_protocol() -> dict[str, Any]:
    """Clearly labelled suggestion; never a registered experiment."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": DRAFT_STATUS,
        "experiment_id": "replace-with-a-unique-experiment-id",
        "title": "DRAFT threshold suggestions for a future experiment",
        "scope": "pilot",
        "comparisons": [
            {
                "id": "Aprime_vs_C",
                "baseline": "A_prime",
                "candidate": "C",
                "role": "confirmatory",
            },
            {
                "id": "Aprime_vs_D",
                "baseline": "A_prime",
                "candidate": "D",
                "role": "confirmatory",
            },
            {
                "id": "Aprime_vs_C_keep",
                "baseline": "A_prime",
                "candidate": "C_keep",
                "role": "exploratory",
            },
        ],
        "sampling": {
            "clusters": "chapter",
            "block_size": None,
            "seed": 20261116,
            "min_paragraphs": 200,
            "max_paragraphs": 300,
            "holdout": {"chapters": [], "verified": False},
        },
        "statistics": {
            "method": "cluster_paired_bootstrap",
            "resamples": 10000,
            "confidence": 0.95,
            "seed": 20261116,
            "min_clusters": 20,
            "min_assessable_fraction": 0.95,
            "multiple_comparisons": "holm",
        },
        "decision": {
            "preference": {"lower_bound": 0.50, "mqm_max_mean_increase": 0.0},
            "omission_max_delta": 0.01,
            "addition_max_delta": 0.01,
            "degradation_max": 0.05,
            "cost_max_ratio": 1.40,
            "severe_new_mistranslation": "failed",
        },
        "draft_notice": (
            "DRAFT suggestions only. They are not user-approved thresholds, were not run, and "
            "must be reviewed and registered before any translation output exists."
        ),
    }


def draft_protocol() -> dict[str, Any]:
    return _draft_protocol()


def write_draft_template(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if target.exists():
        raise ProtocolError(f"Refusing to overwrite an existing file: {target}")
    write_json_atomic(target, _draft_protocol())
    return target
