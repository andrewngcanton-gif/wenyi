"""Diagnostic reflection record schema, privacy and read-side matching."""

from __future__ import annotations

import json

import pytest
from wenyi_core.agents.reflector import ReflectionIssue
from wenyi_core.i18n.policy.models import content_hash
from wenyi_core.models.document import Segment
from wenyi_core.pipeline.translate.reflection_records import (
    build_reflection_record,
    reflection_artifact_key,
    reflection_artifact_status,
)
from wenyi_core.pipeline.translate.translation_batch import ReflectionOutcome

POLICY = {
    "code_identity": "reflective-polish/1",
    "reflective_polish": True,
    "min_severity": "major",
    "clean_action": "polish",
    "reflector_fingerprint": "reflector-hash",
    "translation_fingerprint": "translation-hash",
}
ROUTES = {
    "reflect.body": {
        "profile": "default_strong",
        "model": "critic",
        "provider": "default",
        "inference_fingerprint": "route-hash",
        "max_output_tokens": 4096,
        "fallbacks": [],
    }
}


def segments(targets=("译一", "译二"), before=("初稿一", "初稿二")):
    return [
        Segment(
            index=10 + offset,
            source=("原文一", "原文二")[offset],
            target=target,
            target_before_polish=before[offset],
        )
        for offset, target in enumerate(targets)
    ]


def outcome(**overrides):
    values = {
        "critique": "validated_clean",
        "adoption": "ordinary_polished",
        "adoption_verified": True,
        "positions": (0, 1),
        "critique_attempts": 1,
    }
    values.update(overrides)
    return ReflectionOutcome(**values)


def record(**overrides):
    values = {
        "chapter": 3,
        "start_index": 5,
        "segment_indices": (10, 11),
        "sources": ("原文一", "原文二"),
        "source_sha256": "book-digest",
        "source_language": "ja",
        "target_language": "zh",
        "outcome": outcome(),
        "raw_drafts": ("译一", "译二"),
        "positions": (0, 1),
        "executor_targets": ("译一", "译二"),
        "executor_before_polish": ("译一", "译二"),
        "formal_segments": segments(),
        "policy": POLICY,
        "configured_routes": ROUTES,
        "observed_calls": (),
    }
    values.update(overrides)
    return build_reflection_record(**values)


def identity(**overrides):
    values = {
        "chapter": 3,
        "start_index": 5,
        "source_sha256": "book-digest",
        "source_language": "ja",
        "target_language": "zh",
        "batch_source_sha256": content_hash(["原文一", "原文二"]),
        "segment_indices": [10, 11],
        "policy": POLICY,
    }
    values.update(overrides)
    return values


def test_artifact_key_is_readable_and_position_based():
    assert reflection_artifact_key(3, 5, 2) == "reflection/chapter-0003/batch-000005-0002.json"


def test_record_schema_covers_every_required_evidence_group():
    data = record()
    assert data["schema"] == 1
    for key in (
        "chapter",
        "start_index",
        "segment_indices",
        "source_sha256",
        "batch_source_sha256",
        "critique",
        "adoption",
        "failure",
        "positions",
        "critique_attempts",
        "issues",
        "drafts",
        "policy",
        "routes",
        "targets",
        "targets_before_polish",
        "executor",
    ):
        assert key in data, key
    assert data["targets"]["stage"] == "post_alignment"
    assert data["drafts"]["raw_sha256"] == [
        content_hash("译一"),
        content_hash("译二"),
    ]
    assert data["routes"]["observed_status"] == "unknown"


def test_record_does_not_copy_credentials_or_raw_requests():
    data = record()
    encoded = json.dumps(data, ensure_ascii=False)
    assert "api_key" not in encoded
    assert "messages" not in encoded
    assert "Authorization" not in encoded


def test_matching_record_is_current():
    assert reflection_artifact_status(record(), identity(), segments()) == {
        "status": "current",
        "reason": None,
    }


def test_post_alignment_summary_tracks_the_saved_formal_text():
    data = record()
    changed = segments(targets=("译一", "改写二"))
    assert reflection_artifact_status(data, identity(), changed) == {
        "status": "stale",
        "reason": "targets:post_alignment",
    }
    changed_before = segments(before=("初稿一", "另稿二"))
    assert reflection_artifact_status(data, identity(), changed_before) == {
        "status": "stale",
        "reason": "targets_before_polish:digest",
    }


def test_missing_record_is_never_current():
    assert reflection_artifact_status(None, identity(), segments()) == {
        "status": "missing",
        "reason": "not_found",
    }


def test_unknown_schema_and_missing_identity_are_stale():
    data = record()
    data["schema"] = 2
    assert reflection_artifact_status(data, identity(), segments())["status"] == "stale"
    data = record()
    del data["source_sha256"]
    assert reflection_artifact_status(data, identity(), segments())["reason"] == (
        "identity:source_sha256"
    )


def test_source_language_batch_and_segment_mismatches_are_stale():
    assert reflection_artifact_status(record(), identity(chapter=4), segments())["status"] == (
        "stale"
    )
    assert reflection_artifact_status(record(), identity(target_language="en"), segments())[
        "status"
    ] == ("stale")
    assert reflection_artifact_status(record(), identity(batch_source_sha256="other"), segments())[
        "status"
    ] == ("stale")
    assert reflection_artifact_status(record(), identity(segment_indices=[10]), segments())[
        "status"
    ] == ("stale")


def test_policy_changes_invalidate_without_touching_formal_text():
    changed = dict(POLICY, min_severity="critical")
    assert reflection_artifact_status(record(), identity(policy=changed), segments()) == {
        "status": "stale",
        "reason": "policy:min_severity",
    }
    changed = dict(POLICY, reflector_fingerprint="other")
    assert (
        reflection_artifact_status(record(), identity(policy=changed), segments())["reason"]
        == "policy:reflector_fingerprint"
    )


def test_configured_route_change_is_reported_only_when_expectations_include_it():
    expected = identity(configured_route_fingerprints={"reflect.body": "route-hash"})
    assert reflection_artifact_status(record(), expected, segments())["status"] == "current"
    expected = identity(configured_route_fingerprints={"reflect.body": "new-route"})
    assert reflection_artifact_status(record(), expected, segments()) == {
        "status": "stale",
        "reason": "routes:reflect.body",
    }


def test_provider_fallback_is_recorded_as_observed_not_configured():
    observed = (
        {
            "call_id": "c1",
            "operation": "reflect.body",
            "profile": "fallback_profile",
            "model": "fallback-model",
            "inference_fingerprint": "fallback-hash",
            "attempt": 1,
        },
    )
    data = record(observed_calls=observed)
    assert data["routes"]["observed"][0]["model"] == "fallback-model"
    assert data["routes"]["configured"]["reflect.body"]["model"] == "critic"
    assert data["routes"]["observed_status"] == "recorded"


def test_issue_indices_map_to_stable_segment_identities():
    issue = ReflectionIssue(1, "fluency", "major", "译二", "", "生硬", "改写", "")
    data = record(outcome=outcome(issues=(issue,), gated=(issue,), critique="validated_gated"))
    assert data["issues"][0]["segment_index"] == 11
    assert data["gated"][0]["segment_index"] == 11


def test_no_turn_records_null_positions_without_fabricating_a_critique():
    data = record(
        outcome=outcome(
            critique="skipped_no_turn",
            adoption="standalone_polish_attempted",
            adoption_verified=False,
            positions=None,
            critique_attempts=0,
        ),
        raw_drafts=None,
        positions=None,
    )
    assert data["positions"] is None
    assert data["drafts"]["raw_sha256"] is None
    assert data["issues"] == []
    assert reflection_artifact_status(data, identity(), segments())["status"] == "current"


def test_intentional_empty_translation_is_distinct_from_missing():
    empty = segments(targets=("", "译二"), before=("", "初稿二"))
    data = record(
        executor_targets=("", "译二"),
        formal_segments=empty,
    )
    row = data["targets"]["segments"][0]
    assert row["state"] == "empty" and row["sha256"] is None
    assert reflection_artifact_status(data, identity(), empty)["status"] == "current"
    missing = segments(targets=(None, "译二"), before=(None, "初稿二"))
    assert reflection_artifact_status(data, identity(), missing)["status"] == "stale"


def test_record_construction_rejects_misaligned_batches():
    import pytest

    with pytest.raises(ValueError):
        record(segment_indices=(10,))


@pytest.mark.parametrize("section", ["targets", "targets_before_polish"])
@pytest.mark.parametrize("mutation", ["duplicate", "reordered", "outside", "list", "bool"])
def test_digest_rows_require_exact_stable_segment_coverage(section, mutation):
    data = record()
    rows = data[section]["segments"]
    if mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "reordered":
        rows.reverse()
    else:
        rows[0]["segment_index"] = {"outside": 12, "list": [], "bool": True}[mutation]
    changed = segments()
    if mutation == "duplicate":
        if section == "targets":
            changed[1].target = "已编辑"
        else:
            changed[1].target_before_polish = "已编辑初稿"
    assert reflection_artifact_status(data, identity(), changed)["status"] == "stale"


@pytest.mark.parametrize("indices", [[10, 10], [True, 11], [[], 11], []])
def test_batch_segment_identity_cannot_be_ambiguous(indices):
    data = record()
    data["segment_indices"] = indices
    assert (
        reflection_artifact_status(data, identity(segment_indices=indices), segments())["status"]
        == "stale"
    )


def test_incomplete_identity_on_both_sides_is_not_current():
    data = record()
    expected = identity()
    del data["source_sha256"]
    del expected["source_sha256"]
    assert reflection_artifact_status(data, expected, segments())["status"] == "stale"


def test_explicit_expected_code_identity_is_checked():
    assert (
        reflection_artifact_status(
            record(), identity(code_identity="reflective-polish/2"), segments()
        )["status"]
        == "stale"
    )


def test_boolean_schema_is_not_a_protocol_version():
    data = record()
    data["schema"] = True
    assert reflection_artifact_status(data, identity(), segments())["status"] == "stale"
