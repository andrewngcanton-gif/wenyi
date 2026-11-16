"""Reflective polish integration in the standard batch executor."""

from __future__ import annotations

import json
import re

import pytest
from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.reflector import Reflector
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.llm.limits import RequestStopped
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.translate.translation_batch import (
    ReflectivePolicy,
    TranslationBatchExecutor,
)

from tests.pipeline_fixtures import batch_plan as _plan


def numbered_count(text):
    return len(re.findall(r"^\[(\d+)\]", text, re.MULTILINE))


def clean_critique(n):
    return json.dumps({"issues": [], "reviewed_segments": n, "complete": True})


def major_critique(span="译0", index=0, severity="major"):
    return json.dumps(
        {
            "issues": [
                {
                    "index": index,
                    "category": "fluency",
                    "severity": severity,
                    "span": span,
                    "source_span": "",
                    "accuracy_kind": "",
                    "detail": "生硬",
                    "suggestion": "改写",
                }
            ],
            "reviewed_segments": 2,
            "complete": True,
        },
        ensure_ascii=False,
    )


def make_client(
    *,
    critique=None,
    revise=None,
    continue_polished=None,
    standalone=None,
    translations=None,
):
    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "[Reflect]" in user:
            if isinstance(critique, BaseException):
                raise critique
            return critique if critique is not None else clean_critique(2)
        if "[Revise]" in user:
            if isinstance(revise, BaseException):
                raise revise
            return revise if revise is not None else json.dumps({"polished": ["新0", "新1"]})
        if "Polish the translations from your previous JSON response" in user:
            if isinstance(continue_polished, BaseException):
                raise continue_polished
            return (
                continue_polished
                if continue_polished is not None
                else json.dumps({"polished": ["润0", "润1"]})
            )
        if "prose editor" in system:
            if isinstance(standalone, BaseException):
                raise standalone
            if standalone is not None:
                return standalone
            return json.dumps({"polished": ["全0", "全1", "全2"]}, ensure_ascii=False)
        n = numbered_count(user)
        if translations is not None:
            return translations(n)
        return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

    return FakeClient(handler=handler)


def executor(client, *, reflector=True):
    config = Config.from_dict({"llm": {"preset": "fake"}, "pipeline": {"reflective_polish": True}})
    return TranslationBatchExecutor(
        Translator(client, config),
        Polisher(client, config),
        Reflector(client, config) if reflector else None,
    )


def operations(client):
    return [call["operation"] for call in client.calls]


def test_default_path_is_unchanged_without_reflection():
    client = make_client()
    result = executor(client).execute(_plan(["alpha", "123", "beta"]), polish=True)
    assert operations(client) == ["translation.body", "polish.body"]
    assert result.reflection is None and result.raw_drafts is None
    assert result.targets == ("润0", "123", "润1")


def test_policy_without_a_reflector_uses_the_upstream_path():
    client = make_client()
    result = executor(client, reflector=False).execute(
        _plan(["alpha", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert operations(client) == ["translation.body", "polish.body"]
    assert result.reflection is None


def test_clean_critique_then_ordinary_polish():
    client = make_client()
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert operations(client) == ["translation.body", "reflect.body", "polish.body"]
    assert result.reflection is not None
    assert result.reflection.critique == "validated_clean"
    assert result.reflection.adoption == "ordinary_polished"
    assert result.reflection.adoption_verified is True
    assert result.reflection.ordinary_polish_attempted is True
    assert result.raw_drafts == ("译0", "译1")
    polish_user = client.calls[-1]["messages"][-1]["content"]
    assert "[Reflect]" not in polish_user and "生硬" not in polish_user


def test_clean_validated_keep_skips_the_polish_call():
    client = make_client()
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]),
        polish=True,
        reflective=ReflectivePolicy(clean_action="keep"),
    )
    assert operations(client) == ["translation.body", "reflect.body"]
    assert result.reflection.critique == "validated_clean"
    assert result.reflection.adoption == "draft_kept"
    assert result.reflection.adoption_verified is True
    assert result.targets == result.before_polish == ("译0", "123", "译1")


@pytest.mark.parametrize(
    "critique",
    [
        "not json",
        json.dumps({"issues": [{"index": 9}], "reviewed_segments": 2, "complete": True}),
    ],
)
def test_invalid_critique_never_keeps_the_draft(critique):
    client = make_client(critique=critique)
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]),
        polish=True,
        reflective=ReflectivePolicy(clean_action="keep"),
    )
    assert result.reflection.critique == "invalid"
    assert result.reflection.adoption == "ordinary_polished"
    assert result.targets == ("润0", "123", "润1")
    assert result.reflection.failure in {"invalid_json", "invalid_schema"}
    assert operations(client)[-1] == "polish.body"


def test_provider_failure_falls_back_to_ordinary_polish():
    client = make_client(critique=RuntimeError("boom"))
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]),
        polish=True,
        reflective=ReflectivePolicy(clean_action="keep"),
    )
    assert result.reflection.critique == "provider_failure"
    assert result.reflection.adoption == "ordinary_polished"
    assert result.reflection.failure == "unknown_error"
    assert result.targets == ("润0", "123", "润1")


def test_major_issue_at_major_threshold_adopts_the_revision():
    client = make_client(critique=major_critique())
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert operations(client) == ["translation.body", "reflect.body", "polish.body"]
    assert client.calls[2]["messages"][-1]["content"].startswith("[Revise]")
    assert result.reflection.critique == "validated_gated"
    assert result.reflection.adoption == "revised"
    assert result.reflection.adoption_verified is True
    assert result.reflection.revision_attempts == 1
    assert result.targets == ("新0", "123", "新1")


def test_major_issue_below_critical_threshold_is_treated_as_clean():
    client = make_client(critique=major_critique())
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]),
        polish=True,
        reflective=ReflectivePolicy(min_severity="critical"),
    )
    assert result.reflection.critique == "validated_clean"
    assert result.reflection.gated == ()
    assert operations(client) == ["translation.body", "reflect.body", "polish.body"]
    assert client.calls[2]["messages"][-1]["content"].startswith("[Continue]")


@pytest.mark.parametrize(
    "revise",
    [
        json.dumps({"polished": ["新0"]}),
        json.dumps({"polished": ["新0", 1]}),
        json.dumps({"polished": ["新0", ""]}),
    ],
)
def test_unsafe_revision_falls_back_to_ordinary_polish(revise):
    client = make_client(critique=major_critique(), revise=revise)
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert result.reflection.critique == "validated_gated"
    assert result.reflection.adoption == "ordinary_polished"
    assert result.targets == ("润0", "123", "润1")
    assert result.reflection.revision_attempts == 3


def test_filtered_positions_and_untranslated_segments_stay_unchanged():
    client = make_client(critique=major_critique())
    plan = _plan(["alpha", "123", "beta"])
    result = executor(client).execute(plan, polish=True, reflective=ReflectivePolicy())
    assert result.targets[1] == "123"
    assert "123" not in result.raw_drafts
    assert result.reflection.positions == (0, 2)
    assert plan.segment_indices == (10, 11, 12)


def test_no_turn_records_skipped_and_uses_standalone_polish():
    client = make_client()

    class NoTurnTranslator:
        last_batch_turn = None
        last_batch_indices = None

        def translate_batch(self, sources, **kwargs):
            return ["译0", "译1"]

    config = Config.from_dict({"llm": {"preset": "fake"}, "pipeline": {"reflective_polish": True}})
    batch = TranslationBatchExecutor(
        NoTurnTranslator(), Polisher(client, config), Reflector(client, config)
    )
    result = batch.execute(_plan(["alpha", "beta"]), polish=True, reflective=ReflectivePolicy())
    assert operations(client) == ["polish.body"]
    assert result.reflection.critique == "skipped_no_turn"
    assert result.reflection.adoption == "standalone_polish_attempted"
    assert result.reflection.adoption_verified is False
    assert result.reflection.positions is None
    user = client.calls[0]["messages"][1]["content"]
    assert "[0] alpha" in user and "[1] beta" in user


def test_continuation_failure_labels_only_the_standalone_attempt():
    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "[Reflect]" in user:
            return clean_critique(2)
        if "Polish the translations from your previous JSON response" in user:
            return json.dumps({"polished": ["wrong"]})
        if "prose editor" in system:
            return json.dumps({"polished": ["全0", "全1", "全2"]}, ensure_ascii=False)
        return json.dumps({"translations": ["译0", "译1"]}, ensure_ascii=False)

    client = FakeClient(handler=handler)
    result = executor(client).execute(
        _plan(["alpha", "123", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert operations(client) == ["translation.body", "reflect.body", "polish.body", "polish.body"]
    assert result.reflection.adoption == "standalone_polish_attempted"
    assert result.reflection.adoption_verified is False
    assert result.reflection.ordinary_polish_attempted is True
    assert result.reflection.standalone_polish_attempted is True
    assert result.targets == ("全0", "全1", "全2")


@pytest.mark.parametrize("stage", ["reflect", "revise"])
def test_stops_propagate_without_falling_back(stage):
    if stage == "reflect":
        client = make_client(critique=RequestStopped())
    else:
        client = make_client(critique=major_critique(), revise=RequestStopped())
    with pytest.raises(RequestStopped):
        executor(client).execute(
            _plan(["alpha", "123", "beta"]), polish=True, reflective=ReflectivePolicy()
        )
    assert operations(client) == ["translation.body", "reflect.body"] + (
        ["polish.body"] if stage == "revise" else []
    )


def test_per_segment_fallback_skips_reflection_and_polishes_sources():
    calls = []

    def handler(messages, tier, json_mode):
        calls.append(messages)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "prose editor" in system:
            return json.dumps({"polished": ["ONE", "TWO"]}, ensure_ascii=False)
        if "2 paragraphs, numbered 0 through 1" in user:
            return json.dumps({"translations": ["alignment mismatch"]})
        return json.dumps({"translations": ["translated"]})

    client = FakeClient(handler=handler)
    result = executor(client).execute(
        _plan(["alpha", "beta"]), polish=True, reflective=ReflectivePolicy()
    )
    assert "reflect.body" not in operations(client)
    assert result.reflection.critique == "skipped_no_turn"
    assert result.reflection.adoption == "standalone_polish_attempted"
    assert result.targets == ("ONE", "TWO")
    fallback_user = calls[-1][1]["content"]
    header = "[Current source paragraphs: reference only, not additional output items]"
    assert header in fallback_user
    assert fallback_user.split(header, 1)[1].index("[0] alpha") < fallback_user.split(header, 1)[
        1
    ].index("[1] beta")
