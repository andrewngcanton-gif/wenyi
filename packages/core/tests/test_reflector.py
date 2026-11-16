"""Strict reflection parsing, message boundaries and revision validation."""

from __future__ import annotations

import json
import logging

import pytest
from wenyi_core.agents.reflector import (
    MAX_EXPLANATION_CHARS,
    MAX_SPAN_CHARS,
    Reflection,
    ReflectionIssue,
    Reflector,
    extract_raw_drafts,
    max_issues,
    parse_reflection,
    validate_revision,
)
from wenyi_core.config import Config
from wenyi_core.llm.limits import RequestStopped
from wenyi_core.llm.providers.fake import FakeClient

DRAFTS = ["译文一", "译文二"]
SOURCES = ["原文一", "原文二"]


def critique(issues=None, *, n=2, complete=True):
    data = {"issues": [] if issues is None else issues, "reviewed_segments": n}
    if complete:
        data["complete"] = True
    return data


def issue(**overrides):
    data = {
        "index": 0,
        "category": "fluency",
        "severity": "major",
        "span": "译文一",
        "source_span": "",
        "detail": "读起来生硬",
        "suggestion": "改写为自然表达",
        "accuracy_kind": "",
    }
    data.update(overrides)
    return data


def config(**pipeline):
    values = {"llm": {"preset": "fake"}}
    if pipeline:
        values["pipeline"] = pipeline
    return Config.from_dict(values)


def turn(drafts=DRAFTS):
    return [
        {"role": "system", "content": "你是译者"},
        {"role": "user", "content": "翻译下面的段落"},
        {
            "role": "assistant",
            "content": json.dumps({"translations": list(drafts)}, ensure_ascii=False),
        },
    ]


def reflector(handler):
    config_ = config(reflective_polish=True)
    return Reflector(FakeClient(handler=handler), config_)


def test_empty_but_complete_critique_is_valid():
    assert parse_reflection(critique(), drafts=DRAFTS, sources=SOURCES) == ()


@pytest.mark.parametrize("data", [None, [], "x", {"issues": []}, critique(complete=False)])
def test_complete_flag_is_required(data):
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize("reviewed", [1.0, True, "2", 1, 3, None])
def test_reviewed_segments_requires_the_exact_integer(reviewed):
    data = critique()
    data["reviewed_segments"] = reviewed
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize("issues", [None, {}, "x", 3])
def test_issues_must_be_a_list(issues):
    data = critique()
    data["issues"] = issues
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


def test_any_non_object_item_invalidates_the_whole_critique():
    data = critique([issue(), "not-an-object"])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize("index", [-1, 2, True, "0", 0.0])
def test_issue_index_must_be_an_in_range_integer(index):
    data = critique([issue(index=index)])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "tone"},
        {"severity": "blocking"},
        {"severity": []},
        {"category": []},
    ],
)
def test_unknown_enums_and_unhashable_values_are_invalid(overrides):
    data = critique([issue(**overrides)])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


def test_accuracy_requires_a_kind_and_is_never_minor():
    base = issue(category="accuracy", severity="major", span="", accuracy_kind="")
    assert parse_reflection(critique([base]), drafts=DRAFTS, sources=SOURCES) is None
    minor = issue(category="accuracy", severity="minor", accuracy_kind="omission")
    assert parse_reflection(critique([minor]), drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "accuracy", "accuracy_kind": "omission", "source_span": ""},
        {"category": "accuracy", "accuracy_kind": "omission", "source_span": "不在原文"},
        {"category": "accuracy", "accuracy_kind": "addition", "span": ""},
        {"category": "accuracy", "accuracy_kind": "mistranslation", "span": "不存在"},
        {"category": "fluency", "span": ""},
        {"category": "fluency", "span": "不存在"},
    ],
)
def test_evidence_must_locate_in_the_original_draft_or_source(overrides):
    data = critique([issue(**overrides)])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


def test_omission_may_omit_the_target_span_but_not_the_source_span():
    valid = issue(category="accuracy", accuracy_kind="omission", span="", source_span="原文一")
    parsed = parse_reflection(critique([valid]), drafts=DRAFTS, sources=SOURCES)
    assert parsed is not None and parsed[0].accuracy_kind == "omission"
    fake_target = issue(
        category="accuracy",
        accuracy_kind="omission",
        span="不在初译的片段",
        source_span="原文一",
    )
    assert parse_reflection(critique([fake_target]), drafts=DRAFTS, sources=SOURCES) is None


def test_non_omission_evidence_fields_must_be_empty_and_locate():
    stray_source = issue(source_span="原文一")
    assert parse_reflection(critique([stray_source]), drafts=DRAFTS, sources=SOURCES) is None
    stray_kind = issue(accuracy_kind="addition")
    assert parse_reflection(critique([stray_kind]), drafts=DRAFTS, sources=SOURCES) is None
    too_long = issue(source_span="原" * (MAX_SPAN_CHARS + 1))
    assert parse_reflection(critique([too_long]), drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize("field", ["detail", "suggestion"])
@pytest.mark.parametrize("value", ["", "   ", None, 5])
def test_detail_and_suggestion_must_be_non_empty_strings(field, value):
    data = critique([issue(**{field: value})])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


@pytest.mark.parametrize("field", ["span", "detail", "suggestion"])
def test_evidence_and_explanations_are_bounded(field):
    limit = MAX_SPAN_CHARS if field == "span" else MAX_EXPLANATION_CHARS
    data = critique([issue(**{field: "译" * (limit + 1)})])
    assert parse_reflection(data, drafts=DRAFTS, sources=SOURCES) is None


def test_span_matching_collapses_whitespace_runs_only():
    drafts = ["译文  一", "译文二"]
    spaced = issue(span="译文 一")
    parsed = parse_reflection(critique([spaced]), drafts=drafts, sources=SOURCES)
    assert parsed is not None and len(parsed) == 1
    changed = issue(span="译文 一")
    assert parse_reflection(critique([changed]), drafts=DRAFTS, sources=SOURCES) is None


def test_duplicate_with_earlier_minor_keeps_the_later_critical():
    data = critique(
        [
            issue(severity="minor", detail="轻微", suggestion="稍改"),
            issue(severity="critical", detail="严重", suggestion="重写"),
        ]
    )
    parsed = parse_reflection(data, drafts=DRAFTS, sources=SOURCES)
    assert parsed is not None and len(parsed) == 1
    assert parsed[0].severity == "critical"
    assert (parsed[0].detail, parsed[0].suggestion) == ("严重", "重写")


def test_duplicate_of_equal_severity_keeps_the_first_unchanged():
    data = critique(
        [
            issue(detail="第一条", suggestion="建议一"),
            issue(detail="第二条", suggestion="建议二"),
        ]
    )
    parsed = parse_reflection(data, drafts=DRAFTS, sources=SOURCES)
    assert parsed is not None and len(parsed) == 1
    assert (parsed[0].detail, parsed[0].suggestion) == ("第一条", "建议一")


def test_too_many_issues_invalidates_instead_of_truncating():
    items = [
        issue(index=index % 2, span=DRAFTS[index % 2])
        for index in range(max_issues(len(DRAFTS)) + 1)
    ]
    assert parse_reflection(critique(items), drafts=DRAFTS, sources=SOURCES) is None


def test_gated_uses_inclusive_severity_thresholds():
    reflection = Reflection(
        (
            ReflectionIssue(0, "style", "minor", "译文一", "", "d", "s", ""),
            ReflectionIssue(1, "style", "major", "译文二", "", "d", "s", ""),
        )
    )
    assert [item.severity for item in reflection.gated("major")] == ["major"]
    assert reflection.gated("critical") == ()


def test_extract_raw_drafts_reads_the_assistant_json():
    assert extract_raw_drafts(turn(), n=2) == ("译文一", "译文二")
    assert extract_raw_drafts(turn(), n=3) is None
    assert extract_raw_drafts(turn(), n=0) is None
    assert extract_raw_drafts(turn()[:2], n=2) is None


def test_extract_raw_drafts_rejects_non_string_entries():
    messages = turn()
    messages[-1]["content"] = json.dumps({"translations": ["a", 1]})
    assert extract_raw_drafts(messages, n=2) is None
    messages[-1]["content"] = "not json"
    assert extract_raw_drafts(messages, n=2) is None


def test_critique_copies_the_turn_and_appends_one_user_message():
    calls = []

    def handler(messages, tier, json_mode):
        calls.append(messages)
        return json.dumps(critique(), ensure_ascii=False)

    original = turn()
    snapshot = [dict(message) for message in original]
    subject = reflector(handler)
    result = subject.critique_continue(
        original, drafts=list(DRAFTS), sources=list(SOURCES), next_source="下一段"
    )

    assert result.failure is None and result.reflection is not None
    assert result.attempts == 1
    sent = calls[0]
    assert len(sent) == 4
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("你是译者\n\n[Editorial continuation]")
    assert sent[1:3] == original[1:3]
    assert sent[-1]["role"] == "user"
    assert "[Reflect]" in sent[-1]["content"]
    assert original == snapshot
    assert subject.client.calls[0]["operation"] == "reflect.body"
    assert set(subject.client.calls[0]["messages"][3]) == {"role", "content"}


def test_critique_retries_structural_failures_then_succeeds():
    responses = iter(["not json", json.dumps(critique(), ensure_ascii=False)])
    subject = reflector(lambda *_: next(responses))
    result = subject.critique_continue(turn(), drafts=list(DRAFTS), sources=list(SOURCES))
    assert result.failure is None and result.attempts == 2
    assert len(subject.client.calls) == 2


def test_critique_exhausts_only_the_finite_structural_budget():
    subject = reflector(lambda *_: json.dumps(critique(issues=[issue(index=9)])))
    result = subject.critique_continue(turn(), drafts=list(DRAFTS), sources=list(SOURCES))
    assert result.reflection is None and result.failure == "invalid_schema"
    assert result.attempts == subject.config.pipeline.align_retry_limit + 1
    assert len(subject.client.calls) == result.attempts


def test_critique_provider_failure_returns_a_category_once(caplog):
    def handler(*_):
        raise RuntimeError("secret body must not be logged")

    subject = reflector(handler)
    with caplog.at_level(logging.WARNING):
        result = subject.critique_continue(turn(), drafts=list(DRAFTS), sources=list(SOURCES))
    assert result.reflection is None
    assert result.failure and result.failure not in {"invalid_json", "invalid_schema"}
    assert result.attempts == 1
    assert len(subject.client.calls) == 1
    assert "secret body must not be logged" not in caplog.text


def test_critique_stop_propagates_without_another_request():
    def handler(*_):
        raise RequestStopped()

    subject = reflector(handler)
    with pytest.raises(RequestStopped):
        subject.critique_continue(turn(), drafts=list(DRAFTS), sources=list(SOURCES))
    assert len(subject.client.calls) == 1


def test_critique_without_a_turn_never_calls_the_model():
    subject = reflector(lambda *_: json.dumps(critique()))
    result = subject.critique_continue([], drafts=list(DRAFTS), sources=list(SOURCES))
    assert result.reflection is None and result.failure == "no_turn" and result.attempts == 0
    assert subject.client.calls == []


def gated_issues():
    return (ReflectionIssue(0, "fluency", "major", "译文一", "", "生硬", "改写", ""),)


def test_revision_sends_only_validated_issues_and_keeps_the_original_turn():
    calls = []

    def handler(messages, tier, json_mode):
        calls.append(messages)
        return json.dumps({"polished": ["润一", "润二"]}, ensure_ascii=False)

    original = turn()
    subject = reflector(handler)
    result = subject.revise_continue(
        original,
        gated_issues(),
        drafts_raw=list(DRAFTS),
        drafts_formal=list(DRAFTS),
        next_source="下一段",
    )

    assert result.items == ("润一", "润二") and result.attempts == 1
    sent = calls[0]
    assert len(sent) == 4
    assert sent[0]["content"].startswith("你是译者\n\n[Editorial continuation]")
    assert sent[1:3] == original[1:3]
    payload = sent[-1]["content"]
    assert "[Revise]" in payload and '"index": 0' in payload
    assert "生硬" in payload and "not json" not in payload
    assert subject.client.calls[0]["operation"] == "polish.body"
    assert subject.client.calls[0]["messages"][1:3] == original[1:3]


@pytest.mark.parametrize(
    "raw,formal,items",
    [
        (["译文一", "译文二"], ["译文一", "译文二"], ["only one"]),
        (["译文一", "译文二"], ["译文一", "译文二"], ["润一", 2]),
        (["译文一", "译文二"], ["译文一", "译文二"], ["润一", ""]),
        (["译文一〘object:a〙", "译文二"], ["译文一", "译文二"], ["润一", "润二"]),
        (["译文一", "译文二"], ["译文一", "译文二"], ["润一〘object:a〙", "润二"]),
        (["译文一〘object:a〙", "译文二"], ["译文一", "译文二"], ["润一〘object:a", "润二"]),
    ],
)
def test_unsafe_revision_candidates_are_rejected(raw, formal, items):
    subject = reflector(lambda *_: json.dumps({"polished": items}, ensure_ascii=False))
    result = subject.revise_continue(
        turn(raw), gated_issues(), drafts_raw=raw, drafts_formal=formal
    )
    assert result.items is None
    assert result.attempts == subject.config.pipeline.align_retry_limit + 1


def test_validate_revision_allows_intentional_empty_translations():
    assert validate_revision(
        ["", "润二"], drafts_raw=["", "译文二"], drafts_formal=["", "译文二"]
    ) == ("", "润二")
    assert (
        validate_revision(
            ["", "润二"], drafts_raw=["", "译文二"], drafts_formal=["译文一", "译文二"]
        )
        is None
    )


def test_revision_payload_contains_only_the_given_gated_issues():
    calls = []

    def handler(messages, tier, json_mode):
        calls.append(messages[-1]["content"])
        return json.dumps({"polished": ["润一", "润二"]}, ensure_ascii=False)

    filtered = ReflectionIssue(1, "style", "minor", "译文二", "", "过滤意见", "不改", "")
    subject = reflector(handler)
    subject.revise_continue(
        turn(),
        gated_issues(),
        drafts_raw=list(DRAFTS),
        drafts_formal=list(DRAFTS),
    )
    assert "生硬" in calls[0]
    assert "过滤意见" not in calls[0]
    assert str(filtered.to_record()) not in calls[0]
    assert "reviewed_segments" not in calls[0]


def test_revision_stop_propagates_without_another_request():
    def handler(*_):
        raise RequestStopped()

    subject = reflector(handler)
    with pytest.raises(RequestStopped):
        subject.revise_continue(
            turn(), gated_issues(), drafts_raw=list(DRAFTS), drafts_formal=list(DRAFTS)
        )
    assert len(subject.client.calls) == 1


def test_revision_requires_issues_and_aligned_drafts():
    subject = reflector(lambda *_: json.dumps({"polished": ["润一", "润二"]}))
    assert (
        subject.revise_continue(
            turn(), (), drafts_raw=list(DRAFTS), drafts_formal=list(DRAFTS)
        ).failure
        == "no_issues"
    )
    assert (
        subject.revise_continue(
            turn(), gated_issues(), drafts_raw=list(DRAFTS), drafts_formal=["译文一"]
        ).failure
        == "no_issues"
    )
    assert subject.client.calls == []
