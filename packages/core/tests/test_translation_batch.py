"""Integration of token budgets and translation conversations across batch boundaries."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.translator import Translator
from wenyi_core.config import Config
from wenyi_core.glossary.store import GlossaryStore
from wenyi_core.i18n.resources import read_text
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.document import Chapter, Segment
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.review.review_chunks import ReviewChunkService
from wenyi_core.pipeline.translate.translation import TranslationService
from wenyi_core.pipeline.translate.translation_batch import (
    TranslationBatchExecutor,
    resume_batches,
)
from wenyi_core.storage.file import FileStorage

from tests.fake_llm import routing_handler
from tests.pipeline_fixtures import batch_plan as _plan
from tests.pipeline_fixtures import fake_pipeline_config as _config


def test_resume_and_review_pack_by_tokens_and_preserve_completed_blank():
    segments = [
        Segment(index=10, source="hello world", target=""),
        Segment(index=11, source="foo bar"),
        Segment(index=12, source="x"),
    ]
    assert [[s.index for s in b] for b in resume_batches(segments, 4)] == [[10], [11], [12]]
    assert [[s.index for s in b] for b in ReviewChunkService.pack_contiguous(segments, 4)] == [
        [10, 11],
        [12],
    ]


@pytest.mark.parametrize("genre", ["literary", "academic"])
@pytest.mark.parametrize("invalid_continuation", [False, True])
def test_batch_polish_preserves_filtered_positions_and_fallback(invalid_continuation, genre):
    def handler(messages, tier, json_mode):
        if len(messages) == 4:
            return json.dumps(
                {"polished": ["wrong count"] if invalid_continuation else ["ONE", "TWO"]}
            )
        if f"{genre} translator" in messages[0]["content"]:
            return json.dumps({"translations": ["one", "two"]})
        return json.dumps({"polished": ["ONE", "123", "TWO"]})

    config = Config.from_dict({"llm": {"preset": "fake"}})
    config.set_prompt_genre(genre)
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    plan = _plan(["alpha", "123", "beta"])
    result = executor.execute(plan, polish=True)

    assert result.targets == ("ONE", "123", "TWO")
    assert result.before_polish == ("one", "123", "two")
    assert plan.segment_indices == (10, 11, 12)
    assert [c["operation"] for c in client.calls] == ["translation.body", "polish.body"] + (
        ["polish.body"] if invalid_continuation else []
    )
    assert client.calls[1]["messages"][:2] == client.calls[0]["messages"]
    assert client.calls[1]["messages"][2]["role"] == "assistant"
    if invalid_continuation:
        # The standalone fallback keeps the filtered plan sources as numbered references.
        fallback_user = client.calls[-1]["messages"][1]["content"]
        header = "[Current source paragraphs: reference only, not additional output items]"
        assert header in fallback_user
        source_block = fallback_user.split(header, 1)[1]
        positions = [
            source_block.index(f"[{index}] {source}") for index, source in enumerate(plan.sources)
        ]
        assert positions == sorted(positions)
    rules = dict(config.language_policy().prompt_values)
    for call in client.calls:
        # Continuation repeats the complete rules in its own user turn.
        instruction = call["messages"][-1 if len(call["messages"]) == 4 else 0]["content"]
        assert rules["target_guidance"] in instruction
        assert read_text("shared/target_language.txt").strip() in instruction
        assert rules["punct_rule"] in instruction


@pytest.mark.parametrize("genre", ["literary", "academic"])
def test_per_segment_translation_fallback_polish_receives_plan_sources(genre):
    def handler(messages, tier, json_mode):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "prose editor" in system:
            return json.dumps({"polished": ["ONE", "TWO"]})
        if "2 paragraphs, numbered 0 through 1" in user:
            return json.dumps({"translations": ["alignment mismatch"]})
        return json.dumps({"translations": ["translated"]})

    config = Config.from_dict({"llm": {"preset": "fake"}})
    config.set_prompt_genre(genre)
    client = FakeClient(handler=handler)
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    plan = _plan(["alpha", "beta"])
    result = executor.execute(plan, polish=True)

    assert result.targets == ("ONE", "TWO")
    assert result.before_polish == ("translated", "translated")
    assert [call["operation"] for call in client.calls[-3:]] == [
        "translation.body",
        "translation.body",
        "polish.body",
    ]
    fallback = client.calls[-1]
    assert len(fallback["messages"]) == 2
    user = fallback["messages"][1]["content"]
    header = "[Current source paragraphs: reference only, not additional output items]"
    assert header in user
    source_block = user.split(header, 1)[1]
    assert source_block.index("[0] alpha") < source_block.index("[1] beta")
    assert "exactly 2 items" in user


def test_batch_passes_mineru_blank_allowance_to_translator():
    config = Config.from_dict({"llm": {"preset": "fake"}})
    client = FakeClient(handler=lambda *_: '{"translations": [""]}')
    executor = TranslationBatchExecutor(Translator(client, config), Polisher(client, config))
    result = executor.execute(_plan(["OCR artifact"], allow_empty=True), polish=False)
    assert result.targets == ("",)
    assert result.before_polish == (None,)
    assert len(client.calls) == 1


@pytest.mark.parametrize("backend", ["mineru", "babeldoc", "text"])
def test_chapter_service_limits_blank_allowance_and_resumes_without_retranslation(
    tmp_path, backend
):
    from wenyi_core.agents.translator import AlignmentError
    from wenyi_core.pipeline.orchestrator import Orchestrator
    from wenyi_core.storage.run_store import STATUS_PENDING

    from tests.fake_llm import routing_handler
    from tests.review_fixtures import autofix_config as _config
    from tests.review_fixtures import autofix_store as _store

    store = _store(str(tmp_path))
    manifest = store.load_manifest()
    manifest["fmt"] = "text" if backend == "text" else "pdf"
    manifest["meta"] = {"babeldoc": True} if backend == "babeldoc" else {}
    manifest["chapters"][0]["status"] = STATUS_PENDING
    store.save_manifest(manifest)
    chapter = store.load_chapter(0)
    chapter.text_segments[0].target = None
    store.save_chapter(chapter)
    config = _config(str(tmp_path / "state"))
    config.pipeline.polish = False
    config.pipeline.annotation_alignment = False
    config.pipeline.align_retry_limit = 0

    def handler(messages, tier, json_mode):
        if "literary translator" in messages[0]["content"]:
            return '{"translations": [""]}'
        return routing_handler(messages, tier, json_mode)

    client = FakeClient(handler=handler)
    service = Orchestrator(config, client)._translation
    if backend != "mineru":
        with pytest.raises(AlignmentError):
            service.run(store, book_synopsis="")
        assert store.load_chapter(0).text_segments[0].target is None
        return

    service.run(store, book_synopsis="")
    assert store.load_chapter(0).text_segments[0].target == ""
    assert service.progress_counts(store, [0]) == (1, 1)
    store.set_chapter_status(0, STATUS_PENDING)
    resumed_client = FakeClient(handler=handler)
    Orchestrator(config, resumed_client)._translation.run(store, book_synopsis="")
    assert store.load_chapter(0).text_segments[0].target == ""
    assert store.pending_chapters() == []
    assert not [c for c in resumed_client.calls if c["operation"] == "translation.body"]


class TestTranslationService(unittest.TestCase):
    def test_progress_label_prefers_real_title(self):
        self.assertEqual(TranslationService.chapter_progress_label("引言", 0), "引言")
        self.assertEqual(TranslationService.chapter_progress_label("第一章", 1), "第一章")
        self.assertEqual(TranslationService.chapter_progress_label("", 1), "Chapter 2")

    def test_resume_batch_from_continuation_receives_absolute_annotation_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(os.path.join(directory, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.annotation_alignment = False
            cfg.segment.max_tokens_per_batch = 6
            chapter = Chapter(
                index=0,
                segments=[
                    Segment(
                        index=0,
                        source="aa",
                        target="既译",
                        anchor="tn0_0",
                        meta={
                            "epub_annotations": {
                                "version": 1,
                                "source_length": 6,
                                "items": [
                                    {
                                        "id": "second-piece",
                                        "mode": "point",
                                        "source_start": 3,
                                        "source_end": 3,
                                        "target_key": "notes.xhtml#n2",
                                        "relation": "noteref",
                                    },
                                    {
                                        "id": "third-piece",
                                        "mode": "point",
                                        "source_start": 5,
                                        "source_end": 5,
                                        "target_key": "notes.xhtml#n3",
                                        "relation": "noteref",
                                    },
                                ],
                            }
                        },
                    ),
                    Segment(index=1, source="bb", cont=True),
                    Segment(index=2, source="cc", cont=True),
                ],
            )
            registry = {
                "version": 1,
                "contexts": {
                    "notes.xhtml#n2": {"source_blocks": ["Note two"]},
                    "notes.xhtml#n3": {"source_blocks": ["Note three"]},
                },
            }
            store = FileStorage(os.path.join(directory, "state", "book"))
            store.save_chapter(chapter)
            store.save_manifest(
                {
                    "title": "Book",
                    "fmt": "epub",
                    "source_lang": "en",
                    "target_lang": "zh",
                    "chapters": [{"index": 0, "title": "", "status": "pending"}],
                }
            )
            glossary = GlossaryStore(store.glossary_path)
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            captured: list[list[list[dict[str, str]]]] = []

            def process(plan, **kwargs):
                from wenyi_core.pipeline.translate.translation_batch import BatchResult

                captured.append(plan.annotation_contexts)
                return BatchResult(
                    tuple(f"译{source}" for source in plan.sources), (None,) * len(plan.sources)
                )

            try:
                with (
                    patch.object(orch._translation._batches, "execute", side_effect=process),
                    patch.object(
                        orch._translation,
                        "extract_batch_glossary",
                        return_value={
                            "inserted": 0,
                            "conflict": 0,
                            "unchanged": 0,
                            "updated": 0,
                        },
                    ),
                    patch.object(
                        orch._runtime.extractor,
                        "extract_and_store",
                        return_value={
                            "inserted": 0,
                            "conflict": 0,
                            "unchanged": 0,
                            "updated": 0,
                        },
                    ),
                ):
                    orch._translation.translate_chapter(
                        0,
                        store,
                        glossary,
                        "",
                        translation_history={},
                        source_corpus="aabbcc",
                        annotation_context_registry=registry,
                    )
            finally:
                glossary.close()

            self.assertEqual(
                captured,
                [
                    [
                        [{"target_key": "notes.xhtml#n2", "source": "Note two"}],
                        [{"target_key": "notes.xhtml#n3", "source": "Note three"}],
                    ]
                ],
            )


# ── Reflective polish service integration ────────────────────────────────────────


def _pending_reflection_store(tmp_path):
    from tests.review_fixtures import autofix_store as _store

    store = _store(str(tmp_path))
    manifest = store.load_manifest()
    manifest["chapters"][0]["status"] = "pending"
    store.save_manifest(manifest)
    chapter = store.load_chapter(0)
    for segment in chapter.segments:
        segment.target = None
        segment.target_before_polish = None
    store.save_chapter(chapter)
    return store


def _reflection_config(tmp_path, *, enabled=True):
    from tests.review_fixtures import autofix_config as _config

    config = _config(str(tmp_path / "state"))
    config.pipeline.review_autofix = False
    config.pipeline.annotation_alignment = False
    config.pipeline.align_retry_limit = 0
    config.pipeline.reflective_polish = enabled
    return config


def _reflection_handler(critique=None, revise=None):
    from tests.fake_llm import routing_handler

    def handler(messages, tier, json_mode):
        user = messages[-1]["content"]
        if "[Reflect]" in user:
            if isinstance(critique, BaseException):
                raise critique
            return critique or '{"issues":[],"reviewed_segments":1,"complete":true}'
        if "[Revise]" in user:
            if isinstance(revise, BaseException):
                raise revise
            return revise
        return routing_handler(messages, tier, json_mode)

    return handler


def _batch_events(store):
    return [event for event in store.list_events(limit=0) if event["event"] == "batch_translated"]


def test_service_writes_reflection_artifact_and_event(tmp_path):
    from wenyi_core.i18n.policy.models import content_hash
    from wenyi_core.pipeline.translate.reflection_records import reflection_artifact_key

    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path)
    client = FakeClient(handler=_reflection_handler())
    Orchestrator(config, client)._translation.run(store, book_synopsis="")

    assert store.load_chapter(0).text_segments[0].target == "润0"
    record = store.read_artifact(reflection_artifact_key(0, 0, 1))
    assert record["schema"] == 1
    assert record["critique"] == "validated_clean"
    assert record["adoption"] == "ordinary_polished"
    assert record["source_sha256"] == "0" * 64
    assert record["targets"]["segments"][0]["sha256"] == content_hash("润0")
    assert [call["operation"] for call in client.calls][:3] == [
        "translation.body",
        "reflect.body",
        "polish.body",
    ]
    events = _batch_events(store)
    assert len(events) == 1
    assert events[0]["reflection"]["critique"] == "validated_clean"
    assert events[0]["reflection"]["artifact_status"] == "written"


def test_disabled_reflection_writes_no_artifact_or_event_summary(tmp_path):
    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path, enabled=False)
    client = FakeClient(handler=_reflection_handler())
    Orchestrator(config, client)._translation.run(store, book_synopsis="")

    assert store.list_artifacts("reflection/") == []
    assert "reflect.body" not in [call["operation"] for call in client.calls]
    events = _batch_events(store)
    assert len(events) == 1 and "reflection" not in events[0]


def test_resume_skips_completed_batches_without_reflection(tmp_path):
    from wenyi_core.storage.run_store import STATUS_PENDING

    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path)
    Orchestrator(config, FakeClient(handler=_reflection_handler()))._translation.run(
        store, book_synopsis=""
    )
    first_batches = len(_batch_events(store))
    store.set_chapter_status(0, STATUS_PENDING)
    resumed = FakeClient(handler=_reflection_handler())
    Orchestrator(config, resumed)._translation.run(store, book_synopsis="")

    assert "reflect.body" not in [call["operation"] for call in resumed.calls]
    assert store.load_chapter(0).text_segments[0].target == "润0"
    assert len(_batch_events(store)) == first_batches


def test_artifact_write_failure_keeps_saved_text_and_event(tmp_path, monkeypatch):
    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path)
    original = store.write_artifact

    def failing(key, value):
        if key.startswith("reflection/"):
            raise OSError("disk full")
        return original(key, value)

    monkeypatch.setattr(store, "write_artifact", failing)
    Orchestrator(config, FakeClient(handler=_reflection_handler()))._translation.run(
        store, book_synopsis=""
    )

    assert store.load_chapter(0).text_segments[0].target == "润0"
    events = _batch_events(store)
    assert events[0]["reflection"]["artifact_status"] == "write_failed"
    assert store.list_artifacts("reflection/") == []


def test_save_chapter_failure_propagates_without_artifact(tmp_path, monkeypatch):
    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path)

    def failing(chapter):
        raise OSError("cannot save")

    monkeypatch.setattr(store, "save_chapter", failing)
    with pytest.raises(OSError):
        Orchestrator(config, FakeClient(handler=_reflection_handler()))._translation.run(
            store, book_synopsis=""
        )
    assert store.list_artifacts("reflection/") == []


def test_artifact_stop_propagates_after_the_batch_is_saved(tmp_path, monkeypatch):
    from wenyi_core.llm.limits import RequestStopped

    store = _pending_reflection_store(tmp_path)
    config = _reflection_config(tmp_path)
    original = store.write_artifact

    def stopping(key, value):
        if key.startswith("reflection/"):
            raise RequestStopped()
        return original(key, value)

    monkeypatch.setattr(store, "write_artifact", stopping)
    with pytest.raises(RequestStopped):
        Orchestrator(config, FakeClient(handler=_reflection_handler()))._translation.run(
            store, book_synopsis=""
        )
    assert store.load_chapter(0).text_segments[0].target == "润0"
    assert store.list_artifacts("reflection/") == []
