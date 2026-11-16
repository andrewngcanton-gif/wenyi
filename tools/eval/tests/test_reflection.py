from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wenyi_eval.common import hash_tree
from wenyi_eval.extract import extract_group, segment_id
from wenyi_eval.manifest import variant_config
from wenyi_eval.reflection import expected_reflection_evidence

SOURCE_SHA = "a" * 64


@pytest.mark.parametrize("mutation", ["source", "extra", "order", "kind"])
def test_reflection_batch_must_match_prepared_inventory(tmp_path: Path, mutation: str) -> None:
    from wenyi_core.storage.file import FileStorage

    from tests.helpers import write_run

    prepared = _chapter(0, ["original", "second"], ["甲", "乙"], ["draft", "draft2"])
    expected = _expected([prepared])
    actual = _chapter(0, ["original", "second"], ["甲", "乙"], ["draft", "draft2"])
    if mutation == "source":
        actual.segments[0].source = "changed"
    elif mutation == "extra":
        actual = _chapter(
            0, ["original", "second", "extra"], ["甲", "乙", "丙"], ["draft", "draft2", "draft3"]
        )
    elif mutation == "order":
        actual.segments.reverse()
    else:
        expected["chapters"][0]["segments"][0]["kind"] = "heading"
    config = _config(tmp_path)
    evidence = expected_reflection_evidence(str(config), source_sha256=SOURCE_SHA, source_lang="en")
    assert evidence is not None
    routes = {
        operation: {"inference_fingerprint": fingerprint}
        for operation, fingerprint in evidence["configured_route_fingerprints"].items()
    }
    state = tmp_path / "state"
    run = write_run(state / "book.en" / "targets" / "zh", chapter=actual)
    store = FileStorage(str(run), create=False)
    try:
        store.write_artifact(
            f"reflection/chapter-0000/batch-000000-{len(actual.segments):04d}.json",
            _record(
                chapter=0,
                start=0,
                indices=[s.index for s in actual.segments],
                sources=[s.source for s in actual.segments],
                targets=[s.target for s in actual.segments],
                before=[s.target_before_polish for s in actual.segments],
                policy=evidence["policy"],
                routes=routes,
            ),
        )
    finally:
        store.close()
    result = extract_group(
        group="C",
        state_dir=state,
        expected=expected,
        plan={"config": str(config), "reflection": {"enabled": True}},
    )
    reflection = result["reflection"]
    assert reflection["status"] == "stale"
    assert reflection["segments"]["expected"] == 2
    assert reflection["segments"]["current"] == 0
    assert reflection["adoption_counts"] == {}


def _config(tmp_path: Path) -> Path:
    config = variant_config("C", tmp_path / "state")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _chapter(index: int, rows: list[str], targets: list[str | None], before: list[str | None]):
    from wenyi_core.models.document import Chapter, Segment

    return Chapter(
        index=index,
        title=f"Chapter {index}",
        segments=[
            Segment(
                index=position,
                source=source,
                kind="text",
                target=target,
                target_before_polish=prior,
            )
            for position, (source, target, prior) in enumerate(zip(rows, targets, before))
        ],
    )


def _record(
    *,
    chapter: int,
    start: int,
    indices: list[int],
    sources: list[str],
    targets: list[str | None],
    before: list[str | None],
    policy: dict,
    routes: dict,
    critique: str = "validated_clean",
    adoption: str = "ordinary_polished",
):
    from wenyi_core.models.document import Segment
    from wenyi_core.pipeline.translate.reflection_records import build_reflection_record
    from wenyi_core.pipeline.translate.translation_batch import ReflectionOutcome

    formal = [
        Segment(index=index, source=source, target=target, target_before_polish=prior)
        for index, source, target, prior in zip(indices, sources, targets, before)
    ]
    outcome = ReflectionOutcome(
        critique=critique,
        adoption=adoption,
        adoption_verified=critique == "validated_clean",
        positions=tuple(range(len(indices))) if critique != "skipped_no_turn" else None,
        critique_attempts=1,
    )
    return build_reflection_record(
        chapter=chapter,
        start_index=start,
        segment_indices=indices,
        sources=sources,
        source_sha256=SOURCE_SHA,
        source_language="en",
        target_language="zh",
        outcome=outcome,
        raw_drafts=[target if target is not None else "" for target in targets],
        positions=outcome.positions,
        executor_targets=targets,
        executor_before_polish=before,
        formal_segments=formal,
        policy=policy,
        configured_routes=routes,
        observed_calls=[],
    )


def _expected(chapters: list) -> dict:
    return {
        "source_sha256": SOURCE_SHA,
        "chapters": [
            {
                "index": chapter.index,
                "title": chapter.title,
                "segments": [
                    {
                        "index": segment.index,
                        "source": segment.source,
                        "kind": "text",
                        "id": segment_id(SOURCE_SHA, chapter.index, segment.index),
                    }
                    for segment in chapter.segments
                ],
            }
            for chapter in chapters
        ],
    }


def test_reflection_extraction_classifies_current_stale_invalid_ambiguous_and_missing(
    tmp_path: Path,
) -> None:
    from wenyi_core.storage.file import FileStorage

    from wenyi_eval.snapshot import find_run_dir

    config = _config(tmp_path)
    evidence = expected_reflection_evidence(str(config), source_sha256=SOURCE_SHA, source_lang="en")
    assert evidence is not None
    routes = {
        operation: {"inference_fingerprint": fingerprint}
        for operation, fingerprint in evidence["configured_route_fingerprints"].items()
    }
    policy = evidence["policy"]

    chapters = [
        _chapter(0, ["a", "b", "c", "d"], ["甲", "乙", "丙", "丁"], ["a0", "b0", "c0", "d0"]),
        _chapter(1, ["e", "f"], ["戊", "己"], ["e0", "f0"]),
        _chapter(2, ["g"], ["庚"], ["g0"]),
        _chapter(3, ["h"], ["辛"], ["h0"]),
        _chapter(4, ["i"], ["壬"], ["i0"]),
    ]
    state = tmp_path / "state"
    run = state / "book.en" / "targets" / "zh"
    store = FileStorage(str(run), create=True)
    try:
        for chapter in chapters:
            store.save_chapter(chapter)
        store.save_manifest(
            {
                "title": "book",
                "fmt": "text",
                "source_sha256": SOURCE_SHA,
                "source_lang": "en",
                "target_lang": "zh",
                "meta": {},
                "initialized": True,
                "chapters": [
                    {"index": chapter.index, "title": chapter.title, "status": "done"}
                    for chapter in chapters
                ],
            }
        )
        store.write_artifact(
            "reflection/chapter-0000/batch-000000-0002.json",
            _record(
                chapter=0,
                start=0,
                indices=[0, 1],
                sources=["a", "b"],
                targets=["甲", "乙"],
                before=["a0", "b0"],
                policy=policy,
                routes=routes,
            ),
        )
        store.write_artifact(
            "reflection/chapter-0000/batch-000002-0002.json",
            _record(
                chapter=0,
                start=2,
                indices=[2, 3],
                sources=["c", "d"],
                targets=["旧丙", "旧丁"],
                before=["c0", "d0"],
                policy=policy,
                routes=routes,
            ),
        )
        for key in ("batch-000000-0002", "batch-000000-0001"):
            store.write_artifact(
                f"reflection/chapter-0001/{key}.json",
                _record(
                    chapter=1,
                    start=0,
                    indices=[0],
                    sources=["e"],
                    targets=["戊"],
                    before=["e0"],
                    policy=policy,
                    routes=routes,
                ),
            )
        store.write_artifact(
            "reflection/chapter-0002/batch-000000-0001.json",
            _record(
                chapter=2,
                start=0,
                indices=[0],
                sources=["g"],
                targets=["庚"],
                before=["g0"],
                policy=policy,
                routes=routes,
                critique="skipped_no_turn",
                adoption="standalone_polish_attempted",
            ),
        )
        store.write_artifact(
            "reflection/chapter-0004/batch-000000-0001.json", {"schema": "not-an-integer"}
        )
    finally:
        store.close()
    assert find_run_dir(state) == run

    before = hash_tree(state)
    result = extract_group(
        group="C",
        state_dir=state,
        expected=_expected(chapters),
        plan={
            "reflection": {"enabled": True, "clean_action": "polish", "min_severity": "major"},
            "config": str(config),
        },
    )

    reflection = result["reflection"]
    assert reflection["denominator"] == 9
    assert reflection["segments"] == {
        "expected": 9,
        "current": 3,
        "missing": 1,
        "stale": 2,
        "invalid": 1,
        "ambiguous": 2,
        "not_enabled": 0,
    }
    assert reflection["artifacts"]["current"] == 2
    assert reflection["artifacts"]["stale"] == 1
    assert reflection["artifacts"]["invalid"] == 1
    assert reflection["artifacts"]["ambiguous"] == 2
    assert reflection["adoption_counts"] == {
        "ordinary_polished": 1,
        "standalone_polish_attempted": 1,
    }
    assert reflection["critique_counts"] == {
        "skipped_no_turn": 1,
        "validated_clean": 1,
    }
    assert reflection["status"] == "ambiguous"
    assert reflection["status"] != "current"
    assert all(
        batch["status"] != "current"
        for batch in reflection["batches"]
        if batch["chapter"] in {0, 1, 4}
        and not (batch["chapter"] == 0 and batch["start_index"] == 0)
    )
    assert hash_tree(state) == before


def test_not_enabled_reflection_keeps_the_segment_denominator(tmp_path: Path) -> None:
    from wenyi_core.storage.file import FileStorage

    from tests.helpers import chapter_with, write_run

    chapter = chapter_with([("alpha", "甲"), ("beta", "乙")])
    state = tmp_path / "state"
    write_run(state / "book.en" / "targets" / "zh", chapter=chapter, status="done")
    store = FileStorage(str(state / "book.en" / "targets" / "zh"), create=False)
    store.close()

    result = extract_group(
        group="A_prime",
        state_dir=state,
        expected=_expected([chapter]),
        plan={"reflection": {"enabled": False}, "config": "unused.yaml"},
    )

    assert result["reflection"]["status"] == "not_enabled"
    assert result["reflection"]["segments"]["expected"] == 2
    assert result["reflection"]["segments"]["not_enabled"] == 2
    assert result["reflection"]["segments"]["current"] == 0


def test_reflection_status_flows_through_pairwise_export_and_report(tmp_path: Path) -> None:
    from wenyi_core.storage.file import FileStorage

    from tests.helpers import minimal_manifest
    from wenyi_eval.common import append_jsonl, write_json_atomic
    from wenyi_eval.pairwise import export_pairwise
    from wenyi_eval.report import write_report

    config = _config(tmp_path)
    evidence = expected_reflection_evidence(str(config), source_sha256=SOURCE_SHA, source_lang="en")
    routes = {
        operation: {"inference_fingerprint": fingerprint}
        for operation, fingerprint in evidence["configured_route_fingerprints"].items()
    }
    chapter = _chapter(0, ["a", "b"], ["甲", "乙"], ["a0", "b0"])
    root = tmp_path / "eval"
    state = root / "C" / "state"
    run = state / "book.en" / "targets" / "zh"
    store = FileStorage(str(run), create=True)
    try:
        store.save_chapter(chapter)
        store.save_manifest(
            {
                "title": "book",
                "fmt": "text",
                "source_sha256": SOURCE_SHA,
                "source_lang": "en",
                "target_lang": "zh",
                "meta": {},
                "initialized": True,
                "chapters": [{"index": 0, "title": chapter.title, "status": "done"}],
            }
        )
        store.write_artifact(
            "reflection/chapter-0000/batch-000000-0002.json",
            _record(
                chapter=0,
                start=0,
                indices=[0, 1],
                sources=["a", "b"],
                targets=["旧甲", "旧乙"],
                before=["a0", "b0"],
                policy=evidence["policy"],
                routes=routes,
            ),
        )
    finally:
        store.close()

    extracted = extract_group(
        group="C",
        state_dir=state,
        expected=_expected([chapter]),
        plan={
            "reflection": {"enabled": True, "clean_action": "polish", "min_severity": "major"},
            "config": str(config),
        },
    )
    assert extracted["reflection"]["status"] == "stale"
    assert extracted["reflection"]["adoption_counts"] == {}

    identifiers = [segment_id(SOURCE_SHA, 0, index) for index in range(2)]
    extraction = {
        "groups": {
            "A_prime": {
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
                    "status": "not_enabled",
                    "enabled": False,
                    "segments": {"expected": 2, "current": 0, "not_enabled": 2},
                },
                "chapters": [
                    {
                        "segments": [
                            {"id": identifiers[0], "target": "甲", "status": "translated"},
                            {"id": identifiers[1], "target": "乙", "status": "translated"},
                        ]
                    }
                ],
            },
            "C": extracted,
        }
    }
    append_jsonl(
        root / "samples" / "strata.jsonl",
        [
            {
                "id": identifiers[0],
                "source_text": "a",
                "source_context_before": "",
                "source_context_after": "b",
            }
        ],
    )
    manifest = minimal_manifest(root, diagnostic=True, groups=("A_prime", "C"))
    preregistration = export_pairwise(root=root, manifest=manifest, extraction=extraction)
    assert [item["id"] for item in preregistration["comparisons"]] == ["Aprime_vs_C"]
    assert preregistration["denominator"] == 1

    terminals = {"preparation.prepare": {"status": "succeeded"}}
    for group in ("A_prime", "C"):
        for stage in ("translate", "status", "inspect"):
            terminals[f"{group}.{stage}"] = {
                "status": "succeeded",
                "receipt_validation": {"status": "valid"},
            }
    execution = {
        "terminals": terminals,
        "source_unchanged": True,
        "preparation_unchanged": True,
        "groups": {group: {"durations": {"translate": 1.0}} for group in ("A_prime", "C")},
    }
    cost = {
        "price_status": "unknown",
        "price_date": None,
        "groups": {
            "preparation": {"tokens": {"calls": 1}, "cost": {"status": "unknown"}},
            "A_prime": {"tokens": {"calls": 2}, "cost": {"status": "unknown"}},
            "C": {"tokens": {"calls": 2}, "cost": {"status": "unknown"}},
        },
        "aggregate_cost": {"status": "unknown", "amount": None},
        "aggregate_tokens": {"calls": 5},
    }
    write_json_atomic(root / "samples" / "selection.json", {"selected": 1, "groups": ["C"]})

    report = write_report(
        root=root,
        manifest=manifest,
        extraction=extraction,
        execution=execution,
        cost=cost,
    )

    assert report["reflection"]["C"]["segments"]["stale"] == 2
    assert report["execution_ok"] is False
    assert report["conclusion"] == "failed"
    assert any(
        "without a current reflection artifact" in reason for reason in report["execution_reasons"]
    )


def test_unresolvable_reflection_policy_fails_closed(tmp_path: Path) -> None:
    from tests.helpers import chapter_with, write_run

    chapter = chapter_with([("alpha", "甲"), ("beta", "乙")])
    state = tmp_path / "state"
    write_run(state / "book.en" / "targets" / "zh", chapter=chapter, status="done")

    result = extract_group(
        group="C",
        state_dir=state,
        expected=_expected([chapter]),
        plan={
            "reflection": {"enabled": True, "clean_action": "polish", "min_severity": "major"},
            "config": str(tmp_path / "missing.yaml"),
        },
    )

    assert result["reflection"]["status"] == "invalid"
    assert result["reflection"]["segments"]["invalid"] == 2
    assert result["reflection"]["segments"]["current"] == 0
    assert result["reflection"]["adoption_counts"] == {}
