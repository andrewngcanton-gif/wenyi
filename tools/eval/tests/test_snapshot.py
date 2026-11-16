from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import chapter_with, write_run
from wenyi_eval.common import EvalError, UnsafeSnapshotError, hash_tree
from wenyi_eval.snapshot import create_snapshot, seed_state


def _prepare_run(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    root = tmp_path / "eval"
    run = root / "_prepared" / "staging" / "book.en" / "targets" / "zh"
    chapter = chapter_with(
        [("alpha", None), ("beta", None)],
        meta={"origin": "C:\\private\\chapter-source", "note": "ordinary"},
    )
    write_run(
        run,
        chapter=chapter,
        analysis={"genre": "literary", "origin": "D:\\private\\analysis"},
        manifest_meta={"input": "C:\\private\\book.md", "label": "not-a-path"},
        **overrides,
    )
    (run / "preparation" / "digests").mkdir(parents=True)
    (run / "preparation" / "digests" / "known.json").write_text(
        json.dumps({"source": "D:\\private\\digest", "value": 1}), encoding="utf-8"
    )
    (run / "language-policies").mkdir(parents=True, exist_ok=True)
    (run / "language-policies" / "checkpoint.json").write_text(
        json.dumps({"policy": "ok"}), encoding="utf-8"
    )
    (run / "usage.json").write_text("{}", encoding="utf-8")
    (run / "events.jsonl").write_text('{"event": "body"}\n', encoding="utf-8")
    (run / "reviews" / "review-1").mkdir(parents=True)
    (run / "reviews" / "review-1" / "result.json").write_text("{}", encoding="utf-8")
    (run / ".state.lock").write_text("", encoding="utf-8")
    (run / "glossary.db-wal").write_text("", encoding="utf-8")
    return root, run


def _snapshot(root: Path, run: Path):
    return create_snapshot(
        root=root,
        run_dir=run,
        expected_source_sha256="a" * 64,
        expected_target_lang="zh",
    )


def test_snapshot_copies_only_prepared_state_without_locks_or_targets(tmp_path: Path) -> None:
    root, run = _prepare_run(tmp_path)

    receipt = _snapshot(root, run)

    snapshot = root / "_prepared" / "state"
    relative_files = sorted(
        path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()
    )
    assert "manifest.json" in relative_files
    assert "analysis.json" in relative_files
    assert any(name.startswith("chapters/") for name in relative_files)
    assert "parsed_document.json" not in relative_files
    assert "usage.json" not in relative_files
    assert "events.jsonl" not in relative_files
    assert not any("reviews/" in name for name in relative_files)
    assert not any(name.endswith((".lock", "-wal", "-shm")) for name in relative_files)
    assert receipt["body_targets"] == 0
    assert receipt["chapters"] == 1
    assert receipt["segments"] == 2
    assert receipt["glossary_consistent_backup"] is True
    assert receipt["tree_sha256"] == hash_tree(snapshot)

    # The source preparation run is not rewritten by snapshot creation.
    assert (run / "usage.json").is_file()
    assert (run / "events.jsonl").is_file()
    assert (run / "reviews" / "review-1" / "result.json").is_file()


def test_snapshot_sanitizes_absolute_paths_but_keeps_ordinary_text(tmp_path: Path) -> None:
    root, run = _prepare_run(tmp_path)
    _snapshot(root, run)

    from wenyi_core.storage.file import FileStorage

    store = FileStorage(str(root / "_prepared" / "state"), create=False)
    try:
        manifest = store.load_manifest()
        chapter = store.load_chapter(0)
        analysis = store.load_analysis()
        artifact = store.read_artifact("preparation/digests/known.json")
    finally:
        store.close()

    assert manifest["meta"]["input"] == "<stripped-absolute-path>"
    assert manifest["meta"]["label"] == "not-a-path"
    assert chapter.meta["origin"] == "<stripped-absolute-path>"
    assert chapter.meta["note"] == "ordinary"
    assert analysis["origin"] == "<stripped-absolute-path>"
    assert artifact == {"source": "<stripped-absolute-path>", "value": 1}


def test_snapshot_strips_arbitrary_posix_absolute_metadata_without_touching_prose(
    tmp_path: Path,
) -> None:
    from wenyi_eval.snapshot import absolute_path_strings, sanitize_value

    assert sanitize_value("/data/user/book.epub") == "<stripped-absolute-path>"
    assert sanitize_value("/srv/books/book.epub") == "<stripped-absolute-path>"
    assert sanitize_value("/single-segment") == "<stripped-absolute-path>"
    assert sanitize_value("C:\\private\\book.epub") == "<stripped-absolute-path>"
    assert sanitize_value("\\\\server\\share\\book.epub") == "<stripped-absolute-path>"
    assert sanitize_value("ordinary prose / with a slash") == "ordinary prose / with a slash"
    assert sanitize_value("米拉带着旧地图走向码头。") == "米拉带着旧地图走向码头。"

    payload = {"source_path": "/data/user/book.epub", "nested": ["/mnt/books/a.epub"]}
    assert absolute_path_strings(payload) == ["$.source_path", "$.nested[0]"]
    assert sanitize_value(payload) == {
        "source_path": "<stripped-absolute-path>",
        "nested": ["<stripped-absolute-path>"],
    }


def test_preparation_identity_reasons_cover_every_contract_dimension() -> None:
    from wenyi_eval.snapshot import compare_preparation_identity

    expected = {
        "source_sha256": "a" * 64,
        "target_lang": "zh",
        "ingest_config": {"source_lang": "en", "target_lang": "zh"},
        "analysis_style_policy": "analysis",
        "language_policies": {"analysis": "language-policies/a.json"},
        "chapter_digest_policy": "digest",
        "book_understanding": True,
    }
    actual = {
        "source_sha256": "a" * 64,
        "target_lang": "zh",
        "ingest_config": {"source_lang": "en", "target_lang": "zh"},
        "analysis_style_policy": "analysis",
        "language_policies": {"analysis": "language-policies/a.json"},
        "chapter_digest_policies": ["digest"],
    }
    assert compare_preparation_identity(actual, expected) == []

    mismatched = {
        **actual,
        "source_sha256": "b" * 64,
        "target_lang": "ja",
        "ingest_config": {"source_lang": "fr", "target_lang": "ja"},
        "analysis_style_policy": "other",
        "language_policies": {"analysis": "language-policies/b.json"},
        "chapter_digest_policies": ["other"],
    }
    reasons = compare_preparation_identity(mismatched, expected)
    assert any("source identity" in reason for reason in reasons)
    assert any("target language" in reason for reason in reasons)
    assert any("ingestion" in reason for reason in reasons)
    assert any("analysis policy" in reason for reason in reasons)
    assert any("analysis language policy" in reason for reason in reasons)
    assert any("prescan" in reason for reason in reasons)


def test_snapshot_rejects_a_body_target(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    run = root / "_prepared" / "staging" / "book.en" / "targets" / "zh"
    write_run(run, chapter=chapter_with([("alpha", "译文")]))

    with pytest.raises(UnsafeSnapshotError, match="body target"):
        _snapshot(root, run)


def test_snapshot_rejects_a_non_pending_chapter(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    run = root / "_prepared" / "staging" / "book.en" / "targets" / "zh"
    write_run(run, chapter=chapter_with([("alpha", None)]), status="done")

    with pytest.raises(UnsafeSnapshotError, match="not pending"):
        _snapshot(root, run)


def test_snapshot_rejects_source_hash_or_target_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    run = root / "_prepared" / "staging" / "book.en" / "targets" / "zh"
    write_run(run, chapter=chapter_with([("alpha", None)]))

    with pytest.raises(UnsafeSnapshotError, match="source hash"):
        create_snapshot(
            root=root,
            run_dir=run,
            expected_source_sha256="c" * 64,
            expected_target_lang="zh",
        )
    with pytest.raises(UnsafeSnapshotError, match="target language"):
        create_snapshot(
            root=root,
            run_dir=run,
            expected_source_sha256="a" * 64,
            expected_target_lang="ja",
        )


def test_seed_state_preserves_digest_and_refuses_tampered_or_nonempty_trees(
    tmp_path: Path,
) -> None:
    root, run = _prepare_run(tmp_path)
    receipt = _snapshot(root, run)
    snapshot = root / "_prepared" / "state"

    destination = root / "A" / "state" / "book.en" / "targets" / "zh"
    seed_state(snapshot, destination, expected_digest=receipt["tree_sha256"])
    assert hash_tree(destination) == receipt["tree_sha256"]

    with pytest.raises(EvalError, match="non-empty"):
        seed_state(snapshot, destination, expected_digest=receipt["tree_sha256"])

    (snapshot / "tampered.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(EvalError, match="changed before seeding"):
        seed_state(
            snapshot,
            root / "B" / "state" / "book.en" / "targets" / "zh",
            expected_digest=receipt["tree_sha256"],
        )
