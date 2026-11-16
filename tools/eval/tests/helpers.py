"""Small builders shared by the evaluation-harness tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from wenyi_core.models.document import Chapter, Segment
from wenyi_core.storage.file import FileStorage

from wenyi_eval.common import sha256_file
from wenyi_eval.manifest import BASELINE_SHA, SUPPORTED_GROUPS

CANDIDATE_SHA = "b" * 40


def chapter_with(
    rows: list[tuple[str, str | None]],
    *,
    index: int = 0,
    title: str = "Chapter",
    meta: dict[str, Any] | None = None,
) -> Chapter:
    return Chapter(
        index=index,
        title=title,
        meta=meta or {},
        segments=[
            Segment(index=position, source=source, kind="text", target=target)
            for position, (source, target) in enumerate(rows)
        ],
    )


def write_run(
    run: Path,
    *,
    chapter: Chapter,
    source_sha256: str = "a" * 64,
    target_lang: str = "zh",
    status: str = "pending",
    initialized: bool = True,
    manifest_meta: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    contexts: dict[str, Any] | None = None,
) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    store = FileStorage(str(run), create=True)
    try:
        store.save_chapter(chapter)
        if analysis is not None:
            store.save_analysis(analysis)
        if contexts is not None:
            store.save_annotation_contexts(contexts)
        store.save_manifest(
            {
                "title": chapter.title,
                "fmt": "text",
                "source_sha256": source_sha256,
                "source_lang": "en",
                "target_lang": target_lang,
                "meta": manifest_meta or {},
                "initialized": initialized,
                "chapters": [{"index": chapter.index, "title": chapter.title, "status": status}],
            }
        )
    finally:
        store.close()
    return run


def minimal_manifest(
    root: Path,
    source: Path | None = None,
    *,
    diagnostic: bool = False,
    prices: dict[str, Any] | None = None,
    candidate_sha: str = CANDIDATE_SHA,
    candidate_binding: str = "clean",
) -> dict[str, Any]:
    """Build a manifest that satisfies validate_manifest without running the CLI."""
    root = root.resolve()
    if source is None:
        source = root / "source" / "book.en.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("Synthetic source for tests.\n", encoding="utf-8", newline="\n")
    groups: dict[str, Any] = {}
    for key in SUPPORTED_GROUPS:
        baseline = key in {"A", "A2"}
        expected = BASELINE_SHA if baseline else candidate_sha
        if not baseline and candidate_binding == "unfrozen":
            expected = "unfrozen"
        group_root = root / key
        groups[key] = {
            "label": key,
            "arm": "baseline" if baseline else "candidate",
            "rerun": key.endswith("2"),
            "code_source": str(root / "code" / key),
            "git": {"head_sha": BASELINE_SHA if baseline else candidate_sha},
            "expected_head_sha": expected,
            "binding_status": "clean" if baseline else candidate_binding,
            "adapter": "offline",
            "executable": "python",
            "argv_prefix": ["-m", "wenyi_cli"],
            "command": ["python", "-m", "wenyi_cli"],
            "polish": key != "B",
            "prompt_policy_hashes": {"tasks": "test", "rules": "test", "languages": "test"},
            "routes": {
                "status": "known",
                "translation.body": "test-route",
                "polish.body": "disabled" if key == "B" else "test-route",
                "reflect.body": "unsupported",
            },
            "route_fingerprints": {"status": "known", "fingerprint": "test"},
            "config": str(group_root / "config.yaml"),
            "cwd": str(group_root / "cwd"),
            "state_dir": str(group_root / "state"),
            "output_dir": str(group_root / "output"),
            "cache_dir": str(group_root / "cache"),
            "progress_path": str(group_root / "receipts" / "progress.json"),
            "result_path": str(group_root / "receipts" / "result.json"),
            "status_path": str(root / "records" / f"{key}.status.txt"),
            "inspect_path": str(root / "records" / f"{key}.inspect-run.txt"),
            "stdout_path": str(group_root / "logs" / "translate.stdout.log"),
            "stderr_path": str(group_root / "logs" / "translate.stderr.log"),
        }
    return {
        "protocol_version": 1,
        "eval_id": "test-eval",
        "book": "book.en",
        "created_at": "2026-11-16T00:00:00+00:00",
        "mode": "test",
        "diagnostic": diagnostic,
        "comparability": {
            "status": "not_comparable" if diagnostic else "pending",
            "reason": "test diagnostic" if diagnostic else None,
        },
        "root": str(root),
        "runs_dir": str(root.parent),
        "repo": {"root": str(root / "repo"), "baseline_sha": BASELINE_SHA},
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "format": "markdown",
            "size": source.stat().st_size,
            "language": {"source": "en", "target": "zh"},
        },
        "fixture": {
            "license": "CC0-1.0",
            "provenance": "test fixture",
            "synthetic": True,
            "source_language": "en",
            "target_language": "zh",
        },
        "groups": groups,
        "preparation": {
            "entry": "test",
            "producer_group": "A_prime",
            "snapshot": str(root / "_prepared"),
        },
        "llm": {
            "external_provider": False,
            "adapter": "test adapter",
            "preset": "fake",
            "routes": {"status": "known", "translation.body": "test-route"},
            "route_fingerprints": {"status": "known", "fingerprint": "test"},
            "model_params": {},
            "seed": 1234,
        },
        "prices": prices
        if prices is not None
        else {"status": "unknown", "date": None, "currency": None, "unit": 1_000_000},
        "software": {},
        "unknowns": ["test"],
    }


def init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.txt").write_text("baseline\n", encoding="utf-8")
    commands = (
        ("init", "-q"),
        ("config", "user.email", "eval@example.invalid"),
        ("config", "user.name", "Eval Test"),
        ("add", "README.txt"),
        ("commit", "-q", "-m", "init"),
    )
    for command in commands:
        subprocess.run(
            ["git", "-C", str(path), *command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return path
