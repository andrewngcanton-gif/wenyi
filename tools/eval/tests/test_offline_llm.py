from __future__ import annotations

import json
from pathlib import Path

import pytest

from wenyi_eval.common import (
    OwnershipError,
    PathGuard,
    UnsafePathError,
    bounded_text,
    ownership_lock,
    subprocess_env,
)
from wenyi_eval.offline_llm import handler, polish_text, translate_text


def test_offline_translation_and_polish_are_deterministic_structured_outputs() -> None:
    messages = [
        {"role": "system", "content": "You are a literary translator."},
        {
            "role": "user",
            "content": "[0] Mira carried the old map to the pier.\n[1] Unknown sentence.",
        },
    ]

    first = handler(messages, "strong", True)
    second = handler(messages, "strong", True)

    assert first == second
    payload = json.loads(first)
    assert payload["translations"] == [
        "米拉带着旧地图走向码头。",
        "译文：Unknown sentence.",
    ]
    assert translate_text("Bells rang across the water.") == "钟声越过水面传来。"
    assert (
        translate_text("A sentence outside the lexicon.") == "译文：A sentence outside the lexicon."
    )
    assert polish_text("A sentence outside the lexicon.") == (
        "润色：A sentence outside the lexicon."
    )


def test_offline_handler_answers_review_and_analysis_schemas() -> None:
    review = handler(
        [
            {"role": "system", "content": "You are a translation reviewer."},
            {"role": "user", "content": "[0] one\n[1] two\n[2] three"},
        ],
        "cheap",
        True,
    )
    reviewed = json.loads(review)
    assert reviewed["issues"] == []
    assert reviewed["reviewed_segments"] == 3
    assert reviewed["complete"] is True

    analysis = json.loads(
        handler(
            [{"role": "system", "content": "You are a pre-translation analyst."}],
            "strong",
            True,
        )
    )
    assert analysis["genre"] == "literary"
    assert analysis["characters"][0]["source"] == "Mira"

    detection = json.loads(
        handler([{"role": "system", "content": "You are a language detector."}], "fast", True)
    )
    assert detection == {"language": "en"}


def test_child_environment_excludes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-sentinel")
    monkeypatch.setenv("WENYI_TEST_PASSWORD", "do-not-serialize")

    env = subprocess_env(offline=True)

    assert "sk-test-secret-sentinel" not in env.values()
    assert "do-not-serialize" not in env.values()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["WENYI_EVAL_OFFLINE"] == "1"

    provider_env = subprocess_env(passthrough=["OPENAI_API_KEY"])
    assert "WENYI_EVAL_OFFLINE" not in provider_env
    assert provider_env["OPENAI_API_KEY"] == "sk-test-secret-sentinel"
    assert "do-not-serialize" not in provider_env.values()


def test_path_guard_rejects_escape_and_accepts_children(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    guard = PathGuard(root)

    assert guard.ensure(root / "A" / "state") == (root / "A" / "state").resolve()
    with pytest.raises(UnsafePathError):
        guard.ensure(tmp_path / "outside")
    with pytest.raises(UnsafePathError):
        guard.ensure("relative/path")


def test_ownership_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    with ownership_lock(root):
        with pytest.raises(OwnershipError):
            with ownership_lock(root):
                pass
    assert not (root / ".owner.json").exists()
    with ownership_lock(root):
        assert (root / ".owner.json").is_file()


def test_bounded_text_reports_truncation() -> None:
    text, truncated = bounded_text(b"abcdef", limit=3)
    assert text == "abc"
    assert truncated is True
    text, truncated = bounded_text(b"abc", limit=3)
    assert text == "abc"
    assert truncated is False
