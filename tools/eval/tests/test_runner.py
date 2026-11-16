from __future__ import annotations

from pathlib import Path

import pytest

from wenyi_eval.common import EvalError, OwnershipError
from wenyi_eval.manifest import SUPPORTED_GROUPS
from wenyi_eval.run_variants import GROUP_STAGES, TerminalStore, initialize_root


def _terminal(group: str, stage: str, status: str = "succeeded") -> dict:
    return {
        "protocol_version": 1,
        "group": group,
        "stage": stage,
        "status": status,
    }


def test_terminal_store_requires_every_group_stage_exactly_once(tmp_path: Path) -> None:
    store = TerminalStore(tmp_path / "eval")
    with pytest.raises(EvalError, match="Missing terminal records"):
        store.validate_complete()

    store.write(_terminal("preparation", "prepare"))
    for group in SUPPORTED_GROUPS:
        for stage in GROUP_STAGES:
            store.write(_terminal(group, stage))

    rows = store.validate_complete()
    assert len(rows) == 1 + len(SUPPORTED_GROUPS) * len(GROUP_STAGES)

    with pytest.raises(OwnershipError, match="already exists"):
        store.write(_terminal("A", "translate"))


def test_initialize_root_requires_absolute_safe_and_empty_paths(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="absolute"):
        initialize_root("relative-runs", eval_id="eval", book="book.en")
    with pytest.raises(EvalError, match="Unsafe evaluation path"):
        initialize_root(tmp_path, eval_id="..", book="book.en")
    with pytest.raises(EvalError, match="Unsafe evaluation path"):
        initialize_root(tmp_path, eval_id="eval", book="book/other")

    root = initialize_root(tmp_path, eval_id="eval", book="book.en")
    assert root == (tmp_path / "eval" / "book.en").resolve()
    (root / "occupied.txt").write_text("x", encoding="utf-8")
    with pytest.raises(EvalError, match="overwrite"):
        initialize_root(tmp_path, eval_id="eval", book="book.en")


def test_finish_imports_its_pairwise_export_and_preserves_unrated_rows(tmp_path: Path) -> None:
    from wenyi_core.storage.file import FileStorage

    from tests.helpers import chapter_with, minimal_manifest, write_run
    from wenyi_eval.common import read_json
    from wenyi_eval.run_variants import _finish

    manifest = minimal_manifest(tmp_path, diagnostic=True)
    manifest["sampling"] = {"min": 1, "max": 2}
    source_hash = manifest["source"]["sha256"]
    write_run(
        tmp_path / "_prepared" / "state",
        chapter=chapter_with([("The lantern glowed.", None)]),
        source_sha256=source_hash,
    )
    for group, plan in manifest["groups"].items():
        run_dir = write_run(
            Path(plan["state_dir"]) / "book.en" / "targets" / "zh",
            chapter=chapter_with([("The lantern glowed.", f"{group} translation")]),
            source_sha256=source_hash,
            status="done",
        )
        store = FileStorage(str(run_dir), create=False)
        try:
            store.write_artifact(
                "reviews/review-test/result.json",
                {"issues": [], "reviewed_segments": 1, "complete": True},
            )
        finally:
            store.close()

    terminals = {"preparation.prepare": {"status": "succeeded"}}
    for group in SUPPORTED_GROUPS:
        for stage in GROUP_STAGES:
            terminal = {"status": "succeeded"}
            if stage == "translate":
                terminal["receipt_validation"] = {"status": "valid"}
            terminals[f"{group}.{stage}"] = terminal
    execution = {
        "terminals": terminals,
        "groups": {},
        "source_unchanged": True,
        "preparation_unchanged": True,
    }
    result = _finish(tmp_path, manifest, execution)

    imported = read_json(tmp_path / "samples" / "pairwise_import.json")
    assert imported["denominator"] > 0
    assert imported["rated_rows"] == 0
    assert imported["blank_ratings"] == imported["denominator"]
    assert result["report"]["conclusion"] == "inconclusive"
    assert (tmp_path / "report" / "summary.md").is_file()


def test_cli_help_is_a_real_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    from wenyi_eval.run_variants import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "plan" in output and "run" in output and "extract" in output
    assert "protocol-template" in output


def test_protocol_template_subcommand_writes_a_draft(tmp_path: Path) -> None:
    from wenyi_eval.run_variants import main

    target = tmp_path / "draft.json"

    assert main(["protocol-template", "--out", str(target)]) == 0
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert '"DRAFT"' in text
    assert "draft_notice" in text


def test_child_command_uses_the_declared_executable_and_argv() -> None:
    from wenyi_eval.run_variants import _child_command

    plan = {
        "executable": "/opt/venv/bin/python",
        "argv_prefix": ["-m", "wenyi_cli"],
    }

    assert _child_command(plan, "/tmp/config.yaml", ["status", "book.md"]) == [
        "/opt/venv/bin/python",
        "-m",
        "wenyi_cli",
        "--config",
        "/tmp/config.yaml",
        "status",
        "book.md",
    ]


def test_terminal_validation_accepts_a_selected_subset_and_isolated_preparation(
    tmp_path: Path,
) -> None:
    store = TerminalStore(tmp_path / "eval")
    store.write(_terminal("preparation", "prepare"))
    store.write(_terminal("preparation", "prepare.C"))
    for group in ("A", "C"):
        for stage in GROUP_STAGES:
            store.write(_terminal(group, stage))

    rows = store.validate_complete(groups=("A", "C"), isolated_preparations=("C",))

    assert len(rows) == 2 + 2 * len(GROUP_STAGES)
    assert ("preparation", "prepare.C") in rows
