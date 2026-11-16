from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.helpers import init_git_repo
from wenyi_eval.common import EvalError
from wenyi_eval.manifest import (
    SUPPORTED_GROUPS,
    TEMPLATE_DIR,
    build_group_plans,
    load_group_template,
    require_group_supported,
    write_variant_configs,
)
from wenyi_eval.run_variants import _translate_arguments


def test_every_group_has_a_template_and_unsupported_groups_fail() -> None:
    for name in ("A.yaml", "A_prime.yaml", "B.yaml", "C.yaml", "C_keep.yaml", "D.yaml", "E.yaml"):
        assert (TEMPLATE_DIR / name).is_file()
    for group in SUPPORTED_GROUPS:
        assert load_group_template(group)["pipeline"]["review"] is True

    for group in ("C", "C_keep", "D", "E"):
        with pytest.raises(EvalError, match="unsupported"):
            require_group_supported(group)
    with pytest.raises(EvalError, match="Unknown evaluation group"):
        require_group_supported("Z")


def test_b_disables_polish_and_reflective_templates_are_deferred() -> None:
    def template(name: str) -> dict:
        return yaml.safe_load((TEMPLATE_DIR / name).read_text(encoding="utf-8"))

    assert load_group_template("B")["pipeline"]["polish"] is False
    assert template("C.yaml")["pipeline"]["reflective_polish"] is True
    assert template("C_keep.yaml")["pipeline"]["reflect_clean_batches"] == "keep"
    assert template("D.yaml")["pipeline"]["reflective_polish"] is True
    assert template("E.yaml")["pipeline"]["translation_mode"] == "three_draft_synthesis"


def test_runtime_configs_bind_isolated_state_and_keep_b_unpolished(tmp_path: Path) -> None:
    hashes = write_variant_configs(tmp_path)

    assert set(hashes) == set(SUPPORTED_GROUPS)
    configs = {
        group: yaml.safe_load((tmp_path / group / "config.yaml").read_text(encoding="utf-8"))
        for group in SUPPORTED_GROUPS
    }
    assert configs["B"]["pipeline"]["polish"] is False
    assert configs["A"]["pipeline"]["polish"] is True
    assert configs["A"]["paths"]["state_dir"].endswith("/A/state")
    assert configs["A_prime"]["paths"]["state_dir"].endswith("/A_prime/state")
    assert configs["A"]["paths"]["state_dir"] != configs["A2"]["paths"]["state_dir"]


def test_translate_arguments_follow_the_group_polish_setting(tmp_path: Path) -> None:
    source = tmp_path / "book.en.md"
    output = tmp_path / "book.zh.txt"
    plan = {"progress_path": "progress.json", "result_path": "result.json"}

    polished = _translate_arguments(group="A", source=source, output=output, plan=plan)
    unpolished = _translate_arguments(group="B", source=source, output=output, plan=plan)

    assert "--polish" in polished
    assert "--no-polish" not in polished
    assert "--no-polish" in unpolished
    assert "--polish" not in unpolished
    assert "--review" in unpolished


def test_group_plans_record_explicit_unknown_routes(tmp_path: Path) -> None:
    baseline = init_git_repo(tmp_path / "baseline")
    candidate = init_git_repo(tmp_path / "candidate")
    root = tmp_path / "eval"
    root.mkdir()

    plans = build_group_plans(
        root,
        baseline_source=baseline,
        candidate_source=candidate,
        candidate_expected_sha="unfrozen",
        diagnostic=True,
        baseline_binding={"adapter": "offline"},
        candidate_binding={"adapter": "offline"},
    )

    for plan in plans.values():
        assert plan["routes"]["status"] == "unknown"
        assert plan["route_fingerprints"]["status"] == "unknown"
        assert plan["adapter"] == "offline"
        assert plan["command"]
    assert plans["B"]["polish"] is False
    assert plans["A_prime"]["polish"] is True


def test_user_config_propagates_but_only_overrides_paths_polish_and_review(
    tmp_path: Path,
) -> None:
    from wenyi_eval.manifest import load_user_config, variant_config

    config_path = tmp_path / "user.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {
                    "preset": "deepseek",
                    "providers": {"stub": {"kind": "openai-compatible", "api_key_env": "STUB_KEY"}},
                },
                "pipeline": {
                    "polish": True,
                    "review": False,
                    "review_concurrency": 9,
                    "translation_mode": "three_draft_synthesis",
                },
                "output": {"mono": False},
            }
        ),
        encoding="utf-8",
    )
    base = load_user_config(config_path)

    grouped = variant_config("B", tmp_path / "B" / "state", base=base)

    assert grouped["language"] == {"source": "ja", "target": "zh"}
    assert grouped["llm"]["preset"] == "deepseek"
    assert grouped["llm"]["providers"]["stub"]["api_key_env"] == "STUB_KEY"
    assert grouped["output"]["mono"] is False
    assert grouped["pipeline"]["translation_mode"] == "three_draft_synthesis"
    assert grouped["pipeline"]["polish"] is False
    assert grouped["pipeline"]["review"] is True
    assert grouped["pipeline"]["review_concurrency"] == 1
    assert grouped["paths"]["state_dir"].endswith("/B/state")


def test_user_config_rejects_literal_credentials(tmp_path: Path) -> None:
    from wenyi_eval.manifest import load_user_config

    config_path = tmp_path / "user.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "llm": {
                    "providers": {
                        "stub": {
                            "kind": "openai-compatible",
                            "api_key": "sk-literal-secret",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="literal credential"):
        load_user_config(config_path)
