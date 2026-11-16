from __future__ import annotations

from pathlib import Path

import yaml

from wenyi_eval.manifest import load_group_template

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_configs_declare_languages_without_claiming_a_corpus() -> None:
    for name, source_language in (("en-zh.yaml", "en"), ("ja-zh.yaml", "ja")):
        path = EXAMPLES / name
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)

        assert config["language"] == {"source": source_language, "target": "zh"}
        assert config["pipeline"]["review"] is True
        assert config["pipeline"]["review_autofix"] is False
        assert config["paths"]["state_dir"].startswith("/replace/")
        assert "example only" in text.lower()
        assert "does not bundle, download" in text.lower()
        assert "no evaluation has been executed" in text.lower()


def test_examples_match_the_manifest_common_configuration() -> None:
    template = load_group_template("A_prime")
    for name in ("en-zh.yaml", "ja-zh.yaml"):
        config = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
        for section, values in template.items():
            if section in {"language", "llm", "paths"}:
                continue
            assert config[section] == values
        # Examples carry a real provider preset so a normal run is consumable.
        assert config["llm"]["preset"] != "fake"


def test_example_preset_resolves_its_credential_environment_name() -> None:
    from wenyi_eval.manifest import credential_env_names

    names = credential_env_names(EXAMPLES / "en-zh.yaml")
    assert "DEEPSEEK_API_KEY" in names
    assert all(name.isupper() for name in names)
