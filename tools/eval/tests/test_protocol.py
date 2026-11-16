from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from wenyi_eval.common import write_json_atomic
from wenyi_eval.protocol import (
    BINDING_NAME,
    ProtocolError,
    confirmatory_comparisons,
    content_hash,
    correction_alpha,
    draft_protocol,
    freeze_protocol,
    load_protocol,
    verify_binding,
    write_draft_template,
)


def _protocol(**overrides):
    data = draft_protocol()
    data["status"] = "registered"
    data["comparisons"] = [
        {
            "id": "Aprime_vs_C",
            "baseline": "A_prime",
            "candidate": "C",
            "role": "confirmatory",
        }
    ]
    data["sampling"]["holdout"] = {"chapters": [9], "verified": True}
    data["scope"] = "release"
    data.update(overrides)
    return data


def _registered(tmp_path: Path, **overrides) -> tuple[Path, dict]:
    data = _protocol(**overrides)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def test_draft_template_is_labelled_and_not_registrable(tmp_path: Path) -> None:
    target = write_draft_template(tmp_path / "draft.json")
    draft = json.loads(target.read_text(encoding="utf-8"))

    assert draft["status"] == "DRAFT"
    assert "DRAFT" in draft["draft_notice"]
    assert draft["decision"]["cost_max_ratio"] == 1.40
    with pytest.raises(ProtocolError, match="registered"):
        load_protocol(target)


def test_freeze_then_verify_binds_content_hash(tmp_path: Path) -> None:
    path, data = _registered(tmp_path)

    binding = freeze_protocol(tmp_path / "eval", path)
    state = verify_binding(tmp_path / "eval")

    assert binding["content_sha256"] == content_hash(data)
    assert state["status"] == "valid"
    assert state["content_sha256"] == content_hash(data)
    assert (tmp_path / "eval" / BINDING_NAME).is_file()


def test_protocol_edited_after_freeze_is_invalid(tmp_path: Path) -> None:
    path, data = _registered(tmp_path)
    freeze_protocol(tmp_path / "eval", path)
    changed = dict(data, title="edited after output")
    time.sleep(0.02)
    path.write_text(json.dumps(changed), encoding="utf-8")

    state = verify_binding(tmp_path / "eval")

    assert state["status"] == "invalid"
    assert "modified" in state["reason"] or "no longer matches" in state["reason"]


def test_tampered_binding_content_is_invalid(tmp_path: Path) -> None:
    path, _ = _registered(tmp_path)
    freeze_protocol(tmp_path / "eval", path)
    binding_path = tmp_path / "eval" / BINDING_NAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["protocol"]["decision"]["cost_max_ratio"] = 2.0
    write_json_atomic(binding_path, binding)

    state = verify_binding(tmp_path / "eval")

    assert state["status"] == "invalid"
    assert "hash" in state["reason"]


def test_missing_binding_is_missing_not_legacy_pass(tmp_path: Path) -> None:
    state = verify_binding(tmp_path / "eval")
    assert state["status"] == "missing"
    assert "cannot support a formal quality conclusion" in state["reason"]


def test_multiple_confirmatory_comparisons_require_correction(tmp_path: Path) -> None:
    path, _ = _registered(tmp_path)
    data = _protocol()
    data["comparisons"].append(
        {
            "id": "Aprime_vs_D",
            "baseline": "A_prime",
            "candidate": "D",
            "role": "confirmatory",
        }
    )
    data["statistics"]["multiple_comparisons"] = "none"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProtocolError, match="correction"):
        load_protocol(path)

    data["statistics"]["multiple_comparisons"] = "holm"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_protocol(path)
    assert len(confirmatory_comparisons(loaded)) == 2
    assert correction_alpha(loaded, 0.05) == pytest.approx(0.025)


def test_confirmatory_comparison_must_use_aprime_baseline(tmp_path: Path) -> None:
    data = _protocol()
    data["comparisons"][0]["baseline"] = "A"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProtocolError, match="A_prime"):
        load_protocol(path)


def test_relative_or_overwritten_freeze_is_refused(tmp_path: Path) -> None:
    first, _ = _registered(tmp_path)
    freeze_protocol(tmp_path / "eval", first)
    second = tmp_path / "protocol2.json"
    second.write_text(json.dumps(_protocol(title="another")), encoding="utf-8")

    with pytest.raises(ProtocolError, match="already frozen"):
        freeze_protocol(tmp_path / "eval", second)


def test_nonobject_frozen_binding_is_invalid_without_raising(tmp_path: Path) -> None:
    from wenyi_eval.protocol import BINDING_NAME, verify_binding

    (tmp_path / BINDING_NAME).write_text("[]", encoding="utf-8")
    assert verify_binding(tmp_path)["status"] == "invalid"


def test_frozen_sampling_definition_must_match_the_run() -> None:
    from wenyi_eval.run_variants import _validate_protocol_groups

    data = _protocol()
    with pytest.raises(ProtocolError, match="seed"):
        _validate_protocol_groups(
            protocol=data,
            requested=("A_prime", "C"),
            seed=1,
            human_min=200,
            human_max=300,
        )
    with pytest.raises(ProtocolError, match="paragraph limits"):
        _validate_protocol_groups(
            protocol=data,
            requested=("A_prime", "C"),
            seed=20261116,
            human_min=100,
            human_max=300,
        )
    with pytest.raises(ProtocolError, match="not selected"):
        _validate_protocol_groups(
            protocol=data,
            requested=("A_prime",),
            seed=20261116,
            human_min=200,
            human_max=300,
        )
    _validate_protocol_groups(
        protocol=data,
        requested=("A_prime", "C"),
        seed=20261116,
        human_min=200,
        human_max=300,
    )
