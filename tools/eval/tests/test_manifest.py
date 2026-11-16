from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.helpers import CANDIDATE_SHA, init_git_repo, minimal_manifest
from wenyi_eval.common import EvalError, UnsafePathError
from wenyi_eval.manifest import (
    BASELINE_SHA,
    DEFAULT_GROUPS,
    WP0_SHA,
    build_group_plans,
    normalize_groups,
    reflection_plan,
    require_real_ready,
    validate_manifest,
    variant_config,
    write_manifest,
)


def test_valid_manifest_keeps_group_paths_isolated(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")

    validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))

    paths = [group["state_dir"] for group in manifest["groups"].values()]
    assert len(paths) == len(set(paths))


def test_manifest_rejects_duplicate_state_ownership(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")
    manifest["groups"]["A2"]["state_dir"] = manifest["groups"]["A"]["state_dir"]

    with pytest.raises(EvalError, match="share state_dir"):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_manifest_rejects_paths_outside_the_disposable_root(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")
    manifest["groups"]["B"]["output_dir"] = str(tmp_path / "outside")

    with pytest.raises(UnsafePathError):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_manifest_rejects_real_user_state_even_under_a_repo(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root)
    repo_root = root / "repo"
    manifest["groups"]["A"]["state_dir"] = str(repo_root / "state")

    with pytest.raises(UnsafePathError, match="non-disposable user data"):
        validate_manifest(manifest, repo_root=repo_root)


def test_manifest_rejects_tampered_source_hash(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")
    manifest["source"]["sha256"] = "0" * 64

    with pytest.raises(EvalError, match="source hash"):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_manifest_requires_explicit_price_status(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")
    manifest["prices"] = {}

    with pytest.raises(EvalError, match="Prices"):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_diagnostic_manifest_requires_a_comparability_reason(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval", diagnostic=True)
    manifest["comparability"]["reason"] = None

    with pytest.raises(EvalError, match="not comparable"):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_real_run_requires_clean_pinned_groups_and_known_prices(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    known_prices = {
        "status": "known",
        "date": "2026-11-16",
        "currency": "USD",
        "unit": 1_000_000,
        "providers": {"default": {"input": 1.0, "output": 2.0}},
    }
    clean = minimal_manifest(root, prices=known_prices)
    require_real_ready(clean)

    dirty = copy.deepcopy(clean)
    dirty["groups"]["A_prime"]["binding_status"] = "dirty"
    with pytest.raises(EvalError, match="A_prime:dirty"):
        require_real_ready(dirty)

    unfrozen = minimal_manifest(
        tmp_path / "unfrozen", prices=known_prices, candidate_binding="unfrozen"
    )
    assert unfrozen["groups"]["A_prime"]["git"]["head_sha"] == CANDIDATE_SHA
    with pytest.raises(EvalError, match="A_prime:unfrozen"):
        require_real_ready(unfrozen)

    unknown_prices = minimal_manifest(tmp_path / "unknown")
    # Missing prices leave cost unknown instead of blocking non-cost tooling.
    require_real_ready(unknown_prices)

    unknown_routes = copy.deepcopy(clean)
    unknown_routes["groups"]["A_prime"]["routes"] = {"status": "unknown"}
    with pytest.raises(EvalError, match="routes_unknown"):
        require_real_ready(unknown_routes)

    unknown_fingerprint = copy.deepcopy(clean)
    unknown_fingerprint["groups"]["B"]["route_fingerprints"] = {"status": "unknown"}
    with pytest.raises(EvalError, match="route_fingerprint_unknown"):
        require_real_ready(unknown_fingerprint)

    diagnostic = minimal_manifest(tmp_path / "diagnostic", diagnostic=True)
    with pytest.raises(EvalError, match="diagnostic"):
        require_real_ready(diagnostic)


def test_manifest_requires_b_polish_to_be_disabled(tmp_path: Path) -> None:
    manifest = minimal_manifest(tmp_path / "eval")
    manifest["groups"]["B"]["polish"] = True

    with pytest.raises(EvalError, match="Group B must have polish disabled"):
        validate_manifest(manifest, repo_root=Path(manifest["repo"]["root"]))


def test_write_manifest_refuses_to_overwrite_an_evaluation(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    root.mkdir()
    manifest = minimal_manifest(root)
    write_manifest(root, manifest)

    with pytest.raises(EvalError, match="overwrite"):
        write_manifest(root, manifest)


def test_group_plans_separate_baseline_from_candidate_sources(tmp_path: Path) -> None:
    baseline = init_git_repo(tmp_path / "baseline")
    candidate = init_git_repo(tmp_path / "candidate")
    wp0 = init_git_repo(tmp_path / "wp0")
    root = tmp_path / "eval"
    root.mkdir()

    plans = build_group_plans(
        root,
        baseline_source=baseline,
        candidate_source=candidate,
        wp0_source=wp0,
        wp0_binding={"adapter": "offline"},
        candidate_expected_sha="unfrozen",
        diagnostic=True,
        baseline_binding={"adapter": "offline"},
        candidate_binding={"adapter": "offline"},
    )

    assert plans["A"]["arm"] == "baseline"
    assert plans["A"]["expected_head_sha"] == BASELINE_SHA
    assert plans["A_prime"]["arm"] == "candidate"
    assert plans["A_prime"]["expected_head_sha"] == WP0_SHA
    assert plans["A_prime"]["code_source"] == str(wp0.resolve())
    assert plans["A"]["code_source"] != plans["A_prime"]["code_source"]
    assert plans["A"]["state_dir"] != plans["A2"]["state_dir"]


def test_group_plans_reject_importing_the_working_tree_as_baseline(tmp_path: Path) -> None:
    source = init_git_repo(tmp_path / "source")
    root = tmp_path / "eval"
    root.mkdir()

    with pytest.raises(EvalError, match="distinct code sources"):
        build_group_plans(
            root,
            baseline_source=source,
            candidate_source=source,
            candidate_expected_sha="unfrozen",
            requested_groups=("A",),
            diagnostic=True,
            baseline_binding={"adapter": "offline"},
            candidate_binding={"adapter": "offline"},
        )


@pytest.mark.parametrize("group", ["A_prime", "A_prime2", "B"])
def test_wp0_group_never_falls_back_to_current_candidate(tmp_path: Path, group: str) -> None:
    baseline = init_git_repo(tmp_path / "baseline")
    candidate = init_git_repo(tmp_path / "candidate")
    with pytest.raises(EvalError, match="WP0 source"):
        build_group_plans(
            tmp_path / "eval",
            baseline_source=baseline,
            candidate_source=candidate,
            candidate_expected_sha="unfrozen",
            diagnostic=True,
            requested_groups=(group,),
            candidate_binding={"adapter": "offline"},
        )


def _routing_base() -> dict:
    return {
        "llm": {
            "preset": None,
            "providers": {
                "stub": {
                    "kind": "openai-compatible",
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key_env": "STUB_KEY",
                }
            },
            "models": {
                "polisher": {"provider": "stub", "model": "polisher-model"},
                "critic": {"provider": "stub", "model": "critic-model"},
            },
            "tiers": {"strong": "polisher", "cheap": "polisher", "fast": "polisher"},
        },
        "pipeline": {"polish": True, "reflective_polish": True, "review": False},
    }


def test_group_selection_defaults_and_explicit_subset() -> None:
    assert normalize_groups(None) == DEFAULT_GROUPS
    assert normalize_groups("D,C,A") == ("A", "C", "D")
    assert normalize_groups(["C_keep", "C"]) == ("C", "C_keep")
    with pytest.raises(EvalError, match="unsupported"):
        normalize_groups("E")
    with pytest.raises(EvalError, match="At least one"):
        normalize_groups(" , ")


def test_control_configs_strip_wp2_only_fields_and_reflect_routes(tmp_path: Path) -> None:
    base = _routing_base()
    base["llm"]["routes"] = {"reflect.body": {"model": "critic"}}
    for group in ("A", "B"):
        config = variant_config(group, tmp_path / group / "state", base=base)
        pipeline = config["pipeline"]
        assert "reflective_polish" not in pipeline
        assert "reflect_min_severity" not in pipeline
        assert "reflect_clean_batches" not in pipeline
        assert "reflect.body" not in (config["llm"].get("routes") or {})
    assert variant_config("B", tmp_path / "B" / "state", base=base)["pipeline"]["polish"] is False


def test_reflective_configs_inherit_or_declare_a_distinct_route(tmp_path: Path) -> None:
    base = _routing_base()
    c = variant_config("C", tmp_path / "C" / "state", base=base)
    keep = variant_config("C_keep", tmp_path / "C_keep" / "state", base=base)
    d = variant_config("D", tmp_path / "D" / "state", base=base, reflect_model="critic")

    assert c["pipeline"]["reflective_polish"] is True
    assert c["pipeline"]["reflect_clean_batches"] == "polish"
    assert "reflect.body" not in (c["llm"].get("routes") or {})
    assert keep["pipeline"]["reflect_clean_batches"] == "keep"
    assert d["llm"]["routes"]["reflect.body"] == {"model": "critic"}
    assert reflection_plan("C", c)["route_separation"]["status"] == "inherited"
    separation = reflection_plan("D", d)["route_separation"]
    assert separation["status"] == "distinct"
    assert separation["reflect_model"] == "critic-model"
    assert separation["polish_model"] == "polisher-model"


def test_group_d_requires_an_explicit_distinct_model_profile(tmp_path: Path) -> None:
    base = _routing_base()
    with pytest.raises(EvalError, match="heterogeneous-reflect-model"):
        variant_config("D", tmp_path / "D" / "state", base=base)
    with pytest.raises(EvalError, match="not defined"):
        variant_config("D", tmp_path / "D" / "state", base=base, reflect_model="missing")
    same = variant_config("D", tmp_path / "same" / "state", base=base, reflect_model="polisher")
    with pytest.raises(EvalError, match="different model family"):
        reflection_plan("D", same)


def test_group_plans_bind_upstream_wp0_and_workspace_separately(tmp_path: Path) -> None:
    from wenyi_eval.common import git_identity

    baseline = init_git_repo(tmp_path / "upstream")
    wp0 = init_git_repo(tmp_path / "wp0")
    candidate = init_git_repo(tmp_path / "workspace")
    root = tmp_path / "eval"
    root.mkdir()

    plans = build_group_plans(
        root,
        baseline_source=baseline,
        wp0_source=wp0,
        candidate_source=candidate,
        wp0_expected_sha=git_identity(wp0).head_sha,
        candidate_expected_sha="unfrozen",
        diagnostic=True,
        requested_groups=("A", "A_prime", "B", "C", "C_keep", "D"),
        baseline_binding={"adapter": "offline"},
        wp0_binding={"adapter": "offline"},
        candidate_binding={"adapter": "offline"},
        base_config=_routing_base(),
        reflect_model="critic",
    )

    assert set(plans) == {"A", "A_prime", "B", "C", "C_keep", "D"}
    assert plans["A"]["source_role"] == "baseline"
    assert plans["A_prime"]["source_role"] == "wp0"
    assert plans["B"]["source_role"] == "wp0"
    assert plans["C"]["source_role"] == "candidate"
    assert plans["A_prime"]["code_source"] == str(wp0.resolve())
    assert plans["C"]["code_source"] == str(candidate.resolve())
    assert plans["C"]["reflection"]["clean_action"] == "polish"
    assert plans["C_keep"]["reflection"]["clean_action"] == "keep"
    assert plans["D"]["reflection"]["route_separation"]["status"] == "distinct"
    assert plans["A"]["reflection"]["enabled"] is False


def test_group_plans_reject_wp0_that_reuses_the_candidate(tmp_path: Path) -> None:
    baseline = init_git_repo(tmp_path / "upstream")
    candidate = init_git_repo(tmp_path / "workspace")
    root = tmp_path / "eval"
    root.mkdir()

    with pytest.raises(EvalError, match="distinct code sources"):
        build_group_plans(
            root,
            baseline_source=baseline,
            wp0_source=candidate,
            candidate_source=candidate,
            wp0_expected_sha="unfrozen",
            candidate_expected_sha="unfrozen",
            diagnostic=True,
            requested_groups=("A", "A_prime"),
            baseline_binding={"adapter": "offline"},
            wp0_binding={"adapter": "offline"},
            candidate_binding={"adapter": "offline"},
        )
