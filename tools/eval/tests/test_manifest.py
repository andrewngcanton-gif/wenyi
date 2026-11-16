from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.helpers import CANDIDATE_SHA, init_git_repo, minimal_manifest
from wenyi_eval.common import EvalError, UnsafePathError
from wenyi_eval.manifest import (
    BASELINE_SHA,
    build_group_plans,
    require_real_ready,
    validate_manifest,
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

    assert plans["A"]["arm"] == "baseline"
    assert plans["A"]["expected_head_sha"] == BASELINE_SHA
    assert plans["A_prime"]["arm"] == "candidate"
    assert plans["A_prime"]["expected_head_sha"] == "unfrozen"
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
            diagnostic=True,
            baseline_binding={"adapter": "offline"},
            candidate_binding={"adapter": "offline"},
        )
