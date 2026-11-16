"""End-to-end offline smoke for the WP1a harness.

The smoke replaces only the external LLM boundary. It drives real Wenyi CLI
subprocesses, extracts state through public storage reads, exports and imports
pairwise material, writes cost/report artifacts and runs negative controls.

Diagnostic only: it never supports a translation-quality or provider-cost claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import (
    EvalError,
    git_identity,
    hash_tree,
    new_eval_id,
    ownership_lock,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .controls import run_negative_controls
from .cost import build_cost_summary
from .extract import extract_all
from .fixture import fixture_metadata, write_fixture
from .manifest import (
    BASELINE_REF,
    BASELINE_SHA,
    build_group_plans,
    build_manifest,
    load_manifest,
    write_variant_configs,
)
from .pairwise import export_pairwise, import_pairwise_ratings
from .report import write_report
from .run_variants import initialize_root, start_evaluation
from .sample import build_samples


def _find_baseline(repo_root: Path, override: str | None) -> Path:
    if override:
        candidate = Path(override).resolve()
        if not candidate.is_dir():
            raise EvalError(f"Baseline code source does not exist: {candidate}")
        return candidate
    for candidate in (repo_root / BASELINE_REF, repo_root.parent / BASELINE_REF):
        if candidate.is_dir():
            return candidate.resolve()
    raise EvalError(
        "Cannot find the read-only baseline checkout. Pass --baseline PATH; it must be at "
        f"{BASELINE_SHA}."
    )


def _unknown_notes() -> list[str]:
    return [
        "The A-prime candidate code SHA is not frozen; this smoke is diagnostic and not a "
        "comparable baseline.",
        "WP0 edits are uncommitted; the candidate binding is unfrozen/dirty and cannot be "
        "used for a real comparison.",
        "The external LLM provider boundary is replaced by a deterministic offline adapter.",
        "No licensed corpora, human ratings, or provider price rules are included.",
        "WP2 reflection artifacts are unavailable; reflection is unsupported, not clean.",
        "The smoke validates harness behavior, path isolation and evidence handling, not "
        "translation quality, model behavior or provider cost.",
    ]


def run_smoke(
    *,
    runs_dir: Path,
    eval_id: str,
    baseline: str | None,
    timeout: float,
) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    baseline_source = _find_baseline(repo_root, baseline)
    baseline_identity = git_identity(baseline_source)
    if baseline_identity.head_sha != BASELINE_SHA:
        raise EvalError(
            f"Baseline checkout is at {baseline_identity.head_sha}, expected {BASELINE_SHA}"
        )
    baseline_tree_before = hash_tree(baseline_source / "packages")

    book = "lantern_harbor.en"
    root = initialize_root(runs_dir, eval_id=eval_id, book=book)
    with ownership_lock(root):
        source, fixture_path = write_fixture(root / "source")
        config_hashes = write_variant_configs(root)
        plans = build_group_plans(
            root,
            baseline_source=baseline_source,
            candidate_source=repo_root,
            candidate_expected_sha="unfrozen",
            diagnostic=True,
            baseline_binding={"adapter": "offline"},
            candidate_binding={"adapter": "offline"},
        )
        fixture = fixture_metadata()
        manifest = build_manifest(
            root=root,
            runs_dir=Path(runs_dir).resolve(),
            eval_id=eval_id,
            book=book,
            source_path=source,
            source_format="markdown",
            language={"source": "en", "target": "zh"},
            fixture=fixture,
            plans=plans,
            config_hashes=config_hashes,
            mode="offline_smoke",
            diagnostic=True,
            repo_root=repo_root,
            seed=20261116,
            prices={
                "status": "unknown",
                "date": None,
                "currency": None,
                "unit": 1_000_000,
                "source": None,
                "reason": "No provider price rule is bundled with the offline smoke.",
            },
            preparation_entry=f"{plans['A_prime']['code_source']} wenyi prepare",
            unknown_notes=_unknown_notes(),
        )
        artifact = start_evaluation(
            root=root,
            source=source,
            manifest=manifest,
            repo_root=repo_root,
            timeout=timeout,
        )
        loaded = load_manifest(root / "manifest.json")
        source_hash = loaded["source"]["sha256"]
        if sha256_file(source) != source_hash:
            raise EvalError("Fixture source changed during the smoke")
        extraction = extract_all(root=root, manifest=loaded)
        selection = build_samples(root=root, manifest=loaded, extraction=extraction)
        pairwise = export_pairwise(root=root, manifest=loaded, extraction=extraction)
        imported = import_pairwise_ratings(root, root / "samples" / "pairwise.csv")
        cost = build_cost_summary(
            root=root,
            manifest=loaded,
            extraction=extraction,
            execution=artifact["execution"],
        )
        report = write_report(
            root=root,
            manifest=loaded,
            extraction=extraction,
            execution=artifact["execution"],
            cost=cost,
        )
        controls = run_negative_controls(root, loaded)
        baseline_tree_after = hash_tree(baseline_source / "packages")
        baseline_status_after = git_identity(baseline_source)
        baseline_unchanged = (
            baseline_tree_before == baseline_tree_after
            and baseline_status_after.status_sha256 == baseline_identity.status_sha256
        )
        result = {
            "protocol_version": 1,
            "created_at": utc_now(),
            "eval_id": eval_id,
            "root": str(root),
            "source": str(source),
            "fixture_metadata": str(fixture_path),
            "manifest": str(root / "manifest.json"),
            "extraction": str(root / "extraction.json"),
            "samples": str(root / "samples" / "strata.jsonl"),
            "pairwise": str(root / "samples" / "pairwise.csv"),
            "pairwise_key": str(root / "samples" / "pairwise_key.csv"),
            "pairwise_import": str(root / "samples" / "pairwise_import.json"),
            "cost": str(root / "report" / "cost.json"),
            "report": str(root / "report" / "summary.md"),
            "controls": str(root / "_controls" / "controls.json"),
            "selection": selection,
            "pairwise_preregistration": {
                "expected_pairs": pairwise["expected_pairs"],
                "denominator": pairwise["denominator"],
            },
            "blank_import": {
                "submitted_rows": imported["submitted_rows"],
                "rated_rows": imported["rated_rows"],
                "blank_ratings": imported["blank_ratings"],
                "denominator": imported["denominator"],
            },
            "conclusion": report["conclusion"],
            "execution_ok": report["execution_ok"],
            "execution_reasons": report["execution_reasons"],
            "negative_controls": controls,
            "baseline_unchanged": baseline_unchanged,
            "baseline_tree_sha256": baseline_tree_before,
            "terminals": {
                key: row["status"] for key, row in artifact["execution"]["terminals"].items()
            },
        }
        write_json_atomic(root / "smoke.json", result)
        if not baseline_unchanged:
            raise EvalError("The read-only baseline checkout changed during the smoke")
        if not controls["passed"]:
            raise EvalError("One or more negative controls failed")
        if not report["execution_ok"]:
            raise EvalError("Execution checks failed; see report/summary.md")
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wenyi_eval.smoke")
    parser.add_argument("--offline", action="store_true", help="Use only the offline adapter")
    parser.add_argument("--runs-dir", required=True, help="Absolute disposable runs directory")
    parser.add_argument("--eval-id", default=None, help="New evaluation id (default: timestamp)")
    parser.add_argument("--baseline", default=None, help="Read-only baseline checkout")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="Per-command timeout in seconds"
    )
    args = parser.parse_args(argv)
    if not args.offline:
        print(
            "Refusing to run: only --offline smoke is implemented; no paid or external model "
            "evaluation is authorized.",
            file=sys.stderr,
        )
        return 2
    eval_id = args.eval_id or new_eval_id()
    try:
        result = run_smoke(
            runs_dir=Path(args.runs_dir),
            eval_id=eval_id,
            baseline=args.baseline,
            timeout=args.timeout,
        )
    except EvalError as error:
        print(f"Smoke failed: {error}", file=sys.stderr)
        return 1
    print(f"Eval root: {result['root']}")
    print(f"Manifest: {result['manifest']}")
    print(f"Report: {result['report']}")
    print(f"Conclusion: {result['conclusion']} (diagnostic, not a quality result)")
    print(
        "Terminals: "
        + ", ".join(f"{key}={value}" for key, value in sorted(result["terminals"].items()))
    )
    print(
        "Samples: "
        f"{result['selection']['selected']} suggested human-review rows; "
        f"pairwise denominator {result['pairwise_preregistration']['denominator']}; "
        f"rated rows {result['blank_import']['rated_rows']} (ratings missing by design)."
    )
    print(
        "Negative controls: "
        + ("passed" if result["negative_controls"]["passed"] else "FAILED")
        + f" ({len(result['negative_controls']['controls'])} checks)"
    )
    print(f"Baseline unchanged: {result['baseline_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
