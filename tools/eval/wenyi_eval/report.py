"""Markdown/CSV evaluation report with a conservative tri-state conclusion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import PathGuard, read_json, utc_now, write_json_atomic
from .manifest import SUPPORTED_GROUPS


def _value(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _rate(count: Any, denominator: Any) -> float | None:
    if not isinstance(count, int) or not isinstance(denominator, int) or denominator <= 0:
        return None
    return round(count / denominator, 4)


def _load_optional(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def _execution_check(
    execution: dict[str, Any], extraction: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    terminals = execution.get("terminals", {})
    preparation = terminals.get("preparation.prepare", {})
    if preparation.get("status") != "succeeded":
        reasons.append("preparation did not succeed")
    for group in SUPPORTED_GROUPS:
        for stage in ("translate", "status", "inspect"):
            terminal = terminals.get(f"{group}.{stage}", {})
            if terminal.get("status") != "succeeded":
                reasons.append(f"{group}.{stage} terminal is {terminal.get('status', 'missing')}")
        translate = terminals.get(f"{group}.translate", {})
        validation = translate.get("receipt_validation") or {}
        if validation.get("status") != "valid":
            reasons.append(f"{group} receipt is {validation.get('status', 'missing')}")
        coverage = extraction["groups"].get(group, {}).get("coverage") or {}
        expected = coverage.get("expected")
        if not isinstance(expected, int) or expected <= 0:
            reasons.append(f"{group} has no expected segments")
            continue
        if coverage.get("missing"):
            reasons.append(f"{group} has {coverage['missing']} missing targets")
        if coverage.get("stale"):
            reasons.append(f"{group} has {coverage['stale']} stale targets")
        if coverage.get("error"):
            reasons.append(f"{group} has {coverage['error']} unreadable target records")
        if coverage.get("translated", 0) + coverage.get("empty", 0) != expected:
            reasons.append(f"{group} coverage does not account for every expected segment")
        review = extraction["groups"].get(group, {}).get("review", {})
        if review.get("status") != "available":
            reasons.append(f"{group} review result is {review.get('status', 'missing')}")
    if not execution.get("source_unchanged"):
        reasons.append("fixture source changed")
    if not execution.get("preparation_unchanged"):
        reasons.append("preparation snapshot changed")
    return not reasons, reasons


def _human_decision(
    *,
    root: Path,
    preregistration: dict[str, Any] | None,
    imported: dict[str, Any] | None,
) -> tuple[str, str]:
    if not imported:
        return "inconclusive", "No pairwise ratings have been imported."
    denominator = imported.get("denominator")
    rated = imported.get("rated_rows", 0)
    if not isinstance(denominator, int) or denominator <= 0:
        return "inconclusive", "Pairwise denominator is missing."
    if rated < denominator:
        return (
            "inconclusive",
            f"Only {rated} of {denominator} preregistered pairs have a preference rating; "
            "missing ratings are not imputed.",
        )
    rule = (preregistration or {}).get("decision_rule")
    if not isinstance(rule, dict):
        return "inconclusive", "No preregistered statistical decision rule is available."
    comparison = rule.get("comparison")
    preferred = rule.get("preferred")
    minimum = rule.get("min_rated")
    threshold = rule.get("min_share")
    if (
        comparison not in {item["id"] for item in (preregistration or {}).get("comparisons", [])}
        or preferred not in {"baseline", "candidate"}
        or not isinstance(minimum, int)
        or not isinstance(threshold, (int, float))
    ):
        return "inconclusive", "The preregistered decision rule is incomplete."
    ratings_path = root / "samples" / "pairwise_ratings.jsonl"
    key_path = root / "samples" / "pairwise_key.csv"
    if not ratings_path.is_file() or not key_path.is_file():
        return "inconclusive", "Imported ratings or the pairwise key are missing."
    key = {}
    with open(key_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] == comparison:
                key[row["pair_id"]] = (row["left_group"], row["right_group"])
    candidate_votes = baseline_votes = decisive = 0
    with open(ratings_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["pair_id"] not in key or row["preference"] not in {"left", "right"}:
                continue
            left, right = key[row["pair_id"]]
            baseline_group = next(
                item["baseline"]
                for item in preregistration["comparisons"]
                if item["id"] == comparison
            )
            winner = left if row["preference"] == "left" else right
            decisive += 1
            if winner == baseline_group:
                baseline_votes += 1
            else:
                candidate_votes += 1
    if decisive < minimum:
        return "inconclusive", f"Only {decisive} decisive ratings; {minimum} required."
    share = candidate_votes / decisive
    winner_is_candidate = share >= threshold
    selected = "candidate" if winner_is_candidate else "baseline"
    passed = selected == preferred
    return (
        "passed" if passed else "failed",
        f"Preregistered rule met: {selected} share={share:.3f} (threshold {threshold}) "
        f"over {decisive} decisive ratings.",
    )


def _group_rows(
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    execution: dict[str, Any],
    cost: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    terminals = execution.get("terminals", {})
    compatibility = (execution.get("preparation_compatibility") or {}).get("groups", {})
    for group in SUPPORTED_GROUPS:
        plan = manifest["groups"][group]
        extracted = extraction["groups"].get(group, {})
        coverage = extracted.get("coverage") or {}
        usage = cost["groups"].get(group, {})
        tokens = usage.get("tokens") or {}
        terminal = terminals.get(f"{group}.translate", {})
        validation = terminal.get("receipt_validation") or {}
        review = extracted.get("review") or {}
        expected = coverage.get("expected")
        missing = coverage.get("missing")
        stale = coverage.get("stale")
        translated = coverage.get("translated")
        empty = coverage.get("empty")
        failed = (
            max(0, expected - translated - empty)
            if all(isinstance(value, int) for value in (expected, translated, empty))
            else None
        )
        rows.append(
            {
                "group": group,
                "arm": plan["arm"],
                "code_sha": plan["git"]["head_sha"],
                "binding_status": plan["binding_status"],
                "preparation_status": (compatibility.get(group) or {}).get("status"),
                "preparation_cost": (compatibility.get(group) or {}).get("cost"),
                "translate": terminal.get("status"),
                "status_stage": terminals.get(f"{group}.status", {}).get("status"),
                "inspect_stage": terminals.get(f"{group}.inspect", {}).get("status"),
                "receipt": validation.get("status"),
                "expected": expected,
                "translated": coverage.get("translated"),
                "empty": coverage.get("empty"),
                "missing": missing,
                "stale": stale,
                "translated_rate": _rate(coverage.get("translated"), expected),
                "empty_rate": _rate(coverage.get("empty"), expected),
                "missing_rate": _rate(missing, expected),
                "stale_rate": _rate(stale, expected),
                "failed": failed,
                "failed_rate": _rate(failed, expected),
                "review_status": review.get("status"),
                "review_issues": review.get("issues"),
                "calls": tokens.get("calls"),
                "prompt_tokens": tokens.get("prompt_tokens"),
                "completion_tokens": tokens.get("completion_tokens"),
                "cache_hit_tokens": tokens.get("cache_hit_tokens"),
                "cost_status": usage.get("cost", {}).get("status"),
                "cost": usage.get("cost", {}).get("amount"),
                "wall_seconds": (
                    execution.get("groups", {}).get(group, {}).get("durations") or {}
                ).get("translate"),
            }
        )
    return rows


def write_report(
    *,
    root: Path,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    execution: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    guard = PathGuard(root)
    report_dir = guard.child("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    selection = _load_optional(root / "samples" / "selection.json")
    preregistration = _load_optional(root / "samples" / "pairwise_preregistration.json")
    imported = _load_optional(root / "samples" / "pairwise_import.json")
    execution_ok, execution_reasons = _execution_check(execution, extraction)
    if not execution_ok:
        conclusion = "failed"
        decision_reason = "Execution or coverage checks failed: " + "; ".join(execution_reasons)
    elif manifest.get("diagnostic"):
        conclusion = "inconclusive"
        decision_reason = (
            "Diagnostic offline run: the external model boundary was replaced by a deterministic "
            "adapter, so outputs and costs cannot support a quality conclusion."
        )
    else:
        conclusion, decision_reason = _human_decision(
            root=root, preregistration=preregistration, imported=imported
        )
    rows = _group_rows(manifest, extraction, execution, cost)
    _write_group_csv(report_dir / "group_summary.csv", rows)
    _write_human_csv(
        report_dir / "human_summary.csv",
        selection=selection,
        preregistration=preregistration,
        imported=imported,
        decision_reason=decision_reason,
    )
    report = {
        "protocol_version": 1,
        "generated_at": utc_now(),
        "eval_id": manifest["eval_id"],
        "book": manifest["book"],
        "mode": manifest["mode"],
        "diagnostic": manifest["diagnostic"],
        "conclusion": conclusion,
        "decision_reason": decision_reason,
        "execution_ok": execution_ok,
        "execution_reasons": execution_reasons,
        "preparation_compatibility": execution.get("preparation_compatibility"),
        "groups": rows,
        "coverage": {
            group: extraction["groups"].get(group, {}).get("coverage") for group in SUPPORTED_GROUPS
        },
        "review": {
            group: extraction["groups"].get(group, {}).get("review") for group in SUPPORTED_GROUPS
        },
        "reflection": {
            group: extraction["groups"].get(group, {}).get("reflection")
            for group in SUPPORTED_GROUPS
        },
        "cost": {
            "price_status": cost["price_status"],
            "price_date": cost["price_date"],
            "aggregate": cost["aggregate_cost"],
            "aggregate_tokens": cost["aggregate_tokens"],
        },
        "human": {
            "sample": selection,
            "pairwise": preregistration,
            "imported": imported,
        },
        "unknowns": manifest.get("unknowns", []),
    }
    write_json_atomic(report_dir / "report.json", report)
    markdown = _markdown(
        manifest, report, cost, execution, extraction, selection, preregistration, imported
    )
    (report_dir / "summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    return report


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "arm",
        "code_sha",
        "binding_status",
        "preparation_status",
        "preparation_cost",
        "translate",
        "status_stage",
        "inspect_stage",
        "receipt",
        "expected",
        "translated",
        "empty",
        "missing",
        "stale",
        "translated_rate",
        "empty_rate",
        "missing_rate",
        "stale_rate",
        "failed",
        "failed_rate",
        "review_status",
        "review_issues",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "cache_hit_tokens",
        "cost_status",
        "cost",
        "wall_seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_human_csv(
    path: Path,
    *,
    selection: dict[str, Any] | None,
    preregistration: dict[str, Any] | None,
    imported: dict[str, Any] | None,
    decision_reason: str,
) -> None:
    fields = ["metric", "value", "denominator", "status"]
    rows = [
        {
            "metric": "suggested_strata_rows",
            "value": (selection or {}).get("selected"),
            "denominator": (selection or {}).get("eligible_paragraphs"),
            "status": "unreviewed_suggestion",
        },
        {
            "metric": "human_review_executed",
            "value": (selection or {}).get("human_review_executed"),
            "denominator": "",
            "status": "not_evidence",
        },
        {
            "metric": "pairwise_pairs",
            "value": (preregistration or {}).get("expected_pairs"),
            "denominator": (preregistration or {}).get("denominator"),
            "status": "preregistered",
        },
        {
            "metric": "pairwise_rated",
            "value": (imported or {}).get("rated_rows"),
            "denominator": (imported or {}).get("denominator"),
            "status": "missing" if imported is None else "imported",
        },
        {
            "metric": "decision",
            "value": decision_reason,
            "denominator": "",
            "status": "inconclusive",
        },
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_value(value) for value in row) + " |")
    return lines


def _markdown(
    manifest: dict[str, Any],
    report: dict[str, Any],
    cost: dict[str, Any],
    execution: dict[str, Any],
    extraction: dict[str, Any],
    selection: dict[str, Any] | None,
    preregistration: dict[str, Any] | None,
    imported: dict[str, Any] | None,
) -> str:
    lines = [
        f"# Wenyi evaluation {manifest['eval_id']}",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report["decision_reason"],
        "",
        "## Identity",
        "",
        f"- Book: `{manifest['book']}`",
        f"- Mode: `{manifest['mode']}` (diagnostic={manifest['diagnostic']})",
        f"- Source SHA-256: `{manifest['source']['sha256']}`",
        f"- Fixture license: {manifest['fixture']['license']}",
        f"- Fixture provenance: {manifest['fixture']['provenance']}",
        f"- Seed: {manifest['llm']['seed']}",
        f"- External provider: {manifest['llm']['external_provider']}",
        f"- Adapter: {manifest['llm']['adapter']}",
        f"- Price status/date: {_value(cost['price_status'])} / {_value(cost['price_date'])}",
        "",
        "## Execution",
        "",
    ]
    lines.extend(
        _table(
            [
                "Group",
                "Arm",
                "Code SHA",
                "Binding",
                "Preparation",
                "Translate",
                "Status",
                "Inspect",
                "Receipt",
            ],
            [
                [
                    row["group"],
                    row["arm"],
                    row["code_sha"][:12],
                    row["binding_status"],
                    row["preparation_status"],
                    row["translate"],
                    row["status_stage"],
                    row["inspect_stage"],
                    row["receipt"],
                ]
                for row in report["groups"]
            ],
        )
    )
    lines.extend(["", "## Coverage and translation", ""])
    lines.extend(
        _table(
            [
                "Group",
                "Expected",
                "Translated",
                "Empty",
                "Missing",
                "Stale",
                "Missing rate",
                "Failed rate",
                "Stale rate",
                "Review",
                "Issues",
            ],
            [
                [
                    row["group"],
                    row["expected"],
                    row["translated"],
                    row["empty"],
                    row["missing"],
                    row["stale"],
                    row["missing_rate"],
                    row["failed_rate"],
                    row["stale_rate"],
                    row["review_status"],
                    row["review_issues"],
                ]
                for row in report["groups"]
            ],
        )
    )
    lines.extend(
        [
            "",
            'An empty target (`""`) is preserved and reported separately from a missing '
            "target (`null`). The failed rate is (expected - translated - empty) / expected; "
            "extra stale rows never inflate it. Missing and stale keys are never silently "
            "dropped.",
            "",
            "## Cost and cache",
            "",
            "Preparation usage is recorded once from the preparation run and is excluded from "
            "the variant snapshot. Unknown prices are reported as unknown, never zero.",
            "",
        ]
    )
    lines.extend(
        _table(
            [
                "Group",
                "Calls",
                "Input",
                "Output",
                "Cache read",
                "Reasoning",
                "Cost",
                "Cost status",
                "Wall s",
            ],
            [
                [
                    row["group"],
                    row["calls"],
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    row["cache_hit_tokens"],
                    (cost["groups"].get(row["group"]) or {}).get("reasoning_tokens"),
                    row["cost"],
                    row["cost_status"],
                    row["wall_seconds"],
                ]
                for row in report["groups"]
            ],
        )
    )
    prep = cost["groups"].get("preparation", {})
    prep_tokens = prep.get("tokens") or {}
    lines.extend(
        [
            "",
            f"Preparation calls: {_value(prep_tokens.get('calls'))}; "
            f"input {_value(prep_tokens.get('prompt_tokens'))}; "
            f"output {_value(prep_tokens.get('completion_tokens'))}; "
            f"cost {_value((prep.get('cost') or {}).get('amount'))} "
            f"({_value((prep.get('cost') or {}).get('status'))}).",
            "",
            "## Review and reflection",
            "",
        ]
    )
    lines.extend(
        _table(
            ["Group", "Review status", "Issues", "Reviewed segments", "Reflection"],
            [
                [
                    row["group"],
                    row["review_status"],
                    row["review_issues"],
                    (report["review"].get(row["group"]) or {}).get("reviewed_segments"),
                    (report["reflection"].get(row["group"]) or {}).get("status"),
                ]
                for row in report["groups"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "Reflection is reported as unsupported/unavailable with WP2 artifacts absent; "
            "it is never treated as a clean metric.",
            "",
            "## Human evaluation",
            "",
            f"- Suggested stratified rows: {_value((selection or {}).get('selected'))} "
            f"of {_value((selection or {}).get('eligible_paragraphs'))} eligible; "
            f"reviewed={_value((selection or {}).get('human_review_executed'))}.",
            f"- Pairwise preregistered pairs: {_value((preregistration or {}).get('expected_pairs'))}; "
            f"rated rows: {_value((imported or {}).get('rated_rows'))}.",
            "- Suggested strata and pairwise rows are not counted as human evidence. "
            "Missing ratings remain missing.",
            "- Pairwise ratings use anonymous side labels and a separate key; this is not "
            "described as double blind.",
            "",
            "## Limitations and unknowns",
            "",
        ]
    )
    for item in manifest.get("unknowns", []):
        lines.append(f"- {item}")
    if report["execution_reasons"]:
        lines.append("- Execution check details: " + "; ".join(report["execution_reasons"]))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `manifest.json` — identities, bindings and absolute paths",
            "- `records/terminals/` — one exclusive terminal record per group and stage",
            "- `extraction.json` — expected-key aligned read-only extraction",
            "- `samples/strata.jsonl` — suggested human-review rows (unreviewed)",
            "- `samples/pairwise.csv` / `samples/pairwise_key.csv` — anonymous pairs and key",
            "- `report/cost.json` / `report/stage_costs.csv` — operation and stage accounting",
            "- `report/group_summary.csv` / `report/human_summary.csv` — CSV summaries",
            "",
        ]
    )
    return "\n".join(lines)
