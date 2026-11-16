"""Markdown/CSV evaluation report with a conservative tri-state conclusion."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import PathGuard, read_json, utc_now, write_json_atomic
from .gates import evaluate_gates, load_human_evidence
from .protocol import BINDING_NAME, verify_binding


def _value(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _rate(count: Any, denominator: Any) -> float | None:
    if not isinstance(count, int) or not isinstance(denominator, int) or denominator <= 0:
        return None
    return round(count / denominator, 4)


def _load_optional(path: Path) -> dict[str, Any] | None:
    data = read_json(path) if path.is_file() else None
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return data


def _execution_check(
    execution: dict[str, Any], extraction: dict[str, Any], groups: list[str]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    terminals = execution.get("terminals", {})
    preparation = terminals.get("preparation.prepare", {})
    if preparation.get("status") != "succeeded":
        reasons.append("preparation did not succeed")
    for group, record in (execution.get("isolated_preparation") or {}).items():
        path = record.get("terminal_path")
        status = None
        if isinstance(path, str) and Path(path).is_file():
            status = (read_json(path) or {}).get("status")
        if status != "succeeded":
            reasons.append(f"isolated preparation for {group} is {status or 'missing'}")
    for group in groups:
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
        reflection = extraction["groups"].get(group, {}).get("reflection") or {}
        if reflection.get("enabled"):
            segments = reflection.get("segments") or {}
            uncovered = sum(
                segments.get(status, 0) for status in ("missing", "stale", "invalid", "ambiguous")
            )
            if uncovered:
                reasons.append(
                    f"{group} has {uncovered} segments without a current reflection artifact"
                )
            if (
                reflection.get("status") == "current"
                and segments.get("expected", 0) > 0
                and not reflection.get("adoption_counts")
            ):
                reasons.append(f"{group} reflection evidence has no adoption records")
    if not execution.get("source_unchanged"):
        reasons.append("fixture source changed")
    if not execution.get("preparation_unchanged"):
        reasons.append("preparation snapshot changed")
    return not reasons, reasons


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _protocol_timing(
    *, root: Path, manifest: dict[str, Any], protocol_state: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Detect a binding produced after the manifest or after execution outputs."""
    reasons: list[str] = []
    binding_path = Path(root) / BINDING_NAME
    if protocol_state.get("status") == "missing":
        return False, reasons
    manifest_protocol = manifest.get("protocol")
    if not isinstance(manifest_protocol, dict) or manifest_protocol.get("status") != "valid":
        reasons.append("the manifest does not record a frozen protocol identity")
    elif (
        manifest_protocol.get("content_sha256") != protocol_state.get("content_sha256")
        or manifest_protocol.get("content_sha256") is None
    ):
        reasons.append("the manifest protocol hash differs from the frozen binding")
    created = _parse_time(manifest.get("created_at"))
    frozen = _parse_time(protocol_state.get("frozen_at"))
    if created is not None and frozen is not None and frozen > created:
        reasons.append("the protocol was frozen after the manifest was created")
    execution_path = Path(root) / "records" / "execution.json"
    if binding_path.is_file() and execution_path.is_file():
        if binding_path.stat().st_mtime > execution_path.stat().st_mtime + 2:
            reasons.append("the protocol binding was written after translation outputs")
    data = protocol_state.get("protocol") or {}
    groups = set((manifest.get("groups") or {}).keys())
    for item in data.get("comparisons", []):
        if item.get("role") != "confirmatory":
            continue
        if item.get("baseline") not in groups or item.get("candidate") not in groups:
            reasons.append(f"{item.get('id')} names a group that this run did not execute")
    sampling = data.get("sampling") or {}
    llm = manifest.get("llm") or {}
    if sampling.get("seed") != llm.get("seed"):
        reasons.append("the frozen sampling seed differs from the executed seed")
    manifest_sampling = manifest.get("sampling") or {}
    if manifest_sampling and (
        sampling.get("min_paragraphs") != manifest_sampling.get("min")
        or sampling.get("max_paragraphs") != manifest_sampling.get("max")
    ):
        reasons.append("the frozen paragraph limits differ from the executed sampling limits")
    return bool(reasons), reasons


def _packet_binding_reasons(
    *,
    protocol_state: dict[str, Any],
    preregistration: dict[str, Any] | None,
    edit_preregistration: dict[str, Any] | None,
) -> list[str]:
    if protocol_state.get("status") != "valid":
        return []
    digest = protocol_state.get("content_sha256")
    reasons: list[str] = []
    for label, packet in (
        ("pairwise", preregistration),
        ("edit", edit_preregistration),
    ):
        if packet is None:
            continue
        if packet.get("protocol_content_sha256") != digest:
            reasons.append(
                f"the {label} packet was exported from a different protocol content hash"
            )
    protocol = protocol_state.get("protocol") or {}
    if preregistration:
        exported = {item.get("id") for item in preregistration.get("comparisons", [])}
        required = {
            item["id"]
            for item in protocol.get("comparisons", [])
            if item.get("role") == "confirmatory"
        }
        if not required <= exported:
            reasons.append("the pairwise packet is missing a confirmatory comparison")
    return reasons


def _decision_reason(
    *,
    conclusion: str,
    gates: dict[str, Any],
    manifest: dict[str, Any],
    imported: dict[str, Any] | None,
) -> str:
    failed = gates.get("failed_required") or []
    inconclusive = gates.get("inconclusive_required") or []
    if conclusion == "passed":
        return "Every required pre-registered gate passed on frozen non-synthetic evidence."
    if conclusion == "failed" and failed:
        return "Blocking gate(s) failed: " + ", ".join(failed)
    reasons = []
    if manifest.get("diagnostic"):
        reasons.append(
            "Diagnostic offline run: the external model boundary was replaced by a "
            "deterministic adapter, so outputs and costs cannot support a quality conclusion."
        )
    if gates.get("protocol_status") == "missing":
        reasons.append(
            "No frozen experiment protocol exists; this remains a diagnostic/legacy run."
        )
    if not imported:
        reasons.append("No pairwise ratings have been imported.")
    if inconclusive:
        reasons.append("Inconclusive required gate(s): " + ", ".join(inconclusive))
    if not reasons:
        reasons.append("Required evidence is incomplete; the quality claim is inconclusive.")
    return " ".join(reasons)


def _group_rows(
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    execution: dict[str, Any],
    cost: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    terminals = execution.get("terminals", {})
    compatibility = (execution.get("preparation_compatibility") or {}).get("groups", {})
    for group in manifest["groups"]:
        plan = manifest["groups"][group]
        extracted = extraction["groups"].get(group, {})
        coverage = extracted.get("coverage") or {}
        usage = cost["groups"].get(group, {})
        tokens = usage.get("tokens") or {}
        terminal = terminals.get(f"{group}.translate", {})
        validation = terminal.get("receipt_validation") or {}
        review = extracted.get("review") or {}
        reflection = extracted.get("reflection") or {}
        reflection_segments = reflection.get("segments") or {}
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
                "source_role": plan.get("source_role"),
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
                "reflection_status": reflection.get("status", "not_enabled"),
                "reflection_enabled": bool(reflection.get("enabled")),
                "reflection_current_batches": (reflection.get("artifacts") or {}).get("current"),
                "reflection_stale_batches": (reflection.get("artifacts") or {}).get("stale"),
                "reflection_invalid_batches": (reflection.get("artifacts") or {}).get("invalid"),
                "reflection_ambiguous_batches": (reflection.get("artifacts") or {}).get(
                    "ambiguous"
                ),
                "reflection_current_segments": reflection_segments.get("current"),
                "reflection_uncovered_segments": sum(
                    reflection_segments.get(status, 0)
                    for status in ("missing", "stale", "invalid", "ambiguous")
                ),
                "reflection_write_failed_events": (reflection.get("artifacts") or {}).get(
                    "write_failed_events"
                ),
                "reflection_adoption": reflection.get("adoption_counts"),
                "reflection_critique": reflection.get("critique_counts"),
                "reflection_failure": reflection.get("failure_counts"),
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
    from .annotation_import import verify_imports

    annotation_binding = verify_imports(root, extraction)
    empty_evidence = {
        kind: {"preregistration": None, "import": None, "ratings": [], "key": {}}
        for kind in ("pairwise", "edits")
    }
    try:
        selection = _load_optional(root / "samples" / "selection.json")
        preregistration = _load_optional(root / "samples" / "pairwise_preregistration.json")
        imported = _load_optional(root / "samples" / "pairwise_import.json")
        edit_import = _load_optional(root / "samples" / "edit_import.json")
        edit_preregistration = _load_optional(root / "samples" / "edit_preregistration.json")
        evidence = (
            load_human_evidence(root) if annotation_binding["status"] == "valid" else empty_evidence
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        annotation_binding = {
            "status": "invalid",
            "reason": f"Unreadable annotation evidence: {error}",
        }
        selection = preregistration = imported = edit_import = edit_preregistration = None
        evidence = empty_evidence
    groups = list(manifest["groups"])
    execution_ok, execution_reasons = _execution_check(execution, extraction, groups)
    protocol_state = verify_binding(root)
    protocol_late, protocol_reasons = _protocol_timing(
        root=root, manifest=manifest, protocol_state=protocol_state
    )
    protocol_reasons.extend(
        _packet_binding_reasons(
            protocol_state=protocol_state,
            preregistration=preregistration,
            edit_preregistration=edit_preregistration,
        )
    )
    gate_result = evaluate_gates(
        manifest=manifest,
        extraction=extraction,
        cost=cost,
        execution_ok=execution_ok,
        execution_reasons=execution_reasons,
        protocol_state=protocol_state,
        evidence=evidence,
        protocol_late=protocol_late,
        protocol_reasons=protocol_reasons,
        selection=selection,
        annotation_binding=annotation_binding,
    )
    if not execution_ok:
        conclusion = "failed"
        scope = "execution"
    elif gate_result["failed_required"]:
        conclusion = "failed"
        scope = "quality"
    elif manifest.get("diagnostic"):
        conclusion = "inconclusive"
        scope = "diagnostic"
    elif gate_result.get("protocol_status") == "missing":
        conclusion = "inconclusive"
        scope = "legacy"
    else:
        conclusion = gate_result["quality_conclusion"]
        scope = "quality"
    decision_reason = _decision_reason(
        conclusion=conclusion,
        gates=gate_result,
        manifest=manifest,
        imported=imported,
    )
    rows = _group_rows(manifest, extraction, execution, cost)
    _write_group_csv(report_dir / "group_summary.csv", rows)
    _write_human_csv(
        report_dir / "human_summary.csv",
        selection=selection,
        preregistration=preregistration,
        imported=imported,
        edit_import=edit_import,
        gate_result=gate_result,
        decision_reason=decision_reason,
    )
    report = {
        "protocol_version": 2,
        "generated_at": utc_now(),
        "eval_id": manifest["eval_id"],
        "book": manifest["book"],
        "mode": manifest["mode"],
        "diagnostic": manifest["diagnostic"],
        "conclusion": conclusion,
        "conclusion_scope": scope,
        "quality_conclusion": gate_result["quality_conclusion"],
        "empirical_evidence": gate_result["empirical_evidence"],
        "decision_reason": decision_reason,
        "protocol": {
            "status": protocol_state.get("status"),
            "reason": protocol_state.get("reason"),
            "content_sha256": protocol_state.get("content_sha256"),
            "late_binding": protocol_late,
            "binding_reasons": protocol_reasons,
            "manifest": manifest.get("protocol"),
        },
        "gates": gate_result["gates"],
        "failed_required_gates": gate_result["failed_required"],
        "inconclusive_required_gates": gate_result["inconclusive_required"],
        "execution_ok": execution_ok,
        "execution_reasons": execution_reasons,
        "preparation_compatibility": execution.get("preparation_compatibility"),
        "isolated_preparation": {
            group: {
                "snapshot": record.get("snapshot_dir"),
                "terminal": record.get("terminal_path"),
                "cost": record.get("cost"),
                "usage": record.get("usage"),
            }
            for group, record in (execution.get("isolated_preparation") or {}).items()
        },
        "groups": rows,
        "coverage": {
            group: extraction["groups"].get(group, {}).get("coverage") for group in groups
        },
        "review": {group: extraction["groups"].get(group, {}).get("review") for group in groups},
        "reflection": {
            group: extraction["groups"].get(group, {}).get("reflection") for group in groups
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
            "edit_preregistration": edit_preregistration,
            "edit_imported": edit_import,
            "origin": "caller-reported, not identity-verified; not double blind",
        },
        "unknowns": manifest.get("unknowns", []),
    }
    write_json_atomic(report_dir / "report.json", report)
    markdown = _markdown(
        manifest,
        report,
        cost,
        execution,
        extraction,
        selection,
        preregistration,
        imported,
    )
    (report_dir / "summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    return report


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "arm",
        "source_role",
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
        "reflection_status",
        "reflection_enabled",
        "reflection_current_batches",
        "reflection_stale_batches",
        "reflection_invalid_batches",
        "reflection_ambiguous_batches",
        "reflection_current_segments",
        "reflection_uncovered_segments",
        "reflection_write_failed_events",
        "reflection_adoption",
        "reflection_critique",
        "reflection_failure",
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
    edit_import: dict[str, Any] | None,
    gate_result: dict[str, Any],
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
            "metric": "edited_segments",
            "value": (edit_import or {}).get("assessed_rows"),
            "denominator": (edit_import or {}).get("denominator"),
            "status": "missing" if edit_import is None else "imported",
        },
        {
            "metric": "quality_conclusion",
            "value": gate_result["quality_conclusion"],
            "denominator": "",
            "status": "gate_result",
        },
    ]
    for name, gate in gate_result.get("gates", {}).items():
        rows.append(
            {
                "metric": name,
                "value": gate.get("reason"),
                "denominator": "",
                "status": gate.get("status"),
            }
        )
    rows.append(
        {
            "metric": "decision",
            "value": decision_reason,
            "denominator": "",
            "status": gate_result["quality_conclusion"],
        }
    )
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
        f"**Conclusion: {report['conclusion']}** ({report['conclusion_scope']})",
        "",
        report["decision_reason"],
        "",
        f"- Workflow execution: {'passed' if report['execution_ok'] else 'failed'}",
        f"- Empirical quality: {report['quality_conclusion']} "
        f"(empirical evidence={report['empirical_evidence']})",
        "- A workflow that ran is not a quality result; diagnostic, synthetic and offline "
        "runs never become an empirical pass.",
        "",
        "## Pre-registration",
        "",
        f"- Frozen protocol: {_value((report.get('protocol') or {}).get('status'))}",
        f"- Content SHA-256: {_value((report.get('protocol') or {}).get('content_sha256'))}",
        f"- Late binding: {_value((report.get('protocol') or {}).get('late_binding'))}",
    ]
    protocol_reasons = (report.get("protocol") or {}).get("binding_reasons") or []
    if protocol_reasons:
        lines.append("- Binding problems: " + "; ".join(protocol_reasons))
    lines.extend(["", "## Quality gates", ""])
    lines.extend(
        _table(
            ["Gate", "Required", "Status", "Evidence and reason"],
            [
                [
                    name,
                    gate.get("required"),
                    gate.get("status"),
                    gate.get("reason"),
                ]
                for name, gate in (report.get("gates") or {}).items()
            ],
        )
    )
    lines.extend(
        [
            "",
            "A blocking failure makes the quality conclusion failed; missing evidence makes it "
            "inconclusive. Exploratory comparisons are reported but never decide the overall "
            "pass. Only the frozen protocol defines thresholds; the code does not add hidden "
            "defaults. Resampling happens over pre-declared clusters (whole books, chapters or "
            "contiguous blocks), so repeated paragraphs and reruns are never treated as "
            "independent i.i.d. observations.",
            "",
        ]
    )
    lines.extend(
        [
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
    )
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
    for key in sorted(cost["groups"]):
        if not key.startswith("preparation:") or key == "preparation":
            continue
        isolated = cost["groups"][key]
        tokens = isolated.get("tokens") or {}
        lines.append(
            f"Isolated preparation `{key.split(':', 1)[1]}`: calls "
            f"{_value(tokens.get('calls'))}; cost "
            f"{_value((isolated.get('cost') or {}).get('amount'))} "
            f"({_value((isolated.get('cost') or {}).get('status'))}); charged separately."
        )
    if any(key.startswith("preparation:") for key in cost["groups"]):
        lines.append("")
    lines.extend(
        _table(
            ["Group", "Review status", "Issues", "Reviewed segments", "Reflection"],
            [
                [
                    row["group"],
                    row["review_status"],
                    row["review_issues"],
                    (report["review"].get(row["group"]) or {}).get("reviewed_segments"),
                    row["reflection_status"],
                ]
                for row in report["groups"]
            ],
        )
    )
    lines.extend(["", "### Reflection evidence", ""])
    lines.extend(
        _table(
            [
                "Group",
                "Status",
                "Current batches",
                "Stale",
                "Invalid",
                "Ambiguous",
                "Current segments",
                "Uncovered",
                "Write failed",
                "Adoption",
                "Critique",
                "Failure",
            ],
            [
                [
                    row["group"],
                    row["reflection_status"],
                    row["reflection_current_batches"],
                    row["reflection_stale_batches"],
                    row["reflection_invalid_batches"],
                    row["reflection_ambiguous_batches"],
                    row["reflection_current_segments"],
                    row["reflection_uncovered_segments"],
                    row["reflection_write_failed_events"],
                    json.dumps(row["reflection_adoption"], sort_keys=True),
                    json.dumps(row["reflection_critique"], sort_keys=True),
                    json.dumps(row["reflection_failure"], sort_keys=True),
                ]
                for row in report["groups"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "A `current` artifact only means the record applies to the current post-alignment "
            "text; it is never a clean quality verdict. Missing, stale, invalid or ambiguous "
            "artifacts stay in the segment denominator and are never counted as adoption. "
            "`skipped_no_turn` remains a critique outcome, not a clean batch.",
            "",
        ]
    )
    isolated = report.get("isolated_preparation") or {}
    if isolated:
        lines.extend(["### Isolated preparation", ""])
        for group, record in sorted(isolated.items()):
            lines.append(
                f"- `{group}`: snapshot `{record.get('snapshot')}`; terminal "
                f"`{record.get('terminal')}`; cost "
                f"{(record.get('cost') or {}).get('status')} (charged separately)."
            )
        lines.append("")
    lines.extend(
        [
            "## Human evaluation",
            "",
            f"- Suggested stratified rows: {_value((selection or {}).get('selected'))} "
            f"of {_value((selection or {}).get('eligible_paragraphs'))} eligible; "
            f"reviewed={_value((selection or {}).get('human_review_executed'))}.",
            f"- Pairwise preregistered pairs: {_value((preregistration or {}).get('expected_pairs'))}; "
            f"rated rows: {_value((imported or {}).get('rated_rows'))}.",
            f"- Edited segments assessed: "
            f"{_value(((report.get('human') or {}).get('edit_imported') or {}).get('assessed_rows'))} "
            f"of {_value(((report.get('human') or {}).get('edit_imported') or {}).get('denominator'))}.",
            "- Suggested strata and pairwise rows are not counted as human evidence. "
            "Missing ratings remain missing.",
            "- Pairwise ratings use anonymous side labels and opaque pair ids with a separate key; "
            "this is not described as double blind. Human origin is caller-reported, not "
            "identity-verified.",
            "- MQM errors and severe flags are human annotations bound to exported text hashes; "
            "changed outputs invalidate them.",
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
            "- `experiment_protocol.frozen.json` — frozen pre-registration content and hash",
            "- `manifest.json` — identities, bindings and absolute paths",
            "- `records/terminals/` — one exclusive terminal record per group and stage",
            "- `extraction.json` — expected-key aligned read-only extraction",
            "- `samples/strata.jsonl` — suggested human-review rows (unreviewed)",
            "- `samples/pairwise.csv` / `samples/pairwise_key.csv` — anonymous pairs and key",
            "- `samples/edit_review.csv` / `samples/edit_key.csv` — blind before/after edit packet",
            "- `report/cost.json` / `report/stage_costs.csv` — operation and stage accounting",
            "- `report/group_summary.csv` / `report/human_summary.csv` — CSV summaries",
            "",
        ]
    )
    return "\n".join(lines)
