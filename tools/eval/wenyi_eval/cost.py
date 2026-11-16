"""Operation-level usage, cache and cost attribution with unknown-not-zero rules."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .common import EvalError, PathGuard, read_json, utc_now, write_json_atomic

COST_PROTOCOL = 1
_TOKEN_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_write_tokens",
    "uncached_tokens",
    "cache_unknown_tokens",
    "cache_hit_reported_calls",
    "cache_write_reported_calls",
    "uncached_reported_calls",
)
_CACHE_COUNTERS = {
    "cache_hit_tokens": "cache_hit_reported_calls",
    "cache_write_tokens": "cache_write_reported_calls",
    "uncached_tokens": "uncached_reported_calls",
}


def load_prices(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "unknown",
            "date": None,
            "currency": None,
            "unit": 1_000_000,
            "source": None,
            "reason": "no provider price rule was supplied",
        }
    data = read_json(path)
    if data.get("status") not in {"known", "partial", "unknown"}:
        raise EvalError("Price file must declare status known/partial/unknown")
    return data


def _category(stage: str) -> str:
    if stage.startswith(("analysis.", "synopsis.", "language.detect")):
        return "preparation"
    if stage.startswith(("review.", "autofix.")):
        return "review"
    return "body"


def _count(values: dict[str, Any], field: str) -> int | None:
    """Preserve a missing counter as unknown; reject negative or non-integer counts."""
    value = values.get(field)
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int) or value < 0:
        return None
    return value


def _counter(slot: dict[str, Any], field: str) -> int | None:
    """Read a cache partition counter using the product's presence counters.

    The core ledger records how many samples actually reported each cache
    counter. A zero presence count means the counter was absent, which must not
    be confused with an observed zero.
    """
    presence = slot.get(_CACHE_COUNTERS[field])
    value = slot[field]
    if presence is not None and presence == 0:
        return None
    return value


def _slots(usage: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(usage, dict):
        return {}
    slots = usage.get("by_stage")
    if not isinstance(slots, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, values in slots.items():
        if not isinstance(values, dict):
            continue
        slot = {field: _count(values, field) for field in _TOKEN_FIELDS}
        if isinstance(values.get("reasoning_tokens"), int):
            slot["reasoning_tokens"] = values["reasoning_tokens"]
        result[str(name)] = slot
    return result


def _zero() -> dict[str, int]:
    return {field: 0 for field in _TOKEN_FIELDS}


def _reasoning(slots: dict[str, dict[str, Any]]) -> tuple[int | None, str]:
    values = [slot.get("reasoning_tokens") for slot in slots.values()]
    if slots and all(isinstance(value, int) for value in values):
        return sum(values), "known"
    return None, "unknown"


def _add(into: dict[str, int | None], values: dict[str, Any]) -> None:
    """Add known counters; any missing component makes the total unknown."""
    for field in _TOKEN_FIELDS:
        current, addition = into.get(field), values.get(field)
        into[field] = None if current is None or addition is None else current + addition


def _price_for(prices: dict[str, Any], provider: str) -> dict[str, Any] | None:
    providers = prices.get("providers")
    if not isinstance(providers, dict):
        return None
    value = providers.get(provider) or providers.get("default")
    return value if isinstance(value, dict) else None


def _rate(rule: dict[str, Any], name: str) -> float | None:
    value = rule.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _unknown(amount: None, reason: str) -> dict[str, Any]:
    return {"status": "unknown", "amount": None, "reason": reason}


def _partition(tokens: dict[str, int | None]) -> tuple[dict[str, int] | None, str | None]:
    """Return a complete inclusive input partition or explain why it is ambiguous."""
    prompt = tokens.get("prompt_tokens")
    if prompt is None:
        return None, "prompt token count is missing"
    hit = _counter(tokens, "cache_hit_tokens")
    write = _counter(tokens, "cache_write_tokens")
    uncached = _counter(tokens, "uncached_tokens")
    unknown = tokens.get("cache_unknown_tokens")
    if unknown is None:
        # Counters that were not reported stay in the unclassified residual
        # instead of being assumed to be zero.
        reported = sum(value for value in (hit, write, uncached) if value is not None)
        unknown = prompt - reported
    parts = {
        "cache_hit": hit or 0,
        "cache_write": write or 0,
        "uncached": uncached or 0,
        "unknown": unknown,
    }
    if any(value < 0 for value in parts.values()):
        return None, "cache partition is negative"
    if sum(parts.values()) != prompt:
        return None, "cache partition does not add up to the inclusive prompt token count"
    return parts, None


def _billable(
    tokens: dict[str, int | None],
    prices: dict[str, Any],
    providers: list[str],
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Price an inclusive token ledger without guessing unknown cache rates."""
    if prices.get("status") != "known":
        return _unknown(None, prices.get("reason") or "provider price rules are not known")
    models = models or []
    declared_models = prices.get("models")
    override_rule: dict[str, Any] | None = None
    if len(models) > 1:
        if not isinstance(declared_models, dict) or any(
            model not in declared_models for model in models
        ):
            return _unknown(
                None,
                "multiple models share one provider; declare per-model price rules",
            )
        shapes = {
            tuple(
                _rate(declared_models[model] or {}, field)
                for field in ("input", "output", "cache_read", "cache_write")
            )
            for model in models
        }
        if len(shapes) > 1:
            return _unknown(
                None,
                "multiple model tariffs cannot price aggregate tokens without per-model buckets",
            )
        override_rule = declared_models[models[0]]
    elif len(models) == 1 and isinstance(declared_models, dict) and models[0] in declared_models:
        override_rule = declared_models[models[0]]
    unit = prices.get("unit")
    if not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0:
        return _unknown(None, "price unit is invalid")
    if not prices.get("date") or not prices.get("currency"):
        return _unknown(None, "price rules require a date and currency")
    if len(providers) > 1:
        return _unknown(
            None,
            "tokens cannot be attributed to multiple providers without double counting",
        )
    if not providers:
        return _unknown(None, "no provider attribution was recorded for these tokens")
    prompt = tokens.get("prompt_tokens")
    completion = tokens.get("completion_tokens")
    if prompt is None or completion is None:
        return _unknown(None, "prompt or completion token counts are missing")
    reasoning = tokens.get("reasoning_tokens")
    if isinstance(reasoning, int) and reasoning > completion:
        return _unknown(None, "reasoning tokens exceed the inclusive completion total")
    partition, partition_reason = _partition(tokens)
    if partition is None:
        return _unknown(None, partition_reason)
    total = 0.0
    for provider in providers:
        rule = override_rule or _price_for(prices, provider)
        if rule is None:
            return _unknown(None, f"no price rule for provider {provider!r}")
        input_rate = _rate(rule, "input")
        output_rate = _rate(rule, "output")
        if input_rate is None or output_rate is None:
            return _unknown(
                None, f"price rule for {provider!r} lacks a nonnegative input/output rate"
            )
        read_rate = _rate(rule, "cache_read")
        write_rate = _rate(rule, "cache_write")
        if partition["cache_hit"] and read_rate is None:
            return _unknown(None, f"price rule for {provider!r} lacks a cache_read rate")
        if partition["cache_write"] and write_rate is None:
            return _unknown(None, f"price rule for {provider!r} lacks a cache_write rate")
        if partition["unknown"]:
            # Unclassified input could be a read, a write or ordinary input. It is
            # only priceable when the declared rule proves every allocation costs
            # the same; otherwise the amount is unknown, never zero.
            if (
                read_rate is None
                or write_rate is None
                or not (input_rate == read_rate == write_rate)
            ):
                return _unknown(
                    None,
                    f"price rule for {provider!r} does not prove one rate for unclassified input",
                )
        total += (
            partition["cache_hit"] * (read_rate or 0.0)
            + partition["cache_write"] * (write_rate or 0.0)
            + partition["uncached"] * input_rate
            + partition["unknown"] * input_rate
            + completion * output_rate
        ) / unit
    return {
        "status": "known",
        "amount": round(total, 6),
        "currency": prices.get("currency"),
        "reason": None,
    }


def _usage_rows(
    *,
    scope: str,
    group: str | None,
    usage: dict[str, Any] | None,
    prices: dict[str, Any],
    usage_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slots = _slots(usage)
    totals: dict[str, int | None] = _zero()
    for values in slots.values():
        _add(totals, values)
    if not slots:
        totals = {field: None for field in _TOKEN_FIELDS}
    reasoning_total, reasoning_status = _reasoning(slots)
    providers = sorted((usage.get("by_provider") or {}).keys()) if isinstance(usage, dict) else []
    models = sorted((usage.get("by_model") or {}).keys()) if isinstance(usage, dict) else []
    rows = []
    aggregates: dict[str, dict[str, int]] = {
        category: _zero() for category in ("body", "preparation", "review")
    }
    category_slots: dict[str, dict[str, dict[str, Any]]] = {
        category: {} for category in ("body", "preparation", "review")
    }
    for stage, values in sorted(slots.items()):
        category = _category(stage)
        _add(aggregates[category], values)
        category_slots[category][stage] = values
        cost = _billable(values, prices, providers, models)
        reasoning = values.get("reasoning_tokens")
        rows.append(
            {
                "scope": scope,
                "group": group or "",
                "category": category,
                "operation": stage,
                **values,
                "reasoning_tokens": reasoning if isinstance(reasoning, int) else "unknown",
                "reasoning_status": "known" if isinstance(reasoning, int) else "unknown",
                "cost_status": cost["status"],
                "cost": cost["amount"] if cost["status"] == "known" else "unknown",
                "currency": cost.get("currency") or "",
                "cost_reason": cost.get("reason") or "",
            }
        )
    for category, values in sorted(aggregates.items()):
        if not any(value not in (None, 0) for value in values.values()):
            continue
        cost = _billable(values, prices, providers, models)
        reasoning, reasoning_status = _reasoning(category_slots[category])
        rows.append(
            {
                "scope": scope,
                "group": group or "",
                "category": category,
                "operation": "subtotal",
                **values,
                "reasoning_tokens": reasoning if reasoning is not None else "unknown",
                "reasoning_status": reasoning_status,
                "cost_status": cost["status"],
                "cost": cost["amount"] if cost["status"] == "known" else "unknown",
                "currency": cost.get("currency") or "",
                "cost_reason": cost.get("reason") or "",
            }
        )
    billed_totals = dict(totals)
    if reasoning_total is not None:
        billed_totals["reasoning_tokens"] = reasoning_total
    overall = (
        _billable(billed_totals, prices, providers, models)
        if slots
        else {
            "status": "unknown",
            "amount": None,
            "reason": usage_status,
        }
    )
    hit_rate = (
        round(totals["cache_hit_tokens"] / totals["prompt_tokens"], 4)
        if totals["prompt_tokens"] and totals["cache_hit_tokens"] is not None
        else None
    )
    summary = {
        "usage_status": usage_status,
        "tokens": totals,
        "reasoning_tokens": reasoning_total if reasoning_total is not None else "unknown",
        "reasoning_status": reasoning_status,
        "cost": overall,
        "operations": len(slots),
        "cache_hit_rate": hit_rate,
        "providers": providers,
        "source": "synthetic_offline_adapter" if usage_status == "synthetic" else "provider_ledger",
    }
    return rows, summary


def build_cost_summary(
    *,
    root: Path,
    manifest: dict[str, Any],
    extraction: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    guard = PathGuard(root)
    prices = manifest.get("prices") or {}
    rows: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    preparation = execution.get("preparation", {})
    prep_usage_status = (
        "synthetic" if preparation.get("adapter", "offline") == "offline" else "provider"
    )
    prep_rows, prep_summary = _usage_rows(
        scope="preparation",
        group=None,
        usage=preparation.get("usage"),
        prices=prices,
        usage_status=prep_usage_status,
    )
    rows.extend(prep_rows)
    prep_summary["wall_seconds"] = None
    terminal = preparation.get("terminal_path")
    if isinstance(terminal, str) and Path(terminal).is_file():
        prep_summary["wall_seconds"] = read_json(terminal).get("duration_seconds")
    groups["preparation"] = prep_summary
    for isolated_group, isolated in (execution.get("isolated_preparation") or {}).items():
        isolated_status = (
            "synthetic" if isolated.get("adapter", "offline") == "offline" else "provider"
        )
        isolated_rows, isolated_summary = _usage_rows(
            scope=f"preparation:{isolated_group}",
            group=None,
            usage=isolated.get("usage"),
            prices=prices,
            usage_status=isolated_status,
        )
        isolated_summary["wall_seconds"] = None
        terminal = isolated.get("terminal_path")
        if isinstance(terminal, str) and Path(terminal).is_file():
            isolated_summary["wall_seconds"] = read_json(terminal).get("duration_seconds")
        isolated_summary["isolated_preparation"] = True
        isolated_summary["reason"] = "preparation contract differed from the shared snapshot"
        rows.extend(isolated_rows)
        groups[f"preparation:{isolated_group}"] = isolated_summary
    for group in manifest["groups"]:
        extracted = extraction["groups"].get(group, {})
        usage = extracted.get("usage")
        adapter = (manifest.get("groups", {}).get(group) or {}).get("adapter", "offline")
        usage_status = ("synthetic" if adapter == "offline" else "provider") if usage else "missing"
        _, summary = _usage_rows(
            scope="body_and_review",
            group=group,
            usage=usage,
            prices=prices,
            usage_status=usage_status,
        )
        summary["state_status"] = extracted.get("status")
        summary["wall_seconds"] = (
            execution.get("groups", {}).get(group, {}).get("durations") or {}
        ).get("translate")
        summary["failure_or_retry_events"] = extracted.get("events", {}).get(
            "failure_or_retry_counts", {}
        )
        rows.extend(
            _usage_rows(
                scope="body_and_review",
                group=group,
                usage=usage,
                prices=prices,
                usage_status=usage_status,
            )[0]
        )
        groups[group] = summary
    aggregate_tokens = _zero()
    for summary in groups.values():
        _add(aggregate_tokens, summary["tokens"])
    known_costs = [
        summary["cost"]["amount"]
        for summary in groups.values()
        if summary["cost"].get("status") == "known"
    ]
    all_known = all(summary["cost"].get("status") == "known" for summary in groups.values())
    aggregate_cost = {
        "status": "known" if all_known and prices.get("status") == "known" else "unknown",
        "amount": round(sum(known_costs), 6) if all_known and known_costs else None,
        "currency": prices.get("currency"),
        "reason": None if all_known else "at least one stage has unknown pricing or usage",
    }
    summary = {
        "protocol_version": COST_PROTOCOL,
        "created_at": utc_now(),
        "price_status": prices.get("status"),
        "price_date": prices.get("date"),
        "price_source": prices.get("source"),
        "groups": groups,
        "aggregate_tokens": aggregate_tokens,
        "aggregate_cost": aggregate_cost,
        "rows": rows,
    }
    write_json_atomic(guard.child("report", "cost.json"), summary)
    _write_cost_csv(guard.child("report", "stage_costs.csv"), rows)
    return summary


def _write_cost_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "group",
        "category",
        "operation",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_write_tokens",
        "uncached_tokens",
        "cache_unknown_tokens",
        "reasoning_tokens",
        "reasoning_status",
        "cost_status",
        "cost",
        "currency",
        "cost_reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
