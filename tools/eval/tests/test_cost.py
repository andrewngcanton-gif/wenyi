from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import minimal_manifest
from wenyi_eval.common import EvalError
from wenyi_eval.cost import build_cost_summary, load_prices
from wenyi_eval.manifest import SUPPORTED_GROUPS

KNOWN_PRICES = {
    "status": "known",
    "date": "2026-11-16",
    "currency": "USD",
    "unit": 1_000_000,
    "providers": {"default": {"input": 2.0, "output": 4.0, "cache_read": 0.5, "cache_write": 1.0}},
}

BODY_USAGE = {
    "by_stage": {
        "translation.body": {
            "calls": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cache_hit_tokens": 100,
            "cache_write_tokens": 50,
            "uncached_tokens": 850,
        }
    },
    "by_provider": {"default": {}},
}

PREP_USAGE = {
    "by_stage": {
        "analysis.style": {
            "calls": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "uncached_tokens": 1000,
        }
    },
    "by_provider": {"default": {}},
}


def _execution(root: Path, *, prep_usage=PREP_USAGE) -> dict:
    return {"preparation": {"usage": prep_usage, "terminal_path": str(root / "missing.json")}}


def _extraction(*, usage=BODY_USAGE) -> dict:
    return {
        "groups": {
            group: {"status": "ok", "usage": usage, "events": {}} for group in SUPPORTED_GROUPS
        }
    }


def _summary(root: Path, *, prices: dict, usage=BODY_USAGE, prep_usage=PREP_USAGE):
    manifest = minimal_manifest(root, prices=prices)
    return build_cost_summary(
        root=root,
        manifest=manifest,
        extraction=_extraction(usage=usage),
        execution=_execution(root, prep_usage=prep_usage),
    )


def test_prices_must_be_explicit_and_unknown_is_never_zero(tmp_path: Path) -> None:
    summary = _summary(
        tmp_path / "eval",
        prices={"status": "unknown", "date": None, "currency": None, "unit": 1_000_000},
    )

    assert summary["aggregate_cost"]["status"] == "unknown"
    assert summary["aggregate_cost"]["amount"] is None
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert summary["groups"]["A"]["cost"]["amount"] is None
    assert summary["groups"]["preparation"]["cost"]["amount"] is None
    assert load_prices(None)["status"] == "unknown"

    invalid = tmp_path / "prices.json"
    invalid.write_text(json.dumps({"status": "maybe"}), encoding="utf-8")
    with pytest.raises(EvalError, match="known/partial/unknown"):
        load_prices(invalid)


def test_cache_partitions_are_not_double_counted(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES)
    rows = [row for row in summary["rows"] if row["group"] == "A"]
    body = next(row for row in rows if row["operation"] == "subtotal" and row["category"] == "body")

    # (100 cache reads * 0.5 + 50 writes * 1 + 850 uncached * 2 + 100 output * 4) / 1e6
    assert body["cost"] == pytest.approx(0.0022)
    assert body["prompt_tokens"] == 1000
    assert summary["groups"]["A"]["tokens"]["prompt_tokens"] == 1000


def test_unknown_cache_allocation_is_unknown_not_input_priced(tmp_path: Path) -> None:
    usage = {
        "by_stage": {
            "translation.body": {
                "calls": 1,
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "cache_unknown_tokens": 1000,
            }
        },
        "by_provider": {"default": {}},
    }
    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES, usage=usage)

    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert summary["groups"]["A"]["cost"]["amount"] is None
    assert "does not prove one rate" in summary["groups"]["A"]["cost"]["reason"]


def test_a_declared_uniform_cache_rule_can_price_unclassified_input(tmp_path: Path) -> None:
    prices = {
        **KNOWN_PRICES,
        "providers": {
            "default": {"input": 2.0, "output": 4.0, "cache_read": 2.0, "cache_write": 2.0}
        },
    }
    usage = {
        "by_stage": {
            "translation.body": {
                "calls": 1,
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "cache_unknown_tokens": 1000,
            }
        },
        "by_provider": {"default": {}},
    }

    summary = _summary(tmp_path / "eval", prices=prices, usage=usage)

    assert summary["groups"]["A"]["cost"]["status"] == "known"
    assert summary["groups"]["A"]["cost"]["amount"] == pytest.approx(0.0024)


def test_missing_token_counters_are_unknown_instead_of_zero(tmp_path: Path) -> None:
    usage = {
        "by_stage": {"translation.body": {"calls": 1}},
        "by_provider": {"default": {}},
    }

    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES, usage=usage)

    assert summary["groups"]["A"]["tokens"]["prompt_tokens"] is None
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert summary["groups"]["A"]["cost"]["amount"] is None
    assert "missing" in summary["groups"]["A"]["cost"]["reason"]


def test_real_usage_tracker_ledger_without_cache_counters_stays_unknown(
    tmp_path: Path,
) -> None:
    from wenyi_core.llm.usage import UsageSample, UsageTracker

    tracker = UsageTracker()
    tracker.record(
        "strong",
        UsageSample(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
        "translation.body",
        provider="default",
        model="stub",
    )
    usage = tracker.summary()

    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES, usage=usage)

    assert summary["groups"]["A"]["tokens"]["prompt_tokens"] == 1000
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert summary["groups"]["A"]["cost"]["amount"] is None
    assert summary["groups"]["A"]["cost"]["reason"]


def test_price_rule_requires_date_currency_and_valid_rates(tmp_path: Path) -> None:
    missing_date = {
        **KNOWN_PRICES,
        "date": None,
    }
    summary = _summary(tmp_path / "date", prices=missing_date)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"

    negative_rate = {
        **KNOWN_PRICES,
        "providers": {
            "default": {"input": -1.0, "output": 4.0, "cache_read": 0.5, "cache_write": 1.0}
        },
    }
    summary = _summary(tmp_path / "rate", prices=negative_rate)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "nonnegative" in summary["groups"]["A"]["cost"]["reason"]


def test_reasoning_tokens_must_stay_inside_the_completion_total(tmp_path: Path) -> None:
    usage = {
        "by_stage": {
            "translation.body": {
                **BODY_USAGE["by_stage"]["translation.body"],
                "reasoning_tokens": 101,
            }
        },
        "by_provider": {"default": {}},
    }

    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES, usage=usage)

    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "reasoning tokens exceed" in summary["groups"]["A"]["cost"]["reason"]


def test_multiple_models_require_declared_tariffs(tmp_path: Path) -> None:
    usage = {
        **BODY_USAGE,
        "by_model": {"strong-model": {}, "cheap-model": {}},
    }
    summary = _summary(tmp_path / "unknown", prices=KNOWN_PRICES, usage=usage)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "multiple models" in summary["groups"]["A"]["cost"]["reason"]

    uniform = {
        **KNOWN_PRICES,
        "models": {
            "strong-model": {"input": 2.0, "output": 4.0, "cache_read": 0.5, "cache_write": 1.0},
            "cheap-model": {"input": 2.0, "output": 4.0, "cache_read": 0.5, "cache_write": 1.0},
        },
    }
    summary = _summary(tmp_path / "uniform", prices=uniform, usage=usage)
    assert summary["groups"]["A"]["cost"]["status"] == "known"

    mixed = {
        **uniform,
        "models": {
            "strong-model": {"input": 6.0, "output": 12.0, "cache_read": 1.5, "cache_write": 3.0},
            "cheap-model": {"input": 2.0, "output": 4.0, "cache_read": 0.5, "cache_write": 1.0},
        },
    }
    summary = _summary(tmp_path / "mixed", prices=mixed, usage=usage)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "multiple model tariffs" in summary["groups"]["A"]["cost"]["reason"]


def test_preparation_usage_is_counted_once_and_kept_out_of_variant_totals(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES)

    prep = summary["groups"]["preparation"]
    body = summary["groups"]["A"]
    assert prep["tokens"]["calls"] == 1
    assert prep["tokens"]["prompt_tokens"] == 1000
    assert body["tokens"]["calls"] == 1
    assert body["tokens"]["prompt_tokens"] == 1000
    assert summary["aggregate_tokens"]["calls"] == 6
    assert summary["aggregate_tokens"]["prompt_tokens"] == 6000
    assert summary["aggregate_cost"]["status"] == "known"
    assert summary["aggregate_cost"]["amount"] == pytest.approx(0.0134)


def test_multiple_or_unmatched_providers_leave_cost_unknown(tmp_path: Path) -> None:
    multiple = {
        **BODY_USAGE,
        "by_provider": {"default": {}, "other": {}},
    }
    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES, usage=multiple)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "multiple providers" in summary["groups"]["A"]["cost"]["reason"]

    unmatched = {**BODY_USAGE, "by_provider": {"acme": {}}}
    unmatched_prices = {
        **KNOWN_PRICES,
        "providers": {"other": KNOWN_PRICES["providers"]["default"]},
    }
    other_root = tmp_path / "other"
    summary = _summary(other_root, prices=unmatched_prices, usage=unmatched)
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert "no price rule" in summary["groups"]["A"]["cost"]["reason"]


def test_missing_variant_usage_stays_missing(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    manifest = minimal_manifest(root, prices=KNOWN_PRICES)
    extraction = {
        "groups": {
            group: {"status": "error", "usage": None, "events": {}} for group in SUPPORTED_GROUPS
        }
    }

    summary = build_cost_summary(
        root=root,
        manifest=manifest,
        extraction=extraction,
        execution=_execution(root),
    )

    assert summary["groups"]["A"]["usage_status"] == "missing"
    assert summary["groups"]["A"]["cost"]["status"] == "unknown"
    assert summary["groups"]["A"]["reasoning_tokens"] == "unknown"
    assert summary["groups"]["A"]["reasoning_status"] == "unknown"
    assert summary["aggregate_cost"]["status"] == "unknown"
    assert (root / "report" / "cost.json").is_file()
    assert (root / "report" / "stage_costs.csv").is_file()


def test_reasoning_tokens_are_unknown_when_the_ledger_does_not_expose_them(
    tmp_path: Path,
) -> None:
    summary = _summary(tmp_path / "eval", prices=KNOWN_PRICES)
    assert summary["groups"]["A"]["reasoning_tokens"] == "unknown"

    usage = {
        **BODY_USAGE,
        "by_stage": {
            "translation.body": {
                **BODY_USAGE["by_stage"]["translation.body"],
                "reasoning_tokens": 25,
            }
        },
    }
    known = _summary(tmp_path / "known", prices=KNOWN_PRICES, usage=usage)
    assert known["groups"]["A"]["reasoning_tokens"] == 25
    assert known["groups"]["A"]["reasoning_status"] == "known"
