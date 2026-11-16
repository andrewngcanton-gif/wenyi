"""Batch-level structured critique and critique-informed polishing.

The critique continues the successful translation conversation, so the reviewer sees the
same source, glossary, style guide and context as the translator. An empty critique is a
valid result only after complete strict validation. Callers fall back to ordinary polishing
whenever this agent returns an invalid result or failure.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ..alignment.objects import OBJECT_PREFIX
from ..llm.errors import describe_provider_failure
from ..llm.json_parser import parse_json_loose
from ..markup.ruby import RUBY_MARK_RIGHT, strip_ruby_markers
from . import prompts
from .base import Agent, Messages

_LOGGER = logging.getLogger(__name__)

CATEGORIES = ("style", "fluency", "idiom", "accuracy")
ACCURACY_KINDS = ("omission", "addition", "mistranslation")
SEVERITY_RANK = {"minor": 0, "major": 1, "critical": 2}
MAX_SPAN_CHARS = 2000
MAX_EXPLANATION_CHARS = 1000
_WHITESPACE = re.compile(r"\s+")
# Mirrors the complete-token scan in alignment/objects.py without importing its private
# scanner; the prefix and terminator stay public constants so the transport cannot drift.
_OBJECT_TOKEN = re.compile(
    re.escape(OBJECT_PREFIX) + r"[A-Za-z0-9_.:-]+" + re.escape(RUBY_MARK_RIGHT)
)


@dataclass(frozen=True)
class ReflectionIssue:
    index: int  # Zero-based position in the continued translation response.
    category: str
    severity: str
    span: str
    source_span: str
    detail: str
    suggestion: str
    accuracy_kind: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def max_issues(n: int) -> int:
    return max(8, 2 * n)


def _text(value: Any, limit: int, *, allow_empty: bool = False) -> str | None:
    if not isinstance(value, str) or len(value) > limit:
        return None
    if not allow_empty and not value.strip():
        return None
    return value


def _locate(span: str, text: str) -> bool:
    needle = _normalize(span)
    return bool(needle) and needle in _normalize(text)


def parse_reflection(
    data: Any, *, drafts: list[str], sources: list[str]
) -> tuple[ReflectionIssue, ...] | None:
    """All-or-nothing validation. Any invalid item invalidates the whole critique.

    ``None`` means the caller must fall back to ordinary polishing. It never means
    "clean" and must never be converted into a keep decision.
    """
    n = len(drafts)
    if n != len(sources):
        return None
    if not isinstance(data, dict) or data.get("complete") is not True:
        return None
    reviewed = data.get("reviewed_segments")
    if type(reviewed) is not int or reviewed != n:
        return None
    raw_items = data.get("issues")
    if not isinstance(raw_items, list) or len(raw_items) > max_issues(n):
        return None

    merged: dict[tuple[int, str, str, str, str], ReflectionIssue] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        category = item.get("category")
        severity = item.get("severity")
        if type(index) is not int or not 0 <= index < n:
            return None
        if not isinstance(category, str) or category not in CATEGORIES:
            return None
        if not isinstance(severity, str) or severity not in SEVERITY_RANK:
            return None

        # Validate all supplied evidence fields, including fields not used for gating.
        span = _text(item.get("span", ""), MAX_SPAN_CHARS, allow_empty=True)
        source_span = _text(item.get("source_span", ""), MAX_SPAN_CHARS, allow_empty=True)
        raw_kind = item.get("accuracy_kind", "")
        if span is None or source_span is None or not isinstance(raw_kind, str):
            return None
        accuracy_kind = ""
        if category == "accuracy":
            if raw_kind not in ACCURACY_KINDS or severity == "minor":
                return None
            accuracy_kind = raw_kind
            if accuracy_kind == "omission":
                if not _locate(source_span, sources[index]):
                    return None
                if span.strip() and not _locate(span, drafts[index]):
                    return None
            else:
                if not _locate(span, drafts[index]):
                    return None
                if source_span.strip() and not _locate(source_span, sources[index]):
                    return None
        else:
            if raw_kind or source_span.strip() or not _locate(span, drafts[index]):
                return None

        detail = _text(item.get("detail"), MAX_EXPLANATION_CHARS)
        suggestion = _text(item.get("suggestion"), MAX_EXPLANATION_CHARS)
        if detail is None or suggestion is None:
            return None

        key = (
            index,
            category,
            _normalize(span),
            _normalize(source_span),
            accuracy_kind,
        )
        candidate = ReflectionIssue(
            index=index,
            category=category,
            severity=severity,
            span=span,
            source_span=source_span,
            detail=detail,
            suggestion=suggestion,
            accuracy_kind=accuracy_kind,
        )
        existing = merged.get(key)
        if existing is None or SEVERITY_RANK[severity] > SEVERITY_RANK[existing.severity]:
            merged[key] = candidate
    return tuple(merged.values())


def extract_raw_drafts(turn: Messages, *, n: int) -> tuple[str, ...] | None:
    """Read the untransformed translation array that the critic was shown."""
    if len(turn) < 3 or turn[-1].get("role") != "assistant" or n <= 0:
        return None
    try:
        data = parse_json_loose(turn[-1].get("content", ""))
    except Exception:  # noqa: BLE001 - Malformed transcript is a structural failure.
        return None
    items = data.get("translations") if isinstance(data, dict) else data
    if (
        not isinstance(items, list)
        or len(items) != n
        or any(not isinstance(item, str) for item in items)
    ):
        return None
    return tuple(items)


def _object_counts(text: str) -> Counter[str]:
    return Counter(_OBJECT_TOKEN.findall(text))


def validate_revision(
    items: Any,
    *,
    drafts_raw: list[str],
    drafts_formal: list[str],
) -> tuple[str, ...] | None:
    """Return ruby-normalized candidates, or None when the candidate is unsafe."""
    n = len(drafts_raw)
    if n != len(drafts_formal):
        return None
    if not isinstance(items, list) or len(items) != n:
        return None
    if any(not isinstance(item, str) for item in items):
        return None
    normalized = tuple(strip_ruby_markers(item) for item in items)
    for raw_candidate, raw_draft, formal_draft, candidate in zip(
        items, drafts_raw, drafts_formal, normalized
    ):
        if formal_draft.strip() and not candidate.strip():
            return None
        # Reject malformed new token openers as well as complete-token changes.
        for text in (raw_candidate, raw_draft):
            without_valid_tokens = _OBJECT_TOKEN.sub("", text)
            if OBJECT_PREFIX in without_valid_tokens:
                return None
        if _object_counts(raw_candidate) != _object_counts(raw_draft):
            return None
    return normalized


@dataclass(frozen=True)
class Reflection:
    issues: tuple[ReflectionIssue, ...]

    def gated(self, min_severity: str) -> tuple[ReflectionIssue, ...]:
        threshold = SEVERITY_RANK[min_severity]
        return tuple(issue for issue in self.issues if SEVERITY_RANK[issue.severity] >= threshold)


@dataclass(frozen=True)
class CritiqueAttempt:
    reflection: Reflection | None
    failure: str | None  # "invalid_json" | "invalid_schema" | provider category
    attempts: int


@dataclass(frozen=True)
class RevisionAttempt:
    items: tuple[str, ...] | None  # Ruby-normalized candidates.
    failure: str | None
    attempts: int


def _critique_budget(n: int) -> int:
    return max(1024, min(4096, 512 + 256 * n))


# Revision uses the configured polish.body output limit. A paragraph count is not
# a reliable token budget, especially for a single very long source paragraph.


class Reflector(Agent):
    """Strict structured critique and gated revision on a copied translation turn."""

    def _editor_turn(self, turn: Messages) -> Messages:
        # Copy only; ordinary fallback continues to use the original translator turn.
        messages = [dict(message) for message in turn]
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Expected a validated translation system message")
        messages[0]["content"] += "\n\n" + self.render("reflector_stage_system")
        return messages

    def critique_continue(
        self,
        turn: Messages,
        *,
        drafts: list[str],
        sources: list[str],
        next_source: str = "",
    ) -> CritiqueAttempt:
        n = len(drafts)
        if n <= 0 or n != len(sources) or len(turn) < 3:
            return CritiqueAttempt(None, "no_turn", 0)
        user = self.render(
            "reflector_continue_user",
            n=n,
            next_source=prompts.render_source_reference(next_source),
        )
        messages: Messages = [*self._editor_turn(turn), {"role": "user", "content": user}]
        attempts = self.config.pipeline.align_retry_limit + 1
        failure = "invalid_schema"
        for attempt in range(1, attempts + 1):
            try:
                text = self.client.complete(
                    messages,
                    operation="reflect.body",
                    json_mode=True,
                    max_tokens=_critique_budget(n),
                )
            except Exception as error:  # noqa: BLE001 - Ordinary failures fall back; BaseException propagates.
                failure = describe_provider_failure(error).category
                _LOGGER.warning(
                    "Critique call failed (operation=reflect.body, category=%s)",
                    failure,
                )
                return CritiqueAttempt(None, failure, attempt)
            try:
                data = parse_json_loose(text)
            except Exception:  # noqa: BLE001 - Malformed output is a structural failure.
                data = None
                failure = "invalid_json"
            if data is not None:
                failure = "invalid_schema"
            issues = parse_reflection(data, drafts=drafts, sources=sources)
            if issues is not None:
                return CritiqueAttempt(Reflection(issues), None, attempt)
            _LOGGER.warning("Invalid critique structure (attempt %d/%d)", attempt, attempts)
        return CritiqueAttempt(None, failure, attempts)

    def revise_continue(
        self,
        turn: Messages,
        issues: tuple[ReflectionIssue, ...],
        *,
        drafts_raw: list[str],
        drafts_formal: list[str],
        next_source: str = "",
    ) -> RevisionAttempt:
        n = len(drafts_raw)
        if n <= 0 or n != len(drafts_formal) or not issues:
            return RevisionAttempt(None, "no_issues", 0)
        payload = json.dumps(
            [
                {
                    "index": issue.index,
                    "category": issue.category,
                    "severity": issue.severity,
                    "span": issue.span,
                    "source_span": issue.source_span,
                    "accuracy_kind": issue.accuracy_kind,
                    "detail": issue.detail,
                    "suggestion": issue.suggestion,
                }
                for issue in issues
            ],
            ensure_ascii=False,
        )
        user = self.render(
            "reflector_revise_user",
            n=n,
            issues=payload,
            next_source=prompts.render_source_reference(next_source),
        ) + prompts.protected_object_guidance([message["content"] for message in turn])
        messages: Messages = [*self._editor_turn(turn), {"role": "user", "content": user}]
        attempts = self.config.pipeline.align_retry_limit + 1
        failure = "invalid_revision"
        for attempt in range(1, attempts + 1):
            try:
                text = self.client.complete(
                    messages,
                    operation="polish.body",
                    json_mode=True,
                    max_tokens=None,  # Preserve the configured polish.body limit.
                )
            except Exception as error:  # noqa: BLE001 - Ordinary failures fall back; BaseException propagates.
                failure = describe_provider_failure(error).category
                _LOGGER.warning(
                    "Revision call failed (operation=polish.body, category=%s)",
                    failure,
                )
                return RevisionAttempt(None, failure, attempt)
            try:
                data = parse_json_loose(text)
            except Exception:  # noqa: BLE001 - Malformed output is a structural failure.
                data = None
                failure = "invalid_json"
            if data is not None:
                failure = "invalid_revision"
            items = data.get("polished") if isinstance(data, dict) else data
            validated = validate_revision(
                items,
                drafts_raw=drafts_raw,
                drafts_formal=drafts_formal,
            )
            if validated is not None:
                return RevisionAttempt(validated, None, attempt)
            _LOGGER.warning("Invalid revision structure (attempt %d/%d)", attempt, attempts)
        return RevisionAttempt(None, failure, attempts)


__all__ = [
    "ACCURACY_KINDS",
    "CATEGORIES",
    "MAX_EXPLANATION_CHARS",
    "MAX_SPAN_CHARS",
    "SEVERITY_RANK",
    "CritiqueAttempt",
    "Reflection",
    "ReflectionIssue",
    "Reflector",
    "RevisionAttempt",
    "extract_raw_drafts",
    "max_issues",
    "parse_reflection",
    "validate_revision",
]
