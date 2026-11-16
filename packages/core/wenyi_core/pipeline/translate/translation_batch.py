"""Explicit batch inputs and translate/polish results without formal state mutation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...agents.polisher import Polisher
from ...agents.reflector import ReflectionIssue, Reflector, extract_raw_drafts
from ...agents.translator import Translator
from ...ingest.segmenter import batch_segments
from ...markup.ruby import strip_ruby_markers
from ...models.document import Segment
from ...models.glossary import GlossaryTerm


@dataclass(frozen=True)
class BatchPlan:
    """Capture the source positions and context used for exactly one serial batch."""

    chapter: int
    start_index: int
    segment_indices: tuple[int, ...]
    sources: tuple[str, ...]
    terms: tuple[GlossaryTerm, ...]
    context: str
    style: str
    book_synopsis: str
    chapter_digest: str
    annotation_contexts: list[list[dict[str, str]]]
    next_source: str
    allow_empty_translations: bool = False

    @classmethod
    def capture(
        cls,
        chapter: int,
        start_index: int,
        segments: list[Segment],
        terms: list[GlossaryTerm],
        context: str,
        style: str,
        book_synopsis: str,
        chapter_digest: str,
        annotation_contexts: list[list[dict[str, str]]],
        next_source: str,
        *,
        allow_empty_translations: bool = False,
    ) -> BatchPlan:
        """Detach mutable term and annotation metadata from subsequent chapter updates."""
        return cls(
            chapter,
            start_index,
            tuple(s.index for s in segments),
            tuple(s.source for s in segments),
            tuple(deepcopy(terms)),
            context,
            style,
            book_synopsis,
            chapter_digest,
            deepcopy(annotation_contexts),
            next_source,
            allow_empty_translations,
        )


@dataclass(frozen=True)
class ReflectivePolicy:
    """Configured critique threshold and clean-batch action for one batch."""

    min_severity: str = "major"
    clean_action: str = "polish"


@dataclass(frozen=True)
class ReflectionOutcome:
    """One batch's critique, adoption and failure evidence; never a quality verdict."""

    # validated_clean | validated_gated | invalid | provider_failure | skipped_no_turn
    critique: str
    # revised | ordinary_polished | draft_kept | standalone_polish_attempted
    adoption: str
    adoption_verified: bool  # Confirms the adopted stage only, never semantic quality.
    failure: str | None = None
    positions: tuple[int, ...] | None = None
    issues: tuple[ReflectionIssue, ...] = ()
    gated: tuple[ReflectionIssue, ...] = ()
    critique_attempts: int = 0
    revision_attempts: int = 0
    ordinary_polish_attempted: bool = False
    standalone_polish_attempted: bool = False

    def summary(self) -> dict[str, Any]:
        """Return bounded primitives for the batch event without issue text."""
        return {
            "critique": self.critique,
            "adoption": self.adoption,
            "adoption_verified": self.adoption_verified,
            "failure": self.failure,
            "issue_count": len(self.issues),
            "gated_count": len(self.gated),
            "critique_attempts": self.critique_attempts,
            "revision_attempts": self.revision_attempts,
            "ordinary_polish_attempted": self.ordinary_polish_attempted,
            "standalone_polish_attempted": self.standalone_polish_attempted,
        }


@dataclass(frozen=True)
class BatchResult:
    targets: tuple[str, ...]
    before_polish: tuple[str | None, ...]
    precision_key: str | None = None
    reflection: ReflectionOutcome | None = None
    raw_drafts: tuple[str, ...] | None = None


class TranslationBatchExecutor:
    def __init__(
        self,
        translator: Translator,
        polisher: Polisher,
        reflector: Reflector | None = None,
    ):
        self._translator = translator
        self._polisher = polisher
        self._reflector = reflector

    def execute(
        self,
        plan: BatchPlan,
        *,
        polish: bool,
        reflective: ReflectivePolicy | None = None,
    ) -> BatchResult:
        """Translate and optionally polish; the chapter service alone applies the result."""
        terms = list(plan.terms)
        targets = self._translator.translate_batch(
            list(plan.sources),
            glossary_terms=terms,
            style=plan.style,
            context=plan.context,
            book_synopsis=plan.book_synopsis,
            chapter_digest=plan.chapter_digest,
            annotation_contexts=plan.annotation_contexts,
            next_source=plan.next_source,
            allow_empty_translations=plan.allow_empty_translations,
        )
        targets = [strip_ruby_markers(target) for target in targets]
        before_polish: tuple[str | None, ...] = (None,) * len(targets)
        reflection: ReflectionOutcome | None = None
        raw_drafts: tuple[str, ...] | None = None
        if polish:
            before_polish = tuple(targets)
            turn = self._translator.last_batch_turn
            indices = self._translator.last_batch_indices
            if reflective is not None and self._reflector is not None:
                targets, reflection, raw_drafts = self._reflect_and_polish(
                    plan,
                    targets,
                    terms=terms,
                    turn=turn,
                    indices=indices,
                    policy=reflective,
                )
            else:
                polished = self._continue_or_standalone(
                    plan,
                    targets,
                    terms=terms,
                    turn=turn,
                    indices=indices,
                )
                if len(polished) == len(targets):
                    targets = polished
        return BatchResult(
            tuple(targets), before_polish, reflection=reflection, raw_drafts=raw_drafts
        )

    def _reflect_and_polish(
        self,
        plan: BatchPlan,
        targets: list[str],
        *,
        terms: list[GlossaryTerm],
        turn: list[dict[str, str]] | None,
        indices: list[int] | None,
        policy: ReflectivePolicy,
    ) -> tuple[list[str], ReflectionOutcome, tuple[str, ...] | None]:
        """Critique a successful turn, then adopt the best verified stage.

        Invalid critique, missing evidence and provider failures always fall back to
        ordinary polishing; no such result can become ``validated_clean`` or keep the draft.
        """
        assert self._reflector is not None
        reflector = self._reflector
        positions = tuple(indices) if indices else None
        if turn is None or not indices:
            polished, outcome = self._standalone_with_outcome(
                plan,
                targets,
                terms=terms,
                critique="skipped_no_turn",
                positions=positions,
            )
            return polished, outcome, None

        raw_drafts = extract_raw_drafts(turn, n=len(indices))
        if raw_drafts is None:
            polished, outcome = self._ordinary_polish(
                plan,
                targets,
                terms=terms,
                turn=turn,
                indices=indices,
                critique="invalid",
                failure="invalid_draft",
                positions=positions,
            )
            return polished, outcome, None

        sources = [plan.sources[index] for index in indices]
        critique = reflector.critique_continue(
            turn,
            drafts=list(raw_drafts),
            sources=sources,
            next_source=plan.next_source,
        )
        if critique.reflection is None:
            invalid_structure = critique.failure in {"invalid_json", "invalid_schema", "no_turn"}
            polished, outcome = self._ordinary_polish(
                plan,
                targets,
                terms=terms,
                turn=turn,
                indices=indices,
                critique="invalid" if invalid_structure else "provider_failure",
                failure=critique.failure,
                positions=positions,
                critique_attempts=critique.attempts,
            )
            return polished, outcome, raw_drafts

        reflection = critique.reflection
        gated = reflection.gated(policy.min_severity)
        if not gated:
            if policy.clean_action == "keep":
                return (
                    list(targets),
                    ReflectionOutcome(
                        critique="validated_clean",
                        adoption="draft_kept",
                        adoption_verified=True,
                        positions=positions,
                        issues=reflection.issues,
                        critique_attempts=critique.attempts,
                    ),
                    raw_drafts,
                )
            polished, outcome = self._ordinary_polish(
                plan,
                targets,
                terms=terms,
                turn=turn,
                indices=indices,
                critique="validated_clean",
                positions=positions,
                issues=reflection.issues,
                critique_attempts=critique.attempts,
            )
            return polished, outcome, raw_drafts

        revision = self._reflector.revise_continue(
            turn,
            gated,
            drafts_raw=list(raw_drafts),
            drafts_formal=[targets[index] for index in indices],
            next_source=plan.next_source,
        )
        if revision.items is not None:
            polished = list(targets)
            for index, text in zip(indices, revision.items):
                polished[index] = text
            return (
                polished,
                ReflectionOutcome(
                    critique="validated_gated",
                    adoption="revised",
                    adoption_verified=True,
                    positions=positions,
                    issues=reflection.issues,
                    gated=gated,
                    critique_attempts=critique.attempts,
                    revision_attempts=revision.attempts,
                ),
                raw_drafts,
            )

        polished, outcome = self._ordinary_polish(
            plan,
            targets,
            terms=terms,
            turn=turn,
            indices=indices,
            critique="validated_gated",
            failure=revision.failure,
            positions=positions,
            issues=reflection.issues,
            gated=gated,
            critique_attempts=critique.attempts,
            revision_attempts=revision.attempts,
        )
        return polished, outcome, raw_drafts

    def _ordinary_polish(
        self,
        plan: BatchPlan,
        targets: list[str],
        *,
        terms: list[GlossaryTerm],
        turn: list[dict[str, str]],
        indices: list[int],
        critique: str,
        positions: tuple[int, ...] | None,
        issues: tuple[ReflectionIssue, ...] = (),
        gated: tuple[ReflectionIssue, ...] = (),
        failure: str | None = None,
        critique_attempts: int = 0,
        revision_attempts: int = 0,
    ) -> tuple[list[str], ReflectionOutcome]:
        """Continue the original translation conversation, then use standalone polish."""
        continued = self._polisher.polish_continue(
            turn,
            n=len(indices),
            next_source=plan.next_source,
        )
        if continued is not None and len(continued) == len(indices):
            polished = list(targets)
            for index, text in zip(indices, continued):
                polished[index] = strip_ruby_markers(text)
            return (
                polished,
                ReflectionOutcome(
                    critique=critique,
                    adoption="ordinary_polished",
                    adoption_verified=True,
                    failure=failure,
                    positions=positions,
                    issues=issues,
                    gated=gated,
                    critique_attempts=critique_attempts,
                    revision_attempts=revision_attempts,
                    ordinary_polish_attempted=True,
                ),
            )
        polished = self._standalone(plan, targets, terms=terms)
        return (
            polished,
            ReflectionOutcome(
                critique=critique,
                adoption="standalone_polish_attempted",
                adoption_verified=False,
                failure=failure,
                positions=positions,
                issues=issues,
                gated=gated,
                critique_attempts=critique_attempts,
                revision_attempts=revision_attempts,
                ordinary_polish_attempted=True,
                standalone_polish_attempted=True,
            ),
        )

    def _standalone_with_outcome(
        self,
        plan: BatchPlan,
        targets: list[str],
        *,
        terms: list[GlossaryTerm],
        critique: str,
        positions: tuple[int, ...] | None,
        failure: str | None = None,
        critique_attempts: int = 0,
    ) -> tuple[list[str], ReflectionOutcome]:
        """Use the existing whole-batch polish when no translation turn can be continued."""
        polished = self._standalone(plan, targets, terms=terms)
        return (
            polished,
            ReflectionOutcome(
                critique=critique,
                adoption="standalone_polish_attempted",
                adoption_verified=False,
                failure=failure,
                positions=positions,
                critique_attempts=critique_attempts,
                standalone_polish_attempted=True,
            ),
        )

    def _continue_or_standalone(
        self,
        plan: BatchPlan,
        targets: list[str],
        *,
        terms: list[GlossaryTerm],
        turn: list[dict[str, str]] | None,
        indices: list[int] | None,
    ) -> list[str]:
        """Keep the accepted default path unchanged when reflection is not requested."""
        if turn is not None and indices is not None:
            # Continue the translation conversation and restore filtered source positions.
            continued = self._polisher.polish_continue(
                turn,
                n=len(indices),
                next_source=plan.next_source,
            )
            if continued is not None and len(continued) == len(indices):
                polished = list(targets)
                for index, text in zip(indices, continued):
                    polished[index] = strip_ruby_markers(text)
                return polished
        return self._standalone(plan, targets, terms=terms)

    def _standalone(
        self,
        plan: BatchPlan,
        targets: list[str],
        *,
        terms: list[GlossaryTerm],
    ) -> list[str]:
        return self._polisher.polish(
            targets,
            glossary_terms=terms,
            style=plan.style,
            next_source=plan.next_source,
            sources=list(plan.sources),
        )


def resume_batches(segments: list[Segment], max_tokens: int) -> list[list[Segment]]:
    """Split token-budget batches again at completed/pending boundaries.
    A changed budget may mix saved translations and unset targets in one batch. Group by
    completion state to translate only missing paragraphs and avoid overwriting confirmed
    content. A saved blank string is complete; only None means pending.
    """
    batches: list[list[Segment]] = []
    for raw_batch in batch_segments(segments, max_tokens):
        current: list[Segment] = []
        current_done: bool | None = None
        for segment in raw_batch:
            done = segment.target is not None
            if current and done != current_done:
                batches.append(current)
                current = []
            current.append(segment)
            current_done = done
        if current:
            batches.append(current)
    return batches
