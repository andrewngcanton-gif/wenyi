# Translation workflow

Wenyi prepares a book before translating its chapters, saves completed work as it
goes, then reviews and exports the translation. This guide explains the choices that
affect quality, time, cost, and recovery. For commands, use the [CLI guide](cli.md);
for settings, use the [configuration reference](../reference/configuration.md).
Input preservation and export limitations belong to the
[format reference](../reference/formats.md).

<a id="overview"></a>

## The default workflow

1. Parse and save the source.
2. Prepare chapter digests, style guidance and a whole-book synopsis.
3. Translate and polish chapters, align formatting and update the glossary.
4. Review the complete book and publish revisions when Autofix is enabled.
5. Export the saved translation.

The defaults are standard translation, whole-book understanding, polishing,
annotation alignment, final Review, and **Autofix publishing enabled**.
Three-draft precision and whole-book terminology evidence are opt-in.
The default export is monolingual; a bilingual edition is optional.

<a id="quality-and-cost"></a>

### Quality and cost choices

| Choice | What it buys | Cost or limitation |
| --- | --- | --- |
| Whole-book understanding | Chapter facts and an overview before the first body batch | Reads all chapter source text; long chapters need multiple prescan calls |
| Polishing | A second pass for target-language expression | Adds a model call to a normal standard batch; fluency is not proof of fidelity |
| Reflective polishing | A structured critique before polishing, then a gated revision | Adds a critique call, and a revision or ordinary polish call; an invalid critique falls back rather than saving a call |
| Three-draft precision | Three independent attempts followed by source-aware synthesis | Normally four body-stage calls instead of two with standard translation and polishing |
| Annotation/style alignment | Places links and mixed formatting in translated text | Partial spans can require additional model calls; uncertain placements use format-specific fallbacks |
| Review and Autofix | Whole-book consistency checks, evidence gathering and revisions | Can make several passes and additional Fixer calls; publishing changes the saved translation |
| Larger context or terminology evidence | More cross-passage evidence | Larger prompts and, for evidence production, additional calls |

Disabling a stage reduces its work, not necessarily the total cost in a predictable
ratio. Retries, output length, reasoning tokens, and provider prices matter. Prompt
prefix reuse may help provider caching but does not guarantee a cache hit.
None of these options guarantees a more accurate translation; compare representative
passages and review important results yourself.

<a id="reflective-polish"></a>

### Reflective polishing

`pipeline.reflective_polish` is opt-in and applies only to standard translation
with polishing enabled. Instead of polishing immediately, the batch continues the
translation conversation with a structured critique of style, fluency, idiom and
accuracy. Only issues that pass strict validation and reach
`reflect_min_severity` are sent back with a revision request. Filtered or
untranslated paragraphs keep their draft, and the revision must preserve the
paragraph count, protected objects and ruby markers.

- **Clean result:** a complete, valid critique with no issue at or above the
  threshold. `reflect_clean_batches: polish` then runs ordinary polishing;
  `keep` saves the draft and skips that call.
- **Fallback:** a missing, partial, malformed or failed critique never counts as
  clean. The batch uses ordinary polishing, and if that also fails, the standalone
  polish path. `keep` is never applied to an unvalidated critique.
- **Cost:** a normal enabling batch makes translation → critique → revision, or
  translation → critique → polish when clean. Structural retries and fallback calls
  are billed too. Nothing here guarantees a better translation; compare
  representative passages yourself.

Every enabled batch also writes a diagnostic record under `reflection/` for later
inspection. It is not a checkpoint: standard batches remain atomic, interrupted work
is redone as a whole, and a missing or stale record is never treated as a clean
batch.

**Heterogeneous critic (experimental).** By default `reflect.body` inherits the
effective `polish.body` route, so one model critiques, revises and polishes. An
explicit `llm.routes.reflect.body` entry can move only the structured critique to
another connection and profile. The gated revision still uses the `polish.body`
route, and `translation.body` keeps its route. Both reachable providers need their
own credentials. A prepared, disabled-by-default example with placeholders is
`tools/eval/examples/heterogeneous-critic.experimental.yaml`; field semantics are
in the [configuration reference](../reference/configuration.md).

Inspect a copy locally with `wenyi --config <copy> models list --json`,
`wenyi --config <copy> models explain --operation reflect.body` and
`wenyi --config <copy> models check --for translate`.
These commands read configuration and credential presence; they do not contact a
provider, prove remote model availability or establish translation quality.
With reflection disabled, `check` excludes critic credentials. To verify that
boundary locally, enable reflection in an isolated copy with dummy credentials;
do not translate with that validation copy. An approved evaluation may enable
reflection before human scoring. Daily use still depends on all quality gates; see
`tools/eval/FINAL_ACCEPTANCE.zh.md` and the evaluation guide `tools/eval/README.md`.
Until those gates pass, it is not a recommendation for daily use.

<a id="preparation"></a>

## Preparation before translation

Parsing identifies chapters and paragraphs without calling a translation model.
`wenyi parse` and Web/Desktop upload parsing save a read-only
[source snapshot](../reference/formats.md#source-snapshots). PDF parsing can still
require MinerU or BabelDOC. Parsing alone does not initialize translation state.

`prepare`, or preparation at the start of `translate`, then:

1. Detects the source language when `language.source` is `auto`. An explicit source
   language skips that request.
2. With `pipeline.book_understanding: true`, summarizes the **complete source of
   every chapter**, splitting long chapters into bounded chunks.
3. Builds style and character-voice guidance from original opening, middle and
   ending samples, with chapter digests as factual context.
4. Builds the whole-book synopsis from that guidance and the chapter digests.

Preparation **does not translate body text or extract glossary entries**. It keeps
existing project terms, including imported mappings. Turning off book understanding
skips chapter digests and the synopsis, not style analysis.

With book understanding enabled, a chapter containing text must have a usable digest
before body translation starts. Digest failure stops preparation; a failed whole-book
synopsis produces a warning and allows translation to use the chapter digests instead.
Complete successful digests and style analysis are reusable on a matching retry.
An unfinished chapter's individual prescan chunks are not separately checkpointed.

<a id="context-and-glossary"></a>

## Translation context and the glossary

Chapters and body batches progress in source order. A normal batch receives style
guidance, the book synopsis, its chapter digest, the **full current glossary**,
relevant source notes, recent translated text, the requested source, and one following
source segment as a read-only reference. The following segment helps with continuity;
it is not translated early or included in the expected output count.

Recent translation is reconstructed from saved targets, including preceding chapters.
`pipeline.rolling_context_tokens` defaults to 512 tokens.
It favors complete recent paragraphs; an oversized newest
paragraph contributes only its tail. Setting the budget to zero removes preceding
translation, not the following source reference. BabelDOC uses a
[layout-aware context boundary](../reference/formats.md#babeldoc) instead.

Ordinary glossary extraction starts with completed source-and-translation pairs.
Without imported terms, the first batch therefore has no predefined mappings.
New terms and conflicts inform later batches; they do not automatically rewrite
earlier translations. The glossary is authoritative for name mappings, while style
guidance retains source identities and descriptive character information.

Each pending body batch receives all current mappings, including terms absent from
its chapter. Fresh Reviewer requests receive the complete final glossary.
This favors book-wide consistency at the cost of larger prompts. Evidence queries
and individual Fixers can use relevant subsets.

After EPUB body translation, Wenyi completes the parsed navigation tree and adds its
source-and-translation pairs to the glossary before final Review, including nested,
reused and previously saved titles. Existing glossary translations are retained;
different navigation translations become conflicts for resolution. Review uses these
mappings to check body headings, visible contents and prose. Publishing its revisions
requires Autofix. With Review disabled, navigation terms are still saved, but this
step does not synchronize body text or visible contents during export.

### Optional whole-book terminology evidence

`pipeline.terminology_context` is an experimental, CLI-only option, off by default.
It discovers people before pending translation batches and checks person entries
against matching source passages throughout the book. Other term types retain
ordinary posttranslation extraction. Preparation still does not seed a glossary.

The feature adds discovery, evidence and merge calls, plus bounded relevant notes in
translation, polishing, Review and Fixer requests. It preserves authoritative
translations and protected manual fields; a different proposed translation becomes a
conflict. Notes are supplementary evidence, not permission to reveal future plot
details or invent information. One term still has one main target, not independent
translations for multiple senses. Failed evidence does not publish partial knowledge;
matching successful stages can be reused on retry.

## Three-draft precision

`pipeline.translation_mode: three_draft_synthesis` means **three independent drafts followed
by one source-aware synthesis**, not choosing a winning draft.
Polishing must be enabled.

```text
Same frozen source and context
  ├─ independent draft T1 ─┐
  ├─ independent draft T2 ─┼─ source-aware synthesis → final translation
  └─ independent draft T3 ─┘
```

The three drafts run concurrently and do not see one another. Synthesis receives
their proposals as fallible data, prioritizes the source, and combines useful parts.
There is no independent accuracy judge. The stored before-polish comparison is T1,
not a draft certified to be the most accurate.

A normal batch makes three draft calls and one synthesis call,
versus one translation and one polish call in standard mode. It uses the same model
routes and output limits. Four versus two calls does **not** mean a fixed 2× price:
input sizes, output tokens, retries and the other book stages still count.

If a draft or synthesis cannot produce a usable result after the configured retries,
the batch stops instead of silently switching to single-paragraph translation.
Only the failed stage is retried; compatible successful drafts remain reusable on resume.

Use the same source, mode, context and model settings to reuse pending checkpoints.
Changing the mode does not redo completed formal translations. Commands and
creation-time project choices are documented in the
[CLI guide](cli.md) and [configuration reference](../reference/configuration.md).

## Review and Autofix

Review requires every chapter to be translated. It examines contiguous chunks with
the final glossary, can request evidence for uncertain issues, reconciles conflicting
recommendations, and may propose revisions. Additional rounds recheck the revised
text and add model usage.

**`pipeline.review_autofix` defaults to `true`, so Review can change the saved
translation.** With Autofix disabled, new Review work produces recommendations
without publishing them to the book.

To request recommendations without starting publication:

```bash
wenyi review book.epub --no-autofix
```

Use `pipeline.review_autofix: false` for the same preference in the complete workflow.
`--autofix` forces publishing for an independent Review when that preference is off.
Disabling Autofix prevents new publication, but an already-started, journaled
publication is recovered first and can still write formal targets even when this
invocation uses `--no-autofix`. The switch is not a rollback or a way to cancel
that committed publication.

`pipeline.review: false` or `translate --no-review` skips the automatic Review stage.
An explicit `wenyi review` still runs. Matching content, glossary, guidance, language
policy and model identities reuse a completed result or resume unfinished work;
**each invocation does not necessarily create a new Review**. Changes to those
inputs can require a fresh run and new model calls.

Review may stop at its round budget, with unresolved issues, or after detecting no
progress or a revision cycle. A later pass not reporting an issue again is not proof
that its replacement is correct. Inspect the saved result before relying on a
publication for accuracy-sensitive work.

## Pause and resume

Rerun the same input with the same target language and compatible configuration.
CLI state is isolated under `state/<book>/targets/<target-language>/`; Web and
Desktop keep independent project state. A different file with the same name is not
the same source.

- Completed translation batches remain saved. Changing polishing or translation mode
  does not automatically reprocess them.
- Resume may first finish terminology extraction for an already translated batch;
  this does not mean the batch is being translated again.
- Preparation, precision and Review reuse successful matching checkpoints rather than
  requesting their completed model stages again. In-flight work without a durable
  success record may need another request.
- A language-policy or prompt change can rebuild affected derived guidance and incur
  model usage while preserving completed translations and glossary entries.
- Finishing translation can retry previously failed annotation placements. A
  `translate` invocation on a completed book is therefore not necessarily model-free.

Use `wenyi status` for saved progress and `wenyi inspect-run` for recorded events,
usage and timing. Cumulative token totals include earlier attempts and resumes, not
only the latest invocation. Keep state and source files when recovering; use a
separate project/state directory for intentional fresh translations or comparisons.

<a id="re-export"></a>

## Re-export without translating again

`wenyi assemble` renders saved state **without translation-model calls**. Use it for
another format, bilingual order, source styling, or export-only punctuation changes.
It does not rerun preparation, polishing, Review, or annotation repair.

An export uses one consistent snapshot of the saved book.
Translation may continue elsewhere; export again to include batches saved afterward.
Original templates and, for PDF, the BabelDOC session may still be required.
See [export snapshots](../reference/formats.md#export-snapshots) and
[PDF requirements](../reference/formats.md#pdf).

For checkpoint ownership, publication journals, cache identities and accounting
rules, see the [developer workflow contracts](../development/pipeline.md).
