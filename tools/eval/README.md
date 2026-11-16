# Wenyi evaluation harness

This is an isolated, fork-only harness for exercising the Wenyi translation
pipeline end to end. It is an independent uv project and is not a member of the
repository workspace. It does not import UI, Web, Desktop, or BabelDOC code.

The harness records diagnostic and acceptance evidence. It is not a
model-quality study: offline adapters and dirty or unfrozen code cannot support
translation-quality, model-behavior or provider-cost conclusions.

## Commands

```bash
# From the repository root.
uv run --project tools/eval python -m pytest -q -n auto tools/eval/tests
uv run --project tools/eval python -m wenyi_eval.run_variants --help
uv run --project tools/eval python -m wenyi_eval.smoke \
  --offline --runs-dir /absolute/disposable/eval-runs
```

A real evaluation is planned and run through one entrypoint:

```bash
uv run --project tools/eval python -m wenyi_eval.run_variants run \
  --source /absolute/licensed/book.epub --license CC-BY-4.0 \
  --source-language ja --target-language zh \
  --config tools/eval/examples/ja-zh.yaml \
  --baseline-code /absolute/read-only/upstream \
  --wp0-code /absolute/read-only/wp0-baseline \
  --candidate-code /absolute/candidate \
  --candidate-sha <clean-candidate-commit> \
  --runs-dir /absolute/disposable/eval-runs \
  --candidate-executable /venv/bin/python --candidate-argv=-m --candidate-argv=wenyi_cli \
  --wp0-executable /venv/bin/python --wp0-argv=-m --wp0-argv=wenyi_cli \
  --baseline-executable /venv/bin/python --baseline-argv=-m --baseline-argv=wenyi_cli
```

`plan` writes the manifest using local CLI metadata commands without translating.
`extract`, `export`, `import` and `report` rebuild later stages of an existing
`--root`. Add `--protocol /absolute/registered-protocol.json` to `run` for a
formal pre-registered experiment; without it the run stays diagnostic. Provider
mode runs the selected product factory with no evaluation monkeypatch; the
offline adapter is used only with an explicit `--adapter offline`. The executable
and argv are the command that actually runs, and `models list --json` /
`models explain --operation` metadata is captured and redacted.

## Three code bindings

The harness resolves exactly three source roles and never substitutes one for
another:

| Role | Default checkout | Groups |
| --- | --- | --- |
| `baseline` | `.review/upstream` at `a27a176f…` | `A`, `A2` |
| `wp0` | `.review/wp0-baseline` at `4267537c…` | `A_prime`, `A_prime2`, `B` |
| `candidate` | the current workspace | `C`, `C_keep`, `D` |

`--baseline-code`, `--wp0-code` and `--candidate-code` overrides stay tied to
their expected SHAs. Dirty, unpinned or mismatched bindings are recorded as
`dirty`, `unfrozen` or `mismatch`; they are allowed only in diagnostic runs.
The candidate is never relabeled as A-prime or claimed as a frozen quality
candidate.

## Groups and isolation

The default selection preserves the historical five-group evaluation:
`A,A2,A_prime,A_prime2,B`. Select an explicit subset with `--groups`, for
example `--groups A,A_prime,C,C_keep,D`; only selected groups are planned, run,
extracted and reported, so no unrequested paid run is added.

- `A`/`A2` measure the upstream baseline and its rerun.
- `A_prime`/`A_prime2` measure the frozen WP0 baseline and its rerun.
- `B` matches A-prime except that polish is disabled.
- `C` enables `reflective_polish` with `reflect_clean_batches=polish` and lets
  `reflect.body` inherit its `polish.body` route.
- `C_keep` changes only the clean-batch strategy to `keep`.
- `D` is C plus an explicit heterogeneous `reflect.body` profile. Supply
  `--heterogeneous-reflect-model <profile>` naming a profile in the user's
  `llm.models`. The harness proves the resolved reflect and polish provider/model
  identities differ; family heterogeneity stays an operator declaration. It
  never invents a model ID or credential.
- `E` remains the optional three-draft reference group and is not implemented.

Control and legacy groups never receive WP2-only pipeline fields or
`llm.routes.reflect.body`, even if the user configuration enables reflection.
Requesting `C`, `C_keep` or `D` with a non-standard `translation_mode` fails
explicitly.

Each group has its own absolute config, state, output, cache, cwd, progress,
result, status, inspect, and log paths under `runs/<eval-id>/<book>/<group>/`.
Preparation runs once into `_prepared/`. The sanitized snapshot excludes body
targets, usage/events, reviews, locks, and WAL sidecars; it is copied into each
group's target state.

## Preparation compatibility

The persisted preparation contract is compared strictly against every selected
group: source identity, target language, ingestion settings, analysis policy,
analysis language policy and the prescan chapter-digest policy. Fingerprints are
never relaxed to make reuse succeed.

A full LLM configuration participates in the chapter-digest identity, so D's
explicit reflect route can make the shared snapshot incompatible. When that
happens the run aborts with a written incompatibility record unless
`--allow-isolated-reprepare` is supplied. With that explicit authorization the
harness prepares only the incompatible group under
`<group>/_prepared/`, records a separate terminal and cost, and never
overwrites shared or other-group state. D is described as runnable only when
this isolated path is actually exercised.

## Reflection evidence

Extraction reads the current post-alignment chapters and delegates each
artifact comparison to the product's `reflection_artifact_status`. The expected
batch identity, policy fingerprints and configured route fingerprints are
reconstructed independently from the group configuration, never copied from
the artifact being validated.

Every expected segment stays in the denominator. Per-segment and per-batch
status is one of `current`, `stale`, `missing`, `invalid`, `ambiguous` or
`not_enabled`:

- `current` only means the record applies to the current text. It is never a
  clean verdict and `skipped_no_turn` remains an outcome, not cleanliness.
- stale, missing, invalid and ambiguous records are never counted as adoption.
- duplicate or overlapping batch keys are not selected by mtime and are not
  counted twice.
- a diagnostic write failure is reported separately from a missing artifact.
- untranslated (`None`) and intentionally empty (`""`) targets stay distinct.
- ordinary baseline/B reflection is `not_enabled`.

Reports include per-group reflection coverage, critique, adoption, failure and
observed-route evidence. Missing or stale coverage makes the report fail; it is
never silently dropped or treated as clean.

## Human evaluation and cost

Anonymous pairwise exports compare both the historical pairs and
`A_prime` vs `C` / `C_keep` / `D` when those groups are selected. Blind packets
carry opaque pair ids; the separate key maps ids to comparison and side groups.
Missing ratings and missing/stale evidence remain in the preregistered
denominator. Suggested rows are not executed reviews, and the export is
described as anonymous, not double blind.

Cost accounting uses an inclusive input partition. Cache reads, writes,
uncached input and unclassified input must add up to the prompt total. A
missing counter stays unknown, and unclassified input is only priced when the
declared rule proves every allocation has the same rate. Unknown prices stay
unknown, never zero. Isolated preparation is charged as its own scope.

## Formal protocol and quality gates

A real quality claim needs a pre-registration. `plan`/`run --protocol <file>`
validates and freezes comparisons, cluster unit, holdout chapters, seed,
resampling method and every threshold in `experiment_protocol.frozen.json`
before any translation output exists. Later stages verify the stored hash and
reject a protocol edited after output; a missing protocol keeps the legacy run
diagnostic/inconclusive. Generate a clearly labelled draft with:

```bash
uv run --project tools/eval python -m wenyi_eval.run_variants protocol-template \
  --out /absolute/protocol.draft.json
```

The draft thresholds are suggestions, not an approved study. Confirmatory
comparisons must use `A_prime` as the baseline; multiple confirmatory
comparisons require a declared correction (Holm/Bonferroni). `C_keep` may be
exploratory or explicitly registered. A pilot scope or a holdout without
selected and assessed source evidence can never produce a release-quality pass.
Holdout chapters must exist in the
source inventory, be selected separately from the main sample, and have recorded
paired assessments with sufficient clusters and passing preference, MQM and
severe-error checks. Setting `sampling.holdout.verified` to `true` alone is not
held-out evidence.

Reports evaluate every HLD 9.4 gate separately: preference (cluster-paired
two-sided 95% lower bound > 50% plus human MQM non-deterioration), omission and
addition (one-sided 95% upper bound <= +1pp on paired assessable segments),
severe newly introduced mistranslation (blocks with failed), degradation over
segments the candidate actually edited (one-sided 95% upper bound <= 5%), body
cost ratio <= 1.40 on the same complete workload, completeness, frozen evidence,
multiple-comparison correction and final holdout. All-win/all-loss preference,
zero candidate defects and zero confirmed degradations return unavailable
intervals and inconclusive gates. A binomial bound over clusters cannot establish
the pooled segment error-rate bound when cluster sizes differ, and an all-zero
bootstrap cannot prove a threshold. Workflow execution and empirical quality are reported
separately, and diagnostic/synthetic runs are always inconclusive.

Human material is extended with per-side MQM error counts and severe flags in
`samples/pairwise.csv`, plus a separate blind before/after packet in
`samples/edit_review.csv`. Annotations are bound to source segment and text
hashes; changed outputs or stale keys invalidate the whole submission. Human
origin is caller-reported and the design is anonymous, never double blind.
Combined import validates both CSVs before committing either; an invalid CSV
leaves prior ratings and import summaries unchanged. Reporting revalidates the
current persisted protocol, keys, texts and annotation bindings instead of
trusting an earlier successful import or its summary.

Human scoring remains deferred until the final frozen-material stage. Prepare
the registered protocol, model profiles, licensed data and budget before any
paid study. Toolchain readiness is not empirical quality evidence, and
`reflective_polish` remains off by default pending all required gates.
See [FINAL_ACCEPTANCE.zh.md](FINAL_ACCEPTANCE.zh.md) for the Chinese
final-acceptance checklist.

## Layout and modules

```text
tools/eval/
  README.md
  FINAL_ACCEPTANCE.zh.md
  pyproject.toml
  variants/            # complete per-group config templates
  examples/            # en->zh and ja->zh examples; no corpus included
  wenyi_eval/
    manifest.py        # identities, explicit argv/paths, group bindings
    protocol.py        # frozen pre-registration, content hash, draft template
    sample.py          # pilot selection and stratified human-review packets
    run_variants.py    # prepare once, seed snapshots, run selected groups
    extract.py         # public FileStorage reads aligned by source/index
    reflection.py      # WP2 artifact identity and status evidence
    pairwise.py        # anonymous pairwise export, rating import, hash binding
    edit_review.py     # blind before/after edit packet for degradation
    stats.py           # cluster-paired resampling and conservative sparse handling
    gates.py           # HLD 9.4 gate-by-gate tri-state evaluation
    cost.py            # operation ledger, cache/time, unknown-not-zero cost
    report.py          # Markdown/CSV report and tri-state conclusion
```

The offline bootstrap (`wenyi_eval.offline_cli`) places the selected code source
first on `sys.path` and replaces `build_client` only. The CLI, orchestrator,
storage, receipts, and terminal records are the selected product code. A binding
record proves which `wenyi_core` and `wenyi_cli` modules were imported.

The report is `report/summary.md` with `report/report.json`,
`report/group_summary.csv`, `report/human_summary.csv`, `report/cost.json`, and
`report/stage_costs.csv`. The conclusion is `passed`, `failed`, or
`inconclusive`; missing metrics never pass. Diagnostic runs are always
`inconclusive` unless an execution check fails.

## Known limits

- The current workspace carries uncommitted WP2 code; C/C_keep/D are diagnostic
  and unfrozen, not quality candidates.
- No licensed corpus, human ratings, provider price rules, or provider usage is
  bundled. Only the synthetic CC0 fixture is used.
- Every model boundary in local acceptance is a deterministic adapter or a local
  HTTP stub. No network access, credential or paid endpoint is used.
- The harness verifies that D's resolved reflect and polish identities differ;
  it cannot independently verify semantic model-family heterogeneity.
- Reflection expected policy and routes are reconstructed with the wenyi-core
  installed in `tools/eval`. A run produced by a different candidate should be
  re-extracted with that candidate's harness environment, otherwise records are
  honestly reported as stale.
