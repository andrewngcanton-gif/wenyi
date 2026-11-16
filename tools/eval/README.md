# Wenyi evaluation harness (WP1a)

This is an isolated, fork-only harness for exercising the Wenyi translation
pipeline end to end. It is an independent uv project and is not a member of the
repository workspace. It does not import UI, Web, Desktop, or BabelDOC code.

WP1a is diagnostic until the owner freezes a clean A-prime commit. The offline
smoke replaces only the external LLM client with a deterministic adapter; its
outputs and costs are not translation-quality evidence and are not comparable
with a real provider run.

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
  --baseline-code /absolute/read-only/baseline \
  --candidate-code /absolute/candidate \
  --candidate-sha <clean-candidate-commit> \
  --runs-dir /absolute/disposable/eval-runs \
  --candidate-executable /venv/bin/python --candidate-argv=-m --candidate-argv=wenyi_cli \
  --baseline-executable /venv/bin/python --baseline-argv=-m --baseline-argv=wenyi_cli
```

`plan` writes the manifest using local CLI metadata commands without translating.
`extract`, `export`,
`import` and `report` rebuild later stages of an existing `--root`. Provider
mode runs the selected product factory with no evaluation monkeypatch; the
offline adapter is used only with an explicit `--adapter offline`. The
executable and argv are the command that actually runs, and `models list
--json` / `models explain --operation` metadata is captured and redacted.

A user `--config` propagates to preparation and every group; the harness
overlays only absolute state paths, the group's polish setting and Review
settings. Credentials stay environment references (`api_key_env`) and literal
credentials are rejected. Missing prices leave cost unknown instead of
disabling extraction, sampling or reporting.

`import` reads the completed `samples/pairwise.csv` by default. Use `--ratings`
to supply a separate completed copy of that same pairwise CSV format.

`--runs-dir` must be absolute. A run refuses to reuse a non-empty evaluation
directory. The smoke expects the read-only baseline checkout at
`.review/upstream` at commit `a27a176f42755432478a33db5af0dcf6a1649f97`; use
`--baseline PATH` only for that same commit. The examples in `examples/` are
consumable configuration inputs; no corpus is bundled.

## Layout and modules

```text
tools/eval/
  README.md
  pyproject.toml
  variants/            # complete per-group config templates
  examples/            # en->zh and ja->zh examples; no corpus included
  wenyi_eval/
    manifest.py        # identities, explicit argv/paths, group bindings
    sample.py          # pilot selection and stratified human-review packets
    run_variants.py    # prepare once, seed snapshots, run five groups
    extract.py         # public FileStorage reads aligned by source/index
    pairwise.py        # anonymous pairwise export and rating import
    cost.py            # operation ledger, cache/time, unknown-not-zero cost
    report.py          # Markdown/CSV report and tri-state conclusion
```

The offline bootstrap (`wenyi_eval.offline_cli`) places the selected code source
first on `sys.path` and replaces `build_client` only. The CLI, orchestrator,
storage, receipts, and terminal records are the selected product code. A binding
record proves which `wenyi_core` and `wenyi_cli` modules were imported.

## Groups and isolation

Supported groups are `A`, `A2`, `A_prime`, `A_prime2`, and `B`. `A`/`A2` use the
read-only baseline checkout; `A_prime`/`A_prime2` use the candidate checkout and
must not be the same source. `B` matches A-prime except that polish is disabled
in both its template and its CLI arguments. Rerun groups estimate variability,
not a new treatment.

`C`, `C_keep`, and `D` require the WP2 reflective-polish implementation; `E` is
the optional three-draft reference and is not implemented. Templates exist for
later use, but requesting these groups fails explicitly.

Each group has its own absolute config, state, output, cache, cwd, progress,
result, status, inspect, and log paths under
`runs/<eval-id>/<book>/<group>/`. Preparation runs once through A-prime into
`_prepared/`. The sanitized snapshot excludes body targets, usage/events,
reviews, locks, and WAL sidecars; it is copied into each group's target state.
The CLI then reuses prepared state instead of paying for preparation again.
Manifest routes, route fingerprints, model parameters, prices, and other
unavailable values are recorded as explicit `unknown`, never invented or zero.

## Evidence and conclusions

Extraction uses public `FileStorage` reads keyed by `source_sha256`, chapter, and
stable segment index. Every expected segment stays in the denominator. A
missing group, source-identity mismatch or unreadable chapter is reported as
`missing`, `stale` or `error` with a reason; stale text is never presented as a
current target. An untranslated target (`None`) stays distinct from an
intentional empty target (`""`). Before translation, the prepared snapshot's
ingestion settings, analysis policy and prescan digest policy are compared with
each group's configuration. An incompatible snapshot aborts with a written
incompatibility record rather than being silently reused.

Cost accounting uses an inclusive input partition. Cache reads, writes,
uncached input and unclassified input must add up to the prompt total. A
missing counter stays unknown, and unclassified input is only priced when the
declared rule proves every allocation has the same rate; it is never defaulted
to ordinary input pricing. Empty or incomplete ledgers produce an unknown cost.
Reflection artifacts are unavailable in WP1a and are reported as `unsupported`,
never as clean. Suggested human-review rows are not executed reviews. Pairwise
ratings start blank; missing ratings remain missing and use the preregistered
denominator.

The report is `report/summary.md` with `report/report.json`,
`report/group_summary.csv`, `report/human_summary.csv`, `report/cost.json`, and
`report/stage_costs.csv`. The conclusion is `passed`, `failed`, or
`inconclusive`; missing metrics never pass. Offline smoke runs are always
`inconclusive`.

## Known limits

- A-prime is not frozen; the smoke records an `unfrozen` binding and cannot be a
  real comparison.
- No licensed corpus, human ratings, provider price rules, or provider usage is
  bundled. Only the synthetic CC0 fixture is used.
- The external model boundary is replaced in smoke runs. No network access or
  credentials are used.
- WP2 reflection is unsupported, and C/C_keep/D/E are not runnable.
