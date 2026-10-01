# tests

## Purpose

The offline safety net for the parts of the app that can be checked without a
GPU, network access, model weights, or fixtures: evaluation statistics, LLM
reply parsing, the HTTP response contract, and the browser UI's wiring.

## Ownership

- `test_core.py` owns coverage for `app/metrics.py` and `app/llm.py`, the
  import wiring that proves the package loads in a bare environment, and the
  `/api/*` response shapes through `fastapi.testclient.TestClient`.
- `test_ui_contract.py` owns static checks on `app/static/index.html`: element
  id uniqueness, that every id the script writes to exists, and the honesty
  rules the renderers must not break.

## Local Contracts

- Offline. No test may require CUDA, network access, model weights, or
  `data/fixtures/`. CI runs this suite on `ubuntu-latest` with no GPU, so a test
  that needs one fails there and gets skipped or deleted.
- The `sys.path.insert` at the top of `test_core.py` is deliberate. The package
  is not installed — `[tool.uv] package = false` — so tests import it by path.
- Assert on semantics, not on incidental output. These tests exist to pin down
  what the numbers *mean*: exact-set match grants no partial credit, macro-F1
  depends on pairing rather than marginal counts, McNemar p-values match the
  exact binomial, a degenerate bootstrap input yields a degenerate interval, and
  a paired comparison refuses mismatched item counts.
- **A response-shape test must fail when the shape regresses.** The reason these
  exist: the panel silently stopped rendering `p95_ms`, the top-level `notes`,
  `label_coverage` and `supervised_trained_on` while the backend kept sending
  them, and nothing failed. Assert that each field the UI reads is present, and
  that the UI still reads it.
- **Pin both halves of a field, and pin the exact spelling of each.**
  `test_a_panel_field_is_sent_by_the_backend_and_read_by_the_renderer` takes the
  field, the *specific* producer modules, and the *specific* payload read. Both
  halves are required because either alone passes while the panel renders a blank
  cell. Two mutations found the gaps, and both would have shipped:
  - Searching `app/` for "any module emits this field" is blind when two modules
    emit it. Renaming `label_coverage` in `evaluate.py` left `main.py` emitting
    it, so the guard passed while the panel arm lost the field. Hence the
    per-field producer list.
  - A generic `\.notes` property search is blind because the panel reads `notes`
    twice, as run-level `d.notes` and per-system `s.notes`. Breaking the
    run-level read left `s.notes` to satisfy the search. Hence the exact reader
    expression.
- `TestClient` requests must not need weights. Stub `app.main.get_engine` with
  an object that exposes only the method the endpoint is supposed to call, so a
  regression to the old code path fails loudly instead of returning a dash.
- `test_ui_contract.py` is not a browser test. It parses the file and asserts
  wiring. It exists because there is no compiler: a mistyped or duplicated id
  writes into the wrong element and renders nothing, with no error. A duplicate
  id is the specific trap — `querySelector` returns the first match.
- **A guard that only sees one spelling of the wiring is not a guard.** The id
  checks match literal `$('#name')` lookups, so a renderer that takes the target
  as an argument — `fieldError('rLabels', 'rLabelsErr')` — is invisible to them.
  A validation slot with no matching element threw a `TypeError` at click time
  and every panel run died before sending a request. When wiring can be
  expressed more than one way, assert each way. The real pairings are
  `intField('pLimit','pLimitErr')`, `intField('pBatch','pBatchErr')` and
  `intField('sLimit','sLimitErr')`, plus `fieldError('rLabels','rLabelsErr')`
  and `fieldError('rText','rTextErr')`; a field is validated against its *own*
  slot, never a neighbour's.
- **Do not assert that a hard-coded string is the correct one.** The attention
  guards in `test_core.py` check that no attention literal appears in `info()`
  at all, rather than that `"eager"` is right, because a test pinning the value
  would pass straight back if the checkpoint switched to sdpa. Assert the
  *absence* of an assumption, or resolve the value from the thing under test.
- **Resolve values from the file, not from literals in the test.** A contrast
  check with hardcoded colours keeps asserting 4.5:1 about colours the file no
  longer uses, so it passes forever while the UI is unreadable. Parse `:root`
  and resolve the token. The same applies to a function body: match it by brace
  counting, because a non-greedy regex stops at the first nested block and then
  reports a correct function as broken.
- Match banned patterns against code with comments stripped. This repo comments
  its rules heavily and by name, so a naive substring check matches the
  documentation of the bug it is looking for. This bites in both directions:
  a `focus-visible` check satisfied by the word in a comment, and a
  `prefers-reduced-motion` check that passed on a file whose media query had
  been deleted. Assert on structure, not on a token appearing anywhere.
- Parametrize the LLM reply cases. The cosmetic-variant list is a record of what
  a live endpoint actually returned, so each entry needs the case kept.
- A test that documents a real-world failure keeps that failure in its
  docstring. The collapsed-classifier test is the stated reason the
  majority-class floor exists; the router-confidence test is the stated reason
  the engine stub has no `.model`.
- Test names state the rule, not the function under test:
  `test_compare_refuses_unpaired_item_counts`, not `test_compare_raises`.

## Work Guidance

- New statistics get a degenerate case and a boundary case. `metrics.py` is
  where quiet wrongness hides.
- Prefer small explicit prediction and reference lists over deriving them from
  summary statistics. Macro-F1 cannot be reconstructed from marginal counts, and
  a test that tried would pass for the wrong reason.
- Do not mock `torch` or `gliner2`. Anything needing those belongs in
  `scripts/verify_model.py`, which is the gate for that behaviour.
- When adding a guard, prove it bites: reintroduce the bug, watch the test fail,
  revert. A guard that has never failed is not known to work. **Mutate every
  new guard at least twice**, because a guard that passes on the first try is
  usually not asserting what its name says. Four of the UI guards written during
  the accessibility pass passed against a real defect on the first attempt — a
  contrast check with hardcoded colours, a focus check satisfied by a comment, a
  gridline check that ignored whether the plot read the token, and a
  function-body match that stopped at the first nested brace. A second and third
  mutation caught what the first did not.
- A mutation is only informative if it is realistic. "Delete the whole test" is
  not a mutation; "make the plot draw the old hardcoded colour" is. Where a guard
  was written, the mutation that exposed its blind spot was reintroducing the
  *original* bug it was meant to prevent.
- Playwright would cover more of the UI, but it needs a browser download and CI
  must stay fast. Static checks are the cheap floor; reach for the real browser
  only when a static check cannot express the rule. Layout is the honest limit:
  a 1px control stagger and a 30px row misalignment cannot be expressed as a
  regex, so verify those by measuring `getBoundingClientRect` in a browser, and
  keep the measurement out of CI.

## Verification

- `uv run pytest tests -q`
- `.venv\Scripts\python.exe -m pytest tests -q` on Windows.
- CI runs this suite on every push to `main` and on every pull request.
