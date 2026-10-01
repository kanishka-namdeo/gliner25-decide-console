# tests

## Purpose

The offline safety net for the parts of the app that can be checked without a
GPU, network access, model weights, or fixtures: evaluation statistics, LLM
reply parsing, and import wiring.

## Ownership

- `test_core.py` owns coverage for `app/metrics.py` and `app/llm.py`, plus the
  import wiring that proves the package loads in a bare environment.

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
- Parametrize the LLM reply cases. The cosmetic-variant list is a record of what
  a live endpoint actually returned, so each entry needs the case kept.
- A test that documents a real-world failure keeps that failure in its
  docstring. The collapsed-classifier test is the stated reason the
  majority-class floor exists.
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

## Verification

- `uv run pytest tests -q`
- `.venv\Scripts\python.exe -m pytest tests -q` on Windows.
- CI runs this suite on every push to `main` and on every pull request.
