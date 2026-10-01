# DOX rail

- This file is the project-wide contract for the whole repository.
- Before editing, walk from this file to the target path and read every
  `AGENTS.md` found along the route. The nearest one governs local detail; this
  one governs everything.
- After every meaningful change, update the nearest owning doc and any affected
  parent or child. A DOX pass is part of the task, not a follow-up.

## Purpose

Local evaluation console for `fastino/GLiNER2.5-Decide`, a 340M encoder that
takes a schema of labels as input at call time instead of baking them into the
weights. The repo exists to answer one adoption question with evidence rather
than promotion: given a label set, how accurate is this model, how fast, how
cheap, and does it beat simply fine-tuning something.

The headline finding is deliberately two-sided, and edits must not collapse it
to one half:

- **Stable taxonomy with labelled data available** — fine-tune. On banking77 a
  fine-tuned classifier wins by a wide, significant margin.
- **Taxonomy moves** — the schema-driven model wins categorically. A classifier
  trained on one app's intents scores exactly 0% on another's, because its head
  cannot emit a label it never saw. That is inoperable, not merely worse.

Never let a number, claim, or caveat drift in a direction the measurement does
not support. Reporting only the flattering half is a defect.

## Ownership

| Area | Owned by |
|---|---|
| HTTP surfaces, model engine, evaluation protocol, baselines, fixtures, LLM client | `app/` |
| Single-file browser UI | `app/static/` |
| Operator setup, environment and model gates, weight and fixture fetchers | `scripts/` |
| Offline test suite | `tests/` |
| CI jobs, Pages publishing | `.github/` |
| Dependency pins, CUDA index, pytest config | `pyproject.toml` |
| Narrative results, protocol description, layout, known limits | `README.md` |
| Contributor workflow and evaluation-code rules | `CONTRIBUTING.md` |
| Architecture diagram source and its rendered output | `docs/` |

Runtime artifacts are generated and unowned: `models/`, `data/fixtures/`,
`results/`. All are gitignored. Do not commit them, and do not treat edits to
them as source changes. `.archify/` is diagram-authoring scratch and is likewise
gitignored; `docs/architecture.source.json` is the source of record.

## Local Contracts

### Dependency constraints that break confusingly

- `transformers` must stay `<5`. `gliner2` 2.0.0 hard-pins
  `transformers>=4.38,<5`; 5.x breaks classification with no useful error. If
  you upgrade it, expect to debug the library, not your code.
- `torch` must resolve from the CUDA index declared in `[tool.uv.sources]`. The
  default PyPI wheel is CPU-only. If `torch.version.cuda` is `None`, that
  declaration was lost — fix `pyproject.toml`; do not install torch by hand.
- Python 3.12 only, in practice: CI installs 3.12, the classifiers say 3.12, and
  nothing exercises 3.13. `pyproject.toml` still declares
  `requires-python = ">=3.12,<3.14"`, so a 3.13 interpreter will install the
  package and then fail on an untested path. Treat 3.13 as unsupported; if it ever
  needs to work, add it to CI first. Nothing installs into system Python;
  everything runs from the uv-managed `.venv`.
- Do not adopt `gliner2`'s `[benchmark]` extra. It caps `datasets<5` and breaks
  the `datasets>=5.0.1` requirement.

### Evaluation integrity

These rules are what make the numbers mean anything. They are enforced in code,
not by convention:

- Every system scores identical items in identical order. McNemar and the paired
  bootstrap are invalid otherwise, so the `limit` slice happens once and every
  system consumes it.
- Never drop a system that fails to run. Report it in `unavailable` with a
  reason. A silently omitted arm reads as "we did not test that".
- Never fabricate a number. An unconfigured LLM arm names the missing variables;
  an unpriced token cost says so rather than showing a number that will go stale.
  A null price is not `$0.00`: for a local arm it means genuinely free, for the
  LLM arm it means nobody supplied a token price, and the two are different
  claims. `SystemResult.cost_status` carries which.
- Never grade a row the backend did not grade. A client-side quality verdict
  computed from accuracy thresholds is a defect, not a convenience: on
  `hate_speech` it marked GLiNER2.5-Decide "weak" at 44.7% while marking the
  majority-class floor "usable" at 77.0%, which inverts the finding. If a
  verdict is wanted, the backend sends it.
- Keep the majority-class floor row. It is how a model that collapsed onto one
  class was caught on `hate_speech` — 44.4% accuracy against a 77.6% floor, yet
  a macro-F1 of 0.336 against the floor's 0.291.
- Report macro-F1 next to accuracy. On `hate_speech` the two disagree sharply and
  either alone misleads.
- Never mix `fastino/fast-decisions` — the model's own training distribution —
  into a benchmark number. It appears nowhere in this repository: no fixture, no
  route, no arm. If a change wires it into a demo surface, that surface must stay
  out of every reported figure.
- Confidence values are reported as the library returns them. They are not
  calibrated probabilities and must never be presented as a decision threshold.
- Fixtures are independent public datasets only. The model's own training
  distribution never enters a reported figure.

### Secrets

- `.env` is gitignored and must stay that way. Never paste a key into a prompt,
  source file, fixture, or commit.
- `LLMConfig.status()` reports missing variable names and never the key value.
  Preserve that.

## Work Guidance

Setup, in this order — later steps fail confusingly when earlier ones are
skipped:

```powershell
uv sync --extra dev
uv run python scripts/fetch_model.py            # ~1.9 GB, resumable
uv run python scripts/fetch_fixtures.py --with-train
uv run python scripts/verify_env.py
uv run python scripts/verify_model.py
uv run uvicorn app.main:app --port 8765
```

Go through `uv run`, not a bare `python`: `verify_env.py` asserts you are inside
the project `.venv` and exits non-zero otherwise, which is the gate working, not
the gate misfiring. `--extra dev` is what provides pytest.

Adding a dataset:

- Verify the label mapping by reading samples, not by convention. The yelp
  fixture is wrong on the obvious `{0: negative, 1: positive}` guess: id 1 is
  neutral 3-star prose, id 2 is positive.
- Resolve label descriptions from the source dataset. CLINC150 ships `null` for
  every intent, which is why the described-labels variant is banking77-only.
  Report that variant as unavailable elsewhere rather than silently skipping it.
- Record resolved label names in the fixture manifest so the UI never renders a
  raw integer id.
- A supervised baseline needs a train split. Only banking77 and clinc150 have
  one.

Large downloads: `scripts/fetch_model.py` exists because `uv` and
`huggingface_hub` restart an interrupted transfer from zero, which on a slow link
means a 2 GB file never completes. Use the script, and re-run it after any
interruption.

Documentation: `README.md` carries measured results with their caveats. When a
measurement changes, update the table, the claim, and the caveat together. Keep
stated limits honest — the fine-tuned RoBERTa baseline is a *lower bound* on how
far fine-tuning can pull ahead, not the strongest possible supervised result.

## Verification

- `uv run pytest tests -q` — offline suite, no GPU or network. The fastest
  meaningful check; run it first. It covers the statistics, the LLM parsing, the
  `/api/*` response shapes, and static checks on the UI's wiring.
- `uv run python scripts/verify_env.py` — 8 environment assertions: interpreter
  version, `.venv` isolation, CUDA build, runtime availability, expected device,
  `sm_75` arch coverage, fp16 matmul finiteness, `transformers` 4.x.
- `uv run python scripts/verify_model.py` — model gate against the model card's
  documented examples. Separate from `verify_env.py` because it loads a ~1.9 GB
  checkpoint.
- CI runs the suite on CPU and separately asserts that a CUDA `torch` build and a
  4.x `transformers` resolve from the pins.

Run both `verify_*` gates before opening a pull request that touches
dependencies, the engine, or the device policy.

The UI has no build step, so nothing catches a mistyped or duplicated element id
until a line silently renders empty. `tests/test_ui_contract.py` is that
catch: id uniqueness, that every written id exists, and that the renderers have
not reintroduced a fabricated price or a client-side verdict. It also pairs every
panel field with both of its producers — the backend must still emit it *and* the
renderer must still read it — because checking one side alone passes while the
panel row is blank. Add a check there when adding a renderer, and prove it bites
by reintroducing the bug twice: a guard that passes on its first try is usually
not asserting what its name says.

## Child DOX Index

- `app/AGENTS.md` — FastAPI surfaces, model engine, evaluation protocol,
  baselines, fixtures, LLM client
- `app/static/AGENTS.md` — the single-file browser UI, no build step
- `scripts/AGENTS.md` — environment and model gates, weight and fixture fetchers
- `tests/AGENTS.md` — the offline test suite
- `.github/AGENTS.md` — CI jobs and Pages publishing
- `docs/AGENTS.md` — the architecture diagram and how to regenerate it
