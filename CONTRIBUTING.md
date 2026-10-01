# Contributing

## Getting set up

```powershell
uv sync --extra dev
uv run python scripts/fetch_model.py            # ~1.9 GB, resumable
uv run python scripts/fetch_fixtures.py --with-train
uv run python scripts/verify_env.py
uv run python scripts/verify_model.py
uv run pytest tests -q
```

The two `verify_*` scripts are the real gate. Run both before opening a pull
request — they exist because this project has hard environment constraints that
fail confusingly otherwise. They are invoked through `uv run` because
`verify_env.py` asserts you are inside the project `.venv` and exits non-zero if
you are not, so a bare `python` on a fresh shell fails the gate by design.

The two `fetch_*` scripts come first because `verify_model.py` hard-exits with
"MODEL WEIGHTS NOT FOUND" when `models/GLiNER2.5-Decide/config.json` is absent.
`uv run pytest tests -q` is the cheapest useful check while editing: it needs no
weights, no fixtures and no GPU.

## The two constraints that will bite you

**`transformers` must stay `<5`.** `gliner2` 2.0.0 hard-pins
`transformers>=4.38,<5`. Installing the current 5.x breaks classification with
no useful error. If you upgrade it, expect to debug the library, not your code.

**The PyPI `torch` wheel is CPU-only.** On Windows the default wheel is a
124 MB CPU build; CUDA wheels live on `download.pytorch.org` per CUDA version.
`pyproject.toml` declares that index via `[tool.uv.sources]`. If `torch.version.cuda`
is `None`, that declaration was lost — do not "fix" it by installing torch
manually from PyPI.

For `torch` 2.14.1, only `cu126` and `cu130` publish `cp312-win_amd64`; `cu128`
and `cu129` stop at earlier versions. `cu126` is used because compute capability
7.5 (Turing) is conservatively covered there.

## Rules for evaluation code

These are the parts that are easy to make quietly, wrongly wrong:

- **Every system must score identical items in identical order.** McNemar and
  the paired bootstrap are invalid otherwise, so `compare()` raises on a
  mismatch rather than computing something meaningless.
- **Never drop a system that fails.** Report it in `unavailable` with a reason.
  A silently omitted arm reads as "we didn't test that".
- **Never fabricate a number.** If the LLM endpoint is unset, say which
  variables are missing. Do not default a price to something that looks real —
  and note a null price is not `$0.00`. For a local arm it means genuinely free;
  for the LLM arm it means nobody supplied a token price. Those are different
  claims, so `SystemResult.cost_status` carries which one it is and the client
  must never infer it from a null.
- **Never grade a row the backend did not grade.** A quality verdict computed
  client-side from accuracy thresholds is a defect, not a convenience: it marked
  GLiNER2.5-Decide "weak" at 44.7% on `hate_speech` while marking the
  majority-class floor "usable" at 77.0%, which inverts the finding. If a
  verdict is wanted, the backend sends it.
- **Keep the majority-class floor.** A model that collapses onto one class looks
  respectable on accuracy alone. It is how we caught GLiNER2.5-Decide scoring
  44.4% against a 77.6% floor on `hate_speech`.
- **Report macro-F1 next to accuracy.** Either alone misleads in one direction
  or the other; on `hate_speech` the two disagree sharply.
- **Never mix in-domain and independent data.** `fastino/fast-decisions` is the
  model's own training distribution. It appears nowhere in this repository — no
  fixture, no route, no arm — and must never enter a benchmark number. If a
  change wires it into a demo surface, that surface stays out of every reported
  figure.

## If you add a dataset

Verify the label mapping by reading samples, not by convention. The Yelp fixture
would have been wrong on the obvious guess: id 1 is neutral 3-star prose, not
positive. Where a dataset ships descriptions, resolve them from the source rather
than inventing them — CLINC150 has `null` for every intent, which is why the
described-labels variant is banking77-only.

Record the resolved label names in the fixture manifest so the UI never has to
render a raw integer id.

## Large downloads

`scripts/fetch_model.py` is resumable on purpose. `uv` and `huggingface_hub`
restart an interrupted transfer from zero, which on a slow link means a 2 GB
file never completes. Use the script, not `from_pretrained` directly, and re-run
it after any interruption.

## Style

Follow the surrounding code. The evaluation functions are typed and return
dataclasses; the HTTP layer uses async; nothing imports torch at module scope in
the API path, because the model loads lazily on first inference and importing
torch costs seconds before the UI can respond.