# Contributing

## Getting set up

```powershell
uv sync --extra dev
python scripts/verify_env.py
python scripts/verify_model.py
python -m pytest tests -q
```

The two `verify_*` scripts are the real gate. Run both before opening a pull
request — they exist because this project has hard environment constraints that
fail confusingly otherwise.

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
  variables are missing. Do not default a price to something that looks real.
- **Keep the majority-class floor.** A model that collapses onto one class looks
  respectable on accuracy alone. It is how we caught GLiNER2.5-Decide scoring
  44.4% against a 77.6% floor on `hate_speech`.
- **Report macro-F1 next to accuracy.** Either alone misleads in one direction
  or the other; on `hate_speech` the two disagree sharply.
- **Never mix in-domain and independent data.** `fastino/fast-decisions` is the
  model's own training distribution. It may power demo screens, never a
  benchmark number.

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