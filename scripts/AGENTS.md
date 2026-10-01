# scripts

## Purpose

Operator-facing setup and verification. Four scripts with two distinct roles:
gates that refuse to let you proceed, and fetchers that acquire the artifacts the
gates check.

| Script | Role |
|---|---|
| `verify_env.py` | cheap environment gate: interpreter, `.venv` isolation, CUDA build, runtime, device, arch coverage, fp16 finiteness, dependency pins |
| `verify_model.py` | model gate: load the checkpoint and confirm it behaves as the model card documents |
| `fetch_model.py` | resumable weight download into `models/` |
| `fetch_fixtures.py` | dataset sampling into `data/fixtures/` with manifests |

## Ownership

This folder owns the setup sequence and the gate criteria. `pyproject.toml` owns
the pins the gates assert, so the two are changed together or not at all.

## Local Contracts

- Gates exit non-zero on a hard failure so CI or a pre-commit hook can enforce
  them.
- `verify_model.py` stays split from `verify_env.py`. It loads a ~1.9 GB
  checkpoint and must only run after the cheap checks pass. Do not merge them.
- Only `verify_model.py` distinguishes hard failures from soft observations, and
  only because the model card labels its documented outputs as "potential
  results". A disagreement there is information, not a bug. Shape, finiteness,
  and device problems remain hard failures.
- `verify_model.py` requires the local copy at
  `models/GLiNER2.5-Decide/config.json` and exits 2 with the fetch command when
  it is missing. Never let it fall back to the Hub id — `from_pretrained` would
  restart the download from zero.
- `fetch_model.py` resumes with HTTP range requests, validates by file size, and
  skips already-complete files. That resume path is the entire reason the script
  exists: `uv` and `huggingface_hub` restart from zero, which never completes on
  a slow link. Keep it intact.
- These scripts print human-readable output and return an exit code. They are not
  importable libraries.
- Reconfigure stdout and stderr to UTF-8 before importing `gliner2`
  (`verify_model.py`). Its banner emoji raises `UnicodeEncodeError` on a cp1252
  Windows console.

## Work Guidance

- Keep gate checks numbered and individually reported. A gate that stops at the
  first failure without context is not a gate.
- Adding a dependency pin means adding a matching gate assertion. The
  constraints that bite are the ones something cheap can catch.
- `fetch_fixtures.py` owns the label-id-to-name mapping, and that mapping is
  load-bearing: handed a raw id like `"61"` instead of `card_lost`, the model has
  nothing to condition on.
- Every evaluated fixture written must carry a manifest with resolved label
  names, label counts, class counts, and source. The UI reads it, and it must
  never be made to render an integer id. The `--with-train` splits are the
  exception and write only `<name>_train.csv`: they are supervised input with no
  label vocabulary of their own, and `fixtures.available()` globs
  `*.manifest.json`, so a sibling manifest would advertise a fixture that does
  not exist.
- Sampling is shuffled with a fixed seed so a truncated fixture still covers
  every label rather than only the early ids.
- Record the reason a non-obvious dataset decision was made in the spec entry.
  The yelp mapping note is what stops the obvious guess being re-applied.

## Verification

- `uv run python scripts/verify_env.py` - 8 assertions, must pass before any app
  work. Invoke it through `uv run`: it asserts you are inside the project `.venv`
  and exits non-zero otherwise, so a bare `python` fails the gate by design.
- `uv run python scripts/verify_model.py` - must pass before a pull request
  touching the engine, the device policy, or schema shapes. It loads the
  ~1.9 GB checkpoint and checks the model card's documented examples. Note it
  covers the plain and multi-label schema forms but **not** the described-labels
  form; that one is only exercised by a real banking77 run.
- `uv run pytest tests -q` after changing `verify_env.py` criteria, since CI
  asserts the same pins independently.
