# .github

## Purpose

CI for the repository. Two jobs, chosen to catch the two failure modes this
project actually has rather than to exercise the whole app.

## Ownership

- `workflows/ci.yml` owns the CI definition: triggers, concurrency, and the two
  jobs.

## Local Contracts

- `tests` job: CPU only, no GPU, no weights, no fixtures, `ubuntu-latest`. The
  `uv sync --extra dev` step does use the network, like any install; the
  *test run* is the offline part, and it needs no network of its own. Fast, and
  always meaningful.
- `resolution` job: exists because neither dependency trap shows up in a CPU
  test run. It asserts `torch.version.cuda is not None` — the default PyPI wheel
  is CPU-only, so `[tool.uv.sources]` must keep pointing at
  `download.pytorch.org` — and that `transformers` resolves to 4.x, since
  `gliner2` 2.0.0 pins `<5`.
- Python 3.12 via `uv python install 3.12`, matching `requires-python`.
- Triggers are pushes to `main` and all pull requests. The `concurrency` group
  cancels superseded runs in the same group.
- No job may require a GPU, model weights, or dataset fixtures. Nothing in CI may
  need a secret or a paid endpoint.

## Work Guidance

- Adding a dependency pin means adding a `resolution` assertion for it. That job
  is the cheap place to catch a resolution trap that would otherwise only appear
  on one developer's machine.
- Keep GPU-dependent paths out of CI entirely. `scripts/verify_env.py` and
  `scripts/verify_model.py` are local gates by design and must stay runnable
  offline rather than being promoted into a workflow.
- CI must stay green without reaching any external LLM endpoint. The LLM arm is
  never exercised here.
- No workflow may read `.env`. It is gitignored, so CI does not have it, and a job
  that needs credentials would leak its existence. The offline suite covers
  `load_env_file()` and reply parsing against fixtures instead.

## Verification

- The workflow is itself the verification: it runs on every push and pull
  request.
- Reproduce the `tests` job locally with `uv sync --extra dev` followed by
  `uv run pytest tests -q`.
