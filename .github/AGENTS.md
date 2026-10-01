# .github

## Purpose

CI for the repository, plus the one workflow that publishes the architecture
diagram. Two CI jobs, chosen to catch the two failure modes this project
actually has rather than to exercise the whole app.

## Ownership

| Path | Owns |
|---|---|
| `workflows/ci.yml` | the CI definition: triggers, concurrency, and the two jobs |
| `workflows/pages.yml` | publishing `docs/` to GitHub Pages |

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
- `pages.yml` is deliberately narrow: it uploads `docs/` verbatim and deploys it.
  It is the only workflow with more than `contents: read`, and its extra scopes
  (`pages: write`, `id-token: write`) are the minimum Pages deployment set. It
  needs no token, no model, and no fixture.
- Pages is configured to build from the Actions workflow, not from a branch.
  That is the reason a workflow exists instead of a `gh-pages` branch: the
  diagram is committed once on `main`, next to the code it describes, and a
  second copy on another branch is a copy that can silently go stale.

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
- Anything committed under `docs/` becomes world-readable the moment `pages.yml`
  succeeds. Keep generated gate receipts, capture PNGs, and authoring scratch out
  of it — `.archify/` is gitignored for exactly that reason.

## Verification

- The workflow is itself the verification: it runs on every push and pull
  request.
- Reproduce the `tests` job locally with `uv sync --extra dev` followed by
  `uv run pytest tests -q`.
- After changing anything under `docs/`, confirm the Pages run went green and
  that `https://kanishka-namdeo.github.io/gliner25-decide-console/` serves the
  updated file. GitHub renders a committed `.html` file as sanitised source, so
  the Pages URL is the only way a reader can actually open the diagram.
