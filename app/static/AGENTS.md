# app/static

## Purpose

The entire browser UI in one file: `index.html`, with inline CSS and inline
vanilla JS. No framework, no bundler, no build step. Served at `/` as a
`FileResponse` and mounted at `/static` through `StaticFiles`.

Three tabs, one per product surface: Router against `/api/route`, Evidence panel
against `/api/panel`, Schema swap against `/api/schema-swap`. A header badge
strip reports device and LLM status from `/api/health`.

## Ownership

- `index.html` owns markup, styles, and all client logic. Nothing else in this
  folder is source.

## Local Contracts

- One file. Do not split it or introduce a build step without first recording
  that decision in the repository root AGENTS.md.
- No framework, no bundler, no CDN dependency. The UI must work offline once the
  API is running.
- This is a client of `app/`, not a second source of truth. Every value shown
  comes from an API response. Never compute, round, re-derive, or invent a
  statistic the backend did not send.
- Render `unavailable` and `notes` payloads as received. A system that could not
  run must read as "did not run", never as absent.
- Show latency, accuracy, macro-F1, and the confidence interval together.
  Dropping the interval makes a point estimate look like a fact.
- Label the majority-class row as a floor, not as a system.
- Never display an LLM API key. The backend deliberately never sends one.
- Keep the panel's note that the data is independent of the model's training
  distribution. It is visible on that tab by design.
- Preserve the fixture's own note when present — CLINC150's out-of-scope
  labelling and the yelp mapping are recorded there.

## Work Guidance

- Backend response shapes are the contract. Changing a field in `main.py`,
  `evaluate.py`, or `metrics.py` means updating the renderers here in the same
  change.
- Element ids are the wiring: `rText`, `rLabels`, `rHead`, `rMulti` for the
  router; `pFixture`, `pLimit`, `pBatch`, `pSystems` for the panel; `sFixture`,
  `sLimit` for schema swap; `bDevice`, `bLlm` for the header. Rename one and its
  `getElementById` goes with it.
- Keep new work in the existing visual language: the current CSS custom
  properties, the card / badge / chip classes, and the existing spacing scale.
- Panel and schema-swap requests block on the server. Say so — the LLM arm costs
  seconds per item — rather than letting the UI look hung.

## Verification

- No automated frontend verification exists.
- Manual: start the API with `uv run uvicorn app.main:app --port 8765`, open `/`,
  and confirm the header badges populate, the router returns a decision, and the
  panel renders rows, comparisons, and any `unavailable` entries.
