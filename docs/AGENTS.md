# docs

## Purpose

The architecture diagram: `architecture.source.json`, the candidate it was
rendered from, and `architecture.html`, the rendered artifact. The diagram is the
one place in this repository where the whole system's shape is stated at once,
so it is worth keeping reviewable rather than shipping as an opaque blob.

## Ownership

| File | Owns |
|---|---|
| `architecture.source.json` | the source of record. Every node, edge, boundary and card, with repository-relative `sources` references |
| `architecture.html` | the rendered artifact, self-contained, generated — do not hand-edit |

## Local Contracts

- **Regenerate, never hand-edit `architecture.html`.** It is generated output;
  an edit is lost on the next render. Change the candidate and re-render.
- **The candidate is pinned to a commit.** `meta.repository.revision` names the
  commit the diagram describes, and the archify gates verify every cited source
  line against the committed bytes *at that revision*. Two consequences: commit
  the code first, then render against that commit; and never cite a file that is
  uncommitted at the pinned revision, because there are no committed bytes to
  verify against and the gate cannot confirm the line.
- **Every asserted fact carries a source reference.** If a node claims a
  behaviour, the `sources` array must point at the line that does it. A node
  without evidence is a guess wearing a diagram.
- **Re-anchor source lines after refactors.** The diagram is a snapshot; if
  `main.py` shifts, the citations are wrong even though the picture still looks
  right. Nothing checks them after the fact.
- `architecture.html` is committed even though it is generated. This repository
  has no build step and does not want one for a picture; cloning is enough to
  read the architecture. It is 784 KB, nearly all of it the inlined font.
- **No gate receipts, capture PNGs or authoring scratch in `docs/`.** Those live
  in `.archify/`, which is gitignored. `pages.yml` uploads `docs/` verbatim, so
  anything committed here becomes world-readable the moment that workflow goes
  green — treat the folder as published.
- GitHub serves a committed `.html` file as sanitised source with scripts
  stripped, so `docs/architecture.html` on github.com is *source*, not a viewer.
  The live, interactive copy is the Pages URL; the committed copy is for reading
  after a clone and for diffing.

## Work Guidance

- The diagram must stay true to the code, and the two claims that drifted most
  easily were the ones with no test behind them. A hard-coded attention
  implementation in `info()` and a "20x slower" CPU fallback both survived in
  prose and in the diagram for the same reason: nothing read them. Re-derive
  every number from `README.md` rather than from an older render.
- Only banking77 and clinc150 have train splits, only the plain and multi-label
  schema forms are covered by `scripts/verify_model.py`, and
  `baselines.build_learning_curve()` is called by nothing. If any of these become
  true, update the diagram in the same change that makes them true.
- Read the captures before committing. All four archify gates passing proves the
  file is well-formed, not that it is legible, and this is the artifact a
  stranger lands on first.

## Verification

- `archify finalize architecture docs/architecture.source.json <out.html> --repo-root . --quality showcase --json`
  must report `status: pass` with all four gates green — `validate`, `deliver`,
  `check`, `browser-check`. A non-zero exit is never success.
- `archify check <out.html> --require-provenance` re-verifies the committed
  bytes still match the delivery receipt, which is how you confirm the file in
  this folder is the one that passed.
- `archify visual-check <out.html> --require-provenance --out-dir <dir>` captures
  both themes at two viewport sizes. Inspect the PNGs; do not infer legibility
  from a passing status.
- The sha256 in `console-architecture.delivery.json` must equal the sha256 of the
  committed `architecture.html`. `.gitattributes` normalises `*.html` to LF, so
  confirm no line-ending rewrite changed the bytes.