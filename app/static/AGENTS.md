# app/static

## Purpose

The entire browser UI in one file: `index.html`, with inline CSS and inline
vanilla JS. No framework, no bundler, no build step. Served at `/` as a
`FileResponse` and mounted at `/static` through `StaticFiles`.

Three tabs, one per product surface: Router against `/api/route`, Evidence panel
against `/api/panel`, Schema swap against `/api/schema-swap`. A header badge
strip reports device and LLM status from `/api/health`. The tab is in
`location.hash`, so a surface is linkable and the browser Back button moves
between them.

The panel leads with a dot-and-whisker plot of accuracy against its bootstrap
interval, because overlapping intervals are the finding and a bracketed number
in a table cell is not legible enough to see that.

The console is dark-only, dense, and keyboard-first: it is an operator tool for
reading measurements, not a product surface. Density is a deliberate choice and
the 12px text floor is not negotiable underneath it.

## Ownership

- `index.html` owns markup, styles, and all client logic. Nothing else in this
  folder is source.
- `tests/test_ui_contract.py` (in `tests/`) owns the static checks on this
  file's wiring. It is the closest thing to frontend verification that exists.

## Local Contracts

- One file. Do not split it or introduce a build step without first recording
  that decision in the repository root AGENTS.md.
- No framework, no bundler, no CDN dependency. The UI must work offline once the
  API is running. `test_no_cdn_or_external_asset` enforces this, allowing only
  XML namespace URIs, which are identifiers rather than fetches.
- This is a client of `app/`, not a second source of truth. Every value shown
  comes from an API response. Never compute, round, re-derive, or invent a
  statistic the backend did not send.
- **Never grade a row the backend did not grade.** A client-side
  `strong`/`usable`/`weak` column computed from accuracy thresholds is a defect
  with a name: on an imbalanced fixture it marked GLiNER2.5-Decide "weak" at
  44.7% and the majority-class floor "usable" at 77.0%, inverting the finding
  the repo exists to report. If a verdict is wanted, the backend sends it.
- **A null cost is not `$0.00`.** Branch on `cost_status`: `free`, `priced`,
  `unpriced`. An unpriced LLM arm rendered as `$0.00` reads as free and
  understates the one arm whose cost is why it loses.
- Render `unavailable` and `notes` payloads as received. A system that could not
  run must read as "did not run", never as absent. The panel's top-level `notes`
  carry the LLM arm's per-item cost; render them in `#pRunNotes`.
- Show latency, accuracy, macro-F1, and the confidence interval together —
  **p50 and p95**. Dropping the interval makes a point estimate look like a
  fact, and dropping p95 hides the tail that is the LLM arm's whole story
  (measured p50 9,363 ms against p95 25,321 ms).
- Label the majority-class row as a floor, not as a system. It is dimmed, and it
  is excluded from the forest plot when it has no interval.
- Confidence is the library's score, not a calibrated probability, and never a
  decision threshold. A multi-label head reports the *lowest* confidence of the
  labels emitted, so say so rather than implying it scores one label.
- Never display an LLM API key. The backend deliberately never sends one.
- `#pNote` is permanent: it carries the caveat that the data is independent of
  the model's training distribution. A run writes `#pFixtureNote` instead. This
  is the single most load-bearing line on the tab and it once survived only until
  the first run.
- Preserve the fixture's own note when present — CLINC150's out-of-scope
  labelling and the yelp mapping are recorded there.
- **Element ids must be unique.** A duplicate id is silent: `querySelector`
  returns the first match, so the renderer writes into the wrong element and the
  intended line renders empty. This happened once, with a fixture-detail div
  sharing an id with the fixture `<select>`.
- **Render payload data as text nodes, never as markup.** A template literal
  into `innerHTML` is a sink for strings nobody reviewed: the fixture's own note
  and the comparison system names are free text. It reads as ordinary
  formatting, so it survives review. `el()` and `createTextNode` only.
- **A missing optional key must not destroy a good render.** `comparisons` and
  `unavailable` are read after the systems table is built. Reading them without
  a `|| []` default throws into the catch block, which used to replace a correct
  table with a red box. Losing real numbers to a shape regression is worse than
  losing the section. The catch block now keeps what already rendered.
- **Never `confirm()`/`alert()`/`prompt()`.** A blocking modal freezes the page,
  cannot be styled with the rest of the console, and puts a decision in the
  browser's chrome instead of next to the field it concerns. The slow-LLM
  warning used one. It now renders in-page beside the item-count field and
  offers both the fix and a deliberate override.
- **An estimate shown to an operator must carry its unit conversion.** The
  slow-arm warning read `n * 3` and printed "minutes" over a measurement that is
  ~3 *seconds* per item: 300 items was announced as 900 minutes when it takes
  about 15. An estimate wrong by 60x makes a real ceiling look arbitrary and
  teaches the reader to ignore the warning. `llmEstimate()` owns the
  conversion; call it rather than writing arithmetic in a string.
- **Consent is per value, not per form.** The override that lets a slow run
  proceed is keyed to the item count it was granted for. Editing the field
  un-arms it, because agreeing to 300 items is not agreeing to 5,000.
- **Numeric fields carry a client-side sanity ceiling of 100,000 items,** well
  above the largest fixture slice, so a typo cannot queue a run that reads as
  hung. This is a typo guard, not a policy limit: the server has no item ceiling,
  and the 120-item slow-LLM confirmation in `syncLlmWarning()` is the real
  operator-facing gate. Say which of the two a number is when you add one.
- **The palette has one source of truth: `:root`.** Resolve colours with
  `getPropertyValue` rather than repeating a hex in JS. The forest plot
  hardcoded its gridline, so `--grid` was introduced and the chart kept drawing
  the old value — two sources of truth, and the contrast check was describing
  the one nobody rendered.
- **`--accent` is for marks; `--accent-solid` is for fills under white text.**
  White on `#2f81f7` is 3.75:1 and fails AA, while the same blue as a chart mark
  on the panel is a correct 4.62:1. One token cannot serve both jobs, so
  folding them together is what made every Run button label unreadable.
- **Keyboard focus must be visible on every interactive control.** The UA
  default is a 1px dotted outline that is nearly invisible against `--panel`.
  `:focus-visible` covers the controls, and `outline:none` is forbidden without
  a replacement.
- **Motion is opt-out.** Any `transition` requires a
  `prefers-reduced-motion: reduce` block. A fade on a control is small, but the
  preference is one media query away and the console is already dense.
- **Text has a 12px floor.** The caveat lines are the load-bearing text — what
  a confidence number is not, why a row is missing, what a run cost — and they
  were 11.5px in the dimmest grey on the page, smaller than the table headers
  and the badges. Smallest and greyest text was carrying the most meaning.
- **A wide table scrolls; it does not shrink below legibility.** An overflowing
  div is invisible to Tab, so every `.scroll` wrapper carries `tabindex="0"`,
  `role="region"`, and a label saying what scrolls. The forest plot gets a
  `min-width`: scaling a 560-unit viewBox to a 375px phone renders 11px labels
  at about 7px, which is an unreadable chart rather than a scrollable one.
- **Validation errors belong beside the field, and the row must not move.**
  `intField()` writes to a `#...Err` slot wired by `aria-describedby` and moves
  focus to the first bad field. Those ids are passed as strings, so the literal
  `$('#id')` checks cannot see them; `test_every_field_error_slot_exists` does.
  Every control sits in a `.field-aux` reserve so an appearing error does not
  reflow the row.
- **Give every control an accessible name.** A bare `×` announces as
  "multiplication sign"; the chip remove buttons carry
  `aria-label="Remove label <name>"`. System chips are `<label>` elements, so
  the whole pill is the pointer target and the checkbox inherits its name.
- **Confidence is not rendered in a verdict colour.** It was green, directly
  above a caveat saying it is not a calibrated probability. Green on a number is
  a grade, and this repo has a rule against grading rows the backend did not.

## Work Guidance

- Backend response shapes are the contract. Changing a field in `main.py`,
  `evaluate.py`, or `metrics.py` means updating the renderers here in the same
  change.
- Element ids are the wiring. Full list, because a partial list reads as a
  complete one: `rText`, `rLabels`, `rHead`, `rMulti`, `rRun`, `rChips`, `rOut`,
  `rStatus`, `rResult` for the router; `pFixture`, `pLimit`, `pBatch`,
  `pSystems`, `pForest`, `pResults`, `pRunNotes`, `pStatus`, `pLlmWarn`, `pRun`,
  `pFixtureNote`, `pOut`, `pCompare`, `pUnavailable` for the panel; `sFixture`,
  `sLimit`, `sStatus`, `sRun`, `sOut`, `sResults` for schema swap; `bDevice`,
  `bLlm` for the header; plus the tab trio `tabbtn-route` / `tab-panel` /
  `tab-swap` and their panels `tab-route` / `tab-panel` / `tab-swap`. Each
  numeric field has a matching `#<name>Err` slot. Rename one and its
  `getElementById` goes with it. `test_element_ids_are_unique` and
  `test_every_written_element_exists_in_the_markup` are the mechanical checks;
  this list exists so a rename is a deliberate act.
- `/api/route` returns a fixed shape per head, `{labels: [...], confidence}`.
  Do not re-derive it in the client; that duplication is how the dead confidence
  column survived, because the endpoint's bug was invisible to the renderer.
- Normalise `detail` through `errText()`. FastAPI returns a string for
  `HTTPException` and an array of objects for a 422, and the array renders as
  `[object Object]`.
- Keep new work in the existing visual language: the current CSS custom
  properties, the card / badge / chip classes, and the existing spacing scale.
- `--dur` is the only motion token. Do not copy a duration literal into a new
  rule; if something genuinely needs a different timing, add a named token and
  say why it differs.
- `--control-h` fixes the height of every input, select, and run button, because
  the rows bottom-align and a `<select>` renders 34.4px where a text input
  renders 32.8px. Set an exact `height`, not `min-height`, or the 1px stagger
  returns.
- The tab lives in `location.hash` via `pushState`, so a surface is linkable and
  Back works. Use `pushState` on click and never `replace`, or history entries
  disappear. `popstate` re-selects without pushing, or Back loops.
- **The tab strip is a real ARIA tablist.** Arrow keys move between tabs, Home
  and End jump to the ends, and a roving `tabIndex` keeps exactly one tab in the
  Tab order so the strip is one stop rather than three. Clicking must go through
  the same selection path as a keypress, or the roving index desynchronises from
  the visible selection.
- Panel and schema-swap requests block on the server. Say so — the LLM arm costs
  seconds per item — rather than letting the UI look hung. Progress goes in
  `#pStatus` / `#sStatus` / `#rStatus` (`role="status"`), never by wrapping the
  results table in `aria-live`, which makes a screen reader re-announce every
  cell on each run.
- Prefer inline SVG over a charting dependency. For one mark type it is less
  code than the library's configuration, and it cannot violate the offline rule.
- The comparison box says which system is higher, in words, above the signed
  difference. A bare `-14.9 pts` leaves the reader to work out the direction, and
  getting it backwards inverts the comparison the panel exists to make. A
  non-significant result gets a neutral surface, not a blank one: it is the
  honest answer, not missing information.

## Verification

- `uv run pytest tests -q` includes `tests/test_ui_contract.py`, which checks
  id uniqueness, that every written element exists, that every validation slot
  exists and is wired by `aria-describedby`, that `#pNote` cannot be
  overwritten, that no `$0.00` or client-side verdict returns, that both latency
  percentiles are rendered, that each panel field is both sent by the backend
  and read here, that no payload value reaches `innerHTML`, that
  there is no blocking dialog, that focus is visible and motion is opt-out, that
  the palette clears WCAG AA (4.5:1 text, 3:1 indicators), that the chart
  resolves `--grid` from the stylesheet, that the time estimate converts units,
  and that the file is still one file with no external asset. These run offline
  in about a second; run them first.
- **Prove a new guard bites.** Reintroduce the bug and watch the test fail.
  Four guards here were written wrong and passed against a real defect: a
  contrast check with hardcoded colours, a focus check satisfied by the word
  `focus-visible` in a comment, a gridline check that ignored whether the plot
  read the token, and a function-body match that stopped at the first nested
  brace. A guard that has never failed is not known to work.
- Manual: start the API with `uv run uvicorn app.main:app --port 8765`, open `/`,
  and confirm the header badges populate, the router returns a decision *with a
  confidence and a bar* in neutral text, the panel renders the plot, rows,
  comparisons, and any `unavailable` entries while `#pNote` still reads as the
  independence caveat, and Tab reaches every control with a visible ring.
  Check the console is clean: several defects here were a `TypeError` at click
  time, which no static check would have found.
- Measure layout rather than eyeballing it. `getBoundingClientRect().top` on
  each control in a row is how the 1px stagger and the 30px misalignment from a
  helper line were both caught.
