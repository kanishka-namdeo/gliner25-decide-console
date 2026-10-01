# app

## Purpose

The application package. One engine, three product surfaces:

| Surface | Entry point | Job |
|---|---|---|
| Router | `POST /api/route` | classify arbitrary text against an arbitrary schema |
| Evidence panel | `POST /api/panel` | every system scored on one fixture, with CIs and significance tests |
| Schema swap | `POST /api/schema-swap` | the disjoint-taxonomy experiment that justifies a schema-driven model |

Supporting endpoints: `GET /api/health` (runtime facts, LLM status, fixture
list), `GET /api/fixtures/{name}` (inspect one fixture), `GET /api/llm/probe`
(cheapest possible endpoint round trip), `GET /` (the UI shell).

## Ownership

| File | Owns |
|---|---|
| `main.py` | FastAPI app, pydantic request models, schema construction from request heads |
| `engine.py` | `DecisionEngine`: device policy, fp16, response parsing, timing, disk cache |
| `evaluate.py` | panel orchestration, per-system arms, artifact caching, the schema-swap experiment |
| `metrics.py` | the evaluation protocol: exact-set match, bootstrap CIs, paired bootstrap, exact McNemar, macro-F1, majority floor |
| `baselines.py` | supervised comparison arms — TF-IDF, MiniLM+LR, fine-tuned RoBERTa, learning curve |
| `fixtures.py` | fixture loading and schema construction |
| `llm.py` | OpenAI-compatible `/chat/completions` client, config, reply parsing, pricing |
| `static/` | the browser UI, governed by its own AGENTS.md |

## Local Contracts

### Load order and lazy imports

- The model loads on first inference, not at import. `get_engine()` is a
  process-wide singleton because loading costs ~7s and ~1.8 GB, and `/api/health`
  must answer while the checkpoint is still being read.
- Never import `torch`, `transformers`, `gliner2`, `pandas`, or `sklearn` at
  module scope in the API path. Importing torch costs seconds before the UI can
  respond, which defeats the lazy-load design.
- Reconfigure `sys.stdout` and `sys.stderr` to UTF-8 before importing `gliner2`.
  Its config banner contains an emoji and raises `UnicodeEncodeError` on a
  cp1252 Windows console.

### Device policy (`engine.py`)

- Prefer CUDA with `quantize=True` for fp16. bf16 requires sm_80+ and the target
  GPU is Turing sm_75, so fp16 is the only mixed-precision option available.
- `from_pretrained` may reject `quantize` depending on signature. Catch
  `TypeError` and retry without it rather than dying.
- Attention stays eager. DeBERTa-v2 rejects SDPA in transformers 4.x.
- `info()` reports only capabilities actually obtained. It must never claim fp16
  on a CPU run, and it must never advertise a capability the app did not get.
- CPU fallback is supported and roughly 20x slower. The UI says so rather than
  pretending otherwise.

### Timing

- Warm up before timing. The first call pays lazy init and allocator growth;
  charging that to the first chunk inflated measured p50 by ~3.6x on CPU, which
  would have made the latency column a fiction.
- `torch.cuda.synchronize()` around every timed region on CUDA, or the numbers
  measure kernel launch rather than work.
- Time in chunks so progress reporting does not add per-call overhead.

### Schema shapes

Three forms, all verified by `scripts/verify_model.py`:

```
{"intent": ["a", "b"]}                                 # single label
{"intent": {"labels": {"a": "does A", "b": "does B"}}}  # described labels
{"topics": {"labels": [...], "multi_label": True, "cls_threshold": 0.4}}
```

- A schema is the model's input contract: labels arrive at call time and are
  never baked into weights. Build it from the fixture so the label set is
  exactly the set the gold labels came from. Otherwise the evaluation is not
  honest.
- `classify_text` returns several shapes depending on the head. `engine._parse`
  normalises them; keep that the single normalisation point.
- Label sets above 512 are rejected at the request boundary.

### Caching

- Prediction cache lives in `results/`, keyed by fixture and mode, and validated
  against a schema fingerprint and item count. Change a label set and the
  fingerprint changes, so a stale cache can never be silently reused.
- Fitted baseline artifacts live in `results/artifacts/`. Reuse them. Fine-tuning
  is the expensive part, and re-running the UI must not repeat it.
- Corrupt cache JSON is logged and ignored, never raised.

### Evaluation protocol (`metrics.py`, `evaluate.py`)

The repo-wide integrity rules live in the root AGENTS.md. Locally they are
enforced by these mechanisms:

- `metrics.compare()` raises on mismatched item counts rather than computing a
  meaningless p-value.
- `exact_set_match` compares normalised label *sets*. No partial credit, and a
  superset is wrong.
- Bootstrap is seeded (`DEFAULT_SEED = 0`) so reported intervals reproduce.
- Exact McNemar over the chi-square approximation, because discordant-pair
  counts are small and the approximation is unreliable there.
- `SystemResult.correct` carries the per-item vector needed for pairing;
  `to_public()` strips it before the payload reaches the client.
- The majority-class floor is always appended when a majority label exists.
- A system that cannot run goes into `unavailable` with a reason. It never
  contributes a fabricated number.
- Label descriptions are cheap on small schemas and expensive on large ones —
  measured ~8.6x slower at 77 labels. The panel warns above 25 labels instead of
  letting the latency column surprise the reader.

### Schema swap (`evaluate.py`)

- Split labels deterministically via `fixtures.split_labels` (seed 0) so the
  result reproduces across runs and machines.
- Score only items whose gold label is in that app's schema. An item belonging
  to the other app is out of scope for this schema by construction.
- The supervised arm must be trained on App A only and then asked about App B.
  Its 0% is a structural property of a fixed classification head, not a defect
  to be tuned away.
- The split comparison is the whole argument for a schema-driven model. Keep it
  honest and keep its explanation intact.

### LLM arm (`llm.py`, `evaluate.py`)

- Show the model the bare label list and no per-label descriptions. The encoder
  arm does get descriptions on banking77; handing only one side them would make
  the comparison meaningless.
- `_extract_label` normalises cosmetic variants of a real label — case,
  quotes, code fences, `-` vs `_`, a short lead-in ("The answer is X"), and
  positional indices — because those are the same answer. Observed live: the
  endpoint answered `Refund_not_showing_up` where the schema said
  `refund_not_showing_up`. An invented label outside the schema stays invalid;
  that is a genuine error, not a formatting quirk.
- Match a lead-in only at a separator boundary (the reply must **end** with
  `_<label>`). Splitting on `_` destroys compound labels: it turns
  `card_pin_change` into `card`/`pin`/`change`, which never matches. Matching on
  bare prefix instead credits prose to whatever label shares its first
  characters — `"the customer is unhappy"` must never be scored as
  `theft_report`. Both were real bugs; the boundary rule is what fixes them.
- There is deliberately **no substring fallback**. Once normalisation handles the
  cosmetic cases, a label merely appearing inside a sentence is a wrong answer.
- Bounded concurrency, default 4 (`llm_concurrency` on the request, 1–16). A
  serial loop is correct but useless at the measured ~6-7s per call — 12 items
  took 37s at concurrency 4 versus ~78s serial — while unbounded parallelism
  invites rate limiting.
- Long runs report their per-item cost in `notes` so an operator can choose a
  smaller item count deliberately rather than assuming a hang.
- The UI caps the item count above 120 when the LLM arm is selected and says why.
  Do not raise that ceiling without re-measuring.
- Credentials come from `.env`, read by `load_env_file()`. It is hand-rolled on
  purpose: one more dependency to pin for six lines is a bad trade. Environment
  variables win over the file so an operator can override a bad key locally.
  Never log or return the key value; `status()` names missing variables only.
- Result order must match input order regardless of completion order, because the
  paired significance tests assume identical item ordering.
- Unusable replies count as wrong, and the count is reported.
- Token prices default to 0 and display as unpriced. Do not assert a price.

### HTTP layer

- The API is async; blocking work runs through `asyncio.to_thread`.
- Progress callbacks passed into `run_llm` must be **sync**. `run_llm` awaits
  `progress()` while holding a lock, so an async callback would be created and
  never awaited.
- Report unsupported or unavailable states as data carrying a reason, not as a
  missing section and not as an invented number.

## Work Guidance

- Evaluation functions are typed and return dataclasses.
- Adding a system to the panel means extending `SYSTEM_LABELS` and adding an arm
  in `run_panel`, keeping the failure path reporting into `unavailable`.
- Keep measurements that surprised you in a comment with the number and the
  hardware. A future reader must be able to tell a measurement from a guess.
- Keep the *why* beside each protocol rule. This repo's value is that the
  numbers are trustworthy; an unexplained rule is one someone will optimise away.
- Long-running arms should note their per-item cost so an operator can choose a
  smaller item count deliberately.

## Verification

- `uv run pytest tests -q` covers `metrics.py` and `llm.py` parsing offline.
- `python scripts/verify_env.py` and `python scripts/verify_model.py` gate any
  change to the device policy or schema handling in `engine.py`.
- Changes to `evaluate.py`, `baselines.py`, or `fixtures.py` need a real GPU run
  against a fetched fixture to confirm. CI does not exercise them.

## Child DOX Index

- `static/AGENTS.md` — the single-file browser UI
