# GLiNER2.5-Decide console

[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)

A local app for demonstrating what [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
is good at, and just as importantly, where it is not.

The model is a 340M encoder that takes a **schema of labels as input at call
time** rather than baking them into the weights, and can score several heads in
one forward pass. That makes the interesting questions operational rather than
academic: given your label set, how accurate is it, how fast, how cheap, and
does it beat simply fine-tuning something?

## What it does

Three surfaces, one engine:

1. **Router** — paste text, define any label set, get a decision with
   confidence. Add a label or edit a schema and the model immediately knows
   about it. No retraining. Multiple heads score in one call.
2. **Evidence panel** — run every available system over identical items and
   compare with confidence intervals and significance tests.
3. **Schema swap** — the experiment that justifies a schema-driven model.

## The honest finding

On a **stable taxonomy with labelled data available**, a fine-tuned classifier
wins, and by a lot. Measured here on banking77 (77 intents, 10,000 train rows):

| system | accuracy | 95% CI | macro-F1 | p50 | training labels |
|---|---|---|---|---|---|
| GLiNER2.5-Decide | 72.3% | [67.3, 77.3] | 0.686 | 52 ms | **0** |
| TF-IDF + logistic regression | 86.3% | [82.3, 90.0] | 0.862 | 0.6 ms | 10,000 |
| MiniLM + logistic regression | **93.0%** | [90.0, 95.7] | 0.935 | 5.4 ms | 10,000 |

Both gaps are significant (McNemar p ≈ 6.5e-06). If you have a fixed taxonomy
and 10,000 labelled examples, fine-tune something. That is the honest
recommendation and the app says so.

A frontier LLM does not rescue it either. Adding `qwen3.7-plus` (zero-shot, bare
label list, no per-label descriptions — the encoder arm gets those on banking77)
to the same 60 items:

| system | accuracy | 95% CI | macro-F1 | p50 | p95 |
|---|---|---|---|---|---|
| GLiNER2.5-Decide | 75.0% | [63.3, 85.0] | 0.598 | 66 ms | 72 ms |
| TF-IDF + logistic regression | 96.7% | [91.7, 100.0] | 0.953 | 1 ms | 3 ms |
| qwen3.7-plus (zero-shot) | 91.7% | [83.3, 98.3] | 0.846 | **9,363 ms** | **25,321 ms** |

So the LLM lands between the encoder and a supervised baseline on accuracy, at
roughly **140x the encoder's latency** and non-zero marginal cost. On this axis it
is not the answer — the paper's finding is that it only earns its price where the
schema moves or labels are scarce.

But when the taxonomy **moves**, the ordering inverts categorically. Split
banking77's intents into two disjoint halves:

| system | accuracy |
|---|---|
| GLiNER2.5-Decide on App A (38 intents) | 65.3% [56.1, 74.5] |
| GLiNER2.5-Decide on App B (39 intents, never seen) | **76.5%** [67.6, 84.3] |
| RoBERTa fine-tuned on App A only, asked about App B | **0.0%** |

The supervised classifier scores **exactly zero** because its head cannot emit a
label it never trained on. This is not "worse" — it is inoperable. Same
GLiNER weights, zero retraining, and it serves App B fine.

This reproduces the finding from arXiv:2608.20371 (*When Do LLMs Replace
Fine-Tuned NLU?*), which the evaluation protocol here is modelled on.

## Evaluation protocol

Following that paper, because point estimates on a few hundred examples are not
evidence:

- **Exact-set match.** Correct only if the predicted label set equals the
  reference. The same metric the model card reports.
- **Bootstrap 95% CIs**, 10,000 resamples, on every accuracy.
- **Paired bootstrap** on accuracy differences, plus **exact McNemar** for
  paired significance — McNemar because discordant-pair counts are small and the
  chi-square approximation is unreliable there.
- **Every system sees identical items in identical order**, or no paired test is
  computed at all.
- **A majority-class floor row.** On imbalanced fixtures this catches a model
  that has collapsed onto one class. On `hate_speech` GLiNER2.5-Decide scores
  44.4% accuracy against the floor's 77.6%, while its macro-F1 (0.336) *beats*
  the floor (0.291). Both numbers are reported, because either alone
  misrepresents the model in one direction or the other.
- **Independent data only.** The model's own `fastino/fast-decisions` corpus is
  its training distribution and is never mixed into these numbers.

## Setup

Python 3.12, CUDA, and a `.venv`. Nothing installs into system Python.

```powershell
uv sync                      # creates .venv on a uv-managed CPython 3.12
python scripts/fetch_model.py            # ~1.9 GB, resumable
python scripts/fetch_fixtures.py --with-train
python scripts/verify_env.py    # 8 environment assertions
python scripts/verify_model.py  # API behaviour vs the model card
uv run uvicorn app.main:app --port 8765
```

### Two constraints worth knowing

- **`transformers` must stay `<5`.** `gliner2` 2.0.0 hard-pins `transformers>=4.38,<5`.
  Installing current `transformers` 5.18.0 breaks the classifier. Pinned in
  `pyproject.toml`.
- **The PyPI `torch` wheel is CPU-only.** The default Windows wheel is a 124 MB
  CPU build. CUDA wheels live on `download.pytorch.org` per CUDA version, and
  `pyproject.toml` declares that index explicitly. For `torch` 2.14.1 only
  `cu126` and `cu130` publish `cp312-win_amd64`; `cu126` is used because
  compute capability 7.5 (Turing) is conservatively covered there. Verified via
  `sm_75 in torch.cuda.get_arch_list()`.

### Optional: the LLM comparison arm

Copy `.env.example` to `.env` and fill it in:

```
LLM_BASE_URL=https://your-endpoint/v1
LLM_API_KEY=...
LLM_MODEL=...
```

`app/llm.py` reads `.env` at startup — no dependency, six lines, and existing
environment variables win so a shell export overrides the file. Any
OpenAI-compatible `/chat/completions` endpoint works — the
client speaks plain HTTP rather than using the OpenAI SDK, so it has no SDK
version coupling. Following the paper's protocol, the model is shown the bare
label list and no per-label descriptions, so the comparison is not handicapped
against the encoder arm on banking77. If unset, the panel reports the LLM as
unavailable and names the missing variables rather than omitting it or inventing
a number.

**Do not paste keys into a prompt, issue, or commit.** A key shared in a
conversation should be treated as exposed and rotated.

**The LLM arm is slow — budget for it.** Measured at 6-7 s per call, so a
300-item row takes ~16 minutes at concurrency 4. The UI caps the item count when
the arm is selected, and the panel reports the per-item cost. Token prices
default to 0 and display as unpriced; set `LLM_INPUT_PER_MTOK` and
`LLM_OUTPUT_PER_MTOK` for your endpoint rather than trusting a figure that will go
stale.

## Fixtures

All from public datasets, trimmed to labelled samples in `data/fixtures/`:

| fixture | labels | notes |
|---|---|---|
| banking77 | 77 | ships a description per intent |
| clinc150 | 151 | 150 intents + out-of-scope (`oos`), null-labelled in the DeepPavlov mirror |
| hate_speech | 3 | hate / offensive / neither; heavily imbalanced |
| enron_spam | 2 | spam vs ham, real mailbox |
| yelp | 3 | sentiment; **label ids verified by reading samples**, since the dataset ships no card metadata |

Two data notes that changed results:

- **CLINC150 descriptions are all `null`**, so the described-labels variant is
  only available on banking77. Reported as unavailable elsewhere rather than
  silently skipped.
- **The yelp label mapping was wrong by convention.** The obvious
  `{0: negative, 1: positive}` guess mislabels the 3-star class. Read samples:
  id 1 is neutral ("This is 3 stars, but it's not A OK"), id 2 is positive
  ("this may be my new favorite place").

## Measured behaviour

- p50 latency is **52 ms** for a 77-label schema, 44 ms for 9 labels, on one
  RTX 2070 Super in fp16 — a 10.7x speedup over CPU (480 ms).
- fp16 and CPU produce **identical labels**, confidences agreeing to four
  decimals. Verified, not assumed.
- **Label descriptions are free on small schemas and expensive on large ones**:
  no penalty at 9 labels (43.6 vs 43.9 ms), but **8.6x slower** at 77 labels
  (595 vs 70 ms), because 77 descriptions inflate the input ~3.9x. The panel
  warns when you ask for this on a big schema.
- Batching is nearly pointless on CPU (1.2x) and better on GPU; batch 8–16 is
  the sweet spot, batch 32 is slower.
- The LLM arm costs **6-7 s per call** against a real endpoint, ~2.9 s/item at
  concurrency 4. That is ~140x the encoder's p50 and the reason the UI caps item
  count for that arm.
- LLM replies are normalised before scoring: case, quotes, separators and a
  short lead-in are treated as the same answer. Observed from the live endpoint:
  it answered `Refund_not_showing_up` where the schema said
  `refund_not_showing_up`. An invented label outside the schema is still scored
  wrong — loosening the parser must not also inflate the arm.

## Layout

```
app/
  main.py       FastAPI: /api/route, /api/panel, /api/schema-swap
  engine.py     model wrapper: device policy, fp16, parsing, disk cache
  fixtures.py   fixture loading and schema construction
  metrics.py    bootstrap CIs, paired bootstrap, exact McNemar, macro-F1, floor
  baselines.py  TF-IDF, MiniLM+LR, fine-tuned RoBERTa, learning curve
  evaluate.py   panel orchestration and the schema-swap experiment
  llm.py        OpenAI-compatible client
  static/       single-file UI, no build step
scripts/
  verify_env.py     8 environment assertions (CUDA, arch list, fp16, pins)
  verify_model.py   API behaviour against the model card's documented examples
  fetch_model.py    resumable weight download (uv and hf_hub restart from zero)
  fetch_fixtures.py dataset sampling and manifests
tests/
  test_core.py      48 offline tests: parsing, metrics, degenerate cases
```

Predictions cache to `results/` keyed by a schema fingerprint, so re-running a
sweep is instant. Change a label set and the fingerprint changes, so a stale
cache can never be silently reused.

## Known limits

- English only. `GLiNER2.5-multi-Decide` exists for other languages.
- The moderator-shaped fixtures (`hate_speech`) are the model's weakest showing
  and are included because of it.
- Confidence values are reported as returned by the library. They are not
  treated as calibrated probabilities, and no threshold is presented as a
  decision boundary.
- The fine-tuned RoBERTa baseline uses `roberta-base` to match the paper, not
  the largest available encoder. A stronger supervised baseline would narrow
  the banking77 gap further, so the reported gap is a **lower bound** on how far
  fine-tuning can pull ahead.
- The LLM figures come from **one endpoint and one model** (`qwen3.7-plus`), on
  60 items. That is a demonstration that the arm works end to end, not a
  benchmark of LLMs generally. The interval is wide and the paper's own CLINC150
  result put a comparable model at statistical parity with a supervised encoder
  on a 150-way schema. Re-run it against your own endpoint before drawing
  conclusions.