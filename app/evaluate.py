"""Evidence panel orchestration.

Runs every available system over the same fixture items, scores them with the
paper's protocol, and pairs them up for significance testing.

Two design rules that this file exists to enforce:

1. Every system sees identical items in identical order. Otherwise McNemar and
   the paired bootstrap are meaningless, so `limit` slices the fixture once and
   every system consumes that same slice.
2. A system that could not run is reported as unavailable with a reason. It is
   never silently dropped, and it never contributes a fabricated number.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .fixtures import Fixture, load, split_labels
from .metrics import SystemResult, compare, evaluate_system, majority_baseline

log = logging.getLogger("evaluate")

RESULTS_DIR = Path("results")
ARTIFACT_DIR = RESULTS_DIR / "artifacts"

SYSTEM_LABELS = {
    "gliner": "GLiNER2.5-Decide (340M, zero-shot)",
    "gliner_described": "GLiNER2.5-Decide + label descriptions",
    "tfidf": "TF-IDF + logistic regression",
    "minilm": "MiniLM embeddings + logistic regression",
    "roberta": "RoBERTa-base fine-tuned (supervised)",
    "llm": "LLM (zero-shot)",
}


def _slice(fixture: Fixture, limit: int | None) -> tuple[list[str], list[str]]:
    n = len(fixture.texts) if not limit else min(limit, len(fixture.texts))
    return fixture.texts[:n], fixture.labels[:n]


def _cache_path(fixture: str, kind: str, limit: int | None, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return RESULTS_DIR / f"{fixture}_{kind}{suffix}_n{limit or 'all'}.json"


# --- individual systems --------------------------------------------------

def run_gliner(
    fixture: Fixture, texts: list[str], described: bool = False, batch_size: int = 16
) -> tuple[list[list[str]], float, float, bool]:
    from .engine import get_engine

    engine = get_engine()
    schema = fixture.schema("intent", described=described)
    tag = "described" if described else "plain"
    result = engine.cached_run(
        f"{fixture.name}_gliner_{tag}", texts, schema, batch_size=batch_size
    )
    preds = [list(p["intent"].labels) if "intent" in p else [] for p in result.predictions]
    return preds, result.latency_ms_p50, result.latency_ms_p95, result.cached


def run_baseline(
    kind: str, fixture_name: str, texts: list[str], limit: int | None = None
) -> tuple[list[str], float, int, float]:
    """Fit (or load) a supervised baseline and predict.

    Model artifacts are cached on disk because fine-tuning is the expensive part
    and re-running the UI should not repeat it.
    """
    from . import baselines

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"{fixture_name}_{kind}.joblib"

    if kind in ("tfidf", "minilm"):
        import joblib

        if artifact.exists():
            model = joblib.load(artifact)
        else:
            model = baselines.build_baseline(kind, fixture_name)
            joblib.dump(model, artifact)
    elif kind == "roberta":
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_dir = ARTIFACT_DIR / f"{fixture_name}_roberta"
        if (model_dir / "clf.pt").exists():
            # Reuse a previously fine-tuned head rather than retraining.
            meta = json.loads((model_dir / "clf_meta.json").read_text())
            model = baselines.FineTunedEncoder()
            model.label2id = meta["label2id"]
            model.id2label = meta["id2label"]
            model.trained_on = meta.get("trained_on", 0)
            model.tok = AutoTokenizer.from_pretrained(str(model_dir))
            model.enc = AutoModel.from_pretrained(str(model_dir))
            model.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.enc.to(model.device)
            model.enc.eval()
            model.clf = torch.nn.Linear(model.enc.config.hidden_size, len(model.label2id))
            model.clf.load_state_dict(
                torch.load(model_dir / "clf.pt", map_location=model.device, weights_only=True)
            )
            model.clf.to(model.device)
            model.clf.eval()
        else:
            model = baselines.build_baseline("roberta", fixture_name)
            model.enc.save_pretrained(str(model_dir))
            model.tok.save_pretrained(str(model_dir))
            torch.save(model.clf.state_dict(), model_dir / "clf.pt")
            (model_dir / "clf_meta.json").write_text(
                json.dumps({
                    "label2id": model.label2id,
                    "id2label": model.id2label,
                    "trained_on": getattr(model, "trained_on", 0),
                })
            )
    else:
        raise ValueError(f"unknown baseline {kind!r}")

    out = model.predict(texts)
    return out.predictions, out.latency_ms_p50, out.trained_on, out.latency_ms_p95 or 0.0


async def run_llm(
    texts: list[str],
    labels: list[str],
    timeout_s: float | None = None,
    concurrency: int = 4,
    progress=None,
) -> tuple[list[str | None], float, float, list[str], int]:
    """Classify via the configured endpoint.

    Concurrency is bounded and modest. A serial loop is correct but useless at
    real API latency (measured ~6-7 s per call against DashScope), while
    unbounded parallelism would hammer the endpoint and invite rate limiting.
    Order of results is preserved regardless of completion order, which matters
    because the paired significance tests assume identical item ordering.
    """
    from .llm import LLMClassifier

    client = LLMClassifier()
    if not client.config.configured:
        raise RuntimeError("LLM arm not configured")

    semaphore = asyncio.Semaphore(max(1, concurrency))
    preds: list[str | None] = [None] * len(texts)
    latencies: list[float] = []
    total_cost = 0.0
    errors: list[str] = []
    done = 0
    lock = asyncio.Lock()

    async def one(index: int, text: str) -> None:
        nonlocal done, total_cost
        async with semaphore:
            try:
                resp = await asyncio.wait_for(client.classify(text, labels), timeout=timeout_s)
            except asyncio.TimeoutError:
                async with lock:
                    errors.append("timeout")
                    done += 1
                return
            except Exception as exc:  # noqa: BLE001
                async with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    done += 1
                return
            async with lock:
                preds[index] = resp.label
                if resp.latency_ms:
                    latencies.append(resp.latency_ms)
                total_cost += resp.cost_usd
                if resp.error:
                    errors.append(resp.error)
                done += 1
                if progress:
                    progress(done, len(texts))

    await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
    await client.aclose()

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0
    return preds, p50, p95, total_cost, errors


# --- panel ---------------------------------------------------------------

async def run_panel(
    fixture_name: str,
    limit: int | None = 500,
    systems: list[str] | None = None,
    described: bool = False,
    batch_size: int = 16,
    llm_concurrency: int = 4,
) -> dict[str, Any]:
    """Score every requested system on one fixture and pair them up.

    llm_concurrency matters more than it looks: measured against DashScope at
    ~6-7 s per call, a serial loop would take ~16 minutes for a 300-item row.
    Bounded concurrency cuts that roughly in half while staying polite.
    """
    from .engine import get_engine
    from .llm import LLMConfig

    fixture = load(fixture_name)
    texts, gold = _slice(fixture, limit)
    requested = systems or ["gliner"]
    notes: list[str] = []
    unavailable: list[dict] = []

    results: list[SystemResult] = []

    async def timed_gliner(desc: bool) -> SystemResult:
        key = "gliner_described" if desc else "gliner"
        preds, p50, p95, cached = await asyncio.to_thread(
            run_gliner, fixture, texts, desc, batch_size
        )
        schema = fixture.schema("intent", described=desc)
        n_labelled = len(schema["intent"])
        sys_notes = [
            f"schema: {n_labelled} labels",
            "zero-shot, 0 training examples",
            "cached" if cached else "",
        ]
        if desc and n_labelled > 25:
            # Measured on this GPU: descriptions inflate the input ~3.9x at 77
            # labels and cost ~8.6x latency, because the encoder reads every
            # description. Cheap on small schemas, expensive on large ones.
            sys_notes.append("note: descriptions inflate input ~3.9x here (measured ~8.6x slower)")
        return evaluate_system(
            SYSTEM_LABELS[key], preds, gold, p50_ms=p50, p95_ms=p95, notes=sys_notes
        )

    if "gliner" in requested:
        results.append(await timed_gliner(described))
    if "gliner_described" in requested:
        if fixture.described_coverage() > 0:
            results.append(await timed_gliner(True))
        else:
            unavailable.append({
                "system": SYSTEM_LABELS["gliner_described"],
                "reason": f"{fixture.name} ships no label descriptions "
                          f"(coverage {fixture.described_coverage():.0%})",
            })

    for kind in ("tfidf", "minilm", "roberta"):
        if kind not in requested:
            continue
        try:
            preds, p50, trained_on, p95 = await asyncio.to_thread(
                run_baseline, kind, fixture_name, texts, limit
            )
            results.append(
                evaluate_system(
                    SYSTEM_LABELS[kind],
                    preds,
                    gold,
                    p50_ms=p50,
                    p95_ms=p95,
                    notes=[
                        f"trained on {trained_on:,} labelled examples",
                        f"{fixture.n_labels}-way fixed taxonomy",
                    ],
                )
            )
        except FileNotFoundError as exc:
            unavailable.append({"system": SYSTEM_LABELS[kind], "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001
            unavailable.append({"system": SYSTEM_LABELS[kind], "reason": f"{type(exc).__name__}: {exc}"})

    if "llm" in requested:
        cfg = LLMConfig.from_env()
        if not cfg.configured:
            unavailable.append({
                "system": SYSTEM_LABELS["llm"],
                "reason": cfg.status()["reason"],
            })
        else:
            try:
                def tick(done: int, total: int) -> None:
                    # Deliberately sync: run_llm awaits progress() inside a lock,
                    # so an async callback here would be created but never awaited.
                    log.info("llm progress %d/%d", done, total)

                started = time.perf_counter()
                preds, p50, p95, cost, errors = await run_llm(
                    texts, fixture.label_names, concurrency=llm_concurrency, progress=tick
                )
                llm_seconds = time.perf_counter() - started
                if len(texts) > 20:
                    notes.append(
                        f"llm arm took {llm_seconds:.0f}s for {len(texts)} items "
                        f"(~{llm_seconds / len(texts):.1f}s/item at concurrency "
                        f"{llm_concurrency}); lower the item count for a faster row"
                    )
                invalid = sum(1 for p in preds if p is None)
                cfg = LLMConfig.from_env()
                priced = bool(cfg.input_per_mtok or cfg.output_per_mtok)
                res = evaluate_system(
                    f"LLM ({cfg.model}, zero-shot)",
                    [p if p is not None else "__none__" for p in preds],
                    gold,
                    p50_ms=p50,
                    p95_ms=p95,
                    cost_per_1k_usd=(cost / len(texts) * 1000) if (cost and priced) else None,
                    # Distinguish "we measured a cost" from "nobody set a token
                    # price". The UI must not show $0.00 for the second case.
                    cost_status="priced" if priced else "unpriced",
                    notes=[
                        "zero-shot, label list only (no descriptions)",
                        f"measured over {len(preds) - invalid}/{len(preds)} replies",
                        *([f"{invalid} replies unusable, counted as wrong"] if invalid else []),
                        *(
                            [] if priced
                            else ["cost not priced: set LLM_INPUT_PER_MTOK / LLM_OUTPUT_PER_MTOK"]
                        ),
                        *([f"first error: {errors[0]}"] if errors else []),
                    ],
                )
                results.append(res)
            except Exception as exc:  # noqa: BLE001
                unavailable.append({"system": SYSTEM_LABELS["llm"], "reason": f"{type(exc).__name__}: {exc}"})

    # Floor reference: always predict the most common label. On imbalanced
    # fixtures this can beat a real model on accuracy, and hiding it would make
    # the panel misleading rather than merely incomplete.
    floor = majority_baseline(gold)
    if floor["label"] is not None:
        results.append(
            SystemResult(
                name=f"majority class ({floor['label']})",
                n=len(gold),
                accuracy=floor["accuracy"],
                ci_low=float("nan"),
                ci_high=float("nan"),
                macro_f1=floor["macro_f1"],
                notes=[
                    f"floor, not a system: predicts '{floor['label']}' for everything "
                    f"({floor['count']}/{len(gold)} items)",
                    "a model must beat this to be doing anything",
                ],
            )
        )

    comparisons = []
    gliner_like = [r for r in results if "GLiNER" in r.name]
    others = [r for r in results if "GLiNER" not in r.name and r.correct]
    for base in others:
        for ours in gliner_like:
            if base.n == ours.n:
                comparisons.append(compare(ours, base))

    return {
        "fixture": {
            "name": fixture.name,
            "n_items": len(texts),
            "n_labels": fixture.n_labels,
            "label_coverage": fixture.described_coverage(),
            "source": fixture.manifest.get("source"),
            "note": fixture.manifest.get("note"),
        },
        "systems": [
            {**r.to_public(), "notes": [n for n in r.notes if n]} for r in results
        ],
        "comparisons": comparisons,
        "unavailable": unavailable,
        "notes": notes,
        "runtime": get_engine().info(),
        "metric": "exact-set match (a prediction is correct only if the label set equals the reference)",
    }


def model_classes(kind: str, fixture: Fixture) -> list[str]:
    """Labels a baseline can emit. A fine-tuned head is locked to its training
    taxonomy, which is the entire point of the schema-swap test."""
    return fixture.label_names


async def run_schema_swap(fixture_name: str, limit: int = 300) -> dict[str, Any]:
    """The decisive experiment from arXiv:2608.20371.

    Split the intents into two disjoint halves, A and B. A classifier trained on
    A alone scores 0% on B, because its head physically cannot emit a label it
    never saw. The schema-driven model is handed B's labels and just answers.
    """
    from . import baselines

    fixture = load(fixture_name)
    labels_a, labels_b = split_labels(fixture)
    texts, gold_all = _slice(fixture, limit)

    # GLiNER2.5-Decide, one call per app schema
    out: dict[str, Any] = {
        "fixture": fixture_name,
        "app_a_labels": len(labels_a),
        "app_b_labels": len(labels_b),
        "systems": {},
        "explanation": (
            "A fine-tuned classifier is locked to the taxonomy it trained on. "
            "GLiNER2.5-Decide receives the label set as input at call time, so "
            "the same weights serve a schema they have never seen."
        ),
    }

    for tag, subset in (("app_a", labels_a), ("app_b", labels_b)):
        schema = fixture.schema("intent", label_subset=subset)
        # Only score items whose gold label is in this app's schema; an item from
        # the other app is out of scope for this schema by construction.
        idx = [i for i, g in enumerate(gold_all) if g in subset]
        if not idx:
            out["systems"][tag] = {"error": "no items in fixture for this schema"}
            continue
        sub_texts = [texts[i] for i in idx]
        sub_gold = [gold_all[i] for i in idx]
        preds, p50, p95, _ = await asyncio.to_thread(
            run_gliner, fixture, sub_texts, False, 16
        )

        res = evaluate_system(f"GLiNER2.5-Decide on {tag}", preds, sub_gold, p50_ms=p50)
        out["systems"][tag] = {**res.to_public(), "n_items_scored": len(idx)}

    # Supervised classifier trained on A only, then asked about B
    try:
        from .baselines import FineTunedEncoder

        train_texts, train_labels = _load_train_rows(fixture_name)
        keep = [i for i, y in enumerate(train_labels) if y in set(labels_a)]
        model = FineTunedEncoder(epochs=2).fit(
            [train_texts[i] for i in keep], [train_labels[i] for i in keep]
        )
        idx_b = [i for i, g in enumerate(gold_all) if g in set(labels_b)]
        b_texts = [texts[i] for i in idx_b]
        b_gold = [gold_all[i] for i in idx_b]
        scored = model.predict(b_texts)
        correct = [p == g for p, g in zip(scored.predictions, b_gold)]
        out["systems"]["roberta_trained_on_a_scored_on_b"] = {
            "name": "RoBERTa fine-tuned on App A only, asked about App B",
            "n": len(b_texts),
            "accuracy": sum(correct) / len(correct) if correct else float("nan"),
            "note": "its head cannot emit any App B label, so accuracy is 0 by construction",
        }
        out["supervised_trained_on"] = len(keep)
    except Exception as exc:  # noqa: BLE001
        out["systems"]["roberta_trained_on_a_scored_on_b"] = {
            "error": f"{type(exc).__name__}: {exc}"
        }

    return out


def _load_train_rows(fixture_name: str) -> tuple[list[str], list[str]]:
    from .baselines import _load_train

    return _load_train(fixture_name)