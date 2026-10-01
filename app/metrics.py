"""Evaluation protocol for the evidence panel.

Follows the methodology in arXiv:2608.20371 ("When Do LLMs Replace Fine-Tuned
NLU?"): percentile bootstrap 95% confidence intervals with 10,000 resamples,
paired bootstrap on the accuracy difference between two systems, and exact
McNemar for paired significance.

Two rules this module enforces deliberately:

1. Exact-set match, not per-label accuracy. A prediction counts as correct only
   if the predicted label set equals the reference. This is the metric the
   GLiNER2.5-Decide model card uses, so our numbers are comparable to it.
2. No pairing across systems unless both scored the same items in the same
   order. Unpaired comparisons are reported as such rather than silently given
   a p-value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


def normalise(labels: Any) -> frozenset[str]:
    """Coerce any label payload into a comparable set of strings."""
    if labels is None:
        return frozenset()
    if isinstance(labels, str):
        return frozenset({labels})
    if isinstance(labels, dict):
        # e.g. {"label": "x", "confidence": 0.9} or a list of such dicts
        if "label" in labels:
            return normalise(labels["label"])
        labels = labels.get("labels", [])
    if isinstance(labels, (list, tuple, set, frozenset)):
        out: set[str] = set()
        for item in labels:
            out |= normalise(item)
        return frozenset(out)
    return frozenset({str(labels)})


def exact_set_match(
    predictions: Sequence[Any], references: Sequence[Any]
) -> list[bool]:
    """Per-item correctness under exact-set match."""
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    return [normalise(p) == normalise(g) for p, g in zip(predictions, references)]


def accuracy(correct: Sequence[bool]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def bootstrap_ci(
    correct: Sequence[bool],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean of booleans (i.e. an accuracy)."""
    arr = np.asarray(correct, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(arr[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_bootstrap_diff(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap the accuracy difference (a - b) over paired items.

    Returns the point difference, its CI, and a two-sided p-value derived from
    how often the resampled difference falls on the other side of zero.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired comparison requires identical item counts and order")
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    n = a.size
    diff = float(a.mean() - b.mean()) if n else float("nan")
    if n == 0:
        return {"diff": diff, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided p: resampled differences at or beyond the observed magnitude.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "diff": diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": min(1.0, float(p)),
    }


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, Any]:
    """Exact McNemar test on the discordant pairs.

    b = items a got right and b got wrong; c = the reverse. Under H0 the
    discordant items split 50/50, so we use the exact binomial test rather than
    the chi-square approximation, which is unreliable at these counts.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("McNemar requires identical item counts and order")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"b": b, "c": c, "p_value": min(1.0, 2.0 * tail)}


def macro_f1(
    predictions: Sequence[Any], references: Sequence[Any], labels: Sequence[str] | None = None
) -> float:
    """Unweighted mean F1 across labels. Reported alongside accuracy because
    imbalanced test sets make accuracy alone misleading."""
    preds = [normalise(p) for p in predictions]
    refs = [normalise(g) for g in references]
    if labels is None:
        seen: set[str] = set()
        for s in preds + refs:
            seen |= s
        labels = sorted(seen)
    scores = []
    for label in labels:
        tp = sum(1 for p, r in zip(preds, refs) if label in p and label in r)
        fp = sum(1 for p, r in zip(preds, refs) if label in p and label not in r)
        fn = sum(1 for p, r in zip(preds, refs) if label not in p and label in r)
        if tp == 0 and (fp or fn):
            scores.append(0.0)
        elif tp == 0:
            continue  # label absent from both; exclude rather than score 1.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def majority_baseline(
    references: Sequence[Any],
) -> dict[str, Any]:
    """Accuracy and macro-F1 of always predicting the most common label.

    Without this row a classifier that collapses onto the majority class looks
    respectable. On imbalanced fixtures it can beat the model outright while
    having no real signal: GLiNER2.5-Decide scores 44.4% on hate_speech versus
    77.6% for this baseline, yet its macro-F1 is higher because it does spread
    its predictions. Both numbers are needed to read the panel honestly.
    """
    refs = [normalise(g) for g in references]
    if not refs:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "label": None}
    flat = [next(iter(r)) for r in refs if r]
    if not flat:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "label": None}
    counts: dict[str, int] = {}
    for label in flat:
        counts[label] = counts.get(label, 0) + 1
    majority = max(counts, key=counts.get)
    preds = [{majority} for _ in refs]
    correct = [p == r for p, r in zip(preds, refs)]
    return {
        "accuracy": accuracy(correct),
        "macro_f1": macro_f1(preds, references),
        "label": majority,
        "count": counts[majority],
    }


@dataclass
class SystemResult:
    """One row of the evidence panel."""

    name: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    macro_f1: float
    p50_ms: float | None = None
    p95_ms: float | None = None
    cost_per_1k_usd: float | None = None
    # Why cost_per_1k_usd is what it is. A null price is ambiguous on its own:
    # for a local encoder it means the arm genuinely costs nothing at inference,
    # while for the LLM arm it means nobody supplied a token price. The client
    # used to render both as "$0.00", which reads as free and understates the
    # LLM arm's cost by an unknown amount. The state is decided here so the UI
    # reports it rather than infers it.
    #   "free"     - runs locally, no marginal inference cost
    #   "priced"   - cost_per_1k_usd is a measured figure from a supplied price
    #   "unpriced" - the arm was measured but no token price is configured
    cost_status: str = "free"
    notes: list[str] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list, repr=False)

    def to_public(self) -> dict[str, Any]:
        """Drop the per-item correctness vector before sending to the client."""
        return {
            "name": self.name,
            "n": self.n,
            "accuracy": self.accuracy,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "macro_f1": self.macro_f1,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "cost_per_1k_usd": self.cost_per_1k_usd,
            "cost_status": self.cost_status,
            "notes": self.notes,
        }


def evaluate_system(
    name: str,
    predictions: Sequence[Any],
    references: Sequence[Any],
    *,
    p50_ms: float | None = None,
    p95_ms: float | None = None,
    cost_per_1k_usd: float | None = None,
    cost_status: str = "free",
    notes: Sequence[str] = (),
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> SystemResult:
    correct = exact_set_match(predictions, references)
    lo, hi = bootstrap_ci(correct, n_resamples=n_resamples, seed=seed)
    return SystemResult(
        name=name,
        n=len(correct),
        accuracy=accuracy(correct),
        ci_low=lo,
        ci_high=hi,
        macro_f1=macro_f1(predictions, references),
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        cost_per_1k_usd=cost_per_1k_usd,
        cost_status=cost_status,
        notes=list(notes),
        correct=correct,
    )


def compare(
    a: SystemResult,
    b: SystemResult,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Head-to-head between two systems scored on the same items."""
    if a.n != b.n:
        raise ValueError(
            f"{a.name} scored {a.n} items but {b.name} scored {b.n}; "
            "a paired comparison is not valid"
        )
    diff = paired_bootstrap_diff(a.correct, b.correct, n_resamples=n_resamples, seed=seed)
    diff["mcnemar"] = mcnemar_exact(a.correct, b.correct)
    diff["a"] = a.name
    diff["b"] = b.name
    diff["significant"] = diff["p_value"] < 0.05
    return diff