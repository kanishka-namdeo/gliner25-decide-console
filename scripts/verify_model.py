"""Model gate: load GLiNER2.5-Decide on the verified environment and confirm it
behaves as the model card documents.

Split from verify_env.py on purpose — this downloads ~1.4 GB, so we only reach
it once the cheap environment checks have passed.

Hard failures  : shape / finiteness / device problems. The app cannot work.
Soft observations: agreement with the model card's documented examples. The card
explicitly labels those outputs as "potential results", so a disagreement is
information, not a bug.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ID = "fastino/GLiNER2.5-Decide"
# Require the local copy rather than falling back to the Hub id: from_pretrained
# would restart the ~1.9 GB download from zero on any interruption, which on a
# slow link never completes. scripts/fetch_model.py downloads with range resume.
LOCAL = Path("models") / REPO_ID.split("/")[-1]

if not (LOCAL / "config.json").exists():
    print("=" * 72)
    print("MODEL WEIGHTS NOT FOUND")
    print("=" * 72)
    print(f"Expected: {LOCAL / 'config.json'}")
    print("\nFetch them first (resumable, ~1.9 GB):")
    print(f"    python scripts/fetch_model.py {REPO_ID}")
    print("\nThen re-run this script.")
    print("=" * 72)
    raise SystemExit(2)

MODEL_ID = str(LOCAL.as_posix())

# Verbatim from the model card.
SINGLE_EXAMPLE = (
    "My subscription renewed on April 15 for ¥5,400 after the service was already "
    "down. Can I get that charge refunded?",
    {"intent": ["order_status", "refund_request", "cancel_subscription",
                "update_payment", "login_problem", "shipping_delay",
                "bug_report", "speak_to_human", "other"]},
    "refund_request",  # documented label
)

MULTI_EXAMPLE = (
    "Guest in room 1408 says the AC has been out since yesterday and they want to move "
    "tonight or leave. They also asked for the incidentals hold to be released.",
    {
        "intent": ["maintenance", "room_change", "checkout", "billing", "complaint",
                   "amenity_request"],
        "priority": ["low", "normal", "high", "urgent"],
        "needs_human": ["yes", "no"],
        "topics": {
            "labels": ["hvac", "billing", "housekeeping", "noise", "safety"],
            "multi_label": True,
            "cls_threshold": 0.4,
        },
    },
    4,  # documented: four heads scored in one call
)

MULTI_LABEL_EXAMPLE = (
    "Battery dies before lunch, but the keyboard and the screen are the best I have "
    "used on a laptop.",
    {
        "aspects": {
            "labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
            "multi_label": True,
            "cls_threshold": 0.4,
        }
    },
)

failures: list[str] = []
observations: list[str] = []


def main() -> int:
    # gliner2 prints an emoji in its config banner; the default Windows console
    # codec is cp1252 and raises UnicodeEncodeError on it. Reconfigure before
    # importing gliner2 so the banner cannot kill the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    import torch
    from gliner2 import AutoExtractor

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"device: {device}")
    if not use_cuda:
        observations.append("running on CPU fallback — slower, and fp16 is disabled")

    print(f"\nloading {MODEL_ID} ...")
    t0 = time.perf_counter()
    kwargs = {"map_location": device}
    if use_cuda:
        # fp16 is the only mixed precision Turing supports (bf16 needs sm_80+)
        kwargs["quantize"] = True
    try:
        model = AutoExtractor.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        observations.append("from_pretrained rejected `quantize`; retrying without it")
        kwargs.pop("quantize", None)
        model = AutoExtractor.from_pretrained(MODEL_ID, **kwargs)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    # --- single-label head ----------------------------------------------
    text, schema, documented = SINGLE_EXAMPLE
    out = model.classify_text(text, schema)
    print(f"\nsingle-label result: {out}")
    label = out.get("intent") if isinstance(out, dict) else None
    if not isinstance(label, str):
        failures.append("classify_text did not return a single string label")
    elif label not in schema["intent"]:
        failures.append(f"predicted {label!r} is not in the supplied label set")
    elif label == documented:
        observations.append(f"matches documented example ({label})")
    else:
        observations.append(
            f"differs from documented example: got {label!r}, card shows {documented!r}"
        )

    # --- confidence finiteness (fp16 / DeBERTa NaN risk) ----------------
    conf = model.classify_text(text, schema, include_confidence=True)
    print(f"with confidence:     {conf}")
    value = _extract_confidence(conf)
    if value is None:
        failures.append("include_confidence=True returned no confidence value")
    elif not (0.0 <= value <= 1.0):
        failures.append(f"confidence out of range or NaN: {value!r}")

    # --- multiple heads in one forward pass -----------------------------
    text, schema, n_heads = MULTI_EXAMPLE
    out = model.classify_text(text, schema, include_confidence=True)
    print(f"\nmulti-head result:   {out}")
    if not isinstance(out, dict):
        failures.append("multi-head call did not return a dict")
    else:
        missing = [k for k in ("intent", "priority", "needs_human", "topics") if k not in out]
        if missing:
            failures.append(f"multi-head call dropped heads: {missing}")
        elif len(out) != n_heads:
            failures.append(f"expected {n_heads} heads, got {len(out)}")
        else:
            observations.append(f"all {n_heads} heads returned from a single call")
        topics = out.get("topics")
        if isinstance(topics, list):
            if not topics:
                failures.append("multi_label head returned an empty list")
            else:
                observations.append(f"multi-label head returned {topics}")

    # --- multi-label + threshold ----------------------------------------
    # classify_text always returns a dict keyed by head name; without
    # include_confidence a multi_label head yields plain strings, with it a
    # list of {"label", "confidence"} dicts.
    text, schema = MULTI_LABEL_EXAMPLE
    result = model.classify_text(text, schema)
    print(f"\naspects result:      {result}")
    aspects = result.get("aspects") if isinstance(result, dict) else None
    if not isinstance(aspects, list) or not aspects:
        failures.append(f"multi_label head did not return a non-empty list: {result!r}")
    elif any(not isinstance(a, str) or a not in schema["aspects"]["labels"] for a in aspects):
        failures.append(f"aspect label outside schema: {aspects}")
    else:
        observations.append(f"multi_label head returned {len(aspects)} bare labels: {aspects}")

    # --- latency, for the evidence panel --------------------------------
    text, schema, _ = SINGLE_EXAMPLE
    if use_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.classify_text(text, schema)
        if use_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(samples)
    observations.append(f"p50 latency (warm, batch 1): {p50:.1f} ms on {device}")

    # --- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    for obs in observations:
        print(f"  note: {obs}")
    if failures:
        print(f"\nMODEL GATE FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("\nMODEL GATE PASSED")
    print("=" * 72)
    return 0


def _extract_confidence(payload) -> float | None:
    """Pull the first numeric confidence out of whatever shape came back."""
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "confidence" in node:
                try:
                    return float(node["confidence"])
                except (TypeError, ValueError):
                    return float("nan")
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


if __name__ == "__main__":
    raise SystemExit(main())