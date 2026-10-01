"""Decision engine: one model, loaded once, wrapped for repeated calls.

Device policy, in order of preference:
  1. CUDA in fp16. Required for throughput on the evidence sweeps. The RTX 2070
     Super is compute capability 7.5 (Turing), so bf16 is unavailable (needs
     sm_80+) and eager attention is used because DeBERTa-v2 rejects SDPA in
     transformers 4.x.
  2. CPU fallback. Works, roughly 20x slower, and the UI says so rather than
     pretending otherwise.

Results are cached to disk keyed by (fixture, model, schema fingerprint), so an
evaluation sweep is paid for once and re-running the UI is instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# gliner2 prints an emoji in its config banner, which raises UnicodeEncodeError
# on a cp1252 Windows console. Must happen before gliner2 is imported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

log = logging.getLogger("engine")

MODEL_DIR = Path("models/GLiNER2.5-Decide")
REPO_ID = "fastino/GLiNER2.5-Decide"
CACHE_DIR = Path("results")


@dataclass
class Prediction:
    head: str
    labels: list[str]
    confidence: float | None
    raw: Any


@dataclass
class BatchResult:
    predictions: list[dict[str, Prediction]]
    latency_ms_p50: float
    latency_ms_p95: float
    n: int
    cached: bool = False


def schema_fingerprint(schema: dict) -> str:
    """Stable hash of a schema, so cache keys change when labels change."""
    blob = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class DecisionEngine:
    """Thin, process-wide wrapper around AutoExtractor."""

    def __init__(self, model_path: Path | str | None = None, prefer_cuda: bool = True) -> None:
        self.model_path = Path(model_path or MODEL_DIR)
        self.prefer_cuda = prefer_cuda
        self._model = None
        self.device: str | None = None
        self.fp16: bool = False
        self.load_seconds: float | None = None

    # -- lifecycle --------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"no model at {self.model_path}. "
                f"Run: python scripts/fetch_model.py {REPO_ID}"
            )
        from gliner2 import AutoExtractor
        import torch

        use_cuda = self.prefer_cuda and torch.cuda.is_available()
        kwargs: dict[str, Any] = {"map_location": "cuda" if use_cuda else "cpu"}
        if use_cuda:
            # Turing has no bf16; fp16 is the only mixed-precision option.
            kwargs["quantize"] = True

        t0 = time.perf_counter()
        try:
            self._model = AutoExtractor.from_pretrained(str(self.model_path), **kwargs)
        except TypeError:
            # Older/newer signature without `quantize`; fall back rather than die.
            kwargs.pop("quantize", None)
            self._model = AutoExtractor.from_pretrained(str(self.model_path), **kwargs)
        self.load_seconds = time.perf_counter() - t0

        self.device = "cuda" if use_cuda else "cpu"
        self.fp16 = use_cuda
        log.info(
            "loaded %s on %s in %.1fs (fp16=%s)",
            self.model_path.name, self.device, self.load_seconds, self.fp16,
        )

    def info(self) -> dict:
        """Runtime facts the UI shows. Never claims a capability we did not get."""
        import torch

        cuda = torch.cuda.is_available()
        return {
            "model": self.model_path.name,
            "device": "cuda" if cuda and self.prefer_cuda else "cpu",
            "fp16": bool(cuda and self.prefer_cuda),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "compute_capability": (
                f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
                if cuda else None
            ),
            "attention": "eager (DeBERTa-v2 rejects sdpa in transformers 4.x)",
            "bf16_supported": bool(cuda and torch.cuda.get_device_capability(0)[0] >= 8),
            "loaded": self._model is not None,
            "load_seconds": self.load_seconds,
        }

    # -- inference --------------------------------------------------------

    def classify_one(
        self, text: str, schema: dict, include_confidence: bool = True
    ) -> dict[str, Prediction]:
        raw = self.model.classify_text(
            text, schema, include_confidence=include_confidence
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: Any) -> dict[str, Prediction]:
        """Normalise the several shapes classify_text can return.

        single label -> {"intent": "refund_request"} or
                        {"intent": {"label": ..., "confidence": ...}}
        multi label  -> {"aspects": ["a","b"]} or
                        {"aspects": [{"label":..., "confidence":...}, ...]}
        """
        out: dict[str, Prediction] = {}
        if not isinstance(raw, dict):
            return out
        for head, value in raw.items():
            if isinstance(value, dict):
                out[head] = Prediction(
                    head=head,
                    labels=[str(value.get("label"))],
                    confidence=_as_float(value.get("confidence")),
                    raw=value,
                )
            elif isinstance(value, list):
                labels, confs = [], []
                for item in value:
                    if isinstance(item, dict):
                        labels.append(str(item.get("label")))
                        confs.append(_as_float(item.get("confidence")))
                    else:
                        labels.append(str(item))
                keep = [c for c in confs if c is not None]
                out[head] = Prediction(
                    head=head,
                    labels=labels,
                    confidence=min(keep) if keep else None,
                    raw=value,
                )
            elif value is None:
                out[head] = Prediction(head=head, labels=[], confidence=None, raw=value)
            else:
                out[head] = Prediction(
                    head=head, labels=[str(value)], confidence=None, raw=value
                )
        return out

    def classify_many(
        self,
        texts: list[str],
        schema: dict,
        batch_size: int = 8,
        include_confidence: bool = True,
        progress=None,
    ) -> BatchResult:
        """Classify a list of texts, timing each call.

        batch_classify_text is used even though CPU batching is close to
        pointless (measured ~1.2x); on GPU the same call is markedly better and
        the code path is identical.
        """
        if not texts:
            return BatchResult([], 0.0, 0.0, 0)

        import torch

        def sync() -> None:
            if self.device == "cuda":
                torch.cuda.synchronize()

        raw_results: list[Any] = []
        latencies: list[float] = []

        # Warm up before timing. The first call pays lazy init and allocator
        # growth; charging that to the first chunk inflated measured p50 by
        # ~3.6x on CPU, which would have made the latency column a fiction.
        try:
            self.model.batch_classify_text(
                texts[:1], schema, batch_size=1, include_confidence=include_confidence
            )
            sync()
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            log.warning("warmup call failed, timings may include init cost: %s", exc)

        # Timed in chunks so progress can be reported without per-call overhead.
        chunk = max(batch_size, 1)
        for start in range(0, len(texts), chunk):
            batch = texts[start : start + chunk]
            sync()
            t0 = time.perf_counter()
            out = self.model.batch_classify_text(
                batch,
                schema,
                batch_size=len(batch),
                include_confidence=include_confidence,
            )
            sync()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.extend([elapsed / len(batch)] * len(batch))
            raw_results.extend(out)
            if progress:
                progress(min(start + len(batch), len(texts)), len(texts))

        preds = [self._parse(r) for r in raw_results]
        ordered = sorted(latencies)
        return BatchResult(
            predictions=preds,
            latency_ms_p50=statistics.median(ordered),
            latency_ms_p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            n=len(preds),
        )

    # -- disk cache -------------------------------------------------------

    def cached_run(
        self,
        cache_key: str,
        texts: list[str],
        schema: dict,
        batch_size: int = 8,
        progress=None,
    ) -> BatchResult:
        """classify_many, memoised to results/<cache_key>.json."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{cache_key}.json"
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                if blob.get("fingerprint") == schema_fingerprint(schema) and blob.get("n") == len(texts):
                    preds = [
                        {
                            h: Prediction(h, p["labels"], p.get("confidence"), None)
                            for h, p in item.items()
                        }
                        for item in blob["predictions"]
                    ]
                    return BatchResult(
                        predictions=preds,
                        latency_ms_p50=blob["p50"],
                        latency_ms_p95=blob["p95"],
                        n=blob["n"],
                        cached=True,
                    )
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("ignoring corrupt cache %s: %s", path, exc)

        result = self.classify_many(texts, schema, batch_size=batch_size, progress=progress)
        path.write_text(
            json.dumps(
                {
                    "fingerprint": schema_fingerprint(schema),
                    "n": result.n,
                    "p50": result.latency_ms_p50,
                    "p95": result.latency_ms_p95,
                    "predictions": [
                        {h: {"labels": p.labels, "confidence": p.confidence} for h, p in item.items()}
                        for item in result.predictions
                    ],
                }
            )
        )
        return result


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


_engine: DecisionEngine | None = None


def get_engine() -> DecisionEngine:
    """Process-wide singleton. Loading is ~7s and 1.8 GB; do it once."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine