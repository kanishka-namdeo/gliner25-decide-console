"""Comparison systems for the evidence panel.

Three arms, in increasing order of how much labelled data they need:

  TfidfLogReg      no pretrained weights at all. The on-device floor.
  MiniLM + LogReg  frozen sentence embeddings, one logistic regression.
                   The paper's ST+kNN analogue; cheap and surprisingly strong.
  FineTunedEncoder roberta-base fine-tuned on the real train split. This is the
                   supervised ceiling and the number that matters, because it is
                   what an engineering team would otherwise reach for.

Plus a learning curve over training-set size, since "how many labelled examples
do I need before I should just fine-tune?" is the actual decision an adopter
faces. GLiNER2.5-Decide sits on that curve at zero labels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("baselines")

TRAIN_DIR = Path("data/fixtures")


@dataclass
class BaselineResult:
    predictions: list[str]
    latency_ms_p50: float
    trained_on: int = 0
    latency_ms_p95: float | None = None
    labels_seen: list[str] = field(default_factory=list)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def _load_train(name: str) -> tuple[list[str], list[str]]:
    """Load the train split written alongside a fixture."""
    path = TRAIN_DIR / f"{name}_train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run: python scripts/fetch_fixtures.py {name} --with-train"
        )
    import pandas as pd

    frame = pd.read_csv(path).fillna({"text": "", "label": ""})
    return [str(t) for t in frame["text"]], [str(x) for x in frame["label"]]


def _time_calls(fn, items: list[str]) -> tuple[list[str], float]:
    times = []
    out = []
    for item in items:
        t0 = time.perf_counter()
        out.append(fn(item))
        times.append((time.perf_counter() - t0) * 1000)
    return out, float(np.median(times)) if times else 0.0


class TfidfLogReg:
    """Bag-of-words floor. No pretrained weights, fully supervised."""

    name = "tfidf+logreg"

    def __init__(self) -> None:
        self.vectoriser = None
        self.clf = None
        self.classes_: list[str] = []
        self.trained_on = 0

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogReg":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = self.vectoriser.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        self.trained_on = len(texts)
        return self

    def predict(self, texts: list[str]) -> BaselineResult:
        # Inference latency includes vectorisation, so the comparison against the
        # neural systems is like-for-like.
        times = []
        preds = []
        for t in texts:
            t0 = time.perf_counter()
            preds.append(str(self.clf.predict(self.vectoriser.transform([t]))[0]))
            times.append((time.perf_counter() - t0) * 1000)
        return BaselineResult(
            predictions=preds,
            latency_ms_p50=_percentile(times, 0.5),
            latency_ms_p95=_percentile(times, 0.95),
            trained_on=self.trained_on,
        )


class MiniLMClassifier:
    """Frozen all-MiniLM-L6-v2 embeddings + logistic regression.

    Chosen over sentence-transformers to avoid another dependency; mean pooling
    over the last hidden state reproduces its sentence embeddings closely
    enough for a baseline.
    """

    name = "minilm+logreg"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self.tok = None
        self.enc = None
        self.clf = None
        self.device = "cpu"
        self.trained_on = 0

    def _embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.enc is None:
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.enc = AutoModel.from_pretrained(self.model_id)
            self.enc.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.enc.to(self.device)

        vecs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self.tok(batch, padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(self.device)
                out = self.enc(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs.append(pooled.cpu().numpy())
        return np.vstack(vecs)

    def fit(self, texts: list[str], labels: list[str]) -> "MiniLMClassifier":
        from sklearn.linear_model import LogisticRegression

        X = self._embed(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(X, labels)
        self.trained_on = len(texts)
        return self

    def predict(self, texts: list[str]) -> BaselineResult:
        times = []
        preds = []
        for t in texts:
            t0 = time.perf_counter()
            vec = self._embed([t])
            preds.append(str(self.clf.predict(vec)[0]))
            times.append((time.perf_counter() - t0) * 1000)
        return BaselineResult(
            predictions=preds,
            latency_ms_p50=_percentile(times, 0.5),
            latency_ms_p95=_percentile(times, 0.95),
            trained_on=self.trained_on,
        )


class FineTunedEncoder:
    """roberta-base fine-tuned on the dataset's own train split.

    This is the supervised ceiling the paper reports 89.1 on for CLINC150. It
    is also the honest counterweight to a zero-shot model: give it labels and it
    should win on a fixed taxonomy.
    """

    def __init__(self, model_id: str = "roberta-base", epochs: int = 3,
                 lr: float = 2e-5, batch_size: int = 32, seed: int = 0) -> None:
        self.model_id = model_id
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.tok = None
        self.enc = None
        self.clf = None
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}
        self.device = "cpu"

    def fit(self, texts: list[str], labels: list[str], subset_size: int | None = None) -> "FineTunedEncoder":
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            log.warning("fine-tuning on CPU: this will be slow")

        self.label2id = {lab: i for i, lab in enumerate(sorted(set(labels)))}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

        if subset_size is not None and subset_size < len(texts):
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(texts), size=subset_size, replace=False)
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]
        self.trained_on = len(texts)

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.enc = AutoModel.from_pretrained(self.model_id).to(self.device)

        hidden = self.enc.config.hidden_size
        self.clf = torch.nn.Linear(hidden, len(self.label2id)).to(self.device)

        class _DS(Dataset):
            def __init__(self, t, y):
                self.t, self.y = t, y

            def __len__(self):
                return len(self.t)

            def __getitem__(self, i):
                return self.t[i], self.y[i]

        loader = DataLoader(
            _DS(texts, [self.label2id[y] for y in labels]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        opt = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.clf.parameters()), lr=self.lr
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        self.enc.train()
        self.clf.train()
        for epoch in range(self.epochs):
            total = 0.0
            for batch_text, batch_y in loader:
                enc = self.tok(
                    list(batch_text), padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                ).to(self.device)
                labels_t = torch.as_tensor(list(batch_y), device=self.device)
                opt.zero_grad()
                out = self.enc(**enc).last_hidden_state[:, 0]  # CLS
                loss = loss_fn(self.clf(out), labels_t)
                loss.backward()
                opt.step()
                total += float(loss.detach())
            log.info("epoch %d/%d loss %.4f", epoch + 1, self.epochs, total / max(1, len(loader)))

        self.enc.eval()
        self.clf.eval()
        return self

    def predict(self, texts: list[str]) -> BaselineResult:
        import torch

        preds = []
        times = []
        with torch.no_grad():
            for t in texts:
                t0 = time.perf_counter()
                enc = self.tok([t], padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(self.device)
                logits = self.clf(self.enc(**enc).last_hidden_state[:, 0])
                preds.append(self.id2label[int(logits.argmax(-1)[0])])
                times.append((time.perf_counter() - t0) * 1000)
        return BaselineResult(
            predictions=preds,
            latency_ms_p50=_percentile(times, 0.5),
            latency_ms_p95=_percentile(times, 0.95),
            trained_on=getattr(self, "trained_on", 0),
        )


def build_baseline(kind: str, fixture: str) -> object:
    """Fit a baseline of the requested kind on the fixture's train split."""
    texts, labels = _load_train(fixture)
    log.info("fitting %s on %s: %d train rows", kind, fixture, len(texts))
    if kind == "tfidf":
        return TfidfLogReg().fit(texts, labels)
    if kind == "minilm":
        return MiniLMClassifier().fit(texts, labels)
    if kind == "roberta":
        return FineTunedEncoder().fit(texts, labels)
    raise ValueError(f"unknown baseline {kind!r}")


def build_learning_curve(fixture: str, sizes=(100, 500, 2000), epochs: int = 2) -> list[BaselineResult]:
    """Fine-tune accuracy vs number of labelled examples.

    Each point retrains from scratch on a random subset of that size, so the
    curve answers "how many labels would I need to beat the zero-shot model?"
    """
    import pandas as pd

    texts, labels = _load_train(fixture)
    frame = pd.DataFrame({"text": texts, "label": labels})
    eval_texts = frame["text"].head(300).tolist()

    out = []
    for size in sizes:
        if size > len(texts):
            continue
        model = FineTunedEncoder(epochs=epochs).fit(
            frame["text"].tolist(), frame["label"].tolist(), subset_size=size
        )
        result = model.predict(eval_texts)
        out.append(result)
        log.info("curve point: n=%d trained", result.trained_on)
    return out