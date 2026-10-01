"""Fixture loading and schema construction.

A schema is the model's input contract: the label set is supplied at call time,
not baked into the weights. Building it from a fixture keeps the evaluation
honest, because the label set is exactly the set the gold labels came from.

Schema forms supported by gliner2 2.0.0, all verified in scripts/verify_model.py:

    {"intent": ["a", "b"]}                                  # single label
    {"intent": {"labels": {"a": "does A", "b": "does B"}}}   # described labels
    {"topics": {"labels": [...], "multi_label": True,
                "cls_threshold": 0.4}}                       # multi label
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

FIXTURE_DIR = Path("data/fixtures")


@dataclass
class Fixture:
    """One labelled evaluation sample plus the schema derived from it."""

    name: str
    texts: list[str]
    labels: list[str]
    label_names: list[str]
    descriptions: dict[str, str] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    def schema(
        self,
        head: str = "intent",
        *,
        described: bool = False,
        multi_label: bool = False,
        threshold: float = 0.4,
        label_subset: list[str] | None = None,
    ) -> dict:
        """Build a schema for this fixture.

        label_subset restricts the schema to a disjoint slice of intents. That
        is how the dynamic-schema test works: a classifier trained on App A's
        labels cannot emit App B's, so we hand the model only one app's labels
        at a time and compare.
        """
        names = label_subset if label_subset is not None else self.label_names

        if multi_label:
            return {head: {"labels": list(names), "multi_label": True, "cls_threshold": threshold}}

        if described and self.descriptions:
            usable = {n: self.descriptions[n] for n in names if self.descriptions.get(n)}
            if usable:
                return {head: {"labels": usable}}

        return {head: list(names)}

    def described_coverage(self) -> float:
        """Share of labels that carry a description. CLINC150 is 0.0, banking77 is 1.0."""
        if not self.label_names:
            return 0.0
        return sum(1 for n in self.label_names if self.descriptions.get(n)) / len(self.label_names)


def _manifest_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.manifest.json"


@lru_cache(maxsize=None)
def load(name: str) -> Fixture:
    """Load a fixture written by scripts/fetch_fixtures.py."""
    csv_path = FIXTURE_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"fixture {name!r} not found at {csv_path}. "
            f"Run: python scripts/fetch_fixtures.py {name}"
        )

    import pandas as pd

    frame = pd.read_csv(csv_path).fillna({"text": "", "label": "", "description": ""})
    manifest = {}
    if _manifest_path(name).exists():
        manifest = json.loads(_manifest_path(name).read_text())

    descriptions = {
        str(row["label"]): str(row["description"])
        for _, row in frame.iterrows()
        if str(row.get("description", "")).strip()
    }

    return Fixture(
        name=name,
        texts=[str(t) for t in frame["text"].tolist()],
        labels=[str(x) for x in frame["label"].tolist()],
        label_names=sorted({str(x) for x in frame["label"].tolist()}),
        descriptions=descriptions,
        manifest=manifest,
    )


def available() -> list[dict]:
    """List fixtures on disk, for the UI."""
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.manifest.json")):
        data = json.loads(path.read_text())
        out.append(
            {
                "name": data["name"],
                "n": data.get("n"),
                "n_labels": data.get("n_labels"),
                "note": data.get("note"),
                "source": data.get("source"),
            }
        )
    return out


def split_labels(fixture: Fixture, seed: int = 0, fraction: float = 0.5) -> tuple[list[str], list[str]]:
    """Deterministically split a fixture's labels into two disjoint schemas.

    This is the schema-swap test from arXiv:2608.20371, where a classifier
    trained on one app's intents scores 0% on another's. Deterministic so the
    split is reproducible across runs and machines.
    """
    import random

    labels = sorted(fixture.label_names)
    rng = random.Random(seed)
    rng.shuffle(labels)
    cut = int(len(labels) * fraction)
    return sorted(labels[:cut]), sorted(labels[cut:])