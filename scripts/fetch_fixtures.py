"""Fixture downloader.

Pins trimmed, labelled samples of each public dataset into data/fixtures/*.csv so
the app runs offline after first setup. Sizes are deliberately small: the point
is a reproducible evaluation set, not a corpus.

    python scripts/fetch_fixtures.py              # everything
    python scripts/fetch_fixtures.py banking77    # one fixture
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURES = Path("data/fixtures")

# name -> how to pull one labelled sample out of a public dataset.
#   text/label  : column names in the main config
#   label_text  : optional column already holding a readable name
#   names       : (config, split) exposing id/name[/description] for label ids
SPECS: dict[str, dict] = {
    "banking77": {
        "repo": "DeepPavlov/banking77",
        "config": "default",
        "split": "test",
        "text": "utterance",
        "label": "label",
        "names": ("intents", "intents"),
        "desc": True,  # banking77 is the only set here with real descriptions
        "n": 1500,
        "note": "77 real banking intents; ships a description per intent",
    },
    "clinc150": {
        "repo": "DeepPavlov/clinc150",
        "config": "default",
        "split": "test",
        "text": "utterance",
        "label": "label",
        "names": ("intents", "intents"),
        "desc": False,  # verified: description is null for every CLINC intent
        "n": 1500,
        "note": "150 intents plus out-of-scope",
    },
    "hate_speech": {
        "repo": "tdavidson/hate_speech_offensive",
        "config": "default",
        "split": "train",
        "text": "tweet",
        "label": "class",
        "label_text": None,
        "label_names": {0: "hate_speech", 1: "offensive", 2: "neither"},
        "n": 1500,
        "note": "hate / offensive / neither",
    },
    "yelp": {
        "repo": "bespokelabs/yelp_restaurant_reviews_5k",
        "config": "default",
        "split": "train",
        "text": "text",
        "label": "label",
        "label_text": None,
        "label_names": {0: "negative", 1: "neutral", 2: "positive"},
        "n": 1000,
        # Mapping verified by reading sample reviews, not assumed: id 1 is
        # 3-star prose ("This is 3 stars, but it's not A OK"), id 2 is 4-5 star
        # ("this may be my new favorite place"). The common convention here is
        # wrong, and the file ships no card metadata to check against.
        "note": "3-class sentiment; no star rating in this subset",
    },
    "enron_spam": {
        "repo": "SetFit/enron_spam",
        "config": "default",
        "split": "test",
        "text": "text",
        "label": "label",
        "label_text": "label_text",  # already "spam" / "ham"
        "n": 1000,
        "note": "spam vs ham from a real mailbox; subject column kept",
    },
}


def load_label_names(spec: dict) -> tuple[dict, dict]:
    """Resolve integer label ids to names (and descriptions where present).

    Returns (id -> name, name -> description). A schema-driven classifier needs
    real names: handed the raw id "61" instead of "card_lost" it has nothing to
    condition on, so this join is load-bearing, not cosmetic.
    """
    names: dict = {}
    descs: dict = {}
    if not spec.get("names"):
        return names, descs
    from datasets import load_dataset

    cfg, split = spec["names"]
    meta = load_dataset(spec["repo"], cfg, split=split)
    has_desc = spec.get("desc") and "description" in meta.column_names
    for row in meta:
        key = int(row["id"])
        names[key] = row["name"]
        if has_desc and row.get("description"):
            descs[row["name"]] = row["description"]
    return names, descs


def fetch(name: str) -> bool:
    from datasets import load_dataset

    spec = SPECS[name]
    print(f"\n=== {name} ===")
    print(f"  {spec['repo']} [{spec['config']}/{spec['split']}]  n={spec['n']}")

    ds = load_dataset(spec["repo"], spec["config"], split=spec["split"])
    print(f"  loaded {len(ds):,} rows; columns={ds.column_names}")

    names, descs = load_label_names(spec)
    if names:
        print(f"  resolved {len(names)} label ids -> names ({len(descs)} with descriptions)")

    n_take = min(spec["n"], len(ds))
    # Shuffled selection so a truncated fixture still covers every label,
    # rather than only the early ids.
    sample = ds.shuffle(seed=0).select(range(n_take))

    fallback = spec.get("label_names", {})
    oos_label = spec.get("oos_label")
    rows = []
    for row in sample:
        raw = row[spec["label"]]
        if raw is None:
            # CLINC150 marks out-of-scope utterances with a null label in the
            # DeepPavlov mirror, and the intents config has no entry for it.
            # Keeping it as an explicit label is what gives the "escalate to a
            # human" head real ground truth.
            label = oos_label or "oos"
        elif spec.get("label_text"):
            label = row[spec["label_text"]]
        else:
            key = int(raw)
            label = names.get(key, fallback.get(key, raw))
        rows.append(
            {
                "text": row[spec["text"]],
                "label_id": "" if raw is None else raw,
                "label": str(label),
                "description": descs.get(str(label), ""),
            }
        )

    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"{name}.csv"
    pd = __import__("pandas")
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)

    counts = frame["label"].value_counts()
    manifest = {
        "name": name,
        "source": {"repo": spec["repo"], "config": spec["config"], "split": spec["split"]},
        "note": spec["note"],
        "n": len(frame),
        "n_labels": int(counts.size),
        "labels": sorted(frame["label"].unique().tolist()),
        "labels_with_description": sorted(descs.keys()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }
    (FIXTURES / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"  wrote {out} ({len(frame)} rows, {counts.size} distinct labels)")
    if counts.size > 1:
        top = counts.head(3).to_dict()
        print(f"  most common: {top}")
    print(f"  note: {spec['note']}")
    return True


# Fixtures whose dataset also ships a train split. A supervised baseline is
# defined by having labels, so the learning-curve comparison needs these.
TRAIN_SPECS: dict[str, dict] = {
    "banking77": {"repo": "DeepPavlov/banking77", "config": "default", "split": "train",
                  "n": 10000},
    "clinc150": {"repo": "DeepPavlov/clinc150", "config": "default", "split": "train",
                 "n": 15250},
}


def fetch_train(name: str) -> bool:
    """Write <name>_train.csv so baselines and the learning curve have labels."""
    from datasets import load_dataset

    spec = TRAIN_SPECS[name]
    print(f"\n=== {name} [train split for baselines] ===")
    ds = load_dataset(spec["repo"], spec["config"], split=spec["split"])
    main = SPECS[name]
    names, _ = load_label_names(main)

    take = min(spec["n"], len(ds))
    sample = ds.shuffle(seed=1).select(range(take))
    rows = []
    for row in sample:
        raw = row[main["label"]]
        label = "oos" if raw is None else str(names.get(int(raw), raw))
        rows.append({"text": row[main["text"]], "label_id": "" if raw is None else raw,
                     "label": label, "description": ""})

    pd = __import__("pandas")
    out = FIXTURES / f"{name}_train.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"  wrote {out} ({len(frame)} rows, {frame['label'].nunique()} distinct labels)")
    return True


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--with-train"]
    want_train = "--with-train" in sys.argv[1:]
    targets = argv or list(SPECS)
    unknown = [t for t in targets if t not in SPECS]
    if unknown:
        print(f"unknown fixture(s): {', '.join(unknown)}")
        print(f"known: {', '.join(SPECS)}")
        return 1

    failed = []
    for name in targets:
        try:
            fetch(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)

    if want_train:
        for name in targets:
            if name in TRAIN_SPECS:
                try:
                    fetch_train(name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  TRAIN FAILED: {type(exc).__name__}: {exc}")
                    failed.append(name + "_train")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print(f"\nAll fixtures written to {FIXTURES}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())