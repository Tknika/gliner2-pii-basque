#!/usr/bin/env python3
"""Generate the synthetic PII datasets.

    python scripts/generate_data.py v2-splits [--check]
        Regenerate data/synthetic/v2-{train,eval}.jsonl (seeds 42 / 2025).
        With --check, compare against the committed files instead of
        writing them.

    python scripts/generate_data.py heldout-test [--n 1000] [--seed 2026]
        Build data/synthetic/v2-test-heldout-templates.jsonl from the
        held-out templates, with corrected name weights
        (names_ine_fixed.csv), excluding every text and person value of
        the v2 train/eval splits.
"""

import argparse
from collections import Counter
from pathlib import Path

from gliner2_pii_basque.synthetic.generator import (
    PERSON_TYPES, generate_heldout_test, generate_v2_splits, read_jsonl,
    write_jsonl)
from gliner2_pii_basque.synthetic.sources import Sources

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
TRAIN, EVAL = DATA / "v2-train.jsonl", DATA / "v2-eval.jsonl"
HELDOUT = DATA / "v2-test-heldout-templates.jsonl"


def person_values(records):
    return {m for r in records for t in PERSON_TYPES
            for m in r["entities"].get(t, [])}


def v2_splits(check):
    train, evals = generate_v2_splits(Sources(names_file="names_ine.csv"))
    overlap = (person_values(train) | {r["text"] for r in train}) & \
              (person_values(evals) | {r["text"] for r in evals})
    print(f"train {len(train)}, eval {len(evals)}, "
          f"overlap of texts/person values: {len(overlap)}")
    if check:
        for records, path in ((train, TRAIN), (evals, EVAL)):
            same = records == read_jsonl(path)
            print(f"{path.name}: {'identical' if same else 'DIFFERENT'}")
        return
    write_jsonl(train, TRAIN)
    write_jsonl(evals, EVAL)


def heldout_test(n, seed):
    seen = read_jsonl(TRAIN) + read_jsonl(EVAL)
    excluded = frozenset(person_values(seen) | {r["text"] for r in seen})
    records = generate_heldout_test(
        Sources(names_file="names_ine_fixed.csv"), n=n, seed=seed,
        excluded=excluded)
    write_jsonl(records, HELDOUT)
    meta = [r["meta"] for r in records]
    print(f"{len(records)} records → {HELDOUT}")
    print("languages:", dict(Counter(m["lang"] for m in meta)))
    print("given-name sex:", dict(Counter(m.get("person_sex") for m in meta
                                          if "person_sex" in m)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["v2-splits", "heldout-test"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.what == "v2-splits":
        v2_splits(args.check)
    else:
        heldout_test(args.n, args.seed)
