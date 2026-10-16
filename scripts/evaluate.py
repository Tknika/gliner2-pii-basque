#!/usr/bin/env python3
"""Evaluate one or more models on every benchmark.

    python scripts/evaluate.py \\
        base=fastino/gliner2-privacy-filter-PII-multi \\
        v1=tknika/gliner2-PII-basque \\
        v2=tknika/gliner2-PII-basque-v2

Benchmarks
  nerc_id/test, nerc_od/test   BasqueGLUE NER test splits (never used for
                               training or checkpoint selection)
  pii-heldout                  synthetic PII, held-out templates (primary
                               PII benchmark; sliced by language and by
                               sex of the given name)
  pii-in-template              synthetic v2 eval split; shares templates
                               with training, so it measures in-template
                               generalization only

Writes results/evaluation.json (metrics, 95% bootstrap CIs, paired
deltas between models) and results/predictions/<model>/<benchmark>.jsonl.
"""

import argparse
import json
from pathlib import Path

import torch
from gliner2 import AutoExtractor

from gliner2_pii_basque import basqueglue
from gliner2_pii_basque.metrics import (counts, false_positives_on_negatives,
                                        paired_delta, predict, prf, summarize)
from gliner2_pii_basque.synthetic.generator import read_jsonl
from gliner2_pii_basque.synthetic.templates import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"


def benchmarks(include_validation):
    splits = ["nerc_id/test", "nerc_od/test"]
    if include_validation:
        splits += ["nerc_id/validation", "nerc_od/validation"]
    for split in splits:
        yield split, basqueglue.load(split), basqueglue.NER_TYPES
    yield ("pii-heldout", read_jsonl(DATA / "v2-test-heldout-templates.jsonl"),
           ENTITY_TYPES)
    yield "pii-in-template", read_jsonl(DATA / "v2-eval.jsonl"), ENTITY_TYPES


def slices(records, c_span, types):
    """Held-out slices: micro F1 per language, person-name recall by sex."""
    out = {"language": {}, "person_name_recall_by_sex": {}}
    for lang in ("eu", "es", "fr"):
        rows = [i for i, r in enumerate(records) if r["meta"]["lang"] == lang]
        tp, fp, fn = c_span[rows].sum(axis=(0, 1))
        out["language"][lang] = {"sentences": len(rows),
                                 "f1": round(prf(tp, fp, fn)[2], 3)}
    person = types.index("person name")
    for sex in ("female", "male"):
        rows = [i for i, r in enumerate(records)
                if r["meta"].get("person_sex") == sex]
        tp, _, fn = c_span[rows, person].sum(axis=0)
        out["person_name_recall_by_sex"][sex] = {
            "mentions": int(tp + fn), "recall": round(tp / (tp + fn), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="name=hf_id_or_path")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--include-validation", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation.json")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.models)

    data = list(benchmarks(args.include_validation))
    results = {"models": models, "threshold": args.threshold, "benchmarks": {}}
    span_counts = {}
    for name, path in models.items():
        model = AutoExtractor.from_pretrained(path).to(args.device).eval()
        for bench, records, types in data:
            predictions = predict(model, records, types, args.batch_size,
                                  args.threshold)
            out_dir = args.output.parent / "predictions" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{bench.replace('/', '_')}.jsonl", "w",
                      encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            c_span = counts(records, predictions, types, "span")
            c_string = counts(records, predictions, types, "string")
            span_counts[(bench, name)] = c_span
            entry = {"span": summarize(c_span, types),
                     "string": summarize(c_string, types)}
            negatives, fps = false_positives_on_negatives(records, predictions)
            if negatives:
                entry["negatives"] = {"sentences": negatives,
                                      "false_positives": fps}
            if bench == "pii-heldout":
                entry["slices"] = slices(records, c_span, types)
            results["benchmarks"].setdefault(bench, {})[name] = entry
            m = entry["span"]["micro"]
            print(f"{name:6s} {bench:20s} span F1 {m['f1']:.3f} "
                  f"[{m['f1_ci95'][0]:.3f}, {m['f1_ci95'][1]:.3f}]  "
                  f"string F1 {entry['string']['micro']['f1']:.3f}", flush=True)

    names = list(models)
    for bench, *_ in data:
        results["benchmarks"][bench]["paired_deltas_span_f1"] = {
            f"{a} - {b}": paired_delta(span_counts[(bench, a)],
                                       span_counts[(bench, b)])
            for i, a in enumerate(names) for b in names[:i]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
