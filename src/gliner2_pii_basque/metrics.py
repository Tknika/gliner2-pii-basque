"""Prediction and scoring: exact-match P/R/F1 with bootstrap intervals.

Two matching modes:
  span    (type, start, end) must match a gold span exactly. Primary
          metric: repeated mentions count separately and position matters.
  string  (type, text) sets per sentence. The metric used for the v1/v2
          numbers reported before October 2026; kept for comparability.

Counts are kept per sentence so that 95% intervals can be computed with
a sentence-level bootstrap, and two models can be compared with a paired
bootstrap on the same resamples.
"""

import numpy as np

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0


def predict(model, records, types, batch_size=32, threshold=0.5):
    """[[{"type", "text", "start", "end"}, ...] per record]."""
    outputs = model.batch_extract_entities(
        [r["text"] for r in records], types, batch_size=batch_size,
        threshold=threshold, include_confidence=True, include_spans=True)
    predictions = []
    for output in outputs:
        found = []
        for etype, items in output.get("entities", {}).items():
            for item in items:
                found.append({"type": etype, "text": item["text"],
                              "start": item["start"], "end": item["end"]})
        predictions.append(found)
    return predictions


def _keys(items, mode):
    if mode == "span":
        return {(i["type"], i["start"], i["end"]) for i in items}
    return {(i["type"], i["text"]) for i in items}


def gold_items(record):
    return [{"type": t, "text": record["text"][s:e], "start": s, "end": e}
            for s, e, t in record["spans"]]


def counts(records, predictions, types, mode="span"):
    """Array (sentences, types, [tp, fp, fn])."""
    index = {t: i for i, t in enumerate(types)}
    out = np.zeros((len(records), len(types), 3), dtype=np.int64)
    for n, (record, predicted) in enumerate(zip(records, predictions)):
        gold = _keys(gold_items(record), mode)
        pred = {k for k in _keys(predicted, mode) if k[0] in index}
        for k in pred & gold:
            out[n, index[k[0]], 0] += 1
        for k in pred - gold:
            out[n, index[k[0]], 1] += 1
        for k in gold - pred:
            out[n, index[k[0]], 2] += 1
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _micro_f1(c):
    tp, fp, fn = c.sum(axis=tuple(range(c.ndim - 1)))
    return prf(tp, fp, fn)[2]


def _bootstrap_indices(n):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))


def summarize(c, types):
    """Micro P/R/F1 with 95% CI, per-type P/R/F1 and support."""
    tp, fp, fn = c.sum(axis=(0, 1))
    p, r, f = prf(tp, fp, fn)
    samples = [_micro_f1(c[i]) for i in _bootstrap_indices(len(c))]
    low, high = np.percentile(samples, [2.5, 97.5])
    per_type = {}
    for i, t in enumerate(types):
        ttp, tfp, tfn = c[:, i].sum(axis=0)
        if ttp + tfp + tfn == 0:
            continue
        tp_, tr_, tf_ = prf(ttp, tfp, tfn)
        per_type[t] = {"precision": round(tp_, 3), "recall": round(tr_, 3),
                       "f1": round(tf_, 3), "support": int(ttp + tfn),
                       "predicted": int(ttp + tfp)}
    return {"micro": {"precision": round(p, 3), "recall": round(r, 3),
                      "f1": round(f, 3),
                      "f1_ci95": [round(low, 3), round(high, 3)],
                      "tp": int(tp), "fp": int(fp), "fn": int(fn)},
            "types": per_type, "sentences": len(c)}


def paired_delta(c_a, c_b):
    """Micro-F1 difference a − b with a paired-bootstrap 95% CI."""
    deltas = [_micro_f1(c_a[i]) - _micro_f1(c_b[i])
              for i in _bootstrap_indices(len(c_a))]
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(_micro_f1(c_a) - _micro_f1(c_b), 3),
            "ci95": [round(low, 3), round(high, 3)]}


def false_positives_on_negatives(records, predictions):
    """(negative sentences, predictions made on them)."""
    negatives = [p for r, p in zip(records, predictions) if not r["spans"]]
    return len(negatives), sum(len(p) for p in negatives)
