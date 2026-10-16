"""BasqueGLUE NERC (EIEC) loader: BIO tokens → GLiNER records.

Both NERC subsets are licensed CC BY-NC-SA 4.0 by their authors; any
model trained on them inherits the non-commercial, share-alike terms.

Text is the token sequence joined with single spaces (so punctuation is
space-separated); spans are character offsets into that text.
"""

import pandas as pd

PARQUET = "hf://datasets/orai-nlp/basqueGLUE@refs/convert/parquet"

# Tag order of the original basqueGLUE.py loading script
BIO_TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC",
            "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

# EIEC type → the PII model's entity type
LABEL_MAP = {"PER": "person name", "LOC": "location",
             "ORG": "organization", "MISC": "miscellaneous"}
NER_TYPES = list(LABEL_MAP.values())

# subset: nerc_id (in-domain news) or nerc_od (out-of-domain news);
# split: train / validation / test
SPLITS = ("nerc_id/train", "nerc_id/validation", "nerc_id/test",
          "nerc_od/validation", "nerc_od/test")


def load(split):
    """`nerc_id/test` → [{"text", "entities", "spans"}]."""
    df = pd.read_parquet(f"{PARQUET}/{split}/0000.parquet")
    records = []
    for tokens, tags in zip(df["tokens"], df["tags"]):
        tokens = list(tokens)
        labels = [BIO_TAGS[t] if t < len(BIO_TAGS) else "O" for t in tags]
        starts, position = [], 0
        for token in tokens:
            starts.append(position)
            position += len(token) + 1
        text = " ".join(tokens)
        entities = {t: [] for t in NER_TYPES}
        spans = []
        i = 0
        while i < len(labels):
            if not labels[i].startswith("B-"):
                i += 1
                continue
            etype = labels[i][2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            if etype in LABEL_MAP:
                start = starts[i]
                end = starts[j - 1] + len(tokens[j - 1])
                entities[LABEL_MAP[etype]].append(text[start:end])
                spans.append([start, end, LABEL_MAP[etype]])
            i = j
        records.append({"text": text, "entities": entities, "spans": spans})
    return records
