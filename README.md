# gliner2-pii-basque

Fine-tuning [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) to detect personal information in Basque, Spanish and French text from the Basque Country, with entity types for educational settings (`user name`, `personal url`, `student id`).

| Artifact | Link | License |
|---|---|---|
| Model v2 (current) | [tknika/gliner2-PII-basque-v2](https://huggingface.co/tknika/gliner2-PII-basque-v2) | CC BY-NC-SA 4.0 |
| Model v1 | [tknika/gliner2-PII-basque](https://huggingface.co/tknika/gliner2-PII-basque) | CC BY-NC-SA 4.0 |
| Synthetic dataset v2 | [tknika/pii-synthetic-basque-v2](https://huggingface.co/datasets/tknika/pii-synthetic-basque-v2) | CC BY 4.0 |
| This code | — | Apache-2.0 (`data/sources/streets.csv`, `towns.csv`: ODbL) |

The models are CC BY-NC-SA 4.0 because they are trained on BasqueGLUE NERC, which carries that license.

## Approach

1. **Basque NER**: BasqueGLUE `nerc_id/train` (2,842 hand-annotated newswire sentences) teaches Basque names, including declined forms («Xabier Arzalluzen»).
2. **Experience replay**: 5,000 synthetic PII sentences are mixed into training so the model keeps its PII types. Fine-tuning on NER alone made the base model lose email, phone, date-of-birth and postcode detection.
3. **LoRA** (r = 16) on the encoder and task heads. The adapter is merged into a standalone model afterwards.

## Results (v2)

Exact-span micro F1 with 95% bootstrap confidence intervals. Full per-type tables and slices are in [`cards/model-card-v2.md`](cards/model-card-v2.md) and [`results/evaluation.json`](results/evaluation.json).

| Benchmark | Base | v1 | **v2** |
|---|---|---|---|
| PII, held-out templates (primary) | 0.742 [0.727, 0.757] | 0.845 [0.832, 0.858] | **0.905 [0.895, 0.915]** |
| PII, in-template eval (optimistic) | 0.732 | 0.824 | 0.974 |
| BasqueGLUE nerc_id/test | 0.500 | 0.738 | **0.752** |
| BasqueGLUE nerc_od/test | 0.511 | 0.690 | **0.717** |

## Layout

```
src/gliner2_pii_basque/
  basqueglue.py          BasqueGLUE NERC loader (BIO → GLiNER records with spans)
  metrics.py             exact-span / string P/R/F1, bootstrap CIs, paired deltas
  synthetic/
    sources.py           loads data/sources/ tables
    values.py            checksum-valid IDs, IBANs, cards, phones, emails, dates; Basque declension
    templates.py         training templates and held-out test templates
    generator.py         person contexts, v2 splits, held-out test
    audit.py             data checks
scripts/
  generate_data.py       regenerate / verify the synthetic splits
  audit_data.py          audit JSONL files
  train.py               fine-tune (full or LoRA, optional replay)
  merge_adapter.py       merge LoRA into the base model, verified on a probe
  evaluate.py            all benchmarks, any number of models
  upload_hf.py           publish cards / data to the Hub
  fetch_sources/         rebuild data/sources/ from INE, INSEE, EODA, OSM
data/
  sources/               aggregate source tables (see its README)
  synthetic/             v2-train, v2-eval, v2-test-heldout-templates (JSONL)
cards/                   Hugging Face model and dataset cards
results/                 evaluation.json + per-sentence predictions per model
```

## Reproduce

Requires Python 3.12 and, for training, a CUDA GPU (one L40 takes about 6 minutes).

```bash
uv venv --python 3.12 && uv pip install -e .

# data: verify that the generator reproduces the published v2 splits
python scripts/generate_data.py v2-splits --check
python scripts/audit_data.py data/synthetic/*.jsonl

# train v2 and merge the adapter
python scripts/train.py --mode lora --replay-file data/synthetic/v2-train.jsonl \
    --output-dir outputs/v2
python scripts/merge_adapter.py outputs/v2/best outputs/v2/merged

# evaluate
python scripts/evaluate.py base=fastino/gliner2-privacy-filter-PII-multi \
    v1=tknika/gliner2-PII-basque v2=tknika/gliner2-PII-basque-v2 local=outputs/v2/merged
```

Reproducibility checks done in October 2026:
- `generate_data.py v2-splits --check` reproduces both published JSONL files exactly.
- Merging the original v2 adapter with `merge_adapter.py` reproduces the published `model.safetensors` byte for byte (sha256 `b3fab375…`).
- Retraining is seeded (`--seed 42`), but GPU kernels are not fully deterministic, so expect small differences.

## Evaluation protocol

- **Checkpoint selection** uses `nerc_id/validation` loss only. Reported NER numbers come from the `nerc_id` and `nerc_od` **test** splits.
- **PII** is reported on the **held-out-template** test set: 30 templates never used in training, no shared person values, and given names balanced by sex. The in-template `v2-eval` split is reported only for continuity with the numbers published before October 2026.
- **Matching** is exact (type, start, end). The string metric (type, text) used before October 2026 is also stored in `results/evaluation.json`.
- **Uncertainty** comes from a 1,000-resample sentence-level bootstrap. Model differences use a paired bootstrap on the same resamples.

## Known limitations and next steps

- **No real-text evaluation.** A small hand-annotated set of real (consented or public) Basque student texts is the most valuable next addition.
- **Only 17 of the base model's 42 PII types** are replayed and evaluated, and English is not evaluated.
- **Type confusions remain** in v2: DNI vs. student ID when both appear, user names tagged as person names, and provinces after prepositions.
- **Two-surname names** («Gorka Altuna Zabala») are rare in the training data (1.4%) and get truncated in practice. Planned v3 fix: 40–50% two-surname names, plus a separate truncation metric.
- **Female given names** were missing from the southern-territory training names (see `data/sources/README.md`). This is fixed in the source tables; the v3 data should be generated with `names_ine_fixed.csv`.
- **Single training seed.** Run 3 seeds for v3 to separate seed variance from real gains.
