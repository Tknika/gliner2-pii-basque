---
license: cc-by-nc-sa-4.0
base_model:
- fastino/gliner2-privacy-filter-PII-multi
datasets:
- orai-nlp/basqueGLUE
- tknika/pii-synthetic-basque-v2
library_name: gliner2
language:
- eu
- es
- fr
model-index:
- name: gliner2-PII-basque-v2
  results:
  - task:
      type: token-classification
      name: PII Detection
    dataset:
      type: tknika/pii-synthetic-basque-v2
      name: pii-synthetic-basque-v2 (held-out-template test)
      split: test
    metrics:
    - type: f1
      name: Micro F1 (exact span)
      value: 0.905
    - type: precision
      name: Micro Precision
      value: 0.881
    - type: recall
      name: Micro Recall
      value: 0.931
  - task:
      type: token-classification
      name: Basque Named Entity Recognition
    dataset:
      type: orai-nlp/basqueGLUE
      name: BasqueGLUE NERC in-domain (nerc_id)
      config: nerc_id
      split: test
    metrics:
    - type: f1
      name: Micro F1 (exact span)
      value: 0.752
    - type: f1
      name: Person name F1
      value: 0.861
  - task:
      type: token-classification
      name: Basque Named Entity Recognition
    dataset:
      type: orai-nlp/basqueGLUE
      name: BasqueGLUE NERC out-of-domain (nerc_od)
      config: nerc_od
      split: test
    metrics:
    - type: f1
      name: Micro F1 (exact span)
      value: 0.717
    - type: f1
      name: Person name F1
      value: 0.756
tags:
- gliner2
- named-entity-recognition
- ner
- basque
- euskara
- pii
- privacy
- information-extraction
- education
---

# gliner2-PII-basque-v2

A Basque-adapted fine-tune of [fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) (205M parameters, GLiNER2 architecture, mDeBERTa-v3-base encoder) for detecting personally identifiable information (PII) in Basque, Spanish and French text from the Basque Country, with three entity types for educational settings: `user name` (LMS/forum handles), `personal url` and `student id`.

It is version 2 of [tknika/gliner2-PII-basque](https://huggingface.co/tknika/gliner2-PII-basque). Code, data and evaluation outputs: **[github.com/tknika/gliner2-pii-basque](https://github.com/tknika/gliner2-pii-basque)**.

> **License change (October 2026).** This model was first published as Apache-2.0. It is trained on BasqueGLUE NERC, which its authors license under **CC BY-NC-SA 4.0**, so the model is now distributed under CC BY-NC-SA 4.0 as well: non-commercial use only, with attribution, and derivatives under the same license. The weights did not change.

## Usage

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("tknika/gliner2-PII-basque-v2")

text = ("Kaixo, @mikel_etxeberria naiz ikaslea, ikasle zk 123456B dut. "
        "Nire bloga https://mikeletxeberria.wordpress.com da eta "
        "helbidea Barakaldoko Nagusia kalea 42 da, 48901.")
result = model.extract_entities(
    text,
    ["person name", "email", "phone number", "national id",
     "bank account number", "credit card number", "date of birth",
     "date", "address", "city", "location", "zip code", "age",
     "organization", "user name", "personal url", "student id"],
)
```

All numbers below use the default threshold (0.5).

## Training

- **Basque NER**: BasqueGLUE `nerc_id/train` (2,842 sentences, EIEC newswire), with EIEC labels mapped to the base model's types: PER→`person name`, LOC→`location`, ORG→`organization`, MISC→`miscellaneous`.
- **Synthetic PII replay**: the train split (5,000 sentences) of [tknika/pii-synthetic-basque-v2](https://huggingface.co/datasets/tknika/pii-synthetic-basque-v2), covering 17 PII types in eu/es/fr. It is mixed in so the model does not forget its PII types (experience replay).
- **Method**: LoRA (r = 16, alpha = 32, dropout 0). Targets: the encoder's attention and feed-forward layers (`query_proj`, `key_proj`, `value_proj`, `dense`) plus the span, classifier and count heads. After training, the 13 MB adapter was merged into the base model. This repository holds the merged model.
- **Hyperparameters**: 5 epochs, batch size 16, AdamW (weight decay 0.01), learning rate 1e-5 for the encoder and 5e-4 for the task layers, linear schedule with 10% warmup, bf16, seed 42.
- **Checkpoint selection**: lowest loss on `nerc_id/validation` (epoch 3). The test splits were never used for training or selection.
- **Compute**: one NVIDIA L40, about 6 minutes. Software: gliner2 2.1.0, peft 0.21.2, torch 2.14.1, transformers 5.19.0.

## Evaluation

Scoring is exact match on (type, start, end) spans, micro-averaged over types. Brackets show 95% confidence intervals from a 1,000-sample sentence-level bootstrap. Scripts and per-sentence predictions are in the GitHub repository (`results/`).

### PII detection: held-out templates (primary)

The test set has 1,000 sentences: 573 eu, 257 es, 170 fr. It is built from 30 sentence templates and 24 PII-free sentences that never appear in training, written in school and LMS registers (teacher notes, grant forms, forum posts, private messages). It shares no names, emails, user names or URLs with the training data, and given names are balanced by sex (216 female / 258 male name mentions).

| Model | Precision | Recall | **Micro F1** | Predictions on 159 PII-free sentences |
|---|---|---|---|---|
| Base (zero-shot) | 0.672 | 0.828 | 0.742 [0.727, 0.757] | 225 |
| v1 | 0.793 | 0.904 | 0.845 [0.832, 0.858] | 39 |
| **v2 (this model)** | **0.881** | **0.931** | **0.905 [0.895, 0.915]** | **0** |

Gain of v2 over v1 (paired bootstrap): +0.061 [+0.050, +0.071]. By language, v2 scores eu 0.911, es 0.906 and fr 0.884. Person-name recall is 0.995 for female names and 0.996 for male names.

Per-type F1 on the held-out templates:

| Type | Base | v1 | **v2** | v2 P / R | Support |
|---|---|---|---|---|---|
| person name | 0.885 | 0.956 | 0.893 | 0.810 / 0.996 | 474 |
| phone number | 0.960 | 0.919 | 0.964 | 0.931 / 1.000 | 242 |
| user name | 0.865 | 0.766 | 0.926 | 1.000 / 0.863 | 182 |
| date | 0.607 | 0.941 | 1.000 | 1.000 / 1.000 | 175 |
| email | 0.868 | 0.668 | 0.988 | 0.977 / 1.000 | 171 |
| national id | 0.778 | 0.753 | **0.591** | 0.552 / 0.637 | 168 |
| student id | 0.658 | 0.826 | 0.979 | 0.959 / 1.000 | 162 |
| city | 0.864 | 0.793 | 0.911 | 0.846 / 0.987 | 150 |
| credit card number | 0.928 | 0.984 | 0.927 | 1.000 / 0.863 | 95 |
| location | 0.534 | 0.593 | **0.603** | 0.953 / 0.441 | 93 |
| organization | 0.592 | 0.802 | 0.742 | 0.625 / 0.914 | 93 |
| personal url | 0.277 | 0.915 | 0.970 | 1.000 / 0.943 | 87 |
| bank account number | 0.883 | 0.825 | 0.994 | 0.988 / 1.000 | 80 |
| age | 0.212 | 1.000 | 1.000 | 1.000 / 1.000 | 80 |
| address | 0.015 | 0.741 | 0.971 | 0.971 / 0.971 | 70 |
| date of birth | 1.000 | 0.892 | 1.000 | 1.000 / 1.000 | 70 |
| zip code | 0.558 | 0.754 | 1.000 | 1.000 / 1.000 | 70 |

Main error patterns:
- **National ID vs. student ID.** In sentences with both, the model sometimes swaps the two labels (for example «823181E espedientea … NANaren kopia (63893975K)»). For redaction, both values are still caught; only the label is wrong.
- **User names labelled as person names.** Handles like `laurent_75` sometimes get the `person name` label, which lowers person-name precision.
- **Provinces after a preposition** («centros de Gipuzkoa») are often missed as `location` or absorbed into a longer `organization` span.

### PII detection: in-template eval split (not comparable)

On the `eval` split of the dataset, v2 scores **0.974** [0.970, 0.978] (base 0.732, v1 0.824). That split uses the same 47 templates as v2's training data (85% of its sentences have the same sentence frame as a training sentence), so it measures in-template generalization only. The 0.975 figure first published on this card came from that split. Use the held-out-template numbers above instead.

### Basque NER: BasqueGLUE test splits

| Split | Metric | Base | v1 | **v2** |
|---|---|---|---|---|
| nerc_id/test (1,846 sent.) | micro F1 | 0.500 [0.479, 0.521] | 0.738 [0.722, 0.754] | **0.752 [0.736, 0.769]** |
| | person name F1 | 0.716 | 0.850 | **0.861** |
| | location / organization / misc F1 | 0.562 / 0.420 / 0.056 | 0.759 / 0.643 / 0.467 | 0.769 / 0.672 / 0.490 |
| nerc_od/test (598 sent.) | micro F1 | 0.511 [0.481, 0.541] | 0.690 [0.663, 0.717] | **0.717 [0.692, 0.744]** |
| | person name F1 | 0.721 | 0.766 | 0.756 |

v2 is better than v1 on both test splits (paired bootstrap: nerc_id +0.015 [+0.003, +0.026], nerc_od +0.027 [+0.010, +0.044]). The PER drop from v1 to v2 reported earlier was measured on 500 validation sentences and is not confirmed on the test splits.

## Limitations

- **Synthetic PII evaluation.** Even the held-out templates are generated text. No evaluation on real student texts exists yet, and real-world performance will be lower.
- **Forgetting is measured only for the 17 replayed types.** The base model knows 42 PII types; the other 25 were not replayed or evaluated and may have degraded. English was not evaluated.
- **Name distribution in training.** A bug in the source data left Basque female given names out of the southern-territory training names (about 79% of person names in the training set are male). On the sex-balanced test set, recall shows no gap (0.995 female vs. 0.996 male). Names from outside the Basque, Spanish and French lists (for example Arabic, Romanian or Latin American names) are not represented in training or evaluation.
- **Single training run.** The confidence intervals cover test-set sampling, not variation between training seeds.
- **Not production-tested.** Use it as one layer of a redaction pipeline, with human review for sensitive data.

## Citation

```bibtex
@misc{tknika2026gliner2piibasquev2,
  title  = {gliner2-PII-basque-v2: A Basque-Adapted Multilingual PII Detection Model with Educational Entity Types},
  author = {{TKNIKA} and Ezpeleta Mendikute, Xabier},
  year   = {2026},
  url    = {https://huggingface.co/tknika/gliner2-PII-basque-v2}
}
```

This model builds on:

```bibtex
@misc{zaratiana2026gliner2piimultilingualmodelpersonally,
  title  = {GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction},
  author = {Zaratiana, Urchade and Lewis, Ash and Hurn-Maloney, George},
  year   = {2026},
  eprint = {2605.09973},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.09973}
}

@InProceedings{urbizu2022basqueglue,
  author    = {Urbizu, Gorka and San Vicente, Iñaki and Saralegi, Xabier and Agerri, Rodrigo and Soroa, Aitor},
  title     = {BasqueGLUE: A Natural Language Understanding Benchmark for Basque},
  booktitle = {Proceedings of the Language Resources and Evaluation Conference},
  year      = {2022},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {1603--1612},
  url       = {https://aclanthology.org/2022.lrec-1.172}
}

@misc{he2021debertav3,
  title  = {DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author = {He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  year   = {2021},
  eprint = {2111.09543},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2111.09543}
}
```
