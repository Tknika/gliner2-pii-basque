---
license: cc-by-nc-sa-4.0
base_model:
- fastino/gliner2-privacy-filter-PII-multi
datasets:
- orai-nlp/basqueGLUE
- tknika/pii-synthetic-basque
library_name: gliner2
language:
- eu
- es
- en
model-index:
- name: gliner2-PII-basque
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
      value: 0.845
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
      value: 0.738
    - type: f1
      name: Person name F1
      value: 0.850
tags:
- gliner2
- named-entity-recognition
- ner
- basque
- euskara
- pii
- privacy
- information-extraction
---

# gliner2-PII-basque

A Basque-adapted fine-tune of [fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) (205M parameters, GLiNER2 architecture, mDeBERTa-v3-base encoder), a multilingual PII (personally identifiable information) detection model.

**Superseded by [tknika/gliner2-PII-basque-v2](https://huggingface.co/tknika/gliner2-PII-basque-v2)**, which is better on every benchmark below. Evaluation code: [github.com/tknika/gliner2-pii-basque](https://github.com/tknika/gliner2-pii-basque).

> **License change (October 2026).** This model was first published as Apache-2.0. It is trained on BasqueGLUE NERC, which its authors license under **CC BY-NC-SA 4.0**, so the model is now distributed under CC BY-NC-SA 4.0 as well: non-commercial use only, with attribution, and derivatives under the same license. The weights did not change.

## Training

- **Basque NER**: BasqueGLUE `nerc_id/train` (2,842 sentences, EIEC newswire), with labels mapped to `person name`, `location`, `organization`, `miscellaneous`.
- **Synthetic PII replay**: 6,000 sentences from [tknika/pii-synthetic-basque](https://huggingface.co/datasets/tknika/pii-synthetic-basque) in eu/es/en, covering 14 PII types with checksum-valid values.
- **Method**: LoRA (r = 16, alpha = 32) on the encoder's attention and feed-forward layers and on the span, classifier and count heads. After training, the adapter was merged into the base model. 5 epochs, batch size 16, learning rate 1e-5 for the encoder and 5e-4 for the task layers, bf16. The checkpoint was selected on `nerc_id/validation` loss.

## Evaluation

Scoring is exact match on (type, start, end) spans, micro-averaged. Brackets show 95% bootstrap confidence intervals. The default threshold is 0.5.

| Benchmark | Base (zero-shot) | **v1 (this model)** | v2 |
|---|---|---|---|
| BasqueGLUE nerc_id/test, micro F1 | 0.500 [0.479, 0.521] | **0.738 [0.722, 0.754]** | 0.752 |
| BasqueGLUE nerc_id/test, person name F1 | 0.716 | **0.850** | 0.861 |
| BasqueGLUE nerc_od/test, micro F1 | 0.511 [0.481, 0.541] | **0.690 [0.663, 0.717]** | 0.717 |
| PII, held-out templates (17 types), micro F1 | 0.742 [0.727, 0.757] | **0.845 [0.832, 0.858]** | 0.905 |

The PII benchmark is the held-out-template test split of [pii-synthetic-basque-v2](https://huggingface.co/datasets/tknika/pii-synthetic-basque-v2). This model never saw its three educational types (`user name`, `personal url`, `student id`). It handles `personal url` (0.915) and `student id` (0.826), but `user name` drops below the base model (0.766 vs. 0.865). That drop is consistent with forgetting of types that are absent from the replay data.

Earlier versions of this card reported numbers on 500 BasqueGLUE validation sentences, the same split used for checkpoint selection. Those numbers are replaced by the test-split results above.

## Usage

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("tknika/gliner2-PII-basque")
result = model.extract_entities(
    "Kaixo, Ane Azkune naiz, Bilboko La Casilla kalean bizi naiz. "
    "Posta: ane.azkune@euskaltel.eus, telefonoa 688 123 456.",
    ["person name", "location", "address", "email", "phone number",
     "national id", "date of birth", "city", "zip code"],
)
```

## Limitations

- The PII evaluation uses synthetic text, and performance on real documents will be lower.
- Only 14 of the base model's 42 PII types were replayed; the others may have degraded.
- The training code for this version (v1 synthetic generator) is not maintained; see v2.

## Citation

```bibtex
@misc{tknika2026gliner2piibasque,
  title  = {gliner2-PII-basque: A Basque-Adapted Multilingual PII Detection Model},
  author = {{TKNIKA} and Ezpeleta Mendikute, Xabier},
  year   = {2026},
  url    = {https://huggingface.co/tknika/gliner2-PII-basque}
}

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
  pages     = {1603--1612},
  url       = {https://aclanthology.org/2022.lrec-1.172}
}
```
