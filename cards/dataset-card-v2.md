---
license: cc-by-4.0
language:
- eu
- es
- fr
tags:
- synthetic
- pii
- privacy
- named-entity-recognition
- ner
- basque
- euskara
task_categories:
- token-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: pii-sintetikoa-v2-train.jsonl
  - split: eval
    path: pii-sintetikoa-v2-eval.jsonl
- config_name: heldout_test
  data_files:
  - split: test
    path: pii-sintetikoa-v2-test-heldout-templates.jsonl
---

# pii-synthetic-basque-v2

A fully synthetic dataset of sentences containing personally identifiable information (PII) in Basque, Spanish and French. It covers the whole Basque Country: Araba, Bizkaia, Gipuzkoa, Nafarroa and Iparralde (Pyrénées-Atlantiques). Realistic values (name frequencies, streets, towns, postcodes) come from public aggregate statistics and open map data, never from records of real individuals.

The dataset was built as *experience replay* data for fine-tuning multilingual PII detectors on Basque without losing their original PII types; see [tknika/gliner2-PII-basque-v2](https://huggingface.co/tknika/gliner2-PII-basque-v2). The intended use is educational: anonymizing student texts before sharing or analyzing them.

Generator, audit and evaluation code: **[github.com/tknika/gliner2-pii-basque](https://github.com/tknika/gliner2-pii-basque)**. That repository regenerates the train and eval files below byte for byte.

## Splits

| Config / split | File | Sentences | Templates | Use |
|---|---|---|---|---|
| `default` / `train` | `pii-sintetikoa-v2-train.jsonl` | 5,000 | 47 training templates | fine-tuning / replay |
| `default` / `eval` | `pii-sintetikoa-v2-eval.jsonl` | 1,000 | **same** 47 templates | in-template validation |
| `heldout_test` / `test` | `pii-sintetikoa-v2-test-heldout-templates.jsonl` | 1,000 | 30 **held-out** templates | **evaluation** |

**Use `heldout_test` to report results.** `train` and `eval` share their sentence templates: 85% of eval sentences have the same sentence frame as a training sentence. A model trained on `train` therefore scores much higher on `eval` than on new phrasing. For example, gliner2-PII-basque-v2 scores 0.974 on `eval` and 0.905 on `heldout_test`.

What the splits share:
- **train ↔ eval**: no sentence appears in both, and no person value (name, email, user name, personal URL) is shared. The PII-free sentences are split in disjoint halves. Streets, towns, organizations and numbers may repeat.
- **heldout_test** (added October 2026): built from 30 new templates written in school and LMS registers (teacher notes, grant forms, forum posts, private messages) and 24 new PII-free sentences. It shares no sentence frame, sentence or person value with train or eval. Given names were sampled with corrected sex weights (see Known issues), giving 216 female and 258 male name mentions. Each record also carries `meta` (`lang`, `territory`, `template`, `person_sex`) for sliced evaluation. Languages: 573 eu, 257 es, 170 fr.

## Format

```json
{"text": "Kaixo, Mikel Etxeberria naiz eta Amorebietan bizi naiz. Nire helbidea Santa Luzia auzoa 105 da, 48340 postakodearekin...",
 "entities": {"person name": ["Mikel Etxeberria"], "city": ["Amorebietan"], "address": ["Santa Luzia auzoa 105"], "zip code": ["48340"], "age": [], ...},
 "spans": [[7, 24, "person name"], [40, 49, "city"], [67, 87, "address"], [92, 97, "zip code"], ...]}
```

- `entities` maps each entity type to its mentions. A mention is an exact substring **in its surface form**, declension suffixes included: in «Amorebietan bizi naiz» the mention is `Amorebietan`. Each positive sentence also lists 2–4 absent types with empty lists, as type-level negatives.
- `spans` gives character offsets `[start, end, type]`. In `heldout_test` they are exact. In train and eval they point at the first occurrence of each mention.
- About 15% of the sentences in every split contain no PII.

**Types (17):** `person name`, `email`, `phone number`, `national id`, `bank account number`, `credit card number`, `date of birth`, `date`, `address`, `city`, `location`, `zip code`, `age`, `organization`, plus three education types from the Kaggle *PII Detection, Removal from Educational Data* taxonomy: `user name` (forum/LMS handles), `personal url` and `student id`. `location` is reserved for provinces and historical territories; towns are always `city`.

## Data sources

- **[INE](https://www.ine.es/)** (Spain): given-name and surname frequencies per province (757 given names, 890 surnames).
- **[INSEE](https://www.insee.fr/fr/statistiques/2540004)**: given-name frequencies for department 64 (600 names).
- **[EODA](https://euskaltzaindia.eus/index.php?option=com_ecoeoda)** (Euskaltzaindia): the pool of Basque name and surname candidates queried at INE.
- **[OpenStreetMap](https://www.openstreetmap.org/copyright)** (© OpenStreetMap contributors, ODbL): 17,029 streets in 824 towns, always combined with the town and postcode they really have.

Emails, phone numbers, national IDs (Spanish DNI with mod-23 letter, French NIR with mod-97 key), IBANs (mod-97), card numbers (Luhn) and dates are randomly generated with valid checksums. They are not taken from any real source, but a random value may coincide with a real one by chance.

Territories are sampled at Araba 5%, Bizkaia 25%, Gipuzkoa 25%, Nafarroa 20% and Iparralde 25%. Language is then drawn per territory: eu/es at 70/30 in the Basque Autonomous Community, 35/65 in Nafarroa, and eu/fr at 35/65 in Iparralde. Phone formats, IDs, IBANs and email domains follow the territory.

## Known issues

- **Sex imbalance of given names in train and eval.** A bug in the INE download gave all female given names zero province weights. As a result, southern-territory Basque given names in train and eval are almost always male, and about 79% of person names in train are male. The files are kept as published for reproducibility; `heldout_test` uses corrected weights. Fixed for future versions.
- **Limited template variety.** There are 47 training templates and 24 PII-free sentences, and some Basque organization suffixes are rule-based and not always grammatical. Models trained on this data can overfit to these sentence frames, which `heldout_test` is designed to expose.
- **Synthetic text.** Results on this dataset do not transfer one-to-one to real documents.

## Citation

```bibtex
@misc{tknika2026piisyntheticbasquev2,
  title  = {pii-synthetic-basque-v2: A Synthetic PII Dataset for Basque Experience Replay with Educational Entity Types},
  author = {{TKNIKA} and Ezpeleta Mendikute, Xabier},
  year   = {2026},
  url    = {https://huggingface.co/datasets/tknika/pii-synthetic-basque-v2}
}
```

The synthetic-data approach follows the GLiNER2-PII paper:

```bibtex
@misc{zaratiana2026gliner2piimultilingualmodelpersonally,
  title  = {GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction},
  author = {Zaratiana, Urchade and Lewis, Ash and Hurn-Maloney, George},
  year   = {2026},
  eprint = {2605.09973},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2605.09973}
}
```
