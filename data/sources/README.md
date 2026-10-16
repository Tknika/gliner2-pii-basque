# Source tables

Aggregate, public data used to make the synthetic sentences realistic.
No table holds records of real individuals: given names and surnames are
sampled independently from marginal frequency distributions.

| File | Content | Source | License / terms |
|---|---|---|---|
| `names_ine.csv` | 757 given names: sex, national total, ‰ per province (Araba, Bizkaia, Gipuzkoa, Nafarroa) | [INE](https://www.ine.es/) name widget API | INE public statistics (reuse with attribution) |
| `names_ine_fixed.csv` | Same, with female province weights re-fetched (2026-10-16) | INE | as above |
| `surnames_ine.csv` | 890 surnames: total as first surname, ‰ per province | INE surname widget API | as above |
| `names_fr_64.csv` | 600 given names (top 300 per sex) for department 64 | [INSEE Fichier des prénoms](https://www.insee.fr/fr/statistiques/2540004) | Licence Ouverte / Etalab 2.0 |
| `streets.csv` | 17,029 (street, town, postcode) triples with address counts | [OpenStreetMap](https://www.openstreetmap.org/copyright) via Geofabrik | **ODbL 1.0**, © OpenStreetMap contributors |
| `towns.csv` | 824 towns with territory and main postcode | OpenStreetMap | **ODbL 1.0**, © OpenStreetMap contributors |
| `names_eod.json` | ~5,000 Basque given names (candidate pool for the INE queries) | [EODA, Euskaltzaindia](https://www.euskaltzaindia.eus/index.php?option=com_ecoeoda) | Euskaltzaindia terms of use |

`streets.csv` and `towns.csv` are derivative databases of OpenStreetMap
and are distributed under the ODbL, not under the repository's Apache-2.0
license.

## Known issue: female name weights in `names_ine.csv`

The INE province endpoint takes `sexo=1` for men and `sexo=6` for women.
The script that built `names_ine.csv` sent the male code for every name,
so all 338 female rows got zero province weights. As a result, in the
v2 dataset southern-territory Basque given names are almost always male
(train set: 1,671 male vs. 153 female INE given names; overall ~79% of
person names are male).

`names_ine.csv` is kept unchanged because it reproduces the published
v2 dataset exactly. `names_ine_fixed.csv` (331/338 female names now with
weights) is used for the held-out test set and should be used for any
future training data. `scripts/fetch_sources/fetch_ine.py` has the fix.

## Rebuilding

```bash
uv pip install -e ".[sources]"
python scripts/fetch_sources/fetch_insee.py
python scripts/fetch_sources/fetch_osm.py --download --extract   # ~380 MB download
python scripts/fetch_sources/fetch_eoda_surnames.py              # slow, polite crawl
python scripts/fetch_sources/fetch_ine.py --names --surnames
```

`names_eod.json` was exported from EODA by hand; there is no script for
it. Upstream data changes over time, so rebuilt tables will not be
byte-identical to the committed ones.
