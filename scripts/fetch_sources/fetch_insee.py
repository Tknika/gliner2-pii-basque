#!/usr/bin/env python3
"""Given-name frequencies for department 64 from INSEE's Fichier des prénoms.

Department 64 (Pyrénées-Atlantiques) covers Lapurdi, Nafarroa Beherea
and Zuberoa (plus Béarn). Counts are summed over birth years 1900–2021;
rare-name rows (year "XXXX", names starting with "_") are skipped.

    python scripts/fetch_sources/fetch_insee.py [--top 300]

Writes names_fr_64.csv (name, sex, count) into --out.
"""

import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

URL = "https://www.insee.fr/fr/statistiques/fichier/2540004/dpt2021_csv.zip"
DEPARTMENT = "64"
SEX = {"1": "male", "2": "female"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300, help="names kept per sex")
    ap.add_argument("--out", type=Path, default=Path("data/sources"))
    args = ap.parse_args()

    response = requests.get(URL, timeout=120,
                            headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    totals = defaultdict(int)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"),
                                      delimiter=";"):
                given = row.get("preusuel", "").strip()
                if (row["dpt"] != DEPARTMENT or not row.get("annais", "").isdigit()
                        or not given or given.startswith("_")):
                    continue
                try:
                    totals[(given, row["sexe"])] += int(row.get("nombre", 0))
                except ValueError:
                    continue

    by_sex = defaultdict(list)
    for (given, sex), count in totals.items():
        by_sex[sex].append((given, count))
    rows = [{"name": given.title(), "sex": SEX.get(sex, sex), "count": count}
            for sex, names in by_sex.items()
            for given, count in sorted(names, key=lambda x: -x[1])[:args.top]]

    path = args.out / "names_fr_64.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sex", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
