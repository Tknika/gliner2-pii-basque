#!/usr/bin/env python3
"""Given-name and surname frequencies per province from INE (Spain).

Uses the public JSON endpoints behind INE's name/surname widgets:

  /tnombres/widget?nombre=X                national total + sex
  /tnombres/mapaWidget?nombre=X&sexo=S     per-province ‰ (S: 1 male, 6 female)
  /apellidos/widget?apellido=X             totals as 1st/2nd surname
  /apellidos/mapaWidget?apellido=X&tipo=1  per-province ‰ (1st surname)

Candidates: EODA given names (names_eod.json) and EODA surnames
(surnames_eod.json, from fetch_eoda_surnames.py) plus common Spanish
surnames. Kept if national total ≥ --min-names / --min-surnames.

    python scripts/fetch_sources/fetch_ine.py --names --surnames

Writes names_ine.csv / surnames_ine.csv into --out (default
data/sources/), caching raw responses in <out>/cache_ine/.

Note: the names_ine.csv used for the v2 dataset was produced by an
earlier version that queried every name with the male sex code, leaving
female rows with zero province weights; names_ine_fixed.csv was produced
with this version.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.ine.es"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0"}
DELAY = 0.35
PROVINCES = {"2": "araba", "21": "gipuzkoa", "48": "bizkaia", "32": "nafarroa"}
SEX = {"hombres": "male", "mujeres": "female"}
SEX_CODE = {"male": 1, "female": 6}

SPANISH_SURNAMES = [
    "GARCIA", "RODRIGUEZ", "GONZALEZ", "FERNANDEZ", "LOPEZ", "MARTINEZ",
    "SANCHEZ", "PEREZ", "GOMEZ", "MARTIN", "JIMENEZ", "RUIZ", "HERNANDEZ",
    "DIAZ", "MORENO", "ALVAREZ", "ROMERO", "ALONSO", "GUTIERREZ", "NAVARRO",
    "TORRES", "DOMINGUEZ", "VAZQUEZ", "RAMOS", "GIL", "RAMIREZ", "SERRANO",
    "BLANCO", "MOLINA", "MORALES", "SUAREZ", "ORTEGA", "DELGADO", "CASTRO",
    "ORTIZ", "RUBIO", "MARIN", "SANZ", "IGLESIAS", "MEDINA", "GARRIDO",
    "CORTES", "CASTILLO", "SANTOS", "LOZANO", "GUERRERO", "CANO", "PRIETO",
    "MENDEZ", "CRUZ", "CALVO", "GALLEGOS", "VIDAL", "LEON", "HERRERA",
    "MARQUEZ", "PEÑA", "FLORES", "CARRASCO", "AGUILAR", "VEGA", "FUENTES",
    "CAMPOS", "CABRERA", "NIETO", "REYES", "SALAS", "DIEZ", "PASCUAL",
    "SANTIAGO", "HERRERO", "MONTERO", "LORENZO", "MENDOZA", "GALLARDO",
    "ESTRADA", "SOLIS", "PARDO", "ROJAS", "PADILLA", "JUAREZ",
    "MARCOS", "CARO", "QUINTANA", "OSORIO", "SAEZ", "VILAR", "CORONADO",
]


def normalize(s):
    """INE format: uppercase, no diacritics."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def query(cache, path, params):
    """Cached GET; returns the response object or None."""
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    cached = cache / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
    else:
        data = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(request, timeout=20) as r:
                    data = json.loads(r.read().decode("utf-8", errors="ignore"))
                cached.write_text(json.dumps(data, ensure_ascii=False),
                                  encoding="utf-8")
                time.sleep(DELAY)
                break
            except Exception as e:  # network errors: retry with backoff
                print(f"  {url}: {e} (attempt {attempt + 1})", file=sys.stderr)
                time.sleep(3 + 5 * attempt)
    # widgets wrap the object in a one-item list; errors are plain strings
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    return data if isinstance(data, dict) else None


def provinces(response):
    if not response:
        return {}
    return {PROVINCES[str(r.get("id"))]: r.get("val")
            for r in response.get("regiones", [])
            if isinstance(r, dict) and str(r.get("id")) in PROVINCES}


def write_csv(path, rows, first_columns):
    columns = first_columns + list(PROVINCES.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="0")
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ {len(rows)} rows → {path}")


def fetch_names(out, cache, min_total):
    candidates = json.loads((out / "names_eod.json").read_text(encoding="utf-8"))
    candidates = sorted({normalize(c["name"]) for c in candidates
                         if normalize(c["name"]).isalpha()
                         and 2 <= len(c["name"]) <= 20})
    print(f"{len(candidates)} given-name candidates")
    rows = []
    for name in candidates:
        d = query(cache, "/tnombres/widget", {"nombre": name})
        if not d or d.get("Resultado") != "OK":
            continue
        for series in d.get("Series", []):
            total = series.get("Total") or 0
            sex = SEX.get(series.get("Sexo"))
            if total < min_total or sex is None:
                continue
            row = {"name": name, "sex": sex, "total": total}
            row.update(provinces(query(cache, "/tnombres/mapaWidget",
                                       {"nombre": name, "sexo": SEX_CODE[sex]})))
            rows.append(row)
    write_csv(out / "names_ine.csv", rows, ["name", "sex", "total"])


def fetch_surnames(out, cache, min_total):
    path = out / "surnames_eod.json"
    candidates = [normalize(x) for x in json.loads(path.read_text())] \
        if path.exists() else []
    candidates = sorted({c for c in candidates + SPANISH_SURNAMES
                         if c.isalpha() and 2 <= len(c) <= 25})
    print(f"{len(candidates)} surname candidates")
    rows = []
    for surname in candidates:
        d = query(cache, "/apellidos/widget", {"apellido": surname})
        if not d or d.get("Resultado") != "OK":
            continue
        total = (d.get("Series") or [{}])[0].get("Total1") or 0
        if total < min_total:
            continue
        row = {"surname": surname, "total": total}
        row.update(provinces(query(cache, "/apellidos/mapaWidget",
                                   {"apellido": surname, "tipo": 1})))
        rows.append(row)
    write_csv(out / "surnames_ine.csv", rows, ["surname", "total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--surnames", action="store_true")
    ap.add_argument("--min-names", type=float, default=300)
    ap.add_argument("--min-surnames", type=float, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/sources"))
    args = ap.parse_args()
    if not (args.names or args.surnames):
        ap.error("pass --names and/or --surnames")
    cache = args.out / "cache_ine"
    cache.mkdir(parents=True, exist_ok=True)
    if args.names:
        fetch_names(args.out, cache, args.min_names)
    if args.surnames:
        fetch_surnames(args.out, cache, args.min_surnames)


if __name__ == "__main__":
    main()
