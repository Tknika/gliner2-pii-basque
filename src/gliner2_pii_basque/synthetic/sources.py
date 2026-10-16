"""Real-world value sources for the synthetic generator.

All tables live in ``data/sources/`` (see its README for provenance and
licenses):

    names_ine.csv     name, sex, total, araba, gipuzkoa, bizkaia, nafarroa
    surnames_ine.csv  surname, total, araba, gipuzkoa, bizkaia, nafarroa
    names_fr_64.csv   name, sex, count
    streets.csv       street, town, postcode, territory, addresses, lat, lon
    towns.csv         town, territory, postcode, lat, lon

Province columns hold INE per-mille frequencies; they are used as
sampling weights per territory.
"""

import csv
from pathlib import Path

from .values import clean_town

SOURCES_DIR = Path(__file__).resolve().parents[3] / "data" / "sources"

PROVINCE_COLUMNS = ("araba", "gipuzkoa", "bizkaia", "nafarroa")

# Most common Spanish given names (INE national publications). Value =
# relative weight. Used for the share of Spanish names in the south.
SPANISH_NAMES = {
    "Antonio": 90, "Manuel": 60, "Jose": 80, "Francisco": 50,
    "David": 35, "Juan": 70, "Javier": 30, "Daniel": 28,
    "Carlos": 30, "Jesus": 40, "Alejandro": 25, "Miguel": 40,
    "Rafael": 25, "Pablo": 20, "Angel": 30, "Sergio": 20,
    "Pedro": 40, "Fernando": 25, "Jorge": 20, "Alberto": 20,
    "Luis": 35, "Alvaro": 18, "Adrian": 18, "Diego": 18,
    "Ivan": 12, "Ruben": 12, "Oscar": 13, "Raul": 15,
    "Maria": 90, "Carmen": 50, "Maria Carmen": 25, "Ana": 30,
    "Maria Pilar": 22, "Maria Dolores": 22, "Isabel": 25,
    "Laura": 25, "Sara": 18, "Lucia": 18, "Claudia": 13,
    "Patricia": 13, "Elena": 18, "Marta": 13, "Irene": 10,
    "Sandra": 11, "Paula": 10, "Cristina": 18, "Rocio": 13,
    "Pilar": 20, "Maria Angeles": 18, "Amparo": 12, "Beatriz": 12,
}
SPANISH_FEMALE_NAMES = {
    "Maria", "Carmen", "Maria Carmen", "Ana", "Maria Pilar", "Maria Dolores",
    "Isabel", "Laura", "Sara", "Lucia", "Claudia", "Patricia", "Elena",
    "Marta", "Irene", "Sandra", "Paula", "Cristina", "Rocio", "Pilar",
    "Maria Angeles", "Amparo", "Beatriz",
}

# Most common French surnames + Northern Basque surnames (French spelling)
FRENCH_SURNAMES = {
    "Martin": 100, "Bernard": 90, "Dubois": 80, "Thomas": 80,
    "Robert": 80, "Richard": 70, "Petit": 70, "Durand": 70,
    "Leroy": 65, "Moreau": 65, "Simon": 60, "Laurent": 60,
    "Lefebvre": 55, "Michel": 55, "Bertrand": 50,
    "Roux": 50, "Vincent": 50, "Fournier": 50, "Morel": 45,
    "Girard": 45, "Mercier": 45, "Dupont": 45, "Lambert": 45,
    "Bonnet": 40, "Francois": 40, "Martinez": 40, "Legrand": 40,
    "Faure": 40, "Rousseau": 40, "Blanc": 40, "Guerin": 35,
    "Muller": 35, "Henry": 35, "Roussel": 35, "Nicolas": 35,
    "Perrin": 35, "Morin": 35, "Mathieu": 30, "Clement": 30,
    "Gauthier": 30, "Dumont": 30, "Lopez": 30, "Fontaine": 30,
    "Chevalier": 30, "Robin": 30, "Masson": 30, "Marchand": 30,
    "Etcheverry": 25, "Etcheberry": 20, "Larrieu": 25, "Duhalde": 15,
    "Harispe": 12, "Iriart": 15, "Ithurbide": 12, "Bidegain": 12,
    "Chourrut": 8, "Heguy": 10, "Cazenave": 12, "Lissarrague": 8,
    "Uhalde": 10, "Etchart": 12, "Lacoste": 15, "Diharce": 8,
    "Bordenave": 12, "Jaurgain": 6, "Oyhamburu": 5, "Mendiboure": 8,
}


def _read_csv(folder, name):
    path = Path(folder) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; see data/sources/README.md to rebuild it")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value):
    try:
        return max(float(value or 0), 0.0)
    except ValueError:
        return 0.0


class Sources:
    """Loads every table once and prepares weighted choices.

    `names_file` selects the INE given-name table. The v2 dataset was
    generated with ``names_ine.csv``, whose female rows have zero province
    weights (see data/sources/README.md); ``names_ine_fixed.csv`` has
    them corrected.
    """

    def __init__(self, folder=SOURCES_DIR, names_file="names_ine.csv"):
        self.names = []           # (name, sex, {province: weight})
        for row in _read_csv(folder, names_file):
            weights = {p: _float(row.get(p)) for p in PROVINCE_COLUMNS}
            self.names.append((row["name"].title(), row["sex"], weights))

        self.surnames = []        # (surname, {province|total: weight})
        for row in _read_csv(folder, "surnames_ine.csv"):
            weights = {p: _float(row.get(p)) for p in PROVINCE_COLUMNS}
            weights["total"] = _float(row.get("total"))
            self.surnames.append((row["surname"].title(), weights))

        self.names_fr = [(row["name"], row["sex"], int(row["count"]))
                         for row in _read_csv(folder, "names_fr_64.csv")]

        self.streets = {}         # territory → [(street, town, postcode, weight)]
        for row in _read_csv(folder, "streets.csv"):
            self.streets.setdefault(row["territory"], []).append(
                (row["street"], clean_town(row["town"]), row["postcode"],
                 int(row["addresses"]) ** 0.5))

        towns = {}
        for row in _read_csv(folder, "towns.csv"):
            for territory in row["territory"].split("/"):
                towns.setdefault(territory, set()).add(clean_town(row["town"]))
        self.towns = {t: sorted(v) for t, v in towns.items()}

        self.spanish_names = list(SPANISH_NAMES.items())
        self.french_surnames = list(FRENCH_SURNAMES.items())

        # given name → sex, for per-gender evaluation
        self.name_sex = {n: s for n, s, _ in self.names}
        self.name_sex.update({n: s for n, s, _ in self.names_fr})
        self.name_sex.update({
            n: "female" if n in SPANISH_FEMALE_NAMES else "male"
            for n in SPANISH_NAMES})
