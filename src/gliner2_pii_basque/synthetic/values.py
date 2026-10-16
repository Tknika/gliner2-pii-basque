"""Value generators (checksum-valid identifiers, contacts, dates) and
Basque declension helpers.

Every function draws from the global `random` state, so the call order
defines the generated dataset; do not reorder calls in the generator.
"""

import random
import unicodedata

# ── Lexicons ─────────────────────────────────────────────────────────

# Well-known public organizations per territory
ORGANIZATIONS = {
    "Araba": ["BBVA", "Kutxabank", "Iberdrola", "CAF", "Euskaltel",
              "El Correo", "EITB", "Mondragon", "Osakidetza", "Lanbide"],
    "Bizkaia": ["BBVA", "Kutxabank", "Iberdrola", "Euskaltel",
                "Bilboko Portua", "El Correo", "EITB", "Petronor",
                "Mondragon", "Osakidetza"],
    "Gipuzkoa": ["Kutxabank", "CAF", "Euskaltel", "Orona", "Mondragon",
                 "Siemens Gamesa", "EITB", "Noticias de Gipuzkoa",
                 "Osakidetza"],
    "Nafarroa": ["Gobierno de Navarra", "CENER", "Euskaltel", "Iberdrola",
                 "Diario de Navarra", "Osasuna", "Universidad de Navarra",
                 "MTorres"],
    "Iparralde": ["Credit Agricole", "La Poste", "EDF", "Carrefour",
                  "Banque Populaire", "Bayonne Olympique",
                  "Chambre d'Agriculture", "Suez"],
}

_SOUTH_DOMAINS = [("gmail.com", 40), ("hotmail.com", 18), ("outlook.com", 8),
                  ("euskaltel.eus", 12), ("yahoo.es", 4), ("ehu.eus", 4),
                  ("empresa.eus", 3), ("telefonica.net", 3), ("terra.es", 2),
                  ("mixmail.com", 2)]
EMAIL_DOMAINS = {
    "Araba": _SOUTH_DOMAINS,
    "Bizkaia": _SOUTH_DOMAINS,
    "Gipuzkoa": _SOUTH_DOMAINS,
    "Nafarroa": [("gmail.com", 42), ("hotmail.com", 20), ("outlook.com", 9),
                 ("euskaltel.eus", 6), ("yahoo.es", 5), ("unavarra.es", 2),
                 ("telefonica.net", 3), ("terra.es", 2), ("mixmail.com", 2)],
    "Iparralde": [("gmail.com", 28), ("orange.fr", 15), ("free.fr", 12),
                  ("sfr.fr", 8), ("laposte.net", 6), ("wanadoo.fr", 5),
                  ("yahoo.fr", 6), ("hotmail.com", 10), ("outlook.com", 5),
                  ("ikasle.eus", 3)],
}

EMAIL_PATTERNS = [
    ("name.surname", 25), ("n.surname", 15), ("namesurname", 12),
    ("name_surname", 8), ("name-surname", 5), ("surname.name", 8),
    ("nameN", 10), ("surnameN", 5), ("name", 5), ("initials", 7),
]

# `location` is reserved for provinces / historical territories; towns
# are always `city`. Keeping them lexically distinct makes the types
# learnable.
PROVINCES = ("Araba", "Bizkaia", "Gipuzkoa", "Nafarroa",
             "Lapurdi", "Nafarroa Beherea", "Zuberoa")

MONTHS = {
    "eu": ["urtarrila", "otsaila", "martxoa", "apirila", "maiatza", "ekaina",
           "uztaila", "abuztua", "iraila", "urria", "azaroa", "abendua"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"],
}

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


# ── Identifiers (always checksum-valid) ──────────────────────────────

def dni():
    """Spanish DNI: 8 digits + mod-23 control letter."""
    n = random.randint(10000000, 99999999)
    return f"{n}{DNI_LETTERS[n % 23]}"


def nir():
    """French NIR (department 64): 13 digits + mod-97 key."""
    body = (f"{random.choice('12')}{random.randint(40, 99):02d}"
            f"{random.randint(1, 12):02d}64{random.randint(100, 999):03d}"
            f"{random.randint(1, 999):03d}")
    return f"{body} {97 - int(body) % 97:02d}"


def _iban(country, bban):
    """Add ISO 13616 check digits to a BBAN."""
    digits = "".join(str(int(c, 36)) for c in (bban + country + "00").upper())
    return f"{country}{98 - int(digits) % 97:02d}{bban}"


def iban_es():
    bank = random.choice(["2100", "2095", "0049", "0075", "0182", "0128"])
    return _iban("ES", f"{bank}{random.randint(0, 9999):04d}"
                       f"{random.randint(10, 99):02d}"
                       f"{random.randint(0, 99999999):010d}")


def iban_fr():
    # bank(5) branch(5) account(11) + RIB key(2)
    body = (f"{random.choice(['30004', '20041', '10278', '14706', '10057'])}"
            f"{random.randint(0, 99999):05d}"
            f"{random.randint(0, 99999999999):011d}")
    return _iban("FR", f"{body}{97 - int(body) % 97:02d}")


def credit_card():
    """16 digits, Visa/Mastercard-like prefix, Luhn-valid."""
    digits = [random.choice([4, 5])] + [random.randint(0, 9) for _ in range(14)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        # the check digit is appended afterwards, so parity is flipped
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    digits.append((10 - total % 10) % 10)
    s = "".join(map(str, digits))
    return " ".join(s[i:i + 4] for i in range(0, 16, 4))


def student_id():
    """School/university student number (distinct from national IDs)."""
    return f"{random.randint(100000, 999999)}{random.choice('ABCDE')}"


# ── Contact details ──────────────────────────────────────────────────

def phone_es(mobile=True):
    """6XX/7XX mobile or 9XX landline, optionally +34 / 0034."""
    first = random.choice("67") if mobile else "9"
    s = first + str(random.randint(0, 2)) + "".join(
        str(random.randint(0, 9)) for _ in range(7))
    grouped = f"{s[:3]} {s[3:6]} {s[6:]}"
    r = random.random()
    if r < 0.55:
        return grouped
    if r < 0.80:
        return f"+34 {grouped}"
    return f"0034 {grouped}"


def phone_fr(mobile=True):
    """06/07 mobile or 05 59 / 05 40 landline, optionally +33 / 0033."""
    if mobile:
        local = f"{random.choice(['06', '07'])} " + " ".join(
            f"{random.randint(0, 99):02d}" for _ in range(4))
        intl = "+33 " + local[1:]
    else:
        local = random.choice(["05 59", "05 40"]) + " " + " ".join(
            f"{random.randint(0, 99):02d}" for _ in range(3))
        intl = "+33 5 " + local[3:]
    r = random.random()
    if r < 0.55:
        return local
    if r < 0.80:
        return intl
    return "0033 " + local[1:]


def ascii_lower(s):
    """Strip diacritics and lowercase (María → maria)."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def email(name, surname, territory):
    pattern = random.choices([p for p, _ in EMAIL_PATTERNS],
                             weights=[w for _, w in EMAIL_PATTERNS])[0]
    na = ascii_lower(name.split()[0])     # compound name → first part
    su = ascii_lower(surname)
    year = str(random.randint(60, 99))
    n = f"{random.randint(1, 99)}"
    local = {
        "name.surname": f"{na}.{su}", "n.surname": f"{na[0]}.{su}",
        "namesurname": f"{na}{su}", "name_surname": f"{na}_{su}",
        "name-surname": f"{na}-{su}", "surname.name": f"{su}.{na}",
        "nameN": f"{na}{n}", "surnameN": f"{su}{n}", "name": na,
        "initials": f"{na[0]}{su[0]}",
    }[pattern]
    if pattern in ("nameN", "surnameN") and random.random() < 0.5:
        local = local[:-len(n)] + year
    domains = EMAIL_DOMAINS[territory]
    domain = random.choices([d for d, _ in domains],
                            weights=[w for _, w in domains])[0]
    return f"{local}@{domain}"


def username(name, surname):
    """Forum/LMS handle: mikel_etxeberria, mikel23, metxeberria..."""
    na = ascii_lower(name.split()[0])
    su = ascii_lower(surname)
    return random.choice([
        f"{na}_{su}", f"{na}{su}", f"{na}.{su}", f"{su}.{na}",
        f"{na}{random.randint(1, 99)}", f"{na[0]}{su}",
        f"{na}_{random.randint(70, 99)}", f"{na}{random.randint(1960, 2005)}",
    ])


def personal_url(name, surname):
    """Blog or social profile URL."""
    na = ascii_lower(name.split()[0])
    su = ascii_lower(surname)
    return random.choice([
        f"linkedin.com/in/{na}-{su}", f"twitter.com/{na}{su[0]}",
        f"{na}{su}.wordpress.com", f"instagram.com/{na}.{su}",
        f"{na}{su}.blogspot.com", f"github.com/{na}{su[0]}",
        f"{na}-{su}.eu",
    ])


# ── Dates ────────────────────────────────────────────────────────────

def _format_date(lang, year, month, day):
    if lang == "eu":
        # martxoa → martxoaren
        month_name = MONTHS["eu"][month - 1]
        month_name = (month_name[:-1] + "aren" if month_name.endswith("a")
                      else month_name + "aren")
        return f"{year}eko {month_name} {day}an"
    if lang == "es":
        return f"{day} de {MONTHS['es'][month - 1]} de {year}"
    return f"{day} {MONTHS['fr'][month - 1]} {year}"


def birth_date(lang):
    year, month, day = (random.randint(1945, 2005), random.randint(1, 12),
                        random.randint(1, 28))
    return _format_date(lang, year, month, day)


def recent_date(lang):
    year, month, day = (random.randint(2024, 2026), random.randint(1, 12),
                        random.randint(1, 28))
    return _format_date(lang, year, month, day)


# ── Basque declension ────────────────────────────────────────────────

def clean_town(town):
    """'Barañáin / Barañain' → 'Barañáin' (keep the first official form)."""
    return town.split(" / ")[0].strip()


def inessive(place):
    """Donostia → Donostian, Gasteiz → Gasteizen."""
    return place + ("n" if place[-1].lower() in "aeiou" else "en")


def local_genitive(place):
    """Bilbo → Bilboko, Donostia → Donostiako, Oiartzun → Oiartzungo."""
    exceptions = {"Iruñea": "Iruñeko", "Iruñe": "Iruñeko"}
    if place in exceptions:
        return exceptions[place]
    if place[-1].lower() == "a":
        return place[:-1] + "ako"
    if place[-1].lower() == "n":
        return place + "go"
    return place + "ko"
