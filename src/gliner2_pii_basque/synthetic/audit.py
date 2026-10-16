"""Quality audit of synthetic PII records.

Checks every mention: checksums (DNI mod-23, NIR/IBAN mod-97, Luhn),
formats (phones, emails, handles, URLs, student IDs, postcodes), that
spans point at the mention, and reports type/name/email distributions.
"""

import re
from collections import Counter

from .values import DNI_LETTERS

PHONE_PATTERNS = [
    re.compile(r"^(\+34 |0034 )?[679]\d{2} \d{3} \d{3}$"),
    re.compile(r"^(\+33 |0033 )?[567] \d{2} \d{2} \d{2} \d{2}$"),
    re.compile(r"^0[567] \d{2} \d{2} \d{2} \d{2}$"),
]
EMAIL = re.compile(r"^[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}$")
USERNAME = re.compile(r"^@?[a-z0-9._-]+$")
PERSONAL_URL = re.compile(
    r"^(linkedin\.com/in/|twitter\.com/|instagram\.com/|github\.com/)[a-z0-9._-]+$"
    r"|^[a-z0-9.-]+\.(wordpress|blogspot)\.com$|^[a-z0-9-]+\.eu$")


def dni_valid(value):
    m = re.fullmatch(r"(\d{8})([A-Z])", value)
    return bool(m) and DNI_LETTERS[int(m.group(1)) % 23] == m.group(2)


def nir_valid(value):
    m = re.fullmatch(r"([12]\d{12}) (\d{2})", value)
    return bool(m) and 97 - int(m.group(1)) % 97 == int(m.group(2))


def iban_valid(value):
    if len(value) < 15 or not value[:2].isalpha() or not value[2:4].isdigit():
        return False
    digits = "".join(str(int(c, 36)) for c in (value[4:] + value[:4]).upper())
    return int(digits) % 97 == 1


def luhn_valid(value):
    digits = value.replace(" ", "")
    if not digits.isdigit() or len(digits) != 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        d = int(d) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


CHECKS = {
    "national id": lambda m: nir_valid(m) if " " in m else dni_valid(m),
    "bank account number": iban_valid,
    "credit card number": luhn_valid,
    "phone number": lambda m: any(p.match(m) for p in PHONE_PATTERNS),
    "email": lambda m: bool(EMAIL.match(m)),
    "user name": lambda m: bool(USERNAME.match(m)),
    "personal url": lambda m: bool(PERSONAL_URL.match(m)),
    "student id": lambda m: bool(re.fullmatch(r"\d{6}[A-E]", m)),
    "zip code": lambda m: bool(re.fullmatch(r"\d{5}", m)),
}


def audit(records):
    """Return (errors Counter, report lines)."""
    errors = Counter()
    types = Counter()
    names = Counter()
    domains = Counter()
    negatives = 0
    for r in records:
        text = r["text"]
        if not any(r["entities"].values()):
            negatives += 1
        for etype, mentions in r["entities"].items():
            types[etype] += len(mentions)
            for m in mentions:
                if m not in text:
                    errors[f"{etype}: mention not in text"] += 1
                check = CHECKS.get(etype)
                if check and not check(m):
                    errors[f"{etype}: invalid value"] += 1
                if etype == "person name":
                    names[m] += 1
                elif etype == "email":
                    domains[m.split("@")[-1]] += 1
        for start, end, etype in r.get("spans", []):
            if text[start:end] not in r["entities"].get(etype, []):
                errors[f"{etype}: span does not match a mention"] += 1

    report = [f"records: {len(records)} ({negatives} negatives, "
              f"{100 * negatives / len(records):.0f}%)", "mentions per type:"]
    report += [f"  {t:22s} {c:6d}" for t, c in types.most_common()]
    report.append("top email domains:")
    report += [f"  {d:22s} {c:6d}" for d, c in domains.most_common(8)]
    once = sum(1 for c in names.values() if c == 1)
    report.append(f"person names: {len(names)} distinct, {once} appear once")
    return errors, report
