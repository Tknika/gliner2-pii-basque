"""Synthetic PII sentence generator (Basque Country: eu / es / fr).

Each example samples a territory, a language for that territory, a
person context (name, address, contacts, identifiers...) from real
aggregate sources, and fills a template. ~15% of the sentences are
PII-free negatives, and every positive lists 2–4 absent entity types
with empty lists as type-level hard negatives.

Records are dicts: ``{"text", "entities": {type: [mention, ...]},
"spans": [[start, end, type], ...]}`` (+ ``meta`` for the held-out test).

Determinism: the v2 splits are reproduced exactly from
``Sources(names_file="names_ine.csv")`` with seeds 42 (train) / 2025
(eval). Keep the order of random calls unchanged.
"""

import json
import random
import string

from . import values as v
from .sources import Sources
from .templates import (ENTITY_TYPES, HELDOUT_NEGATIVES, HELDOUT_TEMPLATES,
                        SLOT_TYPES, TRAIN_NEGATIVES, TRAIN_TEMPLATES)

TERRITORIES = [("Araba", 0.05), ("Bizkaia", 0.25), ("Gipuzkoa", 0.25),
               ("Nafarroa", 0.20), ("Iparralde", 0.25)]
LANGUAGES = {
    "Araba": [("eu", 0.70), ("es", 0.30)],
    "Bizkaia": [("eu", 0.70), ("es", 0.30)],
    "Gipuzkoa": [("eu", 0.70), ("es", 0.30)],
    "Nafarroa": [("eu", 0.35), ("es", 0.65)],
    "Iparralde": [("eu", 0.35), ("fr", 0.65)],
}
PROVINCE_COLUMN = {"Araba": "araba", "Bizkaia": "bizkaia",
                   "Gipuzkoa": "gipuzkoa", "Nafarroa": "nafarroa"}
NEGATIVE_RATE = 0.15

# Values that identify a person: never shared between splits
PERSON_TYPES = ("person name", "email", "user name", "personal url")


def _weighted(pairs):
    return random.choices([x for x, _ in pairs], weights=[w for _, w in pairs])[0]


# ── Person context ───────────────────────────────────────────────────

def _name_and_surname(sources, territory, lang):
    if territory == "Iparralde":
        name = random.choices([n for n, _, _ in sources.names_fr],
                              weights=[c for _, _, c in sources.names_fr])[0]
        if random.random() < 0.5:
            surname = _weighted(sources.french_surnames)
        else:
            surname = random.choices(
                [s for s, _ in sources.surnames],
                weights=[w["total"] for _, w in sources.surnames])[0]
        return name, surname
    column = PROVINCE_COLUMN[territory]
    spanish_share = 0.15 if lang == "eu" else 0.55
    if random.random() < spanish_share:
        name = _weighted(sources.spanish_names)
    else:
        name = random.choices([n for n, _, _ in sources.names],
                              weights=[w[column] for _, _, w in sources.names])[0]
    surname = random.choices([s for s, _ in sources.surnames],
                             weights=[w[column] for _, w in sources.surnames])[0]
    return name, surname


def person_context(sources, territory, lang):
    """All slot values for one sentence, coherent with territory/language."""
    name, surname = _name_and_surname(sources, territory, lang)
    north = territory == "Iparralde"
    street, town, postcode, _ = random.choices(
        sources.streets[territory],
        weights=[w for *_, w in sources.streets[territory]])[0]
    number = random.randint(1, 120)
    user = None
    ctx = {
        "given_name": name,
        "name": f"{name} {surname}",
        "street": street, "town": town, "postcode": postcode,
        "town_in": v.inessive(town), "town_ko": v.local_genitive(town),
        "email": v.email(name, surname, territory),
        "phone": v.phone_fr() if north else v.phone_es(),
        "phone2": v.phone_fr(False) if north else v.phone_es(False),
        "iban": v.iban_fr() if north else v.iban_es(),
        "card": v.credit_card(),
        "birth_date": v.birth_date(lang),
        "date": v.recent_date(lang),
        "age": random.randint(18, 85),
        "organization": random.choice(v.ORGANIZATIONS[territory]),
        "dni": v.dni(),
        "nir": v.nir(),
        "username": (user := v.username(name, surname)),
        "username_at": "@" + user,
        "personal_url": v.personal_url(name, surname),
        "student_id": v.student_id(),
    }
    if north:
        ctx["dni"] = ctx["nir"]           # national id in Iparralde = NIR
    ctx["address"] = (f"{number} {street}" if lang == "fr"
                      else f"{street} {number}")
    ctx["address_loc"] = ctx["address"]
    ctx["age_text"] = f"{ctx['age']} " + {"fr": "ans", "es": "años",
                                          "eu": "urte"}[lang]
    org = ctx["organization"]
    ctx["organization_k"] = org + ("k" if org[-1].lower() in "aeiou" else "ek")
    ctx["organization_gen"] = org + ("en" if org[-1].lower() == "r" else "ren")
    ctx["location"] = random.choice(v.PROVINCES)
    ctx["location_in"] = v.inessive(ctx["location"])
    return ctx


# ── Examples ─────────────────────────────────────────────────────────

def _sample_territory_and_language():
    territory = _weighted(TERRITORIES)
    return territory, _weighted(LANGUAGES[territory])


def _add_type_negatives(entities):
    """List 2–4 absent types with empty lists (type-level negatives)."""
    missing = [t for t in ENTITY_TYPES if t not in entities]
    for t in random.sample(missing, min(len(missing), random.randint(2, 4))):
        entities[t] = []


def _train_example(sources, negatives):
    """One example from the training templates → (lang, text, entities)."""
    territory, lang = _sample_territory_and_language()
    if negatives.get(lang) and random.random() < NEGATIVE_RATE:
        return lang, random.choice(negatives[lang]), {t: [] for t in ENTITY_TYPES}
    ctx = person_context(sources, territory, lang)
    template, slots = random.choice(TRAIN_TEMPLATES[lang])
    text = template.format(**ctx)
    entities = {}
    for slot in slots:
        entities.setdefault(SLOT_TYPES[slot], []).append(ctx[slot])
    _add_type_negatives(entities)
    for mentions in entities.values():
        for m in mentions:
            assert m in text, f"{m!r} not found in {text!r}"
    return lang, text, entities


def _first_occurrence_spans(text, entities):
    """Character spans for the v2 files (first occurrence of each mention)."""
    spans = []
    for etype, mentions in entities.items():
        for m in mentions:
            start = text.find(m)
            if start >= 0:
                spans.append([start, start + len(m), etype])
    return sorted(spans)


def _person_values(record):
    return {m for t in PERSON_TYPES for m in record["entities"].get(t, [])}


def generate_v2_splits(sources=None, n_train=5000, n_eval=1000,
                       seed_train=42, seed_eval=2025):
    """The published v2 train/eval splits.

    Disjointness: no identical text across splits, person values (name,
    email, user name, personal url) never shared, and the negative
    sentences split in halves (even index → eval, odd → train). Streets,
    towns, organizations and numbers may repeat across splits.

    Note: both splits use the same TRAIN_TEMPLATES, so `eval` measures
    in-template generalization only; use the held-out test for that.
    """
    sources = sources or Sources()
    neg_eval = {l: s[0::2] for l, s in TRAIN_NEGATIVES.items()}
    neg_train = {l: s[1::2] for l, s in TRAIN_NEGATIVES.items()}

    def generate(n, negatives, excluded=frozenset()):
        records = []
        while len(records) < n:
            _, text, entities = _train_example(sources, negatives)
            record = {"text": text, "entities": entities,
                      "spans": _first_occurrence_spans(text, entities)}
            if text in excluded or _person_values(record) & excluded:
                continue
            records.append(record)
        return records

    random.seed(seed_eval)
    evals = generate(n_eval, neg_eval)
    excluded = {r["text"] for r in evals}.union(*map(_person_values, evals))
    random.seed(seed_train)
    trains = generate(n_train, neg_train, frozenset(excluded))
    return trains, evals


def _fill(template, ctx, slots):
    """Format a template and return exact spans of the slot fields."""
    parts, spans = [], []
    position = 0
    for literal, field, _, _ in string.Formatter().parse(template):
        parts.append(literal)
        position += len(literal)
        if field is None:
            continue
        value = str(ctx[field])
        if field in slots:
            spans.append([position, position + len(value), SLOT_TYPES[field]])
        parts.append(value)
        position += len(value)
    return "".join(parts), spans


def generate_heldout_test(sources, n=1000, seed=2026, excluded=frozenset()):
    """Test set built only from HELDOUT_TEMPLATES / HELDOUT_NEGATIVES.

    `excluded` holds texts and person values from the training and
    evaluation splits; any sentence reusing one is resampled. Each record
    carries `meta` (language, territory, template id, sex of the given
    name) for sliced evaluation.
    """
    random.seed(seed)
    records = []
    while len(records) < n:
        territory, lang = _sample_territory_and_language()
        meta = {"lang": lang, "territory": territory}
        if random.random() < NEGATIVE_RATE:
            index = random.randrange(len(HELDOUT_NEGATIVES[lang]))
            text, spans = HELDOUT_NEGATIVES[lang][index], []
            meta["template"] = f"{lang}-neg-{index}"
        else:
            ctx = person_context(sources, territory, lang)
            index = random.randrange(len(HELDOUT_TEMPLATES[lang]))
            template, slots = HELDOUT_TEMPLATES[lang][index]
            text, spans = _fill(template, ctx, slots)
            meta["template"] = f"{lang}-{index}"
            if "name" in slots:
                meta["person_sex"] = sources.name_sex.get(ctx["given_name"],
                                                          "unknown")
        entities = {}
        for start, end, etype in spans:
            entities.setdefault(etype, []).append(text[start:end])
        if entities:
            _add_type_negatives(entities)
        else:
            entities = {t: [] for t in ENTITY_TYPES}
        record = {"text": text, "entities": entities, "spans": sorted(spans),
                  "meta": meta}
        if spans and (text in excluded or _person_values(record) & excluded):
            continue
        records.append(record)
    return records


# ── I/O ──────────────────────────────────────────────────────────────

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
