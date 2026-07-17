"""
Deterministic synthetic-data generators.

In --mock mode, 01_download_data.py calls these instead of hitting the network.
Each generator emits records shaped like the *real* source schema (nested
protocolSection for trials, flat dicts for USPTO, OpenAlex inverted-index quirks,
etc.) so that 02_parse_normalize.py exercises the same code paths it will use on
real data. All randomness is seeded -> the mock corpus is identical every run.
"""
from __future__ import annotations

import random
from datetime import date

# Small controlled vocab so mock docs cluster into believable therapeutic areas.
_AREAS = [
    ("oncology", ["EGFR inhibitor", "CAR-T", "PD-L1 antibody", "kinase", "tumor"]),
    ("immunology", ["monoclonal antibody", "cytokine", "T-cell", "autoimmune"]),
    ("neurology", ["amyloid", "neurodegeneration", "BBB penetrant", "tau"]),
    ("metabolic", ["GLP-1 agonist", "insulin", "lipid", "glucose transporter"]),
    ("antiviral", ["protease inhibitor", "RNA polymerase", "capsid", "vaccine"]),
]
_ORGS = ["Purdue University", "MIT", "Stanford", "Pfizer", "Merck", "Genentech",
         "Broad Institute", "UCSF", "Novartis", "Eli Lilly"]
_PEOPLE = ["A. Rivera", "S. Chen", "M. Okafor", "L. Novak", "P. Sharma",
           "J. Müller", "K. Tanaka", "R. Ahmed", "E. Rossi", "D. Park"]
_CPC = ["A61K39/00", "A61P35/00", "C07K16/28", "C12N15/113", "A61K31/00"]


def _area(rng):
    return rng.choice(_AREAS)


def gen_clinicaltrials(n: int, seed: int):
    """ClinicalTrials.gov v2 shape: deeply nested protocolSection."""
    rng = random.Random(seed + 1)
    for i in range(n):
        area, terms = _area(rng)
        t = rng.sample(terms, k=min(2, len(terms)))
        nct = f"NCT{rng.randint(10_000_000, 99_999_999)}"
        yield {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct,
                    "briefTitle": f"A study of {t[0]} in {area}",
                    "officialTitle": f"Phase 2 trial evaluating {t[0]} for {area} indications",
                },
                "descriptionModule": {
                    "briefSummary": f"This trial investigates {t[0]} and {t[-1]} "
                    f"as a therapeutic strategy in {area}.",
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": rng.choice(_ORGS)},
                },
                "contactsLocationsModule": {
                    "locations": [{"facility": f"{rng.choice(_ORGS)} Medical Center"}],
                },
            },
            "_source_url": f"https://clinicaltrials.gov/study/{nct}",
        }


def gen_uspto(n: int, seed: int):
    """USPTO / PatentsView shape: flat dict with claims + CPC."""
    rng = random.Random(seed + 2)
    for i in range(n):
        area, terms = _area(rng)
        t = rng.sample(terms, k=min(2, len(terms)))
        pid = f"US{rng.randint(8_000_000, 11_999_999)}B2"
        yield {
            "patent_id": pid,
            "patent_title": f"Compositions and methods for {t[0]} targeting {area}",
            "patent_abstract": f"Disclosed are {t[0]} compounds and methods of use "
            f"for treating {area} conditions, optionally with {t[-1]}.",
            "inventors": rng.sample(_PEOPLE, k=2),
            "assignees": [rng.choice(_ORGS)],
            "cpc_codes": rng.sample(_CPC, k=2),
            "pharma_keywords": t,
            "_source_url": f"https://patents.google.com/patent/{pid}",
        }


def gen_openalex(n: int, seed: int):
    """OpenAlex works shape, incl. the abstract_inverted_index quirk."""
    rng = random.Random(seed + 3)
    for i in range(n):
        area, terms = _area(rng)
        t = rng.sample(terms, k=min(2, len(terms)))
        wid = f"W{rng.randint(1_000_000_000, 4_999_999_999)}"
        abstract = (f"We report a {t[0]} approach for {area}. "
                    f"The {t[-1]} mechanism is characterized in detail.")
        # OpenAlex stores abstracts as an inverted index, not plain text.
        inv: dict[str, list[int]] = {}
        for pos, word in enumerate(abstract.split()):
            inv.setdefault(word, []).append(pos)
        yield {
            "id": f"https://openalex.org/{wid}",
            "title": f"{t[0].capitalize()} strategies in {area}: a study",
            "abstract_inverted_index": inv,
            "authorships": [
                {"author": {"display_name": p},
                 "institutions": [{"display_name": rng.choice(_ORGS)}]}
                for p in rng.sample(_PEOPLE, k=2)
            ],
            "concepts": [{"display_name": area.capitalize()}],
        }


def gen_sbir(n: int, seed: int):
    """SBIR.gov award shape: flat dict."""
    rng = random.Random(seed + 4)
    for i in range(n):
        area, terms = _area(rng)
        t = rng.sample(terms, k=min(2, len(terms)))
        aid = f"SBIR-{rng.randint(100000, 999999)}"
        yield {
            "award_id": aid,
            "award_title": f"Commercialization of {t[0]} for {area}",
            "abstract": f"This SBIR project develops {t[0]} technology addressing "
            f"unmet need in {area}.",
            "firm": rng.choice(_ORGS),
            "pi_name": rng.choice(_PEOPLE),
            "_source_url": f"https://www.sbir.gov/awards/{aid}",
        }


def gen_autm(n: int, seed: int):
    """AUTM Innovation Marketplace listing shape (as students scrape it)."""
    rng = random.Random(seed + 5)
    for i in range(n):
        area, terms = _area(rng)
        t = rng.sample(terms, k=min(2, len(terms)))
        lid = f"AUTM-{rng.randint(10000, 99999)}"
        yield {
            "listing_id": lid,
            "tech_title": f"{t[0].capitalize()} platform for {area}",
            "description": f"University-developed {t[0]} technology available for "
            f"licensing in the {area} space.",
            "university": rng.choice(_ORGS),
            "inventors": rng.sample(_PEOPLE, k=1),
            "_source_url": f"https://example-tto.org/tech/{lid}",
        }


GENERATORS = {
    "clinicaltrials": gen_clinicaltrials,
    "uspto": gen_uspto,
    "openalex": gen_openalex,
    "sbir": gen_sbir,
    "autm": gen_autm,
}
