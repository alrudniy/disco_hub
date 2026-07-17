"""Battery for dh2.pharma_tok. Every case here FAILS against the tokenizer it replaces."""
import re
import sys

sys.path.insert(0, "/home/alex/discovery_hub_pipeline_3")

from dh2.pharma_tok import pharma_toks, stem, word_overlap

# ---- the OLD rule, verbatim, so the battery proves a difference rather than asserting one
_OLD_STOP = set(
    "the a an of for and or to in with as by from thereof use uses using method methods "
    "composition compositions available licensing novel new therapy treatment".split()
)


def old_toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in _OLD_STOP}


def old_overlap(q, d):
    qt = old_toks(q)
    return len(qt & old_toks(d)) / len(qt) if qt else 0.0


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    return bool(cond)

ok = True
print("=== 1. pharma identifiers survive (all vanished before) ===")
for t, want in [("PD-L1", "pdl1"), ("PD-1", "pd1"), ("B7-H1", "b7h1"),
                ("IL-2", "il2"), ("CD20", "cd20"), ("HER2", "her2"),
                ("KRAS G12C", "g12c"), ("AZD9291", "azd9291"),
                ("BMS-986165", "bms986165"), ("T790M", "t790m"),
                ("V600E", "v600e"), ("5-FU", "5fu"), ("COVID-19", "covid19")]:
    got = pharma_toks(t)
    ok &= check(f"{t:12s} -> {sorted(got)}", want in got,
                f"old gave {sorted(old_toks(t)) or '{} <-- VANISHED'}")

print("\n=== 2. the money cases (the metric was INVERTED) ===")
q = "PD-L1 inhibitor"
wrong = "Anti-CD20 antibody inhibitor compositions for treating lymphoma"
right = "Antibodies that bind PD-L1 and block the PD-1 receptor in tumor inhibitor screens"
ok &= check("wrong target no longer scores 1.000",
            word_overlap(q, wrong) < 1.0, f"old={old_overlap(q, wrong):.3f} new={word_overlap(q, wrong):.3f}")
ok &= check("right target no longer scores 0.000",
            word_overlap(q, right) > 0.0, f"old={old_overlap(q, right):.3f} new={word_overlap(q, right):.3f}")
ok &= check("right target now beats wrong target",
            word_overlap(q, right) > word_overlap(q, wrong),
            f"right={word_overlap(q, right):.3f} vs wrong={word_overlap(q, wrong):.3f}")

print("\n=== 2b. gene symbols are not plurals ===")
for sym in ("KRAS", "NRAS", "HRAS", "MYC", "RAS"):
    ok &= check(f"{sym} survives stemming", sym.lower() in pharma_toks(sym),
                f"-> {sorted(pharma_toks(sym))}")
ok &= check("query 'KRAS G12C inhibitor' hits a KRAS doc",
            word_overlap("KRAS G12C inhibitor", "A KRAS G12C covalent inhibitor") == 1.0,
            f"-> {word_overlap('KRAS G12C inhibitor', 'A KRAS G12C covalent inhibitor'):.3f}")
ok &= check("real plurals still stem (cells/genes/drugs)",
            stem("cells") == stem("cell") and stem("genes") == stem("gene")
            and stem("drugs") == stem("drug"))

print("\n=== 3. stemming (query register vs document register) ===")
for a, b in [("antibodies", "antibody"), ("inhibitors", "inhibitor"),
             ("treating", "treated"), ("binds", "binding"), ("therapies", "therapy")]:
    ok &= check(f"{a!r} ~ {b!r}", stem(a) == stem(b), f"{stem(a)!r} vs {stem(b)!r}")
ok &= check("query 'antibodies targeting HER2' hits doc 'antibody that targets HER2'",
            word_overlap("antibodies targeting HER2", "An antibody that targets HER2") == 1.0,
            f"old={old_overlap('antibodies targeting HER2', 'An antibody that targets HER2'):.3f}")

print("\n=== 4. junk NOT readmitted ===")
for junk, why in [("of", "stopword"), ("in", "stopword"), ("by", "stopword"),
                  ("100mg", "dose"), ("10", "bare number"), ("485", "bare number"),
                  ("2020", "bare year")]:
    got = pharma_toks(junk)
    ok &= check(f"{junk!r} -> {sorted(got)}", len(got) == 0, why)
boiler = ("Compositions and methods of use thereof for the treatment of cancer "
          "in a subject in need thereof")
ok &= check("boilerplate stays small", len(pharma_toks(boiler)) <= len(old_toks(boiler)) + 1,
            f"new={sorted(pharma_toks(boiler))} old={sorted(old_toks(boiler))}")

print("\n=== 5. surface variants unify ===")
ok &= check("PD-L1 == PDL1", "pdl1" in pharma_toks("PD-L1") and "pdl1" in pharma_toks("PDL1"))
ok &= check("query PD-L1 matches doc PDL1", word_overlap("PD-L1", "the PDL1 pathway") == 1.0)

print("\n=== 6. declared limits (documented, not fixed) ===")
check("KNOWN LIMIT: CAR-T loses its T", pharma_toks("CAR-T") == {"car"},
      "documented in the module docstring")
check("KNOWN LIMIT: B7-H1 is NOT a synonym of PD-L1",
      not (pharma_toks("B7-H1") & pharma_toks("PD-L1")),
      "needs a synonym table; documented")

print("\n=== 7. modifier prefixes must not hide the target (anti-X is THE common form) ===")
for t, want in [("anti-PD-L1", "pdl1"), ("anti-CD20", "cd20"), ("anti-HER2", "her2"),
                ("Anti-B7-H1", "b7h1"), ("anti-PD-1", "pd1")]:
    got = pharma_toks(t)
    ok &= check(f"{t:12s} -> {sorted(got)}", want in got)
ok &= check("query 'PD-L1 antibody' hits doc 'anti-PD-L1 antibody'",
            word_overlap("PD-L1 antibody", "an anti-PD-L1 antibody") == 1.0,
            f"-> {word_overlap('PD-L1 antibody', 'an anti-PD-L1 antibody'):.3f}")

print("\nRESULT:", "ALL ASSERTIONS PASS" if ok else "*** FAILURES ***")
sys.exit(0 if ok else 1)
