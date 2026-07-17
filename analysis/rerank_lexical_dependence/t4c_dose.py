"""
T4c DOSE-RESPONSE. Is the -5.47 / +1.76 asymmetry dose, or context?

T4 measured two single-term swaps in opposite directions and got magnitudes 3.1x
apart, which rules out a CONSTANT lexical bonus. But the two documents did not carry
the same DOSE: the rank-1 doc has 6 HER2 mentions and the bridge doc has 1. A
per-mention effect with saturation would reconcile -5.47 and +1.76 with no context
term at all. This separates those.

DESIGN: on uspto:US12227591B2 (rerank rank 1, n_tok=319, z=+0.5305), replace the
first k of its 6 HER2 mentions with ErbB2, k = 0..6, and measure z(k). Then repeat
replacing the LAST k, which separates dose from position (occurrence 1 is the title).
Same query, same config, every other byte constant.

PRE-REGISTERED (stated before the run):
  z(1)-z(0) ~= -1.76 +- 0.5 and curve saturates -> dose explains the asymmetry,
      no context term needed
  z(k) linear, slope ~= -5.47/6 = -0.91/mention -> additive; the bridge's +1.76 is
      ~1.9x the per-mention slope -> discrepancy survives
  z(1)-z(0) ~= -5 then flat -> presence/absence, not count; the bridge's +1.76 is
      2.9x smaller -> asymmetry survives, context interaction

Config unchanged: bge-reranker-v2-m3, max_length=512, batch_size=16, Sigmoid head,
query "HER2 targeted therapy for breast cancer". Nothing written. Diagnosis only.
"""
import importlib.util
import math
import re
import sys

sys.path.insert(0, "/workspace/dh_multiagent")

QUERY = "HER2 targeted therapy for breast cancer"
DOC = "uspto:US12227591B2"
RE_HER2 = re.compile(r"\bher[\s\-‐‑‒–—_]*2\b", re.IGNORECASE)

logit = lambda s: math.log(s / (1.0 - s))

spec = importlib.util.spec_from_file_location("rr", "/workspace/dh_multiagent/07_retrieve_rank.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

print("building retriever ...", flush=True)
R = rr.Retriever(mock=False, device="cuda")
ce = R._get_cross_encoder()
tok = ce.tokenizer

text = R.docs[DOC].embedding_text
occ = list(RE_HER2.finditer(text))
print(f"\n{DOC}: {len(occ)} HER2 occurrences")
for i, m in enumerate(occ, 1):
    lo, hi = max(0, m.start() - 34), min(len(text), m.end() + 34)
    ctx = text[lo:hi].replace("\n", " / ")
    print(f"  #{i}  pos {m.start():4d}  {m.group()!r}   ...{ctx}...")


def replace_subset(s, idxs):
    """Replace exactly the occurrences whose 1-based index is in idxs."""
    out, last = [], 0
    for i, m in enumerate(RE_HER2.finditer(s), 1):
        out.append(s[last:m.start()])
        out.append("ErbB2" if i in idxs else m.group())
        last = m.end()
    out.append(s[last:])
    return "".join(out)


def score(t):
    return float(ce.predict([(QUERY, t)], batch_size=16)[0])


n = len(occ)
print("\n" + "=" * 96)
print("T4c  DOSE CURVE   z(k), replacing the FIRST k of "
      f"{n} HER2 mentions with ErbB2")
print("=" * 96)
print(f"{'k':>2} {'n_tok':>6} {'score':>22} {'z':>9} {'dz vs k=0':>10} {'marginal':>9}")
curves = {}
for order in ("first", "last"):
    zs = []
    for k in range(n + 1):
        idxs = set(range(1, k + 1)) if order == "first" else set(range(n - k + 1, n + 1))
        t = replace_subset(text, idxs)
        s = score(t)
        z = logit(s)
        zs.append(z)
        if order == "first":
            marg = "" if k == 0 else f"{z - zs[k-1]:+9.4f}"
            print(f"{k:>2} {len(tok(QUERY, t)['input_ids']):>6} {s!r:>22} "
                  f"{z:>+9.4f} {z - zs[0]:>+10.4f} {marg:>9}")
    curves[order] = zs

print("\n" + "-" * 96)
print("LAST-k order (separates dose from position; occurrence 1 is the title)")
print("-" * 96)
print(f"{'k':>2} {'z':>9} {'dz vs k=0':>10} {'marginal':>9}")
zl = curves["last"]
for k in range(n + 1):
    marg = "" if k == 0 else f"{zl[k] - zl[k-1]:+9.4f}"
    print(f"{k:>2} {zl[k]:>+9.4f} {zl[k] - zl[0]:>+10.4f} {marg:>9}")

zf = curves["first"]
print("\n" + "=" * 96)
print("T4c ANSWERS  (pre-registered criterion)")
print("=" * 96)
print(f"  z(0)                      : {zf[0]:+.4f}")
print(f"  z(1) - z(0)  [first-k]    : {zf[1]-zf[0]:+.4f}   <- compare to the bridge's +1.7618")
print(f"  z(1) - z(0)  [last-k]     : {zl[1]-zl[0]:+.4f}")
print(f"  z({n}) - z(0)               : {zf[n]-zf[0]:+.4f}   (T4(b) full swap was -5.4711)")
print(f"  mean marginal per mention : {(zf[n]-zf[0])/n:+.4f}")
marginals = [zf[k]-zf[k-1] for k in range(1, n+1)]
print(f"  marginals (first-k)       : {[round(m,3) for m in marginals]}")
print(f"  |first marginal| / |mean| : {abs(marginals[0])/abs((zf[n]-zf[0])/n):.2f}x  "
      f"(1.0 = perfectly linear/additive; >1 = front-loaded/saturating)")
print(f"  title occurrence dominant?: first-k dz(1)={zf[1]-zf[0]:+.4f} vs "
      f"last-k dz(1)={zl[1]-zl[0]:+.4f}")
gap = 0.8951
print(f"\n  boundary to beat          : {gap} logits")
print(f"  does ONE mention clear it?: first-k {abs(zf[1]-zf[0]) >= gap}   last-k {abs(zl[1]-zl[0]) >= gap}")
