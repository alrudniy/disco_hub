#!/usr/bin/env python3
"""
generate_queries.py -- synthetic query generation for embedder fine-tuning.

Drop-in for the `discovery_finetune` package. Manufactures (query, positive-doc)
training pairs with NO labeled data, using the InPars / Promptagator recipe with the
register-crossing twist that is the whole point of Discovery Hub:

    documents are written in legal/technical PATENT register;
    queries must be written in the clinical/business register a pharma scout uses.

For each normalized DiscoveryDoc we ask a strong generator LLM to read the
technically-worded invention and emit several queries in the *opposite* register --
the way a business-development or clinical researcher would phrase the need --
deliberately NOT echoing the document's jargon. Those (query -> doc) pairs are the
positives that `build_dataset.py` mines hard negatives against and `train.py`
optimizes with MultipleNegativesRankingLoss.

Why a 753B model is appropriate HERE specifically: this is an OFFLINE BATCH job, so the
generator's size is irrelevant to serving. Better synthetic queries -> a better
fine-tuned embedder -> better TTO<->pharma matching. That is the one place a frontier
model touches retrieval quality, even though it can't sit in the live retrieval path.

Generator endpoint is any OpenAI-compatible API, configured with DEDICATED env vars so
the offline query-gen model stays independent of the online stage-08 explanation model
(DH_LLM_*) -- you can serve a local 7-8B for explanations while generating queries with
GLM-5.2 at the same time. Defaults target GLM-5.2 on the Z.ai API:

    export DH_QUERYGEN_BASE_URL="https://api.z.ai/api/paas/v4"   # any OpenAI-compatible API
    export DH_QUERYGEN_API_KEY="sk-..."                          # provider key (never commit)
    export DH_QUERYGEN_MODEL="glm-5.2"

Mock mode (--mock) needs no key or network: it derives deterministic register-crossed
queries from each doc so the pipeline runs and tests end-to-end, matching the rest of
the project's mock philosophy.

Usage:
    python generate_queries.py --docs data/normalized/docs.jsonl \
                               --out  data/finetune/synthetic_queries.jsonl \
                               --n-queries 3
    python generate_queries.py --mock --docs sample.jsonl --out out.jsonl   # no API/key

Output: JSONL, one row per generated query (a positive pair); join to docs.jsonl on
`doc_id` in build_dataset.py for hard-negative mining.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from datetime import date
from pathlib import Path

# --- dedicated env vars: keep the offline query-gen model independent of DH_LLM_* ---
QUERYGEN_BASE_URL = os.environ.get("DH_QUERYGEN_BASE_URL", "https://api.z.ai/api/paas/v4")
QUERYGEN_API_KEY = os.environ.get("DH_QUERYGEN_API_KEY", "")
QUERYGEN_MODEL = os.environ.get("DH_QUERYGEN_MODEL", "glm-5.2")

MIN_CHARS_DEFAULT = 200          # AUTM noise filter: skip near-empty / boilerplate docs
MIN_Q_WORDS, MAX_Q_WORDS = 4, 40

# Few-shot exemplars that TEACH THE REGISTER CROSS (technical doc -> lay scout query).
# Generic and non-proprietary.
FEWSHOT = [
    {
        "doc": ("Title: Bicyclic heteroaryl compounds as inhibitors of Bruton's tyrosine "
                "kinase. Abstract: Disclosed are substituted pyrazolo[3,4-d]pyrimidine "
                "derivatives that covalently bind Cys481 of BTK, pharmaceutical "
                "compositions thereof, and methods of treating B-cell proliferative "
                "disorders."),
        "queries": [
            "covalent BTK inhibitor for B-cell lymphoma we could in-license",
            "small molecule targeting Bruton's tyrosine kinase for autoimmune indications",
            "oral therapy for relapsed chronic lymphocytic leukemia",
        ],
    },
    {
        "doc": ("Title: Lipid nanoparticle formulations for delivery of messenger RNA. "
                "Abstract: Ionizable cationic lipids and processes for encapsulating mRNA "
                "payloads to enhance endosomal escape and in vivo expression in hepatic "
                "tissue."),
        "queries": [
            "mRNA delivery platform for liver-targeted gene therapy",
            "ionizable lipid nanoparticle technology available for partnership",
            "non-viral vector to improve in vivo expression of RNA therapeutics",
        ],
    },
]


def _client():
    if not QUERYGEN_API_KEY:
        sys.exit("DH_QUERYGEN_API_KEY is not set (needed for real mode; use --mock to "
                 "run without an API).")
    try:
        from openai import OpenAI  # OpenAI-compatible (Z.ai / vLLM / others)
    except ImportError:
        sys.exit("the 'openai' package is required for real mode: pip install openai")
    return OpenAI(base_url=QUERYGEN_BASE_URL, api_key=QUERYGEN_API_KEY)


def _build_prompt(doc_text: str, n: int) -> list[dict]:
    sys_msg = (
        "You generate search queries for a pharmaceutical technology-scouting engine. "
        "You are shown an invention written in dense legal/technical PATENT language. "
        "Produce queries in the DIFFERENT register a pharma business-development or "
        "clinical researcher actually uses: plain clinical/commercial language about the "
        "therapeutic goal, target, indication, modality, or licensing need. Crucially, do "
        "NOT copy the document's technical jargon, chemical names, or rare phrasing -- "
        "paraphrase the intent the way a person searching would phrase it. Vary the angle "
        "across queries (mechanism in lay terms, disease/indication, modality, "
        "business/licensing framing). "
        f"Return ONLY a JSON array of exactly {n} short query strings, nothing else."
    )
    msgs = [{"role": "system", "content": sys_msg}]
    for ex in FEWSHOT:                                  # few-shot demonstrations
        msgs.append({"role": "user", "content": ex["doc"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["queries"][:n])})
    msgs.append({"role": "user", "content": doc_text})
    return msgs


def _parse_array(text: str) -> list[str]:
    """Tolerate ```json fences / stray prose around the JSON array."""
    m = re.search(r"\[.*\]", text.strip(), re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(q) for q in arr if isinstance(q, str)]


def _gen_real(client, doc_text: str, n: int, temperature: float) -> list[str]:
    for attempt in range(3):                            # retry network / rate / parse
        try:
            resp = client.chat.completions.create(
                model=QUERYGEN_MODEL, temperature=temperature,
                messages=_build_prompt(doc_text, n))
            return _parse_array(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                print(f"  ! generation failed ({e}); skipping doc", file=sys.stderr)
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


# ---- deterministic mock generator: derive lay queries without any API ----
_JARGON = re.compile(r"(derivativ|substitut|compositio|heteroaryl|bicyclic|pyrazolo|"
                     r"pyrimidin|ionizable|encapsulat|endosomal|method of|disclosed|"
                     r"comprising|wherein|embodiment)", re.I)
_TEMPLATES = [
    "therapy targeting {kw} for {ind}",
    "{mod} for {ind} available for licensing",
    "partner with technology for {kw}",
    "treatment approach involving {kw}",
    "{kw} candidate for clinical development",
]
_INDICATIONS = ["oncology", "autoimmune disease", "rare disease", "metabolic disorders",
                "neurology", "inflammatory conditions"]
_MODALITIES = ["small molecule", "biologic", "delivery platform", "antibody",
               "gene therapy", "cell therapy"]


def _keywords(doc) -> list[str]:
    kws = list(doc.get("keywords") or [])
    if not kws:                                         # fall back to non-jargon title words
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", doc.get("title", ""))
        kws = [w.lower() for w in words if not _JARGON.search(w)]
    return kws or ["novel therapeutic"]


def _gen_mock(doc, n: int) -> list[str]:
    rng = random.Random(int(hashlib.sha1(doc.get("doc_id", "").encode()).hexdigest(), 16))
    kws, out = _keywords(doc), []
    for _ in range(n * 3):
        q = rng.choice(_TEMPLATES).format(
            kw=rng.choice(kws), ind=rng.choice(_INDICATIONS), mod=rng.choice(_MODALITIES))
        if q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


# ---- quality filters shared by both modes ----
def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip('"').strip()


def _ok(q: str, title: str) -> bool:
    w = q.split()
    if not (MIN_Q_WORDS <= len(w) <= MAX_Q_WORDS):
        return False
    # reject near-verbatim copies of the title (enforce the register cross)
    tset, qset = set(title.lower().split()), set(q.lower().split())
    if tset and len(tset & qset) / max(1, len(qset)) > 0.8:
        return False
    return True


def _filter(qs, title):
    seen, out = set(), []
    for q in qs:
        q = _norm(q)
        if q and q.lower() not in seen and _ok(q, title):
            seen.add(q.lower())
            out.append(q)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="data/normalized/docs.jsonl",
                    help="input normalized DiscoveryDoc JSONL")
    ap.add_argument("--out", default="data/finetune/synthetic_queries.jsonl")
    ap.add_argument("--n-queries", type=int, default=3, help="queries per document")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT,
                    help="AUTM noise filter: skip docs with less text than this")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all (else a cost cap)")
    ap.add_argument("--temperature", type=float, default=0.7, help="query diversity")
    ap.add_argument("--seed", type=int, default=20240611)
    ap.add_argument("--mock", action="store_true", help="no API/key; deterministic")
    args = ap.parse_args()

    random.seed(args.seed)
    docs_path, out_path = Path(args.docs), Path(args.out)
    if not docs_path.exists():
        sys.exit(f"input not found: {docs_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None if args.mock else _client()
    generator = "mock" if args.mock else QUERYGEN_MODEL
    today = date.today().isoformat()

    n_docs = n_kept = n_pairs = 0
    with docs_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            n_docs += 1
            text = (doc.get("embedding_text")
                    or f"{doc.get('title', '')} {doc.get('abstract', '')}").strip()
            if len(text) < args.min_chars:              # AUTM noise filter
                continue
            title = doc.get("title", "")
            doc_text = f"Title: {title}. Abstract: {doc.get('abstract', '')}".strip()

            raw = (_gen_mock(doc, args.n_queries) if args.mock
                   else _gen_real(client, doc_text, args.n_queries, args.temperature))
            queries = _filter(raw, title)[:args.n_queries]
            if not queries:
                continue
            n_kept += 1
            for i, q in enumerate(queries):
                fout.write(json.dumps({
                    "query": q,
                    "doc_id": doc.get("doc_id"),         # positive; join key downstream
                    "source": doc.get("source"),
                    "source_url": doc.get("source_url", ""),
                    "retrieved_date": doc.get("retrieved_date", ""),
                    "generator": generator,
                    "generated_date": today,
                    "query_index": i,
                }) + "\n")
                n_pairs += 1
            if args.max_docs and n_kept >= args.max_docs:
                break

    print(f"docs read={n_docs}  kept={n_kept}  pairs written={n_pairs}  -> {out_path}")
    print("generator: " + generator + ("" if args.mock else f"  @ {QUERYGEN_BASE_URL}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
