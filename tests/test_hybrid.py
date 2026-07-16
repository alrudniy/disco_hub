"""
Known-answer tests for the hybrid-retrieval building blocks: BM25 keyword
scoring, query->graph entity linking, and Reciprocal Rank Fusion. If these pass,
the hybrid signals behave as specified. Run: python -m pytest tests/ -q
"""
from discovery_hub.keyword import BM25Index, tokenize
from discovery_hub.entity_link import build_surface_index, link_query, normalize
from discovery_hub.fusion import reciprocal_rank_fusion, fuse_to_pool


class _Doc:
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.embedding_text = text
        self.title = ""
        self.abstract = ""


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_bm25_term_frequency_ranks_higher():
    docs = [_Doc("A", "kinase kinase inhibitor pathway"),
            _Doc("B", "kinase inhibitor pathway"),
            _Doc("C", "antibody conjugate payload")]
    bm = BM25Index.build(docs)
    res = bm.score("kinase", top_k=10)
    ids = [d for d, _ in res]
    assert ids[0] == "A" and ids[1] == "B"   # A mentions kinase twice
    assert "C" not in ids                     # C has no query term


def test_bm25_rarer_term_weighs_more():
    # "rituximab" appears in 1 doc, "cell" in all 3 -> rarer term scores higher
    docs = [_Doc("A", "rituximab cell therapy"),
            _Doc("B", "cell therapy platform"),
            _Doc("C", "cell biology assay")]
    bm = BM25Index.build(docs)
    s_rare = dict(bm.score("rituximab", top_k=10)).get("A", 0.0)
    s_common = dict(bm.score("cell", top_k=10)).get("A", 0.0)
    assert s_rare > s_common


def test_bm25_is_deterministic_and_persists(tmp_path):
    docs = [_Doc(f"d{i}", f"alpha beta gamma term{i}") for i in range(5)]
    bm = BM25Index.build(docs)
    r1 = bm.score("alpha term3", top_k=5)
    p = tmp_path / "bm25.json"
    bm.save(p)
    bm2 = BM25Index.load(p)
    r2 = bm2.score("alpha term3", top_k=5)
    assert r1 == r2


def test_tokenize_keeps_short_acronyms():
    toks = tokenize("GLP-1 and IL6 modulation")
    # acronyms survive; the standalone single-char "1" is dropped by the len>=2 rule
    assert "glp" in toks and "il6" in toks
    assert "1" not in toks


# --------------------------------------------------------------------------- #
# Entity linking
# --------------------------------------------------------------------------- #
def _nodes():
    return [
        {"node_id": "organization:stanford", "ntype": "organization", "label": "Stanford"},
        {"node_id": "organization:pfizer", "ntype": "organization", "label": "Pfizer"},
        {"node_id": "organization:eli lilly", "ntype": "organization", "label": "Eli Lilly"},
        {"node_id": "organization:genentech inc", "ntype": "organization",
         "label": "Genentech Inc"},
        {"node_id": "facility:pfizer medical center", "ntype": "facility",
         "label": "Pfizer Medical Center"},
        {"node_id": "technology:uspto:US1", "ntype": "technology",
         "label": "a kinase inhibitor", "doc_id": "uspto:US1"},
    ]


def test_link_single_token_org():
    idx = build_surface_index(_nodes())
    assert link_query("research interest in stanford kinase work", idx) == \
        ["organization:stanford"]


def test_link_alias_only_when_token_is_the_whole_name():
    idx = build_surface_index(_nodes())
    # "Genentech Inc" is one name plus a legal suffix, so "genentech" IS the entity
    # and may alias it.
    assert link_query("a genentech antibody", idx) == ["organization:genentech inc"]


def test_link_does_not_alias_a_fragment_of_a_multiword_name():
    idx = build_surface_index(_nodes())
    # This test previously asserted the OPPOSITE -- that "lilly" aliases "Eli Lilly"
    # because it is a globally-unique non-stopword token. That rule was refuted on
    # this graph: uniqueness promotes junk by construction, because in ~926k surfaces
    # a distinctive name is often SHARED while an odd word appears exactly once. It
    # is what made "Oral tablet formulation for CML" link to the real node
    # organization:hot album tansansen tablet inc. See build_surface_index's docstring.
    #
    # Losing "lilly" is the known, deliberate cost of the fix: "Eli Lilly" is two
    # tokens with no legal suffix, so neither word aliases it. That is the
    # precision-over-recall trade the module always claimed to make -- the old
    # mechanism just delivered the reverse. The full surface still links.
    assert link_query("a lilly compound for oncology", idx) == []
    assert link_query("an eli lilly compound", idx) == ["organization:eli lilly"]


def test_link_greedy_longest_match_prefers_full_phrase():
    idx = build_surface_index(_nodes())
    linked = link_query("study at pfizer medical center today", idx)
    assert "facility:pfizer medical center" in linked
    # greedy consumed the 3-gram, so the bare org "pfizer" is NOT separately linked
    assert linked == ["facility:pfizer medical center"]


def test_link_generic_words_do_not_falsely_link():
    idx = build_surface_index(_nodes())
    # "medical"/"center" are stop-aliases; technologies are never linkable
    assert link_query("a medical center kinase inhibitor study", idx) == []


def test_link_abstains_when_no_entity():
    idx = build_surface_index(_nodes())
    assert link_query("novel small molecule for metabolic disease", idx) == []


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def test_rrf_scores_match_formula():
    lists = {"dense": ["x", "y"], "kw": ["y", "z"]}
    fused = reciprocal_rank_fusion(lists, k=60)
    assert abs(fused["x"] - 1 / 61) < 1e-12
    assert abs(fused["y"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["z"] - 1 / 62) < 1e-12


def test_rrf_ranks_doc_in_both_lists_first():
    lists = {"dense": ["x", "y", "z"], "kw": ["y", "w"]}
    pool = fuse_to_pool(lists, top_k=10, k=60)
    assert pool[0][0] == "y"   # appears in both lists


def test_rrf_handles_missing_and_empty_lists():
    # an abstaining (absent) signal just isn't in the dict; empty list is harmless
    lists = {"dense": ["a", "b"], "graph": []}
    fused = reciprocal_rank_fusion(lists, k=60)
    assert set(fused) == {"a", "b"}
    assert fused["a"] > fused["b"]
