"""
Tests for fix #6: the faithfulness / citation verifier (proving it detects
hallucinations and fabricated citations, not just passing vacuously), and the
embedder's batch invariance in mock. Run: python -m pytest tests/ -q
"""
import numpy as np

from discovery_hub import faithfulness as F
from discovery_hub.embedding import MockEmbedder

EV = {"doc1": ("A selective GLP-1 receptor agonist reduces blood glucose and "
               "body weight in patients with type 2 diabetes.")}
URL = "http://ex/doc1"


# --------------------------------------------------------------------------- #
# Faithfulness verifier
# --------------------------------------------------------------------------- #
def test_grounded_claim_with_valid_citation_scores_clean():
    rec = {"doc_id": "doc1", "citation": URL,
           "why": f"A GLP-1 receptor agonist that reduces blood glucose and body "
                  f"weight. [source: {URL}]"}
    r = F.evaluate_recommendations([rec], EV, context="GLP-1 for diabetes", threshold=0.5)
    assert r["grounded_rate"] == 1.0
    assert r["citation_validity_rate"] == 1.0
    assert r["hallucination_rate"] == 0.0


def test_hallucinated_claim_is_caught():
    # content shares no substantive vocabulary with the evidence
    rec = {"doc_id": "doc1", "citation": URL,
           "why": f"This compound cures Alzheimer's and reverses aging in primates "
                  f"within a week. [source: {URL}]"}
    r = F.evaluate_recommendations([rec], EV, context="GLP-1 for diabetes", threshold=0.5)
    assert r["grounded_rate"] == 0.0
    assert r["hallucination_rate"] == 1.0
    assert r["citation_validity_rate"] == 1.0          # citation itself is fine
    assert r["flagged"] and r["flagged"][0]["grounded"] is False


def test_fabricated_citation_is_caught_independently():
    # content is grounded, but it cites a URL it was never given
    rec = {"doc_id": "doc1", "citation": URL,
           "why": "A GLP-1 receptor agonist reducing blood glucose. "
                  "[source: http://evil/fabricated]"}
    r = F.evaluate_recommendations([rec], EV, context="", threshold=0.5)
    assert r["citation_validity_rate"] == 0.0          # invalid citation flagged
    assert r["grounded_rate"] == 1.0                   # but content is supported
    assert r["flagged"][0]["bad_citations"] == ["http://evil/fabricated"]


def test_extract_citations_pulls_urls_in_order():
    txt = f"see [source: {URL}] and also https://b.org/y here"
    assert F.extract_citations(txt) == [URL, "https://b.org/y"]


def test_lexical_support_bounds_and_empty_claim():
    assert F.lexical_support("", "any evidence") == 1.0       # nothing to ground
    s = F.lexical_support("kinase inhibitor", "monoclonal antibody platform")
    assert 0.0 <= s <= 1.0


# --------------------------------------------------------------------------- #
# Batch invariance (mock is invariant by construction)
# --------------------------------------------------------------------------- #
def test_mock_embedder_is_batch_invariant():
    e = MockEmbedder()
    texts = [f"document number {i} about kinase inhibitors" for i in range(7)]
    full = e.encode_queries(texts, batch_size=len(texts))
    one_at_a_time = e.encode_queries(texts, batch_size=1)
    mid = e.encode_queries(texts, batch_size=3)
    assert np.array_equal(full, one_at_a_time)
    assert np.array_equal(full, mid)
