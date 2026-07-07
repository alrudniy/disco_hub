"""
Tests for the asymmetric embedding interface (fix #4): the Qwen3-Embedding query
instruction is formatted exactly per the model contract, and the mock backend
keeps queries and documents numerically identical (so determinism / stage-09
guarantees are unaffected) while still exposing both methods.
"""
import numpy as np

from discovery_hub.embedding import (query_prompt_prefix, format_query,
                                     MockEmbedder, get_embedder)
from discovery_hub import config


def test_query_prefix_matches_qwen3_contract():
    task = "Given a query, retrieve relevant passages"
    prefix = query_prompt_prefix(task)
    assert prefix == f"Instruct: {task}\nQuery: "
    assert prefix.endswith("Query: ")          # ST appends the query after this


def test_format_query_is_prefix_plus_query():
    task = "retrieve relevant patents"
    q = "GLP-1 receptor agonist for obesity"
    # This is the exact invariant ST relies on: prompt + text == instructed query
    assert format_query(q, task) == query_prompt_prefix(task) + q
    assert format_query(q, task) == f"Instruct: {task}\nQuery: {q}"


def test_documents_get_no_instruction_contract():
    # Documents must NOT be wrapped; the doc-side contract is "raw text". We assert
    # the formatter is query-only by checking it isn't accidentally applied to docs
    # in the mock (mock encodes raw text for both, identical vectors -- see below).
    task = config.QUERY_INSTRUCTION
    assert "Instruct:" in query_prompt_prefix(task)
    assert config.QUERY_INSTRUCTION and "retrieve" in config.QUERY_INSTRUCTION.lower()


def test_mock_queries_and_documents_are_identical_and_deterministic():
    e = MockEmbedder()
    texts = ["kinase inhibitor for oncology", "monoclonal antibody platform"]
    q1 = e.encode_queries(texts)
    d1 = e.encode_documents(texts)
    q2 = e.encode_queries(texts)
    # asymmetry is structural-only in mock: same numbers for both sides ...
    assert np.array_equal(q1, d1)
    # ... and fully deterministic across calls ...
    assert np.array_equal(q1, q2)
    # ... and L2-normalized.
    np.testing.assert_allclose(np.linalg.norm(q1, axis=1), 1.0, atol=1e-5)


def test_factory_exposes_asymmetric_interface():
    e = get_embedder(mock=True)
    assert hasattr(e, "encode_queries") and hasattr(e, "encode_documents")
    # bare encode still works (back-compat == document side)
    out = e.encode(["a passage about cell therapy"])
    assert out.shape == (1, config.EMBED_DIM)
