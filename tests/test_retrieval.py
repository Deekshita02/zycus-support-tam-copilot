from src.retrieval import load_kb_chunks, search_kb


def test_kb_chunks_loaded():
    chunks = load_kb_chunks()
    assert len(chunks) > 10
    assert all(c.text for c in chunks)


def test_search_returns_relevant_doc_for_known_error_code():
    results = search_kb("SAML_ASSERTION_EXPIRED", top_k=3)
    assert results
    assert any("authentication-sso" in r["source"] for r in results)


def test_search_empty_query_does_not_crash():
    results = search_kb("", top_k=3)
    assert isinstance(results, list)
