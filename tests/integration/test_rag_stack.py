from __future__ import annotations

import pytest

DEFAULT_QUERY = (
    "How should sponsors integrate feedback from the FDA on HF study protocols "
    "into their development timelines, and what specific documentation is required "
    "for submission in their NDA, BLA, or ANDA applications?"
)
DEFAULT_COLLECTION = "test"


@pytest.mark.integration
def test_retrieve_embedding(require_qdrant, require_openai):
    from backend.app.services.retrieval_service import retrieve_embedding

    passages = retrieve_embedding(DEFAULT_QUERY, DEFAULT_COLLECTION, top_k=5)
    assert isinstance(passages, list)


@pytest.mark.integration
def test_rerank_passages(require_qdrant, require_openai):
    from backend.app.services.retrieval_service import retrieve_embedding
    from backend.app.services.rerank_service import get_ranker, rerank_passages

    passages = retrieve_embedding(DEFAULT_QUERY, DEFAULT_COLLECTION, top_k=20)
    ranker = get_ranker()
    reranked = rerank_passages(DEFAULT_QUERY, passages, top_k=5, ranker=ranker)
    assert isinstance(reranked, list)
    assert len(reranked) <= 5


@pytest.mark.integration
def test_generation_pipeline(require_qdrant, require_openai):
    from backend.app.services.generation_service import (
        build_sources,
        format_context,
        generate_answer,
        get_llm,
        get_parser,
        get_prompt,
    )
    from backend.app.services.retrieval_service import retrieve_embedding
    from backend.app.services.rerank_service import get_ranker, rerank_passages

    passages = retrieve_embedding(DEFAULT_QUERY, DEFAULT_COLLECTION, top_k=20)
    ranker = get_ranker()
    reranked = rerank_passages(DEFAULT_QUERY, passages, top_k=5, ranker=ranker)
    if not reranked:
        pytest.skip(f"No passages in collection {DEFAULT_COLLECTION!r}")

    context = format_context(reranked)
    answer = generate_answer(
        DEFAULT_QUERY,
        context,
        get_prompt(),
        get_llm(),
        get_parser(),
    )
    sources = build_sources(reranked)

    assert isinstance(answer, str)
    assert answer.strip()
    assert len(sources) == len(reranked)


@pytest.mark.integration
def test_run_rag_pipeline(require_qdrant, require_openai):
    from backend.app.core.config import settings
    from backend.app.services.pipeline_service import run_rag_pipeline

    result = run_rag_pipeline(
        DEFAULT_QUERY,
        config={
            "collection_name": settings.DEFAULT_QDRANT_COLLECTION,
            "top_k_initial": 20,
            "top_k_final": 5,
            "rerank_enabled": False,
        },
    )
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)


@pytest.mark.integration
def test_chunk_fixed_synthetic():
    from langchain_core.documents import Document

    from backend.app.services.chunking_service import chunk_fixed, chunks_to_records

    source_docs = [
        Document(
            page_content="word " * 5000,
            metadata={"title": "synthetic", "pdf_id": "synthetic", "page": 1},
        )
    ]
    chunks = chunk_fixed(source_docs, chunk_size=600, chunk_overlap=200)
    records = chunks_to_records(chunks)

    assert len(chunks) > 1
    assert all(len(record["text"]) <= 600 for record in records)
