from __future__ import annotations

from unittest.mock import MagicMock

from backend.app.services.generation_service import (
    build_documents,
    build_sources,
    format_context,
    generate_answer,
)


def test_format_context_numbered(sample_passage):
    passages = [sample_passage, dict(sample_passage, id=1)]
    context = format_context(passages, numbered=True)
    assert context.startswith("[1]")
    assert "[2]" in context


def test_build_sources_truncates_snippet():
    long_text = "x" * 1000
    passages = [
        {
            "text": long_text,
            "metadata": {"title": "Doc", "page": 3, "pdf_id": "122971", "url": "http://x"},
        }
    ]
    sources = build_sources(passages, snippet_max_chars=800)
    assert len(sources) == 1
    assert sources[0]["title"] == "Doc"
    assert sources[0]["page"] == 3
    assert sources[0]["pdf_id"] == "122971"
    assert len(sources[0]["snippet"]) <= 801
    assert sources[0]["snippet"].endswith("…")


def test_build_documents():
    passages = [{"text": "body", "metadata": {"pdf_id": "1"}}]
    docs = build_documents(passages)
    assert docs == [{"text": "body", "metadata": {"pdf_id": "1"}}]


def test_generate_answer_invokes_chain():
    chain = MagicMock()
    chain.invoke.return_value = "Synthesized answer."

    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=chain)
    prompt = MagicMock()
    prompt.__or__ = MagicMock(return_value=mid)
    llm = MagicMock()
    parser = MagicMock()

    result = generate_answer("What is X?", "context block", prompt, llm, parser)

    assert result == "Synthesized answer."
    chain.invoke.assert_called_once_with({"query": "What is X?", "context": "context block"})
