from flashrank import Ranker, RerankRequest

from backend.app.services.passage_format import (
    PASSAGE_FORMAT_RAW,
    format_passage_for_rerank,
    restore_passage_text,
)

DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_CACHE_DIR = "opt/flashrank"
DEFAULT_TOP_K = 5


def get_ranker(
    model_name: str = DEFAULT_RERANK_MODEL,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Ranker:
    return Ranker(model_name=model_name, cache_dir=cache_dir)


def rerank_passages(
    query: str,
    passages: list[dict],
    *,
    top_k: int = DEFAULT_TOP_K,
    ranker: Ranker | None = None,
    model_name: str = DEFAULT_RERANK_MODEL,
    cache_dir: str = DEFAULT_CACHE_DIR,
    passage_format: str = PASSAGE_FORMAT_RAW,
) -> list[dict]:
    if not passages:
        return []

    ranker = ranker or get_ranker(model_name=model_name, cache_dir=cache_dir)

    try:
        if passage_format == PASSAGE_FORMAT_RAW:
            request = RerankRequest(query=query, passages=passages)
        else:
            enriched_passages = []
            for passage in passages:
                enriched = dict(passage)
                enriched["text"] = format_passage_for_rerank(passage, passage_format)
                enriched_passages.append(enriched)
            request = RerankRequest(query=query, passages=enriched_passages)

        results = ranker.rerank(request)
        ranked = results[:top_k]
        if passage_format == PASSAGE_FORMAT_RAW:
            return ranked

        return [restore_passage_text(item, passages) for item in ranked]
    except Exception:
        return passages[:top_k]
