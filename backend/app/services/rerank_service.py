from flashrank import Ranker, RerankRequest

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
) -> list[dict]:
    if not passages:
        return []

    ranker = ranker or get_ranker(model_name=model_name, cache_dir=cache_dir)

    try:
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        return results[:top_k]
    except Exception:
        return passages[:top_k]
