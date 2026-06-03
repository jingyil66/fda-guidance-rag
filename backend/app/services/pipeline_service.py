import os

from backend.app.core.config import settings
from backend.app.services.generation_service import (
    DEFAULT_LLM_MODEL,
    build_documents,
    build_sources,
    format_context,
    generate_answer,
    get_llm,
    get_parser,
    get_prompt,
)
from backend.app.services.retrieval_service import (
    DEFAULT_EMBEDDING_MODEL,
    get_embeddings,
    get_qdrant_client,
    retrieve_embedding,
)
from backend.app.services.passage_format import (
    PASSAGE_FORMAT_LEGACY,
    PASSAGE_FORMAT_RAW,
)
from backend.app.services.rerank_service import (
    DEFAULT_RERANK_MODEL,
    get_ranker,
    rerank_passages,
)

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

DEFAULT_COLLECTION_NAME = settings.DEFAULT_QDRANT_COLLECTION
DEFAULT_TOP_K_INITIAL = 20
DEFAULT_TOP_K_FINAL = 5
DEFAULT_RERANK_ENABLED = False

_client = None
_embeddings = None
_ranker = None
_prompt = get_prompt()
_llm = None
_parser = get_parser()


def _default_client():
    global _client
    if _client is None:
        _client = get_qdrant_client(settings.DEFAULT_QDRANT_URL)
    return _client


def _default_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings(DEFAULT_EMBEDDING_MODEL)
    return _embeddings


def _default_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(DEFAULT_LLM_MODEL)
    return _llm


def _normalize_config(config: dict | None) -> dict:
    config = config or {}
    return {
        "collection_name": config.get("collection_name", DEFAULT_COLLECTION_NAME),
        "top_k_initial": config.get("top_k_initial", DEFAULT_TOP_K_INITIAL),
        "top_k_final": config.get("top_k_final", DEFAULT_TOP_K_FINAL),
        "rerank_enabled": config.get("rerank_enabled", DEFAULT_RERANK_ENABLED),
        "qdrant_url": config.get("qdrant_url", settings.DEFAULT_QDRANT_URL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "llm_model": config.get("llm_model", DEFAULT_LLM_MODEL),
        "rerank_model": config.get("rerank_model", DEFAULT_RERANK_MODEL),
        "rerank_passage_format": config.get("rerank_passage_format", PASSAGE_FORMAT_RAW),
        "context_passage_format": config.get("context_passage_format", PASSAGE_FORMAT_LEGACY),
    }


def _resolve_runtime(config: dict) -> dict:
    qdrant_url = config["qdrant_url"]
    if qdrant_url != settings.DEFAULT_QDRANT_URL:
        client = get_qdrant_client(qdrant_url)
    else:
        client = _default_client()

    embedding_model = config["embedding_model"]
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        embeddings = get_embeddings(embedding_model)
    else:
        embeddings = _default_embeddings()

    rerank_model = config["rerank_model"]
    if config["rerank_enabled"]:
        global _ranker
        if rerank_model != DEFAULT_RERANK_MODEL:
            ranker = get_ranker(model_name=rerank_model)
        else:
            if _ranker is None:
                _ranker = get_ranker()
            ranker = _ranker
    else:
        ranker = None

    llm_model = config["llm_model"]
    if llm_model != DEFAULT_LLM_MODEL:
        llm = get_llm(llm_model)
    else:
        llm = _default_llm()

    return {
        "client": client,
        "embeddings": embeddings,
        "ranker": ranker,
        "llm": llm,
        "prompt": _prompt,
        "parser": _parser,
    }


def run_rag_pipeline(query: str, config: dict | None = None) -> dict:
    if not query or not str(query).strip():
        return {
            "answer": "Query is empty",
            "sources": [],
            "documents": [],
        }

    cfg = _normalize_config(config)
    runtime = _resolve_runtime(cfg)

    passages = retrieve_embedding(
        query,
        cfg["collection_name"],
        top_k=cfg["top_k_initial"],
        client=runtime["client"],
        embeddings=runtime["embeddings"],
        qdrant_url=cfg["qdrant_url"],
    )

    if cfg["rerank_enabled"]:
        ranked = rerank_passages(
            query,
            passages,
            top_k=cfg["top_k_final"],
            ranker=runtime["ranker"],
            model_name=cfg["rerank_model"],
            passage_format=cfg["rerank_passage_format"],
        )
    else:
        ranked = passages[: cfg["top_k_final"]]

    context = format_context(
        ranked,
        passage_format=cfg["context_passage_format"],
        numbered=True,
    )
    answer = generate_answer(
        query,
        context,
        runtime["prompt"],
        runtime["llm"],
        runtime["parser"],
    )
    sources = build_sources(ranked)
    documents = build_documents(ranked)

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents,
    }
