from backend.app.core.config import settings
from backend.app.services.pipeline_service import run_rag_pipeline


def get_answer(
    query: str,
    collection_name: str | None = None,
) -> dict:
    collection_name = collection_name or settings.DEFAULT_QDRANT_COLLECTION
    return run_rag_pipeline(
        query,
        config={
            "collection_name": collection_name,
            "rerank_enabled": False,
        },
    )


if __name__ == "__main__":
    while True:
        query = input("User's query: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = get_answer(query)
        print("LLM answer:", result["answer"])
        print("Source chunks:", result["sources"])
