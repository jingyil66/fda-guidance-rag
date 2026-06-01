from backend.app.services.pipeline_service import run_rag_pipeline


def get_answer(
    query: str,
    collection_name: str = "experiment_subset200_chunk600_overlap200",
) -> dict:
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
