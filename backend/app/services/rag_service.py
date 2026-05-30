from backend.app.services.pipeline_service import run_rag_pipeline


def get_answer(query: str, collection_name="test") -> dict:
    return run_rag_pipeline(query, config={"collection_name": collection_name})


if __name__ == "__main__":
    while True:
        query = input("User's query: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = get_answer(query)
        print("LLM answer:", result["answer"])
        print("Source chunks:", result["sources"])
