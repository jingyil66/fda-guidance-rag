import os
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from backend.app.core.config import settings
from backend.app.services.generation_service import (
    build_documents,
    build_sources,
    format_context,
    generate_answer,
    get_llm,
    get_parser,
    get_prompt,
)
from backend.app.services.retrieval_service import retrieve_embedding
from backend.app.services.rerank_service import get_ranker, rerank_passages

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

client = QdrantClient(url="http://localhost:6333")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

rag_prompt = get_prompt()
llm = get_llm()
str_parser = get_parser()
ranker = get_ranker()


def get_answer(query: str, collection_name="test") -> dict:
    passages = retrieve_embedding(
        query,
        collection_name,
        top_k=20,
        client=client,
        embeddings=embeddings,
    )
    top_5_results = rerank_passages(
        query,
        passages,
        top_k=5,
        ranker=ranker,
    )

    context = format_context(top_5_results)
    answer = generate_answer(query, context, rag_prompt, llm, str_parser)
    sources = build_sources(top_5_results)
    documents = build_documents(top_5_results)

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents,
    }


if __name__ == "__main__":
    while True:
        query = input("User's query: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = get_answer(query)
        print("LLM answer:", result["answer"])
        print("Source chunks:", result["sources"])
