from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 20


def get_qdrant_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name)


def build_vector_store(
    client: QdrantClient,
    collection_name: str,
    embeddings: OpenAIEmbeddings,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def docs_to_passages(docs: list, start_id: int = 0) -> list[dict]:
    return [
        {
            "id": start_id + i,
            "text": doc.page_content,
            "metadata": doc.metadata or {},
            "score": None,
        }
        for i, doc in enumerate(docs)
    ]


def retrieve_embedding(
    query: str,
    collection_name: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    client: QdrantClient | None = None,
    embeddings: OpenAIEmbeddings | None = None,
    qdrant_url: str = DEFAULT_QDRANT_URL,
) -> list[dict]:
    client = client or get_qdrant_client(qdrant_url)
    embeddings = embeddings or get_embeddings()
    vector_store = build_vector_store(client, collection_name, embeddings)
    initial_results = vector_store.similarity_search(query=query, k=top_k)
    return docs_to_passages(initial_results)
