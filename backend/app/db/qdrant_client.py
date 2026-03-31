from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings

def init_qdrant(collection_name="test", qdrant_url="http://localhost:6333", embedding_model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_size = len(embeddings.embed_query("sample text"))
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    return vector_store