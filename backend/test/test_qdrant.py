from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "test"

client = QdrantClient(url=QDRANT_URL)

collections = client.get_collections()
print(collections)

points, next_page = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=10,
    with_vectors=True
)

for point in points:
    print("ID:", point.id)
    print("Vector length:", len(point.vector) if point.vector else "No vector")
    print("Payload:", point.payload)
    print("-"*50)
import langchain
print(langchain.__version__)

