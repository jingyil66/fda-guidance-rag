from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "test"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"✅ Collection '{collection_name}' deleted.")
else:
    print(f"⚠️ Collection '{collection_name}' not found.")