import os
import requests
import time
import tempfile
from multiprocessing import Process, Queue, cpu_count
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def downloader(url_list, process_queue: Queue, headers=None):
    for url in url_list:
        while process_queue.full():
            time.sleep(0.1)
        print(f"Downloading {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        process_queue.put(response.content)
        print(f"Downloaded and queued {url}")

def pdf_chunk(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

def processor(process_queue: Queue, chunk_queue: Queue):
    while True:
        pdf_content = process_queue.get()
        if pdf_content is None:
            break
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = pdf_chunk(pages)
        chunk_queue.put(chunks)
        os.remove(tmp_path)
        
def qdrant_writer(chunk_queue: Queue, collection_name="test", qdrant_url="http://localhost:6333"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_size = len(embeddings.embed_query("sample text"))
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    while True:
        chunks = chunk_queue.get()
        if chunks is None:
            break
        vector_store.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks to Qdrant.")

