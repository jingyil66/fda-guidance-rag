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
import boto3
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from langchain_core.documents import Document
import re, os, tempfile
from app.db.qdrant_client import init_qdrant

def downloader_from_s3(bucket_name, key_list, process_queue: Queue, aws_region="us-east-1"):
    s3_client = boto3.client("s3", region_name=aws_region)
    
    for key in key_list:
        while process_queue.full():
            time.sleep(0.1)
        print(f"Fetching {key} from S3...")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        pdf_bytes = response['Body'].read()
        process_queue.put((key, pdf_bytes))
        print(f"Fetched and queued {key}")

def pdf_chunk(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

def processor(process_queue: Queue, chunk_queue: Queue, pdf_metadata: dict):
    while True:
        item = process_queue.get()
        if item is None:
            break
        pdf_name, pdf_content = item  # S3 key
        
        match = re.search(r"(\d+)", pdf_name)
        media_id = match.group(1) if match else pdf_name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = []

        # 查找 JSON metadata
        meta = pdf_metadata.get(media_id, {})

        from bs4 import BeautifulSoup
        raw_title = meta.get("title", "")
        clean_title = BeautifulSoup(raw_title, "html.parser").get_text()

        for i, page_doc in enumerate(pages):
            page_meta = {
                "title": clean_title,
                "url": meta.get("url", ""),
                "summary": meta.get("summary", ""),
                "field_communication_type": meta.get("field_communication_type", ""),
                "pdf_id": media_id,
                "page": i + 1,
                "field_issue_datetime": meta.get("field_issue_datetime", ""),
                "field_center": meta.get("field_center", ""),
                "field_issuing_office_taxonomy": meta.get("field_issuing_office_taxonomy", ""),
                "term_node_tid": meta.get("term_node_tid", ""),
                "field_topics": meta.get("field_topics", ""),
                "topics_product": meta.get("topics-product", ""),
                "field_regulated_product_field": meta.get("field_regulated_product_field", ""),
                "changed": meta.get("changed", "")
            }
            doc = Document(page_content=page_doc.page_content, metadata=page_meta)
            chunks.append(doc)

        chunk_queue.put(chunks)
        os.remove(tmp_path)
        
def qdrant_writer(chunk_queue: Queue, collection_name="test", qdrant_url="http://localhost:6333", batch_size=50):
    vector_store = init_qdrant(collection_name, qdrant_url)
    processed_pdfs = 0
    buffer = []

    while True:
        chunks = chunk_queue.get()
        if chunks is None:
            # flush remaining
            if buffer:
                vector_store.add_documents(buffer)
                print(f"✅ Processed PDF: {processed_pdfs}, current number of chunks: {len(buffer)} (final batch)")
            break

        buffer.extend(chunks)
        processed_pdfs += 1

        if len(buffer) >= batch_size:
            vector_store.add_documents(buffer)
            print(f"✅ Processed PDF: {processed_pdfs}, current number of chunks: {len(buffer)}")
            buffer = []

    if buffer:
        vector_store.add_documents(buffer)
        print(f"✅ Processed PDF: {processed_pdfs}, current number of chunks: {len(buffer)} (final batch)")

