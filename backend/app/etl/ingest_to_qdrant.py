import os
import time
import tempfile
from multiprocessing import Queue
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import boto3
from langchain_core.documents import Document
import re, os, tempfile
from backend.app.db.qdrant_client import init_qdrant
import openai

def downloader_from_s3(bucket_name, key_list, process_queue: Queue, aws_region="us-east-1"):
    s3_client = boto3.client("s3", region_name=aws_region)
    
    for key in key_list:
        while process_queue.full():
            time.sleep(0.1)
        print(f"Queueing {key}...")
        process_queue.put(key)

def pdf_chunk(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

def processor(process_queue: Queue, chunk_queue: Queue, pdf_metadata: dict):
    import boto3
    s3_client = boto3.client("s3")

    while True:
        item = process_queue.get()
        if item is None:
            break
        pdf_name = item  # S3 key
        
        response = s3_client.get_object(Bucket="04-bucket", Key=pdf_name)
        pdf_content = response['Body'].read()

        match = re.search(r"(\d+)", pdf_name)
        media_id = match.group(1) if match else pdf_name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

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
            while chunk_queue.full():
                time.sleep(0.1)

            chunk_queue.put([doc])

        os.remove(tmp_path)
        
def qdrant_writer(chunk_queue: Queue, collection_name="test", qdrant_url="http://localhost:6333", batch_size=10):
    vector_store = init_qdrant(collection_name, qdrant_url)
    processed_chunks = 0
    processed_pdfs = set()
    buffer = []

    def safe_add_documents(vec_store, docs):
        safe_docs = []
        for doc in docs:
            safe_meta = {k: str(v) if v is not None else "" for k, v in doc.metadata.items()}
            safe_docs.append(Document(page_content=str(doc.page_content), metadata=safe_meta))

        while True:
            try:
                vec_store.add_documents(safe_docs)
                break
            except openai.error.RateLimitError as e:
                print(f"⚠️ Rate limit reached, waiting 1s... ({e})", flush=True)
                time.sleep(1)

    while True:
        chunks = chunk_queue.get()
        if chunks is None:
            if buffer:
                safe_add_documents(vector_store, buffer)
                print(f"✅ PDFs: {len(processed_pdfs)}, chunks: {processed_chunks}, batch size: {len(buffer)} (final batch)", flush=True)
            break

        buffer.extend(chunks)
        processed_chunks += len(chunks)
        for c in chunks:
            processed_pdfs.add(c.metadata.get("pdf_id"))

        if len(buffer) >= batch_size:
            safe_add_documents(vector_store, buffer)
            print(f"✅ PDFs: {len(processed_pdfs)}, chunks: {processed_chunks}, batch size: {len(buffer)}")
            buffer = []

    if buffer:
        safe_add_documents(vector_store, buffer)
        print(f"✅ PDFs: {len(processed_pdfs)}, chunks: {processed_chunks}, batch size: {len(buffer)} (final batch)")