import os
import json
import re
from multiprocessing import Process, Queue
from threading import Thread

import boto3

from backend.app.core.config import OPENAI_API_KEY, OUTPUT_METADATA_JSON
from backend.app.etl.ingest_to_qdrant import downloader_from_s3, processor, qdrant_writer
from backend.app.services.chunking_service import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    collection_name = os.environ.get(
        "QDRANT_COLLECTION",
        "experiment_chunk600_overlap200",
    )
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    chunk_strategy = os.environ.get("CHUNK_STRATEGY", "fixed")
    chunk_size = int(os.environ.get("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))

    process_queue = Queue(maxsize=5)
    chunk_queue = Queue(maxsize=20)

    writer_thread = Thread(
        target=qdrant_writer,
        args=(chunk_queue,),
        kwargs={
            "collection_name": collection_name,
            "qdrant_url": qdrant_url,
        },
    )
    writer_thread.start()

    with open(OUTPUT_METADATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_metadata = {}
    for meta in data:
        media_link = meta.get("field_associated_media_2", "")
        match = re.search(r"/media/(\d+)/download", media_link)
        if match:
            media_id = match.group(1)
            pdf_metadata[media_id] = meta

    s3_bucket = os.environ.get("S3_BUCKET", "04-bucket")
    s3_keys = []
    s3_client = boto3.client("s3")
    continuation_token = None

    while True:
        if continuation_token:
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix="pdfs/",
                ContinuationToken=continuation_token,
            )
        else:
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix="pdfs/",
            )

        for obj in response.get("Contents", []):
            s3_keys.append(obj["Key"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    print(f"{len(s3_keys)} PDFs are found.")
    print(
        f"Ingest config: collection={collection_name}, strategy={chunk_strategy}, "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    processor_kwargs = {
        "bucket_name": s3_bucket,
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    procs = []
    num_workers = 3
    for _ in range(num_workers):
        p = Process(
            target=processor,
            args=(process_queue, chunk_queue, pdf_metadata),
            kwargs=processor_kwargs,
        )
        p.start()
        procs.append(p)

    downloader_from_s3(
        bucket_name=s3_bucket,
        key_list=s3_keys,
        process_queue=process_queue,
    )

    for _ in procs:
        process_queue.put(None)

    for p in procs:
        p.join()

    chunk_queue.put(None)
    writer_thread.join()

    print("All done")
