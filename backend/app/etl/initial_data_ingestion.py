import os
import json
from backend.app.etl.ingest_to_qdrant import downloader_from_s3, processor, qdrant_writer
from backend.app.core.config import OUTPUT_METADATA_JSON, OPENAI_API_KEY
from multiprocessing import Process, Queue
from threading import Thread
import boto3

if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    process_queue = Queue(maxsize=5)
    chunk_queue = Queue(maxsize=20)

    writer_thread = Thread(target=qdrant_writer, args=(chunk_queue,))
    writer_thread.start()

    with open(OUTPUT_METADATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    import re
    pdf_metadata = {}
    for meta in data:
        media_link = meta.get("field_associated_media_2", "")
        match = re.search(r"/media/(\d+)/download", media_link)
        if match:
            media_id = match.group(1)
            pdf_metadata[media_id] = meta
    
    s3_bucket = "04-bucket"
    s3_keys = []
    s3_client = boto3.client("s3")
    continuation_token = None

    while True:
        if continuation_token:
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix="pdfs/",
                ContinuationToken=continuation_token
            )
        else:
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix="pdfs/"
            )

        for obj in response.get("Contents", []):
            s3_keys.append(obj["Key"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    print(f"{len(s3_keys)} PDFs are found.")

    # test
    # s3_keys = ["pdfs/91343"]

    procs = []
    num_workers = 3
    for _ in range(num_workers):
        p = Process(target=processor, args=(process_queue, chunk_queue, pdf_metadata))
        p.start()
        procs.append(p)

    downloader_from_s3(
        bucket_name=s3_bucket,
        key_list=s3_keys,
        process_queue=process_queue
    )

    for _ in procs:
        process_queue.put(None)

    for p in procs:
        p.join()

    chunk_queue.put(None)
    writer_thread.join()

    print("All done")
