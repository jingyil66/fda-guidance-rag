import os
import json
import asyncio
from ingest_to_qdrant import downloader, processor, qdrant_writer
from config import HEADERS, OUTPUT_METADATA_JSON, METADATA_URL, OPENAI_API_KEY
from metadata_pipeline import FDAWorkflow
from multiprocessing import Process, Queue, cpu_count
from bs4 import BeautifulSoup
from threading import Thread

if __name__ == "__main__":
    workflow = FDAWorkflow(
        headers=HEADERS,
        metadata_url=METADATA_URL,
        metadata_path=OUTPUT_METADATA_JSON,
        concurrency=5,
    )

    asyncio.run(workflow.prepare_metadata(force_refresh=False))

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    process_queue = Queue(maxsize=5)
    chunk_queue = Queue(maxsize=10)

    writer_thread = Thread(target=qdrant_writer, args=(chunk_queue,))
    writer_thread.start()

    with open(OUTPUT_METADATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    url_list = []
    for r in data:
        soup = BeautifulSoup(r['field_associated_media_2'], 'html.parser')
        a_tag = soup.find('a')
        if a_tag and a_tag.get('href'):
            pdf_url = "https://www.fda.gov" + a_tag['href']
            url_list.append(pdf_url)
    
    url_list = url_list[:10]

    procs = []
    for _ in range(min(cpu_count(), len(url_list))):
        p = Process(target=processor, args=(process_queue, chunk_queue))
        p.start()
        procs.append(p)

    downloader(url_list=url_list, process_queue=process_queue, headers=HEADERS)

    for _ in procs:
        process_queue.put(None)

    for p in procs:
        p.join()

    chunk_queue.put(None)
    writer_thread.join()

    print("All done")
