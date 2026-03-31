from multiprocessing import Process, Queue, cpu_count
import requests
from bs4 import BeautifulSoup
import boto3
import time
from app.core.config import HEADERS, OUTPUT_METADATA_JSON
import json

s3 = boto3.client("s3")
BUCKET_NAME = "04-bucket"

def download_and_upload(url_list):
    for url in url_list:
        try: 
            print(f"Downloading {url}...")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            filename = url.split("/")[-2]
            s3_key = f"pdfs/{filename}"

            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=response.content,
                ContentType="application/pdf"
            )
            print(f"Uploaded {s3_key} to S3")

        except Exception as e:
            print(f"Failed {url}: {e}")

    print(f"Uploaded successfully!")

if __name__ == "__main__":
    with open(OUTPUT_METADATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    url_list = []
    for r in data:
        soup = BeautifulSoup(r['field_associated_media_2'], 'html.parser')
        a_tag = soup.find('a')
        if a_tag and a_tag.get('href'):
            pdf_url = "https://www.fda.gov" + a_tag['href']
            url_list.append(pdf_url)

    num_processes = min(cpu_count(), len(url_list))
    chunk_size = len(url_list) // num_processes + 1
    chunks = [url_list[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]

    processes = []
    for chunk in chunks:
        p = Process(target=download_and_upload, args=(chunk,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        