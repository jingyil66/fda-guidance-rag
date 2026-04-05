import os
import requests
from bs4 import BeautifulSoup
import json
from backend.app.core.config import HEADERS, OUTPUT_METADATA_JSON

LOCAL_DIR = "./data"
MAX_DOWNLOAD = 50
os.makedirs(LOCAL_DIR, exist_ok=True)

def download_pdfs(url_list):
    for idx, url in enumerate(url_list, 1):
        try:
            print(f"[{idx}/{len(url_list)}] Downloading {url}...")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()

            filename = url.split("/")[-2] + ".pdf"
            local_path = os.path.join(LOCAL_DIR, filename)

            if os.path.exists(local_path):
                print(f"Skipped (already exists): {filename}")
                continue

            with open(local_path, "wb") as f:
                f.write(response.content)

            print(f"Saved: {local_path}")

        except Exception as e:
            print(f"Failed {url}: {e}")

    print("Done! PDFs downloaded to local folder.")

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

    url_list = list(set(url_list))
    url_list = url_list[:MAX_DOWNLOAD]

    download_pdfs(url_list)