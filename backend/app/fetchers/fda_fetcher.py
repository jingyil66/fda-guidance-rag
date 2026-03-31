import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from lxml import html
from asyncio import Semaphore

class FDAMetadataClient:
    def __init__(self, headers, url):
        self.headers = headers
        self.url = url

    def fetch(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch FDA metadata: {e}")
            return []
        
class FDASummaryScraper:
    def __init__(self, headers, concurrency=5):
        self.headers = headers
        self.semaphore = Semaphore(concurrency)

    async def fetch_summary(self, session, item):
        async with self.semaphore:
            soup = BeautifulSoup(item["title"], "html.parser")
            a_tag = soup.find("a")
            if not a_tag or not a_tag.get("href"):
                item["summary"] = ""
                return item

            url = "https://www.fda.gov" + a_tag["href"]
            item["url"] = url

            try:
                async with session.get(url, headers=self.headers) as resp:
                    content = await resp.read()
                    tree = html.fromstring(content)
                    text = tree.xpath("//main//article//text()")
                    item["summary"] = "\n".join([t.strip() for t in text if t.strip()])
            except Exception:
                item["summary"] = ""
        return item

    async def run(self, metadata_list):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_summary(session, m) for m in metadata_list]
            results = []
            for idx, t in enumerate(asyncio.as_completed(tasks), 1):
                item = await t
                results.append(item)
                print(f"Processed {idx}/{len(metadata_list)}")
            return results        