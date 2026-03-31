# services/metadata_service.py
import os
import json
import asyncio
from app.fetchers.fda_fetcher import FDAMetadataClient, FDASummaryScraper
from app.core.config import METADATA_URL, OUTPUT_METADATA_JSON, HEADERS

class FDAWorkflow:
    def __init__(self, headers, metadata_url, metadata_path, concurrency=5):
        self.headers = headers
        self.metadata_url = metadata_url
        self.metadata_path = metadata_path

        self.client = FDAMetadataClient(headers, metadata_url)
        self.scraper = FDASummaryScraper(headers, concurrency)

    async def prepare_metadata(self, force_refresh=False):
        if force_refresh or not os.path.exists(self.metadata_path):
            metadata = self.client.fetch()
            metadata = await self.scraper.run(metadata)
            self._save_metadata(metadata)
        else:
            metadata = self._load_metadata()

        self.metadata = metadata
        print(f"Metadata ready: {len(metadata)} items")

    def _save_metadata(self, metadata):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_metadata(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)