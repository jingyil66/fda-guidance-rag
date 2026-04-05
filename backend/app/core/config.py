import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    DATA_DIR = BASE_DIR / "data"
    PDF_SAVE_DIR = DATA_DIR / "pdfs"
    
    PDF_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents",
    }
    METADATA_URL = "https://www.fda.gov/files/api/datatables/static/search-for-guidance.json"
    OUTPUT_METADATA_JSON = DATA_DIR / "metadata_with_summary.json"

    # --- API Keys ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")

    # --- AWS & Cloud Storage ---
    BUCKET_NAME = os.getenv("BUCKET_NAME", "04-bucket")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    def validate(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file!")

settings = Settings()

