import os

from backend.app.core.config import settings
from backend.app.etl.ingest_pipeline import run_s3_ingest

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

    run_s3_ingest(
        collection_name=os.environ.get("QDRANT_COLLECTION", settings.DEFAULT_QDRANT_COLLECTION),
        qdrant_url=settings.DEFAULT_QDRANT_URL,
        bucket_name=os.environ.get("S3_BUCKET", settings.BUCKET_NAME),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "600")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
        num_workers=int(os.environ.get("INGEST_WORKERS", "3")),
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "64")),
        embed_parallelism=int(os.environ.get("EMBED_PARALLELISM", "1")),
        reset_checkpoint=os.environ.get("RESET_CHECKPOINT", "").lower() in {"1", "true", "yes"},
        retry_failed=os.environ.get("RETRY_FAILED", "").lower() in {"1", "true", "yes"},
        limit=int(os.environ.get("INGEST_LIMIT", "0")),
    )
