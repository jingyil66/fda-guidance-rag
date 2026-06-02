# FDA Guidance RAG

Question answering over [FDA medical guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents) PDFs: dense retrieval (OpenAI embeddings + Qdrant) → `gpt-4o-mini` generation with a strict-context prompt. Flask API and React UI included.

## Production defaults

| Setting | Value |
|--------|--------|
| Collection | `fda_guidance_chunk600_overlap200` (override with `QDRANT_COLLECTION`) |
| Chunking | fixed 600 chars / 200 overlap |
| Retrieval | embedding top-20 → top-5, **rerank off** (`rag_service.get_answer`) |
| Models | `text-embedding-3-small`, `gpt-4o-mini` |

Experiment work on a 200-PDF subset uses collection `experiment_subset200_chunk600_overlap200` and configs under `experiment/configs/` (final: `run_014_subset200_final_dev.json`, sealed test: `run_013_subset200_no_rerank_test.json`). Ablations and numbers: [`experiment/REPORT.md`](experiment/REPORT.md), [`experiment/registry.csv`](experiment/registry.csv).

**Subset eval (GPT judge, embedding-only, 600/200):** dev 30 pairs — e2e 0.83; sealed test 20 pairs — recall@5 0.90, e2e 0.70 (`run_013` in registry). Not full-corpus metrics.

## Layout

```
backend/app/          API, RAG pipeline, ETL (S3 → Qdrant)
backend/mcp/          MCP server (stdio) for IDE clients
experiment/           run configs, evaluators, ingest/smoke scripts
evaluation/           QA dataset tooling (`qa_dataset.json`)
tests/                pytest (unit + integration)
frontend/fda-app/     React + Vite UI
jobs/                 staged ingest / benchmarks (optional)
```

## Quick start

**Prerequisites:** Python 3.11+, Docker (Qdrant), Node 18+ (frontend dev only).

**1. Env** — project-root `.env`:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
# optional: QDRANT_COLLECTION
# ingest via S3 (see below): AWS credentials + BUCKET_NAME
```

**2. Install & test**

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r backend/requirements.txt -r requirements-dev.txt
pytest
```

**3. Qdrant**

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**4. API** (from repo root, after vectors are ingested)

```bash
python -m backend.app.main
# smoke (collection + pipeline, optional /ask):
python experiment/scripts/smoke_production.py
python experiment/scripts/smoke_production.py --api-url http://127.0.0.1:5000/ask
```

`POST /ask` JSON: `{"query": "..."}` — fixed RAG pipeline.  
`POST /ask_agent` JSON: `{"query": "..."}` — tool-calling agent (`search_guidance`, `list_guidance`, `get_guidance_detail`); response includes `steps` trace.  
`GET /health` checks Qdrant + collection.

**5. Full stack (Docker Compose)**

```bash
docker compose up -d --build
```

- UI: http://localhost:8080  
- API: http://127.0.0.1:5000  
- Qdrant dashboard: http://localhost:6333/dashboard  

Edit the Qdrant volume in `docker-compose.yml` if your data path differs. Host-side smoke: `QDRANT_URL=http://localhost:6333`.

**6. Frontend dev** (no Docker)

```bash
cd frontend/fda-app && npm install && npm run dev
```

http://localhost:5173 → `http://127.0.0.1:5000/ask` by default.

## MCP server (stdio)

Exposes the same tools as `/ask_agent` for Cursor, Claude Desktop, or other MCP clients:

| Tool | Purpose |
|------|---------|
| `search_guidance` | Semantic search over Qdrant chunks |
| `list_guidance` | Filter guidance catalog metadata |
| `get_guidance_detail` | Summary for one document by `pdf_id` or title |

**Run** (repo root; Qdrant + `.env` required for search):

```bash
python -m backend.mcp.fda_guidance_server
```

**Cursor** — add to MCP settings (use your absolute paths):

```json
{
  "mcpServers": {
    "fda-guidance-rag": {
      "command": "D:/winter_2025/fda_guidance_rag/venv/Scripts/python.exe",
      "args": ["-m", "backend.mcp.fda_guidance_server"],
      "cwd": "D:/winter_2025/fda_guidance_rag",
      "env": {
        "OPENAI_API_KEY": "your-key",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Demo prompts**

1. *List guidances about adaptive design* → `list_guidance`
2. *What does FDA say about Phase 2 trial design?* → `search_guidance`
3. *Summarize guidance pdf_id=122971* → `get_guidance_detail` then optional `search_guidance` with `pdf_id`
4. *CDER final guidances from 2024* → `list_guidance` with filters
5. *Compare REMS requirements* → `list_guidance` then `search_guidance` on chosen `pdf_id`

## Ingest (optional)

`data/` is gitignored. Full-corpus ingest expects:

- **`data/metadata_with_summary.json`** — FDA guidance catalog + scraped summaries (see step 1).
- **AWS credentials** in the environment and `BUCKET_NAME` (or `S3_BUCKET`) in `.env` for PDF upload / S3-backed stages.
- **`OPENAI_API_KEY`** for embedding.

**1. Metadata** (once; writes `data/metadata_with_summary.json`), from repo root:

```bash
python -c "import asyncio; from backend.app.core.config import settings; from backend.app.services.metadata_service import FDAWorkflow; w = FDAWorkflow(settings.HEADERS, settings.METADATA_URL, settings.OUTPUT_METADATA_JSON); asyncio.run(w.prepare_metadata(force_refresh=True))"
```

**2. Staged ingest (recommended)** — download → chunk → embed with checkpoints:

```bash
python jobs/ingest_stages.py --stage all
# optional: --cache-pdfs (local PDF cache), --limit N, --collection NAME
```

**Alternative (monolithic S3 → Qdrant)** after metadata exists:

```bash
python -m backend.app.etl.download_to_s3
python -m backend.app.etl.initial_data_ingestion
```

**Local PDFs only** (no S3): `python experiment/scripts/ingest_local_pdfs.py` (e.g. `data/`, `data/subset_200/`).

## Experiment runner

```bash
python experiment/run_experiment.py --config experiment/configs/run_014_subset200_final_dev.json --limit 3
```

Runs write under `experiment/runs/` (gitignored).

## Tests

| Command | Needs |
|---------|--------|
| `pytest` | unit only (default) |
| `pytest -m integration` | live Qdrant + `OPENAI_API_KEY` |

CI: `.github/workflows/tests.yml` (unit on push/PR).
