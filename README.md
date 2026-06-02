# FDA Guidance RAG

Question answering over [FDA medical guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents) PDFs: dense retrieval (OpenAI embeddings + Qdrant) → `gpt-4o-mini` generation with a strict-context prompt. Flask API and React UI included.

## Production defaults

| Setting | Value |
|--------|--------|
| Collection | `fda_guidance_chunk600_overlap200` (override with `QDRANT_COLLECTION`) |
| Chunking | fixed 600 chars / 200 overlap |
| Retrieval | embedding top-20 → top-5, **rerank off** (`rag_service.get_answer`) |
| Models | `text-embedding-3-small`, `gpt-4o-mini` |

Experiment work on a 200-PDF subset uses collection `experiment_subset200_chunk600_overlap200` and configs under `experiment/configs/` (final: `run_014_subset200_final_dev.json`, sealed test: `run_013_subset200_no_rerank_test.json`). Ablations and numbers: `experiment/registry.csv`, `experiment/REPORT_OUTLINE.md`.

**Subset eval (GPT judge, embedding-only, 600/200):** dev 30 pairs — e2e 0.83; sealed test 20 pairs — recall@5 0.90, e2e 0.70 (`run_013` in registry). Not full-corpus metrics.

## Layout

```
backend/app/          API, RAG pipeline, ETL (S3 → Qdrant)
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
# optional: QDRANT_COLLECTION, AWS for S3 ingest
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

`POST /ask` JSON: `{"query": "..."}`. `GET /health` checks Qdrant + collection.

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

## Ingest (optional)

Full pipeline (FDA metadata → S3 → Qdrant), from repo root:

```bash
python -m backend.app.fetchers.fda_fetcher
python -m backend.app.etl.download_to_s3
python -m backend.app.etl.initial_data_ingestion.py
```

Local PDFs only: `python experiment/scripts/ingest_local_pdfs.py` (e.g. `data/`, `data/subset_200/`).

Staged ingest with checkpoints: `jobs/ingest_stages.py`.

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
