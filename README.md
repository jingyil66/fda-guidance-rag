# FDA Guidance RAG

Question answering over [FDA medical guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents) PDFs: **fixed RAG** (OpenAI embeddings + Qdrant → `gpt-4o-mini` with a strict-context prompt) and a **tool-calling agent** (`search_guidance`, `list_guidance`, `get_guidance_detail`). Flask API, React UI (toggle Fixed RAG vs Agent), and MCP stdio server included.

## Production defaults

| Setting | Value |
|--------|--------|
| Collection | `fda_guidance_chunk600_overlap200` (override with `QDRANT_COLLECTION`) |
| Chunking | fixed 600 chars / 200 overlap |
| Retrieval | embedding top-20 → top-5, **rerank off** (`rag_service.get_answer`) |
| Models | `text-embedding-3-small`, `gpt-4o-mini` |
| Agent | `gpt-4o-mini`, max 6 LLM turns (`get_agent_answer`) |

Experiment work on a 200-PDF subset uses collection `experiment_subset200_chunk600_overlap200` and configs under `experiment/configs/` (final: `run_014_subset200_final_dev.json`, sealed test: `run_013_subset200_no_rerank_test.json`). Ablations and numbers: [`experiment/REPORT.md`](experiment/REPORT.md), [`experiment/registry.csv`](experiment/registry.csv).

**Subset eval (GPT judge, embedding-only, 600/200):** dev 30 pairs — e2e 0.83; sealed test 20 pairs — recall@5 0.90, e2e 0.70 (`run_013` in registry). Not full-corpus metrics.

## Layout

```
backend/app/          API, RAG pipeline, agent, ETL (S3 → Qdrant)
backend/mcp/          MCP server (stdio); same tools as /ask_agent
experiment/           RAG/agent eval, compare_ask_vs_agent, configs, smoke scripts
evaluation/           QA dataset tooling (`qa_dataset.json`)
tests/                pytest (unit + integration; agent evaluators, MCP)
frontend/fda-app/     React + Vite UI (Fixed RAG / Agent mode toggle)
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

**Agent note:** `search_guidance` needs ingested vectors in Qdrant. `list_guidance` and `get_guidance_detail` also need `data/metadata_with_summary.json` (see [Ingest → Metadata](#ingest-optional)).

**5. Full stack (Docker Compose)**

```bash
docker compose up -d --build
```

- UI: http://localhost:8080 — toggle **Fixed RAG** vs **Agent**; nginx proxies `/api/` → backend (`VITE_API_BASE=/api` at build)  
- API: http://127.0.0.1:5000  
- Qdrant dashboard: http://localhost:6333/dashboard  

Edit the Qdrant volume in `docker-compose.yml` if your data path differs. Host-side smoke: `QDRANT_URL=http://localhost:6333`.

**6. Frontend dev** (no Docker)

```bash
cd frontend/fda-app && npm install && npm run dev
```

http://localhost:5173 → API base `http://127.0.0.1:5000` by default (`VITE_API_BASE` override). UI toggles **Fixed RAG** (`/ask`) vs **Agent** (`/ask_agent`); choice is saved in `localStorage`.

## MCP server (stdio)

Exposes the same tools as `/ask_agent` for Cursor, Claude Desktop, or other MCP clients:

| Tool | Purpose |
|------|---------|
| `search_guidance` | Semantic search over Qdrant chunks |
| `list_guidance` | Filter guidance catalog metadata |
| `get_guidance_detail` | Summary for one document by `pdf_id` or title |

**Run** (repo root; Qdrant + `.env` required for search; **`data/metadata_with_summary.json`** required for `list_guidance` / `get_guidance_detail`):

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

**Agent eval (tool routing + task E2E, dev gold):**

```bash
# Full: 32 tool + 20 task rows, failure taxonomy A1–A5
python experiment/run_agent_eval.py --config experiment/configs/run_agent_eval_full_dev.json --preview

# Tool routing only
python experiment/run_agent_eval.py --config experiment/configs/run_agent_tool_dev.json --limit 3 --skip-task-eval

# Fixed RAG vs agent (same questions; 0 = all 20 task dev rows)
python experiment/compare_ask_vs_agent.py --limit 0

# LangSmith traces (optional)
# set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY
```

**Published dev metrics (full production Qdrant, 2026-06-03):**

- **Agent eval** (`run_agent_eval_full_dev`): tool pass 59%, task success 65%, tool selection accuracy 91% (32 tool + 20 task).
- **Fixed RAG vs agent** (`agent_task_gold_dev`, n=20): avg latency RAG 3.9s vs agent 4.2s (+9%); GPT judge on 14 gold rows — correctness/groundedness tied (0.71 / 0.86), agent relevance higher (0.86 vs 0.71). Agent wins on list/browse latency; fixed RAG wins on pure search Q&A reliability.

Docs: [`experiment/AGENT_EVAL.md`](experiment/AGENT_EVAL.md) (metrics, datasets, failure codes), [`experiment/AGENT_FAILURE_POSTMORTEM.md`](experiment/AGENT_FAILURE_POSTMORTEM.md) (typical A1/A2/empty-retrieval cases).

## Tests

| Command | Needs |
|---------|--------|
| `pytest` | unit only (default); includes `test_agent_evaluators`, `test_mcp_server` |
| `pytest -m integration` | live Qdrant + `OPENAI_API_KEY` |

CI: `.github/workflows/tests.yml` (unit on push/PR).
