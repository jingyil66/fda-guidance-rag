# FDA Guidance Search & RAG Chatbot

FDA-RAG-Assistant is a high-precision, Two-Stage Retrieval-Augmented Generation (RAG) system designed for interactive querying of 2,000+ FDA Medical Guidance documents. By implementing advanced semantic re-ranking and an automated evaluation pipeline, the system resolves "semantic drift" in dense regulatory texts, ensuring responses are both grounded and contextually accurate.

## Dataset

The chatbot uses FDA medical guidance documents as its primary dataset:

- **Source:** [FDA official website](https://www.fda.gov/regulatory-information/search-fda-guidance-documents)

## Engineering Highlights

**1. Scalable Asynchronous ETL Pipeline**

- **The Challenge:** Processing 2,000+ regulatory PDF documents from the FDA website is both I/O bound (downloading) and CPU bound (parsing).

- **The Solution:** Developed an automated ingestion pipeline using a **Hybrid Concurrency Model**.

   - **Async Harvesting:** Used asyncio with **Semaphores** for metadata harvesting to prevent IP rate-limiting.

   - **Multi-processing:** Leveraged Python’s multiprocessing to bypass the **GIL**, parallelizing PDF text extraction and **Recursive Character Splitting** to maintain logical continuity.

**The Impact:** Reduced total indexing time by **75%** while ensuring 100% metadata alignment between the API and vector store.

**2. Two-Stage Retrieval Pipeline (Reranking)**

- **Precision:** Integrated **FlashRank (Cross-Encoder)** to re-score the top-20 candidates from the initial vector search.

- **The Logic:** This architecture significantly improves precision when distinguishing between highly similar regulatory clauses (e.g., distinguishing between Phase 1 vs. Phase 2 clinical trial requirements), which often confuse standard Bi-Encoder similarity searches.

**Result:** Optimized the context window for the LLM by delivering high-density, reranked information, leading to superior generation accuracy.

**3. Metric-Driven Evaluation (MLOps)**

- **The Framework:** Built a robust evaluation suite using **LangSmith** to monitor RAG performance in real-time.

- **Automated QA:** Implemented a **GPT-4o-mini** as a Judge suite, benchmarking the system against a curated **50+ Golden Dataset**.

- **Key Metrics:** Achieved high-tier production scores: 0.84 Correctness, 0.88 Groundedness, and 1.00 Retrieval Relevance.

**4. Cloud-Native Architecture & Memory Management**

- **Persistence:** Engineered a robust uploader using boto3 to stream PDF binaries directly to Amazon S3, creating a scalable Data Lake.

- **Resilience:** Implemented Memory Backpressure using bounded queues and Exponential Backoff to handle OpenAI API rate limits, ensuring the system remains stable under heavy ingestion loads.

---

## Performance Benchmarks

The system achieves "production-ready" scores across all critical RAG metrics:

| Metric | Score (AVG) | Description |
| :--- | :--- | :--- |
| **Correctness** | **0.84** | Accuracy of the answer compared to the ground truth. |
| **Groundedness** | **0.88** | Ability to stay strictly within the provided context (No Hallucination). |
| **Relevance** | **0.88** | How well the answer addresses the user's specific query. |
| **Retrieval Relevance** | **1.00** | Perfect recall: the correct source was found in the retrieval stage 100% of the time. |

## System Architecture

```
backend/
├── app/api/routes.py             # Flask routes (RESTful API endpoints)
├── core/config.py         # Global configurations & environment variables
├── db/qdrant_client.py    # Qdrant vector store initialization & collection management
├── etl/                   # Data Engineering Pipeline
│   ├── download_to_s3.py         # AWS S3 document persistence
│   ├── ingest_to_qdrant.py       # Vector embedding & indexing logic
│   └── initial_data_ingestion.py # Full-cycle ingestion entry point
├── fetchers/fda_fetcher.py# Asynchronous metadata harvesting from FDA APIs
├── services/              # Business Logic Layer
│   ├── metadata_service.py       # Document attribute & filter management
│   └── rag_service.py            # Core RAG engine (Retrieval + Re-ranking + LLM)
└── main.py                # Backend entry point (Flask)

evaluation/                # LangSmith metrics & Golden Dataset
experiment/                # Sandbox for testing new reranking models & splitters
test/                      # Unit tests & Qdrant maintenance scripts
tests/                     # Pytest suite (unit + integration)
```

## Technical Stack

### **Core Architecture**

- **Language**: Python 3.10+
- **Orchestration**: **LangChain** (Chains, Document Loaders, Recursive Splitters)
- **Vector Database**: **Qdrant** (High-performance vector search & collection management)

### **AI & Retrieval Engineering**

- **LLM**: OpenAI **GPT-4o-mini** (Optimized with structured system prompts)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dimensional vectors)
- **Reranker**: **FlashRank** (Cross-Encoder for mitigating semantic drift)

### **Data Engineering (ETL)**

- **Pipeline**: Asynchronous PDF Ingestion for 2,700+ FDA guidance documents.
- **Preprocessing**: **Recursive Character Text Splitting** (Chunk: 600, Overlap: 200).
- **Metadata**: Automated indexing using official FDA Regulatory APIs.

### **MLOps & Quality Assurance**

- **Evaluation**: **LangSmith** (Trace logging, RAG benchmarking, and GPT-4o-as-a-Judge).
- **Version Control**: Git.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm
- Docker (for running Qdrant)

### Backend Setup
**1. Environment Setup**

Clone the repository and initialize the Python environment.
It is highly recommended to use a virtual environment to avoid dependency conflicts:
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**2. Secret Management:**
To ensure security and modularity, the system uses environment variables. Do not hardcode your API keys in config.py.

Create a .env file in the backend/ root directory, add the following configuration to your .env file:
```
# OpenAI API Key for Embeddings and Generation
OPENAI_API_KEY=sk-xxxx...

# LangSmith Configuration (Optional but recommended for evaluation)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxx...
LANGCHAIN_PROJECT=FDA-RAG-Assistant

# Vector Database Configuration
QDRANT_URL=http://localhost:6333
```

**3. Infrastructure (Qdrant Vector Database)**

For local development only, you can run Qdrant alone:

```
docker pull qdrant/qdrant
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

For the **production demo** (Qdrant + API together), use [Docker Compose](#production-demo-docker-compose) instead.

**4. The ETL Pipeline (Data Ingestion)**
This is a three-stage process to move data from the FDA's servers to your Cloud RAG engine.

**Stage A: Metadata Harvesting**
Scrapes official FDA APIs and enriches document summaries using asynchronous workers.

```
python -m app.fetchers.fda_fetcher
```

**Stage B: Cloud Persistence (S3)**
Uses Multi-processing to parallelize the downloading and uploading of 2,000+ PDFs to Amazon S3.

```
python -m app.etl.download_to_s3
```

**Stage C: Vector Indexing (Qdrant)**
The core engine: utilizes a Producer-Consumer pattern to pull from S3, perform recursive chunking, and batch-upload embeddings to Qdrant.

Production defaults (from `backend/app/core/config.py`):

- Collection: `fda_guidance_chunk600_overlap200`
- Chunking: fixed 600 / 200
- Retrieval: embedding top-20 → top-5, **no rerank**

```
# From project root (Windows)
set QDRANT_COLLECTION=fda_guidance_chunk600_overlap200
venv\Scripts\python.exe backend\app\etl\initial_data_ingestion.py

# Local PDFs only (data/ + data/subset_200/)
venv\Scripts\python.exe experiment\scripts\ingest_local_pdfs.py
```

**5. Production smoke test & API (local dev)**

```
venv\Scripts\python.exe experiment\scripts\smoke_production.py
venv\Scripts\python.exe -m backend.app.main
# optional second terminal:
venv\Scripts\python.exe experiment\scripts\smoke_production.py --api-url http://127.0.0.1:5000/ask
```

Experiment ablations use separate collections under `experiment/configs/` (e.g. `experiment_subset200_chunk600_overlap200`).

### Production demo: Docker Compose

After full-corpus ingest into Qdrant, run the production stack (Qdrant + Flask API + React UI) with one command. Data is read from the existing volume on `D:/qdrant/storage` (~252k chunks in `fda_guidance_chunk600_overlap200`).

**Prerequisites**

- Docker Desktop
- Project-root `.env` with `OPENAI_API_KEY` (see step 2 above)
- Qdrant storage already populated (see **Stage C**). Default bind mount: `D:/qdrant/storage`
- Port **6333**, **5000**, and **8080** free (stop any standalone `docker run qdrant` container first)

**Start**

From the project root:

```powershell
docker compose up -d --build
docker compose ps
```

- **UI:** http://localhost:8080 (Nginx serves the React app; `/api/*` proxies to the backend)
- **Qdrant:** http://localhost:6333/dashboard  
- **API (direct):** http://127.0.0.1:5000/health and `POST` http://127.0.0.1:5000/ask with JSON `{"query": "..."}`  
- Inside the backend container, `QDRANT_URL` is `http://qdrant:6333` (set in `docker-compose.yml`). Do **not** put that hostname in `.env` if you run smoke tests from the host.

**Verify**

Run smoke from the host with `localhost` for Qdrant:

```powershell
$env:QDRANT_URL='http://localhost:6333'
venv\Scripts\python.exe experiment\scripts\smoke_production.py --min-points 250000 --api-url http://127.0.0.1:5000/ask
curl.exe -s http://127.0.0.1:5000/health
```

Open http://localhost:8080 and ask a question in the UI.

For local frontend development without Docker (hot reload):

```powershell
cd frontend\fda-app
npm install
npm run dev
```

Open http://localhost:5173 (calls `http://127.0.0.1:5000/ask` by default).

Stop the stack:

```powershell
docker compose down
```

**6. Application Launch (local dev, without Docker)**
Backend (Flask API)
Starts the RAG service, including the FlashRank re-ranker and LangChain orchestration.

```
python main.py
```

Frontend (React)
Navigate to the frontend directory and start the dev server:

```
cd ../frontend
npm install
npm run dev
```

The assistant will be available at http://localhost:5173.

## Future Improvements

**1. Advanced Data Engineering & Orchestration**

- [ ] Automated Pipeline Orchestration: Transition from manual scripts to Apache Airflow or Prefetch to schedule incremental FDA updates and handle retries/monitoring automatically.

- [ ] Change Data Capture (CDC): Implement a hashing mechanism to detect updates in FDA guidance PDFs, ensuring only modified documents are re-processed to save OpenAI embedding costs.

- [ ] Vision-Language Integration: Incorporate Unstructured.io or LayoutLM to parse complex tables and decision flowcharts within FDA PDFs, which are currently treated as plain text.

**2. Retrieval & LLM Optimization**

- [ ] Hybrid Search Implementation: Combine Qdrant's Dense Retrieval (Semantic) with Sparse Retrieval (BM25/Keyword) to improve accuracy for specific regulatory terms and document IDs.

- [ ] Dynamic Context Window: Implement a "Long-Context" strategy using Map-Reduce chains for queries that require summarizing multiple guidance documents simultaneously.

**3. Evaluation & MLOps**

- [ ] A/B Testing Suite: Use LangSmith to run side-by-side comparisons of different chunking strategies (e.g., Fixed-size vs. Semantic Splitting) and different LLM backends.

- [ ] Human-in-the-Loop (HITL): Build a feedback loop where domain experts can "upvote/downvote" answers, using this labeled data to fine-tune a specialized Small Language Model (SLM) for FDA compliance.

**4. Scalability & Security**

- [ ] Multi-modal Persistence: Migrate metadata from JSON to a relational database (e.g., PostgreSQL/pgvector) for more complex relational filtering (e.g., "Find all Phase 3 guidelines issued after 2024").

- [ ] VPC & Private Endpoints: Secure the S3-to-Qdrant data flow within an AWS VPC to simulate enterprise-grade security requirements for sensitive pharmaceutical data.

- [x] Full-Stack Containerization: `docker-compose.yml` orchestrates Flask Backend, React UI (Nginx), and Qdrant (see [Production demo](#production-demo-docker-compose)).

## Testing

Unit tests run by default (no live Qdrant or OpenAI required):

```bash
pip install -r backend/requirements.txt -r requirements-dev.txt
pytest
```

Integration tests (Qdrant at `QDRANT_URL`, `OPENAI_API_KEY` in `.env`):

```bash
pytest -m integration
```

Manual production smoke (Docker stack):

```bash
python experiment/scripts/smoke_production.py
python experiment/scripts/smoke_production.py --api-url http://127.0.0.1:5000/ask
```

Automated tests live under `tests/unit/` and `tests/integration/`.

- [ ] Scalable Cloud Hosting: Deploy the backend to AWS ECS (Fargate) and the frontend to AWS Amplify, utilizing AWS Secrets Manager for secure credential handling.

- [ ] Streaming Responses (UX): Implement Server-Sent Events (SSE) to provide a real-time "typewriter" effect for LLM responses, significantly reducing the perceived latency for end-users.