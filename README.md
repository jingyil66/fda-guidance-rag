# FDA Guidance Search & RAG Chatbot

FDA-RAG-Assistant is a Retrieval-Augmented Generation (RAG) chatbot designed for interactive question answering over FDA medical guidance documents. The system leverages PDF guidelines and associated metadata to provide accurate, context-aware responses, powered by OpenAI LLMs. It combines document retrieval, semantic search, and vector-based embeddings to ensure answers are grounded in official FDA content.

## Dataset

The chatbot uses FDA medical guidance documents as its primary dataset:

- **Source:** [FDA official website](https://www.fda.gov/regulatory-information/search-fda-guidance-documents)

## Features

**1. Metadata Pipeline**

- Fetches FDA document metadata from official APIs.

- Extracts document summaries and URLs asynchronously.

- Saves metadata locally for fast reuse.

**2. PDF Ingestion Pipeline**

- Downloads PDFs from FDA URLs.

- Splits PDFs into chunks for semantic search.

- Generates embeddings using OpenAI embeddings.

- Stores vectors in Qdrant for retrieval.

**3. RAG Question Answering**

- Retrieve chunks from Qdrant based on user queries.

- Generate answers using an LLM (OpenAI GPT-4o-mini).

**4. Web Frontend**

- Simple interface to ask questions and see answers.

- React UI for interactive question input and displaying answers.

- Flask backend provides API endpoints to query Qdrant and generate answers.

---

## Project Structure

```
FDA-RAG-Assistant/
│
├── backend/
│   ├── app.py                  # Flask API server
│   ├── config.py               # API keys, URLs, file paths
│   ├── ingest_to_qdrant.py     # PDF download, chunking, vectorization
│   ├── initial_data_ingestion.py # Orchestrate initial ingestion
│   ├── metadata_fetcher.py     # Fetch metadata from FDA
│   ├── metadata_pipeline.py    # Async metadata + summary pipeline
│   └── rag.py                  # RAG pipeline: query retrieval + LLM
│   └── requirements.txt
│
├── frontend/fda-app/
│   ├── node_modules/
│   ├── public/
│   └── src/                     # React source code
│
└── README.md

```
---

## Tech Stack

### Backend
- **Python 3.9+** – Core programming language  
- **Flask** – REST API server  
- **LangChain** – For RAG pipelines and text processing  
- **OpenAI API** – GPT-4o-mini for question answering  
- **Qdrant** – Vector database for semantic search  
- **Requests / aiohttp / BeautifulSoup / lxml** – Web scraping & HTTP requests  
- **Multiprocessing / Threading** – Parallel PDF downloading and processing  

### Frontend
- **React** – Interactive UI  
- **Node.js / npm** – Frontend runtime and package management  

### DevOps / Infrastructure
- **Docker** – Containerized Qdrant deployment  
- **Git** – Version control  

---

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm
- Docker (for running Qdrant)

### Backend Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/FDA-RAG-Assistant.git
   cd FDA-RAG-Assistant/backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
   
3. Install Python dependencies:
   
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables in config.py (e.g., OPENAI_API_KEY, Qdrant URL).

5. Start Qdrant (Docker):
   ```
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

6.Run `initial_data_ingestion.py`:
   ```
   python initial_data_ingestion.py
   ```

7. Frontend Setup: 
   ```
   npm install
   npm run dev
   ```
   
   The frontend will be available at http://localhost:5173 and will communicate with the Flask backend API.

## Future Improvements

**1. Retrieval & Document Processing**

- [ ] Implement source attribution in answers: LLM responses include citations referencing the retrieved FDA guidance documents, improving answer reliability and traceability.

- [ ] Incremental dataset updates: Automatically detect and ingest new FDA guidance documents.

- [ ] Enhanced document understanding: Extract structured information from tables, figures, and charts in PDFs.

**2. Evaluation & Metrics**

- [ ] Evaluate RAG pipeline using metrics including faithfulness, answer relevance, context precision, citation accuracy, and multi-turn coherence to ensure reliable, context-aware LLM responses.

**3. LLM Optimization**

- [ ] Fine-tune domain-specific LLM on FDA guidance documents for higher accuracy.

- [ ] Experiment with prompt engineering to reduce hallucinations and improve context relevance.


**4. Scalability & Deployment**

- [ ] Containerize with Docker and orchestrate multiple instances for high-volume usage.
