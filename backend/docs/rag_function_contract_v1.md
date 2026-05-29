# RAG Function Contracts (V1 Draft)

## 1) Pipeline Contract

### Function
`run_rag_pipeline(query: str, config: dict | None = None) -> dict`

### Input
- `query` (str): user question, non-empty string
- `config` (dict, optional): runtime knobs
  - `collection_name` (str)
  - `top_k_initial` (int, default 20)
  - `top_k_final` (int, default 5)
  - `rerank_enabled` (bool, default true)
  - `embedding_model` / `llm_model` / `rerank_model` (str)

### Output
- dict with:
  - `answer` (str)
  - `sources` (list[dict])

### Non-breaking requirements
- Always return `answer` and `sources` keys
- Never throw raw exception to API layer
- Empty/invalid query returns graceful response (not 500)

---

## 2) Retrieval Contract

### Function
`retrieve_embedding(query: str, vector_store, top_k: int = 20) -> list[RetrievedDoc]`

### RetrievedDoc schema
- `id` (str | int)
- `text` (str)
- `metadata` (dict)
- `score` (float | None)

### Guarantees
- Return at most `top_k` docs
- Each doc must include `metadata` (empty dict allowed, but key exists)

---

## 3) Rerank Contract

### Function
`rerank_passages(query: str, passages: list[RetrievedDoc], ranker, top_k: int = 5) -> list[RetrievedDoc]`

### Input assumptions
- `passages` must contain `text` and `metadata`

### Guarantees
- Return at most `top_k`
- Preserve original `metadata` for each returned passage
- If rerank fails, caller can fallback to first `top_k` retrieval results

---

## 4) Generation Contract

### Functions
- `format_context(passages: list[RetrievedDoc]) -> str`
- `generate_answer(query: str, context: str, prompt, llm, parser) -> str`
- `build_sources(passages: list[RetrievedDoc]) -> list[dict]`

### Sources schema (minimum)
- `title` (str, default `"Unknown"`)
- `page` (str|int, default `"?"`)
- `pdf_id` (str, default `""`)

### Guarantees
- `generate_answer` returns plain string
- `build_sources` output is directly consumable by frontend
- Missing metadata should be filled with defaults, not crash

---

## 5) API Compatibility Contract (`/ask`)

### Request
`POST /ask`
```json
{ "query": "..." }
```

### Response
```json
{
  "success": true,
  "answer": "...",
  "sources": [...]
}
```

### Must not break
- Frontend can still read `data.answer` and `data.sources`
- Response type/keys stay stable after refactor
