# FDA Guidance RAG — Experiment Report Outline

## 1. Introduction

- Problem: interactive Q&A over FDA medical guidance documents (regulatory text, long PDFs, domain terminology).
- System overview: two-stage RAG — dense retrieval (OpenAI `text-embedding-3-small` + Qdrant) → optional rerank → `gpt-4o-mini` generation with strict-context prompt.
- Evaluation scope: **subset_200** (200 PDFs), **30 dev** + **20 sealed test** QA pairs with `pdf_id` + page gold alignment.
- Report goal: document chunking and retrieval ablations and justify the **final production configuration**.

---

## 2. Experimental Setup

### 2.1 Data and splits

- PDF corpus: `subset_200_v1` (200 guidances).
- QA gold: `qa_gold_dev.json` (30), `qa_gold_test.json` (20, sealed).
- Gold matching: `pdf_id` + `page` primary; text overlap fallback.

### 2.2 Vector store and chunking baseline

- Collection (final): `experiment_subset200_chunk600_overlap200`.
- Baseline chunking: **fixed 600 characters / 200 overlap** (~21k chunks).
- Ingest: PyPDF → LangChain `RecursiveCharacterTextSplitter`.

### 2.3 Retrieval and generation defaults

- Embedding: `text-embedding-3-small`.
- Initial retrieval: top-20; final context: top-5 (unless noted).
- Generation: `gpt-4o-mini`, temperature 0, `strict_context_v1` prompt.
- Answer evaluation: GPT-as-judge — correctness, groundedness, relevance; e2e = correctness ∧ groundedness.

### 2.4 Metrics

| Layer | Metrics |
|-------|---------|
| Retrieval | recall@k, MRR, context_precision@k |
| Answer | correctness, groundedness, relevance, e2e_success_rate |

---

## 3. Chunking Ablation

**Question:** Which fixed chunk size (or section-based strategy) best supports retrieval on subset_200?

| Run | Strategy | Chunk | recall@5 | MRR | precision@5 |
|-----|----------|-------|----------|-----|-------------|
| Baseline | fixed | 600/200 | **0.97** | **0.85** | **0.47** |
| Alt A | fixed | 800/160 | 0.93 | 0.83 | 0.37 |
| Alt B | fixed | 400/80 | 0.93 | 0.82 | 0.38 |
| Section | PyPDF + regex headings | max1200/min200 | 0.90 | 0.76 | 0.31 |

**Findings**

- Larger chunks (800/160) and smaller chunks (400/80) both **underperform** 600/200 on recall and precision.
- Section-based chunking (regex FDA headings) produced fewer, larger chunks (~8.5k) but **hurt** retrieval vs fixed 600/200.
- Semantic chunking was not adopted (cost / no proven gain on dev).

**Decision:** Keep **fixed 600/200** for all subsequent runs.

---

## 4. Retrieval and Rerank Ablation

**Question:** Does FlashRank cross-encoder reranking improve ranking and end-to-end quality on top of embedding retrieval?

All runs use chunk 600/200 collection unless noted. Dev set, 30 queries.

### 4.1 Rerank on vs off

| Config | rerank | recall@5 | MRR | precision@5 | correctness | e2e |
|--------|--------|----------|-----|-------------|-------------|-----|
| Baseline + FlashRank MiniLM | on | 0.97 | 0.85 | 0.47 | **0.90** | **0.90** |
| Embedding-only top-5 | **off** | **1.00** | **0.95** | 0.49 | 0.83 | 0.83 |

**Findings:** Generic MS MARCO rerank **reorders** embedding results in ways that **drop gold chunks** from top-5 (e.g. same wrong PDF filling all slots). Retrieval metrics improve without rerank; answer correctness drops slightly on dev.

### 4.2 Rerank model and candidate pool

| Variant | Change | recall@5 | MRR | precision@5 |
|---------|--------|----------|-----|-------------|
| rank-T5-flan | replace MiniLM | 0.67 | 0.37 | 0.20 |
| top_k_initial = 40 | more candidates | 0.97 | 0.86 | 0.43 |

**Findings:** Larger cross-encoder model **failed** on this domain. Expanding the candidate pool did not beat embedding-only top-5.

### 4.3 Enriched rerank input (Title + Section + chunk)

- Implemented `passage_format: title_section_chunk` for FlashRank and LLM context.
- Section label: `section_title` from ingest, or **`Page {n}`** placeholder for fixed chunks.
- Result vs raw rerank: recall@5 0.97, MRR 0.83, precision@5 0.46 — **no meaningful gain**.

### 4.4 Context size (top_k_final = 3)

- precision@3 identical to first 3 chunks of top-5 embedding run.
- precision@5 headline improves only because fewer slots are filled; **no retrieval benefit**.
- **Decision:** Keep **top_k_final = 5**.

**Decision:** Final retrieval = **embedding-only**, top-20 → take top-5, **no FlashRank**.

---

## 5. Final Configuration

| Component | Setting |
|-----------|---------|
| Chunking | fixed 600/200 |
| Collection | `experiment_subset200_chunk600_overlap200` |
| Retrieval mode | `embedding_only` |
| top_k_initial / top_k_final | 20 / 5 |
| Rerank | disabled |
| LLM | gpt-4o-mini, strict context |

Config files: `run_014_subset200_final_dev.json`, `run_013_subset200_no_rerank_test.json`.

Pipeline defaults updated: `rerank_enabled=false`, subset200 collection.

---

## 6. Final Results

### 6.1 Development set (n=30)

| Metric | Value |
|--------|-------|
| recall@5 | 1.00 |
| MRR | 0.95 |
| context_precision@5 | 0.49 |
| correctness | 0.83 |
| groundedness | 0.93 |
| relevance | 0.90 |
| e2e_success_rate | 0.83 |

### 6.2 Sealed test set (n=20, single run)

| Metric | Value |
|--------|-------|
| recall@5 | 0.90 |
| MRR | 0.90 |
| context_precision@5 | 0.40 |
| correctness | 0.70 |
| groundedness | 0.80 |
| relevance | 0.80 |
| e2e_success_rate | 0.70 |

---

## 7. Error Analysis (recommended section to fill)

- Pull from `experiment/runs/run_007_*/failures.jsonl` and `run_013_*/failures.jsonl`.
- Categories: R1_not_retrieved, wrong PDF dominance, answer correct despite retrieval miss, judge vs gold mismatch.
- Compare failure modes: rerank-on (run_002) vs rerank-off (run_007).

---

## 8. Discussion

### 8.1 Why rerank hurt

- MS MARCO cross-encoders optimize generic QA snippets, not FDA `pdf_id`+page alignment.
- Regulatory boilerplate makes wrong sections **look** relevant to the reranker.
- Embedding order on short chunks (600 char) was already strong; rerank added noise.

### 8.2 Retrieval vs generation trade-off

- Optimizing retrieval (recall/MRR) came at a small cost to dev correctness vs rerank baseline.
- Test e2e (0.70) reflects harder generalization; report as sealed estimate, not tuned on test.

### 8.3 Limitations

- Small QA sets (30/20); subset_200 ≠ full 2000-doc corpus.
- context_precision@5 capped when gold is single-page (at most one exact page hit per five slots).
- Section labels were page placeholders, not true structural sections.
- GPT-as-judge variance; no human eval.

---

## 9. Future Work (optional closing)

- MMR or per-(pdf_id, page) dedup before top-5 (precision, not yet tested post-finalization).
- BM25 + dense hybrid with RRF (code not implemented).
- True section chunking (Unstructured.io or improved regex) with re-ingest.
- Domain-tuned reranker or rerank only when embedding score margin is low.
- Full-corpus ingest and production monitoring.

---

## 10. Conclusion

- **Chunking:** fixed **600/200** wins over 400/80, 800/160, and regex section chunking on subset_200 dev retrieval.
- **Retrieval:** **Disable FlashRank**; use embedding top-5 from top-20 candidates.
- **Reported generalization:** sealed test recall@5 **0.90**, e2e **0.70**.
- The implemented pipeline defaults match the final ablation configuration.

---

## Appendix A — Run ID reference

| run_id | Purpose |
|--------|---------|
| run_002_subset200_dev_baseline | Chunk 600/200 + MiniLM rerank + answer eval |
| run_003 / run_004 | Chunk 800/160, 400/80 ablation |
| run_005 | Section chunk ablation |
| run_007 | No rerank (dev final candidate) |
| run_008 / run_009 / run_012 | Rerank model, top-40, enriched input |
| run_010 | top_k_final=3 |
| run_013 | Sealed test, final config |
| run_014 | Final dev config snapshot |

## Appendix B — Suggested figures

1. Bar chart: recall@5 and MRR across chunk sizes.
2. Bar chart: retrieval metrics rerank on vs off.
3. Table: final dev vs test metrics side by side.
4. (Optional) Example failure: 5 chunks same wrong pdf_id before/after rerank.
