# FDA Guidance RAG: Experiment Report

**Project:** Retrieval-augmented Q&A over FDA medical guidance documents  
**Corpus:** `subset_200_v1` (200 PDFs)  
**Evaluation:** 30 development + 20 sealed test QA pairs  
**Date:** May 2026  

---

## Abstract

We built and evaluated a two-stage RAG pipeline for answering questions over FDA guidance PDFs. Starting from a fixed 600/200 character chunk baseline with FlashRank cross-encoder reranking, we ran systematic ablations on chunking strategy and retrieval configuration. Fixed 600/200 chunking outperformed larger (800/160), smaller (400/80), and regex-based section chunking on retrieval metrics. Surprisingly, disabling reranking and using embedding-only top-5 retrieval improved recall@5 from 0.97 to 1.00 and MRR from 0.85 to 0.95 on the development set, while end-to-end success rate decreased slightly (0.90 → 0.83). Alternative rerank models, expanded candidate pools, and enriched rerank inputs did not beat the no-rerank configuration. On a sealed test set, the final configuration achieved recall@5 0.90 and e2e success rate 0.70. We recommend fixed 600/200 chunking with embedding-only top-5 retrieval as the production default.

---

## 1. Introduction

### 1.1 Problem

FDA guidance documents are long, structurally heterogeneous PDFs written in dense regulatory language. Users need accurate, source-grounded answers to specific compliance questions. A retrieval-augmented generation (RAG) system must (1) locate the correct passage within the correct document, and (2) generate an answer strictly supported by retrieved context.

### 1.2 System overview

The pipeline has three stages:

1. **Ingest:** PyPDF text extraction → LangChain `RecursiveCharacterTextSplitter` → OpenAI `text-embedding-3-small` → Qdrant vector store.
2. **Retrieve:** Dense similarity search (top-20 candidates) → optional FlashRank cross-encoder rerank → top-5 chunks passed to the LLM.
3. **Generate:** `gpt-4o-mini` with temperature 0 and the `strict_context_v1` prompt, constrained to use only retrieved passages.

### 1.3 Evaluation scope and goals

All experiments reported here use the **subset_200** corpus (200 guidances) and a manually curated QA gold set:

| Split | File | Queries | Purpose |
|-------|------|---------|---------|
| Development | `experiment/subsets/qa_gold_dev.json` | 30 | Ablation and config selection |
| Sealed test | `experiment/subsets/qa_gold_test.json` | 20 | Single-run generalization estimate |

Gold alignment uses **`pdf_id` + `page`** as the primary match criterion, with text-overlap fallback when metadata is incomplete (`pdf_id_page_with_text_fallback`).

The goal of this report is to document chunking and retrieval ablations and justify the **final production configuration**, which is captured in `run_014_subset200_final_dev.json` (dev) and `run_013_subset200_no_rerank_test.json` (test).

---

## 2. Experimental Setup

### 2.1 Data

- **PDF subset:** `subset_200_v1` — 200 FDA guidance documents selected from the full corpus.
- **QA gold:** Each item contains `question`, `answer` (reference), `context` (gold passage), and `source_id`, with aligned `pdf_id` and `page` fields added during gold curation.
- **Vector collections:** Separate Qdrant collections per chunking configuration (e.g. `experiment_subset200_chunk600_overlap200`).

### 2.2 Chunking baseline

The baseline strategy is **fixed-size chunking** at 600 characters with 200-character overlap, producing approximately 21,000 chunks for subset_200. This balances passage granularity against embedding context length.

### 2.3 Retrieval and generation defaults

| Parameter | Value |
|-----------|-------|
| Embedding model | `text-embedding-3-small` |
| Initial retrieval | top-20 by cosine similarity |
| Final context | top-5 chunks |
| Rerank model (when enabled) | FlashRank `ms-marco-MiniLM-L-12-v2` |
| LLM | `gpt-4o-mini`, temperature 0 |
| Prompt | `strict_context_v1` |

When reranking is enabled, the pipeline retrieves top-20 embedding hits, reranks them with the cross-encoder, and passes the top-5 reranked passages to the LLM. When disabled (`embedding_only` mode), the top-5 embedding hits are used directly.

### 2.4 Metrics

**Retrieval metrics** (per query, averaged):

- **recall@k:** 1 if any of the top-k retrieved chunks matches gold (`pdf_id`+`page` or text fallback); 0 otherwise.
- **MRR:** reciprocal rank of the first matching chunk in the retrieved list.
- **context_precision@k:** fraction of retrieved chunks (up to k) that match gold. The denominator is the number of chunks actually returned, not always k.

**Answer metrics** (GPT-as-judge, same model as generation):

- **correctness:** factual agreement with the gold reference answer.
- **groundedness:** answer supported by retrieved context, no hallucination.
- **relevance:** answer addresses the question concisely.

**End-to-end success rate:** fraction of queries where both correctness and groundedness pass (`e2e_success_rule: correctness_and_groundedness_pass`).

### 2.5 Reproducibility

Each experiment run stores a config snapshot, results (`results.jsonl`), aggregated metrics (`metrics.json`), and classified failures (`failures.jsonl`) under `experiment/runs/<run_id>/`. Run metadata is appended to `experiment/registry.csv`.

---

## 3. Chunking Ablation

**Research question:** Which chunking strategy best supports retrieval on subset_200?

All chunk ablation runs used the dev set (n=30), FlashRank reranking enabled, and retrieval-only evaluation unless noted.

| Run | Strategy | Chunk params | recall@5 | MRR | context_precision@5 |
|-----|----------|--------------|----------|-----|---------------------|
| run_002 | Fixed | 600 / 200 | **0.97** | **0.85** | **0.47** |
| run_003 | Fixed | 800 / 160 | 0.93 | 0.83 | 0.37 |
| run_004 | Fixed | 400 / 80 | 0.93 | 0.82 | 0.38 |
| run_005 | Section (regex headings) | max 1200 / min 200 | 0.90 | 0.76 | 0.31 |

### 3.1 Fixed-size variants

Increasing chunk size to 800/160 or decreasing to 400/80 both **reduced** recall@5 (0.93 vs 0.97) and context_precision@5 (~0.37–0.38 vs 0.47). Larger chunks merge distinct regulatory statements; smaller chunks fragment coherent passages and increase noise in the embedding space.

### 3.2 Section-based chunking

Section chunking split documents at FDA-style heading patterns (PyPDF + regex), producing fewer, larger chunks (~8,500 vs ~21,000). Retrieval metrics dropped further: recall@5 0.90, MRR 0.76, context_precision@5 0.31. Heading detection on raw PDF text is unreliable; merged sections dilute query–passage similarity.

### 3.3 Decision

**Keep fixed 600/200** for all subsequent retrieval ablations. Semantic chunking was explored briefly but not adopted due to ingest cost and lack of demonstrated gain on dev.

---

## 4. Retrieval and Rerank Ablation

**Research question:** Does cross-encoder reranking improve ranking and end-to-end quality on top of embedding retrieval?

All runs below use the 600/200 collection unless noted. Dev set, n=30.

### 4.1 Rerank on vs off

| Run | Rerank | recall@5 | MRR | context_precision@5 | correctness | e2e |
|-----|--------|----------|-----|---------------------|-------------|-----|
| run_002 | FlashRank MiniLM | 0.97 | 0.85 | 0.47 | **0.90** | **0.90** |
| run_007 | **None (embedding-only)** | **1.00** | **0.95** | 0.49 | 0.83 | 0.83 |

Disabling rerank recovered one dev query that reranking had pushed out of top-5 (qa_index 13, screening assay validation — see Section 7) and improved gold rank on others (e.g. qa_index 7: gold rank 4 → rank 1). Overall retrieval improved, but answer correctness dropped by 7 percentage points on dev.

**Mechanism:** MS MARCO–trained cross-encoders score query–passage relevance for generic web QA. FDA guidances share boilerplate phrasing (“alternative approaches,” “current thinking,” “contact FDA staff”) across unrelated documents. Reranking often promotes semantically similar but **wrong-PDF** passages, filling all five context slots from a single incorrect document.

### 4.2 Rerank model variant

| Run | Change | recall@5 | MRR | context_precision@5 |
|-----|--------|----------|-----|---------------------|
| run_008 | Replace MiniLM with rank-T5-flan | 0.67 | 0.37 | 0.20 |

The larger cross-encoder performed **worse** than both MiniLM rerank and embedding-only retrieval on this domain.

### 4.3 Candidate pool size

| Run | Change | recall@5 | MRR | context_precision@5 |
|-----|--------|----------|-----|---------------------|
| run_009 | top_k_initial = 40 | 0.97 | 0.86 | 0.43 |

Doubling the candidate pool did not reach embedding-only recall@5 (1.00) and slightly lowered precision vs run_007.

### 4.4 Enriched rerank input

We implemented `passage_format: title_section_chunk`, prepending document title and a section label (from ingest metadata, or `Page {n}` for fixed chunks) to each passage before reranking and LLM context formatting.

| Run | Passage format | recall@5 | MRR | context_precision@5 |
|-----|----------------|----------|-----|---------------------|
| run_002 | raw chunk | 0.97 | 0.85 | 0.47 |
| run_012 | title + section + chunk | 0.97 | 0.83 | 0.46 |

Enriched input did not improve over raw rerank and remained below embedding-only metrics.

### 4.5 Context window size (top_k_final = 3)

| Run | top_k_final | recall@3 | context_precision@3 | context_precision@5 |
|-----|-------------|----------|---------------------|---------------------|
| run_007 | 5 | — | — | 0.49 |
| run_010 | 3 | 1.00 | 0.61 | 0.61* |

\*run_010 returns only three chunks; context_precision@5 equals precision@3 because the denominator is `len(retrieved)`, not 5. The first three slots match run_007’s top three — **no actual retrieval benefit**. We keep **top_k_final = 5**.

### 4.6 Decision

Final retrieval configuration:

- Mode: **`embedding_only`**
- top_k_initial / top_k_final: **20 / 5**
- Rerank: **disabled**

---

## 5. Final Configuration

| Component | Setting |
|-----------|---------|
| Chunking | fixed 600/200 |
| Qdrant collection | `experiment_subset200_chunk600_overlap200` |
| Retrieval mode | `embedding_only` |
| Embedding | `text-embedding-3-small` |
| top_k_initial / top_k_final | 20 / 5 |
| Rerank | disabled |
| LLM | `gpt-4o-mini`, `strict_context_v1` |
| Judge | `gpt-4o-mini` |

Config files:

- Dev: `experiment/configs/run_014_subset200_final_dev.json`
- Test: `experiment/configs/run_013_subset200_no_rerank_test.json`

Production pipeline defaults (`backend/app/services/pipeline_service.py`) were updated to `rerank_enabled=false` and the subset_200 collection.

---

## 6. Final Results

### 6.1 Development set (run_007, n=30)

| Metric | Value |
|--------|-------|
| recall@5 | 1.00 |
| MRR | 0.95 |
| context_precision@5 | 0.49 |
| correctness | 0.83 |
| groundedness | 0.93 |
| relevance | 0.90 |
| **e2e_success_rate** | **0.83** |

For comparison, the rerank baseline (run_002) achieved e2e 0.90 with lower retrieval recall (0.97).

### 6.2 Sealed test set (run_013, n=20, single run)

| Metric | Value |
|--------|-------|
| recall@5 | 0.90 |
| MRR | 0.90 |
| context_precision@5 | 0.40 |
| correctness | 0.70 |
| groundedness | 0.80 |
| relevance | 0.80 |
| **e2e_success_rate** | **0.70** |

The test set was evaluated **once** with the config frozen from run_007; no hyperparameter tuning was performed on test queries.

### 6.3 Dev vs test summary

| Metric | Dev (run_007) | Test (run_013) | Δ |
|--------|---------------|----------------|---|
| recall@5 | 1.00 | 0.90 | −0.10 |
| MRR | 0.95 | 0.90 | −0.05 |
| context_precision@5 | 0.49 | 0.40 | −0.09 |
| e2e_success_rate | 0.83 | 0.70 | −0.13 |

The gap is expected given the small test set (n=20) and harder generalization; test metrics should be treated as a point estimate, not an optimized upper bound.

---

## 7. Error Analysis

Failures are classified automatically (`experiment/io_utils.py`):

| Code | Meaning |
|------|---------|
| R1_not_retrieved | Gold chunk absent from top-5 |
| R2_suboptimal_rank | Gold retrieved but not at rank 1 |
| G1_low_correctness | Retrieval OK (rank 1) but answer factually wrong |
| G2_low_groundedness | Answer not supported by context |
| G3_low_relevance | Answer off-topic |

### 7.1 Failure counts

| Run | R1 | R2 | G1 | Total failure records |
|-----|----|----|-----|------------------------|
| run_002 (rerank on) | 1 | 6 | 2 | 9 |
| run_007 (rerank off) | 0 | 3 | 5 | 8 |
| run_013 (test) | 2 | 0 | 4 | 6 |

Disabling rerank **eliminated** R1 failures on dev and reduced R2 from 6 to 3. End-to-end failures shifted toward **generation** (G1 increased from 2 to 5 on dev).

### 7.2 Rerank-induced retrieval loss (run_002)

**qa_index 13** — *“How does the sensitivity of the screening assay impact the validation process…”*

- run_002: recall@5 = **0.0** (gold not in top-5 after rerank); retrieved five chunks from pdf_id `119788`, none matching gold.
- run_007: recall@5 = **1.0**, gold at rank 1; answer still judged correct.

Reranking demoted the correct embedding hit below the top-5 cutoff. Interestingly, run_002 still produced a correct answer (likely from partial overlap in wrong-document text), illustrating that retrieval metrics and answer quality can diverge.

### 7.3 Rerank-induced rank degradation (run_002 vs run_007)

**qa_index 7** — *Vitamin/mineral compositions of test diet vs NRC requirements*

- run_002: gold at **rank 4** (MRR 0.25); correctness failed.
- run_007: gold at **rank 1** (MRR 1.0); retrieval fixed but correctness still failed (G1).

Rerank hurt rank; embedding-only fixed retrieval but the LLM still failed to extract the tabular numeric comparison from context — a **generation/extraction** failure, not retrieval.

### 7.4 Generation failures with perfect retrieval (run_007 dev)

Five dev queries had gold at rank 1 but failed correctness (G1):

- **qa_index 4** — ICH Q3E endorsement vs labeling requirements: model returned empty or non-responsive output (correctness, groundedness, relevance all 0).
- **qa_index 5** — USEPA/ECHA toxicology comparison: grounded and relevant but factually incomplete vs gold.
- **qa_index 6** — Biomarker patient selection (HLA-B*1502): failed all answer metrics despite four of five retrieved chunks from the correct document.
- **qa_index 7** — NRC nutrient table (see above).
- **qa_index 17** — Use-related risk analysis and training decay: grounded but judged not relevant/correct.

These cases share **complex multi-part questions** requiring synthesis of structured or numeric content from long regulatory passages.

### 7.5 Test-set retrieval misses (run_013)

Two test queries had R1 failures:

- **qa_index 12** — CDER vs CBER guidance differences: top-5 retrieved mixed CDER/CBER documents (`114764`, `183874`, `113913`) but none matched gold page alignment; full e2e failure.
- **qa_index 14** — REMS participant requirements: five chunks from pdf_id `164344` (related REMS content) but gold page not hit; model could not ground a correct answer.

Both suggest **embedding confusion among topically similar guidances** from different programs or sections.

### 7.6 Duplicate-chunk pattern

Several failures show the same `pdf_id` repeated across all five slots (e.g. qa_index 5 dev: five copies of `189891`). This inflates recall when any chunk from the correct PDF matches, but leaves little diverse context for synthesis — a known limitation of pure dense retrieval without deduplication.

---

## 8. Discussion

### 8.1 Why generic reranking hurt

Cross-encoders trained on MS MARCO optimize passage-level semantic relevance for open-domain QA. FDA guidances violate this assumption in three ways:

1. **Shared regulatory language** makes wrong documents score highly.
2. **Gold is page-specific**; rerankers do not optimize `pdf_id`+`page` alignment.
3. **600-character chunks** already yield strong embedding rankings; reranking reorders without domain signal and drops gold from top-5.

### 8.2 Retrieval vs generation trade-off

We prioritized **retrieval recall and rank** by disabling rerank. Dev e2e dropped from 0.90 to 0.83. The main residual failure mode is **G1 (low correctness)** with gold at rank 1 — extraction and synthesis limits of `gpt-4o-mini` on dense regulatory text, not missing context.

A hybrid policy (rerank only when embedding score margin is low) remains untested and is listed as future work.

### 8.3 Limitations

- **Small QA sets:** 30 dev / 20 test queries; confidence intervals are wide.
- **Subset corpus:** subset_200 ⊂ full ~2000-document corpus; metrics may not transfer.
- **context_precision@5 ceiling:** single-page gold can contribute at most one hit per five slots (~0.20 per query if only one page is gold).
- **Section labels:** fixed chunks use `Page {n}` placeholders, not true structural sections.
- **GPT-as-judge:** same model family as generator; no independent human evaluation.
- **No hybrid retrieval:** BM25 + dense fusion (RRF) is configured but not implemented or evaluated.

---

## 9. Future Work

1. **Deduplication / MMR** before top-5 to reduce same-PDF slot dominance and improve effective context diversity.
2. **BM25 + dense hybrid** with reciprocal rank fusion for exact-term queries (regulation numbers, assay names).
3. **Improved section chunking** (e.g. Unstructured.io or layout-aware PDF parsing) with re-ingest and re-evaluation.
4. **Conditional reranking** — apply cross-encoder only when top embedding scores are close.
5. **Full-corpus deployment** and production monitoring on live queries.

---

## 10. Conclusion

We evaluated chunking and retrieval strategies for FDA guidance RAG on a 200-document subset with 50 total gold QA pairs.

1. **Chunking:** Fixed **600/200** outperformed 400/80, 800/160, and regex section chunking on dev retrieval.
2. **Retrieval:** **Disable FlashRank**; use embedding top-5 from top-20 candidates. Generic cross-encoder reranking degraded recall and MRR on this domain.
3. **Generalization:** Sealed test recall@5 **0.90**, e2e success rate **0.70** (run_013).
4. **Remaining errors** are primarily generation failures on complex synthesis questions, plus embedding confusion among similar guidances on test.

The production pipeline defaults match this final configuration.

---

## Appendix A — Run reference

| run_id | Purpose |
|--------|---------|
| run_002_subset200_dev_baseline | 600/200 + FlashRank rerank + answer eval (dev baseline) |
| run_003_subset200_chunk800_dev | Chunk 800/160 ablation |
| run_004_subset200_chunk400_dev | Chunk 400/80 ablation |
| run_005_subset200_section_dev | Section chunk ablation |
| run_007_subset200_no_rerank_dev | Embedding-only top-5 (final dev candidate) |
| run_008_subset200_rerank_t5_dev | rank-T5-flan reranker |
| run_009_subset200_topk40_dev | top_k_initial = 40 |
| run_010_subset200_topk3_dev | top_k_final = 3 |
| run_012_subset200_rerank_title_section_dev | Enriched rerank input |
| run_013_subset200_no_rerank_test | Sealed test, final config |
| run_014_subset200_final_dev | Final dev config snapshot |

## Appendix B — Artifact paths

| Artifact | Location |
|----------|----------|
| Run configs | `experiment/configs/run_*.json` |
| Metrics | `experiment/runs/<run_id>/metrics.json` |
| Per-query results | `experiment/runs/<run_id>/results.jsonl` |
| Failure analysis | `experiment/runs/<run_id>/failures.jsonl` |
| Registry | `experiment/registry.csv` |

## Appendix C — Suggested figures

1. Bar chart: recall@5 and MRR across chunk sizes (runs 002–005).
2. Grouped bar chart: retrieval and e2e metrics for run_002 vs run_007.
3. Table: dev vs test final metrics (Section 6.3).
4. Example diagram: five context slots all from one wrong pdf_id after reranking.
