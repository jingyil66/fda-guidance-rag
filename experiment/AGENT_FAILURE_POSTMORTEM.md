# Agent failure postmortems (dev gold)

Run: `run_agent_eval_full_dev` (2026-06-03, full production Qdrant).  
Failure taxonomy: A1 wrong tool · A2 trajectory/args · A3 empty retrieval · A4 answer quality · A5 runtime/max steps.

**Note:** This run had **0 A3 labels** in aggregated counts. A3 only fires when the **first tool is correct**, gold requires `search_guidance`, search ran, and `source_count == 0` (see `classify_agent_failure` in `agent_evaluators.py`). Many empty-retrieval cases were absorbed by **A1** (wrong first tool) or **A2** (args/trajectory) first. Case 3 below is the canonical empty-retrieval pattern.

---

## Case 1 — A1: Wrong first tool (T4 multi-step)

| Field | Value |
|-------|-------|
| **ID** | `agent_t_010` / linked `agent_task_010` |
| **Category** | T4 — detail → scoped search |
| **Code** | **A1** (`wrong_first_tool`) |

### Question

> Look up guidance on crabmeat filth (pdf_id 90349 if needed), then what does it say about alternative approaches?

### Gold trajectory

```json
[
  {"tool": "get_guidance_detail", "args": {"pdf_id": "90349"}},
  {"tool": "search_guidance", "args": {"query": "alternative approaches", "pdf_id": "90349"}}
]
```

### Actual trajectory

```json
[
  {"tool": "search_guidance", "args": {"query": "alternative approaches", "pdf_id": "90349"}},
  {"tool": "list_guidance", "args": {"keyword": "crabmeat filth"}}
]
```

`source_count`: **0** · Answer: *"I cannot find any guidance on crabmeat filth…"*

### Root cause

1. **Tool order inverted** — Agent jumped to scoped search before loading document context via `get_guidance_detail`, failing first-tool accuracy (A1).
2. **Scoped search on cold start** — `search_guidance(pdf_id=90349)` post-filters embedding top-k; with a narrow query and wrong ranking, filtered passages can be empty → no sources merged into context.
3. **Fallback to wrong tool** — Second step `list_guidance(keyword="crabmeat filth")` does substring match on title/summary; CPG titles use *"Crabmeat – Fresh and Frozen – Adulteration with Filth…"* without the literal phrase *"crabmeat filth"*, so list returns no matches.

### Downstream (task E2E)

`agent_task_010` failed **A4**: agent used `list_guidance` twice (`crabmeat filth`, `E. coli`) instead of `get_guidance_detail`, surfaced unrelated E. coli guidances (tree nuts, bottled water). Fixed RAG on the same question retrieved the correct CPG passages (compare run: RAG judge 1.0, agent 0.0).

### Fix directions

- Prompt: *When question mentions `pdf_id`, call `get_guidance_detail` first.*
- Tool: `list_guidance` fuzzy match on title tokens, not single substring.
- Retrieval: apply `pdf_id` as Qdrant metadata filter at query time, not post-filter top-k.

---

## Case 2 — A2: Metadata filter args incomplete (T2 list)

| Field | Value |
|-------|-------|
| **ID** | `agent_t_004` |
| **Category** | T2 — catalog filter |
| **Code** | **A2** (`trajectory_mismatch`) |

### Question

> Which CDER final guidances were issued in 2024? Show up to 15 titles.

### Gold trajectory

```json
[
  {"tool": "list_guidance", "args": {
    "center": "CDER",
    "communication_type": "Final",
    "year": 2024
  }}
]
```

### Actual trajectory

```json
[
  {"tool": "list_guidance", "args": {
    "center": "CDER",
    "year": 2024,
    "limit": 15
  }}
]
```

Answer: *"I cannot answer from the corpus as there are no CDER final guidances issued in 2024."*

### Root cause

1. **Correct tool, wrong args** — First tool is `list_guidance` (passes A1), but **`communication_type: "Final"` omitted**.
2. **`list_guidance` uses substring match** on `field_communication_type` (`agent_tools.py`). Without `"Final"`, results mix Draft/Final/other types; agent then incorrectly concluded no Final guidances exist.
3. **T1 search pass rate is 22%** partly because similar arg-precision issues affect `search_guidance` query strings (e.g. `agent_t_001`: gold query `"human factors validation study NDA"` vs actual long paraphrase).

### Fix directions

- Expose enum hints in tool schema (`communication_type`: Draft | Final | Guidance Document).
- Few-shot in system prompt for filter-heavy list queries.
- Eval: report **arg F1** separately from tool name accuracy (strict trajectory already partial).

---

## Case 3 — Empty retrieval after scoped search (A3 pattern; labeled A2/A4 in run)

| Field | Value |
|-------|-------|
| **ID** | `agent_t_032` / `agent_task_008` |
| **Category** | T4 / L3 — list → search |
| **Code** | Tool routing **passed**; task failed **A4** (`min_sources` / correctness) |

### Question

> Find gene therapy guidances, then summarize FDA expectations for CMC documentation.

### Gold trajectory

```json
[
  {"tool": "list_guidance", "args": {"keyword": "gene therapy"}},
  {"tool": "search_guidance", "args": {"query": "CMC documentation"}}
]
```

### Actual trajectory (7 steps)

```
list_guidance(keyword=gene therapy)
→ get_guidance_detail(pdf_id=72402, 79856, 89036)  ×3
→ search_guidance(query=CMC documentation, pdf_id=<each>)  ×3
```

`source_count`: **0** · Answer acknowledges gene therapy list but *"could not retrieve specific information regarding CMC documentation"*.

### Root cause

1. **Over-scoping search** — Agent picked older gene-therapy PDFs from list, then ran **pdf_id-scoped** CMC search on each. CMC expectations live in dedicated guidances (e.g. *Chemistry, Manufacturing, and Control … Human Gene Therapy INDs*), not in the three chosen docs.
2. **Post-filter empty retrieval** — `search_guidance` embeds globally, then keeps only chunks matching `pdf_id`. Wrong document → **zero passages** after filter (classic A3 symptom).
3. **Why not labeled A3** — Trajectory mismatches gold (extra detail steps, scoped search args) → **A2** checked before A3; task row fails `min_sources: 1` → **A4**.
4. **Fixed RAG contrast** — Same question via `/ask`: RAG retrieved gene therapy CMC guidance passages and produced a substantive answer (compare run).

### Fix directions

- Prompt: *After list, run corpus-wide `search_guidance` unless user specified one pdf_id.*
- Cap redundant `get_guidance_detail` loops; prefer one scoped search with optional `pdf_id`.
- Retrieval: metadata pre-filter in Qdrant; tool output should suggest broadening search when pdf_id filter returns 0 passages.

---

## Summary table

| Case | Code | Symptom | Primary layer |
|------|------|---------|---------------|
| Crabmeat pdf_id (`agent_t_010`) | A1 | Search before detail; list keyword miss | Tool routing + list tool |
| CDER Final 2024 (`agent_t_004`) | A2 | Missing `communication_type` filter | Tool args / metadata |
| Gene therapy CMC (`agent_t_032`) | A4 (A3 pattern) | Scoped search on wrong pdf_ids → 0 sources | Retrieval scoping + multi-step policy |

## Reproduce

```bash
python experiment/run_agent_eval.py --config experiment/configs/run_agent_eval_full_dev.json --preview
python experiment/compare_ask_vs_agent.py --limit 0   # head-to-head on task gold
```

Raw rows: `experiment/runs/run_agent_eval_full_dev/agent_results.jsonl`, failures: `agent_failures.jsonl`.
