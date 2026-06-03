# Agent evaluation (tool routing + task E2E)

Evaluation for `/ask_agent` inspired by BFCL-style tool routing and GAIA-style end-to-end task success.

## Datasets

| File | Rows | Purpose |
|------|------|---------|
| `experiment/subsets/agent_tool_gold_dev.json` | 32 | P0 tool routing (T1–T4) |
| `experiment/subsets/agent_task_gold_dev.json` | 20 | P1 task success (L1–L3) |

### Tool gold (`agent_tool_gold_dev.json`)

| Field | Meaning |
|-------|---------|
| `question` | User query passed to `get_agent_answer` |
| `expected_steps` | Gold `[{tool, args}, ...]` in order |
| `forbidden_tools` | Tools that must not appear |
| `category` | T1 search, T2 list, T3 detail, T4 multi-step |

### Task gold (`agent_task_gold_dev.json`)

| Field | Meaning |
|-------|---------|
| `level` | L1 list-only, L2 search+answer, L3 multi-step |
| `require_tools` | Tool names that must appear in the trace |
| `must_include` | Substrings required in the final answer |
| `min_sources` | Minimum retrieved source count |
| `answer` | Optional gold answer for GPT judge |
| `linked_tool_id` | Optional link to a tool-gold row for joint failure analysis (must share the same `question` text) |

## Metrics

### P0 — Tool routing

| Metric | Definition |
|--------|------------|
| `tool_selection_accuracy` | First predicted tool == first gold tool |
| `trajectory_match_loose` | Gold steps appear in order (extra steps allowed) |
| `trajectory_match_strict` | Same length and pairwise arg match |
| `required_tool_recall` | Every gold tool name appears at least once |
| `pass_rate` | TSA + loose trajectory + required recall + no forbidden tools |

Args use **loose** match: non-empty expected strings must be substrings of actual (case-insensitive).

### P1 — Task E2E

| Metric | Definition |
|--------|------------|
| `task_success_rate` | `require_tools` + `must_include` + `min_sources` + judge pass (when gold answer present) |
| `avg_correctness` / `avg_groundedness` / `avg_relevance` | GPT judge on agent answer vs gold, grounded in agent `sources` |

### P2 — Failure taxonomy

| Code | Meaning |
|------|---------|
| A1 | Wrong tool (first tool or forbidden tool) |
| A2 | Wrong / incomplete trajectory or arguments |
| A3 | Search ran but empty retrieval (`source_count == 0`) |
| A4 | Answer quality (judge or task rules) |
| A5 | Max steps, timeout, or runtime error |

Written to `agent_failures.jsonl` with `failure_code` and `failure_reason`.

**Failure postmortems (3 typical dev cases):** [`AGENT_FAILURE_POSTMORTEM.md`](AGENT_FAILURE_POSTMORTEM.md) — A1 wrong tool order, A2 list filter args, empty retrieval on scoped search (A3 pattern).

### P3 — LangSmith + `/ask` vs `/ask_agent`

- **LangSmith:** `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, optional `LANGCHAIN_PROJECT=fda-guidance-agent`
- **Compare script:** `experiment/compare_ask_vs_agent.py` runs `get_answer` vs `get_agent_answer` on the same questions and writes latency + optional judge scores.

#### Published dev numbers (2026-06-03, full production Qdrant)

**Agent eval** (`run_agent_eval_full_dev`):

| Metric | Value |
|--------|-------|
| Tool pass rate | 0.59 (19/32) |
| Tool selection accuracy | 0.91 |
| Trajectory match (loose) | 0.59 |
| Task success rate | 0.65 (13/20) |
| avg correctness / groundedness | 0.71 / 0.86 |
| Weakest category | T1 search (pass 22%) |
| Weakest task level | L3 multi-step (success 20%) |

**Fixed RAG vs agent** (`compare_ask_vs_agent.py --limit 0`, dataset `agent_task_gold_dev.json`, n=20):

| Metric | Fixed RAG (`/ask`) | Agent (`/ask_agent`) |
|--------|-------------------|----------------------|
| avg latency | 3853 ms | 4190 ms (+9%) |
| agent faster (per-row) | — | 7/20 rows |
| avg correctness | 0.71 | 0.71 |
| avg groundedness | 0.86 | 0.86 |
| avg relevance | 0.71 | 0.86 |

Judge scores computed on 14 rows with `gold_answer` (GPT-4o-mini judge).

**Latency / quality tradeoff:**

1. **List & browse (L1):** Agent skips vector retrieval → often 2–4× faster (e.g. 1.3–1.8s vs 3–5s). Quality depends on `list_guidance` keyword match; failures when metadata filter misses the topic.
2. **Search Q&A (L2):** Similar latency (~4–5s); judge scores usually tied. Fixed RAG is simpler and equally reliable when every question needs passages.
3. **Multi-step (L3):** Agent adds tool rounds → similar or slower latency, more failure modes (missing `search_guidance`, `min_sources`). Fixed RAG can still answer when retrieval hits, but cannot list-then-search by design.
4. **When to use which:** Use **fixed RAG** as default for passage-grounded Q&A; use **agent** when users need catalog browse, filters (center/year/type), or document overview before search.

Output: `experiment/runs/compare_ask_agent_20260603_003822/comparison_summary.json`

## Run

```bash
# Full dev eval (32 tool + 20 task; needs OPENAI_API_KEY, Qdrant, metadata)
python experiment/run_agent_eval.py --config experiment/configs/run_agent_eval_full_dev.json

# Tool-only (legacy config)
python experiment/run_agent_eval.py --config experiment/configs/run_agent_tool_dev.json

# Dry run 3 questions, no GPT judge
python experiment/run_agent_eval.py --limit 3 --preview --skip-judge

# Re-score saved trajectories
python experiment/run_agent_eval.py --score-only --skip-judge \
  --results experiment/runs/run_agent_eval_full_dev/agent_results.jsonl

# Compare fixed RAG vs agent (all 20 task dev rows)
python experiment/compare_ask_vs_agent.py --limit 0
```

Outputs under `experiment/runs/<run_id>/`:

- `agent_results.jsonl` — raw agent runs (answer, steps, sources)
- `agent_task_results.jsonl` — per-task success + judge scores
- `agent_metrics.json` — aggregated tool + task + failure taxonomy
- `agent_failures.jsonl` — failed rows with A1–A5 codes

## Unit tests (no LLM)

```bash
pytest tests/unit/test_agent_evaluators.py -q
```
