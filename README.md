# OBD Insight Bot

A multi-agent conversational system for analysing OBD-II vehicle telemetry and answering vehicle diagnostic questions using locally-hosted LLMs via [Ollama](https://ollama.com).

Two specialised agents handle different question types:
- **PAL Agent** — generates and executes pandas code to answer data-analytical questions over OBD-II CSV logs.
- **RAG Agent** — retrieves and synthesises answers from a structured diagnostic knowledge base.

A routing orchestrator decides which agent handles each question, rewrites ambiguous follow-up questions using conversation history, and persists session context in Redis.

---

## Architecture Overview

```
User question
      │
      ▼
 ask_agent.py  (CLI entry point)
      │
      ▼
 OllamaOrchestrator  (ollama_orchestrator.py)
  ├─ Reference rewriting   (resolves pronouns using session memory)
  ├─ Route classification  (granite3.3 → "pal" or "rag")
  ├─ ConversationMemory    (Redis-backed session store)
  │
  ├─── PAL route ──► OllamaPAL  (ollama_pal.py)
  │                   ├─ Code generation  (granite-code:8b)
  │                   ├─ AST safety validation
  │                   ├─ Sandboxed execution
  │                   ├─ Retry with error feedback
  │                   └─ Result summarisation
  │
  └─── RAG route ──► OllamaDiagnosticsRAG  (ollama_rag.py)
                      ├─ DTC-aware document chunking
                      ├─ Exact DTC code match (Tier 1)
                      ├─ TF-IDF cosine similarity (Tier 2)
                      └─ Answer synthesis  (granite3.3)
```

---

## Models

| Agent | Model | Purpose |
|---|---|---|
| Orchestrator / Router | `granite3.3` | Route classification, reference rewriting |
| PAL | `granite-code:8b` | Pandas code generation and result summarisation |
| RAG | `granite3.3` | Answer synthesis from retrieved context |

---

## Project Structure

```
obd_insight_bot/
├── src/
│   ├── ask_agent.py                  # Unified orchestrator CLI
│   ├── ask_obd.py                    # Standalone PAL CLI
│   ├── ask_diagnostics.py            # Standalone RAG CLI
│   ├── ollama_orchestrator.py        # Router, rewriter, dispatcher
│   ├── ollama_pal.py                 # PAL agent
│   ├── ollama_rag.py                 # RAG agent
│   ├── agent_registry.py             # Agent dispatch interface
│   ├── conversation_memory.py        # Redis session store
│   ├── data_loader.py                # OBD-II CSV loading and preprocessing
│   ├── generate_golden_dataset.py    # PAL golden dataset generator
│   ├── generate_rag_golden_dataset.py# RAG golden dataset generator
│   ├── evaluate_pal.py               # PAL evaluation (ESR + EM)
│   ├── evaluate_rag.py               # RAG evaluation (accuracy + mode breakdown)
│   ├── evaluate_router.py            # Router evaluation (accuracy + confusion matrix)
│   └── test_pal_security.py          # Adversarial AST safety test suite
├── data/
│   ├── obdiidata/                    # Raw OBD-II CSV log files
│   ├── golden/                       # Golden datasets and evaluation results
│   └── processed/                    # Cleaned CSV outputs
├── knowledge/
│   └── diagnostics/
│       └── fault_codes_database.md   # DTC knowledge base for RAG
├── test_pal_memory.ps1               # 2-turn PAL memory smoke test
├── test_pal_conversation.ps1         # 5-test multi-turn PAL conversation suite
└── test_rag_conversation.ps1         # 5-test multi-turn RAG/hybrid conversation suite
```

---

## Getting Started

### 1. Install dependencies

```sh
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull Ollama models

```sh
ollama pull granite-code:8b
ollama pull granite3.3
```

### 3. Start Redis (required for conversation memory)

Using Docker (recommended on Windows):

```sh
docker run -d --name obd-redis -p 6379:6379 redis:7

# If the container already exists:
docker start obd-redis

# Verify Redis is healthy:
docker exec -it obd-redis redis-cli PING
```

Set a custom Redis URL if needed:

```sh
set REDIS_URL=redis://localhost:6379/0
```

---

## Usage

### Unified orchestrator (recommended)

Routes automatically to PAL or RAG based on question intent. Requires `--session-id` for conversation memory.

```sh
# Data analysis question → PAL
python src/ask_agent.py --question "What is the average engine RPM?" --csv data/obdiidata/drive1.csv --session-id sess1 --show-route --show-details

# Diagnostic knowledge question → RAG
python src/ask_agent.py --question "What does P0300 mean?" --session-id sess1 --show-route

# Follow-up using conversation context
python src/ask_agent.py --question "How does that compare to the maximum?" --csv data/obdiidata/drive1.csv --session-id sess1 --show-route
```

**Flags:**
- `--show-route` — prints route decision, rationale, and rewritten question (if rewrite occurred)
- `--show-details` — prints generated pandas code and result preview (PAL) or retrieved chunks (RAG)

### Standalone PAL (no routing, no memory)

```sh
python src/ask_obd.py --csv data/obdiidata/drive1.csv --question "What is the average RPM?"
```

### Standalone RAG (no routing, no memory)

```sh
python src/ask_diagnostics.py --question "What is P0420?"
python src/ask_diagnostics.py --question "What is P0420?" --show-context --top-k 5
```

---

## Source Files

### `src/ollama_orchestrator.py`

Central controller. For every question it:

1. **Rewrites referential follow-ups** — if the question contains a pronoun or vague reference (`"that"`, `"it"`, `"convert that"`, `"based on"`, etc.) and the session has prior history, the LLM rewrites the question into a fully self-contained form using the last 2 turns. Falls back silently to the original if the rewrite fails.
2. **Classifies the route** — LLM (`granite3.3`) returns `{"route": "pal"|"rag", "rationale": "..."}` at temperature 0.0. Falls back from PAL to RAG if no CSV is provided.
3. **Dispatches to the agent** — injects the last 3 turns of session context into the agent call.
4. **Saves the turn** — appends question, route, answer, CSV path, and rationale to Redis (capped at 50 turns).

### `src/ollama_pal.py`

Implements the PAL (Program-Aided Language) pipeline:

1. **Code generation** — `granite-code:8b` receives the DataFrame schema (column names, dtypes, 5-row sample) and is instructed to output Python code only, storing the final answer in `result`. The prompt explicitly forbids answering from conversation context.
2. **Code extraction** — strips fenced code blocks, prose prefix lines, redundant imports, and `df = pd.read_csv(...)` lines from the raw LLM output.
3. **AST safety validation** — blocks dangerous constructs before execution:
   - Node types: `Import`, `ImportFrom`, `Lambda`, `ClassDef`, `FunctionDef`, `Try`, `With`, `Global`, `Nonlocal`, `Delete`, `Raise`, and async variants
   - Names: `eval`, `exec`, `open`, `compile`, `input`, `breakpoint`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__import__`
   - DataFrame methods: `to_csv`, `to_json`, `to_sql`, `to_parquet`, `to_pickle`, `to_hdf`, `to_feather`
4. **Sandboxed execution** — `exec()` with only 8 safe built-ins (`len`, `min`, `max`, `sum`, `round`, `sorted`, `abs`) and the DataFrame as a shallow copy.
5. **Retry** — on first failure, re-calls code generation with the exception message as feedback and retries once.
6. **Fallback** — if the retry also fails, returns a user-facing message asking for clarification rather than raising an exception.
7. **Result summarisation** — the computed result is narrated into natural language by the LLM.

### `src/ollama_rag.py`

Implements the RAG pipeline:

1. **Chunking** — documents are split into 900-character chunks with 180-character overlap. For DTC fault code databases, a line-based parser extracts each code+description pair as a self-contained chunk.
2. **Retrieval (Tier 1)** — if the question contains a DTC code pattern (`[PCBU][0-9A-F]{4}`), chunks containing that code are returned directly.
3. **Retrieval (Tier 2)** — otherwise, TF-IDF cosine similarity is computed between the query and all chunks; top-k by score are returned.
4. **Answer synthesis** — retrieved chunks are passed to `granite3.3` with a system prompt instructing it to answer only from the provided context.

### `src/conversation_memory.py`

Redis-backed session store. Each session is one Redis key (`obd:orchestrator:memory:<session_id>`), storing a JSON object with a `turns` array. Each turn records: timestamp, question (post-rewrite), route, answer, CSV path, and router rationale. Sessions are capped at 50 turns.

### `src/data_loader.py`

Preprocessing pipeline for OBD-II CSV files:
- Handles ragged rows (unequal column counts) via a custom `csv.reader` wrapper
- Sanitises column names to lowercase snake_case and deduplicates
- Coerces 24 known OBD-II metric columns to float (`engine_rpm`, `vehicle_speed`, `coolant_temperature`, `engine_load`, `catalyst_temperature_bank1_sensor1`, `short_term_fuel_trim_bank_1`, `timing_advance`, `control_module_voltage`, and more)
- Optionally parses timestamps, drops duplicates, and drops rows with missing required columns

### `src/agent_registry.py`

Thin dispatch layer exposing `run_pal_agent()`, `run_rag_agent()`, and `run_agent("pal"|"rag", **kwargs)`. Wraps `ollama_pal` and `ollama_rag` into a unified `AgentResponse` dataclass.

---

## Evaluation

### PAL Evaluation

**Generate dataset (100 questions, 20 per CSV file):**

```sh
python src/generate_golden_dataset.py
```

Generates `data/golden/pal_golden_dataset.csv`. Questions cover 18+ categories including basic aggregates, percentiles, IQR, conditional means, percentage thresholds, unit conversions, and cross-column correlations. Selection is interleaved across all categories to ensure balanced coverage.

**Run evaluation:**

```sh
python src/evaluate_pal.py
python src/evaluate_pal.py --limit 20
python src/evaluate_pal.py --row-id 5
```

Reports **ESR** (Execution Success Rate — fraction of questions where generated code ran without error) and **EM** (Exact Match Rate — fraction of executed questions where the answer matched the expected value within tolerance 1×10⁻⁴). Automatically prints a failure mode breakdown classifying both execution failures (e.g. `column_name_error`, `syntax_or_safety_rejection`) and semantic mismatches (e.g. `unconditional_aggregate_for_conditional_query`, `unit_conversion_error`).

**Output:** `data/golden/pal_eval_results.csv`

---

### RAG Evaluation

**Generate dataset (120 questions, 20 per mode across 6 modes):**

```sh
python src/generate_rag_golden_dataset.py
```

Generates `data/golden/rag_golden_dataset.csv`. Each of 60 evenly-spaced DTC codes is tested in 6 question modes:

| Mode | Question form | Tests |
|---|---|---|
| `code_to_description` | `"P0171"` | Exact DTC match retrieval |
| `description_to_code` | Full description | TF-IDF retrieval |
| `nl_what_does_mean` | `"What does P0171 mean?"` | Natural language + exact match |
| `nl_what_is` | `"What is P0171?"` | Natural language + exact match |
| `nl_what_causes` | `"What causes fault code P0171?"` | Natural language + exact match |
| `keyword_fragment` | 2–3 keywords from description | TF-IDF with limited vocabulary |

**Run evaluation:**

```sh
python src/evaluate_rag.py
python src/evaluate_rag.py --limit 20 --model granite3.3 --top-k 4
```

Scoring uses four-tier substring matching (exact → expected-in-model → model-in-expected → mismatch) to handle natural LLM verbosity. Prints accuracy split by question mode after each run.

**Output:** `data/golden/rag_eval_results.csv`

---

### Router Evaluation

**Dataset:** `data/golden/router_golden_dataset.csv` — 60 questions across 3 categories:
- `unambiguous` (30): clearly PAL or clearly RAG
- `hybrid` (10): numerical observations with diagnostic framing (all RAG)
- `follow_up` (20): phrased as conversation continuations (mixed)

**Run evaluation:**

```sh
python src/evaluate_router.py
python src/evaluate_router.py --limit 10
```

Reports overall routing accuracy, per-category accuracy, a confusion matrix, and an asymmetric failure analysis. RAG→PAL misclassifications are flagged as hard failures; PAL→RAG as degraded responses. The evaluator checks whether false positives ≤ false negatives and prints a warning if not.

**Output:** `data/golden/router_eval_results.csv`

---

### PAL Security Test Suite

Tests the AST safety validator directly with 43 adversarial inputs, independent of LLM outputs.

```sh
python src/test_pal_security.py
```

Covers 6 attack vectors:

| Vector | Cases | Examples |
|---|---|---|
| `direct_builtin` | 9 | `eval`, `exec`, `open`, `compile`, `globals`, `locals` |
| `import_based` | 5 | `import os`, `__import__()`, `subprocess`, `socket` |
| `getattr_escalation` | 4 | `getattr(df, 'to_csv')`, `setattr`, `delattr` |
| `dataframe_exfiltration` | 7 | `to_csv`, `to_json`, `to_sql`, `to_parquet`, `to_pickle`, `to_hdf`, `to_feather` |
| `lambda_obfuscation` | 4 | `lambda` wrapping `eval`, `exec`, `__import__`, `open` |
| `safe_pandas` | 10 | Legitimate operations that must NOT be blocked |

Exits with code 0 if all cases pass, 1 if any fail.

---

## Conversation Testing

Multi-turn conversation test scripts for verifying that session memory and reference rewriting work end-to-end:

```sh
# 2-turn smoke test (PAL)
.\test_pal_memory.ps1

# 5-test multi-turn PAL suite
# Tests: aggregate→comparison, filter→unit conversion, percentile→threshold count,
#        correlation→interpretation, three-turn mean→%above→std chain
.\test_pal_conversation.ps1

# 5-test multi-turn RAG/hybrid suite
# Tests: code lookup→cause, description→code→explanation, hybrid PAL→RAG,
#        sequential codes→relationship, three-turn catalyst temp + P0420 chain
.\test_rag_conversation.ps1
```

If PowerShell blocks script execution:

```sh
Set-ExecutionPolicy -Scope Process RemoteSigned
```

---

## Session Memory Management

List all active sessions:
```sh
docker exec -it obd-redis redis-cli KEYS "obd:orchestrator:memory:*"
```

Inspect a session:
```sh
docker exec -it obd-redis redis-cli GET "obd:orchestrator:memory:sess1"

# Pretty-print JSON
$raw = docker exec -i obd-redis redis-cli GET "obd:orchestrator:memory:sess1"
$raw | .\.venv\Scripts\python.exe -m json.tool
```

Clear one session:
```sh
docker exec -it obd-redis redis-cli DEL "obd:orchestrator:memory:sess1"
```

Clear all sessions:
```sh
docker exec -it obd-redis redis-cli EVAL "for _,k in ipairs(redis.call('keys', ARGV[1])) do redis.call('del', k) end return 'OK'" 0 "obd:orchestrator:memory:*"
```

---

## License
MIT
