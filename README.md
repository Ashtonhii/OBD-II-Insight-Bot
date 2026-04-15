# OBD Insight Bot

This repository provides a robust pipeline for analyzing OBD-II vehicle data using Python and the Ollama Granite model. The codebase is organized for reproducible analytics, benchmarking, and CLI interaction. Below is an overview of the main Python files and their roles:

## Multi-Agent Architecture (Current)

- **PAL Agent (`granite-code:8b`)**: Answers questions over OBD CSV data via generated pandas code.
- **RAG Agent (`granite3.3`)**: Answers general vehicle diagnostics questions from a document knowledge base.
- **Orchestrator Agent (`granite3.3`)**: Routes each question to PAL or RAG based on intent.
- **Agent Registry (`src/agent_registry.py`)**: Provides a shared interface to run PAL or RAG directly.

Routing is now automatic via the orchestrator CLI (`src/ask_agent.py`).
The orchestrator now also supports session-based conversational memory.

## Python Files Overview

### src/data_loader.py
- **Purpose:** Loads and processes OBD-II CSV files, preserving all cells and handling ragged rows.
- **Features:**
  - Custom CSV reader for robust parsing
  - Exports cleaned and snapshot CSVs
  - Ensures no data loss during import

**Usage:**
```sh
python src/data_loader.py --input data/obdiidata/drive1.csv --output data/processed/drive1_cleaned.csv
```
*(Arguments may vary; see script for details.)*

### src/ollama_pal.py
- **Purpose:** Implements the PAL (Program-Aided Language) engine for code generation and execution using the Ollama Granite model.
- **Features:**
  - Normalizes LLM-generated code
  - Strips imports and read_csv calls for safety
  - Integrates with the data loader

**Usage:**
This module is imported by other scripts and is not typically run directly.

### src/ask_obd.py
- **Purpose:** Command-line interface (CLI) for querying OBD-II data using PAL.
- **Features:**
  - Accepts user questions and returns answers
  - Uses the PAL engine and loader
  - Provides help and usage instructions

**Usage:**
```sh
python src/ask_obd.py --help
python src/ask_obd.py --csv data/obdiidata/drive1.csv --question "What is the average RPM?"
python src/ask_obd.py --csv data/obdiidata/drive1.csv --question "What is the average RPM?" --model granite --show-code
```

### src/ollama_rag.py
- **Purpose:** Implements a retrieval-augmented generation (RAG) pipeline for vehicle diagnostics Q&A from local documents.
- **Features:**
  - Loads `.md` / `.txt` knowledge documents
  - Chunks and indexes text using TF-IDF style retrieval
  - Uses `granite3.3` to answer from retrieved context only

**Usage:**
This module is imported by `ask_diagnostics.py` and `agent_registry.py`.

### src/ask_diagnostics.py
- **Purpose:** CLI for diagnostics questions not tied to CSV telemetry data.
- **Features:**
  - Retrieves relevant document chunks from `knowledge/diagnostics/`
  - Sends grounded context to Ollama `granite3.3`
  - Optionally prints retrieved context

**Usage:**
```sh
python src/ask_diagnostics.py --help
python src/ask_diagnostics.py --question "What are common causes of P0300?"
python src/ask_diagnostics.py --question "How do I triage P0420?" --top-k 5 --show-context
```

### src/agent_registry.py
- **Purpose:** Shared multi-agent execution layer to call either PAL or RAG agent from one API.
- **Features:**
  - `run_pal_agent(...)`
  - `run_rag_agent(...)`
  - `run_agent("pal" | "rag", **kwargs)`

**Usage:**
Imported by future orchestration/router code. Not typically run directly.

### src/ollama_orchestrator.py
- **Purpose:** LLM router agent that decides whether a user prompt should go to PAL or RAG.
- **Features:**
  - Uses Ollama `granite3.3` for routing decisions
  - Includes recent session memory in routing prompts
  - Persists each turn in Redis for multi-turn context
  - Returns route + rationale
  - Dispatches to PAL for data/CSV computation questions
  - Dispatches to RAG for diagnostics knowledge questions

**Usage:**
Imported by `ask_agent.py`.

### src/ask_agent.py
- **Purpose:** Unified CLI entrypoint for the multi-agent system.
- **Features:**
  - Accepts one user question
  - Uses orchestrator to choose `pal` vs `rag`
  - Maintains conversational memory by `--session-id`
  - Prints route rationale and agent details optionally

**Usage:**
```sh
python src/ask_agent.py --help
python src/ask_agent.py --question "What does P0300 usually indicate?" --session-id tech1 --show-route
python src/ask_agent.py --question "What is average RPM in this log?" --csv data/obdiidata/drive1.csv --session-id tech1 --show-route --show-details
```

### src/conversation_memory.py
- **Purpose:** Persists orchestrator session history for conversational routing context.
- **Features:**
  - Stores turns in Redis (default: `redis://localhost:6379/0`)
  - Sanitizes session IDs for safe Redis keys
  - Uses key prefix `obd:orchestrator:memory:`
  - Formats recent turns for router prompt context

**Usage:**
Imported by `ollama_orchestrator.py`.

### src/generate_golden_dataset.py
- **Purpose:** Builds a golden dataset for benchmarking PAL accuracy.
- **Features:**
  - Uses the robust loader to process raw data
  - Generates reference answers for evaluation

**Usage:**
```sh
python src/generate_golden_dataset.py --input data/obdiidata/ --output data/golden/golden_dataset.csv
```

### src/evaluate_pal.py
- **Purpose:** Evaluates PAL-generated answers against the golden dataset.
- **Features:**
  - Computes ESR/EM metrics for accuracy
  - Automates benchmarking and reporting

**Usage:**
```sh
python src/evaluate_pal.py --golden data/golden/golden_dataset.csv --results data/results/pal_answers.csv
```

### src/evaluate_router.py
- **Purpose:** Evaluates orchestrator routing accuracy (`pal` vs `rag`) on a labeled prompt set.
- **Features:**
  - Runs each prompt through router decision logic
  - Computes overall routing accuracy
  - Exports per-prompt predictions and rationale
  - Prints a confusion table (`expected_route` vs `predicted_route`)

**Usage:**
```sh
python src/evaluate_router.py --golden-csv data/golden/router_golden_dataset.csv --output-csv data/golden/router_eval_results.csv
python src/evaluate_router.py --router-model granite3.3 --limit 10
```

**Labeled Dataset Format (`data/golden/router_golden_dataset.csv`):**
- `id`: unique integer
- `question`: user prompt
- `expected_route`: `pal` or `rag`
- `csv_path` (optional): CSV path hint for data-oriented prompts

## Data Organization
- **data/obdiidata/**: Contains raw OBD-II CSV files for analysis.
- **knowledge/diagnostics/**: Contains RAG knowledge documents (`.md`/`.txt`) for diagnostics Q&A.
- **notebooks/**: (Optional) For exploratory analysis and prototyping.

## Getting Started
1. Install dependencies in a virtual environment:
  ```sh
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
2. Ensure Ollama is running and models are available:
  ```sh
  ollama pull granite-code:8b
  ollama pull granite3.3
  ```
3. Ensure Redis is running for conversation memory:
  ```sh
  # default connection used by conversation_memory.py
  # redis://localhost:6379/0
  ```
  Optional: set a custom Redis URL:
  ```sh
  set REDIS_URL=redis://localhost:6379/0
  ```
  Docker-based Redis (recommended on Windows):
  ```sh
  # Start Docker Desktop first, then run:
  docker run -d --name obd-redis -p 6379:6379 redis:7

  # If container already exists:
  docker start obd-redis

  # Verify Redis is healthy:
  docker exec -it obd-redis redis-cli PING
  ```
4. Run CLI scripts from the `src/` directory using the examples above.
5. Use:
   - `ask_obd.py` for PAL-over-CSV questions
   - `ask_diagnostics.py` for RAG diagnostics questions
  - `ask_agent.py` for automatic routing (orchestrator -> PAL/RAG)
  - `generate_golden_dataset.py` and `evaluate_pal.py` for PAL benchmarking
  - `evaluate_router.py` for router accuracy benchmarking

## Memory Testing (PAL + RAG)

Use the orchestrator entrypoint (`ask_agent.py`) for memory testing. The `ask_obd.py` script bypasses orchestrator memory.

### Provided Test Scripts

- `test_pal_memory.ps1`: runs 2 PAL-oriented turns in the same session (`memtest-pal`)
- `test_rag_memory.ps1`: runs 2 RAG-oriented turns in the same session (`memtest-rag`)

Run from repository root:

```sh
.\test_pal_memory.ps1
.\test_rag_memory.ps1
```

If PowerShell blocks script execution:

```sh
Set-ExecutionPolicy -Scope Process RemoteSigned
```

### Verify Stored Memory in Redis

List memory keys:

```sh
docker exec -it obd-redis redis-cli KEYS "obd:orchestrator:memory:*"
```

Inspect PAL test session:

```sh
docker exec -it obd-redis redis-cli GET "obd:orchestrator:memory:memtest-pal"
```

Inspect RAG test session:

```sh
docker exec -it obd-redis redis-cli GET "obd:orchestrator:memory:memtest-rag"
```

Pretty-print JSON payload:

```sh
$raw = docker exec -i obd-redis redis-cli GET "obd:orchestrator:memory:memtest-pal"
$raw | .\venv\Scripts\python.exe -m json.tool
```

### Clear Memory History

Clear one session history:

```sh
docker exec -it obd-redis redis-cli DEL "obd:orchestrator:memory:memtest-pal"
docker exec -it obd-redis redis-cli DEL "obd:orchestrator:memory:memtest-rag"
```

Clear all orchestrator session histories:

```sh
docker exec -it obd-redis redis-cli EVAL "for _,k in ipairs(redis.call('keys', ARGV[1])) do redis.call('del', k) end return 'OK'" 0 "obd:orchestrator:memory:*"
```

Expected result: each session key stores a JSON object with a `turns` array that grows after each run.

## License
MIT
