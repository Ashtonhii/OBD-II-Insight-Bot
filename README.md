# OBD Insight Bot

This repository provides a robust pipeline for analyzing OBD-II vehicle data using Python and the Ollama Granite model. The codebase is organized for reproducible analytics, benchmarking, and CLI interaction. Below is an overview of the main Python files and their roles:

## Multi-Agent Architecture (Current)

- **PAL Agent (`granite-code:8b`)**: Answers questions over OBD CSV data via generated pandas code.
- **RAG Agent (`granite3.3`)**: Answers general vehicle diagnostics questions from a document knowledge base.
- **Agent Registry (`src/agent_registry.py`)**: Provides a shared interface to run either agent (`pal` or `rag`).

For now, routing is manual (you choose the CLI/agent). This structure is ready for future automatic routing.

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
3. Run CLI scripts from the `src/` directory using the examples above.
4. Use:
   - `ask_obd.py` for PAL-over-CSV questions
   - `ask_diagnostics.py` for RAG diagnostics questions
   - `generate_golden_dataset.py` and `evaluate_pal.py` for PAL benchmarking

## License
MIT
