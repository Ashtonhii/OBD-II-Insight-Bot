# OBD Insight Bot

This repository provides a robust pipeline for analyzing OBD-II vehicle data using Python and the Ollama Granite model. The codebase is organized for reproducible analytics, benchmarking, and CLI interaction. Below is an overview of the main Python files and their roles:

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
- **notebooks/**: (Optional) For exploratory analysis and prototyping.

## Getting Started
1. Install dependencies in a virtual environment:
  ```sh
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
2. Run CLI scripts from the `src/` directory using the examples above.
3. Use `ask_obd.py` for interactive Q&A, `generate_golden_dataset.py` to build benchmarks, and `evaluate_pal.py` to assess PAL performance.

## License
MIT
