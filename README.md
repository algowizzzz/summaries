**Summarization Agent (v1) Technical Documentation**

This document provides detailed technical documentation for the CLI-driven Summarization Agent (v1). It describes the architecture, modules, configuration, data flows, caching, and testing procedures necessary to deploy and operate the agent effectively.

---

## 1. Introduction

### 1.1 Purpose & Scope of the Agent

The Summarization Agent (v1) automates the generation of hierarchical and cross-sectional summaries for structured JSON documents, specifically SEC company filings (40-F and 6-K). It allows users to batch-process entire folders of filings via a command-line interface, producing file-level, node-level (yearly), master-level (multi-year), and thematic cross-sectional summaries. This version focuses on core summarization functionality without user-interactive features.

### 1.2 Intended Audience

* **Risk Analysts & Executives:** Require quick access to structured summaries of large volumes of regulatory filings.
* **Data Engineers & DevOps:** Responsible for installation, configuration, and maintenance of the summarization pipeline.
* **Compliance Officers & Auditors:** Need traceable, versioned outputs for audit and regulatory reporting.

### 1.3 Overview of Functional Capabilities

* **Batch Ingestion:** Recursively reads JSON filings from a well-structured directory.
* **Chunking Logic:** Splits documents into semantic chunks or processes entire files for LLM summarization, with a 50k-word checkpoint.
* **Prompt-Driven Summarization:** Uses static Markdown templates with placeholder substitution.
* **Metadata Annotation:** Attaches unique identifiers, timestamps, key-term extraction, and entity detection to each summary.
* **Output Generation:** Writes summaries and metadata as Markdown and JSON files in a mirrored folder hierarchy.
* **Caching:** Avoids reprocessing unchanged chunks via a simple cache layer.

### 1.4 v1 Feature Summary

* **Non-Interactive CLI:** No runtime prompt editing; all prompts are preconfigured.
* **Four Modes:** `file`, `node`, `master`, and `cross` summarization modes.
* **50k-Word Checkpoint:** Ensures single-document LLM calls or logical paragraph splitting.
* **Retry Logic:** Automatically retries LLM API calls up to two times on transient failures.
* **Cache Manager:** Skips already summarized chunks to reduce API usage and latency.
* **Structured Outputs:** Consistent naming conventions and metadata schemas for downstream integration.

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
[ CLI Frontend ] → [ Orchestrator ] → [ Cache Manager ] → [ Chunking Module ]
                                              ↘ [ Summarization Module ]
                                              ↘ [ Metadata Generator ]
                                              ↘ [ Output Writer ]
```

### 2.2 Module Responsibilities

* **CLI Frontend (summarizer.py):** Parses user arguments and invokes the orchestrator with the specified mode and paths.
* **Orchestrator:** Coordinates workflow, checks cache, and dispatches jobs to child modules.
* **Cache Manager:** Detects and skips chunks that have been previously summarized.
* **Chunking Module:** Converts raw JSON filings or summary files into `Chunk` objects, enforcing the 50k-word limit and splitting on paragraph boundaries when necessary.
* **Summarization Module:** Loads the appropriate prompt template, substitutes metadata placeholders, calls the LLM API, and returns the raw summary text.
* **Metadata Generator:** Constructs a metadata record for each summary, including unique IDs, relationship mappings, extracted key terms, entities, and timestamps.
* **Output Writer:** Persists summaries and metadata to disk, preserving a mirrored folder structure under the output directory.

### 2.3 Data Flow Overview

1. **User Invocation:** `summarizer.py --mode file --input-dir <path> --output-dir <path> --prompt-set sec_prompts_v1.json`
2. **Directory Traversal:** Orchestrator lists JSON files or summary files based on mode.
3. **Cache Check:** Cache Manager loads and compares chunk hashes to skip cached ones.
4. **Chunk Generation:** Chunking Module reads each JSON, splits or wraps into chunks.
5. **LLM Call:** Summarization Module executes each chunk through the LLM with the selected prompt template.
6. **Metadata & Output:** Metadata Generator annotates, Output Writer writes files to the output hierarchy.

### 2.4 Technology Stack & Dependencies

* **Language:** Python 3.10+
* **Libraries:** `openai` (or another LLM client), `tiktoken` for token counting, `PyYAML`/`json`, `argparse`, `pytest` for tests
* **Environment Variables:**

  * `OPENAI_API_KEY` for LLM authentication
  * `OPENAI_API_URL` (optional) to override default endpoint
  * `AGENT_LOG_PATH` to specify custom log file location
* **Caching:** Local JSON cache in `.cache/` directory to avoid reprocessing unchanged chunks
* **Environment:** Virtualenv or Docker container
* **Storage:** Local filesystem or network-mounted storage for input/output

---

## 3. Installation & Setup

### 3.1 Prerequisites

* **Python 3.10+** installed
* **LLM API Credentials:** Exported as `OPENAI_API_KEY`
* **Disk Space:** Minimum 5 GB for input JSONs and generated summaries
* **GitHub Actions:** (optional) for CI/CD integration

### 3.2 Repository Structure

```
model-doc-agent/
├─ summarizer.py             # CLI entrypoint
├─ src/
│  ├─ chunker.py            # Chunking logic
│  ├─ summarization.py      # Summarization orchestrator
│  ├─ cache.py              # Cache management
│  ├─ metadata.py           # Metadata generation
│  └─ writer.py             # Output writer
├─ templates/               # Prompt templates (.md)
│  ├─ sec_file_summary.md
│  ├─ sec_node_summary.md
│  ├─ sec_master_summary.md
│  └─ sec_cross_sectional_summary.md
├─ sec_prompts_v1.json      # Mapping modes to template files
├─ sample_sec/              # Sample data for quick testing
├─ requirements.txt         # Python dependencies
└─ .github/workflows/       # CI pipeline configs
```

### 3.3 Virtual Environment or Docker Setup

**Virtualenv:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Docker:**

```bash
docker build -t summarizer-agent:v1 .
docker run --rm -e OPENAI_API_KEY --mount src=$(pwd)/sample_sec,target=/data/sample_sec summarizer-agent:v1 \
  summarizer.py --mode file --input-dir /data/sample_sec --output-dir /data/out --prompt-set sec_prompts_v1.json
```

### 3.4 Sample Data

The `sample_sec/` directory contains one CIK with a single 40-F JSON for year 2022. Use it to validate chunking, summarization, cache functionality, and output formats. Consider adding a `sample_news/` directory with a similar structure to `Data/news/` for testing news processing.

### 3.5 Input Data Structure

The agent expects input data under a root directory (e.g., `Data/`). The primary subdirectories are `Banks`, `mutual_fund_filings`, and `news`.

**Filings (Banks and Mutual Funds):**

Filings are generally organized in a flat structure within specific entity folders:

```
Data/
├─ Banks/
│  ├─ [Bank_NameA]/             # e.g., Bank_of_Montreal
│  │   ├─ [FilingType]_[Date]_[AccessionNumber].json  # e.g., 6-K_2020-05-27_0001193125-20-152984.json
│  │   └─ ... (other filings for Bank_NameA)
│  ├─ [Bank_NameB]/
│  │   └─ ...
│  └─ ...
├─ mutual_fund_filings/
│  ├─ [Trust_NameA]/            # e.g., iShares_Trust
│  │   ├─ [FilingType]_[Date]_[AccessionNumber].json  # e.g., 497_2024-08-01_0001193125-24-190145.json
│  │   └─ ... (other filings for Trust_NameA)
│  ├─ [Trust_NameB]/
│  │   └─ ...
│  └─ ...
```
*   Filenames typically encode metadata such as `FilingType`, `Date`, and `AccessionNumber`.

**News Articles:**

News articles are organized hierarchically:

```
Data/
└─ news/
   ├─ [Category]/                # e.g., CA Banks, reg, usbank
   │  ├─ [Source_Entity]/        # e.g., cibc, cfpb, usbank (under usbank/news/)
   │  │   ├─ index.json          # Optional index at the entity level
   │  │   └─ [YYYY-MM]/          # Year-Month directory
   │  │       ├─ index.json      # Manifest of articles for the month
   │  │       └─ [article_filename].json # Individual news article JSON
   │  │       └─ ...
   │  └─ ...
   └─ ...
```
*   Individual news article JSON files (e.g., `20250116_…National.json` or `elavon-offers…html.json`) contain the article's content and metadata.
*   `index.json` files within `YYYY-MM` directories often act as manifests, listing articles for that period.

---

## 4. Configuration

Configuration of the Summarization Agent is driven by command‑line flags and the prompt‑set JSON. No code changes are required to adjust behavior in v1.

### 4.1 CLI Usage & Flags

* `--mode <file|node|master|cross>`
* `--input-dir <path>`
* `--output-dir <path>`
* `--prompt-set <path>`
* `--max-words <integer>` (override default 50 000-word checkpoint)
* `--retries <integer>` (default: 2)
* `--no-cache` (force re-summarization)
* `--verbose` (enable detailed logging)

### 4.2 Prompt‑Set JSON (`sec_prompts_v1.json`)

To support multiple filing types across regions (e.g. 40-F, 6-K, 20-F, 10-K, 10-Q, 8-K), `sec_prompts_v1.json` is organized as a nested mapping of filing types to mode-specific templates. For example:

```json
{
  "40-F": {
    "file": "templates/sec_file_summary_40F.md",
    "node": "templates/sec_node_summary_40F.md",
    "master": "templates/sec_master_summary_40F.md",
    "cross": "templates/sec_cross_sectional_40F.md"
  },
  "6-K": {
    "file": "templates/sec_file_summary_6K.md",
    "node": "templates/sec_node_summary_6K.md",
    "master": "templates/sec_master_summary_6K.md",
    "cross": "templates/sec_cross_sectional_6K.md"
  },
  "20-F": {
    "file": "templates/sec_file_summary_20F.md",
    "node": "templates/sec_node_summary_20F.md",
    "master": "templates/sec_master_summary_20F.md",
    "cross": "templates/sec_cross_sectional_20F.md"
  },
  "10-K": {
    "file": "templates/sec_file_summary_10K.md",
    "node": "templates/sec_node_summary_10K.md",
    "master": "templates/sec_master_summary_10K.md",
    "cross": "templates/sec_cross_sectional_10K.md"
  },
  "10-Q": {
    "file": "templates/sec_file_summary_10Q.md",
    "node": "templates/sec_node_summary_10Q.md",
    "master": "templates/sec_master_summary_10Q.md",
    "cross": "templates/sec_cross_sectional_10Q.md"
  },
  "8-K": {
    "file": "templates/sec_file_summary_8K.md",
    "node": "templates/sec_node_summary_8K.md",
    "master": "templates/sec_master_summary_8K.md",
    "cross": "templates/sec_cross_sectional_8K.md"
  }
}
```

* The Orchestrator reads the folder name (`filing_type`) and looks up the corresponding mapping.
* If a filing type is not found, it falls back to a generic template set under the key `"default"`.

\---json
{
"file": "templates/sec\_file\_summary.md",
"node": "templates/sec\_node\_summary.md",
"master": "templates/sec\_master\_summary.md",
"cross": "templates/sec\_cross\_sectional\_summary.md"
}

```

### 4.3 Template Files Directory
All prompt templates are stored under `templates/`. Filenames must match `sec_prompts_v1.json` entries.

### 4.4 Tokenizer & Max‑Token Settings
- Uses `tiktoken` to estimate token counts and guard against LLM input size limits.
- Word threshold = 50 000 (override via `--max-words`).
- Splits long inputs on paragraph boundaries (`\n\n`).

### 4.5 Caching & Incremental Runs
- **Cache Store:** Persists chunk hashes → summary IDs in `.cache/`.
- **Behavior:** Skips cached chunks; use `--no-cache` to ignore.

### 4.6 Environment Variables & Overrides
- `OPENAI_API_URL` to override default endpoint.
- `AGENT_LOG_PATH` to direct logs to a file.

---

## 5. Core Modules

Detailed description of each Python module, public APIs, and key algorithms.

### 5.1 Orchestrator (`src/summarizer.py`)
- **Main Functions:** `main()`, `run_mode()`
- **Responsibilities:** CLI parsing, workflow coordination, cache integration.

### 5.2 Cache Manager (`src/cache.py`)
- **Class:** `CacheManager`
- **Methods:** `load_cache()`, `is_cached()`, `save_cache()`
- **Purpose:** Reduce redundant LLM calls, speed up iterative runs.

### 5.3 Chunking Module (`src/chunker.py`)
- **Class:** `Chunker(mode: str, max_words: int, tokenizer: Tokenizer)`
- **Method:** `chunk(input_path: str) -> List[Chunk]`
- **Strategies:**
  - **File mode:** One chunk per JSON `sections[]` entry.
  - **Node/Master/Cross:** One chunk per summary JSON.
  - **Word Count Checkpoint:** Splits on `\n\n` when > max.
- **Metadata:** Includes `parent_document_id`, `sequence_index`, `is_split`.

### 5.4 Summarization Module (`src/summarization.py`)
- **Function:** `summarize_chunk(chunk: Chunk, prompt_template: str) -> str`
- **Workflow:** Prompt rendering, `openai.ChatCompletion` call, retry logic.

### 5.5 Metadata Generator (`src/metadata.py`)
- **Function:** `generate_metadata(chunk: Chunk, summary_text: str) -> Metadata`
- **Fields:** `summary_id`, `parent_id`, `summary_type`, `relationship_ids`, `generated_timestamp`, `key_terms`, `entities_mentioned`

### 5.6 Output Writer (`src/writer.py`)
- **Functions:** `write_summary()`
- **Behavior:** Creates mode-specific subdirectories, writes Markdown and JSON.

### 5.7 Output Directory Structure & Metadata

**Final Output Directory Layout:**
```

<output-dir>/
file-level/
{CIK}/
{filing\_type}/
{year}/
{chunk\_id}\_summary.md
{chunk\_id}\_meta.json
node-level/
{CIK}/
{filing\_type}/
{year}/
node\_summary.md
node\_summary\_meta.json
master-level/
{CIK}/
{filing\_type}/
master\_summary.md
master\_summary\_meta.json
cross-sectional/
{CIK}/
{filing\_type}/
{theme\_type}\_summary.md
{theme\_type}\_meta.json

````

**Metadata File Schema (`*_meta.json`):**
```json
{
  "summary_id": "string",               // Unique ID of this summary
  "parent_id": "string|null",           // Parent summary or document ID
  "file_id": "string|null",             // Original document or chunk ID
  "category": "string",                 // e.g., "SEC_Filings"
  "domain": "string",                   // e.g., "Market_Risk"
  "doc_type": "string",                 // Filing type (40-F, 6-K, etc.)
  "summary_type": "string",             // file-level/node-level/master-level/cross-sectional
  "relationship_ids": ["string"],       // Related summary IDs for aggregation
  "generated_timestamp": "ISO8601",     // When summary was created
  "key_terms": ["string"],              // Extracted key terms
  "entities_mentioned": ["string"],     // Detected entities
  "theme_type": "string|null"           // For cross-sectional summaries
}
````

### 6. Prompt Templates1 File-Level Templates

`sec_file_summary.md` – per-section summaries.

### 6.2 Node-Level Templates

`sec_node_summary.md` – year-level aggregations.

### 6.3 Master-Level Templates

`sec_master_summary.md` – multi-year roll-ups.

### 6.4 Cross-Sectional Variations

`sec_cross_sectional_summary.md`, plus themed variants.

### 6.5 Placeholder Definitions

`CIK`, `filing_type`, `year`, `section_title`, `start_year`, `end_year`, `theme_type`.

---

## 7. Usage Examples

### 7.1 File-Level Summaries on Sample Data

```bash
# Example for a bank filing
summarizer.py --mode file --input-dir Data/Banks/Bank_of_Montreal --output-dir out/file-level --prompt-set sec_prompts_v1.json --verbose

# Example for a news article (assuming 'file' mode is used, or a new 'news' mode)
summarizer.py --mode file --input-dir Data/news/reg/cfpb/2025-01 --output-dir out/news-summaries --prompt-set sec_prompts_v1.json --verbose
```

### 7.2 Generating Year-Level Summaries

(Node-level summaries are more applicable to structured filings with sections. This example assumes input from a previous file-level summary step if those summaries are structured, or directly from filings if the orchestrator handles section extraction from raw filings for node mode.)
```bash
# Assuming file-level summaries for a bank's filings are in out/file-level/Banks/Bank_of_Montreal
summarizer.py --mode node --input-dir Data/Banks/Bank_of_Montreal --output-dir out/node-level --prompt-set sec_prompts_v1.json
# Note: The orchestrator for 'node' mode will need to identify sections within the JSON filings.
```

### 7.3 Creating Master Summaries

(Master summaries aggregate multiple other summaries or a whole document.)
```bash
# Example: Creating master summaries for each filing in Bank_of_Montreal
summarizer.py --mode master --input-dir Data/Banks/Bank_of_Montreal --output-dir out/master-level --prompt-set sec_prompts_v1.json
```

### 7.4 Cross-Sectional Thematic View

(Cross-sectional summaries typically work on a collection of master summaries or other processed outputs.)
```bash
# Example: Cross-sectional on master summaries of Bank_of_Montreal filings
summarizer.py --mode cross --input-dir out/master-level/Banks/Bank_of_Montreal --output-dir out/cross-sectional/Banks/Bank_of_Montreal/liquidity-risk --prompt-set sec_prompts_v1.json --theme liquidity-risk
```

---

## 8. Testing & Validation

### 8.1 Sample Data Smoke Tests

Run each mode on `sample_sec/` to confirm correct outputs and cache behavior.

### 8.2 Unit Tests

* **Chunker:** splitting logic, metadata correctness.
* **Prompt Rendering:** placeholder substitution.
* **Cache Manager:** hit/miss behavior.
* **Metadata Generator:** schema compliance.

### 8.3 End-to-End Tests

Automate the full pipeline on a small set of filings; compare outputs with stored fixtures.

### 8.4 Quality Gates

* **Code Coverage:** ≥80% via `pytest --cov`.
* **CI/CD:** GitHub Actions workflow to lint, test, and smoke-test on pull requests.

---

## 9. Logging & Error Handling

### 9.1 Log Levels & Formats

* `INFO`, `DEBUG`, `ERROR`; JSON or plain text via `AGENT_LOG_PATH`.

### 9.2 Retry Policies

* Exponential backoff on LLM errors; configurable via `--retries`.

### 9.3 Error Codes

* `0`: Success
* `1`: Warnings (skipped files)
* `2`: Fatal errors

### 9.4 Diagnostics

* Use `--verbose` for raw prompt and response dumps.
* Inspect `.cache/` and metadata files for troubleshooting.

---

## 10. Performance & Scaling

### 10.1 Metrics

* **Latency:** \~1–2s per chunk; **Throughput:** \~30–50 chunks/minute

### 10.2 Parallelization

* Configurable concurrency; use thread pool with rate-limit guard.

### 10.3 Resource Usage

* Memory <200MB; CPU spikes during tokenization.

---

## 11. Future Enhancements (Roadmap)

* Interactive UI & live prompt editing
* Vector store integration for semantic retrieval
* Support for additional document types (policies, SOPs)
* Advanced cross-sectional analytics and visualization
* Role-based summary output templates

---

## 12. Appendix

### 12.1 Data Schemas & Examples

JSON schemas for filings and summary metadata.

### 12.2 Sample Prompt-Set JSON

```json
{ "file": "templates/sec_file_summary.md", "node": "templates/sec_node_summary.md", "master": "templates/sec_master_summary.md", "cross": "templates/sec_cross_sectional_summary.md" }
```

### 12.3 Glossary of Terms

* **Chunk, Node, Master, Cross-Sectional** definitions.

### 12.4 References

* OpenAI API docs, tiktoken guide, prompt engineering best practices.

---

*End of Technical Documentation v1* 