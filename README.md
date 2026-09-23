<div align="center">

# 🏥 Clinical RAG Copilot

### A Production-Grade, Self-Verifying Biomedical RAG System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-FF6B35?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-MedCPT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/ncbi/MedCPT-Article-Encoder)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A fully containerized clinical literature search and question-answering system that retrieves evidence, generates atomic claims, and self-corrects hallucinations — all grounded in source-traceable biomedical text.**

</div>

---

## 📌 Problem Statement

General-purpose LLMs hallucinate in clinical settings at alarming rates. When queried over medical literature, a vanilla RAG pipeline produces claims with **no source traceability**, no verification step, and no mechanism to distinguish between supported evidence and model confabulation. In a clinical decision-support context, a single hallucinated drug dosage or contraindication can be catastrophic.

**Clinical RAG Copilot** solves this through a three-stage architecture:

1. **Specialized biomedical retrieval** using domain-fine-tuned dual-encoders (MedCPT) and a custom-fine-tuned cross-encoder reranker — not generic sentence transformers.
2. **Atomic claim extraction** where every factual statement is pinned to an exact source chunk and document ID with zero tolerance for floating assertions.
3. **Closed-loop agentic self-correction** via a LangGraph state machine that verifies each claim against its cited source, strips unsupported claims, and triggers up to 2 LLM rewrite cycles — all before returning a response.

The result: an **82.3% reduction in the hallucination rate** over a baseline RAG pipeline (82.95% → 8.7%), measured across 100 evaluated clinical questions.

---

## 🏛️ System Architecture & Data Flow

### Query-Time Pipeline (Runtime)

```
User HTTP Request
        │
        ▼
┌──────────────────┐
│   FastAPI Server  │  GET /ask?q=...  or  GET /search?q=...
│    src/api.py     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  Query Encoder (HF Inference) │
│  ncbi/MedCPT-Query-Encoder   │  → 768-dim biomedical query vector
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  pgvector Cosine Similarity Search   │
│  PostgreSQL chunks table             │  → Top-50 candidate chunks
│  (embedding <=> query_vec::vector)   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Cross-Encoder Reranker (HF Inference)│
│  parikshitup7/finetuned-medcpt-reranker│  → Scored & sorted, Top-5 kept
└──────────────────┬───────────────────┘
                   │
         ┌─────────┴─────────────────────────────┐
         │                                         │
    /search                                    /ask
    endpoint                                  endpoint
         │                                         │
         ▼                                         ▼
 Return ranked                     ┌───────────────────────────┐
 SearchResult[]                    │   LangGraph State Machine  │
 (with PubMed URLs)                │       src/graph.py         │
                                   └──────────┬────────────────┘
                                              │
                              ┌───────────────▼────────────────┐
                              │  Node 1: generate_node          │
                              │  LLM → structured ClinicalAnswer│
                              │  (claims[], summary, verified)  │
                              └───────────────┬────────────────┘
                                              │
                              ┌───────────────▼────────────────┐
                              │  Node 2: verify_node            │
                              │  For each Claim:                │
                              │  • Look up cited chunk by index │
                              │  • Call LLM verifier            │
                              │  • Label: SUPPORTED /           │
                              │    UNSUPPORTED / NOT_ENOUGH_INFO│
                              │  • Strip unsupported claims     │
                              └───────────────┬────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Conditional Router  │
                                    └────┬──────────┬─────┘
                                         │          │
                                   All pass?    Failures exist
                                   retries<2?    AND retries≥2?
                                         │          │
                                         ▼          ▼
                              ┌──────────────┐  ┌──────────────┐
                              │ regenerate   │  │    END       │
                              │ _node        │  │ (Return only │
                              │ LLM rewrite  │  │  verified    │
                              │ with feedback│  │  claims)     │
                              └──────┬───────┘  └──────────────┘
                                     │
                              loops back to verify_node
                              (max 2 retries enforced)
```

### Data Ingestion Pipeline (Offline / Setup)

```
scripts/bulk_ingest.py
         │
         ▼
┌─────────────────────────────┐
│  scripts/pubmed_client.py    │  PubMed NCBI E-Utils API (esearch + efetch)
│  fetch_pubmed_abstracts()    │  → 8 clinical domains × ~180 articles
│  → data/raw/{domain}/        │  Raw PubMed XML files saved to disk
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  scripts/bulk_ingest.py      │
│  parse_pubmed_xml()          │  XML → Pydantic Document objects
│  → Document(id, title,       │  PMID extracted, abstract sectioned
│     sections, text, source)  │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  scripts/chunker.py          │  Sliding word-count windows
│  chunk_document()            │  max_words=250 (~330 tokens)
│  → List[Chunk]               │  Section-aware splitting
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  src/repository.py                       │
│  save_document() + save_chunks()         │  UPSERT into PostgreSQL
│  → documents table + chunks table        │  ON CONFLICT DO NOTHING
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  scripts/batch_embed.py                      │
│  generate_embedding() per unembedded chunk   │  HF InferenceClient
│  → ncbi/MedCPT-Article-Encoder              │  768-dim vectors
│  update_chunk_embedding()                    │  UPDATE chunks SET
│  → chunks.embedding = vector(768)            │  embedding = %s
└─────────────────────────────────────────────┘
```

---

## ✨ Key Differentiators

### 1. Domain-Specialized Dual Encoders (MedCPT)
The system uses **two separate** NCBI-trained MedCPT encoders — `MedCPT-Query-Encoder` for user queries and `MedCPT-Article-Encoder` for document chunks. This asymmetric encoding, trained on PubMed query-click data, significantly outperforms symmetric general-purpose sentence transformers in biomedical retrieval. Embedding dimensions are 768, stored as native `pgvector` `vector(768)` columns.

### 2. Fine-Tuned Cross-Encoder Reranker
The reranker (`parikshitup7/finetuned-medcpt-reranker`) is a custom cross-encoder fine-tuned on 250 synthetic (query, positive-chunk, negative-chunk) training triplets generated from the ingested corpus via `scripts/generate_triplets.py`. It is served from HuggingFace Hub via `InferenceClient`, achieving an **MRR of 0.935** on the 100-question evaluation set — a 18.6% improvement over the baseline bi-encoder MRR of 0.787.

### 3. Atomic Claim Citation Architecture
Every claim in a `ClinicalAnswer` carries two mandatory flat fields: `source_id` (the document ID string, e.g., `PMID_12345678`) and `source_chunk_id` (the integer chunk index). The LLM is strictly instructed via `SYSTEM_PROMPT` to produce these as flat top-level properties — no nested objects. This makes every factual statement fully traceable to a specific paragraph of a specific paper.

### 4. LangGraph Closed-Loop Verification
The LangGraph state machine enforces a generate → verify → [regenerate] → verify loop. The verifier makes an independent LLM call for each claim, classifying it as `SUPPORTED`, `UNSUPPORTED`, or `NOT_ENOUGH_INFO`. Unsupported claims are collected into structured feedback, and the regenerator receives this feedback to rewrite the answer. Retry cap is set to 2 cycles. **Hallucination rate drops from 82.95% to 8.7%** after verification.

### 5. pgvector Native SQL Search
Rather than an external vector store (Pinecone, Weaviate, Chroma), all vectors live in PostgreSQL via the `pgvector` extension. This eliminates a separate infrastructure dependency, enables transactional consistency between chunk metadata and vectors, and allows cosine distance search in standard SQL (`embedding <=> query_vec::vector`) — trivially manageable via Docker Compose.

---

## 📁 Repository & Directory Structure

```
clinical-rag-copilot/
│
├── 📦 Runtime Core (served by Docker)
│   └── src/
│       ├── __init__.py                 # Python package marker
│       ├── api.py                      # FastAPI app — /search and /ask endpoints
│       ├── graph.py                    # LangGraph state machine (generate→verify→regenerate)
│       ├── schema.py                   # Pydantic models: Document, Chunk, Claim, ClinicalAnswer
│       ├── embedding_service.py        # HF InferenceClient: MedCPT embeddings + reranker
│       ├── repository.py               # pgvector SQL: save/search/embed chunks
│       ├── database.py                 # psycopg2 connection factory (reads DATABASE_URL)
│       ├── generator.py                # Instructor+OpenAI/Groq: structured answer generation
│       ├── verifier.py                 # Instructor+OpenAI/Groq: per-claim entailment check
│       └── eval/
│           ├── __init__.py
│           ├── loader.py               # Loads tests/fixtures/eval_questions.json
│           ├── retrieval_metrics.py    # Offline: Precision@5, Recall@5, MRR evaluation
│           ├── verifier_metrics.py     # Offline: Verifier precision/recall/F1 evaluation
│           └── e2e_metrics.py          # Offline: End-to-end hallucination rate comparison
│
├── 🔧 Pipeline Scripts (offline, not imported by api.py)
│   └── scripts/
│       ├── __init__.py
│       ├── ingest_pubmed.py            # Single-domain PubMed ingestion pipeline
│       ├── bulk_ingest.py              # 8-domain mass ingestion (Cardiology, Oncology, etc.)
│       ├── pubmed_client.py            # NCBI E-Utils API client (esearch + efetch)
│       ├── parsers.py                  # PubMed XML → Document Pydantic object
│       ├── chunker.py                  # Document → List[Chunk] with word-count windows
│       ├── batch_embed.py              # Embeds unembedded chunks into pgvector
│       ├── generate_triplets.py        # LLM-synthesized (query, positive, negative) triplets
│       └── auto_eval_generator.py      # LLM-synthesized clinical evaluation questions
│
├── 🗄️ Database Migrations
│   ├── alembic.ini                     # Alembic config (sqlalchemy.url)
│   └── alembic/
│       ├── env.py                      # Migration runner (online + offline modes)
│       ├── script.py.mako              # Migration file template
│       └── versions/
│           └── c239fe919c6f_create_documents_and_chunks_tables.py
│               # Revision: creates documents table, chunks table, vector(768) column,
│               #           enables pgvector extension
│
├── 🧪 Test Suite
│   ├── __init__.py
│   ├── test_api.py                     # Unit tests for /search endpoint (mocked)
│   ├── test_api_integration.py         # Integration test for /ask endpoint + graph
│   ├── test_api_verification.py        # Tests claim verification flow
│   ├── test_chunker.py                 # Unit tests for chunk_document()
│   ├── test_embedding_service.py       # Unit tests for generate_query_embedding()
│   ├── test_generator.py               # Unit tests for generate_clinical_answer()
│   ├── test_graph.py                   # LangGraph tests: retry logic, cap enforcement
│   ├── test_main.py                    # Tests ingest pipeline entry point
│   ├── test_parsers.py                 # Unit tests for parse_pubmed_xml_record()
│   ├── test_pubmed_client.py           # Tests NCBI API client (mocked HTTP)
│   ├── test_pubmed_parser.py           # Tests valid/malformed PubMed XML handling
│   ├── test_repository.py              # Tests DB save/search operations
│   ├── test_reranker.py                # Tests cross-encoder ordering
│   ├── test_schema.py                  # Tests Pydantic model validation
│   ├── test_vector_search.py           # Tests pgvector cosine search
│   ├── test_verifier.py                # Tests verify_claim() label output
│   ├── eval/
│   │   ├── test_e2e_metrics.py
│   │   ├── test_loader.py
│   │   ├── test_retrieval_metrics.py
│   │   └── test_verifier_metrics.py
│   └── fixtures/
│       └── eval_questions.json         # 100 LLM-generated clinical Q&A pairs with gold PMIDs
│
├── 📊 Evaluation Reports (gitignored — generated by src/eval/)
│   ├── retrieval_metrics.json
│   ├── verifier_metrics.json
│   ├── e2e_metrics.json
│   └── checkpoint_e2e.json             # Resumable checkpoint (question-by-question)
│
├── 🐳 Container & Config
│   ├── Dockerfile                      # python:3.12-slim, installs requirements, runs uvicorn
│   ├── docker-compose.yml              # Services: pgvector/pgvector:pg16 + api
│   ├── .dockerignore                   # Excludes venv/, models/, .git/
│   ├── .env.example                    # Template for required environment variables
│   └── requirements.txt               # All Python dependencies
│
└── config/
    └── __init__.py                     # Config package namespace (reserved)
```

---

## 🛠️ Technology Stack & Architectural Choices

| Layer | Technology | Why This Choice |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async-native, automatic OpenAPI docs, Pydantic-native request/response validation |
| **Agentic Orchestration** | LangGraph (StateGraph) | Enables explicit typed state, conditional routing, and retry loops — unlike linear LangChain chains, the graph structure is inspectable and testable |
| **LLM Structured Output** | Instructor + OpenAI SDK (Groq endpoint) | `instructor` patches the OpenAI client to force JSON-mode structured outputs conforming to Pydantic models, eliminating manual JSON parsing |
| **LLM Provider** | Groq (`openai/gpt-oss-120b` via OpenAI-compat. API) | Ultra-low latency inference; OpenAI-compatible endpoint means zero vendor lock-in |
| **Query Embedding** | `ncbi/MedCPT-Query-Encoder` (HF Inference) | Pre-trained on 255M PubMed query-click pairs; dramatically outperforms `all-MiniLM` on biomedical queries |
| **Article Embedding** | `ncbi/MedCPT-Article-Encoder` (HF Inference) | Asymmetric partner to the query encoder; 768-dim |
| **Reranking** | `parikshitup7/finetuned-medcpt-reranker` | Custom fine-tuned on domain-specific triplets; dramatically improves MRR over bi-encoder baseline |
| **Vector Store** | PostgreSQL + `pgvector` extension | Native SQL joins between chunk metadata and vectors; transactional guarantees; no separate vector DB process |
| **DB Migration** | Alembic | Schema version control — single migration creates `documents`, `chunks` tables and enables `vector` extension |
| **Containerization** | Docker Compose | Reproducible local environment; `pgvector/pgvector:pg16` official image; health-checked DB startup |
| **DB Driver** | psycopg2 | Native PostgreSQL adapter; direct `vector` column support without ORM overhead |
| **Data Validation** | Pydantic v2 | All domain objects (`Document`, `Chunk`, `Claim`, `ClinicalAnswer`) are validated models |

---

## 🚀 Environment Setup & Quickstart

### Prerequisites

- **Docker Desktop** ≥ 4.x with Docker Compose v2 (`docker compose` subcommand)
- **Python 3.12** (for running scripts and tests outside Docker)
- API keys:
  - `HF_TOKEN` — [HuggingFace token](https://huggingface.co/settings/tokens) with read access (for MedCPT inference)
  - `GROQ_API_KEY` — [Groq API key](https://console.groq.com/) (for LLM generation and verification)
  - `OPENAI_BASE_URL` — Set to `https://api.groq.com/openai/v1` for Groq compatibility

### Step 1: Clone & Configure Environment

```bash
git clone https://github.com/parikshitg7/clinical-rag-copilot.git
cd clinical-rag-copilot

# Copy the template and fill in your keys
cp .env.example .env
```

Edit `.env`:

```env
# LLM (Groq's OpenAI-compatible endpoint)
OPENAI_API_KEY=gsk_your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1

# HuggingFace Inference API (for MedCPT embeddings + reranker)
HF_TOKEN=hf_your_token_here

# Groq SDK (used by auto_eval_generator.py)
GROQ_API_KEY=gsk_your_groq_key_here
```

### Step 2: Start the Docker Stack

```bash
docker compose up --build
```

This will:
1. Pull `pgvector/pgvector:pg16` and start a PostgreSQL instance with the `vector` extension available
2. Build the API image from `Dockerfile` (`python:3.12-slim`)
3. Install all `requirements.txt` dependencies inside the container
4. Start the FastAPI server via `uvicorn src.api:app --host 0.0.0.0 --port 8000`
5. The DB health check (`pg_isready`) ensures the API waits until PostgreSQL is fully ready

> The API will be live at **http://localhost:8000**  
> Swagger UI will be at **http://localhost:8000/docs**

### Step 3: Run Database Migrations

In a new terminal, run Alembic migrations to create the schema:

```bash
docker compose exec api alembic upgrade head
```

This executes migration `c239fe919c6f` which:
- Enables the `pgvector` extension (`CREATE EXTENSION IF NOT EXISTS vector`)
- Creates the `documents` table (`id TEXT PRIMARY KEY`, `source`, `title`, `sections JSONB`, `metadata JSONB`)
- Creates the `chunks` table with a `vector(768)` embedding column and a cascading foreign key to `documents`

---

## 📥 Data Ingestion & Vectorization Pipeline

After the database schema is initialized, populate it with clinical literature. All pipeline scripts run **outside** Docker using your local Python environment against the Docker-exposed PostgreSQL port (`5432`).

```bash
# Set up local Python environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1: Bulk Literature Ingestion (8 Clinical Domains)

```bash
python -m scripts.bulk_ingest
```

This script:
- Iterates over 8 pre-defined MeSH-term queries covering: **Cardiology, Endocrinology, Oncology, Infectious Disease, Neurology, Pharmacology, Pediatrics, General Medicine**
- Calls the NCBI E-Utils `esearch` API to retrieve up to 180 PMIDs per domain
- Calls `efetch` to download full PubMed XML for all retrieved articles
- Saves raw XML to `data/raw/{domain}/pubmed_raw.xml`
- Parses each article: extracts PMID, title, abstract sections (structured or unstructured)
- Upserts `Document` records into the `documents` table
- Splits each document into `Chunk` objects (max 250 words, section-aware) and upserts into `chunks`
- Respects NCBI's rate limit with a 4-second sleep between domains

### Step 2: Generate Vector Embeddings

```bash
python -m scripts.batch_embed
```

This script:
- Queries `SELECT id, text FROM chunks WHERE embedding IS NULL`
- For each unembedded chunk, calls `HF InferenceClient.feature_extraction(text, model="ncbi/MedCPT-Article-Encoder")` to obtain a 768-dim vector
- Automatically handles nested list responses and truncates text to 1,800 characters if needed
- On failure, retries with aggressive truncation to 1,000 characters before skipping
- Updates each chunk with `UPDATE chunks SET embedding = %s WHERE id = %s`

### Step 3: Generate Synthetic Evaluation Questions (Optional)

```bash
python -m scripts.auto_eval_generator
```

- Pulls 100 random chunks (length > 150 chars) from the database
- Calls Groq's `llama-3.3-70b-versatile` via `instructor.from_groq()` to generate one targeted clinical question per chunk
- Saves the dataset to `tests/fixtures/eval_questions.json` in the format:
  ```json
  [{"id": "eval_q_1", "question": "...", "gold_pmids": ["PMID_12345678"]}]
  ```

### Step 4: Generate Fine-Tuning Triplets (Optional — for reranker training)

```bash
python -m scripts.generate_triplets
```

- Pulls 250 random chunks as positive examples and a pool of 1,000 random chunks as hard negatives
- Generates a synthetic clinical query for each positive chunk via LLM
- Saves `(query, positive, negative)` triplets to `data/training_triplets.json` for fine-tuning the cross-encoder

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### `GET /`
Health check endpoint.

**Response:**
```json
{"status": "online", "message": "API is running. Go to /docs for Swagger UI."}
```

---

### `GET /search` — Semantic Search

Returns the top-N ranked chunks most relevant to the query, with direct PubMed links.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | `string` | **required** | Medical query string |
| `top_k` | `integer` | `50` | Number of initial candidates retrieved from pgvector |
| `top_n` | `integer` | `8` | Final number of chunks after cross-encoder reranking |

**Example Request:**
```bash
curl "http://localhost:8000/search?q=metformin+HbA1c+type+2+diabetes&top_k=50&top_n=5"
```

**Example Response:**
```json
[
  {
    "id": 4821,
    "parent_doc_id": "PMID_36485634",
    "section": "RESULTS",
    "text": "Metformin reduced HbA1c by 1.2% compared to placebo in patients with newly diagnosed type 2 diabetes over 24 weeks of treatment (p < 0.001).",
    "score": 0.9312,
    "url": "https://pubmed.ncbi.nlm.nih.gov/36485634/"
  },
  {
    "id": 2134,
    "parent_doc_id": "PMID_29753080",
    "section": "CONCLUSIONS",
    "text": "First-line Metformin therapy significantly lowers fasting plasma glucose and is associated with modest weight reduction in T2DM patients.",
    "score": 0.8874,
    "url": "https://pubmed.ncbi.nlm.nih.gov/29753080/"
  }
]
```

---

### `GET /ask` — Clinical Question Answering

Retrieves evidence, runs the full LangGraph verification pipeline, and returns a structured, verified answer with atomic claims.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | `string` | **required** | Clinical question to answer |
| `top_k` | `integer` | `50` | Initial candidates from pgvector |
| `top_n` | `integer` | `5` | Chunks sent to the LLM after reranking |

**Example Request:**
```bash
curl "http://localhost:8000/ask?q=What+is+the+first-line+treatment+for+type+2+diabetes?"
```

**Example Response:**
```json
{
  "question": "What is the first-line treatment for type 2 diabetes?",
  "claims": [
    {
      "text": "Metformin is recommended as the first-line pharmacological therapy for type 2 diabetes due to its efficacy, safety profile, and low cost.",
      "source_id": "PMID_36485634",
      "source_chunk_id": 2
    },
    {
      "text": "Lifestyle modification including diet and exercise is recommended alongside pharmacotherapy as foundational treatment.",
      "source_id": "PMID_29753080",
      "source_chunk_id": 7
    }
  ],
  "summary": "The first-line treatment for type 2 diabetes combines lifestyle intervention with Metformin pharmacotherapy. Clinical guidelines consistently support Metformin as the initial drug of choice given its robust HbA1c-lowering efficacy and well-established safety record.",
  "verified": true
}
```

**Response Schema (`ClinicalAnswer`):**

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The original query echoed back |
| `claims` | `Claim[]` | List of atomic factual statements, each pinned to a source |
| `claims[].text` | `string` | The exact factual sentence |
| `claims[].source_id` | `string` | Document identifier (e.g., `PMID_12345678`) |
| `claims[].source_chunk_id` | `integer` | Chunk index within the document |
| `summary` | `string` | Synthesized paragraph summarizing all verified claims |
| `verified` | `boolean` | `true` if all claims passed the LangGraph verification loop |

---

## 📊 Evaluation Harness & Quantitative Performance

All evaluation scripts live in `src/eval/` and write results to `reports/`. Run them against a live database populated with the ingested corpus.

### Retrieval Quality — 100-Question Evaluation Set

Evaluated over `tests/fixtures/eval_questions.json` (100 synthetic clinical questions with gold PMIDs). Retrieval pool: Top-15, evaluation at Top-5.

```bash
python -m src.eval.retrieval_metrics
```

| Metric | Baseline (Bi-Encoder Only) | + Cross-Encoder Reranking | Δ Improvement |
|---|---|---|---|
| **Precision@5** | 0.242 | **0.266** | +9.9% |
| **Recall@5** | 0.890 | **0.940** | +5.6% |
| **MRR** | 0.7878 | **0.9350** | **+18.6%** |

> The MRR jump from 0.787 → 0.935 means the correct document is being ranked **first** far more often after reranking, which directly reduces hallucination risk in downstream generation.

---

### Claim Verifier Quality — Hardcoded Clinical Dataset (5 Claims)

```bash
python -m src.eval.verifier_metrics
```

| Metric | Score |
|---|---|
| **Precision** | **1.000** |
| **Recall** | **1.000** |
| **F1 Score** | **1.000** |
| True Positives (caught hallucinations) | 3 |
| False Positives (flagged good claims) | 0 |
| True Negatives (passed good claims) | 2 |
| False Negatives (missed hallucinations) | 0 |

The verifier correctly classified all 5 test cases (SUPPORTED, UNSUPPORTED, NOT_ENOUGH_INFO) with zero false positives or false negatives.

---

### End-to-End Hallucination Rate — 100 Questions, Full Pipeline

```bash
python -m src.eval.e2e_metrics
```

| Condition | Total Claims Made | Unsupported Claims | Hallucination Rate |
|---|---|---|---|
| **Baseline RAG** (generation only, no verification) | 129 | 107 | **82.95%** |
| **Verified LangGraph RAG** (with self-correction loop) | 23 | 2 | **8.70%** |
| **Absolute Reduction** | — | — | **−74.25 percentage points** |

> The LangGraph verification loop achieves a **10.5× reduction** in hallucination rate. The reduction in total claims (129 → 23) reflects that the system only surfaces claims it can actually ground in retrieved evidence — trading quantity for verifiability.

The evaluation framework supports **checkpointing** — if interrupted, it resumes from the last evaluated question using `reports/checkpoint_e2e.json`.

---

## 🧪 Testing & Quality Assurance

The test suite covers unit, integration, and evaluation layers. Run tests against a live Docker stack for integration tests, or in isolation with mocks for unit tests.

### Install Test Dependencies

```bash
pip install pytest pytest-mock
```

### Run Full Test Suite

```bash
# From repository root (with DATABASE_URL pointing to local Docker db)
pytest tests/ -v
```

### Run Unit Tests Only (no DB/API required)

```bash
pytest tests/test_chunker.py tests/test_schema.py tests/test_parsers.py tests/test_pubmed_parser.py -v
```

### Run API Tests (uses FastAPI TestClient with mocks — no live server needed)

```bash
pytest tests/test_api.py tests/test_api_integration.py tests/test_api_verification.py -v
```

### Run LangGraph State Machine Tests

```bash
pytest tests/test_graph.py -v
```

These tests verify:
- **Retry logic**: graph retries exactly once on a failed verification, then succeeds
- **Retry cap**: graph terminates after exactly 2 retries when all claims consistently fail
- **Claim stripping**: unsupported claims are removed from the final response before returning to the user

### Test Coverage Overview

| Test File | What It Tests |
|---|---|
| `test_api.py` | `/search` endpoint: mock embeddings, mock reranker, response shape |
| `test_api_integration.py` | `/ask` endpoint: mock entire pipeline + graph, verifies `verified=True` |
| `test_api_verification.py` | Claim verification flow through the API layer |
| `test_graph.py` | LangGraph retry logic, cap enforcement, claim filtering |
| `test_chunker.py` | Word-count splitting, section-awareness, edge cases |
| `test_parsers.py` | PubMed XML → Document, field extraction correctness |
| `test_pubmed_parser.py` | Valid XML, malformed XML graceful fallback |
| `test_repository.py` | DB upsert, vector search SQL |
| `test_embedding_service.py` | 768-dim output, list flattening |
| `test_reranker.py` | Cross-encoder correctly ranks relevant chunk highest |
| `test_verifier.py` | `verify_claim()` returns correct Label enum |
| `test_schema.py` | Pydantic model validation rules |
| `eval/test_e2e_metrics.py` | E2E evaluation pipeline integrity |
| `eval/test_retrieval_metrics.py` | Precision/Recall/MRR calculation correctness |
| `eval/test_verifier_metrics.py` | Confusion matrix computation |

---

## ☁️ Deployment Architecture

### Local (Default) — Docker Compose

The default setup runs entirely locally via Docker Compose:
- `clinical_rag_db` container: `pgvector/pgvector:pg16` on port `5432`
- `clinical_rag_api` container: FastAPI + Uvicorn on port `8000`
- `pgdata` Docker volume: persistent PostgreSQL data storage

```bash
# Start full stack
docker compose up -d

# View logs
docker compose logs -f api

# Stop and remove containers (data persists in volume)
docker compose down

# Full teardown including volume
docker compose down -v
```

### Cloud Deployment (Option B)

For production or portfolio deployment:

1. **Fine-Tuned Model Weights** — Push `models/finetuned_medcpt_reranker/` to HuggingFace Hub. The `embedding_service.py` already references the Hub model ID `parikshitup7/finetuned-medcpt-reranker` — no code changes needed.

2. **Managed PostgreSQL** — Replace `DATABASE_URL` with a connection string from Neon, Supabase, or Railway. Ensure the `pgvector` extension is enabled on the managed instance.

3. **API Hosting** — Deploy the Docker image to Railway, Render, or any container platform. Set environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `HF_TOKEN`, `DATABASE_URL`) in the platform's config.

```bash
# Build production image
docker build -t clinical-rag-copilot:latest .

# Push to registry (example: Docker Hub)
docker tag clinical-rag-copilot:latest yourusername/clinical-rag-copilot:latest
docker push yourusername/clinical-rag-copilot:latest
```

---

## ⚠️ Limitations & Future Scope

### Current Limitations

| Limitation | Detail |
|---|---|
| **Single-turn only** | The `/ask` endpoint is stateless — no conversation history or follow-up question capability |
| **PubMed-only corpus** | Ingestion pipeline targets PubMed abstracts. Full-text articles, guidelines, and clinical trial registries are not indexed |
| **HF Inference API latency** | Embedding and reranking calls go to HuggingFace Inference API — each request adds network round-trip latency. Local model serving would be faster |
| **Synchronous endpoint** | The `/ask` endpoint is blocking. For production traffic, streaming or a task queue (Celery/RQ) would be needed |
| **Chunk-level citation** | Claims cite chunks, not sentences. Sub-chunk sentence-level provenance would increase precision |
| **Fixed retry cap** | The 2-retry cap is hardcoded in `graph.py`. Configurable retry strategies would improve flexibility |

### Future Scope

- [ ] **Streaming responses** — SSE-based streaming for real-time token output from the generation node
- [ ] **Full-text article indexing** — PubMed Central (PMC) Open Access corpus integration
- [ ] **Sentence-level citation** — Sub-chunk extraction to cite exact sentences, not just paragraph windows
- [ ] **Multi-turn conversation** — Session-based context window management for follow-up questions
- [ ] **Local model serving** — `llama.cpp` or `vLLM` sidecar for offline-capable, low-latency inference
- [ ] **Re-ranking feedback loop** — Log user click-through data to continuously improve the reranker
- [ ] **Clinical NER enrichment** — Extract entities (drugs, conditions, dosages) to enable structured filtering alongside semantic search
- [ ] **Frontend UI** — React/Next.js chat interface with source highlighting

---

## 👤 Author

**Parikshit Gujrathi**

[![GitHub](https://img.shields.io/badge/GitHub-parikshitg7-181717?logo=github)](https://github.com/parikshitg7)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-parikshitup7-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/parikshitup7)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with the conviction that clinical AI must be verifiable, traceable, and honest.**

</div>