# RAGForge

**RAGForge** is a production-ready Document Intelligence REST API that lets you upload PDF, TXT, or DOCX files and ask natural-language questions against them using Retrieval-Augmented Generation (RAG). It is fully free to run locally or on Hugging Face Spaces, with an optional OpenAI upgrade for higher-quality answers.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://python.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://docs.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/YOUR_USERNAME/ragforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ragforge/actions)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client                                  │
│              (curl / Postman / your frontend)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI (main.py)                           │
│  POST /documents/upload │ GET /documents │ POST /query           │
│  GET /evaluate          │ GET /health                            │
└──────────┬──────────────────────────────────────┬───────────────┘
           │ ingest                               │ query
           ▼                                      ▼
┌──────────────────────┐              ┌───────────────────────────┐
│   ingestion.py       │              │      pipeline.py          │
│                      │              │                           │
│  1. Load PDF/TXT/    │              │  1. Embed query           │
│     DOCX bytes       │              │  2. similarity_search()   │
│  2. Extract text     │   embed      │  3. Build context string  │
│  3. Split into  ─────┼──────────►  │  4. Call LLM              │
│     500-token        │   & store   │  5. Return answer +       │
│     chunks           │              │     source chunks +       │
│  4. Add metadata     │              │     confidence + latency  │
└──────────────────────┘              └───────────┬───────────────┘
           │                                      │
           ▼                                      │
┌──────────────────────┐                          │
│   retrieval.py       │◄─────────────────────────┘
│                      │
│  ChromaDB            │   sentence-transformers
│  (local, on-disk)    │◄──all-MiniLM-L6-v2
│                      │   (384-dim embeddings)
│  Document registry   │
│  (JSON, on-disk)     │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   LLM Provider       │
│                      │
│  HuggingFace (free)  │   Mistral-7B-Instruct
│  ── OR ──            │
│  OpenAI (optional)   │   GPT-3.5-turbo / GPT-4
└──────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| REST API | FastAPI | Auto OpenAPI docs, async, Pydantic validation |
| RAG orchestration | LangChain 0.2 | LCEL chains, splitters, retriever abstractions |
| LLM (free default) | HuggingFace Inference API | Free tier, no credit card, Mistral-7B |
| LLM (optional) | OpenAI GPT-3.5/4 | Higher quality answers |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 80 MB, CPU-only, strong quality |
| Vector store | ChromaDB (local) | Zero infra, persists to disk, free |
| PDF parsing | PyPDF2 | Lightweight, page-level extraction |
| DOCX parsing | python-docx | Native OOXML parsing |
| Testing | pytest + pytest-cov | Standard, fixtures, coverage |
| Containerisation | Docker + Compose | Reproducible builds |
| CI | GitHub Actions | Free, parallel jobs, caching |
| Deployment | Hugging Face Spaces | Completely free public URL |

---

## Local Setup

### Prerequisites

- Docker Desktop ≥ 24 (or Python 3.11 + pip for bare-metal)
- A free [Hugging Face token](https://huggingface.co/settings/tokens) (Read scope)

### Quick start with Docker Compose (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ragforge.git
cd ragforge

# 2. Create your .env file
cp .env.example .env

# 3. Edit .env — at minimum set your HuggingFace token:
#    HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 4. Build and start
docker-compose up --build

# API is live at http://localhost:7860
# Swagger UI: http://localhost:7860/docs
```

### Bare-metal (without Docker)

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your HUGGINGFACE_API_KEY

# 4. Run
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

---

## API Documentation

Interactive docs: **http://localhost:7860/docs**

### `GET /health`

Returns application health and component status.

```bash
curl http://localhost:7860/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "components": {
    "vector_store": {"status": "ok", "total_chunks": 42},
    "llm": {"status": "ok", "provider": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.2"},
    "embedding": {"status": "ok", "model": "sentence-transformers/all-MiniLM-L6-v2"}
  },
  "uptime_seconds": 123.45
}
```

---

### `POST /documents/upload`

Upload a PDF, TXT, or DOCX file. Triggers full ingestion pipeline.

```bash
curl -X POST http://localhost:7860/documents/upload \
  -F "file=@docs/sample.pdf"
```

```json
{
  "document_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "filename": "sample.pdf",
  "document_type": "pdf",
  "chunk_count": 42,
  "ingested_at": "2024-01-15T10:30:00.000Z",
  "message": "Document ingested successfully."
}
```

**Save the `document_id`** — use it to scope queries to this document.

---

### `GET /documents`

List all ingested documents.

```bash
curl http://localhost:7860/documents
```

```json
{
  "documents": [
    {
      "document_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
      "filename": "sample.pdf",
      "document_type": "pdf",
      "chunk_count": 42,
      "ingested_at": "2024-01-15T10:30:00.000Z"
    }
  ],
  "total": 1
}
```

---

### `POST /query`

Ask a natural-language question. Returns the LLM answer + source chunks.

```bash
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is retrieval-augmented generation?",
    "top_k": 4
  }'
```

Scope to a single document:

```bash
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main components of RAG?",
    "top_k": 4,
    "document_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
  }'
```

```json
{
  "question": "What is retrieval-augmented generation?",
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language model generation. It retrieves relevant context from a document corpus and uses it to ground the LLM's answer, reducing hallucinations.",
  "source_documents": [
    {
      "content": "RAG systems retrieve relevant passages from a knowledge base...",
      "document_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
      "filename": "sample.pdf",
      "chunk_index": 3,
      "similarity_score": 0.8731,
      "page_number": 2
    }
  ],
  "confidence": "high",
  "retrieval_latency_ms": 45.2,
  "synthesis_latency_ms": 1823.7,
  "total_latency_ms": 1868.9
}
```

**Confidence levels:**
- `high` → mean similarity ≥ 0.75
- `medium` → mean similarity ≥ 0.50
- `low` → mean similarity < 0.50 (answer may be off-topic)

---

### `GET /evaluate`

Run the retrieval and answer-quality evaluation suite.

```bash
curl "http://localhost:7860/evaluate?max_cases=5"
```

```json
{
  "total_cases": 5,
  "retrieval_hit_rate": 1.0,
  "answer_relevance_rate": 0.8,
  "mean_latency_ms": 1950.3,
  "results": [
    {
      "question": "What is retrieval-augmented generation?",
      "retrieved_chunks": 4,
      "retrieval_hit": true,
      "answer_relevance": true,
      "answer_snippet": "RAG combines retrieval with generation...",
      "latency_ms": 1832.5
    }
  ],
  "evaluated_at": "2024-01-15T10:35:00.000Z"
}
```

> **Note:** Evaluation calls the LLM for every test case. Expect 10–30 seconds with 5 cases on the free HF tier.

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run unit tests only (fast, no network/LLM calls)
pytest tests/ -m "not integration" -v

# Run all tests including integration tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -m "not integration" --cov=app --cov-report=term-missing

# Run a single test file
pytest tests/test_ingestion.py -v
```

Expected output (unit tests only, ~5 seconds):

```
tests/test_ingestion.py::TestDetectDocumentType::test_pdf PASSED
tests/test_ingestion.py::TestLoadAndChunk::test_returns_list_of_documents PASSED
...
tests/test_evaluation.py::TestRunEvaluation::test_perfect_score_when_all_keywords_present PASSED

====== 42 passed in 4.81s ======
Coverage: 72%
```

---

## Deployment — Hugging Face Spaces (completely free)

Hugging Face Spaces provides a **permanent public URL** with Docker support at zero cost.

### Step-by-step

**1. Create a new Space**

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and configure:
- **Space name:** `ragforge` (or any name)
- **SDK:** `Docker`
- **Hardware:** `CPU Basic` (free)
- **Visibility:** Public or Private

**2. Add secrets**

In your Space → Settings → Repository secrets, add:

| Secret name | Value |
|---|---|
| `HUGGINGFACE_API_KEY` | Your HF token (`hf_xxx...`) |
| `LLM_PROVIDER` | `huggingface` |

**3. Push the code**

```bash
# Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/ragforge

# Push main branch
git push hf main
```

HF Spaces detects the `Dockerfile` automatically and builds + deploys it. The build takes 3–5 minutes on first push (downloading the embedding model).

**4. Access your API**

Your live URL will be:
```
https://YOUR_HF_USERNAME-ragforge.hf.space
```

Interactive docs:
```
https://YOUR_HF_USERNAME-ragforge.hf.space/docs
```

**5. Upload a document via the live API**

```bash
curl -X POST https://YOUR_HF_USERNAME-ragforge.hf.space/documents/upload \
  -F "file=@docs/sample.pdf"
```

### Notes on HF Spaces limits

| Resource | Free tier |
|---|---|
| CPU | 2 vCPUs |
| RAM | 16 GB |
| Disk | 50 GB (ephemeral — resets on rebuild) |
| Uptime | Sleeps after ~15 min inactivity; wakes on first request |

> **Persistence:** The free tier disk is ephemeral. For production use, persist ChromaDB to S3/GCS by pointing `CHROMA_PERSIST_DIR` at a mounted volume, or upgrade to a paid Space with persistent storage.

---

## Design Decisions

### Why ChromaDB?

ChromaDB runs **in-process** as a Python library — no external service, no Docker dependency for the vector store, no API keys. Data is persisted to a local directory (`./chroma_db`), survives container restarts via Docker volumes, and is compatible with HF Spaces disk storage. For scale-out, swap to Pinecone or pgvector by replacing the `get_vector_store()` factory in `retrieval.py`.

### Chunking strategy

We use `RecursiveCharacterTextSplitter` with:
- **`CHUNK_SIZE = 2000` characters** (≈ 500 tokens at 4 chars/token for English) — large enough to contain complete thoughts, small enough for precise retrieval
- **`CHUNK_OVERLAP = 200` characters** (≈ 50 tokens) — ensures sentences that fall at a boundary aren't lost; improves recall at chunk edges
- **Separator hierarchy:** `["\n\n", "\n", ". ", " ", ""]` — tries to break at paragraph boundaries first, then sentences, then words, avoiding mid-word cuts

### Why sentence-transformers/all-MiniLM-L6-v2?

- **CPU-only:** runs on any hardware with no GPU required
- **Fast:** encodes at ~14k tokens/sec on a single CPU core
- **Small:** 80 MB download, cached after first run
- **Quality:** strong performance on semantic search benchmarks (MTEB BEIR)
- **Free:** local inference, no API key, no rate limits

### Confidence heuristic

Mean cosine similarity of the top-k retrieved chunks is bucketed into `high/medium/low`. This is a **heuristic**, not a calibrated probability. It signals to clients whether the vector store found closely-matching content. Low confidence = the question may be out-of-scope for the ingested documents.

---

## Project Structure

```
ragforge/
├── app/
│   ├── main.py           # FastAPI app, all routes, lifespan hooks
│   ├── models.py         # Pydantic v2 request/response schemas
│   ├── config.py         # Centralised config via pydantic-settings
│   ├── ingestion.py      # PDF/TXT/DOCX loading + chunking
│   ├── retrieval.py      # ChromaDB vector store + document registry
│   ├── pipeline.py       # Full RAG chain (retrieve → synthesise)
│   └── evaluation.py     # Eval framework: hit rate + answer relevance
├── tests/
│   ├── conftest.py        # Shared fixtures (env isolation, singleton reset)
│   ├── test_ingestion.py  # Chunking and loading tests
│   ├── test_retrieval.py  # Vector search and registry tests
│   ├── test_pipeline.py   # RAG pipeline tests (mocked LLM)
│   └── test_evaluation.py # Evaluation framework tests
├── docs/
│   └── sample.pdf         # Sample document for testing
├── .github/
│   └── workflows/
│       └── ci.yml         # Lint + test + Docker build on push
├── docker-compose.yml     # Local development stack
├── Dockerfile             # Multi-stage build (builder + runtime)
├── requirements.txt       # Pinned Python dependencies
├── pytest.ini             # pytest configuration
├── .env.example           # Environment variable template
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
