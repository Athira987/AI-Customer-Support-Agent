# 🤖 NovaTech AI Customer Support Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.com/)
[![Qwen](https://img.shields.io/badge/Qwen-2.5--7B-purple.svg)](https://ollama.com/library/qwen2.5)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, full-stack AI Customer Support Agent powered by **Retrieval-Augmented Generation (RAG)**, **FastAPI**, **ChromaDB**, **Sentence Transformers**, and **Qwen 2.5 7B** running locally via **Ollama**.

The system retrieves relevant knowledge-base documents from a fictional NovaTech Electronics catalog, generates strictly grounded responses with source citations, and intelligently escalates unsupported or low-confidence queries to human support specialists. Built entirely without heavy black-box frameworks (no LangChain or LlamaIndex) to ensure complete architectural control over vector retrieval, similarity thresholds, prompt grounding, and fallback escalation.

---
## 📸 Project Screenshots

<p align="center">
  <img src="screenshots/01-home.png" width="32%" alt="NovaTech Support AI home screen">
  <img src="screenshots/02-rag-response.png" width="32%" alt="RAG-based support response">
  <img src="screenshots/03-human-escalation.png" width="32%" alt="Human escalation workflow">
</p>

---
## ✨ Key Features

- **⚡ FastAPI Backend**: High-performance asynchronous REST API with Pydantic v2 schema validation, structured error handling, and interactive OpenAPI documentation.
- **🔍 Local RAG Pipeline by Default**: Dense vector retrieval using ChromaDB and Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) — runs locally without external embedding API calls.
- **🤖 Local LLM Inference**: Private on-device generation with Qwen 2.5 7B (configurable via environment variables) executed via Ollama's OpenAI-compatible `/v1` endpoint.
- **🎯 Strictly Grounded Answers**: Prompt design enforcing answers derived exclusively from verified company documents, mitigating hallucinations with document source citations (filename, category, page, snippet).
- **🗂️ Category-Aware Retrieval**: Topic selection restricts ChromaDB vector search to the corresponding knowledge-base file via a `where` metadata filter without duplicate collections.
- **🛡️ Multi-Tier Human Escalation**: Proactively escalates queries involving explicit human requests, out-of-domain topics (cosine distance exceeding threshold), or missing/conflicting knowledge-base information.
- **🔄 LLM Query Rewriting**: Automatically rewrites user queries into high-recall standalone search terms and resolves follow-up references across multi-turn dialogues.
- **⚡ Conversational Shortcuts**: Fast regex routing answers common greetings and closings instantly without triggering vector search or LLM generation.
- **🔒 Prompt Injection Defense**: System prompt isolates retrieved context as read-only reference material to resist adversarial injection attempts.
- **💻 Integrated Chat Interface**: Modern, responsive single-page chat UI built with vanilla HTML, CSS, and JavaScript, featuring topic switching, live status indicators, and keyboard shortcuts.
- **🧪 Automated Test Suite**: Comprehensive automated test coverage across 7 test modules covering API validation, category filtering, escalation rules, LLM parsing fallbacks, and vector search operations.

---

## 🏗️ Tech Stack

| Layer | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python 3.10+) | Asynchronous REST API routing, dependency injection, and schema enforcement |
| **ASGI Server** | Uvicorn | Lightweight, production-grade ASGI web server |
| **Local LLM Runtime** | Ollama | Local execution runtime providing an OpenAI-compatible `/v1` interface |
| **Language Model** | Qwen 2.5 7B | On-device model (configurable via `OLLAMA_MODEL`) for query rewriting and response synthesis |
| **Client Library** | OpenAI Python SDK (v1.20+) | Client library used to connect to Ollama's local OpenAI-compatible endpoint |
| **Embedding Model** | Sentence Transformers | `all-MiniLM-L6-v2` generating 384-dimensional dense vector embeddings |
| **Vector Database** | ChromaDB | Persistent local vector store with cosine similarity index and metadata filtering |
| **Data Validation** | Pydantic v2 | Strongly-typed request/response validation and environment settings management |
| **Document Processing** | PyPDF & Python Standard Library | Text extraction (.txt, .pdf) and section-aware semantic chunking |
| **Frontend UI** | HTML5, CSS3, JavaScript | Vanilla single-page chat application with category sidebar and live status |
| **Testing** | Pytest & HTTPX | Automated test suite validating API, retrieval, escalation, and RAG pipelines |

---

## 📁 Project Structure

```text
AI-Customer-Support-Agent/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API routes (/health, /chat, /knowledge-base/status)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Pydantic BaseSettings application configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic request, response, and structured output models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Local SentenceTransformer embedding generator
│   │   ├── escalation.py        # Pre-retrieval regex and distance-based escalation engine
│   │   ├── ingestion.py         # Document parsing, normalization, and chunking service
│   │   ├── llm.py               # Ollama LLM client with structured JSON parsing & fallback
│   │   ├── rag.py               # Complete end-to-end RAG coordinator pipeline
│   │   └── retrieval.py         # ChromaDB persistence, upserting, and vector query logic
│   ├── __init__.py
│   └── main.py                  # FastAPI application entrypoint, middleware, and static mount
├── frontend/
│   ├── app.js                   # Client-side chat logic, API communication, and state
│   ├── index.html               # Single-page chat interface with topic sidebar
│   └── style.css                # Interface styling and responsive layout
├── knowledge_base/
│   ├── faq/
│   │   ├── payment_faq.txt      # Accepted payment methods, billing, and installments
│   │   └── troubleshooting_guide.txt # Device reset procedures, Bluetooth pairing, audio fixes
│   ├── policies/
│   │   ├── return_and_refund.txt # 30-day return policy, restocking fee, and return process
│   │   ├── shipping_policy.txt  # Domestic, international, and expedited delivery timelines
│   │   └── warranty_policy.txt  # 1-year and 2-year warranty terms, battery coverage, exclusions
│   └── products/
│       └── nova_products.txt    # Product specifications, storage options, ports, and pricing
├── scripts/
│   └── ingest_documents.py      # CLI script to chunk, embed, and index knowledge-base files
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # Health check, chat endpoint validation, and category routing
│   ├── test_category_retrieval.py # Category mapping, metadata isolation, and fallback tests
│   ├── test_escalation.py       # Human trigger regex, distance threshold, and escalation formatting
│   ├── test_llm_service.py      # JSON output parsing, markdown fence stripping, and fallback handling
│   ├── test_rag_evaluation.py   # RAG benchmark dataset verification and response grounding
│   ├── test_rag_pipeline.py     # Standalone query rewriting, multi-turn context, and pipeline tests
│   └── test_retrieval.py        # Chunking logic, ChromaDB upsert/query, and 384-d embeddings
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns (ignores .env, .venv, data/chroma/)
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
└── requirements.txt             # Project Python dependencies
```

> **Note:** The local virtual environment (`.venv/`), environment configuration (`.env`), and generated vector database (`data/chroma/`) are intentionally excluded from the Git repository via `.gitignore`.

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** installed:
  ```bash
  python --version
  ```
- **Git** installed:
  ```bash
  git --version
  ```
- **Ollama** installed ([Download Ollama](https://ollama.com/download))

### 2. Clone the Repository

```bash
git clone https://github.com/Athira987/AI-Customer-Support-Agent.git
cd AI-Customer-Support-Agent
```

### 3. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create your local `.env` configuration from the provided template:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

The default values in `.env.example` work out of the box for local deployment:

```env
# Ollama LLM Configuration (Local Response Generation - No API Key required)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b

# Local SentenceTransformers Embedding Model (No API Key required)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=novatech_knowledge_base

# RAG & Retrieval Parameters
TOP_K=4
SIMILARITY_DISTANCE_THRESHOLD=0.65

# Ingestion & Chunking Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Server Configuration
API_HOST=127.0.0.1
API_PORT=8000
```

### 6. Run / Start Required Services

#### Start Ollama and Pull Model
In a separate terminal, start the Ollama service and pull the default Qwen 2.5 7B model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

#### Ingest the Knowledge Base
Populate the local ChromaDB vector database from the knowledge-base documents:

```bash
python scripts/ingest_documents.py
```

To wipe and re-index the collection from scratch:
```bash
python scripts/ingest_documents.py --reset
```

> **Knowledge Base Indexing Details:**
> - The knowledge base consists of **6 source documents** across 3 categories (`faq`, `policies`, `products`).
> - The number of indexed chunks depends on the knowledge-base documents and chunking configuration. For example, the provided knowledge base yields **117 chunks** when ingested with the `.env.example` settings (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`), or **67 chunks** with the `config.py` defaults (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`).
> - The ChromaDB vector database is generated locally inside `data/chroma/` and does not need to be committed to version control.

### 7. Run the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 8. Open in Browser

Open your browser and navigate to:
- **Chat Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 💡 How It Works

The end-to-end RAG workflow executes in the following sequence:

1. **Request Validation**: The incoming `POST /chat` request is validated by Pydantic against message length constraints (1–2000 characters) and optional fields (`category`, `conversation_history`).
2. **Conversational Shortcut Check**: Fast regex pattern matching intercepts greetings ("hello", "hi") and closings ("thanks", "bye"), returning immediate conversational responses without database or LLM overhead.
3. **Pre-Retrieval Escalation**: Regex patterns detect explicit customer requests to contact a human agent (e.g., "speak to a representative"), returning an immediate escalation notice and bypassing vector retrieval.
4. **LLM Query Rewriting**: If conversation history is present, the local LLM synthesizes a concise, standalone semantic query resolving ambiguous pronouns or references to earlier dialogue turns.
5. **Dense Vector Embedding**: The search query is embedded into a 384-dimensional vector using the local `sentence-transformers/all-MiniLM-L6-v2` model.
6. **Category Resolution & Vector Query**: If a support category was selected in the UI (e.g. `"warranty"`), it is resolved to its source filename (`"warranty_policy.txt"`) and queried in ChromaDB using a `where={"filename": ...}` filter.
7. **Retrieval Fallback Diagnostic**: If the rewritten query's best cosine distance exceeds the internal fallback threshold of `0.40`, the system re-queries with the original user message and selects whichever result set achieved a lower cosine distance.
8. **Relevance Threshold Gate**: If the best retrieved chunk has a cosine distance greater than the configured similarity threshold (default: `0.65`, configurable via `SIMILARITY_DISTANCE_THRESHOLD`), the system concludes the topic is outside the knowledge base and escalates to a human agent before invoking the LLM.
9. **Grounded Response Generation**: The retrieved context chunks and conversation history are formatted into a structured prompt sent to Qwen 2.5 7B. The model generates a structured JSON response containing `answer`, `needs_escalation`, `reason`, and `used_sources`.
10. **Source Citation Matching**: Filenames cited by the LLM are validated against retrieved chunks and returned as structured source citations (`document`, `category`, `page`, `snippet`).

---

## 🔌 API Endpoints

### Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check and version metadata |
| `POST` | `/chat` | Core customer support RAG interaction endpoint |
| `GET` | `/knowledge-base/status` | Vector store status, chunk count, and model configuration |

---

## 🔮 Future Improvements

- [ ] **Helpdesk Webhook Integration**: Forward escalated inquiries directly to Zendesk, Freshdesk, or Slack channels via webhooks.
- [ ] **Hybrid Search**: Combine BM25 sparse keyword retrieval with ChromaDB dense vector search using Reciprocal Rank Fusion (RRF).
- [ ] **User Feedback Rating**: Add thumbs-up / thumbs-down buttons to log retrieval quality metrics and tune similarity thresholds.
- [ ] **Streaming Responses**: Implement Server-Sent Events (SSE) for token-by-token streaming in the chat interface.
- [ ] **Expanded Knowledge Base**: Support automated ingestion of warranty PDFs, customer order lookup APIs, and extended product lines.

---

