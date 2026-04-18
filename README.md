# 📄 Production-Ready RAG AI Agent

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload PDFs and ask questions about their content. Built with **FastAPI**, **Inngest** (retries + observability), **Qdrant** (vector database), and **Groq** (free LLM) — running locally with **zero OpenAI costs**.

## ✨ Features

- **Upload PDFs** → Automatic chunking and embedding via Streamlit UI
- **Ask questions** → Semantic search + LLM answers with source citations
- **Production-ready** → Retries, logging, observability via Inngest
- **100% free** → No OpenAI API costs (uses `sentence-transformers` + Groq)
- **Modular codebase** → Clean separation of data loading, vector storage, and API logic

## 🏗️ Architecture

```
Streamlit (Frontend)
       │
       ▼ fires Inngest events
Inngest Dev Server (Orchestration + Observability)
       │
       ▼ triggers functions
FastAPI (Backend — main.py)
       │
       ├──► data_loader.py     → LlamaIndex PDF chunking + sentence-transformers embeddings
       ├──► vector_db.py       → Qdrant upsert / semantic search
       └──► custom_types.py    → Pydantic response models
              │
              ▼
         Groq API (LLM — llama-3.3-70b-versatile)
```

## 🗂️ Project Structure

```
production-rag-agent/
│
├── main.py              # FastAPI app + Inngest function handlers
├── data_loader.py       # PDF loading (LlamaIndex), chunking, embedding (sentence-transformers)
├── vector_db.py         # Qdrant client wrapper — upsert & semantic search
├── custom_types.py      # Pydantic models for typed responses
├── streamlit_app.py     # Streamlit UI — upload PDFs + ask questions
├── test_embed.py        # Quick sanity check for embeddings
├── pyproject.toml       # Project dependencies (managed with uv)
├── .env                 # API keys 
└── uploads/             # Uploaded PDFs saved here temporarily
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| Orchestration | Inngest (retries, logging, observability) |
| Vector Database | Qdrant (Docker) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Frontend | Streamlit |
| PDF Processing | LlamaIndex (`SimpleDirectoryReader`, `SentenceSplitter`) |
| Data Validation | Pydantic |
| Package Manager | uv |

## 📋 Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- Node.js (for Inngest CLI)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)
- Groq API key — free at [console.groq.com](https://console.groq.com)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ghassen236/production-rag-agent.git
cd production-rag-agent
```


### 2. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

```bash
# Install uv if you don't have it
pip install uv

# Install all project dependencies
uv sync
```

Or install packages individually:

```bash
uv add fastapi uvicorn inngest llama-index-core llama-index-readers-file \
       python-dotenv qdrant-client streamlit sentence-transformers groq pydantic
```

### 3. Set up environment variables

you need a free Groq API key. Here's how to get one:

1. Go to https://console.groq.com/home
2. Sign up or log in
3. In the left sidebar click **API Keys**
4. Click **Create API Key**, give it a name, and copy it somewhere safe


Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Start Qdrant with Docker

```bash
docker run -d --name qdrantRagDb -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5. Run all three services

**Terminal 1 — FastAPI Backend:**

```bash
uv run uvicorn main:app --reload
```

**Terminal 2 — Inngest Dev Server:**

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

**Terminal 3 — Streamlit Frontend:**

```bash
uv run streamlit run streamlit_app.py
```

### 6. Open the interfaces

| Interface | URL |
|-----------|-----|
| Streamlit UI | http://localhost:8501 |
| Inngest Observability | http://localhost:8288 |
| FastAPI Docs | http://localhost:8000/docs |

## 📖 Usage

1. Open the **Streamlit UI** and upload a PDF
2. Wait for ingestion to complete (watch progress in the Inngest dashboard)
3. Type a question in the "Ask a question" form
4. Get an answer with source citations

**Example questions:**
- "What position is the person applying for?"
- "What technologies are mentioned?"
- "Summarize the main topic of this document."

## 🔧 How It Works

### PDF Ingestion Flow

1. User uploads a PDF via Streamlit
2. Streamlit fires a `rag/ingest_pdf` Inngest event
3. FastAPI handles the event and calls `data_loader.py`
4. LlamaIndex loads and splits the PDF into chunks (1000 tokens, 200 overlap)
5. `sentence-transformers` encodes each chunk into a 384-dim vector
6. Vectors and text are stored in Qdrant via `vector_db.py`

### Query Flow

1. User submits a question via Streamlit
2. Streamlit fires a `rag/query_pdf_ai` event and polls for the result
3. FastAPI embeds the question using the same model
4. Qdrant performs cosine similarity search and returns the top-k chunks
5. Chunks + question are sent to Groq's LLM
6. Answer is returned to Streamlit with source citations

## 🧪 Testing Embeddings

Run a quick sanity check to verify your embedding setup:

```bash
uv run python test_embed.py
# Expected output: Embedding dimension: 384
```

## 📝 Notes

- Uploaded PDFs are saved temporarily in the `uploads/` folder
- The Qdrant collection is persisted in `qdrant_storage/` via Docker volume
- Inngest runs in local dev mode (`is_production=False`) — no cloud account needed
