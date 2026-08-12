# RAG Document Q&A — Production Edition

A production-ready Retrieval-Augmented Generation app that lets you upload documents and ask questions grounded in their actual content. FastAPI + LangChain + Chroma on the backend, React on the frontend — with real-time streaming, cross-encoder re-ranking, multi-format support, and Docker deployment.

## What changed from v1

The first version was a working RAG pipeline — upload a PDF, ask questions, get answers with citations. This version makes it production-grade:

**Real-time streaming** — Answers stream token-by-token via Server-Sent Events instead of making you wait for the full response. The frontend renders each token as it arrives.

**Cross-encoder re-ranking** — v1 retrieved the top 4 chunks by embedding similarity and sent them straight to the LLM. v2 retrieves 10 chunks, then re-ranks them with a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) that reads the question and each chunk together to score relevance more accurately. Only the top 4 after re-ranking go to the LLM. This measurably improves answer quality on ambiguous questions.

**Multi-format support** — Now handles PDF, DOCX, TXT, and CSV files. Each uses the appropriate LangChain loader.

**Chunk deduplication** — Re-uploading the same document doesn't create duplicate chunks. Each chunk is hashed and checked before indexing.

**Document management** — API endpoints to list and delete indexed documents. The frontend sidebar shows what's indexed and lets you remove documents.

**Rate limiting** — `slowapi` prevents abuse: 10 uploads/min, 30 queries/min per IP.

**Metrics endpoint** — `/metrics` reports request counts, average/p95 latency per endpoint, error count, and uptime. Swap this for Prometheus in production.

**Evaluation pipeline** — `evaluate.py` uses DeepEval to measure answer relevancy, faithfulness (hallucination), and contextual relevancy across test cases.

**Docker deployment** — `docker compose up` runs the full stack. Frontend served via Nginx with API proxying and SSE support configured.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────┐
│  FastAPI Backend                        │
│                                         │
│  1. Retrieve 10 chunks (Chroma)         │
│  2. Re-rank with cross-encoder → top 4  │
│  3. Stuff into prompt                   │
│  4. Stream LLM response via SSE         │
└─────────────────────────────────────────┘
     │
     ▼
React Frontend (token-by-token rendering)
```

**Ingestion pipeline:**
```
Upload (PDF/DOCX/TXT/CSV)
  → Load with appropriate LangChain loader
  → Split into 1000-char chunks with 150-char overlap
  → Hash each chunk for deduplication
  → Embed with text-embedding-3-small
  → Store in Chroma with metadata (source, page, file_type, timestamp)
```

## Setup

### Local development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your OpenAI API key to .env

uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### Docker (one command)

```bash
cp backend/.env.example backend/.env
# Add your OpenAI API key to backend/.env

docker compose up --build
```

Open http://localhost.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with vector DB status |
| POST | `/upload` | Upload and index a document |
| POST | `/chat` | Standard RAG response |
| POST | `/chat/stream` | SSE streaming RAG response |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{name}` | Remove a document |
| GET | `/metrics` | Operational metrics |

## Evaluation

```bash
cd backend
pip install deepeval
deepeval test run evaluate.py
```

Edit `evaluate.py` to add your own test cases from your documents. The evaluation measures answer relevancy (is the answer on-topic?), faithfulness (does it hallucinate?), and contextual relevancy (did retrieval pull the right chunks?).

## Engineering decisions

**Why re-rank instead of just retrieving fewer chunks?** Embedding similarity is a rough approximation — it compares vectors, not meaning in context. A cross-encoder reads the question and chunk text together, which catches relevance that pure vector distance misses. The tradeoff is ~100ms extra latency per query, which is worth it for noticeably better answers.

**Why SSE instead of WebSockets?** For this use case (one-directional streaming from server to client), SSE is simpler, works over standard HTTP, and auto-reconnects. WebSockets would be overkill since the client only sends the initial question via a regular POST.

**Why local Chroma instead of Pinecone/Weaviate?** For a portfolio project that someone clones and runs locally, zero external dependencies is the right call. The code is structured so swapping to a hosted vector DB is a one-file change in `ingest.py`.

**Why `gpt-4o-mini` and not a local model?** Cost vs. quality tradeoff. For a demo, gpt-4o-mini gives high-quality answers at ~$0.15/1M input tokens. The `.env` makes it configurable — swap to Ollama/Llama for a fully free setup.

## Tech Stack

**Backend:** Python, FastAPI, LangChain, ChromaDB, OpenAI, sentence-transformers, slowapi

**Frontend:** React 18, Vite, react-markdown

**Deployment:** Docker, Nginx, docker-compose

## What I'd build next

- Add conversation memory so follow-up questions work ("what about page 3?" after a previous question)
- Add hybrid search (keyword BM25 + semantic) for better retrieval on technical documents
- Deploy to AWS/GCP with a hosted vector DB for persistence across redeploys
- Add user authentication and per-user document isolation
- Integrate LangSmith for production tracing and observability
