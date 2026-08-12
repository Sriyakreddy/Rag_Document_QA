"""
main.py
-------
Production-ready FastAPI app for RAG Document Q&A.

Enhancements over v1:
  - SSE streaming responses (token-by-token)
  - Rate limiting (slowapi)
  - Request logging with structured output
  - Health check with dependency status
  - Metrics endpoint (request count, latency, errors)
  - CORS configured for dev and production
  - Graceful error handling throughout

Run locally:
  uvicorn main:app --reload --port 8000
"""

import time
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from ingest import ingest_file, get_vectorstore, SUPPORTED_EXTENSIONS
from rag_chain import answer_question, stream_answer
from metrics import metrics

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-api")

# ── Rate Limiter ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API starting up")
    yield
    logger.info("RAG API shutting down")


app = FastAPI(
    title="RAG Document Q&A API",
    version="2.0.0",
    description="Production-ready RAG with streaming, re-ranking, and multi-format support.",
    lifespan=lifespan,
)
app.state.limiter = limiter


# Rate-limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )


# CORS — allow React dev server + common production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:80",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ───────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    metrics.record_request(request.url.path, duration_ms)
    logger.info(f"{request.method} {request.url.path} — {duration_ms:.0f}ms")
    return response


# ── Request / Response Models ────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    k: int = 4  # number of chunks to retrieve


class DocumentInfo(BaseModel):
    filename: str
    chunks_added: int


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check with dependency status."""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        doc_count = collection.count()
        db_status = "connected"
    except Exception as e:
        doc_count = 0
        db_status = f"error: {e}"

    return {
        "status": "ok",
        "version": "2.0.0",
        "vector_db": db_status,
        "documents_indexed": doc_count,
    }


@app.post("/upload", response_model=DocumentInfo)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT, CSV)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    try:
        chunk_count = await ingest_file(file, source_name=file.filename)
        metrics.record_upload(file.filename, chunk_count)
        logger.info(f"Indexed '{file.filename}' — {chunk_count} chunks")
    except Exception as e:
        logger.error(f"Ingest failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    return DocumentInfo(filename=file.filename, chunks_added=chunk_count)


@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    """Standard (non-streaming) RAG response."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(req.question, k=req.k)
        metrics.record_query(req.question)
        return result
    except Exception as e:
        metrics.record_error("chat", str(e))
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")


@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest):
    """SSE streaming RAG response — tokens arrive in real-time."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        generator = stream_answer(req.question, k=req.k)
        metrics.record_query(req.question)
        return StreamingResponse(generator, media_type="text/event-stream")
    except Exception as e:
        metrics.record_error("chat_stream", str(e))
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@app.get("/documents")
def list_documents():
    """List all indexed documents and their chunk counts."""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        all_metadata = collection.get()["metadatas"]

        doc_map = {}
        for meta in all_metadata:
            source = meta.get("source", "unknown")
            doc_map[source] = doc_map.get(source, 0) + 1

        return {
            "documents": [
                {"name": name, "chunks": count}
                for name, count in sorted(doc_map.items())
            ],
            "total_chunks": sum(doc_map.values()),
        }
    except Exception:
        return {"documents": [], "total_chunks": 0}


@app.delete("/documents/{filename}")
@limiter.limit("5/minute")
async def delete_document(request: Request, filename: str):
    """Remove a document's chunks from the vector store."""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        # Find IDs with matching source
        results = collection.get(where={"source": filename})
        ids_to_delete = results["ids"]
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"No document found: {filename}")
        collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for '{filename}'")
        return {"deleted": filename, "chunks_removed": len(ids_to_delete)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Operational metrics for monitoring."""
    return metrics.summary()
