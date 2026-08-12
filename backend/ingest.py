"""
ingest.py
---------
Production-ready document ingestion pipeline.

Enhancements over v1:
  - Supports PDF, DOCX, TXT, and CSV file types
  - Async file handling for better performance
  - Chunk deduplication to avoid indexing the same content twice
  - Better metadata tagging (source, page, file_type, ingested_at)
  - Configurable chunk size via environment variables

Flow:
  1. Detect file type and load with the appropriate LangChain loader.
  2. Split into overlapping chunks.
  3. Tag each chunk with rich metadata.
  4. Embed and store in Chroma (skipping duplicates).
"""

import os
import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = str(Path(__file__).parent / "chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore():
    """Load the existing vector store (or an empty one if none exists yet)."""
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def _load_document(file_path: str, file_type: str):
    """Pick the right LangChain loader based on file extension."""
    if file_type == ".pdf":
        return PyPDFLoader(file_path).load()
    elif file_type == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(file_path).load()
        except ImportError:
            raise ImportError("Install docx2txt: pip install docx2txt")
    elif file_type == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif file_type == ".csv":
        return CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _chunk_hash(text: str) -> str:
    """Create a short hash of chunk content for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


async def ingest_file(upload: UploadFile, source_name: str) -> int:
    """
    Save uploaded file to temp, load it, chunk it, embed it, store it.
    Returns the number of NEW chunks added (after dedup).
    """
    file_ext = "." + source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""

    # Save upload to a temp file so loaders can read from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = tmp.name

    try:
        pages = _load_document(tmp_path, file_ext)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Tag every page with source metadata
    now = datetime.now(timezone.utc).isoformat()
    for page in pages:
        page.metadata["source"] = source_name
        page.metadata["file_type"] = file_ext
        page.metadata["ingested_at"] = now

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    # Deduplicate: skip chunks whose content already exists in the store
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    existing = set()
    try:
        existing_meta = collection.get()["metadatas"]
        existing = {m.get("content_hash") for m in existing_meta if m.get("content_hash")}
    except Exception:
        pass

    new_chunks = []
    for chunk in chunks:
        h = _chunk_hash(chunk.page_content)
        if h not in existing:
            chunk.metadata["content_hash"] = h
            new_chunks.append(chunk)
            existing.add(h)

    if new_chunks:
        vectorstore.add_documents(new_chunks)

    return len(new_chunks)
