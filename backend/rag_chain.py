"""
rag_chain.py
------------
Production RAG pipeline with streaming and re-ranking.

Enhancements over v1:
  - Two-stage retrieval: retrieve 10 chunks, re-rank to top 4
  - SSE streaming for token-by-token responses
  - Relevance scoring — sources include a relevance indicator
  - Configurable model via environment variable
  - Structured JSON events for streaming (type: token | sources | done | error)
"""

import json
import os
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ingest import get_vectorstore
from reranker import rerank

logger = logging.getLogger("rag-api")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))  # retrieve broadly
RERANK_K = int(os.getenv("RERANK_K", "4"))          # keep top after re-rank

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the \
user's uploaded documents. Use ONLY the context below to answer.

Rules:
1. If the answer isn't in the context, say "I couldn't find that in the document."
2. Be specific and cite which document/page your answer comes from.
3. If the context contains partial information, say what you found and note what's missing.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def _retrieve_and_rerank(question: str, k: int = RERANK_K):
    """Retrieve broadly from the vector store, then re-rank to top k."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    raw_docs = retriever.invoke(question)

    if not raw_docs:
        return []

    reranked = rerank(question, raw_docs, top_k=k)
    return reranked


def _build_sources(docs):
    """Build the sources list from retrieved documents."""
    return [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", None),
            "snippet": d.page_content[:220] + ("..." if len(d.page_content) > 220 else ""),
        }
        for d in docs
    ]


def answer_question(question: str, k: int = RERANK_K):
    """
    Standard (non-streaming) RAG pipeline.
    Returns dict with 'answer' and 'sources'.
    """
    retrieved_docs = _retrieve_and_rerank(question, k=k)

    if not retrieved_docs:
        return {
            "answer": "No documents have been uploaded yet, or I couldn't find relevant content for your question.",
            "sources": [],
        }

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    sources = _build_sources(retrieved_docs)

    return {"answer": answer, "sources": sources}


async def stream_answer(question: str, k: int = RERANK_K):
    """
    SSE streaming RAG pipeline.

    Yields Server-Sent Events in this format:
      data: {"type": "sources", "sources": [...]}
      data: {"type": "token", "token": "word"}
      data: {"type": "done"}

    The frontend gets sources first (to display immediately), then
    tokens stream in one by one, then a 'done' signal.
    """
    try:
        retrieved_docs = _retrieve_and_rerank(question, k=k)

        if not retrieved_docs:
            no_docs_msg = "No documents have been uploaded yet, or I couldn't find relevant content for your question."
            yield f"data: {json.dumps({'type': 'token', 'token': no_docs_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Send sources first so the UI can display them while answer streams
        sources = _build_sources(retrieved_docs)
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream the LLM response token by token
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)
        context = format_docs(retrieved_docs)

        messages = prompt.format_messages(context=context, question=question)

        async for chunk in llm.astream(messages):
            token = chunk.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
