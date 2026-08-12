"""
reranker.py
-----------
Cross-encoder re-ranking to improve retrieval quality.

Why re-rank?
  Vector similarity search (embedding cosine distance) is fast but rough.
  A cross-encoder reads the QUESTION and each CHUNK together, scoring
  relevance more accurately. We retrieve more chunks than we need (e.g. 10),
  re-rank them, then keep only the top-k (e.g. 4).

  This two-stage approach (retrieve broadly, then re-rank precisely) is
  standard in production RAG systems and measurably improves answer quality.

Uses the `sentence-transformers` cross-encoder, which runs locally (no API cost).
Falls back to no re-ranking if the model isn't available.
"""

import logging
import os

logger = logging.getLogger("rag-api")

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (only on first use)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(RERANK_MODEL)
            logger.info(f"Loaded re-ranker model: {RERANK_MODEL}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — re-ranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            _cross_encoder = False  # sentinel: don't retry
        except Exception as e:
            logger.warning(f"Failed to load re-ranker: {e}")
            _cross_encoder = False
    return _cross_encoder


def rerank(question: str, documents: list, top_k: int = 4) -> list:
    """
    Re-rank retrieved documents by cross-encoder relevance score.

    Args:
        question: The user's question.
        documents: List of LangChain Document objects from retrieval.
        top_k: How many to keep after re-ranking.

    Returns:
        The top_k most relevant documents, sorted by score (highest first).
    """
    if not RERANK_ENABLED or not documents:
        return documents[:top_k]

    encoder = _get_cross_encoder()
    if not encoder:
        return documents[:top_k]

    # Score each (question, chunk) pair
    pairs = [(question, doc.page_content) for doc in documents]
    scores = encoder.predict(pairs)

    # Attach scores and sort
    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Log score spread for debugging
    if scored:
        top_score = scored[0][1]
        bottom_score = scored[-1][1]
        logger.info(
            f"Re-rank: {len(documents)} docs, scores {bottom_score:.3f}–{top_score:.3f}, "
            f"keeping top {top_k}"
        )

    return [doc for doc, _ in scored[:top_k]]
