"""
Local cross-encoder reranker for search result relevance scoring.
Uses a multilingual model trained on mMARCO for Indonesian support.
"""

import asyncio
import threading
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_model: CrossEncoder = None
_lock = threading.Lock()


def load():
    """Pre-load the cross-encoder model at startup."""
    _get_model()


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model on first use."""
    global _model
    if _model is None:
        print(f"[Reranker] Loading model: {_MODEL_NAME}...")
        _model = CrossEncoder(_MODEL_NAME)
        print("[Reranker] Model loaded.")
    return _model


def _rerank_sync(query: str, results: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Synchronous reranking, guarded by a lock for thread safety."""
    if not results:
        return []

    model = _get_model()

    # Build (query, passage) pairs for scoring
    pairs = []
    for r in results:
        passage = f"{r.get('title', '')} {r.get('description', '')}".strip()
        pairs.append((query, passage))

    with _lock:
        scores = model.predict(pairs)

    # Attach scores and sort descending
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [r for r, _ in scored_results[:top_k]]


async def rerank(query: str, results: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Rerank search results by relevance to the query using a cross-encoder.
    Runs in a thread to avoid blocking the event loop.

    Args:
        query: The search query / claim being verified.
        results: List of search result dicts (must have 'title' and 'description').
        top_k: Number of top results to return after reranking.

    Returns:
        Top-k results sorted by relevance (most relevant first).
    """
    return await asyncio.to_thread(_rerank_sync, query, results, top_k)
