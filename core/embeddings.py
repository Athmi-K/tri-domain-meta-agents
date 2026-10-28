"""
core/embeddings.py

Thread-safe singleton provider for the shared SentenceTransformer embedding model.
Ensures that all components (career tools, RAG retrievers) reuse a single
in-memory model instance on CPU instead of allocating multiple native runtimes.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_shared_model: Optional[Any] = None
_model_lock = threading.Lock()


def get_shared_embedding_model(
    model_name: Optional[str] = None,
) -> Any:
    """
    Get or initialize the shared SentenceTransformer model instance (thread-safe singleton).

    Args:
        model_name: Optional override for the model identifier. Defaults to
                    the EMBEDDING_MODEL environment variable or 'all-MiniLM-L6-v2'.

    Returns:
        The shared SentenceTransformer model instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        Exception: If model instantiation fails.
    """
    global _shared_model

    if _shared_model is not None:
        return _shared_model

    with _model_lock:
        if _shared_model is not None:
            return _shared_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error("sentence-transformers is not installed: %s", exc)
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        target_model = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        hf_token = os.getenv("HF_TOKEN") or None

        logger.info("Initializing shared embedding model: %s", target_model)
        if hf_token:
            _shared_model = SentenceTransformer(target_model, token=hf_token)
        else:
            _shared_model = SentenceTransformer(target_model)

        logger.info("Shared embedding model initialized successfully: %s", target_model)
        return _shared_model


def is_shared_embedding_model_loaded() -> bool:
    """Check whether the shared model singleton has already been loaded."""
    return _shared_model is not None


def reset_shared_embedding_model() -> None:
    """
    Reset the shared model singleton (primarily for testing and mock isolation).
    """
    global _shared_model
    with _model_lock:
        _shared_model = None
