"""
tests/test_embeddings.py

Focused unit tests for core.embeddings singleton, verifying:
- Singleton identity (same object returned across calls)
- Thread safety (concurrent access yields single instantiation)
- Lazy loading and reset functionality
- Error handling on missing dependency
- Caller compatibility (career_tools and all three RAG retrievers share the same instance)
"""
from __future__ import annotations

import concurrent.futures
import sys
from unittest.mock import MagicMock, patch

import pytest

from core.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    get_shared_embedding_model,
    is_shared_embedding_model_loaded,
    reset_shared_embedding_model,
)


@pytest.fixture(autouse=True)
def clean_shared_model(monkeypatch):
    """Ensure every test starts with an uninitialized shared model and clean env."""
    reset_shared_embedding_model()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    yield
    reset_shared_embedding_model()


def test_singleton_identity():
    """Calling get_shared_embedding_model repeatedly returns the exact same object."""
    mock_instance = MagicMock(name="MockSentenceTransformer")
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
        m1 = get_shared_embedding_model()
        m2 = get_shared_embedding_model()

        assert m1 is m2
        assert m1 is mock_instance
        mock_cls.assert_called_once_with(DEFAULT_EMBEDDING_MODEL)


def test_thread_safe_initialization():
    """Concurrent calls across threads instantiate the model exactly once."""
    mock_instance = MagicMock(name="MockSentenceTransformer")
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
        results = []

        def worker():
            return get_shared_embedding_model()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(16)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        assert len(results) == 16
        for instance in results:
            assert instance is mock_instance
        mock_cls.assert_called_once()


def test_lazy_loading_and_reset():
    """Model is not instantiated until requested, and reset clears the singleton."""
    assert not is_shared_embedding_model_loaded()

    mock_instance = MagicMock(name="MockSentenceTransformer")
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance):
        m = get_shared_embedding_model()
        assert is_shared_embedding_model_loaded()
        assert m is mock_instance

        reset_shared_embedding_model()
        assert not is_shared_embedding_model_loaded()


def test_missing_sentence_transformers_raises_importerror(monkeypatch):
    """If sentence_transformers cannot be imported, raise an informative ImportError."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match="sentence-transformers is required"):
        get_shared_embedding_model()


def test_custom_model_and_token(monkeypatch):
    """Model name override and HF_TOKEN are passed to constructor."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
    mock_instance = MagicMock(name="MockSentenceTransformer")
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
        m = get_shared_embedding_model("custom-model-id")
        assert m is mock_instance
        mock_cls.assert_called_once_with("custom-model-id", token="hf_test_secret_token")


def test_career_tools_caller_compatibility():
    """tools.career_tools.get_embedding_model returns the shared singleton."""
    from tools import career_tools

    mock_instance = MagicMock(name="MockSentenceTransformer")
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
        career_model = career_tools.get_embedding_model()
        shared_model = get_shared_embedding_model()

        assert career_model is shared_model
        assert career_model is mock_instance
        mock_cls.assert_called_once()


def test_retrievers_share_single_instance(monkeypatch):
    """All three RAG retrievers obtain the same shared model without separate instantiations."""
    import rag.finance_retriever as f_ret
    import rag.health_retriever as h_ret
    import rag.retriever as g_ret

    # Reset any existing loaded models in retriever modules
    g_ret._model = None
    h_ret._model = None
    f_ret._model = None

    mock_instance = MagicMock(name="MockSentenceTransformer")
    mock_instance.encode.return_value = MagicMock()

    # Mock FAISS read_index and pickle load to avoid disk dependencies
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("faiss.read_index", lambda path: MagicMock())
    monkeypatch.setattr("pickle.load", lambda f: {0: {"domain": "career", "source": "kb", "text": "text"}})
    monkeypatch.setattr("builtins.open", MagicMock())

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance) as mock_cls:
        g_ret._load()
        h_ret._load()
        f_ret._load()

        assert g_ret._model is mock_instance
        assert h_ret._model is mock_instance
        assert f_ret._model is mock_instance
        assert g_ret._model is h_ret._model is f_ret._model

        # Only 1 constructor call across all 3 retrievers
        mock_cls.assert_called_once()
