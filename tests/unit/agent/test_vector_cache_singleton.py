"""
Unit tests for vector cache embedding model singleton.
"""
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from external_llm.agent.vector_cache import (
    VectorCacheManager,
    get_configured_embedding_model_name,
    get_global_embedding_model,
    reset_global_embedding_model,
)

EXPECTED_MODEL = get_configured_embedding_model_name()


def test_global_embedding_model_singleton():
    """Test that get_global_embedding_model() returns the same instance."""
    # Mock SentenceTransformer to avoid actual model loading
    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True):
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        # Reset global state before test
        reset_global_embedding_model()

        # First call should create model
        model1 = get_global_embedding_model()
        # Second call should return same instance
        model2 = get_global_embedding_model()

        assert model1 is model2
        mock_st.assert_called_once_with(EXPECTED_MODEL)


def test_vector_cache_manager_shares_model():
    """Test that VectorCacheManager instances share the same embedding model."""
    # Mock SentenceTransformer and dependencies
    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True), \
         patch('external_llm.agent.vector_cache.HAS_NUMPY', True), \
         patch('external_llm.agent.vector_cache.HAS_FAISS', False):
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        # Reset global state before test
        reset_global_embedding_model()

        # Create two managers (model is lazy-loaded, not at construction)
        manager1 = VectorCacheManager("/tmp/test_cache1")
        manager2 = VectorCacheManager("/tmp/test_cache2")

        # Trigger lazy loading on the first manager
        manager1._ensure_model_loaded()
        manager2._ensure_model_loaded()

        # Both should share the same embedding model instance
        assert manager1.embedding_model is manager2.embedding_model

        # Model should have been loaded only once
        mock_st.assert_called_once_with(EXPECTED_MODEL)


def test_reset_global_embedding_model():
    """Test that reset_global_embedding_model() clears the singleton."""
    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True):
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        # Reset and get model
        reset_global_embedding_model()
        model1 = get_global_embedding_model()

        # Reset again
        reset_global_embedding_model()

        # Get model again - should create new instance
        model2 = get_global_embedding_model()

        # Since we mocked SentenceTransformer, both will be the same mock instance
        # but the mock should have been called twice
        assert mock_st.call_count == 2
        assert model1 is model2  # Both are the same mock instance


def test_get_global_embedding_dimension():
    """Test get_global_embedding_dimension() returns correct dimension."""
    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True):
        mock_instance = Mock()
        mock_instance.get_embedding_dimension.return_value = 512
        mock_st.return_value = mock_instance

        reset_global_embedding_model()
        get_global_embedding_model()

        # Import the function
        from external_llm.agent.vector_cache import get_global_embedding_dimension
        dimension = get_global_embedding_dimension()

        assert dimension == 512


def test_falls_back_to_cached_model_when_preferred_unavailable():
    """Preferred model failing to load should fall back, not disable embeddings."""
    from external_llm.agent.vector_cache import (
        FALLBACK_EMBEDDING_MODELS,
        get_loaded_embedding_model_name,
    )

    fallback = FALLBACK_EMBEDDING_MODELS[0]

    def fake_st(name):
        if name != fallback:
            raise OSError("couldn't connect to huggingface.co")
        inst = Mock()
        inst.get_sentence_embedding_dimension.return_value = 384
        return inst

    with patch('external_llm.agent.vector_cache.SentenceTransformer', side_effect=fake_st), \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True):
        reset_global_embedding_model()
        model = get_global_embedding_model()

        assert model is not None  # did not disable; used the fallback
        assert get_loaded_embedding_model_name() == fallback


def test_all_models_failing_returns_none():
    """When every candidate fails, the loader returns None (graceful no-op)."""
    with patch('external_llm.agent.vector_cache.SentenceTransformer',
               side_effect=OSError("offline")), \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True):
        reset_global_embedding_model()
        assert get_global_embedding_model() is None


def test_vector_cache_manager_without_dependencies():
    """Test VectorCacheManager when dependencies are missing."""
    # Temporarily patch the flags
    with patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', False):
        with patch('external_llm.agent.vector_cache.HAS_NUMPY', False):
            with patch('external_llm.agent.vector_cache.HAS_FAISS', False):
                reset_global_embedding_model()
                manager = VectorCacheManager("/tmp/test_cache_no_deps")

                # Model should be None
                assert manager.embedding_model is None
                # Dimension should fall back to default
                assert manager.dimension == 384


def test_warmup_does_not_double_load_with_concurrent_caller():
    """Warmup running concurrently with a real caller must load exactly once.

    The core invariant: get_global_embedding_model's lock + double-check must
    guarantee a single SentenceTransformer(...) instantiation even when a
    background warmup thread and the main thread both call the loader. This is
    the safety property that makes background warmup correct.
    """
    import threading

    from external_llm.agent.vector_cache import warmup_embedding_model

    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True), \
         patch('external_llm.agent.vector_cache.HAS_NUMPY', True), \
         patch('external_llm.agent.vector_cache.HAS_FAISS', True):
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        reset_global_embedding_model()

        barrier = threading.Barrier(2)

        def _warmup_then_get():
            barrier.wait()  # release both threads simultaneously
            warmup_embedding_model()

        t = threading.Thread(target=_warmup_then_get, daemon=True)
        t.start()
        # Main thread races the warmup thread into the loader.
        barrier.wait()
        model_main = get_global_embedding_model()
        t.join(timeout=5)

        assert model_main is mock_instance
        assert get_global_embedding_model() is mock_instance  # still singleton
        # Exactly one instantiation despite two concurrent callers.
        assert mock_st.call_count == 1


def test_warmup_noop_when_deps_missing():
    """warmup_embedding_model must be a safe no-op when deps are absent."""
    from external_llm.agent.vector_cache import warmup_embedding_model

    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', False):
        reset_global_embedding_model()
        warmup_embedding_model()  # must not raise, must not load
        assert mock_st.call_count == 0
        assert get_global_embedding_model() is None


def test_warmup_noop_when_already_loaded():
    """warmup_embedding_model must not re-invoke the loader when warm."""
    from external_llm.agent.vector_cache import warmup_embedding_model

    with patch('external_llm.agent.vector_cache.SentenceTransformer') as mock_st, \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True), \
         patch('external_llm.agent.vector_cache.HAS_NUMPY', True), \
         patch('external_llm.agent.vector_cache.HAS_FAISS', True):
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        reset_global_embedding_model()
        get_global_embedding_model()  # first load
        assert mock_st.call_count == 1
        warmup_embedding_model()  # already loaded — must short-circuit
        assert mock_st.call_count == 1  # unchanged


def test_warmup_swallows_loader_exception():
    """A loader failure must not propagate from the warmup primitive."""
    from external_llm.agent.vector_cache import warmup_embedding_model

    with patch('external_llm.agent.vector_cache.SentenceTransformer',
               side_effect=RuntimeError("boom")), \
         patch('external_llm.agent.vector_cache.HAS_SENTENCE_TRANSFORMERS', True), \
         patch('external_llm.agent.vector_cache.HAS_NUMPY', True), \
         patch('external_llm.agent.vector_cache.HAS_FAISS', True):
        reset_global_embedding_model()
        # Must not raise even though all candidates fail.
        warmup_embedding_model()
if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
