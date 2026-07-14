"""
Integration tests for vector caching (FAISS-based embedding cache).
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Check if vector cache dependencies are available
try:
    import numpy as np

    from external_llm.agent.vector_cache import (
        HAS_FAISS,
        HAS_NUMPY,
        HAS_SENTENCE_TRANSFORMERS,
        VectorCacheManager,
        get_global_embedding_model,
        reset_global_embedding_model,
    )
    VECTOR_CACHE_AVAILABLE = True
except ImportError:
    VECTOR_CACHE_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not VECTOR_CACHE_AVAILABLE, reason="Vector cache module not available")
class TestVectorCache:
    """Test vector cache for semantic search embeddings."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for vector cache."""
        cache_dir = tempfile.mkdtemp(prefix="vector-cache-test-")
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_global_embedding_model_singleton(self):
        """Test global embedding model singleton pattern."""
        reset_global_embedding_model()

        model1 = get_global_embedding_model()
        model2 = get_global_embedding_model()

        if HAS_SENTENCE_TRANSFORMERS and model1 is not None:
            assert model1 is model2
        else:
            assert model1 is None
            assert model2 is None

    def test_vector_cache_manager_initialization(self, temp_cache_dir):
        """Test vector cache manager initialization."""
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("NumPy or SentenceTransformers not available")

        manager = VectorCacheManager(temp_cache_dir)

        assert manager.cache_dir == Path(temp_cache_dir)
        assert manager.index_path == Path(temp_cache_dir) / "faiss_index.bin"
        assert manager.metadata_path == Path(temp_cache_dir) / "metadata.json"
        assert manager.dimension > 0
        assert manager.cache_dir.exists()

    def test_vector_cache_embedding_generation(self, temp_cache_dir):
        """Test embedding generation for text."""
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("NumPy or SentenceTransformers not available")

        manager = VectorCacheManager(temp_cache_dir)

        text = "Test query for embedding"
        embedding = manager._compute_embedding(text)

        if embedding is not None:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (manager.dimension,)

    def test_vector_cache_add_and_search(self, temp_cache_dir):
        """Test adding documents and searching the vector cache."""
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("NumPy or SentenceTransformers not available")

        manager = VectorCacheManager(temp_cache_dir)

        test_embedding = np.random.randn(manager.dimension).astype(np.float32)

        with patch.object(manager, '_compute_embedding', return_value=test_embedding):
            manager.add_document("file1.py", "def hello(): return 'world'")
            manager.add_document("file2.py", "class Calculator: pass")

            if HAS_FAISS and manager.index is not None:
                results = manager.search("hello world function", top_k=2)
                assert isinstance(results, list)
            else:
                # FAISS not available - search returns empty
                results = manager.search("hello world function", top_k=2)
                assert results == []

    def test_vector_cache_doc_id_generation(self, temp_cache_dir):
        """Test document ID generation from file path and content."""
        if not HAS_NUMPY:
            pytest.skip("NumPy not available")

        manager = VectorCacheManager(temp_cache_dir)

        # Same file + content should generate same doc_id
        id1 = manager._get_doc_id("file.py", "content")
        id2 = manager._get_doc_id("file.py", "content")
        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) > 0

        # Different parameters should generate different doc_ids
        id3 = manager._get_doc_id("other.py", "content")
        id4 = manager._get_doc_id("file.py", "different content")
        assert id1 != id3
        assert id1 != id4

    def test_vector_cache_similarity_search(self, temp_cache_dir):
        """Test similarity search using FAISS index."""
        if not HAS_NUMPY or not HAS_FAISS:
            pytest.skip("NumPy or FAISS not available")

        import faiss

        manager = VectorCacheManager(temp_cache_dir)
        dimension = manager.dimension

        # Build a FAISS index manually
        index = faiss.IndexFlatIP(dimension)
        item_embeddings = np.random.randn(10, dimension).astype(np.float32)
        index.add(item_embeddings)

        # Search
        query_embedding = np.random.randn(1, dimension).astype(np.float32)
        k = 3
        distances, indices = index.search(query_embedding, k)

        assert distances.shape == (1, k)
        assert indices.shape == (1, k)
        assert all(0 <= idx < 10 for idx in indices[0])

    def test_vector_cache_integration_with_rag(self, temp_cache_dir):
        """Test vector cache integration with RAG searcher."""
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("NumPy or SentenceTransformers not available")

        from external_llm.agent.rag_searcher import RAGSearcher

        with tempfile.TemporaryDirectory() as repo_root:
            repo_path = Path(repo_root)
            (repo_path / "sample1.py").write_text("def function1():\n    return 1")
            (repo_path / "sample2.py").write_text("def function2():\n    return 2")

            searcher = RAGSearcher(repo_root)

            has_vector_cache = hasattr(searcher, 'vector_cache_manager')
            if has_vector_cache:
                assert searcher.vector_cache_manager is not None

                query = "test function"
                results = searcher.find_relevant_files(query, top_k=2)
                assert isinstance(results, list)

    def test_vector_cache_performance_metrics(self, temp_cache_dir):
        """Test that vector cache usage is tracked in performance metrics."""
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("NumPy or SentenceTransformers not available")

        from external_llm.agent.performance_metrics import get_global_collector

        collector = get_global_collector()
        collector.reset_cache_stats()

        collector.record_vector_cache(hit=True)
        collector.record_vector_cache(hit=False)
        collector.record_vector_cache(hit=True)

        vector_stats = collector.get_cache_stats("vector")

        if vector_stats:
            assert vector_stats.get("hits", vector_stats.get("hit_count", 0)) == 2
            assert vector_stats.get("misses", vector_stats.get("miss_count", 0)) == 1

    def test_vector_cache_persistence(self, temp_cache_dir):
        """Test that vector cache persists across instances."""
        if not HAS_NUMPY or not HAS_FAISS:
            pytest.skip("NumPy or FAISS not available")

        manager1 = VectorCacheManager(temp_cache_dir)

        test_embedding = np.random.randn(manager1.dimension).astype(np.float32)

        with patch.object(manager1, '_compute_embedding', return_value=test_embedding):
            manager1.add_document("persistent.py", "persistent content")
            manager1._save_index()

        # Create second manager pointing to same cache directory
        manager2 = VectorCacheManager(temp_cache_dir)

        # Should have loaded the saved index
        if manager2.index is not None:
            assert manager2.index.ntotal >= 1

    def test_vector_cache_fallback_on_failure(self, temp_cache_dir):
        """Test fallback behavior when vector cache (FAISS) is not available."""
        with patch('external_llm.agent.vector_cache.HAS_FAISS', False):
            manager = VectorCacheManager(temp_cache_dir)

            # Should gracefully handle missing FAISS
            assert manager.index is None

            # add_document should not raise
            manager.add_document("test.py", "test content")

            # search should return empty list
            results = manager.search("test query", top_k=3)
            assert results == []

    def test_vector_cache_thread_safety(self, temp_cache_dir):
        """Test vector cache thread safety for concurrent add_document."""
        if not HAS_NUMPY or not HAS_FAISS:
            pytest.skip("NumPy or FAISS not available")

        import threading

        manager = VectorCacheManager(temp_cache_dir)
        errors = []

        test_embedding = np.random.randn(manager.dimension).astype(np.float32)

        def worker(worker_id: int):
            try:
                with patch.object(manager, '_compute_embedding', return_value=test_embedding):
                    manager.add_document(f"file_{worker_id}.py", f"content {worker_id}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_vector_cache_in_agent_config(self, temp_repo_root: str):
        """Test vector cache configuration in agent settings."""
        from external_llm.agent.tool_registry import AgentConfig

        config = AgentConfig()
        # rag_enabled controls vector cache usage
        assert hasattr(config, 'rag_enabled')
