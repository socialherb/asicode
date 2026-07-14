#!/usr/bin/env python3
"""
Quick test to verify embedding model singleton works.
"""
import logging

logging.basicConfig(level=logging.INFO)

from external_llm.agent.vector_cache import (
    HAS_FAISS,
    HAS_NUMPY,
    HAS_SENTENCE_TRANSFORMERS,
    VectorCacheManager,
    get_global_embedding_model,
    reset_global_embedding_model,
)

print("Testing embedding model singleton...")
print(f"Dependencies: HAS_SENTENCE_TRANSFORMERS={HAS_SENTENCE_TRANSFORMERS}, HAS_NUMPY={HAS_NUMPY}, HAS_FAISS={HAS_FAISS}")

if HAS_SENTENCE_TRANSFORMERS and HAS_NUMPY:
    # Reset first
    reset_global_embedding_model()

    # First call - should load model
    print("\n1. First call to get_global_embedding_model():")
    model1 = get_global_embedding_model()
    print(f"   Model: {model1}")

    # Second call - should return same instance
    print("\n2. Second call to get_global_embedding_model():")
    model2 = get_global_embedding_model()
    print(f"   Model: {model2}")

    # Verify they're the same object
    print(f"\n3. Same instance? {model1 is model2}")

    # Test VectorCacheManager sharing
    print("\n4. Creating VectorCacheManager instances:")
    manager1 = VectorCacheManager("/tmp/test_cache_1")
    manager2 = VectorCacheManager("/tmp/test_cache_2")
    print(f"   Manager1 embedding_model: {manager1.embedding_model}")
    print(f"   Manager2 embedding_model: {manager2.embedding_model}")
    print(f"   Same model? {manager1.embedding_model is manager2.embedding_model}")

    # Test dimension
    from external_llm.agent.vector_cache import get_global_embedding_dimension
    dimension = get_global_embedding_dimension()
    print(f"\n5. Global embedding dimension: {dimension}")
    print(f"   Manager1 dimension: {manager1.dimension}")
    print(f"   Manager2 dimension: {manager2.dimension}")

    # Clean up
    import shutil
    for path in ["/tmp/test_cache_1", "/tmp/test_cache_2"]:
        shutil.rmtree(path, ignore_errors=True)
    print("\n6. Cleaned up test directories.")
else:
    print("Dependencies not available, skipping model loading test.")
