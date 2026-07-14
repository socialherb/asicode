"""
Vector Cache for asicode Agent

FAISS-based embedding cache for semantic search, integrated with RAG searcher.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from external_llm.common.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_hf_progress():
    """Suppress HF/transformers tqdm bars + chatter during a model load.

    SentenceTransformer/transformers write ``Loading weights: 100%|...| 199/199``
    to stderr flush-left from a worker thread with ``leave=True``, so the final
    flush can land out-of-band and break the CLI's column alignment. The bar is
    emitted via ``transformers.utils.logging.tqdm`` / ``huggingface_hub`` progress
    callbacks — NOT stdlib ``logging`` — so disabling those progress-bar APIs is
    sufficient. We must NOT call ``logging.disable()`` here: that would also
    silence our own ``logger.info("Loading SentenceTransformer model ...")`` and
    ``logger.info("Model loaded with dimension ...")`` messages, hiding the load
    from the user. Only the tqdm bars are suppressed.
    """
    _bar_restores = []
    for _mod, _off, _on in (
        ("huggingface_hub.utils", "disable_progress_bars", "enable_progress_bars"),
        ("transformers.utils.logging", "disable_progress_bar", "enable_progress_bar"),
    ):
        try:
            _m = __import__(_mod, fromlist=[_off, _on])
            getattr(_m, _off)()
            _bar_restores.append(getattr(_m, _on))
        except Exception as e:
            # API drift (e.g. transformers renames disable_progress_bar) would
            # otherwise let the tqdm bar leak back to the terminal silently.
            # Log at DEBUG so the leak is diagnosable without being noisy.
            logger.debug("_suppress_hf_progress: %s.%s unavailable: %s", _mod, _off, e)
    try:
        yield
    finally:
        for _restore in _bar_restores:
            try:
                _restore()
            except Exception:
                logger.debug("_suppress_hf_progress: _restore() failed", exc_info=True)

# Embedding model. Default to the multilingual MiniLM so non-English requests
# (this project's prompts are often Korean) embed well; it is the same 384-dim
# space as the previous English-only all-MiniLM-L6-v2, so FAISS index structure
# is unchanged — only the semantic content differs, which the on-disk cache
# invalidates by model name (see VectorCacheManager). Override via env for
# experiments or to pin the old model.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# Loaded in order until one succeeds. The multilingual default is preferred for
# non-English requests, but it must be fetched once from the network; when that
# is impossible (offline / HF unreachable) we fall back to the previous default,
# which is small and usually already cached locally — keeping embeddings working
# rather than disabling them. Both are 384-dim, so the FAISS index is unaffected.
FALLBACK_EMBEDDING_MODELS = ("all-MiniLM-L6-v2",)


def get_configured_embedding_model_name() -> str:
    """Resolve the embedding model name (env override → multilingual default)."""
    return (os.environ.get("ASICODE_EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL


def _embedding_model_candidates() -> list:
    """Ordered, de-duplicated model names to attempt loading."""
    candidates = [get_configured_embedding_model_name()]
    for name in FALLBACK_EMBEDDING_MODELS:
        if name not in candidates:
            candidates.append(name)
    return candidates

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not installed, vector cache disabled")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed, vector cache disabled")

# SentenceTransformer is an optional dependency
# Detect availability WITHOUT loading the heavy sentence_transformers/transformers/torch stack (~4s).
# The actual import is deferred to _ensure_st_imported(), called from get_global_embedding_model() etc.
import importlib.util as _importlib_util

HAS_SENTENCE_TRANSFORMERS = _importlib_util.find_spec("sentence_transformers") is not None
if not HAS_SENTENCE_TRANSFORMERS:
    logger.warning("SentenceTransformers not installed, vector cache disabled")
# Module-level attribute kept for patch() compatibility in tests.
# Replaced with the real class on first lazy import via _ensure_st_imported().
SentenceTransformer = None


def _ensure_st_imported() -> None:
    """Lazy import of sentence_transformers (first call ~4s, subsequent calls no-op).

    **Caller must hold** ``_embedding_model_lock`` — the global ``SentenceTransformer``
    and ``HAS_SENTENCE_TRANSFORMERS`` writes are NOT internally synchronized.
    """
    global HAS_SENTENCE_TRANSFORMERS, SentenceTransformer
    if SentenceTransformer is not None:
        return
    if not HAS_SENTENCE_TRANSFORMERS:
        return
    try:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
    except ImportError:
        HAS_SENTENCE_TRANSFORMERS = False
        logger.warning("SentenceTransformers not installed, vector cache disabled")

# Global embedding model singleton
_global_embedding_model: "Optional[SentenceTransformer]" = None
_embedding_model_lock = threading.Lock()
_embedding_model_dimension: int = 384  # Multilingual MiniLM-L12-v2 is also 384-d
_loaded_embedding_model_name: "Optional[str]" = None


def _read_embedding_dimension(model: "SentenceTransformer", fallback: int = 384) -> int:
    """Best-effort embedding dimension, tolerant of SentenceTransformer API drift.

    Prefer the current ``get_embedding_dimension`` name; ``get_sentence_embedding_dimension``
    is a deprecated alias that emits a FutureWarning on newer versions.
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            try:
                dim = getter()
                if dim:
                    return int(dim)
            except Exception:
                logger.debug(
                    "_read_embedding_dimension: %s() failed", attr, exc_info=True
                )
    return fallback


def get_global_embedding_model() -> "Optional[SentenceTransformer]":
    """Get or create global SentenceTransformer instance."""
    global _global_embedding_model, _embedding_model_dimension, _loaded_embedding_model_name

    if _global_embedding_model is not None:
        return _global_embedding_model

    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    with _embedding_model_lock:
        # Double-check after acquiring lock
        if _global_embedding_model is not None:
            return _global_embedding_model

        _ensure_st_imported()
        candidates = _embedding_model_candidates()
        for i, model_name in enumerate(candidates):
            try:
                logger.info(f"Loading SentenceTransformer model {model_name!r}...")
                with _suppress_hf_progress():
                    model = SentenceTransformer(model_name)
                _global_embedding_model = model
                _embedding_model_dimension = _read_embedding_dimension(model)
                _loaded_embedding_model_name = model_name
                if i > 0:
                    logger.warning(
                        "Embedding model %r unavailable; fell back to %r. "
                        "Run online once to fetch the preferred model.",
                        candidates[0], model_name,
                    )
                logger.info(f"Model loaded with dimension {_embedding_model_dimension}")
                return _global_embedding_model
            except Exception as e:
                # Not fatal yet if a fallback remains — log softly and try the next.
                level = logging.ERROR if i == len(candidates) - 1 else logging.WARNING
                logger.log(level, "Failed to load embedding model %r: %s", model_name, e)

        logger.error("No embedding model could be loaded; semantic features disabled.")
        return None


def set_active_embedding_model(model_name: str) -> "Optional[SentenceTransformer]":
    """Force-load a specific model and install it as the global singleton.

    Bypasses the preferred→fallback candidate order of ``get_global_embedding_model``.
    Used after an explicit, user-approved download so we activate *exactly* what
    was fetched — e.g. if the user declined the multilingual default and chose the
    lighter fallback, loading must not silently re-fetch the preferred model.
    Returns the loaded model, or None on failure / when deps are missing.
    """
    global _global_embedding_model, _embedding_model_dimension, _loaded_embedding_model_name

    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    with _embedding_model_lock:
        _ensure_st_imported()
        try:
            logger.info(f"Loading SentenceTransformer model {model_name!r}...")
            with _suppress_hf_progress():
                model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error("Failed to load embedding model %r: %s", model_name, e)
            return None
        _global_embedding_model = model
        _embedding_model_dimension = _read_embedding_dimension(model)
        _loaded_embedding_model_name = model_name
        return model


def get_global_embedding_dimension() -> int:
    """Get the dimension of the global embedding model."""
    return _embedding_model_dimension


def get_loaded_embedding_model_name() -> Optional[str]:
    """Name of the model actually loaded, or None if not yet loaded."""
    return _loaded_embedding_model_name


def reset_global_embedding_model():
    """Reset global embedding model (for testing)."""
    global _global_embedding_model, _embedding_model_dimension, _loaded_embedding_model_name
    with _embedding_model_lock:
        _global_embedding_model = None
        _embedding_model_dimension = 384
        _loaded_embedding_model_name = None


def warmup_embedding_model() -> None:
    """Best-effort, non-blocking pre-load of the global embedding model.

    Drives :func:`get_global_embedding_model`, which loads the model under the
    existing ``_embedding_model_lock`` with a double-check. Safe to call from a
    background thread: a concurrent first real caller (e.g. ``RAGSearcher``
    during ``ToolRegistry`` construction) will block on the lock until the
    warmup finishes, then reuse the *same* singleton instance — never loading
    twice. Latency is therefore never worse than without warmup, and usually
    better (the load overlaps with other startup work on the main thread).

    No-op when deps are missing or the model is already loaded, so a background
    caller may invoke this unconditionally. Whether a *network* fetch is
    permitted is a policy decision left to the caller — this primitive just
    triggers the loader; guard it yourself if you must avoid network access.

    All exceptions are swallowed (DEBUG-logged): a failed warmup must never
    crash its thread, and the real call path will surface genuine errors.

    INFO-level logs from our own logger and from sentence_transformers are
    temporarily suppressed so the background thread doesn't disrupt the REPL
    prompt.  The same messages still appear verbatim when a real, user-driven
    call path triggers the model load (e.g. the first RAG query).


    """
    if not (HAS_FAISS and HAS_NUMPY and HAS_SENTENCE_TRANSFORMERS):
        return
    if _global_embedding_model is not None:
        return
    # Temporarily suppress INFO chatter during background load —
    # the REPL prompt is already visible and background noise is distracting.
    _st_logger = logging.getLogger("sentence_transformers")
    _old_st = _st_logger.level
    _st_logger.setLevel(logging.WARNING)
    # Also silence the sentence_transformers.base.model logger directly
    # (transformers' logging adapter may set its own level, bypassing the parent).
    _st_model_logger = logging.getLogger("sentence_transformers.base.model")
    _old_stm = _st_model_logger.level
    _st_model_logger.setLevel(logging.WARNING)
    _old_vc = logger.level
    logger.setLevel(logging.WARNING)
    try:
        get_global_embedding_model()
    except Exception as e:
        logger.debug("embedding model warmup failed: %s", e)
    finally:
        _st_logger.setLevel(_old_st)
        _st_model_logger.setLevel(_old_stm)
        logger.setLevel(_old_vc)


class VectorCacheManager:
    """Manages FAISS-based vector cache for semantic search."""

    def __init__(self, cache_dir: str, dimension: int = 384):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.index_path = self.cache_dir / "faiss_index.bin"
        self.metadata_path = self.cache_dir / "metadata.json"
        # Records which embedding model produced the persisted vectors. A cache
        # built with a different model lives in a different semantic space and
        # must not be reused — see _load_or_create_index.
        self.model_marker_path = self.cache_dir / "embedding_model.txt"

        # Lazy: model is loaded on first _ensure_model_loaded() call,
        # not at construction time. This avoids ~2-4s blocking during
        # ToolRegistry creation at REPL startup.
        self.embedding_model = None

        # Tie the cache identity to the model that actually loaded (which may be
        # a fallback), not the one we asked for — the vectors are produced by the
        # loaded model. Falls back to the configured name when nothing loaded.
        self.model_name = get_loaded_embedding_model_name() or get_configured_embedding_model_name()

        # Load or create FAISS index
        self.index, self.id_to_doc = self._load_or_create_index()

        # Reverse lookup: doc_id → row index, so add_document's duplicate check
        # is O(1) instead of an O(n) linear scan of id_to_doc.values(). Built
        # once here from the loaded metadata and kept in sync by add_document /
        # clear (the only two sites that mutate id_to_doc).
        self._doc_id_to_idx: dict[str, int] = {
            doc["doc_id"]: idx
            for idx, doc in self.id_to_doc.items()
            if isinstance(doc, dict) and "doc_id" in doc
        }

        # Statistics
        self.hit_count = 0
        self.miss_count = 0

        # Dirty flag: set True whenever the in-memory index/metadata mutates
        # (add_document / clear), cleared by a successful _save_index(). This
        # lets __del__ skip the O(n) full dump when nothing changed since the
        # last save — e.g. exiting after only searches (no adds), or right
        # after a 100-doc checkpoint save.
        #
        # Starts False: _load_or_create_index only returns a populated index
        # when it was read cleanly from disk (so in-memory == on-disk), and a
        # freshly created empty index has nothing to write. If the on-disk
        # cache was stale/mismatched, the load path already discarded it and
        # the next startup rebuilds — so a clean exit with no adds never needs
        # to re-dump.
        self._dirty = False

    def _cached_model_matches(self) -> bool:
        """True if the persisted cache was built with the current embedding model.

        A missing marker is treated as a mismatch: legacy caches (pre-marker,
        built with the old English model) are discarded rather than silently
        reused in a new model's semantic space.
        """
        try:
            stored = self.model_marker_path.read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return False
        return stored == self.model_name

    def _load_or_create_index(self) -> tuple[Optional[Any], dict[int, dict]]:
        """Load existing FAISS index and metadata, or create new ones.

        Existing vectors are reused only when they were produced by the current
        embedding model; otherwise the index is rebuilt from scratch.
        """
        if not HAS_NUMPY or not HAS_FAISS:
            return None, {}

        if self.index_path.exists() and self.metadata_path.exists():
            if not self._cached_model_matches():
                logger.info(
                    "Vector cache was built with a different embedding model; "
                    "rebuilding for %s", self.model_name,
                )
            else:
                try:
                    index = faiss.read_index(str(self.index_path))
                    with open(self.metadata_path, encoding="utf-8") as f:
                        _raw = json.load(f)
                    # JSON object keys are strings; restore the int row indices
                    # expected by id_to_doc[idx] lookups (idx is numpy.int64).
                    id_to_doc = {int(k): v for k, v in _raw.items()}
                    # The FAISS index (row count) and metadata (id_to_doc) are
                    # persisted as two separate files; a torn write or an earlier
                    # add desync can leave the index with rows that have no
                    # metadata key. Such rows raise KeyError on every search, so
                    # discard the pair and rebuild rather than load a broken cache.
                    if index.ntotal != len(id_to_doc):
                        logger.warning(
                            "Vector cache index/metadata mismatch "
                            "(index=%d rows, metadata=%d entries); rebuilding",
                            index.ntotal, len(id_to_doc),
                        )
                    else:
                        logger.info(f"Loaded vector cache with {index.ntotal} documents")
                        return index, id_to_doc
                except Exception as e:
                    logger.warning(f"Failed to load vector cache: {e}; removing stale files")
                    # Discard both files so on-disk state matches the fresh empty
                    # in-memory state that follows — no stale metadata lingers.
                    for _p in (self.index_path, self.metadata_path):
                        try:
                            _p.unlink(missing_ok=True)
                        except OSError:
                            pass

        # Create new index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        id_to_doc = {}
        return index, id_to_doc

    def _write_model_marker(self):
        """Persist the model name alongside the index for invalidation checks."""
        try:
            self.model_marker_path.write_text(self.model_name, encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to write embedding model marker: {e}")

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        if not HAS_NUMPY or not HAS_FAISS or self.index is None:
            return
        try:
            faiss.write_index(self.index, str(self.index_path))
            atomic_write_json(self.metadata_path, self.id_to_doc, indent=None, ensure_ascii=True)
            self._write_model_marker()
            # Persisted state now matches in-memory state.
            self._dirty = False
        except Exception as e:
            logger.warning(f"Failed to save vector cache: {e}")

    def _ensure_model_loaded(self) -> None:
        """Load the embedding model on first use (lazy init)."""
        if self.embedding_model is not None:
            return
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            return
        self.embedding_model = get_global_embedding_model()
        if self.embedding_model is not None:
            self.dimension = get_global_embedding_dimension()
            self.model_name = get_loaded_embedding_model_name() or self.model_name

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        self._ensure_model_loaded()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not available")
        # SentenceTransformer returns numpy array
        return self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def _get_doc_id(self, file_path: str, content: str) -> str:
        """Generate unique ID for a document."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{file_path}:{content_hash}"

    def add_document(self, file_path: str, content: str):
        """Add a document to the vector cache."""
        self._ensure_model_loaded()
        if not HAS_NUMPY or not HAS_FAISS or self.index is None or self.embedding_model is None:
            return

        try:
            # Check if document already exists
            doc_id = self._get_doc_id(file_path, content)
            if doc_id in self._doc_id_to_idx:
                logger.debug(f"Document already in cache: {file_path}")
                return

            # Compute embedding
            embedding = self._compute_embedding(content)
            embedding = embedding.reshape(1, -1).astype('float32')

            # Build metadata BEFORE touching the index so that a failure here
            # (e.g. np.linalg.norm) cannot leave the FAISS index with a row that
            # has no id_to_doc entry — an index/metadata desync that breaks every
            # subsequent search with a KeyError on the orphaned row.
            metadata = {
                'file_path': file_path,
                'content': content,
                'doc_id': doc_id,
                # Coerce numpy float32 scalar → Python float so the metadata
                # dict is JSON-serializable (metadata is persisted as JSON, not
                # pickle). Note: embedding_norm is not read back by any caller.
                'embedding_norm': float(np.linalg.norm(embedding)),
            }

            # Normalize so FAISS IndexFlatIP inner product = cosine similarity.
            # search() already normalizes the query vector (L370); the indexed
            # vectors must also be normalized for the math to work correctly.
            faiss.normalize_L2(embedding)

            # Add to index, then record metadata. If the index mutated but the
            # dict assignment never ran we'd desync; rebuilding metadata first
            # keeps the two-step commit as tight as possible.
            idx = self.index.ntotal
            self.index.add(embedding)
            self.id_to_doc[idx] = metadata
            # Keep the reverse lookup in sync with id_to_doc so the next
            # add_document's duplicate check stays O(1).
            self._doc_id_to_idx[doc_id] = idx

            # Save periodically (every 100 documents).
            # NOTE: idx is the row index BEFORE add() (i.e. ntotal-1 after add).
            # Using idx would save on the very first document (0 % 100 == 0);
            # idx+1 = post-add count, so we save at 100, 200, ... as intended.
            if (idx + 1) % 100 == 0:
                self._save_index()
            else:
                # Mutation happened but not yet at a checkpoint — mark dirty so
                # __del__ persists it on exit instead of losing the tail.
                self._dirty = True

            logger.debug(f"Added document to vector cache: {file_path} (idx={idx})")
        except Exception as e:
            logger.warning(f"Failed to add document to vector cache: {e}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for documents similar to query."""
        self._ensure_model_loaded()
        if not HAS_NUMPY or not HAS_FAISS or self.index is None or self.index.ntotal == 0 or self.embedding_model is None:
            self.miss_count += 1
            return []

        try:
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')

            # Normalize for cosine similarity (FAISS inner product expects normalized vectors)
            # We'll normalize both query and indexed vectors
            faiss.normalize_L2(query_embedding)

            # Search
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

            results = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx != -1:
                    # idx is numpy.int64 from FAISS. Use .get() so an orphaned
                    # row (index/metadata desync) is skipped rather than raising
                    # KeyError and failing the whole search.
                    doc = self.id_to_doc.get(int(idx))
                    if doc is None:
                        logger.warning(
                            "Vector cache row %s has no metadata; skipping", int(idx),
                        )
                        continue
                    # Convert inner product to cosine similarity (since vectors are normalized)
                    cosine_sim = max(0.0, min(1.0, float(dist)))
                    results.append({
                        "file_path": doc["file_path"],
                        "content": doc["content"],
                        "score": cosine_sim,
                        "from_cache": True
                    })

            if results:
                self.hit_count += 1
            else:
                self.miss_count += 1

            return results
        except Exception as e:
            logger.warning(f"Vector cache search failed: {e}")
            self.miss_count += 1
            return []

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def clear(self):
        """Clear the vector cache."""
        if HAS_NUMPY and HAS_FAISS and self.index is not None:
            self.index.reset()
            self.id_to_doc.clear()
            self._doc_id_to_idx.clear()
            self._save_index()
            logger.info("Vector cache cleared")

    def __del__(self):
        """Save index on destruction. Avoid logger calls (logging may already be shut down).

        Skipped when the in-memory state is clean (``_dirty == False``): a clean
        state means the last periodic save already wrote the exact same data, so
        re-dumping would be redundant O(n) I/O. The dirty flag is set by
        add_document/clear and cleared by a successful _save_index.
        """
        if not HAS_NUMPY or not HAS_FAISS or self.index is None:
            return
        if not getattr(self, "_dirty", False):
            return
        try:
            faiss.write_index(self.index, str(self.index_path))
            atomic_write_json(self.metadata_path, self.id_to_doc, indent=None, ensure_ascii=True)
            # Keep the model marker in sync so the next process doesn't discard
            # this index as stale.
            self.model_marker_path.write_text(self.model_name, encoding="utf-8")
        except Exception:
            pass
