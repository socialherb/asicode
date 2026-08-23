"""
Vector Cache for asicode Agent

FAISS-based embedding cache for semantic search, integrated with RAG searcher.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util as _importlib_util
import json
import logging
import os
import sys
import tempfile
import threading
import weakref
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy

from external_llm.common.atomic_io import atomic_write_json, atomic_write_text

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


# Detect availability WITHOUT loading the full numpy/faiss modules (~100ms combined, +175 imported modules).
# Actual imports are deferred to _ensure_np_imported() / _ensure_faiss_imported().

HAS_NUMPY = _importlib_util.find_spec("numpy") is not None
if not HAS_NUMPY:
    logger.warning("NumPy not installed, vector cache disabled")

HAS_FAISS = _importlib_util.find_spec("faiss") is not None
if not HAS_FAISS:
    logger.warning("FAISS not installed, vector cache disabled")

# Lazy module references — actual imports at first use
_np = None
_faiss = None

# SentenceTransformer is an optional dependency
# Detect availability WITHOUT loading the heavy sentence_transformers/transformers/torch stack (~4s).
# The actual import is deferred to _ensure_st_imported(), called from get_global_embedding_model() etc.
HAS_SENTENCE_TRANSFORMERS = _importlib_util.find_spec("sentence_transformers") is not None
if not HAS_SENTENCE_TRANSFORMERS:
    logger.warning("SentenceTransformers not installed, vector cache disabled")
# Module-level attribute kept for patch() compatibility in tests.
# Replaced with the real class on first lazy import via _ensure_st_imported().
SentenceTransformer = None


def _ensure_np_imported() -> None:
    """Lazy import of numpy (first call ~40ms, subsequent calls no-op)."""
    global _np
    if _np is not None:
        return
    if not HAS_NUMPY:
        return
    try:
        import numpy as _n

        _np = _n
    except ImportError:
        logger.debug("numpy import failed (HAS_NUMPY was stale)")


def _ensure_faiss_imported() -> None:
    """Lazy import of faiss (first call ~60ms, subsequent calls no-op)."""
    global _faiss
    if _faiss is not None:
        return
    if not HAS_FAISS:
        return
    try:
        import faiss as _f

        _faiss = _f
    except ImportError:
        logger.debug("faiss import failed (HAS_FAISS was stale)")


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
        from sentence_transformers import SentenceTransformer as _ST  # noqa: N814 — private lazy-import alias

        SentenceTransformer = _ST
    except ImportError:
        HAS_SENTENCE_TRANSFORMERS = False
        logger.warning("SentenceTransformers not installed, vector cache disabled")


# Global embedding model singleton
_global_embedding_model: Any | None = None
_embedding_model_lock = threading.Lock()
_embedding_model_dimension: int = 384  # Multilingual MiniLM-L12-v2 is also 384-d
_loaded_embedding_model_name: str | None = None


def _read_embedding_dimension(model: Any, fallback: int = 384) -> int:
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
                    return int(dim)  # type: ignore[arg-type]  # getattr result is object; API returns int/float
            except Exception:
                logger.debug("_read_embedding_dimension: %s() failed", attr, exc_info=True)
    return fallback


def get_global_embedding_model() -> Any | None:
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
        assert SentenceTransformer is not None  # HAS_SENTENCE_TRANSFORMERS gate guarantees import succeeded
        candidates = _embedding_model_candidates()
        for i, model_name in enumerate(candidates):
            try:
                logger.info("Loading SentenceTransformer model %r...", model_name)
                with _suppress_hf_progress():
                    model = SentenceTransformer(model_name)
                _global_embedding_model = model
                _embedding_model_dimension = _read_embedding_dimension(model)
                _loaded_embedding_model_name = model_name
                if i > 0:
                    logger.warning(
                        "Embedding model %r unavailable; fell back to %r. "
                        "Run online once to fetch the preferred model.",
                        candidates[0],
                        model_name,
                    )
                logger.info("Model loaded with dimension %s", _embedding_model_dimension)
            except Exception as e:
                # Not fatal yet if a fallback remains — log softly and try the next.
                level = logging.ERROR if i == len(candidates) - 1 else logging.WARNING
                logger.log(level, "Failed to load embedding model %r: %s", model_name, e)
            else:
                return _global_embedding_model

        logger.error("No embedding model could be loaded; semantic features disabled.")
        return None


def set_active_embedding_model(model_name: str) -> Any | None:
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
        assert SentenceTransformer is not None  # HAS_SENTENCE_TRANSFORMERS gate guarantees import succeeded
        try:
            logger.info("Loading SentenceTransformer model %r...", model_name)
            with _suppress_hf_progress():
                model = SentenceTransformer(model_name)
        except Exception as e:
            logger.exception("Failed to load embedding model %r: %s", model_name, e)
            return None
        _global_embedding_model = model
        _embedding_model_dimension = _read_embedding_dimension(model)
        _loaded_embedding_model_name = model_name
        return model


def get_global_embedding_dimension() -> int:
    """Get the dimension of the global embedding model."""
    return _embedding_model_dimension


def get_loaded_embedding_model_name() -> str | None:
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


# ── Exit-time flush ──────────────────────────────────────────────────────────
# __del__ cannot be relied on to persist the index. When the manager survives
# until the final GC pass, the interpreter is already finalizing: sys.modules
# has been purged and sys.meta_path is None, so the first lazy import inside
# the write path (os.fdopen -> `import io`) raises
#   ImportError: sys.meta_path is None, Python is likely shutting down
# CPython can only report that as "Exception ignored while calling
# deallocator", and the flush dies half-done — faiss_index.bin already
# rewritten, metadata.json not — which _load_or_create_index then rejects on
# the next start (index/metadata row-count mismatch) for a full re-embed.
#
# So flush from atexit, which runs while the import system is still intact
# (measured: is_finalizing() is False and sys.meta_path is populated inside an
# atexit hook; both are already gone by the time __del__ runs at shutdown).
# __del__ keeps handling only the live-interpreter case (manager dropped
# mid-session).
_live_managers: weakref.WeakSet[VectorCacheManager] = weakref.WeakSet()
_atexit_registered = False
_atexit_guard = threading.Lock()


def _flush_live_caches() -> None:
    """Persist every dirty live manager before the interpreter tears down."""
    for manager in list(_live_managers):
        try:
            manager._flush_if_dirty()
        except Exception as e:  # exit hook: never let one manager abort the rest
            logger.warning("Vector cache exit flush failed for one manager: %s", e)


def _register_exit_flush() -> None:
    """Register :func:`_flush_live_caches` once per process (idempotent)."""
    global _atexit_registered
    with _atexit_guard:
        if not _atexit_registered:
            _atexit_registered = True
            atexit.register(_flush_live_caches)


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

        # Lazy: FAISS index and metadata are loaded on first use (add_document,
        # search, or clear).  This avoids ~124 MB JSON parse + 561 MB RSS during
        # ToolRegistry construction at REPL startup when the cache may never be
        # queried (e.g. rag_enabled=False).
        self.index = None
        self.id_to_doc: dict[int, dict] = {}
        self._doc_id_to_idx: dict[str, int] = {}

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

        # Monotonic mutation counter: _save_index snapshots it under the lock
        # and clears _dirty only if it is unchanged when the disk I/O commits —
        # a mutation landing during the I/O must leave the flag set so the
        # next flush persists it.
        self._generation = 0

        # Serializes FAISS index mutation/read on THIS instance.  FAISS
        # add/search are not safe to interleave on one index: multi-tool
        # batches dispatch read tools (vector search) in parallel threads
        # while a write tool's RAG invalidation adds documents, and the
        # shared session VCM is used from concurrent search_design_history
        # calls.  RLock so nested acquisition (_save_index called from
        # add_document) is safe.  The embedding encode() itself is NOT held —
        # model inference must not block concurrent searches.
        self._io_lock = threading.RLock()

        # Persist from atexit rather than trusting __del__ to run early enough
        # to still have a working import system — see _flush_live_caches.
        _live_managers.add(self)
        _register_exit_flush()

    def _ensure_index_loaded(self) -> None:
        """Lazily load (or create) the FAISS index and metadata on first use.

        Safe to call multiple times — no-op after the first successful load.
        """
        if self.index is not None:
            return
        # Resolve the model BEFORE touching the index: _load_or_create_index
        # compares the persisted marker against self.model_name and builds
        # IndexFlatIP(self.dimension), and both values are only correct after
        # _ensure_model_loaded corrected them (an offline fallback changes the
        # name, a non-384 model changes the dimension). Resolving after the
        # load used to: discard a valid fallback-built cache on every session
        # (marker compared against the configured name → full re-embed loop),
        # mix two semantic spaces into one index (marker-matched cache reused
        # under a fallback session), and build a 384-dim index for non-384
        # models (every add silently failed).
        self._ensure_model_loaded()
        self.index, self.id_to_doc = self._load_or_create_index()
        # Rebuild reverse-lookup from loaded metadata.
        self._doc_id_to_idx = {
            doc["doc_id"]: idx for idx, doc in self.id_to_doc.items() if isinstance(doc, dict) and "doc_id" in doc
        }

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

    def _load_or_create_index(self) -> tuple[Any | None, dict[int, dict]]:
        """Load existing FAISS index and metadata, or create new ones.

        Existing vectors are reused only when they were produced by the current
        embedding model; otherwise the index is rebuilt from scratch.
        """
        if not HAS_NUMPY or not HAS_FAISS:
            return None, {}
        _ensure_faiss_imported()

        if self.index_path.exists() and self.metadata_path.exists():
            if not self._cached_model_matches():
                logger.info(
                    "Vector cache was built with a different embedding model; rebuilding for %s",
                    self.model_name,
                )
            else:
                try:
                    assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
                    index = _faiss.read_index(str(self.index_path))
                    if index.d != self.dimension:
                        # The marker can match by name while the persisted
                        # vectors were built at a different width (model-name
                        # collision, or an index written by an older build).
                        # A width mismatch cannot be tail-recovered — every
                        # search would be garbage — so treat the pair as stale:
                        # the except below removes both files and a fresh index
                        # is built at the current model's width.
                        logger.warning(
                            "Vector cache dimension %d != model dimension %d; rebuilding",
                            index.d,
                            self.dimension,
                        )
                        raise ValueError("dimension mismatch")
                    with open(self.metadata_path, encoding="utf-8") as f:
                        _raw = json.load(f)
                    # JSON object keys are strings; restore the int row indices
                    # expected by id_to_doc[idx] lookups (idx is numpy.int64).
                    id_to_doc = {int(k): v for k, v in _raw.items()}
                    # The FAISS index (row count) and metadata (id_to_doc) are
                    # persisted as two separate files; a torn write or an earlier
                    # add desync can leave the index with rows that have no
                    # metadata key. Such rows raise KeyError on every search.
                    #
                    # index > metadata is ALWAYS a tail overflow: metadata keys
                    # are the gapless row sequence 0..n-1 (append-only), so a
                    # save that died after committing the index but before the
                    # metadata write leaves orphan rows at the end. Drop just
                    # the tail and keep the rest — the alternative (discarding
                    # the pair) re-embeds the entire corpus for what is, in the
                    # wild, a handful of rows (observed: 33 orphans costing
                    # 7700 re-embeddings). The lost tail is bounded by the
                    # checkpoint cadence, same as any crash.
                    _recovered = False
                    if index.ntotal > len(id_to_doc) and id_to_doc and set(id_to_doc) == set(range(len(id_to_doc))):
                        _ensure_np_imported()
                        _before = index.ntotal
                        _orphan = _before - len(id_to_doc)
                        assert _np is not None  # HAS_NUMPY gate guarantees import succeeded
                        index.remove_ids(_np.arange(len(id_to_doc), _before))
                        # Persist the trimmed state on the next save.
                        self._dirty = True
                        _recovered = True
                        logger.warning(
                            "Vector cache: dropped %d orphan tail rows "
                            "(index=%d, metadata=%d); recovered without re-embedding",
                            _orphan,
                            _before,
                            len(id_to_doc),
                        )
                    if not _recovered and index.ntotal != len(id_to_doc):
                        logger.warning(
                            "Vector cache index/metadata mismatch (index=%d rows, metadata=%d entries); rebuilding",
                            index.ntotal,
                            len(id_to_doc),
                        )
                    else:
                        # ── Legacy-entry migration: drop persisted 'content' ──
                        # Entries written before content was removed from the
                        # metadata carry a full copy of each file (~35 KB x 3359
                        # ≈ 117 MB of a 124 MB metadata.json — 96% of it). Simply
                        # not writing content for NEW entries leaves those alone
                        # forever: the load path would read them back and
                        # _save_index dumps id_to_doc verbatim, re-persisting the
                        # bulk on every save. So strip on load, and mark dirty so
                        # the next save rewrites the file WITHOUT content — the
                        # 124 MB shrinks to ~5 MB once, permanently, instead of
                        # every session paying a 0.33 s parse and ~500 MB of RSS
                        # to reconstruct data nothing reads.
                        _stripped = 0
                        for _doc in id_to_doc.values():
                            if isinstance(_doc, dict) and "content" in _doc:
                                del _doc["content"]
                                _stripped += 1
                        if _stripped:
                            self._dirty = True
                            logger.info(
                                "Vector cache: dropped legacy 'content' from %d entries; "
                                "metadata will be rewritten without it on next save",
                                _stripped,
                            )
                        logger.info("Loaded vector cache with %s documents", index.ntotal)
                        return index, id_to_doc
                except Exception as e:
                    logger.warning("Failed to load vector cache: %s; removing stale files", e)
                    # Discard both files so on-disk state matches the fresh empty
                    # in-memory state that follows — no stale metadata lingers.
                    for _p in (self.index_path, self.metadata_path):
                        with suppress(OSError):
                            _p.unlink(missing_ok=True)

        # Create new index
        assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
        index = _faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        id_to_doc = {}
        return index, id_to_doc

    def _write_model_marker(self):
        """Persist the model name alongside the index for invalidation checks."""
        try:
            # Atomic: a truncated/empty marker would read as a model MISMATCH
            # on the next start and discard an otherwise-valid index for a
            # full re-embed.
            atomic_write_text(self.model_marker_path, self.model_name)
        except OSError as e:
            logger.warning("Failed to write embedding model marker: %s", e)

    def _save_index(self):
        """Save FAISS index and metadata to disk.

        The index is staged in a sibling ``.atomic_*`` temp file (the same
        prefix :func:`external_llm.common.atomic_io.sweep_stale_temp_files`
        reclaims after a kill -9), then committed with ONE ``os.replace``, and
        only then is the metadata written atomically.  POSIX cannot commit two
        files as one unit, but this shrinks the vulnerable window between the
        pair from two multi-MB writes (tens of ms) to a single rename
        (microseconds): a crash mid-save leaves the PREVIOUS consistent pair
        on disk (which the load path accepts) instead of a truncated index;
        a crash between the replace and the metadata write leaves index >
        metadata, which ``_load_or_create_index`` recovers by dropping the
        orphan tail.

        The index is serialized and the metadata is snapshotted under the lock
        (fast, in-memory), but the disk I/O itself runs OUTSIDE it: a multi-MB
        write must not block concurrent searches. The dirty flag is cleared
        only if the mutation counter (``_generation``) is unchanged when the
        I/O commits — a mutation landing during the I/O keeps the flag set so
        a subsequent flush persists it.
        """
        if not HAS_NUMPY or not HAS_FAISS or self.index is None:
            return
        try:
            with self._io_lock:
                _ensure_faiss_imported()
                assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
                _blob = _faiss.serialize_index(self.index)
                _meta = dict(self.id_to_doc)
                _gen = self._generation
        except Exception as e:
            logger.warning("Failed to serialize vector cache: %s", e)
            return
        try:
            _fd, _tmp = tempfile.mkstemp(dir=str(self.index_path.parent), prefix=".atomic_", suffix=".tmp")
            os.close(_fd)
        except OSError as e:
            # The cache dir vanished (rm -rf .asicode, pytest tmp teardown):
            # _flush_if_dirty's is_dir() guard is a TOCTOU, not a guarantee.
            # Log and skip — the save is best-effort by contract.
            logger.warning("Failed to stage vector cache temp file: %s", e)
            return
        try:
            with open(_tmp, "wb") as _fh:
                _fh.write(_blob)
            # fsync the staged file BEFORE the rename: os.replace only
            # guarantees rename atomicity, not data durability. Without
            # this, a power loss can commit "new name + old/empty data" —
            # and the metadata leg already fsyncs (atomic_io), so the
            # pair's durability would be asymmetric.
            _sfd = os.open(_tmp, os.O_RDONLY)
            try:
                os.fsync(_sfd)
            finally:
                os.close(_sfd)
            os.replace(_tmp, self.index_path)
            atomic_write_json(self.metadata_path, _meta, indent=None, ensure_ascii=True)
            self._write_model_marker()
        except Exception as e:
            with suppress(OSError):
                os.unlink(_tmp)
            logger.warning("Failed to save vector cache: %s", e)
            return
        # Commit: clear the dirty flag only if nothing mutated while the disk
        # I/O ran — otherwise the later mutation's flag survives and the next
        # flush persists it (no lost tail).
        with self._io_lock:
            if self._generation == _gen:
                self._dirty = False

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

    def _compute_embedding(self, text: str) -> numpy.ndarray:
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
        with self._io_lock:
            self._ensure_index_loaded()
            self._ensure_model_loaded()
            if not HAS_NUMPY or not HAS_FAISS or self.index is None or self.embedding_model is None:
                return

            _ensure_np_imported()
            _ensure_faiss_imported()

            # Check if document already exists (under the lock: the reverse
            # lookup is mutated by concurrent adds).
            doc_id = self._get_doc_id(file_path, content)
            if doc_id in self._doc_id_to_idx:
                logger.debug("Document already in cache: %s", file_path)
                return

        # Compute embedding OUTSIDE the lock: encode() is model inference and
        # must not block concurrent searches.
        try:
            embedding = self._compute_embedding(content)
            embedding = embedding.reshape(1, -1).astype("float32")

            with self._io_lock:
                # Re-check under the lock: a concurrent add may have inserted
                # this doc while we were embedding.
                if doc_id in self._doc_id_to_idx:
                    return

                # Build metadata BEFORE touching the index so that a failure here
                # (e.g. np.linalg.norm) cannot leave the FAISS index with a row that
                # has no id_to_doc entry — an index/metadata desync that breaks every
                # subsequent search with a KeyError on the orphaned row.
                assert _np is not None  # HAS_NUMPY gate guarantees import succeeded
                metadata = {
                    "file_path": file_path,
                    # Intentionally NOT storing full content: content is the ~96%
                    # of metadata.json size (35 KB per entry x 3359 files ≈ 117 MB),
                    # and snippets are extracted from the BM25 _doc_texts in the
                    # RAGSearcher layer.  The BM25 index and vector cache index are
                    # built from the same documents so _doc_texts is always
                    # populated; vector cache search falls back to "" for the rare
                    # desync case, which _vector_search handles gracefully (empty
                    # snippet, re-read from disk).
                    "doc_id": doc_id,
                    # Coerce numpy float32 scalar → Python float so the metadata
                    # dict is JSON-serializable (metadata is persisted as JSON, not
                    # pickle). Note: embedding_norm is not read back by any caller.
                    "embedding_norm": float(_np.linalg.norm(embedding)),
                }

                # Normalize so FAISS IndexFlatIP inner product = cosine similarity.
                # search() already normalizes the query vector; the indexed
                # vectors must also be normalized for the math to work correctly.
                assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
                _faiss.normalize_L2(embedding)

                # Add to index, then record metadata. If the index mutated but the
                # dict assignment never ran we'd desync; rebuilding metadata first
                # keeps the two-step commit as tight as possible.
                idx = self.index.ntotal
                self.index.add(embedding)
                self.id_to_doc[idx] = metadata
                # Keep the reverse lookup in sync with id_to_doc so the next
                # add_document's duplicate check stays O(1).
                self._doc_id_to_idx[doc_id] = idx
                self._generation += 1

                # Save periodically (every 100 documents).
                # NOTE: idx is the row index BEFORE add() (i.e. ntotal-1 after add).
                # Using idx would save on the very first document (0 % 100 == 0);
                # idx+1 = post-add count, so we save at 100, 200, ... as intended.
                # Mark dirty BEFORE the checkpoint attempt: _save_index clears
                # the flag only on SUCCESS, so a failed checkpoint save must
                # leave the tail dirty for the exit flush to retry. (The flag
                # used to be set only in the non-checkpoint branch — a failed
                # checkpoint save left _dirty False and the entire
                # un-persisted tail was silently dropped.)
                self._dirty = True
                _checkpoint = (idx + 1) % 100 == 0
            if _checkpoint:
                # Save OUTSIDE the lock: the multi-MB disk I/O must not block
                # concurrent searches.
                self._save_index()
            logger.debug("Added document to vector cache: %s (idx=%s)", file_path, idx)
        except Exception as e:
            logger.warning("Failed to add document to vector cache: %s", e)

    def add_documents(self, items: list[tuple[str, str]]):
        """Add multiple documents in a single batched encode pass.

        Mirrors ``add_document`` semantics (dedup by doc_id, cosine-normalised,
        checkpoint every 100 docs) but computes embeddings for all *new*
        documents in one ``encode([...])`` call. SentenceTransformer's batched
        encode amortises per-call Python/torch overhead across the list, which
        on a cold-start RAG build (hundreds of files) is 1.6x-2.2x faster than
        per-file ``add_document``. Callers that only have one document should
        still use ``add_document``.
        """
        if not items:
            return
        with self._io_lock:
            self._ensure_index_loaded()
            self._ensure_model_loaded()
            if not HAS_NUMPY or not HAS_FAISS or self.index is None or self.embedding_model is None:
                return

            _ensure_np_imported()
            _ensure_faiss_imported()

            # Dedup against existing entries (O(1) reverse lookup) AND against
            # items already accepted earlier in this same batch. _doc_id_to_idx
            # is only updated AFTER the loop, so without the in-batch ``seen``
            # set a duplicate (file_path, content) pair appearing twice in one
            # call would pass the check twice: two identical FAISS rows, the
            # reverse lookup overwriting to keep only the last idx (orphan first
            # row), and duplicate documents surfacing in search results.
            new_items: list[tuple[str, str, str]] = []  # (doc_id, file_path, content)
            seen: set[str] = set()
            for file_path, content in items:
                doc_id = self._get_doc_id(file_path, content)
                if doc_id in self._doc_id_to_idx or doc_id in seen:
                    logger.debug("Document already in cache: %s", file_path)
                    continue
                seen.add(doc_id)
                new_items.append((doc_id, file_path, content))
            if not new_items:
                return

        # Batched embedding OUTSIDE the lock: encode() is model inference and
        # must not block concurrent searches.
        try:
            contents = [c for (_did, _fp, c) in new_items]
            embeddings = self.embedding_model.encode(contents, convert_to_numpy=True, show_progress_bar=False)
            assert _np is not None  # HAS_NUMPY gate guarantees import succeeded
            embeddings = _np.asarray(embeddings, dtype="float32")
            # encode() may return 1-D for a single-item list; normalise to 2-D.
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            with self._io_lock:
                # Re-filter under the lock: a concurrent add may have inserted
                # some of these docs while we were embedding.  Drop them now
                # (their embeddings were wasted, but dedup stays strict).
                kept = [i for i, (doc_id, _fp, _c) in enumerate(new_items) if doc_id not in self._doc_id_to_idx]
                if not kept:
                    return
                kept_items = [new_items[i] for i in kept]
                kept_embeddings = embeddings[kept]

                # Capture pre-normalization norms (matches add_document: the norm is
                # read before normalize_L2). embedding_norm is not read back by any
                # caller, but we keep the value consistent to avoid silent drift.
                assert _np is not None  # HAS_NUMPY gate guarantees import succeeded
                norms = _np.linalg.norm(kept_embeddings, axis=1)
                assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
                _faiss.normalize_L2(kept_embeddings)

                # Build ALL metadata BEFORE touching the index so a failure here
                # cannot leave FAISS with rows that have no id_to_doc entry (the
                # index/metadata desync that breaks every subsequent search).
                metadatas = []
                for (doc_id, file_path, _content), pre_norm in zip(kept_items, norms, strict=True):
                    metadatas.append(
                        {
                            "file_path": file_path,
                            "doc_id": doc_id,
                            "embedding_norm": float(pre_norm),
                        }
                    )

                # Add the whole batch in one FAISS call (cheaper than N single-row
                # adds), then record metadata + reverse lookups in lockstep.
                base = self.index.ntotal
                self.index.add(kept_embeddings)
                for offset, (doc_id, _fp, _content) in enumerate(kept_items):
                    idx = base + offset
                    self.id_to_doc[idx] = metadatas[offset]
                    self._doc_id_to_idx[doc_id] = idx
                self._generation += 1

                # Checkpoint if this batch crossed a 100-doc boundary, mirroring
                # add_document's every-100 cadence. ``base`` is the pre-add ntotal;
                # the whole batch already committed atomically to the index in one
                # add() call, so a single save here captures every row up to the
                # current ntotal. Loss-on-crash is bounded to <100 docs since the
                # last boundary crossing — the same invariant as per-file add_document.
                #
                # NOTE: do NOT use ``ntotal % 100 == 0`` here. A cold-start build
                # flushes the entire file walk as ONE batch (rag_searcher calls
                # add_documents once), so ntotal is just the file count — almost
                # never an exact multiple of 100. That left the whole (expensive)
                # encode relying solely on __del__, which never runs under
                # SIGKILL/OOM/os._exit, silently discarding the entire batch and
                # forcing a full re-encode on the next cold start.
                # Dirty BEFORE the checkpoint: same contract as add_document —
                # a failed save must leave the tail dirty for the exit flush.
                self._dirty = True
                _checkpoint = (self.index.ntotal // 100) > (base // 100)
            if _checkpoint:
                # Save OUTSIDE the lock: the multi-MB disk I/O must not block
                # concurrent searches.
                self._save_index()
            logger.debug("Batch-added %s documents to vector cache (ntotal=%s)", len(kept_items), self.index.ntotal)
        except Exception as e:
            logger.warning("Failed to batch-add documents to vector cache: %s", e)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for documents similar to query."""
        with self._io_lock:
            self._ensure_index_loaded()
            self._ensure_model_loaded()
            if (
                not HAS_NUMPY
                or not HAS_FAISS
                or self.index is None
                or self.index.ntotal == 0
                or self.embedding_model is None
            ):
                return []
            # Every other _faiss user guards itself; search() reaches _faiss only
            # when self.index is already set, which today only _load_or_create_index
            # does (and it imports faiss). Guard anyway — the except below turns a
            # None _faiss into a silent empty result, not a crash.
            _ensure_faiss_imported()

        try:
            # Compute query embedding (outside the lock: model inference must
            # not block concurrent adds).
            query_embedding = self._compute_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype("float32")

            with self._io_lock:
                # Normalize for cosine similarity (FAISS inner product expects normalized vectors)
                # We'll normalize both query and indexed vectors
                assert _faiss is not None  # HAS_FAISS gate guarantees import succeeded
                _faiss.normalize_L2(query_embedding)

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
                                "Vector cache row %s has no metadata; skipping",
                                int(idx),
                            )
                            continue
                        # Convert inner product to cosine similarity (since vectors are normalized)
                        cosine_sim = max(0.0, min(1.0, float(dist)))
                        results.append(
                            {
                                "file_path": doc["file_path"],
                                "content": "",  # content stored in BM25 _doc_texts, not here
                                "score": cosine_sim,
                                "from_cache": True,
                            }
                        )
                return results
        except Exception as e:
            logger.warning("Vector cache search failed: %s", e)
            return []

    def clear(self):
        """Clear the vector cache."""
        with self._io_lock:
            self._ensure_index_loaded()
            if HAS_NUMPY and HAS_FAISS and self.index is not None:
                self.index.reset()
                self.id_to_doc.clear()
                self._doc_id_to_idx.clear()
                self._generation += 1
                # Dirty BEFORE the save attempt: _save_index clears the flag
                # only on success, so a failed save leaves it set and the exit
                # flush retries — otherwise the old disk rows resurrect on the
                # next session.
                self._dirty = True
                self._save_index()
                logger.info("Vector cache cleared")

    def _flush_if_dirty(self) -> None:
        """Persist un-saved state — shared by ``__del__`` and the atexit hook.

        Skipped when the in-memory state is clean (``_dirty == False``): a clean
        state means the last periodic save already wrote the exact same data, so
        re-dumping would be redundant O(n) I/O. The dirty flag is set by
        add_document/clear and cleared by a successful _save_index.

        Also skipped when the cache dir has disappeared under us (``rm -rf
        .asicode``, a pytest ``tmp_path`` teardown): that cache is not ours to
        recreate, and attempting it only logs a failed faiss write per dead
        manager.
        """
        if not getattr(self, "_dirty", False):
            return
        if not self.cache_dir.is_dir():
            return
        self._save_index()  # already best-effort: logs and swallows

    def __del__(self):
        """Save index when the manager is dropped mid-session.

        Skipped during interpreter finalization, where the write path cannot
        run at all (the import system is gone) and a raised exception only
        surfaces as unraisable stderr noise — the atexit hook
        (:func:`_flush_live_caches`) has already flushed by then.
        """
        # ``sys.is_finalizing`` exists on every supported Python (added 3.5;
        # this module's floor is 3.10), but a deallocator must never raise:
        # sandboxed or stubbed ``sys`` modules (embedded/restricted
        # interpreters, packaging tools) may omit it. Treat "function absent"
        # as "not finalizing" so the mid-session flush still runs — the
        # ``suppress`` below must never be what saves us.
        if getattr(sys, "is_finalizing", lambda: False)():
            return
        # Best-effort disk I/O in a deallocator: anything escaping here becomes
        # an "Exception ignored" traceback, so swallow more than OSError
        # (faiss surfaces disk failures as RuntimeError; a partially
        # constructed object can raise AttributeError).
        with suppress(OSError, ValueError, TypeError, AttributeError, RuntimeError):
            self._flush_if_dirty()
