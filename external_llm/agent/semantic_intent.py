"""
semantic_intent.py — Embedding-based intent matching for keyword-fallback paths.

Replaces hardcoded verb/noun keyword lists + substring matching with cosine
similarity over a small set of labeled example phrases. This solves the synonym
and multilingual gaps a fixed keyword list leaves open ("wipe out" / "없애" /
"purge"), per the CLAUDE.md design insight (keyword/regex → bm25/rag/embedding).

Positioning: this is a *fallback enhancer*, not a primary classifier. LLM
classification stays the main path, and keyword matching stays the fast first
pass. When both miss — and the embedding model is available — semantic
similarity provides a multilingual, synonym-tolerant backstop. When the
embedding model is absent (no sentence-transformers / numpy), every method
degrades to a no-op so callers keep their prior keyword-based decision.

Example phrases are encoded once (lazily, thread-safe) into an in-memory
L2-normalized matrix; classification is a single matrix-vector product. No FAISS
index or disk persistence is used — the example set is tiny and fixed, so the
VectorCacheManager machinery (built for large evolving document corpora) would
be overkill. The shared embedding-model singleton from vector_cache is reused.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .vector_cache import get_global_embedding_model

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None


class SemanticIntentMatcher:
    """Cosine-similarity classifier over a fixed set of labeled example phrases.

    Construct with ``{label: [example phrase, ...]}``, a ``threshold`` (floor on
    the winning label's score) and a ``margin`` (how far the winner must beat the
    runner-up label). ``classify`` scores the query by its *mean* cosine to each
    label's examples, then returns the top label when it clears both gates, else
    ``None``.

    Why mean-per-label + margin rather than nearest-example argmax: embedding
    models add a roughly constant offset to all cosine scores (anisotropy), and
    short look-alike phrases — e.g. a "create X" request sharing imperative
    surface form with a "remove X" example — can spike the single nearest row.
    Averaging over each label's examples suppresses that surface-form spike, and
    the runner-up margin cancels the constant offset, so the same gates transfer
    across models. Always provide contrastive negative examples (an ``"other"``
    label) so the margin has something to measure against.
    """

    def __init__(self, examples: dict[str, list[str]], threshold: float,
                 margin: float = 0.0, name: str = ""):
        self._examples = {label: list(phrases) for label, phrases in examples.items()}
        self._threshold = float(threshold)
        self._margin = float(margin)
        self._name = name or "matcher"
        self._lock = threading.Lock()
        self._built = False
        self._available = False
        self._model = None
        self._matrix = None  # (N, D) float32, L2-normalized rows
        self._label_rows: dict[str, Any] = {}  # label -> int index array into _matrix

    def _ensure_built(self) -> None:
        """Encode all example phrases once. Idempotent and thread-safe.

        Leaves ``_available`` False (so the matcher is a no-op) when numpy or the
        embedding model is unavailable, or when encoding fails.
        """
        if self._built:
            return
        with self._lock:
            if self._built:
                return
            self._built = True  # mark attempted even on failure — never retry-loop

            if np is None:
                logger.debug("semantic matcher '%s' disabled: numpy unavailable", self._name)
                return
            model = get_global_embedding_model()
            if model is None:
                logger.debug("semantic matcher '%s' disabled: no embedding model", self._name)
                return

            phrases: list[str] = []
            labels: list[str] = []
            for label, examples in self._examples.items():
                for phrase in examples:
                    if phrase and phrase.strip():
                        phrases.append(phrase)
                        labels.append(label)
            if not phrases:
                return

            try:
                matrix = model.encode(
                    phrases,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as e:
                logger.debug("semantic matcher '%s' build failed: %s", self._name, e)
                return

            self._model = model
            self._matrix = np.asarray(matrix, dtype="float32")
            # Precompute per-label row index arrays for mean aggregation.
            self._label_rows = {
                label: np.asarray([i for i, lab in enumerate(labels) if lab == label])
                for label in dict.fromkeys(labels)
            }
            self._available = True

    def classify(self, text: str) -> Optional[tuple[str, float]]:
        """Return ``(best_label, mean_cosine_score)`` if the top label clears the
        threshold and beats the runner-up label by the margin, else ``None``.

        ``None`` also covers empty input and an unavailable embedding model;
        callers treat it as "no semantic signal" and keep their prior decision.
        """
        if not text or not text.strip():
            return None
        self._ensure_built()
        if not self._available:
            return None
        try:
            query = self._model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Rows and query are L2-normalized, so the dot product is cosine sim.
            sims = self._matrix @ np.asarray(query, dtype="float32")[0]
            # Mean cosine per label, ranked high to low.
            ranked = sorted(
                ((label, float(sims[rows].mean())) for label, rows in self._label_rows.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            best_label, best_score = ranked[0]
            runner_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
            if best_score < self._threshold:
                return None
            if (best_score - runner_score) < self._margin:
                return None
            return best_label, best_score
        except Exception as e:
            logger.debug("semantic matcher '%s' classify failed: %s", self._name, e)
            return None

    def matches(self, text: str, label: str) -> bool:
        """True iff the nearest example for *text* carries *label* above threshold."""
        result = self.classify(text)
        return result is not None and result[0] == label
