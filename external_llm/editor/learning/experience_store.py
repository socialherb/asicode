"""
Experience store: records and retrieves execution experiences with problem signatures.

Builds on InMemoryRunStore for persistence. Adds problem signature matching
and structured experience records.
"""
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.common.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

MAX_EXPERIENCES = 200
SIMILARITY_THRESHOLD = 0.3

# Reciprocal Rank Fusion parameters (same algorithm as rag_searcher._merge_results).
# k dampens the rank influence; SEMANTIC_LIMIT bounds the fusion candidate set.
_RRF_K = 60
_SEMANTIC_LIMIT = 30


@dataclass
class ExperienceRecord:
    """A single execution experience."""
    problem_signature: dict[str, Any] = field(default_factory=dict)
    strategy_used: str = ""
    success: bool = False
    attempts: int = 1
    failure_type: str = ""
    risk_level: str = "unknown"
    composite_risk_score: int = 0
    graph_confidence: float = 0.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # P8: alignment-enriched fields
    alignment_score: float = -1.0       # -1 = not computed; 0~1 = P7 result
    termination_decision: str = ""      # SUCCESS | PARTIAL | STOP | REPAIR_REQUIRED
    replan_count: int = 0
    repair_methods_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_signature": self.problem_signature,
            "strategy_used": self.strategy_used,
            "success": self.success,
            "attempts": self.attempts,
            "failure_type": self.failure_type,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
            "alignment_score": self.alignment_score,
            "termination_decision": self.termination_decision,
            "replan_count": self.replan_count,
            "repair_methods_used": self.repair_methods_used,
        }


class ExperienceStore:
    """
    Stores and retrieves execution experiences with problem signature matching.

    JSON-file backed, append-only with max size limit.
    Never raises — returns empty results on error.
    """

    def __init__(self, store_path: Optional[str] = None, max_experiences: int = MAX_EXPERIENCES):
        self._max = max_experiences
        if store_path is not None:
            self._store_path = store_path
            self._explicit_path = True
        else:
            from external_llm.editor.learning.strategy_state import get_path
            self._store_path = get_path()
            self._explicit_path = False
        self._experiences: deque = deque(maxlen=self._max)
        self._load()

        # Semantic index state. Kept lazy and invalidated on every mutation
        # (record/clear) so it can never drift from self._experiences. Building
        # it is O(n) embedding calls, so it is deferred until the first
        # find_similar_semantic() call — most stores are never queried that way.
        self._embed_matrix = None      # np.ndarray (n, d) | None
        self._embed_dirty = True       # True ⇒ rebuild before next semantic query

    def record(self, record: ExperienceRecord) -> None:
        """Record a new experience."""
        try:
            if record.timestamp == 0.0:
                record.timestamp = time.time()
            self._experiences.append(record.to_dict())
            self._save()
            # The deque mutated; any cached embedding matrix is now stale.
            self._embed_dirty = True
        except Exception as e:
            logger.debug("Experience record failed: %s", e)

    def get_all(self) -> list[dict[str, Any]]:
        """Return all experiences."""
        return list(self._experiences)

    def find_similar(
        self,
        signature,
        threshold: float = SIMILARITY_THRESHOLD,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find experiences with similar problem signatures.
        Returns experiences sorted by similarity (descending).
        """
        results = []

        try:
            from external_llm.editor.learning.problem_signature import ProblemSignature

            if isinstance(signature, ProblemSignature):
                query_sig = signature
            elif isinstance(signature, dict):
                query_sig = ProblemSignature(**{k: v for k, v in signature.items() if hasattr(ProblemSignature, k)})
            else:
                return results

            for exp in self._experiences:
                exp_sig_data = exp.get("problem_signature", {})
                exp_sig = ProblemSignature(**{k: v for k, v in exp_sig_data.items() if hasattr(ProblemSignature, k)})
                sim = query_sig.similarity_score(exp_sig)
                if sim >= threshold:
                    results.append({**exp, "_similarity": sim})

            results.sort(key=lambda x: -x.get("_similarity", 0))
            return results[:limit]

        except Exception as e:
            logger.debug("Experience similarity search failed: %s", e)

        return results

    # ------------------------------------------------------------------
    # Semantic recall (embedding-augmented matching)
    # ------------------------------------------------------------------
    #
    # Why this exists alongside find_similar(): the exact-match
    # similarity_score() returns 0 when failure_type / module / symbol differ
    # at all, so semantically related past failures (e.g. a "rename" failure
    # on UserAuth vs LoginManager; an "indentation mismatch" in edit_text vs
    # anchor_edit) are never recalled. Embedding cosine over a text
    # serialization of the signature catches those, and Reciprocal Rank
    # Fusion merges the two rankings without disturbing the exact path.
    #
    # FAISS is deliberately NOT used here: the store caps at MAX_EXPERIENCES
    # (200), so a numpy brute-force cosine is tens of microseconds and avoids
    # the separate index file + dirty-flag + __del__ persistence the
    # VectorCacheManager carries. We only borrow the (already-loaded)
    # embedding model singleton.

    def _ensure_embed_index(self) -> bool:
        """Build self._embed_matrix from self._experiences if dirty.

        Returns True if an embedding matrix is available for the current
        contents (which includes the empty case — an empty matrix is valid,
        it just means semantic search will return nothing). Returns False
        only when embeddings are genuinely unavailable (deps missing / model
        load failed), in which case callers fall back to exact-only.
        """
        if not self._embed_dirty and self._embed_matrix is not None:
            return True
        self._embed_dirty = False
        if not self._experiences:
            self._embed_matrix = None
            return True
        try:
            import numpy as np

            from external_llm.agent.vector_cache import HAS_NUMPY, HAS_SENTENCE_TRANSFORMERS, get_global_embedding_model
        except Exception:
            self._embed_matrix = None
            return False
        if not HAS_NUMPY or not HAS_SENTENCE_TRANSFORMERS:
            self._embed_matrix = None
            return False
        model = get_global_embedding_model()
        if model is None:
            self._embed_matrix = None
            return False
        try:
            from external_llm.editor.learning.problem_signature import ProblemSignature

            texts = []
            for exp in self._experiences:
                sig_data = exp.get("problem_signature", {}) if isinstance(exp, dict) else {}
                sig = ProblemSignature(**{k: v for k, v in sig_data.items() if hasattr(ProblemSignature, k)})
                texts.append(sig.to_text() or ProblemSignature(request_type="unknown").to_text())
            mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype('float32')
            # L2-normalize rows so dot product == cosine similarity.
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embed_matrix = mat / norms
            return True
        except Exception as e:
            logger.debug("Experience semantic index build failed: %s", e)
            self._embed_matrix = None
            return False

    def find_similar_semantic(
        self,
        signature,
        threshold: float = SIMILARITY_THRESHOLD,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find similar experiences using RRF fusion of exact-match and embedding cosine.

        Augments find_similar() with semantic recall so that related-but-not-
        identical past failures are surfaced. Pure no-op when embeddings are
        unavailable: returns an empty list, leaving the caller to use the
        exact-match path unchanged.
        """
        results: list[dict[str, Any]] = []
        if not self._experiences:
            return results
        try:
            import numpy as np
        except Exception:
            return results
        if not self._ensure_embed_index():
            return results
        if self._embed_matrix is None:
            return results

        try:
            from external_llm.agent.vector_cache import get_global_embedding_model
            from external_llm.editor.learning.problem_signature import ProblemSignature

            if isinstance(signature, ProblemSignature):
                query_sig = signature
            elif isinstance(signature, dict):
                query_sig = ProblemSignature(**{k: v for k, v in signature.items() if hasattr(ProblemSignature, k)})
            else:
                return results

            # --- Exact-match ranking (same logic as find_similar) ---
            exact_rank: list[int] = []  # deque indices, best first
            scored: list[tuple] = []
            for i, exp in enumerate(self._experiences):
                sig_data = exp.get("problem_signature", {}) if isinstance(exp, dict) else {}
                exp_sig = ProblemSignature(**{k: v for k, v in sig_data.items() if hasattr(ProblemSignature, k)})
                sim = query_sig.similarity_score(exp_sig)
                if sim >= threshold:
                    scored.append((sim, i))
            scored.sort(key=lambda t: (-t[0], t[1]))
            exact_rank = [i for _, i in scored]

            # --- Semantic ranking (cosine) ---
            query_text = query_sig.to_text() or ProblemSignature(request_type="unknown").to_text()
            model = get_global_embedding_model()
            if model is None:
                return results
            q = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False).astype('float32')
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return results
            q = q / q_norm
            # (n, d) @ (d,) → (n,) cosine similarities.
            sims = self._embed_matrix @ q.reshape(-1)
            sem_order = list(np.argsort(-sims)[:_SEMANTIC_LIMIT])

            # --- Reciprocal Rank Fusion ---
            rrf: dict[int, float] = {}
            for rank, i in enumerate(exact_rank[:_SEMANTIC_LIMIT]):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
            for rank, i in enumerate(sem_order):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)

            fused = sorted(rrf.items(), key=lambda kv: -kv[1])
            for i, score in fused[:limit]:
                exp = self._experiences[i]
                results.append({**exp, "_semantic_score": float(score)})
            return results
        except Exception as e:
            logger.debug("Experience semantic search failed: %s", e)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        total = len(self._experiences)
        successes = sum(1 for e in self._experiences if e.get("success"))
        return {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all experiences."""
        self._experiences.clear()
        self._save()
        self._embed_matrix = None
        self._embed_dirty = True

    def _load(self) -> None:
        try:
            if self._explicit_path:
                if os.path.isfile(self._store_path):
                    with open(self._store_path) as f:
                        data = json.load(f)
            else:
                from external_llm.editor.learning.strategy_state import read_namespace, write_namespace
                data = read_namespace("experience_store")
                # Fallback: migrate legacy file (~/.asicode/experience_store.json)
                if data is None:
                    legacy_path = os.path.join(
                        os.path.expanduser("~"), ".asicode", "experience_store.json",
                    )
                    if os.path.isfile(legacy_path):
                        try:
                            with open(legacy_path) as f:
                                legacy_data = json.load(f)
                            if isinstance(legacy_data, list) and legacy_data:
                                write_namespace("experience_store", legacy_data)
                                data = legacy_data
                                logger.debug("ExperienceStore: migrated from %s", legacy_path)
                        except Exception:
                            pass
            if isinstance(data, list):
                for item in data[-self._max:]:
                    self._experiences.append(item)
        except Exception as e:
            logger.debug("ExperienceStore._load: failed to load from %s: %s", self._store_path, e)

    def _save(self) -> None:
        try:
            if self._explicit_path:
                atomic_write_json(self._store_path, list(self._experiences), indent=None, ensure_ascii=True)
            else:
                from external_llm.editor.learning.strategy_state import write_namespace
                write_namespace("experience_store", list(self._experiences))
        except Exception as e:
            logger.debug("ExperienceStore._save: failed to save to %s: %s", self._store_path, e)
