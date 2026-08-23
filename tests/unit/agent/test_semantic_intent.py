"""
Unit tests for SemanticIntentMatcher (embedding-based intent fallback).

The matcher is a fallback enhancer for keyword paths: it must classify by
cosine similarity when an embedding model is available, and degrade to a no-op
(returning None / False) when it is not. These tests use a deterministic fake
embedding model so they don't depend on the real SentenceTransformer.
"""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

np = pytest.importorskip("numpy")

from external_llm.agent.semantic_intent import SemanticIntentMatcher


class _FakeModel:
    """Maps text to a 2-D unit vector: axis 0 = removal-ish, axis 1 = additive."""

    _REMOVAL = ("remove", "delete", "drop", "purge", "rid", "wipe")
    _ADD = ("add", "create", "new", "fix", "refactor")

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for t in texts:
            low = t.lower()
            r = 1.0 if any(k in low for k in self._REMOVAL) else 0.0
            a = 1.0 if any(k in low for k in self._ADD) else 0.0
            if r == 0.0 and a == 0.0:
                a = 1.0  # neutral text leans non-removal
            rows.append([r, a])
        arr = np.asarray(rows, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


EXAMPLES = {
    "removal": ["remove the import", "delete this function", "purge dead code"],
    "other": ["add a feature", "fix the bug", "refactor the class"],
}


def _matcher(monkeypatch, threshold=0.45):
    monkeypatch.setattr(
        "external_llm.agent.semantic_intent.get_global_embedding_model",
        lambda: _FakeModel(),
    )
    return SemanticIntentMatcher(EXAMPLES, threshold=threshold, name="test")


def test_classifies_removal_synonym(monkeypatch):
    m = _matcher(monkeypatch)
    label, score = m.classify("please get rid of this helper")
    assert label == "removal"
    assert score == pytest.approx(1.0, abs=1e-5)
    assert m.matches("wipe out the old code", "removal")


def test_rejects_additive_intent(monkeypatch):
    m = _matcher(monkeypatch)
    assert m.matches("add error handling here", "other")
    assert not m.matches("add error handling here", "removal")


def test_empty_text_returns_none(monkeypatch):
    m = _matcher(monkeypatch)
    assert m.classify("") is None
    assert m.classify("   ") is None
    assert not m.matches("", "removal")


def test_threshold_floor_rejects_low_similarity(monkeypatch):
    # Threshold above any achievable cosine (max is 1.0) → never matches.
    m = _matcher(monkeypatch, threshold=1.5)
    assert m.classify("remove the import") is None


def test_no_model_degrades_to_noop(monkeypatch):
    monkeypatch.setattr(
        "external_llm.agent.semantic_intent.get_global_embedding_model",
        lambda: None,
    )
    m = SemanticIntentMatcher(EXAMPLES, threshold=0.45, name="test")
    assert m.classify("remove the import") is None
    assert not m.matches("remove the import", "removal")


def test_numpy_absent_degrades_to_noop(monkeypatch):
    monkeypatch.setattr("external_llm.agent.semantic_intent.np", None)
    monkeypatch.setattr(
        "external_llm.agent.semantic_intent.get_global_embedding_model",
        lambda: _FakeModel(),
    )
    m = SemanticIntentMatcher(EXAMPLES, threshold=0.45, name="test")
    assert m.classify("remove the import") is None


import math


class _AngleModel:
    """Encodes text 'ANG:<degrees>' as a 2-D unit vector at that angle.

    Lets tests place examples and queries at precise cosine separations so the
    mean-per-label and margin logic can be exercised deterministically.
    """

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for t in texts:
            deg = float(t.split("ANG:")[1])
            r = math.radians(deg)
            rows.append([math.cos(r), math.sin(r)])
        return np.asarray(rows, dtype="float32")


ANGLE_EXAMPLES = {
    # removal centroid at 0°, other centroid at 90°
    "removal": ["ANG:-5", "ANG:0", "ANG:5"],
    "other": ["ANG:85", "ANG:90", "ANG:95"],
}


def _angle_matcher(monkeypatch, threshold=0.0, margin=0.0):
    monkeypatch.setattr(
        "external_llm.agent.semantic_intent.get_global_embedding_model",
        lambda: _AngleModel(),
    )
    return SemanticIntentMatcher(ANGLE_EXAMPLES, threshold=threshold, margin=margin, name="angle")


def test_mean_aggregation_picks_nearer_label(monkeypatch):
    m = _angle_matcher(monkeypatch)
    # 30° is nearer the removal centroid (0°) than the other centroid (90°).
    label, _ = m.classify("ANG:30")
    assert label == "removal"


def test_margin_rejects_ambiguous_query(monkeypatch):
    # 44° sits almost equidistant between the two centroids → tiny margin.
    near = _angle_matcher(monkeypatch, margin=0.02)
    assert near.classify("ANG:44") is not None  # small margin tolerated
    strict = _angle_matcher(monkeypatch, margin=0.2)
    assert strict.classify("ANG:44") is None  # large margin rejects it


def test_margin_keeps_confident_query(monkeypatch):
    # 10° is clearly removal; even a strict margin keeps it.
    m = _angle_matcher(monkeypatch, margin=0.2)
    result = m.classify("ANG:10")
    assert result is not None and result[0] == "removal"


def test_build_is_idempotent_and_cached(monkeypatch):
    calls = {"n": 0}
    real = _FakeModel()

    def counting_get():
        calls["n"] += 1
        return real

    monkeypatch.setattr(
        "external_llm.agent.semantic_intent.get_global_embedding_model",
        counting_get,
    )
    m = SemanticIntentMatcher(EXAMPLES, threshold=0.45, name="test")
    m.classify("remove x")
    m.classify("delete y")
    assert calls["n"] == 1  # model fetched once, examples encoded once


# ── RED→GREEN: 남은 브랜치 (빌드 실패/빈 예시/이중 빌드/분류 실패) ──────────


class _RaisesAtQueryModel:
    """빌드 시 encode는 성공, classify 쿼리 encode(1개 텍스트)에서만 실패."""

    def __init__(self, np_):
        self._np = np_

    def encode(self, texts, **kwargs):
        if len(texts) == 1:
            raise RuntimeError("embedding model crashed")
        arr = self._np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        return arr / self._np.linalg.norm(arr, axis=1, keepdims=True)


class TestMatcherEdgePaths:
    def test_empty_examples_disables_matcher(self, monkeypatch):
        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _FakeModel(),
        )
        m = SemanticIntentMatcher({"x": ["", "   "]}, threshold=0.1)
        assert m.classify("anything") is None

    def test_build_encode_failure_disables_matcher(self, monkeypatch):
        class _BoomModel:
            def encode(self, texts, **kwargs):
                raise RuntimeError("model load failed")

        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _BoomModel(),
        )
        m = SemanticIntentMatcher(EXAMPLES, threshold=0.1)
        assert m.classify("remove this") is None

    def test_double_build_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _FakeModel(),
        )
        m = SemanticIntentMatcher(EXAMPLES, threshold=0.0)
        m._ensure_built()
        m._ensure_built()  # 두 번째는 _built 가드에서 즉시 반환
        assert m._available is True

    def test_classify_query_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _RaisesAtQueryModel(np),
        )
        m = SemanticIntentMatcher(EXAMPLES, threshold=0.0)
        assert m.classify("remove the import") is None

    def test_empty_text_short_circuits_before_build(self, monkeypatch):
        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _FakeModel(),
        )
        m = SemanticIntentMatcher(EXAMPLES, threshold=0.1)
        assert m.classify("   ") is None
        assert m._built is False  # 빌드 전에 거름


class _GateLock:
    """첫 번째 진입자만 게이트에서 대기시키는 락 프록시 (결정적 레이스 재현)."""

    def __init__(self):
        self._real = threading.Lock()
        self._first = True
        self.entered = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        if self._first:
            self._first = False
            self.entered.set()
            self.release.wait(5)
        self._real.acquire()
        return self

    def __exit__(self, *a):
        self._real.release()


class TestInLockDoubleBuildGuard:
    def test_second_thread_sees_built_flag_inside_lock(self, monkeypatch):
        """락 내부 이중 확인 가드: B가 77행을 통과한 뒤 대기하고, A가 빌드를
        완료하면 B는 락 진입 후 `if self._built: return`(81행)에서 끝난다."""
        monkeypatch.setattr(
            "external_llm.agent.semantic_intent.get_global_embedding_model",
            lambda: _FakeModel(),
        )
        m = SemanticIntentMatcher(EXAMPLES, threshold=0.1)
        m._lock = _GateLock()

        t = threading.Thread(target=m._ensure_built)
        t.start()
        assert m._lock.entered.wait(5)  # B가 락 게이트에서 대기
        m._ensure_built()  # A: 락 자유 — 빌드 완료
        m._lock.release.set()  # B가 락 획득 → _built True → 81행
        t.join(timeout=5)
        assert not t.is_alive()
        assert m._available is True
