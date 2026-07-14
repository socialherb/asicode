"""
Unit tests for SemanticIntentMatcher (embedding-based intent fallback).

The matcher is a fallback enhancer for keyword paths: it must classify by
cosine similarity when an embedding model is available, and degrade to a no-op
(returning None / False) when it is not. These tests use a deterministic fake
embedding model so they don't depend on the real SentenceTransformer.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

np = pytest.importorskip("numpy")

from external_llm.agent.semantic_intent import SemanticIntentMatcher


class _FakeModel:
    """Maps text to a 2-D unit vector: axis 0 = removal-ish, axis 1 = additive."""

    _REMOVAL = ("remove", "delete", "drop", "purge", "rid", "wipe")
    _ADD = ("add", "create", "new", "fix", "refactor")

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
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

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
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
    return SemanticIntentMatcher(ANGLE_EXAMPLES, threshold=threshold, margin=margin,
                                 name="angle")


def test_mean_aggregation_picks_nearer_label(monkeypatch):
    m = _angle_matcher(monkeypatch)
    # 30° is nearer the removal centroid (0°) than the other centroid (90°).
    label, _ = m.classify("ANG:30")
    assert label == "removal"


def test_margin_rejects_ambiguous_query(monkeypatch):
    # 44° sits almost equidistant between the two centroids → tiny margin.
    near = _angle_matcher(monkeypatch, margin=0.02)
    assert near.classify("ANG:44") is not None      # small margin tolerated
    strict = _angle_matcher(monkeypatch, margin=0.2)
    assert strict.classify("ANG:44") is None         # large margin rejects it


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
