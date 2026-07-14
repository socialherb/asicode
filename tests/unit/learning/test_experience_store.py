"""Tests for ExperienceStore."""
import pytest

from external_llm.editor.learning.experience_store import ExperienceRecord, ExperienceStore
from external_llm.editor.learning.problem_signature import ProblemSignature


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "exp.json")
    return ExperienceStore(store_path=path)


class TestExperienceRecord:
    def test_to_dict(self):
        r = ExperienceRecord(strategy_used="refactor", success=True)
        d = r.to_dict()
        assert d["strategy_used"] == "refactor"
        assert d["success"] is True


class TestExperienceStore:
    def test_record_and_retrieve(self, store):
        r = ExperienceRecord(strategy_used="refactor", success=True,
                             problem_signature={"failure_type": "test_failure"})
        store.record(r)
        all_exp = store.get_all()
        assert len(all_exp) == 1
        assert all_exp[0]["strategy_used"] == "refactor"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "exp.json")
        store1 = ExperienceStore(store_path=path)
        store1.record(ExperienceRecord(strategy_used="x", success=True,
                                        problem_signature={}))

        store2 = ExperienceStore(store_path=path)
        assert len(store2.get_all()) == 1

    def test_max_experiences(self, tmp_path):
        path = str(tmp_path / "exp.json")
        store = ExperienceStore(store_path=path, max_experiences=5)
        for i in range(10):
            store.record(ExperienceRecord(strategy_used=f"s{i}", problem_signature={}))
        assert len(store.get_all()) == 5

    def test_find_similar(self, store):
        for _i in range(5):
            store.record(ExperienceRecord(
                strategy_used="update_callers",
                success=True,
                problem_signature={"failure_type": "test_failure", "module": "mod.agent"},
            ))

        sig = ProblemSignature(failure_type="test_failure", module="mod.agent")
        similar = store.find_similar(sig)
        assert len(similar) > 0

    def test_find_similar_no_match(self, store):
        store.record(ExperienceRecord(
            strategy_used="x", problem_signature={"failure_type": "apply_failed"},
        ))
        sig = ProblemSignature(failure_type="test_failure")
        similar = store.find_similar(sig, threshold=0.5)
        assert len(similar) == 0

    def test_get_stats(self, store):
        store.record(ExperienceRecord(success=True, problem_signature={}))
        store.record(ExperienceRecord(success=False, problem_signature={}))
        stats = store.get_stats()
        assert stats["total"] == 2
        assert stats["successes"] == 1

    def test_clear(self, store):
        store.record(ExperienceRecord(problem_signature={}))
        store.clear()
        assert len(store.get_all()) == 0


class TestExperienceStoreDict:
    def test_find_similar_with_dict(self, store):
        store.record(ExperienceRecord(
            strategy_used="refactor",
            success=True,
            problem_signature={"failure_type": "test_failure"},
        ))
        similar = store.find_similar({"failure_type": "test_failure"})
        assert len(similar) > 0


class TestProblemSignatureToText:
    def test_to_text_includes_labels_and_omits_unknown(self):
        sig = ProblemSignature(
            failure_type="test_failure", module="mod.agent",
            symbol="UserAuth.login", request_type="bugfix",
        )
        txt = sig.to_text()
        assert "failure_type test_failure" in txt
        assert "module mod.agent" in txt
        assert "symbol UserAuth.login" in txt
        assert "request_type bugfix" in txt
        # unknown-valued fields are omitted so they don't dilute the embedding.
        assert "risk_level" not in txt
        assert "impact_size" not in txt

    def test_to_text_empty_signature(self):
        assert ProblemSignature().to_text() == ""


class _FakeEmbeddingModel:
    """Deterministic embedding model substitute.

    Maps each text to a fixed unit vector indexed by a stable hash, so tests
    are reproducible and run without loading the real model.
    """

    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        import hashlib

        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for r, t in enumerate(texts):
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
            mat[r, h % self.dim] = 1.0
        # L2-normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class TestFindSimilarSemantic:
    """Tests for the embedding-augmented semantic recall path."""

    def test_no_op_when_store_empty(self, store):
        # No experiences recorded → semantic search returns nothing, never raises.
        result = store.find_similar_semantic(ProblemSignature(failure_type="x"))
        assert result == []

    def test_semantic_recalls_recorded_experience(self, store, monkeypatch):
        store.record(ExperienceRecord(
            strategy_used="update_callers", success=True,
            problem_signature={"failure_type": "test_failure", "module": "mod"},
        ))
        fake = _FakeEmbeddingModel()
        # The module imports get_global_embedding_model lazily inside methods;
        # patch the source module so the lazy import resolves to the fake.
        import external_llm.agent.vector_cache as vc
        monkeypatch.setattr(vc, "HAS_NUMPY", True, raising=False)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True, raising=False)
        monkeypatch.setattr(vc, "get_global_embedding_model", lambda: fake, raising=False)

        result = store.find_similar_semantic(
            ProblemSignature(failure_type="test_failure", module="mod"),
        )
        assert len(result) == 1
        assert result[0]["strategy_used"] == "update_callers"
        assert "_semantic_score" in result[0]

    def test_rrf_fuses_exact_and_semantic(self, store, monkeypatch):
        """RRF should surface items found by EITHER exact or semantic ranking."""
        # Two experiences: one exact-matchable, one only semantically reachable.
        store.record(ExperienceRecord(
            strategy_used="exact_match_strategy", success=True,
            problem_signature={"failure_type": "test_failure", "module": "mod"},
        ))
        store.record(ExperienceRecord(
            strategy_used="semantic_only_strategy", success=False,
            problem_signature={"failure_type": "other_type", "module": "other"},
        ))
        import external_llm.agent.vector_cache as vc
        fake = _FakeEmbeddingModel()
        monkeypatch.setattr(vc, "HAS_NUMPY", True, raising=False)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True, raising=False)
        monkeypatch.setattr(vc, "get_global_embedding_model", lambda: fake, raising=False)

        # Query matches the first experience exactly.
        result = store.find_similar_semantic(
            ProblemSignature(failure_type="test_failure", module="mod"),
            limit=5,
        )
        # The exact-match item must appear (RRF boosts it strongly).
        strategies = [r["strategy_used"] for r in result]
        assert "exact_match_strategy" in strategies

    def test_index_invalidated_on_record(self, store, monkeypatch):
        # Recording must mark the embedding index dirty so it rebuilds.
        import external_llm.agent.vector_cache as vc
        fake = _FakeEmbeddingModel()
        monkeypatch.setattr(vc, "HAS_NUMPY", True, raising=False)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True, raising=False)
        monkeypatch.setattr(vc, "get_global_embedding_model", lambda: fake, raising=False)

        store.record(ExperienceRecord(
            problem_signature={"failure_type": "a"},
        ))
        store.find_similar_semantic(ProblemSignature(failure_type="a"))
        assert store._embed_dirty is False
        assert store._embed_matrix is not None
        # A new record must invalidate the cached matrix.
        store.record(ExperienceRecord(
            problem_signature={"failure_type": "b"},
        ))
        assert store._embed_dirty is True

    def test_no_op_when_model_unavailable(self, store, monkeypatch):
        """When the embedding model cannot load, semantic returns [] (graceful)."""
        store.record(ExperienceRecord(
            problem_signature={"failure_type": "test_failure"},
        ))
        import external_llm.agent.vector_cache as vc
        monkeypatch.setattr(vc, "HAS_NUMPY", True, raising=False)
        monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True, raising=False)
        monkeypatch.setattr(vc, "get_global_embedding_model", lambda: None, raising=False)
        result = store.find_similar_semantic(ProblemSignature(failure_type="test_failure"))
        assert result == []
        # And the exact path is untouched.
        assert len(store.find_similar(ProblemSignature(failure_type="test_failure"))) > 0
