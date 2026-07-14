"""Tests for P13: Cross-Language Learning.

Tests strategy abstraction, JSONL-backed store, and transfer engine.
"""
import time

import pytest

from external_llm.editor.cross_language.models import (
    AbstractIntent,
    AbstractStrategy,
    CrossLanguageRecord,
    Language,
)
from external_llm.editor.cross_language.strategy_abstraction import (
    abstract_intent,
    abstract_strategy,
    local_strategies,
    peer_strategies,
    to_cross_language_record,
)
from external_llm.editor.cross_language.transfer_engine import TransferEngine
from external_llm.editor.learning.unified_run_record import UnifiedRunRecord
from external_llm.editor.learning.unified_store import UnifiedStore

# ═══════════════════════════════════════════════════════════════════
# 1. Models
# ═══════════════════════════════════════════════════════════════════

class TestModels:
    """Basic model tests."""

    def test_language_enum(self):
        assert Language.PYTHON == "python"
        assert Language.TYPESCRIPT == "typescript"

    def test_abstract_intent_values(self):
        assert AbstractIntent.ADD_SYMBOL == "add_symbol"
        assert AbstractIntent.MODIFY_SYMBOL == "modify_symbol"
        assert AbstractIntent.RENAME_SYMBOL == "rename_symbol"

    def test_abstract_strategy_values(self):
        assert AbstractStrategy.MINIMAL_CHANGE == "minimal_change"
        assert AbstractStrategy.STRUCTURED_EDIT == "structured_edit"
        assert AbstractStrategy.CROSS_FILE == "cross_file"

    def test_cross_language_record_scope(self):
        r = CrossLanguageRecord(
            language="python", abstract_intent="add_symbol",
            abstract_strategy="structured_edit", original_strategy="create_symbol",
            original_intent="CREATE_SYMBOL", success=True, reward=0.8,
            affected_files=1,
        )
        assert r.scope == "single"

        r2 = CrossLanguageRecord(
            language="typescript", abstract_intent="modify_symbol",
            abstract_strategy="cross_file", original_strategy="cross_file",
            original_intent="modify_function", success=False, reward=-0.5,
            affected_files=3,
        )
        assert r2.scope == "multi"

    def test_cross_language_record_context_key(self):
        r = CrossLanguageRecord(
            language="python", abstract_intent="add_symbol",
            abstract_strategy="structured_edit", original_strategy="create_symbol",
            original_intent="CREATE_SYMBOL", success=True, reward=0.8,
            affected_files=1, context_key="add_symbol:single",
        )
        assert r.abstract_context_key == "add_symbol:single"

    def test_cross_language_record_auto_context_key(self):
        r = CrossLanguageRecord(
            language="python", abstract_intent="add_symbol",
            abstract_strategy="structured_edit", original_strategy="create_symbol",
            original_intent="CREATE_SYMBOL", success=True, reward=0.8,
            affected_files=2,
        )
        assert r.abstract_context_key == "add_symbol:multi"


# ═══════════════════════════════════════════════════════════════════
# 2. Strategy Abstraction
# ═══════════════════════════════════════════════════════════════════

class TestStrategyAbstraction:
    """Strategy abstraction mapping tests."""

    # -- TS strategies --

    def test_ts_minimal_patch(self):
        assert abstract_strategy("typescript", "minimal_patch") == \
            AbstractStrategy.MINIMAL_CHANGE

    def test_ts_symbol_edit(self):
        assert abstract_strategy("typescript", "symbol_edit") == \
            AbstractStrategy.STRUCTURED_EDIT

    def test_ts_cross_file(self):
        assert abstract_strategy("typescript", "cross_file") == \
            AbstractStrategy.CROSS_FILE

    def test_ts_graph_repair(self):
        assert abstract_strategy("typescript", "graph_repair") == \
            AbstractStrategy.ERROR_DRIVEN_REPAIR

    def test_ts_body_replace(self):
        assert abstract_strategy("typescript", "body_replace") == \
            AbstractStrategy.BODY_REPLACEMENT

    # -- Python strategies --

    def test_py_generic_create(self):
        assert abstract_strategy("python", "generic_create") == \
            AbstractStrategy.TEMPLATE_BASED

    def test_py_modify_symbol(self):
        assert abstract_strategy("python", "modify_symbol") == \
            AbstractStrategy.STRUCTURED_EDIT

    def test_py_move_symbol(self):
        assert abstract_strategy("python", "move_symbol") == \
            AbstractStrategy.CROSS_FILE

    def test_py_delete_symbol(self):
        assert abstract_strategy("python", "delete_symbol") == \
            AbstractStrategy.MINIMAL_CHANGE

    # -- Unknown --

    def test_unknown_strategy(self):
        assert abstract_strategy("typescript", "nonexistent") == \
            AbstractStrategy.UNKNOWN

    # -- Intent mapping --

    def test_ts_intent_add_function(self):
        assert abstract_intent("typescript", "add_function") == \
            AbstractIntent.ADD_SYMBOL

    def test_ts_intent_modify_function(self):
        assert abstract_intent("typescript", "modify_function") == \
            AbstractIntent.MODIFY_SYMBOL

    def test_py_intent_create_symbol(self):
        assert abstract_intent("python", "CREATE_SYMBOL") == \
            AbstractIntent.ADD_SYMBOL

    def test_py_intent_modify_symbol(self):
        assert abstract_intent("python", "modify_symbol") == \
            AbstractIntent.MODIFY_SYMBOL

    def test_unknown_intent(self):
        assert abstract_intent("python", "nonexistent") == \
            AbstractIntent.UNKNOWN

    # -- Reverse mapping --

    def test_local_strategies_ts_structured(self):
        locals_ = local_strategies("typescript", "structured_edit")
        assert "symbol_edit" in locals_

    def test_local_strategies_py_structured(self):
        locals_ = local_strategies("python", "structured_edit")
        assert "modify_symbol" in locals_

    def test_local_strategies_unknown(self):
        locals_ = local_strategies("typescript", "nonexistent_abstract")
        assert locals_ == []

    # -- Peer strategies --

    def test_peer_strategies_ts_to_py(self):
        peers = peer_strategies("typescript", "symbol_edit")
        assert "python" in peers
        assert "modify_symbol" in peers["python"]

    def test_peer_strategies_py_to_ts(self):
        peers = peer_strategies("python", "modify_symbol")
        assert "typescript" in peers
        assert "symbol_edit" in peers["typescript"]

    def test_peer_strategies_unknown(self):
        peers = peer_strategies("typescript", "nonexistent")
        assert peers == {}

    # -- to_cross_language_record --

    def test_to_cross_language_record_ts(self):
        r = to_cross_language_record(
            language="typescript",
            intent="modify_function",
            strategy="symbol_edit",
            success=True,
            reward=0.85,
            affected_files=2,
        )
        assert r.language == "typescript"
        assert r.abstract_intent == "modify_symbol"
        assert r.abstract_strategy == "structured_edit"
        assert r.original_strategy == "symbol_edit"
        assert r.context_key == "modify_symbol:multi"
        assert r.success is True
        assert r.reward == 0.85

    def test_to_cross_language_record_py(self):
        r = to_cross_language_record(
            language="python",
            intent="CREATE_SYMBOL",
            strategy="generic_create",
            success=True,
            reward=0.7,
        )
        assert r.abstract_intent == "add_symbol"
        assert r.abstract_strategy == "template_based"
        assert r.context_key == "add_symbol:single"


# ═══════════════════════════════════════════════════════════════════
# 3. Unified Store (Cross-Language Queries)
# ═══════════════════════════════════════════════════════════════════

class TestCrossLanguageStore:
    """JSONL-backed unified store tests (cross-language queries)."""

    @pytest.fixture
    def store(self):
        s = UnifiedStore(":memory:")
        yield s

    def _make_record(self, **kwargs) -> UnifiedRunRecord:
        """Helper: create a UnifiedRunRecord for cross-language tests."""
        defaults = dict(
            run_id="",
            timestamp=time.time(),
            language="typescript",
            request="",
            intent="modify_function",
            strategy="symbol_edit",
            success=True,
            reward=0.8,
            repair_rounds=0,
            affected_files=1,
            error_types=[],
            context_key="modify_symbol:single",
            abstract_strategy="structured_edit",
            final_status="success",
        )
        defaults.update(kwargs)
        return UnifiedRunRecord(**defaults)

    def test_insert_and_count(self, store):
        store.insert(self._make_record())
        assert store.count() == 1

    def test_count_by_language(self, store):
        store.insert(self._make_record(language="typescript"))
        store.insert(self._make_record(language="python"))
        store.insert(self._make_record(language="typescript"))
        assert store.count("typescript") == 2
        assert store.count("python") == 1

    def test_clear(self, store):
        store.insert(self._make_record())
        store.insert(self._make_record())
        store.clear()
        assert store.count() == 0

    def test_load_strategy_scores(self, store):
        store.insert(self._make_record(reward=0.8))
        store.insert(self._make_record(reward=0.6))
        rows = store.load_strategy_scores("structured_edit")
        assert len(rows) == 2
        # Most recent first
        assert rows[0][1] == 0.6  # raw reward
        assert rows[0][2] > 0     # decayed reward (very recent, ≈ raw)

    def test_load_strategy_scores_exclude_language(self, store):
        store.insert(self._make_record(language="typescript", reward=0.8))
        store.insert(self._make_record(language="python", reward=0.7))
        rows = store.load_strategy_scores(
            "structured_edit", exclude_language="typescript")
        assert len(rows) == 1
        assert rows[0][0] == "python"

    def test_load_strategy_scores_with_context(self, store):
        store.insert(self._make_record(
            context_key="modify_symbol:single", reward=0.8))
        store.insert(self._make_record(
            context_key="add_symbol:single", reward=0.6))
        rows = store.load_strategy_scores(
            "structured_edit", context_key="modify_symbol:single")
        assert len(rows) == 1
        assert rows[0][1] == 0.8

    def test_aggregate_strategy_score(self, store):
        store.insert(self._make_record(language="python", reward=0.8))
        store.insert(self._make_record(language="python", reward=0.6))
        mean, count = store.aggregate_strategy_score("structured_edit")
        assert count == 2
        # Very recent records → decayed ≈ raw
        assert abs(mean - 0.7) < 0.05

    def test_aggregate_excludes_language(self, store):
        store.insert(self._make_record(language="typescript", reward=0.9))
        store.insert(self._make_record(language="python", reward=0.5))
        mean, count = store.aggregate_strategy_score(
            "structured_edit", exclude_language="typescript")
        assert count == 1
        assert abs(mean - 0.5) < 0.05

    def test_load_context_scores(self, store):
        store.insert(self._make_record(
            abstract_strategy="structured_edit", reward=0.8,
            context_key="modify_symbol:single"))
        store.insert(self._make_record(
            abstract_strategy="minimal_change", reward=0.6,
            context_key="modify_symbol:single",
            strategy="minimal_patch"))
        scores = store.load_context_scores("modify_symbol:single")
        assert "structured_edit" in scores
        assert "minimal_change" in scores
        assert scores["structured_edit"][1] == 1  # count

    def test_compaction(self):
        store = UnifiedStore(":memory:", max_records=5)
        for i in range(8):
            store.insert(self._make_record(reward=float(i)))
        assert store.count() == 5

    def test_decay_applied_on_load(self, store):
        """Verify age-based decay is applied."""
        store.insert(self._make_record(
            timestamp=time.time() - 30 * 86400,  # 30 days ago
            language="python", intent="modify_symbol",
            strategy="modify_symbol", abstract_strategy="structured_edit",
            success=True, reward=1.0,
            context_key="modify_symbol:single",
        ))

        rows = store.load_strategy_scores("structured_edit")
        assert len(rows) == 1
        raw = rows[0][1]
        decayed = rows[0][2]
        assert raw == 1.0
        # 30 days with tau=14 days → exp(-30/14) ≈ 0.117
        assert decayed < 0.2
        assert decayed > 0.05

    def test_load_skips_and_heals_corrupt_lines(self, tmp_path):
        """Corrupt/non-object lines are skipped on load, then atomically dropped
        from the file (self-heal) so they don't persist and re-warn forever."""
        import json as _json
        path = tmp_path / "run_history.jsonl"
        good1 = _json.dumps({"run_id": "a", "language": "python",
                             "strategy": "modify_symbol", "reward": 0.8})
        good2 = _json.dumps({"run_id": "b", "language": "typescript",
                             "strategy": "symbol_edit", "reward": 0.6})
        path.write_text(
            "this is not json\n"           # unparseable
            + "[1, 2, 3]\n"                # valid json, but non-object
            + good1 + "\n"
            + "INFO  some redirected console spam\n"  # unparseable
            + good2 + "\n",
            encoding="utf-8",
        )
        store = UnifiedStore(str(path))
        assert store.count() == 2                       # only good records loaded
        # File healed: only the 2 good records remain on disk
        remaining = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
        assert len(remaining) == 2
        # Idempotent: reloading a healed file keeps both records, no further drops
        store2 = UnifiedStore(str(path))
        assert store2.count() == 2

    def test_load_all_garbage_heals_to_empty(self, tmp_path):
        """A file containing only garbage is healed to an empty store file."""
        path = tmp_path / "run_history.jsonl"
        path.write_text("not json\nalso not json\n", encoding="utf-8")
        store = UnifiedStore(str(path))
        assert store.count() == 0
        assert path.read_text(encoding="utf-8") == ""


# ═══════════════════════════════════════════════════════════════════
# 4. Transfer Engine
# ═══════════════════════════════════════════════════════════════════

class TestTransferEngine:
    """Transfer engine blending tests."""

    @pytest.fixture
    def engine(self):
        store = UnifiedStore(":memory:")
        eng = TransferEngine(store)
        yield eng

    def test_record_and_retrieve(self, engine):
        engine.record(
            language="typescript",
            intent="modify_function",
            strategy="symbol_edit",
            success=True,
            reward=0.85,
        )
        assert engine.store.count() == 1
