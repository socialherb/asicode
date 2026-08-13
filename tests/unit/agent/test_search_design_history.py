"""Tests for search_design_history improvements (P0 + P2)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.design_session import DesignSessionManager


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo root with .asicode directory."""
    asr_dir = tmp_path / ".asicode"
    asr_dir.mkdir()
    return tmp_path


@pytest.fixture
def session_mgr(tmp_repo):
    """Create a DesignSessionManager pointing at tmp_repo."""
    return DesignSessionManager(repo_root=str(tmp_repo))


def _make_loop(session_mgr, session_id="test-session-1"):
    """Helper: create a DesignChatLoop with session_id and _session_mgr set."""
    mock_llm = MagicMock()
    mock_registry = MagicMock()
    mock_registry.repo_root = str(session_mgr.repo_root)
    mock_registry.repo_config = MagicMock()
    mock_registry.repo_config.project_type = "python"

    loop = DesignChatLoop(
        llm_client=mock_llm,
        registry=mock_registry,
        model="test-model",
    )
    loop.session_id = session_id
    loop._session_mgr = session_mgr
    return loop


@pytest.fixture
def populated_session_mgr(session_mgr, tmp_repo):
    """Create a session manager with a populated session containing turns + decisions + summary."""
    session = session_mgr.get_or_create("test-session-1")

    # Add turns (some compressed/old)
    session.turns = [
        {"role": "user", "content": "I want to add logging to the handler module.", "timestamp": 1000000.0},
        {"role": "assistant", "content": "Let me look at the handler.py file for existing logging patterns.", "timestamp": 1000010.0},
        {"role": "user", "content": "Actually, can we add validation for empty inputs instead?", "timestamp": 1000020.0},
        {"role": "assistant", "content": "Sure, let's add input validation using the existing validator pattern.", "timestamp": 1000030.0},
        {"role": "user", "content": "Also need to handle the edge case for None values.", "timestamp": 1000040.0},
        {"role": "assistant", "content": "The None handler is in utils.py. Let's add a guard clause there.", "timestamp": 1000050.0},
    ]

    # Add compressed summary (covers first 4 turns)
    session.compressed_summary = (
        "The user requested adding logging to the handler module, "
        "then changed to input validation for empty inputs instead. "
        "The assistant suggested using the existing validator pattern."
    )
    session.compressed_up_to = 4

    # Add decisions
    session.decisions = [
        "Use existing validator pattern for input validation",
        "Add guard clause in utils.py for None values",
        "Handler module will use centralized logging",
    ]

    # Persist
    session_mgr._save(session)

    return session_mgr


@pytest.fixture
def loop_with_session(populated_session_mgr):
    """Fixture: DesignChatLoop pointed at test-session-1."""
    return _make_loop(populated_session_mgr, "test-session-1")


class TestSessionListingP0:
    """P0: Session listing feature."""

    def test_list_sessions_empty(self, session_mgr):
        """Query 'list sessions' on empty repo returns appropriate message."""
        loop = _make_loop(session_mgr, "test-session-1")
        result = loop._search_design_history("list sessions")
        assert "No sessions found" in result

    def test_list_sessions_with_data(self, loop_with_session):
        """Query 'list sessions' returns formatted session list."""
        result = loop_with_session._search_design_history("list sessions")
        assert "Found 1 session(s)" in result
        assert "test-session-1" in result
        assert "turns=6" in result
        assert "📋" in result  # has summary marker

    def test_list_sessions_korean(self, loop_with_session):
        """Query '세션 목록' in Korean returns session list."""
        result = loop_with_session._search_design_history("세션 목록")
        assert "Found 1 session(s)" in result
        assert "test-session-1" in result

    def test_list_sessions_variant(self, loop_with_session):
        """Variant query 'list all sessions' also triggers listing."""
        result = loop_with_session._search_design_history("list all sessions")
        assert "Found 1 session(s)" in result

    def test_show_sessions(self, loop_with_session):
        """Query 'show sessions' triggers listing."""
        result = loop_with_session._search_design_history("show sessions")
        assert "Found 1 session(s)" in result

    def test_list_sessions_prefix(self, loop_with_session):
        """Queries starting with 'list session' also match."""
        result = loop_with_session._search_design_history("list session with logging context")
        assert "Found 1 session(s)" in result

    def test_normal_search_not_affected(self, loop_with_session):
        """Normal keyword search should not trigger session listing."""
        result = loop_with_session._search_design_history("logging handler")
        assert "Found 1 session(s)" not in result
        assert "turn(s)" in result


class TestFieldSpecificSearchP2:
    """P2: Field-specific search feature."""

    def test_search_decisions_field(self, loop_with_session):
        """Search decisions field finds matching decisions."""
        result = loop_with_session._search_design_history(
            "validation", search_field="decisions"
        )
        assert "Found 1 match(es) in decisions" in result
        assert "[Decision]" in result
        assert "validator pattern" in result

    def test_search_decisions_field_no_match(self, loop_with_session):
        """Search decisions field with non-matching keyword."""
        result = loop_with_session._search_design_history(
            "database", search_field="decisions"
        )
        assert "No matches found" in result
        assert "decisions" in result

    def test_search_summary_field(self, loop_with_session):
        """Search summary field finds matching summary."""
        result = loop_with_session._search_design_history(
            "logging", search_field="summary"
        )
        assert "Found 1 match(es) in summary" in result
        assert "[Summary]" in result
        assert "handler" in result

    def test_search_summary_field_no_match(self, loop_with_session):
        """Search summary field with non-matching keyword."""
        result = loop_with_session._search_design_history(
            "database", search_field="summary"
        )
        assert "No matches found" in result
        assert "summary" in result

    def test_search_content_field_explicit(self, loop_with_session):
        """Explicit search_field='content' works same as default."""
        result = loop_with_session._search_design_history(
            "validation", search_field="content"
        )
        assert "turn(s)" in result
        assert "validation" in result.lower()

    def test_default_search_field_is_content(self, loop_with_session):
        """Default (no search_field) searches turn content."""
        result = loop_with_session._search_design_history("handler")
        assert "turn(s)" in result

    def test_invalid_search_field_fallback(self, loop_with_session):
        """Invalid search_field falls back to content search."""
        result = loop_with_session._search_design_history(
            "handler", search_field="nonexistent"
        )
        assert "turn(s)" in result

    def test_search_all_field_combines(self, loop_with_session):
        """search_field='all' searches turn content (compressed portion)."""
        result = loop_with_session._search_design_history(
            "logging", search_field="all"
        )
        assert "turn(s)" in result
        assert "Found" in result

    def test_no_decisions_available(self, session_mgr):
        """Search decisions when none exist returns appropriate message."""
        loop = _make_loop(session_mgr, "no-decision-session")
        result = loop._search_design_history(
            "validation", search_field="decisions"
        )
        assert "No matches found" in result

    def test_no_summary_available(self, session_mgr):
        """Search summary when none exists returns appropriate message."""
        session = session_mgr.get_or_create("no-summary-session")
        session.turns = [{"role": "user", "content": "Hello", "timestamp": 1000000.0}]
        session_mgr._save(session)

        loop = _make_loop(session_mgr, "no-summary-session")
        result = loop._search_design_history(
            "logging", search_field="summary"
        )
        assert "No compressed summary available" in result

    def test_decisions_summary_search_skips_archive_load(self, populated_session_mgr, monkeypatch):
        """P27-1: decisions/summary field searches must NOT load the turn archive.

        The archive (archive.jsonl) is only needed for content/all searches —
        decisions/summary are session-level docs. Loading it on those fields
        pays a full file read + JSON parse for nothing.
        """
        loop = _make_loop(populated_session_mgr, "test-session-1")
        calls: list[str] = []
        orig = populated_session_mgr.load_archived_turns

        def _counting(sid):
            calls.append(sid)
            return orig(sid)

        monkeypatch.setattr(populated_session_mgr, "load_archived_turns", _counting)

        loop._search_design_history("validation", search_field="decisions")
        assert calls == [], "decisions search must not touch the archive"

        loop._search_design_history("logging", search_field="summary")
        assert calls == [], "summary search must not touch the archive"

        loop._search_design_history("validation", search_field="content")
        assert calls == ["test-session-1"], "content search loads the archive once"


class TestCrossSessionSearch:
    """Cross-session search with search_field."""

    def test_cross_session_search_decisions(self, session_mgr):
        """Search decisions field on a different session."""
        # Populate two sessions
        session1 = session_mgr.get_or_create("session-alpha")
        session1.turns = [{"role": "user", "content": "Alpha content", "timestamp": 1000000.0}]
        session1.decisions = ["Alpha decision about caching"]
        session_mgr._save(session1)

        session2 = session_mgr.get_or_create("session-beta")
        session2.turns = [{"role": "user", "content": "Beta content", "timestamp": 1000000.0}]
        session2.decisions = ["Beta decision about logging"]
        session_mgr._save(session2)

        loop = _make_loop(session_mgr, "session-alpha")

        # Search beta's decisions from alpha's session
        result = loop._search_design_history(
            "logging", search_field="decisions",
            target_session_id="session-beta",
        )
        assert "Found 1 match(es) in decisions" in result
        assert "Beta decision" in result


class TestProcessToolCallIntegration:
    """Integration test: _process_tool_call correctly extracts search_field from args."""

    def test_process_tool_call_passes_search_field(self, loop_with_session):
        """_process_tool_call extracts and passes search_field to _search_design_history."""
        mock_tc = MagicMock()
        mock_tc.name = "search_design_history"
        mock_tc.args = {
            "query": "validation",
            "max_results": 3,
            "search_field": "decisions",
        }

        mock_result = MagicMock()
        mock_result.tool_calls_made = []
        mock_result.tool_results = []

        result = loop_with_session._process_tool_call(mock_tc, stream_callback=None, result=mock_result)

        assert "Found 1 match(es) in decisions" in result
        assert "validator pattern" in result

    def test_process_tool_call_without_search_field(self, loop_with_session):
        """_process_tool_call without search_field defaults to content search."""
        mock_tc = MagicMock()
        mock_tc.name = "search_design_history"
        mock_tc.args = {
            "query": "handler",
        }

        mock_result = MagicMock()
        mock_result.tool_calls_made = []
        mock_result.tool_results = []

        result = loop_with_session._process_tool_call(mock_tc, stream_callback=None, result=mock_result)

        assert "turn(s)" in result


class TestToolSchema:
    """Verify the tool schema is correctly updated."""

    def test_search_field_in_schema(self):
        """search_field parameter is present in schema with correct enum values."""
        from external_llm.agent.tool_schemas import AGENT_TOOL_SCHEMAS

        search_schema = None
        for s in AGENT_TOOL_SCHEMAS:
            if s["name"] == "search_design_history":
                search_schema = s
                break

        assert search_schema is not None, "search_design_history schema not found"

        props = search_schema["parameters"]["properties"]
        assert "search_field" in props, "search_field not in schema properties"

        sf = props["search_field"]
        assert sf["type"] == "string"
        assert "enum" in sf
        assert "content" in sf["enum"]
        assert "decisions" in sf["enum"]
        assert "summary" in sf["enum"]
        assert "all" in sf["enum"]
        assert sf.get("default") == "content"


class TestSemanticRerankCorpusGate:
    """Pin the corpus-size gate that skips the embedding-model load on small corpora.

    Below ``_SEMANTIC_RERANK_MIN_CORPUS`` the searcher runs BM25-only
    (``vector_cache=None``), so the ~8s sentence-transformers import/load is
    never triggered.  Verified by asserting the VCM factory
    (``_get_session_vcm``) is NOT called for small corpora and IS called once
    for a corpus at/above the threshold.
    """

    def test_small_corpus_skips_vcm(self, session_mgr, monkeypatch):
        """Few turns → VCM factory never invoked → no model load."""
        import external_llm.agent.design_chat_loop as _dcl

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        calls = []
        monkeypatch.setattr(_dcl, "_get_session_vcm", lambda: calls.append(1))

        session = session_mgr.get_or_create("small")
        session.turns = [
            {"role": "user", "content": f"turn {i} about topic", "timestamp": float(i)}
            for i in range(6)
        ]
        session.compressed_up_to = 6
        session_mgr._save(session)

        loop = _make_loop(session_mgr, "small")
        loop._search_design_history("topic")
        assert calls == [], "small corpus must not touch the VCM (no model load)"

    def test_large_corpus_uses_vcm(self, session_mgr, monkeypatch):
        """≥ threshold turns → VCM factory invoked exactly once."""
        import external_llm.agent.design_chat_loop as _dcl

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        calls = []
        # Return a TRUTHY VCM stub so _SessionSearcher uses the cached value
        # directly.  A None return (e.g. ``lambda: calls.append(1)``) would make
        # _get_shared_vcm cache None, and the constructor's vector_cache=None
        # branch would then call _get_session_vcm() AGAIN (auto-create fallback),
        # inflating the count to 2.  .search → [] keeps RRF fusion exception-free.
        _stub_vcm = MagicMock()
        _stub_vcm.search.return_value = []

        def _factory():
            calls.append(1)
            return _stub_vcm

        monkeypatch.setattr(_dcl, "_get_session_vcm", _factory)

        n = _dcl._SEMANTIC_RERANK_MIN_CORPUS + 5
        session = session_mgr.get_or_create("large")
        session.turns = [
            {"role": "user", "content": f"turn {i} about topic keyword", "timestamp": float(i)}
            for i in range(n)
        ]
        session.compressed_up_to = n
        session_mgr._save(session)

        loop = _make_loop(session_mgr, "large")
        loop._search_design_history("topic")
        assert len(calls) == 1, "large corpus must engage the VCM exactly once"

    def test_decisions_field_typical_skips_vcm(self, session_mgr, monkeypatch):
        """decisions field with a handful of items → VCM not loaded (typical case)."""
        import external_llm.agent.design_chat_loop as _dcl

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        calls = []
        monkeypatch.setattr(_dcl, "_get_session_vcm", lambda: calls.append(1))

        session = session_mgr.get_or_create("dec")
        session.turns = [{"role": "user", "content": "x", "timestamp": 1.0}]
        session.decisions = ["decision alpha", "decision beta", "decision gamma"]
        session.compressed_up_to = 1
        session_mgr._save(session)

        loop = _make_loop(session_mgr, "dec")
        loop._search_design_history("decision", search_field="decisions")
        assert calls == [], "typical decisions corpus (3 items) must stay BM25-only"

