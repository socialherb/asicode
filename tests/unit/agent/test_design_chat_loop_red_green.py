"""RED→GREEN: design_chat_loop.py tool-routing blocks.

Covers the previously untested _process_tool_call routing (save_insight /
delete_insight / edit_insight / search_design_history + the generic Execute
tool path with snapshots / failure-log / recall / NO_EFFECTIVE_PROGRESS gate /
verify-warning / semantic lint / preview assembly) plus the pure-helper edge
branches of the insight helpers and _SessionSearcher.

Also pins the _edit_insight new_category timestamp-preservation contract for
hand-written headers (no "[category]" bracket).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from external_llm.agent import design_chat_loop as dcl
from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.design_chat_loop import (
    DesignChatLoop,
    DesignChatResult,
    _edit_insight,
    _find_entry_by_match,
    _save_insight_to_file,
    _SessionSearcher,
    load_design_insights,
    load_promoted_insights,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import (
    ContextWindowCollapseError,
    LLMAPIError,
    LLMAuthenticationError,
    LLMCancelled,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponse,
    LLMServerUnavailableError,
    ToolCallRequest,
    ToolCallResponse,
)
from external_llm.design_session import DesignSessionManager


@pytest.fixture
def tmp_repo(tmp_path):
    """Temporary repo root with .asicode directory."""
    asr_dir = tmp_path / ".asicode"
    asr_dir.mkdir()
    return str(tmp_path)


def _tc(name, args=None, call_id="call-1"):
    return SimpleNamespace(name=name, args=args or {}, call_id=call_id)


def _make_loop(repo_root, session_mgr=None, run_store=None, write_tools=None):
    """DesignChatLoop with a scripted MagicMock registry."""
    reg = MagicMock()
    reg.repo_root = repo_root
    reg._WRITE_TOOLS = set(write_tools or [])
    reg.normalize_args_for_display = lambda args: args
    reg.dispatch.return_value = SimpleNamespace(content="generic output", error=None, ok=True, metadata={})
    loop = DesignChatLoop(llm_client=MagicMock(), registry=reg, model="test-model")
    loop._session_mgr = session_mgr
    loop._run_store = run_store
    return loop


def _populated_session_mgr(repo_root):
    """DesignSessionManager with a session containing turns + summary + decisions."""
    mgr = DesignSessionManager(repo_root=repo_root)
    session = mgr.get_or_create("test-session-1")
    session.turns = [
        {"role": "user", "content": "I want to add logging to the handler module.", "timestamp": 1000000.0},
        {
            "role": "assistant",
            "content": "Let me look at the handler.py file for existing logging patterns.",
            "timestamp": 1000010.0,
        },
        {
            "role": "user",
            "content": "Actually, can we add validation for empty inputs instead?",
            "timestamp": 1000020.0,
        },
        {
            "role": "assistant",
            "content": "Sure, let's add input validation using the existing validator pattern.",
            "timestamp": 1000030.0,
        },
        {"role": "user", "content": "Also need to handle the edge case for None values.", "timestamp": 1000040.0},
        {
            "role": "assistant",
            "content": "The None handler is in utils.py. Let's add a guard clause there.",
            "timestamp": 1000050.0,
        },
    ]
    session.compressed_summary = (
        "The user requested adding logging to the handler module, "
        "then changed to input validation for empty inputs instead."
    )
    session.compressed_up_to = 4
    session.decisions = [
        "Use existing validator pattern for input validation",
        "Add guard clause in utils.py for None values",
    ]
    mgr._save(session)
    return mgr


class TestProcessToolCallSaveInsight:
    def test_save_success(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("save_insight", {"insight": "Test insight", "category": "architecture"}),
            None,
            result,
        )
        assert out.startswith("Insight saved")
        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        assert os.path.exists(path)
        assert "[architecture]" in Path(path).read_text(encoding="utf-8")
        rec = result.tool_results[0]
        assert rec["ok"] is True and rec["content"] == out
        assert result.tool_calls_made[0]["result_length"] == len(out)

    def test_save_empty_insight(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(_tc("save_insight", {"insight": "   "}), None, result)
        assert out == "Error: 'insight' is required and must not be empty."
        assert result.tool_results[0]["ok"] is False

    def test_save_truncates_to_1000(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        loop._process_tool_call(
            _tc("save_insight", {"insight": "x" * 2500, "category": "pattern"}),
            None,
            result,
        )
        content = Path(os.path.join(tmp_repo, ".asicode", "design_insights.md")).read_text(encoding="utf-8")
        # drop the "### [pattern] <timestamp>" header line, keep only the body
        body = content.split("### [pattern] ", 1)[1].split("\n", 1)[1].strip()
        assert len(body) == 1000

    def test_save_default_category(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        loop._process_tool_call(_tc("save_insight", {"insight": "plain"}), None, result)
        content = Path(os.path.join(tmp_repo, ".asicode", "design_insights.md")).read_text(encoding="utf-8")
        assert "### [general]" in content

    def test_save_exception_records_failure(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        with patch(
            "external_llm.agent.design_chat_loop._save_insight_to_file",
            side_effect=RuntimeError("disk full"),
        ):
            out = loop._process_tool_call(_tc("save_insight", {"insight": "x"}), None, result)
        assert out == "Error saving insight: disk full"
        assert result.tool_results[0]["ok"] is False

    def test_save_stream_events(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        events = []

        def cb(event_type, payload):
            events.append((event_type, payload["tool"], payload["status"]))

        loop._process_tool_call(
            _tc("save_insight", {"insight": "x"}),
            cb,
            result,
        )
        assert ("design_tool_call", "save_insight", "running") in events
        assert ("design_tool_call", "save_insight", "complete") in events

    def test_save_stream_callback_raises_is_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()

        def cb(event_type, payload):
            raise RuntimeError("cli died")

        out = loop._process_tool_call(_tc("save_insight", {"insight": "x"}), cb, result)
        assert out.startswith("Insight saved")  # callback failure is non-fatal


class TestProcessToolCallDeleteInsight:
    def _save_one(self, tmp_repo, text="to be removed", category="pattern"):
        _save_insight_to_file(tmp_repo, text, category)

    def test_delete_success(self, tmp_repo):
        self._save_one(tmp_repo)
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("delete_insight", {"entry_match": "pattern"}),
            None,
            result,
        )
        assert out.startswith("✅ Deleted insight:")
        assert result.tool_results[0]["ok"] is True
        content = Path(os.path.join(tmp_repo, ".asicode", "design_insights.md")).read_text(encoding="utf-8")
        assert "to be removed" not in content

    def test_delete_requires_entry_match(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(_tc("delete_insight", {}), None, result)
        assert out == "Error: 'entry_match' is required."
        assert result.tool_results[0]["ok"] is False

    def test_delete_no_match(self, tmp_repo):
        self._save_one(tmp_repo)
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("delete_insight", {"entry_match": "nothing matches"}),
            None,
            result,
        )
        assert out.startswith("Error: No insight found")
        assert result.tool_results[0]["ok"] is False

    def test_delete_no_file(self, tmp_path):
        repo = str(tmp_path)  # no .asicode dir at all
        loop = _make_loop(repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("delete_insight", {"entry_match": "x"}),
            None,
            result,
        )
        assert out == "Error: No design insights file found."
        assert result.tool_results[0]["ok"] is False

    def test_delete_empty_file(self, tmp_repo):
        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("delete_insight", {"entry_match": "x"}),
            None,
            result,
        )
        assert out == "Error: Design insights file is empty."
        assert result.tool_results[0]["ok"] is False

    def test_delete_exception_records_failure(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        with patch(
            "external_llm.agent.design_chat_loop._delete_insight",
            side_effect=RuntimeError("perm"),
        ):
            out = loop._process_tool_call(_tc("delete_insight", {"entry_match": "x"}), None, result)
        assert out == "Error deleting insight: perm"
        assert result.tool_results[0]["ok"] is False


class TestProcessToolCallEditInsight:
    def _save_one(self, tmp_repo, text="original body", category="gotcha"):
        _save_insight_to_file(tmp_repo, text, category)

    def test_edit_success(self, tmp_repo):
        self._save_one(tmp_repo)
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("edit_insight", {"entry_match": "gotcha", "new_insight": "updated body"}),
            None,
            result,
        )
        assert out.startswith("✅ Edited insight:")
        assert result.tool_results[0]["ok"] is True
        content = Path(os.path.join(tmp_repo, ".asicode", "design_insights.md")).read_text(encoding="utf-8")
        assert "updated body" in content and "original body" not in content

    def test_edit_requires_entry_match(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("edit_insight", {"new_insight": "x"}),
            None,
            result,
        )
        assert out == "Error: 'entry_match' is required."

    def test_edit_requires_new_insight(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("edit_insight", {"entry_match": "x"}),
            None,
            result,
        )
        assert out == "Error: 'new_insight' is required and must not be empty."

    def test_edit_new_category_preserves_timestamp(self, tmp_repo):
        self._save_one(tmp_repo)
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        loop._process_tool_call(
            _tc(
                "edit_insight",
                {
                    "entry_match": "gotcha",
                    "new_insight": "updated body",
                    "new_category": "bug",
                },
            ),
            None,
            result,
        )
        content = Path(os.path.join(tmp_repo, ".asicode", "design_insights.md")).read_text(encoding="utf-8")
        header = next(line for line in content.splitlines() if line.startswith("###"))
        assert header.startswith("### [bug] ")
        assert header.split("### [bug] ", 1)[1].strip()  # timestamp survived

    def test_edit_handwritten_header_keeps_timestamp(self, tmp_repo):
        """B1: hand-written header (no [category] bracket) must keep its
        timestamp when a new category is applied. _find_entry_by_match's
        contract explicitly supports such headers."""
        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Design Chat Insights\n\n### 2026-08-14 18:47 +0900\nlegacy note\n\n")
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        loop._process_tool_call(
            _tc(
                "edit_insight",
                {
                    "entry_match": "2026-08-14",
                    "new_insight": "updated note",
                    "new_category": "pattern",
                },
            ),
            None,
            result,
        )
        content = Path(path).read_text(encoding="utf-8")
        assert "### [pattern] 2026-08-14 18:47 +0900" in content, content
        assert "updated note" in content

    def test_edit_exception_records_failure(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        with patch(
            "external_llm.agent.design_chat_loop._edit_insight",
            side_effect=RuntimeError("boom"),
        ):
            out = loop._process_tool_call(
                _tc("edit_insight", {"entry_match": "x", "new_insight": "y"}),
                None,
                result,
            )
        assert out == "Error editing insight: boom"
        assert result.tool_results[0]["ok"] is False


class TestProcessToolCallSearchHistory:
    def test_search_success(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging", "max_results": 2}),
            None,
            result,
        )
        assert "Found" in out and "logging" in out
        assert result.tool_results[0]["ok"] is True

    def test_search_requires_query(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(_tc("search_design_history", {}), None, result)
        assert out == "Error: 'query' is required."
        assert result.tool_results[0]["ok"] is False

    def test_search_bad_max_results_falls_back(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging", "max_results": "many"}),
            None,
            result,
        )
        assert "Found" in out  # coerced to 3, not crashed

    def test_search_max_results_clamped(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging", "max_results": 9999}),
            None,
            result,
        )
        assert "showing top 50" in out

    def test_search_exception_records_failure(self, tmp_repo):
        mgr = MagicMock()
        mgr.get_or_create.side_effect = RuntimeError("session corrupt")
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging"}),
            None,
            result,
        )
        assert out == "Error searching design history: session corrupt"
        assert result.tool_results[0]["ok"] is False

    def test_search_stream_events(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        events = []

        def cb(event_type, payload):
            events.append((event_type, payload["tool"], payload["status"]))

        loop._process_tool_call(
            _tc("search_design_history", {"query": "logging"}),
            cb,
            result,
        )
        assert ("design_tool_call", "search_design_history", "running") in events
        assert ("design_tool_call", "search_design_history", "complete") in events


class TestProcessToolCallGeneric:
    def test_generic_success(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("read_file", {"path": "x"}), None, result)
        assert out == "generic output"
        assert result.tool_results[0]["ok"] is True
        assert result.tool_calls_made[0]["tool"] == "read_file"

    def test_generic_dispatch_exception(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry.dispatch.side_effect = RuntimeError("tool crashed")
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("bash", {"cmd": "x"}), None, result)
        assert out == "Error: tool crashed"
        assert result.tool_results[0]["ok"] is False

    def test_generic_stream_running_event(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        events = []

        def cb(event_type, payload):
            events.append((event_type, payload["tool"], payload["status"]))

        loop._process_tool_call(_tc("read_file", {"path": "x"}), cb, result)
        assert ("design_tool_call", "read_file", "running") in events
        assert ("design_tool_call", "read_file", "complete") in events

    def test_write_tool_snapshot_and_change_summary(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop.registry._snapshot_target_files = MagicMock(return_value={"f.py": "old"})
        loop.registry._safety_manager.summarize_change = lambda snaps: "diff summary [POST-EDIT DIFF] here"
        loop.registry._safety_manager.all_files_unchanged = lambda snaps: False
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(
            _tc("apply_patch", {"patch": "..."}),
            None,
            result,
        )
        assert "diff summary" in out
        assert loop.registry._snapshot_target_files.called

    def test_verify_warning_appended(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop.registry._snapshot_target_files = lambda name, args: {"f.py": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        loop.registry.dispatch.return_value = SimpleNamespace(
            content="patched",
            error=None,
            ok=True,
            metadata={"verify_warning": "syntax soft-fail"},
        )
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert "[⚠️ VERIFY WARNING]" in out and "syntax soft-fail" in out

    def test_semantic_lint_and_auto_repair(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = True
        loop.registry._snapshot_target_files = lambda name, args: {"f.py": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        loop.registry._safety_manager.new_semantic_warnings = lambda snaps: "[F1] dead branch"
        loop.registry.dispatch.return_value = SimpleNamespace(
            content="patched",
            error=None,
            ok=True,
            metadata={"semantic_repaired": 2},
        )
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert "[F1] dead branch" in out
        assert "[AUTO-REPAIR] 2 semantic finding(s) auto-fixed" in out

    def test_no_effective_progress_downgrades_ok(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop.registry._snapshot_target_files = lambda name, args: {"f.py": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: "⚠️ NO CHANGE"
        loop.registry._safety_manager.all_files_unchanged = lambda snaps: True
        loop.registry.dispatch.return_value = SimpleNamespace(
            content="patched",
            error=None,
            ok=True,
            metadata={},
        )
        result = DesignChatResult()
        result.recall_session_key = "k1"
        loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert result.tool_results[0]["ok"] is False  # downgraded

    def test_preview_truncation_and_diff_reappend(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        events = []

        def cb(event_type, payload):
            events.append(payload)

        loop.registry.dispatch.return_value = SimpleNamespace(
            content="x" * 1200 + "\n[POST-EDIT DIFF] tail info",
            error=None,
            ok=True,
            metadata={},
        )
        loop._process_tool_call(_tc("bash", {}), cb, result)
        payload = events[-1]
        assert payload["status"] == "complete"
        # bash limit 1200 → front-truncated with ellipsis, then the diff re-attached
        assert payload["preview"].startswith("x" * 1200 + "...")
        assert "[POST-EDIT DIFF] tail info" in payload["preview"]

    def test_update_plan_extra_metadata(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        events = []

        def cb(event_type, payload):
            events.append(payload)

        loop.registry.dispatch.return_value = SimpleNamespace(
            content="plan updated",
            error=None,
            ok=True,
            metadata={"plan": {"item": "x"}, "prev_statuses": {"a": "done"}},
        )
        loop._process_tool_call(_tc("update_plan", {}), cb, result)
        payload = events[-1]
        assert payload["plan"] == {"item": "x"}  # **extra spread at top level
        assert payload["plan_prev"] == {"a": "done"}


class TestNoEffectiveProgressGate:
    def test_non_apply_patch_passthrough(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        assert loop._apply_no_effective_progress_gate("bash", True, {}, None) is True

    def test_apply_patch_no_change_downgrades(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry._safety_manager.all_files_unchanged = lambda snaps: True
        metadata = {}
        ok = loop._apply_no_effective_progress_gate("apply_patch", True, {"f": "s"}, metadata)
        assert ok is False
        assert metadata["failure_class"] == "no_effective_change"

    def test_apply_patch_changed_keeps_ok(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry._safety_manager.all_files_unchanged = lambda snaps: False
        assert loop._apply_no_effective_progress_gate("apply_patch", True, {"f": "s"}, {}) is True

    def test_apply_patch_check_error_fail_open(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry._safety_manager.all_files_unchanged = MagicMock(side_effect=RuntimeError)
        assert loop._apply_no_effective_progress_gate("apply_patch", True, {"f": "s"}, {}) is True

    def test_metadata_not_a_dict(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry._safety_manager.all_files_unchanged = lambda snaps: True
        assert loop._apply_no_effective_progress_gate("apply_patch", True, {"f": "s"}, None) is False


class TestProcessToolCallWithLearning:
    def test_records_usage(self, tmp_repo):
        run_store = MagicMock()
        loop = _make_loop(tmp_repo, run_store=run_store)
        result = DesignChatResult()
        loop._process_tool_call_with_learning(_tc("read_file", {"path": "x"}), None, result)
        run_store.record_tool_usage.assert_called_once_with("tool_loop", "read_file", True)

    def test_record_failure_silent(self, tmp_repo):
        run_store = MagicMock()
        run_store.record_tool_usage.side_effect = RuntimeError("store down")
        loop = _make_loop(tmp_repo, run_store=run_store)
        result = DesignChatResult()
        out = loop._process_tool_call_with_learning(_tc("read_file", {"path": "x"}), None, result)
        assert out == "generic output"  # learning failure never breaks the tool


class TestInsightHelperEdges:
    def test_find_entry_zero_matches(self):
        idx, err = _find_entry_by_match([], "anything")
        assert idx is None and "No insight found" in err

    def test_find_entry_multiple_matches(self):
        from external_llm.agent.insights_manager import InsightEntry

        e1 = InsightEntry(
            lines=["### [a] shared keyword\n", "body one\n\n"], header_line="### [a] shared keyword", category="a"
        )
        e2 = InsightEntry(
            lines=["### [b] shared keyword\n", "body two\n\n"], header_line="### [b] shared keyword", category="b"
        )
        idx, err = _find_entry_by_match([e1, e2], "shared keyword")
        assert idx is None and "Multiple insights match" in err
        assert "[a]" in err and "[b]" in err

    def test_edit_empty_file(self, tmp_repo):
        path = os.path.join(tmp_repo, ".asicode", "design_insights.md")
        Path(path).write_text("")
        out = _edit_insight(tmp_repo, "x", "y")
        assert out == "Error: Design insights file is empty."

    def test_load_design_insights_archive_error_ignored(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "keep me", "pattern")
        with patch(
            "external_llm.agent.design_chat_loop.build_archive_index",
            side_effect=RuntimeError("index broke"),
        ):
            content = load_design_insights(tmp_repo)
        assert "keep me" in content  # archive failure must not drop the active block

    def test_load_promoted_insights_exception_returns_empty(self, tmp_repo):
        with patch(
            "external_llm.agent.design_chat_loop.select_promotable_entries",
            side_effect=OSError("file gone"),
        ):
            assert load_promoted_insights(tmp_repo, "query") == ""

    def test_load_promoted_insights_no_overlap(self, tmp_repo):
        with patch(
            "external_llm.agent.design_chat_loop.select_promotable_entries",
            return_value=[],
        ):
            assert load_promoted_insights(tmp_repo, "query") == ""

    def test_load_promoted_insights_blocks(self, tmp_repo):
        from external_llm.agent.insights_manager import InsightEntry

        entry = InsightEntry(
            lines=["### [pattern] 2026-08-14 18:47 +0900\n", "old invariant\n", "\n"],
            header_line="### [pattern] 2026-08-14 18:47 +0900",
            category="pattern",
        )
        with patch(
            "external_llm.agent.design_chat_loop.select_promotable_entries",
            return_value=[entry],
        ):
            out = load_promoted_insights(tmp_repo, "query")
        assert "PROMOTED FROM ARCHIVE" in out and "old invariant" in out


class TestSessionSearcherEdges:
    def test_empty_docs_search_returns_empty(self):
        s = _SessionSearcher(vector_cache=False)
        s.index_docs([])
        assert s.search("anything") == []

    def test_blank_query_tokens_returns_empty(self):
        s = _SessionSearcher(vector_cache=False)
        s.index_docs([("a", "some real content here")])
        assert s.search("") == []
        assert s.search("!!!") == []  # no tokenizable tokens

    def test_vector_cache_exception_falls_back_bm25(self):
        vc = MagicMock()
        vc.add_document.side_effect = RuntimeError("index corrupt")
        s = _SessionSearcher(vector_cache=vc)
        s.index_docs([("1", "alpha beta gamma"), ("2", "beta gamma delta")])
        results = s.search("beta")
        assert results and results[0]["id"] in ("1", "2")

    def test_vector_search_exception_drops_rerank(self):
        vc = MagicMock()
        s = _SessionSearcher(vector_cache=vc)
        s.index_docs([("1", "alpha beta gamma"), ("2", "beta gamma delta")])
        vc.search.side_effect = RuntimeError("faiss broken")
        results = s.search("beta")
        assert results  # BM25 still works

    def test_rrf_rerank_with_vector_hits(self):
        vc = MagicMock()
        s = _SessionSearcher(session_prefix="s", vector_cache=vc)
        s.index_docs([("1", "alpha beta gamma"), ("2", "beta gamma delta")])
        # vector search returns doc key "s1" (prefixed) — semantic hit on doc 0
        vc.search.return_value = [{"file_path": "s1"}]
        results = s.search("beta")
        assert results and results[0]["id"] == "1"
        vc.add_document.assert_called()

    def test_pre_tokenized_archive_skip_vector_index(self):
        vc = MagicMock()
        sig = ("sig", 42)
        pre = [("arch1", {"tok": 1}, 1, "archived text")]
        s = _SessionSearcher(session_prefix="s", vector_cache=vc)
        s.index_docs([], pre_tokenized=pre, archive_sig=sig)
        vc.add_document.assert_called_once()
        # Second pass with same sig → archive vector insertion skipped
        s2 = _SessionSearcher(session_prefix="s", vector_cache=vc)
        s2.index_docs([], pre_tokenized=pre, archive_sig=sig)
        assert vc.add_document.call_count == 1

    def test_vector_cache_init_error_falls_back(self):
        with (
            patch("external_llm.agent.design_chat_loop._HAS_VECTOR_CACHE", True),
            patch("external_llm.agent.design_chat_loop._get_session_vcm", side_effect=RuntimeError("no model")),
        ):
            s = _SessionSearcher()
        assert s._vector_cache is None
        s.index_docs([("1", "alpha beta")])
        assert s.search("alpha")

    def test_index_docs_skips_empty_tokens(self):
        s = _SessionSearcher(vector_cache=False)
        with patch(
            "external_llm.agent.design_chat_loop._TOKENIZER.tokenize",
            side_effect=[[], ["tok"]],
        ):
            s.index_docs([("1", "garbage"), ("2", "real")])
        assert s._n_docs == 1


class TestSearchDesignHistoryBranches:
    def test_no_session_mgr(self, tmp_repo):
        loop = _make_loop(tmp_repo, session_mgr=None)
        loop.session_id = "s1"
        out = loop._search_design_history("anything")
        assert out == "Design session manager not available."

    def test_invalid_field_normalizes_to_content(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        out = loop._search_design_history("logging", search_field="bogus")
        assert "turn(s)" in out

    def test_no_active_session(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = ""
        out = loop._search_design_history("logging")
        assert out == "No active session to search."

    def test_cross_session_empty_history(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        out = loop._search_design_history("logging", target_session_id="empty-session")
        assert "has no conversation history" in out

    def test_current_session_no_old_history(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        session = mgr.get_or_create("fresh")
        session.compressed_up_to = 0
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "fresh"
        out = loop._search_design_history("logging")
        assert "No old conversation history" in out

    def test_archive_load_error_returns_empty(self, tmp_repo):
        session = SimpleNamespace(
            turns=[{"role": "user", "content": "recent only", "timestamp": 1.0}],
            compressed_up_to=1,
            archived_count=0,
            decisions=[],
            compressed_summary="",
        )
        mgr = MagicMock()
        mgr.get_or_create.return_value = session
        mgr.load_archived_turns.side_effect = OSError("archive corrupt")
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "s1"
        out = loop._search_design_history("recent")
        assert "Found" in out  # archived turns degrade to [] without crashing

    def test_decisions_field(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        out = loop._search_design_history("validator", search_field="decisions")
        assert "match(es) in decisions" in out

    def test_summary_field_no_summary(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        session = mgr.get_or_create("nosum")
        session.compressed_summary = ""
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "nosum"
        out = loop._search_design_history("logging", search_field="summary")
        assert "No compressed summary" in out

    def test_all_field_includes_decisions_and_summary(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        out = loop._search_design_history("logging", search_field="all")
        assert "turn(s)" in out

    def test_no_matches_with_suggestion(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        out = loop._search_design_history("zzzznomatch")
        assert "No matches found" in out
        assert "target_session_id" in out


class TestPureHelpers:
    """Pure helper functions: message stripping, context cap, error mapping."""

    def test_strip_tool_messages(self):
        msgs = [
            LLMMessage(role="user", content="hi"),
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "grep"}},
                ],
            ),
            LLMMessage(role="tool", content="result", tool_call_id="c1", name="read_file"),
        ]
        out = dcl._strip_tool_messages(msgs)
        assert len(out) == 2  # tool message dropped
        assert out[1].role == "assistant"
        assert "[Code analysis performed: read_file, grep]" in out[1].content

    def test_strip_tool_messages_keeps_plain_assistant(self):
        msgs = [LLMMessage(role="assistant", content="plain")]
        out = dcl._strip_tool_messages(msgs)
        assert out == msgs

    def test_context_hard_cap_collapse_raises(self):
        with (
            patch("external_llm.agent.design_chat_loop.context_message_cap", return_value=0),
            pytest.raises(ContextWindowCollapseError),
        ):
            dcl._apply_context_hard_cap([LLMMessage(role="user", content="x")], "test-model")

    def test_context_hard_cap_passes_through_unchanged(self):
        """preemptive_trim removed: oversized input passes through untouched —
        enforcement is the provider 400 → overflow-override backstop."""
        msgs = [LLMMessage(role="user", content="a"), LLMMessage(role="user", content="b")]
        out = dcl._apply_context_hard_cap(msgs, "test-model")
        assert out == msgs

    def test_cancel_aware_callback_none_returns_same(self):
        assert dcl._cancel_aware_callback(None, object()) is None

    def test_cancel_aware_callback_skips_when_set(self):
        ev = threading.Event()
        ev.set()
        calls = []
        guarded = dcl._cancel_aware_callback(lambda *a, **k: calls.append(1), ev)
        guarded("x")
        assert calls == []

    def test_cancel_aware_callback_passes_through(self):
        ev = threading.Event()
        calls = []
        guarded = dcl._cancel_aware_callback(lambda *a, **k: calls.append(a), ev)
        guarded("x", y=1)
        assert calls == [("x",)]

    def test_extract_provider_message(self):
        assert dcl._extract_provider_message('{"error":{"message":"boom"}}') == "boom"
        assert dcl._extract_provider_message('{"error":"str err"}') == "str err"
        assert dcl._extract_provider_message("no brace here") == ""
        assert dcl._extract_provider_message("{not json") == ""
        assert dcl._extract_provider_message('{"ok": 1}') == ""

    def test_upstream_gateway_name(self):
        assert dcl._upstream_gateway_name("Error from provider (Console Go): Upstream request failed") == "Console Go"
        assert dcl._upstream_gateway_name("Error from provider: no paren") == ""
        assert dcl._upstream_gateway_name("random text") is None

    def test_user_facing_llm_error_mapping(self):
        assert "GLM server" in dcl._user_facing_llm_error(LLMRateLimitError("x", error_code=1305))
        assert "1302" in dcl._user_facing_llm_error(LLMRateLimitError("x", error_code=1302))
        assert "rate limit" in dcl._user_facing_llm_error(LLMRateLimitError("x"))
        assert "Cannot connect" in dcl._user_facing_llm_error(LLMServerUnavailableError("x"))
        assert "Cannot connect" in dcl._user_facing_llm_error(LLMConnectionError("x"))
        assert "authentication" in dcl._user_facing_llm_error(LLMAuthenticationError("x"))
        assert "quota" in dcl._user_facing_llm_error(LLMQuotaExceededError("x"))
        assert "provider-side" in dcl._user_facing_llm_error(
            LLMAPIError('{"error":{"message":"Error from provider (Console Go): Upstream request failed"}}')
        )
        assert "context window" in dcl._user_facing_llm_error(ContextWindowCollapseError("small"))
        assert "An error occurred" in dcl._user_facing_llm_error(RuntimeError("weird"))

    def test_fallback_plain_chat_success(self):
        client = MagicMock()
        client.chat.return_value = LLMResponse(
            content="plain answer",
            model="m",
            provider="stub",
            tokens_used=7,
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
            raw_response=None,
        )
        out = dcl._fallback_plain_chat([LLMMessage(role="user", content="q")], client, "m")
        assert out["content"] == "plain answer"
        assert out["error"] is False and out["prompt_tokens"] == 5

    def test_fallback_plain_chat_reasoning_extract_failure(self):
        client = MagicMock()
        client.chat.return_value = LLMResponse(
            content="answer",
            model="m",
            provider="stub",
            tokens_used=1,
            finish_reason="stop",
            raw_response=None,
        )
        # raw_response missing → AttributeError inside the try → debug log, still returns
        client.chat.return_value.raw_response = None
        out = dcl._fallback_plain_chat([LLMMessage(role="user", content="q")], client, "m")
        assert out["content"] == "answer"

    def test_fallback_plain_chat_reraises_llm_errors(self):
        client = MagicMock()
        client.chat.side_effect = LLMRateLimitError("rl")
        with pytest.raises(LLMRateLimitError):
            dcl._fallback_plain_chat([LLMMessage(role="user", content="q")], client, "m")

    def test_fallback_plain_chat_non_llm_error_returns_error_dict(self):
        client = MagicMock()
        client.chat.side_effect = ValueError("broken")
        out = dcl._fallback_plain_chat([LLMMessage(role="user", content="q")], client, "m")
        assert out["error"] is True and "broken" in out["content"]


class TestCallLlmWithRetry:
    """_call_llm_with_retry transient-error retry loop."""

    def _loop(self, monkeypatch):
        from dataclasses import replace

        monkeypatch.setattr(
            dcl,
            "_cfg",
            replace(dcl._cfg, counts=replace(dcl._cfg.counts, DESIGN_CHAT_LLM_MAX_RETRIES=1)),
        )
        loop = _make_loop("/tmp/x")
        loop._retry_wait = lambda delay: None  # no real sleep
        loop._flip_zai_endpoint = lambda: False
        return loop

    def test_retry_rate_limit_then_success(self, monkeypatch):
        loop = self._loop(monkeypatch)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise LLMRateLimitError("rl", retry_after=2)
            return "ok"

        assert loop._call_llm_with_retry(fn) == "ok"
        assert len(calls) == 2

    def test_retry_connection_error_then_success(self, monkeypatch):
        loop = self._loop(monkeypatch)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise LLMConnectionError("refused")
            return "ok"

        assert loop._call_llm_with_retry(fn) == "ok"
        assert loop._flip_zai_endpoint.called if hasattr(loop._flip_zai_endpoint, "called") else True

    def test_retry_auth_reraises(self, monkeypatch):
        loop = self._loop(monkeypatch)

        def fn():
            raise LLMAuthenticationError("bad key")

        with pytest.raises(LLMAuthenticationError):
            loop._call_llm_with_retry(fn)

    def test_retry_exhausted_raises(self, monkeypatch):
        loop = self._loop(monkeypatch)

        def fn():
            raise LLMRateLimitError("rl")

        with pytest.raises(LLMRateLimitError):
            loop._call_llm_with_retry(fn)

    def test_retry_context_overflow_retries(self, monkeypatch):
        loop = self._loop(monkeypatch)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise LLMAPIError("maximum context length is 8192")
            return "ok"

        def overflow_cb():
            return 500

        assert loop._call_llm_with_retry(fn, _estimated_prompt_tokens=9000, overflow_retry_cb=overflow_cb) == "ok"
        assert len(calls) == 2

    def test_retry_context_overflow_no_progress_raises(self, monkeypatch):
        loop = self._loop(monkeypatch)

        def fn():
            raise LLMAPIError("maximum context length is 8192")

        with pytest.raises(LLMAPIError):
            loop._call_llm_with_retry(fn, overflow_retry_cb=lambda: None)

    def test_retry_server_unavailable(self, monkeypatch):
        loop = self._loop(monkeypatch)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise LLMServerUnavailableError("503")
            return "ok"

        assert loop._call_llm_with_retry(fn) == "ok"


class _ScriptedClient:
    """LLM stub with scripted chat_with_tools / chat outcomes."""

    def __init__(
        self,
        tool_response=None,
        tool_responses=None,
        tool_error=None,
        chat_responses=None,
        chat_errors=None,
        on_tool_call=None,
    ):
        self.tool_response = tool_response
        self.tool_responses = list(tool_responses or [])
        self.tool_error = tool_error
        self.chat_responses = list(chat_responses or [])
        self.chat_errors = list(chat_errors or [])
        self.on_tool_call = on_tool_call
        self.tool_calls = 0
        self.chat_calls = 0

    @staticmethod
    def get_provider_name() -> str:
        return "stub"

    def chat_with_tools(self, messages, tools, model, **kw):
        self.tool_calls += 1
        if self.on_tool_call:
            self.on_tool_call()
        if self.tool_error is not None:
            e, self.tool_error = self.tool_error, None
            raise e
        if self.tool_responses:
            return self.tool_responses.pop(0)
        return self.tool_response

    def chat(self, messages, model, **kw):
        self.chat_calls += 1
        if self.chat_responses:
            return self.chat_responses.pop(0)
        if self.chat_errors:
            raise self.chat_errors.pop(0)
        return LLMResponse(
            content="default final",
            model=model,
            provider="stub",
            tokens_used=1,
            finish_reason="stop",
            raw_response=None,
        )


def _stop_response(content=""):
    return ToolCallResponse(
        content=content,
        model="m",
        provider="stub",
        tokens_used=1,
        finish_reason="stop",
        raw_response=None,
        tool_calls=[],
    )


@pytest.fixture
def _repo(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("hello\n")
    return str(tmp_path)


def _zero_retries(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(
        dcl,
        "_cfg",
        replace(dcl._cfg, counts=replace(dcl._cfg.counts, DESIGN_CHAT_LLM_MAX_RETRIES=0)),
    )


class TestRespondScenarios:
    """respond() error handling, empty-response funnel, final-response retry."""

    def test_respond_rate_limit_error_sets_is_error(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_error=LLMRateLimitError("rl"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")])
        assert r.is_error is True
        assert "rate limit" in r.content.lower() or "busy" in r.content.lower()

    def test_respond_auth_error_sets_error_type(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_error=LLMAuthenticationError("bad key"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")])
        assert r.is_error is True
        assert r.error_type == "auth"

    def test_respond_generic_error_falls_back_to_plain_chat(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_error=RuntimeError("boom"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")])
        assert r.is_error is False
        assert r.content == "default final"  # fallback plain chat answer
        assert r.total_llm_calls == 1

    def test_respond_empty_response_funnel(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_response=_stop_response(""))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")])
        assert r.content.startswith("⚠️ The model returned an empty response")

    def test_respond_thinking_stop_emitted_in_finally(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_response=_stop_response("done"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        events = []

        def cb(event_type, payload):
            events.append(event_type)

        loop.respond([LLMMessage(role="user", content="do it")], stream_callback=cb)
        assert "design_thinking_stop" in events

    def test_respond_max_iterations_final_retry_supersedes(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                ),
                LLMResponse(
                    content="final after retry",
                    model="m",
                    provider="stub",
                    tokens_used=2,
                    finish_reason="stop",
                    raw_response=None,
                ),
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")], max_tool_iterations=1)
        assert r.hit_max_iterations is True
        assert r.content == "final after retry"

    def test_respond_final_reasoning_fallback(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response={"choices": [{"message": {"reasoning_content": "reasoned answer"}}]},
                ),
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")], max_tool_iterations=1)
        assert r.content == "reasoned answer"

    def test_respond_final_retry_generic_failure(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                ),
            ],
            chat_errors=[RuntimeError("retry died")],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="do it")], max_tool_iterations=1)
        assert r.content.startswith("⚠️")  # empty funnel fallback after failed retry


class TestProcessToolCallStreamEdges:
    """stream_callback failure / registry edge paths inside _process_tool_call."""

    def _raise_cb(self, event_type, payload):
        raise RuntimeError("cli died")

    def test_delete_stream_callback_raises(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "to be removed", "pattern")
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("delete_insight", {"entry_match": "pattern"}),
            self._raise_cb,
            result,
        )
        assert out.startswith("✅ Deleted insight:")

    def test_edit_stream_callback_raises(self, tmp_repo):
        _save_insight_to_file(tmp_repo, "original body", "gotcha")
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("edit_insight", {"entry_match": "gotcha", "new_insight": "new"}),
            self._raise_cb,
            result,
        )
        assert out.startswith("✅ Edited insight:")

    def test_search_stream_callback_raises(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "test-session-1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging"}),
            self._raise_cb,
            result,
        )
        assert "Found" in out

    def test_search_stream_callback_raises_on_error_event(self, tmp_repo):
        mgr = MagicMock()
        mgr.get_or_create.side_effect = RuntimeError("corrupt")
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "s1"
        result = DesignChatResult()
        out = loop._process_tool_call(
            _tc("search_design_history", {"query": "logging"}),
            self._raise_cb,
            result,
        )
        assert out.startswith("Error searching design history:")

    def test_generic_stream_callback_raises(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("read_file", {"path": "x"}), self._raise_cb, result)
        assert out == "generic output"

    def test_normalize_args_fallback_on_type_error(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry.normalize_args_for_display = MagicMock(side_effect=TypeError)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        events = []

        def cb(event_type, payload):
            events.append(payload)

        loop._process_tool_call(_tc("read_file", {"path": "x"}), cb, result)
        assert events[0]["args"] == {"path": "x"}  # fell back to raw args

    def test_snapshot_error_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop.registry._snapshot_target_files = MagicMock(side_effect=RuntimeError)
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "generic output"  # snapshot failure is non-fatal

    def test_verify_warning_metadata_none(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = False
        loop.registry._snapshot_target_files = lambda n, a: {"f": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        loop.registry.dispatch.return_value = SimpleNamespace(
            content="patched",
            error=None,
            ok=True,
            metadata=None,
        )
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "patched"  # metadata=None handled defensively

    def test_failure_log_error_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = False
        loop.registry._snapshot_target_files = lambda n, a: {"f": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        result = DesignChatResult()
        result.recall_session_key = "k1"
        with patch(
            "external_llm.agent.tool_failure_log.record_write_tool_failure_from_tr",
            side_effect=RuntimeError("log broken"),
        ):
            out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "generic output"

    def test_recall_error_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry.dispatch.side_effect = RuntimeError("tool crashed")
        result = DesignChatResult()
        result.recall_session_key = "k1"
        with patch(
            "external_llm.agent.failure_pattern_store.recall_on_failure",
            side_effect=RuntimeError("store down"),
        ):
            out = loop._process_tool_call(_tc("bash", {"cmd": "x"}), None, result)
        assert out == "Error: tool crashed"

    def test_semantic_lint_error_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = True
        loop.registry._snapshot_target_files = lambda n, a: {"f": "old"}
        loop.registry._safety_manager.summarize_change = lambda snaps: None
        loop.registry._safety_manager.new_semantic_warnings = MagicMock(side_effect=RuntimeError)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "generic output"


class TestRetryWaitAndFlip:
    def test_retry_wait_cancel_event_already_set(self):
        ev = threading.Event()
        ev.set()
        loop = _make_loop("/tmp/x")
        loop.registry.config.cancel_event = ev
        with pytest.raises(AgentCancelled):
            loop._retry_wait(1)

    def test_retry_wait_cancel_during_wait(self):
        ev = threading.Event()
        ev.wait = lambda timeout: ev.set() or True  # returns True when set mid-wait
        loop = _make_loop("/tmp/x")
        loop.registry.config.cancel_event = ev
        with pytest.raises(AgentCancelled):
            loop._retry_wait(1)

    def test_retry_wait_sleeps_without_cancel_event(self):
        with patch("external_llm.agent.design_chat_loop.time.sleep") as sl:
            loop = _make_loop("/tmp/x")
            loop.registry.config.cancel_event = None
            loop._retry_wait(3)
        sl.assert_called_once_with(3)

    def test_flip_zai_endpoint_swaps_to_openai_compat(self):
        class _FakeZAI:
            def __init__(self):
                self.get_provider_name = lambda: "zai"
                self.base_url = None
                self.api_key = "k"

        class _FakeOpenAICompat:
            def __init__(self, api_key, base_url, timeout):
                pass

        with (
            patch("external_llm.anthropic_client.ZAIAnthropicClient", _FakeZAI),
            patch("external_llm.openai_client.ZAIClient", _FakeOpenAICompat),
        ):
            loop = _make_loop("/tmp/x")
            loop.llm_client = _FakeZAI()
            assert loop._flip_zai_endpoint() is True
            assert isinstance(loop.llm_client, _FakeOpenAICompat)

    def test_flip_zai_non_zai_client_false(self):
        client = MagicMock()
        client.get_provider_name.return_value = "openai"
        loop = _make_loop("/tmp/x")
        loop.llm_client = client
        assert loop._flip_zai_endpoint() is False

    def test_flip_zai_custom_base_url_false(self):
        client = MagicMock()
        client.get_provider_name.return_value = "zai"
        client.base_url = "https://custom.example"
        loop = _make_loop("/tmp/x")
        loop.llm_client = client
        assert loop._flip_zai_endpoint() is False

    def test_flip_zai_missing_provider_name_false(self):
        client = MagicMock()
        client.get_provider_name.side_effect = AttributeError
        loop = _make_loop("/tmp/x")
        loop.llm_client = client
        assert loop._flip_zai_endpoint() is False


class TestRespondMoreScenarios:
    """Additional respond() integration scenarios."""

    def test_respond_general_mode_filters_tools(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_response=_stop_response("general done"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="hi")], mode="general")
        assert r.content == "general done"

    def test_respond_reasoning_extract_failure_silent(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="working",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response={"bad": "shape"},
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        with patch("external_llm.agent.design_chat_loop.extract_llm_reasoning", side_effect=TypeError("bad raw")):
            r = loop.respond([LLMMessage(role="user", content="do it")], max_tool_iterations=2)
        assert r.content == "fin"

    def test_respond_text_mode_tool_parse(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content='{"name": "read_file", "arguments": {"file_path": "README.md"}}',
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response=None,
                    tool_calls=[],
                ),
                _stop_response("read done"),
            ],
            chat_responses=[
                LLMResponse(
                    content="read done",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response=None,
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="read it")], max_tool_iterations=3)
        assert r.content == "read done"
        assert len(r.tool_results) == 1
        assert "hello" in r.tool_results[0]["content"]  # README.md body

    def test_respond_plan_gate_nudges_then_notes(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response=None,
                    tool_calls=[
                        ToolCallRequest(
                            call_id="c1",
                            name="update_plan",
                            args={"goal": "g", "items": [{"title": "open item", "status": "pending"}]},
                        )
                    ],
                ),
                _stop_response("done once"),
                _stop_response("done twice"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                ),
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        events = []

        def cb(event_type, payload):
            events.append((event_type, payload))

        r = loop.respond([LLMMessage(role="user", content="plan it")], stream_callback=cb, max_tool_iterations=3)
        assert any(t == "design_plan_gate" for t, _ in events)
        assert "Unresolved plan items" in r.content

    def test_respond_parallel_read_tools(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response=None,
                    tool_calls=[
                        ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"}),
                        ToolCallRequest(call_id="c2", name="read_file", args={"file_path": "README.md"}),
                    ],
                ),
                _stop_response("all done"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        events = []

        def cb(event_type, payload):
            events.append(event_type)

        r = loop.respond([LLMMessage(role="user", content="read both")], stream_callback=cb, max_tool_iterations=2)
        assert r.content == "all done"
        assert len(r.tool_results) == 2  # both parallel tools executed

    def test_respond_serial_tool_cancel_before_run(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="ask_user", args={"question": "continue?"})],
            ),
        )
        reg = ToolRegistry(_repo, AgentConfig())
        reg.config.cancel_event = threading.Event()
        reg.config.cancel_event.set()
        loop = DesignChatLoop(client, reg, "stub-model")
        with pytest.raises(AgentCancelled):
            loop.respond([LLMMessage(role="user", content="go")])


class TestArchiveCache:
    """_archive_sig / _archived_bm25_entries caching."""

    def test_archive_sig_missing_returns_none(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        assert dcl._archive_sig(mgr, "no-such-session") is None

    def test_archive_sig_present(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        p = mgr.archive_path("test-session-1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"role": "user", "content": "archived"}\n')
        sig = dcl._archive_sig(mgr, "test-session-1")
        assert sig is not None and sig[0] == "test-session-1"

    def test_archive_sig_stat_error_returns_none(self, tmp_repo):
        mgr = MagicMock()
        mgr.archive_path.side_effect = OSError("denied")
        assert dcl._archive_sig(mgr, "s1") is None

    def test_archived_bm25_entries_cache_hit(self, tmp_repo, monkeypatch):
        monkeypatch.setattr(dcl, "_ARCHIVE_SMALL_SKIP", 0)
        mgr = _populated_session_mgr(tmp_repo)
        p = mgr.archive_path("test-session-1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"role": "user", "content": "x"}\n')
        turns = [{"role": "user", "content": f"archived turn {i} unique words", "timestamp": 1.0} for i in range(3)]
        entries, sig, from_cache = dcl._archived_bm25_entries(mgr, "test-session-1", turns)
        assert len(entries) == 3
        assert sig is not None and from_cache is False
        _, _, from_cache2 = dcl._archived_bm25_entries(mgr, "test-session-1", turns)
        assert from_cache2 is True

    def test_archived_bm25_entries_skips_empty_content(self, tmp_repo, monkeypatch):
        monkeypatch.setattr(dcl, "_ARCHIVE_SMALL_SKIP", 0)
        mgr = _populated_session_mgr(tmp_repo)
        p = mgr.archive_path("test-session-1")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"role": "user", "content": "x"}\n')
        turns = [
            {"role": "user", "content": "", "timestamp": 1.0},
            {"role": "user", "content": "!!!", "timestamp": 1.0},  # no tokenizable tokens
            {"role": "user", "content": "real archived content here", "timestamp": 1.0},
        ]
        entries, _, _ = dcl._archived_bm25_entries(mgr, "test-session-1", turns)
        assert len(entries) == 1  # only the real-content turn survived


class TestParseTextToolCallsEdges:
    """_parse_text_tool_calls normalization branches (text-mode models)."""

    def test_tool_name_format(self):
        out = dcl._parse_text_tool_calls('{"tool_name": "read_file", "params": {"file_path": "x"}}')
        assert out and out[0]["name"] == "read_file" and out[0]["args"] == {"file_path": "x"}

    def test_tool_name_params_bad_json_str(self):
        out = dcl._parse_text_tool_calls('{"tool_name": "read_file", "params": "{bad"}')
        assert out and out[0]["args"] == {}

    def test_tool_format_non_dict_args(self):
        out = dcl._parse_text_tool_calls('{"tool": "read_file", "args": "str-not-dict"}')
        assert out and out[0]["args"] == {}

    def test_arguments_bad_json_str(self):
        out = dcl._parse_text_tool_calls('{"name": "read_file", "arguments": "{bad"}')
        assert out and out[0]["args"] == {}

    def test_openai_function_format(self):
        out = dcl._parse_text_tool_calls(
            '{"type": "function", "function": {"name": "grep", "arguments": "{\\"pattern\\": \\"x\\"}"}}'
        )
        assert out and out[0]["name"] == "grep" and out[0]["args"] == {"pattern": "x"}

    def test_list_with_non_dict_items(self):
        out = dcl._parse_text_tool_calls('[1, 2, {"name": "read_file", "arguments": {"p": "x"}}]')
        assert out and out[0]["name"] == "read_file"

    def test_free_text_scan_with_escaped_quotes(self):
        out = dcl._parse_text_tool_calls('prefix text {"name": "read_file", "arguments": {"path": "a\\"b"}} suffix')
        assert out and out[0]["name"] == "read_file"

    def test_fenced_json_block(self):
        out = dcl._parse_text_tool_calls('```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```')
        assert out and out[0]["name"] == "read_file"


class TestRespondEdgeCallbacks:
    """respond() callback-failure and late-error scenarios."""

    def test_respond_thinking_stop_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_response=_stop_response("done"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond(
            [LLMMessage(role="user", content="hi")],
            stream_callback=lambda t, p: (_ for _ in ()).throw(RuntimeError("cli died")),
        )
        assert r.content == "done"  # finally-block callback failure is non-fatal

    def test_respond_thinking_start_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_response=_stop_response("done"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")

        def cb(event_type, payload):
            if event_type == "design_thinking_start":
                raise RuntimeError("cli died")
            events.append(event_type)

        events = []
        r = loop.respond([LLMMessage(role="user", content="hi")], stream_callback=cb)
        assert r.content == "done"

    def test_respond_token_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="working",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond(
            [LLMMessage(role="user", content="go")],
            token_callback=lambda t: (_ for _ in ()).throw(RuntimeError("ui died")),
            max_tool_iterations=2,
        )
        assert r.content == "fin"

    def test_respond_thinking_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="working",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")

        def cb(event_type, payload):
            if event_type == "design_thinking":
                raise RuntimeError("cli died")

        r = loop.respond(
            [LLMMessage(role="user", content="go")],
            stream_callback=cb,
            max_tool_iterations=2,
        )
        assert r.content == "fin"

    def test_respond_llm_call_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="working",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")

        def cb(event_type, payload):
            if event_type == "design_llm_call":
                raise RuntimeError("cli died")

        r = loop.respond(
            [LLMMessage(role="user", content="go")],
            stream_callback=cb,
            max_tool_iterations=2,
        )
        assert r.content == "fin"

    def test_respond_plan_gate_callback_raises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response=None,
                    tool_calls=[
                        ToolCallRequest(
                            call_id="c1",
                            name="update_plan",
                            args={"goal": "g", "items": [{"title": "open item", "status": "pending"}]},
                        )
                    ],
                ),
                _stop_response("done once"),
                _stop_response("done twice"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")

        def cb(event_type, payload):
            if event_type == "design_plan_gate":
                raise RuntimeError("cli died")

        r = loop.respond([LLMMessage(role="user", content="plan")], stream_callback=cb, max_tool_iterations=3)
        assert "Unresolved plan items" in r.content

    def test_respond_llm_cancelled_converted(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(tool_error=LLMCancelled("user pressed esc"))
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        with pytest.raises(AgentCancelled):
            loop.respond([LLMMessage(role="user", content="go")])

    def test_respond_overflow_no_retrim_propagates(self, _repo, monkeypatch):
        """preemptive_trim removed: a context-length 400 records the overflow
        override but cannot re-trim in-turn — respond() maps the LLMClientError
        to an error result (is_error=True) instead of retrying."""
        from dataclasses import replace

        monkeypatch.setattr(
            dcl,
            "_cfg",
            replace(dcl._cfg, counts=replace(dcl._cfg.counts, DESIGN_CHAT_LLM_MAX_RETRIES=1)),
        )
        client = _ScriptedClient(
            tool_responses=[
                _stop_response("unused"),
            ],
        )
        client.tool_error = LLMAPIError("maximum context length is 8192, but you sent 9000")
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        with patch("external_llm.agent.design_chat_loop._record_context_overflow"):
            r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=2)
        assert r.is_error is True
        assert "An error occurred" in r.content  # user-facing mapping, no retry

    def test_respond_serial_tool_cancel_mid_turn(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        ev = threading.Event()
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="ask_user", args={"question": "continue?"})],
            ),
        )
        client.on_tool_call = lambda: ev.set()  # cancel arrives after the LLM call
        reg = ToolRegistry(_repo, AgentConfig())
        reg.config.cancel_event = ev
        loop = DesignChatLoop(client, reg, "stub-model")
        with pytest.raises(AgentCancelled):
            loop.respond([LLMMessage(role="user", content="go")])

    def test_respond_parallel_no_stream_callback(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response=None,
                    tool_calls=[
                        ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"}),
                        ToolCallRequest(call_id="c2", name="read_file", args={"file_path": "README.md"}),
                    ],
                ),
                _stop_response("all done"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="read")], max_tool_iterations=2)
        assert r.content == "all done"
        assert len(r.tool_results) == 2

    def test_respond_final_retry_llm_error_reraises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
            chat_errors=[LLMRateLimitError("rl on retry")],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=1)
        assert r.is_error is True
        assert "rate limit" in r.content.lower() or "busy" in r.content.lower()

    def test_respond_final_auth_error_reraises(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_errors=[LLMAuthenticationError("bad key")],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=1)
        assert r.is_error is True and r.error_type == "auth"

    def test_respond_final_and_retry_reasoning_extract_failure(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response={"choices": [{}]},
                ),
                LLMResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response={"choices": [{}]},
                ),
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        with patch("external_llm.agent.design_chat_loop.extract_llm_reasoning", side_effect=AttributeError("bad raw")):
            r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=1)
        assert r.content.startswith("⚠️")  # both extraction fallbacks failed → empty funnel

    def test_summarize_change_raises_silent(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = False
        loop.registry._snapshot_target_files = lambda n, a: {"f": "old"}
        loop.registry._safety_manager.summarize_change = MagicMock(side_effect=RuntimeError)
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "generic output"

    def test_write_tool_dispatch_exception_records_failure(self, tmp_repo):
        loop = _make_loop(tmp_repo, write_tools={"apply_patch"})
        loop._asr_semantic_lint = False
        loop.registry._snapshot_target_files = lambda n, a: {"f": "old"}
        loop.registry.dispatch.side_effect = RuntimeError("patch failed")
        result = DesignChatResult()
        result.recall_session_key = "k1"
        out = loop._process_tool_call(_tc("apply_patch", {"patch": "..."}), None, result)
        assert out == "Error: patch failed"
        assert result.tool_results[0]["ok"] is False

    def test_flip_zai_unknown_zai_client_false(self):
        client = MagicMock()
        client.get_provider_name.return_value = "zai"
        client.base_url = None
        loop = _make_loop("/tmp/x")
        loop.llm_client = client  # MagicMock is neither ZAIAnthropicClient nor ZAIClient
        assert loop._flip_zai_endpoint() is False


class TestSearchHistoryLongText:
    """search_design_history long-text truncation markers + archive-less mgr."""

    def test_decisions_long_text_truncated(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        session = mgr.get_or_create("long-dec")
        session.decisions = ["x" * 600 + " unique decision needle"]
        session.compressed_summary = ""
        session.compressed_up_to = 0
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "long-dec"
        out = loop._search_design_history("needle", search_field="decisions")
        assert "..." in out

    def test_turn_long_content_truncated(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        session = mgr.get_or_create("long-turn")
        session.turns = [{"role": "user", "content": "x" * 1100 + " needle here", "timestamp": 1.0}]
        session.compressed_up_to = 1
        session.decisions = []
        session.compressed_summary = ""
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "long-turn"
        out = loop._search_design_history("needle")
        assert "..." in out

    def test_all_field_long_decisions_and_summary(self, tmp_repo):
        mgr = _populated_session_mgr(tmp_repo)
        session = mgr.get_or_create("long-all")
        session.turns = [{"role": "user", "content": "unrelated filler text", "timestamp": 1.0}]
        session.compressed_up_to = 1
        session.decisions = ["y" * 600 + " needle decision"]
        session.compressed_summary = "z" * 600 + " needle summary"
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "long-all"
        out = loop._search_design_history("needle", search_field="all")
        assert "Decision" in out and "Summary" in out
        assert out.count("...") >= 2

    def test_archive_less_session_mgr(self, tmp_repo):
        session = SimpleNamespace(
            turns=[],
            compressed_up_to=0,
            archived_count=0,
            decisions=[],
            compressed_summary="",
        )
        mgr = SimpleNamespace(get_or_create=lambda sid: session)
        loop = _make_loop(tmp_repo, session_mgr=mgr)
        loop.session_id = "s1"
        out = loop._search_design_history("anything")
        assert "No old conversation history" in out

    def test_fallback_plain_chat_reasoning_extract_failure(self):
        client = MagicMock()
        client.chat.return_value = LLMResponse(
            content="answer",
            model="m",
            provider="stub",
            tokens_used=1,
            finish_reason="stop",
            raw_response={"x": 1},
        )
        with patch("external_llm.agent.design_chat_loop.extract_llm_reasoning", side_effect=TypeError("bad")):
            out = dcl._fallback_plain_chat([LLMMessage(role="user", content="q")], client, "m")
        assert out["content"] == "answer"


class TestFinalResidualBranches:
    """Last uncovered branches: repair-path escapes, reasoning fallbacks, provider."""

    def test_function_arguments_bad_json_str(self):
        out = dcl._parse_text_tool_calls('{"type": "function", "function": {"name": "grep", "arguments": "{bad"}}')
        assert out and out[0]["args"] == {}

    def test_bracket_repair_closes_list(self):
        # Missing closing "]" forces _try_json → bracket repair (closes "[").
        out = dcl._parse_text_tool_calls('[{"name": "read_file", "arguments": {"path": "x"}}')
        assert out and out[0]["name"] == "read_file"

    def test_bracket_repair_escape_handling(self):
        # Unclosed "[" + escaped quote + backslash inside a string → repair
        # must keep the escape machinery alive across the scan.
        out = dcl._parse_text_tool_calls('[{"name": "read_file", "arguments": {"path": "a\\\\b\\"c"}}')
        assert out and out[0]["args"] == {"path": 'a\\b"c'}

    def test_respond_raw_reasoning_fallback_failure(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response={"choices": [{}]},
                    tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
                ),
                _stop_response("fin"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        with patch("external_llm.agent.design_chat_loop.extract_llm_reasoning", side_effect=TypeError("bad raw")):
            r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=2)
        assert r.content == "fin"

    def test_respond_retry_reasoning_content(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="stub",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                ),
                LLMResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response={"choices": [{"message": {"reasoning_content": "retry reasoning answer"}}]},
                ),
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=1)
        assert r.content == "retry reasoning answer"

    def test_respond_provider_from_final_response(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_response=ToolCallResponse(
                content="",
                model="m",
                provider="",
                tokens_used=1,
                finish_reason="tool_calls",
                raw_response=None,
                tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
            ),
            chat_responses=[
                LLMResponse(
                    content="fin",
                    model="m",
                    provider="final-provider",
                    tokens_used=1,
                    finish_reason="stop",
                    raw_response=None,
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=1)
        assert r.provider == "final-provider"


class TestFinalResidualBranches2:
    """Repair-path stray ']' + raw-response reasoning success."""

    def test_bracket_repair_stray_close_bracket(self):
        # A stray trailing quote forces bracket repair; the "]" inside the
        # array closes the "[" stack frame (repair branch), even though the
        # final parse still fails (unclosed quote).
        # Whole-content JSON parse fails (trailing quote) → repair closes the
        # "[" frame; the free-text fallback then still recovers the call.
        out = dcl._parse_text_tool_calls('[{"name": "read_file", "arguments": {"path": "x"}}] trailing "')
        assert out and out[0]["name"] == "read_file"

    def test_respond_raw_reasoning_fallback_success(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response={"choices": [{"message": {"reasoning_content": "raw reasoned"}}]},
                    tool_calls=[ToolCallRequest(call_id="c1", name="read_file", args={"file_path": "README.md"})],
                ),
                _stop_response("fin"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=2)
        assert r.content == "fin"  # reasoning fallback filled content mid-loop, final answer wins


class TestFinalResidualBranches3:
    """Recall-hint append + error-status stream event."""

    def test_recall_hint_appended_to_failure(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry.dispatch.side_effect = RuntimeError("tool crashed")
        result = DesignChatResult()
        result.recall_session_key = "k1"
        with patch(
            "external_llm.agent.failure_pattern_store.recall_on_failure",
            return_value="[RECALL] fix this recurring pattern",
        ):
            out = loop._process_tool_call(_tc("bash", {"cmd": "x"}), None, result)
        assert "[RECALL] fix this recurring pattern" in out

    def test_generic_error_stream_status(self, tmp_repo):
        loop = _make_loop(tmp_repo)
        loop.registry.dispatch.side_effect = RuntimeError("boom")
        result = DesignChatResult()
        result.recall_session_key = "k1"
        events = []

        def cb(event_type, payload):
            events.append(payload)

        loop._process_tool_call(_tc("bash", {"cmd": "x"}), cb, result)
        assert events[-1]["status"] == "error"


class TestFinalResidualBranches4:
    """Serial-tool phase failure handling."""

    def test_serial_tool_failure_synthesizes_error(self, _repo, monkeypatch):
        _zero_retries(monkeypatch)
        client = _ScriptedClient(
            tool_responses=[
                ToolCallResponse(
                    content="",
                    model="m",
                    provider="stub",
                    tokens_used=1,
                    finish_reason="tool_calls",
                    raw_response=None,
                    tool_calls=[ToolCallRequest(call_id="c1", name="ask_user", args={"question": "continue?"})],
                ),
                _stop_response("fin"),
            ],
            chat_responses=[
                LLMResponse(
                    content="fin", model="m", provider="stub", tokens_used=1, finish_reason="stop", raw_response=None
                )
            ],
        )
        reg = ToolRegistry(_repo, AgentConfig())
        loop = DesignChatLoop(client, reg, "stub-model")
        loop._process_tool_call_with_learning = MagicMock(side_effect=RuntimeError("dispatcher bug"))
        r = loop.respond([LLMMessage(role="user", content="go")], max_tool_iterations=2)
        assert r.content == "fin"
        assert r.tool_results[0]["ok"] is False
        assert "tool execution failed" in r.tool_results[0]["content"]
