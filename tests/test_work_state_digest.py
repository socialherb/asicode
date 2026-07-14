"""
Tests for work_state_digest.py — deterministic turn work-state digest,
plus its integration with DesignSessionManager / build_context_messages.
"""
from __future__ import annotations

import pytest

from external_llm.agent.work_state_digest import build_work_state_digest


def _entry(tool, args=None, content="", ok=True):
    return {"tool": tool, "args": args or {}, "content": content, "ok": ok}


# ── build_work_state_digest ─────────────────────────────────────────────────

class TestBuildWorkStateDigest:
    def test_empty_input(self):
        assert build_work_state_digest([]) == ""
        assert build_work_state_digest(None) == ""

    def test_reads_deduped_in_order(self):
        digest = build_work_state_digest([
            _entry("read_file", {"path": "a.py"}),
            _entry("read_file", {"path": "b.py"}),
            _entry("read_file", {"path": "a.py"}),
        ])
        assert digest == "read: a.py, b.py"

    def test_read_symbol_includes_symbol_name(self):
        digest = build_work_state_digest([
            _entry("read_symbol", {"path": "mod.py", "name": "MyClass"}),
        ])
        assert "mod.py:MyClass" in digest

    def test_writes_include_tool_and_failure_marker(self):
        digest = build_work_state_digest([
            _entry("edit_text", {"path": "c.py"}),
            _entry("apply_patch", {"path": "d.py"}, content="Error: rejected", ok=False),
        ])
        assert "modified: c.py (edit_text), d.py (apply_patch FAILED)" in digest
        assert "failed: apply_patch d.py — Error: rejected" in digest

    def test_commands_show_excerpt_and_status(self):
        digest = build_work_state_digest([
            _entry("bash", {"command": "git status"}),
            _entry("run_tests", {}, content="2 failed", ok=False),
        ])
        assert "ran: bash: git status → ok; run_tests → FAILED" in digest

    def test_long_command_truncated(self):
        digest = build_work_state_digest([
            _entry("bash", {"command": "x" * 200}),
        ])
        assert "x" * 60 + "…" in digest
        assert "x" * 61 not in digest

    def test_searches_include_query(self):
        digest = build_work_state_digest([
            _entry("grep", {"pattern": "TODO"}),
            _entry("find_symbol", {"name": "foo"}),
        ])
        assert "searched: grep TODO; find_symbol foo" in digest

    def test_ignored_tools_produce_empty_digest(self):
        digest = build_work_state_digest([
            _entry("ask_user", {"question": "?"}),
            _entry("save_insight", {"insight": "x"}),
        ])
        assert digest == ""

    def test_unknown_tool_listed_by_name(self):
        digest = build_work_state_digest([_entry("brand_new_tool", {"x": 1})])
        assert "other tools: brand_new_tool" in digest

    def test_failure_error_first_line_truncated(self):
        long_err = "E" * 300 + "\nsecond line"
        digest = build_work_state_digest([
            _entry("edit_ast", {"path": "e.py"}, content=long_err, ok=False),
        ])
        assert "second line" not in digest
        assert "E" * 120 + "…" in digest

    def test_section_cap_with_overflow_marker(self):
        entries = [_entry("read_file", {"path": f"f{i}.py"}) for i in range(15)]
        digest = build_work_state_digest(entries)
        assert "f9.py" in digest
        assert "f10.py" not in digest.replace("(+5 more)", "")
        assert "(+5 more)" in digest

    def test_malformed_entries_do_not_crash(self):
        digest = build_work_state_digest([
            None, "garbage", {}, {"tool": "read_file", "args": "not-a-dict"},
            _entry("read_file", {"path": "ok.py"}),
        ])
        assert "ok.py" in digest

    def test_deterministic(self):
        entries = [
            _entry("read_file", {"path": "a.py"}),
            _entry("edit_text", {"path": "a.py"}),
            _entry("bash", {"command": "pytest"}),
        ]
        assert build_work_state_digest(entries) == build_work_state_digest(entries)


# ── Session integration ─────────────────────────────────────────────────────

class TestSessionDigestIntegration:
    @pytest.fixture
    def mgr(self, tmp_path):
        from external_llm.design_session import DesignSessionManager
        return DesignSessionManager(str(tmp_path))

    def test_add_turn_stores_and_persists_digest(self, mgr, tmp_path):
        mgr.add_turn("s1", "user", "fix the bug")
        mgr.add_turn("s1", "assistant", "fixed", digest="modified: a.py (edit_text)")

        # round-trip: clear cache, reload from disk
        mgr._cache.clear()
        session = mgr.get_or_create("s1")
        assert session.turns[1]["digest"] == "modified: a.py (edit_text)"
        assert "digest" not in session.turns[0]

    def test_context_injects_work_state_for_assistant_turn(self, mgr):
        mgr.add_turn("s2", "user", "fix the bug")
        mgr.add_turn("s2", "assistant", "fixed", digest="read: a.py\nmodified: a.py (edit_text)")
        mgr.add_turn("s2", "user", "now do b")

        session = mgr.get_or_create("s2")
        messages = mgr.build_context_messages(session, skip_core_prompt=True)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "[WORK STATE — tools used in this turn]" in assistant_msgs[0]["content"]
        assert "modified: a.py (edit_text)" in assistant_msgs[0]["content"]
        # user turns never get a digest block
        for m in messages:
            if m["role"] == "user":
                assert "[WORK STATE" not in m["content"]

    def test_context_rendering_is_byte_stable(self, mgr):
        mgr.add_turn("s3", "user", "q1")
        mgr.add_turn("s3", "assistant", "a1", digest="read: x.py")
        session = mgr.get_or_create("s3")
        first = mgr.build_context_messages(session, skip_core_prompt=True)
        second = mgr.build_context_messages(session, skip_core_prompt=True)
        assert first == second
