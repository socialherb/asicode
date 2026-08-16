"""Contract tests for agent_tools.py (AgentToolsMixin + _read_local_snippet).

Covers: _read_local_snippet streaming window, _tool_update_memory, the
delegate_to_helper dispatch surface, and the ask_user guard branches
(disabled / limit / no-callback / callback exception) that the checkpoint
contract tests do not reach.
"""
from __future__ import annotations

import threading
import types

from external_llm.agent.tool_handlers.agent_tools import (
    AgentToolsMixin,
    _read_local_snippet,
)
from external_llm.agent.tool_registry import AgentConfig


class _Host(AgentToolsMixin):
    """Minimal mixin host exposing only what the handlers touch."""

    def __init__(self, repo_root=".", local_assistant=None, config=None):
        self.repo_root = repo_root
        self.local_assistant = local_assistant
        self.config = config or AgentConfig()

    def _make_result(self, **kwargs):
        fields = {
            "ok": False,
            "content": "",
            "error": None,
            "metadata": {},
            "execution_time": 0.0,
            "partial_failure": False,
            "retryable": True,
            "retry_count": 0,
        }
        fields.update(kwargs)
        return types.SimpleNamespace(**fields)

    def _ensure_asicode_gitignored(self):
        return None


# ---------------------------------------------------------------------------
# _read_local_snippet
# ---------------------------------------------------------------------------


def test_read_local_snippet_streams_numbered_window(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
    text = _read_local_snippet(f, 3, 6)
    lines = text.split("\n")
    assert lines[0].startswith("3: line 3")
    assert lines[-1].startswith("6: line 6")
    assert len(lines) == 4


def test_read_local_snippet_bounds_window_to_requested(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line {i}" for i in range(1, 5000)), encoding="utf-8")
    text = _read_local_snippet(f, 1, 80)
    lines = text.split("\n")
    assert len(lines) == 80
    assert lines[0].startswith("1: line 1")
    assert lines[-1].startswith("80: line 80")


# ---------------------------------------------------------------------------
# _tool_update_memory
# ---------------------------------------------------------------------------


def test_update_memory_requires_note(tmp_path):
    host = _Host(repo_root=str(tmp_path))
    res = host._tool_update_memory({"section": "s"})
    assert res.ok is False
    assert "'note' is required" in res.error


def test_update_memory_appends_section_note(tmp_path):
    host = _Host(repo_root=str(tmp_path))
    res = host._tool_update_memory({"note": "hello", "section": "design"})
    assert res.ok is True
    memory = (tmp_path / ".asicode" / "memory.md").read_text(encoding="utf-8")
    assert "hello" in memory
    assert "### design (" in memory
    assert res.metadata["path"] == ".asicode/memory.md"
    assert res.metadata["section"] == "design"


def test_update_memory_no_section_uses_comment_marker(tmp_path):
    host = _Host(repo_root=str(tmp_path))
    res = host._tool_update_memory({"note": "plain note"})
    assert res.ok is True
    memory = (tmp_path / ".asicode" / "memory.md").read_text(encoding="utf-8")
    assert "<!-- " in memory
    assert "plain note" in memory


def test_update_memory_truncates_long_note(tmp_path):
    host = _Host(repo_root=str(tmp_path))
    res = host._tool_update_memory({"note": "x" * 5000})
    assert res.ok is True
    memory = (tmp_path / ".asicode" / "memory.md").read_text(encoding="utf-8")
    assert len(memory.strip()) <= 1000 + 40  # note cap + marker


def test_update_memory_failure_returns_error(tmp_path, monkeypatch):
    host = _Host(repo_root=str(tmp_path))

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)
    res = host._tool_update_memory({"note": "n"})
    assert res.ok is False
    assert "Failed to update memory" in res.error


# ---------------------------------------------------------------------------
# _tool_delegate_to_helper
# ---------------------------------------------------------------------------


class _FakeLocalAssistant:
    def __init__(self, result=None, exc=None):
        self._result = result or {}
        self._exc = exc
        self.last_kwargs = None

    def delegate_single_task(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._result


def test_delegate_requires_local_assistant():
    host = _Host(local_assistant=None)
    res = host._tool_delegate_to_helper({"role": "r", "instruction": "i"})
    assert res.ok is False
    assert "Helper not available" in res.error


def test_delegate_requires_role():
    host = _Host(local_assistant=_FakeLocalAssistant())
    res = host._tool_delegate_to_helper({"instruction": "i"})
    assert res.ok is False
    assert "'role' is required" in res.error


def test_delegate_requires_instruction():
    host = _Host(local_assistant=_FakeLocalAssistant())
    res = host._tool_delegate_to_helper({"role": "r"})
    assert res.ok is False
    assert "'instruction' is required" in res.error


def test_delegate_success_with_direct_context():
    la = _FakeLocalAssistant(
        result={
            "success": True,
            "language": "python",
            "code": "def f(): pass",
            "issues": ["nit"],
            "validation": {"ok": True},
        }
    )
    host = _Host(local_assistant=la)
    res = host._tool_delegate_to_helper(
        {
            "role": "implementer",
            "instruction": "write f",
            "function_signature": "def f():",
            "context_code": "class A: pass",
        }
    )
    assert res.ok is True
    assert "def f(): pass" in res.content
    assert "Issues noted" in res.content
    assert res.metadata["role"] == "implementer"
    assert la.last_kwargs["function_signature"] == "def f():"
    assert la.last_kwargs["context_code"] == "class A: pass"


def test_delegate_failure_reports_error():
    la = _FakeLocalAssistant(result={"success": False, "error": "model refused"})
    host = _Host(local_assistant=la)
    res = host._tool_delegate_to_helper(
        {
            "role": "r",
            "instruction": "i",
            "function_signature": "def f():",
            "context_code": "x = 1",
        }
    )
    assert res.ok is False
    assert "model refused" in res.error


def test_delegate_exception_reports_error():
    la = _FakeLocalAssistant(exc=RuntimeError("sdk down"))
    host = _Host(local_assistant=la)
    res = host._tool_delegate_to_helper(
        {
            "role": "r",
            "instruction": "i",
            "function_signature": "def f():",
            "context_code": "x = 1",
        }
    )
    assert res.ok is False
    assert "sdk down" in res.error


def test_delegate_builds_context_from_target_symbol(tmp_path, monkeypatch):
    """function_signature omitted + target_symbol set -> symbol search fill."""
    import external_llm.agent.symbol_search as ss

    class _FakeSearcher:
        def get_symbol_info(self, name, file_path=None):
            return {"signature": "def target():", "line": 10}

    class _FakeBuilder:
        def __init__(self, repo_root):
            self.repo_root = repo_root

        def build(self, **kwargs):
            return types.SimpleNamespace(content="## Context\nhelper code")

    monkeypatch.setattr(ss, "get_symbol_searcher", lambda root: _FakeSearcher())
    monkeypatch.setattr(
        "external_llm.context.context_packs.HelperContextBuilder", _FakeBuilder
    )
    la = _FakeLocalAssistant(result={"success": True, "code": "ok"})
    host = _Host(repo_root=str(tmp_path), local_assistant=la)
    res = host._tool_delegate_to_helper(
        {
            "role": "r",
            "instruction": "i",
            "target_symbol": "target",
            "file_path": "mod.py",
        }
    )
    assert res.ok is True
    assert la.last_kwargs["function_signature"] == "def target():"
    assert "helper code" in la.last_kwargs["context_code"]


def test_delegate_builds_context_from_file_snippet(tmp_path, monkeypatch):
    """context_code omitted + file_path set -> local snippet read and handed
    to the builder, which wraps it into the pack content -> context_code."""
    f = tmp_path / "mod.py"
    f.write_text("\n".join(f"line {i}" for i in range(1, 30)), encoding="utf-8")

    seen = {}

    class _FakeBuilder:
        def __init__(self, repo_root):
            pass

        def build(self, **kwargs):
            seen["local_snippet"] = kwargs.get("local_snippet")
            # builder wraps the snippet into pack content (its real contract)
            return types.SimpleNamespace(content=f"## Context\n{kwargs.get('local_snippet')}")

    monkeypatch.setattr(
        "external_llm.context.context_packs.HelperContextBuilder", _FakeBuilder
    )
    la = _FakeLocalAssistant(result={"success": True, "code": "ok"})
    host = _Host(repo_root=str(tmp_path), local_assistant=la)
    res = host._tool_delegate_to_helper(
        {"role": "r", "instruction": "i", "file_path": "mod.py"}
    )
    assert res.ok is True
    assert "1: line 1" in seen["local_snippet"]
    assert "1: line 1" in la.last_kwargs["context_code"]


def test_delegate_context_build_failure_is_suppressed(tmp_path, monkeypatch):
    """Builder raising -> context build skipped, delegation still proceeds."""
    import external_llm.agent.symbol_search as ss

    def _raise(*a, **k):
        raise RuntimeError("searcher unavailable")

    monkeypatch.setattr(ss, "get_symbol_searcher", _raise)
    la = _FakeLocalAssistant(result={"success": True, "code": "ok"})
    host = _Host(repo_root=str(tmp_path), local_assistant=la)
    res = host._tool_delegate_to_helper(
        {"role": "r", "instruction": "i", "target_symbol": "x"}
    )
    # outer try/except -> context build failure logged, delegation proceeds
    assert res.ok is True


# ---------------------------------------------------------------------------
# _tool_ask_user guard branches
# ---------------------------------------------------------------------------


def test_ask_user_requires_question():
    host = _Host()
    res = host._tool_ask_user({})
    assert res.ok is False
    assert "'question' is required" in res.error


def test_ask_user_disabled_uses_default():
    cfg = AgentConfig(user_checkpoint_enabled=False)
    host = _Host(config=cfg)
    res = host._tool_ask_user({"question": "q", "default": "dflt"})
    assert res.ok is True
    assert res.metadata["status"] == "disabled"
    assert res.metadata["answer"] == "dflt"


def test_ask_user_no_callback_uses_default():
    cfg = AgentConfig(user_checkpoint_enabled=True, user_checkpoint_callback=None)
    host = _Host(config=cfg)
    res = host._tool_ask_user({"question": "q", "default": "dflt"})
    assert res.ok is True
    assert res.metadata["status"] == "no_callback"
    assert res.metadata["answer"] == "dflt"


def test_ask_user_question_limit_reached():
    cfg = AgentConfig(
        user_checkpoint_enabled=True,
        user_checkpoint_max_questions=1,
        user_checkpoint_callback=lambda qd: {"status": "answered", "answer": "a"},
    )
    cfg._user_checkpoint_count = 1
    host = _Host(config=cfg)
    res = host._tool_ask_user({"question": "q", "default": "dflt"})
    assert res.ok is True
    assert res.metadata["status"] == "limit_reached"
    assert res.metadata["answer"] == "dflt"


def test_ask_user_callback_exception_uses_default():
    def _boom(qd):
        raise RuntimeError("callback failed")

    cfg = AgentConfig(user_checkpoint_enabled=True, user_checkpoint_callback=_boom)
    host = _Host(config=cfg)
    res = host._tool_ask_user({"question": "q", "default": "dflt"})
    assert res.ok is True
    assert res.metadata["status"] == "error"
    assert res.metadata["answer"] == "dflt"


def test_ask_user_cancel_event_not_consulted():
    """ask_user does not check cancel_event; it blocks on the callback."""
    cfg = AgentConfig(user_checkpoint_enabled=True)
    cfg.cancel_event = threading.Event()
    cfg.cancel_event.set()
    cfg.user_checkpoint_callback = lambda qd: {"status": "answered", "answer": "still-asks"}
    host = _Host(config=cfg)
    res = host._tool_ask_user({"question": "q"})
    assert res.metadata["status"] == "answered"
