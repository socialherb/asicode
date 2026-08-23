"""grep's cancel channel must observe the per-call scope, not just ESC.

``_tool_grep`` handed ``_run_search_bounded`` a predicate reading only
``config.cancel_event`` — the whole-turn ESC. A search abandoned by ITS
caller (MCP ``wait_for`` timeout, aborted ``dispatch_parallel`` batch) sets a
per-call scope event the poll never saw, so the rg/grep process group kept
walking the tree unowned up to the 120 s bound.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from external_llm.agent.cancel_scope import call_cancel_scope
from external_llm.agent.tool_handlers.read_tools import ReadToolsMixin, SearchCancelled


class _Host(ReadToolsMixin):
    """Duck-typed host: just what ``_tool_grep`` touches."""

    def __init__(self, root: str, config):
        self.repo_root = root
        self.config = config

    # Registry-level helpers the mixin borrows from its host (ToolRegistry).
    def _correct_bias_path(self, p):
        return p

    def _secure_path(self, p):
        return p

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


@pytest.fixture
def host(tmp_path):
    return _Host(str(tmp_path), types.SimpleNamespace(cancel_event=threading.Event()))


def test_grep_cancel_predicate_observes_per_call_scope(host, monkeypatch):
    """The predicate ``_tool_grep`` hands ``_run_search_bounded`` must OR the
    live per-call scope in — an abandoned dispatch stops an already-running
    search, not just a whole-turn ESC."""
    captured = {}

    def _fake_run(cmd, cwd, timeout, retain, *, cancelled=None, max_line_chars=None):
        captured["cancelled"] = cancelled
        return 0, [], 0, ""

    monkeypatch.setattr(host, "_run_search_bounded", _fake_run)

    scope_ev = threading.Event()
    scope_ev.set()  # caller abandoned the dispatch; config.cancel_event stays unset
    with call_cancel_scope(scope_ev):
        res = host._tool_grep({"pattern": "needle"})
        # The predicate reads its sources at CALL time (fresh per poll) — so
        # it must be probed while the scope is still installed.
        tripped = captured["cancelled"]()

    assert res["ok"]  # the search itself "succeeded"; only the wiring is under test
    assert captured["cancelled"] is not None
    assert tripped, "scope set → the search poll must observe cancel"
    # And it is a LIVE read, not a snapshot: unwound scope → inert again.
    assert not captured["cancelled"]()


def test_search_cancelled_mid_run_when_scope_set(host):
    """End-to-end: a per-call scope set MID-search raises SearchCancelled and
    tears the process group down promptly, instead of running to completion
    (or the 120 s bound) with the abandon unobserved."""
    scope_ev = threading.Event()

    def _abandon():
        time.sleep(0.4)
        scope_ev.set()

    threading.Thread(target=_abandon, daemon=True).start()
    t0 = time.monotonic()
    with call_cancel_scope(scope_ev), pytest.raises(SearchCancelled):
        host._run_search_bounded(
            ["sh", "-c", "echo one; sleep 10"],
            host.repo_root,
            60,
            100,
            cancelled=host._search_cancel_requested,
        )
    assert time.monotonic() - t0 < 5.0, "mid-run abandon must not wait out the child"
