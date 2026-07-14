"""Regression guard for the design_insights auto-compaction 401-spam bug.

``_compact_insights_interactive`` (asi.py) makes a synchronous LLM call to
the helper model. Without suppression, a persistent helper-model auth/quota
failure surfaces the provider's raw ``logger.error("DeepSeek authentication
failed (401)")`` on the terminal every time compaction runs (and it re-runs
each turn the insights file is over budget). The background-compress path
(context_manager.compress_old_turns) already suppresses this via
``_SuppressInfoFilter`` + routes a single user-facing notice via
``_compress_failure_notice``; this test pins the SAME contract on the
synchronous insights-compaction path.

These are source-contract tests (inspect.getsource) — they verify the wiring
exists without importing asi (which has heavy import-time side effects).
"""
import logging

import pytest

from external_llm.agent import context_manager as cm


def _get_compact_insights_source() -> str:
    """Extract ``_compact_insights_interactive`` source from asi.py.

    asi.py is a large script with import-time side effects (it constructs
    services, reads .env, etc.), so we parse the source text instead of
    importing. The function is a closure defined inside ``run_repl``; we locate
    it by its ``def`` line and balance-indent the body.
    """
    import re
    src_path = "asi.py"
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Find the def line
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^    def _compact_insights_interactive\(\) -> bool:", ln):
            start = i
            break
    if start is None:
        pytest.skip("_compact_insights_interactive not found in asi.py")
    # Collect until the next top-level/statement at the same or lower indent
    # that isn't part of the body. The function is indented 4 spaces; its body
    # is indented 8+. We stop at the first line at indent <= 4 that follows a
    # blank or the body (i.e. the next sibling def/statement).
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        # A line at indent <= 4 (and non-blank, non-comment-only) ends the func.
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("\t"):
            # but allow decorators/continuation — a def at 4-space indent ends it
            if re.match(r"^    def ", ln) or re.match(r"^def ", ln) or re.match(r"^    [a-zA-Z_]", ln):
                break
        body.append(ln)
    return "".join(body)


class TestInsightsCompactSuppress:
    def test_suppress_filter_installed_on_external_llm_logger(self):
        """The provider logger.error must be suppressed during the LLM call.

        providers.py emits ``logger = getLogger(__name__)`` where __name__ ==
        "external_llm", so the filter MUST attach to ``logging.getLogger(
        "external_llm")`` (the parent), not to a child logger. See
        context_manager.py:665-674 for the same rationale in the background path.
        """
        src = _get_compact_insights_source()
        assert 'logging.getLogger("external_llm")' in src, (
            "filter must attach to the 'external_llm' parent logger to intercept "
            "providers.py's logger.error"
        )
        assert "addFilter" in src, "filter must be added before the LLM call"
        assert "removeFilter" in src, "filter must be removed in finally"

    def test_uses_canonical_suppress_filter_and_notice_helper(self):
        """Must reuse the SAME helpers as the background path — single source of truth."""
        src = _get_compact_insights_source()
        assert "_SuppressInfoFilter" in src, (
            "must import _SuppressInfoFilter from context_manager (reuse, not re-implement)"
        )
        assert "_compress_failure_notice" in src, (
            "must import _compress_failure_notice for the one-shot user-facing notice"
        )

    def test_filter_removed_in_finally(self):
        """The filter MUST be removed even on exception — else it leaks and suppresses
        ALL subsequent provider logging (not just during this call)."""
        src = _get_compact_insights_source()
        # The removeFilter call must be inside a finally block.
        assert "finally:" in src, "filter removal must be in a finally block"
        # Find the finally block and confirm removeFilter is within it.
        finally_idx = src.index("finally:")
        remove_idx = src.index("removeFilter")
        assert remove_idx > finally_idx, "removeFilter must come after finally:"

    def test_failure_notice_routed_on_exception(self):
        """On LLM failure, a one-shot user-facing notice must be routed (not the raw
        provider error). The notice is keyed by failure class so auth/quota/rate
        each get one notice, then stay silent."""
        src = _get_compact_insights_source()
        assert "_compress_failure_notice(" in src, (
            "except block must call _compress_failure_notice to route the notice"
        )
        # The original exception must be passed through (not re-wrapped), because
        # _compress_failure_notice branches on isinstance(exc, LLMAuthenticationError).
        assert "_cie" in src, "the caught exception must be passed to the notice helper"

    def test_notice_printed_before_generic_fallback_message(self):
        """When a notice is available (auth/quota/rate), print IT instead of the
        generic 'compaction failed (LLM call error)' — the generic message hides
        the actionable cause (bad helper key)."""
        src = _get_compact_insights_source()
        assert "_ci_notice" in src, "notice variable must be threaded to the output path"
        assert 'if _ci_notice:' in src or 'if _ci_notice :' in src, (
            "notice must be checked before the generic fallback message"
        )

    def test_generic_fallback_skipped_when_notice_present(self):
        """The generic 'compaction failed' message must NOT be printed when a
        specific notice was already shown (avoids double-messaging)."""
        src = _get_compact_insights_source()
        # The generic message must be guarded by `if not _ci_notice`
        assert "if not _ci_notice" in src, (
            "generic fallback message must be skipped when a specific notice was shown"
        )


class TestSuppressFilterBehavior:
    """Pin the _SuppressInfoFilter contract: suppress ALL levels (ERROR included)."""

    def test_filter_suppresses_error_level(self):
        """The filter must return False for ERROR records — this is the exact
        record type providers.py emits on 401."""
        f = cm._SuppressInfoFilter()
        record = logging.LogRecord(
            name="external_llm", level=logging.ERROR, pathname="x", lineno=1,
            msg="DeepSeek authentication failed (401)", args=(), exc_info=None,
        )
        assert f.filter(record) is False, "ERROR records must be suppressed"

    def test_filter_suppresses_warning_level(self):
        f = cm._SuppressInfoFilter()
        record = logging.LogRecord(
            name="external_llm", level=logging.WARNING, pathname="x", lineno=1,
            msg="warn", args=(), exc_info=None,
        )
        assert f.filter(record) is False

    def test_filter_suppresses_info_level(self):
        f = cm._SuppressInfoFilter()
        record = logging.LogRecord(
            name="external_llm", level=logging.INFO, pathname="x", lineno=1,
            msg="info", args=(), exc_info=None,
        )
        assert f.filter(record) is False
