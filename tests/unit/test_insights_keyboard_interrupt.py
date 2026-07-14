"""Regression: Ctrl+C during ``/insights compact`` / ``/insights verify`` must
not crash the REPL with a raw traceback.

Root cause (locked in here): ``except Exception`` does NOT catch
``KeyboardInterrupt`` (a ``BaseException`` subclass). Both insights closures
(``_compact_insights_interactive`` / ``_verify_insights_interactive``) document a
"never raises" contract yet previously relied solely on ``except Exception``
around their blocking LLM / agent-tool-loop calls — so a Ctrl+C during the call
escaped the closure, propagated out of the REPL command dispatch (which only
wraps *input* collection in try/except, not command execution), and crashed the
whole process with a raw traceback.

These tests lock in BOTH the language invariant and the presence of the
``except KeyboardInterrupt`` handler in each closure, so a future refactor that
removes the handler — e.g. on the mistaken belief that ``except Exception``
covers it — is caught at test time.
"""
import ast
from pathlib import Path

import asi


# ═══════════════════════════════════════════════════════════════
# Language invariant — the load-bearing reason the fix is needed
# ═══════════════════════════════════════════════════════════════

def test_except_exception_does_not_catch_keyboard_interrupt():
    """If this ever becomes False, the except-keyboardinterrupt handlers become
    dead code and can be removed. Until then they are load-bearing."""
    def _inner():
        try:
            raise KeyboardInterrupt
        except Exception:  # noqa: B902 — the whole point of the test
            return "caught-by-exception"

    try:
        _inner()
    except KeyboardInterrupt:
        return  # expected — KeyboardInterrupt escaped except Exception
    raise AssertionError(
        "KeyboardInterrupt was caught by `except Exception` — the "
        "except-keyboardinterrupt handlers in the insights closures are no "
        "longer needed"
    )


# ═══════════════════════════════════════════════════════════════
# Fix pattern — except KeyboardInterrupt + finally still runs
# ═══════════════════════════════════════════════════════════════

def test_keyboard_interrupt_handler_runs_finally_and_returns_false():
    """Mirror the exact try/except Exception/except KeyboardInterrupt/finally
    shape used by the two insights closures. Verifies that the finally cleanup
    (spinner stop / filter removal) still runs and the function returns False
    (honoring the 'never raises' contract) instead of propagating."""
    log = []

    def _compact_like():
        notice = None
        try:
            raise KeyboardInterrupt
        except Exception:
            notice = "exception-path"
        except KeyboardInterrupt:
            notice = "cancelled"
        finally:
            log.append("finally-ran")
        log.append(("notice", notice))
        return False

    assert _compact_like() is False
    assert "finally-ran" in log, "finally cleanup must run before return"
    assert ("notice", "cancelled") in log


# ═══════════════════════════════════════════════════════════════
# Source-level mutation guard — both closures must keep the handler
# ═══════════════════════════════════════════════════════════════

_TREE = ast.parse(Path(asi.__file__).read_text())


def _find_func(tree, name):
    """Find a (possibly nested) function def by name via full AST walk."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _has_keyboard_interrupt_except(func_node):
    """True if the function body contains an ``except KeyboardInterrupt`` handler
    (covers both ``except KeyboardInterrupt:`` and ``except (..., KI):``)."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            t = node.type
            if isinstance(t, ast.Name) and t.id == "KeyboardInterrupt":
                return True
            if isinstance(t, ast.Tuple) and any(
                isinstance(e, ast.Name) and e.id == "KeyboardInterrupt" for e in t.elts
            ):
                return True
    return False


def test_compact_insights_interactive_has_keyboard_interrupt_handler():
    func = _find_func(_TREE, "_compact_insights_interactive")
    assert func is not None, "_compact_insights_interactive not found in asi"
    assert _has_keyboard_interrupt_except(func), (
        "_compact_insights_interactive must catch KeyboardInterrupt around its "
        "blocking LLM call — otherwise Ctrl+C escapes the 'never raises' contract "
        "and crashes the REPL with a raw traceback."
    )


def test_verify_insights_interactive_has_keyboard_interrupt_handler():
    func = _find_func(_TREE, "_verify_insights_interactive")
    assert func is not None, "_verify_insights_interactive not found in asi"
    assert _has_keyboard_interrupt_except(func), (
        "_verify_insights_interactive must catch KeyboardInterrupt around its "
        "blocking agent-tool-loop call — otherwise Ctrl+C escapes the 'never "
        "raises' contract and crashes the REPL with a raw traceback."
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
