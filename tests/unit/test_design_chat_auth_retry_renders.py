"""The design-chat auth retry must not swallow the answer it just paid for.

Regression: the retry lived *inside* ``if chat_result.is_error:`` and rebound
``chat_result`` to a fresh, successful result — but every renderer sits in that
statement's ``else:`` branch, which is not re-entered. So a retry ran a whole
tool loop, spent real tokens, recorded the assistant turn in session history,
and then displayed nothing; the REPL dropped back to the prompt with no final
summary and no ``/copy`` target.

The structure is inside ``run_repl`` (a several-thousand-line function driven by
an interactive prompt), so this is pinned as an AST contract rather than by
executing the loop: the retry must be reachable from OUTSIDE the branch whose
``else:`` owns the renderer.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPL = pathlib.Path(__file__).resolve().parents[2] / "external_llm" / "repl" / "repl_impl.py"

RENDER_MARKER = "_split_work_state"      # first thing the render branch does
RETRY_MARKER = "_prompt_auth_retry_key"  # entry point of the auth retry


def _calls_named(node: ast.AST, name: str) -> bool:
    """True if *name* is called anywhere under *node*."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name) and fn.id == name:
                return True
            if isinstance(fn, ast.Attribute) and fn.attr == name:
                return True
    return False


def _mentions(node: ast.AST, name: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == name:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == name:
            return True
    return False


@pytest.fixture(scope="module")
def render_ifs():
    """Every ``if`` whose else-branch contains the design-chat renderer."""
    tree = ast.parse(REPL.read_text(encoding="utf-8"))
    found = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and n.orelse
        and any(_mentions(s, RENDER_MARKER) for s in n.orelse)
    ]
    assert found, (
        f"no if/else owning {RENDER_MARKER!r} found — the design-chat render "
        "structure moved; update this contract test rather than deleting it"
    )
    return found


def test_auth_retry_is_not_trapped_in_the_non_render_branch(render_ifs):
    for node in render_ifs:
        for stmt in node.body:
            assert not _calls_named(stmt, RETRY_MARKER), (
                f"{RETRY_MARKER} is nested in the branch whose else: renders the "
                f"answer (repl_impl.py:{node.lineno}). A successful retry rebinds "
                "chat_result but can never reach the renderer, so its answer is "
                "computed and silently discarded. Run the retry BEFORE this "
                "if/else so the fresh result takes the normal path."
            )


def test_retry_result_can_still_reach_the_renderer():
    """The retry must precede the render branch in the same statement list."""
    tree = ast.parse(REPL.read_text(encoding="utf-8"))
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        retry_at = render_at = None
        for i, stmt in enumerate(body):
            if retry_at is None and _calls_named(stmt, RETRY_MARKER):
                retry_at = i
            if render_at is None and isinstance(stmt, ast.If) and stmt.orelse \
                    and any(_mentions(s, RENDER_MARKER) for s in stmt.orelse):
                render_at = i
        if retry_at is not None and render_at is not None:
            assert retry_at < render_at, (
                "the auth retry runs after the render branch — a recovered "
                "answer would never be displayed"
            )
            return
    pytest.fail(
        "auth retry and the render if/else are not siblings; the retry cannot "
        "feed the renderer (see this module's docstring)"
    )


def test_original_error_is_not_printed_twice_when_the_prompt_is_skipped():
    """Skipping the key prompt must not double-report the same error.

    Moving the retry out of the error branch means both blocks can print; the
    guard flag is what keeps a skipped prompt to a single message.
    """
    src = REPL.read_text(encoding="utf-8")
    assert "_dc_error_shown" in src, (
        "the retry was moved out of the error branch without the "
        "double-print guard — a skipped prompt now reports the same error twice"
    )
