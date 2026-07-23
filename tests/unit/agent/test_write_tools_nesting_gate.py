"""Block-introducer AST nesting gate — split out for PUBLIC coverage.

This class was previously inside ``test_write_tools_bugfixes.py``, which also
imported an *excluded* editor symbol-handler module at module scope. That
import matches the ``_COUPLED_TEST_PAT`` in ``scripts/export_public.py``, so
the *entire* file (including these gate tests) was silently dropped from the
public release snapshot.

The nesting gate is pure-Python (``ast`` only) and has no dependency on any
excluded module, so splitting it into its own file lets it ship publicly. This
file imports ONLY from ``external_llm.agent.tool_handlers.write_tools`` and
``external_llm.agent.tool_registry`` — both of which are public modules.
"""

import pytest

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_registry import ToolResult


# ── Minimal test harnesses (copies of the ones in test_write_tools_bugfixes;
# duplicated here to keep this file free of the operation_handlers import) ──


class _Harness(WriteToolsMixin):
    """Minimal concrete host for WriteToolsMixin handlers."""

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._repo_root_override = None
        self._applied_patches = []

    @property
    def _effective_repo_root(self):
        return self.repo_root

    def _make_result(self, **kwargs):
        kwargs.setdefault("content", "")
        return ToolResult(**kwargs)

    def _run_syntax_check_for_file(self, path):
        return {"ok": True, "skipped": True, "reason": "test"}

    def _secure_path(self, path, *, confine=False):
        from pathlib import Path as _Path

        resolved = _Path(self.repo_root) / path
        return resolved if resolved.exists() else None

    def _invalidate_cache_after_write(self, files):
        pass


class _SemanticHarness(_Harness):
    """Harness whose _run_syntax_check_for_file returns non-skipped diagnostics."""

    def _run_syntax_check_for_file(self, path):
        return {
            "ok": True,
            "skipped": False,
            "language": "python",
            "errors": [],
            "semantic_diagnostics": [
                {"file_path": str(path), "line": 3, "col": 5,
                 "message": "Cannot access member 'bar' for type 'Foo'",
                 "severity": "error", "code": "attr-defined"},
            ],
        }


@pytest.fixture
def sem_harness(tmp_path):
    return _SemanticHarness(tmp_path)


class TestBlockIntroducerAstNestingGate:
    """Post-insert AST structural gate: a defense-in-depth backstop behind
    the text-based indent-correction above, for shapes it cannot fix."""

    def test_swallowed_trailing_code_rejected(self, sem_harness, tmp_path):
        """REGRESSION: inserting a block-introducer snippet mid-body (i.e.
        the anchor is NOT the block's last statement) corrects the new
        def's indent to a sibling level, but the pre-existing statement(s)
        that followed the anchor are left at their original (deeper) indent
        with nothing to close the new def's block first — so they silently
        become part of the new def's body instead of remaining in the
        original function. This is syntactically valid Python (so the
        separate syntax gate can't catch it) but corrupts the file's
        behavior. Must be rejected pre-write, file left untouched."""
        target = tmp_path / "mod.py"
        original = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        x = 1\n"
            "        return x\n"  # trailing sibling statement after anchor
        )
        target.write_text(original, encoding="utf-8")

        snippet = (
            "# ── Cache section ──\n"
            "_CACHE: dict = {}\n"
            "\n"
            "def helper():\n"
            "    return 1\n"
        )
        result = sem_harness._tool_anchor_edit({
            "file_path": "mod.py",
            "anchor_pattern": "x = 1",
            "edit_mode": "insert_after",
            "code_snippet": snippet,
        })

        assert not result.ok
        assert result.metadata.get("failure_class") == "structural_gate_violation"
        assert target.read_text(encoding="utf-8") == original

    def test_introducer_at_block_end_succeeds(self, sem_harness, tmp_path):
        """Sanity check: when the anchor IS the last statement of its block,
        there is no trailing code to swallow, so the insert must succeed."""
        target = tmp_path / "mod.py"
        target.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        x = 1\n"
            "        return x\n",
            encoding="utf-8",
        )

        snippet = (
            "# ── Cache section ──\n"
            "_CACHE: dict = {}\n"
            "\n"
            "def helper():\n"
            "    return 1\n"
        )
        result = sem_harness._tool_anchor_edit({
            "file_path": "mod.py",
            "anchor_pattern": "return x",
            "edit_mode": "insert_after",
            "code_snippet": snippet,
        })

        assert result.ok, f"should succeed: {result.error}"
        content = target.read_text(encoding="utf-8")
        assert "\n    def helper():" in content

    def test_legitimate_inner_helper_not_rejected(self, sem_harness, tmp_path):
        """REGRESSION (turn ~301): a snippet that introduces a function
        CONTAINING an inner helper def (a legitimate closure — outer + inner
        move together in the same insertion) must NOT be rejected. The gate
        used to flag the inner helper as "nested inside the outer function's
        body" even though that outer function was itself part of the SAME
        insertion, so nothing landed in anyone else's body. Real-world trigger:
        inserting ``_suggest_missing_paths`` which defines ``_score``/``_keep``
        closures."""
        target = tmp_path / "mod.py"
        target.write_text("def pre_existing():\n    pass\n", encoding="utf-8")
        snippet = (
            "def with_helper():\n"
            "    def _inner(x):\n"
            "        return x + 1\n"
            "    return _inner\n"
        )
        result = sem_harness._tool_anchor_edit({
            "file_path": "mod.py",
            "anchor_pattern": "pass",
            "edit_mode": "insert_after",
            "code_snippet": snippet,
        })
        assert result.ok, f"legitimate inner helper wrongly rejected: {result.error}"
        content = target.read_text(encoding="utf-8")
        assert "\ndef with_helper():" in content
        assert "def _inner(x):" in content

    # ── Direct unit tests for _check_block_introducer_nesting ──────────────
    # The `nested_in_function` branch had NO direct coverage before this change;
    # the integration tests above only ever exercised `swallowed_trailing_code`.

    def test_gate_legitimate_inner_helper_returns_none(self):
        """Inner helper defined inside a function that is ALSO in the insert
        range is a closure, not a landing accident → must return None."""
        from external_llm.agent.tool_handlers.write_tools import (
            _check_block_introducer_nesting,
        )
        src = (
            "import os\n"                       # pre-existing
            "def pre():\n    pass\n"            # pre-existing
            "def outer(self):\n"                # inserted (0-based line 3)
            "    def _score(rel):\n"            # inserted inner helper
            "        return rel\n"
            "    return _score\n"
            "def after():\n    pass\n"          # pre-existing
        )
        assert _check_block_introducer_nesting(src, 3, 7) is None

    def test_gate_real_nested_in_preexisting_caught(self):
        """A def that lands inside a PRE-EXISTING function (the enclosing
        function is NOT part of the insert) must still be flagged."""
        from external_llm.agent.tool_handlers.write_tools import (
            _check_block_introducer_nesting,
        )
        src = (
            "def pre_existing():\n    x = 1\n"  # pre-existing
            "    def oops():\n        pass\n"   # inserted, nested in pre_existing
            "    return x\n"
        )
        err = _check_block_introducer_nesting(src, 2, 4)
        assert err is not None
        assert "pre_existing" in err

    def test_gate_transitive_new_def_in_preexisting_caught(self):
        """Even when an introduced def nests inside ANOTHER introduced def, if
        that chain bottoms out in a pre-existing function, it must still be
        flagged (the nearest pre-existing enclosing function is reported)."""
        from external_llm.agent.tool_handlers.write_tools import (
            _check_block_introducer_nesting,
        )
        src = (
            "def pre_existing():\n    x = 1\n"          # pre-existing
            "    def new_a():\n"                        # inserted, nested in pre_existing
            "        def new_b():\n"                    # inserted inner closure
            "            pass\n"
            "    return x\n"                            # pre-existing
        )
        # inserted 0-based lines 2..5 (new_a + new_b + new_b body)
        err = _check_block_introducer_nesting(src, 2, 5)
        assert err is not None
        assert "pre_existing" in err

    def test_gate_method_inside_inserted_class_not_flagged(self):
        """A method inside a class that is ALSO inserted is legitimate — ClassDef
        never counts as an enclosing function (a method is not "nested")."""
        from external_llm.agent.tool_handlers.write_tools import (
            _check_block_introducer_nesting,
        )
        src = (
            "def pre():\n    pass\n"            # pre-existing
            "class C:\n"                        # inserted
            "    def m(self):\n        return 1\n"
            "def after():\n    pass\n"          # pre-existing
        )
        assert _check_block_introducer_nesting(src, 2, 5) is None

    def test_gate_async_def_treated_as_function(self):
        """REGRESSION: ``async def`` must be governed by the SAME rules as
        ``def`` — it is ``ast.AsyncFunctionDef``, a sibling of ``FunctionDef``.
        Two guarantees: (1) an ``async def`` landing inside a PRE-EXISTING
        function is still flagged (violation preserved); (2) an ``async def``
        that itself contains an inner helper is legitimate (no false reject),
        mirroring the sync ``_suggest_missing_paths`` closure case."""
        from external_llm.agent.tool_handlers.write_tools import (
            _check_block_introducer_nesting,
        )
        # (1) async def inserted INTO a pre-existing function body → flagged.
        #     The async def is parsed as a child of ``pre`` (deep indent), so
        #     it genuinely lands in someone else's body.
        src1 = "def pre():\n    pass\n" "    async def a():\n        return 1\n"
        assert _check_block_introducer_nesting(src1, 2, 4) is not None
        # (2) async def WITH an inner closure, inserted at top level → OK.
        src2 = (
            "def pre():\n    pass\n"
            "async def outer():\n"
            "    def _inner():\n"
            "        return 1\n"
            "    return _inner()\n"
        )
        assert _check_block_introducer_nesting(src2, 2, 6) is None
