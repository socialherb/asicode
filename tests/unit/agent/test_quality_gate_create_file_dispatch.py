"""Regression tests for the create_file/overwrite_file quality-gate bypass
surfaced by log analysis of run_20260608_052739 (a JS→TS migration).

Bug: _after_operation_hook only dispatched MODIFY_SYMBOL / INSERT_* / ANCHOR_EDIT
/ EXTRACT_FUNCTION to _run_quality_gate. CREATE_FILE and OVERWRITE_FILE were
omitted, so a created .ts/.js file passed with ZERO syntax/compile validation
(the migration's server.ts did not type-check, yet the run reported success).

Two fixes are locked here:
  1. CREATE_FILE / OVERWRITE_FILE now reach _run_quality_gate.
  2. The Python-package (__init__.py) check inside the gate is Python-only, so
     non-Python creates (.ts/.json/...) don't emit spurious "not a valid Python
     package" warnings now that they reach the gate.
"""
import os
import tempfile
from unittest.mock import MagicMock

from external_llm.agent.operation_models import Operation, OperationKind, PlanMode


def _make_agent_loop(repo_root: str):
    from external_llm.agent.agent_loop import AgentLoop
    loop = AgentLoop.__new__(AgentLoop)
    registry = MagicMock()
    registry.repo_root = repo_root
    loop.registry = registry
    loop.slash_commands = None
    loop._cb = MagicMock()
    loop.config = MagicMock()
    loop.config.run_tests = False
    return loop


def _make_state():
    state = MagicMock()
    state.plan_mode = PlanMode.EDIT
    state.completed_ops = {}
    state.failed_ops = {}
    state.force_finish = False
    state.checkpoint_log = []
    state.pre_op_file_snapshots = {}
    return state


class TestQualityGateDispatch:
    """Fix 1 — file-producing ops must be dispatched to the quality gate."""

    def _hook_calls_gate(self, kind, path, symbol=None):
        loop = _make_agent_loop("/tmp")
        loop._run_quality_gate = MagicMock()
        op = Operation(id="op1", kind=kind, path=path, symbol=symbol)
        loop._after_operation_hook(op, _make_state(), {"status": "success"})
        return loop._run_quality_gate.called

    def test_create_file_dispatched(self):
        assert self._hook_calls_gate(OperationKind.CREATE_FILE, "a/server.ts")

    def test_overwrite_file_dispatched(self):
        assert self._hook_calls_gate(OperationKind.OVERWRITE_FILE, "a/package.json")

    def test_modify_symbol_still_dispatched(self):
        assert self._hook_calls_gate(OperationKind.MODIFY_SYMBOL, "a/x.py", symbol="foo")

    def test_non_edit_kind_not_dispatched(self):
        # DELETE_SYMBOL_RANGE is intentionally NOT in the dispatch set.
        assert not self._hook_calls_gate(OperationKind.DELETE_SYMBOL_RANGE, "a/x.py")


class TestQualityGatePythonPackageGuard:
    """Fix 2 — the __init__.py / package check fires for .py only."""

    def _compat_issue_fired(self, filename, content):
        with tempfile.TemporaryDirectory() as td:
            sub = os.path.join(td, "pkgless")
            os.makedirs(sub)  # directory deliberately has NO __init__.py
            fpath = os.path.join(sub, filename)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            loop = _make_agent_loop(td)
            op = Operation(id="op1", kind=OperationKind.CREATE_FILE, path=fpath)
            loop._run_quality_gate(op, _make_state(), {"status": "success"})
            kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
            return "quality_gate_compatibility_issue" in kinds

    def test_non_python_create_no_package_warning(self):
        # Valid JSON in a non-package dir must NOT raise the package warning.
        assert not self._compat_issue_fired("tsconfig.json", '{"a": 1}\n')

    def test_python_create_in_non_package_dir_still_warns(self):
        # A .py file in a dir without __init__.py still triggers the check.
        assert self._compat_issue_fired("mod.py", "x = 1\n")
