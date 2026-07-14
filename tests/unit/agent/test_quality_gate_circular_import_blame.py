"""Unit tests for _run_quality_gate circular import blame model (agent_loop.py).

Verifies that _run_quality_gate correctly distinguishes:
  - Pre-existing circular imports → INFO only, NOT a critical failure
  - Newly introduced circular imports → critical failure (existing behavior)

The blame model: _before_operation_hook saves pre-op file content in
state.pre_op_file_snapshots[op.id]. In _run_quality_gate, the AST-based
circular import check reads the pre-edit imports to determine if the cycle
was already present before this op ran.
"""
from unittest.mock import MagicMock

from external_llm.agent.operation_models import (
    Operation,
    OperationKind,
    PlanMode,
)


def _make_state_with_snapshot(op_id: str, abs_path: str, content: str):
    state = MagicMock()
    state.plan_mode = PlanMode.EDIT
    state.completed_ops = {}
    state.failed_ops = {}
    state.force_finish = False
    state.pre_op_file_snapshots = {op_id: {abs_path: content}}
    return state


def _make_state_no_snapshot():
    state = MagicMock()
    state.plan_mode = PlanMode.EDIT
    state.completed_ops = {}
    state.failed_ops = {}
    state.force_finish = False
    state.pre_op_file_snapshots = {}
    return state


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


def _build_circular_pair(tmp_path):
    """Create two direct-module-file circular imports:
    prj/executor.py imports prj.handlers  ↔  prj/handlers.py imports prj.executor
    Returns (repo_root, executor_py_abs, executor_py_rel).
    """
    repo_root = str(tmp_path)
    pkg = tmp_path / "prj"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    # handlers.py: imports from prj.executor (direct module file)
    handlers_py = pkg / "handlers.py"
    handlers_py.write_text(
        "from prj.executor import do_exec\n\ndef do_handler(): pass\n"
    )

    # executor.py: imports from prj.handlers (direct module file)
    executor_py = pkg / "executor.py"
    executor_rel = str(executor_py.relative_to(tmp_path))
    return repo_root, executor_py, executor_rel


class TestCircularImportBlameModel:
    """_run_quality_gate must not fail critically for pre-existing circular imports."""

    def test_preexisting_circular_import_no_critical_failure(self, tmp_path):
        """When the edited file had the same circular import BEFORE the edit,
        no critical failure should be raised."""
        repo_root, executor_py, executor_rel = _build_circular_pair(tmp_path)

        # POST-edit: imports handlers + newly added function
        executor_current = (
            "from prj.handlers import do_handler\n\n"
            "def do_exec(): pass\n\n"
            "def new_func(): pass\n"
        )
        executor_py.write_text(executor_current)

        # PRE-edit: same import was already present — only new_func was missing
        executor_pre = (
            "from prj.handlers import do_handler\n\n"
            "def do_exec(): pass\n"
        )

        op = Operation(
            id="op1",
            kind=OperationKind.MODIFY_SYMBOL,
            path=executor_rel,
            symbol="do_exec",
            intent="add new_func",
        )

        state = _make_state_with_snapshot(op.id, str(executor_py), executor_pre)
        loop = _make_agent_loop(repo_root)

        loop._run_quality_gate(op, state, {"status": "success"})

        assert not state.force_finish, (
            "force_finish must not be set for pre-existing circular import"
        )
        assert op.id not in state.failed_ops, (
            "op must not be in failed_ops for pre-existing circular import"
        )

    def test_newly_introduced_circular_import_is_critical(self, tmp_path):
        """When the edit ADDS an import that creates a new cycle, it IS critical."""
        repo_root, executor_py, executor_rel = _build_circular_pair(tmp_path)

        # POST-edit: NOW imports handlers (creating the cycle)
        executor_current = (
            "from prj.handlers import do_handler\n\n"
            "def do_exec(): pass\n"
        )
        executor_py.write_text(executor_current)

        # PRE-edit: did NOT import handlers → no cycle before this op
        executor_pre = "def do_exec(): pass\n"

        op = Operation(
            id="op2",
            kind=OperationKind.MODIFY_SYMBOL,
            path=executor_rel,
            symbol="do_exec",
            intent="add import handlers",
        )

        state = _make_state_with_snapshot(op.id, str(executor_py), executor_pre)
        loop = _make_agent_loop(repo_root)

        loop._run_quality_gate(op, state, {"status": "success"})

        assert state.force_finish or op.id in state.failed_ops, (
            "Newly introduced circular import must trigger critical failure"
        )

    def test_no_snapshot_is_conservative(self, tmp_path):
        """When no pre-op snapshot exists, treat as newly introduced (conservative)."""
        repo_root, executor_py, executor_rel = _build_circular_pair(tmp_path)

        executor_py.write_text(
            "from prj.handlers import do_handler\n\n"
            "def do_exec(): pass\n"
        )

        op = Operation(
            id="op3",
            kind=OperationKind.MODIFY_SYMBOL,
            path=executor_rel,
            symbol="do_exec",
            intent="modify do_exec",
        )

        state = _make_state_no_snapshot()  # no snapshot → unknown → conservative
        loop = _make_agent_loop(repo_root)

        loop._run_quality_gate(op, state, {"status": "success"})

        assert state.force_finish or op.id in state.failed_ops, (
            "Without pre-op snapshot, circular import must be treated as critical (conservative)"
        )
