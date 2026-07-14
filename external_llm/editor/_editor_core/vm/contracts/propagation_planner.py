"""propagation_planner.py — Plan contract-aware cross-file repairs.

When a function's contract changes, this planner generates primitive
ops to update all call sites across the project.

Ported from ts_vm/contract/propagation_planner.py — language-agnostic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from external_llm.editor._editor_core.vm.contracts.contract_models import (
    ContractChange,
    ContractChangeKind,
    ContractDiffResult,
)
from external_llm.editor._editor_core.vm.contracts.project_graph import ProjectGraph
from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.models import CallSite, PrimitiveKind, PrimitiveOp

logger = logging.getLogger(__name__)


@dataclass
class PropagationOp:
    """A single propagation operation targeting one file."""
    file_path: str
    ops: list[PrimitiveOp] = field(default_factory=list)
    description: str = ""


@dataclass
class PropagationPlan:
    """Ordered list of per-file propagation operations."""
    operations: list[PropagationOp] = field(default_factory=list)
    description: str = ""
    has_work: bool = False


class PropagationPlanner:
    """Plans cross-file propagation from contract changes."""

    def propagate(
        self,
        graph: ProjectGraph,
        diff: ContractDiffResult,
    ) -> PropagationPlan:
        """Generate a plan to propagate contract changes."""
        operations: list[PropagationOp] = []

        for change in diff.changes:
            ops = self._plan_change(graph, diff, change)
            operations.extend(ops)

        return PropagationPlan(
            operations=operations,
            description=f"Propagate {len(diff.changes)} contract change(s) for '{diff.symbol}'",
            has_work=len(operations) > 0,
        )

    def _plan_change(
        self,
        graph: ProjectGraph,
        diff: ContractDiffResult,
        change: ContractChange,
    ) -> list[PropagationOp]:
        kind = change.kind

        if kind == ContractChangeKind.SYMBOL_RENAMED:
            return self._propagate_rename(graph, diff, change)

        if kind == ContractChangeKind.PARAM_REMOVED:
            return self._propagate_param_removed(graph, diff, change)

        if kind == ContractChangeKind.PARAM_ADDED:
            return self._propagate_param_added(graph, diff, change)

        if kind == ContractChangeKind.RETURN_TYPE_CHANGED:
            return self._propagate_return_type_change(graph, diff, change)

        return []

    def _propagate_rename(
        self, graph: ProjectGraph, diff: ContractDiffResult,
        change: ContractChange,
    ) -> list[PropagationOp]:
        """Rename symbol across all using files."""
        old_name = change.old_value or diff.symbol
        new_name = change.new_value or diff.symbol
        using_files = graph.files_using(old_name)
        ops = []
        for path in using_files:
            if path == diff.file_path:
                continue  # definition file already updated
            ops.append(PropagationOp(
                file_path=path,
                ops=[PrimitiveOp(
                    kind=PrimitiveKind.RENAME_SYMBOL,
                    payload={
                        "old_name": old_name,
                        "new_name": new_name,
                    },
                )],
                description=f"Rename '{old_name}' → '{new_name}' in {path}",
            ))
        return ops

    def _propagate_param_removed(
        self, graph: ProjectGraph, diff: ContractDiffResult,
        change: ContractChange,
    ) -> list[PropagationOp]:
        """When a param is removed, update call sites to drop the arg."""
        if diff.new_contract is None or diff.old_contract is None:
            return []

        using_files = graph.files_using(diff.symbol)

        # Find the position of removed param
        removed_name = change.old_value or ""
        removed_pos = None
        for p in diff.old_contract.params:
            if p.name == removed_name:
                removed_pos = p.position
                break

        if removed_pos is None:
            return []

        ops = []
        for path in using_files:
            if path == diff.file_path:
                continue

            source = graph.file_sources.get(path)
            if source is None:
                continue

            # Use CodeContext to find call sites
            call_sites = self._find_call_sites(source, path, diff.symbol)
            for cs in call_sites:
                call_text = source[cs.start_byte:cs.end_byte]
                new_call = self._remove_arg_at(call_text, removed_pos)
                if new_call and new_call != call_text:
                    ops.append(PropagationOp(
                        file_path=path,
                        ops=[PrimitiveOp(
                            kind=PrimitiveKind.UPDATE_CALL,
                            payload={
                                "callee": diff.symbol,
                                "new_call": new_call,
                            },
                        )],
                        description=f"Remove arg at pos {removed_pos} in {path}",
                    ))
                    break  # one op per file

        return ops

    def _propagate_param_added(
        self, graph: ProjectGraph, diff: ContractDiffResult,
        change: ContractChange,
    ) -> list[PropagationOp]:
        """When a param is added, callers may need updating."""
        if diff.new_contract is None:
            return []

        # Find the new param
        new_param = None
        for p in diff.new_contract.params:
            if p.name == change.new_value:
                new_param = p
                break

        if new_param is None or new_param.has_default or new_param.is_optional:
            return []  # optional params don't break callers

        # Required param added — we can't auto-generate the value
        logger.info(
            "Required param '%s' added to '%s' — callers need manual update",
            new_param.name, diff.symbol,
        )
        return []

    def _propagate_return_type_change(
        self, graph: ProjectGraph, diff: ContractDiffResult,
        change: ContractChange,
    ) -> list[PropagationOp]:
        """When return type changes, signal but don't auto-fix."""
        logger.info(
            "Return type changed for '%s': %s → %s",
            diff.symbol, change.old_value, change.new_value,
        )
        return []

    # ── Helpers ──────────────────────────────────────────────────────

    def _find_call_sites(
        self, source: str, file_path: str, callee: str,
    ) -> list[CallSite]:
        """Find call sites of *callee* in *source*."""
        import os

        from external_llm.languages.models import LanguageId

        ext_map = {
            ".py": LanguageId.PYTHON,
            ".java": LanguageId.JAVA,
            ".kt": LanguageId.KOTLIN,
            ".go": LanguageId.GO,
        }
        _, ext = os.path.splitext(file_path)
        lang = ext_map.get(ext.lower())
        if lang is None:
            return []

        ctx = CodeContext(source, file_path, lang)
        call_sites = ctx.get_call_sites()
        return [cs for cs in call_sites if cs.callee == callee]

    def _remove_arg_at(self, call_text: str, position: int) -> Optional[str]:
        """Remove the arg at *position* from a call expression."""
        paren_open = call_text.find("(")
        paren_close = call_text.rfind(")")
        if paren_open == -1 or paren_close == -1:
            return None

        args_str = call_text[paren_open + 1:paren_close]
        args = [a.strip() for a in args_str.split(",")]

        if position < 0 or position >= len(args):
            return None

        args.pop(position)
        new_args = ", ".join(args)
        return call_text[:paren_open + 1] + new_args + call_text[paren_close:]
