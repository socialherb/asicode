"""contract_diff.py — Diff two function contracts.

Detects what changed between an old and new contract:
param add/remove/rename/type-change, return type change.

Ported from ts_vm/contract/contract_diff.py — language-agnostic.
"""
from __future__ import annotations

from external_llm.editor._editor_core.vm.contracts.contract_models import (
    ContractChange,
    ContractChangeKind,
    ContractDiffResult,
    FunctionContract,
)


class ContractDiffer:
    """Computes the diff between two function contracts."""

    def diff(
        self,
        old: FunctionContract,
        new: FunctionContract,
    ) -> ContractDiffResult:
        changes: list[ContractChange] = []

        # Name change
        if old.name != new.name:
            changes.append(ContractChange(
                kind=ContractChangeKind.SYMBOL_RENAMED,
                detail=f"{old.name} → {new.name}",
                old_value=old.name, new_value=new.name,
            ))

        # Return type change
        if old.return_type != new.return_type:
            changes.append(ContractChange(
                kind=ContractChangeKind.RETURN_TYPE_CHANGED,
                detail=f"{old.return_type} → {new.return_type}",
                old_value=old.return_type, new_value=new.return_type,
            ))

        # Async change
        if old.is_async != new.is_async:
            changes.append(ContractChange(
                kind=ContractChangeKind.ASYNC_CHANGED,
                detail=f"async: {old.is_async} → {new.is_async}",
                old_value=str(old.is_async),
                new_value=str(new.is_async),
            ))

        # Param changes
        self._diff_params(old, new, changes)

        return ContractDiffResult(
            symbol=new.name,
            file_path=new.file_path,
            changes=changes,
            old_contract=old,
            new_contract=new,
        )

    def diff_multi(
        self,
        old_contracts: list[FunctionContract],
        new_contracts: list[FunctionContract],
    ) -> list[ContractDiffResult]:
        """Diff multiple contracts between old and new versions.

        Matches by name first, then by parameter/return type
        similarity to detect renames.
        """
        old_by_name = {c.name: c for c in old_contracts}
        new_by_name = {c.name: c for c in new_contracts}
        matched_old: set = set()
        matched_new: set = set()

        results: list[ContractDiffResult] = []

        # Phase 1: Exact name matches
        for name in new_by_name:
            if name in old_by_name:
                result = self.diff(old_by_name[name], new_by_name[name])
                if result.has_changes:
                    results.append(result)
                matched_old.add(name)
                matched_new.add(name)

        # Phase 2: Detect renames (unmatched symbols with similar params)
        unmatched_old = [c for c in old_contracts if c.name not in matched_old]
        unmatched_new = [c for c in new_contracts if c.name not in matched_new]

        for new_c in list(unmatched_new):
            best_match = None
            best_score = -1
            for old_c in unmatched_old:
                score = self._similarity_score(old_c, new_c)
                if score > best_score:
                    best_score = score
                    best_match = old_c

            if best_match and best_score >= 2:  # at least 2 matching params
                # This is a rename
                diff = self.diff(best_match, new_c)
                diff.symbol = new_c.name
                diff.file_path = new_c.file_path
                # Add rename change if not already detected
                has_rename = any(c.kind == ContractChangeKind.SYMBOL_RENAMED
                                for c in diff.changes)
                if not has_rename:
                    diff.changes.insert(0, ContractChange(
                        kind=ContractChangeKind.SYMBOL_RENAMED,
                        detail=f"{best_match.name} → {new_c.name}",
                        old_value=best_match.name,
                        new_value=new_c.name,
                    ))
                results.append(diff)
                unmatched_old.remove(best_match)
                unmatched_new.remove(new_c)
                matched_old.add(best_match.name)
                matched_new.add(new_c.name)

        # Phase 3: Remaining unmatched = added/removed
        for c in unmatched_new:
            results.append(ContractDiffResult(
                symbol=c.name,
                file_path=c.file_path,
                changes=[ContractChange(
                    kind=ContractChangeKind.SYMBOL_REMOVED,
                    detail=f"Symbol '{c.name}' added (no old contract)",
                    new_value=c.name,
                )],
                new_contract=c,
            ))

        for c in unmatched_old:
            results.append(ContractDiffResult(
                symbol=c.name,
                file_path=c.file_path,
                changes=[ContractChange(
                    kind=ContractChangeKind.SYMBOL_REMOVED,
                    detail=f"Symbol '{c.name}' removed",
                    old_value=c.name,
                )],
                old_contract=c,
            ))

        return results

    @staticmethod
    def _similarity_score(
        a: FunctionContract, b: FunctionContract,
    ) -> int:
        """Score how similar two contracts are (higher = more similar).

        Used for rename detection.
        """
        score = 0
        # Same arity
        if a.arity == b.arity:
            score += 1
        # Same param names (positional)
        shared_params = 0
        for i in range(min(len(a.params), len(b.params))):
            if a.params[i].name == b.params[i].name:
                shared_params += 1
        score += shared_params
        # Same return type
        if a.return_type == b.return_type and a.return_type is not None:
            score += 1
        return score

    def _diff_params(
        self,
        old: FunctionContract,
        new: FunctionContract,
        changes: list[ContractChange],
    ) -> None:
        old_names: set[str] = set(old.param_names)
        new_names: set[str] = set(new.param_names)

        # Added params
        for name in new_names - old_names:
            p = next(p for p in new.params if p.name == name)
            changes.append(ContractChange(
                kind=ContractChangeKind.PARAM_ADDED,
                detail=f"param '{name}' added at position {p.position}",
                new_value=name,
            ))

        # Removed params
        for name in old_names - new_names:
            changes.append(ContractChange(
                kind=ContractChangeKind.PARAM_REMOVED,
                detail=f"param '{name}' removed",
                old_value=name,
            ))

        # Shared params: check type
        for name in old_names & new_names:
            old_p = next(p for p in old.params if p.name == name)
            new_p = next(p for p in new.params if p.name == name)

            if old_p.type_name != new_p.type_name:
                changes.append(ContractChange(
                    kind=ContractChangeKind.PARAM_TYPE_CHANGED,
                    detail=f"param '{name}': {old_p.type_name} → {new_p.type_name}",
                    old_value=old_p.type_name,
                    new_value=new_p.type_name,
                ))

        # Position-based rename detection
        for i in range(min(len(old.params), len(new.params))):
            if (
                old.params[i].name != new.params[i].name
                and old.params[i].name not in new_names
                and new.params[i].name not in old_names
            ):
                changes.append(ContractChange(
                    kind=ContractChangeKind.PARAM_RENAMED,
                    detail=f"param at position {i}: {old.params[i].name} → {new.params[i].name}",
                    old_value=old.params[i].name,
                    new_value=new.params[i].name,
                ))
