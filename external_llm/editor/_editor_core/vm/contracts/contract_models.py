"""contract_models.py — Language-agnostic contract models.

A contract captures the externally visible "shape" of a symbol:
- Function: params + return type
- Class: public methods + properties

When a contract changes, all callers/consumers must be updated.

Ported from ts_vm/contract/contract_models.py with no TS dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContractChangeKind(Enum):
    """What changed in a contract."""

    PARAM_ADDED = "param_added"
    PARAM_REMOVED = "param_removed"
    PARAM_TYPE_CHANGED = "param_type_changed"
    PARAM_RENAMED = "param_renamed"
    RETURN_TYPE_CHANGED = "return_type_changed"
    ASYNC_CHANGED = "async_changed"
    SYMBOL_RENAMED = "symbol_renamed"
    SYMBOL_REMOVED = "symbol_removed"


@dataclass
class ParamContract:
    """Contract for a single parameter."""

    name: str
    type_name: Optional[str] = None
    has_default: bool = False
    is_optional: bool = False
    position: int = 0


@dataclass
class FunctionContract:
    """Externally visible shape of a function."""

    name: str
    params: list[ParamContract] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_exported: bool = False
    file_path: str = ""

    @property
    def arity(self) -> int:
        return len(self.params)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]


@dataclass
class ContractChange:
    """A single change between old and new contract."""

    kind: ContractChangeKind
    detail: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None


@dataclass
class ContractDiffResult:
    """Result of diffing two contracts."""

    symbol: str
    file_path: str
    changes: list[ContractChange] = field(default_factory=list)
    old_contract: Optional[FunctionContract] = None
    new_contract: Optional[FunctionContract] = None

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    @property
    def has_signature_change(self) -> bool:
        return any(c.kind in (
            ContractChangeKind.PARAM_ADDED,
            ContractChangeKind.PARAM_REMOVED,
            ContractChangeKind.PARAM_TYPE_CHANGED,
            ContractChangeKind.RETURN_TYPE_CHANGED,
        ) for c in self.changes)
