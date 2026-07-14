"""semantic_rewrite_models.py — Phase C.3: Rewrite Operation Models.

Defines the data structures for AST-based semantic rewrites:
- RewriteOperation: a single AST transformation
- RewritePlan: ordered list of operations on one file
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RewriteOpType:
    REORDER_CALLS = "reorder_calls"
    REPLACE_CALL_ARGS = "replace_call_args"
    REWRITE_RETURN = "rewrite_return"
    MOVE_STATEMENT = "move_statement"


@dataclass
class RewriteOperation:
    """A single AST-level rewrite operation."""
    op_type: str                  # RewriteOpType constant
    target_function: str          # Function to modify
    payload: dict[str, Any] = field(default_factory=dict)
    # Payload varies by op_type:
    #   REORDER_CALLS:    {"order": ["get_user", "verify_password", "create_access_token"]}
    #   REPLACE_CALL_ARGS: {"call_name": "verify_password", "new_args": ["password", "user.hashed_password"]}
    #   REWRITE_RETURN:   {"new_return": '{"id": entity.id, "status": "created"}'}
    #   MOVE_STATEMENT:   {"call_name": "get_user", "before": "create_access_token"}
    description: str = ""
    contract_name: str = ""       # Which contract triggered this


@dataclass
class RewritePlan:
    """Ordered list of rewrite operations for a single file."""
    file_path: str
    operations: list[RewriteOperation] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.operations) == 0


@dataclass
class RewriteResult:
    """Result of applying a rewrite plan."""
    success: bool = False
    applied_ops: list[str] = field(default_factory=list)
    skipped_ops: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "applied_ops": self.applied_ops,
            "skipped_ops": self.skipped_ops,
            "files_modified": self.files_modified,
            "error": self.error,
        }
