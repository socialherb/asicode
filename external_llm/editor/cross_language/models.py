"""models.py — Cross-language learning data models.

Language-neutral representations for sharing strategy knowledge
between Python and TS/JS execution engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Language(str, Enum):
    """Supported execution engine languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"


class AbstractIntent(str, Enum):
    """Language-neutral intent categories.

    Maps from both Python OperationKind and TS IntentKind.
    """
    ADD_SYMBOL = "add_symbol"
    MODIFY_SYMBOL = "modify_symbol"
    RENAME_SYMBOL = "rename_symbol"
    DELETE_SYMBOL = "delete_symbol"
    MOVE_SYMBOL = "move_symbol"
    ADD_IMPORT = "add_import"
    REFACTOR = "refactor"
    FIX_ERROR = "fix_error"
    CREATE_FILE = "create_file"
    UNKNOWN = "unknown"


class AbstractStrategy(str, Enum):
    """Language-neutral strategy categories.

    Strategies that have semantic equivalents across languages
    are mapped to the same abstract strategy.
    """
    MINIMAL_CHANGE = "minimal_change"        # smallest edit
    STRUCTURED_EDIT = "structured_edit"      # AST-level ops
    CROSS_FILE = "cross_file"               # multi-file coordinated
    ERROR_DRIVEN_REPAIR = "error_driven"    # error→fix loop
    BODY_REPLACEMENT = "body_replacement"   # full body swap
    TEMPLATE_BASED = "template_based"       # from template/reference
    TEST_DRIVEN = "test_driven"             # test-first / TDD approach
    UNKNOWN = "unknown"


@dataclass
class CrossLanguageRecord:
    """A single cross-language learning record.

    Stored in the shared SQLite DB. Language-neutral representation
    of a strategy execution outcome.
    """
    language: str               # Language enum value
    abstract_intent: str        # AbstractIntent value
    abstract_strategy: str      # AbstractStrategy value
    original_strategy: str      # language-specific strategy name
    original_intent: str        # language-specific intent name
    success: bool
    reward: float
    repair_rounds: int = 0
    affected_files: int = 1
    error_types: list[str] = field(default_factory=list)
    context_key: str = ""       # abstract context key

    @property
    def scope(self) -> str:
        return "single" if self.affected_files <= 1 else "multi"

    @property
    def abstract_context_key(self) -> str:
        """Language-neutral context key for policy grouping."""
        if self.context_key:
            return self.context_key
        return f"{self.abstract_intent}:{self.scope}"
