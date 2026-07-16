"""
Language models for multi-language support.

Defines language identification, capabilities, and validation result types.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

_EXT_MAP = {
    ".py": "PYTHON",
    ".pyi": "PYTHON",
    ".ts": "TYPESCRIPT",
    ".tsx": "TYPESCRIPT",
    ".mts": "TYPESCRIPT",
    ".cts": "TYPESCRIPT",
    ".js": "JAVASCRIPT",
    ".jsx": "JAVASCRIPT",
    ".mjs": "JAVASCRIPT",
    ".cjs": "JAVASCRIPT",
    ".go": "GO",
    ".java": "JAVA",
    ".kt": "KOTLIN",
    ".kts": "KOTLIN",
    ".json": "JSON",
    ".jsonc": "JSON",
    ".css": "CSS",
    ".scss": "CSS",
    ".less": "CSS",
    ".html": "HTML",
    ".htm": "HTML",
    # Parse-only → languages with full AST support
    ".rs": "RUST",
    ".c": "C",
    ".h": "C",
    ".cpp": "CPP",
    ".cc": "CPP",
    ".cxx": "CPP",
    ".hpp": "CPP",
    ".hh": "CPP",
    ".rb": "RUBY",
    ".php": "PHP",
    ".cs": "CSHARP",
    ".swift": "SWIFT",
    ".scala": "SCALA",
    ".sc": "SCALA",
    ".lua": "LUA",
    ".sh": "BASH",
    ".bash": "BASH",
}

# Language "callability families": groups of file extensions whose definitions
# are mutually callable.  JS and TS parse as different languages (different
# LanguageId / tree-sitter grammar) but a function defined in .ts *can* be
# called from .js/.jsx/.tsx, so they form one family — hence a group, not a
# single LanguageId.
#
# Covers every extension that maps to a full-AST-support language
# (_SYMBOL_QUERIES + _CALL_QUERIES + _IMPORT_QUERIES + _REFERENCE_QUERIES all
# populated).  An extension absent here causes two live bugs:
#   (a) caller_search_extensions returns the broad fallback union instead of a
#       tight family glob — every other language's files are scanned;
#   (b) _get_language_group returns -1, silently bypassing the cross-language
#       resolution guard.
#
# Single source of truth, consumed by:
#   * cross-file caller search (ripgrep glob set — see caller_search_extensions)
#   * cross-language resolution guard (SpecGraphEnricher — see _get_language_group)
_LANGUAGE_EXTENSION_GROUPS: list[frozenset[str]] = [
    frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"}),  # JS/TS family
    frozenset({".py", ".pyi"}),                                  # Python (+type stubs)
    frozenset({".go"}),                                          # Go
    frozenset({".java"}),                                        # Java
    frozenset({".kt", ".kts"}),                                  # Kotlin
    frozenset({".rs"}),                                          # Rust
    frozenset({".rb"}),                                          # Ruby
    frozenset({".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}),  # C/C++ family
    frozenset({".php"}),                                         # PHP
    frozenset({".cs"}),                                          # C#
    frozenset({".swift"}),                                       # Swift
    frozenset({".scala", ".sc"}),                                # Scala
    frozenset({".lua"}),                                         # Lua
    frozenset({".sh", ".bash"}),                                 # Bash
]


def _get_language_group(ext: str) -> int:
    """Return the family index (0-based) for a file extension, or -1 if unknown."""
    ext = ext.lower()
    for i, group in enumerate(_LANGUAGE_EXTENSION_GROUPS):
        if ext in group:
            return i
    return -1


def caller_search_extensions(file_path: str | None) -> list[str]:
    """Return file extensions (with leading dot) that may *call* functions defined
    in *file_path*, based on its language family.

    A definition is only callable from its own family (e.g. ``.ts`` ↔ ``.js``),
    so cross-file caller search should scope globs to that family rather than a
    single hardcoded language.  When the family is unknown (unrecognized
    extension or ``None``), returns the union of all known code-language
    extensions as a safe broad fallback — strictly better than hardcoding one
    language and silently missing the rest.  Sorted for deterministic output.
    """
    if file_path:
        _, ext = os.path.splitext(file_path)
        group_idx = _get_language_group(ext)
        if group_idx >= 0:
            return sorted(_LANGUAGE_EXTENSION_GROUPS[group_idx])
    all_exts: set[str] = set()
    for g in _LANGUAGE_EXTENSION_GROUPS:
        all_exts |= g
    return sorted(all_exts)


class LanguageId(Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JSON = "json"
    CSS = "css"
    HTML = "html"
    # Parse-only → languages with full AST support
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "c_sharp"
    SWIFT = "swift"
    SCALA = "scala"
    LUA = "lua"
    BASH = "bash"
    UNKNOWN = "unknown"

    @staticmethod
    @lru_cache(maxsize=128)
    def from_path(file_path: str) -> "LanguageId":
        """Map file extension to LanguageId."""
        _, ext = os.path.splitext(file_path)
        name = _EXT_MAP.get(ext.lower())
        if name is None:
            return LanguageId.UNKNOWN
        return LanguageId[name]


@dataclass(frozen=True)
class SyntaxError_:
    """A single syntax/semantic diagnostic in a file.

    ``severity`` and ``code`` were added to carry semantic diagnostics
    (pyright/tsc type errors, undefined names, missing imports). Both default
    so that existing call sites that only report syntax errors stay compatible.
    """
    file: str
    line: int
    col: int
    message: str
    severity: str = "error"  # "error" | "warning" | "info"
    code: str = ""  # tool-specific code, e.g. "reportUndefinedVariable", "TS2304"


@dataclass
class SyntaxValidationResult:
    """Result of validating a file's syntax."""
    ok: bool
    errors: list[SyntaxError_] = field(default_factory=list)
    language: LanguageId = LanguageId.UNKNOWN


@dataclass
class SymbolPattern:
    """A regex pattern for finding a symbol definition.

    The ``regex`` field may contain a ``{name}`` placeholder that should
    be replaced with the actual (regex-escaped) symbol name before use.

    When the index pass substitutes ``{name}`` with an *unknown* name — i.e.
    building a reverse index of every symbol in the tree — ``name_capture``
    supplies the regex group that captures the actual identifier.  It defaults
    to ``\\w+`` (single word, suits Go/Java/Rust identifiers), but CSS uses
    a broader class that includes hyphens (``[-\\w]+``) so kebab-case class /
    id / custom-property names like ``btn-primary`` or ``--primary-color`` are
    not truncated at the first hyphen.
    """
    kind: str
    regex: str
    description: str = ""
    name_capture: str = r"\w+"


@dataclass
class LanguageCapabilities:
    """Boolean flags describing what a language provider supports."""
    has_ast_parser: bool = False
    has_syntax_validator: bool = False
    has_semantic_validator: bool = False
    has_linter: bool = False
    has_test_runner: bool = False
    has_symbol_search: bool = False
    has_tree_sitter: bool = False
    supports_modify_symbol: bool = False
    supports_insert_after_symbol: bool = False
