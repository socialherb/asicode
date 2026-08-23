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
    # Canonical extension → language-name map (values are LanguageId member
    # names).  The structural-scan walk selects a subset of these
    # (SCAN_EXTS, external_llm/analysis/scan_walk.py) and deliberately
    # excludes the stub/module/script variants below; the boundary is pinned
    # by test_scan_ext_map_boundary_is_exhaustive.  The tree-sitter
    # extension → grammar-key map (tree_sitter_utils._EXT_TO_GRAMMAR_KEY) is
    # DERIVED from this map for languages with full AST query support.
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
    # zsh / ksh are '#'-comment shells that share the bash grammar's common
    # subset.  Without these entries they fell to UNKNOWN → no comment skipping
    # → a '#' comment bracket was over-counted, re-opening the same F2
    # multi-line-expansion data-loss class this family classification prevents
    # (see test_zsh_ksh_resolve_to_bash_comment_syntax).
    ".zsh": "BASH",
    ".ksh": "BASH",
}

# Language "callability families": groups of file extensions whose definitions
# are mutually callable.  JS and TS parse as different languages (different
# LanguageId / tree-sitter grammar) but a function defined in .ts *can* be
# called from .js/.jsx/.tsx, so they form one family — hence a group, not a
# single LanguageId.
#
# The hand-maintained fact is the LANGUAGE-LEVEL partition below
# (_LANGUAGE_FAMILIES): which LanguageId member names form one callability
# family.  The extension sets are DERIVED from _EXT_MAP (the canonical
# extension → language map) by _derive_language_extension_groups — the old
# 33-entry literal table is gone, so extension drift is structurally
# impossible.  The partition must name exactly the full-AST-support languages
# (the _EXT_TO_GRAMMAR_KEY domain, tree_sitter_utils — pinned by
# test_family_groups_match_grammar_map); an extension absent from its family
# causes two live bugs:
#   (a) caller_search_extensions returns the broad fallback union instead of a
#       tight family glob — every other language's files are scanned;
#   (b) _get_language_group returns -1, silently bypassing the cross-language
#       resolution guard.
#
# The tuple ORDER is part of the contract: _get_language_group returns group
# indices that the cross-language resolution guard compares (pinned by
# test_group_indices_are_stable) — keep family order stable.
#
# Single source of truth, consumed by:
#   * cross-file caller search (ripgrep glob set — see caller_search_extensions)
#   * cross-language resolution guard (see _get_language_group)
_LANGUAGE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"TYPESCRIPT", "JAVASCRIPT"}),  # JS/TS family
    frozenset({"PYTHON"}),  # Python (+type stubs)
    frozenset({"GO"}),  # Go
    frozenset({"JAVA"}),  # Java
    frozenset({"KOTLIN"}),  # Kotlin
    frozenset({"RUST"}),  # Rust
    frozenset({"RUBY"}),  # Ruby
    frozenset({"C", "CPP"}),  # C/C++ family
    frozenset({"PHP"}),  # PHP
    frozenset({"CSHARP"}),  # C#
    frozenset({"SWIFT"}),  # Swift
    frozenset({"SCALA"}),  # Scala
    frozenset({"LUA"}),  # Lua
    frozenset({"BASH"}),  # Bash
)


def _derive_language_extension_groups() -> list[frozenset[str]]:
    """Resolve the language-level families to extension sets via _EXT_MAP.

    Fails at import time (fail-fast) when a family names a language that has
    no extensions in _EXT_MAP — a typo'd member name, or a language whose
    entries were removed, would otherwise silently shrink/empty the family,
    widening caller search to the broad fallback union and bypassing the
    cross-language resolution guard.
    """
    known = set(_EXT_MAP.values())
    groups: list[frozenset[str]] = []
    for family in _LANGUAGE_FAMILIES:
        unknown = family - known
        if unknown:
            raise ValueError(
                f"_LANGUAGE_FAMILIES names {sorted(unknown)} which has no "
                "extensions in _EXT_MAP (external_llm/languages/models.py) — "
                "fix the family name or add the language's extensions"
            )
        exts = frozenset(ext for ext, lang in _EXT_MAP.items() if lang in family)
        if not exts:
            raise ValueError(f"family {sorted(family)} resolved to no extensions")
        groups.append(exts)
    return groups


_LANGUAGE_EXTENSION_GROUPS: list[frozenset[str]] = _derive_language_extension_groups()


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
    def from_path(file_path: str) -> LanguageId:
        """Map file extension to LanguageId."""
        _, ext = os.path.splitext(file_path)
        name = _EXT_MAP.get(ext.lower())
        if name is None:
            return LanguageId.UNKNOWN
        return LanguageId[name]


@dataclass(frozen=True)
class SyntaxError_:  # noqa: N801 — trailing underscore avoids builtin shadow
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
    """Result of validating a file's syntax.

    ``checked`` separates "the tool ran and found nothing" from "no tool ran".
    Both used to be ``ok=True, errors=[]``, and downstream that is
    indistinguishable from a clean verdict — so a user with no pyright
    installed had every Python edit reported to the model as semantically
    checked. Semantic validation skips for several ordinary reasons (the
    toolchain is not installed, it timed out, the project has no config for it),
    and none of them is evidence about the file.

    Syntax validation always genuinely runs — it is tree-sitter or the
    language's own parser, always present — so the default is True and only the
    semantic paths construct the False case, via :meth:`unchecked`.
    """

    ok: bool
    errors: list[SyntaxError_] = field(default_factory=list)
    language: LanguageId = LanguageId.UNKNOWN
    checked: bool = True
    skip_reason: str = ""

    @classmethod
    def unchecked(cls, language: LanguageId, reason: str) -> SyntaxValidationResult:
        """A result meaning "nothing examined this file", with the why.

        *reason* reaches the model verbatim, so it names the missing tool
        rather than the code path — "pyright is not installed", not
        "FileNotFoundError".
        """
        return cls(ok=True, errors=[], language=language, checked=False, skip_reason=reason)


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
