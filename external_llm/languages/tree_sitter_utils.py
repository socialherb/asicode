"""
Tree-sitter integration utilities (optional dependency).

When tree-sitter is installed, provides precise AST-based symbol range
detection.  When not installed, all functions gracefully return None / empty
so callers can fall back to regex-based heuristics.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional import ──────────────────────────────────────────────────────────

_HAS_TREE_SITTER = False
try:
    import tree_sitter as _ts

    _HAS_TREE_SITTER = True
except ImportError:
    pass

# Language module cache: language name → tree_sitter.Language object (thread-safe)
_LANG_CACHE: dict[str, object] = {}
_LANG_CACHE_LOCK = threading.RLock()

# Parser cache: per-thread. tree-sitter's TSParser is stateful and NOT thread-safe
# ("not safe to call ts_parser_parse from multiple threads at once" — api.h).
# Languages are immutable/thread-safe, so they stay in the shared _LANG_CACHE;
# only the Parser (which holds mutable parse state) is isolated per thread.
# Thread-pool workers are reused, so each worker pays the Parser-construction
# cost once and then reuses its own instance with no locking on the parse path.
_PARSER_TLS = threading.local()

# Sentinel distinguishing "no entry yet" from a cached None (negative cache
# from a failed language-binding load). A module-level object is identity-safe
# across threads. Used by the parser TLS cache AND the compiled-query cache
# (_compile_query) — the single sentinel serves both.
_MISS = object()

# Memoised UTF-8 encoding: the same content is encoded by query_captures,
# query_matches, extract_import_names, and find_anchor_node.  Caching avoids
# re-encoding a 300 KB file 4 times per scan pipeline.
@lru_cache(maxsize=128)
def _encode_content(content: str) -> bytes:
    """Memoised UTF-8 encoding of *content*."""
    return content.encode("utf-8")


# Leading import-keyword prefix stripped from an @source capture that spans
# the whole import declaration.  Scala's grammar inlines the dotted path into
# separate identifier nodes, so we capture the declaration node and drop the
# keyword here.  No-op for languages whose @source capture is already the bare
# module path (identifier / qualified_name / string literal).  Scala only uses
# `import` (never `using`) for imports, so that is the only keyword handled.
_IMPORT_KW_RE = re.compile(r"^import\s+")


# Map our language ids to the tree-sitter language module import path
_LANG_MODULE_MAP = {
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",  # same package exports language_tsx()
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "kotlin": "tree_sitter_kotlin",
    "python": "tree_sitter_python",
    "html": "tree_sitter_html",
    "css": "tree_sitter_css",
    # Full AST support (symbol/call/import queries available)
    "rust": "tree_sitter_rust",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
    "php": "tree_sitter_php",
    "c_sharp": "tree_sitter_c_sharp",
    "swift": "tree_sitter_swift",
    "scala": "tree_sitter_scala",
    "lua": "tree_sitter_lua",
    "bash": "tree_sitter_bash",
}
# JS/TS file-extension → tree-sitter grammar key mapping.
# Single source of truth so that code_integrity.py, base.py, and other callers
# don't duplicate this logic.
_EXT_TO_GRAMMAR_KEY: dict[str, str] = {
    # JS/TS
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Go / Java / Kotlin / Python
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".py": "python",
    ".pyi": "python",
    # C / C++ family
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    # Other full-AST languages (symbol / call / import queries available)
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sc": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}


def grammar_key_for_path(file_path: str) -> str | None:
    """Return tree-sitter grammar key for *file_path*, or ``None`` for unknown extensions.

    Single canonical mapping for file extension → tree-sitter grammar key.
    Covers all languages with full AST query support (symbol, call, import, reference queries).

    ``None`` tells the caller to use its own default (typically ``"typescript"``
    or ``lang_id.value``), so this function stays a pure extension-to-key mapper
    without imposing a fallback policy.
    """
    if not file_path:
        return None
    return _EXT_TO_GRAMMAR_KEY.get(os.path.splitext(file_path)[1].lower())


def grammar_key_for_ext(ext: str) -> str | None:
    """Return tree-sitter grammar key for a file *ext* (including leading dot), or ``None``.

    Same canonical mapping as :func:`grammar_key_for_path` but takes a bare extension
    (e.g. ``.tsx``, ``.go``) instead of a full file path.  Use when the caller already
    has the extension and wants to avoid constructing a dummy path.
    """
    return _EXT_TO_GRAMMAR_KEY.get(ext.lower())


# ── tree-sitter-language-pack fallback ───────────────────────────────────────
# language-pack bundles 300+ prebuilt grammars behind a unified get_language()
# API (1.9 MB, full platform wheel coverage). It is used as a FALLBACK when the
# individual tree_sitter_<lang> modules are not installed. Resolved lazily so
# the module loads fine without it.  _resolve_lang_pack() re-probes on every call
# (no negative cache), but _get_language() DOES cache failures (None) — after a
# miss, subsequent calls return None without re-probing.  Live pickup after a late
# pip-install works because the dependency checker calls _LANG_CACHE.clear() first.
_LANG_PACK_GET_LANGUAGE = None


def _resolve_lang_pack():
    """Return language-pack's get_language() callable, or None if unavailable.

    Probes on every call while unresolved so a late install is detected.
    """
    global _LANG_PACK_GET_LANGUAGE
    if _LANG_PACK_GET_LANGUAGE is not None:
        return _LANG_PACK_GET_LANGUAGE
    try:
        from tree_sitter_language_pack import get_language as _gl

        _LANG_PACK_GET_LANGUAGE = _gl
    except ImportError:
        _LANG_PACK_GET_LANGUAGE = None
    return _LANG_PACK_GET_LANGUAGE


# Our language ids that differ from language-pack naming conventions.
_LANG_PACK_ALIASES = {"c_sharp": "csharp"}

@dataclass
class QueryCapture:
    """A single capture from a tree-sitter query."""
    capture_name: str       # e.g., "sym", "def", "call"
    node_type: str          # e.g., "function_definition", "identifier"
    text: str               # source text of the captured node
    start_line: int         # 1-indexed
    end_line: int           # 1-indexed
    start_byte: int
    end_byte: int


# tree-sitter node types that represent top-level symbol definitions
_SYMBOL_NODE_TYPES = {
    # TypeScript / JavaScript
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "method_definition",
    "field_definition",
    "public_field_definition",
    "lexical_declaration",
    "export_statement",
    "enum_declaration",
    # Go
    "method_declaration",
    "type_declaration",
    "var_declaration",
    "const_declaration",
    "short_var_declaration",
    # Java / Kotlin
    "constructor_declaration",
    "object_declaration",
    # Python
    "function_definition",
    "class_definition",
    "decorated_definition",
    # Rust
    "function_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "type_item",
    "const_item",
    "static_item",
    # C
    "struct_specifier",
    "enum_specifier",
    "union_specifier",
    "type_definition",
    # C++ only
    "class_specifier",
    "namespace_definition",
    # Ruby
    "class",
    "module",
    "method",
    # PHP
    "trait_declaration",
    # C#
    "namespace_declaration",
    "struct_declaration",
    "delegate_declaration",
    # Swift
    "protocol_declaration",
    # Scala
    "object_definition",
    "trait_definition",
    # Lua
    # Bash
    # CSS — selectors are definition sites; "declaration" is also included
    # because CSS custom properties (``--name``) live in declaration nodes.
    # _extract_name filters declarations to only those whose property_name
    # starts with "--", so ordinary ``color: red`` declarations are skipped.
    "class_selector",
    "id_selector",
    "declaration",
}

# Container types that nest other symbols (class/interface/enum bodies).
# When _collect() encounters these, it records them BUT continues
# descending into children to find nested methods, fields, and inner
# classes — unlike leaf types (function_declaration, method_definition,
# field_definition) where we stop after recording.
_CONTAINER_NODE_TYPES = frozenset({
    "class_declaration", "interface_declaration", "enum_declaration",
    "export_statement",
})

# Per-language tree-sitter queries for extracting top-level symbol definitions.
# Each query captures:
#   @def — the definition node (for line range)
#   @name — the name identifier (for the symbol name string)
#   @kind — optional: a node whose type encodes the symbol kind
_SYMBOL_QUERIES: dict[str, str] = {
    "python": """
(function_definition name: (identifier) @name) @def
(class_definition name: (identifier) @name) @def
(module (expression_statement (assignment left: (identifier) @name)) @def)
""",
    "typescript": """
(function_declaration name: (identifier) @name) @def
(class_declaration name: (type_identifier) @name) @def
(interface_declaration name: (type_identifier) @name) @def
(type_alias_declaration name: (type_identifier) @name) @def
(enum_declaration name: (identifier) @name) @def
(lexical_declaration (variable_declarator name: (identifier) @name)) @def
(method_definition name: (property_identifier) @name) @def
(assignment_expression left: (member_expression) @name right: (arrow_function)) @def
(assignment_expression left: (member_expression) @name right: (function_expression)) @def
""",
    "javascript": """
(function_declaration name: (identifier) @name) @def
(class_declaration name: (identifier) @name) @def
(lexical_declaration (variable_declarator name: (identifier) @name)) @def
(method_definition name: (property_identifier) @name) @def
(assignment_expression left: (member_expression) @name right: (arrow_function)) @def
(assignment_expression left: (member_expression) @name right: (function_expression)) @def
""",
    "go": """
(function_declaration name: (identifier) @name) @def
(method_declaration name: (field_identifier) @name) @def
(type_declaration (type_spec name: (type_identifier) @name)) @def
(var_declaration (var_spec name: (identifier) @name)) @def
(const_declaration (const_spec name: (identifier) @name)) @def
""",
    "java": """
(class_declaration name: (identifier) @name) @def
(interface_declaration name: (identifier) @name) @def
(enum_declaration name: (identifier) @name) @def
(method_declaration name: (identifier) @name) @def
(constructor_declaration name: (identifier) @name) @def
""",
    "kotlin": """
(class_declaration name: (type_identifier) @name) @def
(interface_declaration name: (type_identifier) @name) @def
(object_declaration name: (type_identifier) @name) @def
(function_declaration name: (simple_identifier) @name) @def
""",
    "rust": """
(function_item name: (identifier) @name) @def
(struct_item name: (type_identifier) @name) @def
(enum_item name: (type_identifier) @name) @def
(trait_item name: (type_identifier) @name) @def
(type_item name: (type_identifier) @name) @def
(const_item name: (identifier) @name) @def
(static_item name: (identifier) @name) @def
""",
    "c": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
(struct_specifier name: (type_identifier) @name) @def
(enum_specifier name: (type_identifier) @name) @def
(union_specifier name: (type_identifier) @name) @def
(type_definition declarator: (type_identifier) @name) @def
""",
    "cpp": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def
(struct_specifier name: (type_identifier) @name) @def
(enum_specifier name: (type_identifier) @name) @def
(union_specifier name: (type_identifier) @name) @def
(type_definition declarator: (type_identifier) @name) @def
(class_specifier name: (type_identifier) @name) @def
(namespace_definition name: (namespace_identifier) @name) @def
""",
    "ruby": """
(class name: (constant) @name) @def
(module name: (constant) @name) @def
(method name: (identifier) @name) @def
""",
    "php": """
(class_declaration name: (name) @name) @def
(interface_declaration name: (name) @name) @def
(trait_declaration name: (name) @name) @def
(enum_declaration name: (name) @name) @def
(function_definition name: (name) @name) @def
""",
    "c_sharp": """
(namespace_declaration name: (identifier) @name) @def
(class_declaration name: (identifier) @name) @def
(struct_declaration name: (identifier) @name) @def
(interface_declaration name: (identifier) @name) @def
(enum_declaration name: (identifier) @name) @def
(delegate_declaration name: (identifier) @name) @def
(method_declaration name: (identifier) @name) @def
""",
    "swift": """
(class_declaration name: (type_identifier) @name) @def
(protocol_declaration name: (type_identifier) @name) @def
(function_declaration name: (simple_identifier) @name) @def
""",
    "scala": """
(class_definition name: (identifier) @name) @def
(object_definition name: (identifier) @name) @def
(trait_definition name: (identifier) @name) @def
(function_definition name: (identifier) @name) @def
""",
    "lua": """
(function_declaration name: (identifier) @name) @def
""",
    "bash": """
(function_definition name: (word) @name) @def
""",
}


_SYMBOL_QUERIES["tsx"] = _SYMBOL_QUERIES["typescript"]
# Per-language queries for extracting call expressions.
# Captures:
#   @call  — the call expression node itself
#   @callee — the name being called (identifier or property)
_CALL_QUERIES: dict[str, str] = {
    "python": """
(call function: (identifier) @callee) @call
(call function: (attribute attribute: (identifier) @callee)) @call
""",
    "typescript": """
(call_expression function: (identifier) @callee) @call
(call_expression function: (member_expression property: (property_identifier) @callee)) @call
""",
    "javascript": """
(call_expression function: (identifier) @callee) @call
(call_expression function: (member_expression property: (property_identifier) @callee)) @call
""",
    "go": """
(call_expression function: (identifier) @callee) @call
(call_expression function: (selector_expression field: (field_identifier) @callee)) @call
""",
    "java": """
(method_invocation name: (identifier) @callee) @call
""",
    "kotlin": """
(call_expression (simple_identifier) @callee) @call
(call_expression (navigation_expression (simple_identifier) @callee)) @call
""",
    "rust": """
(call_expression function: (identifier) @callee) @call
(call_expression function: (field_expression field: (field_identifier) @callee)) @call
(call_expression function: (scoped_identifier name: (identifier) @callee)) @call
""",
    "c": """
(call_expression function: (identifier) @callee) @call
""",
    "cpp": """
(call_expression function: (identifier) @callee) @call
""",
    "ruby": """
(call method: (identifier) @callee) @call
""",
    "php": """
(function_call_expression function: (name) @callee) @call
""",
    "c_sharp": """
(invocation_expression function: (identifier) @callee) @call
(invocation_expression function: (member_access_expression name: (identifier) @callee)) @call
""",
    "swift": """
(call_expression (simple_identifier) @callee) @call
""",
    "scala": """
(call_expression function: (identifier) @callee) @call
(call_expression function: (field_expression field: (identifier) @callee)) @call
""",
    "lua": """
(function_call name: (identifier) @callee) @call
""",
    "bash": """
(command name: (command_name) @callee) @call
""",
}

_CALL_QUERIES["tsx"] = _CALL_QUERIES["typescript"]
# Per-language queries for extracting import statements.
# Captures:
#   @import — the import statement node
#   @source — the module path string content
_IMPORT_QUERIES: dict[str, str] = {
    "python": """
(import_statement name: (dotted_name) @source) @import
(import_from_statement module_name: (dotted_name) @source) @import
""",
    "typescript": """
(import_statement source: (string (string_fragment) @source)) @import
""",
    "javascript": """
(import_statement source: (string (string_fragment) @source)) @import
""",
    "go": """
(import_declaration (import_spec path: (interpreted_string_literal) @source)) @import
""",
    "java": """
(import_declaration (scoped_identifier) @source) @import
""",
    "kotlin": """
(import_header (identifier) @source) @import
""",
    "rust": """
(use_declaration (scoped_identifier) @source) @import
""",
    "c": """
(preproc_include (system_lib_string) @source) @import
(preproc_include (string_literal) @source) @import
""",
    "cpp": """
(preproc_include (system_lib_string) @source) @import
(preproc_include (string_literal) @source) @import
""",
    "php": """
(namespace_use_declaration (namespace_use_clause) @source) @import
""",
    "c_sharp": """
(using_directive (identifier) @source) @import
(using_directive (qualified_name) @source) @import
""",
    "swift": """
(import_declaration (identifier) @source) @import
""",
    "scala": """
(import_declaration) @source
""",
    # Ruby: require/require_relative "<gem>" — @source is the string argument.
    "ruby": """
(call (identifier) @_fn (#match? @_fn "^(require|require_relative)$") (argument_list (string) @source))
""",
    # Lua: require "mod" / require("mod") — @source is the string argument.
    "lua": """
(function_call (identifier) @_fn (#eq? @_fn "require") (arguments (string) @source))
""",
    # Bash: source file / . file — @source is the word *or* string argument.
    # Bare words (source ./script.sh) parse as (word); quoted strings
    # (source "$dir/script.sh") parse as (string).  Both must be captured.
    "bash": """
(command (command_name) @_cmd (#match? @_cmd "^(source|\\.)$")
  [
    (word) @source
    (string) @source
  ]
)
""",
}


_IMPORT_QUERIES["tsx"] = _IMPORT_QUERIES["typescript"]
# Per-language queries for extracting identifier references.
# Captures:
#   @ref — an identifier node in a potentially-referencing context
# Note: tree-sitter does not distinguish Load vs Store; the consumer
# must filter out identifiers that appear in definition positions
# (e.g., function name, class name, variable declarator name).
_REFERENCE_QUERIES: dict[str, str] = {
    "python": "(identifier) @ref",
    "typescript": """
(identifier) @ref
(property_identifier) @ref
""",
    "javascript": """
(identifier) @ref
(property_identifier) @ref
""",
    "go": """
(identifier) @ref
(field_identifier) @ref
""",
    "java": "(identifier) @ref",
    "kotlin": "(simple_identifier) @ref",
    "rust": """
(identifier) @ref
(type_identifier) @ref
(field_identifier) @ref
""",
    "c": "(identifier) @ref",
    "cpp": "(identifier) @ref",
    "ruby": "(identifier) @ref",
    "php": "(name) @ref",
    "c_sharp": "(identifier) @ref",
    "swift": "(simple_identifier) @ref",
    "scala": "(identifier) @ref",
    "lua": "(identifier) @ref",
    "bash": "(variable_name) @ref",
}


_REFERENCE_QUERIES["tsx"] = _REFERENCE_QUERIES["typescript"]
def get_available_languages() -> set[str]:
    """Return set of language names whose tree-sitter bindings are installed."""
    available = set()
    for lang in _LANG_MODULE_MAP:
        if _get_language(lang) is not None:
            available.add(lang)
    return available


def is_available() -> bool:
    """Return True if tree-sitter core library is installed."""
    return _HAS_TREE_SITTER


def _get_language(language: str) -> object | None:
    """Get a tree-sitter Language object for *language*, or None (thread-safe)."""
    if not _HAS_TREE_SITTER:
        return None

    with _LANG_CACHE_LOCK:
        if language in _LANG_CACHE:
            return _LANG_CACHE[language]

        # All languages are imported in the order registered in _LANG_MODULE_MAP.
        # Unregistered languages also fall back via standard naming convention (tree_sitter_<lang>).
        # Also handles non-standard modules using language_<lang>() naming like PHP.
        module_name = _LANG_MODULE_MAP.get(language) or f"tree_sitter_{language}"

        try:
            import importlib

            mod = importlib.import_module(module_name)
            # tree-sitter-typescript exposes .language_typescript() and .language_tsx()
            if language == "typescript":
                raw = mod.language_typescript()
            elif language == "tsx":
                raw = mod.language_tsx()
            else:
                # Standard convention: module.language()
                try:
                    raw = mod.language()
                except AttributeError:
                    # Fallback for modules (e.g., tree-sitter-php) that use
                    # module.language_<lang>() naming convention
                    raw = getattr(mod, f"language_{language}")()
            # tree-sitter ≥0.23 returns PyCapsule; wrap with Language()
            if not isinstance(raw, _ts.Language):
                raw = _ts.Language(raw)
            _LANG_CACHE[language] = raw
            return raw
        except (ImportError, AttributeError, TypeError) as e:
            # Fallback: tree-sitter-language-pack bundles 300+ prebuilt grammars
            # behind a unified get_language() API. Used when individual
            # tree_sitter_<lang> modules are not installed. (c_sharp → csharp
            # is the only id that differs from our naming.)
            lp = _resolve_lang_pack()
            if lp is not None:
                try:
                    raw = lp(_LANG_PACK_ALIASES.get(language, language))
                except Exception as e2:  # pack may lack the grammar
                    logger.debug(
                        "tree-sitter language-pack %s not available: %s", language, e2
                    )
                else:
                    if not isinstance(raw, _ts.Language):
                        raw = _ts.Language(raw)
                    _LANG_CACHE[language] = raw
                    return raw
            logger.debug("tree-sitter language %s not available: %s", language, e)
            _LANG_CACHE[language] = None  # type: ignore[assignment]
            return None


def get_parser(language: str):
    """Return a tree-sitter Parser for *language*, or None if not installed.

    Parsers are **per-thread** (``threading.local``). tree-sitter's
    ``TSParser`` holds mutable parse state and is explicitly not safe to call
    concurrently from multiple threads (see ``api.h``). Thread-pool workers
    reuse their worker thread, so each worker constructs its parser once and
    then hits a cached instance on every subsequent call — no locking on the
    hot parse path, and no cross-thread sharing.

    Languages are immutable and thread-safe, so they live in the shared
    ``_LANG_CACHE`` (guarded by ``_LANG_CACHE_LOCK``).
    """
    if not _HAS_TREE_SITTER:
        return None

    lang_obj = _get_language(language)
    if lang_obj is None:
        return None

    cache = getattr(_PARSER_TLS, "cache", None)
    if cache is None:
        cache = {}
        _PARSER_TLS.cache = cache

    cached = cache.get(language, _MISS)
    if cached is not _MISS:
        return cached

    try:
        parser = _ts.Parser(lang_obj)
    except Exception as e:
        logger.debug("Failed to create parser for %s: %s", language, e)
        # Negative-cache the failure per thread so we don't retry the failing
        # language-binding import/construction on every call.
        cache[language] = None
        return None

    cache[language] = parser
    return parser


# ── Query API ─────────────────────────────────────────────────────────────────

# Compiled Query cache: (language_name, query_string) → tree_sitter.Query | None (thread-safe).
# Query objects are immutable once compiled and are safe to share across threads
# (only QueryCursor, which is created fresh per call, carries per-match state).
# Query strings come from a small finite set of module constants (_SYMBOL_QUERIES,
# _CALL_QUERIES, ...), so this cache has effectively unbounded hit rate with a
# tiny memory footprint. Compilation is non-trivial, so caching avoids rebuilding
# the same Query on every parse.
_QUERY_CACHE: dict[tuple[str, str], object] = {}
_QUERY_CACHE_LOCK = threading.RLock()

# _MISS sentinel is defined at module level (see above) — reused here for
# the query cache's absence-check pattern.

def invalidate_caches() -> None:
    """Atomically clear all tree-sitter caches (language, parse, query).

    Called by the dependency checker after a late pip-install so that newly
    installed grammars take effect without a process restart.
    All three caches are cleared under the ``_LANG_CACHE_LOCK`` (an RLock) to
    prevent interleaved reads from seeing a partially-invalidated state.
    ``_QUERY_CACHE_LOCK`` is also acquired in the correct lock order
    (``_LANG_CACHE_LOCK`` → ``_QUERY_CACHE_LOCK``, matching ``_get_language``)
    to avoid deadlock.
    """
    with _LANG_CACHE_LOCK:
        _LANG_CACHE.clear()
        parse_to_tree.cache_clear()
        with _QUERY_CACHE_LOCK:
            _QUERY_CACHE.clear()


def _compile_query(language: str, lang_obj, query_string: str):
    """Compile (or fetch a cached) tree-sitter Query for *lang_obj*.

    Returns a ``tree_sitter.Query`` or None if the query string is invalid.
    Thread-safe: Query objects are immutable, so a shared cache guarded by
    ``_QUERY_CACHE_LOCK`` is sufficient.

    Cache keyed by ``(language, query_string)`` — the language name string,
    not ``id(lang_obj)``, to avoid stale hits after a GC reuses an address.
    Failed compilations (None) are also cached via a sentinel so that an
    invalid query string does not trigger re-compilation on every call.
    """
    if not _HAS_TREE_SITTER:
        return None

    cache_key = (language, query_string)
    with _QUERY_CACHE_LOCK:
        cached = _QUERY_CACHE.get(cache_key, _MISS)
        if cached is not _MISS:
            return cached

    try:
        query = lang_obj.query(query_string)
    except AttributeError:
        try:
            query = _ts.Query(lang_obj, query_string)
        except Exception:
            query = None
    except Exception:
        # Invalid query string (QueryError) or other issue
        query = None

    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE[cache_key] = query
    return query


def query_captures(
    content: str, language: str, query_string: str,
) -> list[QueryCapture]:
    """Run a tree-sitter query and return all captured nodes.

    Each capture includes metadata (capture name, node type, source text,
    1-indexed line range, byte offsets).

    Returns an empty list if tree-sitter is unavailable, the language grammar
    is missing, parsing fails, or the query is invalid.
    """
    lang_obj = _get_language(language)
    if lang_obj is None:
        return []

    # parse_to_tree is memoised (@lru_cache): callers that run several
    # queries against the same source (imports, exports, symbols, calls)
    # share a single parse instead of re-parsing each time.
    tree = parse_to_tree(content, language)
    if tree is None:
        return []

    query = _compile_query(language, lang_obj, query_string)
    if query is None:
        return []

    try:
        cursor = _ts.QueryCursor(query)
        captures_raw = cursor.captures(tree.root_node)
    except Exception:
        return []

    code_bytes = _encode_content(content)
    results: list[QueryCapture] = []

    for capture_name, nodes in captures_raw.items():
        for node in nodes:
            results.append(QueryCapture(
                capture_name=capture_name,
                node_type=node.type,
                text=code_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            ))

    results.sort(key=lambda c: c.start_byte)
    return results


def query_matches(
    content: str, language: str, query_string: str,
) -> list[dict[str, list[QueryCapture]]]:
    """Run a tree-sitter query and return matches grouped by pattern.

    Each element in the returned list corresponds to one pattern match.
    The dict maps capture name to a list of ``QueryCapture`` objects for
    that capture within the match.  This is useful when captures within
    the same pattern need to be associated (e.g., ``@def`` and ``@name``
    from the same function definition).
    """
    lang_obj = _get_language(language)
    if lang_obj is None:
        return []

    # parse_to_tree is memoised (@lru_cache): callers that run several
    # queries against the same source (imports, exports, symbols, calls)
    # share a single parse instead of re-parsing each time.
    tree = parse_to_tree(content, language)
    if tree is None:
        return []

    query = _compile_query(language, lang_obj, query_string)
    if query is None:
        return []

    try:
        cursor = _ts.QueryCursor(query)
        matches_raw = cursor.matches(tree.root_node)
    except Exception:
        return []

    code_bytes = _encode_content(content)
    results: list[dict[str, list[QueryCapture]]] = []

    for (_pattern_idx, captures_dict) in matches_raw:
        match_result: dict[str, list[QueryCapture]] = {}
        for capture_name, nodes in captures_dict.items():
            match_result[capture_name] = [
                QueryCapture(
                    capture_name=capture_name,
                    node_type=node.type,
                    text=code_bytes[node.start_byte:node.end_byte].decode("utf-8"),
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
                for node in nodes
            ]
        results.append(match_result)

    return results


def has_error(content: str, language: str) -> Optional[bool]:
    """Check whether *content* has syntax errors for *language*.

    Returns True if the parse tree contains ERROR or MISSING nodes,
    False if the parse is clean, None if tree-sitter is unavailable.

    The tree traversal is iterative (explicit stack) rather than recursive so
    that deeply nested / machine-generated inputs — which can exceed Python's
    default recursion limit (1000) and would otherwise raise ``RecursionError``
    — are handled safely. Callers (e.g. ``validate_syntax``) rely on this as a
    hard gate on every write path, so it must never propagate.
    """
    tree = parse_to_tree(content, language)
    if tree is None:
        return None

    # Fast-path: root_node.has_error is an O(1) cached flag maintained by
    # tree-sitter internally, covering both ERROR and MISSING descendant nodes.
    # Benchmarked ~99,000x faster than a full DFS for a 2000-line valid file.
    if not tree.root_node.has_error:
        return False

    # has_error is definitionally equivalent to "any descendant is ERROR or
    # MISSING" — verified empirically across C/Java/Go error variants (see
    # test_tree_sitter_utils.py for exhaustive coverage).  The remaining DFS
    # would always yield the same True, so we short-circuit to the return.
    return True


@dataclass(frozen=True)
class SyntaxErrorNode:
    """Structured syntax error from tree-sitter ERROR/MISSING node.

    Used by failure classifier Layer A for structural syntax error detection.
    """
    kind: str            # "ERROR" | "MISSING"
    missing_token: str   # MISSING node's expected token (e.g. ";", ")")
    line: int            # 0-based line number
    column: int          # 0-based column number
    context_snippet: str # Surrounding source code (for repair prompts)


def find_error_nodes(content: str, language: str) -> Optional[list[SyntaxErrorNode]]:
    """Collect all ERROR/MISSING nodes from tree-sitter parse.

    Returns list of SyntaxErrorNode if tree-sitter is available,
    None if tree-sitter is unavailable (fallback signal for classifier).

    MISSING nodes have node.type as the expected token (e.g. ";", ")"),
    which provides a free FixHint for repair strategies.

    Uses iterative DFS (same as has_error) to avoid recursion limit issues.
    """
    tree = parse_to_tree(content, language)
    if tree is None:
        return None

    errors: list[SyntaxErrorNode] = []
    lines = content.splitlines()

    # Iterative DFS: collect all ERROR/MISSING nodes
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            # Extract context snippet (surrounding lines)
            start_line = node.start_point[0]
            end_line = min(node.end_point[0] + 1, len(lines))
            context = "\n".join(lines[max(0, start_line - 1):end_line + 1])

            errors.append(SyntaxErrorNode(
                kind="MISSING" if node.is_missing else "ERROR",
                missing_token=node.type if node.is_missing else "",
                line=node.start_point[0],  # 0-based
                column=node.start_point[1],
                context_snippet=context,
            ))
        stack.extend(node.children)

    return errors


def extract_symbol_at_position(
    content: str, language: str, line: int, column: int,
) -> Optional[str]:
    """Extract identifier/type at (line, column) using tree-sitter.

    Line and column are 1-based (matching VerifyError convention).
    Returns the identifier text if found, None otherwise.

    This replaces regex-based symbol extraction with structural lookup,
    handling qualified names (pkg.Foo), generics, backticks, and Unicode
    identifiers correctly.
    """
    tree = parse_to_tree(content, language)
    if tree is None:
        return None

    # Convert to 0-based for tree-sitter
    point = (line - 1, column - 1)
    node = tree.root_node.descendant_for_point_range(point, point)
    if node is None:
        return None

    # Walk up to find identifier/type_identifier node
    current = node
    while current is not None:
        if current.type in ("identifier", "type_identifier", "field_identifier"):
            return current.text.decode("utf-8") if isinstance(current.text, bytes) else current.text
        current = current.parent

    # Fallback: return the node's text if it looks like an identifier
    text = node.text.decode("utf-8") if isinstance(node.text, bytes) else node.text
    if text and text.isidentifier():
        return text

    return None


def extract_calls(
    content: str, language: str,
) -> list[tuple[str, int]]:
    """Extract call sites: ``[(callee_name, line), ...]``.

    Lines are 1-indexed.  Returns an empty list if tree-sitter is unavailable
    or no call query is defined for *language*.
    """
    query_str = _CALL_QUERIES.get(language)
    if query_str is None:
        return []

    caps = query_captures(content, language, query_str)
    results: list[tuple[str, int]] = []
    seen: set = set()
    for c in caps:
        if c.capture_name != "callee":
            continue
        key = (c.text, c.start_line)
        if key in seen:
            continue
        seen.add(key)
        results.append((c.text, c.start_line))
    return results


def extract_imports(
    content: str, language: str,
) -> list[tuple[str, int]]:
    """Extract import statements: ``[(imported_module, line), ...]``.

    The module string is cleaned: surrounding quotes and trailing semicolons
    are stripped.

    Lines are 1-indexed.  Returns an empty list if tree-sitter is unavailable
    or no import query is defined for *language*.
    """
    query_str = _IMPORT_QUERIES.get(language)
    if query_str is None:
        return []

    caps = query_captures(content, language, query_str)
    results: list[tuple[str, int]] = []
    seen: set = set()
    for c in caps:
        if c.capture_name != "source":
            continue
        # Clean module path: strip surrounding whitespace, quotes, and semicolons.
        module = c.text.strip().strip('\"\';')
        # Scala captures the whole import_declaration node, which includes the
        # leading `import`/`using` keyword; strip it.  Every other language's
        # @source capture is a child node that already excludes the keyword, so
        # the strip is a no-op there — but scope it to Scala to avoid corrupting
        # a pathological module path such as lua `require("import foo")`.
        if language == "scala":
            module = _IMPORT_KW_RE.sub("", module)
            # Strip namespace selectors and wildcard suffixes from module path.
            # tree-sitter-scala has no scoped_identifier node, so @source captures
            # the entire declaration including selectors like `{c, d}` / `_` / `*`.
            #   e.g., "a.b.{c, d}" → "a.b",  "a.b._" → "a.b",  "a.b.*" → "a.b"
            module = re.sub(r"\.(?:\{[^}]*\}|_|\*)\s*$", "", module)
        key = (module, c.start_line)
        if key in seen:
            continue
        seen.add(key)
        results.append((module, c.start_line))
    return results


def extract_import_names(
    content: str, language: str,
) -> list[tuple[str, str]]:
    """Extract names bound by imports: ``[(module_path, name), ...]``.

    Unlike ``extract_imports`` (module paths only), this returns the
    individual symbol names each import binds.  Both the original exported
    name and the local alias are emitted when they differ —
    ``import { A as B } from './m'`` yields ``('./m', 'A')`` and
    ``('./m', 'B')`` — because dead-code analysis needs the original name
    (it matches the definition in the source module) while reference
    counting needs the local alias.

    Re-exports (``export { X } from './m'``) are included: they reference
    the symbol in the source module just like imports do.

    TypeScript/JavaScript only.  Returns an empty list for other languages
    or when tree-sitter is unavailable.
    """
    if language not in ("typescript", "javascript", "tsx"):
        return []
    tree = parse_to_tree(content, language)
    if tree is None:
        return []
    code_bytes = _encode_content(content)
    results: list[tuple[str, str]] = []

    def _text(n) -> str:
        return get_node_text(code_bytes, n)

    def _emit_specifiers(clause_node, module: str) -> None:
        for sub in clause_node.children:
            if sub.type == "identifier":  # default import binding
                results.append((module, _text(sub)))
            elif sub.type == "namespace_import":  # import * as ns
                for nch in sub.children:
                    if nch.type == "identifier":
                        results.append((module, _text(nch)))
            elif sub.type in ("named_imports", "export_clause"):
                for spec in sub.children:
                    if spec.type not in ("import_specifier", "export_specifier"):
                        continue
                    name_node = spec.child_by_field_name("name")
                    alias_node = spec.child_by_field_name("alias")
                    if name_node is not None:
                        results.append((module, _text(name_node)))
                    if alias_node is not None:
                        results.append((module, _text(alias_node)))

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. Mirrors the original
    # recursive _walk: pre-order traversal over ALL children (named + unnamed).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("import_statement", "export_statement"):
            src = node.child_by_field_name("source")
            if src is not None:
                module = _text(src).strip().strip("'\"")
                for ch in node.children:
                    if ch.type == "import_clause":
                        _emit_specifiers(ch, module)
                    elif ch.type == "export_clause":
                        _emit_specifiers(node, module)
                        break
        stack.extend(reversed(node.children))
    return results


def _extract_name(node) -> Optional[str]:
    """Extract the symbol name from a tree-sitter node."""
    # CSS selectors: class_selector → class_name child, id_selector → id_name
    # child. These node types are CSS-only, so no language guard is needed.
    if node.type == "class_selector":
        for ch in node.named_children:
            if ch.type == "class_name":
                return ch.text.decode("utf-8")
        return None
    if node.type == "id_selector":
        for ch in node.named_children:
            if ch.type == "id_name":
                return ch.text.decode("utf-8")
        return None
    # CSS custom property (--name): lives inside a declaration's property_name.
    # Ordinary declarations (``color: red``) have non-"--" property names and
    # return None here, so they are skipped by the caller's symbol collection.
    if node.type == "declaration":
        for ch in node.named_children:
            if ch.type == "property_name":
                name = ch.text.decode("utf-8")
                if name.startswith("--"):
                    return name
                return None  # ordinary CSS property — not a symbol
        return None

    # For export statements, look inside
    if node.type == "export_statement":
        for child in node.children:
            if child.type in _SYMBOL_NODE_TYPES:
                return _extract_name(child)
        # export default function/class
        for child in node.children:
            name = _extract_name(child)
            if name:
                return name
        return None

    # For lexical_declaration (const/let/var), get the variable name
    if node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
        return None

    # For type_declaration (Go: type X struct{...}), find the type_spec
    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
        return None

    # For Python decorated_definition, look inside for the actual definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_name(child)
        return None

    # Go var_declaration: "var x int" or "var x = 1" or "var (...)"
    if node.type == "var_declaration":
        for child in node.children:
            if child.type == "var_spec":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
        return None

    # Go const_declaration: "const x = 1" or "const (...)"
    if node.type == "const_declaration":
        for child in node.children:
            if child.type == "const_spec":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
        return None

    # Standard: node has a "name" field
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    # Fallback for grammars that expose the symbol name as a positional named
    # child rather than a "name" field. Kotlin is the known case: its
    # function_declaration / class_declaration / object_declaration / etc. carry
    # the name as a bare simple_identifier / type_identifier named child, so
    # child_by_field_name("name") returns None. Without this fallback every
    # Kotlin symbol is skipped by find_all_symbols, forcing the caller
    # (symbol_modify_tool._find_symbol_line_range) onto the naive
    # brace-counting range heuristic — which miscounts braces inside string /
    # comment literals and corrupts the file on edit.
    #
    # The first identifier-typed named child is the symbol name in every such
    # grammar: modifiers (private/public/override/...) precede the name but are
    # not identifier-typed, and post-name constructs (parameter lists, return
    # types, bodies) come after. Verified against Kotlin/Scala/Swift/Java/Go
    # grammars — only Kotlin lacks the "name" field, but this is grammar-version
    # agnostic and will cover any future grammar with the same shape.
    for child in node.named_children:
        if child.type in ("simple_identifier", "type_identifier", "identifier"):
            return child.text.decode("utf-8")
    return None


# ── Node-type → kind SSOT ────────────────────────────────────────────────────
# Single source of truth shared by BOTH the manual-walk path (``_node_kind``)
# and the query path (``_node_kind_from_type``).  Keeping one dict prevents the
# two from silently drifting on shared node types — a class of bug that bit us
# before (``lexical_declaration``/``object_declaration`` mapped inconsistently
# between the two paths).  ``test_walk_and_query_agree_on_common_keys`` pins this.
_BASE_KIND_MAP = {
    # TypeScript / JavaScript
    "function_declaration": "function",
    "method_definition": "function",
    "method_declaration": "function",
    "field_definition": "assignment",
    "public_field_definition": "assignment",
    "expression_statement": "assignment",
    "constructor_declaration": "function",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "lexical_declaration": "assignment",
    "variable_declaration": "assignment",
    # Go
    "type_declaration": "type",
    "var_declaration": "variable",
    "const_declaration": "constant",
    "short_var_declaration": "variable",
    # Python
    "function_definition": "function",
    "class_definition": "class",
    "async_function_definition": "function",
    # Rust
    "function_item": "function",
    "struct_item": "class",
    "enum_item": "enum",
    "trait_item": "interface",
    "type_item": "type",
    "const_item": "constant",
    "static_item": "constant",
    # C
    "struct_specifier": "class",
    "enum_specifier": "enum",
    "union_specifier": "class",
    "type_definition": "type",
    # C++ only
    "class_specifier": "class",
    "namespace_definition": "namespace",
    # Ruby
    "class": "class",
    "module": "namespace",
    "method": "function",
    # PHP
    "trait_declaration": "class",
    # C#
    "namespace_declaration": "namespace",
    "struct_declaration": "class",
    "delegate_declaration": "function",
    # Kotlin
    "object_declaration": "class",
    # Swift
    "protocol_declaration": "interface",
    # Scala
    "object_definition": "class",
    "trait_definition": "interface",
}

# CSS-only overlay — walk path only.  CSS uses no declarative query
# (``_SYMBOL_QUERIES`` has no CSS entry), so these never reach the query path.
# Selectors and custom properties get CSS-specific kinds (not the generic
# "class"/"variable") so find_symbol's dispatch can route them distinctly and
# they don't collide with Go structs / Rust consts.
_CSS_KIND_MAP = {
    "class_selector": "css_class",
    "id_selector": "css_id",
    "declaration": "css_variable",
}

_WALK_KIND_MAP = {**_BASE_KIND_MAP, **_CSS_KIND_MAP}


def _node_kind(node) -> str:
    """Map tree-sitter node type to our kind strings (manual-walk path)."""
    t = node.type
    if t == "export_statement":
        for child in node.children:
            if child.type in _SYMBOL_NODE_TYPES:
                return _node_kind(child)
        return "function"
    if t == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition",
                              "async_function_definition"):
                return _node_kind(child)
        return "function"
    return _WALK_KIND_MAP.get(t, "function")


def find_symbol_range(
    content: str, symbol_name: str, language: str
) -> Optional[tuple[int, int]]:
    """Find (start_line, end_line) of *symbol_name* using tree-sitter AST.

    Lines are 1-indexed.  Returns None if tree-sitter is unavailable or
    the symbol is not found.

    Tries a declarative query first, then falls back to manual tree walk
    for languages without a symbol query defined.
    """
    query_str = _SYMBOL_QUERIES.get(language)
    if query_str is not None:
        # Query-based path: fast, declarative, catch decorated/export wrappers
        matches = query_matches(content, language, query_str)
        for match_group in matches:
            name_caps = match_group.get("name", [])
            def_caps = match_group.get("def", [])
            if name_caps and def_caps:
                if name_caps[0].text == symbol_name:
                    return (def_caps[0].start_line, def_caps[0].end_line)
        return None

    # Manual traversal fallback for languages without a query defined.
    # parse_to_tree is memoised (@lru_cache), so this shares a single parse
    # with any concurrent query-based analysis of the same source.
    tree = parse_to_tree(content, language)
    if tree is None:
        return None

    root = tree.root_node

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. Returns the FIRST match in
    # pre-order (mirrors the original recursive _search).
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in _SYMBOL_NODE_TYPES:
            name = _extract_name(node)
            if name == symbol_name:
                return (node.start_point.row + 1, node.end_point.row + 1)
        # Extend in reverse so children are visited in original order.
        stack.extend(reversed(node.children))
    return None


def find_all_symbols(
    content: str, language: str
) -> list[tuple[str, str, int, int]]:
    """Extract all top-level symbols: ``[(name, kind, start_line, end_line), ...]``.

    Lines are 1-indexed.  Returns empty list if tree-sitter is unavailable.

    Merges results from BOTH declarative query and manual tree walk.
    The manual walk catches symbol types not covered by queries
    (e.g., ``field_definition`` in TypeScript, which breaks multi-pattern
    queries when included).

    Duplicates (same name + same line range) are removed.
    """
    # Phase 1: Declarative query (fast, handles standard constructs)
    results: list[tuple[str, str, int, int]] = []
    query_str = _SYMBOL_QUERIES.get(language)
    if query_str is not None:
        _query_results = _find_all_symbols_via_query(content, language, query_str)
        results.extend(_query_results)

    # Phase 2: Manual tree walk (catches field_definition and other
    # symbol types excluded from the query because they break
    # multi-pattern matching in tree-sitter).
    # parse_to_tree is memoised (@lru_cache), so this shares a single parse
    # with Phase 1's query (which also goes through parse_to_tree).
    tree = parse_to_tree(content, language)
    if tree is None:
        return results

    root = tree.root_node

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. Mirrors the original
    # _collect() descent rules exactly.
    seen: set = set()
    stack = [root]
    while stack:
        node = stack.pop()
        descend = True
        if node.type in _SYMBOL_NODE_TYPES:
            name = _extract_name(node)
            if name:
                kind = _node_kind(node)
                start = node.start_point.row + 1
                end = node.end_point.row + 1
                # Deduplicate against existing results (O(1) set lookup)
                dedup_key = (name, start, end)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    results.append((name, kind, start, end))
                # Container types (class/interface/enum/export): record AND
                # descend into children to find nested symbols (methods,
                # fields, inner classes, etc.). Non-containers do NOT descend.
                if node.type not in _CONTAINER_NODE_TYPES:
                    descend = False
            # name extraction failure → still descend (descend stays True)
        if descend:
            # Extend in reverse so children are visited in original order
            # (we pop from the end of the stack).
            stack.extend(reversed(node.children))


    # Remove any query-result duplicates that manual walk may have
    # produced (e.g., both catch the same class_declaration).
    seen: set = set()
    deduped: list[tuple[str, str, int, int]] = []
    for item in results:
        dedup_key = (item[0], item[2], item[3])
        if dedup_key not in seen:
            seen.add(dedup_key)
            deduped.append(item)

    return deduped


def _find_all_symbols_via_query(
    content: str, language: str, query_str: str,
) -> list[tuple[str, str, int, int]]:
    """Extract top-level symbols using a declarative tree-sitter query."""
    matches = query_matches(content, language, query_str)
    results: list[tuple[str, str, int, int]] = []
    seen: set = set()

    for match_group in matches:
        name_caps = match_group.get("name", [])
        def_caps = match_group.get("def", [])
        if not name_caps or not def_caps:
            continue
        name = name_caps[0].text
        node_type = def_caps[0].node_type
        start = def_caps[0].start_line
        end = def_caps[0].end_line

        # Deduplicate (same name + same line range)
        dedup_key = (name, start, end)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        kind = _node_kind_from_type(node_type)
        results.append((name, kind, start, end))

    return results


def _node_kind_from_type(node_type: str) -> str:
    """Map a tree-sitter node type string to kind string (query path).

    Shares ``_BASE_KIND_MAP`` with ``_node_kind`` so the two paths cannot drift
    on common node types.  CSS-only types are absent here by design — CSS has no
    declarative query, so those node types never reach this path.
    """
    return _BASE_KIND_MAP.get(node_type, "function")


# ── CST utility functions (new in Phase 1) ──────────────────────────────────


def find_anchor_node(
    content: str, language: str, line: int,
) -> Optional[dict]:
    """Find the enclosing statement node at *line* (1-indexed).

    Uses tree-sitter to locate the syntax node covering *line*.
    Returns a dict with keys:
      - start_line, end_line (1-indexed)
      - start_byte, end_byte
      - text: exact source text of the node
      - node_type: tree-sitter node type (e.g. "if_statement", "expression_statement")
    Returns None if tree-sitter is unavailable or no suitable node found.

    ★ Resolution strategy: first finds the deepest named node at *line*, then
      walks UP the tree to the nearest enclosing statement node (if_statement,
      else_clause, expression_statement, etc.). This ensures that when multiple
      nodes occupy the same line (e.g. "} else if (…) {"), the enclosing
      statement is used as the anchor rather than a child body node.

    This is the anchor-resolution primitive for structural editing.
    Instead of regex-matching text lines, we resolve the anchor to a real AST
    node — eliminating hallucination (non-existent anchors), non-unique matches,
    and block-structure corruption by design.
    """
    if not content.strip():
        return None
    tree = parse_to_tree(content, language)
    if tree is None:
        return None
    root = tree.root_node
    zero_line = line - 1  # convert to 0-indexed

    def _descend(n):
        if n.start_point[0] <= zero_line <= n.end_point[0]:
            for child in n.children:
                result = _descend(child)
                if result is not None:
                    return result
            if n.is_named and n.start_point[0] <= zero_line <= n.end_point[0]:
                return n
        return None

    node = _descend(root)
    if node is None:
        return None

    # ── Walk UP to nearest enclosing statement node ──────────────────────
    # When multiple AST nodes share the same line (e.g. "} else if (cond) {"),
    # the deepest named node may be a punctuation-like block body. We need
    # the enclosing if_statement/else_clause/expression_statement instead.
    _ENCLOSING_TYPES = frozenset({
        "if_statement", "else_clause", "for_statement", "while_statement",
        "do_statement", "expression_statement", "switch_expression",
        "switch_case", "labeled_statement", "try_statement",
        "catch_clause", "finally_clause", "function_declaration",
        "method_definition", "class_declaration", "program", "module",
        "statement_block", "lexical_declaration", "variable_declaration",
        "return_statement", "throw_statement", "break_statement",
        "continue_statement", "debugger_statement",
    })
    _parent = node.parent
    while _parent is not None:
        if _parent.type in _ENCLOSING_TYPES:
            # Only promote if parent still covers the target line
            if _parent.start_point[0] <= zero_line <= _parent.end_point[0]:
                node = _parent
                break
        _parent = _parent.parent
    # (If no enclosing statement found, keep the original node)

    # Skip punctuation-only nodes (bare braces, semicolons)
    if node.type in ("{", "}", "(", ")", "[", "]", ";", ","):
        return None

    # ── Parent & sibling info for placement validation ──────────────────
    # Used by _handle_insert_after_line to detect "end-of-block" ambiguity:
    # when insert_after targets the LAST statement in a block, the inserted
    # code lands INSIDE the block, which may not be the intent (the planner
    # may have meant "after the entire block").
    _parent_node = node.parent
    _parent_type = _parent_node.type if _parent_node is not None else None
    _parent_end_line = None
    # Check if this node has a next_named_sibling — if it's the last named
    # child of its parent, insert_after means "insert at end of block body"
    # (inside the block), not "after the block closing brace".
    _has_next_named = False
    if _parent_node is not None:
        _next = node.next_named_sibling
        if _next is not None:
            _has_next_named = True
        _parent_end_line = _parent_node.end_point[0] + 1  # 1-indexed

    code_bytes = _encode_content(content)
    return {
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "text": code_bytes[node.start_byte:node.end_byte].decode("utf-8"),
        "node_type": node.type,
        "parent_type": _parent_type,
        "has_next_named_sibling": _has_next_named,
        "parent_end_line": _parent_end_line,
        "_ts_node": node,  # raw tree-sitter node for sibling walking etc.
    }


@lru_cache(maxsize=64)
def parse_to_tree(content: str, language: str):
    """Parse *content* and return the tree-sitter Tree object.

    Returns None if tree-sitter is unavailable or parsing fails.
    Callers receive the same Tree object that ``Parser.parse()`` returns;
    they can traverse ``tree.root_node`` and its children directly.

    Memoised: analysis scanners parse the same source several times per run
    (``__all__`` extraction, def collection, reference collection — and again
    per scanner in a pipeline).  Trees are read-only in this codebase, so
    sharing the object is safe.
    """
    parser = get_parser(language)
    if parser is None:
        return None
    try:
        return parser.parse(_encode_content(content))
    except Exception:
        return None


def structural_hash(node) -> str:
    """Compute a structural hash from node types and field names.

    Walks the CST tree and builds a SHA-256 digest from:
      - node types
      - field names of named children (via ``field_name_for_child``)
    Leaf text content (variable names, string literals), whitespace, and
    comments are excluded.  Two trees that differ in variable naming,
    literal values, or formatting produce the *same* hash, but different
    node structure or field assignments yield different hashes.

    This granularity catches structural regressions (missing body,
    swapped children, wrong node type) while tolerating the naming and
    formatting variation that gate comparisons need to survive.

    Returns a 16-character hex digest string.  Callers should compare
    for equality.

    The traversal is iterative (explicit stack) rather than recursive so
    that deeply nested / machine-generated inputs — which can exceed
    Python's default recursion limit (1000) — are handled safely.
    """
    import hashlib

    # Iterative post-order walk. Each frame holds:
    #   parts  — signature fragments accumulated so far for this node
    #            (starts as [node.type])
    #   children — pending list of (child, field_prefix) not yet folded in
    # When a frame's children are exhausted we join its parts into a
    # signature string. If a parent exists, that signature is appended to the
    # parent's parts (with the parent's field prefix); otherwise it is the
    # final result.
    root_children = deque(
        (ch, (node.field_name_for_child(i) or ""))
        for i, ch in enumerate(node.children)
        if ch.is_named
    )
    # Each frame: (parts, pending_children, field_prefix)
    #   field_prefix — this frame's own field name (prefixed onto its
    #                  signature when appended to the parent)
    stack: list[tuple[list[str], deque, str]] = [
        ([node.type], root_children, ""),
    ]
    final_sig = None

    while stack:
        parts, children, field_prefix = stack[-1]
        if children:
            child, child_field = children.popleft()
            child_children = deque(
                (ch, (child.field_name_for_child(j) or ""))
                for j, ch in enumerate(child.children)
                if ch.is_named
            )
            stack.append(([child.type], child_children, child_field))
            continue
        # All children consumed: finalize this node's signature.
        stack.pop()
        sig = "|".join(parts)
        if stack:
            parent_parts, _parent_children, _ = stack[-1]
            parent_parts.append(f"{field_prefix}:{sig}" if field_prefix else sig)
        else:
            final_sig = sig

    return hashlib.sha256(final_sig.encode("utf-8")).hexdigest()[:16]


def get_node_text(code_bytes: bytes, node) -> str:
    """Extract exact source text for *node* using byte-range slicing.

    Args:
        code_bytes: The full source encoded as UTF-8 bytes (the same
            bytes passed to ``parser.parse()``).
        node: A tree-sitter node with ``start_byte`` and ``end_byte``.

    Returns:
        The exact substring of *code_bytes* that corresponds to *node*,
        decoded to str.
    """
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8")


# ── Structural analysis helpers (replace numeric/regex guards) ──────────


def count_method_statements(
    code: str, method_name: str, language: str,
) -> Optional[int]:
    """Count AST statements in a method's body using tree-sitter.

    Walks the tree for ``method_definition`` or ``function_declaration``
    nodes whose name matches *method_name*, then counts the children of
    the ``statement_block`` that are actual statements (not braces).

    Returns the statement count, or None if tree-sitter is unavailable
    or the method is not found.

    This replaces numeric line-count heuristics (``_new_nonempty <= 2``
    or ``< 30%`` of old line count) with a structural AST measurement.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return None

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. First-match wins (mirrors
    # the original recursive _walk).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        # method_definition — class methods in TS/JS
        if node.type == "method_definition":
            _name_node = None
            _body_node = None
            for child in node.named_children:
                if child.type == "property_identifier":
                    _name_node = child
                elif child.type == "statement_block":
                    _body_node = child
            if (_name_node is not None and _body_node is not None
                    and _name_node.text.decode("utf-8") == method_name):
                stmts = [c for c in _body_node.children
                         if c.type not in ("{", "}")]
                return len(stmts)

        # function_declaration — top-level functions, also arrow-function
        # assigned to const/let/var
        elif node.type == "function_declaration":
            _name_node = node.child_by_field_name("name")
            _body_node = None
            for child in node.named_children:
                if child.type == "statement_block":
                    _body_node = child
            if (_name_node is not None and _body_node is not None
                    and _name_node.text.decode("utf-8") == method_name):
                stmts = [c for c in _body_node.children
                         if c.type not in ("{", "}")]
                return len(stmts)

        # arrow_function — const fn = () => { ... }
        elif node.type == "arrow_function":
            _body_node = None
            for child in node.named_children:
                if child.type == "statement_block":
                    _body_node = child
            if _body_node is not None:
                stmts = [c for c in _body_node.children
                         if c.type not in ("{", "}")]
                return len(stmts)

        stack.extend(reversed(node.named_children))

    return None


def get_class_member_names(
    code: str, class_name: str, language: str,
) -> Optional[tuple[set[str], set[str]]]:
    """Get (method_names, field_names) for a class using tree-sitter.

    Returns unique member name sets (duplicates collapsed).
    Returns None if tree-sitter is unavailable or the class is not found.

    Use this instead of count_class_members when you need to detect
    duplicate definitions within a class (Pattern 3).
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return None

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. First-match wins (mirrors
    # the original recursive _walk).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "class_declaration":
            _name_node = node.child_by_field_name("name")
            if (_name_node is not None
                    and _name_node.text.decode("utf-8") == class_name):
                _body_node = None
                for child in node.named_children:
                    if child.type == "class_body":
                        _body_node = child
                        break
                if _body_node is None:
                    # Class found but has no body — original returned None here
                    # (resumes search at siblings, does NOT descend). `continue`
                    # mirrors that by not pushing this node's children.
                    continue
                _method_names: set[str] = set()
                _field_names: set[str] = set()
                for c in _body_node.named_children:
                    if c.type == "method_definition":
                        _prop = c.child_by_field_name("name")
                        if _prop is not None:
                            _method_names.add(_prop.text.decode("utf-8"))
                    elif c.type in ("field_definition", "public_field_definition"):
                        _prop = c.child_by_field_name("name")
                        if _prop is not None:
                            _field_names.add(_prop.text.decode("utf-8"))
                return (_method_names, _field_names)

        stack.extend(reversed(node.named_children))

    return None


def count_unique_class_members(
    code: str, class_name: str, language: str,
) -> Optional[tuple[int, int]]:
    """Count unique (method_count, field_count) for a class using tree-sitter.

    Unlike count_class_members, this deduplicates members with the same name.
    If a class has two methods named 'lockPiece', the unique method count will
    be 1 (not 2), enabling duplicate detection (Pattern 3).

    Returns (unique_method_count, unique_field_count) or None if tree-sitter
    is unavailable or the class is not found.
    """
    _names = get_class_member_names(code, class_name, language)
    if _names is None:
        return None
    return (len(_names[0]), len(_names[1]))


def extract_this_references(code: str, language: str) -> list[str]:
    """Extract all `this.xxx` member access expressions from code.

    Walks the AST for `member_expression` nodes where the object is `this`,
    and returns the property names. This is used to detect hallucinated
    method/field calls (Pattern 4).

    Examples::

      this.board.forEach(...)  → ["board"]
      this.lockPiece()         → ["lockPiece"]
      this.tryRotate(current)  → ["tryRotate"]

    Returns an empty list if tree-sitter is unavailable or no references found.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return []

    _refs: list[str] = []

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. When a member_expression
    # matches (this.prop), we do NOT descend into it (mirrors the original).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        descend = True
        if node.type == "member_expression":
            _obj = node.child_by_field_name("object")
            _prop = node.child_by_field_name("property")
            if (_obj is not None and _prop is not None
                    and _obj.type == "this"):
                _refs.append(_prop.text.decode("utf-8"))
                descend = False  # Don't descend into matched expression
        if descend:
            stack.extend(reversed(node.named_children))

    return _refs


def symbol_exists_deep(code: str, name: str, language: str) -> bool:
    """Check if a symbol name exists ANYWHERE in the AST (deep traversal).

    Unlike ``find_all_symbols`` which skips descent into matched symbols
    (class bodies, etc.), this function walks ALL nodes in the tree,
    including class body children such as ``field_definition`` and
    ``method_definition``.

    This is the correct check for "does this symbol exist in the code?"
    and replaces regex-based fallback patterns that were prone to
    false positives with string contents and type annotations.

    Returns False if tree-sitter is unavailable or the name is not found.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return False

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. Returns True on the first
    # matching node (short-circuit, mirrors the original recursive _walk).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        # Check if this node defines a name
        _name_node = node.child_by_field_name("name")
        if _name_node is not None:
            _text = _name_node.text.decode("utf-8")
            if _text == name or _text == name.split(".")[-1]:
                return True

        # For method_definition, the name is a property_identifier child
        if node.type == "method_definition":
            for child in node.named_children:
                if child.type == "property_identifier":
                    if child.text.decode("utf-8") == name:
                        return True

        # For lexical_declaration (const/let/var), check variable name
        if node.type == "lexical_declaration":
            for child in node.named_children:
                if child.type == "variable_declarator":
                    _vn = child.child_by_field_name("name")
                    if _vn is not None and _vn.text.decode("utf-8") == name:
                        return True

        stack.extend(reversed(node.named_children))

    return False


def count_class_members(
    code: str, class_name: str, language: str,
) -> Optional[tuple[int, int]]:
    """Count (method_count, field_count) for a class using tree-sitter.

    Walks the tree for ``class_declaration`` with *class_name*, then
    counts ``method_definition`` children (= methods) and ``field_definition``
    children (= fields) directly from the AST.

    Returns ``(method_count, field_count)`` or None if tree-sitter is
    unavailable or the class is not found.

    This replaces character-count ratio (F3: ``< 30%``) and line-count
    threshold (F5: ``> 10`` and ``<= 3``) with exact AST member counting.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return None

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. First-match wins (mirrors
    # the original recursive _walk).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "class_declaration":
            _name_node = node.child_by_field_name("name")
            if (_name_node is not None
                    and _name_node.text.decode("utf-8") == class_name):
                # Found the class — count members from the class body
                _body_node = None
                for child in node.named_children:
                    if child.type == "class_body":
                        _body_node = child
                        break
                if _body_node is None:
                    # Class found but has no body — resumes search at siblings
                    # without descending (mirrors original `return None`).
                    continue
                _methods = sum(
                    1 for c in _body_node.named_children
                    if c.type == "method_definition"
                )
                _fields = sum(
                    1 for c in _body_node.named_children
                    if c.type in ("field_definition", "public_field_definition")
                )
                return (_methods, _fields)

        stack.extend(reversed(node.named_children))

    return None


def extract_class_methods(
    code: str, class_name: str, language: str,
) -> list[tuple[str, int, int]]:
    """Return ``[(method_name, start_line, end_line), ...]`` for a class.

    Handles multi-language tree-sitter node structures:
      Python:   class_definition → block → function_definition
      TS/JS:    class_declaration → class_body → method_definition
      Java:     class_declaration → class_body → method_declaration, constructor_declaration
      Kotlin:   class_declaration → class_body → function_declaration
      Go:       method_declaration (receiver-based, not inside struct body)

    Returns an empty list if tree-sitter is unavailable or the class is not found.
    Lines are 1-indexed.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return []

    results: list[tuple[str, int, int]] = []

    if language == "go":
        # Go methods are declared externally with a receiver:
        #   func (r *MyStruct) Method() { ... }
        # Iterative DFS (explicit stack) — avoids Python recursion-limit
        # blow-up on deeply nested / machine-generated inputs.
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "method_declaration":
                _receiver_node = node.child_by_field_name("receiver")
                if _receiver_node is not None:
                    _recv_text = _receiver_node.text.decode("utf-8")
                    # Check if receiver type matches class_name.
                    # Format: (r *MyStruct) or (r MyStruct) or (r pkg.MyStruct)
                    # Strip parens, take last space-delimited token, strip star
                    _recv_clean = _recv_text.strip("()").strip()
                    _parts = _recv_clean.split()
                    _recv_type = _parts[-1] if len(_parts) >= 2 else _recv_clean
                    _recv_type = _recv_type.replace("*", "").strip()
                    if class_name in (_recv_type, _recv_type.split(".")[-1]):
                        _name_node = node.child_by_field_name("name")
                        if _name_node is not None:
                            _name = _name_node.text.decode("utf-8")
                            results.append((
                                _name,
                                node.start_point.row + 1,
                                node.end_point.row + 1,
                            ))
            stack.extend(reversed(node.named_children))

        return results

    # For class_body-based languages: find the class, then scan its body.
    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. First-match wins (mirrors
    # the original recursive _find_class_body).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        matched = False
        if node.type in ("class_declaration", "class_definition"):
            _name_node = node.child_by_field_name("name")
            if _name_node is not None:
                _cname = _name_node.text.decode("utf-8")
                if _cname == class_name or _cname.split(".")[-1] == class_name:
                    matched = True
                else:
                    # Check simple_identifier for Kotlin class names
                    for _ch in node.named_children:
                        if (_ch.type == "simple_identifier"
                                and _ch.text.decode("utf-8") == class_name):
                            matched = True
                            break
            else:
                # No name field — original fell through to "Found the class".
                matched = True

        if matched:
            # Found the class — find its body node
            _body_node = None
            for child in node.named_children:
                if child.type in ("class_body", "block", "body"):
                    _body_node = child
                    break
            if _body_node is None:
                # Class matched but has no body — original returned None here
                # (resumes search at siblings without descending).
                continue
            # Scan body for method-like definitions
            for item in _body_node.named_children:
                _item_type = item.type
                if _item_type in (
                    "function_definition",          # Python
                    "async_function_definition",     # Python async
                    "method_definition",             # TS/JS
                    "method_declaration",             # Java
                    "constructor_declaration",        # Java
                    "function_declaration",            # Kotlin
                ):
                    _method_name = None
                    # Try standard "name" field
                    _mn = item.child_by_field_name("name")
                    if _mn is not None:
                        _method_name = _mn.text.decode("utf-8")
                    else:
                        # Fallback: property_identifier (TS/JS method_definition)
                        for _ch in item.named_children:
                            if _ch.type == "property_identifier":
                                _method_name = _ch.text.decode("utf-8")
                                break
                    if _method_name is not None:
                        results.append((
                            _method_name,
                            item.start_point.row + 1,
                            item.end_point.row + 1,
                        ))
            return results

        stack.extend(reversed(node.named_children))

    return results


def extract_symbol_body(
    code: str, symbol_name: str, language: str,
) -> Optional[tuple[int, int]]:
    """Return ``(body_start_line, body_end_line)`` for a function/method's body.

    The body is the indented block (Python) or brace-delimited block
    (C-family languages) — the executable statements without the signature.

    Returns None if tree-sitter is unavailable or the symbol is not found.
    Lines are 1-indexed.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return None

    # Iterative DFS (explicit stack) — avoids Python recursion-limit blow-up
    # on deeply nested / machine-generated inputs. Mirrors the original
    # recursive _walk: scan in pre-order; on the first matching definition
    # node with a body, return its line range. A matching definition without
    # a body falls through to descend into its children (the original
    # `return None` after a name match only short-circuited THAT node's own
    # child loop, not the whole search).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        _is_def = node.type in (
            "function_definition",          # Python
            "async_function_definition",    # Python async
            "function_declaration",         # Go, Kotlin
            "method_declaration",           # Go, Java
            "method_definition",            # TS/JS
            "constructor_declaration",      # Java
        )
        found_body = False
        if _is_def:
            _name_node = node.child_by_field_name("name")
            # TS/JS method_definition: name is property_identifier
            if _name_node is None and node.type == "method_definition":
                for _ch in node.named_children:
                    if _ch.type == "property_identifier":
                        _name_node = _ch
                        break
            if _name_node is not None:
                _text = _name_node.text.decode("utf-8")
                if _text == symbol_name or _text.rsplit(".", 1)[-1] == symbol_name:
                    # Find body child node
                    for child in node.named_children:
                        if child.type in ("block", "statement_block", "body"):
                            return (child.start_point.row + 1, child.end_point.row + 1)
                    # Name matched but no body found — do NOT descend further
                    # into this definition (mirrors the original early return).
                    found_body = True
        if not found_body:
            stack.extend(reversed(node.named_children))

    return None


# ── Close-brace depth analysis (structural replacement for text-level lstrip) ──

# Block-statement types that contribute to brace depth when they end at a line.
# These are the types whose closing `}` adds one level of structure to close.
_BLOCK_DEPTH_TYPES = frozenset({
    "if_statement", "else_clause", "for_statement", "while_statement",
    "do_statement", "try_statement", "catch_clause", "finally_clause",
    "function_declaration", "method_definition", "class_declaration",
    "switch_expression", "switch_case", "labeled_statement",
    "statement_block", "lexical_declaration",
})


def find_collection_literal_closing_brace(
    content: str, language: str, symbol_name: str,
) -> Optional[int]:
    """Find the closing brace line of a collection literal assigned to *symbol_name*.

    For map/array/struct literals like:
        var handlers = map[string]Handler{
            "add": cmdAdd,
            "list": cmdList,
        }
    This returns the 1-indexed line of the closing ``}`` (line 5 in this example).

    Returns None if tree-sitter is unavailable, symbol not found, or the symbol
    is not a collection literal.
    """
    if not _HAS_TREE_SITTER or not content.strip():
        return None
    tree = parse_to_tree(content, language)
    if tree is None:
        return None

    # Composite literal node types by language
    _COMPOSITE_LITERAL_TYPES = frozenset({
        "composite_literal",   # Go: map[string]X{...}, []X{...}, X{...}
        "object",              # JS/TS: {...}
        "array",               # JS/TS: [...]
        "object_literal_expression",  # Kotlin
        "array_literal_expression",   # Kotlin
    })

    # Walk the tree to find the symbol's var_declaration (or equivalent)
    # that contains a composite literal, then find the literal's closing brace.
    root = tree.root_node

    def _find_symbol_decl(node) -> Optional[int]:
        """Find the declaration node for *symbol_name*, search its subtree
        for a composite literal, and return the closing brace line (1-indexed).
        Returns None if symbol not found or not a collection literal."""
        if node.type in _SYMBOL_NODE_TYPES:
            name = _extract_name(node)
            if name == symbol_name:
                # Recursively search the declaration's subtree for composite literal
                def _find_literal_in_subtree(n) -> Optional[int]:
                    if n.type in _COMPOSITE_LITERAL_TYPES:
                        return n.end_point[0] + 1
                    for child in n.children:
                        result = _find_literal_in_subtree(child)
                        if result is not None:
                            return result
                    return None
                return _find_literal_in_subtree(node)
            # Container types: descend into children to find nested symbols
            if node.type not in _CONTAINER_NODE_TYPES:
                return None
        for child in node.children:
            result = _find_symbol_decl(child)
            if result is not None:
                return result
        return None

    return _find_symbol_decl(root)


def compute_close_brace_depth(
    content: str, language: str, line: int,
) -> Optional[int]:
    """Compute how many block-statement levels end at *line* (1-indexed).

    For a closing-brace line like ``}`` (pure brace), this returns the
    number of enclosing block-statement ancestors whose span ends at this
    same line.  This represents the structural "depth of closure" at the
    anchor point.

    Example — for ``}`` closing an ``if_statement`` inside a function::

        function foo() {       // function_declaration start
            if (cond) {        // if_statement start
                // body        // (statement_block body)
            }                   // ← both statement_block AND if_statement end here
        }                       //   → close_brace_depth = 2 at this line
                                //   → close_brace_depth = 1 (function_declaration only)

    This is used by the close-brace merge (insert_after_line) to validate
    that the snippet's leading ``}}`` count is structurally consistent with
    the anchor's nesting depth.

    Returns an int (≥ 0), or None if tree-sitter is unavailable.
    """
    if not _HAS_TREE_SITTER or not content.strip():
        return None
    tree = parse_to_tree(content, language)
    if tree is None:
        return None
    root = tree.root_node
    zero_line = line - 1

    # Step 1: find the deepest named node covering *line*
    def _deepest(n):
        if n.start_point[0] <= zero_line <= n.end_point[0]:
            for child in n.children:
                result = _deepest(child)
                if result is not None:
                    return result
            if n.is_named and n.start_point[0] <= zero_line <= n.end_point[0]:
                return n
        return None

    node = _deepest(root)
    if node is None:
        return 0

    # Step 2: walk up the ancestry; count block types whose span ends at *line*
    depth = 0
    current = node
    while current is not None:
        if current.type in _BLOCK_DEPTH_TYPES and current.end_point[0] == zero_line:
            depth += 1
        current = current.parent

    return depth
