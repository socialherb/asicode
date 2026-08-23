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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any  # f821-protected

from external_llm.languages.models import _EXT_MAP, LanguageId

logger = logging.getLogger(__name__)

# ── Optional import ──────────────────────────────────────────────────────────

_HAS_TREE_SITTER = False
_ts = None  # type: ignore[assignment]  # real module bound on successful import
try:
    import tree_sitter as _ts  # type: ignore[assignment]

    _HAS_TREE_SITTER = True
except ImportError:
    # tree-sitter is optional — every public entry point checks
    # _HAS_TREE_SITTER and falls back to regex heuristics. `_ts` stays
    # None (bound above) so the module remains importable without it.
    _ts = None  # type: ignore[assignment]  # explicit fallback binding

# Language module cache: language name → tree_sitter.Language object (thread-safe)
_LANG_CACHE: dict[str, object] = {}
_LANG_CACHE_LOCK = threading.RLock()

# One lock per language, so a cold resolve serialises only the threads asking
# for THAT language. _LANG_CACHE_LOCK guards this dict itself and nothing slow;
# see _get_language for why the resolve must not run under it.
#
# Growth matches _LANG_CACHE exactly — one entry per distinct language string
# ever requested, both unbounded by the same argument (a failed resolve is
# negative-cached, so a given string is only ever resolved once). Entries are
# not reclaimed after resolution on purpose: dropping one lets a thread that
# arrives before the cache store build a second lock and duplicate a resolve
# that can cost half a second.
_LANG_RESOLVE_LOCKS: dict[str, threading.Lock] = {}

# Parser cache: per-thread. tree-sitter's TSParser is stateful and NOT thread-safe
# ("not safe to call ts_parser_parse from multiple threads at once" — api.h).
# Languages are immutable/thread-safe, so they stay in the shared _LANG_CACHE;
# only the Parser (which holds mutable parse state) is isolated per thread.
# Thread-pool workers are reused, so each worker pays the Parser-construction
# cost once and then reuses its own instance with no locking on the parse path.
_PARSER_TLS = threading.local()

# Generation counter for invalidating the PER-THREAD parser caches.
#
# invalidate_caches() can only reach the thread-local dict of the thread that
# calls it, so clearing directly would leave every OTHER thread holding a Parser
# still bound to the previous Language — defeating the documented purpose of
# invalidate_caches() ("newly installed grammars take effect without a process
# restart"). Instead the counter is bumped there and each thread compares it
# lazily on its next get_parser() call, rebuilding when it has fallen behind.
_PARSER_GENERATION = 0

# Sentinel distinguishing "no entry yet" from a cached None (negative cache
# from a failed language-binding load). A module-level object is identity-safe
# across threads. Used by the parser TLS cache AND the compiled-query cache
# (_compile_query) — the single sentinel serves both.
_MISS = object()

# Both memo caches below key on the FULL source text, so an entry cap alone
# bounds the entry COUNT, not the bytes held — the two caches are filled by
# whatever the agent last looked at, and the biggest sources are both the most
# expensive to retain and the least likely to repeat.
#
# Measured on this repo (scan the 200 largest .py files, hold no references,
# then diff RSS across cache_clear): the old 64/128-entry settings retained
# 89 MB, these retain 17 MB. A parsed Tree costs ~19.5x its source, which is
# where the bulk goes. Meanwhile a realistic find_symbol workload (10 lookups)
# got hits=0/misses=36 — find_symbol has its own per-file mtime cache, so
# nothing repeats at this layer. The documented win is intra-pipeline reuse
# (one source parsed by several helpers in a row), which needs a handful of
# slots, not 64.
#
# So: skip the cache above _MAX_CACHED_SOURCE_CHARS and keep few slots. The
# size distribution here is median 6.5 KB / p90 42 KB, so a 64 KB gate still
# caches ~95% of files while capping one parse entry near 1.3 MB. Larger
# sources are re-parsed on repeat — correct, just not memoised.
_MAX_CACHED_SOURCE_CHARS = 64 * 1024


# Memoised UTF-8 encoding: the same content is encoded by query_captures,
# query_matches, and extract_import_names.  Caching avoids
# re-encoding the same file multiple times per scan pipeline.
@lru_cache(maxsize=32)
def _encode_content_cached(content: str) -> bytes:
    return content.encode("utf-8")


def _encode_content(content: str) -> bytes:
    """UTF-8 encode *content*, memoised for sources small enough to be worth it."""
    if len(content) > _MAX_CACHED_SOURCE_CHARS:
        return content.encode("utf-8")
    return _encode_content_cached(content)


# Leading import-keyword prefix stripped from an @source capture that spans
# the whole import declaration.  Scala's grammar inlines the dotted path into
# separate identifier nodes, so we capture the declaration node and drop the
# keyword here.  No-op for languages whose @source capture is already the bare
# module path (identifier / qualified_name / string literal).  Scala only uses
# `import` (never `using`) for imports, so that is the only keyword handled.
_IMPORT_KW_RE = re.compile(r"^import\s+")


# The grammar key → tree-sitter module import path map (_LANG_MODULE_MAP) is
# DERIVED below, after the per-language query maps — its domain is defined by
# them (see the "grammar key → tree-sitter module map (DERIVED)" section).
# extension → grammar-key map (_EXT_TO_GRAMMAR_KEY) and the
# grammar_key_for_path / grammar_key_for_ext helpers: DERIVED from _EXT_MAP
# after the per-language query maps below (the query maps define which
# languages have full AST support, so the map must follow them).


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
        logger.debug("tree_sitter_language_pack unavailable; grammar lookups will fail", exc_info=True)
        _LANG_PACK_GET_LANGUAGE = None
    return _LANG_PACK_GET_LANGUAGE


# Our language ids that differ from language-pack naming conventions.
_LANG_PACK_ALIASES = {"c_sharp": "csharp"}


@dataclass
class QueryCapture:
    """A single capture from a tree-sitter query."""

    capture_name: str  # e.g., "sym", "def", "call"
    node_type: str  # e.g., "function_definition", "identifier"
    text: str  # source text of the captured node
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
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
    # C# — the active tree-sitter-csharp grammar emits method_declaration
    # (shared with Go, above) for class methods and local_function_statement
    # for top-level methods / local functions (both carry a "name" field,
    # probed). Without local_function_statement C# top-level methods were
    # never extracted, so Allman/K&R method headers were invisible to
    # _find_block_end_line.
    "local_function_statement",
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
_CONTAINER_NODE_TYPES = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "export_statement",
    }
)

# Per-language tree-sitter queries for extracting top-level symbol definitions.
# Each query captures:
#   @def — the definition node (for line range)
#   @name — the name identifier (for the symbol name string)
#   @kind — optional: a node whose type encodes the symbol kind
_SYMBOL_QUERIES: dict[str, str] = {
    # The two module-child alternatives are the same statement under two
    # grammars: standalone tree-sitter-python wraps a statement-position
    # assignment in `expression_statement`, the language-pack bundle emits
    # `assignment` directly under `module`. Only the pack is a declared
    # dependency, so the wrapper-only pattern found no module-level symbols at
    # all on a real install.
    "python": """
(function_definition name: (identifier) @name) @def
(class_definition name: (identifier) @name) @def
(module [
  (expression_statement (assignment left: (identifier) @name))
  (assignment left: (identifier) @name)
 ] @def)
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


# ── extension → grammar key (DERIVED) ───────────────────────────────────────
# _EXT_TO_GRAMMAR_KEY used to be a hand-written parallel of _EXT_MAP
# (languages/models.py — the package's canonical extension → language map).
# It is now DERIVED so the ext → language → grammar-key chain cannot drift:
#   * domain — extensions whose language has full AST query support, defined
#     by the four query maps above (a grammar key present in all four is
#     full-AST; parse-only JSON/CSS/HTML get no entry);
#   * key — the LanguageId value, except the single documented override
#     (.tsx parses with the separate "tsx" grammar of the same package).
# Import-time fail-fast (see _derive_ext_to_grammar_key): an override key
# with no query maps, or a derived key missing from _LANG_MODULE_MAP, is a
# hard error — the latter would otherwise resolve through the language-pack
# download fallback AND read as unavailable to is_language_available(), so
# the gate's fail-closed probe would fail on a resolvable grammar.
_GRAMMAR_KEY_OVERRIDES: dict[str, str] = {".tsx": "tsx"}

_FULL_AST_GRAMMAR_KEYS: frozenset[str] = (
    frozenset(_SYMBOL_QUERIES) & frozenset(_CALL_QUERIES) & frozenset(_IMPORT_QUERIES) & frozenset(_REFERENCE_QUERIES)
)


# ── grammar key → tree-sitter module map (DERIVED) ──────────────────────────
# _LANG_MODULE_MAP used to be a hand-written 19-entry literal.  It is now
# DERIVED so its key set cannot drift from the query maps:
#   * domain — every full-AST grammar key (the four query maps' intersection,
#     _FULL_AST_GRAMMAR_KEYS) plus the parse-only keys editing/symbol tooling
#     still parses with tree-sitter (_PARSE_ONLY_GRAMMAR_KEYS: html/css walk
#     the syntax tree without queries).  JSON is deliberately NOT included —
#     nothing parses .json with tree-sitter (see is_language_available).
#   * value — the standard tree_sitter_<key> import convention, except the
#     single documented override (tsx's grammar lives in the typescript
#     package, which exports language_typescript() AND language_tsx()).
# Import-time fail-fast: a domain key that is neither a LanguageId value nor
# the tsx alias is a hard error — a typo'd query-map key or parse-only entry
# would otherwise build a phantom tree_sitter_<typo> module path that only
# fails at first parse.
_PARSE_ONLY_GRAMMAR_KEYS: frozenset[str] = frozenset({"html", "css"})

_MODULE_NAME_OVERRIDES: dict[str, str] = {"tsx": "typescript"}


def _derive_lang_module_map() -> dict[str, str]:
    """Derive the grammar key → tree-sitter module import path map.

    Domain: _FULL_AST_GRAMMAR_KEYS | _PARSE_ONLY_GRAMMAR_KEYS; values follow
    the tree_sitter_<key> convention except _MODULE_NAME_OVERRIDES.  Raises
    ValueError (import time) on a domain key that is neither a LanguageId
    value nor the tsx alias.
    """
    language_values = {member.value for member in LanguageId}
    out: dict[str, str] = {}
    for key in sorted(_FULL_AST_GRAMMAR_KEYS | _PARSE_ONLY_GRAMMAR_KEYS):
        if key != "tsx" and key not in language_values:
            raise ValueError(
                f"grammar key {key!r} is not a LanguageId value — a typo'd "
                "query-map key or _PARSE_ONLY_GRAMMAR_KEYS entry "
                "(LanguageId, external_llm/languages/models.py)"
            )
        out[key] = f"tree_sitter_{_MODULE_NAME_OVERRIDES.get(key, key)}"
    return out


_LANG_MODULE_MAP: dict[str, str] = _derive_lang_module_map()


def _derive_ext_to_grammar_key() -> dict[str, str]:
    """Derive the extension → grammar-key map from ``_EXT_MAP``.

    Every extension whose language has full AST query support (present in all
    four query maps) maps to that language's value, except the ``.tsx``
    override.  Raises ValueError (import time) on an override key without
    query maps or a derived key missing from ``_LANG_MODULE_MAP``.
    """
    out: dict[str, str] = {}
    for ext, name in _EXT_MAP.items():
        try:
            value = LanguageId[name].value
        except KeyError:
            raise ValueError(
                f"_EXT_MAP[{ext!r}] = {name!r} is not a LanguageId member (external_llm/languages/models.py)"
            ) from None
        if value not in _FULL_AST_GRAMMAR_KEYS:
            continue  # parse-only language (JSON/CSS/HTML): no grammar-key entry
        key = _GRAMMAR_KEY_OVERRIDES.get(ext, value)
        if key not in _FULL_AST_GRAMMAR_KEYS:
            raise ValueError(
                f"grammar-key override {ext!r} -> {key!r} has no query maps "
                "(tree_sitter_utils _SYMBOL/_CALL/_IMPORT/_REFERENCE_QUERIES)"
            )
        if key not in _LANG_MODULE_MAP:
            raise ValueError(
                f"grammar key {key!r} (for {ext!r}) has no entry in "
                "_LANG_MODULE_MAP — is_language_available() would report it "
                "missing and the gate would fail closed on a resolvable grammar"
            )
        out[ext] = key
    return out


_EXT_TO_GRAMMAR_KEY: dict[str, str] = _derive_ext_to_grammar_key()


def grammar_key_for_path(file_path: str) -> str | None:
    """Return tree-sitter grammar key for *file_path*, or ``None`` for unknown extensions.

    Single canonical mapping for file extension → tree-sitter grammar key
    (derived from ``_EXT_MAP``; see above).
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


def get_available_languages() -> set[str]:
    """Return set of language names whose tree-sitter bindings are installed.

    Resolving the whole set imports EVERY mapped grammar module (~50 ms here,
    where 18 of the 19 standalone packages happen to be installed). Callers that
    only need to know about one language should use :func:`is_language_available`
    instead — it is exactly equivalent per-language and resolves just that one.
    """
    available = set()
    for lang in _LANG_MODULE_MAP:
        if _get_language(lang) is not None:
            available.add(lang)
    return available


def is_language_available(language: str) -> bool:
    """True iff *language* would appear in :func:`get_available_languages`.

    Exactly equivalent to ``language in get_available_languages()`` — the set
    is built by filtering ``_LANG_MODULE_MAP`` through ``_get_language``, so
    membership is ``mapped AND resolvable`` and this reproduces both halves.
    It just resolves the ONE language asked about instead of all 19.

    The ``_LANG_MODULE_MAP`` check must come first, and not only for speed:
    ``_get_language`` falls back to ``tree_sitter_language_pack`` for anything
    it cannot import, and that pack resolves ~306 languages by DOWNLOADING the
    grammar on first use (measured: a cold ``_get_language("json")`` took
    644 ms, and json is not in the map). Probing an unmapped language would
    therefore turn a symbol lookup into a network fetch AND report available
    for a language the set-based callers treat as missing — a behaviour change,
    not just a slow one. Gating on the map keeps this a pure optimisation.
    """
    if language not in _LANG_MODULE_MAP:
        return False
    return _get_language(language) is not None


def is_available() -> bool:
    """Return True if tree-sitter core library is installed."""
    return _HAS_TREE_SITTER


def _resolve_language_uncached(language: str) -> object | None:
    """Import and build the Language object for *language*. Never touches a cache.

    Split out of :func:`_get_language` so the slow part runs with NO lock held.

    The cost that motivates the split is the FIRST-EVER resolve of a language on
    a given machine, which the language-pack fallback pays to materialise the
    prebuilt grammar before caching it on disk. Measured here, first ever:
    ocaml 660 ms, nim 587 ms, crystal 583 ms, erlang 550 ms. Every later
    process loads the same grammars in well under 1 ms, so this is a first-run
    effect and not a steady-state one — do not quote these numbers as a
    recurring per-process cost.

    It still matters, for two reasons. pyproject declares only
    ``tree-sitter-language-pack`` (no standalone ``tree_sitter_<lang>``
    packages), so on a clean user install EVERY language resolves through the
    except-branch below and pays this once — a dev machine with standalone
    grammar packages installed takes the fast branch and hides it entirely.
    And a mixed-language repo indexed in parallel hits several of these at
    once, which under a global lock serialised them AND every unrelated cache
    hit behind them.
    """
    # All languages are imported in the order registered in _LANG_MODULE_MAP.
    # Unregistered languages also fall back via standard naming convention (tree_sitter_<lang>).
    # Also handles non-standard modules using language_<lang>() naming like PHP.
    assert _ts is not None  # caller (_get_language) already returned None on !_HAS_TREE_SITTER
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
                logger.debug("tree-sitter language-pack %s not available: %s", language, e2)
            else:
                if not isinstance(raw, _ts.Language):
                    raw = _ts.Language(raw)
                return raw
        logger.debug("tree-sitter language %s not available: %s", language, e)
        return None
    else:
        return raw


def _get_language(language: str) -> object | None:
    """Get a tree-sitter Language object for *language*, or None (thread-safe).

    The resolution runs OUTSIDE ``_LANG_CACHE_LOCK``. Holding that global lock
    across the import made one thread's first-ever resolve of a language block
    every other thread's *cache hit* for an unrelated language: measured, a
    first-ever ``haskell`` resolve blocked an already-cached ``python`` lookup
    on another thread for 587 ms. That is the hot parse path — ``get_parser``
    is called from thread-pool workers, and a mixed-language repo resolves
    several grammars concurrently. See :func:`_resolve_language_uncached` for
    why the expensive case is first-run rather than per-process.

    Concurrency is handled in two layers instead:

      * a per-language resolve lock, so N threads asking for the same cold
        language do the work once rather than N times — while a thread asking
        for a DIFFERENT language is never blocked by it;
      * a generation re-check before storing. ``invalidate_caches()`` clears
        ``_LANG_CACHE`` (after a late pip-install of a grammar) and bumps
        ``_PARSER_GENERATION`` under the same lock. A clear landing while a
        resolve is in flight would otherwise be undone by the store that
        follows, resurrecting the very Language the invalidation discarded —
        the "read → slow collect → store" hazard this repo's other per-root
        caches guard the same way.
    """
    if not _HAS_TREE_SITTER:
        return None

    with _LANG_CACHE_LOCK:
        if language in _LANG_CACHE:
            return _LANG_CACHE[language]
        resolve_lock = _LANG_RESOLVE_LOCKS.get(language)
        if resolve_lock is None:
            resolve_lock = _LANG_RESOLVE_LOCKS[language] = threading.Lock()
        generation = _PARSER_GENERATION

    with resolve_lock:
        # Another thread may have finished this exact language while we queued.
        with _LANG_CACHE_LOCK:
            if language in _LANG_CACHE:
                return _LANG_CACHE[language]

        result = _resolve_language_uncached(language)

        with _LANG_CACHE_LOCK:
            if generation != _PARSER_GENERATION:
                # Invalidated mid-resolve: hand this caller its (correct, freshly
                # built) object but do not re-populate the cache the invalidation
                # just emptied.
                return result
            _LANG_CACHE[language] = result  # type: ignore[assignment]
            return result


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
    assert _ts is not None  # _HAS_TREE_SITTER implies the import succeeded

    lang_obj = _get_language(language)
    if lang_obj is None:
        return None

    cache = getattr(_PARSER_TLS, "cache", None)
    # Drop this thread's parsers when a cache invalidation happened since they
    # were built — they still reference the previous Language object.
    if cache is not None and getattr(_PARSER_TLS, "generation", None) != _PARSER_GENERATION:
        cache = None
    if cache is None:
        cache = {}
        _PARSER_TLS.cache = cache
        _PARSER_TLS.generation = _PARSER_GENERATION

    cached = cache.get(language, _MISS)
    if cached is not _MISS:
        return cached

    try:
        # lang_obj is a tree_sitter.Language, but _get_language types it as
        # object (it can also be a PyCapsule pre-wrap); Parser accepts Language.
        parser = _ts.Parser(lang_obj)  # type: ignore[arg-type]
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
    """Atomically clear all tree-sitter caches (language, parse, query, parsers).

    Called by the dependency checker after a late pip-install so that newly
    installed grammars take effect without a process restart.
    The caches are cleared under the ``_LANG_CACHE_LOCK`` (an RLock) to
    prevent interleaved reads from seeing a partially-invalidated state.
    ``_QUERY_CACHE_LOCK`` is also acquired in the correct lock order
    (``_LANG_CACHE_LOCK`` → ``_QUERY_CACHE_LOCK``, matching ``_get_language``)
    to avoid deadlock.

    The per-thread parser caches are invalidated via ``_PARSER_GENERATION``
    rather than cleared: this function only sees its OWN thread's
    ``_PARSER_TLS``, so a direct clear left every other thread holding a Parser
    bound to the discarded Language — the newly installed grammar then never took
    effect on those threads, which is exactly what this function exists to fix.
    """
    global _PARSER_GENERATION
    with _LANG_CACHE_LOCK:
        _LANG_CACHE.clear()
        # cache_clear/cache_info are attached to the public names below the lru_cache
        # wrappers (see the assignments after parse_to_tree); pyright sees only the
        # plain function type.
        parse_to_tree.cache_clear()  # type: ignore[attr-defined]
        _PARSER_GENERATION += 1
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
    assert _ts is not None  # _HAS_TREE_SITTER implies the import succeeded

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


def _prepare_query(
    content: str,
    language: str,
    query_string: str,
    tree=None,
) -> tuple | None:
    """Shared scaffolding for ``query_captures`` and ``query_matches``.

    Resolves the language, parses the source, compiles the query, and encodes
    the content as UTF-8 bytes.  ``parse_to_tree`` is memoised (@lru_cache):
    callers that run several queries against the same source (imports,
    exports, symbols, calls) share a single parse instead of re-parsing.
    *tree*, when given, must be ``parse_to_tree(content, language)`` — the
    caller parsed once and wants every query to share that tree (P5,
    2026-08-11: sources above ``_MAX_CACHED_SOURCE_CHARS`` bypass the memo, so
    a multi-query extraction used to parse them once per query).

    Returns ``(tree, query, code_bytes)`` or ``None`` if tree-sitter is
    unavailable, the language grammar is missing, parsing fails, or the query
    is invalid.
    """
    lang_obj = _get_language(language)
    if lang_obj is None:
        return None

    if tree is None:
        tree = parse_to_tree(content, language)
    if tree is None:
        return None

    query = _compile_query(language, lang_obj, query_string)
    if query is None:
        return None

    return tree, query, _encode_content(content)


def _node_text(node) -> str | None:
    """UTF-8 source text of a tree-sitter node, or None if it carries none.

    tree-sitter's ``Node.text`` is ``bytes | None`` (anonymous nodes and some
    error nodes have no source span).  All structural helpers decode via this
    so the Optional is handled in exactly one place.
    """
    raw = node.text
    if raw is None:
        return None
    return raw.decode("utf-8")


def _make_capture(
    capture_name: str,
    node,
    code_bytes: bytes,
) -> QueryCapture:
    """Build a ``QueryCapture`` from a tree-sitter node + encoded source."""
    return QueryCapture(
        capture_name=capture_name,
        node_type=node.type,
        text=code_bytes[node.start_byte : node.end_byte].decode("utf-8"),
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
    )


def query_captures(
    content: str,
    language: str,
    query_string: str,
    tree=None,
) -> list[QueryCapture]:
    """Run a tree-sitter query and return all captured nodes.

    Each capture includes metadata (capture name, node type, source text,
    1-indexed line range, byte offsets).

    *tree* optionally supplies a pre-parsed tree for *content* (see
    ``_prepare_query``) so several queries share one parse.

    Returns an empty list if tree-sitter is unavailable, the language grammar
    is missing, parsing fails, or the query is invalid.
    """
    prepared = _prepare_query(content, language, query_string, tree=tree)
    if prepared is None:
        return []
    tree, query, code_bytes = prepared

    assert _ts is not None  # _prepare_query returned None when tree-sitter is missing
    try:
        cursor = _ts.QueryCursor(query)
        captures_raw = cursor.captures(tree.root_node)
    except Exception:
        return []

    results: list[QueryCapture] = []
    for capture_name, nodes in captures_raw.items():
        results.extend(_make_capture(capture_name, node, code_bytes) for node in nodes)

    results.sort(key=lambda c: c.start_byte)
    return results


def query_matches(
    content: str,
    language: str,
    query_string: str,
    tree=None,
) -> list[dict[str, list[QueryCapture]]]:
    """Run a tree-sitter query and return matches grouped by pattern.

    Each element in the returned list corresponds to one pattern match.
    The dict maps capture name to a list of ``QueryCapture`` objects for
    that capture within the match.  This is useful when captures within
    the same pattern need to be associated (e.g., ``@def`` and ``@name``
    from the same function definition).

    *tree* optionally supplies a pre-parsed tree for *content* (see
    ``_prepare_query``) so several queries share one parse.
    """
    prepared = _prepare_query(content, language, query_string, tree=tree)
    if prepared is None:
        return []
    tree, query, code_bytes = prepared

    assert _ts is not None  # _prepare_query returned None when tree-sitter is missing
    try:
        cursor = _ts.QueryCursor(query)
        matches_raw = cursor.matches(tree.root_node)
    except Exception:
        return []

    results: list[dict[str, list[QueryCapture]]] = []

    for _pattern_idx, captures_dict in matches_raw:
        match_result: dict[str, list[QueryCapture]] = {}
        for capture_name, nodes in captures_dict.items():
            match_result[capture_name] = [_make_capture(capture_name, node, code_bytes) for node in nodes]
        results.append(match_result)

    return results


def has_error(content: str, language: str) -> bool | None:
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
    # has_error is definitionally equivalent to "any descendant is ERROR or
    # MISSING" — verified empirically across C/Java/Go error variants (see
    # test_tree_sitter_utils.py for exhaustive coverage).  The remaining DFS
    # would always yield the same True, so we short-circuit to the return.
    return bool(tree.root_node.has_error)


@dataclass(frozen=True)
class SyntaxErrorNode:
    """Structured syntax error from tree-sitter ERROR/MISSING node.

    Used by failure classifier Layer A for structural syntax error detection.
    """

    kind: str  # "ERROR" | "MISSING"
    missing_token: str  # MISSING node's expected token (e.g. ";", ")")
    line: int  # 0-based line number
    column: int  # 0-based column number


def find_error_nodes(content: str, language: str) -> list[SyntaxErrorNode] | None:
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
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            errors.append(
                SyntaxErrorNode(
                    kind="MISSING" if node.is_missing else "ERROR",
                    missing_token=node.type if node.is_missing else "",
                    line=node.start_point[0],  # 0-based
                    column=node.start_point[1],
                )
            )
        stack.extend(node.children)

    return errors


def extract_symbol_at_position(
    content: str,
    language: str,
    line: int,
    column: int,
) -> str | None:
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


def _extract_query_pairs(
    content: str,
    language: str,
    query_map: dict[str, str],
    capture_name: str,
    clean: Callable[[str], str] | None = None,
    tree=None,
) -> list[tuple[str, int]]:
    """Run a query and collect ``(value, line)`` pairs for *capture_name*.

    Shared scaffolding for ``extract_calls`` and ``extract_imports``.
    *query_map* selects the per-language query string; *capture_name* filters
    which captures to keep; *clean* optionally transforms each captured text
    (e.g. stripping quotes / selectors from module paths); *tree* optionally
    supplies a pre-parsed tree for *content* so several queries share one
    parse (P5, 2026-08-11).  Results are de-duplicated by ``(value, line)``
    and lines are 1-indexed.
    """
    query_str = query_map.get(language)
    if query_str is None:
        return []

    caps = query_captures(content, language, query_str, tree=tree)
    results: list[tuple[str, int]] = []
    seen: set = set()
    for c in caps:
        if c.capture_name != capture_name:
            continue
        value = clean(c.text) if clean is not None else c.text
        key = (value, c.start_line)
        if key in seen:
            continue
        seen.add(key)
        results.append((value, c.start_line))
    return results


def extract_calls(
    content: str,
    language: str,
    tree=None,
) -> list[tuple[str, int]]:
    """Extract call sites: ``[(callee_name, line), ...]``.

    Lines are 1-indexed.  Returns an empty list if tree-sitter is unavailable
    or no call query is defined for *language*.  *tree* optionally supplies a
    pre-parsed tree for *content* so several queries share one parse (P5,
    2026-08-11).
    """
    return _extract_query_pairs(content, language, _CALL_QUERIES, "callee", tree=tree)


def extract_imports(
    content: str,
    language: str,
    tree=None,
) -> list[tuple[str, int]]:
    """Extract import statements: ``[(imported_module, line), ...]``.

    The module string is cleaned: surrounding quotes and trailing semicolons
    are stripped.  *tree* optionally supplies a pre-parsed tree for *content*
    so several queries share one parse (P5, 2026-08-11).

    Lines are 1-indexed.  Returns an empty list if tree-sitter is unavailable
    or no import query is defined for *language*.
    """

    def _clean_module(module: str) -> str:
        module = module.strip().strip("\"';")
        # Scala captures the whole import_declaration node, which includes the
        # leading `import`/`using` keyword; strip it.  Every other language's
        # @source capture is a child node that already excludes the keyword, so
        # the strip is a no-op there — but scope it to Scala to avoid corrupting
        # a pathological module path such as lua `require("import foo")`.
        if language != "scala":
            return module
        module = _IMPORT_KW_RE.sub("", module)
        # Strip namespace selectors and wildcard suffixes from module path.
        # tree-sitter-scala has no scoped_identifier node, so @source captures
        # the entire declaration including selectors like `{c, d}` / `_` / `*`.
        #   e.g., "a.b.{c, d}" → "a.b",  "a.b._" → "a.b",  "a.b.*" → "a.b"
        return re.sub(r"\.(?:\{[^}]*\}|_|\*)\s*$", "", module)

    return _extract_query_pairs(
        content,
        language,
        _IMPORT_QUERIES,
        "source",
        _clean_module,
        tree=tree,
    )


def extract_import_names(
    content: str,
    language: str,
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
                results.extend((module, _text(nch)) for nch in sub.children if nch.type == "identifier")
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


def _extract_name(node) -> str | None:
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
    "local_function_statement": "function",
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
            if child.type in ("function_definition", "class_definition", "async_function_definition"):
                return _node_kind(child)
        return "function"
    return _WALK_KIND_MAP.get(t, "function")


def find_symbol_range(content: str, symbol_name: str, language: str) -> tuple[int, int] | None:
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
            if name_caps and def_caps and name_caps[0].text == symbol_name:
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
    content: str,
    language: str,
    tree=None,
) -> list[tuple[str, str, int, int]]:
    """Extract all top-level symbols: ``[(name, kind, start_line, end_line), ...]``.

    Lines are 1-indexed.  Returns empty list if tree-sitter is unavailable.

    Merges results from BOTH declarative query and manual tree walk.
    The manual walk catches symbol types not covered by queries
    (e.g., ``field_definition`` in TypeScript, which breaks multi-pattern
    queries when included).

    *tree* optionally supplies a pre-parsed tree for *content* so the query
    phase and the manual walk share the caller's single parse (P5, 2026-08-11).

    Duplicates (same name + same line range) are removed.
    """
    # Phase 1: Declarative query (fast, handles standard constructs)
    results: list[tuple[str, str, int, int]] = []
    query_str = _SYMBOL_QUERIES.get(language)
    if query_str is not None:
        _query_results = _find_all_symbols_via_query(content, language, query_str, tree=tree)
        results.extend(_query_results)

    # Phase 2: Manual tree walk (catches field_definition and other
    # symbol types excluded from the query because they break
    # multi-pattern matching in tree-sitter).
    # parse_to_tree is memoised (@lru_cache), so this shares a single parse
    # with Phase 1's query (which also goes through parse_to_tree).
    if tree is None:
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
    content: str,
    language: str,
    query_str: str,
    tree=None,
) -> list[tuple[str, str, int, int]]:
    """Extract top-level symbols using a declarative tree-sitter query."""
    matches = query_matches(content, language, query_str, tree=tree)
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


def _parse_to_tree_uncached(content: str, language: str):
    parser = get_parser(language)
    if parser is None:
        return None
    try:
        return parser.parse(_encode_content(content))
    except Exception:
        logger.debug("tree_sitter parse failed for %s", language, exc_info=True)
        return None


@lru_cache(maxsize=16)
def _parse_to_tree_cached(content: str, language: str):
    return _parse_to_tree_uncached(content, language)


def parse_to_tree(content: str, language: str):
    """Parse *content* and return the tree-sitter Tree object.

    Returns None if tree-sitter is unavailable or parsing fails.
    Callers receive the same Tree object that ``Parser.parse()`` returns;
    they can traverse ``tree.root_node`` and its children directly.

    Memoised: analysis scanners parse the same source several times per run
    (``__all__`` extraction, def collection, reference collection — and again
    per scanner in a pipeline).  Trees are read-only in this codebase, so
    sharing the object is safe.

    Sources larger than ``_MAX_CACHED_SOURCE_CHARS`` bypass the memo and are
    re-parsed on every call (see that constant for the measurements). Callers
    must therefore not assume object identity across calls — only that every
    call returns a valid tree for the content they passed.
    """
    if len(content) > _MAX_CACHED_SOURCE_CHARS:
        return _parse_to_tree_uncached(content, language)
    return _parse_to_tree_cached(content, language)


# Keep the lru_cache surface on the public name: invalidate_caches() calls
# parse_to_tree.cache_clear() to make a late-installed grammar take effect, and
# losing it would silently break grammar hot-reload.
parse_to_tree.cache_clear = _parse_to_tree_cached.cache_clear  # type: ignore[attr-defined]
parse_to_tree.cache_info = _parse_to_tree_cached.cache_info  # type: ignore[attr-defined]
_encode_content.cache_clear = _encode_content_cached.cache_clear  # type: ignore[attr-defined]
_encode_content.cache_info = _encode_content_cached.cache_info  # type: ignore[attr-defined]


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
    return code_bytes[node.start_byte : node.end_byte].decode("utf-8")


# ── Structural analysis helpers (replace numeric/regex guards) ──────────


def _extract_go_class_methods(
    tree: Any,
) -> dict[str, list[tuple[str, int, int]]]:
    """Group Go ``method_declaration`` nodes by normalized receiver type.

    One iterative DFS (explicit stack — avoids Python recursion-limit blow-up
    on deeply nested / machine-generated inputs).  The receiver normalization
    matches ``extract_class_methods``'s former inline rules exactly:

      ``(r *MyStruct)`` / ``(r MyStruct)`` / ``(r pkg.MyStruct)``
      → strip parens, take last space-delimited token, strip ``*``.

    Methods stay in source order within each class (the DFS visits nodes in
    document order).
    """
    grouped: dict[str, list[tuple[str, int, int]]] = {}
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "method_declaration":
            _receiver_node = node.child_by_field_name("receiver")
            if _receiver_node is not None:
                _recv_text = _node_text(_receiver_node)
                if _recv_text is None:
                    continue  # anonymous receiver — nothing to group under
                _recv_clean = _recv_text.strip("()").strip()
                _parts = _recv_clean.split()
                _recv_type = _parts[-1] if len(_parts) >= 2 else _recv_clean
                _recv_type = _recv_type.replace("*", "").strip()
                _name_node = node.child_by_field_name("name")
                if _name_node is not None:
                    _name_text = _node_text(_name_node)
                    if _name_text is None:
                        continue  # anonymous name — nothing to record
                    grouped.setdefault(_recv_type, []).append(
                        (
                            _name_text,
                            node.start_point.row + 1,
                            node.end_point.row + 1,
                        )
                    )
        stack.extend(reversed(node.named_children))
    return grouped


def _filter_go_class_methods(
    grouped: dict[str, list[tuple[str, int, int]]],
    class_name: str,
) -> list[tuple[str, int, int]]:
    """Pick the methods whose receiver type matches *class_name*."""
    results: list[tuple[str, int, int]] = []
    for _recv_type, _methods in grouped.items():
        if class_name in (_recv_type, _recv_type.split(".")[-1]):
            results.extend(_methods)
    return results


def extract_all_class_methods(
    code: str,
    language: str,
) -> dict[str, list[tuple[str, int, int]]] | None:
    """Return ``{class_name: [(method_name, start_line, end_line), ...]}``.

    Batch variant of :func:`extract_class_methods`: parses *code* once and
    groups every class's methods, so N per-class lookups on one file cost one
    parse + one walk instead of N parses.  Currently implemented for Go
    (receiver-based ``method_declaration`` grouping); other languages return
    ``None`` and callers fall back to the per-class path.

    Returns ``None`` when tree-sitter is unavailable or the parse fails —
    distinct from ``{}``, a valid "no methods in this file" answer.
    Lines are 1-indexed.
    """
    tree = parse_to_tree(code, language)
    if tree is None:
        return None
    if language == "go":
        return _extract_go_class_methods(tree)
    return None


def extract_class_methods(
    code: str,
    class_name: str,
    language: str,
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
        # One walk groups every method by receiver type; the class filter is
        # then a dict lookup. _extract_go_class_methods is the single source
        # of the receiver-normalization rules (shared with
        # extract_all_class_methods), so per-class and batch queries cannot
        # drift.
        return _filter_go_class_methods(_extract_go_class_methods(tree), class_name)

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
                _cname = _node_text(_name_node)
                if _cname is not None and (_cname == class_name or _cname.split(".")[-1] == class_name):
                    matched = True
                elif _cname is not None:
                    # Check simple_identifier for Kotlin class names
                    for _ch in node.named_children:
                        if _ch.type == "simple_identifier" and _node_text(_ch) == class_name:
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
                    "function_definition",  # Python
                    "async_function_definition",  # Python async
                    "method_definition",  # TS/JS
                    "method_declaration",  # Java
                    "constructor_declaration",  # Java
                    "function_declaration",  # Kotlin
                ):
                    _method_name = None
                    # Try standard "name" field
                    _mn = item.child_by_field_name("name")
                    if _mn is not None:
                        _method_name = _node_text(_mn)
                    else:
                        # Fallback: property_identifier (TS/JS method_definition)
                        for _ch in item.named_children:
                            if _ch.type == "property_identifier":
                                _method_name = _node_text(_ch)
                                break
                    if _method_name is not None:
                        results.append(
                            (
                                _method_name,
                                item.start_point.row + 1,
                                item.end_point.row + 1,
                            )
                        )
            return results

        stack.extend(reversed(node.named_children))

    return results


def extract_symbol_body(
    code: str,
    symbol_name: str,
    language: str,
) -> tuple[int, int] | None:
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
            "function_definition",  # Python
            "async_function_definition",  # Python async
            "function_declaration",  # Go, Kotlin
            "method_declaration",  # Go, Java
            "method_definition",  # TS/JS
            "constructor_declaration",  # Java
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
                _text = _node_text(_name_node)
                if _text is not None and (_text == symbol_name or _text.rsplit(".", 1)[-1] == symbol_name):
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
