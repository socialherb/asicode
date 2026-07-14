"""Shared utilities for AgentLoop and DesignChatLoop.


Both systems share ToolRegistry, LLMClient, and AgentConfig but previously
duplicated context building, tool result wrapping, and schema filtering.
This module consolidates those common patterns.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import json
import os
import re as _re
import threading
import warnings
from pathlib import Path
from typing import Any, Optional

from .operation_models import OpStatus


def compile_quiet(source: str, filename: str, mode: str = "exec"):
    """``compile()`` with ``SyntaxWarning`` silenced — for syntax gates only.

    The write/modify syntax gates compile candidate user source purely to detect
    a hard ``SyntaxError`` before touching disk. ``compile()`` ALSO emits
    ``SyntaxWarning`` (e.g. an invalid escape ``"\\w"`` in a non-raw string)
    straight to ``stderr`` via the warnings machinery — and during a live
    agent-stream render that stray stderr line lands inside the in-place tool
    status row, corrupting it (the pending ``○`` line can no longer be
    overwritten, so it splits into a stranded ``○`` line + a fresh ``✓`` line).

    Silence ``SyntaxWarning`` here so the gate never leaks into the TUI. A real
    ``SyntaxError`` still propagates unchanged, so gate behaviour is identical.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return compile(source, filename, mode)

# Fail statuses imported by executor modules (replicated from operation_executor.py)
# WARNING: OpStatus.FAILED was missing until 2026-05-31 — ops with status="failed"
# were routed to _handle_op_success_path, miscounting failures as completed ops.
# Keep this set in sync with ALL terminal failure statuses in OpStatus enum.
_FAIL_STATUSES: frozenset = frozenset({
    OpStatus.ERROR, OpStatus.NOT_FOUND, OpStatus.FAILED,
    OpStatus.VERIFICATION_FAILED, OpStatus.EXECUTION_ERROR, OpStatus.PREFLIGHT_FAILED,
})

# Auto-sync guard: every OpStatus whose name contains "FAIL" or "ERROR"
# must be in _FAIL_STATUSES.  This catches omissions when new terminal failure
# statuses are added to OpStatus.
assert _FAIL_STATUSES.issuperset(
    s for s in OpStatus if "FAIL" in s.name or "ERROR" in s.name
), (
    f"_FAIL_STATUSES missing failure status(es): "
    f"{ {s for s in OpStatus if ('FAIL' in s.name or 'ERROR' in s.name) and s not in _FAIL_STATUSES} }"
)

# TS/JS file extensions (source of truth — no local redefinitions in consumers)
# tuple (not frozenset) so it is usable directly with str.endswith();
# this is the single source of truth consumed by _walk_ts_js_files below.
_TS_JS_EXTENSIONS: tuple = ('.ts', '.tsx', '.js', '.jsx')

# ── Shared repo file walkers ─────────────────────────────────────────────────
# Consolidated here so symbol_search and call_graph share ONE walk
# implementation + ONE process-global cache (previously each module walked
# independently; call_graph had no cache at all and re-rglobbed every build).
#
# The skip-set is the union of the two former implementations — the stricter of
# each — so both consumers now also exclude venv/site-packages/*.egg-info dirs
# that call_graph previously indexed.
import time as _walk_time

_WALK_SKIP_DIRS: frozenset = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env", ".tox", "dist", "build",
    ".eggs", "worktrees",
})

# Per-root file-list cache. rglob over a large repo costs ~250ms; repeated
# find_symbol / call-graph builds would pay it every time. Short TTL so newly
# created files become visible quickly. Best-effort: callers tolerate a
# slightly-stale list (missing files → "not found this round").
_WALK_CACHE_TTL: float = 30.0
# Bounded entry cap (P4): these path-keyed caches grew unboundedly in a long-
# lived REPL that visited many repos (each holding a full file list). FIFO
# eviction under the GIL stays consistent with the lock-free, single-threaded
# design; the current repo is the newest entry, stale repos are evicted first.
_WALK_CACHE_MAX_ENTRIES: int = 8
# 3‑tuple: (timestamp, files, was_truncated).  ``was_truncated`` is True when
# the walk exited early because ``max_files`` was reached — on cache hit the
# caller's own cap must be checked (no truncated list may masquerade as a full
# one for a larger cap; see ``_walk_repo_files`` cache-hit logic).
_PY_WALK_CACHE: dict[str, tuple[float, list, bool]] = {}
_TS_WALK_CACHE: dict[str, tuple[float, list, bool]] = {}


def _capped_put(cache: dict, key, value, cap: int = _WALK_CACHE_MAX_ENTRIES) -> None:
    """Set ``cache[key] = value`` then FIFO-evict the oldest entry if over *cap*.

    Pure-dict, GIL-atomic — no lock needed (consistent with the lock-free cache
    family). ``dict`` insertion order (3.7+) yields the oldest via
    ``next(iter(cache))``; the most-recently-inserted path is the current repo,
    so stale repos are the correct eviction candidates.
    """
    cache[key] = value
    while len(cache) > cap:
        _oldest = next(iter(cache))
        cache.pop(_oldest, None)


def _walk_should_skip_dir(d: str) -> bool:
    """True if directory name *d* must be excluded from repo walks.

    Single pruning predicate shared by both ``_walk_py_files`` and
    ``_walk_ts_js_files`` so the two walkers cannot drift. They previously
    diverged: the TS/JS walker carried a redundant ``node_modules`` substring
    check (already in ``_WALK_SKIP_DIRS`` as an exact match) while *missing*
    ``venv*`` (e.g. ``venv310``, ``myvenv``) and ``site-packages`` dirs —
    letting vendored JS/TS bundled inside a Python package pollute the index.
    """
    return (
        d.startswith(".")
        or d in _WALK_SKIP_DIRS
        or d.endswith(".egg-info")
        or d.startswith("venv")
        or "site-packages" in d
    )


def _walk_repo_files(root, max_files: int, cache: dict, keep) -> list:
    """Shared walk engine behind :func:`_walk_py_files` / :func:`_walk_ts_js_files`.

    Returns every file under *root* for which ``keep(name)`` is true, skipping
    hidden/vendor/venv dirs via the single :func:`_walk_should_skip_dir`
    predicate. ``dirnames`` is pruned in-place so ``os.walk`` makes a single
    descent — a whole-tree walk including node_modules/.venv would visit tens of
    thousands of irrelevant files. Early-exits at ``max_files`` so a huge vendor
    tree can't exhaust memory/time before the caller's cap check runs, and
    memoizes the result in *cache* (per-root, TTL-bounded via
    :data:`_WALK_CACHE_TTL`, FIFO-bounded via :func:`_capped_put`).

    The two callers pass *distinct* caches so an extension set never
    masquerades as the other, and a single ``os.walk`` pass per extension set
    so one extension (e.g. ``.js``) cannot fill ``max_files`` and exclude
    ``.ts``/``.tsx``.
    """
    key = str(root)
    cached = cache.get(key)
    if cached is not None:
        ts, files, was_truncated = cached
        if (_walk_time.monotonic() - ts) < _WALK_CACHE_TTL:
            if not was_truncated or len(files) >= max_files:
                # Slice to the caller's cap THEN shallow-copy. A complete walk
                # cached under a large cap (e.g. vulture max_files=4000) must
                # not hand a smaller caller (symbol_search max_files=600) more
                # files than it asked for — callers consume the list directly
                # without re-slicing, so an over-long result causes redundant
                # symbol indexing. The copy prevents cache pollution from
                # callers that mutate the result (.append() / .sort()).
                return list(files[:max_files])
            # Truncated and the cached result doesn't have enough files for this
            # caller's cap — re-walk to collect the required number.

    results: list = []
    _was_truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _walk_should_skip_dir(d)]
        for name in filenames:
            if keep(name):
                results.append(Path(dirpath) / name)
                if len(results) >= max_files:
                    _capped_put(cache, key, (_walk_time.monotonic(), results, True))
                    return results
    _capped_put(cache, key, (_walk_time.monotonic(), results, False))
    return results


def _walk_py_files(root, max_files: int) -> list:
    """Walk *root* returning .py files, skipping hidden/vendor/venv dirs.

    Results are cached per root for ``_WALK_CACHE_TTL`` seconds.
    """
    return _walk_repo_files(root, max_files, _PY_WALK_CACHE, lambda n: n.endswith(".py"))


def _walk_ts_js_files(root, max_files: int) -> list:
    """Walk *root* returning TS/JS files, skipping hidden/vendor/node_modules.

    Cached per root with the same TTL scheme as :func:`_walk_py_files`. A single
    ``os.walk`` pass collects all four extensions (``.ts/.tsx/.js/.jsx``) so one
    extension (e.g. ``.js``) cannot fill ``max_files`` and exclude ``.ts``/
    ``.tsx`` — the primary source files of a TypeScript project.
    """
    return _walk_repo_files(
        root, max_files, _TS_WALK_CACHE, lambda n: n.endswith(_TS_JS_EXTENSIONS)
    )


def make_tool_signature(tool_name: str, tool_args: Any) -> str:
    """Return a stable cross-process signature for a (tool_name, tool_args) pair.

    Used for tool success/failure memory, failure-loop (fail_streak) detection,
    and any other per-call keying that must be collision-resistant and
    invariant across process restarts.

    Why not `hash(json.dumps(...))`?  Two problems with the older pattern:

      1. Collision / false positives — built-in `hash()` returns a 64-bit
         int. Two genuinely different `tool_args` dicts can collide, causing
         one call's failure to be charged to a *different* call's key. For
         loop detection this means an unrelated call could trip
         `fail_streak[key] == threshold`, producing a spurious STRATEGY
         WARNING for a call that never actually failed.
      2. PYTHONHASHSEED instability — `hash()` of str/bytes is randomized
         per interpreter launch. A signature persisted in one run (e.g.
         checkpoint/resume, weight-learning stores) becomes unreadable in
         the next, silently losing memory.

    `hashlib.sha256` (already the project convention — see
    tool_result_cache._make_key and learning.problem_signature) avoids both.

    Returns the hex digest (full length) so consumers can truncate if they
    need a shorter key.
    """
    import hashlib

    stable_args = json.dumps(tool_args, sort_keys=True, default=str)
    key_str = f"{tool_name}:{stable_args}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def is_test_block_description(name: str, file_content: str) -> bool:
    """Check if `name` looks like a test block description string in file_content.

    Test frameworks (Jest, Vitest, Mocha) use patterns like:
      describe('sum', () => { ... })
      it('should add numbers', () => { ... })
      test('compact removes falsy values', () => { ... })

    These 'names' are human-readable description strings, NOT code symbols.
    Returns True when `name` appears as a test block argument in the file.
    """
    if not name or not file_content:
        return False
    _ename = _re.escape(name)
    for _quote in ("'", '"', '`'):
        _eq = _re.escape(_quote)
        _pattern = _re.compile(
            r"""(?:describe|it|test|beforeEach|afterEach|beforeAll|afterAll)\("""
            rf"""\s*{_eq}{_ename}{_eq}"""
        )
        if _pattern.search(file_content):
            return True
    return False


# ── TS/JS brace-matching helpers ────────────────────────────────────────


def _brace_skip_str_literal(text: str, start: int) -> int:
    """Skip past a string/template literal starting at *start*.

    Handles single-quoted, double-quoted, and backtick-template strings
    with escape sequences.  Returns the index after the closing quote.
    """
    _quote = text[start]
    _i = start + 1
    while _i < len(text):
        if text[_i] == '\\' and _i + 1 < len(text):
            _i += 2  # skip escaped char
            continue
        if text[_i] == _quote:
            return _i + 1  # past closing quote
        _i += 1
    return len(text)  # unterminated — consume rest


def _brace_skip_line_comment(text: str, start: int) -> int:
    """Skip past a //-style line comment starting at *start*.

    The caller must ensure text[start:start+2] == '//'.
    Returns the index after the newline (or end of text).
    """
    nl = text.find('\n', start + 2)
    return nl + 1 if nl >= 0 else len(text)


def _brace_skip_block_comment(text: str, start: int) -> int:
    """Skip past a /* */ block comment starting at *start*.

    The caller must ensure text[start:start+2] == '/*'.
    Returns the index after '*/'.
    """
    end = text.find('*/', start + 2)
    return end + 2 if end >= 0 else len(text)  # unterminated


def _brace_skip_regex_literal(text: str, start: int) -> int:
    """Attempt to skip past a regex literal starting at *start*.

    TS regex detection is tricky (needs full parser context); this is a
    best-effort heuristic: forward-scan for an unescaped '/' that ends
    the regex, counting nested '[' ']' pairs.

    Returns index after closing '/' on success, or *start* (no skip) on
    uncertainty to avoid false skips.
    """
    if text[start] != '/':
        return start
    # A regex literal can appear after: =, (, ,, !, &, |, ?,
    # :, ;, {, return, typeof, etc.  Skip detection if preceding
    # char suggests we're in a division context.
    if start > 0 and text[start - 1].isalnum() and text[start - 1] not in ('n', 'r'):
        # 'n' for 'return', 'r' for 'typeof' — too complex, be conservative
        return start
    _i = start + 1
    _bracket_depth = 0
    while _i < len(text):
        if text[_i] == '\\' and _i + 1 < len(text):
            _i += 2
            continue
        if text[_i] == '[':
            _bracket_depth += 1
        elif text[_i] == ']':
            _bracket_depth -= 1
        elif text[_i] == '/' and _bracket_depth == 0:
            return _i + 1  # past closing /
        _i += 1
    return start  # uncertainty — don't skip


def _brace_match_depth(text: str, start: int, initial_depth: int = 1) -> int:
    """Scan *text* from *start*, tracking brace depth with string/comment skipping.

    Returns the index AFTER the brace that brings depth to 0, or len(text)
    if no matching brace is found.

    Skips braces inside:
    - String/template literals ('...', "...", '...')
    - Single-line comments (//)
    - Block comments (/* */)
    - Regex literals (best-effort)
    """
    _depth = initial_depth
    _i = start
    while _i < len(text):
        _ch = text[_i]
        # String / template literals
        if _ch in ('"', "'", '`'):
            _i = _brace_skip_str_literal(text, _i)
            continue
        # Comments
        if _ch == '/' and _i + 1 < len(text):
            if text[_i + 1] == '/':
                _i = _brace_skip_line_comment(text, _i)
                continue
            if text[_i + 1] == '*':
                _i = _brace_skip_block_comment(text, _i)
                continue
        # Regex literal (best-effort)
        if _ch == '/':
            _next = _brace_skip_regex_literal(text, _i)
            if _next > _i:
                _i = _next
                continue
        # Brace tracking
        if _ch == '{':
            _depth += 1
        elif _ch == '}':
            _depth -= 1
            if _depth == 0:
                return _i + 1
        _i += 1
    return len(text)


def _find_class_body_range(source: str, class_name: str) -> Optional[tuple]:
    """Find a class's body byte range in TS/JS source, handling strings/comments.

    Returns (open_brace_byte + 1, close_brace_byte) — the range of content
    INSIDE the class braces.  Returns None if class is not found.

    Unlike naive brace counting, this helper uses ``_brace_match_depth`` to
    skip braces inside string literals, template literals, and comments.
    """
    import re
    _class_header_re = re.compile(
        rf"(?:export\s+)?(?:abstract\s+)?class\s+{re.escape(class_name)}\s*"
        r"(?:extends\s+\S+(?:\s*,\s*\S+)*\s*)?"
        r"(?:implements\s+\S+(?:\s*,\s*\S+)*\s*)?\{"
    )
    _match = _class_header_re.search(source)
    if not _match:
        return None

    _after_brace = source[_match.end():]
    _scope_end = _brace_match_depth(_after_brace, 0, initial_depth=1)
    return (_match.end(), _match.end() + _scope_end)




def _ts_class_scan_methods(source: str, class_name: str) -> Optional[tuple[int, int, str]]:
    """Scan a TS/JS class body and return the last method's (line, end_line, name).

    1-indexed. Returns None if class or no method found.
    Uses regex-based detection — sufficient for anchor-fallback purposes.
    """
    import re as _re
    _body = _find_class_body_range(source, class_name)
    if _body is None:
        return None
    _body_start_byte, _body_end_byte = _body
    _class_body = source[_body_start_byte:_body_end_byte]
    _class_body_lines = _class_body.splitlines(keepends=False)
    if not _class_body_lines:
        return None

    _method_re = _re.compile(
        r'^\s*(?:public|private|protected|static|readonly|async|\s)*\s*'
        r'(?:get\s+|set\s+)?'
        r'(?P<name>[a-zA-Z_$]\w*)\s*[(<]'
    )
    _last_method = None  # (line_1idx_in_source, name)
    _current_line_1idx = source[:_body_start_byte].count('\n') + 1
    for _line_text in _class_body_lines:
        _m = _method_re.match(_line_text)
        if _m:
            _name = _m.group('name')
            if _name not in ('constructor', 'new', class_name):
                _last_method = (_current_line_1idx, _name)
        _current_line_1idx += 1

    if _last_method is None:
        return None
    _method_line, _method_name = _last_method

    # End line: scan forward for next sibling method or class closing brace
    _all_lines = source.splitlines(keepends=False)
    _depth = 0
    _found_self = False
    for _i in range(_method_line - 1, len(_all_lines)):
        _ln = _all_lines[_i]
        _depth += _ln.count('{') - _ln.count('}')
        if not _found_self:
            if _i == _method_line - 1:
                _found_self = True
            continue
        # Next sibling at same or lower depth?
        if _depth <= 0:
            return (_method_line, _i + 1, _method_name)
        if _depth <= 1 and _re.match(
            r'^\s*(?:public|private|protected|static|readonly|async|\s)*\s*'
            r'(?:get\s+|set\s+)?'
            r'[a-zA-Z_$]\w*\s*[(<]',
            _ln,
        ):
            return (_method_line, _i, _method_name)
    return (_method_line, len(_all_lines), _method_name)


# ── TS/JS symbol detection ──────────────────────────────────────────────


def _is_real_ts_symbol(name: str, file_content: str, file_path: str = "") -> bool:
    """Check if `name` is a real TypeScript/JavaScript symbol using tree-sitter AST.

    Uses tree-sitter AST parsing when available (most accurate).
    Falls back to deep tree-sitter traversal (``symbol_exists_deep``).

    Returns True if:
      - name appears in a function/class/const/let/var/interface/type/enum declaration
      - name is a method, accessor (get/set), or abstract method in a class
      - name is an export default or export list entry
      - name is an import specifier
      - name is defined as a const arrow function
    Returns False if name only appears as a test block description string.

    Originally defined in ts_aware_strategy.py; migrated here as part of
    TSAwareCandidateStrategy removal (2026-06-04).
    """
    if not name or not file_content:
        return False

    # ── Primary: tree-sitter AST (most accurate) ────────────────────
    _lang = "typescript"
    if file_path:
        _ext = os.path.splitext(file_path)[1].lower()
        _ts_lang_map = {".ts": "typescript", ".tsx": "typescript",
                        ".js": "javascript", ".jsx": "javascript"}
        _detected = _ts_lang_map.get(_ext)
        if _detected:
            _lang = _detected
            try:
                from external_llm.languages.tree_sitter_utils import find_all_symbols
                _symbols = find_all_symbols(file_content, _lang)
                if _symbols:
                    _bare = name.split(".")[-1]
                    for _sym_name, _kind, _start, _end in _symbols:
                        if _sym_name == _bare or _sym_name == name:
                            return True
            except Exception:
                pass

    # ── Deep tree-sitter traversal (replaces regex fallback) ────────
    from external_llm.languages.tree_sitter_utils import symbol_exists_deep
    return symbol_exists_deep(file_content, name, _lang)


# ── Tree-sitter class member node types (TS/JS grammar) ───────────────
_TS_CLASS_MEMBER_TYPES: frozenset = frozenset({
    "method_definition",
    "field_definition",
    "public_field_definition",
    "abstract_method_signature",
})

# ── Tree-sitter top-level definition node types (TS/JS) ────────────────
_TS_TOP_LEVEL_TYPES: frozenset = frozenset({
    "function_declaration",
    "class_declaration",
    "abstract_class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "lexical_declaration",
})


def _ts_unwrap_export(node):
    """Unwrap ``export_statement`` to reveal the inner definition.

    Returns the inner definition node (function, class, interface, etc.),
    or ``None`` if the export is a re-export (``export { ... }`` /
    ``export * from ...``) or has no recognised inner node.
    """
    n = node
    while n.type == "export_statement":
        for child in n.children:
            if child.type in _TS_TOP_LEVEL_TYPES:
                n = child
                break
        else:
            return None  # re-export or anonymous default
    return n


def _ts_unwrap_decorator(node):
    """If *node* is a ``decorated_definition``, return the inner definition.

    E.g. ``@Bind() method() { ... }`` → the ``method_definition`` node.
    Returns ``None`` if *node* is not a decorated definition.
    """
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in (_TS_CLASS_MEMBER_TYPES | _TS_TOP_LEVEL_TYPES):
                return child
    return None


def _ts_extract_def_name(node):
    """Extract the name of a top-level or class-member definition node."""
    if node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
        return None
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    return None


def _ts_member_name(node):
    """Extract the member name from a class-body child node.

    Handles ``method_definition``, ``field_definition``,
    ``public_field_definition``, and ``decorated_definition`` wrappers.
    Returns ``None`` for non-member nodes (semicolons, index signatures, …).
    """
    inner = _ts_unwrap_decorator(node)
    if inner is None:
        inner = node
    if inner.type in _TS_CLASS_MEMBER_TYPES:
        name_node = inner.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
    return None


def _ts_has_top_level_def(root, symbol: str) -> bool:
    """Return ``True`` if a top-level definition named *symbol* exists.

    Searches all direct children of *root*, unwrapping both
    ``export_statement`` and ``decorated_definition`` wrappers.
    """
    for child in root.children:
        target = _ts_unwrap_export(child)
        if target is None:
            continue
        target = _ts_unwrap_decorator(target) or target
        if target.type not in _TS_TOP_LEVEL_TYPES:
            continue
        if _ts_extract_def_name(target) == symbol:
            return True
    return False


def _ts_class_has_member(root, class_name: str, member_name: str) -> bool:
    """Return ``True`` if *class_name* has a member named *member_name*."""
    for child in root.children:
        target = _ts_unwrap_export(child)
        if target is None:
            continue
        if target.type not in ("class_declaration", "abstract_class_declaration"):
            continue
        name_node = target.child_by_field_name("name")
        if name_node is None:
            continue
        _decoded = name_node.text.decode("utf-8")
        if _decoded != class_name:
            continue
        body = target.child_by_field_name("body")
        if body is None:
            continue
        for member in body.children:
            if _ts_member_name(member) == member_name:
                return True
    return False


def _ts_plain_name_as_member(root, symbol: str) -> bool:
    """Check if a bare *symbol* is a member name inside any class body.

    Used when *symbol* is not dotted — a name like ``lockPiece`` may be
    a class method rather than a top-level definition.
    """
    for child in root.children:
        target = _ts_unwrap_export(child)
        if target is None:
            continue
        if target.type not in ("class_declaration", "abstract_class_declaration"):
            continue
        body = target.child_by_field_name("body")
        if body is None:
            continue
        for member in body.children:
            if _ts_member_name(member) == symbol:
                return True
    return False


def _ts_symbol_exists(root, symbol: str) -> bool:
    """Check if *symbol* exists in the tree-sitter AST *root*.

    For dotted names (``Game.lockPiece``): finds the class and walks its
    body children using grammar-aware member detection.

    For plain names: checks top-level definitions first, then falls back
    to searching inside all class bodies for a matching member name.
    """
    if "." in symbol:
        parts = symbol.split(".", 1)
        return _ts_class_has_member(root, parts[0], parts[1])
    return _ts_has_top_level_def(root, symbol) or _ts_plain_name_as_member(root, symbol)


def ts_symbol_exists_in_file(file_path: str, symbol: str) -> bool:
    """Check if a TS/JS symbol exists in the given file.

    **Primary path:** tree-sitter AST traversal — grammar-aware, no regex.
    Detection is precise for all edge cases (decorators, getters/setters,
    template literals, string escapes, regex literals, …).

    **Fallback path:** regex heuristics (tree-sitter unavailable).

    Supports:
    - Plain names: ``SHAPES``, ``game``, ``randomPiece``
    - Dotted names: ``Game.lockPiece``, ``Game.SHAPES``
    - Top-level: function, const/let, class, interface, type alias, enum
    - Class methods/fields: ``ClassName.memberName`` or bare ``memberName``
    - Export/decorator wrappers — handled automatically by the AST walk

    Args:
        file_path: Absolute path to the file.
        symbol: Symbol name (optionally dotted like ``ClassName.method``).

    Returns:
        True if the symbol is confirmed present in the file.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as _fh:
            content = _fh.read()
    except OSError:
        return False

    # ── Tree-sitter AST path (primary) ──────────────────────────────
    ext = os.path.splitext(file_path)[1].lower()
    _LANG_MAP = {'.ts': 'typescript', '.tsx': 'typescript',
                  '.js': 'javascript', '.jsx': 'javascript'}
    language = _LANG_MAP.get(ext)
    if language:
        try:
            from ..languages.tree_sitter_utils import is_available, parse_to_tree
            if is_available():
                tree = parse_to_tree(content, language)
                if tree:
                    if _ts_symbol_exists(tree.root_node, symbol):
                        return True
        except Exception:
            pass  # fall through to legacy fallback

    # ── Legacy fallback (tree-sitter unavailable) ──────────────────
    return _legacy_regex_symbol_exists(content, symbol)


def _legacy_regex_symbol_exists(content: str, symbol: str) -> bool:
    """Legacy regex-based symbol detection (tree-sitter not available).

    ═══════════════════════════════════════════════════════════════════
    This function exists solely as a fallback for environments where
    tree-sitter and its language grammars are not installed.

    Regex-based symbol matching is inherently fragile (template literals,
    string escapes, decorators, getter/setter syntax, …).  When
    tree-sitter *is* available (the common case), it is never called.
    ═══════════════════════════════════════════════════════════════════
    """
    import re

    if "." in symbol:
        parts = symbol.split(".", 1)
        class_name = parts[0]
        member_name = parts[1]
        _body_range = _find_class_body_range(content, class_name)
        if _body_range is None:
            return False
        _class_body = content[_body_range[0]:_body_range[1]]
        _method_re = re.compile(
            rf"(?:public|private|protected|static|readonly|async|\s)*\b{re.escape(member_name)}\s*[\(=:<]"
        )
        return bool(_method_re.search(_class_body))

    _patterns = [
        rf"(?:export\s+)?(?:async\s+)?function\s+{re.escape(symbol)}\s*[\(<]",
        rf"(?:export\s+)?(?:const|let|var)\s+{re.escape(symbol)}\s*[=:]",
        rf"(?:export\s+)?(?:abstract\s+)?class\s+{re.escape(symbol)}\s*(?:extends|implements|<|\{{)",
        rf"(?:export\s+)?interface\s+{re.escape(symbol)}\s*(?:extends|<|\{{)",
        rf"(?:export\s+)?type\s+{re.escape(symbol)}\s*(?:=|<)",
        rf"(?:export\s+)?(?:const\s+)?enum\s+{re.escape(symbol)}\s*\{{",
    ]
    for _pat in _patterns:
        if re.search(_pat, content, re.MULTILINE):
            return True

    _class_header_re = re.compile(
        r"(?:export\s+)?(?:abstract\s+)?class\s+\w+\s*"
        r"(?:extends\s+\S+(?:\s*,\s*\S+)*\s*)?"
        r"(?:implements\s+\S+(?:\s*,\s*\S+)*\s*)?\{"
    )
    _method_re = re.compile(
        rf"(?:public|private|protected|static|readonly|async|\s)*\b{re.escape(symbol)}\s*[\(=:<]"
    )
    for _class_match in _class_header_re.finditer(content):
        _after_brace = content[_class_match.end():]
        _scope_end = _brace_match_depth(_after_brace, 0, initial_depth=1)
        _class_body = _after_brace[:_scope_end]
        if _method_re.search(_class_body):
            return True

    return False



# ── LLM cost estimation ──────────────────────────────────────────────────────

# (input_per_M_usd, output_per_M_usd)
# Provider-level pricing — fallback when no model-specific match.
_COST_PER_M: dict[str, tuple[float, float]] = {
    "google":    (0.10,  0.40),
    "openai":    (5.00, 15.00),
    "anthropic": (3.00, 15.00),
    "deepseek":  (0.27,  1.10),
    "ollama":    (0.00,  0.00),
    "zai":       (1.40,  4.40),
    # OpenRouter serves many vendors; no single representative price. Default to
    # a low DeepSeek-tier rate since the common OpenRouter workloads (DeepSeek
    # Flash/Pro) are cheap — model-specific entries in _MODEL_COST_PER_M win.
    "openrouter": (0.27, 1.10),
}

# Model-specific pricing (prefix-matched, checked before provider fallback).
# Sources (verified 2026-06):
#   DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
#   Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
#   OpenAI:    https://openai.com/api/pricing/
#   Google:    https://ai.google.dev/pricing
#   Z.AI:      https://docs.z.ai/guides/overview/pricing
_MODEL_COST_PER_M: dict[str, tuple[float, float]] = {
    # DeepSeek — V4-Pro 75% discount made permanent 2026-05-22
    "deepseek-v4-flash":    (0.14,  0.28),
    "deepseek-v4-pro":      (0.435, 0.87),
    "deepseek-reasoner":    (0.55,  2.19),
    "deepseek-r1":          (0.55,  2.19),
    "deepseek-chat":        (0.27,  1.10),
    # Anthropic
    "claude-fable-5":       (15.00, 75.00),
    "claude-mythos-5":      (15.00, 75.00),
    "claude-4-opus":        (15.00, 75.00),
    "claude-opus-4-8":      (15.00, 75.00),
    "claude-opus-4-7":      (15.00, 75.00),
    "claude-sonnet-5":      (3.00, 15.00),
    "claude-3-5-sonnet":    (3.00, 15.00),
    "claude-sonnet-4-6":    (3.00, 15.00),
    "claude-sonnet-4-5":    (3.00, 15.00),
    "claude-3-5-haiku":     (0.80,  4.00),
    "claude-haiku-4-5":     (0.80,  4.00),
    "claude-3-opus":        (15.00, 75.00),
    "claude-3-sonnet":      (3.00, 15.00),
    "claude-3-haiku":       (0.25,  1.25),
    # OpenAI
    "gpt-4o-mini":          (0.15,  0.60),
    "gpt-4o":               (2.50, 10.00),
    "gpt-4.1":              (2.00,  8.00),
    "o3-mini":              (1.10,  4.40),
    "o4-mini":              (1.10,  4.40),
    # Google
    "gemini-2.0-flash":     (0.10,  0.40),
    "gemini-2.5-pro":       (1.25,  5.00),
    # OpenRouter / third-party models served via OpenAI-compatible API.
    # OpenRouter slugs use the ``<vendor>/<model>`` form. These are
    # LONGEST-prefix-matched, so a vendor-prefixed slug (e.g.
    # ``deepseek/deepseek-v4-flash``) must be listed explicitly to win over the
    # bare ``deepseek-...`` entries above — otherwise the cheaper OpenRouter
    # rate would be shadowed by the native DeepSeek price.
    # Source (verified 2026-06): https://openrouter.ai/models
    "deepseek/deepseek-v4-flash":  (0.09,  0.18),   # 35% cheaper than native
    "deepseek/deepseek-v4-pro":    (0.435, 0.87),    # same as native
    "qwen/qwen3.6":                (0.289, 2.40),
    # Z.AI — source: https://docs.z.ai/guides/overview/pricing (verified 2026-06)
    "glm-5.2":              (1.40,  4.40),
    "glm-5.1":              (1.40,  4.40),
    "glm-5-turbo":          (1.20,  4.00),
    "glm-5":                (1.00,  3.20),
    "glm-4.7":              (0.60,  2.20),
    "glm-4.6":              (0.60,  2.20),
    "glm-4.5":              (0.60,  2.20),
}

# Fraction of input rate charged for cached tokens (e.g. 0.1 → 10%).
# Provider-level default — used when no model-specific match applies.
# NOTE: DeepSeek's cache discount varies widely by model — v4-flash/v4-pro are
# ~2%/0.8% while deprecated chat/reasoner are ~26%. Model-specific cached rates
# in ``_MODEL_CACHE_RATE`` take precedence; this fallback covers any unknown
# DeepSeek model (conservative 26% ≈ average of deprecated models).
# OpenRouter applies a flat 10% discount on its own input rate (not the native
# provider's rate), per DEEPSEEK_CACHE_READ_MULTIPLIER in their docs.
_CACHE_DISCOUNT: dict[str, float] = {
    "anthropic": 0.1,
    "deepseek":  0.26,
    "openrouter": 0.1,
}

# Model-specific cached-input rate ($/1M tokens), stored directly rather than as
# a discount fraction. Z.AI charges a *different* rate per model tier, so deriving
# a single provider-level discount would be inaccurate AND floating-point division
# (cached_rate / in_rate) introduces rounding error. Storing the rate verbatim
# from the price sheet keeps cost math bit-exact against the official numbers.
# Prefix-matched against ``model`` exactly like ``_MODEL_COST_PER_M`` in ``_get_rates``.
# Source (verified 2026-06): https://docs.z.ai/guides/overview/pricing
# Z.AI's Cached Input Storage is "Limited-time Free", so no cache-creation premium.
_MODEL_CACHE_RATE: dict[str, float] = {
    # DeepSeek — cache-hit rates per model (source: api-docs.deepseek.com/quick_start/pricing)
    # Stored as $/1M tokens (not a discount fraction) to avoid rounding error.
    "deepseek-v4-flash":  0.0028,
    "deepseek-v4-pro":    0.003625,
    "deepseek-chat":      0.07,
    "deepseek-reasoner":  0.14,
    "deepseek-r1":        0.14,
    # Z.AI GLM models — source: https://docs.z.ai/guides/overview/pricing
    "glm-5.2":     0.26,
    "glm-5.1":     0.26,
    "glm-5-turbo": 0.24,
    "glm-5":       0.20,
    "glm-4.7":     0.11,
    "glm-4.6":     0.11,
    "glm-4.5":     0.11,
}


def _is_zai_payg_url(base_url: str) -> bool:
    """Detect whether a z.ai base URL indicates pay-as-you-go billing.

    z.ai has three endpoint families:
    - /anthropic/v1...     → Coding Plan (prompt-unit billing, no cache discount)
    - /coding/paas/v4...   → Coding Plan (prompt-unit billing, no cache discount)
    - /paas/v4...          → Pay-as-you-go (token-unit billing, cache discount applies)

    Returns True when the URL matches the pay-as-you-go endpoint.
    When ``base_url`` is empty (not provided), returns False (safe default = Coding Plan).
    """
    if not base_url:
        return False
    url_lower = base_url.lower()
    has_paas_v4 = "/paas/v4" in url_lower
    has_coding = "/coding/" in url_lower
    has_anthropic = "/anthropic/" in url_lower
    return has_paas_v4 and not has_coding and not has_anthropic


# Providers whose reported prompt/input token count EXCLUDES cached tokens.
# For these, cache_read / cache_creation are reported SEPARATELY and are NOT a
# subset of prompt_tokens, so they must be added on top (read at a discount,
# write at a premium) rather than re-priced within prompt_tokens.
#   - Anthropic: usage.input_tokens excludes cache_read_input_tokens and
#     cache_creation_input_tokens.
#   - zai: served via ZAIAnthropicClient over the Anthropic Messages API, so its
#     usage shape is identical to Anthropic — input_tokens EXCLUDES the
#     separately-reported cache_read_input_tokens. Routing it through the
#     subset formula yields >100% hit rates (e.g. 3241% cached) and mis-costs (cache
#     reads get capped inside a too-small prompt_tok). ZAIClient (OpenAI
#     protocol, used as the failover sibling of ZAIAnthropicClient) inherits
#     OpenAI's subset semantics — prompt_tokens INCLUDES cached_tokens — but
#     re-normalizes to the separate shape at its boundary
#     (ZAIClient._normalize_cache_accounting: prompt_tokens -= cached). So
#     "zai" is always separate-accounting by the time tokens reach these
#     formulas, regardless of which z.ai facade served the request.
#   - DeepSeek/OpenAI: prompt_tokens INCLUDES cache_read tokens as a subset.
_CACHE_TOKENS_SEPARATE: set = {"anthropic", "zai"}

# Multiplier on the input rate charged for cache-WRITE (creation) tokens.
# Anthropic charges a 25% premium to write the cache (1.25× input rate).
_CACHE_CREATION_MULT: dict[str, float] = {"anthropic": 1.25}


def _longest_prefix_match(model_lower: str, table: dict[str, Any]):
    """Return the value for the LONGEST matching prefix in ``table``, or None.

    Cost tables are prefix-matched (e.g. ``"glm-5"`` matches ``"glm-5.2-x"``).
    A naive first-match scan is insertion-order dependent — if ``"glm-5"`` were
    listed before ``"glm-5.2"``, the more specific rate would be shadowed. This
    helper matches on the *longest* prefix so model resolution is order-independent,
    making new-model additions safe regardless of dict ordering.
    """
    _best_prefix, _best_val = "", None
    for prefix, val in table.items():
        if model_lower.startswith(prefix) and len(prefix) > len(_best_prefix):
            _best_prefix, _best_val = prefix, val
    return _best_val


def _get_rates(provider: str, model: str = "") -> tuple[float, float]:
    """Get (input_per_M_usd, output_per_M_usd) for a provider+model combination.

    Tries model-specific pricing first (longest-prefix-matched against
    ``_MODEL_COST_PER_M``), then falls back to provider-level pricing via
    ``_COST_PER_M``.
    """
    if model:
        rates = _longest_prefix_match(model.lower(), _MODEL_COST_PER_M)
        if rates is not None:
            return rates
    return _COST_PER_M.get(provider.lower(), (0.0, 0.0))


def estimate_cost(provider: str, prompt_tok: int, completion_tok: int, model: str = "") -> float:
    """Return estimated USD cost for the given token counts. Optional ``model`` for model-specific rates."""
    in_rate, out_rate = _get_rates(provider, model)
    return (prompt_tok * in_rate + completion_tok * out_rate) / 1_000_000

def _get_cached_input_rate(
    provider: str, in_rate: float, model: str = "", base_url: str = ""
) -> Optional[float]:
    """Return the per-M-token rate charged for cached input tokens.

    Tries a model-specific cached rate first (prefix-matched against
    ``_MODEL_CACHE_RATE``), then derives one from the provider-level discount
    in ``_CACHE_DISCOUNT``. Returns ``None`` when the provider does not offer a
    cache discount (full input price applies to cached tokens).

    ``base_url`` is used for z.ai only to detect the billing model:
    Coding Plan endpoints (/anthropic/v1, /coding/paas/v4) do NOT offer
    a cache discount, while the pay-as-you-go endpoint (/paas/v4) does.
    When empty (default), assumes Coding Plan (no discount).
    """
    if provider.lower() == "zai" and not _is_zai_payg_url(base_url):
        # z.ai Coding Plan: cached tokens billed at full input rate (no discount).
        return None
    if model:
        cached_rate = _longest_prefix_match(model.lower(), _MODEL_CACHE_RATE)
        if cached_rate is not None:
            return cached_rate
    discount = _CACHE_DISCOUNT.get(provider.lower())
    return in_rate * discount if discount is not None else None


def estimate_cache_adjusted_cost(
    provider: str,
    prompt_tok: int,
    completion_tok: int,
    cache_read_tok: int = 0,
    cache_creation_tok: int = 0,
    model: str = "",
    base_url: str = "",
) -> float:
    """Return estimated USD cost accounting for prompt-caching pricing.

    ``model`` enables model-specific per-token rates (see ``_MODEL_COST_PER_M``)
    and model-specific cached-input rates (see ``_MODEL_CACHE_RATE``).

    ``base_url`` is forwarded to ``_get_cached_input_rate`` for z.ai
    billing-model detection (Coding Plan vs. pay-as-you-go).

    Token accounting differs by provider:

    - Anthropic (``_CACHE_TOKENS_SEPARATE``): ``prompt_tok`` (input_tokens)
      EXCLUDES cached tokens. cache_read and cache_creation are billed
      separately — reads at a discount, writes at a premium — and are added on
      top of the full-priced uncached prompt.
    - DeepSeek / OpenAI: ``prompt_tok`` INCLUDES ``cache_read_tok`` as a
      subset, so the cached portion is re-priced from the full rate down to the
      cached rate (a refund against ``raw``).

    Applying the subset formula to a separate-accounting provider (or vice
    versa) yields nonsensical values such as >100% hit rates or negative cost.
    """
    prov = provider.lower()
    in_rate, out_rate = _get_rates(provider, model)
    cached_rate = _get_cached_input_rate(provider, in_rate, model, base_url=base_url)

    if prov in _CACHE_TOKENS_SEPARATE:
        # Disjoint accounting: prompt_tok is full-priced uncached input; add the
        # separately-reported cached tokens on top.
        cost = prompt_tok * in_rate + completion_tok * out_rate
        read_rate = cached_rate if cached_rate is not None else in_rate
        creation_rate = in_rate * _CACHE_CREATION_MULT.get(prov, 1.0)
        cost += cache_read_tok * read_rate + cache_creation_tok * creation_rate
        return cost / 1_000_000

    # Subset accounting: cache_read_tok ⊆ prompt_tok. Re-price the cached part:
    # subtract cached tokens at the full rate and add them back at the cached rate.
    raw = prompt_tok * in_rate + completion_tok * out_rate
    if cache_read_tok and cached_rate is not None:
        cached = min(cache_read_tok, prompt_tok)  # guard against malformed inputs
        raw -= cached * (in_rate - cached_rate)
    return raw / 1_000_000


def total_input_tokens(
    provider: str, prompt_tok: int, cache_read_tok: int, cache_creation_tok: int = 0
) -> int:
    """Total input context size sent to the model for a single LLM call.

    Provider-aware per ``_CACHE_TOKENS_SEPARATE``:
      - separate (Anthropic/zai): ``prompt_tok`` reports only the uncached input;
        both ``cache_read_tok`` AND ``cache_creation_tok`` are reported OUTSIDE it.
        The true context size the model ingested is therefore
        prompt_tok + cache_read_tok + cache_creation_tok. Omitting
        ``cache_creation_tok`` understates occupancy on cache-WRITE turns (cold
        start / post-eviction prefix re-write), making the ``↑`` display drop
        spuriously even though the context actually grew.
      - subset (OpenAI/DeepSeek): cached reads are a SUBSET of ``prompt_tok``,
        so total = prompt_tok (adding cache_read_tok would double-count);
        ``cache_creation_tok`` is always 0 for these providers.

    Use this for context-window-occupancy display (e.g. ``↑48k``); use
    ``cache_hit_pct`` for the cache-read percentage. Both must agree on the
    same denominator, so this helper exists to share it.
    """
    if provider.lower() in _CACHE_TOKENS_SEPARATE:
        return (prompt_tok or 0) + (cache_read_tok or 0) + (cache_creation_tok or 0)
    return prompt_tok or 0
def cache_hit_pct(
    provider: str, prompt_tok: int, cache_read_tok: int, cache_creation_tok: int = 0
) -> float:
    """Return cache-read tokens as a percentage of total input tokens.

    Uses the correct denominator per provider via ``total_input_tokens``:
    for separate-accounting providers (Anthropic/zai) total input = prompt +
    cache_read + cache_creation (cache-WRITE tokens are part of the context the
    model ingested but were NOT served from cache, so they belong in the
    denominator and correctly lower the ratio on cache-WRITE turns); for subset
    providers (DeepSeek/OpenAI) ``prompt_tok`` already includes the cached reads
    and ``cache_creation_tok`` is 0.
    """
    if not cache_read_tok:
        return 0.0
    total_in = total_input_tokens(provider, prompt_tok, cache_read_tok, cache_creation_tok)
    if total_in <= 0:
        return 0.0
    return cache_read_tok * 100.0 / total_in


def cache_cost_summary(
    provider: str,
    prompt_tok: int,
    completion_tok: int,
    cache_read_tok: int = 0,
    cache_creation_tok: int = 0,
    model: str = "",
    base_url: str = "",
) -> tuple[float, float, float]:
    """Return ``(full_cost, actual_cost, hit_pct)`` for cost display.

    - ``full_cost``   — counterfactual USD cost if nothing were cached.
    - ``actual_cost`` — true billed USD cost (cache discounts/premiums applied).
    - ``hit_pct``     — cache-read tokens as % of total input.

    ``model`` enables model-specific per-token rates. ``base_url`` is forwarded
    for z.ai billing-model detection (Coding Plan vs. pay-as-you-go).
    For separate-accounting providers the counterfactual adds the cached tokens
    back at the full input rate so a ``full → actual`` display reads as a real
    saving; for subset providers ``prompt_tok`` already represents the full
    input, so ``full_cost`` matches the legacy behaviour exactly.
    """
    if provider.lower() in _CACHE_TOKENS_SEPARATE:
        full_in = prompt_tok + cache_read_tok + cache_creation_tok
    else:
        full_in = prompt_tok
    full_cost = estimate_cost(provider, full_in, completion_tok, model=model)
    actual_cost = estimate_cache_adjusted_cost(
        provider, prompt_tok, completion_tok,
        cache_read_tok, cache_creation_tok, model, base_url=base_url,
    )
    return full_cost, actual_cost, cache_hit_pct(
        provider, prompt_tok, cache_read_tok, cache_creation_tok
    )


def extract_files_from_patch(patch_text: str) -> list:
    """Extract unique file paths from a unified diff patch.

    Supports both ``+++ b/...`` and ``diff --git a/... b/...`` formats.
    Returns deduplicated list in order of first appearance.
    """
    files = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            f = line[6:].strip()
            if f and f not in files:
                files.append(f)
        elif line.startswith("diff --git a/"):
            _parts = line.split(' b/', 1)
            if len(_parts) >= 2:
                f = _parts[1].strip()
                if f and f not in files:
                    files.append(f)
    return files


def _discover_repo_files(repo_root: str, max_files: int = 120) -> list:
    """Discover files in repo for auto-generating project.md."""
    result = []
    try:
        for _root, _dirs, _files in os.walk(repo_root):
            _dirs[:] = [d for d in _dirs
                        if not d.startswith(".") and d != "__pycache__"
                        and d not in ("node_modules", "venv", ".venv", "dist", "build", ".git")]
            for _f in _files:
                if _f.startswith("."):
                    continue
                _rel = os.path.relpath(os.path.join(_root, _f), repo_root)
                if len(_rel) < 120:
                    result.append(_rel)
                if len(result) >= max_files:
                    return result
    except Exception:
        pass
    return result


def load_project_context_md(repo_root: str) -> str:
    """Read .asicode/project.md and return a formatted context block.

    Both AgentLoop (session-start) and DesignChat (every-turn) inject this
    file to give the model a static architecture reference.
    """
    path = os.path.join(repo_root, ".asicode", "project.md")
    _asicode_dir = os.path.join(repo_root, ".asicode")
    try:
        if not os.path.isfile(path):
            # Auto-generate project.md
            _all = _discover_repo_files(repo_root)
            _parts = [f"# {os.path.basename(repo_root)}", "", "## Repository Structure"]
            for _f in _all[:120]:
                _parts.append(f"- {_f}")
            _content = "\n".join(_parts)
            try:
                os.makedirs(_asicode_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as _fw:
                    _fw.write(_content)
            except Exception as e:
                logger.warning(
                    "Failed to auto-generate %s: %s (disk full or permission denied)",
                    path, e
                )
            return _content
        with open(path, encoding="utf-8") as _f:
            content = _f.read().strip()
        if not content:
            return ""
        return (
            "## ═══ PROJECT CONTEXT (.asicode/project.md) ═══\n"
            "Static architecture reference — use to skip exploratory file reads:\n\n"
            + content
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not load project context: %s", e)
        return ""


# ── Token estimation & context trimming (shared by AgentLoop and DesignChatLoop) ──

CHARS_PER_TOKEN: float = 3.0
"""Rough estimate: ~4 chars/token for English, ~2 for code-heavy text.
Conservative default of 3 chars/token avoids underestimation.
Primarily used by estimate_tokens_from_tool_schemas (English-only schema text)."""

MAX_SAFE_TOKENS: int = 80000
"""Conservative safety margin below typical 128k-200k model limits."""


def _cjk_aware_tokens(text: str) -> int:
    """Estimate tokens via ``utf8_bytes // 2`` (conservative upper bound).

    English/ASCII (~1 byte/char) yields ~2 chars/token.
    CJK text (~3 bytes/char) yields ~1.5 chars/token — a conservative
    upper bound that avoids the 2-3× underestimation of ``chars//3`` alone.
    Returns 0 for empty/None text.

    This is the single canonical token estimator for message content across
    the guard path (``estimate_tokens_from_msgs``, ``preemptive_trim``).
    """
    if not text:
        return 0
    return len(text.encode('utf-8')) // 2 + 1


def estimate_tokens(text: str) -> int:
    """Estimate token count CJK-aware: ``utf8_bytes // 2``."""
    return _cjk_aware_tokens(text)


# ══════════════════════════════════════════════════════════════════════════════
# Wire-block token registry (single source of truth for "which raw_content block
# types we know how to count").  Consumed by ``_estimate_single_message_tokens``.
#
# Wire-drift hazard: providers periodically emit NEW content-block types
# (reasoning_content, thinking, redacted_thinking, gemini functionCall parts …).
# Before this registry each new type silently fell through an if/elif chain and
# was under-counted (often to ~0), which later surfaced as a context-overflow
# 400 — the exact failure the budget subsystem exists to prevent.
#
# The registry seals the class three ways:
#   1. Single source of truth — recognise a new type by registering ONE tokenizer.
#   2. Fail-safe runtime fallback — an UNKNOWN block type is counted wholesale
#      (over-count, never under-count) and logged once-per-type so drift is
#      observable instead of silent.
#   3. Contract test — asserts the registry covers the canonical provider set;
#      adding a fixture for a new provider type fails the test until registered.
# ══════════════════════════════════════════════════════════════════════════════


def _tok_tool_use(block: dict) -> int:
    """tool_use: ``input`` holds actual tool args (bash, patches, …)."""
    n = 0
    inp = block.get('input')
    if isinstance(inp, dict):
        n += len(json.dumps(inp, ensure_ascii=False, default=str)) // 3 + 1
    elif isinstance(inp, str):
        n += _cjk_aware_tokens(inp)
    tname = block.get('name', '')
    if tname:
        n += (len(tname) + 10) // 3 + 1
    return n


def _tok_tool_result(block: dict) -> int:
    """tool_result: ``content`` holds tool output (file reads, bash stdout, …)."""
    n = 0
    _tr_content = block.get('content')
    if isinstance(_tr_content, str):
        n += _cjk_aware_tokens(_tr_content)
    elif isinstance(_tr_content, list):
        for sub in _tr_content:
            if isinstance(sub, dict):
                stext = sub.get('text', '')
                if stext:
                    n += _cjk_aware_tokens(stext)
                elif sub.get('type') == 'image':
                    n += _IMAGE_BLOCK_TOKEN_ESTIMATE
    return n


def _tok_thinking(block: dict) -> int:
    """thinking (Anthropic/zai-native): reasoning trace sent alongside text."""
    return _cjk_aware_tokens(block.get('thinking', ''))


def _tok_redacted_thinking(block: dict) -> int:
    """redacted_thinking: opaque signature payload, still on the wire."""
    return _cjk_aware_tokens(block.get('data', ''))


def _tok_function_call(block: dict) -> int:
    """Gemini functionCall part (typed or content-key form)."""
    fc = block.get('functionCall') or block.get('function_call')
    if isinstance(fc, dict):
        return len(json.dumps(fc, ensure_ascii=False, default=str)) // 3 + 1
    return 0


def _tok_function_response(block: dict) -> int:
    """Gemini functionResponse part (typed or content-key form)."""
    fr = block.get('functionResponse') or block.get('function_response')
    if isinstance(fr, dict):
        return len(json.dumps(fr, ensure_ascii=False, default=str)) // 3 + 1
    return 0


# Providers charge images by pixel geometry (Anthropic: ~(w*h)/750, capped
# around 1.6k tokens per image), NOT by base64 payload length — wholesale
# json-counting a 300 KB screenshot would yield ~130k "tokens" and starve the
# budget after a handful of images.  Without dimensions the real cost is
# unknowable here, so use the provider cap as a flat upper bound (over-counts
# small images, never under-counts).
_IMAGE_BLOCK_TOKEN_ESTIMATE = 1600


def _tok_image(block: dict) -> int:
    """image (Anthropic raw_content form): flat provider-cap estimate."""
    return _IMAGE_BLOCK_TOKEN_ESTIMATE


# type-field value → payload tokenizer.  Plain 'text' blocks are counted by the
# generic text pre-pass in ``_estimate_single_message_tokens`` and intentionally
# NOT listed here (their payload IS the ``text`` field).
_WIRE_BLOCK_TOKENIZERS: dict[str, Any] = {
    'tool_use': _tok_tool_use,
    'tool_result': _tok_tool_result,
    'thinking': _tok_thinking,
    'redacted_thinking': _tok_redacted_thinking,
    'functionCall': _tok_function_call,
    'functionResponse': _tok_function_response,
    'image': _tok_image,
}

# Gemini native ``parts`` carry the type as a TOP-LEVEL KEY rather than in a
# ``type`` field, so type dispatch misses them.  These markers re-route such
# untyped blocks to the matching tokenizer (preserving pre-registry behaviour).
_WIRE_CONTENT_KEY_MARKERS: dict[str, Any] = {
    'functionCall': _tok_function_call,
    'functionResponse': _tok_function_response,
}

# The canonical set of wire block types this subsystem must recognise.  The
# contract test asserts the registry covers exactly this set.  When a provider
# adds a new type, add its fixture here AND register a tokenizer.
CANONICAL_WIRE_BLOCK_TYPES: frozenset[str] = frozenset(_WIRE_BLOCK_TOKENIZERS) | {'text'}

# Module-level counters for unknown wire block types (type → occurrences).
# Replaces the original one-time-per-type set so the stats are observable
# via ``get_unknown_block_type_counts()`` instead of only through logs.
#
# Guarded by ``_unknown_block_types_lock``: token estimation may run concurrently
# across requests, and the read-modify-write below (load → add → store) is not
# atomic under the GIL.  Without the lock, concurrent writers could lose
# increments and the one-time warning could fire more than once per type.
_warned_unknown_block_types: dict[str, int] = {}
_unknown_block_types_lock = threading.Lock()


def _count_block_wholesale(block: dict) -> int:
    """Fail-safe tokenizer for an unrecognised wire block.

    Dumps the entire block to JSON and counts it, guaranteeing we never
    under-count a whole unknown block (under-counting is what causes the
    context-overflow 400s this subsystem prevents).  Slight over-counting is
    always safe — it only trims the budget marginally sooner.
    """
    try:
        chars = len(json.dumps(block, ensure_ascii=False, default=str))
    except Exception:
        chars = sum(len(str(v)) for v in block.values()) if block else 0
    return chars // 3 + 1


def _warn_unknown_block_type(btype: str) -> None:
    """Record an unknown wire block type occurrence.

    Increments a per-type counter (observable via
    ``get_unknown_block_type_counts()``) and emits a one-time warning so wire
    drift is surfaced via both logs and the stats endpoint.

    The warning is emitted outside the lock so a slow log handler never blocks
    other counters; only the RMW on ``_warned_unknown_block_types`` is guarded.
    """
    with _unknown_block_types_lock:
        prev = _warned_unknown_block_types.get(btype, 0)
        _warned_unknown_block_types[btype] = prev + 1
        should_warn = prev == 0
    if should_warn:
        logger.warning(
            "Unknown LLM content-block type %r encountered during token estimation — "
            "counted wholesale (fail-safe). Add it to _WIRE_BLOCK_TOKENIZERS in "
            "external_llm/agent/_shared_utils.py and update CANONICAL_WIRE_BLOCK_TYPES.",
            btype,
        )


def get_unknown_block_type_counts() -> dict[str, int]:
    """Return a snapshot of unknown wire block type occurrence counts.

    Returns a copy so callers cannot mutate the module-level counter.  Holds
    the lock briefly so a concurrent writer cannot observe a partially-updated
    dict.
    """
    with _unknown_block_types_lock:
        return dict(_warned_unknown_block_types)


def reset_unknown_block_type_counts() -> dict[str, int]:
    """Atomically clear the unknown-block-type counters.

    Returns the **pre-reset** snapshot so callers (monitoring, the
    ``/stats/wire-drift?reset=1`` endpoint, tests) observe exactly what was
    cleared — enabling "snapshot → reset → measure delta over a window"
    workflows without a separate read call that could miss events in between.

    After a reset, the next occurrence of a previously-seen type re-emits the
    one-time log warning (``_warn_unknown_block_type`` treats ``prev == 0`` as a
    first sighting).  This is intentional: a reset starts a fresh observation
    window.  The snapshot-and-clear is a single critical section guarded by
    ``_unknown_block_types_lock`` so it is atomic w.r.t. concurrent
    ``_warn_unknown_block_type`` RMW.
    """
    with _unknown_block_types_lock:
        snapshot = dict(_warned_unknown_block_types)
        _warned_unknown_block_types.clear()
        return snapshot


def _estimate_single_message_tokens(m: object) -> int:
    """Compute and cache token estimate for a single message object.

    Caches the result on ``m._msg_token_estimate`` so repeated calls (e.g.
    pre-trim + post-trim + overflow-retry in the same turn) skip re-counting
    and re-``json.dumps`` for messages that survive trimming.  The cache lives
    as long as the message object, which is exactly the turn lifetime.

    .. warning::
        The cache has **no invalidation mechanism**.  It is safe **only** because
        every mutation path (``_evict_consumed_tool_results`` → ``_stub_tool_result``)
        creates a **copy-on-write** via ``dataclasses.replace`` (new object, no cached
        attr).  Any future in-place mutation of ``.content`` / ``.raw_content`` on an
        already-estimated message **will** silently return a stale count.  If such a
        path is added, invalidate the cache (``del m._msg_token_estimate``) in the
        mutator or key the cache against a content-length fingerprint.

    Supports both ``LLMMessage`` objects (cache via attribute) and plain
    ``dict`` messages (always recompute — dict has no writable ``__dict__``).
    """
    # Cache check — only for mutable objects with __dict__ (not plain dicts).
    _can_cache = not isinstance(m, dict) and hasattr(m, '__dict__')
    if _can_cache:
        cached = getattr(m, '_msg_token_estimate', None)
        if cached is not None:
            return cached

    mt = 0
    # Content (CJK-aware) — skip when raw_content is present because it is the
    # authoritative wire form; content is a derived mirror. Counting both would
    # double-count assistant text (anthropic/zai/native assistant messages include
    # the same text in both content and raw_content text blocks).
    content = getattr(m, 'content', '') or ''
    rc = getattr(m, 'raw_content', None)
    # Type-guard: raw_content is typed Optional[list[dict]]; a non-list truthy
    # value (type violation, stray JSON string) must NOT be treated as
    # "content is covered" — that would under-count the message to ~0 tokens,
    # which is precisely the failure-mode this subsystem prevents.
    if not isinstance(rc, list) or not rc:
        mt += _cjk_aware_tokens(content)
    # Reasoning content (DeepSeek reasoner) — separate field sent on wire alongside content.
    # NOT covered by raw_content; this is a parallel attribute that always needs counting.
    reasoning_attr = getattr(m, 'reasoning_content', None)
    if reasoning_attr:
        mt += _cjk_aware_tokens(reasoning_attr if isinstance(reasoning_attr, str) else str(reasoning_attr))
    # Tool calls (JSON args are programming text — chars//3 is adequate)
    tc = getattr(m, 'tool_calls', None)
    if tc:
        try:
            for t in tc:
                args = t.get("args", t.get("function", {}).get("arguments", ""))
                if isinstance(args, dict):
                    mt += len(json.dumps(args, ensure_ascii=False, default=str)) // 3 + 1
                elif isinstance(args, str):
                    mt += len(args) // 3 + 1
                elif args:
                    mt += len(str(args)) // 3 + 1
                name = t.get("name", t.get("function", {}).get("name", ""))
                if name:
                    mt += (len(name) + 10) // 3 + 1
        except Exception:
            mt += len(tc) * 100
    # Raw content blocks (Anthropic/zai-native tool payloads), counted via the
    # wire-block token registry — the single source of truth for which block
    # types this subsystem recognises.  rc is already fetched above for the
    # content-skip check.
    if rc:
        for block in rc:
            if not isinstance(block, dict):
                continue
            # Generic text pre-pass: counts plain text blocks and any inline
            # ``text`` field regardless of block type (harmless when absent).
            text = block.get('text', '')
            if text:
                mt += _cjk_aware_tokens(text)
            btype = block.get('type')
            if btype is not None and not isinstance(btype, str):
                # Malformed block (client-supplied raw_content can carry a
                # non-string type).  Normalize to str so dict lookup and the
                # warned-set below never hit an unhashable key — the pre-registry
                # ``==`` chain tolerated these, so must we.
                btype = str(btype)
            tokenizer = _WIRE_BLOCK_TOKENIZERS.get(btype) if btype else None
            if tokenizer is None and not btype:
                # Gemini ``parts`` carry the type as a top-level key, not a
                # ``type`` field — re-route them via content-key inference.
                for _marker, _fn in _WIRE_CONTENT_KEY_MARKERS.items():
                    if _marker in block:
                        tokenizer = _fn
                        break
            if tokenizer is not None:
                mt += tokenizer(block)
            elif btype in (None, '', 'text'):
                # 'text' blocks and blocks without a type whose payload is covered
                # by the generic text pre-pass above — these are safe to skip.
                # BUT: Gemini untyped parts (inlineData, fileData, executableCode,
                # etc.) that have NO 'text' field and matched no content-key marker
                # are NOT covered by any pre-pass — they reach here as btype=None
                # and silently produce 0 tokens. Count them wholesale as fail-safe.
                if btype is None and not text and not any(
                    _m in block for _m in _WIRE_CONTENT_KEY_MARKERS
                ):
                    mt += _count_block_wholesale(block)
                    _warn_unknown_block_type("<untyped-gemini-part>")
            else:
                # Unknown wire block type — fail-safe toward OVER-counting so a
                # new provider block type can never silently trigger a context
                # overflow (the exact failure this subsystem prevents).  Drift is
                # surfaced via a one-time-per-type warning.
                mt += _count_block_wholesale(block)
                _warn_unknown_block_type(btype)
    # Images (provider-cap flat estimate — see _IMAGE_BLOCK_TOKEN_ESTIMATE docstring)
    images = getattr(m, 'images', None)
    if images:
        mt += len(images) * _IMAGE_BLOCK_TOKEN_ESTIMATE

    # Cache on the message object for the turn lifetime (only for cacheable objects).
    if _can_cache:
        m._msg_token_estimate = mt
    return mt


def estimate_tokens_from_msgs(messages: list) -> int:
    """Estimate total token count from a list of LLMMessage objects.

    Counts content (CJK-aware) + tool_calls JSON args + raw_content blocks
    + images (provider-cap flat estimate via ``_IMAGE_BLOCK_TOKEN_ESTIMATE``).
    Uses per-message caching (``_msg_token_estimate``) so repeated calls in the
    same turn skip re-counting unchanged messages.
    """
    return sum(_estimate_single_message_tokens(m) for m in messages)


_tool_schema_token_cache: dict[int, int] = {}
"""Bounded ``id(tool_schemas)`` → token-count cache.

The no-filter path (``tool_registry.AGENT_TOOL_SCHEMAS``) always returns the
*same* list object — 100% permanent cache hit.  The lang_filter path returns a
*fresh* list each call, but within a single turn all call sites pass the same
object.  A small bounded cache eliminates repeated ``json.dumps`` of the
(identical or same-reference) schemas.

The id()-reuse risk (GC → same address → different content) is negligible:
tool schemas are loaded once and never mutated, so any reuse produces the same
token count.  Worst-case mismatch is one turn with a slightly stale count.
"""


def estimate_tokens_from_tool_schemas(tool_schemas: Optional[list]) -> int:
    """Estimate tokens consumed by serialised tool/function schemas.

    OpenAI-compatible and Ollama chat APIs serialise the ``tools`` array (name,
    description, JSON-schema parameters) into the model prompt, so these tokens
    count against the context window even though they are not chat messages.
    Omitting them under-counts the real prompt size — fatal on small local
    windows (e.g. Ollama num_ctx=8192) where a full prompt leaves zero
    generation budget (done_reason='length', eval_count=1).
    """
    if not tool_schemas:
        return 0
    # Bounded id()-keyed cache: same list object → skip json.dumps.
    _cache_id = id(tool_schemas)
    _cached = _tool_schema_token_cache.get(_cache_id)
    if _cached is not None:
        return _cached
    try:
        _chars = len(json.dumps(tool_schemas, ensure_ascii=False, default=str))
    except Exception:
        _chars = sum(len(str(s)) for s in tool_schemas)
    result = int(_chars / CHARS_PER_TOKEN) + 1
    if len(_tool_schema_token_cache) >= 8:
        _tool_schema_token_cache.clear()
    _tool_schema_token_cache[_cache_id] = result
    return result


def context_message_cap(ctx_limit: int, safety_margin: int,
                        tool_schemas: Optional[list] = None,
                        tool_tokens: Optional[int] = None) -> int:
    """Max prompt-token budget for chat messages, reserving room for output.

    Subtracts (a) an output reserve so the prompt never fills the whole window —
    critical for small local models like Ollama (num_ctx=8192), where a prompt
    that consumes the entire window leaves 0 tokens to generate, so the model
    emits exactly one token with done_reason='length'; and (b) the tokens used
    by serialised tool schemas, which are sent alongside the messages.

    The reserve is ``max(safety_margin, min(4096, ctx_limit // 5))`` so small
    windows reserve a meaningful slice (8192 -> ~1638) while large cloud windows
    cap the reserve at 4096 (negligible vs. their size).

    Pass ``tool_tokens`` to skip re-serialization of tool schemas (caller has
    already computed the token count).  When omitted, falls back to
    ``estimate_tokens_from_tool_schemas(tool_schemas)``.
    """
    _output_reserve = max(safety_margin, min(4096, ctx_limit // 5))
    if tool_tokens is None:
        _tool_tokens = estimate_tokens_from_tool_schemas(tool_schemas)
    else:
        _tool_tokens = tool_tokens
    return max(512, ctx_limit - _output_reserve - _tool_tokens)


def preemptive_trim(
    messages: list,
    max_tokens: int = MAX_SAFE_TOKENS,
    preserve_last: int = 2,
    tag: str = "PREEMPTIVE_TRIM",
) -> list:
    """Preemptively trim conversation history to stay within token limits.

    Preserves system prompt (first message) + last N messages,
    decrementing N until under cap. Returns trimmed list (or original if under limit).

    Uses CJK-aware estimation (``estimate_tokens_from_msgs``) so CJK-heavy
    conversations are not underestimated and do not provoke HTTP 400.
    Prefer ``_evict_consumed_tool_results`` (context-smart) over this blunt
    front-trim when the goal is gentle eviction.

    Args:
        messages: List of LLMMessage or dict message objects.
        max_tokens: Maximum allowed estimated tokens (default: MAX_SAFE_TOKENS).
        preserve_last: Initial number of recent messages to preserve (default: 2).
        tag: Log prefix for debugging (e.g. "DESIGN_CHAT_PREEMPTIVE_TRIM").
    """
    if not messages:
        return messages

    _est_tokens = estimate_tokens_from_msgs(messages)
    if _est_tokens <= max_tokens:
        return messages

    # Progressive trim: keep first (system) + last N messages
    _n = preserve_last
    while _n >= 0 and len(messages) > 1:
        _raw = messages[:1] + messages[max(1, len(messages) - _n - 1):]
        _kept_est = estimate_tokens_from_msgs(_raw)
        if _kept_est <= max_tokens:
            logger.warning(
                "[%s] %d->%d estimated tokens (%d->%d messages, preserve_last=%d->%d)",
                tag, _est_tokens, _kept_est, len(messages), len(_raw),
                preserve_last, _n,
            )
            return _raw
        if _n == 0:
            # Still over limit with system + last 1 message → fall through to system-only fallback
            break
        _n -= 1

    # Last resort: keep system message + last user/tool-result message.
    # messages[:1] alone drops the current user turn entirely, which causes
    # "messages must contain a user turn" errors on some providers and always
    # loses the user's actual request.  Including the last message gives the API
    # a complete (if oversized) request; the actual API limit and overflow
    # backstop handle final enforcement.
    # Guard against duplicate when len(messages)==1 (messages[:1] is msg[-1:]).
    _fallback = messages[:1] if len(messages) <= 1 else messages[:1] + messages[-1:]
    _fallback_est = estimate_tokens_from_msgs(_fallback)
    logger.warning(
        "[%s] last resort: %d->%d tokens (%d->%d messages)",
        tag, _est_tokens, _fallback_est,
        len(messages), len(_fallback),
    )
    return _fallback
