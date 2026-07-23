"""
Symbol search for asicode Agent (tree-sitter + AST hybrid).

Python files: tree-sitter based with AST fallback.
Other languages (JS/TS/Java/Go/Rust/…): language provider patterns + ripgrep.

Public API
----------
SymbolSearcher(repo_root)
  .find_symbol(name, *, kind, search_path)       -> List[SymbolDef]
  .find_references(name, *, search_path)          -> List[SymbolRef]
  .get_symbol_info(name, *, file_path, kind, defs) -> Optional[dict]
"""
from __future__ import annotations

import ast
import difflib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..languages import (
    LanguageId,  # S8 fix: missing module-level import
    LanguageRegistry,
)
from ._shared_utils import (
    _WALK_CACHE_TTL,
)
from ._shared_utils import (
    _walk_py_files as _shared_walk_py_files,
)
from ._shared_utils import (
    _walk_ts_js_files as _shared_walk_ts_js_files,
)
from .config.thresholds import config as _cfg
from .rag_configs import CodeTokenizer
from .rag_searcher import _bm25_score as _bm25
# Module-level lazy tokenizer singleton — avoids re-constructing
# CodeTokenizer (which compiles internal regex-alternative scanners)
# on every find_references call (a hot path).
_TOKENIZER: Any = None

# ── Tree-sitter availability ─────────────────────────────────────────────
try:
    from ..languages.tree_sitter_utils import (
        _LANG_MODULE_MAP as _TS_LANG_MODULE_MAP,
    )
    from ..languages.tree_sitter_utils import (
        find_all_symbols as _ts_find_all_symbols,
    )
    from ..languages.tree_sitter_utils import (
        get_available_languages as _ts_available_languages,
    )
    from ..languages.tree_sitter_utils import (  # type: ignore
        get_node_text as _ts_get_node_text,
    )
    from ..languages.tree_sitter_utils import (
        parse_to_tree as _ts_parse_to_tree,
    )
    _HAS_TS = True
except ImportError:
    _HAS_TS = False
    _TS_LANG_MODULE_MAP = {}

logger = logging.getLogger(__name__)

# Per-process dedup for the "grammar not installed" warning. Emitted at most
# once per language so the hot find_symbol path never spams the log. See
# _warn_missing_grammar — fires only for languages that are tree-sitter
# supported, have their grammar missing, AND expose no regex fallback (the
# "silent zero-results" trap introduced when the CSS regex path was retired
# in favor of the authoritative AST path).
_warned_missing_grammar: set[str] = set()

# Max files to AST-scan before stopping (avoids very large repos).
# Passed into the shared walkers (._shared_utils._walk_*_files) which apply
# the cap INSIDE the walk loop so a huge node_modules (tens of thousands of
# .js files) can't OOM or stall the walk before the caller's own
# SEARCH_RESULTS_CAP check runs.
_MAX_PY_FILES = _cfg.counts.SYMBOL_MAX_PY_FILES
_MAX_TS_FILES = _cfg.counts.SYMBOL_MAX_TS_FILES

# _WALK_CACHE_TTL is re-exported from ._shared_utils (shared with call_graph).
# _time is still needed for the non-Python index TTL check below.
import time as _time

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolDef:
    file: str
    line: int
    kind: str   # function | async_function | method | class | variable | import
    name: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    bases: Optional[list[str]] = None        # class base classes
    methods: Optional[list[str]] = None      # class method names
    decorators: Optional[list[str]] = None
    end_line: Optional[int] = None           # 1-indexed inclusive end (from AST end_lineno)
    parent_class: Optional[str] = None      # set when symbol is a method inside a class
    file_mtime: Optional[float] = None      # os.stat(file).st_mtime at resolve time — staleness guard


@dataclass
class SymbolRef:
    file: str
    line: int
    col: int
    context: str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_definition_line(file_path: str, line: str, name: str) -> bool:
    """Check whether *line* defines *name* using language-aware patterns.

    Uses the LanguageProvider's ``get_symbol_patterns()`` for the file's
    language — covers Python (def/async def/class), JS/TS (function/const/class),
    Go (func), Rust (fn), Kotlin (fun), Java, and other registered languages.
    Falls back to generic patterns when no provider or patterns are available.
    """
    stripped = line.lstrip()
    escaped = re.escape(name)

    # ── Primary: language-provider patterns (language-agnostic) ──────────
    try:
        registry = LanguageRegistry.instance()
        provider = registry.get(file_path)
        if provider is not None:
            patterns = provider.get_symbol_patterns(kind="any")
            if patterns:
                for sp in patterns:
                    pattern = sp.regex.replace("{name}", escaped)
                    if re.match(pattern, stripped):
                        return True
                # Provider had patterns but none matched — definitively not a definition
                return False
    except Exception:
        pass  # fall through to heuristic fallback

    # ── Fallback: generic patterns for unrecognised providers ────────────
    return bool(
        stripped.startswith(f"def {name}")
        or stripped.startswith(f"async def {name}")
        or stripped.startswith(f"class {name}")
        or re.match(rf"^(function\s+{escaped}|const\s+{escaped}\s*=)", stripped)
    )
def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (SyntaxError, TypeError, AttributeError):
        return ""


def _get_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a human-readable signature string."""
    args = node.args
    parts: list[str] = []

    # positional args
    num_no_default = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        annotation = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        if i < num_no_default:
            parts.append(f"{arg.arg}{annotation}")
        else:
            default = _unparse(args.defaults[i - num_no_default])
            parts.append(f"{arg.arg}{annotation}={default}")

    # *args
    if args.vararg:
        ann = f": {_unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only
    for i, arg in enumerate(args.kwonlyargs):
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        kd = args.kw_defaults[i]
        default = f"={_unparse(kd)}" if kd is not None else ""
        parts.append(f"{arg.arg}{ann}{default}")

    # **kwargs
    if args.kwarg:
        ann = f": {_unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    ret = f" -> {_unparse(node.returns)}" if node.returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)}){ret}"


def _build_ts_function_signature(fn: Any) -> str:
    """Build a human-readable signature from an IRFunction."""
    params = []
    for p in getattr(fn, "params", []):
        s = p.name
        if p.type_ref:
            s += f": {p.type_ref.name}"
        if p.has_default:
            s += " = ..."
        if p.is_rest:
            s = f"...{s}"
        params.append(s)
    ret = ""
    if fn.return_type:
        ret = f": {fn.return_type.name}"
    prefix = "async function" if fn.is_async else "function"
    return f"{prefix} {fn.name}({', '.join(params)}){ret}"


def _build_ts_method_signature(class_name: str, method: Any) -> str:
    """Build a human-readable signature from an IRMethod."""
    params = []
    for p in getattr(method, "params", []):
        s = p.name
        if p.type_ref:
            s += f": {p.type_ref.name}"
        params.append(s)
    ret = ""
    if method.return_type:
        ret = f": {method.return_type.name}"
    prefix = "async " if method.is_async else ""
    static = "static " if method.is_static else ""
    return f"{class_name}.{static}{prefix}{method.name}({', '.join(params)}){ret}"


def _walk_py_files(root: Path) -> list[Path]:
    """Walk repo, returning .py files, skipping hidden/vendor dirs.

    Thin wrapper over the shared walker (._shared_utils._walk_py_files) so
    symbol_search and call_graph share one implementation + one process-global
    cache. Best-effort: a concurrent write is fine because callers tolerate a
    slightly-stale file list (missing files → "not found this round").
    """
    return _shared_walk_py_files(root, _MAX_PY_FILES)


def _walk_ts_js_files(root: Path) -> list[Path]:
    """Return TS/JS files under *root*, skipping hidden/vendor/node_modules.

    Thin wrapper over the shared walker so the file list is shared across
    calls and across consumers. The rglob is the dominant cost of
    find_symbol's TS/JS fallback path (~6s on this repo uncached).
    """
    return _shared_walk_ts_js_files(root, _MAX_TS_FILES)


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter helpers (Python symbol extraction)
# ─────────────────────────────────────────────────────────────────────────────

def _ts_build_py_function_signature(node, code_bytes: bytes) -> str:
    """Build function signature from a tree-sitter function_definition node."""
    try:
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        ret_node = node.child_by_field_name("return_type")
        fn_name = _ts_get_node_text(code_bytes, name_node) if name_node else ""
        parts: list[str] = []
        if params_node:
            for child in params_node.children:
                if child.type == "identifier":
                    parts.append(_ts_get_node_text(code_bytes, child))
                elif child.type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
                    n = child.child_by_field_name("name")
                    # Some Python tree-sitter grammars don't have "name" as a
                    # named field on typed_parameter — fall back to first child.
                    if n is None and child.children:
                        n = child.children[0]
                    t = child.child_by_field_name("type")
                    pname = _ts_get_node_text(code_bytes, n) if n else ""
                    ptype = f": {_ts_get_node_text(code_bytes, t)}" if t else ""
                    parts.append(f"{pname}{ptype}")
        ret = f" -> {_ts_get_node_text(code_bytes, ret_node)}" if ret_node else ""
        return f"def {fn_name}({', '.join(parts)}){ret}"
    except Exception:
        return ""


def _ts_extract_decorators(node, code_bytes: bytes) -> list[str]:
    """Extract decorator names from a decorated_definition or function node."""
    decs: list[str] = []
    try:
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    d_text = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    decs.append(d_text.lstrip("@").strip())
        else:
            # Function may have decorator_list child
            dec_list = node.child_by_field_name("decorator")
            if dec_list:
                for child in dec_list.children:
                    if child.type == "decorator":
                        d_text = _ts_get_node_text(code_bytes, child)
                        decs.append(d_text.lstrip("@").strip())
    except Exception:
        pass
    return decs


def _ts_extract_docstring(node, code_bytes: bytes) -> Optional[str]:
    """Extract docstring from a function/class tree-sitter node."""
    try:
        body = node.child_by_field_name("body")
        if body and body.children:
            first = body.children[0]
            if first.type == "expression_statement":
                expr = first.child_by_field_name("expression")
                if expr and expr.type == "string":
                    text = _ts_get_node_text(code_bytes, expr)
                    # Strip quotes
                    if text.startswith(('"""', "'''")):
                        text = text[3:-3]
                    elif text.startswith(("'", '"')):
                        text = text[1:-1]
                    return text[:150] or None
    except Exception:
        pass
    return None


def _ts_extract_class_bases(node, code_bytes: bytes) -> list[str]:
    """Extract base class names from a class_definition node."""
    bases: list[str] = []
    try:
        super_node = node.child_by_field_name("superclass")
        if super_node:
            for child in super_node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type in ("identifier", "attribute", "call"):
                            bases.append(_ts_get_node_text(code_bytes, arg))
                        elif arg.type == "comment":
                            continue
    except Exception:
        pass
    return bases


def _ts_collect_class_methods(node, code_bytes: bytes) -> list[str]:
    """Collect method names from a class_definition's body."""
    methods: list[str] = []
    try:
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        methods.append(_ts_get_node_text(code_bytes, name_node))
    except Exception:
        pass
    return methods


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SymbolSearcher:
    """
    Tree-sitter + AST hybrid symbol search for Python; rg-based for other languages.
    All paths are validated to stay within repo_root.
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        # ── per-file Python parse cache ────────────────────────────────────
        # Key: abs file path -> (mtime, full_symbol_map)
        # full_symbol_map: {name -> [SymbolDef, ...]} covering ALL definitions
        # (functions, classes, methods, constants) found in that file, including
        # parent_class scope info so name-filtered lookups reproduce the exact
        # results of a fresh _find_in_python call without re-parsing.
        self._py_file_cache: dict[str, tuple] = {}
        # ── non-Python persistent definition index (mtime-invalidated) ─────
        # Key: search_root str -> (fingerprint, {name -> [SymbolDef, ...]})
        # Built once per (root, file-set mtime fingerprint); reused across
        # find_symbol calls for non-Python languages.
        self._nonpy_index_cache: dict[str, tuple] = {}
        # ── TS/JS per-file module analysis cache ───────────────────────────
        # Key: abs file path -> (mtime, {name -> [SymbolDef]}) covering every
        # symbol kind TSSemanticTracer exposes. analyze_core costs ~15ms/file
        # and is pure function of content, so caching it removes the dominant
        # cost of repeated find_symbol over TS/JS repos.
        self._ts_file_cache: dict[str, tuple] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def fuzzy_find_symbol(
        self,
        name: str,
        *,
        kind: str = "any",
        search_path: Optional[str] = None,
    ) -> Optional[SymbolDef]:
        """
        Fuzzy match symbol names using difflib.
        """
        candidates = self.find_symbol(name, kind=kind, search_path=search_path)

        if not candidates:
            return None

        names = [c.name for c in candidates]

        matches = difflib.get_close_matches(
            name,
            names,
            n=1,
            cutoff=0.6
        )

        if not matches:
            return None

        for c in candidates:
            if c.name == matches[0]:
                return c

        return None


    def find_symbol(
        self,
        name: str,
        *,
        kind: str = "any",
        search_path: Optional[str] = None,
        prefer_files: Optional[list[str]] = None,
    ) -> list[SymbolDef]:
        """
        Find definition(s) of a symbol by name.

        kind: "function" | "class" | "variable" | "any"
        prefer_files: when provided, results in these files are ranked first.
            This disambiguates when the same symbol exists in multiple files.
        Returns at most 20 results.
        """
        if not name:
            return []
        root = self._resolve_search_root(search_path)
        if root is None:
            return []

        # Handle dotted names like "ClassName.method_name" or "Outer.Inner.method"
        # Keep the FULL class chain as parent_class so nested classes are supported.
        # e.g. "A.B.method" → search_name="method", parent_class="A.B"
        _parts = name.split(".") if "." in name else [name]
        search_name = _parts[-1]
        parent_class = ".".join(_parts[:-1]) if len(_parts) >= 2 else None

        results: list[SymbolDef] = []

        # ── Dotted name resolution: find parent class first, then search in its file ──
        if parent_class and search_path:
            # search_path provided with dotted name — do class-scoped search within
            # that file so we return the method belonging to the correct class.
            # Without this, bare `__init__` returns the FIRST __init__ in the file,
            # which may belong to a different class earlier in the file.
            # Guard: root may be a directory (e.g. "playground/galaga"), not a file.
            # _find_in_python calls read_text() which raises IsADirectoryError.
            if root.is_file() and LanguageId.from_path(str(root)) == LanguageId.PYTHON:
                parent_results = self._find_in_python_cached(root, search_name, kind,
                                                             parent_class=parent_class)
            elif root.is_file() and LanguageId.from_path(str(root)) == LanguageId.GO:
                parent_results = self._find_in_go(root, search_name, kind,
                                                  parent_class=parent_class)
            else:
                parent_results = []
            if parent_results:
                return parent_results[:20]
            # Fall through to whole-file scan if not found in that class

        if parent_class and not search_path:
            parent_defs = self.find_symbol(parent_class, kind="class")
            if parent_defs:
                # Search for the method only in the parent class's file
                parent_file = Path(parent_defs[0].file) if parent_defs[0].file else None
                # Resolve to absolute path so _find_in_python can call relative_to(repo_root)
                if parent_file and not parent_file.is_absolute():
                    parent_file = self.repo_root / parent_file
                if parent_file and parent_file.exists():
                    parent_results = self._find_in_python_cached(parent_file, search_name, kind,
                                                                parent_class=parent_class)
                    if parent_results:
                        return parent_results[:20]

                    # @dataclass fallback: if searching for __init__ in a @dataclass,
                    # there's no explicit __init__ — return the class body instead.
                    if search_name == "__init__" and parent_defs[0].kind == "class":
                        _is_dataclass = self._is_dataclass(parent_file, parent_class)
                        if _is_dataclass:
                            logger.info(
                                "find_symbol: %s.__init__ not found — @dataclass detected, "
                                "returning class definition instead",
                                parent_class,
                            )
                            # Return the class itself so modify_symbol can edit its body
                            return [parent_defs[0]]

        # ── Python AST scan ──────────────────────────────────────────────────
        py_files = [root] if root.is_file() and LanguageId.from_path(str(root)) == LanguageId.PYTHON else _walk_py_files(root)
        for pf in py_files:
            results.extend(self._find_in_python_cached(pf, search_name, kind))
            if len(results) >= _cfg.counts.SEARCH_RESULTS_CAP:
                break

        # ── TS/JS rich scan via TSSemanticTracer ─────────────────────────────
        try:
            from config import MULTILANG_SYMBOL_SEARCH as _ML_SYM
        except Exception:
            _ML_SYM = True  # non-critical — never block execution
        if _ML_SYM and not results:
            if root.is_file() and LanguageId.from_path(str(root)) in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
                results.extend(self._find_in_ts_js(root, search_name, kind))
            elif root.is_dir():
                _ts_count = 0
                for _tf in _walk_ts_js_files(root):
                    results.extend(self._find_in_ts_js(_tf, search_name, kind))
                    _ts_count += 1
                    if len(results) >= _cfg.counts.SEARCH_RESULTS_CAP or _ts_count >= self._MAX_TS_FILES:
                        break

        # ── Provider-aware search for registered languages (persistent index)
        if (not results or kind == "any") and kind != "variable":
            registry = LanguageRegistry.instance()
            has_nonpy_provider = any(
                p.language_id().value not in ("python", "typescript", "javascript")
                for p in set(registry._providers.values())
            )
            if has_nonpy_provider:
                # The persistent index already aggregates all non-Python
                # providers in one rg pass; filter to this name/kind.
                _idx = self._nonpy_index_for(root)
                for d in _idx.get(search_name, []):
                    if kind in ("function", "method", "any") and d.kind in (
                        "function", "async_function", "method",
                    ):
                        results.append(d)
                    elif kind in ("class", "any") and d.kind in (
                        # All type/aggregate-like declarations across languages.
                        # "class"-group covers: OOP classes, interfaces, type
                        # aliases, enums, CSS selectors/custom properties, plus
                        # the struct/trait/record/module/protocol kinds emitted
                        # by the Rust/C#/Ruby/PHP/Swift providers & AST path.
                        # "namespace" covers Ruby modules / AST-normalized
                        # module-kind symbols.
                        "class", "interface", "type", "enum",
                        "struct", "trait", "record", "module", "protocol",
                        "namespace",
                        "css_class", "css_id", "css_variable",
                    ):
                        results.append(d)
                    elif kind == "any":
                        results.append(d)

        # ── Legacy rg fallback (_find_in_other_langs) — RETIRED ──────────────
        # Every non-Python language now has either a tree-sitter binding
        # (AST path in _nonpy_index_for) or a registered provider whose
        # get_symbol_patterns feeds the same index. The hardcoded rg+regex
        # fallback was pure redundancy (and the source of the leading "-"/"#"
        # shell-arg trap that originally motivated the -e flag). Removed.

        # Deduplicate by (file, line)
        seen: set = set()
        unique: list[SymbolDef] = []
        for d in results:
            key = (d.file, d.line)
            if key not in seen:
                seen.add(key)
                unique.append(d)

        # ── Disambiguation: rank results when same symbol in multiple files ──
        if len(unique) > 1 and prefer_files:
            unique = self._rank_symbol_results(unique, prefer_files)

        return unique[:20]

    @staticmethod
    def _rank_symbol_results(
        results: list[SymbolDef],
        prefer_files: list[str],
    ) -> list[SymbolDef]:
        """Rank symbol results by file preference and structural heuristics.

        Scoring (per result):
        - Tier 1: File match (+4.0) — result.file is in prefer_files
        - Tier 2: Directory proximity (+2.0) — same directory as a prefer_file
        - Tier 3: Test penalty (-2.0) — test files deprioritized
        - Tier 4: Definition kind (+1.0) — class/function preferred over variable
        """

        _prefer_set = set(prefer_files)
        _prefer_basenames = {os.path.basename(f): f for f in prefer_files}
        _prefer_dirs = {os.path.dirname(f) for f in prefer_files if f}

        _test_patterns = ('/test', '_test', '/tests/', 'test_', '/fixtures/')
        _strong_kinds = {'class', 'function', 'async_function', 'method'}

        scores: dict[int, float] = {}
        for i, d in enumerate(results):
            score = 0.0

            # Tier 1: exact file match
            if d.file in _prefer_set:
                score += 4.0
            elif os.path.basename(d.file) in _prefer_basenames:
                score += 3.0  # basename match (slightly lower)

            # Tier 2: directory proximity
            if d.file and os.path.dirname(d.file) in _prefer_dirs:
                score += 2.0

            # Tier 3: test penalty
            if d.file:
                _lower = d.file.lower()
                if any(tp in _lower for tp in _test_patterns):
                    score -= 2.0

            # Tier 4: definition kind preference
            if d.kind in _strong_kinds:
                score += 1.0

            scores[i] = score

        # Stable sort by score descending
        ranked = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        return [results[i] for i in ranked]

    def find_references(
        self,
        name: str,
        *,
        search_path: Optional[str] = None,
        include_definitions: bool = False,
    ) -> list[SymbolRef]:
        """
        Find all usages of a symbol (using rg word-boundary search).
        Returns at most 40 results.
        """
        if not name:
            return []
        root = self._resolve_search_root(search_path) or self.repo_root

        pattern = rf"\b{re.escape(name)}\b"
        try:
            cmd = [
                "rg", "--no-heading", "--line-number",
                "-m", "5", pattern, str(root),
            ]
            proc = subprocess.run(
                cmd, cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=10,
            )
            refs: list[SymbolRef] = []
            for line in (proc.stdout or "").splitlines()[:80]:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                try:
                    rel = str(Path(parts[0]).relative_to(self.repo_root))
                    lineno = int(parts[1])
                    ctx = parts[2].strip()
                    stripped = ctx.lstrip()
                    if not include_definitions and _is_definition_line(parts[0], stripped, name):
                        continue
                    col = ctx.find(name)
                    refs.append(SymbolRef(file=rel, line=lineno, col=max(col, 0), context=ctx[:120]))
                except (AttributeError, TypeError, ValueError):
                    pass

            # BM25 ranking: sort by relevance to the symbol name before capping.
            # Treats each reference's file+context as a pseudo-document and scores
            # against the symbol name — so references with richer surrounding context
            # (more identifier tokens matching the name) rank higher.
            if len(refs) > 1:
                from collections import Counter
                global _TOKENIZER
                if _TOKENIZER is None:
                    _TOKENIZER = CodeTokenizer()
                _tok = _TOKENIZER
                _qtokens = _tok.tokenize(name)
                if _qtokens:
                    _docs = [f"{r.file}:{r.context}" for r in refs]
                    _tokenized = [_tok.tokenize(d) for d in _docs]
                    _doc_tc: list[dict[str, int]] = [dict(Counter(t)) for t in _tokenized]
                    _doc_lens = [len(t) for t in _tokenized]
                    _n = len(refs)
                    _avgdl = sum(_doc_lens) / _n
                    _df: dict[str, int] = {}
                    for qt in _qtokens:
                        _df[qt] = sum(1 for tc in _doc_tc if qt in tc)
                    _scores = [
                        _bm25(_qtokens, _doc_tc[i], _doc_lens[i], _df, _n, _avgdl)
                        for i in range(_n)
                    ]
                    # Sort by score only — SymbolRef has no ordering, so a
                    # plain reverse sort would compare SymbolRef on score ties
                    # and raise TypeError ('<' not supported).
                    refs = [
                        r for _, r in sorted(
                            zip(_scores, refs, strict=False), key=lambda x: x[0], reverse=True
                        )
                    ]
            return refs[:40]
        except FileNotFoundError:
            # rg not installed — graceful skip
            return []
        except Exception as e:
            logger.warning("find_references failed: %s", e)
            return []

    def get_symbol_info(
        self,
        name: str,
        *,
        file_path: Optional[str] = None,
        kind: str = "any",
        defs: Optional[list[SymbolDef]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Returns symbol metadata including definitions, references, and callers.
        Includes signature, docstring, bases/methods (for classes), subclasses, reference count.

        defs: pre-fetched find_symbol results — pass them to skip the internal
            lookup and guarantee enrichment targets the same definitions the
            caller already displayed.
        """
        if defs is None:
            defs = self.find_symbol(name, kind=kind, search_path=file_path)
        if not defs:
            return None

        sym = defs[0]
        info: dict[str, Any] = {
            "name": sym.name,
            "kind": sym.kind,
            "file": sym.file,
            "line": sym.line,
        }
        if sym.signature:
            info["signature"] = sym.signature
        if sym.docstring:
            info["docstring"] = sym.docstring
        if sym.bases is not None:
            info["bases"] = sym.bases
        if sym.methods is not None:
            info["methods"] = sym.methods
        if sym.decorators:
            info["decorators"] = sym.decorators

        if sym.kind == "class":
            subs = self._find_subclasses(name)
            if subs:
                info["subclasses"] = subs[:8]

        # Reference summary
        refs = self.find_references(name, search_path=file_path)
        info["reference_count"] = len(refs)
        if refs:
            ref_files = list(dict.fromkeys(r.file for r in refs))[:5]
            info["referenced_in"] = ref_files
            info["sample_references"] = [
                {"file": r.file, "line": r.line, "context": r.context[:400]}
                for r in refs[:4]
            ]

        if len(defs) > 1:
            info["other_definitions"] = [
                {"file": d.file, "line": d.line, "kind": d.kind}
                for d in defs[1:5]
            ]

        # Add read guidance for the agent
        if "file" in info and "line" in info:
            info["read_guidance"] = (
                f"To examine this symbol, use bash (cat -n) on "
                f"`{info['file']}` starting around line {max(1, info['line'] - 10)}. "
                "Read 30-50 lines around the definition to understand context."
            )
        else:
            info["read_guidance"] = "Unable to generate read guidance due to missing file or line information."

        return info

    def get_file_outline(self, file_path: str) -> list[SymbolDef]:
        """Return all top-level symbols (classes, functions, constants) in a single file.

        For Python files: uses AST for precise results.
        For other languages: falls back to ripgrep patterns.
        Returns symbols sorted by line number, capped at 120 entries.
        """
        try:
            p = (self.repo_root / file_path).resolve()
            if not p.is_relative_to(self.repo_root):
                return []
            if not p.is_file():
                return []
            rel = str(p.relative_to(self.repo_root))
        except Exception:
            return []  # non-critical — never block execution

        if LanguageId.from_path(str(p)) == LanguageId.PYTHON:
            return self._outline_python(p, rel)
        elif LanguageId.from_path(str(p)) in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
            try:
                from config import MULTILANG_OUTLINE as _ML_OL
            except Exception:
                _ML_OL = True  # non-critical — never block execution
            if _ML_OL:
                _ts_outline = self._outline_ts_js(p, rel)
                if _ts_outline:
                    return _ts_outline
            _ast_outline = self._outline_treesitter(p, rel)
            if _ast_outline:
                return _ast_outline
            return self._outline_ripgrep(p, rel)
        else:
            # AST-first: tree-sitter gives an accurate (start, end) per symbol
            # and handles modifiers/annotations structurally. Fall back to the
            # provider-regex rg path only when tree-sitter is unavailable or the
            # grammar is not installed (e.g. Kotlin before tree_sitter_kotlin).
            _ast_outline = self._outline_treesitter(p, rel)
            if _ast_outline:
                return _ast_outline
            return self._outline_ripgrep(p, rel)

    def _outline_python(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """Tree-sitter-based outline for a single Python file (AST fallback)."""
        results: list[SymbolDef] = []
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (AttributeError, TypeError):
            return results
        code_bytes = source.encode("utf-8")

        # ── Primary: tree-sitter ─────────────────────────────────────────
        if _HAS_TS:
            try:
                tree = _ts_parse_to_tree(source, "python")
                if tree is not None:

                    def _walk_outline(node, parent_class: str = "") -> None:
                        """Walk tree-sitter tree and collect outline entries."""
                        if node.type == "function_definition":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                fn_name = _ts_get_node_text(code_bytes, name_node)
                                if not parent_class:  # only top-level
                                    sig = _ts_build_py_function_signature(node, code_bytes)
                                    doc = _ts_extract_docstring(node, code_bytes)
                                    decs = _ts_extract_decorators(node, code_bytes)
                                    results.append(SymbolDef(
                                        file=rel, line=node.start_point.row + 1,
                                        kind="function", name=fn_name,
                                        signature=sig or None, docstring=doc,
                                        decorators=decs or None,
                                    ))
                            return  # stop descent for function bodies

                        elif node.type == "class_definition":
                            name_node = node.child_by_field_name("name")
                            if name_node:
                                cls_name = _ts_get_node_text(code_bytes, name_node)
                                bases = _ts_extract_class_bases(node, code_bytes) or None
                                methods = _ts_collect_class_methods(node, code_bytes) or None
                                doc = _ts_extract_docstring(node, code_bytes)
                                results.append(SymbolDef(
                                    file=rel, line=node.start_point.row + 1,
                                    kind="class", name=cls_name,
                                    bases=bases, methods=methods, docstring=doc,
                                ))
                            return  # don't descend into class body for outline

                        elif node.type == "decorated_definition":
                            # Unwrap: the real definition is inside
                            for child in node.children:
                                if child.type in ("function_definition", "class_definition"):
                                    _walk_outline(child, parent_class)
                            return

                        elif node.type == "expression_statement":
                            # Top-level assignment (constant)
                            if not parent_class:
                                # Python tree-sitter grammar: expression_statement has
                                # no "expression" named field — child_by_field_name
                                # always returns None. Use children[0] instead.
                                expr = node.children[0] if node.children else None
                                if expr and expr.type == "assignment":
                                    left = expr.child_by_field_name("left")
                                    right = expr.child_by_field_name("right")
                                    if left and left.type == "identifier":
                                        name = _ts_get_node_text(code_bytes, left)
                                        val = _ts_get_node_text(code_bytes, right)[:60] if right else ""
                                        results.append(SymbolDef(
                                            file=rel, line=node.start_point.row + 1,
                                            kind="constant", name=name,
                                            signature=f"{name} = {val}" if val else None,
                                        ))
                            return

                        # Recurse into children (skip function/class bodies)
                        for child in node.children:
                            _walk_outline(child, parent_class)

                    _walk_outline(tree.root_node)
                    results.sort(key=lambda s: s.line)
                    return results
            except Exception:
                pass  # fall through to AST

        # ── Fallback: AST ────────────────────────────────────────────────
        try:
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, TypeError, AttributeError):
            return results

        for node in ast.iter_child_nodes(tree):
            # ── functions ──
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = _get_function_signature(node)
                doc = ast.get_docstring(node) or None
                decs = [_unparse(d) for d in node.decorator_list] or None
                nk = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                results.append(SymbolDef(
                    file=rel, line=node.lineno, kind=nk, name=node.name,
                    signature=sig, docstring=doc, decorators=decs,
                ))

            # ── classes ──
            elif isinstance(node, ast.ClassDef):
                bases = [_unparse(b) for b in node.bases] or None
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                doc = ast.get_docstring(node) or None
                results.append(SymbolDef(
                    file=rel, line=node.lineno, kind="class", name=node.name,
                    bases=bases, methods=methods or None, docstring=doc,
                ))

            # ── top-level assignments (constants) ──
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        val = _unparse(node.value)[:60]
                        results.append(SymbolDef(
                            file=rel, line=node.lineno, kind="constant",
                            name=target.id,
                            signature=f"{target.id} = {val}" if val else None,
                        ))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    ann = _unparse(node.annotation)
                    val = _unparse(node.value)[:40] if node.value else ""
                    sig = f"{node.target.id}: {ann}" + (f" = {val}" if val else "")
                    results.append(SymbolDef(
                        file=rel, line=node.lineno, kind="constant",
                        name=node.target.id, signature=sig,
                    ))

        results.sort(key=lambda s: s.line)
        return results

    def _outline_treesitter(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """Tree-sitter outline for non-Python files (Go/Java/Rust/Ruby/...).

        Primary path for any language whose tree-sitter binding is installed.
        Shares the same ``find_all_symbols`` extractor the cross-file index uses
        (``_index_via_treesitter``), so outline and index agree on the same
        symbol set — a single source of truth, no per-language regex drift.

        Unlike the rg path, the AST yields BOTH the start and the end line of
        each construct, so ``SymbolDef.end_line`` is populated (callers such as
        modify_symbol benefit from an exact extent instead of brace-balancing).

        Returns an empty list when tree-sitter is unavailable or the grammar is
        not installed, so the caller can transparently fall back to
        ``_outline_ripgrep``. Installing a grammar (e.g. ``tree_sitter_kotlin``)
        therefore enables outline for that language with no code change.
        """
        if not _HAS_TS:
            return []
        lang_id = LanguageId.from_path(str(file_path)).value
        # Non-Python only (Python has _outline_python). TS/JS keep their richer
        # TSSemanticTracer path via _outline_ts_js.
        if (
            lang_id not in _TS_LANG_MODULE_MAP
            or lang_id in ("python", "typescript", "javascript")
        ):
            return []
        if lang_id not in _ts_available_languages():
            return []  # grammar mapped but not installed → caller falls back
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            syms = _ts_find_all_symbols(content, lang_id)
        except Exception:
            return []  # non-critical — fall back to _outline_ripgrep
        src_lines = content.splitlines()
        results: list[SymbolDef] = []
        for name, kind, start_line, end_line in syms:
            sig = None
            if 0 < start_line <= len(src_lines):
                sig = src_lines[start_line - 1].strip() or None
            results.append(SymbolDef(
                file=rel, line=start_line, kind=kind, name=name,
                signature=sig, end_line=end_line,
            ))
        results.sort(key=lambda s: s.line)
        return results

    def _outline_ripgrep(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """Ripgrep-based outline for non-Python files (JS/TS/Go/Rust/etc.).

        Uses LanguageProvider.get_symbol_patterns() when available for the
        detected language, falling back to minimal hardcoded patterns for
        languages without a registered provider (e.g. Rust).
        """
        results: list[SymbolDef] = []
        seen: set = set()

        # Build patterns from LanguageProvider when possible
        patterns: list = []
        provider = LanguageRegistry.instance().get(str(file_path))
        if provider is not None:
            for sp in provider.get_symbol_patterns(kind="any"):
                # Convert {name} placeholder to capture group for outline mode.
                # Use the pattern's OWN name_capture (default \w+) — NOT a hardcoded
                # \w+ — so providers that capture broader names work here too: Lua
                # [\w.:]+ for dotted/colon methods (M.foo / Account:bar), CSS [-\w]+
                # for kebab-case. With a hardcoded \w+ the dotted form silently failed
                # to match (the whole regex aborted at the '.'), dropping the symbol.
                # This MUST mirror the repo-index substitution in _nonpy_index_for.
                outline_regex = sp.regex.replace("{name}", f"({sp.name_capture})")
                patterns.append((outline_regex, sp.kind))
        else:
            # Fallback for languages without a registered provider (Rust, etc.)
            patterns = [
                (r"^\s*(?:pub\s+)?fn\s+(\w+)", "function"),   # Rust
                (r"^\s*(?:pub\s+)?struct\s+(\w+)", "class"),   # Rust
                (r"^\s*(?:pub\s+)?enum\s+(\w+)", "enum"),      # Rust
            ]

        for pat, kind in patterns:
            try:
                # --with-filename is mandatory: with a single FILE argument, rg omits
                # the path prefix and emits "lineno:content", which would collapse the
                # 3-part split below (path:lineno:content) and silently drop every
                # match — see _index_via_treesitter (L1967) which uses the same flag.
                proc = subprocess.run(
                    ["rg", "--no-heading", "--with-filename", "--line-number", "-m", "50", pat, str(file_path)],
                    capture_output=True, text=True, timeout=5,
                )
                for line in (proc.stdout or "").splitlines():
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        continue
                    ctx = parts[2].strip()
                    # Prefer the pattern's own capture group — provider patterns
                    # capture the symbol name as (\w+) via {name} -> (\\w+) above.
                    # The generic heuristic below mishandles declarations where the
                    # name is NOT the last token (e.g. Go "type Server struct" would
                    # extract "struct"). Mirrors _index_via_treesitter's fallback.
                    rm = re.search(pat, ctx)
                    name = rm.group(1) if (rm and rm.groups()) else ""
                    if not name:
                        _h = ctx.split("(")[0].split("{")[0].rsplit(None, 1)[-1] if ctx else ""
                        m = re.search(r"(\w+)", _h)
                        name = m.group(1) if m else ctx[:30]
                    key = (lineno, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(SymbolDef(
                        file=rel, line=lineno, kind=kind, name=name,
                        signature=ctx,
                    ))
            except (AttributeError, TypeError):
                continue

        results.sort(key=lambda s: s.line)
        return results

    def _resolve_search_root(self, search_path: Optional[str]) -> Optional[Path]:
        if not search_path:
            return self.repo_root
        try:
            p = (self.repo_root / search_path).resolve()
            if p.is_relative_to(self.repo_root):
                return p
        except Exception:
            pass  # non-critical — never block execution
        return None

    # ── Python per-file symbol extraction + mtime cache ────────────────────
    # The cache stores, per file, a {name -> [SymbolDef]} map of ALL symbols
    # (functions/classes/methods/constants, with parent_class). This lets
    # find_symbol filter by name in O(1) instead of re-parsing the file on
    # every lookup. Invalidated by mtime.

    def _extract_all_python_symbols(
        self, file_path: Path, rel: str,
    ) -> dict[str, list[SymbolDef]]:
        """Extract ALL symbols from a Python file into a {name: [defs]} map.

        Reproduces the exact SymbolDef fields that _find_in_python emits for a
        single name, but collected for every name in one pass. This is the
        cache primitive: find_symbol then becomes a dict lookup + kind/parent
        filter instead of a full re-parse.
        """
        out: dict[str, list[SymbolDef]] = {}
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return out

        # ── Primary: tree-sitter (collect every symbol) ──────────────────
        if _HAS_TS:
            try:
                code_bytes = source.encode("utf-8")
                tree = _ts_parse_to_tree(source, "python")
                if tree is not None:
                    self._ts_collect_all(tree.root_node, code_bytes, rel, "", out)
                    return out
            except Exception:
                pass  # fall through to AST

        # ── Fallback: AST ──────────────────────────────────────────────────
        try:
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, TypeError, AttributeError):
            return out

        # Build an enclosing-class map once (AST-level O(n)) instead of the
        # O(n²) nested ast.walk that _find_in_python does per matched node.
        enclosing: dict[int, str] = {}  # id(node) -> class name

        def _record_enclosing(cls_node: ast.ClassDef, prefix: str) -> None:
            full = f"{prefix}.{cls_node.name}" if prefix else cls_node.name
            for child in cls_node.body:
                enclosing[id(child)] = full
                # recurse into nested classes
                for nc in ast.walk(child):
                    if isinstance(nc, ast.ClassDef) and nc is not child:
                        # only record direct children of nested class via separate pass
                        pass

        # Single-pass: walk and record direct enclosing class for top-level body items
        def _walk_body(body: list, parent_class: str) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    full = f"{parent_class}.{node.name}" if parent_class else node.name
                    # record the class itself
                    self._ast_add_class(out, node, rel, parent_class)
                    # record its direct members under `full`
                    for child in node.body:
                        enclosing[id(child)] = full
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._ast_add_func(out, child, rel, full)
                        elif isinstance(child, ast.Assign):
                            self._ast_add_assign(out, child, rel, full)
                        elif isinstance(child, ast.AnnAssign):
                            self._ast_add_annassign(out, child, rel, full)
                        elif isinstance(child, ast.ClassDef):
                            _walk_body([child], full)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._ast_add_func(out, node, rel, parent_class)
                elif isinstance(node, ast.Assign):
                    self._ast_add_assign(out, node, rel, parent_class)
                elif isinstance(node, ast.AnnAssign):
                    self._ast_add_annassign(out, node, rel, parent_class)

        _walk_body(tree.body, "")
        return out

    @staticmethod
    def _ast_add_func(
        out: dict[str, list[SymbolDef]], node: ast.AST, rel: str, parent_class: str,
    ) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        sig = _get_function_signature(node)
        doc = (ast.get_docstring(node) or "")[:150] or None
        decs = [_unparse(d) for d in node.decorator_list if d] or None
        nk = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        out.setdefault(node.name, []).append(SymbolDef(
            file=rel, line=node.lineno, kind=nk, name=node.name,
            signature=sig, docstring=doc, decorators=decs,
            end_line=getattr(node, "end_lineno", None),
            parent_class=parent_class or None,
        ))

    @staticmethod
    def _ast_add_class(
        out: dict[str, list[SymbolDef]], node: ast.ClassDef, rel: str, parent_class: str,
    ) -> None:
        bases = [_unparse(b) for b in node.bases] or None
        methods = [
            n.name for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        doc = (ast.get_docstring(node) or "")[:150] or None
        out.setdefault(node.name, []).append(SymbolDef(
            file=rel, line=node.lineno, kind="class", name=node.name,
            bases=bases, methods=methods[:25] or None, docstring=doc,
            end_line=getattr(node, "end_lineno", None),
            parent_class=parent_class or None,
        ))

    @staticmethod
    def _ast_add_assign(
        out: dict[str, list[SymbolDef]], node: ast.Assign, rel: str, parent_class: str,
    ) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                val = _unparse(node.value)[:80]
                out.setdefault(target.id, []).append(SymbolDef(
                    file=rel, line=node.lineno, kind="constant", name=target.id,
                    signature=f"{target.id} = {val}" if val else None,
                    end_line=getattr(node, "end_lineno", None),
                    parent_class=parent_class or None,
                ))

    @staticmethod
    def _ast_add_annassign(
        out: dict[str, list[SymbolDef]], node: ast.AnnAssign, rel: str, parent_class: str,
    ) -> None:
        if isinstance(node.target, ast.Name):
            ann = _unparse(node.annotation)
            val = _unparse(node.value)[:60] if node.value else ""
            sig = f"{node.target.id}: {ann}" + (f" = {val}" if val else "")
            out.setdefault(node.target.id, []).append(SymbolDef(
                file=rel, line=node.lineno, kind="constant", name=node.target.id,
                signature=sig,
                end_line=getattr(node, "end_lineno", None),
                parent_class=parent_class or None,
            ))

    def _ts_collect_all(
        self, node, code_bytes: bytes, rel: str, parent_class: str,
        out: dict[str, list[SymbolDef]],
    ) -> None:
        """tree-sitter counterpart of _extract_all_python_symbols AST pass.

        Walks the tree once and records every function/class/constant into
        *out* under its name key. Mirrors _ts_find_symbol_in_tree but without
        the name/parent filter.
        """
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn_name = _ts_get_node_text(code_bytes, name_node)
                sig = _ts_build_py_function_signature(node, code_bytes)
                doc = _ts_extract_docstring(node, code_bytes)
                decs = _ts_extract_decorators(node, code_bytes)
                out.setdefault(fn_name, []).append(SymbolDef(
                    file=rel, line=node.start_point.row + 1,
                    kind="function" if not parent_class else "method",
                    name=fn_name, signature=sig or None,
                    docstring=doc, decorators=decs or None,
                    end_line=node.end_point.row + 1,
                    parent_class=parent_class or None,
                ))
            return  # do not descend into function bodies

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cls_name = _ts_get_node_text(code_bytes, name_node) if name_node else ""
            if cls_name:
                bases = _ts_extract_class_bases(node, code_bytes) or None
                methods = _ts_collect_class_methods(node, code_bytes) or None
                doc = _ts_extract_docstring(node, code_bytes)
                out.setdefault(cls_name, []).append(SymbolDef(
                    file=rel, line=node.start_point.row + 1,
                    kind="class", name=cls_name,
                    bases=bases, methods=methods, docstring=doc,
                    end_line=node.end_point.row + 1,
                    parent_class=parent_class or None,
                ))
            new_parent = f"{parent_class}.{cls_name}" if parent_class else cls_name
            for child in node.children:
                self._ts_collect_all(child, code_bytes, rel, new_parent, out)
            return

        elif node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    self._ts_collect_all(child, code_bytes, rel, parent_class, out)
            return

        elif node.type == "expression_statement":
            expr = node.children[0] if node.children else None
            if expr and expr.type == "assignment":
                left = expr.child_by_field_name("left")
                right = expr.child_by_field_name("right")
                if left and left.type == "identifier":
                    var_name = _ts_get_node_text(code_bytes, left)
                    val = _ts_get_node_text(code_bytes, right)[:80] if right else ""
                    out.setdefault(var_name, []).append(SymbolDef(
                        file=rel, line=node.start_point.row + 1,
                        kind="constant", name=var_name,
                        signature=f"{var_name} = {val}" if val else None,
                        end_line=node.end_point.row + 1,
                        parent_class=parent_class or None,
                    ))
            return

        for child in node.children:
            self._ts_collect_all(child, code_bytes, rel, parent_class, out)

    def _find_in_python_cached(
        self, file_path: Path, name: str, kind: str,
        parent_class: str = "",
    ) -> list[SymbolDef]:
        """mtime-cached wrapper around per-file full symbol extraction.

        Equivalent to _find_in_python(file_path, name, kind, parent_class) but
        amortizes parsing: the file is parsed once and every symbol is cached,
        so subsequent lookups for any name in the same file are O(1).
        """
        key = str(file_path)
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            return []
        cached = self._py_file_cache.get(key)
        if cached is None or cached[0] != mtime:
            try:
                rel = str(file_path.relative_to(self.repo_root))
            except ValueError:
                rel = str(file_path)
            full_map = self._extract_all_python_symbols(file_path, rel)
            self._py_file_cache[key] = (mtime, full_map)
        else:
            full_map = cached[1]

        defs = full_map.get(name, [])
        # Apply kind + parent_class filter (same semantics as _find_in_python)
        out: list[SymbolDef] = []
        for d in defs:
            # parent_class filter: empty means "any scope"; otherwise exact match.
            if parent_class and (d.parent_class or "") != parent_class:
                continue
            if kind in ("function", "method", "any") and d.kind in ("function", "async_function", "method"):
                out.append(d)
            elif kind in ("class", "any") and d.kind == "class":
                out.append(d)
            elif kind in ("variable", "constant", "any") and d.kind == "constant":
                out.append(d)
            elif kind == "any":
                out.append(d)
        return out

    @staticmethod
    def _is_dataclass(file_path: Path, class_name: str) -> bool:
        """Check if a class has @dataclass decorator (tree-sitter + AST fallback)."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (Exception):
            return False

        # ── Primary: tree-sitter ───────────────────────────
        if _HAS_TS:
            try:
                tree = _ts_parse_to_tree(source, "python")
                if tree is not None:
                    query = f"""
(decorated_definition
  decorator: (decorator) @dec
  definition: (class_definition name: (identifier) @name)
  (#eq? @name "{class_name}")
)
"""
                    from ..languages.tree_sitter_utils import query_matches as _ts_qm
                    matches = _ts_qm(source, "python", query)
                    for match_group in matches:
                        dec_caps = match_group.get("dec", [])
                        for dc in dec_caps:
                            dtext = dc.text.strip()
                            if dtext.startswith("dataclass") or dtext.startswith("@dataclass"):
                                return True
                    return False
            except Exception:
                pass  # fall through to AST

        # ── Fallback: AST ──────────────────────────────────
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for dec in node.decorator_list:
                        dec_name = ""
                        if isinstance(dec, ast.Name):
                            dec_name = dec.id
                        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                            dec_name = dec.func.id
                        elif isinstance(dec, ast.Attribute):
                            dec_name = dec.attr
                        if dec_name == "dataclass":
                            return True
            return False
        except (SyntaxError, AttributeError, TypeError):
            return False

    # ── Go dotted name resolution via GoSyntaxProvider ─────────────────────
    def _find_in_go(
        self, file_path: Path, name: str, kind: str,
        parent_class: str = "",
    ) -> list[SymbolDef]:
        """Find a Go method scoped to a specific struct.

        When ``find_symbol("TodoList.Add")`` is called, *parent_class* is
        ``"TodoList"`` and *name* is ``"Add"``.  This method reads the file,
        gets the Go provider, and calls ``find_class_methods()`` to locate
        only those methods belonging to *parent_class*.
        """
        results: list[SymbolDef] = []
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            rel = str(file_path.relative_to(self.repo_root))
        except Exception:
            return results

        provider = LanguageRegistry.instance().get(str(file_path))
        if provider is None:
            return results

        try:
            methods = provider.find_class_methods(source, parent_class)
        except Exception:
            return results

        for mname, mstart, mend in methods:
            if mname != name:
                continue
            # Read the method signature line for context
            lines = source.split("\n")
            sig_line = lines[mstart - 1].strip() if 0 <= mstart - 1 < len(lines) else ""
            results.append(SymbolDef(
                file=rel, line=mstart, kind="method",
                name=name, signature=sig_line,
                end_line=mend,
                parent_class=parent_class,
            ))
            break  # only the first match

        return results

    # ── TS/JS rich symbol search via TSSemanticTracer ──────────────────────
    _MAX_TS_FILES = _cfg.counts.SYMBOL_MAX_TS_FILES

    # Per-file TS/JS module cache: path -> (mtime, {name -> [SymbolDef]}).
    # analyze_core is ~15ms/file and pure function of content, so caching the
    # full extracted symbol map removes the dominant cost of repeated
    # find_symbol over TS/JS repos (was ~700ms for 47 files, now ~0ms warm).

    def _ts_module_map(self, file_path: Path) -> dict[str, list[SymbolDef]]:
        """Return {name -> [SymbolDef]} for a TS/JS file, mtime-cached."""
        key = str(file_path)
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            return {}
        cached = self._ts_file_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        full_map: dict[str, list[SymbolDef]] = {}
        try:
            from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

            from ..languages.models import LanguageId as _LID
            content = file_path.read_text(encoding="utf-8", errors="replace")
            rel = str(file_path.relative_to(self.repo_root))
            lang_str = "typescript" if _LID.from_path(str(file_path)) == _LID.TYPESCRIPT else "javascript"
            tracer = TSSemanticTracer(language=lang_str)
            module = tracer.analyze_core(content, str(file_path))
            full_map = self._ts_extract_all(module, rel)
        except Exception:
            pass  # non-critical — never block execution
        self._ts_file_cache[key] = (mtime, full_map)
        return full_map

    def _ts_extract_all(self, module, rel: str) -> dict[str, list[SymbolDef]]:
        """Extract ALL symbols from a parsed TS/JS module into {name: [defs]}.

        Covers every kind _find_in_ts_js returns so name lookups become O(1)
        dict filters against a cached module instead of re-running analyze_core.
        """
        out: dict[str, list[SymbolDef]] = {}

        for fn in module.functions:
            sig = _build_ts_function_signature(fn)
            out.setdefault(fn.name, []).append(SymbolDef(
                file=rel,
                line=fn.meta.start_line if fn.meta else fn.start_line,
                kind="async_function" if fn.is_async else "function",
                name=fn.name, signature=sig,
            ))
        for cls in module.classes:
            methods = [m.name for m in cls.methods]
            bases = []
            if cls.extends:
                bases.append(cls.extends)
            bases.extend(cls.implements or [])
            out.setdefault(cls.name, []).append(SymbolDef(
                file=rel,
                line=cls.meta.start_line if cls.meta else cls.start_line,
                kind="class", name=cls.name,
                methods=methods[:25] or None, bases=bases or None,
            ))
            for method in cls.methods:
                msig = _build_ts_method_signature(cls.name, method)
                out.setdefault(method.name, []).append(SymbolDef(
                    file=rel,
                    line=method.meta.start_line if method.meta else method.start_line,
                    kind="method", name=f"{cls.name}.{method.name}", signature=msig,
                ))
        for iface in module.interfaces:
            out.setdefault(iface.name, []).append(SymbolDef(
                file=rel,
                line=iface.meta.start_line if iface.meta else iface.start_line,
                kind="interface", name=iface.name,
                methods=iface.methods[:25] or None,
            ))
        for ta in module.type_aliases:
            out.setdefault(ta.name, []).append(SymbolDef(
                file=rel,
                line=ta.meta.start_line if ta.meta else ta.start_line,
                kind="type", name=ta.name,
            ))
        for en in module.enums:
            out.setdefault(en.name, []).append(SymbolDef(
                file=rel,
                line=en.meta.start_line if en.meta else en.start_line,
                kind="enum", name=en.name,
            ))
        for var in module.variables:
            sig = f"{var.decl_kind} {var.name}"
            if var.type_ref:
                sig += f": {var.type_ref.name}"
            out.setdefault(var.name, []).append(SymbolDef(
                file=rel,
                line=var.meta.start_line if var.meta else var.start_line,
                kind="variable", name=var.name, signature=sig,
            ))
        return out

    def _find_in_ts_js(self, file_path: Path, name: str, kind: str) -> list[SymbolDef]:
        """TSSemanticTracer-based rich symbol search for TS/JS files (cached)."""
        full_map = self._ts_module_map(file_path)
        defs = full_map.get(name, [])
        out: list[SymbolDef] = []
        for d in defs:
            if kind in ("function", "method", "any") and d.kind in ("function", "async_function", "method"):
                out.append(d)
            elif kind in ("class", "interface", "any") and d.kind in ("class", "interface"):
                out.append(d)
            elif kind in ("type", "any") and d.kind == "type":
                out.append(d)
            elif kind in ("enum", "any") and d.kind == "enum":
                out.append(d)
            elif kind in ("variable", "any") and d.kind == "variable":
                out.append(d)
            elif kind == "any":
                out.append(d)
        return out

    def _outline_ts_js(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """TSSemanticTracer-based file outline for TS/JS files."""
        try:
            from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

            from ..languages.models import LanguageId
        except ImportError:
            return self._outline_ripgrep(file_path, rel)

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (ImportError, AttributeError):
            return []

        lang_id = LanguageId.from_path(str(file_path))
        lang_str = "typescript" if lang_id == LanguageId.TYPESCRIPT else "javascript"

        try:
            tracer = TSSemanticTracer(language=lang_str)
            module = tracer.analyze_core(content, str(file_path))
        except Exception:
            return self._outline_ripgrep(file_path, rel)  # non-critical — never block execution

        results: list[SymbolDef] = []

        for fn in module.functions:
            sig = _build_ts_function_signature(fn)
            results.append(SymbolDef(
                file=rel,
                line=fn.meta.start_line if fn.meta else fn.start_line,
                kind="async_function" if fn.is_async else "function",
                name=fn.name,
                signature=sig,
            ))

        for cls in module.classes:
            methods = [m.name for m in cls.methods]
            results.append(SymbolDef(
                file=rel,
                line=cls.meta.start_line if cls.meta else cls.start_line,
                kind="class",
                name=cls.name,
                methods=methods[:25] or None,
            ))

        for iface in module.interfaces:
            results.append(SymbolDef(
                file=rel,
                line=iface.meta.start_line if iface.meta else iface.start_line,
                kind="interface",
                name=iface.name,
            ))

        for ta in module.type_aliases:
            results.append(SymbolDef(
                file=rel,
                line=ta.meta.start_line if ta.meta else ta.start_line,
                kind="type",
                name=ta.name,
            ))

        for en in module.enums:
            results.append(SymbolDef(
                file=rel,
                line=en.meta.start_line if en.meta else en.start_line,
                kind="enum",
                name=en.name,
            ))

        for var in module.variables:
            results.append(SymbolDef(
                file=rel,
                line=var.meta.start_line if var.meta else var.start_line,
                kind="variable",
                name=var.name,
            ))

        results.sort(key=lambda s: s.line)
        return results

    def _rg_path_to_rel(self, raw: str) -> str:
        """Normalize an rg-emitted path to a repo-relative string.

        rg runs with cwd=repo_root and emits paths like './foo/bar.go' or
        'foo/bar.go'. A bare relative_to(repo_root) fails because the path is
        not absolute. This handles all three shapes (absolute, './'-prefixed,
        bare-relative) robustly so the non-Python index and the legacy
        _find_in_other_langs path agree on canonical relative paths.
        """
        p = Path(raw)
        try:
            if p.is_absolute():
                return str(p.relative_to(self.repo_root))
        except ValueError:
            pass
        s = str(p)
        if s.startswith("./"):
            s = s[2:]
        return s

    def _index_via_treesitter(
        self, provider, search_root: Path,
        index: dict[str, list[SymbolDef]], seen: set,
    ) -> None:
        """Index a provider's files by parsing each with tree-sitter.

        Replaces the rg+regex path for languages whose tree-sitter binding is
        installed. For CSS this is the authoritative source: class selectors,
        id selectors, and custom properties (``--name``) are extracted from the
        AST, so no regex pattern ever becomes an rg positional/flag arg (the
        ``--name`` leading-dash trap is structurally impossible here).

        Failures (unreadable file, parse error) are skipped per-file — the
        index simply lacks those symbols, matching the rg path's tolerance.
        """
        lang_id = provider.language_id().value
        for glob in provider.get_file_globs():
            try:
                proc = subprocess.run(
                    ["rg", "--files", "--glob", glob,
                     "--glob", "!node_modules*", str(search_root)],
                    cwd=str(self.repo_root),
                    capture_output=True, text=True, timeout=8,
                )
            except subprocess.SubprocessError:
                continue
            for fpath in (proc.stdout or "").splitlines():
                if not fpath:
                    continue
                try:
                    abs_path = self.repo_root / fpath
                    content = abs_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError):
                    continue
                try:
                    syms = _ts_find_all_symbols(content, lang_id)
                except Exception:
                    continue
                rel = self._rg_path_to_rel(fpath)
                for name, kind, start_line, _end_line in syms:
                    # Dedup by (file, name, line): multiple distinct symbols can
                    # share a line (e.g. ``.x { color: red; --real-var: 1; }``
                    # has both a class selector and a custom property on line 1),
                    # so name must be part of the key to avoid dropping them.
                    key = (rel, name, start_line)
                    if key in seen:
                        continue
                    seen.add(key)
                    # CSS custom properties are stored in the AST with their
                    # leading "--" (e.g. "--primary-color"), but callers search
                    # by the bare identifier ("primary-color"). Normalize both
                    # the index key and the stored name so lookup matches
                    # regardless of whether the caller includes the dashes.
                    if kind == "css_variable" and name.startswith("--"):
                        norm_name = name[2:]
                        index.setdefault(norm_name, []).append(SymbolDef(
                            file=rel, line=start_line, kind=kind, name=norm_name,
                            signature="",
                        ))
                        # Also index under the dashed form so "--primary-color"
                        # lookups resolve too.
                        index.setdefault(name, []).append(SymbolDef(
                            file=rel, line=start_line, kind=kind, name=name,
                            signature="",
                        ))
                    else:
                        index.setdefault(name, []).append(SymbolDef(
                            file=rel, line=start_line, kind=kind, name=name,
                            signature="",
                        ))

    def _nonpy_index_for(self, search_root: Path) -> dict[str, list[SymbolDef]]:
        """Build (once, TTL-cached) a {name -> [SymbolDef]} index of ALL
        non-Python definitions under *search_root*.

        Invalidation is TTL-based (same scheme as _walk_py_files): an mtime
        fingerprint would itself require a full directory walk (~6s here),
        defeating the cache. The TTL is generous (30s) and find_symbol tolerates
        a briefly-stale index because edited non-Python files re-converge on the
        next TTL expiry.
        """
        cache_key = str(search_root)
        cached = self._nonpy_index_cache.get(cache_key)
        if cached is not None:
            ts, index = cached
            if (_time.monotonic() - ts) < _WALK_CACHE_TTL:
                return index

        index: dict[str, list[SymbolDef]] = {}
        seen: set = set()
        ts_langs = _ts_available_languages() if _HAS_TS else set()
        registry = LanguageRegistry.instance()
        for provider in set(registry._providers.values()):
            lang_id = provider.language_id().value
            if lang_id in ("python", "typescript", "javascript"):
                continue  # handled by AST/TS tracer paths

            # ── Grammar-missing detection ──────────────────────────────────
            # A language that (a) is tree-sitter supported, (b) has its grammar
            # not installed, and (c) exposes no regex fallback (empty
            # get_symbol_patterns) would be indexed by NEITHER path → silent
            # zero results with no signal. Warn once so the cause is obvious.
            # (CSS hit this after its regex path was retired for the AST path.)
            if (
                _HAS_TS
                and lang_id in _TS_LANG_MODULE_MAP
                and lang_id not in ts_langs
                and not provider.get_symbol_patterns(kind="any")
                and lang_id not in _warned_missing_grammar
            ):
                _warned_missing_grammar.add(lang_id)
                logger.warning(
                    "[symbol-search] %s symbols skipped: tree-sitter grammar "
                    "'%s' not installed. Install it or symbol search for this "
                    "language will return nothing. (warned once per process)",
                    lang_id,
                    _TS_LANG_MODULE_MAP[lang_id].replace("_", "-"),
                )

            # ── AST-first path: if this provider's language has a tree-sitter
            # binding installed, index its files by parsing the AST directly —
            # no rg subprocess, no regex patterns. This is the single source
            # of truth for CSS (class/id/custom-property), where the regex
            # approach previously hit the leading "-"/"#" shell-arg trap.
            if lang_id in ts_langs:
                self._index_via_treesitter(provider, search_root, index, seen)
                continue  # skip the provider-regex rg spawn below

            for glob in provider.get_file_globs():
                for sp in provider.get_symbol_patterns(kind="any"):
                    # Capture the name with the pattern's own name_capture
                    # group (default \w+) instead of substituting a literal.
                    # CSS uses [-\w]+ so kebab-case names like "btn-primary"
                    # or "--primary-color" are not truncated at the hyphen.
                    pat = sp.regex.replace("{name}", f"({sp.name_capture})")
                    try:
                        cmd = [
                            "rg", "--no-heading", "--with-filename", "--line-number", "-m", "5",
                            "--glob", glob,
                            "--glob", "!node_modules*",
                            "--glob", "!*.py",
                            # Pass the pattern via -e so a pattern that starts
                            # with '-' (e.g. the CSS "--{name}" custom-property
                            # pattern) is not misparsed as a flag.
                            "-e", pat, str(search_root),
                        ]
                        proc = subprocess.run(
                            cmd, cwd=str(self.repo_root),
                            capture_output=True, text=True, timeout=8,
                        )
                        for line in (proc.stdout or "").splitlines()[:50]:
                            parts = line.split(":", 2)
                            if len(parts) < 3:
                                continue
                            try:
                                rel = self._rg_path_to_rel(parts[0])
                                lineno = int(parts[1])
                                ctx = parts[2].strip()
                                key = (rel, lineno)
                                if key in seen:
                                    continue
                                seen.add(key)
                                # Prefer the regex's own capture group, then a
                                # generic identifier heuristic as a fallback.
                                rm = re.search(pat, ctx)
                                cap_name = rm.group(1) if (rm and rm.groups()) else ""
                                if not cap_name:
                                    m = re.search(r"\b(\w+)\s*[\(\{<]", ctx)
                                    cap_name = m.group(1) if m else ""
                                if not cap_name:
                                    continue
                                index.setdefault(cap_name, []).append(SymbolDef(
                                    file=rel, line=lineno, kind=sp.kind, name=cap_name,
                                    signature=ctx,
                                ))
                            except (ValueError, AttributeError, TypeError):
                                pass
                    except (AttributeError, TypeError, subprocess.SubprocessError):
                        pass

        self._nonpy_index_cache[cache_key] = (_time.monotonic(), index)
        return index

    def _find_subclasses(self, base_class: str) -> list[str]:
        """Find names of classes that inherit from base_class."""
        pattern = rf"class\s+(\w+)\s*\(.*\b{re.escape(base_class)}\b"
        names: list[str] = []
        try:
            cmd = ["rg", "--no-heading", "-m", "3", pattern, str(self.repo_root)]
            proc = subprocess.run(
                cmd, cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=5,
            )
            for line in (proc.stdout or "").splitlines()[:20]:
                m = re.search(r"class\s+(\w+)", line)
                if m and m.group(1) != base_class:
                    names.append(m.group(1))
        except (OSError, subprocess.SubprocessError):
            pass
        return list(dict.fromkeys(names))
