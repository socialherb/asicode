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
get_symbol_searcher(repo_root)             -> process-shared pooled SymbolSearcher
"""
from __future__ import annotations

import ast
import contextlib
import difflib
import logging
import os
import re
import shutil
import subprocess
import threading
import time as _time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ..common.text_reading import read_line_window
from ..common.walk_policy import _WALK_SKIP_DIRS
from ..languages import (
    LanguageId,  # S8 fix: missing module-level import
    LanguageRegistry,
)
from ..languages.tree_sitter_utils import (
    _LANG_MODULE_MAP as _TS_LANG_MODULE_MAP,
)
from ..languages.tree_sitter_utils import (
    find_all_symbols as _ts_find_all_symbols,
)
from ..languages.tree_sitter_utils import (
    get_available_languages as _ts_available_languages,
)
from ..languages.tree_sitter_utils import (
    get_node_text as _ts_get_node_text,
)
from ..languages.tree_sitter_utils import (
    is_language_available as _ts_language_available,
)
from ..languages.tree_sitter_utils import (
    parse_to_tree as _ts_parse_to_tree,
)

# Walk-cache introspection — lets find_symbol distinguish a genuine miss
# ("symbol absent") from a truncated index ("symbol may live in un-indexed
# files"). Both caches are module-global in ._shared_utils.
from ._shared_utils import _PY_WALK_CACHE as _SHARED_PY_WALK_CACHE
from ._shared_utils import _TS_WALK_CACHE as _SHARED_TS_WALK_CACHE
from ._shared_utils import (
    _WALK_CACHE_TTL,
    _capped_put,
)
from ._shared_utils import (
    _walk_py_files as _shared_walk_py_files,
)
from ._shared_utils import _walk_truncated_for as _shared_walk_truncated_for
from ._shared_utils import (
    _walk_ts_js_files as _shared_walk_ts_js_files,
)
from ._thread_pool import shared_pool as _shared_pool
from .bm25 import bm25_rank
from .config.thresholds import config as _cfg
from .rag_configs import CodeTokenizer

# Cap on how long the speculative non-Python probe (submitted to _shared_pool
# before the Python scan) may be awaited. The probe runs on the SAME pool a
# dispatch may already be occupying (P1: execute_parallel dispatches on
# _thread_pool.shared_pool), so N concurrent find_symbol calls each blocking on
# a still-queued probe would exhaust the pool and deadlock permanently; with a
# cap the worst case is this stall followed by the inline fallback below.
_NONPY_PROBE_TIMEOUT_SEC = 10.0

# Module-level lazy tokenizer singleton — avoids re-constructing
# CodeTokenizer (which compiles internal regex-alternative scanners)
# on every find_references call (a hot path).
_TOKENIZER: Any = None

# ── Tree-sitter availability ─────────────────────────────────────────────
# tree_sitter_utils guards its own tree-sitter import (get_parser returns
# None / _ts_language_available False when the grammar is missing), so the
# module itself can never fail to import — the old try/except ImportError
# fallback was dead code. _HAS_TS stays as a module flag so tests can force
# the regex fallback path via monkeypatch.
_HAS_TS = True

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

# ── Superlinear tree-sitter guard for Python ────────────────────────────────
# tree-sitter's Python grammar parses a long RUN of indented comment lines
# inside a function body quadratically in the run length: measured with
# 200-char comments, 1000 lines → 0.50 s, 2000 → 1.98 s, 3000 → 4.53 s,
# 4000 → 7.88 s. The same lines at module level cost 8 ms, 4000 non-comment
# lines 14 ms, class-body comments 12 ms — the run of indented comments is
# the trigger, not the line count or the line width (4000x20-char comments
# still cost 0.89 s). Real code never forms such runs (the largest in this
# repo is 32 consecutive indented comments), so the guard only fires on
# generated files, license/docstring walls and synthetic fixtures.
# ast.parse is linear and extracts the same symbols for valid files, so
# such files are routed to the AST path; tree-sitter remains the last
# resort when ast fails on a syntax-broken file.
_TS_SKIP_MIN_LINES = 300  # below this, keep tree-sitter (error tolerance)
_TS_SKIP_COMMENT_RUN = 50  # consecutive indented comment lines that trigger


def _python_ts_parse_too_costly(source: str) -> bool:
    """True when tree-sitter Python parsing of *source* is superlinear.

    Counts consecutive indented ``#`` comment lines — the measured O(n²)
    trigger. Short-circuits as soon as the run crosses the threshold past
    the minimum line count, so the scan costs O(run) on the offending
    prefix and one cheap pass otherwise.
    """
    run = 0
    for n_total, line in enumerate(source.splitlines(), start=1):
        if line[:1].isspace() and line.lstrip().startswith("#"):
            run += 1
            if run >= _TS_SKIP_COMMENT_RUN and n_total >= _TS_SKIP_MIN_LINES:
                return True
        else:
            run = 0
    return False


# _WALK_CACHE_TTL is re-exported from ._shared_utils (shared with call_graph).
# _time is still needed for the non-Python index TTL check below.

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
    with contextlib.suppress(KeyError, TypeError, ValueError, AttributeError, re.error):  # fall through to heuristic fallback
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

    # ── Fallback: generic patterns for unrecognised providers ────────────
    return bool(
        stripped.startswith((f"def {name}", f"async def {name}", f"class {name}"))
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


def _ts_node_end_line(node: Any) -> Optional[int]:
    """Extent of a TS/JS IR node, or None when it carries no usable one.

    Mirrors the ``meta or bare attribute`` shape the start lines already use,
    via getattr because not every IR node declares both (IRImport/IRExport
    carry only ``meta``). ``or None`` normalises the dataclass default of 0,
    which would otherwise render as a "line N-0" range.
    """
    _m = getattr(node, "meta", None)
    _e = getattr(_m, "end_line", None) if _m else getattr(node, "end_line", None)
    return _e or None


def _walk_py_files(root: Path) -> list[Path]:
    """Walk repo, returning .py files, skipping hidden/vendor dirs.

    Thin wrapper over the shared walker (._shared_utils._walk_py_files) so
    symbol_search and call_graph share one implementation + one process-global
    cache. Best-effort: a concurrent write is fine because callers tolerate a
    slightly-stale file list (missing files → "not found this round").
    """
    return _shared_walk_py_files(root, _MAX_PY_FILES)


def _nonpy_index_globs() -> list[str]:
    """File globs the non-Python index actually covers.

    Read from the same provider registry and with the same python/ts/js
    exclusion that ``_index_via_treesitter_batch`` uses, so the probe and the
    index can never disagree about what is in scope.

    Owning a glob is NOT enough — the provider must have a path that can emit a
    symbol. ``_nonpy_index_for`` has exactly two: the tree-sitter batch (needs
    the grammar installed) and the regex loop (needs non-empty
    ``get_symbol_patterns``). A provider with neither owns globs whose files the
    index walks past in silence, so counting them makes the probe answer a
    question the build cannot: it returns True and the caller pays a whole-repo
    build that provably cannot contain the token.

    Measured on this repo, where ``json`` is exactly such a provider (no
    grammar in ``_LANG_MODULE_MAP``, no regex patterns): its ``*.json`` glob
    contributed 158 of the 163 files in the probe's scope while only 5 were
    indexable at all, and 19 of 40 sampled real symbol names probed True purely
    on a .json mention — a 75 ms index build each, for a set that could never
    match. Filtering here is the whole fix; the build itself was already right.

    Grammar availability is a property of the INSTALL, not of the repo, so this
    is recomputed rather than frozen: installing ``tree-sitter-json`` makes json
    indexable and its glob correctly re-enters scope.
    """
    globs: list[str] = []
    registry = LanguageRegistry.instance()
    for provider in sorted(
        set(registry._providers.values()), key=lambda p: p.language_id().value
    ):
        lang_id = provider.language_id().value
        if lang_id in ("python", "typescript", "javascript"):
            continue
        # Patterns first, grammar second — deliberately, not stylistically. A
        # provider with regex patterns is indexable no matter what is installed,
        # so asking about its grammar is a question we do not need the answer
        # to, and answering it imports that grammar module. Ordering this way
        # leaves only the pattern-less providers (css/html/json here) to
        # resolve, on what is now find_symbol's hot path — the whole-set form
        # imported all 19 mapped grammars (~50 ms) to decide 28 globs.
        if provider.get_symbol_patterns(kind="any") or (_HAS_TS and _ts_language_available(lang_id)):
            globs.extend(provider.get_file_globs())
    return globs


# In-process probing is only a win while the indexable set stays small: it
# trades one rg spawn (~9 ms, flat) for reading N files (~0.04 ms each here).
# Above these caps the spawn is cheaper and bounded, so we defer to rg.
_NONPY_INPROC_MAX_FILES = 200
_NONPY_INPROC_MAX_BYTES = 8 * 1024 * 1024


def _too_big_to_parse_inproc(st_size: int, file_path: Any) -> bool:
    """True when one file is too large to read + parse in this process.

    P26-4 gated the BATCH tree-sitter walker (``_index_via_treesitter``) at
    ``_NONPY_INPROC_MAX_BYTES`` because "a single minified dist/*.js (tens of
    MB is common) was read + tree-sitter-parsed in full".  Every PER-FILE
    entry point had the identical hole, and each of them already holds the
    answer: ``_find_in_python_cached`` / ``_go_class_methods_map`` /
    ``_ts_module_map`` all ``stat()`` the file and then spend ``st_size`` on
    nothing but a cache signature.

    Measured on a generated 32 MB bundle before this gate: read_file's
    over-cap refusal — a 131-character message that returns no file content
    at all — cost 13.31 s and 1.65 GB of peak RSS (8.67 s of it a single
    ``tree_sitter.Parser.parse``), and find_symbol over the same tree cost
    21.87 s to answer "not found".  A 13 MB .py cost 3.40 s / 684 MB by the
    same route.  That is ~20x the whole-process peak-RSS budget 0.2.15
    established when it capped tree-sitter parse memos at 76 MB.

    Callers degrade, never fail: an empty map sends ``get_file_outline`` down
    its documented ``_outline_treesitter`` → ``_outline_ripgrep`` fallback and
    leaves find_symbol to the rg path — the same behaviour an unparsable file
    already produces.  The cost is that a symbol defined in an 8 MiB+ source
    file is not reachable by exact-parse lookup; that is the trade P26-4
    already made for the batch walker, made consistent here.
    """
    if st_size <= _NONPY_INPROC_MAX_BYTES:
        return False
    logger.debug(
        "[symbol-search] skipping in-process parse of %s (%d bytes > %d)",
        file_path, st_size, _NONPY_INPROC_MAX_BYTES,
    )
    return True
# Stream size for _word_in_files. A whole-word match needs at most
# len(token) + 2 bytes of context (one word char on each side), so carrying
# that many trailing bytes across the seam keeps the lookarounds exact.
_NONPY_SCAN_CHUNK = 64 * 1024
# {root: (timestamp, files, total_bytes)} — files is None when unanswerable.
_NONPY_FILES_CACHE: dict[str, tuple] = {}


def _nonpy_indexable_files(root: Path) -> Optional[tuple[list[str], int]]:
    """TTL-cached ``(paths, total_bytes)`` of files the non-Python index reads.

    One ``rg --files`` walk shared across every token, where the probe's own
    query is per-token. That is the whole point: the walk is the expensive half
    and it does not depend on the token, so hoisting it turns each subsequent
    probe into an in-process scan.

    Returns None when the answer cannot be trusted (rg missing/error), so the
    caller keeps its existing fallback rather than treating "unknown" as empty.
    """
    key = str(root)
    hit = _NONPY_FILES_CACHE.get(key)
    if hit is not None and (_time.monotonic() - hit[0]) < _WALK_CACHE_TTL:
        return None if hit[1] is None else (hit[1], hit[2])
    rg = shutil.which("rg")
    if not rg:
        return None
    glob_args: list[str] = []
    for g in _nonpy_index_globs():
        glob_args += ["--glob", g]
    if not glob_args:
        return None
    try:
        proc = subprocess.run(
            [rg, "--files", "--no-ignore-vcs", *glob_args, "."],
            cwd=str(root), capture_output=True, text=True, timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("nonpy file list: rg failed (%s)", e)
        _capped_put(_NONPY_FILES_CACHE, key, (_time.monotonic(), None, 0))
        return None
    if proc.returncode not in (0, 1):
        _capped_put(_NONPY_FILES_CACHE, key, (_time.monotonic(), None, 0))
        return None
    files: list[str] = []
    total = 0
    for line in proc.stdout.splitlines():
        if not line:
            continue
        p = str(root / line)
        files.append(p)
        try:
            total += os.path.getsize(p)
        except OSError as e:
            # Vanished between walk and stat — the scan skips it too, so the
            # only consequence is a slightly low byte total for the cap.
            logger.debug("nonpy file list: cannot size %s (%s)", p, e)
    _capped_put(_NONPY_FILES_CACHE, key, (_time.monotonic(), files, total))
    return files, total


def _word_in_files(files: list[str], token: str) -> bool:
    """True if *token* occurs as a whole word in any of *files*.

    Mirrors rg's ``--word-regexp --fixed-strings``: the match must not be
    flanked by word characters. ``(?<!\\w)…(?!\\w)`` is used rather than
    ``\\b…\\b`` because ``\\b`` is defined relative to the adjacent pattern
    character, so a token starting or ending in a non-word character (CSS's
    ``--var``) would anchor the wrong way; the lookarounds mean the same thing
    for plain identifiers and stay correct for those.

    Unreadable files are skipped, matching ``_index_via_treesitter_batch`` —
    a file the build cannot read holds no indexable symbol either.

    Files are streamed in ``_NONPY_SCAN_CHUNK`` chunks rather than slurped:
    a hit near the top of a large generated file no longer forces the whole
    file into memory, and ``len(token) + 2`` trailing bytes are carried
    across the seam so tokens split by a chunk boundary still match.
    """
    pat = re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")
    carry = len(token) + 2
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                tail = ""
                for chunk in iter(lambda: fh.read(_NONPY_SCAN_CHUNK), ""):
                    buf = tail + chunk
                    if pat.search(buf):
                        return True
                    tail = buf[-carry:]
        except OSError as e:
            logger.debug("nonpy probe: unreadable %s (%s) — skipped", f, e)
            continue
    return False


# Memo for token SEQUENCES: _word_in_files re-reads the whole (capped) file
# set on every probe, and find_symbol probes once per call — SymbolSearcher
# instances are per-lookup, so the per-instance index cache cannot dedupe the
# repeat lookups a turn issues. The first probe pays ONE full read and caches
# the set's content as one blob; every later token answers with a single
# literal search over it, no file I/O at all.
#
# The streaming core keeps its early-exit-on-hit; the probe path deliberately
# trades it: a HIT triggers a cold whole-repo index build anyway, and the
# probe runs concurrently with the Python scan (speculative), so the one extra
# read is hidden on the wall clock while the miss-heavy repeat case drops from
# a full re-read per token to a memchr-speed scan.
#
# Freshness mirrors _nonpy_indexable_files: the key carries the file LIST (a
# newly created file changes the list and the key), and per-file
# (st_mtime_ns, st_size) signatures are re-verified on every probe, so a
# just-written file invalidates the blob without waiting for the TTL.
# invalidate_nonpy_caches clears both layers together.
#
# FIFO-capped BELOW the sibling caches on purpose: a value is content (up to
# _NONPY_INPROC_MAX_BYTES each), not paths, so the cap is also the memory
# bound (4 x 8 MB worst case).
_NONPY_BLOB_CACHE: dict[tuple, tuple[float, dict[str, Optional[tuple[int, int]]], str]] = {}
_NONPY_BLOB_MAX_ENTRIES: int = 4

# The lookaround form (?<!\w)...(?!\w) costs ~16 ns/byte (the engine visits
# every position), which dominates the probe on multi-MB sets. The equivalent
# literal search + manual boundary check is ~0.6 ns/byte — the same semantics
# (a seam/newline is not \w, so joined files behave exactly like the per-file
# scan) at memchr speed.
_WORD_CHAR = re.compile(r"\w")


def _blob_contains_word(blob: str, token: str) -> bool:
    """Whether *token* occurs as a whole word in *blob*.

    Exact equivalent of ``(?<!\\w)`` + escape + ``(?!\\w)``: a literal hit
    counts only when neither flanking char is a word char (start/end of the
    blob count as clean). Bad-boundary hits are skipped by resuming the
    search one char past them, so repeated matches cannot loop — the empty
    token included, which matches exactly where the lookarounds would.

    The ``_pos <= len(blob)`` bound exists for the empty-pattern edge: a
    literal-only search at pos > len() clamps back to the end and returns an
    empty match, which would otherwise pin ``_pos`` forever.
    """
    _lit = re.compile(re.escape(token))
    _pos = 0
    while _pos <= len(blob):
        m = _lit.search(blob, _pos)
        if m is None:
            return False
        _s, _e = m.start(), m.end()
        if (_s == 0 or not _WORD_CHAR.match(blob, _s - 1)) and (
            _e == len(blob) or not _WORD_CHAR.match(blob, _e)
        ):
            return True
        _pos = _s + 1
    return False


def _path_sig(path: str) -> Optional[tuple[int, int]]:
    """(st_mtime_ns, st_size) of *path*, or None when it cannot be stat-ed."""
    try:
        st = os.stat(path)
    except OSError as e:
        logger.debug("nonpy blob: cannot stat %s (%s) — treated as absent", path, e)
        return None
    return (st.st_mtime_ns, st.st_size)


def _sigs_valid(sigs: dict[str, Optional[tuple[int, int]]], files: list[str]) -> bool:
    """Whether every *file* still matches its cached signature.

    A stored None (file was missing at build time) stays valid while the file
    stays missing; a file that reappears or changes produces a tuple that
    mismatches, forcing a rebuild. An open-failed-but-stat-able file keeps its
    REAL signature, so a permission-denied file does not rebuild on every
    probe — the "becomes readable mid-TTL" edge is covered by the TTL, the
    same staleness horizon as the file-list cache.
    """
    return all(sigs.get(f) == _path_sig(f) for f in files)


def _nonpy_blob(files: list[str]) -> tuple[dict[str, Optional[tuple[int, int]]], str]:
    """``(sigs, content)`` — one full read of *files* joined by ``"\\n"``.

    Unreadable files are skipped with their signature still recorded, matching
    _word_in_files' skip semantics. The newline separator is a non-word char,
    so the ``(?<!\\w)/(?!\\w)`` lookarounds see a seam exactly like a file
    boundary: whole-word semantics survive the join. (Hence the newline guard
    in the caller — a token containing one could otherwise match across seams.)
    """
    sigs: dict[str, Optional[tuple[int, int]]] = {}
    parts: list[str] = []
    for f in files:
        try:
            st = os.stat(f)
        except OSError as e:
            sigs[f] = None
            logger.debug("nonpy blob: cannot stat %s (%s) — skipped", f, e)
            continue
        sigs[f] = (st.st_mtime_ns, st.st_size)
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                parts.append(fh.read())
        except OSError as e:
            logger.debug("nonpy blob: unreadable %s (%s) — skipped", f, e)
    return sigs, "\n".join(parts)


def _word_in_files_cached(root: Path, files: list[str], token: str) -> bool:
    """``_word_in_files`` with the per-root content memo above.

    A token containing a newline falls back to the streaming core, where the
    per-file scan cannot match across files (the ``"\\n"`` join would).
    """
    if "\n" in token:
        return _word_in_files(files, token)
    key = (str(root), tuple(files))
    hit = _NONPY_BLOB_CACHE.get(key)
    if hit is not None and (_time.monotonic() - hit[0]) < _WALK_CACHE_TTL:
        _ts, _sigs, _blob = hit
        if _sigs_valid(_sigs, files):
            return _blob_contains_word(_blob, token)
    # Cold: ONE full read builds the memo, and the answer comes from the same
    # pass (see the module comment — the extra read vs. early-exit streaming
    # is hidden by the speculative probe, and it turns every later token into
    # a no-I/O literal search).
    _sigs, _blob = _nonpy_blob(files)
    _capped_put(_NONPY_BLOB_CACHE, key, (_time.monotonic(), _sigs, _blob),
                _NONPY_BLOB_MAX_ENTRIES)
    return _blob_contains_word(_blob, token)


def _rg_token_in_nonpy_files(root: Path, token: str) -> Optional[bool]:
    """Whether *token* appears as a bare word in any indexed non-Python file.

    Companion to :func:`_rg_py_files_containing` for the non-Python index: a
    file that DEFINES ``token`` must mention it, so a False here means the
    whole-repo non-Python index cannot possibly contain the symbol and building
    it is pure waste.

    Returns None when the answer cannot be trusted (rg missing, error, timeout)
    so the caller builds the index exactly as before.

    Scope is the provider globs, NOT ``--type-not py``. The looser form both
    cost more (1335 files scanned instead of 163 here, 26-97ms instead of
    8-12ms) and answered the wrong question — it reported True for tokens that
    appear only in .txt baselines, which the index never indexes, so the build
    it triggered could not have matched. ``--no-ignore-vcs`` matches
    _rg_py_files_containing; see its docstring.
    """
    rg = shutil.which("rg")
    if not rg:
        return None
    # Fast path: the indexable set is walked ONCE per root (TTL-cached, token
    # independent) and scanned in-process, so repeat lookups cost a read rather
    # than a spawn — and after the first miss, a single in-memory scan over the
    # cached blob (_word_in_files_cached). Falls through to rg when the set is
    # unknown or too big.
    _listed = _nonpy_indexable_files(root)
    if _listed is not None:
        _files, _bytes = _listed
        if len(_files) <= _NONPY_INPROC_MAX_FILES and _bytes <= _NONPY_INPROC_MAX_BYTES:
            return _word_in_files_cached(root, _files, token)
    glob_args: list[str] = []
    for g in _nonpy_index_globs():
        glob_args += ["--glob", g]
    if not glob_args:
        return None  # no non-Python providers -> nothing to assert; build as before
    try:
        proc = subprocess.run(
            [rg, "--quiet", "--no-ignore-vcs", "--word-regexp", "--fixed-strings",
                *glob_args, "--", token, "."],
            cwd=str(root), capture_output=True, text=True, timeout=10,
                check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("nonpy index probe: rg failed (%s) — building index", e)
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    logger.debug("nonpy index probe: rg exit %s — building index", proc.returncode)
    return None


def _rg_py_files_containing(root: Path, token: str) -> Optional[set[str]]:
    """Absolute paths of .py files under *root* containing *token* as a word.

    A prefilter for the Python symbol scan: a file that DEFINES ``token`` must
    contain ``token`` as a bare word, so any file rg does not list cannot hold
    the definition. Narrowing to these files avoids tree-sitter-parsing the
    whole repo to answer one lookup (measured: 1102 parses / 2.2s cold on this
    repo, versus 2-61 candidate files for real symbols).

    Returns None when the filter cannot be trusted — rg missing, rg error, or
    timeout — so the caller falls back to scanning every file. An empty set is
    a real answer (no file contains the token) and is NOT None.

    ``--no-ignore-vcs`` is deliberate: .gitignore in this repo has hidden real
    source before (bare filename patterns un-tracking exploration/ modules), and
    a prefilter that inherits that blindness would silently drop definitions.
    Vendor/hidden-dir policy is not rg's job here — the caller intersects this
    set with ``_walk_py_files``, which remains the single source of truth for
    which files are in scope.
    """
    return _rg_list_py_files(root, ["--word-regexp", "--fixed-strings", "--", token])


# {(root, matcher_args): (timestamp, files)} for the Python prefilter spawns.
# Keyed by QUERY, so it is sized for the distinct symbols looked up inside one
# TTL window (a handful), not for repos like the walk caches it sits beside —
# hence a larger cap than their 8. Not larger still: a value is a set of
# absolute paths, and the word-match fallback can return most of the repo, so
# the cap is also the memory bound.
_RG_PY_FILTER_CACHE: dict[tuple, tuple[float, set[str]]] = {}
_RG_PY_FILTER_MAX_ENTRIES: int = 32


def invalidate_py_prefilter_cache() -> None:
    """Drop the Python prefilter memo so a just-written file is visible.

    Sibling of :meth:`SymbolSearcher.invalidate_nonpy_caches`, and needed for
    the same reason: the memo below is TTL-based, and 30 s of staleness is fine
    for drift but NOT for the agent's own writes — it edits a file and looks up
    the symbol it just added in the same turn. Without this the memo would
    reintroduce "find_symbol answers 'No definitions found' for a function that
    is on disk" (commit 77008787) on the *Python* side, which is the one side
    that was already correct.

    Cleared wholesale rather than per-path: the key is the rg QUERY, so there is
    no path to match against — an entry's value is a file set, and a newly
    created file is absent from every one of them.
    """
    _RG_PY_FILTER_CACHE.clear()


def _rg_list_py_files(root: Path, matcher_args: list[str]) -> Optional[set[str]]:
    """Core rg runner for the find_symbol prefilters.

    Returns absolute paths of .py files under *root* that rg matches with
    *matcher_args*, or None when the answer cannot be trusted (rg missing,
    error, timeout). Shared by the word-match and definition-pattern
    prefilters so the trust contract (None vs empty set) stays identical.

    Memoized per ``(root, matcher_args)`` for :data:`_WALK_CACHE_TTL`, the same
    window the sibling walk caches use. The repeat this exists for is not a
    retry but a SEQUENCE: ``find_symbol(X)`` followed by ``read_symbol(X)``
    issues the identical query twice (measured: 12 subprocess spawns in a 30
    tool-call stub run, of which 2 were byte-identical rg invocations at ~29 ms
    each), and the tool-result cache cannot dedupe those because they arrive
    under different tool names.

    Only trustworthy answers are memoized. A None means "fall back to scanning
    every file", and caching that would keep a transient rg timeout suppressing
    the prefilter for a full TTL — recomputing it is both correct and cheap
    (the rg-missing path is one ``shutil.which``).
    """
    rg = shutil.which("rg")
    if not rg:
        return None
    # AFTER the rg check, deliberately. Reading the memo first would let a warm
    # entry answer for a machine where rg has since disappeared, turning the
    # documented "None = prefilter untrustworthy, scan everything" contract into
    # a stale set — caught by test_rg_missing_returns_none_for_full_scan_fallback.
    # The spawn is what this skips, and the spawn is downstream of here anyway.
    _key = (str(root), tuple(matcher_args))
    _hit = _RG_PY_FILTER_CACHE.get(_key)
    if _hit is not None and (_time.monotonic() - _hit[0]) < _WALK_CACHE_TTL:
        # Copy: callers own the result and the intersection at the call site
        # mutates it, which would otherwise poison every later hit.
        return set(_hit[1])
    try:
        # --no-ignore-vcs is deliberate: .gitignore bare-filename patterns
        # have hidden real source files before, and the prefilter inheriting
        # their blind spot would be a correctness regression (not a perf one).
        # The walk that filters the rg output already prunes _WALK_SKIP_DIRS
        # via _walk_should_skip_dir, so omitting --glob exclusions here costs
        # only a wasted descent — rg walks vendor trees whose every .py match
        # is discarded at the intersection.  Deriving --glob rules from the
        # SAME _WALK_SKIP_DIRS set cuts that waste without touching .gitignore.
        _skip_globs: list[str] = []
        for _d in _WALK_SKIP_DIRS:
            _skip_globs.extend(["--glob", f"!**/{_d}/**"])
        proc = subprocess.run(
            [rg, "--files-with-matches", "--type", "py",
             "--no-ignore-vcs", *_skip_globs, *matcher_args, "."],
            cwd=str(root), capture_output=True, text=True, timeout=10,
             check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("find_symbol prefilter: rg failed (%s) — scanning all files", e)
        return None
    # 0 = matches, 1 = no matches (a real, trustworthy empty answer), 2+ = error.
    if proc.returncode not in (0, 1):
        logger.debug(
            "find_symbol prefilter: rg exit %s — scanning all files", proc.returncode
        )
        return None
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        if line:
            # No .resolve(): the intersection at the call site compares
            # against _walk_py_files paths, which are built from the same
            # already-resolved *root* (repo_root/_resolve_search_root resolve
            # at the boundary). Both sides being plain joins of one resolved
            # root makes them string-identical; per-path realpath() here cost
            # ~1,150 lstat-walking calls per find_symbol on this repo.
            out.add(str(root / line))
    _capped_put(_RG_PY_FILTER_CACHE, _key, (_time.monotonic(), set(out)),
                _RG_PY_FILTER_MAX_ENTRIES)
    return out


# A definition line must literally contain one of these shapes — the complete
# definition surface of _extract_all_python_symbols/_ts_collect_all (class,
# def/async def, simple or chained assignment, annotated assignment). The
# assignment piece deliberately over-matches (kwargs `f(X=1)`, slices `a[X:]`,
# lambda params): a false positive only keeps a file the word-match set would
# have kept anyway; it can never drop one.
_DEF_PATTERNS: dict[str, str] = {
    "class": r"^\s*class\s+{t}\b",
    "function": r"^\s*(async\s+def|def)\s+{t}\b",
    "variable": r"\b{t}\s*=([^=]|$)|\b{t}\s*:",
}
_DEF_PATTERNS["method"] = _DEF_PATTERNS["function"]
_DEF_PATTERNS["constant"] = _DEF_PATTERNS["variable"]

_PLAIN_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _rg_py_files_defining(root: Path, token: str, kind: str) -> Optional[set[str]]:
    """Files that can DEFINE *token* (per kind), or None when not answerable.

    A strict subset of :func:`_rg_py_files_containing`: a definition line
    contains the token, but for widely-imported names the mention set is
    dominated by importers that provably hold no definition (measured on this
    repo: ToolRegistry — 82 mentioning files, 1 defining; dispatch 120 → 12).
    Parsing only the defining set is the entire win of this prefilter.

    Trust contract mirrors the word-match prefilter, with one addition: the
    caller must treat an EMPTY set as "fall back to the word-match set", not
    as a real answer. The regex cannot see definitions split across physical
    lines (`X \\` + `= 1`), so empty means "the regex saw nothing", while the
    extractor might still find something in a mentioning file. Non-empty is
    safe to use directly: a file whose only definition is regex-invisible
    while ANOTHER file defines the name visibly is the one shape this filter
    can drop, and no such construct survives review in practice.

    Non-identifier tokens (regex metacharacters, unicode names) return None —
    the fixed-string word-match path handles those.
    """
    if not _PLAIN_IDENTIFIER_RE.fullmatch(token):
        return None
    pattern = _DEF_PATTERNS.get(kind)
    if pattern is None:  # "any" or an unrecognized kind → union of all shapes
        pattern = "|".join(
            _DEF_PATTERNS[k] for k in ("class", "function", "variable")
        )
    return _rg_list_py_files(root, ["--", pattern.replace("{t}", token)])


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
    """Build function signature from a tree-sitter function_definition node.

    No local exception fence: failures propagate to the caller boundary in
    ``_extract_all_python_symbols`` (the TS→AST fallback), so a broken node
    shape degrades to full AST extraction instead of a silently empty
    signature.
    """
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


def _ts_extract_decorators(node, code_bytes: bytes) -> list[str]:
    """Extract decorator names from a decorated_definition or function node.

    No local exception fence — a broken node shape propagates to the
    ``_extract_all_python_symbols`` TS→AST fallback boundary.
    """
    decs: list[str] = []
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
    return decs


def _ts_extract_docstring(node, code_bytes: bytes) -> Optional[str]:
    """Extract docstring from a function/class tree-sitter node.

    Two independent shape problems are handled here:

    1. ``expression_statement`` has NO ``expression`` named field in the Python
       grammar, so ``child_by_field_name("expression")`` returned None and this
       function never extracted anything under EITHER grammar (verified against
       both). The docstring is ``children[0]``. The same trap is already noted at
       the ``_walk_outline`` site below.
    2. The wrapper may not be there at all: standalone ``tree-sitter-python``
       gives ``block → expression_statement → string``, while the
       ``tree-sitter-language-pack`` bundle gives ``block → string``.

    No local exception fence — a broken node shape propagates to the
    ``_extract_all_python_symbols`` TS→AST fallback boundary.
    """
    body = node.child_by_field_name("body")
    if body and body.children:
        first = body.children[0]
        if first.type in ("expression_statement", "string"):
            expr = first if first.type == "string" else (
                first.children[0] if first.children else None
            )
            if expr is not None and expr.type == "string":
                text = _ts_get_node_text(code_bytes, expr)
                # Strip quotes
                if text.startswith(('"""', "'''")):
                    text = text[3:-3]
                elif text.startswith(("'", '"')):
                    text = text[1:-1]
                return text[:150] or None
    return None


def _ts_extract_class_bases(node, code_bytes: bytes) -> list[str]:
    """Extract base class names from a class_definition node.

    No local exception fence — a broken node shape propagates to the
    ``_extract_all_python_symbols`` TS→AST fallback boundary.
    """
    bases: list[str] = []
    super_node = node.child_by_field_name("superclass")
    if super_node:
        for child in super_node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type in ("identifier", "attribute", "call"):
                        bases.append(_ts_get_node_text(code_bytes, arg))
                    elif arg.type == "comment":
                        continue
    return bases


def _ts_collect_class_methods(node, code_bytes: bytes) -> list[str]:
    """Collect method names from a class_definition's body.

    No local exception fence — a broken node shape propagates to the
    ``_extract_all_python_symbols`` TS→AST fallback boundary.
    """
    methods: list[str] = []
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    methods.append(_ts_get_node_text(code_bytes, name_node))
    return methods


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

# ── per-file symbol cache LRU bounds ────────────────────────────────────────
# _py_file_cache, _ts_file_cache and _go_file_cache are plain dicts keyed on
# the resolved abs path. Without a bound, a long session touching many files
# would grow memory without limit (each entry holds a per-file symbol map).
# All three are capped with the same LRU discipline: the dict is
# insertion-ordered, a cache hit moves the entry to the MRU end
# (_cache_get_lru), and a put evicts the oldest entry once the cap is exceeded
# (_cache_put_lru).
_PY_FILE_CACHE_MAX_ENTRIES = 512
_TS_FILE_CACHE_MAX_ENTRIES = 256  # TS entries carry heavier module analysis
_GO_FILE_CACHE_MAX_ENTRIES = 512  # Go entries are light method tuples
# _realpath_memo shares the same LRU discipline (see _cache_get_lru /
# _cache_put_lru). Entries are two small strings, so the cap is far larger than
# the file caches; the >4096 full clear it replaces threw away every memoised
# path at once, while LRU eviction drops only the least-recently-used key.
_REALPATH_MEMO_MAX_ENTRIES = 4096


def _cache_get_lru(cache: dict, key: str):
    """dict.get() that refreshes LRU recency — a hit moves the entry to MRU.

    Plain dicts preserve insertion order but have no ``move_to_end``, so the
    recency refresh is a delete + re-insert (position resets to the end).
    """
    val = cache.get(key)
    if val is not None:
        del cache[key]
        cache[key] = val
    return val


def _cache_put_lru(cache: dict, key: str, value, max_entries: int) -> None:
    """Insert into a bounded LRU cache, evicting the oldest entry over the cap.

    ``pop(key, None)`` before assignment is a no-op for a fresh key and
    refreshes recency when a stale entry is rebuilt in place. ``next(iter(
    cache))`` is the least-recently-used entry because hits and puts both move
    their key to the MRU end.
    """
    # Delegate to the shared FIFO/LRU eviction SSOT (``_capped_put`` pop-then-
    # assign = move-to-end, matching this helper's documented LRU contract).
    # ``_capped_put`` is strictly more defensive: it guards ``next(iter())``
    # against free-threaded dict resize (RuntimeError/StopIteration), which the
    # previous bare ``pop(next(iter(cache)))`` did not.
    _capped_put(cache, key, value, cap=max_entries)


class SymbolSearcher:
    """
    Tree-sitter + AST hybrid symbol search for Python; rg-based for other languages.
    All paths are validated to stay within repo_root.
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        # ── per-file Python parse cache ────────────────────────────────────
        # Key: abs file path -> (mtime, full_symbol_map)
        # LRU-capped at _PY_FILE_CACHE_MAX_ENTRIES (see _cache_put_lru).
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
        # LRU-capped at _TS_FILE_CACHE_MAX_ENTRIES (see _cache_put_lru).
        # symbol kind TSSemanticTracer exposes. analyze_core costs ~15ms/file
        # and is pure function of content, so caching it removes the dominant
        # cost of repeated find_symbol over TS/JS repos.
        self._ts_file_cache: dict[str, tuple] = {}
        # ── Go per-file class→methods cache ──────────────────────────────────
        # Key: abs file path -> (mtime, {class_name -> [(name, start, end)]})
        # LRU-capped at _GO_FILE_CACHE_MAX_ENTRIES (see _cache_put_lru).
        # _find_in_go used to read_text + re-parse the file (tree-sitter walk)
        # on every dotted lookup; the map is built once per signature and
        # lookups become O(1) dict gets. Invalidation rides the same
        # invalidate_file_caches() post-write path as the Python/TS maps.
        self._go_file_cache: dict[str, tuple] = {}
        # Memoised os.path.realpath for the three per-file caches above. Both are
        # keyed on the RESOLVED path so invalidate_file_caches() can pop in O(1)
        # instead of scanning: the write path and the search path reach the same
        # file by different spellings (repo-relative vs absolute, and on macOS
        # /var vs the resolved /private/var), and comparing those by realpath at
        # invalidation time cost ~3 ms per write over a 200-entry cache —
        # measured, and paid on the COMMON path since most writes touch files
        # that were never symbol-searched. Resolving once at key construction
        # moves that syscall to the cache-miss path, where a full file parse
        # already dominates it.
        # LRU-capped at _REALPATH_MEMO_MAX_ENTRIES (see _cache_put_lru). The
        # old bound cleared the whole memo past 4096 entries; LRU eviction
        # instead drops only the least-recently-used key.
        self._realpath_memo: dict[str, str] = {}

    def _cache_key(self, file_path) -> str:
        """Resolved, memoised cache key for the per-file symbol caches."""
        raw = str(file_path)
        hit = _cache_get_lru(self._realpath_memo, raw)
        if hit is not None:
            return hit
        try:
            resolved = os.path.realpath(raw)
        except OSError:
            resolved = raw
        _cache_put_lru(self._realpath_memo, raw, resolved,
                       _REALPATH_MEMO_MAX_ENTRIES)
        return resolved

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
                    # parent_defs[0] comes from the same cached per-file symbol map
                    # _find_in_python_cached just used, so checking its decorators
                    # replaces a whole re-read + re-parse of the file (former
                    # _is_dataclass helper).
                    if search_name == "__init__" and parent_defs[0].kind == "class":
                        _is_dataclass = any(
                            d.startswith(("dataclass", "@dataclass"))
                            for d in (parent_defs[0].decorators or [])
                        )
                        if _is_dataclass:
                            logger.info(
                                "find_symbol: %s.__init__ not found — @dataclass detected, "
                                "returning class definition instead",
                                parent_class,
                            )
                            # Return the class itself so modify_symbol can edit its body
                            return [parent_defs[0]]

        # ── Non-Python probe, started early and collected below ──────────────
        # The two prefilters this function runs are independent — both are a
        # function of (root, search_name) alone — but they were sequential with
        # the whole Python parse between them: rg #1 (definition patterns, ~14 ms
        # here), parse, rg #2 (does any non-Python file even mention the name,
        # ~8 ms). Starting #2 now overlaps it with both, so its cost comes off
        # the wall clock rather than adding to it.
        #
        # Only for kind="any", which is the default and the one kind that reaches
        # the collection point unconditionally (see the branch below: every other
        # kind gets there only when Python found nothing). Speculating for those
        # would spawn an rg whose answer is usually discarded — the point is to
        # move work already certain to happen, not to add any.
        _nonpy_probe = None
        if kind == "any":
            try:
                _nonpy_probe = _shared_pool.submit(
                    self._nonpy_index_worth_building, root, search_name
                )
            except RuntimeError:
                # Pool shut down (interpreter teardown) — fall back to the
                # inline call at the collection point.
                _nonpy_probe = None

        # ── Python AST scan ──────────────────────────────────────────────────
        py_files = [root] if root.is_file() and LanguageId.from_path(str(root)) == LanguageId.PYTHON else _walk_py_files(root)
        # Narrow to files that actually contain the name before parsing any of
        # them. Intersecting (rather than using rg's list directly) keeps
        # _walk_py_files as the sole authority on which files are in scope, so
        # this can only ever remove files that provably cannot hold the
        # definition. A None result means the prefilter is untrustworthy —
        # scan everything, exactly as before.
        if len(py_files) > 1:
            # Definition-pattern pass first: for widely-imported names the
            # word-match set is dominated by importers (82 files for
            # ToolRegistry on this repo, 1 of which defines it), and every
            # candidate is tree-sitter parsed. Empty-or-None falls back to
            # the word-match set, so this can only skip files whose text
            # provably contains no definition shape the extractor records.
            _candidates = _rg_py_files_defining(root, search_name, kind)
            if not _candidates:
                _candidates = _rg_py_files_containing(root, search_name)
            if _candidates is not None:
                # str(p), not str(p.resolve()): both sides are plain joins of
                # the same resolved root (see _rg_list_py_files), and
                # resolving every walked file cost ~1,150 realpath calls per
                # lookup — most of find_symbol's non-parse overhead.
                py_files = [p for p in py_files if str(p) in _candidates]
        for pf in py_files:
            results.extend(self._find_in_python_cached(pf, search_name, kind))
            if len(results) >= _cfg.counts.SEARCH_RESULTS_CAP:
                break

        # ── TS/JS rich scan via TSSemanticTracer ─────────────────────────────
        from config import MULTILANG_SYMBOL_SEARCH as _ML_SYM  # config defines it — no fallback needed
        if _ML_SYM and not results:
            if root.is_file() and LanguageId.from_path(str(root)) in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
                results.extend(self._find_in_ts_js(root, search_name, kind))
            elif root.is_dir():
                for _ts_count, _tf in enumerate(_walk_ts_js_files(root), start=1):
                    results.extend(self._find_in_ts_js(_tf, search_name, kind))
                    if len(results) >= _cfg.counts.SEARCH_RESULTS_CAP or _ts_count >= _MAX_TS_FILES:
                        break

        # ── Provider-aware search for registered languages (persistent index)
        if not results or kind == "any":
            registry = LanguageRegistry.instance()
            # NOTE: this is a property of the STATIC provider registry, not of
            # the repo — the built-in providers always include non-Python ones,
            # so it is always True and filters nothing. Repo-level "is there
            # anything to find here" is what _nonpy_index_worth_building answers.
            has_nonpy_provider = any(
                p.language_id().value not in ("python", "typescript", "javascript")
                for p in set(registry._providers.values())
            )
            # Collect the probe started before the Python scan. Guarded by
            # has_nonpy_provider so the short-circuit the `and` used to give is
            # preserved — that flag is documented above as always True today, but
            # this must not become the reason it is.
            #
            # A probe failure must not lose the branch, so it falls back to the
            # inline call — the same answer, just without the overlap.
            #
            # The wait is also capped (_NONPY_PROBE_TIMEOUT_SEC): the probe was
            # submitted to the same _shared_pool a dispatch may be running on
            # (P1), so under pool saturation a queued probe could otherwise be
            # awaited forever — N such waiters would wedge the whole pool.
            # On timeout we cancel the queued future (best-effort — a running
            # probe is left to finish and its answer discarded) and retry
            # inline, so saturation degrades to a stall, never a deadlock.
            _worth = False
            if has_nonpy_provider:
                if _nonpy_probe is not None:
                    try:
                        _worth = _nonpy_probe.result(timeout=_NONPY_PROBE_TIMEOUT_SEC)
                    except Exception as e:
                        _nonpy_probe.cancel()  # best-effort: don't leave it queued
                        logger.debug("nonpy probe failed (%s) — retrying inline", e)
                        _worth = self._nonpy_index_worth_building(root, search_name)
                else:
                    _worth = self._nonpy_index_worth_building(root, search_name)
            if _worth:
                # The persistent index already aggregates all non-Python
                # providers in one rg pass; filter to this name/kind.
                _idx = self._nonpy_index_for(root)
                # Filter loop kept (PERF401 rejected): the kind-matching
                # predicate below spans ~50 lines with per-language comments;
                # folding it into a list comprehension would bury that
                # documentation for a micro-optimization the rule itself
                # calls negligible.
                for d in _idx.get(search_name, []):
                    if (
                        (
                            kind in ("function", "method", "any")
                            and d.kind
                            in (
                                "function",
                                "async_function",
                                "method",
                            )
                        )
                        or (
                            kind in ("variable", "any")
                            and d.kind
                            in (
                                # Variable/constant declarations across languages.
                                # "variable" covers Go var/short_var, "constant" covers
                                # Go const + Rust const/static, "css_variable" covers
                                # CSS custom properties (--name).
                                "variable",
                                "constant",
                                "css_variable",
                            )
                        )
                        or (
                            kind in ("class", "any")
                            and d.kind
                            in (
                                # All type/aggregate-like declarations across languages.
                                # "class"-group covers: OOP classes, interfaces, type
                                # aliases, enums, CSS selectors (NOT custom properties —
                                # those are in the variable group above), plus the
                                # struct/trait/record/module/protocol kinds emitted by the
                                # Rust/C#/Ruby/PHP/Swift providers & AST path.
                                # "namespace" covers Ruby modules / AST-normalized
                                # module-kind symbols.
                                "class",
                                "interface",
                                "type",
                                "enum",
                                "struct",
                                "trait",
                                "record",
                                "module",
                                "protocol",
                                "namespace",
                                "css_class",
                                "css_id",
                            )
                        )
                        or kind == "any"
                    ):
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
                check=False,
            )
            refs: list[SymbolRef] = []
            for line in (proc.stdout or "").splitlines()[:80]:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    rel = str(Path(parts[0]).relative_to(self.repo_root))
                    lineno = int(parts[1])
                    ctx = parts[2].strip()
                    stripped = ctx.lstrip()
                    if not include_definitions and _is_definition_line(parts[0], stripped, name):
                        continue
                    col = ctx.find(name)
                    refs.append(SymbolRef(file=rel, line=lineno, col=max(col, 0), context=ctx[:120]))

            # BM25 ranking: sort by relevance to the symbol name before capping.
            # Treats each reference's file+context as a pseudo-document and scores
            # against the symbol name — so references with richer surrounding context
            # (more identifier tokens matching the name) rank higher.
            if len(refs) > 1:
                global _TOKENIZER
                if _TOKENIZER is None:
                    _TOKENIZER = CodeTokenizer()
                _tok = _TOKENIZER
                _qtokens = _tok.tokenize(name)
                if _qtokens:
                    # Pseudo-documents ranked by bm25_rank (single-sourced in
                    # agent/bm25.py — this setup used to be a copy of the
                    # read_tools twin; scores are bit-identical).
                    _docs = [f"{r.file}:{r.context}" for r in refs]
                    _scores = bm25_rank(_qtokens, [_tok.tokenize(d) for d in _docs])
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

        # NOTE: no "read_guidance" key. It used to be built here on every call
        # and no caller ever read it — both consumers cherry-pick specific keys
        # (read_tools takes subclasses/reference_count/…, agent_tools takes
        # signature/line). Its text also told the model to read with
        # `bash (cat -n)`, which omits read_file's │N│ indent gutter, so the day
        # someone serialised this dict wholesale the agent would have been
        # steered off the tool that exists to prevent old_string mismatches.
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
            # Gate BEFORE the language dispatch: every branch below reads and
            # parses the whole file, and this is the entry point read_file's
            # over-cap guidance calls — the path that spent 13.31 s / 1.65 GB
            # to decorate a refusal message. See _too_big_to_parse_inproc.
            if _too_big_to_parse_inproc(p.stat().st_size, p):
                return []
            rel = str(p.relative_to(self.repo_root))
        except (OSError, RuntimeError, ValueError):  # vanished file / symlink loop / path outside root
            return []  # non-critical — never block execution

        if LanguageId.from_path(str(p)) == LanguageId.PYTHON:
            return self._outline_python(p, rel)
        if LanguageId.from_path(str(p)) in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
            from config import MULTILANG_OUTLINE as _ML_OL  # config defines it — no fallback needed
            if _ML_OL:
                _ts_outline = self._outline_ts_js(p, rel)
                if _ts_outline:
                    return _ts_outline
            _ast_outline = self._outline_treesitter(p, rel)
            if _ast_outline:
                return _ast_outline
            return self._outline_ripgrep(p, rel)
        # AST-first: tree-sitter gives an accurate (start, end) per symbol
        # and handles modifiers/annotations structurally. Fall back to the
        # provider-regex rg path only when tree-sitter is unavailable or the
        # grammar is not installed (e.g. Kotlin before tree_sitter_kotlin).
        _ast_outline = self._outline_treesitter(p, rel)
        if _ast_outline:
            return _ast_outline
        return self._outline_ripgrep(p, rel)

    def _outline_python(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """Outline for a single Python file, derived from the shared per-file
        symbol map (the same map ``_find_in_python_cached`` builds via
        :meth:`_python_symbol_map`).

        The map is the single source of truth, so outline and lookup share one
        parse per file: whichever runs first warms ``_py_file_cache`` and the
        other reuses it — a ``find_symbol`` immediately followed by
        ``get_file_outline`` (or vice versa) parses the file once instead of
        twice. Previously the outline ran its own dedicated tree-sitter walk
        that ignored the cache entirely.

        Returns top-level symbols only (``parent_class is None``), sorted by
        line — the same contract as the former dedicated walk, including the
        ``end_line`` extent the caller reads ranges with. One deliberate
        additive change: class entries now carry ``decorators`` (populated by
        ``_extract_all_python_symbols``), matching what find_symbol exposes;
        and in the AST fallback, docstrings are truncated at 150 chars and
        method lists at 25 entries, exactly as the map already stores them.
        """
        full_map = self._python_symbol_map(file_path)
        out = [
            d for defs in full_map.values() for d in defs
            if d.parent_class is None
        ]
        out.sort(key=lambda s: s.line)
        return out

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
        if not _ts_language_available(lang_id):
            return []  # grammar mapped but not installed → caller falls back
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            syms = _ts_find_all_symbols(content, lang_id)
        except Exception as _err:
            logger.debug(
                "[symbol-search] tree-sitter outline failed for %s (%s) — "
                "falling back to ripgrep outline", file_path, _err,
            )
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
            with contextlib.suppress(AttributeError, TypeError, OSError, subprocess.SubprocessError):
                # OSError covers rg-absent FileNotFoundError (rg is an OPTIONAL dep,
                # see pyproject [search]); SubprocessError covers TimeoutExpired on
                # a hung rg — both degrade gracefully: skip this pattern and return
                # whatever the other patterns found (possibly empty).
                # --with-filename is mandatory: with a single FILE argument, rg omits
                # the path prefix and emits "lineno:content", which would collapse the
                # 3-part split below (path:lineno:content) and silently drop every
                # match — see `_index_via_treesitter` which uses the same flag.
                proc = subprocess.run(
                    ["rg", "--no-heading", "--with-filename", "--line-number", "-m", "50", pat, str(file_path)],
                    capture_output=True, text=True, timeout=5,
                    check=False,
                )
                for line in (proc.stdout or "").splitlines():
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        logger.debug("symbol_search: unparsable rg line number in %r", line)
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

        results.sort(key=lambda s: s.line)
        return results

    def _resolve_search_root(self, search_path: Optional[str]) -> Optional[Path]:
        if not search_path:
            return self.repo_root
        with contextlib.suppress(OSError, RuntimeError):  # resolve() on broken path / symlink loop
            p = (self.repo_root / search_path).resolve()
            if p.is_relative_to(self.repo_root):
                return p
        return None

    def index_was_truncated(self, search_path: Optional[str] = None) -> bool:
        """True if the most recent file walk for *search_path* hit the cap.

        find_symbol walks the candidate file set lazily; on a MISS the result
        is only authoritative if the walk was complete. A truncated walk means
        the symbol may simply live in an un-indexed file. The caller (see
        read_tools._tool_find_symbol) uses this to annotate the empty result so
        the agent does not wrongly conclude "symbol does not exist".

        Each cache is queried with the cap THIS module walks at, not with the
        flag alone: a higher-cap caller (vulture_scanner uses 4000) can leave a
        complete cache entry that is nonetheless sliced down for our 3000, and
        a flag-only reading would call that shortened list complete.
        """
        root = self._resolve_search_root(search_path)
        if root is None:
            return False
        return (
            _shared_walk_truncated_for(root, _SHARED_PY_WALK_CACHE, _MAX_PY_FILES)
            or _shared_walk_truncated_for(root, _SHARED_TS_WALK_CACHE, _MAX_TS_FILES)
        )

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
        except OSError:
            return out

        # ── Primary: tree-sitter (collect every symbol) ──────────────────
        # Comment-wall files are superlinear in tree-sitter (see
        # _python_ts_parse_too_costly) — the AST path below is identical.
        if _HAS_TS and not _python_ts_parse_too_costly(source):
            with contextlib.suppress(Exception):  # fall through to AST
                code_bytes = source.encode("utf-8")
                tree = _ts_parse_to_tree(source, "python")
                if tree is not None:
                    self._ts_collect_all(tree.root_node, code_bytes, rel, "", out)
                    return out

        # ── Fallback: AST ──────────────────────────────────────────────────
        try:
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, TypeError, AttributeError):
            # Syntax-broken file: ast cannot extract anything, but the
            # error-tolerant tree-sitter parse still finds symbols — use it
            # as the last resort (the slow-parse case is exactly the kind of
            # file that is unlikely to be broken, so this stays rare). The
            # guard applies here too: a broken comment wall costs seconds
            # for symbols the brokenness already makes untrustworthy.
            if _HAS_TS and not _python_ts_parse_too_costly(source):
                with contextlib.suppress(Exception):
                    code_bytes = source.encode("utf-8")
                    tree = _ts_parse_to_tree(source, "python")
                    if tree is not None:
                        self._ts_collect_all(tree.root_node, code_bytes, rel, "", out)
            return out

        # Single-pass walk: thread parent_class as a parameter (O(n)) instead of
        # the O(n²) nested ast.walk that recomputes the enclosing class per node.
        def _walk_body(body: list, parent_class: str) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    full = f"{parent_class}.{node.name}" if parent_class else node.name
                    # record the class itself
                    self._ast_add_class(out, node, rel, parent_class)
                    # record its direct members under `full`
                    for child in node.body:
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
        decs = [_unparse(d) for d in node.decorator_list if d] or None
        out.setdefault(node.name, []).append(SymbolDef(
            file=rel, line=node.lineno, kind="class", name=node.name,
            bases=bases, methods=methods[:25] or None, docstring=doc,
            decorators=decs,
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

        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cls_name = _ts_get_node_text(code_bytes, name_node) if name_node else ""
            if cls_name:
                bases = _ts_extract_class_bases(node, code_bytes) or None
                methods = _ts_collect_class_methods(node, code_bytes) or None
                doc = _ts_extract_docstring(node, code_bytes)
                # Decorated classes appear as class_definition under a
                # decorated_definition parent — pull decorators from there so
                # find_symbol's @dataclass fallback can rely on the cached map.
                decs = None
                _parent = getattr(node, "parent", None)
                if _parent is not None and _parent.type == "decorated_definition":
                    decs = _ts_extract_decorators(_parent, code_bytes) or None
                out.setdefault(cls_name, []).append(SymbolDef(
                    file=rel, line=node.start_point.row + 1,
                    kind="class", name=cls_name,
                    bases=bases, methods=methods, docstring=doc, decorators=decs,
                    end_line=node.end_point.row + 1,
                    parent_class=parent_class or None,
                ))
            new_parent = f"{parent_class}.{cls_name}" if parent_class else cls_name
            for child in node.children:
                self._ts_collect_all(child, code_bytes, rel, new_parent, out)
            return

        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    self._ts_collect_all(child, code_bytes, rel, parent_class, out)
            return

        if node.type in ("expression_statement", "assignment"):
            # Bare `assignment` is the language-pack grammar's shape for the same
            # statement the standalone grammar wraps in `expression_statement`.
            expr = node if node.type == "assignment" else (
                node.children[0] if node.children else None
            )
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

    def _python_symbol_map(self, file_path: Path) -> dict[str, list[SymbolDef]]:
            """Return the per-file ``{name -> [SymbolDef]}`` map, building and
            caching it on miss.

            Shared by ``_find_in_python_cached`` and ``_outline_python`` so both
            warm the same ``_py_file_cache``: a find_symbol + get_file_outline pair
            on one file parses it once instead of twice, in either order. The
            key/signature semantics are identical to the former inline code in
            ``_find_in_python_cached``: ``(st_mtime_ns, st_size)`` signature plus
            explicit per-path drops via :meth:`invalidate_file_caches` for the
            agent's own writes. The cache is LRU-capped at
            ``_PY_FILE_CACHE_MAX_ENTRIES`` — the least-recently-used file is
            evicted on overflow, so long sessions cannot grow memory without
            limit.
            """
            key = self._cache_key(file_path)
            try:
                _st = file_path.stat()
                sig = (_st.st_mtime_ns, _st.st_size)
            except OSError:
                return {}
            if _too_big_to_parse_inproc(_st.st_size, file_path):
                return {}
            cached = _cache_get_lru(self._py_file_cache, key)
            if cached is not None and cached[0] == sig:
                return cached[1]
            try:
                rel = str(file_path.relative_to(self.repo_root))
            except ValueError:
                rel = str(file_path)
            full_map = self._extract_all_python_symbols(file_path, rel)
            _cache_put_lru(self._py_file_cache, key, (sig, full_map),
                           _PY_FILE_CACHE_MAX_ENTRIES)
            return full_map
    def _find_in_python_cached(
        self, file_path: Path, name: str, kind: str,
        parent_class: str = "",
    ) -> list[SymbolDef]:
        """Signature-cached wrapper around per-file full symbol extraction.

        Equivalent to _find_in_python(file_path, name, kind, parent_class) but
        amortizes parsing: the file is parsed once and every symbol is cached,
        so subsequent lookups for any name in the same file are O(1).

        The key is ``(st_mtime_ns, st_size)``, matching parse_cache._stat_key
        and the insights_manager signature family. A bare ``st_mtime`` was not
        enough: on a filesystem with coarse mtime granularity (container bind
        mounts, NFS/SMB) — or on any tree whose mtimes were restored by
        tar/rsync -t/cp -p — two different contents share one stat value, and
        this cache then hands back pre-edit LINE NUMBERS for a file the agent
        just wrote (measured: target reported at line 5 while it had moved to
        line 9, and a newly added symbol was invisible).

        Note this is the belt, not the suspenders: a same-size edit on such a
        filesystem still collides. The guarantee for the agent's own writes
        comes from :meth:`invalidate_file_caches`, called by the post-write
        invalidation path — that one needs no filesystem assumptions.
        """
        full_map = self._python_symbol_map(file_path)

        defs = full_map.get(name, [])
        # Apply kind + parent_class filter (same semantics as _find_in_python)
        out: list[SymbolDef] = []
        for d in defs:
            # parent_class filter: empty means "any scope"; otherwise exact match.
            if parent_class and (d.parent_class or "") != parent_class:
                continue
            if (
                (kind in ("function", "method", "any") and d.kind in ("function", "async_function", "method"))
                or (kind in ("class", "any") and d.kind == "class")
                or (kind in ("variable", "constant", "any") and d.kind == "constant")
                or kind == "any"
            ):
                out.append(d)
        return out

    # ── Go dotted name resolution via GoSyntaxProvider ─────────────────────
    def _go_class_methods_map(
        self, file_path: Path,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """Return ``{class_name: [(method_name, start_line, end_line)]}`` for a
        Go file, ``(st_mtime_ns, st_size)``-cached in ``_go_file_cache``.

        Same signature rationale as :meth:`_find_in_python_cached` — see there
        for why a bare ``st_mtime`` let a just-edited file answer from cache.
        The batch provider call (``find_all_class_methods``) parses the file
        once regardless of how many structs it has, so repeated dotted
        lookups like ``find_symbol("TodoList.Add")`` on one file cost one
        parse total instead of one read + tree-sitter walk per call.
        """
        key = self._cache_key(file_path)
        try:
            _st = file_path.stat()
            sig = (_st.st_mtime_ns, _st.st_size)
        except OSError:
            return {}
        if _too_big_to_parse_inproc(_st.st_size, file_path):
            return {}
        cached = _cache_get_lru(self._go_file_cache, key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {}
        provider = LanguageRegistry.instance().get(str(file_path))
        if provider is None:
            return {}
        try:
            methods_map = provider.find_all_class_methods(source)
        except Exception as _err:
            logger.debug(
                "[symbol-search] Go class-methods parse failed for %s (%s)",
                file_path, _err,
            )
            return {}
        _cache_put_lru(self._go_file_cache, key, (sig, methods_map),
                       _GO_FILE_CACHE_MAX_ENTRIES)
        return methods_map

    def _find_in_go(
        self, file_path: Path, name: str, kind: str,
        parent_class: str = "",
    ) -> list[SymbolDef]:
        """Find a Go method scoped to a specific struct.

        When ``find_symbol("TodoList.Add")`` is called, *parent_class* is
        ``"TodoList"`` and *name* is ``"Add"``.  The per-file class→methods
        map (:meth:`_go_class_methods_map`) is parsed once and cached, and
        the method list for *parent_class* is a dict lookup.
        """
        results: list[SymbolDef] = []
        methods = self._go_class_methods_map(file_path).get(parent_class, [])
        for mname, mstart, mend in methods:
            if mname != name:
                continue
            try:
                rel = str(file_path.relative_to(self.repo_root))
            except ValueError:
                rel = str(file_path)
            # Read the method signature line for context (lazy: only on a match).
            # P26-3: window-read ONE line instead of the whole file — the
            # class-methods map above already parsed the file, and re-reading
            # it in full for a single signature line materialized huge
            # generated Go files (e.g. *.pb.go) on every lookup.
            # read_line_window is the shared P25-1 primitive (O(1) memory,
            # \n-only line semantics matching the AST line model).
            sig_line = ""
            try:
                window = read_line_window(file_path, mstart - 1, 1)
                sig_line = window[0].strip() if window else ""
            except Exception as e:
                logger.debug("Go signature line read failed for %s: %s", file_path, e)
            results.append(SymbolDef(
                file=rel, line=mstart, kind="method",
                name=name, signature=sig_line,
                end_line=mend,
                parent_class=parent_class,
            ))
            break  # only the first match

        return results

    # ── TS/JS rich symbol search via TSSemanticTracer ──────────────────────
    # `_MAX_TS_FILES` is the module-level constant; _walk_ts_js_files already
    # uses it, so find_symbol's TS/JS cap reads the same SSOT rather than a class copy.

    # Per-file TS/JS module cache: path -> ((mtime_ns, size), {name -> [SymbolDef]}).
    # analyze_core is ~15ms/file and pure function of content, so caching the
    # full extracted symbol map removes the dominant cost of repeated
    # find_symbol AND get_file_outline over TS/JS repos (was ~700ms for 47
    # files, now ~0ms warm). The outline derives from the same map
    # (_outline_ts_js), so a find+outline pair parses once — Python parity.
    #
    # This is the FOURTH cache holding non-Python symbol state, alongside the
    # two invalidate_nonpy_caches() drops and the Go class→methods map
    # (_go_file_cache). It is per-file and signature-keyed rather than TTL'd,
    # so it is cleared by invalidate_file_caches() on the post-write path
    # instead.

    def _ts_module_map(self, file_path: Path) -> dict[str, list[SymbolDef]]:
        """Return {name -> [SymbolDef]} for a TS/JS file, (mtime_ns,size)-cached.

        Same signature rationale as :meth:`_find_in_python_cached` — see there
        for why a bare ``st_mtime`` let a just-edited file answer from cache.
        """
        key = self._cache_key(file_path)
        try:
            _st = file_path.stat()
            sig = (_st.st_mtime_ns, _st.st_size)
        except OSError:
            return {}
        if _too_big_to_parse_inproc(_st.st_size, file_path):
            return {}
        cached = _cache_get_lru(self._ts_file_cache, key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        full_map: dict[str, list[SymbolDef]] = {}
        with contextlib.suppress(Exception):  # non-critical — never block execution
            from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

            from ..languages.models import LanguageId as _LID  # noqa: N814 — private lazy-import alias
            content = file_path.read_text(encoding="utf-8", errors="replace")
            rel = str(file_path.relative_to(self.repo_root))
            lang_str = "typescript" if _LID.from_path(str(file_path)) == _LID.TYPESCRIPT else "javascript"
            tracer = TSSemanticTracer(language=lang_str)
            module = tracer.analyze_core(content, str(file_path))
            full_map = self._ts_extract_all(module, rel)
        _cache_put_lru(self._ts_file_cache, key, (sig, full_map),
                       _TS_FILE_CACHE_MAX_ENTRIES)
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
                end_line=_ts_node_end_line(fn),
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
                end_line=_ts_node_end_line(cls),
            ))
            for method in cls.methods:
                msig = _build_ts_method_signature(cls.name, method)
                out.setdefault(method.name, []).append(SymbolDef(
                    file=rel,
                    line=method.meta.start_line if method.meta else method.start_line,
                    kind="method", name=f"{cls.name}.{method.name}", signature=msig,
                    end_line=_ts_node_end_line(method),
                ))
        for iface in module.interfaces:
            out.setdefault(iface.name, []).append(SymbolDef(
                file=rel,
                line=iface.meta.start_line if iface.meta else iface.start_line,
                kind="interface", name=iface.name,
                methods=iface.methods[:25] or None,
                end_line=_ts_node_end_line(iface),
            ))
        for ta in module.type_aliases:
            out.setdefault(ta.name, []).append(SymbolDef(
                file=rel,
                line=ta.meta.start_line if ta.meta else ta.start_line,
                kind="type", name=ta.name,
                end_line=_ts_node_end_line(ta),
            ))
        for en in module.enums:
            out.setdefault(en.name, []).append(SymbolDef(
                file=rel,
                line=en.meta.start_line if en.meta else en.start_line,
                kind="enum", name=en.name,
                end_line=_ts_node_end_line(en),
            ))
        for var in module.variables:
            sig = f"{var.decl_kind} {var.name}"
            if var.type_ref:
                sig += f": {var.type_ref.name}"
            out.setdefault(var.name, []).append(SymbolDef(
                file=rel,
                line=var.meta.start_line if var.meta else var.start_line,
                kind="variable", name=var.name, signature=sig,
                end_line=_ts_node_end_line(var),
            ))
        return out

    def _find_in_ts_js(self, file_path: Path, name: str, kind: str) -> list[SymbolDef]:
        """TSSemanticTracer-based rich symbol search for TS/JS files (cached)."""
        full_map = self._ts_module_map(file_path)
        defs = full_map.get(name, [])
        out: list[SymbolDef] = []
        # Filter loop kept (PERF401 rejected): same rationale as the
        # non-Python index loop — multi-line kind predicate with per-language
        # comments, comprehension fold would hurt readability for negligible
        # gain.
        for d in defs:
            if (
                (kind in ("function", "method", "any") and d.kind in ("function", "async_function", "method"))
                or (kind in ("class", "interface", "any") and d.kind in ("class", "interface"))
                or (kind in ("type", "any") and d.kind == "type")
                or (kind in ("enum", "any") and d.kind == "enum")
                or (kind in ("variable", "any") and d.kind == "variable")
                or kind == "any"
            ):
                out.append(d)
        return out

    def _outline_ts_js(self, file_path: Path, rel: str) -> list[SymbolDef]:
        """Outline for a single TS/JS file, derived from the shared per-file
        symbol map (the same map ``_find_in_ts_js`` builds via
        :meth:`_ts_module_map`).

        The map is the single source of truth, so outline and lookup share one
        parse per file: whichever runs first warms ``_ts_file_cache`` and the
        other reuses it — a ``find_symbol`` immediately followed by
        ``get_file_outline`` (or vice versa) parses the file once instead of
        twice. Previously the outline ran its own dedicated TSSemanticTracer
        walk that ignored the cache entirely. This is the TS/JS side of the
        Python parity established by ``_outline_python``.

        Returns top-level symbols only (``kind != "method"``), sorted by line —
        the same contract as the former dedicated walk, including the
        ``end_line`` extent the caller reads ranges with. One deliberate
        additive change: class entries now carry ``bases``, matching what
        find_symbol exposes (the same spirit as the Python outline carrying
        decorators).

        On any failure the map is empty and this returns ``[]``; the caller's
        fallback chain (``_outline_treesitter`` → ``_outline_ripgrep``) then
        takes over, exactly as it does when the tracer yields nothing.
        """
        full_map = self._ts_module_map(file_path)
        out = [
            d for defs in full_map.values() for d in defs
            if d.kind != "method"
        ]
        out.sort(key=lambda s: s.line)
        return out

    def _rg_path_to_rel(self, raw: str) -> str:
        """Normalize an rg-emitted path to a repo-relative string.

        rg runs with cwd=repo_root and emits paths like './foo/bar.go' or
        'foo/bar.go'. A bare relative_to(repo_root) fails because the path is
        not absolute. This handles all three shapes (absolute, './'-prefixed,
        bare-relative) robustly so the non-Python index and the legacy
        _find_in_other_langs path agree on canonical relative paths.
        """
        p = Path(raw)
        with contextlib.suppress(ValueError):
            if p.is_absolute():
                return str(p.relative_to(self.repo_root))
        s = str(p)
        if s.startswith("./"):
            s = s[2:]
        return s

    def _index_via_treesitter_batch(
        self, providers: list, search_root: Path,
        index: dict[str, list[SymbolDef]], seen: set,
    ) -> None:
        """Index every provider's files from a SINGLE ``rg --files`` walk.

        Replaces the rg+regex path for languages whose tree-sitter binding is
        installed. For CSS this is the authoritative source: class selectors,
        id selectors, and custom properties (``--name``) are extracted from the
        AST, so no regex pattern ever becomes an rg positional/flag arg (the
        ``--name`` leading-dash trap is structurally impossible here).

        rg accepts repeated ``--glob``, so every provider's globs go into one
        invocation and the repo tree is walked once instead of once per
        provider. Each returned path is dispatched back to its language by
        matching the file NAME against the globs (``fnmatch``, first match in
        provider order wins) — this handles non-``*.ext`` globs too, which a
        plain suffix map would not.

        Failures (unreadable file, parse error) are skipped per-file — the
        index simply lacks those symbols, matching the rg path's tolerance.
        """
        # (glob, lang_id) in stable provider order — first match wins, so a
        # path matching two providers' globs resolves deterministically.
        glob_lang: list[tuple[str, str]] = []
        for provider in providers:
            lang_id = provider.language_id().value
            glob_lang.extend((g, lang_id) for g in provider.get_file_globs())
        if not glob_lang:
            return

        try:
            cmd = ["rg", "--files"]
            for g, _ in glob_lang:
                cmd += ["--glob", g]
            cmd += ["--glob", "!node_modules*", str(search_root)]
            proc = subprocess.run(
                cmd, cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as _err:
            # OSError covers rg-absent FileNotFoundError (rg is OPTIONAL — pyproject
            # [search]). Without it, a base `pip install asicode` has no rg and the
            # default find_symbol(kind="any") crashes here every time. Graceful:
            # empty tree-sitter index this pass.
            logger.debug(
                "[symbol-search] batched rg --files failed for %d glob(s) "
                "under %s: %s — tree-sitter index will be empty this pass",
                len(glob_lang), search_root, _err,
            )
            return
        if proc.returncode not in (0, 1):
            # rg exit 2+ (regex/flag error) yields empty stdout and would otherwise
            # silently produce an empty index with no signal.
            logger.debug(
                "[symbol-search] batched rg --files exit %s under %s — "
                "tree-sitter index will be empty this pass",
                proc.returncode, search_root,
            )
            return

        _total_batch_bytes = 0
        _batch_file_count = 0

        for fpath in (proc.stdout or "").splitlines():
            if not fpath:
                continue
            try:
                abs_path = self.repo_root / fpath
                _size = abs_path.stat().st_size
            except OSError as _err:
                logger.debug(
                    "[symbol-search] stat failed for %s (%s) — skipped",
                    fpath, _err,
                )
                continue
            # P26-4: sibling _rg_token_in_nonpy_files bounds the indexable set
            # at _NONPY_INPROC_MAX_BYTES per file and in total; the batch
            # walker had neither gate — a single minified dist/*.js (tens of
            # MB is common) was read + tree-sitter-parsed in full.  Skip
            # oversized files and stop when the cumulative budget is spent.
            if _size > _NONPY_INPROC_MAX_BYTES:
                logger.debug(
                    "[symbol-search] skipping oversized %s (%d bytes > %d)",
                    fpath, _size, _NONPY_INPROC_MAX_BYTES,
                )
                continue
            if _total_batch_bytes + _size > _NONPY_INPROC_MAX_BYTES:
                logger.debug(
                    "[symbol-search] batch byte budget exhausted at %s "
                    "(cumulative %d bytes) — stopping walk",
                    fpath, _total_batch_bytes,
                )
                break
            if _batch_file_count >= _NONPY_INPROC_MAX_FILES:
                logger.debug(
                    "[symbol-search] batch file cap reached (%d) — stopping walk",
                    _NONPY_INPROC_MAX_FILES,
                )
                break
            _base = PurePosixPath(fpath).name
            lang_id = ""
            for g, lid in glob_lang:
                if fnmatch(_base, g):
                    lang_id = lid
                    break
            if not lang_id:
                continue
            try:
                abs_path = self.repo_root / fpath
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError) as _err:
                logger.debug(
                    "[symbol-search] unreadable %s (%s) — skipped", fpath, _err,
                )
                continue
            _total_batch_bytes += _size
            _batch_file_count += 1
            try:
                syms = _ts_find_all_symbols(content, lang_id)
            except Exception as _err:
                logger.debug(
                    "[symbol-search] tree-sitter parse failed for %s (%s): %s "
                    "— skipped", fpath, lang_id, _err,
                )
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

    def _nonpy_index_worth_building(self, search_root: Path, token: str) -> bool:
        """Whether consulting the non-Python index for *token* is worth its cost.

        The index is whole-repo and TTL-cached, so a COLD lookup pays a full
        build (123ms here) even when the caller already found the symbol in
        Python — ``find_symbol``'s default ``kind="any"`` reaches this branch
        unconditionally. When no non-Python file even mentions the token, that
        build cannot produce a match, so skip it.

        Only the cold path is probed. On a warm cache the index lookup is a
        dict hit, which is cheaper than the rg probe would be — probing there
        would make the fast path slower.
        """
        cached = self._nonpy_index_cache.get(str(search_root))
        if cached is not None and (_time.monotonic() - cached[0]) < _WALK_CACHE_TTL:
            return True
        found = _rg_token_in_nonpy_files(search_root, token)
        # None = probe untrustworthy -> build, exactly as before the probe existed.
        return found is not False

    def invalidate_nonpy_caches(self) -> None:
        """Drop the non-Python symbol caches so a just-written file is visible.

        Both are TTL-based (30 s) because an mtime fingerprint would need the
        very directory walk they exist to avoid. That TTL is fine for drift, but
        NOT for the agent's own writes: it edits a file and immediately looks up
        the symbol it just added. ``_invalidate_cache_after_write`` already
        clears six caches for exactly that reason — its comment records
        find_symbol answering "No definitions found" for a function on disk —
        but these two were not among them, so the symptom survived for every
        non-Python language while Python edits were visible at once.

        Two layers, and both must go — the probe's own memo is a third:

        * ``_nonpy_index_cache`` — the built {name: [SymbolDef]} index. Stale for
          an EDITED file.
        * ``_NONPY_FILES_CACHE`` — the shared file-list walk behind the probe.
          Stale for a NEWLY CREATED file: the probe scans that cached list, so a
          new .go file is invisible even once the index above is rebuilt.
        * ``_NONPY_BLOB_CACHE`` — the probe's content memo. Its per-file
          signatures already catch edits, and its key carries the file list, so
          clearing the list cache above would rebuild it on the next probe via
          a new key — cleared anyway so the two stay coupled in one place.

        Kept here rather than reaching into the module from tool_registry so the
        caller does not have to know how many caches there are.
        """
        self._nonpy_index_cache.clear()
        _NONPY_FILES_CACHE.clear()
        _NONPY_BLOB_CACHE.clear()

    def invalidate_file_caches(self, paths: Optional[Iterable[str]] = None) -> None:
        """Drop the per-file symbol maps for *paths* after they were written.

        The companion to :meth:`invalidate_nonpy_caches`, for the three
        per-file caches that method does NOT cover: ``_py_file_cache``
        (Python), ``_ts_file_cache`` (TS/JS) and ``_go_file_cache`` (Go
        class→methods). All are per-file and keyed on a
        ``(st_mtime_ns, st_size)`` signature, which is why they were never in
        the TTL-drop list — but a signature only detects a change the
        filesystem bothered to record. Where mtime granularity is coarse
        (container bind mounts, NFS/SMB) or mtimes were restored wholesale
        (tar, ``rsync -t``, ``cp -p``), a rewrite can land on the same
        signature and the cache keeps answering with pre-edit line numbers.

        The agent's own writes are the mutation source that matters here and we
        know their paths exactly, so this drops them outright instead of hoping
        the stat differs. Same belt-AND-suspenders split ``insights_manager``
        makes: signature for drift, explicit bump for our own writes.

        Unknown-scope callers (``bash`` can write anything) pass no paths and
        get a full clear — cheap, since all three caches refill per file on
        demand.

        Matching is resolution-independent, and O(1) per path. A key that never
        matched would make this method a silent no-op — the exact failure mode
        it exists to close — and there are two ways to miss: the callers
        disagree on form (``_snapshot_target_files`` builds absolute paths, the
        patch-mixin's ``touched``/``written`` lists are repo-relative), and
        either side may carry an unresolved symlink (on macOS ``repo_root`` is
        resolved to ``/private/var/...`` while a caller-supplied path can still
        read ``/var/...``). Both are handled by resolving through the same
        :meth:`_cache_key` the caches are keyed on, rather than by scanning them
        — an earlier realpath scan over the cache cost ~3 ms per write.

        ``None`` and ``[]`` are NOT the same request. ``None`` means "scope
        unknown" (the bash path) and drops both caches wholesale; an empty list
        means "these writes touched nothing cacheable" and must be a no-op —
        conflating them would evict a live cache on every write that reported no
        paths, giving back exactly the parse this cache exists to avoid.
        """
        if paths is None:
            self._py_file_cache.clear()
            self._ts_file_cache.clear()
            self._go_file_cache.clear()
            return
        for p in paths:
            _p = str(p)
            if not _p:
                continue
            _abs = _p if os.path.isabs(_p) else os.path.join(str(self.repo_root), _p)
            _key = self._cache_key(_abs)
            self._py_file_cache.pop(_key, None)
            self._ts_file_cache.pop(_key, None)
            self._go_file_cache.pop(_key, None)

    def _nonpy_index_for(self, search_root: Path) -> dict[str, list[SymbolDef]]:
        """Build (once, TTL-cached) a {name -> [SymbolDef]} index of ALL
        non-Python definitions under *search_root*.

        Regex-fallback providers are indexed with ONE batched rg spawn per
        provider (all globs via repeated ``--glob``, all patterns via repeated
        ``-e``) instead of one spawn per (glob, pattern) pair — spawn count is
        O(providers), not O(globs x patterns).  Output is bounded per file
        (``-m 5 x pattern-count``) and per provider (``50 x patterns x
        globs`` lines) so pathological matches cannot balloon memory; the caps
        scale with the merged counts so no single pattern is starved of its
        per-pattern budget.

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
        # Providers whose grammar is installed are indexed by ONE batched
        # rg --files walk after this loop (see _index_via_treesitter_batch);
        # collecting them here keeps the tree walk count at 1 instead of one
        # per provider. Sorted for a deterministic glob-dispatch order.
        _ts_providers: list = []
        for provider in sorted(
            set(registry._providers.values()),
            key=lambda p: p.language_id().value,
        ):
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
                _ts_providers.append(provider)
                continue  # skip the provider-regex rg spawn below

            globs = provider.get_file_globs()
            patterns = provider.get_symbol_patterns(kind="any")
            if not globs or not patterns:
                continue
            # ONE batched rg spawn per provider instead of one per (glob,
            # pattern) pair — the old triple loop spawned 94 rg processes
            # (~1.1s here), each re-walking the tree; merging cuts that to one
            # spawn per provider (~0.13s).  All globs and patterns ride in a
            # single invocation via repeated --glob / -e.
            pats = [sp.regex.replace("{name}", f"({sp.name_capture})") for sp in patterns]
            # Caps scale with the merged pattern/glob counts so a single
            # pattern cannot starve the others of their per-pattern budget:
            # -m is a per-FILE total across the merged alternation, and the
            # line slice is a per-provider total.
            _per_file_cap = 5 * len(pats)
            _line_cap = 50 * len(pats) * len(globs)
            cmd = ["rg", "--no-heading", "--with-filename", "--line-number", "-m", str(_per_file_cap)]
            for glob in globs:
                cmd += ["--glob", glob]
            cmd += ["--glob", "!node_modules*", "--glob", "!*.py"]
            # Pass each pattern via -e so a pattern that starts with '-' (e.g.
            # the CSS "--{name}" custom-property pattern) is not misparsed as
            # a flag.
            for pat in pats:
                cmd += ["-e", pat]
            cmd.append(str(search_root))
            with contextlib.suppress(AttributeError, TypeError, OSError, subprocess.SubprocessError):
                # OSError covers rg-absent FileNotFoundError (rg is OPTIONAL —
                # pyproject [search]). Graceful: skip this provider.
                proc = subprocess.run(
                    cmd, cwd=str(self.repo_root),
                    capture_output=True, text=True, timeout=8,
                    check=False,
                )
                for line in (proc.stdout or "").splitlines()[:_line_cap]:
                    parts = line.split(":", 2)
                    if len(parts) < 3:
                        continue
                    with contextlib.suppress(ValueError, AttributeError, TypeError):
                        rel = self._rg_path_to_rel(parts[0])
                        lineno = int(parts[1])
                        ctx = parts[2].strip()
                        key = (rel, lineno)
                        if key in seen:
                            continue
                        seen.add(key)
                        # Classify the line with the FIRST pattern whose regex
                        # matches, using that pattern's own name_capture group
                        # (default \w+).  CSS uses [-\w]+ so kebab-case names
                        # like "btn-primary" or "--primary-color" are not
                        # truncated at the hyphen.
                        cap_name = ""
                        kind = "any"
                        for sp, pat in zip(patterns, pats, strict=True):
                            rm = re.search(pat, ctx)
                            if rm and rm.groups():
                                cap_name = rm.group(1)
                                kind = sp.kind
                                break
                        if not cap_name:
                            m = re.search(r"\b(\w+)\s*[\(\{<]", ctx)
                            cap_name = m.group(1) if m else ""
                        if not cap_name:
                            continue
                        index.setdefault(cap_name, []).append(SymbolDef(
                            file=rel, line=lineno, kind=kind, name=cap_name,
                            signature=ctx,
                        ))

        # One rg --files walk for every tree-sitter-capable provider at once.
        # Runs after the regex loop rather than inside it; the two paths key
        # ``seen`` with different tuple arities ((rel, lineno) vs
        # (rel, name, line)), so they can never collide and ordering between
        # them does not affect the result.
        if _ts_providers:
            self._index_via_treesitter_batch(
                _ts_providers, search_root, index, seen,
            )

        self._nonpy_index_cache[cache_key] = (_time.monotonic(), index)
        return index

    def _find_subclasses(self, base_class: str) -> list[str]:
        """Find names of classes that inherit from base_class."""
        pattern = rf"class\s+(\w+)\s*\(.*\b{re.escape(base_class)}\b"
        names: list[str] = []
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            cmd = ["rg", "--no-heading", "-m", "3", pattern, str(self.repo_root)]
            proc = subprocess.run(
                cmd, cwd=str(self.repo_root),
                capture_output=True, text=True, timeout=5,
                check=False,
            )
            for line in (proc.stdout or "").splitlines()[:20]:
                m = re.search(r"class\s+(\w+)", line)
                if m and m.group(1) != base_class:
                    names.append(m.group(1))
        return list(dict.fromkeys(names))


# ── process-shared SymbolSearcher pool ─────────────────────────────────────
# SymbolSearcher carries five per-instance caches: per-file Python/TS/Go parse
# maps (_py_file_cache/_ts_file_cache/_go_file_cache), the whole-repo non-
# Python definition index (_nonpy_index_cache — cold build ~123 ms, see
# _nonpy_index_worth_building) and the realpath memo. A fresh instance re-pays
# a cold build on first use. Several call sites construct one PER CALL
# (patch_engine apply/salvage fallback, service eval fallback, agent_tools
# helper pack) — repeated tool calls over one repo re-paid that cold build
# every time. This pool shares a small LRU-capped set of instances across ALL
# call sites, keyed by the resolved repo root.
#
# Bonus: tool_registry's post-write invalidation (invalidate_nonpy_caches /
# invalidate_file_caches) runs on the POOLED instance, so the per-call sites
# now see subagent writes immediately instead of after the 30 s non-Python
# TTL. Previously they built a private instance whose caches no invalidator
# could reach. Concurrency is unchanged: the tool_registry instance already
# served parallel read-only tool dispatches, and the module-level walk caches
# (_NONPY_FILES_CACHE etc.) were already shared across instances.
_SEARCHER_POOL_MAX_ENTRIES = 4
_searcher_pool: OrderedDict[str, SymbolSearcher] = OrderedDict()
_searcher_pool_lock = threading.Lock()


def get_symbol_searcher(repo_root: str) -> SymbolSearcher:
    """Return the process-shared :class:`SymbolSearcher` for *repo_root*.

    Instances are keyed by the RESOLVED root string and LRU-capped at
    ``_SEARCHER_POOL_MAX_ENTRIES``: a get refreshes recency, so repeated
    lookups over the same repo reuse warm caches while memory stays bounded.
    The pool is process-wide, so any invalidator (tool_registry's post-write
    path) reaches the same instances the per-call sites use.
    """
    key = str(Path(repo_root).resolve())
    with _searcher_pool_lock:
        searcher = _searcher_pool.get(key)
        if searcher is not None:
            _searcher_pool.move_to_end(key)
            return searcher
        searcher = SymbolSearcher(key)
        _capped_put(_searcher_pool, key, searcher, _SEARCHER_POOL_MAX_ENTRIES)
        return searcher
