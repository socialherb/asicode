"""Shared utilities for AgentLoop and DesignChatLoop.


Both systems share ToolRegistry, LLMClient, and AgentConfig but previously
duplicated context building, tool result wrapping, and schema filtering.
This module consolidates those common patterns.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time as _walk_time
import warnings
from pathlib import Path
from typing import Any, NoReturn  # NoReturn f821-protected

from external_llm.languages.comment_syntax import CommentSyntax, comment_syntax_for
from external_llm.languages.models import _LANGUAGE_EXTENSION_GROUPS

from ..common.cache_utils import (  # noqa: F401  # re-export (SSOT: common/cache_utils.py)
    _WALK_CACHE_MAX_ENTRIES,
    _capped_put,
)
from ..common.walk_policy import (  # noqa: F401  # re-export (SSOT: common/walk_policy.py)
    _WALK_DEPRIORITIZED_DIRS,
    _WALK_SKIP_DIRS,
    _WALK_SKIP_FILE_SUFFIXES,
    _path_is_walk_admissible,
    _rel_under_skipped_dir,
    _walk_dir_sort_key,
    _walk_should_skip_dir,
)
from ..languages import LanguageId
from .operation_models import OpStatus

logger = logging.getLogger(__name__)


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
_FAIL_STATUSES: frozenset = frozenset(
    {
        OpStatus.ERROR,
        OpStatus.NOT_FOUND,
        OpStatus.FAILED,
        OpStatus.VERIFICATION_FAILED,
        OpStatus.EXECUTION_ERROR,
        OpStatus.PREFLIGHT_FAILED,
    }
)

# Auto-sync guard: every OpStatus whose name contains "FAIL" or "ERROR"
# must be in _FAIL_STATUSES.  This catches omissions when new terminal failure
# statuses are added to OpStatus.
assert _FAIL_STATUSES.issuperset(s for s in OpStatus if "FAIL" in s.name or "ERROR" in s.name), (
    f"_FAIL_STATUSES missing failure status(es): "
    f"{ {s for s in OpStatus if ('FAIL' in s.name or 'ERROR' in s.name) and s not in _FAIL_STATUSES} }"
)

# File-extension sets for the repo walkers, derived from the language-family
# SSOT (_LANGUAGE_EXTENSION_GROUPS in languages/models.py). A hardcoded tuple
# here drifted from that SSOT: .pyi (Python) and .mts/.cts/.mjs/.cjs (JS/TS)
# were first-class in _EXT_MAP + provider globs + _LANGUAGE_EXTENSION_GROUPS but
# absent from the walkers — so find_symbol / call_graph silently returned
# nothing for symbols defined in those files (confirmed regression: a function
# in mod.mts or a class in stub.pyi was invisible to find_symbol). Deriving
# from the family groups keeps the walkers structurally in sync with every
# other SSOT dimension: adding an extension to a family now propagates here
# automatically. tuple (not frozenset) so it is usable directly with
# str.endswith(); sorted for deterministic ordering.


# Guard for the SSOT-derived constants below: if the .ts or .py group
# is ever removed from _LANGUAGE_EXTENSION_GROUPS, the next(..., None)
# would silently return None, and "None or f()" would call this. Raise
# a clear error rather than crashing at module level with StopIteration.
def _stop_iter_fallback(name: str) -> NoReturn:
    raise RuntimeError(f"SSOT invariant broken: no group containing {name} found in _LANGUAGE_EXTENSION_GROUPS")


_TS_JS_EXTENSIONS: tuple = tuple(
    sorted(
        next(
            (g for g in _LANGUAGE_EXTENSION_GROUPS if ".ts" in g),
            None,
        )
        or _stop_iter_fallback("_TS_JS_EXTENSIONS (.ts)")
    )
)
_PY_EXTENSIONS: tuple = tuple(
    sorted(
        next(
            (g for g in _LANGUAGE_EXTENSION_GROUPS if ".py" in g),
            None,
        )
        or _stop_iter_fallback("_PY_EXTENSIONS (.py)")
    )
)

# ── Shared repo file walkers ─────────────────────────────────────────────────
# Consolidated here so symbol_search and call_graph share ONE walk
# implementation + ONE process-global cache (previously each module walked
# independently; call_graph had no cache at all and re-rglobbed every build).
#
# Walk ADMISSION policy (skip dirs / suffixes / descent ordering) is defined in
# common/walk_policy.py and re-exported at the top of this module — the single
# source of truth every walker (agent + graph layer) consumes (B2' parity
# contract, 2026-08-11).


# Module-level guard so the truncation warning fires at most once per
# (root, cap) per process — a long-lived REPL re-walks every TTL window and
# would otherwise spam. Reset implicitly when the entry is re-cached.
_WALK_TRUNCATION_WARNED: set[tuple[str, int]] = set()


def _warn_walk_truncated(root_key: str, cap: int, collected: int) -> None:
    """Log (once) that a repo walk hit the file cap and is therefore incomplete.

    A truncated walk silently hides files from find_symbol / call_graph /
    find_relevant_files — the agent then concludes "symbol does not exist"
    (fail-silent). Surfacing the cap as an explicit, deduped warning makes the
    incompleteness visible without changing any return type.
    """
    marker = (root_key, cap)
    if marker in _WALK_TRUNCATION_WARNED:
        return
    _WALK_TRUNCATION_WARNED.add(marker)
    logger.warning(
        "File walk for %s truncated at cap %d (collected %d); files beyond the "
        "cap are INVISIBLE to find_symbol / find_relevant_files / "
        "analyze_change_impact. Raise the cap if the repo is larger.",
        root_key,
        cap,
        collected,
    )


def _walk_truncated_for(root, cache: dict, max_files: int | None = None) -> bool:
    """True if the walk *this caller* would receive for *root* is incomplete.

    Callers (e.g. find_symbol on a miss) consult this to distinguish "symbol
    genuinely absent" from "symbol may exist in un-indexed files". Returns
    False when no cached walk exists (treat as: not known-truncated).

    Truncation is a property of ``(walk, max_files)``, NOT of the cache entry
    alone, so *max_files* must be the cap the caller itself passes to
    :func:`_walk_repo_files`. A cache entry produced by a HIGHER-cap caller can
    be complete (``was_truncated=False``) yet still be sliced down for a
    lower-cap one — ``_walk_repo_files`` returns ``files[:max_files]`` on the
    hit path. Reading only the stored flag then reports "complete" for a result
    that is missing files, resurrecting the exact fail-silent behaviour this
    helper exists to prevent.

    Live example of the mismatch: ``vulture_scanner`` walks with
    ``max_files=4000`` while ``symbol_search``/``call_graph`` use 3000, so on a
    3000-4000 file repo a structural scan inside the cache TTL would hand
    find_symbol a silently shortened list. Passing *max_files* closes that.
    Omitting it preserves the old flag-only reading for callers that genuinely
    do not have a cap in hand.
    """
    cached = cache.get(str(root))
    if cached is None:
        return False
    # cached = (timestamp, files, was_truncated)
    _ts, _files, _was_truncated = cached
    if _was_truncated:
        return True
    return max_files is not None and len(_files) > max_files


# Per-root file-list cache. rglob over a large repo costs ~250ms; repeated
# find_symbol / call-graph builds would pay it every time. Short TTL so newly
# created files become visible quickly. Best-effort: callers tolerate a
# slightly-stale list (missing files → "not found this round").
_WALK_CACHE_TTL: float = 30.0
# Bounded entry cap (P4): these path-keyed caches grew unboundedly in a long-
# lived REPL that visited many repos (each holding a full file list). FIFO
# eviction under the GIL stays consistent with the lock-free, single-threaded
# design; the current repo is the newest entry, stale repos are evicted first.
# Generation counters — bumped by post-write invalidation so a walk that was
# already in flight cannot resurrect its pre-write result into the cache (the
# `pop()` alone loses the race: it runs while os.walk is still collecting, and
# the store that follows re-inserts the stale list under a FRESH timestamp, so
# the staleness lasts a full TTL rather than being cut short).
#
# A single-element LIST, not a bare int, and that is load-bearing twice over:
#   * cross-module — the invalidator reaches these via ``from _shared_utils
#     import _PY_WALK_GEN``, and ``+= 1`` on an imported int rebinds only the
#     importer's local, leaving this module's value at 0. Silently. Mutating a
#     shared list element is visible to every holder without ``global``.
#   * in-module — ``_walk_repo_files`` reads the counter before the walk and
#     again before the store, so it needs the counter OBJECT, not a copy of the
#     value taken at call time.
# Both mistakes are no-ops that read as working code, so keep the list.
_PY_WALK_GEN: list[int] = [0]
_TS_WALK_GEN: list[int] = [0]
# 3-tuple: (timestamp, files, was_truncated).  ``was_truncated`` is True when
# the walk exited early because ``max_files`` was reached — on cache hit the
# caller's own cap must be checked (no truncated list may masquerade as a full
# one for a larger cap; see ``_walk_repo_files`` cache-hit logic).
_PY_WALK_CACHE: dict[str, tuple[float, list, bool]] = {}
_TS_WALK_CACHE: dict[str, tuple[float, list, bool]] = {}


def _walk_repo_files(root, max_files: int, cache: dict, keep, gen_counter: list[int]) -> list:
    """Shared walk engine behind :func:`_walk_py_files` / :func:`_walk_ts_js_files`.

    Returns every file under *root* for which ``keep(name)`` is true, skipping
    hidden/vendor/venv dirs via the single :func:`_walk_should_skip_dir`
    predicate. ``dirnames`` is pruned in-place so ``os.walk`` makes a single
    descent — a whole-tree walk including node_modules/.venv would visit tens of
    thousands of irrelevant files. Early-exits at ``max_files`` so a huge vendor
    tree can't exhaust memory/time before the caller's cap check runs, and
    memoizes the result in *cache* (per-root, TTL-bounded via
    :data:`_WALK_CACHE_TTL`, FIFO-bounded via :func:`_capped_put`).

    *gen_counter* is the caller's generation counter ITSELF — pass
    ``_PY_WALK_GEN``, never ``[_PY_WALK_GEN]``. Wrapping it builds a private
    box holding a copy of the value, so the re-check below compares a snapshot
    against itself and can never fire (that shipped once, inert but
    reading as correct). Read before the slow walk; if it moves while we
    collect, an invalidation landed mid-flight and the result must NOT be
    cached. See :data:`_PY_WALK_GEN`.

    The two callers pass *distinct* caches so an extension set never
    masquerades as the other, and a single ``os.walk`` pass per extension set
    so one extension (e.g. ``.js``) cannot fill ``max_files`` and exclude
    ``.ts``/``.tsx``.
    """
    key = str(root)
    cached = cache.get(key)
    if cached is not None:
        ts, files, was_truncated = cached
        if (_walk_time.monotonic() - ts) < _WALK_CACHE_TTL and (not was_truncated or len(files) >= max_files):
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

    # Read generation BEFORE the slow os.walk — if invalidation bumps it while
    # we collect, the result is stale and must NOT be cached.
    _gen_before = gen_counter[0]

    results: list = []
    _was_truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune vendor/noise dirs, then SORT the survivors so the descent is
        # (a) deterministic across machines/clones — os.walk otherwise yields
        # filesystem-enumeration order — and (b) source-prioritized: real code
        # subtrees (tier 0) are visited before tests/fixtures/generated (tier
        # 1), so the file cap is filled with source, not tests. Sorting both
        # dirnames and filenames keeps file order reproducible too.
        dirnames[:] = sorted(
            (d for d in dirnames if not _walk_should_skip_dir(d)),
            key=_walk_dir_sort_key,
        )
        filenames.sort()
        for name in filenames:
            if keep(name):
                results.append(Path(dirpath) / name)
                if len(results) >= max_files:
                    _was_truncated = True
                    if gen_counter[0] != _gen_before:
                        return list(results[:max_files])  # invalidated mid-walk
                    _capped_put(cache, key, (_walk_time.monotonic(), results, True))
                    _warn_walk_truncated(key, max_files, len(results))
                    # Slice + shallow-copy, mirroring the cache-HIT path above. Without
                    # the copy we'd hand back the very list object stored in *cache*, so
                    # a caller that .append()/.sort()s the result would pollute the cache
                    # for every subsequent caller — the invariant documented at the HIT
                    # path must hold on both paths.
                    return list(results[:max_files])
    if gen_counter[0] != _gen_before:
        return list(results[:max_files])  # invalidated mid-walk
    _capped_put(cache, key, (_walk_time.monotonic(), results, _was_truncated))
    # See note above: return a copy, not the cached object.
    return list(results[:max_files])


def _walk_py_files(root, max_files: int) -> list:
    """Walk *root* returning .py files, skipping hidden/vendor/venv dirs.

    Results are cached per root for ``_WALK_CACHE_TTL`` seconds.
    """
    return _walk_repo_files(root, max_files, _PY_WALK_CACHE, lambda n: n.endswith(_PY_EXTENSIONS), _PY_WALK_GEN)


def _walk_ts_js_files(root, max_files: int) -> list:
    """Walk *root* returning TS/JS files, skipping hidden/vendor/node_modules
    dirs and ``*.min.js`` bundles.

    Cached per root with the same TTL scheme as :func:`_walk_py_files`. A single
    ``os.walk`` pass collects all four extensions (``.ts/.tsx/.js/.jsx``) so one
    extension (e.g. ``.js``) cannot fill ``max_files`` and exclude ``.ts``/
    ``.tsx`` — the primary source files of a TypeScript project.
    """
    return _walk_repo_files(
        root,
        max_files,
        _TS_WALK_CACHE,
        lambda n: n.endswith(_TS_JS_EXTENSIONS) and not n.endswith(_WALK_SKIP_FILE_SUFFIXES),
        _TS_WALK_GEN,
    )


def invalidate_walk_caches() -> None:
    """Drop the file-walk caches so a just-written file is visible.

    Both layers must move together, which is the whole reason this lives here
    rather than in ``tool_registry``: dropping the entries without bumping the
    generation leaves an in-flight walk free to re-insert its pre-write result
    (see :data:`_PY_WALK_GEN`). Callers should not have to know that — the same
    reasoning ``SymbolSearcher.invalidate_nonpy_caches`` records for its own
    two-layer drop.

    Clears EVERY entry, and takes no root, because these caches are not keyed by
    the repo root — they are keyed by whatever root was walked.
    ``find_symbol(..., search_path="external_llm/agent")`` resolves that to a
    SUBDIRECTORY (``SymbolSearcher._resolve_search_root``) and caches under it,
    so a caller popping one repo-root key left every scoped entry behind:
    measured, a repo-root pop cleared 1 of 2 live keys, and the surviving
    subtree entry then answered a scoped find_symbol with a pre-write file list
    for the full TTL. That is the "cannot find the symbol it just wrote" symptom
    this fan-out exists to kill, and the generation counter cannot catch it —
    the counter only stops an in-flight walk from storing, never an entry that
    is already stored.

    Wholesale is also what makes the root moot: the two callers disagreed on
    ``repo_root`` vs ``_effective_repo_root``, and neither matched the searcher's
    own root reliably. Cost is bounded — ``_WALK_CACHE_MAX_ENTRIES`` caps these
    at 8 entries, and the generation counter is already process-wide, so this
    adds no over-invalidation that was not already accepted. The sibling that
    got this right from the start is ``invalidate_nonpy_caches``, which likewise
    clears rather than pops.
    """
    for _cache in (_PY_WALK_CACHE, _TS_WALK_CACHE):
        _cache.clear()
    for _gen in (_PY_WALK_GEN, _TS_WALK_GEN):
        _gen[0] += 1


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


def _scan_to_line_state(
    lines,
    end_lineno: int,
    comment_syntax: CommentSyntax | None = None,
) -> tuple[str | None, bool, str | None]:
    """Scan ``lines[0:end_lineno]`` and return the literal/block-comment state
    ENTERING line ``end_lineno`` — ``(in_str, in_triple, block_close)``.

    Seeds :func:`_net_bracket_delta` (and the F2 forward scan) with correct
    prior-line context so a ``replace_line`` whose anchor sits INSIDE a
    multi-line block comment (``/* */``, Lua ``--[[ ]]``) or triple-quoted
    string is not mis-counted. Without this seed the per-line tally treats the
    anchor's brackets as real code, falsely tripping the F2 expansion and
    ``del``-ing the real code after the comment (confirmed data-loss vector;
    the forward scan likewise started from empty state and mis-identified the
    close line).

    Cost is O(end_lineno) line scans — acceptable for an interactive edit op,
    and the common case (anchor in normal code) simply yields the empty state
    ``(None, False, None)`` (measured <45ms even for a 10k-line file).

    .. note:: Unclosed-quote fallback. A non-triple single/double quote that
       stays open across a line boundary is NEVER a legitimate multi-line
       literal in any supported language — multi-line literals use
       triple-quotes (tracked via ``in_triple``) or backticks (JS/TS template
       literals, kept as ``in_str='`'``). An open ``'``/``"`` at end-of-prefix
       is therefore either a Rust *lifetime* (``'a``, ``'static`` — a single
       ``'`` with no closer, which the scanner cannot distinguish from a char
       literal without a real grammar) or a genuine syntax error. Returning a
       poisoned seed in that case lets the F2 forward scan mis-identify the
       close line and ``del`` real code (confirmed: an odd-count lifetime above
       the anchor deletes a victim line below it). Falling back to the empty
       seed is provably safe — no valid code is affected — and closes the
       documented Rust residual risk without a per-language special case.
       Triple-quote and block-comment seeds are unaffected (they are legit
       multi-line constructs); only the non-triple ``'``/``"`` case resets.
    """
    if comment_syntax is None:
        comment_syntax = comment_syntax_for(LanguageId.UNKNOWN)
    _in_str, _in_triple, _block_close = None, False, None
    for _i in range(0, min(end_lineno, len(lines))):
        _, _in_str, _in_triple, _block_close = _scan_line_brackets_delta(
            lines[_i], _in_str, _in_triple, _block_close, comment_syntax
        )
    # Defensive fallback: an open non-triple ' / " crossing a line boundary is a
    # Rust lifetime or a syntax error, never a legit literal — a poisoned seed
    # here is a confirmed data-loss vector (F2 forward scan mis-identifies the
    # close line). Triple-quote (`in_triple`) and backtick seeds are legit and
    # kept; only the non-triple ' / " case resets to the empty state.
    if _in_str in ('"', "'") and not _in_triple:
        return None, False, None
    return _in_str, _in_triple, _block_close


def _net_bracket_delta(
    text: str,
    comment_syntax: CommentSyntax | None = None,
    *,
    in_str: str | None = None,
    in_triple: bool = False,
    block_close: str | None = None,
) -> int:
    """Net bracket delta (``{}``, ``()``, ``[]``) outside string/comment content.

    Language-aware via *comment_syntax* (a typed policy from
    :mod:`external_llm.languages.comment_syntax`, looked up per-file via
    :func:`comment_syntax_for`): the scanner skips exactly the line- and
    block-comment tokens the target language uses — e.g. ``#`` for Python/Ruby/
    Bash, ``//`` + ``/* */`` for the C-family, BOTH for PHP, ``--`` for Lua.

    This replaces the prior binary ``c_style_comments`` flag, which classified
    every non-Python language as C-style and thus mis-counted brackets inside
    ``#`` comments for Ruby / Bash / PHP (genuine ``#``-comment languages) — a
    latent data-loss vector (a bracket in a ``#`` comment was counted, falsely
    tripping the guard and triggering the F2 multi-line expansion that ``del``
    real code). Centralising the classification in a typed policy means a new
    language can never silently re-introduce the bug: its ``LanguageId`` simply
    maps to its ``CommentSyntax``.

    *comment_syntax* defaults to ``None`` (skip no comments) — the safest
    default, since counting a bracket inside a comment is far less dangerous
    than wrongly skipping a bracket inside real code. Callers that care about
    comment-awareness MUST pass an explicit policy.

    String/char/template literals (with escape and triple-quote handling) are
    ALWAYS skipped, regardless of language.

    The optional *in_str* / *in_triple* / *block_close* seed the scanner with
    the prior-line state (computed by :func:`_scan_to_line_state`) so a line
    that sits INSIDE a multi-line block comment or triple-quoted string opened
    on an earlier line is counted correctly — its brackets are part of the
    literal/comment, not real code. Without this, a ``replace_line`` whose
    anchor is inside a ``/* */`` block would mis-trigger the F2 expansion and
    ``del`` the real code after the comment (confirmed data-loss vector).

    .. note:: Rust lifetime limitation — a single ``'`` opens a char/string
       literal for ALL languages, which is correct for C/C++/Java (``'a'``),
       JS/TS/Go (``'...'``) and Python (``'...'``). But Rust *lifetimes*
       (``'a``, ``'static``) use a SINGLE quote with no closing quote, so the
       scanner enters an unterminated "string" and swallows any subsequent
       bracket — e.g. ``foo::<'a>(x)`` tallies as ``0`` instead of ``+1`` (the
       ``(`` is consumed). This is a per-line false negative kept intentionally:
       distinguishing a lifetime from a char literal needs a real grammar, and a
       heuristic would risk re-introducing the exact mis-count for char literals
       in every OTHER language. Impact is now BENIGN: at worst a missed
       bracket-balance guard (→ a possible syntax error, never data loss),
       because the guard compares ``_old_delta`` vs ``_new_delta`` on the same
       line and a miss simply skips the F2 expansion. The dangerous case — a
       poisoned seed reaching the anchor from an odd-count lifetime ABOVE it and
       mis-directing the F2 forward scan to ``del`` a victim line — is closed by
       :func:`_scan_to_line_state`'s unclosed-quote fallback (a non-triple
       ``'``/``"`` open at a line boundary is reset to the empty seed).

    Returns the net count of opening minus closing brackets across all three
    bracket families, ignoring any bracket that appears inside a literal or
    comment. This is the SSOT per-line bracket tally for the bracket-balance
    guard in ``anchor_edit`` (``replace_line``) — both the tool path
    (``write_tools._tool_anchor_edit``) and the editor path
    (``symbol_handlers_anchor._handle_anchor_edit``) consume it so the two
    cannot desync.
    """
    if comment_syntax is None:
        comment_syntax = comment_syntax_for(LanguageId.UNKNOWN)
    _line_tokens = comment_syntax.line_tokens
    _block_pairs = comment_syntax.block_pairs

    _delta = 0
    _in_str = in_str
    _in_triple = in_triple
    _esc = False
    _j = 0
    _n = len(text)
    # Entering mid-block-comment: skip to the block's close token first. The
    # whole line up to the close is comment content (brackets ignored); if the
    # close is not on this line, the entire line is inside the comment.
    if block_close is not None:
        _end = text.find(block_close)
        if _end < 0:
            return 0
        _j = _end + len(block_close)
    while _j < _n:
        _ch = text[_j]
        if _in_str is not None:
            # Inside a string literal. A backslash escapes the next character;
            # we track escape state with ``_esc`` rather than the naive
            # ``prev != '\\'`` look-back, which mis-counts ``"C:\\"`` (an escaped
            # backslash — the trailing ``"`` is a real closer) and leaves the
            # literal open, swallowing the code after it. Mirrors the verified
            # escape state machine in :mod:`external_llm.providers`.
            if _in_triple:
                if not _esc and text[_j : _j + 3] == _in_str * 3:
                    _in_str = None
                    _in_triple = False
                    _esc = False
                    _j += 3
                    continue
            elif not _esc and _ch == _in_str:
                _in_str = None
                _esc = False
                _j += 1
                continue
            if _esc:
                _esc = False
            elif _ch == "\\":
                _esc = True
            _j += 1
            continue
        # Not inside a literal: a quote opens one (incl. triple), else fall
        # through to comment / bracket accounting.
        if _ch in ('"', "'", "`"):
            if _j + 2 < _n and text[_j : _j + 3] == _ch * 3:
                _in_str = _ch
                _in_triple = True
                _esc = False
                _j += 3
                continue
            _in_str = _ch
            _in_triple = False
            _esc = False
            _j += 1
            continue
        # Block comments — checked BEFORE line tokens so a block open
        # that shares a prefix with a line token (Lua '--[[' vs '--')
        # wins. (Also handles PHP '/* */' alongside its '#' line token.)
        _skipped = False
        for _open, _close in _block_pairs:
            if text[_j : _j + len(_open)] == _open:
                _end = text.find(_close, _j + len(_open))
                _j = _n if _end < 0 else _end + len(_close)
                _skipped = True
                break
        if _skipped:
            continue
        # Line comments — '#' (Python/Ruby/Bash/php), '//' (C-family),
        # '--' (Lua); PHP matches BOTH '#' and '//'.
        for _tok in _line_tokens:
            if text[_j : _j + len(_tok)] == _tok:
                _nl = text.find("\n", _j)
                _j = _n if _nl < 0 else _nl
                _skipped = True
                break
        if _skipped:
            continue
        if _ch in "({[":
            _delta += 1
        elif _ch in ")}]":
            _delta -= 1
        _j += 1
    return _delta


def _scan_line_brackets_delta(
    line: str,
    in_str: str | None,
    in_triple: bool,
    block_close: str | None,
    comment_syntax: CommentSyntax | None = None,
) -> tuple[int, str | None, bool, str | None]:
    """Scan ONE line for net bracket delta, carrying string/comment state.

    Stateful companion to :func:`_net_bracket_delta` for the multi-line F2
    bracket-expansion scan in ``anchor_edit(replace_line)``. The per-line
    ``_net_bracket_delta`` is stateless (resets string state each call), so it
    CANNOT walk consecutive lines where a string or block-comment that opened
    on a prior line is still open. This helper threads that state
    (``in_str`` / ``in_triple`` / ``block_close``) across line boundaries so the
    expansion scan never mis-counts a bracket living inside a multi-line
    construct — which previously caused it to mis-identify the close line and
    ``del`` real code (e.g. a ``)`` inside a Python ``#`` comment was counted,
    terminating the scan early and deleting the function's real arguments).

    Language-aware via *comment_syntax* (a typed policy; see
    :func:`_net_bracket_delta`). The ``block_close`` state carries the CLOSE
    token of whichever block comment is currently open (e.g. ``*/`` for a
    C-family block, ``]]`` for a Lua long comment), or ``None`` when no
    block comment is open. Carrying the close token (rather than a bare bool)
    means languages with different block-comment styles coexist correctly.
    Triple-quoted strings carry their open state across lines too.

    Returns ``(delta, in_str, in_triple, block_close)`` — the state to pass
    into the next line's call.
    """
    if comment_syntax is None:
        comment_syntax = comment_syntax_for(LanguageId.UNKNOWN)
    _line_tokens = comment_syntax.line_tokens
    _block_pairs = comment_syntax.block_pairs

    _delta = 0
    _esc = False
    _j = 0
    _n = len(line)
    while _j < _n:
        _ch = line[_j]
        # 1. Inside a block comment: look ONLY for its close token.
        if block_close is not None:
            if line[_j : _j + len(block_close)] == block_close:
                _j += len(block_close)
                block_close = None
                continue
            _j += 1
            continue
        # 2. Inside a string literal. ``_esc`` tracks backslash escapes within
        #    the line (mirrors :func:`_net_bracket_delta`); it is reset to False
        #    at the start of every line rather than threaded across boundaries,
        #    because ``\<newline>`` in a multi-line literal is a line
        #    continuation (the backslash escapes the newline, not the first
        #    char of the next line), so each line starts unescaped.
        if in_str is not None:
            if in_triple:
                if not _esc and line[_j : _j + 3] == in_str * 3:
                    in_str = None
                    in_triple = False
                    _esc = False
                    _j += 3
                    continue
            elif not _esc and _ch == in_str:
                in_str = None
                _esc = False
                _j += 1
                continue
            if _esc:
                _esc = False
            elif _ch == "\\":
                _esc = True
            _j += 1
            continue
        # 3. String/template-literal open.
        if _ch in ('"', "'", "`"):
            if _j + 2 < _n and line[_j : _j + 3] == _ch * 3:
                in_str = _ch
                in_triple = True
                _esc = False
                _j += 3
                continue
            in_str = _ch
            in_triple = False
            _esc = False
            _j += 1
            continue
        # 4. Block-comment open — checked BEFORE line tokens so a block open
        #    that shares a prefix with a line token (Lua '--[[' vs '--') wins.
        _skipped = False
        for _open, _close in _block_pairs:
            if line[_j : _j + len(_open)] == _open:
                _rest = line[_j + len(_open) :]
                _end = _rest.find(_close)
                if _end >= 0:
                    _j = _j + len(_open) + _end + len(_close)
                else:
                    block_close = _close
                    _j = _j + len(_open)
                _skipped = True
                break
        if _skipped:
            continue
        # 5. Line comment — rest of line ignored.
        for _tok in _line_tokens:
            if line[_j : _j + len(_tok)] == _tok:
                _j = _n  # break out of the while loop
                _skipped = True
                break
        if _skipped:
            continue
        # 6. Brackets.
        if _ch in "({[":
            _delta += 1
        elif _ch in ")}]":
            _delta -= 1
        _j += 1
    return _delta, in_str, in_triple, block_close


# ── LLM cost estimation ──────────────────────────────────────────────────────

# (input_per_M_usd, output_per_M_usd)
# Provider-level pricing — fallback when no model-specific match.
_COST_PER_M: dict[str, tuple[float, float]] = {
    "google": (0.10, 0.40),
    "openai": (5.00, 15.00),
    "anthropic": (3.00, 15.00),
    "deepseek": (0.27, 1.10),
    "ollama": (0.00, 0.00),
    "zai": (1.40, 4.40),
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
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-r1": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    # Anthropic
    "claude-fable-5": (15.00, 75.00),
    "claude-mythos-5": (15.00, 75.00),
    "claude-4-opus": (15.00, 75.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 5.00),
    # OpenRouter / third-party models served via OpenAI-compatible API.
    # OpenRouter slugs use the ``<vendor>/<model>`` form. These are
    # LONGEST-prefix-matched, so a vendor-prefixed slug (e.g.
    # ``deepseek/deepseek-v4-flash``) must be listed explicitly to win over the
    # bare ``deepseek-...`` entries above — otherwise the cheaper OpenRouter
    # rate would be shadowed by the native DeepSeek price.
    # Source (verified 2026-06): https://openrouter.ai/models
    "deepseek/deepseek-v4-flash": (0.09, 0.18),  # 35% cheaper than native
    "deepseek/deepseek-v4-pro": (0.435, 0.87),  # same as native
    "qwen/qwen3.6": (0.289, 2.40),
    # Z.AI — source: https://docs.z.ai/guides/overview/pricing (verified 2026-06)
    "glm-5.2": (1.40, 4.40),
    "glm-5.1": (1.40, 4.40),
    "glm-5-turbo": (1.20, 4.00),
    "glm-5": (1.00, 3.20),
    "glm-4.7": (0.60, 2.20),
    "glm-4.6": (0.60, 2.20),
    "glm-4.5": (0.60, 2.20),
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
    "deepseek": 0.26,
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
    "deepseek-v4-flash": 0.0028,
    "deepseek-v4-pro": 0.003625,
    "deepseek-chat": 0.07,
    "deepseek-reasoner": 0.14,
    "deepseek-r1": 0.14,
    # Z.AI GLM models — source: https://docs.z.ai/guides/overview/pricing
    "glm-5.2": 0.26,
    "glm-5.1": 0.26,
    "glm-5-turbo": 0.24,
    "glm-5": 0.20,
    "glm-4.7": 0.11,
    "glm-4.6": 0.11,
    "glm-4.5": 0.11,
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
# Anthropic charges a 25% premium to write the cache (1.25x input rate).
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


def _get_cached_input_rate(provider: str, in_rate: float, model: str = "", base_url: str = "") -> float | None:
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


def total_input_tokens(provider: str, prompt_tok: int, cache_read_tok: int, cache_creation_tok: int = 0) -> int:
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


def coerce_token_count(value: Any) -> int:
    """Coerce a provider usage-field value to a safe int.

    Usage payloads are contractually ints, but non-conforming responses must
    not TypeError the per-turn ``+=`` accumulation — a crash in token
    bookkeeping kills the whole agent loop. Non-int values are treated as 0,
    i.e. "no usage reported":

      * Mock auto-attributes — ``getattr(mock, name, default)`` never returns
        the default for an unset name; it fabricates a truthy Mock.
      * JSON-decoded strings from gateway shims / non-conforming providers.
      * ``None`` (no usage report in the response).

    Only ``int`` (``bool`` included, as a subclass) passes through; anything
    else — including floats, which usage payloads never legitimately carry —
    is treated as 0.
    """
    return value if isinstance(value, int) else 0


def cache_hit_pct(provider: str, prompt_tok: int, cache_read_tok: int, cache_creation_tok: int = 0) -> float:
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
        provider,
        prompt_tok,
        completion_tok,
        cache_read_tok,
        cache_creation_tok,
        model,
        base_url=base_url,
    )
    return full_cost, actual_cost, cache_hit_pct(provider, prompt_tok, cache_read_tok, cache_creation_tok)


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
            _parts = line.split(" b/", 1)
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
            _dirs[:] = sorted(
                d
                for d in _dirs
                if not d.startswith(".")
                and d != "__pycache__"
                and d not in ("node_modules", "venv", ".venv", "dist", "build", ".git")
            )
            for _f in sorted(_files):
                if _f.startswith("."):
                    continue
                _rel = os.path.relpath(os.path.join(_root, _f), repo_root)
                if len(_rel) < 120:
                    result.append(_rel)
                if len(result) >= max_files:
                    return result
    except Exception as e:
        # Best-effort discovery for project.md auto-generation. A subtree that
        # cannot be walked (permission denied, vanished dir, cross-drive
        # relpath) must not crash session start — partial results are kept and
        # the failure stays traceable at debug level (silent-swallow gate).
        logger.debug("_discover_repo_files: walk of %s failed: %s", repo_root, e)
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
            _parts.extend(f"- {_f}" for _f in _all[:120])
            _content = "\n".join(_parts)
            try:
                os.makedirs(_asicode_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as _fw:
                    _fw.write(_content)
            except Exception as e:
                logger.warning("Failed to auto-generate %s: %s (disk full or permission denied)", path, e)
            return _content
        with open(path, encoding="utf-8") as _f:
            content = _f.read().strip()
        if not content:
            return ""
        return (
            "## ═══ PROJECT CONTEXT (.asicode/project.md) ═══\n"
            "Static architecture reference — use to skip exploratory file reads:\n\n" + content
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Could not load project context: %s", e)
        return ""


# ── Token estimation (shared by AgentLoop and DesignChatLoop) ──


def _cjk_aware_tokens(text: str) -> int:
    """Estimate tokens via ``utf8_bytes // 2`` (conservative upper bound).

    English/ASCII (~1 byte/char) yields ~2 chars/token.
    CJK text (~3 bytes/char) yields ~1.5 chars/token — a conservative
    upper bound that avoids the 2-3x underestimation of ``chars//3`` alone.
    Returns 0 for empty/None text.

    This is the single canonical token estimator for message content across
    the guard path (``estimate_tokens_from_msgs``).
    """
    if not text:
        return 0
    return len(text.encode("utf-8")) // 2 + 1


def estimate_tokens(text: str) -> int:
    """Estimate token count CJK-aware: ``utf8_bytes // 2``."""
    return _cjk_aware_tokens(text)


def _cjk_tokens_from_jsonable(obj: object) -> int:
    """Token estimate for an arbitrary JSON-serialisable object.

    Dumps to JSON with ``ensure_ascii=False`` (so CJK stays as multi-byte
    characters, one Python char each) then counts via the canonical byte-based
    estimator :func:`_cjk_aware_tokens`.  This keeps every JSON-args / wholesale
    path on the SAME fail-safe estimator as message content, so a CJK-heavy tool
    payload (Korean edit content, CJK bash output, ...) can never be
    under-counted into a context-overflow 400 — the exact failure this subsystem
    exists to prevent.

    The previous ``len(json.dumps(...)) // 3`` under-counted CJK ~2-3x because a
    3-byte Korean char counted as ~1/3 token instead of ~1.5.  Over-counting
    ASCII is safe and matches the documented budget philosophy.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return _cjk_aware_tokens(s)


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
    inp = block.get("input")
    if isinstance(inp, dict):
        n += _cjk_tokens_from_jsonable(inp)
    elif isinstance(inp, str):
        n += _cjk_aware_tokens(inp)
    tname = block.get("name", "")
    if tname:
        n += (len(tname) + 10) // 3 + 1
    return n


def _tok_tool_result(block: dict) -> int:
    """tool_result: ``content`` holds tool output (file reads, bash stdout, …)."""
    n = 0
    _tr_content = block.get("content")
    if isinstance(_tr_content, str):
        n += _cjk_aware_tokens(_tr_content)
    elif isinstance(_tr_content, list):
        for sub in _tr_content:
            if isinstance(sub, dict):
                stext = sub.get("text", "")
                if stext:
                    n += _cjk_aware_tokens(stext)
                elif sub.get("type") == "image":
                    n += _IMAGE_BLOCK_TOKEN_ESTIMATE
    return n


def _tok_thinking(block: dict) -> int:
    """thinking (Anthropic/zai-native): reasoning trace sent alongside text."""
    return _cjk_aware_tokens(block.get("thinking", ""))


def _tok_redacted_thinking(block: dict) -> int:
    """redacted_thinking: opaque signature payload, still on the wire."""
    return _cjk_aware_tokens(block.get("data", ""))


def _tok_function_call(block: dict) -> int:
    """Gemini functionCall part (typed or content-key form)."""
    fc = block.get("functionCall") or block.get("function_call")
    if isinstance(fc, dict):
        return _cjk_tokens_from_jsonable(fc)
    return 0


def _tok_function_response(block: dict) -> int:
    """Gemini functionResponse part (typed or content-key form)."""
    fr = block.get("functionResponse") or block.get("function_response")
    if isinstance(fr, dict):
        return _cjk_tokens_from_jsonable(fr)
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
    "tool_use": _tok_tool_use,
    "tool_result": _tok_tool_result,
    "thinking": _tok_thinking,
    "redacted_thinking": _tok_redacted_thinking,
    "functionCall": _tok_function_call,
    "functionResponse": _tok_function_response,
    "image": _tok_image,
}

# Gemini native ``parts`` carry the type as a TOP-LEVEL KEY rather than in a
# ``type`` field, so type dispatch misses them.  These markers re-route such
# untyped blocks to the matching tokenizer (preserving pre-registry behaviour).
_WIRE_CONTENT_KEY_MARKERS: dict[str, Any] = {
    "functionCall": _tok_function_call,
    "functionResponse": _tok_function_response,
}

# ── Intra-type drift guard ───────────────────────────────────────────────────
# The fail-safes above catch an unknown block TYPE and an untyped Gemini part.
# Neither catches drift INSIDE a known type: a registered tokenizer's answer was
# final, so a payload arriving under a key that tokenizer does not read counted
# as ~0.  Measured before this guard: a `thinking` block's `signature` — which
# Anthropic sends on EVERY extended-thinking block and which this client mirrors
# back verbatim (anthropic_client.py appends `raw_content` unchanged) — counted 0
# tokens at 615 chars; a 40 KB payload under `thinking.summary` or
# `tool_use.partial_json` (a real streaming field) likewise counted 0 and 4.
# That is the same silent under-count, and the same context-overflow 400, that
# the unknown-type fail-safe exists to prevent.
#
# So: each tokenizer declares the keys it consumes, and whatever is left over is
# counted wholesale. Drift then fails toward OVER-counting, matching the policy
# the rest of this subsystem already follows.
_WIRE_BLOCK_CONSUMED_KEYS: dict[str, frozenset[str]] = {
    "tool_use": frozenset({"input", "name"}),
    "tool_result": frozenset({"content"}),
    "thinking": frozenset({"thinking"}),
    "redacted_thinking": frozenset({"data"}),
    "functionCall": frozenset({"functionCall", "function_call"}),
    "functionResponse": frozenset({"functionResponse", "function_response"}),
    # image is a flat provider-cap estimate that deliberately ignores the
    # base64 payload (see _IMAGE_BLOCK_TOKEN_ESTIMATE) — counting `source`
    # wholesale would reintroduce the ~130k-token screenshot it exists to avoid.
    "image": frozenset({"source", "data"}),
}

# Keys that are pure wire structure. They ride along on every block and carry no
# payload, so counting them would inflate every correct-shape estimate without
# protecting against anything. `text` is here because the generic text pre-pass
# in _estimate_single_message_tokens already counted it for EVERY block type.
_WIRE_STRUCTURAL_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "index",
        "cache_control",
        "text",
    }
)

# Keys that a tokenizer does not read but that legitimately ride on the block.
# They are COUNTED (they are on the wire and billed) but never WARNED about:
# they are not drift, they are fields the tokenizers were simply never taught.
# Warning on them would make the drift counter fire on every single request and
# turn a signal that exists to catch a real regression into constant noise.
_WIRE_EXPECTED_EXTRA_KEYS: dict[str, frozenset[str]] = {
    "tool_use": frozenset({"id"}),
    "tool_result": frozenset({"tool_use_id", "is_error"}),
    # Anthropic sends `signature` on every extended-thinking block and this
    # client mirrors it back unchanged, so it is billed on every such turn.
    # Counting it is the leak this guard was written for; it is expected, so it
    # must not also raise a drift warning forever.
    "thinking": frozenset({"signature"}),
    "redacted_thinking": frozenset({"signature"}),
}


def _count_unconsumed_payload(block: dict, btype: str) -> tuple[int, tuple[str, ...]]:
    """Tokens in *block* its tokenizer did not read, and which of those are drift.

    Returns ``(tokens, drift_keys)``. Everything unconsumed is counted; only the
    keys that are neither consumed nor expected are reported as drift, so the
    warning stays a regression signal rather than a per-request refrain.
    ``drift_keys`` is empty in the overwhelmingly common case, established by one
    set difference.
    """
    consumed = _WIRE_BLOCK_CONSUMED_KEYS.get(btype)
    if consumed is None:
        return 0, ()
    residual = {
        k: v
        for k, v in block.items()
        if k not in consumed and k not in _WIRE_STRUCTURAL_KEYS and v not in (None, "", [], {})
    }
    if not residual:
        return 0, ()
    expected = _WIRE_EXPECTED_EXTRA_KEYS.get(btype, frozenset())
    drift = tuple(sorted(k for k in residual if k not in expected))
    return _cjk_tokens_from_jsonable(residual), drift


# The canonical set of wire block types this subsystem must recognise.  The
# contract test asserts the registry covers exactly this set.  When a provider
# adds a new type, add its fixture here AND register a tokenizer.
CANONICAL_WIRE_BLOCK_TYPES: frozenset[str] = frozenset(_WIRE_BLOCK_TOKENIZERS) | {"text"}

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

    Dumps the entire block to JSON and counts it via the canonical byte-based
    estimator (:func:`_cjk_tokens_from_jsonable`), guaranteeing we never
    under-count a whole unknown block — including CJK payloads, which the old
    ``chars // 3`` formula under-counted ~2-3x (under-counting is what causes
    the context-overflow 400s this subsystem prevents).  Slight over-counting is
    always safe — it only trims the budget marginally sooner.
    """
    return _cjk_tokens_from_jsonable(block)


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


def _warn_block_key_drift(btype: str, key: str) -> None:
    """Record a payload key a registered tokenizer does not read.

    Shares the counter and the one-time-per-entry discipline with
    :func:`_warn_unknown_block_type` (entries are namespaced ``type.key``, which
    cannot collide with a bare type name), but says something different: the
    block type IS known, so the fix is to teach its tokenizer — or, if the key
    is expected and simply unread, to list it in
    ``_WIRE_EXPECTED_EXTRA_KEYS`` so it stops being reported as drift.
    """
    entry = f"{btype}.{key}"
    with _unknown_block_types_lock:
        prev = _warned_unknown_block_types.get(entry, 0)
        _warned_unknown_block_types[entry] = prev + 1
        should_warn = prev == 0
    if should_warn:
        logger.warning(
            "LLM content-block %r carries payload key %r that its tokenizer does not "
            "read — counted wholesale (fail-safe). Teach the %s tokenizer to read it, "
            "or add it to _WIRE_EXPECTED_EXTRA_KEYS in "
            "external_llm/agent/_shared_utils.py if it is expected.",
            btype,
            key,
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


def _images_ocr_len(images: object) -> int:
    """Total length of any pre-computed ``ocr_text`` across *images*.

    Non-list ``images`` and non-dict elements yield 0 rather than raising: this
    runs on every cache probe and on every estimate, and the token estimator
    must degrade rather than crash on a malformed message.
    """
    if not isinstance(images, list):
        return 0
    return sum(len(img.get("ocr_text") or "") for img in images if isinstance(img, dict))


def _msg_token_fingerprint(m: object) -> tuple:
    """Length-based signature of every field that feeds the token estimate.

    A change in any of these lengths signals the cached ``_msg_token_estimate``
    is stale and must be recomputed.  Length (not a deep hash) keeps the guard
    cheap enough to run on every cache probe; it catches the realistic mutation
    paths (replacement, trimming, append, stubbing a tool result) which all
    change a container length.  Nested in-place edits that preserve every length
    are not detected — but no current mutator produces those, and the
    copy-on-write discipline remains the primary safety mechanism.

    ``images`` is the one known in-place mutator: ``_images_to_text`` writes an
    ``ocr_text`` key into each image dict without changing the list length, and
    that text feeds the estimate (see ``_estimate_single_message_tokens``).  So
    the OCR length is fingerprinted alongside the list length — otherwise the
    pre-OCR flat estimate stays cached and the message is under-counted, which
    is the context-overflow failure this subsystem exists to prevent.
    """
    rc = getattr(m, "raw_content", None)
    tc = getattr(m, "tool_calls", None)
    images = getattr(m, "images", None)
    reasoning = getattr(m, "reasoning_content", None)
    return (
        len(getattr(m, "content", "") or ""),
        len(rc) if isinstance(rc, (list, str)) else 0,
        len(tc) if isinstance(tc, list) else 0,
        len(images) if isinstance(images, list) else 0,
        _images_ocr_len(images),
        len(reasoning) if isinstance(reasoning, str) else (1 if reasoning else 0),
    )


def _msg_field(m: object, field: str, default: Any = "") -> Any:
    """Read *field* from an LLMMessage (attribute) or plain dict (key).

    Dict messages (Ollama wire format) use ``m[key]`` access; LLMMessage
    objects use ``getattr``.  This union lets ``_estimate_single_message_tokens``
    support both types without callers having to normalise first.
    """
    if isinstance(m, dict):
        return m.get(field, default)
    return getattr(m, field, default)


def _estimate_single_message_tokens(m: object) -> int:
    """Compute and cache token estimate for a single message object.

    Caches the result on ``m._msg_token_estimate`` so repeated calls (e.g.
    pre-trim + post-trim + overflow-retry in the same turn) skip re-counting
    and re-``json.dumps`` for messages that survive trimming.  The cache lives
    as long as the message object, which is exactly the turn lifetime.

    .. note::
        The cache is self-healing: every probe compares a length fingerprint
        (``_msg_token_fp``) of the counted fields against the current message and
        recomputes on mismatch.  This catches in-place mutation of
        ``.content`` / ``.raw_content`` / ``.tool_calls`` (replacement, trimming,
        append) even if a future mutator bypasses the copy-on-write discipline
        (``dataclasses.replace`` via ``_evict_consumed_tool_results`` ->
        ``_stub_tool_result``) that remains the primary safety mechanism.
        Nested edits that preserve every container length are not detected.

    Supports both ``LLMMessage`` objects (cache via attribute) and plain
    ``dict`` messages (always recompute — dict has no writable ``__dict__``).
    """
    # Cache check — only for mutable objects with __dict__ (not plain dicts).
    _can_cache = not isinstance(m, dict) and hasattr(m, "__dict__")
    if _can_cache:
        cached = getattr(m, "_msg_token_estimate", None)
        if cached is not None and getattr(m, "_msg_token_fp", None) == _msg_token_fingerprint(m):
            # Fingerprint guard: recompute if any counted field's length changed
            # since the estimate was cached.  Self-heals in-place mutation that
            # bypasses copy-on-write (see .. note:: above).  Lengths keep the
            # guard cheap while catching every realistic mutation path.
            return cached
        # fp mismatch -> fall through and recompute a fresh estimate.

    mt = 0
    # Content (CJK-aware) — skip when raw_content is present because it is the
    # authoritative wire form; content is a derived mirror. Counting both would
    # double-count assistant text (anthropic/zai/native assistant messages include
    # the same text in both content and raw_content text blocks).
    content = _msg_field(m, "content", "") or ""
    rc = _msg_field(m, "raw_content", None)
    # Type-guard: raw_content is typed Optional[list[dict]]; a non-list truthy
    # value (type violation, stray JSON string) must NOT be treated as
    # "content is covered" — that would under-count the message to ~0 tokens,
    # which is precisely the failure-mode this subsystem prevents.
    if not isinstance(rc, list) or not rc:
        mt += _cjk_aware_tokens(content)
    # Reasoning content (DeepSeek reasoner) — separate field sent on wire alongside content.
    # NOT covered by raw_content; this is a parallel attribute that always needs counting.
    reasoning_attr = _msg_field(m, "reasoning_content", None)
    if reasoning_attr:
        mt += _cjk_aware_tokens(reasoning_attr if isinstance(reasoning_attr, str) else str(reasoning_attr))
    # Tool calls — args are routed through the canonical byte-based estimator so
    # CJK-heavy payloads (Korean edit content, etc.) are not under-counted.  The
    # tool/function *name* is an ASCII identifier; the +10 covers the JSON
    # envelope and //3 is an adequate over-count for pure-ASCII names.
    tc = _msg_field(m, "tool_calls", None)
    if tc:
        try:
            for t in tc:
                args = t.get("args", t.get("function", {}).get("arguments", ""))
                if isinstance(args, dict):
                    mt += _cjk_tokens_from_jsonable(args)
                elif isinstance(args, str):
                    mt += _cjk_aware_tokens(args)
                elif args:
                    mt += _cjk_aware_tokens(str(args))
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
            text = block.get("text", "")
            if text:
                mt += _cjk_aware_tokens(text)
            btype = block.get("type")
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
                # A registered tokenizer reads the keys it knows; anything else
                # on the block is still on the wire and still billed. Counting
                # the residual is what keeps intra-type drift from going silent
                # (see _WIRE_BLOCK_CONSUMED_KEYS).
                _extra, _drift_keys = _count_unconsumed_payload(block, btype or "")
                if _extra:
                    mt += _extra
                    for _k in _drift_keys:
                        _warn_block_key_drift(btype or "", _k)
            elif btype in (None, "", "text"):
                # 'text' blocks and blocks without a type whose payload is covered
                # by the generic text pre-pass above — these are safe to skip.
                # BUT: Gemini untyped parts (inlineData, fileData, executableCode,
                # etc.) that have NO 'text' field and matched no content-key marker
                # are NOT covered by any pre-pass — they reach here as btype=None
                # and silently produce 0 tokens. Count them wholesale as fail-safe.
                if btype is None and not text and not any(_m in block for _m in _WIRE_CONTENT_KEY_MARKERS):
                    mt += _count_block_wholesale(block)
                    _warn_unknown_block_type("<untyped-gemini-part>")
            else:
                # Unknown wire block type — fail-safe toward OVER-counting so a
                # new provider block type can never silently trigger a context
                # overflow (the exact failure this subsystem prevents).  Drift is
                # surfaced via a one-time-per-type warning.
                mt += _count_block_wholesale(block)
                _warn_unknown_block_type(btype)
    # Images (provider-cap flat estimate — see _IMAGE_BLOCK_TOKEN_ESTIMATE docstring).
    # When an image dict carries a pre-computed ``ocr_text`` (set by
    # _images_to_text on first call), use its token-equivalent length
    # as a floor so that text-only model paths never under-count
    # Korean-heavy OCR output (which can be ~2x the flat cap).
    images = _msg_field(m, "images", None)
    if images:
        for img in images:
            ocr_len = len(img.get("ocr_text") or "") if isinstance(img, dict) else 0
            mt += max(_IMAGE_BLOCK_TOKEN_ESTIMATE, ocr_len // 2)

    # Cache on the message object for the turn lifetime (only for cacheable
    # objects).  Store a length fingerprint alongside so a later in-place
    # mutation is detected by the cache probe above (self-healing guard).
    if _can_cache:
        m._msg_token_estimate = mt  # type: ignore[attr-defined]  # cacheable objects (non-dict, has __dict__) accept dynamic attrs
        m._msg_token_fp = _msg_token_fingerprint(m)  # type: ignore[attr-defined]  # cacheable objects (non-dict, has __dict__) accept dynamic attrs
    return mt


def estimate_tokens_from_msgs(messages: list) -> int:
    """Estimate total token count from a list of LLMMessage objects.

    Counts content (CJK-aware) + tool_calls JSON args + raw_content blocks
    + images (provider-cap flat estimate via ``_IMAGE_BLOCK_TOKEN_ESTIMATE``).
    Uses per-message caching (``_msg_token_estimate``) so repeated calls in the
    same turn skip re-counting unchanged messages.
    """
    return sum(_estimate_single_message_tokens(m) for m in messages)


_tool_schema_token_cache: dict[tuple[int, tuple[str, ...]], int] = {}
"""Bounded ``content-fingerprint -> token-count`` cache.

Keyed on ``(len(tool_schemas), tuple(tool_names))`` instead of ``id()``: the
old id()-keyed cache was silently poisoned by address reuse (a freed small
schema list's id was handed to a freshly-allocated large schema list, yielding
a 10x+ under-count that survived because the freed id stayed in the cache).
The fingerprint is content-based, so it is immune to GC address reuse, while
staying cheaper than the ``json.dumps`` it exists to avoid.  Tool schemas are
loaded once per process and never mutated, so the name-set uniquely identifies
the (immutable) schema content -- the cache is both safe and collision-free
for the in-repo usage.
"""


def _tool_schema_fingerprint(tool_schemas: list) -> tuple[int, tuple[str, ...]]:
    """Cheap content key for :data:`_tool_schema_token_cache`.

    Reads only each schema's ``name`` (or ``function.name`` for OpenAI-style
    wrappers) -- O(n) string extraction, far cheaper than the full
    ``json.dumps`` the cache exists to avoid.  Mutation guard: revert the key
    to ``id(tool_schemas)`` -> test_tool_schema_token_cache tests FAIL.
    """
    names: list[str] = []
    for s in tool_schemas:
        nm = ""
        if isinstance(s, dict):
            nm = s.get("name") or ""
            if not nm:
                fn = s.get("function")
                nm = fn.get("name", "") if isinstance(fn, dict) else ""
        names.append(str(nm))
    return len(tool_schemas), tuple(names)


def estimate_tokens_from_tool_schemas(tool_schemas: list | None) -> int:
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
    # Content-fingerprint cache: identical name-set -> skip json.dumps. The key
    # is content-based (not id()) so GC address reuse can never poison it.
    _fp = _tool_schema_fingerprint(tool_schemas)
    _cached = _tool_schema_token_cache.get(_fp)
    if _cached is not None:
        return _cached
    # CJK-aware byte-based estimator (same fail-safe as message content): the
    # old chars/3 count under-estimated Korean/CJK descriptions ~2-3x, which
    # inflated context_message_cap and could 400 on CJK-heavy schemas.
    result = _cjk_tokens_from_jsonable(tool_schemas)
    # FIFO-bounded via the shared SSOT helper (same family as the file-index and
    # walk caches) — evicts only the oldest entry instead of nuking the whole
    # cache on the 9th distinct schema, so recently-used schemas stay warm.
    _capped_put(_tool_schema_token_cache, _fp, result, cap=8)
    return result


# The smallest message budget at which a prompt can still be useful. When
# ``ctx_limit - output_reserve - tool_tokens`` falls below this, the window is
# structurally too small for the current toolset — ``context_message_cap``
# logs a diagnosis (once per signature) instead of silently returning its 512
# floor, and ``_record_context_overflow`` (context_budget.py) refuses to
# reduce the window below the matching structural floor.
MIN_USABLE_MESSAGE_BUDGET: int = 2048

# (ctx_limit, output_reserve, tool_tokens) signatures already diagnosed —
# prevents one identical ERROR per LLM call while an impossible budget persists.
_IMPOSSIBLE_BUDGET_WARNED: set = set()


def context_message_cap(
    ctx_limit: int, safety_margin: int, tool_schemas: list | None = None, tool_tokens: int | None = None
) -> int:
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
    _tool_tokens = estimate_tokens_from_tool_schemas(tool_schemas) if tool_tokens is None else tool_tokens
    _raw = ctx_limit - _output_reserve - _tool_tokens
    if _raw < MIN_USABLE_MESSAGE_BUDGET:
        # Structurally impossible: even with zero chat history the output
        # reserve + tool schemas already exceed the window. Surface the numbers
        # once per signature instead of silently returning the 512 floor —
        # which hides the cause (and the fix: smaller toolset / larger window)
        # forever.
        _sig = (ctx_limit, _output_reserve, _tool_tokens)
        if _sig not in _IMPOSSIBLE_BUDGET_WARNED:
            _IMPOSSIBLE_BUDGET_WARNED.add(_sig)
            logger.error(
                "Context budget is structurally impossible for this model: "
                "window=%d, output reserve=%d, tool schemas=%d → only %d tokens "
                "left for messages (below the %d minimum). Reduce the toolset "
                "or use a larger context window.",
                ctx_limit,
                _output_reserve,
                _tool_tokens,
                _raw,
                MIN_USABLE_MESSAGE_BUDGET,
            )
    return max(512, _raw)


def render_file_diagnostics_block(diags: Any) -> str:
    """Render a ``<file_diagnostics>`` guidance block from raw diagnostic dicts.

    Shared by the inline path (``AgentLoop._append_semantic_diagnostics``) and
    the turn-end deferred-settlement path
    (``TurnPipelineMixin._settle_deferred_semantics``) so a check that was
    coalesced to turn end reaches the model through the SAME formatted channel
    as an inline one, instead of only as raw JSON in ``metadata``. The block is
    the channel the surrounding code documents as "which the LLM parses
    reliably"; without this, every in-turn write (all deferred) would lose it.

    Keeps error/warning severities only, de-duplicates by
    ``(file_path, line, message)`` and caps the shown list at 15 (the running
    totals are still reported). Returns ``""`` when nothing remains, so callers
    can treat a falsy result as "append nothing" — identical to the prior
    inline behaviour where an empty filtered list left ``content`` untouched.
    """
    seen: set = set()
    filtered: list = []
    total = 0
    n_err = 0
    n_warn = 0
    for d in diags or []:
        if not isinstance(d, dict):
            continue
        sev = (d.get("severity") or "error").lower()
        if sev not in ("error", "warning"):
            continue
        key = (d.get("file_path", ""), d.get("line"), d.get("message", ""))
        if key in seen:
            continue
        seen.add(key)
        total += 1
        if sev == "error":
            n_err += 1
        else:
            n_warn += 1
        if len(filtered) >= 15:
            continue
        filtered.append(d)
    if not filtered:
        return ""
    suppressed = total - len(filtered)
    lines = ["\n\n<file_diagnostics>"]
    lines.append(
        f"Semantic check found {total} unique issue(s) "
        f"({n_err} error, {n_warn} warning), showing {len(filtered)} below. "
        f"The edit was applied, but these may cause runtime failures — "
        f"consider fixing them next."
    )
    if suppressed > 0:
        lines.append(
            f"... {suppressed} more {'issues' if suppressed > 1 else 'issue'} "
            f"suppressed (run the validator directly for full output)"
        )
    for d in filtered:
        sev = (d.get("severity") or "error").lower()
        tag = "Error" if sev == "error" else "Warn"
        loc = ""
        if d.get("line") is not None:
            col = d.get("column") or d.get("col")
            loc = f":{d.get('line')}" + (f":{col}" if col else "")
        file_ = d.get("file_path", "") or ""
        file_short = file_.rsplit("/", 1)[-1] if file_ else ""
        code = d.get("code")
        code_str = f" [{code}]" if code else ""
        lines.append(f"{tag}: {file_short}{loc}{code_str} {d.get('message', '').strip()}")
    lines.append("</file_diagnostics>")
    return "\n".join(lines)
