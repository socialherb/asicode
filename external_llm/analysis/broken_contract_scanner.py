"""Broken-contract scanner — finds writer/reader pairs split by migration.

Detects the regression class first seen in two real bugs:

  * ``mark_f821_protected()`` — writer whose original caller
    (``repair_core.py``) was dropped when F821 handling migrated to
    ``tool_safety.py``; the writer (and its in-memory cache) survived but
    was never re-wired to the new call site (fixed in commit ``407f766e``).
  * ``set_pending_impl_spec()`` / ``pop_pending_impl_spec()`` — a
    writer/reader pair over a shared session field. The writer was removed
    in ``1a6839a4`` (``switch_to_planner`` simplification deleted the
    confirm-handoff workflow) but the reader stayed live, so the reader
    silently returned ``None`` forever (reader side deleted subsequently).

Why the other dead-code scanners miss it
----------------------------------------
``public_dead_code_scanner`` and ``dead_block_scanner`` reason about a
*single* symbol's reference count. In a broken pair **exactly one half is
still alive**, so its reference count is ≥ 1 and it looks healthy — the
other (already-removed) half is no longer in the AST to be flagged. The
contract between the two is invisible to per-symbol reachability.

Detection strategy (structural, not keyword-based)
---------------------------------------------------
1. **Core-name grouping.** Functions are clustered by their *core* name —
   the name with a small set of state-access verbs stripped from the front
   (``set_``/``get_``/``pop_``/``peek_``/``mark_``/``clear_``/``reset_``/
   ``push_``/``add_``/``remove_``/``register_``/``unregister_``/``fetch_``/
   ``load_``/``save_``/``store_``/``delete_``/``has_``/``is_``). A pair
   ``set_pending_impl_spec`` / ``pop_pending_impl_spec`` shares the core
   ``pending_impl_spec``. Only names that survive stripping (non-trivial
   core) are grouped, so ``get()`` / ``set()`` are excluded.
2. **Shared-state gate (false-positive suppression).** Two functions that
   merely share a core name are *not* necessarily a contract. The scanner
   verifies that the bodies touch a common piece of state via AST
   inspection: the writer side must *mutate* (assignment target /
   subscript-store / attribute-store / ``open(path, 'w')``) and the reader
   side must *read* (attribute-load / subscript-load / ``open(path, 'r')``)
   the same name/path. Pairs without shared state are dropped. This is what
   distinguishes a real contract from an accidental name collision.
3. **Caller-count asymmetry.** For each verified pair the graph reports
   the number of caller edges per member. A pair is *broken* when the
   members' caller counts differ and the minimum is 0 — one half is
   unreachable (orphan reader / orphan writer). When *both* are 0 the pair
   is wholly dead and is left to ``public_dead_code_scanner``.

The scanner needs the repository graph (``get_symbols_in_file`` /
``get_callers``) and therefore declares ``requires_graph=True``. It runs
over Python only (the verb-prefix heuristic is calibrated for Python
naming); tree-sitter extraction for other languages is a future
extension.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..languages import LanguageId
from . import parse_cache

logger = logging.getLogger(__name__)

__all__ = ["BrokenContractCandidate", "scan_broken_contracts"]


# Verbs stripped to derive a function's *core* name. Ordered so multi-word
# prefixes (e.g. ``unregister_``) are tried before their substrings. Only a
# leading, single verb is stripped — ``reset_all_stats`` → ``all_stats``,
# never ``all`` (the second token is never a verb).
_ACCESS_VERBS: tuple[str, ...] = (
    "unregister_",
    "register_",
    "remove_",
    "delete_",
    "clear_",
    "reset_",
    "push_",
    "store_",
    "save_",
    "load_",
    "fetch_",
    "append_",
    "update_",
    "mark_",
    "set_",
    "get_",
    "pop_",
    "peek_",
    "has_",
    "is_",
    "add_",
)


@dataclass
class BrokenContractCandidate:
    """A writer/reader pair where exactly one half is unreachable.

    ``role`` identifies which member is the orphan so the fix is obvious:
    re-wire the missing caller to it (or, if the contract was intentionally
    deleted, remove the orphan too).
    """
    file: str
    core_name: str
    # The two members of the pair. Each dict carries the bare name, kind,
    # lineno and the state-read/write summary used to validate the pair.
    members: list[dict[str, Any]] = field(default_factory=list)
    orphan_role: str = ""  # "reader" | "writer" — which half has 0 callers
    orphan_name: str = ""
    shared_state: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "core_name": self.core_name,
            "members": [dict(m) for m in self.members],
            "orphan_role": self.orphan_role,
            "orphan_name": self.orphan_name,
            "shared_state": list(self.shared_state),
        }


# ── AST state-access extraction ──────────────────────────────────────────────


def _strip_access_verb(name: str) -> str:
    """Return *name* with one leading access verb removed, or "" if trivial.

    ``set_pending_impl_spec`` → ``pending_impl_spec``;
    ``mark_f821_protected`` → ``f821_protected``;
    ``get()`` → "" (single-token, trivial — not a contract candidate).
    """
    if not name or "_" not in name:
        return ""
    for verb in _ACCESS_VERBS:
        if name.startswith(verb):
            core = name[len(verb):]
            # Reject single-token cores (``get_x`` → ``x``) that are too
            # generic to be meaningful, and require at least one underscore
            # so the core itself reads as a noun phrase.
            if core and "_" in core:
                return core
            return ""
    return ""


def _dotted_target(node: ast.AST) -> Optional[str]:
    """Rebuild ``self._cache`` / ``self._cache["k"]`` target string.

    Returns a stable string capturing the *location* being touched
    (attribute chain or subscript base) so writer/reader bodies can be
    compared for a shared-state overlap. The exact subscript key is
    deliberately dropped — ``self._d["a"] = 1`` and ``self._d["a"]`` share
    the base ``self._d``.
    """
    if isinstance(node, ast.Attribute):
        base = _dotted_target(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        base = _dotted_target(node.value)
        return base  # drop the key, keep the base
    return None


def _state_accesses(body: list[ast.stmt]) -> tuple[set[str], set[str]]:
    """Return ``(writes, reads)`` — the set of state locations a body touches.

    A *write* is any assignment target, a subscript-store, an attribute-store
    (``self.x = ...``), or an ``open(path, 'w'|'a')`` call (file-backed state).
    A *read* is any attribute-load / subscript-load on the same kinds of
    targets, or ``open(path, 'r')``. ``self`` attribute access dominates —
    module-level globals are also captured via ``Name``.

    The split lets the caller confirm the writer half actually *mutates* and
    the reader half actually *reads* — the contract invariant.
    """
    writes: set[str] = set()
    reads: set[str] = set()

    for node in body:
        # Assignment targets → writes.
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                loc = _dotted_target(tgt)
                if loc:
                    writes.add(loc)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            loc = _dotted_target(getattr(node, "target", None))
            if loc:
                writes.add(loc)

        # Walk the whole body once for attribute/subscript reads & writes
        # and for open() calls.
        for sub in ast.walk(node):
            # Attribute store: ``self.x.y = ...`` appears as an Assign target
            # (already handled above) but also inside augmented assigns and
            # calls like ``self.buf.append(..)`` where the *method call*
            # mutates — we treat any attribute-load whose base looks like a
            # private store as a write candidate when a known mutator method
            # is called on it.
            if isinstance(sub, ast.Call):
                _classify_open_call(sub, writes, reads)
                _classify_mutator_call(sub, writes)
                _classify_dynamic_attr_call(sub, writes, reads)

    # Second pass: attribute/subscript loads that are *not* assignment
    # targets are reads. We re-walk the top-level body statements so we can
    # distinguish load-in-target-context (handled) from pure reads.
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                loc = _dotted_target(sub)
                # Skip the outermost ``self`` token alone.
                if loc and loc not in ("self", "cls"):
                    if loc in writes:
                        continue
                    reads.add(loc)
            elif isinstance(sub, ast.Subscript):
                loc = _dotted_target(sub)
                if loc and loc not in ("self", "cls"):
                    if loc in writes:
                        continue
                    reads.add(loc)

    return writes, reads


# Common method names that mutate the receiver. Calling any of these on an
# attribute is treated as a write of that attribute's location.
_MUTATOR_METHODS: frozenset = frozenset({
    "append", "extend", "insert", "add", "update", "pop", "popleft",
    "remove", "discard", "setdefault", "clear", "sort", "reverse",
    "put", "set", "write", "writelines", "seek",
})


def _classify_mutator_call(call: ast.Call, writes: set[str]) -> None:
    """Record ``self.x.append(..)`` / ``self.x.update(..)`` as a write of ``self.x``."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in _MUTATOR_METHODS:
        loc = _dotted_target(func.value)
        if loc and loc not in ("self", "cls"):
            writes.add(loc)


def _classify_dynamic_attr_call(call: ast.Call, writes: set[str], reads: set[str]) -> None:
    """Recognize ``getattr(obj, "name")`` / ``setattr(obj, "name", v)`` / ``delattr``.

    These dynamic accessors appear in lazy-initialised attributes (the real
    ``pop_pending_impl_spec`` reader uses ``getattr(self, "_pending_impl_spec",
    None)``). Without recognising them the reader looks stateless and the
    shared-state gate rejects the pair — exactly the bug we are trying to catch.

    The attribute name must be a literal string; dynamic names are dropped to
    avoid false pairing. The location is keyed as ``<base>.<name>`` so it
    matches the ``self._x`` shape produced by direct attribute access.
    """
    if not isinstance(call.func, ast.Name):
        return
    fn = call.func.id
    if fn not in ("getattr", "hasattr", "setattr", "delattr"):
        return
    if len(call.args) < 2:
        return
    base = _dotted_target(call.args[0])
    if not base:
        return
    # ``self``/``cls`` IS a valid base for getattr/setattr — the whole point
    # is to catch ``getattr(self, "_x", None)``. Only reject empty bases.
    name = _literal_str(call.args[1])
    if not name:
        return
    loc = f"{base}.{name}"
    if fn in ("setattr", "delattr"):
        writes.add(loc)
    else:  # getattr / hasattr → read
        reads.add(loc)


def _classify_open_call(call: ast.Call, writes: set[str], reads: set[str]) -> None:
    """Treat ``open(path, mode)`` as file-state access keyed by the path."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "open"):
        return
    if not call.args:
        return
    path = _literal_path(call.args[0])
    if not path:
        return
    mode = ""
    if len(call.args) >= 2:
        mode = _literal_str(call.args[1]) or ""
    key = f"open({path})"
    if "w" in mode or "a" in mode or "+" in mode:
        writes.add(key)
    else:
        reads.add(key)


def _literal_str(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _literal_path(node: ast.AST) -> Optional[str]:
    """Return a stable key for an ``open()`` path argument.

    Literal strings are used verbatim; ``self._path`` style attributes are
    used as-is (without resolving the runtime value). Anything dynamic is
    dropped (returns None) so we never falsely pair on a meaningless key.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        loc = _dotted_target(node)
        return loc
    if isinstance(node, ast.Name):
        return node.id
    return None


# ── Pairing & caller asymmetry ───────────────────────────────────────────────


def _member_caller_count(graph: Any, name: str, repo_root: str, file_path: str) -> int:
    """Number of caller edges for *name* in the repo graph.

    Cross-file referenced-name semantics mirror ``cross_file_refs``:
    ``resident_entry_point_names`` (scanner ``scan_*`` callables) are treated
    as live so a scanner entry point is never misclassified as an orphan.
    """
    if graph is None:
        return -1
    try:
        callers = graph.get_callers(name) or []
        if callers:
            return len(callers)
    except Exception:
        return -1
    # Suffix-fallback already handled inside get_callers, but the index keys
    # qualified names (``Class.method``). Try the file-scoped definition name
    # as well in case the bare name does not index-match.
    try:
        syms = graph.get_symbols_in_file(file_path) or []
        for sym in syms:
            sym_name = getattr(sym, "name", "") or getattr(sym, "symbol_name", "")
            if sym_name and sym_name.endswith(f".{name}"):
                callers = graph.get_callers(sym_name) or []
                if callers:
                    return len(callers)
    except Exception:
        pass
    return 0


def _is_scanner_entry_point(name: str) -> bool:
    """True when *name* is a registered scanner callable name.

    Scanner entry points (``scan_dead_blocks`` etc.) are alive by construction
    via ``ScannerRegistry.register`` but have no static call edge. Without
    this suppression they would be flagged as orphans — the same suppression
    ``cross_file_refs._scanner_resident_entry_points`` applies.
    """
    if not name.startswith("scan_"):
        return False
    try:
        from ..agent.scanner_registry import get_registry
        return name in get_registry().resident_entry_point_names()
    except Exception:
        return False


# ── Scanner entry point ──────────────────────────────────────────────────────


def scan_broken_contracts(
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = 20,
    repo_graph: object = None,
    **_unused: Any,
) -> list[BrokenContractCandidate]:
    """Find writer/reader pairs split by migration (one half unreachable).

    Args:
        repo_root: Repository root (for graph caller queries).
        file_paths: Python files to scan (other extensions are skipped).
        max_per_file: Cap on candidates returned per file.
        repo_graph: Repository graph facade (``get_callers`` /
            ``get_symbols_in_file``). Required for caller asymmetry; when
            absent the scanner returns no candidates rather than guess.

    Returns:
        Candidates ordered by file then core name.
    """
    if repo_graph is None or not file_paths:
        return []
    if not (hasattr(repo_graph, "get_callers") and hasattr(repo_graph, "get_symbols_in_file")):
        logger.debug(
            "[BROKEN_CONTRACT] graph lacks caller/symbol API (%s) — skipping",
            type(repo_graph).__name__,
        )
        return []

    results: list[BrokenContractCandidate] = []
    for fpath in file_paths:
        if LanguageId.from_path(fpath) is not LanguageId.PYTHON:
            continue
        abs_path = fpath if os.path.isabs(fpath) else os.path.join(repo_root, fpath)
        tree = parse_cache.parse_ast(abs_path)
        if tree is None:
            continue
        per_file = _scan_module(tree, fpath, repo_graph, repo_root, max_per_file)
        results.extend(per_file)
        if len(results) >= max_per_file * max(1, len(file_paths)):
            break
    return results


def _scan_module(
    tree: ast.Module,
    rel_path: str,
    graph: Any,
    repo_root: str,
    max_per_file: int,
) -> list[BrokenContractCandidate]:
    """Group top-level & method defs by core name, validate pairs."""
    # Collect every function/method def with a non-trivial core name.
    grouped: dict[str, list[dict[str, Any]]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("_") and not name.startswith("__"):
            # Private helpers are usually fine; the contract bugs so far were
            # public-facing set_/pop_ methods. Keep __dunder__ for completeness.
            pass
        core = _strip_access_verb(name)
        if not core:
            continue
        writes, reads = _state_accesses(node.body)
        grouped.setdefault(core, []).append({
            "name": name,
            "core": core,
            "lineno": node.lineno,
            "end_lineno": getattr(node, "end_lineno", node.lineno),
            "writes": writes,
            "reads": reads,
            "node": node,
        })

    candidates: list[BrokenContractCandidate] = []
    for core, members in grouped.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                cand = _evaluate_pair(members[i], members[j], core, rel_path, graph, repo_root)
                if cand is not None:
                    candidates.append(cand)
                    if len(candidates) >= max_per_file:
                        return candidates
    candidates.sort(key=lambda c: (c.file, c.core_name))
    return candidates


def _evaluate_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    core: str,
    rel_path: str,
    graph: Any,
    repo_root: str,
) -> Optional[BrokenContractCandidate]:
    """Validate one candidate pair; return a BrokenContractCandidate or None."""
    a_writes, a_reads = a["writes"], a["reads"]
    b_writes, b_reads = b["writes"], b["reads"]

    # Shared-state gate: one side must write and the other read the same loc.
    shared = _shared_state(a_writes, a_reads, b_writes, b_reads)
    if not shared:
        return None

    # Caller asymmetry via the graph.
    a_calls = _member_caller_count(graph, a["name"], repo_root, rel_path)
    b_calls = _member_caller_count(graph, b["name"], repo_root, rel_path)
    # Negative = graph unavailable for this symbol; cannot decide.
    if a_calls < 0 or b_calls < 0:
        return None
    # Both reachable → healthy pair. Both dead → leave to public_dead_code.
    if a_calls > 0 and b_calls > 0:
        return None
    if a_calls > 0 or b_calls > 0:
        # Exactly one is unreachable → broken contract.
        orphan = a if a_calls == 0 else b
        live = b if a_calls == 0 else a
    else:
        # Both zero — wholly dead, defer to public_dead_code_scanner.
        return None

    # Scanner entry points are alive by construction; never report them.
    if _is_scanner_entry_point(orphan["name"]) or _is_scanner_entry_point(live["name"]):
        return None

    orphan_role = _role_of(orphan)
    return BrokenContractCandidate(
        file=rel_path,
        core_name=core,
        members=[
            _member_summary(a),
            _member_summary(b),
        ],
        orphan_role=orphan_role,
        orphan_name=orphan["name"],
        shared_state=sorted(shared),
    )


def _shared_state(
    a_w: set[str], a_r: set[str],
    b_w: set[str], b_r: set[str],
) -> set[str]:
    """Locations where one side writes and the other reads (or both write).

    A real contract means the pair shares state: side A writes it, side B
    reads it (or vice-versa). Pure overlap of reads (both getters query the
    same cache but neither writes) is *not* a contract — it is two readers.

    Access locations are matched with a *base-prefix* rule so that a write of
    ``_cache`` and a read of ``_cache.get`` (or ``self._d["k"]`` vs
    ``self._d``) count as the same store. Without this, dict/cache patterns
    — where the writer assigns the whole mapping and the reader calls a
    method on it — would slip through, including the ``mark_f821_protected``
    regression (writes ``_cache[..]``, read-side ``_cache.get(..)``).
    """
    b_reads = b_r | b_w  # a read OR write on B's side touches the state
    a_reads = a_r | a_w
    shared: set[str] = set()
    # A writes, B touches (reads or writes).
    for loc in a_w:
        if _loc_overlaps(loc, b_reads):
            shared.add(loc)
    # B writes, A touches.
    for loc in b_w:
        if _loc_overlaps(loc, a_reads):
            shared.add(loc)
    return shared


def _loc_overlaps(loc: str, others: set[str]) -> bool:
    """True when *loc* shares a state base with any location in *others*.

    Two locations overlap when one is a prefix of the other up to a ``.``
    boundary (``_cache`` ~ ``_cache.get``) or they are equal. Subscript
    accessors like ``self._d["k"]`` already collapsed to ``self._d`` in
    ``_dotted_target``, so they match plain ``self._d`` exactly.
    """
    for o in others:
        if loc == o:
            return True
        # loc is a prefix of o (loc == "_cache", o == "_cache.get")
        if o.startswith(loc + "."):
            return True
        # o is a prefix of loc (loc == "_cache.get", o == "_cache")
        if loc.startswith(o + "."):
            return True
    return False


def _role_of(member: dict[str, Any]) -> str:
    """Classify a member as "reader" or "writer" by its dominant access."""
    if member["writes"]:
        return "writer"
    return "reader"


def _member_summary(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": m["name"],
        "lineno": m["lineno"],
        "end_lineno": m["end_lineno"],
        "role": _role_of(m),
        "caller_count": m.get("caller_count", -1),
    }
