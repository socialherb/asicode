"""auto_correction.py — Operation Auto-Correction Layer.

Sits between planner output and executor input.  For each operation,
resolves the actual file/symbol state and rewrites misclassified ops
so the executor never attempts impossible actions (e.g. modify a
symbol that doesn't exist, anchor_edit with None pattern).

Architecture:
  OperationPlan
    → validate_operation_preflight(op)     # fail-fast for invalid ops
    → resolve_operation_facts(op)          # deterministic file/symbol scan
    → auto_correct_operation(op, facts)    # rule-based rewrite
    → execute(corrected_op)                # executor runs corrected op

Key invariant: if facts show the op is already valid, it passes through
unchanged (action=keep).  Only misclassified ops are rewritten.

Rules:
  Rule A: modify_symbol + symbol exists       → keep
  Rule B: modify_symbol + symbol missing      → rewrite/repair/skip
  Rule C: create_file + file exists           → keep (downstream)
  Rule D: insert_after_symbol + anchor missing → fallback
  Rule E: anchor_edit + missing pattern/file  → skip
  Rule F: anchor_edit + anchor not found      → rewrite to insert_after or skip
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Set

from ..code_structure_utils import find_last_top_level_def
from ..languages import LanguageId  # S8 fix: missing module-level import
from ._shared_utils import ts_symbol_exists_in_file
from .config.thresholds import config
from .operation_models import Operation, OperationKind, normalize_op_semantic_fields

logger = logging.getLogger(__name__)

# ── Symbol similarity helpers ─────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (O(m*n) — strings are short)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _tokenize_symbol(name: str) -> list[str]:
    """Split a symbol name into lowercase tokens.

    Handles:
    - UPPER_SNAKE_CASE  → ["upper", "snake", "case"]
    - _private_const    → ["private", "const"]  (leading underscores stripped)
    - CamelCase         → ["camel", "case"]
    - mixedCase         → ["mixed", "case"]

    Unifies auto_correction simplicity with target_resolver completeness:
    both re.split on underscores AND split CamelCase boundaries.
    """
    # strip leading/trailing underscores
    s = name.strip("_")
    # split on underscores, then split each CamelCase token
    # replace '.' with '_' so qualified names (e.g., "Tetris._update") split properly
    s = s.replace('.', '_')
    tokens: list[str] = []
    for part in s.split('_'):
        if not part:
            continue
        # split CamelCase within each underscore-delimited part (char scanner, no regex)
        # If ALL characters are uppercase, treat as a single token (not individual chars)
        if part.isupper():
            tokens.append(part.lower())
        else:
            res: list[str] = []
            part_len = len(part)
            for i, ch in enumerate(part):
                if ch.isupper() and res:
                    prev_is_lower = part[i - 1].islower()
                    next_is_lower = (i + 1 < part_len) and part[i + 1].islower()
                    if prev_is_lower:
                        res.append('_')  # myMethod → my_method
                    elif part[i - 1].isupper() and next_is_lower:
                        res.append('_')  # HTTPRequest → http_request
                res.append(ch.lower())
            camel_split = ''.join(res)
            tokens.extend(t for t in camel_split.split('_') if t)
    tokens.append(s.lower())  # stripped — leading _ ignored for scoring
    return tokens


def compute_symbol_similarity(a: str, b: str) -> float:
    """Return a [0, 1] similarity score for two symbol names.

    Combines:
    - Jaccard similarity on token sets               (weight 0.45)
    - Prefix/substring match bonus between tokens    (weight 0.20)
    - Edit-distance similarity on normalised strings (weight 0.35)

    Keeps leading underscores out of scoring — ``_FOO`` and ``FOO`` are the
    same base name, just with different visibility conventions.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a_tokens = set(_tokenize_symbol(a))
    b_tokens = set(_tokenize_symbol(b))

    if not a_tokens or not b_tokens:
        return 0.0

    # Jaccard on token sets
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    jaccard = len(intersection) / len(union)

    # Token prefix/substring bonus (each qualifying pair contributes once)
    prefix_bonus = 0.0
    for ta in a_tokens:
        if len(ta) < 3:
            continue
        for tb in b_tokens:
            if len(tb) < 3:
                continue
            if ta == tb:
                continue  # already captured by Jaccard
            short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            if long.startswith(short) or long.endswith(short):
                prefix_bonus = max(prefix_bonus, len(short) / len(long))

    # No cap — ratio is already ≤ 1.0; capping at 0.40 was penalising legitimate prefix matches

    # Edit-distance similarity on full normalised strings
    a_norm = "_".join(_tokenize_symbol(a))
    b_norm = "_".join(_tokenize_symbol(b))

    # Identical after normalisation (e.g. case variation, leading _) → same name
    if a_norm == b_norm:
        return 1.0

    ld = _levenshtein(a_norm, b_norm)
    max_len = max(len(a_norm), len(b_norm))
    edit_sim = 1.0 - (ld / max_len) if max_len > 0 else 0.0

    # prefix_bonus is naturally ≤ 1.0 (short/long ratio) — no extra cap needed
    score = config.scores.SIM_WEIGHT_JACCARD * jaccard + config.scores.SIM_WEIGHT_PREFIX * prefix_bonus + config.scores.SIM_WEIGHT_EDIT * edit_sim
    return round(min(score, 1.0), 4)


# Similarity thresholds for auto-correction decisions
_SIM_AUTO_CORRECT = config.scores.AUTO_CORRECT   # ≥ this → rewrite op to use nearest symbol
_SIM_HINT_ONLY    = config.scores.HINT_ONLY   # ≥ this → add "did you mean X?" to replan_hint


# ── Magic methods that are "creatable" inside a class ────────────────────

_MAGIC_METHODS = {
    "__post_init__", "__init__", "__repr__", "__str__", "__eq__",
    "__hash__", "__lt__", "__le__", "__gt__", "__ge__", "__len__",
    "__iter__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__enter__", "__exit__", "__call__",
}

# ── Validation-related keywords (Korean + English) ───────────────────────

@dataclass(frozen=True)
class _ValidationClassifier:
    """Case-insensitive keyword classifier — no regex.

    Multi-word keywords use substring ``in``.  Single-word alphanumeric
    keywords use ``split()`` for word-boundary matching (equivalent to
    ``\b`` but without regex).  Non-alphanumeric tokens (e.g. ``"len("``)
    use ``in`` — too distinctive for false positives.
    """
    name: str
    keywords: frozenset

    _SENTINEL = object()  # sentinel for non-alphanumeric token detection

    def match(self, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        words = None  # lazy split
        for kw in self.keywords:
            if " " in kw or not (kw[-1].isalnum() or kw[-1] == "_"):
                # Multi-word or non-alphanumeric token → substring ``in``
                if kw in text_lower:
                    return True
            else:
                # Single-word alphanumeric → word-boundary via split()
                if words is None:
                    words = text_lower.split()
                if kw in words:
                    return True
        return False


_VALIDATION_KEYWORDS = _ValidationClassifier(
    name="validation",
    keywords=frozenset({
        "검증", "validation", "validate", "validator", "제약", "constraint",
        "최소", "minimum", "min_length", "max_length", "len(", "assert", "raise ValueError",
    }),
)


def _is_import_binding_in_source(symbol: str, source: str) -> bool:
    """Check if *symbol* is an import binding (ESM or CJS) in TS/JS source.

    Import bindings are names introduced by import / require statements.
    They are valid anchors for insert_after_symbol but are NOT tracked by
    the dependency graph (which only records definitions).  Detecting them
    here prevents RULE_D from redirecting import-anchored insertions to
    the last top-level definition (e.g. shutdown()), which would place
    import code at EOF instead of at the file header.

    Detection strategy (hybrid):

    1. **Tree-sitter AST** — parses ESM import/export statements via
       ``extract_import_names``.  Avoids false positives from comments
       and string literals (a blind spot of pure regex).  Only available
       when ``tree_sitter_utils`` is installed (optional dep).

    2. **CJS require() regex** — always runs as a supplement because
       ``extract_import_names`` does not cover CommonJS require bindings.
       These patterns are narrow enough (``require(``) that false positives
       in comments/strings are very rare in practice.

    3. **Full ESM regex fallback** — only runs when tree-sitter is NOT
       available, preserving the original 7-pattern coverage for
       environments without the optional tree-sitter package.
    """
    if not symbol or not source:
        return False

    # ── Phase 1: tree-sitter AST (ESM imports/exports only) ────────────────
    # Authoritative for ESM: tree-sitter covers every ESM form (default /
    # named / namespace / type) and ignores import-like text inside comments
    # and strings — the exact blind spot of the Phase 3 regex fallback.
    # ``_ts_ran`` therefore gates Phase 3: if tree-sitter ran and did NOT
    # match, the symbol is not an ESM import, and the regex must NOT run to
    # reintroduce those false positives.
    _ts_ran = False
    try:
        from ..languages.tree_sitter_utils import (  # noqa: I001
            extract_import_names as _ts_extract_import_names,
            is_available as _ts_available,
        )
        if _ts_available():
            _ts_ran = True
            for _lang in ("typescript", "javascript"):
                for _mod, _name in _ts_extract_import_names(source, _lang):
                    if _name == symbol:
                        return True
    except Exception:
        pass  # Fall through to regex paths below

    _esc = re.escape(symbol)

    # ── Phase 2: CJS require() patterns (tree-sitter does not capture these) ─
    # const WebSocket = require('ws')
    if re.search(rf'(?:const|let|var)\s+{_esc}\s*=\s*require\s*\(', source):
        return True
    # const { WebSocket, Server } = require('ws')
    # The `= require(` tail anchor is required: without it, any local
    # destructuring (e.g. `const { cfg } = loadConfig()`) would be misread
    # as an import binding, suppressing RULE_D's missing-anchor redirect.
    if re.search(
        rf'(?:const|let|var)\s*\{{[^}}]*\b{_esc}\b[^}}]*\}}\s*=\s*require\s*\(',
        source,
    ):
        return True

    # ── Phase 3: ESM regex fallback (only when tree-sitter did NOT run) ──────
    # When tree-sitter is available it is authoritative for ESM imports, so we
    # skip these regexes to avoid reintroducing comment/string false positives.
    if not _ts_ran:
        # ESM default import:   import WebSocket from 'ws'
        if re.search(rf'^import\s+{_esc}\b', source, re.MULTILINE):
            return True
        # ESM named import:     import { WebSocket } from 'ws'
        if re.search(rf'import\s*\{{[^}}]*\b{_esc}\b[^}}]*\}}', source):
            return True
        # ESM namespace import: import * as WebSocket from 'ws'
        if re.search(rf'import\s+\*\s+as\s+{_esc}\b', source):
            return True
        # ESM type import:       import type { WebSocket } from 'ws'
        if re.search(rf'import\s+type\s*\{{[^}}]*\b{_esc}\b[^}}]*\}}', source):
            return True
        # ESM import type:       import { type WebSocket } from 'ws'
        if re.search(rf'import\s*\{{[^}}]*type\s+{_esc}\b[^}}]*\}}', source):
            return True

    return False


# ── Class field detection helper ────────────────────────────────────────


def _is_dotted_name_class_field(file_path: str, class_name: str, dotted_name: str) -> bool:
    """Check if dotted_name (e.g. "ToolConfig.timeout") refers to a class field.

    Scans the file's AST for ``class ClassName`` and checks if the second
    component of ``dotted_name`` is an ``AnnAssign`` or ``Assign`` in the
    class body.  Used as safety net when ``_find_symbol_node`` (P8-S2) fails
    to resolve dataclass fields.
    """
    try:
        parts = dotted_name.split(".")
        if len(parts) != 2:
            return False
        _child_name = parts[1]
        with open(file_path) as _f:
            _tree = ast.parse(_f.read())
        for _node in ast.iter_child_nodes(_tree):
            if isinstance(_node, ast.ClassDef) and _node.name == class_name:
                for _child in ast.iter_child_nodes(_node):
                    if isinstance(_child, ast.AnnAssign):
                        if isinstance(_child.target, ast.Name) and _child.target.id == _child_name:
                            return True
                    elif isinstance(_child, ast.Assign):
                        for _target in _child.targets:
                            if isinstance(_target, ast.Name) and _target.id == _child_name:
                                return True
    except Exception:
        return False
    return False


# ── Data Models ──────────────────────────────────────────────────────────

@dataclass
class ResolutionFacts:
    """Deterministic file/symbol state resolved before execution."""
    file_exists: bool = False
    symbol_exists: bool = False
    parent_symbol_exists: bool = False
    parent_symbol: str = ""
    symbol_kind: str = "unknown"       # class / function / method / unknown
    is_magic_method: bool = False
    creatable: bool = False
    anchor_found: bool = False         # for anchor_edit: whether anchor pattern matches
    anchor_line: int = -1              # matched line number (-1 = not found)
    anchor_match_count: int = 0        # number of matches for anchor pattern
    reason: str = ""
    is_import_binding: bool = False    # TS/JS: symbol is an import binding (not tracked by graph)
    abs_path: str = ""                 # resolved absolute path (set by resolve_operation_facts)
    # When symbol_exists=False and parent_name is set (dotted name), this flag
    # indicates that the CHILD part of the dotted name (e.g. "_attach_edit_contracts"
    child_exists_at_module_level: bool = False
    child_name_bare: str = ""          # bare child name when child_exists_at_module_level
    # When symbol_exists=False, the top-scored similar symbols found in the same
    # file.  Each entry is (symbol_name, similarity_score).  Only populated for
    # Python files; TS/JS leaves this empty (the tracer already handles renaming).
    nearest_symbols: list[tuple[str, float]] = field(default_factory=list)
    # TS/JS type alias names from the file's AST.  Used to detect and fix
    # "TypeName.Member" member access where TypeName is a type alias (not an
    # enum), which would be a runtime error.  Populated by resolve_operation_facts.
    type_alias_names: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "file_exists": self.file_exists,
            "symbol_exists": self.symbol_exists,
            "parent_symbol_exists": self.parent_symbol_exists,
            "parent_symbol": self.parent_symbol,
            "symbol_kind": self.symbol_kind,
            "is_magic_method": self.is_magic_method,
            "creatable": self.creatable,
            "anchor_found": self.anchor_found,
            "anchor_line": self.anchor_line,
            "anchor_match_count": self.anchor_match_count,
            "reason": self.reason,
        }
        if self.nearest_symbols:
            d["nearest_symbols"] = self.nearest_symbols
        return d


@dataclass
class AutoCorrectionDecision:
    """Result of auto-correction evaluation for one operation."""
    action: str = "keep"               # keep / rewrite / skip / route_to_repair
    original_kind: str = ""
    corrected_kind: str = ""
    corrected_symbol: str = ""
    corrected_anchor: str = ""
    rationale: str = ""
    confidence: float = 1.0
    repair_kind: str = ""              # e.g. "add_validation" when route_to_repair
    repair_detail: dict[str, Any] = field(default_factory=dict)
    # Preflight replan signal: when True, executor should trigger evidence-based
    # replan instead of silently skipping. Only set when skip has concrete evidence
    # that the planner can act on (symbol/anchor not found in known file).
    should_replan: bool = False
    replan_hint: str = ""              # human-readable evidence for planner

    def to_dict(self) -> dict[str, Any]:
        d = {
            "action": self.action,
            "original_kind": self.original_kind,
            "corrected_kind": self.corrected_kind,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }
        if self.corrected_symbol:
            d["corrected_symbol"] = self.corrected_symbol
        if self.corrected_anchor:
            d["corrected_anchor"] = self.corrected_anchor
        if self.repair_kind:
            d["repair_kind"] = self.repair_kind
        if self.repair_detail:
            d["repair_detail"] = self.repair_detail
        return d


# ── Preflight Validation ────────────────────────────────────────────────

def validate_operation_preflight(op: Operation) -> Optional[str]:
    """Validate an operation has required fields before execution.

    Delegates to Operation.validate() — the single source of truth.
    """
    return op.validate()


# ── Resolution ───────────────────────────────────────────────────────────

def resolve_operation_facts(
    op: Operation,
    repo_root: str,
) -> ResolutionFacts:
    """Resolve the actual file/symbol state for an operation.

    Pure function — reads filesystem and AST, no side effects.
    """
    facts = ResolutionFacts()
    path = op.path or ""
    symbol = op.symbol or ""

    if not path:
        facts.reason = "no_path"
        return facts

    abs_path = os.path.join(repo_root, path) if not os.path.isabs(path) else path
    facts.abs_path = abs_path
    facts.file_exists = os.path.isfile(abs_path)

    # Fallback: if repo_root-based resolution fails, try the raw path
    # (handles cases where the CWD is already the repo root or the path
    # is provided as an absolute path without repo_root context).
    if not facts.file_exists and repo_root:
        _raw_path = path if os.path.isabs(path) else os.path.abspath(path)
        if _raw_path != abs_path and os.path.isfile(_raw_path):
            logger.info(
                "[RESOLVE_FACTS] path resolution fallback — repo_root %r did not "
                "resolve %r; found via raw path %r",
                repo_root, path, _raw_path,
            )
            facts.abs_path = _raw_path
            facts.file_exists = True

    if not facts.file_exists:
        if not repo_root:
            logger.debug(
                "[RESOLVE_FACTS] file_not_found for %r — repo_root is empty; "
                "auto-correction may be running without repo context",
                path,
            )
        facts.reason = "file_not_found"
        return facts

    # For anchor_edit: check if anchor pattern exists in file
    if op.kind == OperationKind.ANCHOR_EDIT and op.anchor_pattern:
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            pattern = op.anchor_pattern
            match_count = 0
            first_match = -1
            for i, line in enumerate(lines):
                if pattern in line:
                    match_count += 1
                    if first_match < 0:
                        first_match = i
            facts.anchor_found = match_count > 0
            facts.anchor_match_count = match_count
            facts.anchor_line = first_match
        except Exception:
            pass  # non-critical — never block execution

    if not symbol:
        facts.reason = "no_symbol" if op.kind != OperationKind.ANCHOR_EDIT else "anchor_edit_no_symbol"
        return facts

    # Parse parent/child from dotted symbol
    parent_name = ""
    child_name = symbol
    if "." in symbol:
        parts = symbol.split(".")
        parent_name = parts[0]
        child_name = parts[-1]

    facts.is_magic_method = child_name in _MAGIC_METHODS

    # Read source
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception as e:
        facts.reason = f"read_error: {e}"
        return facts

    # ── Dispatch: TS/JS TSSemanticTracer vs Provider-based vs Python AST ──
    _lang_id = LanguageId.from_path(path)
    _is_ts_js = _lang_id in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT)
    _is_python = _lang_id == LanguageId.PYTHON

    if _is_ts_js:
        # TS/JS: use TSSemanticTracer for symbol resolution
        _all_symbols: dict[str, str] = {}  # name → kind
        _class_methods: dict[str, list[str]] = {}  # class → [methods]
        _has_classes: dict[str, bool] = {}
        try:
            from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer
            _lang_id = LanguageId.from_path(path)
            _lang_str = "typescript" if _lang_id == LanguageId.TYPESCRIPT else "javascript"
            _tracer = TSSemanticTracer(language=_lang_str)
            _module = _tracer.analyze_core(source, path)
            for fn in _module.functions:
                _all_symbols[fn.name] = "function"
            for cls in _module.classes:
                _all_symbols[cls.name] = "class"
                _has_classes[cls.name] = True
                _meths = []
                for m in cls.methods:
                    _all_symbols[f"{cls.name}.{m.name}"] = "method"
                    _meths.append(m.name)
                _class_methods[cls.name] = _meths
            for iface in _module.interfaces:
                _all_symbols[iface.name] = "interface"
            for var in _module.variables:
                _all_symbols[var.name] = "variable"
            for en in _module.enums:
                _all_symbols[en.name] = "enum"
            for ta in _module.type_aliases:
                _all_symbols[ta.name] = "type_alias"
            facts.type_alias_names = {ta.name for ta in _module.type_aliases}
        except Exception as e:
            logger.debug("auto_correction: TS parse failed %s: %s", path, e)
            # Fallback: use SyntaxProvider.find_symbol_in_file
            try:
                from ..languages import LanguageRegistry
                _prov = LanguageRegistry.instance().get(path)
                if _prov:
                    _result = _prov.find_symbol_in_file(abs_path, child_name, source)
                    if _result:
                        _all_symbols[child_name] = "function"
            except (ImportError, AttributeError):
                pass

        # Check symbol existence
        if symbol in _all_symbols:
            facts.symbol_exists = True
            facts.symbol_kind = _all_symbols[symbol]
        elif child_name in _all_symbols and not parent_name:
            facts.symbol_exists = True
            facts.symbol_kind = _all_symbols[child_name]

        # Check parent
        if parent_name:
            if parent_name in _has_classes:
                facts.parent_symbol_exists = True
                facts.parent_symbol = parent_name
                facts.symbol_kind = "method"
                if not facts.symbol_exists:
                    facts.creatable = True

        if not facts.symbol_exists and not parent_name:
            facts.creatable = True

        # For INSERT_AFTER_SYMBOL ops, the symbol field should be the anchor
        # (an existing symbol).  If it was NOT found by the tracer or regex
        # fallback above, do NOT override to True — let Rule D in
        # auto_correct_operation handle the missing-anchor recovery (e.g.
        # redirect to last top-level def).  Overriding to True here would
        # bypass Rule D and send a hallucinated symbol straight to the TS VM.
        # (Bug: run_20260606_230851 — handleMessage was incorrectly marked
        # as existing, bypassing auto-correction.)
        # ══ Import binding detection (TS/JS) ════════════════════════════
        # TSSemanticTracer and regex fallbacks only detect top-level
        # definitions (functions/classes/variables).  Import bindings
        # (e.g. 'WebSocket' from 'import WebSocket from "ws"') are valid
        # anchors for insert_after_symbol, but are NOT tracked by the
        # dependency graph.  Detect them here so RULE_D does not redirect
        # them to the last top-level definition.
        if not facts.symbol_exists:
            facts.is_import_binding = _is_import_binding_in_source(
                symbol or child_name, source
            )

        # ══ Fallback: regex-based ts_symbol_exists_in_file ═══════════════
        # TSSemanticTracer misses certain symbol types:
        #   - bare class method names (``constructor`` vs ``Game.constructor``)
        # The regex-based fallback in _shared_utils handles these.
        # Note: type_aliases (``type PieceType = ...``) are no longer missed
        # — they are collected in the type_aliases loop above.
        if not facts.symbol_exists:
            try:
                if ts_symbol_exists_in_file(abs_path, symbol):
                    facts.symbol_exists = True
            except Exception:
                pass  # non-critical

        facts.reason = "resolved"
        return facts

    # ── Provider-based resolution (Go, Java, Kotlin, etc.) ──
    if not _is_python:
        _provider = None
        try:
            from ..languages import LanguageRegistry
            _provider = LanguageRegistry.instance().get(path)
        except Exception:
            pass

        if _provider is not None:
            _all_symbols: dict[str, str] = {}
            _has_classes: dict[str, bool] = {}

            # Build search names: dotted full name + bare child name
            _lines = source.splitlines()
            _search_names: list[str] = []
            if "." in symbol:
                _search_names.append(symbol)      # e.g. "TodoList.Add"
            if child_name:
                _search_names.append(child_name)  # e.g. "Add" (or "Todo" for bare)

            for _search_name in _search_names:
                if _search_name in _all_symbols:
                    continue  # already found
                _esc_name = re.escape(_search_name)
                for sp in _provider.get_symbol_patterns(kind="any"):
                    _pat = re.compile(sp.regex.replace("{name}", _esc_name))
                    for _line in _lines:
                        if _pat.search(_line):
                            _all_symbols[_search_name] = sp.kind
                            break
                    if _search_name in _all_symbols:
                        break

            # Detect parent classes/types for dotted-name resolution
            if parent_name:
                _esc_parent = re.escape(parent_name)
                for _kind in ("class", "type"):
                    if _has_classes.get(parent_name):
                        break
                    for sp in _provider.get_symbol_patterns(kind=_kind):
                        if sp.regex:
                            _pat = re.compile(sp.regex.replace("{name}", _esc_parent))
                            for _line in _lines:
                                if _pat.search(_line):
                                    _has_classes[parent_name] = True
                                    break

            # Check symbol existence
            if symbol in _all_symbols:
                facts.symbol_exists = True
                facts.symbol_kind = _all_symbols[symbol]
            elif child_name in _all_symbols:
                # Parent-aware: if parent exists, treat as method; otherwise bare symbol
                if parent_name:
                    if _has_classes.get(parent_name):
                        facts.symbol_exists = True
                        facts.symbol_kind = "method"
                else:
                    facts.symbol_exists = True
                    facts.symbol_kind = _all_symbols[child_name]

            # Check parent
            if parent_name:
                if parent_name in _has_classes:
                    facts.parent_symbol_exists = True
                    facts.parent_symbol = parent_name
                    if not facts.symbol_kind:
                        facts.symbol_kind = "method"
                    if not facts.symbol_exists:
                        facts.creatable = True
                elif not facts.symbol_exists:
                    # Parent not found but symbol is dotted — child may still
                    # exist at module level (unqualified)
                    facts.creatable = True

            if not facts.symbol_exists and not parent_name:
                facts.creatable = True

            facts.reason = "resolved"
            return facts

        # ── UNKNOWN language: no provider registered ──
        # Simple fallback: bare-name constant assignment detection
        _lines = source.splitlines()
        _found = False
        for _line in _lines:
            _s = _line.strip()
            if _s.startswith(f"{symbol} =") and not _s.startswith(" "):
                _found = True
                facts.symbol_kind = "constant"
                break

        if _found:
            facts.symbol_exists = True
        elif not parent_name:
            facts.creatable = True

        facts.reason = "resolved"
        return facts

    # ── Python: existing AST path (unchanged) ──
    try:
        tree = ast.parse(source)
    except Exception as e:
        facts.reason = f"parse_error: {e}"
        return facts

    # Collect all top-level and nested definitions
    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.symbols: dict[str, str] = {}       # name → kind
            self.classes: dict[str, ast.ClassDef] = {}
            self.methods: dict[str, list[str]] = {}  # class_name → [method_names]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.symbols[node.name] = "class"
            self.classes[node.name] = node
            meths = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.symbols[f"{node.name}.{item.name}"] = "method"
                    meths.append(item.name)
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            self.symbols[f"{node.name}.{tgt.id}"] = "class_attribute"
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        self.symbols[f"{node.name}.{item.target.id}"] = "class_attribute"
            self.methods[node.name] = meths
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name not in self.symbols:
                self.symbols[node.name] = "function"
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.name not in self.symbols:
                self.symbols[node.name] = "function"
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.asname or (alias.name.split(".")[0] if "." in alias.name else alias.name)
                self.symbols.setdefault(name, "import")

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                self.symbols.setdefault(name, "import")

    visitor = _Visitor()
    visitor.visit(tree)

    # Also collect module-level constant assignments (Assign / AnnAssign at tree.body level)
    # These are not functions or classes, so the visitor skips them — but they DO exist
    # and should be treated as "symbol_exists=True" (kind="constant") so that
    # auto-correction does NOT rewrite modify_symbol to insert_after_symbol.
    _module_vars: dict[str, str] = {}
    for _stmt in tree.body:
        if isinstance(_stmt, ast.Assign):
            for _tgt in _stmt.targets:
                if isinstance(_tgt, ast.Name):
                    _module_vars[_tgt.id] = "constant"
        elif isinstance(_stmt, ast.AnnAssign):
            if isinstance(_stmt.target, ast.Name):
                _module_vars[_stmt.target.id] = "constant"
        elif isinstance(_stmt, ast.Import):
            for _alias in _stmt.names:
                _name = _alias.asname or (_alias.name.split(".")[0] if "." in _alias.name else _alias.name)
                _module_vars.setdefault(_name, "import")
        elif isinstance(_stmt, ast.ImportFrom):
            for _alias in _stmt.names:
                if _alias.name != "*":
                    _name = _alias.asname or _alias.name
                    _module_vars.setdefault(_name, "import")

    # Check symbol existence
    if symbol in visitor.symbols:
        facts.symbol_exists = True
        facts.symbol_kind = visitor.symbols[symbol]
    elif child_name in visitor.symbols and not parent_name:
        facts.symbol_exists = True
        facts.symbol_kind = visitor.symbols[child_name]
    elif child_name in _module_vars and not parent_name:
        facts.symbol_exists = True
        facts.symbol_kind = _module_vars[child_name]
    elif symbol in _module_vars:
        facts.symbol_exists = True
        facts.symbol_kind = _module_vars[symbol]

    # Check parent
    if parent_name:
        if parent_name in visitor.classes:
            facts.parent_symbol_exists = True
            facts.parent_symbol = parent_name
            if not facts.symbol_kind:
                facts.symbol_kind = "method"

            if not facts.symbol_exists and facts.is_magic_method:
                facts.creatable = True
            elif not facts.symbol_exists:
                facts.creatable = True

            # ── Module-level unqualification check ──────────────────────────
            # If the dotted name "Class.method" is not found as a class method
            if not facts.symbol_exists:
                _child_kind = visitor.symbols.get(child_name) or _module_vars.get(child_name)
                if _child_kind in ("function", "class", "variable"):
                    facts.child_exists_at_module_level = True
                    facts.child_name_bare = child_name

    if not facts.symbol_exists and not parent_name:
        facts.creatable = True

    # For INSERT_AFTER_SYMBOL ops, the symbol field may hold the NEW symbol
    # name being created rather than the anchor.  If the file exists and has
    # parseable definitions, the handler will resolve the anchor — treat as
    # "exists" so auto-correction does not skip or redirect the operation.
    #
    # Virtual anchors: __imports__ and __doc__ are routing directives, not
    # literal file symbols.  Mark them as "exists" unconditionally so Rule D
    # does not redirect them to a real symbol (last top-level def), which
    # would place import code at the end of the file instead of the header.
    if op.kind == OperationKind.INSERT_AFTER_SYMBOL and symbol in ("__imports__", "__doc__"):
        facts.symbol_exists = True
    if not facts.symbol_exists and op.kind == OperationKind.INSERT_AFTER_SYMBOL:
        if facts.file_exists and (visitor.symbols or _module_vars):
            facts.symbol_exists = True

    # ── Nearest-symbol candidates (only when symbol is missing) ──────────
    # Build a flat candidate pool: bare symbol names from visitor + module vars.
    # For dotted names (Class.method), skip — nearest-symbol search only makes
    # sense for module-level rename targets.
    if not facts.symbol_exists and not parent_name and symbol:
        _candidate_names = (
            # module-level functions / classes (bare names only)
            [n for n in visitor.symbols if "." not in n]
            + list(_module_vars.keys())
        )
        _scored: list[tuple[str, float]] = []
        for _cand in _candidate_names:
            _sim = compute_symbol_similarity(symbol, _cand)
            if _sim > 0.0:
                _scored.append((_cand, _sim))
        # Keep top-5 by score, filter out noise (score < 0.15)
        _scored.sort(key=lambda x: x[1], reverse=True)
        facts.nearest_symbols = [
            (n, s) for n, s in _scored[:5] if s >= 0.15
        ]

    facts.reason = "resolved"
    return facts


# ── Correction Rule Helpers ──────────────────────────────────────────────


def _find_def_near_line(path: str, anchor_line: int) -> Optional[str]:
    """Scan backwards from *anchor_line* for the nearest ``def``/``class``.

    Pure text scan — no AST, no tree-sitter.  Handles both indented methods
    and module-level definitions.  Used by RULE D to recover a precise
    anchor location from FixSpec's ``anchor_line`` metadata.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as _fh:
            _lines = _fh.readlines()
    except OSError:
        return None
    for _ln in range(min(anchor_line, len(_lines)), 0, -1):
        _stripped = _lines[_ln - 1].lstrip()
        if _stripped.startswith(('def ', 'class ')):
            return _stripped.split(None, 2)[1]
        if _stripped.startswith('async '):
            _rest = _stripped[6:].lstrip()
            if _rest.startswith(('def ', 'class ')):
                return _rest.split(None, 2)[1]
    return None


# ── Correction Rules ─────────────────────────────────────────────────────

def auto_correct_operation(
    op: Operation,
    facts: ResolutionFacts,
    plan_metadata: Optional[dict[str, Any]] = None,
    intent: str = "",
) -> AutoCorrectionDecision:
    """Apply deterministic correction rules to a single operation.

    Rules (in priority order):
      A: modify_symbol + symbol exists       → keep
      B: modify_symbol + symbol missing      → rewrite/repair/skip
      C: create_file + file exists           → keep (downstream)
      D: insert_after_symbol + anchor missing → fallback
      E: anchor_edit + missing/invalid       → skip with reason
      F: anchor_edit + anchor not found      → rewrite or skip

    Returns a decision describing what (if anything) to change.
    """
    decision = AutoCorrectionDecision(
        original_kind=op.kind.value if isinstance(op.kind, OperationKind) else str(op.kind),
    )
    plan_metadata = plan_metadata or {}

    # ── Rule A: modify_symbol + symbol exists → keep ─────────────────
    if op.kind == OperationKind.MODIFY_SYMBOL and facts.symbol_exists:
        decision.action = "keep"
        decision.corrected_kind = op.kind.value
        decision.rationale = "symbol_exists"
        return decision

    # ── Rule B: modify_symbol + symbol missing ───────────────────────
    if op.kind == OperationKind.MODIFY_SYMBOL and not facts.symbol_exists:
        if facts.parent_symbol_exists:
            # B.1: Magic method with validation intent → route_to_repair
            if facts.is_magic_method and _has_validation_intent(op, plan_metadata, intent):
                decision.action = "route_to_repair"
                decision.corrected_kind = "route_to_change_spec_repair"
                decision.corrected_symbol = facts.parent_symbol
                decision.rationale = "missing_magic_method_with_validation_intent"
                decision.repair_kind = "add_validation"
                decision.repair_detail = {
                    "parent_class": facts.parent_symbol,
                    "method_name": (op.symbol or "").split(".")[-1],
                }
                decision.confidence = 0.95
                return decision

            # B.2: Magic method without validation intent → insert_after_symbol
            if facts.is_magic_method:
                decision.action = "rewrite"
                decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
                decision.corrected_symbol = facts.parent_symbol
                decision.corrected_anchor = facts.parent_symbol
                decision.rationale = "missing_magic_method_insert"
                decision.confidence = 0.90
                return decision

            # B.3a: Dotted name is actually a module-level symbol (wrong qualification).
            # e.g. "PlannerAgent._attach_edit_contracts" where _attach_edit_contracts
            if facts.child_exists_at_module_level:
                logger.info(
                    "auto_correct[B.3a]: qualified name '%s' → bare module-level symbol '%s' "
                    "(wrong qualification by planner; parent class '%s' exists but '%s' is not a method)",
                    op.symbol, facts.child_name_bare, facts.parent_symbol, facts.child_name_bare,
                )
                decision.action = "rewrite"
                decision.corrected_kind = OperationKind.MODIFY_SYMBOL.value
                decision.corrected_symbol = facts.child_name_bare
                decision.rationale = "qualified_name_is_module_level_symbol"
                decision.confidence = 0.90
                return decision

            # B.3b-1: Dotted name is a class field (AnnAssign/Assign), not a missing method.
            # modify_symbol for class fields (e.g. dataclass field deletion) should
            # pass through as-is — rewriting to insert_after_symbol would misplace the op.
            if _is_dotted_name_class_field(facts.abs_path, facts.parent_symbol, op.symbol):
                logger.info(
                    "auto_correct[B.3b-1]: dotted name '%s' is a class field in '%s' "
                    "— keeping modify_symbol op as-is",
                    op.symbol, facts.parent_symbol,
                )
                decision.action = "keep"
                decision.corrected_kind = op.kind.value
                decision.rationale = "symbol_is_class_field"
                decision.confidence = 0.90
                return decision

            # B.3b: Regular method missing in class → insert_after_symbol
            decision.action = "rewrite"
            decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
            decision.corrected_symbol = facts.parent_symbol
            decision.corrected_anchor = facts.parent_symbol
            decision.rationale = "missing_method_insert_after_parent"
            decision.confidence = 0.85
            return decision

        # B.4: No parent, symbol simply doesn't exist
        if facts.creatable:
            # For rename/refactor operations the source symbol MUST already exist.
            # Auto-creating it via insert_after_symbol would produce a dangling
            # new definition instead of renaming — worse, insert_after also fails
            # because it cannot find a non-existent anchor.
            _action_hint = (op.metadata or {}).get("action_hint", "")
            if _action_hint in ("refactor", "rename"):
                # B.4-R: Nearest-symbol fuzzy correction for rename/refactor.
                # If a close-enough match exists in the file, auto-correct the op
                _nearest = facts.nearest_symbols  # [(name, score), ...]
                if _nearest and _nearest[0][1] >= _SIM_AUTO_CORRECT:
                    _best_name, _best_score = _nearest[0]
                    logger.info(
                        "auto_correct[B.4-R]: refactor '%s' → nearest '%s' "
                        "(sim=%.3f ≥ %.2f) in %s",
                        op.symbol, _best_name, _best_score, _SIM_AUTO_CORRECT, op.path,
                    )
                    decision.action = "rewrite"
                    decision.corrected_kind = OperationKind.MODIFY_SYMBOL.value
                    decision.corrected_symbol = _best_name
                    decision.rationale = "refactor_nearest_symbol_autocorrect"
                    decision.confidence = _best_score
                    return decision

                # Not close enough to auto-correct — skip with an informative hint
                decision.action = "skip"
                decision.corrected_kind = op.kind.value
                decision.rationale = "refactor_symbol_not_found"
                decision.confidence = 0.90
                decision.should_replan = True
                if _nearest and _nearest[0][1] >= _SIM_HINT_ONLY:
                    _hint_names = ", ".join(
                        f"'{n}' (sim={s:.2f})" for n, s in _nearest[:3]
                    )
                    decision.replan_hint = (
                        f"Symbol '{op.symbol}' not found in {op.path}. "
                        f"Did you mean one of: {_hint_names}? "
                        f"This is a rename/refactor — the source symbol must exist."
                    )
                else:
                    # No useful match — list all constants/functions so the planner
                    # can select the right one.
                    _all_names = [n for n, _ in (_nearest or [])][:8]
                    _avail = ", ".join(f"'{n}'" for n in _all_names) if _all_names else "(none found)"
                    decision.replan_hint = (
                        f"Symbol '{op.symbol}' not found in {op.path}. "
                        f"This is a rename/refactor — the source symbol must already "
                        f"exist. Available symbols in file include: {_avail}."
                    )
                return decision

            # B.4-G (removed): General rewrite to INSERT_AFTER_SYMBOL at last
            # top-level def was harmful.  Contract 3 already handles legitimate
            # new-symbol-in-modify_symbol cases (intent with code block).  When
            # ── B.4-FN: File name used as symbol ──────────────────────────────
            # Bug (2026-06-02): Planner LLM frequently generates ops where the
            # symbol field is the file's basename (e.g. symbol='constants' for
            # constants.py, symbol='game' for game.py).  These are not real AST
            # symbols — skip with a targeted replan_hint.
            if op.path and op.symbol:
                _file_stem = os.path.splitext(os.path.basename(op.path))[0]
                if op.symbol == _file_stem:
                    decision.action = "skip"
                    decision.corrected_kind = op.kind.value
                    decision.rationale = "file_name_used_as_symbol"
                    decision.confidence = 0.95
                    decision.should_replan = True
                    decision.replan_hint = (
                        f"Symbol '{op.symbol}' matches the file name '{_file_stem}{os.path.splitext(op.path)[1]}' "
                        f"in {op.path} — this is a file/module name, not an AST symbol. "
                        f"Use an actual symbol from the file (class, function, or constant definition). "
                        f"Available symbols: {', '.join(n for n, _ in (facts.nearest_symbols or [])[:5])}"
                    )
                    return decision

            # Contract 3 skips (prose-only intent), the symbol is almost certainly
            # hallucinated or a system-level issue (wrong path, context truncation).
            # Rewriting it to INSERT_AFTER_SYMBOL produces code in the wrong location.
            # Skip with replan instead — safer, and the planner gets a retry.
            # Inject facts.nearest_symbols into replan_hint so the planner knows
            # which symbols actually exist in the file and avoids re-hallucinating
            # the same wrong name.  (Design insight 2026-05-22: hallucination loop fix)
            _hint_available = ""
            if facts.nearest_symbols:
                _hint_names = ", ".join(
                    f"'{n}' (sim={s:.2f})" for n, s in facts.nearest_symbols[:5]
                )
                _hint_available = f" Did you mean one of: {_hint_names}?"
            else:
                # No close matches — at least confirm the file was parsed
                _hint_available = (
                    " No similar symbols found in file. Try verifying the symbol "
                    "name and file path."
                )
            decision.action = "skip"
            decision.corrected_kind = op.kind.value
            decision.rationale = "symbol_not_found_hallucinated"
            decision.confidence = 0.90
            decision.should_replan = True
            decision.replan_hint = (
                f"Symbol '{op.symbol}' not found in {op.path}."
                f"{_hint_available}"
            )
            return decision

        # B.5: Not creatable — fail fast with replan evidence
        _hint_avail_b5 = ""
        if facts.nearest_symbols:
            _hint_names_b5 = ", ".join(
                f"'{n}' (sim={s:.2f})" for n, s in facts.nearest_symbols[:5]
            )
            _hint_avail_b5 = f" Did you mean one of: {_hint_names_b5}?"
        decision.action = "skip"
        decision.corrected_kind = op.kind.value
        decision.rationale = "symbol_not_found_not_creatable"
        decision.confidence = 0.90
        decision.should_replan = True
        decision.replan_hint = (
            f"Symbol '{op.symbol}' not found in {op.path} and cannot be created."
            f"{_hint_avail_b5}"
        )
        return decision

    # ── Rule C: create_file + file exists → keep (handled downstream) ─
    if op.kind == OperationKind.CREATE_FILE and facts.file_exists:
        decision.action = "keep"
        decision.corrected_kind = op.kind.value
        decision.rationale = "create_file_exists_handled_downstream"
        return decision

    # ── Rules D-pre & D: INSERT_AFTER_SYMBOL ──────────────────────────
    if op.kind == OperationKind.INSERT_AFTER_SYMBOL:
        _sym_stripped = (op.symbol or "").strip()

        # Rule D-pre: import statement as symbol
        if _sym_stripped.startswith(("from ", "import ")):
            logger.info(
                "[RULE_D_PRE] INSERT_AFTER_SYMBOL symbol looks like an import statement '%s' "
                "— converting to INSERT_IMPORT op",
                _sym_stripped[:80],
            )
            decision.action = "rewrite"
            decision.corrected_kind = OperationKind.INSERT_IMPORT.value
            decision.corrected_symbol = _sym_stripped
            decision.rationale = "insert_after_symbol_is_import_stmt_converted_to_insert_import"
            decision.confidence = 0.95
            return decision

        # Rule D: virtual anchors (passthrough)
        if _sym_stripped in ("__imports__", "__doc__"):
            decision.action = "keep"
            decision.corrected_kind = op.kind.value
            decision.rationale = "virtual_anchor_passthrough"
            return decision
    if op.kind == OperationKind.INSERT_AFTER_SYMBOL and not facts.symbol_exists:
        if facts.parent_symbol_exists:
            decision.action = "rewrite"
            decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
            decision.corrected_symbol = facts.parent_symbol
            decision.corrected_anchor = facts.parent_symbol
            decision.rationale = "insert_anchor_missing_fallback_to_parent"
            decision.confidence = 0.80
            return decision

        if facts.file_exists:
            # ── Line-based recovery (FixSpec path) ────────────────────────
            # When the op is from FixSpec materialization, the metadata
            # preserves the original anchor_line.  Use it to find the nearest
            # existing definition above that line — this is far more precise
            # than jumping to the last top-level definition in the file.
            _anchor_line: Optional[int] = None
            if op.metadata:
                _anchor_line = op.metadata.get("fixspec_anchor_line")
            if _anchor_line is not None and facts.abs_path:
                _nearest = _find_def_near_line(facts.abs_path, _anchor_line)
                if _nearest:
                    logger.info(
                        "[RULE_D] INSERT_AFTER_SYMBOL anchor '%s' not in graph — "
                        "redirecting to definition at line-based anchor '%s' "
                        "(via fixspec_anchor_line=%s)",
                        op.symbol, _nearest, _anchor_line,
                    )
                    decision.action = "rewrite"
                    decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
                    decision.corrected_symbol = _nearest
                    decision.corrected_anchor = _nearest
                    decision.rationale = (
                        "insert_anchor_redirected_to_fixspec_anchor_line"
                    )
                    decision.confidence = 0.85
                    return decision

            # ── Import binding guard ─────────────────────────────────────
            # Import bindings (e.g. 'WebSocket' from 'import WebSocket from "ws"')
            # are valid AST anchors that ts_vm_bridge / operation_executor
            # know how to handle natively.  The dependency graph does not
            # track them (it only records definitions), so symbol_exists=False.
            # Redirecting to the last top-level definition (e.g. shutdown())
            # would place the insertion at EOF, breaking import placement.
            # Keep the original anchor and let the language VM handle it.
            if facts.is_import_binding:
                logger.info(
                    "[RULE_D] INSERT_AFTER_SYMBOL anchor '%s' is an import binding "
                    "— keeping (language VM will handle)",
                    op.symbol,
                )
                decision.action = "keep"
                decision.corrected_kind = op.kind.value
                decision.rationale = "insert_anchor_is_import_binding"
                return decision

            # ── File-level fallback (DPB principle) ───────────────────────
            # When no line-based recovery is available, find the last top-level
            # function/class as anchor.  This handles LLM specifying a non-symbol
            # (variable assignment, logger declaration, etc.) as anchor.
            _last_anchor = find_last_top_level_def(facts.abs_path or op.path)
            if _last_anchor:
                logger.info(
                    "[RULE_D] INSERT_AFTER_SYMBOL anchor '%s' not in graph — "
                    "redirecting to last top-level definition '%s' (DPB principle)",
                    op.symbol, _last_anchor,
                )
                decision.action = "rewrite"
                decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
                decision.corrected_symbol = _last_anchor
                decision.corrected_anchor = _last_anchor
                decision.rationale = "insert_anchor_redirected_to_last_toplevel_def"
                decision.confidence = 0.75
                return decision
            # No top-level definitions found (empty/unparseable file) — keep
            decision.action = "keep"
            decision.corrected_kind = op.kind.value
            decision.rationale = "insert_anchor_missing_file_level_attempt"
            decision.confidence = 0.60
            return decision

        # ── Module-level insert fallback check ──
        # If this is a module-level insert (bare symbol, no parent class context),
        # check if the parent directory exists.  If so, file-level rewriting
        # (create file then insert at top) is feasible during replan even though
        # the file does not exist yet.
        _is_module_level = (
            op.kind == OperationKind.INSERT_AFTER_SYMBOL
            and op.symbol
            and "." not in op.symbol
        )
        _parent_dir = os.path.dirname(facts.abs_path) if facts.abs_path else ""
        _dir_exists = bool(_parent_dir) and os.path.isdir(_parent_dir)

        decision.action = "skip"
        decision.corrected_kind = op.kind.value
        decision.rationale = "insert_anchor_missing_no_fallback"
        decision.confidence = 0.90
        decision.should_replan = True
        if not facts.file_exists:
            # File doesn't exist — the replan should either create the file first
            # or redirect to an existing file with a valid anchor.
            if _is_module_level and _dir_exists:
                decision.replan_hint = (
                    f"INSERT_AFTER_SYMBOL anchor '{op.symbol}' not found — "
                    f"file {op.path} does not exist (parent directory exists). "
                    f"This is a module-level insert; consider prepending a "
                    f"CREATE_FILE op for {op.path} before the INSERT, or "
                    f"redirecting the INSERT to an existing file with a valid anchor."
                )
            else:
                decision.replan_hint = (
                    f"INSERT_AFTER_SYMBOL anchor '{op.symbol}' not found in "
                    f"{op.path} and file does not exist. "
                    f"Check if the anchor symbol name or file path is correct."
                )
        else:
            decision.replan_hint = (
                f"INSERT_AFTER_SYMBOL anchor '{op.symbol}' not found in "
                f"existing file {op.path}. "
                f"Check if the anchor symbol name is correct, or use a known "
                f"top-level definition as anchor."
            )
        return decision

    # ── Rule E: anchor_edit + missing required fields → skip ─────────
    if op.kind == OperationKind.ANCHOR_EDIT:
        if not op.anchor_pattern:
            decision.action = "skip"
            decision.corrected_kind = op.kind.value
            decision.rationale = "anchor_edit_missing_pattern"
            decision.confidence = 1.0
            return decision

        if not facts.file_exists:
            decision.action = "skip"
            decision.corrected_kind = op.kind.value
            decision.rationale = "anchor_edit_file_not_found"
            decision.confidence = 1.0
            return decision

    # ── Rule F: anchor_edit + anchor not found in file ───────────────
    if op.kind == OperationKind.ANCHOR_EDIT and not facts.anchor_found:
        # Try to rewrite to a symbol-based op if we have a symbol
        if op.symbol and facts.symbol_exists:
            decision.action = "rewrite"
            decision.corrected_kind = OperationKind.MODIFY_SYMBOL.value
            decision.corrected_symbol = op.symbol
            decision.rationale = "anchor_not_found_rewrite_to_modify_symbol"
            decision.confidence = 0.75
            return decision

        if op.symbol and facts.parent_symbol_exists:
            decision.action = "rewrite"
            decision.corrected_kind = OperationKind.INSERT_AFTER_SYMBOL.value
            decision.corrected_symbol = facts.parent_symbol
            decision.corrected_anchor = facts.parent_symbol
            decision.rationale = "anchor_not_found_rewrite_to_insert_after_parent"
            decision.confidence = 0.70
            return decision

        # No good fallback — skip with replan evidence
        decision.action = "skip"
        decision.corrected_kind = op.kind.value
        decision.rationale = "anchor_not_found_no_fallback"
        decision.confidence = 0.85
        decision.should_replan = True
        decision.replan_hint = (
            f"ANCHOR_EDIT pattern '{op.anchor_pattern}' not found in {op.path}. "
            f"The anchor string may have changed. "
            f"Use a different anchor or switch to symbol-based editing."
        )
        return decision

    # ── Rule G: DELETE_SYMBOL_RANGE + symbol missing → skip (already satisfied) ─
    # A prior write op on the same file already deleted this symbol's range.
    # Executing with stale metadata line numbers would either ERROR (start_line
    # > file length) or silently delete wrong content. Skip cleanly instead.
    if op.kind == OperationKind.DELETE_SYMBOL_RANGE and not facts.symbol_exists:
        # ── G1: symbol="" → check metadata line-range fallback ────────
        if not op.symbol:
            # If metadata has explicit start_line/end_line, let the handler
            # resolve the range from metadata lines (line-range fallback).
            # The handler checks metadata.start_line/end_line BEFORE requiring
            # a symbol name — skipping here would block that fallback.
            _meta = op.metadata or {}
            try:
                _has_line_range = (
                    int(_meta.get("start_line", 0)) > 0
                    and int(_meta.get("end_line", 0)) > 0
                )
            except (TypeError, ValueError):
                _has_line_range = False
            if _has_line_range:
                decision.action = "keep"
                decision.corrected_kind = op.kind.value
                decision.rationale = "delete_range_resolved_via_metadata_lines"
                decision.confidence = 0.85
                decision.should_replan = False
                return decision

                # ── G1b: removed (2026-06-02) ──────────────────────────
                # _intent_fallback was removed from planner_plan_create.py.
                # DELETE_SYMBOL_RANGE with empty symbol and no line_range
                # is now skipped at plan parsing time (before auto_correction runs).

            decision.action = "skip"
            decision.corrected_kind = op.kind.value
            decision.rationale = "delete_range_missing_symbol"
            decision.confidence = 0.80
            decision.should_replan = True
            decision.replan_hint = (
                f"DELETE_SYMBOL_RANGE op for '{op.path}' has symbol='' (empty). "
                f"Re-emit with a valid symbol name or provide a line range "
                f"in op.metadata.start_line/end_line."
            )
            return decision

        # ── G2: symbol set but not in file → already satisfied ─
        decision.action = "skip"
        decision.corrected_kind = op.kind.value
        decision.rationale = "delete_range_symbol_already_removed"
        decision.confidence = 1.0
        decision.should_replan = False
        return decision

    # ── Rule H: DELETE_SYMBOL_RANGE + class_attribute → guard ─────────
    # Class-level attributes are API-contract definitions.  Cross-file
    # instance-attribute access (result.metadata) is undetectable via
    # static analysis.  Even when the scanner flags one as "dead,"
    # deleting it is high-risk without explicit confirmation.
    if op.kind == OperationKind.DELETE_SYMBOL_RANGE and facts.symbol_kind == "class_attribute":
        decision.action = "skip"
        decision.corrected_kind = op.kind.value
        decision.rationale = "class_attribute_delete_requires_confirmation"
        decision.confidence = 0.85
        decision.should_replan = False
        return decision

    # ── Default: keep as-is ──────────────────────────────────────────
    decision.action = "keep"
    decision.corrected_kind = op.kind.value
    decision.rationale = "no_correction_needed"
    return decision


def _fix_type_alias_member_access_opcode(op: Operation, facts: ResolutionFacts) -> bool:
    """Fix ``TypeName.Member`` access in op.code_snippet for TS type aliases.

    TypeScript ``type`` aliases are erased at compile time — they do not exist
    at runtime.  When the planner generates ``PieceType.I`` (treating a type
    alias as an enum), this must be corrected to ``'I'`` (string literal).

    Mutates *op.code_snippet* in-place.  Returns True if any fix was applied.
    """
    if not op.code_snippet or not facts.type_alias_names:
        return False
    _lang = LanguageId.from_path(op.path or "")
    if _lang not in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
        return False

    import re
    _original = op.code_snippet
    _fixed = _original
    for _name in sorted(facts.type_alias_names, key=len, reverse=True):
        # Match \bTypeName\.Member where Member starts with uppercase (enum-like)
        # Replace TypeName.Member → 'Member'
        _pattern = re.compile(r'\b' + re.escape(_name) + r'\.([A-Z][A-Za-z0-9]*)\b')
        _fixed = _pattern.sub(r"'\1'", _fixed)

    if _fixed != _original:
        op.code_snippet = _fixed
        logger.info(
            "[TYPE_ALIAS_FIX] Fixed member access in op %s: %s → %s",
            op.id, repr(_original), repr(_fixed),
        )
        return True
    return False


def apply_correction(
    op: Operation,
    decision: AutoCorrectionDecision,
) -> Optional[Operation]:
    """Create a corrected Operation based on the decision.

    Returns None if action is 'skip' or 'route_to_repair'.
    Returns the original op if action is 'keep'.
    Returns a new Operation if action is 'rewrite'.
    """
    if decision.action == "keep":
        return op

    if decision.action in ("skip", "route_to_repair"):
        return None

    if decision.action == "rewrite":
        try:
            new_kind = OperationKind(decision.corrected_kind)
        except ValueError:
            logger.warning("Unknown corrected_kind: %s, keeping original", decision.corrected_kind)
            return op

        new_symbol = decision.corrected_symbol or op.symbol or ""
        new_intent = op.intent or ""

        # Enrich intent for rewritten ops
        if decision.rationale.startswith("missing_magic_method"):
            child = (op.symbol or "").split(".")[-1]
            new_intent = (
                f"[AUTO-CORRECTED: modify_symbol→insert_after_symbol]\n"
                f"Add new method '{child}' to class '{new_symbol}'.\n"
                f"Original intent: {op.intent or '(none)'}"
            )
        elif decision.rationale == "missing_method_insert_after_parent":
            child = (op.symbol or "").split(".")[-1]
            new_intent = (
                f"[AUTO-CORRECTED: modify_symbol→insert_after_symbol]\n"
                f"Add new method '{child}' to class '{new_symbol}'.\n"
                f"Original intent: {op.intent or '(none)'}"
            )
        elif decision.rationale == "insert_anchor_missing_fallback_to_parent":
            new_intent = (
                f"[AUTO-CORRECTED: anchor fallback to parent '{new_symbol}']\n"
                f"{op.intent or ''}"
            )
        elif decision.rationale.startswith("anchor_not_found_rewrite"):
            new_intent = (
                f"[AUTO-CORRECTED: anchor_edit→{new_kind.value}]\n"
                f"Anchor pattern not found. Applying change via symbol-based operation.\n"
                f"Original intent: {op.intent or '(none)'}"
            )
        elif decision.rationale == "insert_anchor_redirected_to_last_toplevel_def":
            # Rule D: non-graph anchor (variable assignment) was redirected to last
            # top-level function definition. Update intent so the LLM does not read
            # the original anchor text and override the corrected symbol in its JSON.
            orig_anchor = op.symbol or "(unknown)"
            new_intent = (
                f"[AUTO-CORRECTED: INSERT anchor redirected to '{new_symbol}']\n"
                f"Original anchor '{orig_anchor}' is not a graph symbol. "
                f"Use '{new_symbol}' as the anchor (last top-level def in file).\n"
                f"Original intent: {op.intent or '(none)'}"
            )
        elif decision.rationale == "insert_after_symbol_is_import_stmt_converted_to_insert_import":
            # Rule D-pre: the symbol itself was an import statement — set intent to
            # the import statement so _handle_insert_import can parse it directly.
            new_intent = new_symbol
        # Inherit context_hints as a copy so mutations don't affect the original op.
        _new_hints = dict(op.context_hints or {})

        # For INSERT_AFTER_SYMBOL corrections: enrich context_hints with the
        # produced symbol name so normalize_op_semantic_fields can populate
        # produces/scope/anchor correctly.
        if new_kind == OperationKind.INSERT_AFTER_SYMBOL:
            if not _new_hints.get("new_symbol_name"):
                _orig_sym = op.symbol or ""
                if "." in _orig_sym:
                    # e.g. "MyClass.method" → method is the produced symbol
                    _child = _orig_sym.split(".")[-1]
                    _new_hints["new_symbol_name"] = _child
                    # parent class is the anchor (new_symbol)
                    if new_symbol and new_symbol != _child:
                        _new_hints.setdefault("parent_class", new_symbol)
                elif _orig_sym and _orig_sym != new_symbol:
                    # anchor was redirected (Rule D); original sym is the produced symbol
                    _new_hints["new_symbol_name"] = _orig_sym

        # Carry intent_assertions from original op, updating target_symbol when
        # the symbol was corrected (e.g. B.3a qualified→bare).  Without this the
        # assertions are silently dropped and assertion_verdict is never evaluated.
        _old_sym = op.symbol or ""
        _corrected_assertions = []

        # ── Remove stale assertions when kind changes ──
        # When switching from modify_symbol to insert_after_symbol, existing assertions
        # (GUARD_IN_SCOPE, BEHAVIORAL_CONTRACT, SYMBOL_HAS_PARAM, etc.) are not valid
        # for the new kind. (CONTRACT_NORM handles this identically at the planner
        # stage — planner_agent.py L5335)
        _old_kind = op.kind
        _kind_changed_to_insert = (
            _old_kind is not None
            and _old_kind == OperationKind.MODIFY_SYMBOL
            and new_kind is not None
            and new_kind == OperationKind.INSERT_AFTER_SYMBOL
        )
        if _kind_changed_to_insert:
            logger.info(
                "[AUTO_CORRECT] clearing %d stale intent_assertions for %s: "
                "MODIFY_SYMBOL → INSERT_AFTER_SYMBOL",
                len(list(op.intent_assertions or [])), op.id,
            )
        else:
            for _ia in (op.intent_assertions or []):
                if _old_sym and new_symbol and _ia.target_symbol == _old_sym:
                    import dataclasses as _dc
                    _ia = _dc.replace(_ia, target_symbol=new_symbol)
                _corrected_assertions.append(_ia)

        corrected = Operation(
            id=op.id,
            kind=new_kind,
            path=op.path,
            symbol=new_symbol,
            intent=new_intent,
            depends_on=op.depends_on,
            acceptance=op.acceptance,
            context_hints=_new_hints,
            metadata=op.metadata or {},
            # Preserve fields not re-derived by normalize_op_semantic_fields
            intent_assertions=_corrected_assertions,
            atomic=op.atomic,
            edit_contract=op.edit_contract,
            code_snippet=op.code_snippet,
            anchor_pattern=op.anchor_pattern,
            upstream_ops=list(op.upstream_ops or []),
            downstream_ops=list(op.downstream_ops or []),
            cross_ref=dict(op.cross_ref or {}),
        )
        normalize_op_semantic_fields(corrected)
        return corrected

    return op


# ── Helpers ──────────────────────────────────────────────────────────────

def _has_validation_intent(
    op: Operation,
    plan_metadata: dict[str, Any],
    intent: str = "",
) -> bool:
    """Check if the op or plan has validation-related intent."""
    full_text = (op.intent or "") + " " + intent
    if _VALIDATION_KEYWORDS.match(full_text):
        return True

    cs = plan_metadata.get("change_spec")
    if isinstance(cs, dict):
        for change in cs.get("changes", []):
            kind = change.get("kind", "")
            if kind in ("add_validation", "add_field"):
                return True
            desc = change.get("description", "")
            if _VALIDATION_KEYWORDS.match(desc):
                return True

    return False


# ── Diff Purity Gate ──────────────────────────────────────────────────────────

_IMPORT_START = ('import ', 'from ')
_COMMENT_START = '#'


def is_diff_churn_only(before: str, after: str) -> bool:
    """Return True if the diff between `before` and `after` is purely cosmetic.

    A "churn-only" diff contains ONLY changes to:
      - blank lines (added or removed)
      - comment lines (lines starting with #)
      - import statements (import X / from X import Y)

    If at least one changed line is a semantic code line (a non-blank, non-comment,
    non-import line that was added or removed), the diff has real content → return False.

    Used as a diff purity gate: patches that only change imports/comments/blanks
    without touching actual code are almost certainly wrong (LLM hallucinated churn).
    """
    if before == after:
        return False  # No diff at all → not a churn issue (it's a no-op)

    # Use sequence-based unified diff so that duplicate lines (common in large files)
    # are correctly attributed to their position.  The former set-difference approach
    import difflib as _difflib
    diff_lines = _difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        lineterm="",
    )

    semantic_count = 0
    for diff_line in diff_lines:
        # Skip unified diff header lines and context lines
        if diff_line.startswith(("---", "+++", "@@")):
            continue
        if not diff_line.startswith(("+", "-")):
            continue  # context line (unchanged)

        content = diff_line[1:]  # strip leading +/-
        stripped = content.strip()
        # Blank line
        if not stripped:
            continue
        # Comment line
        if stripped.startswith(_COMMENT_START):
            continue
        # Import line
        if stripped.startswith(_IMPORT_START):
            continue
        # Everything else is semantic code
        semantic_count += 1
        break  # one semantic line is enough

    return semantic_count == 0


def is_diff_dead_import_cleanup(before: str, after: str) -> bool:
    """Return True iff the diff exclusively removes imports unused in `after`.

    Used as a *negative-protection* signal for two intent-blind gates:
      • DIFF_REGRESSION_PURE_DELETION (deletions-only diff)
      • Diff purity gate (import/comment/blank-only diff classified as churn)

    Both gates correctly flag suspicious churn in general, but a legitimate
    "remove unused import" cleanup looks identical structurally. This helper
    provides the AST-grounded escape: if every removed import name is genuinely
    unreferenced in the new file content, the diff is the *correct* outcome,
    not a defective edit.

    Required:
      • before / after both parse as Python
      • at least one import bound name removed
      • no new import bound names added
      • each removed bound name not referenced anywhere in `after`
        (Name node id, or attribute-access root id)
    """
    if before == after:
        return False
    try:
        before_tree = ast.parse(before)
        after_tree = ast.parse(after)
    except SyntaxError:
        return False

    def _import_bound_names(tree: ast.AST) -> set:
        names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
        return names

    before_names = _import_bound_names(before_tree)
    after_names = _import_bound_names(after_tree)
    removed = before_names - after_names
    added = after_names - before_names
    if not removed or added:
        return False

    after_referenced: set = set()
    for node in ast.walk(after_tree):
        if isinstance(node, ast.Name):
            after_referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                after_referenced.add(base.id)

    _still_referenced = [n for n in removed if n in after_referenced]
    if not _still_referenced:
        return True  # all removed names genuinely unused → dead import cleanup

    # Some removed import name(s) are still referenced in the file.
    # This can still be legitimate if the removed import was a *local* import
    # (e.g. inside a function body) and a module-level import of the same name
    # exists — removing a redundant local import is a real improvement, not churn.
    _module_level_imports: set = set()
    for _stmt in after_tree.body:
        if isinstance(_stmt, ast.Import):
            for _alias in _stmt.names:
                _module_level_imports.add(_alias.asname or _alias.name.split(".")[0])
        elif isinstance(_stmt, ast.ImportFrom):
            for _alias in _stmt.names:
                _module_level_imports.add(_alias.asname or _alias.name)

    if all(n in _module_level_imports for n in _still_referenced):
        return True  # redundant local import removal, module-level import covers it

    return False


# ── Diff Locality Gate ────────────────────────────────────────────────────────

def get_symbol_line_ranges(source: str, symbols: list[str]) -> dict[str, tuple]:
    """Return {symbol_name: (start_line, end_line)} for each symbol found in source.

    Supports qualified names like ``UIManager.__init__`` which will look for
    ``__init__`` inside the ``UIManager`` class definition.  Bare names like
    ``draw`` still match the first occurrence at file scope (depth-first).

    Line numbers are 1-based (matching ast node lineno).
    Returns an empty dict on parse failure.
    """
    ranges: dict[str, tuple] = {}
    if not source or not symbols:
        return ranges

    # Separate qualified names (ClassName.method) from bare names
    qualified_pairs: list = []  # [(class_name, method_name, orig_name), ...]
    bare_names: set = set()
    for s in symbols:
        if "." in s:
            parts = s.rsplit(".", 1)
            if len(parts) == 2:
                qualified_pairs.append((parts[0], parts[1], s))
                continue
        bare_names.add(s)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ranges

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Qualified-name lookup: find methods inside this class only
            for cls_name, method_name, orig_name in qualified_pairs:
                if node.name == cls_name and orig_name not in ranges:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                end_line = getattr(child, "end_lineno", child.lineno)
                                ranges[orig_name] = (child.lineno, end_line)
                                break
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in bare_names and node.name not in ranges:
                end_line = getattr(node, "end_lineno", node.lineno)
                ranges[node.name] = (node.lineno, end_line)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in bare_names and tgt.id not in ranges:
                    end_line = getattr(node, "end_lineno", node.lineno)
                    ranges[tgt.id] = (node.lineno, end_line)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in bare_names and node.target.id not in ranges:
                end_line = getattr(node, "end_lineno", node.lineno)
                ranges[node.target.id] = (node.lineno, end_line)
    return ranges


def is_diff_locality_ok(
    before: str,
    after: str,
    target_symbols: list[str],
) -> bool:
    """Return True if the diff has semantic changes near/within the target symbols.

    Return False (locality failure) when:
      - Target symbols are found in `before`
      - AND semantic changed lines (non-import, non-blank, non-comment) exist
      - AND NONE of the semantic changed lines falls within any target symbol's
        line range in `before`

    This catches "wrong-region" patches: the LLM modified unrelated code (e.g.,
    DateField, TimeField) while the target was FilePathField.

    Returns True (pass) when:
      - No target symbols found in before (new symbol / file) → can't check
      - No semantic changes outside the symbol range (e.g., import-only additions
        alongside the actual symbol fix) → already caught by churn gate
      - At least one semantic changed line is inside a target symbol's range
    """
    if not target_symbols or before == after:
        return True

    # Find line ranges of target symbols in the before file
    sym_ranges = get_symbol_line_ranges(before, target_symbols)
    if not sym_ranges:
        # Symbols not found in before (new code) — can't enforce locality
        return True

    # Use difflib unified_diff for accurate line-level diff detection
    # (set-based approach loses duplicate lines — e.g. two `pass` lines where
    # only one is removed would be invisible to set difference).
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    import difflib as _difflib
    import re as _re
    _diff = _difflib.unified_diff(
        before_lines, after_lines, lineterm="", n=0
    )
    removed_lines_with_idx: list = []
    added_lines_with_idx: list = []
    _before_lineno = 0
    _after_lineno = 0
    for _dl in _diff:
        if _dl.startswith("@@"):
            _m = _re.search(r'-(\d+)', _dl)
            if _m:
                _before_lineno = int(_m.group(1))
            _m = _re.search(r'\+(\d+)', _dl)
            if _m:
                _after_lineno = int(_m.group(1))
        elif _dl.startswith("-") and not _dl.startswith("---"):
            removed_lines_with_idx.append((_before_lineno, _dl[1:]))
            _before_lineno += 1
        elif _dl.startswith("+") and not _dl.startswith("+++"):
            added_lines_with_idx.append((_after_lineno, _dl[1:]))
            _after_lineno += 1

    all_changed = removed_lines_with_idx + added_lines_with_idx

    if not all_changed:
        return True

    # Check if any semantic changed line is within a symbol's range
    def _is_semantic(line_content: str) -> bool:
        stripped = line_content.strip()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return False
        if stripped.startswith(_IMPORT_START):
            return False
        return True

    # Collect ranges as a flat set of covered line numbers.
    # No upward buffer: extending above start would bleed into preceding class/function.
    # Upward buffer (-3): covers blank lines / comments immediately before the def
    # that some rewriters include in their replacement range.  SymbolSearcher
    # reports the `def` line as start, but AST-based rewriters often capture
    # 2–3 pre-function lines — without the upward buffer these fall outside
    # covered_lines and trigger false-positive scope_violations.
    # Downward buffer (+5): covers trailing blank lines / closing brackets.
    # Decorators: ast node.lineno already includes decorator in Python ≥3.8.
    covered_lines: set = set()
    for _sym, (start, end) in sym_ranges.items():
        covered_lines.update(range(max(1, start - 3), end + 6))

    has_semantic_in_range = False
    has_semantic_out_of_range = False

    for lineno, content in removed_lines_with_idx:
        if _is_semantic(content):
            if lineno in covered_lines:
                has_semantic_in_range = True
            else:
                has_semantic_out_of_range = True

    # For added lines we check position in after — approximate mapping
    for lineno, content in added_lines_with_idx:
        if _is_semantic(content):
            if lineno in covered_lines:
                has_semantic_in_range = True
            else:
                has_semantic_out_of_range = True

    # ── Comment-bridge expansion ───────────────────────────────────────
    # When semantic changes fall outside every symbol range but the first
    if has_semantic_out_of_range and not has_semantic_in_range:
        if _comment_bridge_connects_semantic_block(
            after_lines, sym_ranges, added_lines_with_idx,
        ):
            has_semantic_in_range = True
            has_semantic_out_of_range = False

    # Failure: semantic changes exist but NONE are near/within the target symbol
    if has_semantic_out_of_range and not has_semantic_in_range:
        return False

    return True


def prune_to_locality(
    before: str,
    after: str,
    target_symbols: list[str],
) -> str:
    """Remove changes outside target symbol ranges from the diff.

    Uses difflib.SequenceMatcher to identify structured diff hunks,
    then selectively applies only hunks that overlap with target symbols'
    line ranges (plus buffer).  Hunks outside all target symbols are
    reverted to the ``before`` state.

    Returns the pruned content if out-of-range changes were removed,
    or ``after`` unchanged if no out-of-range changes exist (or if
    symbol ranges cannot be determined).

    This is the auto-pruning counterpart of ``is_diff_locality_ok``:
    instead of rejecting the patch, it surgically removes the
    problematic changes.
    """
    if not target_symbols or before == after:
        return after

    # Find symbol ranges in before content
    sym_ranges = get_symbol_line_ranges(before, target_symbols)
    if not sym_ranges:
        return after  # Can't determine ranges — keep as-is

    # Build covered line set (same buffer as is_diff_locality_ok: -3/+5)
    covered_lines: set = set()
    for _sym, (start, end) in sym_ranges.items():
        covered_lines.update(range(max(1, start - 3), end + 6))

    before_lines = before.splitlines()
    after_lines = after.splitlines()

    import difflib
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)

    result_lines: list = []
    n_pruned = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result_lines.extend(after_lines[j1:j2])
        elif tag == 'replace':
            # Lines i1:i2 in before are replaced by j1:j2 in after
            # Check if the replacement falls within coverage
            in_coverage = any(
                lineno in covered_lines
                for lineno in range(i1 + 1, i2 + 1)  # 1-based
            )
            if in_coverage:
                result_lines.extend(after_lines[j1:j2])
            else:
                # Revert replacement: keep before lines
                result_lines.extend(before_lines[i1:i2])
                n_pruned += 1
        elif tag == 'delete':
            # Lines i1:i2 removed from before
            in_coverage = any(
                lineno in covered_lines
                for lineno in range(i1 + 1, i2 + 1)
            )
            if not in_coverage:
                # Revert deletion: keep the lines
                result_lines.extend(before_lines[i1:i2])
                n_pruned += 1
            # If in coverage, deletion was intentional — don't add
        elif tag == 'insert':
            # Lines j1:j2 added in after (no corresponding before)
            # Check if nearby line in before (around insertion point)
            # falls within coverage
            in_coverage = any(
                lineno in covered_lines
                for lineno in range(max(1, i1), i1 + 2)
            )
            if not in_coverage:
                n_pruned += 1
                # Skip the insertion entirely
            else:
                result_lines.extend(after_lines[j1:j2])

    pruned = '\n'.join(result_lines)

    if n_pruned > 0:
        logger.info(
            "[PRUNE] prune_to_locality: pruned %d hunk(s) outside target symbol(s) %s",
            n_pruned, target_symbols,
        )
        return pruned

    return after


def _comment_bridge_connects_semantic_block(
    after_lines: list[str],
    sym_ranges: dict[str, tuple],
    added_lines_with_idx: list[tuple[int, str]],
) -> bool:
    """Return True when every out-of-range semantic addition forms a single
    contiguous block connected to the *closest* symbol end by a comment/blank
    bridge.

    Safeguards against false negatives:
      (1) imports are treated as semantic — they break the bridge.
      (2) the closest symbol is selected; a far-away symbol cannot validate
          an insertion near a different function.
      (3) the entire semantic block must be contiguous — no existing
          non-comment code may sit between scattered semantic additions.
      (4) GAP_THRESHOLD=3: ≥3 consecutive blank/comment lines separate
          semantic clusters; only the first cluster is bridge-validated.
      (5) top-level symbol: the first semantic line after the bridge must
          not be indented — indented code after a top-level function end
          signals a misplaced body line.
    """
    MAX_BRIDGE = config.counts.MAX_BRIDGE_LINES
    GAP_THRESHOLD = 3  # consecutive blank/comment lines → separate cluster

    # Gather out-of-range semantic additions.
    # Imports ARE semantic here: `# comment / import os / actual_code` is a
    # broken bridge — the import signals a genuinely different region.
    _semantic_out: list[tuple[int, str]] = []
    _semantic_out_set: set = set()
    for ln, content in added_lines_with_idx:
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Check against every symbol's covered range (end + 3 buffer).
        _in_any = any(
            ln in range(start, end + 4)
            for start, end in sym_ranges.values()
        )
        if not _in_any:
            _semantic_out.append((ln, content))
            _semantic_out_set.add(ln)

    if not _semantic_out:
        return False

    _semantic_out.sort(key=lambda x: x[0])
    first_sem_ln = _semantic_out[0][0]
    last_sem_ln = _semantic_out[-1][0]

    # ── Select the CLOSEST symbol (shortest bridge to first semantic line).
    #    A far-away symbol must not validate an insertion near a different
    #    function just because bridge_len happens to be small.
    best_bridge_start = None
    best_bridge_end = None
    best_bridge_len = float("inf")
    best_sym_start = None

    for _sym, (start, end) in sym_ranges.items():
        if first_sem_ln <= end:
            continue

        _bs = end + 1
        _be = first_sem_ln - 1
        _blen = _be - _bs + 1
        if _blen < best_bridge_len:
            best_bridge_len = _blen
            best_bridge_start = _bs
            best_bridge_end = _be
            best_sym_start = start

    if best_bridge_start is None or best_bridge_len > MAX_BRIDGE:
        return False

    _added_set: set = {ln for ln, _ in added_lines_with_idx}

    # ── Bridge check: symbol end → first semantic line ──
    #    Every line in the gap must be blank or a comment.
    #    Added lines (comments or in-range code) are part of the LLM's block
    #    and are skipped — they don't break the bridge.
    for ln in range(best_bridge_start, best_bridge_end + 1):
        if ln > len(after_lines):
            return False
        if ln in _added_set:
            # Added line within the bridge — only fails if it's semantic
            # (non-comment, non-blank) code, like a misplaced import.
            stripped = after_lines[ln - 1].strip()
            if stripped and not stripped.startswith("#"):
                return False
            continue
        stripped = after_lines[ln - 1].strip()
        if stripped and not stripped.startswith("#"):
            return False

    # ── Contiguity check: first → last semantic line ──
    #    Every line between the first and last out-of-range semantic addition
    _consecutive_gap = 0
    _found_gap_break = False
    for ln in range(first_sem_ln + 1, last_sem_ln):
        if ln > len(after_lines):
            break
        if ln in _semantic_out_set:
            if _found_gap_break:
                return False
            _consecutive_gap = 0
            continue
        if ln in _added_set:
            # Non-semantic added line (comment/blank) — fine.
            _consecutive_gap = 0
            continue
        # Line exists in BEFORE (not added).  Must be blank or comment.
        stripped = after_lines[ln - 1].strip()
        if stripped and not stripped.startswith("#"):
            return False
        _consecutive_gap += 1
        if _consecutive_gap > GAP_THRESHOLD:
            _found_gap_break = True

    # If a gap break was detected but last_sem_ln was the loop boundary (only
    # two semantic additions), the inner loop never had a chance to reject.
    # Reject now: the second cluster is not bridge-validated.
    if _found_gap_break:
        return False

    # ── Indentation guard: top-level symbol → first semantic line ──
    #    After a top-level (col 0) function/class ends, the next semantic
    #    line should also be at column 0.  Indented code signals a body line
    #    placed outside the function scope.
    if best_sym_start is not None and best_sym_start <= len(after_lines):
        _sym_line = after_lines[best_sym_start - 1]
        _sym_indent = len(_sym_line) - len(_sym_line.lstrip())
        if _sym_indent == 0:
            _first_content = _semantic_out[0][1]
            _first_indent = len(_first_content) - len(_first_content.lstrip())
            if _first_indent > 0:
                return False

    return True


# ── Non-Python import detection ──────────────────────────────────────────────

# Extensions that can't appear in a valid Python import path
_NON_PYTHON_EXTENSIONS = frozenset({
    '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.yaml', '.yml',
    '.xml', '.md', '.txt', '.png', '.jpg', '.gif', '.svg',
})


def has_suspicious_new_imports(before: str, after: str) -> bool:
    """Return True if the patch added any obviously hallucinated import lines.

    Hallucinated imports share a pattern: they reference file-system paths
    (containing .js, .html, etc.) that cannot be valid Python module names.
    """
    try:
        before_imports: set[str] = set()
        after_imports: set[str] = set()
        for line in before.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                before_imports.add(stripped)
        for line in after.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                after_imports.add(stripped)

        new_imports = after_imports - before_imports
        for imp in new_imports:
            _imp_lower = imp.lower()
        if any(ext in _imp_lower for ext in _NON_PYTHON_EXTENSIONS):
                return True
        return False
    except Exception:
        return False  # non-critical — never block execution


# ── AST-level structure integrity check ──────────────────────────────────────


def is_structure_preserved(before: str, after: str) -> bool:
    """Return False if any top-level class or function definition was removed.

    Detects bugs like the LLM accidentally deleting `class Foo(Bar):` while
    editing a method inside it.
    """
    def _top_level_names(source: str) -> set[str]:
        try:
            tree = ast.parse(source)
            return {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(getattr(node, 'col_offset', 1), int)
                and node.col_offset == 0  # top-level only
            }
        except SyntaxError:
            return set()

    before_names = _top_level_names(before)
    after_names = _top_level_names(after)
    if not before_names:
        return True  # can't determine; don't block
    removed = before_names - after_names
    return len(removed) == 0


# ── Boot-time API surface validation ───────────────────────────────────────
# Names listed here are imported lazily (from inside function bodies) by
# external modules.  A rename or removal will surface as a silent degradation
# (caught by the caller's except Exception), so we verify them at load time.
_LAZY_IMPORT_NAMES = (
    "resolve_operation_facts",
    "auto_correct_operation",
    "apply_correction",
    "validate_operation_preflight",
    "AutoCorrectionDecision",
    "_IMPORT_START",
    "_COMMENT_START",
    "is_diff_churn_only",
    "is_diff_dead_import_cleanup",
    "is_diff_locality_ok",
    "prune_to_locality",
    "has_suspicious_new_imports",
    "is_structure_preserved",
)
for _name in _LAZY_IMPORT_NAMES:
    if _name not in globals():
        raise NameError(
            f"[AUTO_CORRECTION_API] lazy-import name '{_name}' is not defined. "
            f"Update _LAZY_IMPORT_NAMES or re-add the name to this module."
        )
del _name, _LAZY_IMPORT_NAMES
