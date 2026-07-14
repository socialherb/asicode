"""
Planner Pipeline Mixin for AgentLoop.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits PlannerPipelineMixin, so all methods have full access to
self.config, self.registry, etc.
"""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Optional

from ..languages import LanguageId
from .execution_spec import ResolvedExecutionSpec

logger = logging.getLogger(__name__)

# Additive edit_kinds that inherently create new symbols.
# When edit_kind is one of these, the graph enricher is expected to have
# low confidence because the target symbols don't exist yet.  The guard
# should not block even if no INSERT_AFTER_SYMBOL/CREATE_FILE ops remain
# after candidate selection (those ops may have been transformed into
# MODIFY_SYMBOL by the candidate ranker).
_ADDITIVE_EDIT_KINDS: frozenset = frozenset({
    "add_field",
    "add_validation",
    "add_call",
    "guard_add",
    "add_logging",
})


# ---------------------------------------------------------------------------
def _validate_target_symbols_against_files(
    target_files: list[str],
    target_symbols: list[str],
    repo_root: str,
) -> list[str]:
    """Validate target_symbols exist in target_files.  Corrects hallucinated symbols
    by finding the nearest matching symbol in the same file via similarity scoring.

    Returns corrected target_symbols list.  Does NOT validate symbols for new files
    (where the symbol is being created, not modified).
    """
    if not target_files or not target_symbols:
        return target_symbols

    from external_llm.agent.auto_correction import compute_symbol_similarity as _sim

    # Build file -> all-definition-names map (lazy, only for files we actually need)
    _file_defs: dict[str, list[str]] = {}

    def _get_defs(fpath: str) -> list[str]:
        if fpath in _file_defs:
            return _file_defs[fpath]
        abs_path = os.path.join(repo_root, fpath) if repo_root else fpath
        if not os.path.isfile(abs_path):
            _file_defs[fpath] = []
            return []
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            _file_defs[fpath] = []
            return []
        # ── Unified path: LanguageId-based tree-sitter symbol extraction ──
        try:
            from external_llm.languages.tree_sitter_utils import find_all_symbols
            _lang = LanguageId.from_path(fpath).value
            _symbols = find_all_symbols(content, _lang)
            if _symbols is None:
                _symbols = []
            # Collect all unique definition names (top-level + nested)
            _names = list({s[0] for s in _symbols})
            # Also include qualified names (e.g. "Game.render") for exact
            # match against Planner's qualified symbol references, avoiding
            # fuzzy-match fallback that risks low-similarity misses.
            _container_kinds = {"class", "interface", "enum"}
            _containers = [(s[0], s[2], s[3]) for s in _symbols
                           if s[1] in _container_kinds]
            _qual_names: list[str] = []
            for s_name, s_kind, s_start, s_end in _symbols:
                if s_kind in _container_kinds:
                    continue
                for c_name, c_start, c_end in _containers:
                    if c_start < s_start and c_end > s_end:
                        _qual_names.append(f"{c_name}.{s_name}")
                        break
            if _qual_names:
                _names.extend(qn for qn in _qual_names if qn not in _names)
            _file_defs[fpath] = _names
            return _names
        except Exception:
            _file_defs[fpath] = []
            return []

    corrected: list[str] = []
    for sym in target_symbols:
        if not sym:
            corrected.append(sym)
            continue

        # Check across all target files
        found = False
        for fpath in target_files:
            defs = _get_defs(fpath)
            if sym in defs:
                corrected.append(sym)
                found = True
                break

        if found:
            continue

        # Not found — try nearest match across all target files
        best_name: Optional[str] = None
        best_score = 0.0
        best_fpath: Optional[str] = None
        for fpath in target_files:
            defs = _get_defs(fpath)
            for def_name in defs:
                score = _sim(sym, def_name)
                if score > best_score:
                    best_score = score
                    best_name = def_name
                    best_fpath = fpath

        if best_name and best_score >= 0.55 and best_fpath:
            logger.warning(
                "[SPEC_VALIDATION] Symbol %r not found — corrected to nearest %r (sim=%.2f) in %s",
                sym, best_name, best_score, best_fpath,
            )
            corrected.append(best_name)
        else:
            if best_name:
                logger.warning(
                    "[SPEC_VALIDATION] Symbol %r not found in %s — "
                    "passing through (nearest %r sim=%.2f below threshold 0.55)",
                    sym, target_files, best_name, best_score,
                )
            else:
                logger.info(
                    "[SPEC_VALIDATION] Symbol %r not found in %s — "
                    "passing through (no definitions found in target files)",
                    sym, target_files,
                )
            corrected.append(sym)

    return corrected


# ---------------------------------------------------------------------------
# Structural NL detection — replaces curated _NL_KEYWORDS frozenset.
#
# A valid project-specific code symbol almost always has at least one
# structural marker (underscore, digit, ALL_CAPS, >=2 uppercase letters).
# Tokens lacking ALL of these markers are likely natural-language noise
# that leaked from LLM intent resolution (e.g. "three", "new", "Handler").
#
# Key design choice: this only affects *warning-level* assertions.
# False positives are harmless — real symbols are found by the
# collect_defined_names check and never reach this filter.
# ---------------------------------------------------------------------------


def _is_nl_keyword(name: str) -> bool:
    """Return True if name looks like natural language, not a project-specific code symbol.

    Uses structural heuristics — zero curated keywords.  A valid code symbol
    almost always has at least one structural marker (underscore, digit,
    ALL_CAPS, multi-segment CamelCase).  Tokens lacking ALL markers are
    likely NL noise from LLM intent resolution.
    """
    if not name:
        return False
    if "_" in name:
        return False   # snake_case / __dunder__
    if any(c.isdigit() for c in name):
        return False   # has digit
    if name.isupper():
        return False   # ALL_CAPS constant
    upper_count = sum(1 for c in name if c.isupper())
    if upper_count >= 2:
        return False   # multi-segment CamelCase/PascalCase
    # TitleCase (single uppercase) threshold: <=8 chars catches "Handler"(7), "Service"(7)
    if upper_count == 1 and name[0].isupper():
        return len(name) <= 8
    # Pure lowercase: <=5 chars catches "three"(5), "new"(3); passes "render"(6)
    return len(name) <= 5


# ---------------------------------------------------------------------------
# Module-level helper (moved from agent_loop.py for co-location with its
# single caller inside _create_operation_plan_with_scanner).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory: implementation_spec → ResolvedExecutionSpec
# Single source of truth replacing 3 duplicate sites in the switch_to_planner
# pipeline (agent_stream.py, asi.py, agent_loop.py).
# ---------------------------------------------------------------------------


def build_prebuilt_spec_from_impl_spec(
    request_text: str,
    implementation_spec: dict,
    source: str = "design_chat_analysis",
    repo_root: str = "",
) -> Optional["ResolvedExecutionSpec"]:
    """Convert an implementation_spec dict into a ResolvedExecutionSpec.

    Normalises target_files / new_files (str | dict → str), preserves
    target_symbols, and infers request_type from presence of target_files.
    If repo_root is provided, it is used to resolve relative file paths.
    Returns None when the spec contains no actionable targets.
    """
    if not isinstance(implementation_spec, dict):
        return None

    def _normalize(items):
        result = []
        for item in (items or []):
            if isinstance(item, str):
                _path = item
            elif isinstance(item, dict):
                _path = item.get("file") or item.get("path") or ""
            else:
                continue
            if _path:
                if repo_root:
                    _path = os.path.normpath(os.path.join(repo_root, _path))
                else:
                    _path = os.path.normpath(_path)
                result.append(_path)
        return result

    _target_files = _normalize(implementation_spec.get("target_files"))
    _new_files = _normalize(implementation_spec.get("new_files"))
    _target_symbols = [
        s for s in (implementation_spec.get("target_symbols") or [])
        if isinstance(s, str)
    ]

    # Validate target_symbols against target_files to catch hallucinated symbols
    # before Planner ever sees them.  New files skip validation (symbols may not
    # exist yet).
    if _target_files and _target_symbols:
        _target_symbols = _validate_target_symbols_against_files(
            _target_files, _target_symbols, repo_root,
        )

    if not _target_files and not _new_files and not _target_symbols:
        return None

    _request_type = "modify" if (_target_files or _new_files) else "read"

    _edit_kind = (implementation_spec.get("edit_kind") or "").strip().lower()

    # Structural fallback: dotted target_symbol (e.g. "Game.lockPiece") → add_field.
    # Pure name-based inference — no semantic keyword matching, so zero false positives.
    if not _edit_kind and _target_symbols:
        for _sym in _target_symbols:
            if "." in _sym:
                _edit_kind = "add_field"
                logger.info(
                    "[EDIT_KIND] inferred add_field from dotted target_symbol: %s", _sym
                )
                break

    _code_context = []
    _raw_cc = implementation_spec.get("code_context") or []
    for _cc_item in _raw_cc:
        if isinstance(_cc_item, dict) and _cc_item.get("file") and _cc_item.get("snippet"):
            _code_context.append({
                "reason": _cc_item.get("reason", ""),
                "file": _cc_item["file"],
                "snippet": _cc_item["snippet"],
            })

    # ── Analysis notes (from Design Chat — proposed new files / structural reasoning) ──
    _analysis_notes = []
    _raw_an = implementation_spec.get("analysis_notes") or []
    for _an_item in _raw_an:
        if isinstance(_an_item, dict) and _an_item.get("file") and _an_item.get("note"):
            _analysis_notes.append(_an_item)
            logger.info(
                "[ANALYSIS_NOTE] extracted analysis_note for '%s': reason=%s, note_preview=%s",
                _an_item.get("file", "?"),
                _an_item.get("reason", "N/A"),
                _an_item["note"][:200],
            )
    logger.info("[ANALYSIS_NOTE] total %d analysis_note(s) from implementation_spec", len(_analysis_notes))

    # ── Multi-region uncovered detection ────────────────────────────────────
    # When analysis_notes outnumber target_symbols, the spec describes more
    # distinct code regions than symbols can cover.  DPB only sees target_symbols
    # and would miss the uncovered regions — route to LLM Planner instead.
    # (Heuristic: #analysis_notes > #target_symbols.  No content parsing.)
    _multi_region_uncovered = bool(
        _target_symbols and _analysis_notes
        and len(_analysis_notes) > len(_target_symbols)
    )
    if _multi_region_uncovered:
        logger.info(
            "[MULTI_REGION_UNCOVERED] %d analysis_notes > %d target_symbols"
            " → DPB would miss uncovered edits, routing to LLM Planner",
            len(_analysis_notes), len(_target_symbols),
        )

    # Warn when both target_symbols AND code_context are empty
    # -> planner operates blind (re-grounding risk, drift false positives)
    if not _target_symbols and not _code_context:
        logger.warning(
            "[BLIND_SPEC] Both target_symbols and code_context are empty. "
            "Planner will operate blind - re-grounding risk is high. "
            "request_text=%s target_files=%s",
            request_text[:80], _target_files,
        )

    from .execution_spec import ResolvedExecutionSpec

    return ResolvedExecutionSpec(
        original_request=request_text,
        intent=implementation_spec.get("purpose", request_text),
        request_type=_request_type,
        target_files=_target_files,
        target_symbols=_target_symbols,
        new_files=_new_files,
        reference_files=_normalize(implementation_spec.get("reference_files")),
        reference_symbols=_normalize(implementation_spec.get("reference_symbols")),
        code_context=_code_context,
        analysis_notes=_analysis_notes,
        metadata={
            "source": source,
            "skip_grounding": True,
            "implementation_spec": implementation_spec,
            "edit_kind": _edit_kind,
            "multi_region_uncovered": _multi_region_uncovered,
        },
        authoritative=True,
    )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

from ..client import LLMServerUnavailableError
from .agent_loop_types import (
    AgentCancelled,
    AgentResult,
    _PlannerLaneOutcome,
    _SpecResolutionResult,
)
from .operation_models import (
    ExecutionPhase,
    OperationKind,
)
from .operation_models import (
    GroundingSummary as _GroundingSummary,
)
from .operation_models import (
    SemanticFitAction as _SFA,
)


def _checkpoint_plan_files(
    mode: str | None, operations, repo_root: str | None = None
) -> tuple[bool, list | None]:
    """Decide the pre-write checkpoint action for ASICODE_CHECKPOINT_ON_WRITE.

    Returns ``(enabled, files)`` where ``files is None`` means a full-repo
    snapshot and a list means a scoped snapshot of exactly those paths.

    Mode semantics (case-insensitive; unset/empty defaults to "scoped"):
      "0"/"off"/"false"/"no"  → disabled
      "full"                  → full-repo snapshot
      anything else           → scoped snapshot of the plan's target files
                                (op.path); disabled when the plan has no
                                resolvable paths — the expensive full walk is
                                never paid implicitly.

    ``repo_root`` scopes the existence check: a *scoped* snapshot only retains
    target paths that resolve to an existing regular file under ``repo_root``
    (mirrors ``CheckpointStore._scan_listed_files``). A create-only plan has
    target paths but zero resolvable ones — snapshotting it would yield an
    empty checkpoint whose ``restore()`` is a silent no-op reported as success,
    so the checkpoint is skipped entirely (no Undo button). When ``repo_root``
    is None, paths are resolved against the process CWD as a defensive
    fallback (production always passes the registry's repo_root).
    """
    mode = (mode or "scoped").strip().lower() or "scoped"
    if mode in ("0", "off", "false", "no") or not operations:
        return False, None
    if mode == "full":
        return True, None
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    candidates = sorted({op.path for op in operations if getattr(op, "path", None)})
    files: list = []
    for p in candidates:
        cand = Path(p)
        if not cand.is_absolute():
            cand = root / p
        try:
            cand = cand.resolve()
            cand.relative_to(root)  # reject "../" escapes outside the repo
        except (ValueError, OSError):
            continue
        if cand.is_file():
            files.append(p)
    return (True, files) if files else (False, None)


class PlannerPipelineMixin:
    """
    Mixin providing plan-creation and execution methods for AgentLoop.

    Requires the host class to expose:
        - self.config           (AgentConfig)
        - self.registry         (ToolRegistry)
        - self._operation_executor, self._planner_agent
        - self._cb(), self._build_planner_summary(), self._build_planner_fallback_context()
        - self._rollback_patches(), self._execute_plan_sequence_direct()
        - self._handle_analyze_first_escalation()  (EscalationPipelineMixin)
        - self.performance_collector, self._routing_intent_hint
    """

    # ------------------------------------------------------------------
    # _run_planner_lane
    # ------------------------------------------------------------------

    def _run_planner_lane(
        self,
        request: str,
        context: str,
        route: Any,
        git_state: Any,
        session_id: str,
        turns: Optional[list] = None,
    ) -> "_PlannerLaneOutcome":
        """PLANNER lane: spec resolution → plan → execution → AgentResult.

        Extracted from run() to reduce the 3500-line monolith.
        Returns _PlannerLaneOutcome with result=non-None to return directly,
        or fallback_context=non-None to update context and continue to LLM loop.
        """
        try:
            # ── Cancel check before any LLM call in planner lane ──
            if self.config.cancel_event and self.config.cancel_event.is_set():
                raise AgentCancelled("cancelled by user before planner lane init")

            self._init_hybrid_components()

            # ── Apply token offset from prior phases (Design Chat, main agent) ──
            _tok_off_p = getattr(self.config, '_token_offset_prompt_tokens', 0) or 0
            _tok_off_c = getattr(self.config, '_token_offset_completion_tokens', 0) or 0
            _tok_off_r = getattr(self.config, '_token_offset_cache_read_tokens', 0) or 0
            if _tok_off_p or _tok_off_c:
                self._planner_agent.apply_token_offset(_tok_off_p, _tok_off_c, _tok_off_r)
                # Clear so re-entry won't double-count
                self.config._token_offset_prompt_tokens = 0
                self.config._token_offset_completion_tokens = 0
                self.config._token_offset_cache_read_tokens = 0
                logger.info(
                    "[TOKEN_CONTINUITY] applied prior-phase offset: "
                    "prompt=%d completion=%d cache_read=%d",
                    _tok_off_p, _tok_off_c, _tok_off_r,
                )

            if self._operation_executor and self._planner_agent:
                # ── Pipeline stage: spec resolution + semantic-fit judge ──
                _prebuilt = self.config.prebuilt_spec_for_planner
                if _prebuilt is not None:
                    # Analysis-backed path: design chat already analyzed the code.
                    # Skip heavy SpecResolver — use the pre-built spec directly.
                    # Make an immutable copy of list/dict fields so early bail-out
                    # mutations in _build_and_execute_plan don't affect the original.
                    import copy as _copy_mod
                    from dataclasses import replace as _dc_replace
                    # Use deepcopy for metadata to avoid sharing nested structures
                    # (implementation_spec, etc.) with the original spec.
                    # Shallow copy (dict(...)) would preserve references to nested
                    # dicts/lists, causing silent corruption of the original config.
                    _prebuilt = _dc_replace(
                        _prebuilt,
                        target_files=list(getattr(_prebuilt, "target_files", [])),
                        new_files=list(getattr(_prebuilt, "new_files", [])),
                        target_symbols=list(getattr(_prebuilt, "target_symbols", [])),
                        metadata=_copy_mod.deepcopy(getattr(_prebuilt, "metadata", {})),
                    )
                    # ANALYSIS_BACKED: spec comes from design-chat analysis, so
                    # grounding_confidence defaults HIGH (0.80) instead of 0.0.
                    # The symbol validation below may adjust this downward.
                    _analysis_gc = 0.80
                    _spec_result = _SpecResolutionResult(
                        spec=_prebuilt,
                        fit_verdict=None,
                        grounding_summary=_GroundingSummary(grounding_confidence=_analysis_gc),
                        is_read_only_intent=getattr(_prebuilt, "request_type", "") == "read",
                    )
                    # Write to spec.metadata so downstream readers
                    # (GroundingSummary.from_spec_meta, etc.) get the correct value.
                    _prebuilt.metadata["grounding_confidence"] = _analysis_gc
                    _prebuilt_files = getattr(_prebuilt, "target_files", [])
                    _prebuilt_impl = (getattr(_prebuilt, "metadata", {}) or {}).get("implementation_spec") or {}
                    logger.info(
                        "[ANALYSIS_BACKED] spec active: purpose=%r target_files=%s skip_grounding=True",
                        _prebuilt_impl.get("purpose", getattr(_prebuilt, "intent", ""))[:120],
                        _prebuilt_files,
                    )

                    # ── Prebuilt spec target_files validation ─────────────────────────
                    # Files referenced by Design Chat must actually exist on disk.
                    _prebuilt_repo_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
                    _prebuilt_valid_files = []
                    _prebuilt_missing_files = []
                    for _tf in _prebuilt_files:
                        _tf_abs = _tf if os.path.isabs(_tf) else os.path.join(_prebuilt_repo_root, _tf)
                        if os.path.isfile(_tf_abs):
                            _prebuilt_valid_files.append(_tf)
                        else:
                            _prebuilt_missing_files.append(_tf)
                    if _prebuilt_missing_files:
                        # Move missing target_files to new_files instead of discarding.
                        # Design Chat may put files-to-create in target_files instead of
                        # new_files (e.g., when the analysis spec lacks a new_files field).
                        # Removing them silently causes EMPTY_SPEC -> 200-file injection
                        # (344K token waste).  Moving them to new_files preserves the
                        # creation intent and avoids grounding failure.
                        _prebuilt_new_files = list(getattr(_prebuilt, "new_files", []) or [])
                        for _mf in _prebuilt_missing_files:
                            if _mf not in _prebuilt_new_files:
                                _prebuilt_new_files.append(_mf)
                        _prebuilt.target_files = _prebuilt_valid_files
                        _prebuilt.new_files = _prebuilt_new_files
                        logger.warning(
                            "[ANALYSIS_BACKED] %d target_file(s) not found on disk: %s -- "
                            "moved to new_files",
                            len(_prebuilt_missing_files), _prebuilt_missing_files,
                        )
                    # ── Prebuilt spec new_files validation ─────────────────────────────
                    # Fix 2: If any new_files already exist on disk, move them to
                    # target_files. Prevents create_file op from overwriting existing files.
                    _pnf_validated = list(getattr(_prebuilt, "new_files", []) or [])
                    _pnf_moved_to_target = []
                    for _pnf in list(_pnf_validated):
                        _pnf_abs = _pnf if os.path.isabs(_pnf) else os.path.join(_prebuilt_repo_root, _pnf)
                        if os.path.isfile(_pnf_abs):
                            if _pnf not in _prebuilt.target_files:
                                _prebuilt.target_files.append(_pnf)
                            _pnf_validated.remove(_pnf)
                            _pnf_moved_to_target.append(_pnf)
                    if _pnf_moved_to_target:
                        _prebuilt.new_files = _pnf_validated
                        logger.info(
                            "[ANALYSIS_BACKED] %d new_file(s) already exist on disk -- "
                            "moved to target_files: %s",
                            len(_pnf_moved_to_target), _pnf_moved_to_target,
                        )
                    if not _prebuilt.target_files and not getattr(_prebuilt, "new_files", []):
                        # Use _prebuilt.target_files (updated by new_files validation) instead of
                        # _prebuilt_valid_files (stale after new_files → target_files promotion).
                        logger.warning(
                            "[ANALYSIS_BACKED] no valid target_files or new_files -- "
                            "grounding failure; plan will likely produce no effective changes",
                        )
                        _analysis_gc = min(_analysis_gc, 0.30)
                        _spec_result.grounding_summary.grounding_confidence = _analysis_gc
                        _prebuilt.metadata["grounding_confidence"] = _analysis_gc

                    # ── Prebuilt spec symbol validation ─────────────────────────────
                    # Design-chat may have hallucinated symbol names. Verify
                    # target_symbols exist in target_files before using them.
                    _prebuilt_syms = getattr(_prebuilt, "target_symbols", []) or []
                    if _prebuilt_syms:
                        _ens_repo = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
                        _ens_target_files = getattr(_prebuilt, "target_files", []) or []
                        _ens_valid_syms = []
                        _ens_invalid_syms = []
                        for _sym in _prebuilt_syms:
                            _found_in_any = False
                            for _tf in _ens_target_files:
                                _abs = _tf if os.path.isabs(_tf) else os.path.join(_ens_repo, _tf)
                                if os.path.isfile(_abs):
                                    try:
                                        from external_llm.languages.tree_sitter_utils import find_all_symbols
                                        _lang = LanguageId.from_path(_abs).value
                                        if _lang != "unknown":
                                            with open(_abs, encoding="utf-8", errors="replace") as _fh:
                                                _src = _fh.read()
                                            _symbols = find_all_symbols(_src, _lang)
                                            _bare = _sym.split(".")[-1] if "." in _sym else _sym
                                            if any(_sym_name in (_bare, _sym)
                                                   for _sym_name, _, _, _ in _symbols):
                                                _found_in_any = True
                                                break
                                    except Exception:
                                        pass
                            if _found_in_any:
                                _ens_valid_syms.append(_sym)
                            else:
                                # Symbol not found in existing files.  Distinguish:
                                #   - NEW symbol to add: at least one target_file exists
                                #     (symbol is a new function for that existing file)
                                #   - Truly hallucinated: NO target_file exists (symbol
                                #     references a file that doesn't exist)
                                _parent_file_exists = any(
                                    os.path.isfile(
                                        _tf if os.path.isabs(_tf) else os.path.join(_ens_repo, _tf)
                                    )
                                    for _tf in _ens_target_files
                                )
                                if _parent_file_exists:
                                    # For dotted symbols (Class.method or module.func),
                                    # verify that the parent class/module actually exists
                                    # in the target file. Design Chat can hallucinate
                                    # method names even when the parent file exists
                                    # (e.g., InstructionHandler._validate_inserted_code
                                    # when InstructionHandler class doesn't exist).
                                    # Non-dotted symbols are kept as-is (new top-level
                                    # functions/variables to add).
                                    _is_dotted = "." in _sym
                                    _parent_verified = not _is_dotted  # non-dotted: no parent check needed
                                    if _is_dotted:
                                        _parent_part = _sym.split(".")[0]
                                        # Short-circuit: if parent_part matches a target file stem
                                        # (e.g., "helper.process_data" where helper is the module name),
                                        # treat it as a valid module.func reference.
                                        _target_stems = {
                                            os.path.splitext(os.path.basename(_tf))[0]
                                            for _tf in _ens_target_files
                                        }
                                        if _parent_part in _target_stems:
                                            _parent_verified = True
                                        else:
                                            for _tf in _ens_target_files:
                                                _abs_pf = _tf if os.path.isabs(_tf) else os.path.join(_ens_repo, _tf)
                                                if os.path.isfile(_abs_pf):
                                                    try:
                                                        with open(_abs_pf, encoding="utf-8", errors="replace") as _fh:
                                                            _src_check = _fh.read()
                                                        from external_llm.languages.tree_sitter_utils import (
                                                            find_all_symbols as _find_syms,
                                                        )
                                                        _parent_lang = LanguageId.from_path(_abs_pf).value
                                                        if _parent_lang != "unknown" and _parent_part in {s[0] for s in _find_syms(_src_check, _parent_lang)}:
                                                            _parent_verified = True
                                                            break
                                                    except Exception:
                                                        pass
                                    if _parent_verified:
                                        # New symbol to be added to an existing file.
                                        # File exists → trust Design Chat's intent.
                                        # NL keyword filtering (removed) caused false positives
                                        # for legitimate new symbols (Class name patterns etc).
                                        _ens_valid_syms.append(_sym)
                                    else:
                                        # Dotted symbol's parent class/module not found in any
                                        # target file → likely hallucination (wrong class or module).
                                        _ens_invalid_syms.append(_sym)

                        if _ens_invalid_syms:
                            logger.warning(
                                "[ANALYSIS_BACKED] %d target_symbol(s) not found in target_files: %s — "
                                "will be removed from spec (preserved in metadata for reference)",
                                len(_ens_invalid_syms), _ens_invalid_syms,
                            )
                            _prebuilt.target_symbols = _ens_valid_syms
                            _prebuilt.metadata["_design_chat_removed_symbols"] = _ens_invalid_syms
                            # Downgrade confidence proportionally to hallucination ratio.
                            # Both truly hallucinated (file not found) and dotted-symbol
                            # hallucination (parent class/module not found in file) count.
                            _hallucinated_ratio = len(_ens_invalid_syms) / len(_prebuilt_syms)
                            _analysis_gc = max(0.30, 0.80 * (1.0 - _hallucinated_ratio))
                            _spec_result.grounding_summary.grounding_confidence = _analysis_gc
                            _prebuilt.metadata["grounding_confidence"] = _analysis_gc
                            logger.info(
                                "[ANALYSIS_BACKED] adjusted grounding_confidence=%.2f "
                                "(%d/%d symbols validated, %d hallucinated)",
                                _analysis_gc, len(_ens_valid_syms), len(_prebuilt_syms),
                                len(_ens_invalid_syms),
                            )

                    # ── Semantic-fit judge for ANALYSIS_BACKED spec ──
                    # Design Chat may produce incomplete or hallucinated specs.
                    # Run the semantic-fit judge to validate before planning.
                    try:
                        # ── Cancel check before semantic-fit judge ──
                        if self.config.cancel_event and self.config.cancel_event.is_set():
                            raise AgentCancelled("cancelled by user during planner judge")
                        _af_verdict = self._planner_agent.judge_semantic_fit(
                            request=request,
                            spec=_prebuilt,
                            exploration_facts=None,
                            grounding_summary=_GroundingSummary(
                                grounding_confidence=_analysis_gc,
                            ),
                            emit_verdict=True,
                            target_symbol_sources=None,
                        )
                        # RE_EXPLORE: Design Chat analysis is incomplete or misleading.
                        # Direct fallback to Design Chat tool loop for refinement,
                        # bypassing BUILD phase entirely.
                        if _af_verdict.action == _SFA.RE_EXPLORE:
                            logger.info(
                                "[ANALYSIS_BACKED_SEMANTIC_FIT] RE_EXPLORE → "
                                "Design Chat fallback (reason=%s)",
                                _af_verdict.reason[:200],
                            )
                            return _PlannerLaneOutcome(
                                result=None,
                                fallback_context=(
                                    f"Semantic-fit judge determined re-exploration needed: "
                                    f"{_af_verdict.reason[:300]}. "
                                    "Routing back to Design Chat for analysis refinement."
                                ),
                            )
                        _spec_result.fit_verdict = _af_verdict
                        logger.info(
                            "[ANALYSIS_BACKED_SEMANTIC_FIT] fit=%s action=%s conf=%.2f",
                            _af_verdict.semantic_fit, _af_verdict.action.value,
                            _af_verdict.confidence,
                        )
                    except Exception as _af_exc:
                        logger.warning(
                            "[ANALYSIS_BACKED_SEMANTIC_FIT] judge failed: %s", _af_exc,
                        )
                else:
                    # SpecResolver removed: PLANNER lane now requires a prebuilt
                    # spec produced by Design Chat's analysis/grounding. Without
                    # one, there is nothing to plan against — fall back to the
                    # Design Chat / MAIN_AGENT tool loop instead of resolving a
                    # spec from a raw request (the old SpecResolver path).
                    logger.info(
                        "[PLANNER_LANE] No prebuilt spec available — "
                        "routing to Design Chat / MAIN_AGENT tool loop"
                    )
                    return _PlannerLaneOutcome(
                        result=None,
                        fallback_context=(
                            "PLANNER lane requires a pre-resolved spec (from Design Chat "
                            "analysis). Routing to Design Chat / MAIN_AGENT for fallback."
                        ),
                    )
                _spec = _spec_result.spec
                _fit_verdict = _spec_result.fit_verdict
                _grounding_summary = _spec_result.grounding_summary
                _is_read_only_intent = _spec_result.is_read_only_intent

                # CLARIFY: emit question and stop instead of fabricating a plan.
                if _fit_verdict is not None and _fit_verdict.action == _SFA.CLARIFY:
                    logger.info("[SEMANTIC_FIT] CLARIFY → surfacing clarification question")
                    _clar_q = [{
                        "id": "semantic_fit_clarify",
                        "question": (
                            _fit_verdict.reason
                            or "Please narrow your request scope or specify which area (file/module/feature) you want to modify."
                        ),
                        "type": "free_text",
                        "options": [],
                        "reason": "semantic-fit judge: insufficient evidence",
                        "default": "",
                    }]
                    _checkpoint_cb = self.config.user_checkpoint_callback
                    _user_answer = ""
                    if self.config.user_checkpoint_enabled and _checkpoint_cb:
                        try:
                            _resp = _checkpoint_cb({
                                "question_id": _clar_q[0]["id"],
                                "question": _clar_q[0]["question"],
                                "type": "free_text",
                                "options": [],
                                "reason": _clar_q[0]["reason"],
                                "default": "",
                                "source": "semantic_fit",
                            })
                            _user_answer = (_resp or {}).get("answer", "") or ""
                        except Exception as _cpe:
                            logger.warning("CLARIFY checkpoint failed: %s", _cpe)
                    if _user_answer.strip():
                        context = (context or "") + f"\n\nUser clarification: {_user_answer}"
                        # On affirmative answer with a redirect target, retarget the
                        # spec to scanner-suggested symbols.
                        _redir_syms = getattr(_fit_verdict, "redirect_symbols", None)
                        from ._user_intent import UserApproval, classify_user_approval
                        _affirmative = classify_user_approval(_user_answer) == UserApproval.APPROVED
                        if _redir_syms and _affirmative:
                            _redir_files_actual = getattr(_fit_verdict, "redirect_files", None)
                            _spec.intent_symbols = list(_redir_syms)
                            if _redir_files_actual:
                                _spec.intent_files = list(_redir_files_actual)
                                _spec.target_files = list(_redir_files_actual)
                            _spec.metadata["redirected_from"] = list(
                                getattr(_spec, "target_symbols", []) or []
                            )
                            logger.info(
                                "[SEMANTIC_FIT] CLARIFY redirect accepted → "
                                "retargeted spec to %s",
                                _redir_syms,
                            )
                        _spec.metadata["planner_action"] = _SFA.ANALYZE_FIRST.value
                        logger.info(
                            "[SEMANTIC_FIT] CLARIFY answered → continuing as ANALYZE_FIRST"
                        )
                    else:
                        logger.info(
                            "[SEMANTIC_FIT] CLARIFY unanswered → terminal "
                            "clarification_needed (skipping plan generation)"
                        )
                        self.performance_collector.end_session()
                        _perf_summary = self.performance_collector.get_summary()
                        return _PlannerLaneOutcome(result=AgentResult(
                            status="clarification_needed",
                            turns=turns,
                            applied_patches=self.registry.applied_patches,
                            final_message=(
                                _fit_verdict.reason
                                or "The judge detected a structural premise mismatch between the request and the source. "
                                   "Please check your question and refine your request."
                            ),
                            metadata={
                                "session_id": session_id,
                                "git_state": git_state,
                                "clarification_questions": _clar_q,
                                "semantic_fit_verdict": _fit_verdict.to_dict(),
                                "performance": _perf_summary,
                            },
                        ))

                # ── Cancel check before plan execution ──
                if self.config.cancel_event and self.config.cancel_event.is_set():
                    raise AgentCancelled("cancelled by user before planner execution")
                # ── Pipeline stage: plan creation + execution ──
                _bep_outcome = self._build_and_execute_plan(
                    request=request,
                    context=context,
                    route=route,
                    git_state=git_state,
                    session_id=session_id,
                    turns=turns,
                    spec_result=_spec_result,
                )
                return _bep_outcome

            # ── Hybrid init failed: PLANNER components not available ──
            # _init_hybrid_components() failed silently (logged as warning).
            # _operation_executor and _planner_agent are both None.
            # Return fallback_context so Design Chat can take over.
            logger.warning(
                "[PLANNER_INIT_FAIL] Hybrid components not initialized — "
                "_init_hybrid_components() failed earlier; returning fallback_context"
            )
            return _PlannerLaneOutcome(
                result=None,
                fallback_context=(
                    "Hybrid PLANNER components failed to initialize. "
                    "Routing to Design Chat for fallback exploration."
                ),
            )
        except LLMServerUnavailableError as exc:
            logger.error("LLM server unavailable — aborting run: %s", exc)
            return _PlannerLaneOutcome(result=AgentResult(
                status="error",
                turns=[],
                final_message=f"Aborting due to LLM server unavailability: {exc}",
                applied_patches=self.registry.applied_patches,
                metadata={
                    "operation_executor": True,
                    "server_unavailable": True,
                    "planner_error": str(exc),
                    "session_id": session_id,
                    "git_state": git_state,
                },
            ))
        except RuntimeError as exc:
            # Intentional fallback signal from _build_and_execute_plan:
            #   - "targets not found" (spec had no existing target files)
            #   - "0 ops completed" (empty operation plan)
            # These are NOT errors — they signal "try MAIN_AGENT instead".
            # Return fallback_context instead of partial_success so the
            # drift guard can distinguish intentional fallback from bugs.
            logger.info(
                "[PLANNER_FALLBACK] Intentional RuntimeError — returning fallback_context: %s",
                str(exc)[:500],
            )
            # Use full exc for LLM context (not truncated); metadata dump was
            # already removed from the RuntimeError message (keys-only).
            return _PlannerLaneOutcome(
                result=None,
                fallback_context=(
                    f"PLANNER lane produced no operations: {exc}. "
                    "Routing to Design Chat for fallback exploration."
                ),
            )

        except AgentCancelled:
            raise  # Re-raise → _run_with_cancel (asi.py) catches → REPL creates paused state

        except Exception as exc:
            logger.warning(
                "Operation executor failed — returning partial result: %s\n%s",
                exc, traceback.format_exc(),
            )
            return _PlannerLaneOutcome(result=AgentResult(
                status="partial_success",
                turns=[],
                final_message=f"PLANNER lane failed: {exc}",
                applied_patches=self.registry.applied_patches,
                metadata={
                    "operation_executor": True,
                    "planner_fallback_blocked": True,
                    "planner_error": str(exc),
                    "session_id": session_id,
                    "git_state": git_state,
                },
            ))

    # ------------------------------------------------------------------
    # _build_and_execute_plan
    # ------------------------------------------------------------------

    def _build_and_execute_plan(
        self,
        request: str,
        context: str,
        route: Any,
        git_state: Any,
        session_id: str,
        turns: Optional[list],
        spec_result: "_SpecResolutionResult",
    ) -> "_PlannerLaneOutcome":
        """Plan creation, execution, and result assembly for the PLANNER lane.

        Extracted from _run_planner_lane(). Covers:
        - Early bail-out for non-existent target files
        - Plan creation + scanner injection + new symbol contracts
        - User checkpoint Q&A and pre-execution analysis gates
        - ANALYZE_FIRST escalation (re-resolve with wider scope)
        - Operation execution
        - Result collection → _PlannerLaneOutcome
        """
        turns = turns or []
        _spec = spec_result.spec
        _fit_verdict = spec_result.fit_verdict
        _grounding_summary = spec_result.grounding_summary
        _is_read_only_intent = spec_result.is_read_only_intent
        _llm_hints = dict(spec_result.llm_hints or {})
        _pending_guidance = spec_result.pending_guidance
        _prev_spec_fingerprint = spec_result.prev_spec_fingerprint
        _escalation_attempt = spec_result.escalation_attempt

        # Early bail-out: move non-existing target_files to new_files; skip PLANNER if empty.
        if _spec and _spec.target_files:
            _repo = self.registry.repo_root or ""
            _existing = []
            _missing = []
            for f in _spec.target_files:
                if os.path.exists(os.path.join(_repo, f)):
                    _existing.append(f)
                else:
                    _missing.append(f)
            if _missing:
                _is_analysis_backed = bool(
                    (getattr(_spec, 'metadata', None) or {}).get("skip_grounding") or False
                )
                if _is_analysis_backed:
                    # Issue 2: Design Chat analysis may hallucinate file paths.
                    # Log warning but do NOT promote to new_files automatically.
                    # Only keep files that actually exist.
                    logger.warning(
                        "[ANALYSIS_BACKED] %d target_file(s) not found (likely hallucination): %s — "
                    "NOT promoting to new_files. Only existing files kept.",
                        len(_missing), _missing,
                    )
                    # Do NOT promote to new_files — likely hallucination.
                    _spec.target_files = _existing
                else:
                    _spec.target_files = _existing
                    for mf in _missing:
                        if mf not in _spec.new_files:
                            _spec.new_files.append(mf)
                    logger.info(
                        "[PLANNER] moved non-existing target_files to new_files: %s",
                        _missing,
                    )
            _spec_has_targets = len(_existing) > 0 or len(getattr(_spec, 'new_files', []) or []) > 0
            if not _spec_has_targets:
                logger.info(
                    "Spec has no existing target files and no new_files (%s) — skipping PLANNER",
                    _spec.target_files,
                )
                self._cb("pipeline_stage", {
                    "stage": "spec_bailout",
                    "status": "skipped",
                    "detail": "No existing target files and no new_files → skipping PLANNER",
                })
                raise RuntimeError(
                    f"Spec targets not found: {_spec.target_files}. "
                    "Cannot use structured ops on non-existent files."
                )

        # Presupposed-missing gate: stop before any LLM call if core intent symbols
        # are confirmed absent from the target file. Emit symbol_not_found and return.
        if _spec is not None:
            _pm_all = (getattr(_spec, "metadata", None) or {}).get(
                "presupposed_missing_symbols", {}
            )
            _pm_ask = {
                sym: info for sym, info in _pm_all.items()
                if info.get("status") == "ask_clarification"
            }
            if _pm_ask:
                _spec_meta = getattr(_spec, "metadata", None) or {}
                _orig_syms = set(
                    _spec_meta.get("intent_search_terms", [])
                    or getattr(_spec, "target_symbols", None)
                    or []
                )
                # If search_term matches a target file's module name (stem), it's a file specifier — exclude
                _target_stems = {
                    os.path.splitext(os.path.basename(f))[0]
                    for f in (getattr(_spec, "target_files", None) or [])
                }
                _all_are_file_stems = (
                    len(_target_stems) > 0
                    and len(_pm_ask) > 0
                    and all(sym in _target_stems for sym in _pm_ask)
                )
                if _all_are_file_stems:
                    logger.info(
                        "[SYMBOL_NOT_FOUND] all %d presupposed-missing symbol(s) are "
                        "target file stems (%s) — treating as file-scope request",
                        len(_pm_ask), sorted(_pm_ask.keys()),
                    )
                    _pm_core = {}
                else:
                    _pm_core = {
                        sym: info for sym, info in _pm_ask.items()
                        if sym in _orig_syms and sym not in _target_stems
                    }
                if _pm_core:
                    _missing_details = [
                        {
                            "symbol": sym,
                            "target_file": info.get("target_file", ""),
                            "candidates": [
                                c.get("name") for c in (info.get("candidates") or [])[:3]
                                if c.get("name")
                            ],
                        }
                        for sym, info in _pm_core.items()
                    ]
                    logger.info(
                        "[SYMBOL_NOT_FOUND] aborting before planner — "
                        "core intent symbols absent from target file: %s",
                        list(_pm_core.keys()),
                    )
                    self._cb("symbol_not_found", {
                        "symbols": _missing_details,
                        "message": (
                            "The requested symbol does not exist in the file. "
                            "Please check the symbol name and try again."
                        ),
                    })
                    self.performance_collector.end_session()
                    return _PlannerLaneOutcome(result=AgentResult(
                        status="clarification_needed",
                        turns=[],
                        applied_patches=[],
                        final_message=(
                            "Operation aborted because the requested symbol does not exist in the file: "
                            + ", ".join(
                                f"'{sym}'"
                                + (f" (candidates: {info['candidates'][:2]})" if info.get("candidates") else "")
                                for sym, info in _pm_core.items()
                            )
                        ),
                        metadata={"symbol_not_found": _missing_details},
                    ))

        # ── N1: _low_confidence_warning flag from graph enricher ──
        # Soft warning instead of hard block — defer decision until after plan
        # creation so we can detect additive ops (INSERT_AFTER_SYMBOL, CREATE_FILE,
        # etc.) that legitimately reference non-existent symbols.
        if _spec is not None:
            _gc = (getattr(_spec, "metadata", None) or {}).get("graph_context", {})
            if _gc.get("_low_confidence_warning"):
                _unresolved = _gc.get("unresolved_symbols", [])
                logger.info(
                    "[GRAPH_LOW_CONFIDENCE] graph_confidence=0.00 with %d unresolved "
                    "symbol(s) — will re-check after plan creation",
                    len(_unresolved),
                )
                self._cb("pipeline_stage", {
                    "stage": "graph_confidence_warning",
                    "status": "warning",
                    "detail": "graph_confidence=0.00, deferring to post-plan check",
                })

        self._cb("pipeline_stage", {"stage": "decomposition", "status": "running", "detail": "Creating operation plan..."})
        op_plan, _plan_context = self._create_operation_plan_with_scanner(
            request=request, context=context, spec=_spec, grounding_summary=_grounding_summary
        )  # <-- N1a re-check inserted below

        # ── N1a: Re-check graph_confidence with ops knowledge ──
        # If the plan creates new symbols (INSERT_AFTER_SYMBOL, CREATE_FILE, etc.),
        # low graph_confidence is expected -- clear the block.
        if _spec is not None and op_plan is not None:
            _gc = (getattr(_spec, "metadata", None) or {}).get("graph_context", {})
            if _gc.get("_low_confidence_warning"):
                from .operation_models import _ADDITIVE_OP_KINDS
                # Extract edit_kind for additive edit_kind bypass
                _spec_edit_kind = (
                    (getattr(_spec, "metadata", None) or {})
                    .get("edit_kind", "")
                )
                if any(
                    op.kind in _ADDITIVE_OP_KINDS or bool(op.produces)
                    for op in (op_plan.operations or [])
                ):
                    _gc["_low_confidence_warning"] = False
                    _gc["_adjusted_for_additive_plan"] = True
                    logger.info(
                        "[GRAPH_CONFIDENCE_ADJUSTED] plan has %d additive ops -- "
                        "cleared _low_confidence_warning",
                        sum(1 for op in (op_plan.operations or [])
                            if op.kind in _ADDITIVE_OP_KINDS),
                    )
                    self._cb("pipeline_stage", {
                        "stage": "graph_confidence_adjusted",
                        "status": "adjusted",
                        "detail": "additive ops detected, confidence block lifted",
                    })
                elif _spec_edit_kind in _ADDITIVE_EDIT_KINDS:
                    _gc["_low_confidence_warning"] = False
                    _gc["_adjusted_for_additive_plan"] = True
                    logger.info(
                        "[GRAPH_CONFIDENCE_ADJUSTED] edit_kind=%s is additive "
                        "(%d unresolved symbols expected) — cleared "
                        "_low_confidence_warning",
                        _spec_edit_kind,
                        len(_gc.get("unresolved_symbols", [])),
                    )
                    self._cb("pipeline_stage", {
                        "stage": "graph_confidence_adjusted",
                        "status": "adjusted_by_edit_kind",
                        "detail": (
                            f"edit_kind={_spec_edit_kind} is additive, "
                            "confidence block lifted"
                        ),
                    })
                else:
                    _unresolved = _gc.get("unresolved_symbols", [])
                    _target_syms = getattr(_spec, "target_symbols", None) or []

                    # ── skip_grounding bypass ──────────────────────────────────
                    # Analysis-backed spec (from Design Chat / switch_to_planner)
                    # has already grounded target_files and target_symbols before
                    # reaching the planner.  Low graph confidence is expected
                    # because the graph is stale or the symbols are new and don't
                    # exist yet.  Skip the GRAPH_LOW_CONFIDENCE block entirely.
                    _spec_meta = getattr(_spec, "metadata", None) or {}
                    if _spec_meta.get("skip_grounding"):
                        logger.info(
                            "[GRAPH_LOW_CONFIDENCE] skip_grounding=True — "
                            "analysis-backed spec, bypassing confidence block "
                            "(unresolved=%d)",
                            len(_unresolved),
                        )
                        _gc["_low_confidence_warning"] = False
                        _gc["_adjusted_for_skip_grounding"] = True
                    else:
                        # ── Partial fallback: filter ops referencing unresolved symbols ──
                        # Instead of discarding the entire plan, keep ops that don't
                        # reference unresolved symbols (e.g. READ_SYMBOL on resolved
                        # symbols, CREATE_FILE for new files) and only route the
                        # unresolved-symbol ops to Design Chat fallback.
                        _resolved_names = {
                            r.get("name") for r in _gc.get("resolved_symbols", [])
                        }
                        _salvage_ops = [
                            op for op in (op_plan.operations or [])
                            if not (op.symbol and op.symbol in _unresolved
                                    and op.symbol not in _resolved_names)
                        ]
                        if len(_salvage_ops) < len(op_plan.operations or []):
                            _dropped = len(op_plan.operations or []) - len(_salvage_ops)
                            logger.info(
                                "[GRAPH_LOW_CONFIDENCE] partial fallback — "
                                "removed %d op(s) referencing unresolved symbols, "
                                "keeping %d ops, %d unresolved symbol(s): %s",
                                _dropped, len(_salvage_ops),
                                len(_unresolved), _unresolved,
                            )
                            if _salvage_ops:
                                # Clear warning — partial plan is viable
                                _gc["_low_confidence_warning"] = False
                                _gc["_partial_salvage_ops"] = True
                                _gc["_dropped_unresolved_ops"] = _dropped
                                # Replace plan with filtered ops
                                object.__setattr__(op_plan, "operations", _salvage_ops)
                                self._cb("pipeline_stage", {
                                    "stage": "graph_confidence_adjusted",
                                    "status": "partial_salvage",
                                    "detail": (
                                        f"removed {_dropped} unresolved ops, "
                                        f"keeping {len(_salvage_ops)} ops"
                                    ),
                                })
                            else:
                                logger.warning(
                                    "[GRAPH_LOW_CONFIDENCE] blocking -- no additive ops found, "
                                    "%d unresolved symbol(s) (%s) -- falling back to Design Chat tool loop",
                                    len(_unresolved), _target_syms,
                                )
                                # ── Graph confidence fallback ──
                                # Instead of returning a hard error (dead end), signal PLANNER
                                # fallthrough with context so the caller (agent_loop.py) runs
                                # the Design Chat tool loop.  Pass unresolved symbol details
                                # so the fallback can skip redundant find_symbol calls.
                                _fb_exec_result: dict = {
                                    "metadata": {
                                        "symbol_not_found": [
                                            {"symbol": s} for s in _unresolved
                                        ]
                                    }
                                }
                                return _PlannerLaneOutcome(
                                    result=None,
                                    fallback_context=self._build_planner_fallback_context(
                                        exc=(
                                            "GRAPH_LOW_CONFIDENCE: graph analysis could not resolve "
                                            f"{len(_unresolved)} symbol(s) in codebase: "
                                            f"{', '.join(str(s) for s in _unresolved)}"
                                        ),
                                        _spec=_spec,
                                        op_plan=op_plan,
                                        exec_result=_fb_exec_result,
                                    ),
                                )
                        else:
                            # All ops reference resolved symbols — still block
                            # (this is the original behavior for additive-op check failure)
                            logger.warning(
                                "[GRAPH_LOW_CONFIDENCE] blocking -- no additive ops found, "
                                "%d unresolved symbol(s) (%s) -- falling back to Design Chat tool loop",
                                len(_unresolved), _target_syms,
                            )
                            _fb_exec_result: dict = {
                                "metadata": {
                                    "symbol_not_found": [
                                        {"symbol": s} for s in _unresolved
                                    ]
                                }
                            }
                            return _PlannerLaneOutcome(
                                result=None,
                                fallback_context=self._build_planner_fallback_context(
                                    exc=(
                                        "GRAPH_LOW_CONFIDENCE: graph analysis could not resolve "
                                        f"{len(_unresolved)} symbol(s) in codebase: "
                                        f"{', '.join(str(s) for s in _unresolved)}"
                                    ),
                                    _spec=_spec,
                                    op_plan=op_plan,
                                    exec_result=_fb_exec_result,
                                ),
                            )

        # ── P1-alt: StageContext for typed pipeline stage boundary
        from .operation_models import StageContext
        stage_ctx = StageContext()

        # ── Task drift detection ───────────────────────────────────────────
        # Compare plan operations against spec to detect when the agent
        # drifts from the original request (modifying unrelated files,
        # targeting wrong symbols, using wrong operation kinds).
        _ref_files = list(getattr(_spec, "reference_files", None) or []) if _spec is not None else []
        # reference_files: exclude read-only source files from drift
        # detection so the system does not flag them as "missing targets".
        if _spec is not None:
            try:
                from .output_normalizer import detect_task_drift
                _drift_ref_syms = list(getattr(_spec, "reference_symbols", None) or [])
                _drift_report = detect_task_drift(
                    spec_target_files=list(getattr(_spec, "target_files", None) or []),
                    spec_target_symbols=list(getattr(_spec, "target_symbols", None) or []),
                    plan_operations=list(op_plan.operations or []),
                    request_type=getattr(_spec, "request_type", "") or "",
                    reference_symbols=_drift_ref_syms,
                    reference_files=_ref_files,
                )
                if _drift_report.has_drift:
                    logger.warning(
                        "[TASK_DRIFT] severity=%s: %s",
                        _drift_report.severity,
                        _drift_report.summary,
                    )
                    stage_ctx.task_drift = {
                        "severity": _drift_report.severity,
                        "summary": _drift_report.summary,
                        "untargeted_files": _drift_report.untargeted_files,
                        "drifted_kinds": _drift_report.drifted_kinds,
                    }
                    #── severity -based execution block ──
                    # TASK_DRIFT checks file-based drift (untargeted_files)
                    # and kind drift (drifted_kinds) are checked.
                    # Symbol-based drift (untargeted_symbols, missing_target_symbols) is handled by the
                    # spec_alignment quality gate in candidate ranking.
                    # To prevent whack-a-mole issues with symbol matching, TASK_DRIFT does not check
                    # symbols at all.
                    if _drift_report.severity in ("high", "medium"):
                        logger.warning(
                            "[TASK_DRIFT] severity=%s — blocking execution",
                            _drift_report.severity,
                        )
                        stage_ctx.task_drift_block = True
                else:
                    logger.debug("[TASK_DRIFT] no drift detected: %s", _drift_report.summary)
            except Exception as _de:
                logger.debug("[TASK_DRIFT] detection skipped: %s", _de)

        # ── Hard block check ─────────────────────────────────────────
        # If the candidate ranker marked all candidates as hard_blocked
        # (spec_alignment < 0.2 across all candidates), block execution.
        # This replaces TASK_DRIFT's string-pattern symbol matching which
        # was a constant source of false positives.
        if op_plan.metadata and op_plan.metadata.get("hard_blocked"):
            logger.warning(
                "[CANDIDATE_RANKING] hard_blocked=true â "
                "all candidates below spec_alignment threshold, blocking execution"
            )
            stage_ctx.task_drift_block = True
        # Self-planning is handled inside create_operation_plan — no re-run here.
        _sp_meta = (op_plan.metadata or {}).get("self_planning", {})
        if _sp_meta.get("iterations"):
            self._cb("pipeline_stage", {
                "stage": "self_planning", "status": "complete",
                "detail": f"{len(op_plan.operations)} ops after review (iter={_sp_meta['iterations']})",
            })
            logger.info("Plan review complete (inside planner): %d operations, iter=%d",
                        len(op_plan.operations), _sp_meta["iterations"])
        else:
            self._cb("pipeline_stage", {"stage": "self_planning", "status": "skipped", "detail": ""})

        # Pre-execution: detect analyze_then_modify plans with no edit ops.
        _pre_pp = getattr(op_plan, "plan_policy", None)
        _pre_policy_kind = _pre_pp.kind if _pre_pp is not None else ""
        _pre_requires_changes = _pre_pp.requires_code_changes if _pre_pp is not None else False
        _atm_no_edit = False
        if _pre_requires_changes and op_plan.operations:
            from external_llm.editor._editor_core.lane.operation_executor import _EDIT_OP_KINDS as _PRECHECK_EDIT_KINDS
            _pre_has_edit_op = any(
                op.kind in _PRECHECK_EDIT_KINDS for op in op_plan.operations
            )
            _pre_is_phase1 = (
                op_plan.execution_phase == ExecutionPhase.PHASE1_ANALYSIS
                and op_plan.requires_phase2
            )
            if not _pre_has_edit_op and not _pre_is_phase1:
                logger.info(
                    "[PLAN_VALIDITY] plan has no edit ops despite "
                    "requires_code_changes=True (policy=%r ops=%d)",
                    _pre_policy_kind, len(op_plan.operations),
                )
                _atm_no_edit = True

        # ── Checkpoint: snapshot repo state before write operations ──
        # Default ON, *scoped*: only the files the plan targets (op.path) are
        # snapshotted — a handful of small reads per write turn — so /agent/undo
        # works out of the box. ASICODE_CHECKPOINT_ON_WRITE controls the mode:
        #   unset / "1" / "scoped"      → scoped snapshot of plan target files
        #   "full"                      → full-repo snapshot (expensive; every
        #                                 source file is read + stored)
        #   "0" / "off" / "false" / "no" → disabled
        # A scoped snapshot with zero resolvable target paths is skipped (nothing
        # to roll back to; the full walk is never paid implicitly).
        _checkpoint_id = None
        _cp_root = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
        _cp_enabled, _cp_files = _checkpoint_plan_files(
            os.environ.get("ASICODE_CHECKPOINT_ON_WRITE"),
            op_plan.operations,
            repo_root=_cp_root,
        )
        if not _is_read_only_intent and _cp_enabled:
            try:
                from external_llm.agent.checkpoint_store import CheckpointStore
                _cp_store = CheckpointStore(_cp_root)
                _checkpoint_id = _cp_store.create(
                    f"Pre-execution snapshot for {session_id}", files=_cp_files
                )
                op_plan.metadata['checkpoint_id'] = _checkpoint_id
                logger.info(
                    "Created %s checkpoint %s before operation execution",
                    "full" if _cp_files is None else f"scoped({len(_cp_files)} files)",
                    _checkpoint_id,
                )
            except Exception as _cp_e:
                logger.warning("Failed to create checkpoint before execution: %s", _cp_e)
        _plan_seq = (op_plan.metadata or {}).get("plan_sequence") or []
        if len(_plan_seq) >= 2:
            self._cb("pipeline_stage", {
                "stage": "execution", "status": "running",
                "detail": (f"Executing {len(op_plan.operations)} operations"
                           f" across {len(_plan_seq)} files (sequential)..."),
            })
            exec_result = self._execute_plan_sequence_direct(
                _plan_seq, request, context,
                stage_ctx=stage_ctx,
            )
        else:
            self._cb("pipeline_stage", {"stage": "execution", "status": "running", "detail": f"Executing {len(op_plan.operations)} operations..."})
            exec_result = self._operation_executor.execute_plan(
                op_plan,
                original_request=request,
                context=context,
                max_operations=50,
                stage_ctx=stage_ctx,
            )
        logger.info("[TIMING] EXECUTOR_RETURN reached")
        _completed = exec_result.get("completed", 0)
        _failed = exec_result.get("failed", 0)
        _exec_status = "complete" if _failed == 0 else "error"
        _exec_detail = f"{_completed} completed" + (f", {_failed} failed" if _failed else "")
        self._cb("pipeline_stage", {"stage": "execution", "status": _exec_status, "detail": _exec_detail})
        logger.info(
            "Operation executor result: completed=%s, failed=%s, output_len=%s",
            _completed, _failed,
            len(exec_result.get("output", "")),
        )
        logger.info("[TIMING] BEFORE_SSE_EMIT")

        _esc = self._handle_analyze_first_escalation(
            request=request, context=context,
            op_plan=op_plan, exec_result=exec_result,
            spec=_spec, atm_no_edit=_atm_no_edit,
            is_read_only_intent=_is_read_only_intent,
            escalation_attempt=_escalation_attempt,
            llm_hints=_llm_hints,
            pending_guidance=_pending_guidance,
            prev_spec_fingerprint=_prev_spec_fingerprint,
            completed=_completed, failed=_failed,
            exec_status=_exec_status, exec_detail=_exec_detail,
        )
        if _esc.early_outcome is not None:
            return _esc.early_outcome
        exec_result = _esc.exec_result
        op_plan = _esc.op_plan
        _spec = _esc.spec
        _completed = _esc.completed
        _failed = _esc.failed
        _exec_status = _esc.exec_status
        _exec_detail = _esc.exec_detail

        self._cb("operation_execution_result", {
            "completed": _completed,
            "failed": _failed,
            "completed_ids": exec_result.get("completed_ids", []),
            "failed_ids": exec_result.get("failed_ids", []),
            "skipped_ids": exec_result.get("skipped_ids", []),
            "output_preview": (exec_result.get("output", "") or "")[:500],
            "failure_class": exec_result.get("final_failure_class", "") or "",
        })

        completed = exec_result.get("completed", 0)
        failed = exec_result.get("failed", 0)
        final_message = self._build_planner_summary(
            request, op_plan, exec_result, completed, failed,
        )

        if completed == 0 and failed == 0:
            logger.warning("Operation executor completed 0 ops — falling back to tool loop")
            raise RuntimeError(
                "Empty operation plan: 0 operations completed "
                f"(strategy={getattr(op_plan, 'execution_strategy', 'N/A')}, "
                f"target_files={getattr(op_plan, 'target_files', [])}, "
                f"target_symbols={getattr(op_plan, 'target_symbols', [])}, "
                f"metadata_keys={list(getattr(op_plan, 'metadata', {}).keys())})"
            )

        # PLANNER writes files directly; use modified_files as applied_patches proxy.
        _exec_modified = exec_result.get("modified_files", [])
        _planner_patches = (
            _exec_modified
            if _exec_modified
            else self.registry.applied_patches
        )

        # Use final_status (not op-count) — executor may set blocking verdicts on all-ok ops.
        _exec_final_status = exec_result.get("final_status", "")
        _is_blocking_verdict = _exec_final_status in {
            "failed", "verification_failed", "rollback", "execution_error",
            "unfulfilled", "invalidated",
        }
        _exec_status_str = (
            "already_satisfied"
            if _exec_final_status == "already_satisfied"
            else (
                "partial_success"
                if (failed > 0 or _is_blocking_verdict)
                else "success"
            )
        )
        # ── Include planner token totals in result metadata for session tracking ──
        _planner_tok = self._planner_agent.incremental_tokens() if self._planner_agent else {}

        return _PlannerLaneOutcome(result=AgentResult(
            status=_exec_status_str,
            turns=[],
            final_message=final_message,
            applied_patches=_planner_patches,
            metadata={
                "operation_executor": True,
                "completed_ops": completed,
                "failed_ops": failed,
                "noop_ops": exec_result.get("noop_ops", 0),
                "mode": op_plan.mode.value,
                "session_id": session_id,
                "git_state": git_state,
                "tokens": _planner_tok,
                "checkpoint_id": _checkpoint_id,
            },
        ))

    # ------------------------------------------------------------------
    # _create_operation_plan_with_scanner
    # ------------------------------------------------------------------

    def _create_operation_plan_with_scanner(
        self,
        request: str,
        context: str,
        spec: Any,
        grounding_summary: Any,
    ) -> tuple[Any, str]:
        """Create operation plan, inject RUN_SCANNER ops, and record symbol expectations.

        Returns (op_plan, plan_context) where plan_context is the GSG-enriched
        context used for planning — needed by the caller for Q&A replanning.
        """
        _plan_context = context
        _gsg_ctx = ""
        if spec is not None:
            _gsg_ctx = spec.gsg_context or spec.metadata.get("gsg_planning_context", "")
        if _gsg_ctx:
            _plan_context = _plan_context + "\n\n" + _gsg_ctx if _plan_context else _gsg_ctx
        # ── Cancel check before create_operation_plan (LLM call) ──
        if self.config.cancel_event and self.config.cancel_event.is_set():
            raise AgentCancelled("cancelled by user before planner plan creation")
        op_plan = self._planner_agent.create_operation_plan(
            request, _plan_context, spec=spec,
            routing_intent_hint=self._routing_intent_hint,
        )
        logger.info(
            "[PP_TRACE] after create_operation_plan: plan.plan_policy=%s",
            getattr(op_plan, "plan_policy", "__missing__"),
        )
        # Enforce ANALYZE_FIRST as a structural contract (no-op when not applicable).
        op_plan = self._planner_agent.enforce_analyze_first_structure(
            op_plan, spec
        )
        # Enforce evaluator verdicts: strip modify ops targeting symbol pairs that
        # the LLM pair evaluator confidently deemed NOT actionable (conf >= 0.90).
        op_plan = self._planner_agent.enforce_evaluator_verdicts(
            op_plan, spec
        )
        # Store evaluator verdicts in plan metadata so executor's FixSpec
        # materialization (Phase 2 of analyze_then_modify) can also enforce
        # them — initial plan has only READ ops, verdicts are checked later
        # when FixSpec generates new MODIFY ops.
        if spec is not None:
            _spec_meta = getattr(spec, "metadata", None) or {}
            _ev = _spec_meta.get("evaluator_verdicts") or []
            if _ev:
                op_plan.metadata["_evaluator_verdicts"] = _ev
        if op_plan is not None and grounding_summary is not None:
            op_plan.grounding_summary = grounding_summary

        # Inject RUN_SCANNER ops when spec targets include scanner modules (SL58 fix).
        if spec is not None:
            from .operation_models import Operation
            from .scanner_registry import _SCANNER_REGISTRY

            _scanner_names = _SCANNER_REGISTRY.names_for_spec_target_files(
                list(getattr(spec, "target_files", []) or [])
            )
            if _scanner_names:
                _scan_target_files = [
                    tf for tf in (getattr(spec, "target_files", []) or [])
                    if LanguageId.from_path(tf) is LanguageId.PYTHON
                ]
                _scanner_ops: list = []
                for _sn in _scanner_names:
                    _so = Operation(
                        id=f"scan_{_sn}",
                        kind=OperationKind.RUN_SCANNER,
                        path="",
                        intent=f"Execute {_sn} analysis on target files",
                        metadata={
                            "scanner_name": _sn,
                            "paths": list(_scan_target_files),
                            "source": "planner_injection",
                        },
                        depends_on=[],
                    )
                    _scanner_ops.append(_so)

                # Add scanner op as dependency for READ_SYMBOL ops targeting scanner files.
                _scanner_file_stems: set = set()
                for _sn in _scanner_names:
                    _scanner_file_stems.add(f"{_sn}.py")
                    _scanner_file_stems.add(f"analysis/{_sn}.py")
                    _scanner_file_stems.add(f"external_llm/analysis/{_sn}.py")

                _rewritten: list = list(_scanner_ops)
                for _op in op_plan.operations:
                    _op_path = _op.path or ""
                    if _op.kind == OperationKind.READ_SYMBOL and any(
                        _sf in _op_path for _sf in _scanner_file_stems
                    ):
                        _op.depends_on = list(
                            dict.fromkeys(
                                [_scanner_ops[0].id]
                                + (list(_op.depends_on) if _op.depends_on else [])
                            )
                        )
                    _rewritten.append(_op)

                op_plan.operations = _rewritten
                logger.info(
                    "[INJECT_RUN_SCANNER] injected %d RUN_SCANNER op(s) "
                    "for %s (scanning %d files)",
                    len(_scanner_ops), _scanner_names, len(_scan_target_files),
                )

        # Record planned new symbols (absent from codebase) for the missing-callee helper step.
        if spec and isinstance(op_plan.metadata, dict):
            import os as _ens_os
            import re as _ens_re
            # ── Identifier validation ──────────────────────────────────────────────
            # Filter out natural-language keywords that accidentally leaked into
            # target_symbols (e.g. "Create", "TypeScript" in pure-creation requests).
            # A valid code identifier must match ^[a-zA-Z_][a-zA-Z0-9_]*$.
            def _is_code_id(s: str) -> bool:
                return bool(_ens_re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s))

            # _is_likely_natural_language removed — use module-level _is_nl_keyword instead.

            # ── CREATE-only guard ───────────────────────────────────────────────────
            # When target_files is empty and new_files exists (pure file creation),
            # target_symbols is likely natural-language noise, not real symbols.
            # Skip tentative_new_symbols inference entirely in this case.
            _ens_target_files = getattr(spec, "target_files", None) or []
            _ens_new_files = getattr(spec, "new_files", None) or []
            _ens_is_pure_create = bool(_ens_new_files) and not _ens_target_files
            _ens_repo = str(self.registry.repo_root) if hasattr(self.registry, 'repo_root') else '.'
            _ens_new: list = []
            _ens_meta = (getattr(spec, "metadata", None) or {})
            _ens_structural = (
                set(_ens_meta.get("structural_merged_symbols", []))
                | set(_ens_meta.get("similarity_expanded_symbols", []))
            )
            if _ens_is_pure_create:
                logger.debug(
                    "[CONTRACT] pure CREATE — skipping tentative_new_symbols "
                    "from target_symbols (no target_files to verify against)"
                )
            else:
                for _ens_sym in (getattr(spec, "target_symbols", None) or []):
                    _ens_bare = _ens_sym.split(".")[-1] if "." in _ens_sym else _ens_sym
                    if _ens_bare in _ens_structural or _ens_sym in _ens_structural:
                        continue
                    _ens_found = False
                    for _ens_tf in (getattr(spec, "target_files", None) or []):
                        _ens_abs = _ens_tf if _ens_os.path.isabs(_ens_tf) else _ens_os.path.join(_ens_repo, _ens_tf)
                        if _ens_os.path.isfile(_ens_abs):
                            _ens_lang = LanguageId.from_path(_ens_abs).value
                            if _ens_lang == "unknown":
                                continue
                            try:
                                with open(_ens_abs, encoding="utf-8", errors="replace") as _ens_f:
                                    _ens_src = _ens_f.read()
                                from external_llm.languages.tree_sitter_utils import find_all_symbols as _ens_find_syms
                                if _ens_bare in {s[0] for s in _ens_find_syms(_ens_src, _ens_lang)}:
                                    _ens_found = True
                                    break
                            except Exception:
                                pass
                    if not _ens_found and _is_code_id(_ens_bare) and not _is_nl_keyword(_ens_bare):
                        _ens_new.append(_ens_bare)
            if _ens_new:
                op_plan.tentative_new_symbols = _ens_new
                op_plan.symbol_expectation_source = "intent_result"
                logger.info(
                    "[CONTRACT] tentative_new_symbols set (Phase 0 guess, NOT confirmed): %s",
                    _ens_new,
                )
                if (
                    isinstance(op_plan.metadata, dict)
                    and op_plan.metadata.get("_structural_seed_winner")
                    and not op_plan.confirmed_new_symbols
                ):
                    op_plan.confirmed_new_symbols = list(_ens_new)
                    logger.info(
                        "[STRUCTURAL_SEED] auto-confirmed intent_result "
                        "tentative_new_symbols → confirmed: %s",
                        op_plan.confirmed_new_symbols,
                    )

            # expected_new_attributes: attribute/field kinds not detectable by collect_defined_names.
            _DATA_ATTR_KINDS = frozenset({
                "attribute", "field", "config", "threshold",
                "constant", "parameter", "property",
            })
            _ens_nsk = (getattr(spec, "metadata", None) or {}).get(
                "new_symbol_kinds"
            ) or {}
            _ens_attrs = [
                name for name, kind in _ens_nsk.items()
                if (kind or "").lower() in _DATA_ATTR_KINDS
            ]
            if _ens_attrs:
                op_plan.metadata["expected_new_attributes"] = _ens_attrs
                logger.info("[CONTRACT] expected_new_attributes: %s", _ens_attrs)
        self._cb("pipeline_stage", {
            "stage": "decomposition", "status": "complete",
            "detail": f"{len(op_plan.operations)} operations planned",
        })
        self._cb("operation_plan", {
            "description": op_plan.description or "",
            "operations": [
                {
                    "id": op.id,
                    "kind": op.kind.value if hasattr(op.kind, "value") else str(op.kind),
                    "path": op.path or "",
                    "symbol": op.symbol or "",
                    "intent": op.intent or "",
                    "action_hint": (op.metadata or {}).get("action_hint", ""),
                    "preferred_output_mode": (
                        getattr(getattr(op, "edit_contract", None), "preferred_output_mode", "")
                    ),
                }
                for op in op_plan.operations
            ],
            "mode": op_plan.mode.value if hasattr(op_plan.mode, "value") else str(op_plan.mode),
        })
        logger.info(
            "Operation plan created: %d operations, mode=%s",
            len(op_plan.operations), op_plan.mode.value
        )

        return op_plan, _plan_context
