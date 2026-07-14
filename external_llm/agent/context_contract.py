"""context_contract.py — Declares what context each operation kind needs.

_gather_symbol_context() skips expensive collection steps not required by the
contract (file_content_window, callee_sources, call-graph traversal).

Design principle: contracts answer "what is needed" structurally, not "how much".
Numeric limits (token budgets, char caps) are safety ceilings, not the selector.
Each contract also declares per-section priority so the _CtxAccumulator drops
the least-important evidence first when the prompt budget is tight.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Section priority constants — match _CtxAccumulator.P_* in operation_executor
# Defined here (no circular import) so ContextContract can use them directly.
# ---------------------------------------------------------------------------
CTX_P_HIGH = 2       # structural evidence: callees, callers, imports, siblings (default)
CTX_P_MEDIUM = 3     # supplementary: tests, inferred intent hints
CTX_P_LOW = 4        # optional: plan_progress, prior_read_files, learning_hint


@dataclass(frozen=True)
class ContextContract:
    """Declares what context an operation needs from _gather_symbol_context.

    Structural flags (what to collect):
      needs_file_window       — include a windowed view of the file around the symbol.
      needs_callees           — include callee source snippets.
      needs_callers           — include caller signatures.
      needs_references        — include cross-file reference data.
      source_symbol_override  — use a different symbol for context extraction.
      class_methods_mode      — how to select related sibling method signatures:
          "none"    — omit siblings entirely.
          "ranked"  — evidence-based selection: graph/AST/intent/prefix/private tiers
                      determine which siblings are evidence for this op; budget cap
                      (measured residual) applied after ranking so top-tier sigs survive.
      prior_read_files_mode   — how to filter state.read_files for prior_read_files:
          "all"           — include every file in state.read_files (default).
          "skip_own"      — exclude the op's own target file (imports already in
                            context["imports"]; 300-line pre-load is redundant).
          "type_defs_only" — only files injected as type-def references (Enum/
                            dataclass/TypedDict), skip all others.

    Evidence priority fields (op-kind-specific importance of each section):
      Each field controls the _CtxAccumulator priority for that section.
      Default = CTX_P_HIGH so nothing drops unless explicitly downgraded.
      Override in canonical contracts to reflect what evidence matters for each op kind.

      callers_priority       — callers section (who calls the target symbol)
      callees_priority       — callee signatures and source snippets
      related_tests_priority — test functions that exercise the target
      siblings_priority      — class sibling method signatures
      prior_analysis_priority — SUMMARIZE_ANALYSIS results from planner
      class_init_priority    — class __init__ body (attribute names)
    """

    # ── Structural collection flags ──────────────────────────────────────────
    needs_file_window: bool = False
    needs_callees: bool = False
    needs_callers: bool = False
    needs_references: bool = False
    source_symbol_override: str = ""
    class_methods_mode: str = "ranked"       # "none" | "ranked"
    prior_read_files_mode: str = "all"       # "all" | "skip_own" | "type_defs_only"

    # ── Per-section evidence priority ────────────────────────────────────────
    callers_priority: int = CTX_P_HIGH
    callees_priority: int = CTX_P_HIGH
    related_tests_priority: int = CTX_P_MEDIUM
    siblings_priority: int = CTX_P_HIGH
    prior_analysis_priority: int = CTX_P_HIGH
    class_init_priority: int = CTX_P_HIGH


# ---------------------------------------------------------------------------
# Canonical contracts — import these in operation handlers
# ---------------------------------------------------------------------------

# Full MODIFY_SYMBOL: callee awareness matters, callers needed for interface compat.
CONTRACT_MODIFY = ContextContract(
    needs_file_window=True,
    needs_callees=True,
    needs_callers=True,
    class_methods_mode="ranked",
    prior_read_files_mode="all",
    # callers: HIGH — signature compat is critical
    callers_priority=CTX_P_HIGH,
    # callees: HIGH — target's dependencies must be understood
    callees_priority=CTX_P_HIGH,
    # tests: MEDIUM — useful reference but not blocking
    related_tests_priority=CTX_P_MEDIUM,
    # siblings: HIGH — naming and helper patterns
    siblings_priority=CTX_P_HIGH,
    prior_analysis_priority=CTX_P_HIGH,
    class_init_priority=CTX_P_HIGH,
)

# MODIFY_SYMBOL body-only: signature is frozen, callers less critical,
# but callees are needed for structural anomaly detection (Pattern C bare_append).
CONTRACT_MODIFY_BODY_ONLY = ContextContract(
    needs_file_window=True,
    needs_callees=True,    # callees needed for Pattern C (bare_append) anomaly detection
    needs_callers=False,
    class_methods_mode="ranked",
    prior_read_files_mode="all",
    # callers: MEDIUM — signature frozen, interface compat not at stake
    callers_priority=CTX_P_MEDIUM,
    callees_priority=CTX_P_HIGH,
    related_tests_priority=CTX_P_MEDIUM,
    siblings_priority=CTX_P_HIGH,
    prior_analysis_priority=CTX_P_HIGH,
    class_init_priority=CTX_P_HIGH,
)

# INSERT_AFTER_SYMBOL: inserting a new symbol; callers/callees of anchor not critical.
CONTRACT_INSERT = ContextContract(
    needs_file_window=True,
    needs_callees=False,
    needs_callers=False,
    class_methods_mode="ranked",
    prior_read_files_mode="skip_own",
    # callers: LOW — insert doesn't affect existing callers
    callers_priority=CTX_P_LOW,
    # callees: LOW — new symbol doesn't have callees yet
    callees_priority=CTX_P_LOW,
    # tests: LOW — not yet relevant for a new insertion
    related_tests_priority=CTX_P_LOW,
    # siblings: HIGH — naming conventions and helper availability are critical
    siblings_priority=CTX_P_HIGH,
    prior_analysis_priority=CTX_P_HIGH,
    # class_init: HIGH — need correct self.xxx attribute names in new method
    class_init_priority=CTX_P_HIGH,
)

# INSERT_AFTER_SYMBOL positional-only anchor.
CONTRACT_INSERT_POSITIONAL = ContextContract(
    needs_file_window=False,
    needs_callees=False,
    needs_callers=False,
    class_methods_mode="none",
    prior_read_files_mode="skip_own",
    callers_priority=CTX_P_LOW,
    callees_priority=CTX_P_LOW,
    related_tests_priority=CTX_P_LOW,
    siblings_priority=CTX_P_LOW,
    prior_analysis_priority=CTX_P_MEDIUM,
    class_init_priority=CTX_P_LOW,
)

# INSERT_AFTER_SYMBOL when anchor is a class method (naming + attributes matter more).
# Callers/callees/tests are still LOW (new insertion), but siblings and class_init are HIGH
# because the new method must follow class conventions and use correct self.xxx names.
CONTRACT_INSERT_METHOD = ContextContract(
    needs_file_window=True,
    needs_callees=False,
    needs_callers=False,
    class_methods_mode="ranked",
    prior_read_files_mode="skip_own",
    callers_priority=CTX_P_LOW,
    callees_priority=CTX_P_LOW,
    related_tests_priority=CTX_P_LOW,
    siblings_priority=CTX_P_HIGH,   # same as CONTRACT_INSERT
    prior_analysis_priority=CTX_P_HIGH,
    class_init_priority=CTX_P_HIGH,
)

# EXTRACT_FUNCTION Phase A: INSERT the new helper function.
# Context gathered on the SOURCE symbol (what we're extracting FROM), not the anchor.
CONTRACT_EXTRACT_INSERT = ContextContract(
    needs_file_window=False,          # positional anchor; DPB provides region content
    needs_callees=True,               # source fn's callees needed for helper generation
    needs_callers=False,
    class_methods_mode="ranked",
    prior_read_files_mode="skip_own",
    callers_priority=CTX_P_LOW,
    callees_priority=CTX_P_HIGH,
    related_tests_priority=CTX_P_LOW,
    siblings_priority=CTX_P_HIGH,
    prior_analysis_priority=CTX_P_HIGH,
    class_init_priority=CTX_P_HIGH,
)

# EXTRACT_FUNCTION Phase B: MODIFY source symbol to replace region with helper call.
# After Phase A the helper exists — cross-op context carries its signature.
CONTRACT_EXTRACT_MODIFY = ContextContract(
    needs_file_window=True,
    needs_callees=False,              # helper handles callees now
    needs_callers=True,
    class_methods_mode="ranked",
    prior_read_files_mode="all",
    callers_priority=CTX_P_MEDIUM,
    callees_priority=CTX_P_LOW,
    related_tests_priority=CTX_P_MEDIUM,
    siblings_priority=CTX_P_LOW,
    prior_analysis_priority=CTX_P_HIGH,
    class_init_priority=CTX_P_MEDIUM,
)

# MOVE_SYMBOL: currently deterministic (GraphRefactorEngine), no LLM prompting.
# Contract defined for future LLM analysis/verification phase.
# Primary concern: ALL callers of the moved symbol need import updates.
CONTRACT_MOVE_SYMBOL = ContextContract(
    needs_file_window=False,
    needs_callees=False,
    needs_callers=True,               # all callers need import updates after move
    class_methods_mode="none",
    prior_read_files_mode="all",
    callers_priority=CTX_P_HIGH,      # caller impact is the core concern for move
    callees_priority=CTX_P_LOW,
    related_tests_priority=CTX_P_MEDIUM,
    siblings_priority=CTX_P_LOW,
    prior_analysis_priority=CTX_P_MEDIUM,
    class_init_priority=CTX_P_LOW,
)

# Pure read (READ_SYMBOL) or minimal lookup: symbol_source only.
CONTRACT_READ = ContextContract(
    needs_file_window=False,
    needs_callees=False,
    needs_callers=False,
    class_methods_mode="none",
    prior_read_files_mode="all",
    callers_priority=CTX_P_LOW,
    callees_priority=CTX_P_LOW,
    related_tests_priority=CTX_P_LOW,
    siblings_priority=CTX_P_LOW,
    prior_analysis_priority=CTX_P_MEDIUM,
    class_init_priority=CTX_P_LOW,
)
