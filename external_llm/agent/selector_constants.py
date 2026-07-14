"""Shared constants for representation selector policy.

_infer_representation_constraints (planner_agent.py) and
_family_representation_ladder (symbol_handlers.py) must use the same
thresholds so proposed_representation and the family ladder never drift apart.
"""

# local_patch: above this size, ast_direct_body (full-body regen) is cheaper
# and more reliable than surgical_edit's large diff overhead.
LOCAL_PATCH_LARGE_SYM_LINES: int = 80

# helper_extraction / delegate_common_logic: above this size, exclude
# ast_direct_body as fallback — full-body regeneration of a 100+ line function
# risks indentation inconsistency (SL58 pattern: unindent mismatch).
HELPER_EXTRACT_LARGE_SYM_LINES: int = 100

# Dict literal size threshold for surgical_edit ban in the selector.
# Below this line count: ban surgical_edit — prefer ast_direct_body or
# replace_symbol_body (full-value replacement) because small dict literals
# are cheap to regenerate and surgical_edit's exact-formatting reproduction
# requirement often produces byte-identical output (see TYPE_CONFIG case).
# Above this threshold: keep surgical_edit to avoid expensive full-body
# regeneration of large dicts — the safety net in repair_core.py's
# _handle_empty_or_no_diff catches content_unchanged cases as NO_DIFF_GENERATED
# and triggers fallback to an alternative representation.
DICT_LITERAL_LARGE_LINES: int = 20
