"""
User-prompt builder functions — one per LLM edit mode.

Each function takes (context_block, file_path, symbol, symbol_source) and
returns the user_prompt string for that mode. Dispatched via
_resolve_prompt_mode() + _USER_PROMPT_BUILDERS dict.

Each builder also accepts a keyword `op` (Operation, optional). When
present, builders inject representation-specific REINFORCEMENT instructions
derived from `op.edit_contract` (semantic_change_family, preserve set,
control_flow_preservation_required, must_keep_return_paths, etc.).  This
narrows the LLM's degrees of freedom in the way that matters most for the
chosen representation — e.g. ast_direct_body prompts emphasise control-flow
preservation; replace_symbol_body prompts emphasise the preserve set;
surgical_edit prompts emphasise minimal scope.
"""
from __future__ import annotations

from typing import Any, Optional

from external_llm.agent.operation_models import EditInstructionKind
from external_llm.languages import LanguageId


def _resolve_prompt_mode(
    expected_kind: Optional[EditInstructionKind],
    use_targeted_patch: bool,
    use_search_replace: bool,
) -> str:
    if use_targeted_patch and use_search_replace:
        return "surgical_edit"
    if use_targeted_patch:
        return "ast_op"
    if expected_kind == EditInstructionKind.INSERT_AFTER_SYMBOL:
        return "insert_after_symbol"
    if expected_kind == EditInstructionKind.CREATE_FILE:
        return "create_file"
    return "replace_symbol_body"


# ── Representation-specific reinforcement helpers ─────────────────────────
# These pull semantic / preserve signals off `op.edit_contract` and turn

def _contract_of(op: Any) -> Any:
    """Return op.edit_contract or None when op is missing/unattached."""
    if op is None:
        return None
    return getattr(op, "edit_contract", None)


def _preserve_block(contract: Any) -> str:
    """Emit the change family as neutral context. Structural constraints are
    enforced by the semantic verifier post-edit, not repeated here."""
    if contract is None:
        return ""
    family = (getattr(contract, "semantic_change_family", "") or "").strip()
    if family:
        return f"Change family: {family}\n"
    return ""


def _surgical_edit_reinforcement(contract: Any) -> str:
    return _preserve_block(contract)


def _surgical_edit_line_range_constraint(op: Any) -> str:
    """Return line-range constraint string if op.metadata has start_line/end_line.

    Tier 3: When the planner preserved original line-range metadata (e.g. after
    DELETE_SYMBOL_RANGE fallback), inject a scope constraint into the LLM prompt
    so the surgical edit stays within the intended lines — preventing the LLM from
    expanding the edit scope beyond the originally targeted region.
    """
    if op is None:
        return ""
    metadata = getattr(op, "metadata", {}) or {}
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")
    if start_line is not None and end_line is not None:
        return (
            "IMPORTANT: Your edit MUST be confined to lines "
            f"{start_line}–{end_line}.\n"
            "Do NOT modify code outside this line range.\n"
        )
    if start_line is not None:
        return (
            "IMPORTANT: Your edit MUST start at line "
            f"{start_line} or later.\n"
            "Do NOT modify code before this line.\n"
        )
    return ""


def _ast_op_reinforcement(contract: Any) -> str:
    return _preserve_block(contract)


def _replace_symbol_body_reinforcement(contract: Any) -> str:
    return _preserve_block(contract)


def _ast_direct_body_reinforcement(contract: Any) -> str:
    return _preserve_block(contract)


def render_representation_reinforcement(rep: str, op: Any) -> str:
    """Public entry point used by callers that don't go through
    _USER_PROMPT_BUILDERS (e.g. _try_ast_rewrite_fallback).

    `rep` ∈ {"ast_op", "ast_direct_body", "surgical_edit", "replace_symbol_body"}.
    Returns "" when no contract or unknown rep — safe to drop into a prompt
    via simple string concatenation.
    """
    contract = _contract_of(op)
    if contract is None:
        return ""
    if rep == "surgical_edit":
        return _surgical_edit_reinforcement(contract)
    if rep == "ast_op":
        return _ast_op_reinforcement(contract)
    if rep == "replace_symbol_body":
        return _replace_symbol_body_reinforcement(contract)
    if rep == "ast_direct_body":
        return _ast_direct_body_reinforcement(contract)
    return ""


def _build_surgical_edit_user_prompt(
    context_block: str, file_path: str, symbol: str, symbol_source: str,
    *, op: Any = None,
) -> str:
    _reinforce = _surgical_edit_reinforcement(_contract_of(op))
    _line_constraint = _surgical_edit_line_range_constraint(op)

    # ── Class indent guidance ──
    # When the target symbol is a class, guide the LLM to place new def/class
    # at class body level (4-space indent) rather than inside a method body.
    _indent_guide = ""
    if symbol_source and symbol_source.lstrip().startswith("class "):
        _indent_guide = (
            "INDENT RULE: The target is a class. New methods must use 4-space indentation "
            "(class body level). Do NOT place new def/class inside an existing method body.\n\n"
        )

    return (
        f"{context_block}\n\n"
        f"{_line_constraint}"
        f"{_reinforce}"
        "Apply targeted change using search/replace blocks.\n\n"
        "Output MUST be a JSON object:\n"
        '{\n'
        '  "kind": "SURGICAL_EDIT",\n'
        f'  "file_path": "{file_path}",\n'
        f'  "symbol": "{symbol}",\n'
        '  "data": {\n'
        '    "edits": [\n'
        '      {"search": "<exact text to find, copy verbatim>", "replace": "<replacement text>"}\n'
        '    ]\n'
        '  }\n'
        '}\n\n'
        "RULES:\n"
        '1. "search" must appear EXACTLY in the source (copy from FULL SYMBOL SOURCE below).\n'
        f"{_indent_guide}"
        "2. Use multiple edits if needed (applied in order, top to bottom).\n\n"
        f"FUNCTION SOURCE:\n```python\n{symbol_source or ''}\n```\n"
    )


def _build_ast_op_user_prompt(
    context_block: str, file_path: str, symbol: str, symbol_source: str,
    *, op: Any = None,
) -> str:
    _reinforce = _ast_op_reinforcement(_contract_of(op))
    return (
        f"{context_block}\n\n"
        f"{_reinforce}"
        f"Specify targeted change using typed AST operations.\n\n"
        "Output MUST be a JSON object:\n"
        '{\n'
        '  "kind": "AST_OP",\n'
        f'  "file_path": "{file_path}",\n'
        f'  "symbol": "{symbol}",\n'
        '  "data": {\n'
        '    "ops": [\n'
        '      <one or more op objects, see below>\n'
        '    ]\n'
        '  }\n'
        '}\n\n'
        "Available op types (use the minimum needed):\n"
        '  {"type": "replace_expr", "old": "<SINGLE-LINE expression to find>", "new": "<replacement>"}\n'
        '  {"type": "add_import",   "import": "<import statement, e.g. from x import y>"}\n'
        '  {"type": "add_guard",    "statement": "<guard, e.g. if x is None: return None>"}\n'
        '  {"type": "delete_stmt",  "pattern": "<text — every line containing this is removed>"}\n\n'
        "RULES:\n"
        '1. replace_expr "old" must be a SINGLE LINE — no newlines. Must appear EXACTLY in the source.\n'
        "2. Use multiple ops in order if needed (e.g. add_import first, then replace_expr).\n"
        "3. For import additions always use add_import.\n\n"
        "EXAMPLE — fix wrong method call:\n"
        '  "ops": [{"type": "replace_expr", "old": "data.get(key)", "new": "data.get(key, {})"}]\n\n'
        "EXAMPLE — add missing import + guard:\n"
        '  "ops": [\n'
        '    {"type": "add_import", "import": "from typing import Optional"},\n'
        '    {"type": "add_guard",  "statement": "if self._cache is None: return"}\n'
        '  ]\n\n'
        f"FUNCTION SOURCE:\n```python\n{symbol_source or ''}\n```\n"
    )


def _build_insert_user_prompt(
    context_block: str, file_path: str, symbol: str, symbol_source: str = "",
    *, op: Any = None,
) -> str:
    _src_block = (
        f"\n═══ SOURCE REFERENCE (extract or adapt from this code) ═══\n"
        f"{symbol_source}\n"
        f"═══ END SOURCE REFERENCE ═══\n"
    ) if symbol_source and symbol_source.strip() else ""

    # BR: Inject code_snippet when available — tells the LLM exactly what code to insert
    _code_snippet = getattr(op, 'code_snippet', '') if op is not None else ''
    _code_block = (
        f"\n═══ EXACT CODE TO INSERT ═══\n"
        f"{_code_snippet}\n"
        f"═══ END EXACT CODE ═══\n"
    ) if _code_snippet and _code_snippet.strip() else ''

    # ── Build return value ──
    _parts = [
        context_block + '\n',
        _src_block,
        _code_block,
        "Output MUST be a JSON object with structure:\n"
        '{\n'
        '  "kind": "INSERT_AFTER_SYMBOL",\n'
        f'  "file_path": "{file_path}",\n'
        f'  "symbol": "{symbol}",\n'
        '  "data": { "inserted_code": "<complete new code to insert, with correct indentation>" }\n'
        '}\n\n'
        "Rules for inserted_code:\n"
        "1. Include the COMPLETE new code (def line + body).\n"
        "2. If inserting a CLASS METHOD: use 4-space indentation for the def line and 8-space for the body.\n"
        "3. If inserting a MODULE-LEVEL function: use 0-space indentation.\n"
        "4. Include only the new code to add (not the anchor symbol's existing code).\n"
        "5. The inserted code must be syntactically valid.\n"
    ]
    return "".join(_parts)


def _build_create_file_user_prompt(
    context_block: str, file_path: str, symbol: str = "", symbol_source: str = "",
    *, op: Any = None,
) -> str:
    _py_rule = (
        "5. Python source file (.py): content must be valid Python code "
        "(no Markdown, no prose — start with imports or a shebang line).\n"
    ) if file_path and LanguageId.from_path(file_path) is LanguageId.PYTHON else ""
    return (
        f"{context_block}\n\n"
        "Output the COMPLETE raw source code for the new file. "
        "No JSON, no markdown fences, no explanations.\n\n"
        "RULES:\n"
        "1. Generate the COMPLETE content for a new file.\n"
        "2. Include all necessary imports at the top.\n"
        "3. Follow the project's existing code style.\n"
        "4. The content must be syntactically valid.\n"
        f"{_py_rule}"
    )


def _build_replace_symbol_body_user_prompt(
    context_block: str, file_path: str, symbol: str, symbol_source: str,
    *, op: Any = None,
) -> str:
    _rsb_src = symbol_source or ""
    _reinforce = _replace_symbol_body_reinforcement(_contract_of(op))
    return (
        f"{context_block}\n\n"
        f"{_reinforce}"
        "Output kind: REPLACE_SYMBOL_BODY (not SURGICAL_EDIT or AST_OP).\n\n"
        "Output MUST be a JSON object with structure:\n"
        '{\n'
        '  "kind": "REPLACE_SYMBOL_BODY",\n'
        f'  "file_path": "{file_path}",\n'
        f'  "symbol": "{symbol}",\n'
        '  "data": { "new_body": "<complete function body>" }\n'
        '}\n\n'
        "RULES:\n"
        "1. data.new_body must be the COMPLETE function (def line through end), not a fragment.\n\n"
        + (f"FUNCTION SOURCE:\n```python\n{_rsb_src}\n```\n" if _rsb_src else "")
    )


_USER_PROMPT_BUILDERS = {
    "surgical_edit":      _build_surgical_edit_user_prompt,
    "ast_op":             _build_ast_op_user_prompt,
    "insert_after_symbol": _build_insert_user_prompt,
    "create_file":        _build_create_file_user_prompt,
    "replace_symbol_body": _build_replace_symbol_body_user_prompt,
}
