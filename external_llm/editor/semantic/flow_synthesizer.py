"""flow_synthesizer.py — Phase F.3: Primitive Sequence → Code Synthesis.

Takes a SynthesisPlan (ordered primitives) and generates a complete
Python function by composing code templates.

No LLM. No domain hardcoding. Pure template composition.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Optional

from external_llm.editor.semantic.primitive_code_templates import get_code_template
from external_llm.editor.semantic.synthesis_planner import SynthesisPlan

logger = logging.getLogger(__name__)


def synthesize_function(plan: SynthesisPlan) -> Optional[str]:
    """Generate a complete Python function from a synthesis plan.

    Returns function source code, or None on failure.
    """
    entity = plan.entity or "Item"
    entity_lower = entity.lower()

    # Build template context
    context: dict[str, Any] = {
        "entity": entity,
        "entity_lower": entity_lower,
        "action_type": plan.action_type,
        "params": plan.params,
        "persist_style": "memory",
    }

    # Infer identifier param (first string-like param)
    identifier = "identifier"
    password_param = "password"
    for p in plan.params:
        if p in ("username", "email", "user_id", "login", "name"):
            identifier = p
            break
    for p in plan.params:
        if "password" in p.lower() or "pw" in p.lower():
            password_param = p
            break
    context["identifier"] = identifier
    context["password_param"] = password_param
    context["_sequence"] = list(plan.sequence)  # For dedup (e.g., validate+branch merge)

    # Compose templates in sequence order
    imports: list[str] = []
    body_lines: list[str] = []
    all_imports = set()

    for prim_name in plan.sequence:
        tmpl = get_code_template(prim_name, context)

        if tmpl.imports and tmpl.imports not in all_imports:
            imports.append(tmpl.imports)
            all_imports.add(tmpl.imports)

        if tmpl.body:
            # Indent body lines
            for line in tmpl.body.splitlines():
                body_lines.append(f"    {line}" if line.strip() else "")

    if not body_lines:
        return None

    # Build function signature
    params_str = ", ".join(plan.params) if plan.params else ""
    func_lines: list[str] = []

    # Imports
    func_lines.append("from fastapi import APIRouter, HTTPException")
    for imp in imports:
        if imp not in func_lines:
            func_lines.append(imp)
    func_lines.append("")

    # Decorator
    if plan.decorator:
        func_lines.append(plan.decorator)

    # Def line
    func_lines.append(f"def {plan.action_name}({params_str}):")
    func_lines.append(f'    """Auto-generated {plan.action_type} handler for {entity}."""')

    # Body
    func_lines.extend(body_lines)

    # Ensure trailing newline
    code = "\n".join(func_lines)
    if not code.endswith("\n"):
        code += "\n"

    # Validate syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.debug("[SYNTH] syntax error: %s\n%s", e, code[:200])
        return None

    logger.info(
        "[SYNTH] generated %s() (%d lines, %d primitives, source=%s)",
        plan.action_name, len(func_lines), len(plan.sequence), plan.source,
    )

    return code


def synthesize_from_ir(
    primitive_ir: Any,
    sequence_store: Any = None,
    scope: Optional[set[str]] = None,
) -> dict[str, str]:
    """Synthesize functions for all actions in a PrimitiveIR.

    Args:
        primitive_ir: IR whose sequences are candidates for synthesis.
        sequence_store: Optional primitive sequence store for learned patterns.
        scope: Optional relevance gate — a set of bare function names that are
            'in scope' for synthesis. When provided, sequences whose
            ``action_name`` is not in the set are skipped regardless of
            coverage. ``None`` disables the gate (legacy behaviour, used by
            synthesis-only unit tests).

    Returns {action_name: generated_code}.
    """
    from external_llm.editor.semantic.synthesis_planner import plan_synthesis

    results: dict[str, str] = {}
    skipped_out_of_scope = 0

    for seq in primitive_ir.sequences:
        if scope is not None and seq.action_name not in scope:
            skipped_out_of_scope += 1
            continue
        if seq.coverage >= 0.9:
            continue  # Already good enough

        plan = plan_synthesis(
            action_name=seq.action_name,
            action_type=seq.action_type,
            entity=seq.entity,
            params=[],  # Params not available from IR
            sequence_store=sequence_store,
            missing_primitives=seq.missing_names,
        )

        code = synthesize_function(plan)
        if code:
            results[seq.action_name] = code

    if results:
        logger.info("[SYNTH] generated %d functions: %s", len(results), list(results.keys()))
    if skipped_out_of_scope:
        logger.info(
            "[SYNTH] relevance gate skipped %d out-of-scope sequences",
            skipped_out_of_scope,
        )

    return results
