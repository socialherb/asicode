"""primitive_code_templates.py — Phase F.3: Primitive → Code Templates.

Parameterized code fragments for each semantic primitive.
Templates use {placeholders} for entity, variable, identifier names.

Design: domain-independent. Entity/var names are injected by caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeTemplate:
    """A parameterized code template for a primitive."""
    primitive: str
    imports: str = ""          # Import lines (may be empty)
    body: str = ""             # Function body lines
    variables_produced: list[str] = field(default_factory=list)
    # Variables this template creates (for downstream reference)
    variables_consumed: list[str] = field(default_factory=list)
    # Variables this template needs from earlier steps


# ── Template Definitions ──────────────────────────────────────────────────────

def _tmpl_lookup(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "Item")
    entity_lower = ctx.get("entity_lower", entity.lower())
    # Resolve identifier from params: prefer _id suffix, then username/email, then first param
    params = ctx.get("params", [])
    identifier = ctx.get("identifier", "identifier")
    for p in (params if isinstance(params, list) else []):
        if p.endswith("_id") or p in ("username", "email", "name", "slug"):
            identifier = p
            break
    if identifier == "identifier" and params:
        identifier = params[0] if isinstance(params, list) else "identifier"
    return CodeTemplate(
        primitive="lookup",
        body=f"{entity_lower} = get_{entity_lower}({identifier})\n"
             f"if not {entity_lower}:\n"
             f"    raise ValueError(f\"{entity} not found: {{{identifier}}}\")",
        variables_produced=[entity_lower],
        variables_consumed=[identifier],
    )


def _tmpl_validate(ctx: dict[str, str]) -> CodeTemplate:
    entity_lower = ctx.get("entity_lower", "user")
    password_param = ctx.get("password_param", "password")
    action_type = ctx.get("action_type", "")

    if action_type == "login":
        return CodeTemplate(
            primitive="validate",
            body=f"if not verify_password({password_param}, {entity_lower}.hashed_password):\n"
                 f"    raise ValueError(\"Invalid credentials\")",
            variables_consumed=[password_param, entity_lower],
        )
    return CodeTemplate(
        primitive="validate",
        body=f"if not {entity_lower}:\n"
             f"    raise ValueError(\"Invalid input\")",
        variables_consumed=[entity_lower],
    )


def _tmpl_branch_on_failure(ctx: dict[str, str]) -> CodeTemplate:
    # If validate is also in the sequence, validate already includes the branch.
    # Emit empty to avoid duplicate.
    sequence = ctx.get("_sequence", [])
    if "validate" in sequence:
        return CodeTemplate(primitive="branch_on_failure")  # No-op: merged into validate
    entity_lower = ctx.get("entity_lower", "result")
    return CodeTemplate(
        primitive="branch_on_failure",
        body=f"if not {entity_lower}:\n"
             f"    raise ValueError(\"Operation failed\")",
        variables_consumed=[entity_lower],
    )


def _tmpl_authorize(ctx: dict[str, str]) -> CodeTemplate:
    identifier = ctx.get("identifier", "username")
    return CodeTemplate(
        primitive="authorize",
        imports="",
        body=f"token = create_access_token({{\"sub\": {identifier}}})\n",
        variables_produced=["token"],
        variables_consumed=[identifier],
    )


def _tmpl_create_entity(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "Item")
    entity_lower = ctx.get("entity_lower", entity.lower())
    params = ctx.get("params", [])

    if params:
        args = ", ".join(f"{p}={p}" for p in params if p not in ("self", "db", "session"))
    else:
        args = ""

    return CodeTemplate(
        primitive="create_entity",
        body=f"{entity_lower} = {entity}({args})",
        variables_produced=[entity_lower],
        variables_consumed=list(params) if params else [],
    )


def _tmpl_input_bind(ctx: dict[str, str]) -> CodeTemplate:
    # Input binding is implicit in create_entity args
    # This produces a "data" variable when needed
    params = ctx.get("params", [])
    if params:
        fields = ", ".join(f'"{p}": {p}' for p in params if p not in ("self", "db", "session"))
        return CodeTemplate(
            primitive="input_bind",
            body=f"data = {{{fields}}}",
            variables_produced=["data"],
            variables_consumed=list(params),
        )
    return CodeTemplate(primitive="input_bind")


def _tmpl_update_entity(ctx: dict[str, str]) -> CodeTemplate:
    entity_lower = ctx.get("entity_lower", "item")
    params = ctx.get("params", [])
    updates = "\n".join(
        f"if {p} is not None:\n    {entity_lower}.{p} = {p}"
        for p in params if p not in ("self", "db", "session", "id", f"{entity_lower}_id")
    )
    return CodeTemplate(
        primitive="update_entity",
        body=updates or f"# Update {entity_lower} fields",
        variables_consumed=[entity_lower, *list(params)],
    )


def _tmpl_delete_entity(ctx: dict[str, str]) -> CodeTemplate:
    entity_lower = ctx.get("entity_lower", "item")
    return CodeTemplate(
        primitive="delete_entity",
        body=f"db.delete({entity_lower})\ndb.commit()",
        variables_consumed=[entity_lower],
    )


def _tmpl_persist_state(ctx: dict[str, str]) -> CodeTemplate:
    entity_lower = ctx.get("entity_lower", "item")
    persist_style = ctx.get("persist_style", "memory")
    action_type = ctx.get("action_type", "create")

    # For delete actions, persist = commit the deletion
    if action_type == "delete":
        if persist_style in ("sqlalchemy", "session"):
            return CodeTemplate(
                primitive="persist_state",
                body="db.commit()",
            )
        return CodeTemplate(
            primitive="persist_state",
            body=f"# {entity_lower} removed from store",
        )

    if persist_style == "sqlalchemy":
        return CodeTemplate(
            primitive="persist_state",
            body=f"db.add({entity_lower})\ndb.commit()\ndb.refresh({entity_lower})",
            variables_consumed=[entity_lower],
        )
    if persist_style == "session":
        return CodeTemplate(
            primitive="persist_state",
            body=f"session.add({entity_lower})\nsession.commit()",
            variables_consumed=[entity_lower],
        )
    # Memory fallback
    return CodeTemplate(
        primitive="persist_state",
        body=f"_{entity_lower}s.append({entity_lower})",
        variables_consumed=[entity_lower],
    )


def _tmpl_list_or_query(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "Item")
    entity_lower = ctx.get("entity_lower", entity.lower())
    return CodeTemplate(
        primitive="list_or_query",
        body=f"items = get_all_{entity_lower}s()\nreturn {{\"items\": items, \"total\": len(items)}}",
        variables_produced=["items"],
    )


def _tmpl_produce_output(ctx: dict[str, str]) -> CodeTemplate:
    action_type = ctx.get("action_type", "create")
    entity_lower = ctx.get("entity_lower", "item")

    if action_type == "login":
        return CodeTemplate(
            primitive="produce_output",
            body='return {"access_token": token, "token_type": "bearer"}',
            variables_consumed=["token"],
        )
    if action_type in ("list", "get"):
        return CodeTemplate(
            primitive="produce_output",
            body=f'return {{{entity_lower}}}',
            variables_consumed=[entity_lower],
        )
    if action_type == "delete":
        return CodeTemplate(
            primitive="produce_output",
            body=f'return {{"deleted": True, "id": getattr({entity_lower}, "id", None)}}',
            variables_consumed=[entity_lower],
        )
    if action_type == "update":
        return CodeTemplate(
            primitive="produce_output",
            body=f'return {{"updated": True, "id": getattr({entity_lower}, "id", None)}}',
            variables_consumed=[entity_lower],
        )
    # Default: create-like
    return CodeTemplate(
        primitive="produce_output",
        body=f'return {{"id": getattr({entity_lower}, "id", 1), "status": "created"}}',
        variables_consumed=[entity_lower],
    )


def _tmpl_delegate_action(ctx: dict[str, str]) -> CodeTemplate:
    # Delegation is structural — no code template needed
    return CodeTemplate(primitive="delegate_action")


# ── Extended Primitives (#9) ─────────────────────────────────────────────────

def _tmpl_paginate(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "Item")
    return CodeTemplate(
        primitive="paginate",
        body=f"    items = items[offset:offset + limit]\n"
             f"    return {{\"{entity.lower()}s\": items, \"total\": total, \"offset\": offset, \"limit\": limit}}",
        imports="from typing import List",
    )


def _tmpl_cache(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "item")
    return CodeTemplate(
        primitive="cache",
        body=f"    cache_key = f\"{entity.lower()}_{{identifier}}\"\n"
             f"    cached = _cache.get(cache_key)\n"
             f"    if cached is not None:\n"
             f"        return cached",
    )


def _tmpl_rate_limit(ctx: dict[str, str]) -> CodeTemplate:
    return CodeTemplate(
        primitive="rate_limit",
        body="    if not rate_limiter.allow(client_id):\n"
             "        raise RuntimeError(\"Rate limit exceeded\")",
    )


def _tmpl_transform(ctx: dict[str, str]) -> CodeTemplate:
    entity = ctx.get("entity", "item")
    return CodeTemplate(
        primitive="transform",
        body=f"    result = {entity.lower()}.to_dict() if hasattr({entity.lower()}, 'to_dict') else {entity.lower()}",
    )


# ── Registry ──────────────────────────────────────────────────────────────────

_TEMPLATE_BUILDERS = {
    # Base 12
    "lookup": _tmpl_lookup,
    "validate": _tmpl_validate,
    "branch_on_failure": _tmpl_branch_on_failure,
    "authorize": _tmpl_authorize,
    "create_entity": _tmpl_create_entity,
    "input_bind": _tmpl_input_bind,
    "update_entity": _tmpl_update_entity,
    "delete_entity": _tmpl_delete_entity,
    "persist_state": _tmpl_persist_state,
    "list_or_query": _tmpl_list_or_query,
    "produce_output": _tmpl_produce_output,
    "delegate_action": _tmpl_delegate_action,
    # Extended 4 (#9)
    "paginate": _tmpl_paginate,
    "cache": _tmpl_cache,
    "rate_limit": _tmpl_rate_limit,
    "transform": _tmpl_transform,
}


def get_code_template(primitive: str, context: dict[str, str]) -> CodeTemplate:
    """Get a parameterized code template for a primitive."""
    builder = _TEMPLATE_BUILDERS.get(primitive)
    if builder:
        return builder(context)
    return CodeTemplate(primitive=primitive, body=f"# TODO: {primitive}")
