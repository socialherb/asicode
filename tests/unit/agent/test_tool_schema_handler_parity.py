"""Every tool advertised to a model must be dispatchable on that surface.

`AGENT_TOOL_SCHEMAS` is what the model is told exists; `_TOOL_HANDLER_MAP` is
what `dispatch()` can actually run. Nothing tied the two together, and they
drifted: `save_insight`, `edit_insight`, `delete_insight` and
`search_design_history` are dispatched by DesignChatLoop (which intercepts them
by name before dispatch), yet `get_tool_schemas()` handed them to the coding
agent lane as well — where calling one returns

    Unknown tool: save_insight. Available tools: [...]

An "unknown tool" is indistinguishable from a bug in the harness from the
model's side, so the failure is silent and unrecoverable rather than merely
useless. These tests pin both directions of the contract.
"""
from __future__ import annotations

import pytest

from external_llm.agent.tool_registry import ToolRegistry
from external_llm.agent.tool_schemas import (
    AGENT_TOOL_SCHEMAS,
    DESIGN_CHAT_ONLY_TOOL_NAMES,
    TOOL_NAME_VARIANTS,
    TOOL_SCHEMA_VARIANTS,
)


def _names(schemas):
    return {s["name"] for s in schemas}


def test_every_advertised_tool_has_a_handler(tool_registry):
    """The agent lane must not advertise a tool `dispatch()` cannot run."""
    advertised = _names(tool_registry.get_tool_schemas())
    missing = sorted(n for n in advertised if not tool_registry.has_tool_handler(n))
    assert not missing, (
        f"advertised to the agent lane but not dispatchable: {missing}. "
        "Either add a handler to _TOOL_HANDLER_MAP or mark the schema "
        '"x_design_chat_only": True.'
    )


def test_the_design_chat_surface_is_the_only_one_that_advertises_them(tool_registry):
    assert DESIGN_CHAT_ONLY_TOOL_NAMES, "the marker flag stopped matching any schema"
    agent = _names(tool_registry.get_tool_schemas())
    design = _names(tool_registry.get_tool_schemas(design_chat=True))
    assert not (DESIGN_CHAT_ONLY_TOOL_NAMES & agent)
    assert design >= DESIGN_CHAT_ONLY_TOOL_NAMES
    assert design - agent == DESIGN_CHAT_ONLY_TOOL_NAMES


@pytest.mark.parametrize("name", sorted(DESIGN_CHAT_ONLY_TOOL_NAMES))
def test_a_design_chat_tool_is_genuinely_undispatchable_here(tool_registry, name):
    """The reason they must stay hidden — asserted against the real dispatch,
    so the day a handler IS added this test fails and says to unmark it."""
    assert not tool_registry.has_tool_handler(name)
    result = tool_registry.dispatch(name, {})
    assert not result.ok
    assert "Unknown tool" in (result.error or "")


def test_names_and_schemas_agree_on_every_variant(tool_registry):
    """`get_tool_names` gates validation and `get_tool_schemas` gates
    advertisement; a disagreement rejects a tool the model was just offered."""
    for key, schemas in TOOL_SCHEMA_VARIANTS.items():
        assert TOOL_NAME_VARIANTS[key] == _names(schemas), key


def test_variants_have_stable_identity(tool_registry):
    """One shared list per variant, not one per registry or per call.

    Not for the token cache — that keys on a content fingerprint, explicitly
    "not id()". This just pins that the variant tables stay module-level, so
    reintroducing a per-instance memo is a visible change rather than a silent
    one.
    """
    other = ToolRegistry(tool_registry.repo_root, tool_registry.config)
    assert tool_registry.get_tool_schemas() is tool_registry.get_tool_schemas()
    assert tool_registry.get_tool_schemas() is other.get_tool_schemas(), (
        "variant lists must be shared across registries, not memoized per instance"
    )


def test_python_only_and_design_chat_filters_compose(tool_registry):
    """The two flags are independent; combining them must not drop the other's
    exclusions."""
    from external_llm.languages import LanguageId

    ts_agent = _names(tool_registry.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT))
    ts_design = _names(
        tool_registry.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT, design_chat=True)
    )
    python_only = {s["name"] for s in AGENT_TOOL_SCHEMAS if s.get("x_python_only")}
    assert python_only, "no schema carries x_python_only any more"
    assert not (python_only & ts_agent)
    assert not (python_only & ts_design)
    assert not (DESIGN_CHAT_ONLY_TOOL_NAMES & ts_agent)
    assert ts_design >= DESIGN_CHAT_ONLY_TOOL_NAMES


# ── the other direction: a handler nothing advertises ───────────────────────
# `find_tests_for_symbol` had a working handler, a working backend
# (SymbolAwareTestFinder), an entry in work_state_digest's tool taxonomy and
# render support in asi.py — and no schema and no _TOOL_HANDLER_MAP entry, so no
# model could ever call it. That drift is invisible from the model's side (the
# tool simply does not exist) and invisible from the code's side (every piece
# looks wired to the piece next to it), which is why it needs a gate of its own.

# Handlers that are dispatched by something other than a model tool call.
# Adding a handler without a schema is a decision, so it must be made HERE.
INTERNAL_ONLY_HANDLERS = frozenset({
    # Reached through delegate_to_helper, not chosen by the model.
    "delegate_to_helper",
    "delegate_to_local_model",
    "edit_file",
    # Removed from the schemas because bash is equivalent and more flexible;
    # kept dispatchable for internal callers and backward compatibility.
    "run_tests",
    "run_lint",
    # Internal bookkeeping invoked by the loop, never advertised.
    "update_memory",
})


def test_every_handler_is_reachable_or_declared_internal():
    """A dispatchable tool the model is never told about is dead weight."""
    advertised = _names(AGENT_TOOL_SCHEMAS)
    orphaned = sorted(
        n for n in ToolRegistry._TOOL_HANDLER_MAP
        if n not in advertised and n not in INTERNAL_ONLY_HANDLERS
    )
    assert not orphaned, (
        f"handler exists but nothing advertises it: {orphaned}. "
        "Either add a schema to AGENT_TOOL_SCHEMAS or list it in "
        "INTERNAL_ONLY_HANDLERS with the reason."
    )


def test_internal_only_list_has_no_stale_entries():
    """The allowlist must not outlive the handlers it excuses."""
    stale = sorted(INTERNAL_ONLY_HANDLERS - set(ToolRegistry._TOOL_HANDLER_MAP))
    assert not stale, f"listed as internal-only but no longer a handler: {stale}"
    advertised = _names(AGENT_TOOL_SCHEMAS)
    now_public = sorted(INTERNAL_ONLY_HANDLERS & advertised)
    assert not now_public, (
        f"advertised to models but still listed as internal-only: {now_public}"
    )
