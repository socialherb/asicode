"""Regression: WRITE_TOOL_NAMES is the single source of truth for the set of
file-modifying tool names.

Both ``ToolRegistry._WRITE_TOOLS`` (file-locking, failure-logging, cache
invalidation) and ``TurnContext.write_tools`` (reads_since_last_edit reset,
write_tool_used detection, test-impact invalidation) must derive from
``WRITE_TOOL_NAMES``. A write tool added to one literal but not the other
lets edits via it silently bypass up to six correctness mechanisms — this
test makes that drift fail loudly instead of silently bypassing the gates.
"""
from external_llm.agent.agent_loop_types import WRITE_TOOL_NAMES, TurnContext
from external_llm.agent.tool_registry import ToolRegistry

_EXPECTED = frozenset({
    "apply_patch", "write_plan", "edit_ast", "edit_file",
    "edit_text", "modify_symbol", "anchor_edit",
})


def test_write_tool_names_value_is_pinned():
    """SSOT value must be exactly the seven file-modifying tools.

    ``create_file`` is only a ``write_plan`` *op* (no handler, no dispatch
    key in ``_TOOL_HANDLER_MAP``) so it must NOT be a top-level write tool,
    while ``edit_file`` IS a real internal tool (dispatched via
    ``delegate_to_helper``) and MUST be present.
    """
    assert WRITE_TOOL_NAMES == _EXPECTED
    assert "create_file" not in WRITE_TOOL_NAMES
    assert "edit_file" in WRITE_TOOL_NAMES


def test_tool_registry_write_tools_derive_from_ssot():
    """``ToolRegistry._WRITE_TOOLS`` must equal the SSOT set.

    Guards against a future revert to a divergent literal (e.g. dropping a
    name) that would let one write tool bypass locking / failure-logging /
    cache invalidation.
    """
    assert set(ToolRegistry._WRITE_TOOLS) == set(WRITE_TOOL_NAMES)


def test_turn_context_write_tools_default_derives_from_ssot():
    """``TurnContext.write_tools`` default must equal the SSOT set.

    Guards against a future revert to a divergent literal that would let one
    write tool bypass reads_since_last_edit reset, write_tool_used detection,
    and test-impact invalidation.
    """
    default = TurnContext.__dataclass_fields__["write_tools"].default_factory()
    assert default == set(WRITE_TOOL_NAMES)
    # Each instantiation must be an independent mutable copy, not the shared
    # ClassVar / SSOT object itself (a shared set would let one turn context's
    # writes leak into another).
    other = TurnContext.__dataclass_fields__["write_tools"].default_factory()
    assert default is not other
    default.add("sentinel")
    assert "sentinel" not in other
