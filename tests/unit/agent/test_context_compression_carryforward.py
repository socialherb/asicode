"""Regression tests for multi-cycle compressed-context carry-forward.

Validates the fix for the duplicate ``[END COMPRESSED CONTEXT]`` marker that
appeared on the FIRST re-compression cycle (the next cycle's marker was carried
forward verbatim from the previous body before a new one was appended), plus
the structural invariants that must hold across arbitrarily many cycles:

  * exactly one ``[END ...]`` marker
  * no nested ``Previous summary (carried forward):`` sections (the previous
    cycle's carry-forward is stripped before being carried again)
  * the fresh categorisation of the current cycle is preserved
  * size stays roughly linear (not quadratic) across cycles
"""
from __future__ import annotations

from external_llm.agent.context_manager import (
    SlidingWindowConfig,
    SlidingWindowContext,
)
from external_llm.client import LLMMessage

_START = "[COMPRESSED CONTEXT]"
_END = "[END COMPRESSED CONTEXT]"
_CF = "Previous summary (carried forward):"


def _compress(prev_compacted, fresh):
    """One compression cycle: build a new compacted message from a previous
    compacted message (if any) plus a batch of fresh messages."""
    dropped = []
    if prev_compacted is not None:
        dropped.append(prev_compacted)
    dropped.extend(fresh)
    cfg = SlidingWindowConfig()
    sw = SlidingWindowContext(cfg)
    return sw._build_compressed_message(dropped)


def _fresh_batch(label: str):
    """A small batch of categorisable messages."""
    return [
        LLMMessage(role="assistant", content=f"working on {label}"),
        LLMMessage(role="user", content=f"continue {label}"),
    ]


def _assert_invariants(msg, cycle):
    body = msg.content
    # exactly one END marker
    assert body.count(_END) == 1, (
        f"cycle {cycle}: expected exactly 1 END marker, got {body.count(_END)}\n{body}"
    )
    # at most one carry-forward section (zero after stripping on re-compress)
    n_cf = body.count(_CF)
    assert n_cf <= 1, (
        f"cycle {cycle}: nested carry-forward sections ({n_cf}) — infinite growth\n{body}"
    )
    # body must START with the header and END with the marker (structural shape)
    assert body.startswith(_START), f"cycle {cycle}: missing header"
    assert body.rstrip().endswith(_END), f"cycle {cycle}: trailing marker corrupted"


def test_first_recompress_no_duplicate_end_marker():
    """The reported bug: cycle 1 carried cycle-0's END marker forward and a
    second one was appended, producing two ``[END COMPRESSED CONTEXT]`` lines.

    The END-strip fix removes the marker from the *carried-forward payload*
    (the body of cycle 0) before it is spliced into cycle 1.  Cycle 1 then
    appends exactly one END marker of its own.  So the invariant is:
    exactly one END marker in the whole message, AND the carried-forward body
    (between the CF marker and the message's own END) contains no END marker.
    """
    c0 = _compress(None, _fresh_batch("zero"))
    c1 = _compress(c0, _fresh_batch("one"))
    _assert_invariants(c0, 0)
    _assert_invariants(c1, 1)

    # Exactly one END marker in the whole message.
    assert c1.content.count(_END) == 1

    # The carried-forward body (CF marker ... message END) must be pure
    # content — no embedded END marker that would duplicate on the next cycle.
    cf_idx = c1.content.find(_CF)
    end_idx = c1.content.rfind(_END)
    assert cf_idx != -1 and end_idx != -1 and cf_idx < end_idx
    carried = c1.content[cf_idx + len(_CF):end_idx]
    assert _END not in carried, (
        f"END marker leaked into carried-forward payload:\n{carried}"
    )


def test_carryforward_marker_stripped_on_recompress():
    """Each carried-forward body must NOT itself contain a carry-forward
    section — otherwise summaries grow quadratically across cycles."""
    c0 = _compress(None, _fresh_batch("zero"))
    c1 = _compress(c0, _fresh_batch("one"))
    # The CF section in c1 should carry cycle-0's *fresh* content only, with
    # no nested CF marker inside it.
    cf_idx = c1.content.find(_CF)
    assert cf_idx != -1, "cycle 1 should carry cycle 0 forward"
    cf_section = c1.content[cf_idx:]
    assert _CF not in cf_section[len(_CF):], (
        f"nested CF marker inside carried-forward section\n{cf_section}"
    )


def test_multi_cycle_size_growth_is_subquadratic():
    """Across 8 compression cycles, total size must stay roughly linear
    (bounded by carry_forward_bytes), not accumulate every prior cycle."""
    sizes = []
    prev = None
    for i in range(8):
        prev = _compress(prev, _fresh_batch(f"c{i}"))
        _assert_invariants(prev, i)
        sizes.append(len(prev.content.encode("utf-8")))
    # The final size must be within 2x of the first carry-forward size
    # (capped content, not the sum of all 8 cycles).
    assert sizes[-1] < 2 * sizes[1], (
        f"size not bounded across cycles: {sizes}"
    )


def test_fresh_categorisation_preserved_on_recompress():
    """The fresh batch's content must appear in the categorised section, not
    be lost when an old summary is carried forward."""
    c0 = _compress(None, _fresh_batch("zero"))
    c1 = _compress(c0, [LLMMessage(role="assistant", content="FRESH_MARKER_XYZ")])
    # The fresh content must appear BEFORE the carry-forward section.
    cf_idx = c1.content.find(_CF)
    fresh_idx = c1.content.find("FRESH_MARKER_XYZ")
    assert fresh_idx != -1, "fresh categorisation lost"
    assert fresh_idx < cf_idx, (
        f"fresh content not before carry-forward (fresh={fresh_idx}, cf={cf_idx})"
    )


# ── Anthropic-native tool_result categorisation (P1 regression) ──────────────
#
# Anthropic tool_result blocks carry ``tool_use_id`` but NO ``name`` field.
# The classifier must recover the tool name from the preceding assistant
# message's ``tool_use`` block (which has ``id`` + ``name``) so results are
# bucketed into the SAME categories the standard role="tool" path uses.
# Without recovery, every Anthropic tool result collapses into "other_tools".

import json as _json


def _anthropic_tool_pair(tool_name: str, tool_use_id: str, *, ok: bool, body: str):
    """The exact Anthropic message shape produced by agent_loop.py:
    an assistant turn carrying a ``tool_use`` block, followed by a user turn
    carrying a ``tool_result`` block (no ``name`` field, only ``tool_use_id``)."""
    return [
        LLMMessage(
            role="assistant",
            content=f"calling {tool_name}",
            raw_content=[
                {"type": "text", "text": f"calling {tool_name}"},
                {"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": {}},
            ],
        ),
        LLMMessage(
            role="user",
            content="tool_result",
            raw_content=[
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _json.dumps(
                        {"ok": ok, "content": body, "error": "boom" if not ok else None}
                    ),
                },
            ],
        ),
    ]


def _anthropic_section(body: str, label: str) -> str | None:
    """Return the text of a categorised section (up to the next blank line),
    or None if the section header is absent."""
    idx = body.find(label)
    if idx == -1:
        return None
    rest = body[idx + len(label):]
    return rest.split("\n\n", 1)[0]


def test_anthropic_tool_result_classified_by_recovered_name():
    """An Anthropic ``apply_patch`` result must land under 'Applied changes',
    NOT 'Other tool calls' — the name is recovered via tool_use_id -> name."""
    dropped = _anthropic_tool_pair("apply_patch", "toolu_01A", ok=True, body="patched foo.py")
    msg = _compress(None, dropped)
    changes = _anthropic_section(msg.content, "Applied changes")
    assert changes is not None and "apply_patch" in changes, (
        f"apply_patch not in Changes section\n{msg.content}"
    )
    other = _anthropic_section(msg.content, "Other tool calls")
    assert other is None or "apply_patch" not in other, (
        f"apply_patch mis-bucketed into Other tool calls\n{msg.content}"
    )


def test_anthropic_tool_result_search_bucket():
    """A ``find_symbol`` Anthropic result must land under the search section."""
    dropped = _anthropic_tool_pair("find_symbol", "toolu_02B", ok=True, body="foo at L10")
    msg = _compress(None, dropped)
    search = _anthropic_section(msg.content, "Symbol / search results")
    assert search is not None and "find_symbol" in search, (
        f"find_symbol not in search section\n{msg.content}"
    )


def test_anthropic_tool_result_failure_bucket():
    """A failed Anthropic tool result (ok=False) must land under 'Failed tool
    calls' and surface the error body."""
    dropped = _anthropic_tool_pair("apply_patch", "toolu_03C", ok=False, body="patched")
    msg = _compress(None, dropped)
    errors = _anthropic_section(msg.content, "Failed tool calls")
    assert errors is not None and "apply_patch" in errors and "boom" in errors, (
        f"failed result not in errors section\n{msg.content}"
    )


def test_anthropic_tool_result_unmapped_id_falls_back():
    """When the tool_use_id has no matching ``tool_use`` block (e.g. the
    assistant turn was already dropped), the result must still be summarised
    under 'Other tool calls' rather than dropped or crashed."""
    dropped = [
        LLMMessage(
            role="user",
            content="orphan tool_result",
            raw_content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_UNKNOWN",
                    "content": _json.dumps({"ok": True, "content": "orphan body"}),
                },
            ],
        ),
    ]
    msg = _compress(None, dropped)
    other = _anthropic_section(msg.content, "Other tool calls")
    assert other is not None and "orphan body" in other, (
        f"unmapped tool_result not summarised\n{msg.content}"
    )
