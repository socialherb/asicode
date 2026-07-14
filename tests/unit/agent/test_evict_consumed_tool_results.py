"""Regression: ``_evict_consumed_tool_results`` must actually bound context.

Previous implementation treated "referenced by a tool_call" as "preserve
forever". But by construction every tool_result is referenced by its
preceding assistant tool_call, so the keep_recent budget never applied in
normal conversations — the function was a silent no-op (10 tool results in,
10 out). These tests pin the corrected behaviour:

* content of tool_results beyond the recent window is stubbed (context shrinks);
* the message shell stays so the tool_call<->tool_result pairing never breaks
  (which would trigger orphaned-tool_call HTTP 400 — the reason
  ``repair_tool_message_sequence`` exists);
* the function is idempotent across turns and never grows context by stubbing
  results smaller than the stub itself.
"""
from external_llm.agent.agent_turn_pipeline import (
    _EVICT_MIN_CONTENT_LEN,
    _EVICTED_MARKER,
    _evict_consumed_tool_results,
)
from external_llm.client import LLMMessage


def _pair(idx: int, content: str, name: str = "read_file") -> tuple[LLMMessage, LLMMessage]:
    """Build an assistant tool_call + matching tool result pair."""
    tid = f"call_{idx}"
    asst = LLMMessage(
        role="assistant",
        content="",
        tool_calls=[{"id": tid, "function": {"name": name, "arguments": "{}"}}],
    )
    tool = LLMMessage(role="tool", content=content, tool_call_id=tid, name=name)
    return asst, tool


def _tool_call_ids(msgs):
    ids = set()
    for m in msgs:
        if getattr(m, "role", "") == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                ids.add(tc.get("id"))
    return ids


def _tool_result_ids(msgs):
    return {getattr(m, "tool_call_id", None) for m in msgs if getattr(m, "role", "") == "tool"}


def test_old_tool_results_are_stubbed_not_kept():
    """10-turn tool conversation: recent 6 verbatim, older 4 content stubbed."""
    msgs = []
    for i in range(10):
        a, t = _pair(i, f"BIG FILE CONTENT #{i} " * 200)
        msgs += [a, t]

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    big_surviving = [m for m in out if getattr(m, "role", "") == "tool" and "BIG FILE CONTENT" in (m.content or "")]
    stubbed = [m for m in out if getattr(m, "role", "") == "tool" and (m.content or "").startswith(_EVICTED_MARKER)]
    assert len(big_surviving) == 6
    assert len(stubbed) == 4


def test_no_messages_removed_pairing_preserved():
    """Stubbing must never drop messages or break the tool_call<->result pairing."""
    msgs = []
    for i in range(8):
        a, t = _pair(i, "X" * 500)
        msgs += [a, t]

    out = _evict_consumed_tool_results(msgs, keep_recent=3)

    assert len(out) == len(msgs), "no messages may be removed"
    assert _tool_call_ids(out) == _tool_result_ids(out), "no orphaned tool_calls/results"
    # tool_call_id + name retained on every (even stubbed) result
    for m in out:
        if getattr(m, "role", "") == "tool":
            assert m.tool_call_id
            assert m.name == "read_file"


def test_idempotent_across_turns():
    """Re-running on an already-evicted list must not re-stub or change counts."""
    msgs = []
    for i in range(10):
        a, t = _pair(i, "Y" * 600)
        msgs += [a, t]

    once = _evict_consumed_tool_results(msgs, keep_recent=6)
    once_stubbed = sum(1 for m in once if (getattr(m, "content", "") or "").startswith(_EVICTED_MARKER))
    once_snapshot = [m.content for m in once if getattr(m, "role", "") == "tool"]

    twice = _evict_consumed_tool_results(once, keep_recent=6)
    twice_stubbed = sum(1 for m in twice if (getattr(m, "content", "") or "").startswith(_EVICTED_MARKER))
    twice_snapshot = [m.content for m in twice if getattr(m, "role", "") == "tool"]

    assert once_stubbed == twice_stubbed == 4
    assert once_snapshot == twice_snapshot, "idempotent — content stable on second pass"


def test_tiny_results_left_verbatim():
    """Stubbing a result smaller than the stub would grow context — keep verbatim."""
    msgs = []
    for i in range(8):
        a, t = _pair(i, "ok", name="noop")
        msgs += [a, t]

    out = _evict_consumed_tool_results(msgs, keep_recent=2)

    verbatim = [m for m in out if getattr(m, "role", "") == "tool" and m.content == "ok"]
    stubbed = [m for m in out if getattr(m, "role", "") == "tool" and (m.content or "").startswith(_EVICTED_MARKER)]
    assert verbatim and len(verbatim) == 8
    assert stubbed == []


def test_boundary_size_floor_kept_just_under_threshold():
    """Result at exactly the floor is kept; just above is stubbed (when old)."""
    msgs = []
    # one under-threshold old result, then keep_recent fresh pairs push it out
    a0, t0 = _pair(0, "A" * _EVICT_MIN_CONTENT_LEN)  # exactly at floor -> verbatim
    msgs += [a0, t0]
    a1, t1 = _pair(1, "B" * (_EVICT_MIN_CONTENT_LEN + 1))  # above floor -> stubbed
    msgs += [a1, t1]
    for i in range(2, 8):  # 6 fresh pairs fill keep_recent=6 window
        a, t = _pair(i, "C" * (_EVICT_MIN_CONTENT_LEN + 1))
        msgs += [a, t]

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    at_floor = [m for m in out if getattr(m, "role", "") == "tool" and (m.content or "").startswith("A" * 10)]
    stubbed_b = [m for m in out if getattr(m, "role", "") == "tool" and (m.content or "").startswith(_EVICTED_MARKER)]
    assert len(at_floor) == 1, "result exactly at floor left verbatim even when old"
    # t1 is the only old result above the floor -> stubbed
    assert len(stubbed_b) == 1


def test_non_tool_messages_untouched():
    """user/assistant plain messages pass through unchanged."""
    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hello"),
    ]
    out = _evict_consumed_tool_results(msgs, keep_recent=6)
    assert [m.content for m in out] == ["sys", "hello"]


def test_empty_and_short_input():
    """Degenerate inputs return without error."""
    assert _evict_consumed_tool_results([], keep_recent=6) == []
    a, t = _pair(0, "X" * 500)
    out = _evict_consumed_tool_results([a, t], keep_recent=6)
    assert len(out) == 2 and out[1].content == "X" * 500  # within window, verbatim


def test_copy_on_write_leaves_original_intact():
    """Stubbing must NOT mutate the original message object — ctx.messages
    entries are shared with event payloads / run records, so an in-place
    ``m.content = ...`` would retroactively rewrite recorded history. The
    stubbed entry must be a NEW object (dataclasses.replace)."""
    msgs = list(_pair(0, "ORIG " * 200))  # big tool result beyond window
    for i in range(1, 7):  # 6 fresh pairs fill keep_recent=6
        msgs += list(_pair(i, "FRESH " * 200))
    original_tool = msgs[1]
    original_content = original_tool.content

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    # original object untouched
    assert original_tool.content == original_content, \
        "original message object was mutated in place (not copy-on-write)"
    # the evicted slot in `out` is a distinct object carrying the stub
    evicted = next(
        m for m in out
        if getattr(m, "role", "") == "tool"
        and (m.content or "").startswith(_EVICTED_MARKER)
    )
    assert evicted is not original_tool, "evicted entry must be a copy, not the same object"


def test_multipart_raw_content_is_stubbed():
    """Anthropic/Gemini carry the tool_result payload in raw_content (content[]
    / parts[]) with content == ''. Ignoring raw_content leaves eviction a silent
    no-op for those providers — the stub must measure AND clear raw_content."""
    a = LLMMessage(
        role="assistant", content="",
        tool_calls=[{"id": "c0", "function": {"name": "read_file", "arguments": "{}"}}],
    )
    big_blocks = [{"type": "text", "text": "Z" * 500} for _ in range(3)]
    t = LLMMessage(
        role="tool", content="", tool_call_id="c0", name="read_file",
        raw_content=big_blocks,
    )
    msgs = [a, t]
    for i in range(1, 7):
        msgs += list(_pair(i, "FRESH " * 200))

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    evicted = [
        m for m in out
        if getattr(m, "role", "") == "tool"
        and (m.content or "").startswith(_EVICTED_MARKER)
    ]
    assert len(evicted) == 1, "multipart (raw_content) result beyond window must be stubbed"
    assert evicted[0].raw_content is None, \
        "raw_content must be cleared — else the provider still sends the full payload"
    # copy-on-write: original raw_content untouched
    assert t.raw_content == big_blocks, "original raw_content was mutated in place"


def test_hysteresis_pending_count_cadence():
    """Simulate 30 turns and verify true N-turn cadence.

    With keep_recent=6 and batch_evict_threshold=6, eviction fires only when
    pending (non-stubbed) tool results reach 12.  After each batch the pending
    count resets to ~6, so over 30 turns we expect at most
    ceil((30 - 12) / 6) + 1 ≈ 4 eviction events, not 19.
    """
    keep_recent = 6
    threshold = 6
    msgs: list[LLMMessage] = []
    eviction_turns: list[int] = []

    for turn in range(30):
        # Each turn adds one fresh assistant+tool pair (big content)
        a, t = _pair(turn, f"TURN {turn} BIG DATA " * 200)
        msgs += [a, t]
        # Simulate history growth: pre-existing stubs from prior evictions
        out = _evict_consumed_tool_results(msgs, keep_recent=keep_recent,
                                            batch_evict_threshold=threshold)
        # Count *new* stubs introduced this turn (deterministic tool_call_id match)
        evicted_ids = {
            getattr(m, "tool_call_id", None)
            for m in msgs
            if getattr(m, "role", "") == "tool"
            and isinstance(getattr(m, "content", ""), str)
            and m.content.startswith(_EVICTED_MARKER)
        }
        new_stubs = sum(
            1 for m in out
            if getattr(m, "role", "") == "tool"
            and isinstance(getattr(m, "content", ""), str)
            and m.content.startswith(_EVICTED_MARKER)
            and getattr(m, "tool_call_id", None) not in evicted_ids
        )
        if new_stubs:
            eviction_turns.append(turn)
        msgs = out

    # Maximum expected evictions: ceil((30 - keep_recent) / threshold)
    max_expected = (30 - keep_recent + threshold - 1) // threshold
    # Sanity: with default (no hysteresis) you'd get 30 - keep_recent = 24 events
    naive_count = 30 - keep_recent
    assert len(eviction_turns) <= max_expected + 1, (
        f"Hysteresis cadence broken: got {len(eviction_turns)} eviction turns "
        f"(naive = {naive_count}), expected ≤ {max_expected + 1}"
    )
    # Verify at least one eviction happened (threshold was crossed)
    assert len(eviction_turns) >= 1, "No eviction ever fired — hysteresis too aggressive?"


def test_below_occupancy_does_not_fire_eviction():
    """Occupancy gate: while the prompt sits below the trigger fraction of the
    model's cap, ``_evict_for_loop`` must evict NOTHING.

    Eviction's only unconditional benefit is bounding context; firing it earlier
    just mints a self-inflicted cache-miss (the prefix rewrite) with no window
    pressure to justify it. So a small loop on a large-window model keeps every
    tool result verbatim and the prefix cache stays warm.
    """
    from external_llm.agent.agent_turn_pipeline import _evict_for_loop

    # A handful of turns on a 200K-window model — nowhere near 0.75 × cap.
    msgs: list[LLMMessage] = []
    for turn in range(15):
        a, t = _pair(turn, f"SMALL LOOP TURN {turn} DATA " * 200)
        msgs += [a, t]
    out = _evict_for_loop(msgs, model="claude-haiku-4-5")

    stubbed = sum(
        1 for m in out
        if getattr(m, "role", "") == "tool"
        and isinstance(getattr(m, "content", ""), str)
        and m.content.startswith(_EVICTED_MARKER)
    )
    assert stubbed == 0, (
        f"Below-occupancy loop evicted {stubbed} results — a self-inflicted "
        "cache-miss. Eviction must fire only as the prompt nears the cap."
    )


def test_above_occupancy_fires_eviction(monkeypatch):
    """Occupancy gate (mechanism regression): WHEN eviction is enabled and the
    estimated prompt exceeds the trigger fraction of a SMALL window,
    ``_evict_for_loop`` fires and stubs older tool results.

    Eviction is disabled by default (see ``_EVICTION_ENABLED``). This test
    temporarily re-enables it to prove the occupancy gate + keep_recent floor
    still behave correctly, so flipping the flag back on is the ONLY change
    needed to restore gentle eviction.
    """
    import external_llm.agent.agent_turn_pipeline as atp

    monkeypatch.setattr(atp, "_EVICTION_ENABLED", True)
    _EVICTION_KEEP_RECENT = atp._EVICTION_KEEP_RECENT

    # Big enough tool payloads to cross 0.75 × cap on a 64K window
    # (deepseek-r1 cap ≈ 59.9K → trigger ≈ 44.9K; this builds ~85K est tokens).
    msgs: list[LLMMessage] = []
    for turn in range(40):
        a, t = _pair(turn, f"BIG TURN {turn} " + ("PAYLOAD " * 800))
        msgs += [a, t]
    out = atp._evict_for_loop(msgs, model="deepseek-r1")

    stubbed = sum(
        1 for m in out
        if getattr(m, "role", "") == "tool"
        and isinstance(getattr(m, "content", ""), str)
        and m.content.startswith(_EVICTED_MARKER)
    )
    kept = sum(
        1 for m in out
        if getattr(m, "role", "") == "tool"
        and isinstance(getattr(m, "content", ""), str)
        and not m.content.startswith(_EVICTED_MARKER)
    )
    assert stubbed > 0, (
        "Above-occupancy loop evicted nothing — the context bound is not firing "
        "even with _EVICTION_ENABLED=True."
    )
    # The quality floor: the most-recent keep_recent results stay verbatim.
    assert kept >= _EVICTION_KEEP_RECENT, (
        f"Only {kept} verbatim results kept; the recent working set "
        f"(>= {_EVICTION_KEEP_RECENT}) must never be stubbed."
    )


def test_occupancy_trigger_config():
    """Pin the occupancy design's knobs and its default-off master switch:
    a fixed model-independent quality floor (keep_recent), a fractional trigger
    strictly inside (0, 1), and eviction disabled by default."""
    from external_llm.agent.agent_turn_pipeline import (
        _EVICTION_KEEP_RECENT,
        _EVICTION_OCCUPANCY_TRIGGER,
        _EVICTION_ENABLED,
    )
    assert _EVICTION_KEEP_RECENT == 6
    assert _EVICTION_ENABLED is False, (
        "Occupancy-gated eviction must be disabled by default; it mints "
        "self-inflicted cache-miss spikes. Flip _EVICTION_ENABLED to re-enable."
    )
    assert 0.0 < _EVICTION_OCCUPANCY_TRIGGER < 1.0, (
        "Trigger must be a fraction of the cap; a value >= 1 defeats the point "
        "(eviction would never preempt the hard-cap front-trim)."
    )


def test_eviction_disabled_by_default_does_not_fire():
    """Default behaviour: occupancy-gated eviction is OFF. A prompt that far
    exceeds the trigger fraction of a SMALL window is still left untouched by
    ``_evict_for_loop`` — only the hard-cap front-trim (a later, cruder,
    overflow-only cut) bounds the window, so a routine loop never pays a prefix
    rewrite.
    """
    from external_llm.agent.agent_turn_pipeline import _evict_for_loop, _EVICTION_ENABLED

    assert _EVICTION_ENABLED is False
    # Same oversized payload as the enabled-flag regression above.
    msgs: list[LLMMessage] = []
    for turn in range(40):
        a, t = _pair(turn, f"BIG TURN {turn} " + ("PAYLOAD " * 800))
        msgs += [a, t]
    out = _evict_for_loop(msgs, model="deepseek-r1")

    stubbed = sum(
        1 for m in out
        if getattr(m, "role", "") == "tool"
        and isinstance(getattr(m, "content", ""), str)
        and m.content.startswith(_EVICTED_MARKER)
    )
    assert stubbed == 0, (
        f"Default-off eviction fired {stubbed} stubs above occupancy — the "
        "gentle gate must stay disabled; only the hard-cap front-trim bounds the window."
    )


def test_production_eviction_paths_cannot_diverge():
    """Structural invariant: every production tool-loop applies the SAME trigger.

    Regression for a divergence bug: the design-chat loop (where the eviction
    cost-symptom was originally observed) was hand-coding its own eviction
    numbers while MAIN_AGENT used a different tuned path — so a fix to one never
    reached the other.

    The fix introduced a single-source-of-truth wrapper ``_evict_for_loop``.
    The wrapper accepts only model *identity* (``model``) and the ``tool_schemas``
    sent alongside the prompt — never a raw threshold — so the firing decision
    (now an occupancy gate) lives in exactly ONE place. This test pins that BOTH
    production call sites route through it AND that neither passes a raw eviction
    knob, so they can never drift apart again.
    """
    import inspect

    from external_llm.agent import agent_turn_pipeline
    from external_llm.agent import design_chat_loop

    # (1) The wrapper takes only IDENTITY params — never a raw eviction knob.
    #     Passing ``model``/``tool_schemas`` cannot make two call sites diverge
    #     (the trigger is derived centrally from those); passing
    #     ``keep_recent``/``batch_evict_threshold``/``occupancy`` could.
    sig = inspect.signature(agent_turn_pipeline._evict_for_loop)
    _allowed = {"messages", "model", "tool_schemas"}
    _forbidden = {"keep_recent", "batch_evict_threshold", "occupancy", "trigger"}
    params = set(sig.parameters)
    assert params <= _allowed, (
        "_evict_for_loop may take ONLY identity params (messages/model/"
        "tool_schemas); a raw eviction knob would let call sites diverge again. "
        "Found: " + str(sig.parameters)
    )
    assert not (params & _forbidden), (
        "_evict_for_loop must NOT expose an eviction knob — the trigger must stay "
        "derived inside the wrapper. Found: " + str(sig.parameters)
    )

    # (2) Behavioural: a small loop on a large-window model is below the
    #     occupancy trigger, so the wrapper evicts nothing (prefix stays warm).
    msgs: list[LLMMessage] = []
    for turn in range(15):
        a, t = _pair(turn, f"WRAPPER TURN {turn} DATA " * 200)
        msgs += [a, t]
    out = agent_turn_pipeline._evict_for_loop(msgs, model="claude-haiku-4-5")
    stubbed = sum(
        1 for m in out
        if getattr(m, "role", "") == "tool"
        and isinstance(getattr(m, "content", ""), str)
        and m.content.startswith(_EVICTED_MARKER)
    )
    assert stubbed == 0, (
        f"_evict_for_loop evicted {stubbed} below the occupancy trigger — "
        "production paths would mint self-inflicted cache-misses."
    )

    # (3) Structural: both production modules route through the wrapper, and
    #     neither hand-codes a raw eviction knob at the call site.
    src_main = inspect.getsource(agent_turn_pipeline)
    src_chat = inspect.getsource(design_chat_loop)
    assert "ctx.messages = _evict_for_loop(" in src_main, (
        "MAIN_AGENT must route eviction through _evict_for_loop."
    )
    assert "_evict_for_loop" in dir(design_chat_loop), (
        "design-chat loop must import + use _evict_for_loop."
    )
    for _src, _name in ((src_chat, "design-chat loop"), (src_main, "MAIN_AGENT")):
        _before_def = _src.split("def _evict_for_loop")[0]
        assert "batch_evict_threshold=" not in _before_def, (
            f"{_name} must not hand-code a raw eviction knob before the wrapper def."
        )


# ── BUG-2: Anthropic parallel tool_results must get PER-BLOCK stubs ────────
# Anthropic batches N parallel tool calls' results into ONE role="user" message
# with N tool_result blocks. The old code gave every block the SAME stub (wrong
# name "tool" + aggregated size). The fix builds tool_use_id → name from the
# preceding assistant tool_use blocks and stubs each block with its OWN size.
def test_anthropic_parallel_results_get_per_block_stubs():
    """Two parallel Anthropic tool_results in one user message: each block keeps
    its own tool_use_id and gets a per-block stub naming the correct tool."""
    # Preceding assistant with two parallel tool_use blocks (names known here).
    asst = LLMMessage(
        role="assistant", content="",
        raw_content=[
            {"type": "text", "text": "I'll run two tools in parallel."},
            {"type": "tool_use", "id": "tu_1", "name": "read_symbol"},
            {"type": "tool_use", "id": "tu_2", "name": "grep"},
        ],
    )
    # One user message carrying BOTH results (the parallel-batch shape).
    big = "X" * 600
    t = LLMMessage(
        role="user", content="",
        raw_content=[
            {"type": "tool_result", "tool_use_id": "tu_1", "content": big},
            {"type": "tool_result", "tool_use_id": "tu_2", "content": big},
        ],
    )
    msgs = [asst, t]
    # Pad with 6 standard fresh results so the parallel batch falls outside the
    # keep_recent=6 window and gets stubbed.
    for i in range(1, 7):
        msgs += list(_pair(i, "FRESH " * 200))

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    # The parallel-batch message must have been stubbed (raw_content shrunk, not
    # nulled — native shape is preserved).
    stubbed_msgs = [
        m for m in out
        if getattr(m, "role", "") == "user"
        and isinstance(getattr(m, "raw_content", None), list)
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            and (b.get("content") or "").startswith(_EVICTED_MARKER)
            for b in m.raw_content
        )
    ]
    assert len(stubbed_msgs) == 1, "parallel batch must be stubbed exactly once"
    blocks = [b for b in stubbed_msgs[0].raw_content if b.get("type") == "tool_result"]
    assert len(blocks) == 2, "both tool_result blocks preserved"

    # BUG-2 core: each block names the CORRECT tool via the id→name map.
    assert "read_symbol" in blocks[0]["content"], \
        "block tu_1 stub must name read_symbol (recovered from tool_use map)"
    assert "grep" in blocks[1]["content"], \
        "block tu_2 stub must name grep (recovered from tool_use map)"
    # tool_use_id pairing preserved so the API does not reject the request.
    assert blocks[0]["tool_use_id"] == "tu_1"
    assert blocks[1]["tool_use_id"] == "tu_2"
    # Per-block size (not aggregated 1200): each claims ~600.
    assert "600 chars" in blocks[0]["content"], \
        "per-block stub must report the block's own size, not the aggregated size"


# ── GAP-3: Gemini functionResponse results must be stubbed (not nulled) ────
# Gemini carries tool results as role="user" with functionResponse parts. Without
# a Gemini-aware stub handler, _stub_tool_result would clear raw_content (standard
# path), destroying the part structure the Gemini API expects. The fix stubs each
# functionResponse part's inner response.content in place.
def test_gemini_function_response_is_stubbed_in_place():
    """A Gemini tool result (role=user + functionResponse parts) must be stubbed
    by rewriting each part's inner response.content, NOT by clearing raw_content."""
    asst = LLMMessage(
        role="assistant", content="",
        raw_content=[
            {"text": "Calling tool."},
            {"functionCall": {"name": "read_file", "args": {}}},
        ],
    )
    big = "Y" * 600
    t = LLMMessage(
        role="user", content="",
        raw_content=[
            {"functionResponse": {
                "name": "read_file",
                "response": {"content": big},
            }},
        ],
    )
    msgs = [asst, t]
    for i in range(1, 7):
        msgs += list(_pair(i, "FRESH " * 200))

    out = _evict_consumed_tool_results(msgs, keep_recent=6)

    stubbed_msgs = [
        m for m in out
        if getattr(m, "role", "") == "user"
        and isinstance(getattr(m, "raw_content", None), list)
        and any(
            isinstance(b, dict) and "functionResponse" in b
            and ((b["functionResponse"] or {}).get("response", {}) or {})
                .get("content", "").startswith(_EVICTED_MARKER)
            for b in m.raw_content
        )
    ]
    assert len(stubbed_msgs) == 1, "Gemini functionResponse must be stubbed"
    fr = stubbed_msgs[0].raw_content[0]["functionResponse"]
    assert fr["name"] == "read_file", "functionResponse name preserved"
    assert fr["response"]["content"].startswith(_EVICTED_MARKER), \
        "inner response.content replaced with stub"
    assert "read_file" in fr["response"]["content"], \
        "Gemini stub names the tool (from the part's own name)"
    assert "600 chars" in fr["response"]["content"], \
        "per-part size reported"
