"""Contract tests for the wire-block token registry.

Seals the wire-drift bug class: when a provider emits a NEW content-block type,
the token estimator must either (a) count it correctly via the registry or
(b) count it wholesale via the fail-safe — never silently drop it to ~0 tokens.
A silent under-count is what causes the context-overflow 400s the budget
subsystem exists to prevent (see ``_estimate_single_message_tokens``).
"""

from __future__ import annotations

import json

from external_llm.agent._shared_utils import (
    CANONICAL_WIRE_BLOCK_TYPES,
    _WIRE_BLOCK_TOKENIZERS,
    _WIRE_CONTENT_KEY_MARKERS,
    _count_block_wholesale,
    _estimate_single_message_tokens,
    _warn_unknown_block_type,
    estimate_tokens_from_msgs,
    get_unknown_block_type_counts,
    reset_unknown_block_type_counts,
)
from external_llm.client import LLMMessage


def _msg_with_raw_content(blocks: list) -> LLMMessage:
    return LLMMessage(role="assistant", content="", raw_content=blocks)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Registry completeness contract
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistryContract:
    def test_canonical_types_are_all_registered(self):
        """Every canonical wire block type has a tokenizer (text is the lone
        exception — counted by the generic text pre-pass)."""
        unregistered = CANONICAL_WIRE_BLOCK_TYPES - set(_WIRE_BLOCK_TOKENIZERS) - {"text"}
        assert unregistered == set(), f"Canonical types missing tokenizers: {unregistered}"

    def test_registry_keys_are_subset_of_canonical(self):
        """No phantom registry entries without canonical-set membership."""
        extra = set(_WIRE_BLOCK_TOKENIZERS) - CANONICAL_WIRE_BLOCK_TYPES
        assert extra == set(), f"Registered types not in canonical set: {extra}"

    def test_every_tokenizer_is_callable(self):
        for name, fn in _WIRE_BLOCK_TOKENIZERS.items():
            assert callable(fn), f"Tokenizer for {name!r} is not callable"

    def test_gemini_content_key_markers_have_tokenizers(self):
        assert set(_WIRE_CONTENT_KEY_MARKERS) <= set(_WIRE_BLOCK_TOKENIZERS)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Known block types are counted (not zero)
# ══════════════════════════════════════════════════════════════════════════════

class TestKnownBlockTypesCounted:
    def test_tool_use_block_counted(self):
        msg = _msg_with_raw_content([
            {"type": "tool_use", "id": "t1", "name": "bash",
             "input": {"command": "ls -la /usr/bin"}}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_tool_result_string_content_counted(self):
        msg = _msg_with_raw_content([
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "total 4096\ndrwxr-xr-x bin"}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_tool_result_list_content_counted(self):
        msg = _msg_with_raw_content([
            {"type": "tool_result",
             "content": [{"type": "text", "text": "output line one"}]}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_thinking_block_counted(self):
        msg = _msg_with_raw_content([
            {"type": "thinking", "thinking": "Let me analyse the request step by step."}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_redacted_thinking_block_counted(self):
        msg = _msg_with_raw_content([
            {"type": "redacted_thinking", "data": "opaque-signature-payload-data"}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_function_call_typed_counted(self):
        msg = _msg_with_raw_content([
            {"type": "functionCall", "functionCall": {"name": "search", "args": {"q": "tokyo"}}}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_function_response_typed_counted(self):
        msg = _msg_with_raw_content([
            {"type": "functionResponse",
             "functionResponse": {"name": "search", "content": {"result": "found"}}}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_gemini_key_inferred_function_call_counted(self):
        """Gemini parts may omit ``type`` and carry it as a top-level key."""
        msg = _msg_with_raw_content([
            {"functionCall": {"name": "search", "args": {"q": "paris"}}}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_gemini_key_inferred_function_response_counted(self):
        msg = _msg_with_raw_content([
            {"functionResponse": {"name": "search", "content": {"result": "none"}}}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_plain_text_block_counted(self):
        msg = _msg_with_raw_content([
            {"type": "text", "text": "Running analysis on the dataset."}
        ])
        assert _estimate_single_message_tokens(msg) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Fail-safe: unknown block types are NEVER silently zero (the core invariant)
# ══════════════════════════════════════════════════════════════════════════════

class TestFailSafeUnknownTypes:
    def test_unknown_block_counted_wholesale_not_zero(self):
        """The seal: a brand-new provider block type must not vanish."""
        unknown = {"type": "future_citations_block",
                   "citations": ["doc1", "doc2", "doc3"],
                   "metadata": {"k": "v" * 50}}
        assert _count_block_wholesale(unknown) > 0

    def test_unknown_block_via_message_estimator_not_zero(self):
        msg = _msg_with_raw_content([
            {"type": "future_reasoning_signature", "signature": "x" * 200}
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_wholesale_approximates_json_size(self):
        block = {"type": "new", "payload": "abcdefghij"}  # ~10-char payload
        dumped = json.dumps(block, ensure_ascii=False)
        # wholesale count is chars//3+1 → at least a quarter of the json length
        assert _count_block_wholesale(block) >= len(dumped) // 4

    def test_unknown_type_not_in_canonical_set(self):
        assert "future_citations_block" not in CANONICAL_WIRE_BLOCK_TYPES

    def test_non_string_type_field_does_not_crash(self):
        """Client-supplied raw_content may carry a malformed (unhashable) type
        field; the pre-registry ``==`` chain tolerated it, so must dispatch."""
        msg = _msg_with_raw_content([
            {"type": ["weird"], "text": "payload"},
            {"type": {"nested": True}, "data": "x" * 30},
        ])
        assert _estimate_single_message_tokens(msg) > 0

    def test_gemini_untyped_part_counted_not_zero(self):
        """Gemini ``parts`` carry the type as a TOP-LEVEL KEY, not a ``type``
        field.  A part whose only key is ``inlineData`` (base64 image), or
        ``fileData``, or ``executableCode`` matches no registry tokenizer AND
        has no ``text`` field AND matches no content-key marker — so without
        the wholesale fail-safe it falls through to 0 tokens.  This is the
        exact wire-drift class the registry exists to prevent (a new Gemini
        part shape silently under-counting toward a context-overflow 400).

        Seals the branch at ``_estimate_single_message_tokens`` where
        ``btype is None and not text and not any(marker)`` must route to
        ``_count_block_wholesale``.  If a future refactor collapses that
        branch into the plain ``btype in (None, '', 'text')`` skip, this test
        fails — preventing silent regression.
        """
        for part in [
            {"inlineData": {"mimeType": "image/png", "data": "iVBOR" * 40}},
            {"fileData": {"mimeType": "image/png", "fileUri": "gs://b/o.png"}},
            {"executableCode": {"language": "PYTHON", "code": "print(1)" * 20}},
        ]:
            msg = _msg_with_raw_content([part])
            est = _estimate_single_message_tokens(msg)
            assert est > 0, f"untyped Gemini part counted as 0 tokens: {part}"

    def test_gemini_untyped_part_with_text_does_not_double_count(self):
        """A Gemini part that has BOTH a ``text`` field AND a top-level key
        (e.g. some providers nest text inside a part that also carries
        metadata) is counted via the text pre-pass, then reaches the
        untyped branch with ``text`` truthy — it must NOT also be wholesale
        counted (that would double-count the text)."""
        part = {"text": "hello world", "inlineData": {"data": "x"}}
        est = _estimate_single_message_tokens(_msg_with_raw_content([part]))
        # text alone → ~4 tokens; inlineData alone (wholesale) → ~6 tokens.
        # Combined (text counted, inlineData NOT wholesale-counted) → ~4.
        assert est < 15, f"untyped part with text was double-counted: {est}"

    def test_image_block_uses_flat_estimate_not_wholesale(self):
        """A base64 image must be charged at the provider cap (~1.6k), not by
        payload length — a 300 KB screenshot is ~1.6k real tokens, not ~130k."""
        import base64
        fake = base64.b64encode(b"\x00" * 300_000).decode()
        msg = _msg_with_raw_content([
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": fake}}
        ])
        est = _estimate_single_message_tokens(msg)
        assert 0 < est <= 2000, f"image over-counted: {est}"

    def test_known_type_does_not_trip_fail_safe(self):
        """A known type must dispatch through the registry, not the wholesale
        fallback, so its count reflects the real payload (not the whole block)."""
        # tool_use with a small input: registry counts input; wholesale would
        # count the whole block (larger). Registry count must be strictly less.
        msg = _msg_with_raw_content([
            {"type": "tool_use", "id": "x", "name": "n", "input": {"a": 1}}
        ])
        registry_count = _estimate_single_message_tokens(msg)
        wholesale = _count_block_wholesale(
            {"type": "tool_use", "id": "x", "name": "n", "input": {"a": 1}}
        )
        assert registry_count < wholesale


# ══════════════════════════════════════════════════════════════════════════════
# 4. Regression: end-to-end via estimate_tokens_from_msgs
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndEstimation:
    def test_mixed_blocks_all_counted(self):
        msg = _msg_with_raw_content([
            {"type": "text", "text": "Running analysis."},
            {"type": "tool_use", "name": "bash", "input": {"command": "echo hi"}},
            {"type": "tool_result", "content": "hi"},
            {"type": "thinking", "thinking": "step one"},
        ])
        text_only = _msg_with_raw_content([{"type": "text", "text": "Running analysis."}])
        assert _estimate_single_message_tokens(msg) > _estimate_single_message_tokens(text_only)

    def test_estimate_tokens_from_msgs_sums_multiple(self):
        m1 = _msg_with_raw_content([{"type": "text", "text": "alpha beta gamma"}])
        m2 = _msg_with_raw_content([{"type": "thinking", "thinking": "delta epsilon"}])
        assert estimate_tokens_from_msgs([m1, m2]) > _estimate_single_message_tokens(m1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Wire-drift counter contract (get_unknown_block_type_counts)
# ══════════════════════════════════════════════════════════════════════════════

class TestWireDriftCounter:
    """Verifies the public API of the unknown-block-type occurrence counter.

    The counter feeds ``/stats/wire-drift`` and must satisfy:
      - each occurrence increments the count for that type
      - the returned snapshot is an isolated copy (mutation-safe)
      - the wholesale fallback path in ``_estimate_single_message_tokens``
        actually records occurrences (end-to-end integration).

    Uses unique per-type names so assertions are robust against pre-existing
    counter entries left by other tests (the counter is module-level mutable
    state shared across the test class).
    """

    def test_single_occurrence_recorded(self):
        _uid = f"__test_single_{id(self)}__"
        before = get_unknown_block_type_counts().get(_uid, 0)
        _warn_unknown_block_type(_uid)
        assert get_unknown_block_type_counts().get(_uid, 0) == before + 1

    def test_multiple_occurrences_accumulate(self):
        _uid = f"__test_multi_{id(self)}__"
        before = get_unknown_block_type_counts().get(_uid, 0)
        for _ in range(3):
            _warn_unknown_block_type(_uid)
        assert get_unknown_block_type_counts().get(_uid, 0) == before + 3

    def test_returned_dict_is_snapshot(self):
        """Assert that mutating the returned dict does NOT affect internal state."""
        before = get_unknown_block_type_counts()
        # Wild mutation that should be invisible to the real counter.
        before["__test_ghost__"] = 999
        after = get_unknown_block_type_counts()
        assert "__test_ghost__" not in after

    def test_wholesale_fallback_records_counter(self):
        """End-to-end: an unknown block type fed through the message estimator
        must appear in ``get_unknown_block_type_counts()``."""
        _uid = f"__test_end_to_end_{id(self)}__"
        before = get_unknown_block_type_counts().get(_uid, 0)
        msg = _msg_with_raw_content([{"type": _uid, "payload": "x" * 30}])
        assert _estimate_single_message_tokens(msg) > 0
        assert get_unknown_block_type_counts().get(_uid, 0) == before + 1

    def test_reset_returns_presets_snapshot_and_clears(self):
        """reset_unknown_block_type_counts() returns the pre-reset counts and
        then zeroes the counter for the affected type.

        Uses a unique type name (robust against pre-existing entries shared by
        the module-level counter) and asserts the post-reset count is 0 — the
        contract the ``/stats/wire-drift?reset=1`` monitoring baseline relies on.
        """
        _uid = f"__test_reset_{id(self)}__"
        for _ in range(2):
            _warn_unknown_block_type(_uid)
        assert get_unknown_block_type_counts().get(_uid, 0) >= 2
        snapshot = reset_unknown_block_type_counts()
        # Pre-reset snapshot still reflects the accumulated count.
        assert snapshot.get(_uid, 0) >= 2
        # After reset the live counter no longer carries this type.
        assert get_unknown_block_type_counts().get(_uid, 0) == 0
        # A fresh occurrence after reset restarts at 1 (new observation window).
        _warn_unknown_block_type(_uid)
        assert get_unknown_block_type_counts().get(_uid, 0) == 1
