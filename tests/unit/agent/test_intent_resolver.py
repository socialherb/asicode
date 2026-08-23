"""Unit tests for intent_resolver.py — RED→GREEN (IR-B1) + full coverage.

IR-B1 (real bug): ``_build_intent_result`` calls ``Scope(scope_hint)``
unconditionally.  Any non-canonical LLM output — ``"project-wide"``
(hyphenated), ``"SINGLE_FILE"`` (enum-name form), a non-string — raises
``ValueError``, which the outer ``except Exception`` in ``_resolve_with_llm``
silently swallows.  The ENTIRE successful LLM resolution then collapses to the
minimal fallback (confidence 0.1, intent unknown, all search terms lost).
``_parse_llm_response`` validates intent_type/lane_hint but not scope_hint —
asymmetric validation leaves this hole open.

Fix: defensive ``try/except ValueError`` around ``Scope(...)`` → default
``Scope.SINGLE_FILE`` while preserving every other LLM-extracted field.

Test map:
  - create_intent_resolver factory / config validation
  - resolve(): empty request, strip, cache hit/TTL/LRU/disabled
  - _resolve_with_llm(): chat & chat_with_tools paths, error taxonomy,
    truncation retry (finish_reason + structural), reasoning_content fallback
  - _parse_llm_response(): JSON extraction, recovery, validation/clamping
  - _build_intent_result(): role classification, new_files, edit_kind/guard,
    code_concepts, scope/complexity mapping, boolean flags, IR-B1 variants
  - _recover_truncated_json(): string/escape/bracket-aware comma scanning
  - _fallback_extraction(): words, stop words, file patterns, lane hints
"""

from __future__ import annotations

import json
import logging

import pytest

import external_llm.agent.intent_resolver as ir_module
from external_llm.agent.enums import Complexity, Scope
from external_llm.agent.intent_models import IntentResolutionConfig
from external_llm.agent.intent_resolver import IntentResolver, create_intent_resolver
from external_llm.client import LLMServerUnavailableError

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_VALID_JSON = json.dumps(
    {
        "intent_type": "bugfix",
        "lane_hint": "planner",
        "scope_hint": "single_file",
        "complexity_hint": "normal",
        "is_test_write": False,
        "is_style_fix": False,
        "is_filesystem_op": False,
        "is_ui_change": False,
        "is_interface_preserving": False,
        "modify_symbols": ["ConnectionPool.release"],
        "new_symbols": [],
        "reference_symbols": [],
        "search_terms": ["ConnectionPool", "release"],
        "confidence": 0.9,
        "metadata": {"language_detected": "en"},
        "normalized_query": "fix bug in ConnectionPool.release",
        "code_concepts": {"data_fields": [], "behavioral_kind": "fix", "scope_phase": "execution"},
    }
)


class FakeResponse:
    """Minimal stand-in for the client's response object."""

    def __init__(self, content="", finish_reason=None, raw_response=None):
        self.content = content
        self.finish_reason = finish_reason
        self.raw_response = raw_response or {}


class FakeClient:
    """Chat-only LLM client recording calls; pops canned responses."""

    def __init__(self, responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def chat(self, messages, model, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        if not self.responses:
            return None
        return self.responses.pop(0)


class ToolsOnlyClient:
    """Client exposing only ``chat_with_tools`` (no ``chat`` attr)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_with_tools(self, messages, tools, model, temperature, max_tokens):
        self.calls.append({"max_tokens": max_tokens, "tools": tools})
        return self.responses.pop(0)


class FailingRetryClient(FakeClient):
    """First call returns truncated content; retry raises a generic error."""

    def chat(self, messages, model, temperature, max_tokens):
        self.calls.append({"max_tokens": max_tokens})
        if len(self.calls) == 2:
            raise RuntimeError("retry failed")
        return self.responses.pop(0)


def make_resolver(client=None, enable_cache=False, **kw) -> IntentResolver:
    cfg = IntentResolutionConfig(
        llm_client=client,
        model="test-model",
        enable_cache=enable_cache,
        **kw,
    )
    return IntentResolver(cfg)


def base_dict(**overrides) -> dict:
    d = {
        "normalized_query": "fix the bug",
        "search_terms": ["ConnectionPool", "release"],
        "intent_type": "bugfix",
        "lane_hint": "planner",
        "scope_hint": "single_file",
        "complexity_hint": "normal",
        "is_test_write": False,
        "is_style_fix": False,
        "is_filesystem_op": False,
        "is_ui_change": False,
        "is_interface_preserving": False,
        "modify_symbols": ["ConnectionPool.release"],
        "reference_symbols": [],
        "new_symbols": [],
        "target_files": [],
        "confidence": 0.9,
        "metadata": {"language_detected": "en"},
        "edit_kind": "",
        "guard_statement": "",
        "code_concepts": {},
        "new_files": [],
    }
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Factory / __init__
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryAndInit:
    def test_create_factory_defaults(self):
        client = FakeClient([])
        r = create_intent_resolver(client, "model-x", enable_cache=True)
        assert r.config.enable_cache is True
        assert r.config.cache_ttl_seconds == 300
        assert r.config.max_search_terms == 10
        assert r.config.max_target_files == 5
        assert r._model == "model-x"
        assert r._llm_client is client

    def test_create_factory_cache_default_on(self):
        r = create_intent_resolver(None, "model-x")
        assert r.config.enable_cache is True

    def test_create_factory_no_model_raises(self):
        with pytest.raises(ValueError, match="requires a model"):
            create_intent_resolver(None, "")

    def test_init_without_client_ok(self):
        r = make_resolver(None)
        assert r._llm_client is None
        assert r._cache_max == 128


# ═══════════════════════════════════════════════════════════════════════════
# resolve() — cache / empty / strip
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveCache:
    def test_resolve_empty_request(self):
        client = FakeClient([])
        resolver = make_resolver(client, enable_cache=True)
        for req in ("", "   ", "\t\n"):
            r = resolver.resolve(req)
            assert r.intent_type == "unknown"
            assert r.confidence == 0.0
            assert r.metadata == {"source": "empty_request"}
            assert r.original_request == ""
        assert client.calls == []

    def test_resolve_strips_request(self):
        client = FakeClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")])
        resolver = make_resolver(client, enable_cache=True)
        r = resolver.resolve("   fix ConnectionPool.release   ")
        assert r.original_request == "fix ConnectionPool.release"
        user_prompt = client.calls[0]["messages"][1].content
        assert user_prompt == "User request: fix ConnectionPool.release"

    def test_cache_hit_returns_same_object(self):
        client = FakeClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")])
        resolver = make_resolver(client, enable_cache=True)
        r1 = resolver.resolve("fix ConnectionPool.release")
        r2 = resolver.resolve("fix ConnectionPool.release")
        assert r1 is r2
        assert len(client.calls) == 1

    def test_cache_key_uses_stripped_request(self):
        client = FakeClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")])
        resolver = make_resolver(client, enable_cache=True)
        resolver.resolve("  fix ConnectionPool.release  ")
        resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 1

    def test_cache_ttl_expiry(self, monkeypatch):
        client = FakeClient(
            [
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
            ]
        )
        resolver = make_resolver(client, enable_cache=True, cache_ttl_seconds=300)
        now = {"t": 1000.0}
        monkeypatch.setattr(ir_module.time, "monotonic", lambda: now["t"])
        resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 1
        now["t"] += 299
        resolver.resolve("fix ConnectionPool.release")  # fresh
        assert len(client.calls) == 1
        now["t"] += 2  # 301s elapsed → expired
        resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2

    def test_cache_lru_eviction(self):
        client = FakeClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")] * 6)
        resolver = make_resolver(client, enable_cache=True)
        resolver._cache_max = 3
        for i in range(4):
            resolver.resolve(f"fix ConnectionPool.release issue {i}")
        assert len(client.calls) == 4
        # First entry evicted → must re-resolve
        resolver.resolve("fix ConnectionPool.release issue 0")
        assert len(client.calls) == 5

    def test_cache_disabled_no_caching(self):
        client = FakeClient(
            [
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
            ]
        )
        resolver = make_resolver(client, enable_cache=False)
        resolver.resolve("fix ConnectionPool.release")
        resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_with_llm — client paths / errors / truncation / reasoning fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveWithLlm:
    def test_no_client_falls_back(self):
        resolver = make_resolver(None)
        r = resolver.resolve("fix the ConnectionPool release bug")
        assert r.intent_type == "unknown"
        assert r.confidence == 0.1
        assert r.metadata == {"source": "minimal_fallback"}
        assert "ConnectionPool" in r.search_terms

    def test_chat_path_parses_result(self):
        client = FakeClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"
        assert r.modify_symbols == ["ConnectionPool.release"]
        assert r.confidence == 0.9
        assert client.calls[0]["temperature"] == 0.1

    def test_chat_with_tools_path(self):
        client = ToolsOnlyClient([FakeResponse(content=_VALID_JSON, finish_reason="stop")])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"
        assert client.calls[0]["tools"] == []
        assert client.calls[0]["max_tokens"] == resolver.config.max_tokens

    def test_generic_error_falls_back(self):
        client = FakeClient([], error=RuntimeError("boom"))
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.metadata == {"source": "minimal_fallback"}
        assert r.intent_type == "unknown"

    def test_parse_stage_exception_falls_back(self, monkeypatch, caplog):
        # Coverage gap: the outer ``except Exception`` — the LLM call itself
        # succeeded, but the parse stage raised a non-LLM exception.  Must log
        # and fall back, never propagate (LLMServerUnavailableError is the
        # only exception that is re-raised).
        resolver = make_resolver(FakeClient([FakeResponse(content=_VALID_JSON)]))

        def _boom(raw, req):
            raise RuntimeError("parse crash")

        monkeypatch.setattr(resolver, "_parse_llm_response", _boom)
        with caplog.at_level(logging.ERROR, logger="external_llm.agent.intent_resolver"):
            r = resolver.resolve("fix ConnectionPool.release")
        assert r.metadata == {"source": "minimal_fallback"}
        assert r.intent_type == "unknown"
        assert "LLM resolution failed" in caplog.text

    def test_llm_unavailable_propagates(self):
        client = FakeClient([], error=LLMServerUnavailableError("server down"))
        resolver = make_resolver(client)
        with pytest.raises(LLMServerUnavailableError):
            resolver.resolve("fix ConnectionPool.release")

    def test_none_response_falls_back(self):
        client = FakeClient([None])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.metadata == {"source": "minimal_fallback"}

    def test_client_without_any_call_method_falls_back(self):
        class NoMethodClient:
            pass

        resolver = make_resolver(NoMethodClient())
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.metadata == {"source": "minimal_fallback"}

    def test_finish_reason_from_raw_response_dict(self):
        # Client wrapper without .finish_reason attr; reason lives in raw_response.
        resp = FakeResponse(
            content=_VALID_JSON,
            raw_response={"choices": [{"finish_reason": "stop"}]},
        )
        del resp.finish_reason
        client = FakeClient([resp])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 1  # "stop" → no retry
        assert r.intent_type == "bugfix"

    def test_str_response_parsed(self):
        client = FakeClient([_VALID_JSON])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"
        assert r.search_terms == ["ConnectionPool", "release"]

    def test_reasoning_content_fallback(self):
        raw_resp = {"choices": [{"message": {"content": "", "reasoning_content": _VALID_JSON}}]}
        client = FakeClient([FakeResponse(content="", finish_reason="stop", raw_response=raw_resp)])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"
        assert r.modify_symbols == ["ConnectionPool.release"]

    def test_truncation_retry_doubles_budget(self):
        truncated = (
            '{"intent_type": "bugfix", "search_terms": ["ConnectionPool", "release"], "modify_symbols": ["Connection'
        )
        client = FakeClient(
            [
                FakeResponse(content=truncated, finish_reason="length"),
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
            ]
        )
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2
        assert client.calls[0]["max_tokens"] == resolver.config.max_tokens
        assert client.calls[1]["max_tokens"] == resolver.config.max_tokens * 2
        assert r.intent_type == "bugfix"
        assert r.modify_symbols == ["ConnectionPool.release"]

    def test_finish_reason_truncated_retries(self):
        client = FakeClient(
            [
                FakeResponse(content='{"intent_type": "feature"}', finish_reason="truncated"),
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
            ]
        )
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2
        assert r.intent_type == "bugfix"  # retry content wins

    def test_json_looks_truncated_retries(self):
        client = FakeClient(
            [
                FakeResponse(content='{"intent_type": "feature"', finish_reason="stop"),
                FakeResponse(content=_VALID_JSON, finish_reason="stop"),
            ]
        )
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2
        assert r.intent_type == "bugfix"

    def test_retry_failure_uses_partial_content(self):
        # Partial content ends with a complete top-level field so the JSON
        # recovery can cut at the last top-level comma boundary.
        truncated = '{"intent_type": "feature", "nested": {"x": 1}, "search_terms": ["alpha", "be'
        client = FailingRetryClient([FakeResponse(content=truncated, finish_reason="length")])
        resolver = make_resolver(client)
        r = resolver.resolve("add alpha beta support")
        assert len(client.calls) == 2
        assert r.intent_type == "feature"
        assert r.search_terms == []  # truncated mid-array → field dropped

    def test_retry_empty_content_keeps_original(self):
        truncated = '{"intent_type": "feature", "nested": {"x": 1}, "search_terms": ["alpha"]'
        client = FakeClient(
            [
                FakeResponse(content=truncated, finish_reason="length"),
                FakeResponse(content="", finish_reason="stop"),  # empty retry
            ]
        )
        resolver = make_resolver(client)
        r = resolver.resolve("add alpha support")
        assert len(client.calls) == 2
        assert r.intent_type == "feature"  # original partial recovered
        assert r.search_terms == []

    def test_retry_plain_str_response(self):
        truncated = '{"intent_type": "feature", "search_terms": ["alpha"]'
        client = FakeClient(
            [
                FakeResponse(content=truncated, finish_reason="length"),
                _VALID_JSON,  # retry returns a plain string
            ]
        )
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        assert len(client.calls) == 2
        assert r.intent_type == "bugfix"

    def test_retry_unavailable_propagates(self):
        class UnavailableRetryClient(FakeClient):
            def chat(self, messages, model, temperature, max_tokens):
                self.calls.append({"max_tokens": max_tokens})
                if len(self.calls) == 2:
                    raise LLMServerUnavailableError("still down")
                return self.responses.pop(0)

        truncated = '{"intent_type": "feature"'
        client = UnavailableRetryClient([FakeResponse(content=truncated, finish_reason="length")])
        resolver = make_resolver(client)
        with pytest.raises(LLMServerUnavailableError):
            resolver.resolve("fix ConnectionPool.release")


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════


class TestPrompts:
    def test_system_prompt_contains_guides(self):
        p = make_resolver(None)._build_system_prompt()
        assert "intent_type" in p
        assert "lane_hint" in p
        assert "scope_hint" in p
        assert "bugfix" in p
        assert "read_only" in p
        assert "guard_add" in p

    def test_user_prompt_format(self):
        assert make_resolver(None)._build_user_prompt("hello") == "User request: hello"


# ═══════════════════════════════════════════════════════════════════════════
# _recover_truncated_json
# ═══════════════════════════════════════════════════════════════════════════


class TestRecoverTruncatedJson:
    def test_not_object_none(self):
        assert make_resolver(None)._recover_truncated_json("[1, 2") is None

    def test_mid_field_recovers_prefix(self):
        r = make_resolver(None)._recover_truncated_json('{"a": 1, "b": 2, "c": "trunc')
        assert r == {"a": 1, "b": 2}

    def test_comma_inside_string_ignored(self):
        r = make_resolver(None)._recover_truncated_json('{"a": "x,y", "b":')
        assert r == {"a": "x,y"}

    def test_escaped_quotes_in_string(self):
        r = make_resolver(None)._recover_truncated_json('{"a": "va\\"lue", "b": 1')
        assert r == {"a": 'va"lue'}

    def test_truncation_inside_array_no_recovery(self):
        assert make_resolver(None)._recover_truncated_json('{"a": [1, 2, 3') is None

    def test_nested_object_truncation_partial(self):
        r = make_resolver(None)._recover_truncated_json('{"a": 1, "b": {"x": 1, "y": 2')
        assert r == {"a": 1}

    def test_complete_json_returns_most_fields_prefix(self):
        # Only reached when json.loads already failed (e.g. multi-object text);
        # documents the "drop trailing fields" semantics.
        r = make_resolver(None)._recover_truncated_json('{"a": 1, "b": 2}')
        assert r == {"a": 1}

    def test_invalid_candidate_continues_to_previous(self, monkeypatch):
        # Coverage gap: a candidate closed at a top-level comma is normally
        # always valid (the comma implies a completed field), so the
        # JSONDecodeError → continue path is defensive.  Force the LAST
        # candidate to fail and verify recovery falls back to the previous
        # comma instead of abandoning the whole recovery.
        resolver = make_resolver(None)
        real_loads = ir_module.json.loads
        calls = {"n": 0}

        def flaky_loads(s, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise json.JSONDecodeError("boom", s, 0)
            return real_loads(s, *a, **k)

        monkeypatch.setattr(ir_module.json, "loads", flaky_loads)
        r = resolver._recover_truncated_json('{"a": 1, "b": 2, "c": 3,')
        assert r == {"a": 1, "b": 2}
        assert calls["n"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# _parse_llm_response — extraction / recovery / validation / clamping
# ═══════════════════════════════════════════════════════════════════════════


class TestParseLlmResponse:
    def test_no_brace_fallback(self):
        d = make_resolver(None)._parse_llm_response("no json here", "orig")
        assert d["intent_type"] == "unknown"
        assert d["confidence"] == 0.2
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_close_before_open_fallback(self):
        d = make_resolver(None)._parse_llm_response("} {", "orig")
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_code_fence_wrapper_parsed(self):
        raw = "```json\n" + _VALID_JSON + "\n```"
        d = make_resolver(None)._parse_llm_response(raw, "orig")
        assert d["intent_type"] == "bugfix"

    def test_decode_error_recovery_succeeds(self):
        # A closed nested field provides the '}' needed to pass brace
        # extraction; truncation after the last top-level comma → recovery.
        raw = '{"intent_type": "feature", "search_terms": ["a", "b"], "nested": {"x": 1}, "normalized_query": "trunc'
        d = make_resolver(None)._parse_llm_response(raw, "orig")
        assert d["intent_type"] == "feature"
        assert d["search_terms"] == ["a", "b"]
        assert d["nested"] == {"x": 1}

    def test_decode_error_recovery_drops_truncated_array(self):
        # Truncation inside the array → the whole field is dropped.
        raw = '{"intent_type": "feature", "nested": {"x": 1}, "search_terms": ["a", "b"]'
        d = make_resolver(None)._parse_llm_response(raw, "orig")
        assert d["intent_type"] == "feature"
        assert d.get("search_terms", []) == []

    def test_decode_error_no_recovery_fallback(self):
        d = make_resolver(None)._parse_llm_response('{"intent_type": "feature"', "orig")
        assert d["intent_type"] == "unknown"
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_non_dict_result_fallback(self):
        d = make_resolver(None)._parse_llm_response("[1, 2, 3]", "orig")
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_decode_error_unrecoverable_fallback(self):
        # Coverage gap: a real '}' exists so brace extraction passes, but
        # every comma sits inside an unclosed array → no top-level separator →
        # recovery returns None → fallback dict.  (Complements
        # test_decode_error_no_recovery_fallback, which exercised the
        # no-'}'-at-all early return instead.)
        d = make_resolver(None)._parse_llm_response('{"a": [1, 2}', "orig")
        assert d["intent_type"] == "unknown"
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_non_dict_json_loads_result_fallback(self, monkeypatch):
        # Coverage gap: '{'…'}' brace extraction guarantees a syntactically
        # valid result is always a dict, so this defensive branch is
        # unreachable naturally — force json.loads to return a non-dict.
        resolver = make_resolver(None)
        monkeypatch.setattr(ir_module.json, "loads", lambda *a, **k: [1, 2, 3])
        d = resolver._parse_llm_response('{"intent_type": "bugfix"}', "orig")
        assert d["metadata"] == {"source": "llm_parse_failed"}

    def test_missing_normalized_query_filled(self):
        d = make_resolver(None)._parse_llm_response('{"intent_type": "bugfix"}', "my original request")
        assert d["normalized_query"] == "my original request"

    def test_non_list_list_fields_coerced(self):
        d = make_resolver(None)._parse_llm_response(
            '{"intent_type": "bugfix", "search_terms": "notalist", "modify_symbols": 42, "new_symbols": {"name": "x"}}',
            "orig",
        )
        assert d["search_terms"] == []
        assert d["modify_symbols"] == []
        assert d["new_symbols"] == []

    def test_confidence_clamped_and_defaulted(self):
        p = make_resolver(None)._parse_llm_response
        assert p('{"intent_type": "bugfix", "confidence": 2.5}', "o")["confidence"] == 1.0
        assert p('{"intent_type": "bugfix", "confidence": -3}', "o")["confidence"] == 0.0
        assert p('{"intent_type": "bugfix", "confidence": "abc"}', "o")["confidence"] == 0.5
        assert p('{"intent_type": "bugfix"}', "o")["confidence"] == 0.5

    def test_invalid_intent_type_unknown(self):
        d = make_resolver(None)._parse_llm_response('{"intent_type": "delete_everything"}', "orig")
        assert d["intent_type"] == "unknown"

    def test_invalid_lane_question_read_only(self):
        d = make_resolver(None)._parse_llm_response('{"intent_type": "question", "lane_hint": "bogus_lane"}', "orig")
        assert d["lane_hint"] == "read_only"

    def test_invalid_lane_non_question_planner(self):
        p = make_resolver(None)._parse_llm_response
        assert p('{"intent_type": "bugfix", "lane_hint": "bogus"}', "o")["lane_hint"] == "planner"
        assert p('{"intent_type": "exploration"}', "o")["lane_hint"] == "planner"

    def test_invalid_metadata_defaulted(self):
        d = make_resolver(None)._parse_llm_response('{"intent_type": "bugfix", "metadata": "nope"}', "orig")
        assert d["metadata"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# _build_intent_result — role classification / spec hints / guard / concepts
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildIntentResult:
    def test_basic_role_aware_fields(self):
        r = make_resolver(None)._build_intent_result("req", base_dict())
        assert r.intent_type == "bugfix"
        assert r.modify_symbols == ["ConnectionPool.release"]
        assert r.reference_symbols == []
        assert r.target_symbols == ["ConnectionPool.release"]
        assert r.search_terms == ["ConnectionPool", "release"]
        assert r.confidence == 0.9
        assert r.lane_hint == "planner"
        assert r.scope_hint == Scope.SINGLE_FILE
        assert r.complexity_hint == Complexity.MEDIUM
        assert r.normalized_query == "fix the bug"

    def test_search_terms_capped(self):
        d = base_dict(search_terms=[f"t{i}" for i in range(20)])
        r = make_resolver(None)._build_intent_result("req", d)
        assert len(r.search_terms) == 10

    def test_target_files_filtered(self):
        d = base_dict(target_files=["a.py", "  ", 42, "b/c.py"])
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.target_files == ["a.py", "b/c.py"]

    def test_old_style_target_symbols_backward_compat(self):
        d = base_dict(modify_symbols=[], reference_symbols=[], target_symbols=["Foo", "Bar"])
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.modify_symbols == ["Foo", "Bar"]
        assert r.reference_symbols == []
        assert r.target_symbols == ["Foo", "Bar"]

    def test_role_fields_filter_non_strings(self):
        d = base_dict(modify_symbols=["Good", 42, ""], reference_symbols=["Ref", None, "x"])
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.modify_symbols == ["Good"]
        assert r.reference_symbols == ["Ref", "x"]

    def test_new_files_spec_hints(self):
        d = base_dict(new_files=["src/new_module.py", "tests/test_x.py", "plainword", 42, "noext"])
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.spec_hints == {"new_files": ["src/new_module.py", "tests/test_x.py"]}

    def test_no_new_files_empty_spec_hints(self):
        r = make_resolver(None)._build_intent_result("req", base_dict())
        assert r.spec_hints == {}

    def test_new_symbols_filtered(self):
        d = base_dict(
            new_symbols=[
                {"name": "validate", "kind": "method", "parent": "UserModel"},
                {"name": ""},
                {},
                "junk",
                42,
                {"name": "ok"},
            ]
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert [s["name"] for s in r.new_symbols] == ["validate", "ok"]

    def test_edit_kind_valid_and_normalized(self):
        r = make_resolver(None)._build_intent_result("req", base_dict(edit_kind="Guard_Add"))
        assert r.edit_kind == "guard_add"
        r = make_resolver(None)._build_intent_result("req", base_dict(edit_kind="signature_change"))
        assert r.edit_kind == "signature_change"

    def test_edit_kind_invalid_empty(self):
        r = make_resolver(None)._build_intent_result("req", base_dict(edit_kind="total_rewrite"))
        assert r.edit_kind == ""

    def test_guard_add_valid_builds_guard_spec(self):
        d = base_dict(edit_kind="guard_add", guard_statement="if not candidates: return None")
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.guard_statement == "if not candidates: return None"
        assert r.guard_spec is not None
        assert r.guard_spec.is_parsed
        assert r.guard_spec.control == "return"

    def test_guard_add_syntax_error_discarded(self):
        d = base_dict(edit_kind="guard_add", guard_statement="if if if")
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.guard_statement == ""
        assert r.guard_spec is None

    def test_guard_add_non_guard_statement(self):
        d = base_dict(edit_kind="guard_add", guard_statement="x = 1")
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.guard_statement == "x = 1"  # syntax-valid → kept
        assert r.guard_spec is None  # but not a guard → no IR

    def test_guard_statement_ignored_unless_guard_add(self):
        d = base_dict(edit_kind="body_only", guard_statement="if x: return")
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.guard_statement == ""
        assert r.guard_spec is None

    def test_code_concepts_valid_and_capped(self):
        d = base_dict(
            code_concepts={
                "data_fields": [f"field{i}" for i in range(10)] + ["x", "bad field!"],
                "behavioral_kind": "enforcement",
                "scope_phase": "verification",
            }
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.code_concepts["data_fields"] == [f"field{i}" for i in range(8)]
        assert r.code_concepts["behavioral_kind"] == "enforcement"
        assert r.code_concepts["scope_phase"] == "verification"

    def test_code_concepts_all_invalid_empty(self):
        d = base_dict(code_concepts={"data_fields": [], "behavioral_kind": "delete", "scope_phase": "now"})
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.code_concepts == {}

    def test_code_concepts_partial_valid(self):
        d = base_dict(code_concepts={"data_fields": ["count"], "behavioral_kind": "bogus", "scope_phase": "execution"})
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.code_concepts["data_fields"] == ["count"]
        assert r.code_concepts["behavioral_kind"] == ""
        assert r.code_concepts["scope_phase"] == "execution"

    def test_code_concepts_non_dict_empty(self):
        r = make_resolver(None)._build_intent_result("req", base_dict(code_concepts="junk"))
        assert r.code_concepts == {}

    def test_project_wide_clears_scope_phase(self):
        d = base_dict(
            scope_hint="project_wide",
            code_concepts={
                "data_fields": ["x_field"],
                "behavioral_kind": "fix",
                "scope_phase": "planning",
            },
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.scope_hint == Scope.PROJECT_WIDE
        assert r.code_concepts["scope_phase"] == ""
        assert r.code_concepts["data_fields"] == ["x_field"]

    def test_project_wide_clears_scope_phase_keeps_behavioral_kind(self):
        d = base_dict(
            scope_hint="project_wide",
            code_concepts={
                "data_fields": [],
                "behavioral_kind": "fix",
                "scope_phase": "planning",
            },
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.code_concepts == {"data_fields": [], "behavioral_kind": "fix", "scope_phase": ""}

    def test_project_wide_empties_concepts_without_fields(self):
        d = base_dict(
            scope_hint="project_wide",
            code_concepts={
                "data_fields": [],
                "behavioral_kind": "",
                "scope_phase": "planning",
            },
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.code_concepts == {}

    def test_complexity_mapping(self):
        for hint, expected in [
            ("trivial", Complexity.LOW),
            ("normal", Complexity.MEDIUM),
            ("complex", Complexity.HIGH),
            ("bogus", Complexity.LOW),
            (None, Complexity.LOW),
        ]:
            r = make_resolver(None)._build_intent_result("req", base_dict(complexity_hint=hint))
            assert r.complexity_hint == expected

    def test_boolean_flags(self):
        d = base_dict(
            is_test_write=True,
            is_style_fix=True,
            is_filesystem_op=True,
            is_ui_change=True,
            is_interface_preserving=True,
        )
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.is_test_write and r.is_style_fix and r.is_filesystem_op
        assert r.is_ui_change and r.is_interface_preserving

    def test_metadata_passthrough(self):
        r = make_resolver(None)._build_intent_result("req", base_dict(metadata={"language_detected": "ko"}))
        assert r.metadata == {"language_detected": "ko"}


# ═══════════════════════════════════════════════════════════════════════════
# IR-B1 — invalid scope_hint must NOT collapse the LLM resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestScopeHintRobustness:  # RED → GREEN (IR-B1)
    @pytest.mark.parametrize("bad_scope", ["project-wide", "SINGLE_FILE", "multi", 42])
    def test_resolve_invalid_scope_hint_preserves_llm_result(self, bad_scope):
        payload = json.loads(_VALID_JSON)
        payload["scope_hint"] = bad_scope
        client = FakeClient([FakeResponse(content=json.dumps(payload), finish_reason="stop")])
        resolver = make_resolver(client)
        r = resolver.resolve("fix ConnectionPool.release")
        # Current bug: ValueError from Scope() is swallowed by the outer
        # except → entire result collapses to minimal fallback.
        assert r.intent_type == "bugfix"
        assert r.confidence == 0.9
        assert r.modify_symbols == ["ConnectionPool.release"]
        assert r.search_terms == ["ConnectionPool", "release"]
        assert r.scope_hint == Scope.SINGLE_FILE
        assert r.metadata == {"language_detected": "en"}

    def test_direct_build_no_raise_on_bad_scope(self):
        d = base_dict(scope_hint="project-wide")
        r = make_resolver(None)._build_intent_result("req", d)
        assert r.scope_hint == Scope.SINGLE_FILE
        assert r.intent_type == "bugfix"

    def test_complexity_hint_weird_value_preserved(self):
        # Control group: complexity already fails safe (dict.get default).
        payload = json.loads(_VALID_JSON)
        payload["complexity_hint"] = "Complex"
        client = FakeClient([FakeResponse(content=json.dumps(payload), finish_reason="stop")])
        r = make_resolver(client).resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"
        assert r.complexity_hint == Complexity.LOW

    def test_non_string_search_terms_filtered_not_collapse(self):
        # IR-B3: search_terms is the only role field WITHOUT an isinstance(str)
        # filter (siblings modify/reference/target_symbols all filter).  An
        # unhashable entry (nested list) raised TypeError inside
        # IntentResult.__post_init__ (set membership) → outer except collapsed
        # the whole resolution to minimal fallback.
        payload = json.loads(_VALID_JSON)
        payload["search_terms"] = [[1, 2], "valid", 42, ""]
        client = FakeClient([FakeResponse(content=json.dumps(payload), finish_reason="stop")])
        r = make_resolver(client).resolve("fix ConnectionPool.release")
        assert r.intent_type == "bugfix"  # ← currently collapses to "unknown"
        assert r.search_terms == ["valid"]
        assert r.confidence == 0.9  # ← currently 0.1


# ═══════════════════════════════════════════════════════════════════════════
# _fallback_extraction / _create_empty_result / _create_fallback_dict
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackExtraction:
    def test_word_extraction_and_stop_words(self):
        r = make_resolver(None).resolve("please fix the ConnectionPool release bug")
        assert "ConnectionPool" in r.search_terms
        assert "bug" in r.search_terms
        assert "the" not in r.search_terms
        assert "please" in r.search_terms

    def test_mixed_korean_text(self):
        r = make_resolver(None).resolve("버그 수정해줘")
        assert "버그" in r.search_terms
        assert "수정해줘" in r.search_terms
        assert r.intent_type == "unknown"
        assert r.lane_hint == "planner"

    def test_file_extraction_and_main_agent_lane(self):
        r = make_resolver(None).resolve("update styles.css and fix src/utils/helper.ts")
        assert "styles.css" in r.target_files
        assert "src/utils/helper.ts" in r.target_files
        assert r.lane_hint == "main_agent"
        assert r.spec_hints["modify_files"] == r.target_files

    def test_code_files_planner_lane(self):
        r = make_resolver(None).resolve("fix app.py and helpers.py")
        assert r.lane_hint == "planner"

    def test_no_files_planner_no_hints(self):
        r = make_resolver(None).resolve("버그 수정해줘")
        assert r.lane_hint == "planner"
        assert r.target_files == []
        assert r.spec_hints == {}

    def test_target_files_capped(self):
        r = make_resolver(None).resolve(" ".join(f"file{i}.py" for i in range(10)))
        assert len(r.target_files) == 5

    def test_search_terms_capped(self):
        r = make_resolver(None).resolve(" ".join(f"word{i}" for i in range(20)))
        assert len(r.search_terms) == 10

    def test_file_punctuation_stripped(self):
        r = make_resolver(None).resolve("check out foo.py;")
        assert "foo.py" in r.target_files

    def test_trailing_period_in_filename(self):
        r = make_resolver(None).resolve("read file.txt.")
        assert "file.txt" in r.target_files


class TestCreateEmptyAndFallback:
    def test_empty_result_fields(self):
        r = make_resolver(None)._create_empty_result("")
        assert r.original_request == ""
        assert r.intent_type == "unknown"
        assert r.confidence == 0.0
        assert r.metadata == {"source": "empty_request"}
        assert r.spec_hints == {}
        assert r.lane_hint == "planner"

    def test_fallback_dict_fields(self):
        d = make_resolver(None)._create_fallback_dict("my request")
        assert d["normalized_query"] == "my request"
        assert d["intent_type"] == "unknown"
        assert d["confidence"] == 0.2
        assert d["metadata"] == {"source": "llm_parse_failed"}
        assert d["search_terms"] == []
        assert d["target_files"] == []
        assert d["target_symbols"] == []
