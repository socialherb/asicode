"""Defense-depth parity: _handle_max_turns_reached must accumulate cache tokens
(cache_read_input_tokens, cache_creation_input_tokens) into ctx — just like the
main turn loop (L228-239) already does.

Previously the two accumulation sites in _handle_max_turns_reached (L381-388
and L407-414) only accumulated prompt/completion tokens, silently dropping
cache token consumption. For cache-heavy providers (Anthropic, z.ai) this
underreported cache_read_tokens and cache_creation_tokens, skewing
cache_adjusted_cost_usd and cache_hit_ratio in the final metadata.
"""
from __future__ import annotations

from unittest import mock

from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin
from external_llm.agent.agent_loop_types import TurnContext


def _make_response(prompt=80, completion=20, cache_read=500, cache_creation=100, content="done", tool_calls=None):
    """Build a mock LLM response with split-token fields."""
    r = mock.MagicMock()
    r.prompt_tokens = prompt
    r.completion_tokens = completion
    r.tokens_used = prompt + completion
    r.cache_read_input_tokens = cache_read
    r.cache_creation_input_tokens = cache_creation
    r.content = content
    r.tool_calls = tool_calls or []
    r.finish_reason = "stop"
    # _rget tries .get() first (dict path), then getattr — raise AttributeError
    # so _rget falls through to getattr (mirrors real LLMResponse dataclass which
    # has no .get() method).
    r.get = mock.MagicMock(side_effect=AttributeError("no .get"))
    return r


def _make_ctx():
    """Build a minimal TurnContext with token counters initialized to 0."""
    ctx = TurnContext.__new__(TurnContext)
    ctx.messages = []
    ctx.read_only_request = True  # skip the write-intent guard
    ctx.turns = []
    ctx.total_prompt_tokens = 0
    ctx.total_completion_tokens = 0
    ctx.total_cache_read_tokens = 0
    ctx.total_cache_creation_tokens = 0
    ctx.last_call_prompt_tokens = 0
    ctx.last_call_completion_tokens = 0
    ctx.provider_name = "anthropic"
    ctx.model_name = "claude-test"
    ctx.base_url = ""
    ctx.plan = None
    ctx.tdd_fail_count = 0
    ctx.tdd_total_runs = 0
    ctx.tdd_total_pass = 0
    ctx.has_native_tools = False
    ctx.request = "test"
    ctx.context = None
    ctx.tier = 1
    ctx.git_state = None
    ctx.rollback_performed = False
    ctx.rollback_result = None
    ctx.write_tool_used = False
    ctx.budget_warned = False
    ctx.session_id = "test-session"
    ctx.route = None
    return ctx


def _make_loop():
    """Build a TurnPipelineMixin instance with minimal mocked dependencies."""
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    loop.config = mock.MagicMock()
    loop.config.max_turns = 5
    loop.config.model_name = "claude-test"
    loop.config.self_review_enabled = False
    loop.config.make_token_callback.return_value = None
    loop.llm_client = mock.MagicMock()
    loop.llm_client.get_provider_name.return_value = "anthropic"
    loop.registry = mock.MagicMock()
    loop.registry.applied_patches = []
    loop.performance_collector = mock.MagicMock()
    loop.performance_collector.get_summary.return_value = {}
    loop._save_session_log = mock.MagicMock()
    loop._is_trivial_edit_request = lambda req: True
    return loop


def test_max_turns_accumulates_cache_tokens_no_tool_calls():
    """When max_turns final call has no tool_calls, cache tokens must be accumulated."""
    loop = _make_loop()
    ctx = _make_ctx()
    resp = _make_response(cache_read=500, cache_creation=100)
    loop._llm_call_with_tools = mock.MagicMock(return_value=resp)

    result = loop._handle_max_turns_reached(ctx)

    assert ctx.total_cache_read_tokens == 500, "cache_read_tokens were silently dropped"
    assert ctx.total_cache_creation_tokens == 100, "cache_creation_tokens were silently dropped"
    assert ctx.total_prompt_tokens == 80
    assert ctx.total_completion_tokens == 20
    # metadata reflects the accumulated cache tokens
    tokens_meta = result.metadata["tokens"]
    assert tokens_meta["cache_read_tokens"] == 500
    assert tokens_meta["cache_creation_tokens"] == 100


def test_max_turns_accumulates_cache_tokens_with_wrap_up_retry():
    """When the first max_turns call still has tool_calls, a wrap-up retry happens.
    Both calls' cache tokens must be accumulated."""
    loop = _make_loop()
    ctx = _make_ctx()
    # First call returns tool_calls (triggers wrap-up retry), second returns none
    resp1 = _make_response(cache_read=300, cache_creation=50, tool_calls=[{"id": "t1"}])
    resp2 = _make_response(cache_read=400, cache_creation=60, content="final answer")
    loop._llm_call_with_tools = mock.MagicMock(side_effect=[resp1, resp2])

    result = loop._handle_max_turns_reached(ctx)

    # Both calls' cache tokens must be accumulated
    assert ctx.total_cache_read_tokens == 700, "wrap-up retry cache_read_tokens were silently dropped"
    assert ctx.total_cache_creation_tokens == 110, "wrap-up retry cache_creation_tokens were silently dropped"
    assert ctx.total_prompt_tokens == 160  # 80 + 80
    assert ctx.total_completion_tokens == 40   # 20 + 20
    tokens_meta = result.metadata["tokens"]
    assert tokens_meta["cache_read_tokens"] == 700
    assert tokens_meta["cache_creation_tokens"] == 110
