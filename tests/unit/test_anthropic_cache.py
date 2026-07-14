"""Tests for Anthropic prompt-cache breakpoint placement.

Covers two complementary cache contracts:

1. Anthropic explicit cache_control breakpoints — sliding breakpoint on the
   final message (so a tool loop caches the growing conversation prefix) and
   static system-prompt breakpoints.
2. OpenAI-compatible providers (ZAI/GLM, DeepSeek, Ollama) must NOT receive the
   Anthropic-only ``cache_breakpoint_offset`` kwarg in their request payload —
   they use automatic prefix matching and silently ignore the field today, but
   leaking it risks future strict-validation 400s.
"""


from external_llm.anthropic_client import AnthropicClient
from external_llm.client import LLMMessage
from external_llm.openai_client import OpenAIClient, ZAIClient

EPHEMERAL = {"type": "ephemeral"}


class TestMarkLastMessageForCaching:
    mark = staticmethod(AnthropicClient._mark_last_message_for_caching)

    def test_string_content_promoted_to_text_block(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.mark(msgs)
        assert msgs[-1]["content"] == [
            {"type": "text", "text": "hello", "cache_control": EPHEMERAL}
        ]

    def test_list_content_marks_last_block(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}],
        }]
        self.mark(msgs)
        assert msgs[-1]["content"][-1]["cache_control"] == EPHEMERAL

    def test_does_not_mutate_caller_blocks(self):
        orig = [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]
        msgs = [{"role": "user", "content": orig}]
        self.mark(msgs)
        # The original block/list must be untouched (we rebuild a copy).
        assert "cache_control" not in orig[-1]

    def test_only_last_message_marked(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "c"}]},
        ]
        self.mark(msgs)
        assert msgs[-1]["content"][-1]["cache_control"] == EPHEMERAL
        # earlier assistant block stays unmarked
        assert "cache_control" not in msgs[1]["content"][-1]

    def test_empty_messages_noop(self):
        msgs = []
        self.mark(msgs)
        assert msgs == []

    def test_empty_string_content_skipped(self):
        msgs = [{"role": "user", "content": ""}]
        self.mark(msgs)
        # empty content block is invalid → left as-is
        assert msgs[-1]["content"] == ""


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        }


class _FakeSession:
    def __init__(self):
        self.captured = None

    def post(self, url, headers=None, json=None, timeout=None, **kw):
        self.captured = json
        return _FakeResponse()


def _make_client():
    c = AnthropicClient.__new__(AnthropicClient)
    c.api_key = "test"
    c.base_url = None
    c.timeout = 30
    c._session = _FakeSession()
    return c


class TestChatWithToolsCaching:
    def test_payload_has_system_and_last_message_breakpoints(self):
        client = _make_client()
        system_text = "## Identity\n" + ("You are a careful assistant. " * 40) + \
                      "\n## Rules\n" + ("Follow the rules closely. " * 40)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "first request"},
        ]
        tools = [{"name": "read_file", "description": "Read a file",
                  "parameters": {"type": "object", "properties": {}}}]

        client.chat_with_tools(messages, tools, model="claude-sonnet-4-20250514")
        payload = client._session.captured

        # System prompt is split into cached chunks.
        sys_blocks = payload["system"]
        assert any(b.get("cache_control") == EPHEMERAL for b in sys_blocks), sys_blocks

        # Last message carries a sliding breakpoint.
        last = payload["messages"][-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == EPHEMERAL

        # Breakpoint count stays within Anthropic's limit of 4.
        n = sum(1 for b in sys_blocks if b.get("cache_control"))
        n += sum(
            1 for m in payload["messages"]
            if isinstance(m["content"], list)
            for blk in m["content"] if isinstance(blk, dict) and blk.get("cache_control")
        )
        assert n <= 4


class _FakeOpenAIResponse:
  """Minimal response shape for OpenAIClient.chat/chat_with_tools parsing."""
  status_code = 200
  text = "{}"
  headers: dict = {}

  def json(self):
      return {
          "choices": [{"message": {"content": "ok", "tool_calls": None}}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 1},
      }


class _FakeOpenAISession:
  """Captures the json payload sent via session.post()."""

  def __init__(self):
      self.captured = None

  def post(self, url, headers=None, json=None, timeout=None, **kw):
      self.captured = json
      resp = _FakeOpenAIResponse()
      # streaming path reads iter_lines / iter_content
      resp.iter_lines = lambda: iter([])
      resp.iter_content = lambda *a, **k: iter([])
      resp.close = lambda: None
      return resp


def _make_openai_client(cls):
  c = cls.__new__(cls)
  c.api_key = "test"
  c.base_url = "https://api.openai.com/v1"
  c.timeout = 30
  c._session = _FakeOpenAISession()
  return c


class TestOpenAIPayloadDoesNotLeakCacheBreakpoint:
  """``cache_breakpoint_offset`` is Anthropic-only. OpenAI-compatible providers
  must never see it in the request payload (they use automatic prefix caching
  and the field risks future strict-validation 400s). Regression guard for the
  pop() added to OpenAIClient.chat / chat_with_tools."""

  def _msgs(self):
      return [LLMMessage(role="user", content="hi")]

  def test_openai_client_chat_strips_breakpoint_offset(self):
      c = _make_openai_client(OpenAIClient)
      c.chat(self._msgs(), model="gpt-4o", cache_breakpoint_offset=42)
      assert "cache_breakpoint_offset" not in c._session.captured

  def test_openai_client_chat_with_tools_strips_breakpoint_offset(self):
      c = _make_openai_client(OpenAIClient)
      c.chat_with_tools(
          self._msgs(), [], model="gpt-4o",
          cache_breakpoint_offset=42, token_callback=lambda _: None,
      )
      assert "cache_breakpoint_offset" not in c._session.captured

  def test_zai_client_chat_strips_breakpoint_offset(self):
      # ZAIClient overrides chat() but delegates to super(); the pop must
      # still fire on the parent path so ZAI's payload stays clean.
      c = _make_openai_client(ZAIClient)
      c.chat(self._msgs(), model="glm-4.6", cache_breakpoint_offset=42)
      assert "cache_breakpoint_offset" not in c._session.captured

  def test_zai_client_chat_with_tools_strips_breakpoint_offset(self):
      c = _make_openai_client(ZAIClient)
      c.chat_with_tools(
          self._msgs(), [], model="glm-4.6",
          cache_breakpoint_offset=42, thinking_mode=True,
          token_callback=lambda _: None,
      )
      assert "cache_breakpoint_offset" not in c._session.captured

  def test_zai_client_thinking_field_preserved(self):
      # Ensure stripping the Anthropic-only kwarg does not also drop ZAI's
      # own thinking/reasoning params.
      c = _make_openai_client(ZAIClient)
      c.chat_with_tools(
          self._msgs(), [], model="glm-4.6",
          thinking_mode=True, cache_breakpoint_offset=42,
          token_callback=lambda _: None,
      )
      payload = c._session.captured
      assert payload.get("thinking") == {"type": "enabled"}
      assert "cache_breakpoint_offset" not in payload
