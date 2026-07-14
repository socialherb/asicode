"""
Tests for llm_utils.py — simple_llm_call.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from utils.llm_utils import simple_llm_call


class TestSimpleLlmCallChatClient:
    """Tests for simple_llm_call with a client that has .chat()."""

    def test_returns_content_from_chat(self):
        """chat() response content is returned on success."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello, world!"
        mock_client.chat.return_value = mock_response

        result = simple_llm_call(mock_client, "gpt-4", [{"role": "user", "content": "hi"}])

        assert result == "Hello, world!"
        mock_client.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
        )

    def test_chat_preferred_over_chat_with_tools(self):
        """When both methods exist, chat() is used, not chat_with_tools()."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock(return_value=MagicMock(content="from chat"))
        mock_client.chat_with_tools = MagicMock(return_value=MagicMock(content="from tools"))

        result = simple_llm_call(mock_client, "gpt-4", [{"role": "user", "content": "hi"}])

        assert result == "from chat"
        mock_client.chat.assert_called_once()
        mock_client.chat_with_tools.assert_not_called()

    def test_kwargs_forwarded_to_chat(self):
        """**kwargs are forwarded to chat()."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(content="result")

        simple_llm_call(
            mock_client,
            "gpt-4",
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            thinking_mode=False,
        )

        mock_client.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            temperature=0.5,
            thinking_mode=False,
        )

    def test_messages_and_model_passed_correctly(self):
        """messages and model arguments are forwarded verbatim."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(content="ok")

        messages = [{"role": "system", "content": "You are a bot."}, {"role": "user", "content": "Hello"}]
        simple_llm_call(mock_client, "claude-3", messages)

        mock_client.chat.assert_called_once_with(
            messages=messages,
            model="claude-3",
        )

    def test_empty_messages_list(self):
        """Empty messages list does not raise an error."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(content="response")

        result = simple_llm_call(mock_client, "gpt-4", [])

        assert result == "response"
        mock_client.chat.assert_called_once_with(messages=[], model="gpt-4")


class TestSimpleLlmCallChatWithToolsClient:
    """Tests for simple_llm_call with a client that only has chat_with_tools()."""

    def test_returns_content_from_chat_with_tools(self):
        """chat_with_tools() response content is returned when chat() is absent."""
        mock_client = MagicMock()
        # Only has chat_with_tools, not chat
        del mock_client.chat
        mock_response = MagicMock()
        mock_response.content = "Fallback response"
        mock_client.chat_with_tools.return_value = mock_response

        result = simple_llm_call(mock_client, "claude-3", [{"role": "user", "content": "hi"}])

        assert result == "Fallback response"
        mock_client.chat_with_tools.assert_called_once_with(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="claude-3",
        )

    def test_kwargs_forwarded_to_chat_with_tools(self):
        """**kwargs are forwarded to chat_with_tools() when chat() is absent."""
        mock_client = MagicMock()
        del mock_client.chat
        mock_client.chat_with_tools.return_value = MagicMock(content="result")

        simple_llm_call(
            mock_client,
            "claude-3",
            [{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=100,
        )

        mock_client.chat_with_tools.assert_called_once_with(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="claude-3",
            temperature=0.7,
            max_tokens=100,
        )


class TestSimpleLlmCallErrorHandling:
    """Tests for error handling in simple_llm_call."""

    def test_chat_exception_returns_empty_string(self):
        """When chat() raises an exception, return empty string."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("API failure")

        result = simple_llm_call(mock_client, "gpt-4", [{"role": "user", "content": "hi"}])

        assert result == ""

    def test_chat_with_tools_exception_returns_empty_string(self):
        """When chat_with_tools() raises an exception, return empty string."""
        mock_client = MagicMock()
        del mock_client.chat
        mock_client.chat_with_tools.side_effect = ConnectionError("Network error")

        result = simple_llm_call(mock_client, "claude-3", [{"role": "user", "content": "hi"}])

        assert result == ""

    def test_response_without_content_returns_empty_string(self):
        """Response object lacking a content attribute returns empty string."""
        mock_client = MagicMock()
        mock_client.chat.return_value = object()  # no .content attribute

        result = simple_llm_call(mock_client, "gpt-4", [{"role": "user", "content": "hi"}])

        assert result == ""

    def test_response_content_is_none_returns_empty_string(self):
        """Response with content=None returns empty string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_client.chat.return_value = mock_response

        result = simple_llm_call(mock_client, "gpt-4", [{"role": "user", "content": "hi"}])

        assert result == ""
