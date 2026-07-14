"""
Integration tests for SSE streaming and event sequencing.
"""

import pytest

from tests.integration.helpers import capture_sse_events, verify_event_sequence


@pytest.mark.integration
class TestSSEStreaming:
    """Test Server-Sent Events streaming functionality."""

    def test_sse_basic_stream_structure(self, test_client, temp_repo_root: str):
        """Test basic SSE stream structure and format."""
        # /agent/run/stream is a GET endpoint with query params
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Test SSE streaming",
            },
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Read first few lines
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        # Even if external LLM is disabled, should return at least one SSE event
        assert len(lines) > 0
        # All lines should be valid SSE format: a field line (event:/data:/id:/
        # retry:) or a comment (starts with ":"). The stream emits a `retry:`
        # directive first to disable EventSource auto-reconnect.
        for line in lines:
            assert (
                line.startswith(("event:", "data:", "id:", "retry:"))
                or line.startswith(":")
            )

    def test_sse_event_sequence_basic_agent(self, test_client, temp_repo_root: str):
        """Test event sequence for basic agent execution."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Read sample.py",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 6:
                    break

        # Parse events
        parsed_events = capture_sse_events(lines)
        # Should have at least one event
        assert len(parsed_events) > 0

    def test_sse_planning_events(self, test_client, temp_repo_root: str):
        """Test SSE response with planning_enabled parameter."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Plan and execute fix",
                "planning_enabled": "true",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_tdd_cycle_events(self, test_client, temp_repo_root: str):
        """Test SSE response with auto_test_on_patch parameter."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Test with TDD",
                "auto_test_on_patch": "true",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_cancellation_events(self, test_client, temp_repo_root: str):
        """Test SSE stream can be connected to."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Task to cancel",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 2:
                    break

        assert len(lines) > 0

    def test_sse_error_events(self, test_client, temp_repo_root: str):
        """Test SSE error when request_text is empty."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 2:
                    break

        parsed_events = capture_sse_events(lines)
        # Should receive an error event for empty request_text
        event_types = [e.get("event") for e in parsed_events if "event" in e]
        assert "error" in event_types

    def test_sse_multi_agent_events(self, test_client, temp_repo_root: str):
        """Test SSE response with multi_agent parameter."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Complex multi-file task",
                "multi_agent": "true",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_context_trimmed_event(self, test_client, temp_repo_root: str):
        """Test SSE stream with small context window size."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Long conversation test",
                "max_turns": "3",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_auto_observation_event(self, test_client, temp_repo_root: str):
        """Test SSE stream responds with events for agent tasks."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Apply patch and observe",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_review_events(self, test_client, temp_repo_root: str):
        """Test SSE response with self_review_enabled parameter."""
        with test_client.stream(
            "GET",
            "/agent/run/stream",
            params={
                "repo_root": temp_repo_root,
                "request_text": "Task with self-review",
                "self_review_enabled": "true",
            },
        ) as response:
            assert response.status_code == 200

            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(line.decode("utf-8") if isinstance(line, bytes) else line)
                if len(lines) >= 4:
                    break

        parsed_events = capture_sse_events(lines)
        assert len(parsed_events) > 0

    def test_sse_event_data_structure(self):
        """Test that SSE event data has consistent structure."""
        # Test helper function
        sample_events = [
            'event: test_event',
            'data: {"session_id": "123", "value": 42}',
            '',
            'event: another_event',
            'data: {"session_id": "123", "status": "ok"}',
            ''
        ]

        parsed = capture_sse_events(sample_events)

        assert len(parsed) == 2
        assert parsed[0]["event"] == "test_event"
        assert parsed[0]["data"]["session_id"] == "123"
        assert parsed[0]["data"]["value"] == 42
        assert parsed[1]["event"] == "another_event"

    def test_sse_sequence_verification(self):
        """Test SSE event sequence verification helper."""
        events = [
            {"event": "start", "data": {}},
            {"event": "tool_call_preview", "data": {}},
            {"event": "tool_call", "data": {}},
            {"event": "complete", "data": {}}
        ]

        # Valid sequence
        valid_pattern = ["start", "tool_call_preview", "tool_call", "complete"]
        assert verify_event_sequence(events, valid_pattern) is True

        # Invalid sequence
        invalid_pattern = ["start", "complete", "tool_call"]  # Wrong order
        assert verify_event_sequence(events, invalid_pattern) is False

        # Different length
        short_pattern = ["start", "tool_call"]
        assert verify_event_sequence(events, short_pattern) is False
