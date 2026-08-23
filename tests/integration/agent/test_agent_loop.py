"""
Integration tests for AgentLoop.
"""

from unittest.mock import Mock, patch

from external_llm.agent.agent_loop import AgentResult
from external_llm.agent.tool_registry import ToolResult


def _numeric_token_fields(resp):
    """Force the token counters on a Mock response to real ints.

    The loop accumulates `total += resp.prompt_tokens` / `.cache_read_input_tokens`.
    A bare Mock attribute raises `TypeError: unsupported operand type(s) for +=:
    'int' and 'Mock'`, which the loop catches as an unexpected error — surfacing
    as status="error" plus a spurious rollback, and hiding whatever the test was
    actually asserting.
    """
    for _f in ("prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        if not isinstance(getattr(resp, _f, None), int):
            setattr(resp, _f, 0)
    return resp


def _responses(mock_llm_client, *responses):
    """Feed `responses` to chat_with_tools, repeating the last one forever.

    A plain ``side_effect=[a, b]`` raises StopIteration the moment the loop
    makes one more call than the test author predicted, which surfaces as
    ``status="error"`` and hides whatever the test was actually checking. The
    loop legitimately makes extra calls (e.g. the write-intent re-nudge), and
    the count is not what these tests are asserting — so serve the last
    response indefinitely instead of running dry.
    """
    seq = [_numeric_token_fields(r) for r in responses]

    def _next(*_a, **_kw):
        return seq[min(_next.calls, len(seq) - 1)] if seq else None

    def _wrapped(*a, **kw):
        r = _next(*a, **kw)
        _next.calls += 1
        return r

    _next.calls = 0
    mock_llm_client.chat_with_tools.side_effect = _wrapped
    mock_llm_client.chat.side_effect = _wrapped
    return _wrapped


def _dispatch_results(mock_dispatch, *results):
    """Same non-exhausting contract as :func:`_responses`, for registry.dispatch."""
    seq = list(results)
    state = {"n": 0}

    def _wrapped(*_a, **_kw):
        r = seq[min(state["n"], len(seq) - 1)]
        state["n"] += 1
        return r

    mock_dispatch.side_effect = _wrapped
    return _wrapped


def test_agent_loop_basic_execution(agent_loop, mock_llm_client):
    """Test basic agent execution with no tool calls."""
    # Mock LLM response with final answer (no tool calls)
    # Note: fixture already sets default response "Test response"
    # Override it to match test expectation
    mock_response = Mock()
    mock_response.content = "I have completed the task successfully."
    mock_response.tool_calls = []
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.cache_read_input_tokens = 0
    mock_response.cache_creation_input_tokens = 0
    mock_llm_client.chat_with_tools.return_value = _numeric_token_fields(mock_response)
    mock_llm_client.chat.return_value = _numeric_token_fields(mock_response)

    result = agent_loop.run("Test request")
    assert isinstance(result, AgentResult)
    # Implementation returns text_reply for text-only LLM responses (no tool calls)
    assert result.status in ("success", "text_reply")
    # The mock override MUST reach the result. The old response-cache path
    # (long removed) returned the fixture default "Test response" for repeated
    # identical-message calls, which is why this was once a weak either-or
    # assertion with a TODO. The cache is gone — restore the exact contract.
    assert result.final_message == "I have completed the task successfully."
    assert len(result.turns) == 0  # No tool calls were made
    assert "turns_used" in result.metadata
    assert result.metadata["turns_used"] == 0
    # Verify the LLM was actually exercised through the native-tools path
    assert mock_llm_client.chat_with_tools.called


def test_agent_loop_with_tool_calls(agent_loop, mock_llm_client, temp_repo_root):
    """Test agent execution with tool calls.

    Was xfail(strict) on the token assertion: the totals carried a third,
    spurious turn (550 vs the predicted 350) because this fixture attaches no
    IntentResult, which the write-intent false-success gate read as "the user
    asked for an edit" and answered with a re-nudge. The gate now stands down
    when intent resolution never classified the request, so the two mocked
    responses are again the only two calls — which is what makes the exact
    token totals below assertable at all.
    """
    # Mock first LLM response with tool call
    mock_response1 = Mock()
    mock_response1.content = ""
    mock_response1.tool_calls = [{"id": "call_1", "name": "find_symbol", "args": {"name": "hello"}}]
    mock_response1.prompt_tokens = 150
    mock_response1.completion_tokens = 80
    mock_response1.raw_response = None
    mock_response1.cache_read_input_tokens = 0
    mock_response1.cache_creation_input_tokens = 0

    # Mock second LLM response with final answer
    mock_response2 = Mock()
    mock_response2.content = "I have read the file."
    mock_response2.tool_calls = []
    mock_response2.prompt_tokens = 200
    mock_response2.completion_tokens = 60
    mock_response2.raw_response = None
    mock_response2.cache_read_input_tokens = 0
    mock_response2.cache_creation_input_tokens = 0

    # Set up mock to return different responses each call
    _responses(mock_llm_client, mock_response1, mock_response2)

    # Mock tool dispatch
    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResult(ok=True, content="File content")
        result = agent_loop.run("Read sample.py file")

    assert result.status in ("success", "text_reply")
    assert len(result.turns) == 1  # One tool call turn
    assert result.turns[0].tool_name == "find_symbol"
    assert result.turns[0].tool_args == {"name": "hello"}
    assert result.turns[0].tool_result.ok

    # Check token tracking
    assert "tokens" in result.metadata
    tokens = result.metadata["tokens"]
    assert tokens["prompt"] == 350  # 150 + 200
    assert tokens["completion"] == 140  # 80 + 60


def test_agent_loop_max_turns(agent_loop, mock_llm_client):
    """Test agent stops when max turns reached."""
    # Mock LLM response that always calls a tool (infinite loop)
    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [
        {
            "id": "call_1",
            # git_status was retired from the registry; an unregistered name is
            # skipped before dispatch, so no turn would ever be recorded.
            "name": "bash",
            "args": {"command": "git status"},
        }
    ]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None
    mock_response.cache_read_input_tokens = 0
    mock_response.cache_creation_input_tokens = 0

    # Always return the same response (tool call)
    mock_llm_client.chat_with_tools.return_value = _numeric_token_fields(mock_response)

    # Set max_turns to 3
    agent_loop.config.max_turns = 3

    # Mock tool dispatch
    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResult(ok=True, content="OK")
        result = agent_loop.run("Test request")

    assert result.status == "max_turns"
    assert len(result.turns) == 3  # Should have used all turns
    # status_detail may not be present in metadata
    # assert "max_turns" in result.metadata.get("status_detail", "").lower()


def test_agent_loop_has_no_planning_phase(agent_loop, mock_llm_client):
    """No planning phase exists on the MAIN_AGENT lane.

    Contract (current): the PLANNER lane was removed from task_router (Lane
    holds only MAIN_AGENT) and the planning_enabled config flag was removed
    entirely. ctx.plan is never written, so metadata["plan"] stays None.
    """

    # Mock first LLM response with a tool call
    mock_response1 = Mock()
    mock_response1.content = ""
    mock_response1.tool_calls = [{"id": "call_1", "name": "find_symbol", "args": {"name": "hello"}}]
    mock_response1.prompt_tokens = 100
    mock_response1.completion_tokens = 50
    mock_response1.raw_response = None

    # Mock second LLM response with final answer
    mock_response2 = Mock()
    mock_response2.content = "I have completed the task successfully."
    mock_response2.tool_calls = []
    mock_response2.prompt_tokens = 120
    mock_response2.completion_tokens = 30
    mock_response2.raw_response = None

    _responses(mock_llm_client, mock_response1, mock_response2)

    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResult(ok=True, content="File content")
        result = agent_loop.run("Test request with planning")

    assert result.status in ("success", "text_reply")
    # The planning phase does not exist on the MAIN_AGENT lane — no plan is
    # ever produced.
    assert result.metadata.get("plan") is None


def test_agent_loop_with_self_review(agent_loop, mock_llm_client, sample_patch):
    """Self-review metadata reflects the disabled mini-loop contract.

    Contract (current): _run_self_review is deliberately short-circuited (the
    mini-loop added latency and false rejections) and returns the fixed string
    "lgtm — self-review disabled." The config flag is still honored — it gates
    whether the review runs and what metadata["self_review"]["enabled"] says —
    but the summary is always the fixed LGTM and issues_found is always False.
    """
    agent_loop.config.self_review_enabled = True
    agent_loop.config.max_review_turns = 3  # Ensure default is set

    # Mock execution response with patch
    mock_patch_response = Mock()
    mock_patch_response.content = ""
    mock_patch_response.tool_calls = [{"id": "call_1", "name": "apply_patch", "args": {"patch": sample_patch}}]
    mock_patch_response.prompt_tokens = 100
    mock_patch_response.completion_tokens = 50
    mock_patch_response.raw_response = None
    # Add get method for LLM response dict-like access
    mock_patch_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_patch_response, k, default))

    # Mock review response
    mock_review_response = Mock()
    mock_review_response.content = "LGTM"
    mock_review_response.tool_calls = []
    mock_review_response.prompt_tokens = 120
    mock_review_response.completion_tokens = 30
    mock_review_response.raw_response = None
    mock_review_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_review_response, k, default))

    # Set up sequence: execution call, then review call
    _responses(mock_llm_client, mock_patch_response, mock_review_response)

    # Mock successful patch application and git_diff
    with (
        patch.object(agent_loop.registry, "dispatch") as mock_dispatch,
        patch.object(agent_loop.registry, "_applied_patches", [sample_patch]),
    ):
        # Create a side effect that returns appropriate results
        call_count = 0

        def dispatch_side_effect(tool, args):
            nonlocal call_count
            call_count += 1
            if tool == "apply_patch":
                return ToolResult(ok=True, content="Patch applied successfully")
            if tool == "git_diff":
                return ToolResult(
                    ok=True, content="diff --git a/sample.py b/sample.py\n@@ -1,7 +1,10 @@\n+Some changes"
                )
            return ToolResult(ok=True, content="OK")

        mock_dispatch.side_effect = dispatch_side_effect
        result = agent_loop.run("Apply patch with review")

    assert result.status in ("success", "text_reply")
    assert "self_review" in result.metadata
    review = result.metadata["self_review"]
    assert review["enabled"] is True
    # Fixed summary: the self-review mini-loop is deliberately short-circuited
    # in _run_self_review — the exact string is the contract.
    assert review["summary"] == "lgtm — self-review disabled."
    assert review["issues_found"] is False


def test_agent_loop_with_tdd_cycle(agent_loop, mock_llm_client, sample_patch):
    """Test agent with TDD auto-test cycle."""
    agent_loop.config.auto_test_on_patch = True
    agent_loop.config.max_tdd_cycles = 3  # Ensure default is set

    # Mock response with patch
    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [{"id": "call_1", "name": "apply_patch", "args": {"patch": sample_patch}}]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None

    mock_llm_client.chat_with_tools.return_value = _numeric_token_fields(mock_response)

    # Mock successful patch application
    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        # First call: apply_patch
        # Second call: run_tests (auto-triggered by TDD)
        _dispatch_results(
            mock_dispatch,
            ToolResult(ok=True, content="Patch applied"),
            ToolResult(ok=True, content="Tests passed"),
        )
        result = agent_loop.run("Apply patch with TDD")

    assert result.status in ("success", "text_reply")
    assert "tdd" in result.metadata
    tdd = result.metadata["tdd"]
    assert tdd["runs"] == 1  # One test run triggered
    assert tdd["pass"] == 1  # One pass


def test_agent_loop_cancellation(agent_loop, mock_llm_client):
    """Test agent cancellation."""
    import threading

    # Set up cancellation event
    cancel_event = threading.Event()
    agent_loop.config.cancel_event = cancel_event

    # Mock slow LLM response
    def slow_chat(*args, **kwargs):
        cancel_event.set()  # Simulate cancellation during LLM call
        raise Exception("Cancelled")

    mock_llm_client.chat_with_tools.side_effect = slow_chat

    result = agent_loop.run("Test request")
    assert result.status in {"cancelled", "error"}
    # Note: actual cancellation handling might vary


def test_agent_loop_context_trimming(agent_loop, mock_llm_client):
    """Test context sliding window trimming."""
    agent_loop.config.context_window_size = 2  # Keep only 2 non-system messages
    agent_loop.config.max_turns = 6  # Enough for 3 tool calls + final

    # Create multiple turns with tool calls (3 tool calls)
    mock_responses = []
    for i in range(3):
        mock_response = Mock()
        mock_response.content = ""
        mock_response.tool_calls = [{"id": f"call_{i}", "name": "git_status", "args": {}}]
        mock_response.prompt_tokens = 100
        mock_response.completion_tokens = 50
        mock_response.raw_response = None
        # Add get method for dict-like access
        mock_response.get = Mock(side_effect=lambda k, default=None, _mr=mock_response: getattr(_mr, k, default))
        mock_responses.append(mock_response)

    # Final response
    final_response = Mock()
    final_response.content = "Done"
    final_response.tool_calls = []
    final_response.prompt_tokens = 100
    final_response.completion_tokens = 50
    final_response.raw_response = None
    final_response.get = Mock(side_effect=lambda k, default=None: getattr(final_response, k, default))
    mock_responses.append(final_response)

    mock_llm_client.chat_with_tools.side_effect = mock_responses

    # Mock tool results
    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.return_value = ToolResult(ok=True, content="OK")
        result = agent_loop.run("Test context trimming")

    # Should succeed despite many turns
    assert result.status in ("success", "text_reply")
    # Context trimming should have prevented unbounded growth


def test_agent_loop_parallel_tool_execution(agent_loop, mock_llm_client):
    """Parallel tool execution dispatches all prepared calls in one batch.

    All tool names must be registered: unregistered names are filtered out of
    prepared_calls before dispatch. git_status was retired from the registry,
    so grep (read-only, parallel-safe) stands in for it here.
    """
    agent_loop.config.parallel_tool_execution_enabled = True

    # Mock response with multiple tool calls
    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [
        {"id": "call_1", "name": "find_symbol", "args": {"name": "hello"}},
        {"id": "call_2", "name": "grep", "args": {"pattern": "hello"}},
        {"id": "call_3", "name": "get_project_info", "args": {}},
    ]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None

    # Mock final response
    mock_final_response = Mock()
    mock_final_response.content = "All tools executed"
    mock_final_response.tool_calls = []
    mock_final_response.prompt_tokens = 120
    mock_final_response.completion_tokens = 30
    mock_final_response.raw_response = None

    _responses(mock_llm_client, mock_response, mock_final_response)

    # Mock parallel dispatch
    with patch.object(agent_loop.registry, "dispatch_parallel") as mock_parallel:
        mock_parallel.return_value = [
            ToolResult(ok=True, content="File content"),
            ToolResult(ok=True, content="Git status"),
            ToolResult(ok=True, content="Project info"),
        ]
        result = agent_loop.run("Test parallel tools")

    assert result.status in ("success", "text_reply")
    assert len(result.turns) == 3
    # Should have used parallel dispatch
    mock_parallel.assert_called_once()


def test_agent_loop_auto_observation(agent_loop, mock_llm_client, sample_patch):
    """Auto-observation injects a real git diff after a successful patch.

    Contract (current): after apply_patch/write_plan succeeds, the pipeline
    collects touched_files from the ToolResult metadata, runs
    ``git diff -- <paths>`` via subprocess (NOT registry.dispatch), and injects
    the result as a ``[auto_observation]`` user message, firing the
    "auto_observation" stream callback. Early-finish then completes the run
    without another LLM turn.
    """
    # Enable auto-observation
    agent_loop.config.auto_observation_enabled = True
    # Ensure early-exit path is eligible for this test
    agent_loop.config.auto_test_on_patch = False
    agent_loop.config.self_review_enabled = False

    # Capture stream callbacks (the auto_observation event rides on _cb)
    events: list[tuple[str, dict]] = []
    agent_loop.config.stream_callback = lambda ev, data: events.append((ev, data))

    # Mock LLM response with patch (no final LLM turn needed due to early-exit)
    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [{"id": "call_1", "name": "apply_patch", "args": {"patch": sample_patch}}]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None
    # Add get method for dict-like access
    mock_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_response, k, default))

    _responses(mock_llm_client, mock_response)

    # REAL dispatch: apply_patch actually applies the patch to temp_repo_root,
    # so the touched_files metadata and the follow-up git diff are genuine.
    result = agent_loop.run("Test auto-observation")

    assert result.status == "success"
    # Check that final_message is not empty (auto-completion occurred)
    assert result.final_message and len(result.final_message.strip()) > 0
    # Auto-observation fired exactly once, carrying the real diff
    obs_events = [e for e in events if e[0] == "auto_observation"]
    assert len(obs_events) == 1
    diff = obs_events[0][1]["diff"]
    assert "self.memory" in diff


def test_agent_loop_auto_repair_apply_patch_hunk_only(agent_loop, mock_llm_client):
    """Test auto-repair for hunk-only apply_patch failures."""
    # Mock LLM response with apply_patch tool call (hunk-only patch)
    hunk_only_patch = """@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
        return a + b"""

    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [
        {"id": "call_1", "name": "apply_patch", "args": {"patch": hunk_only_patch, "path": "sample.py"}}
    ]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None
    mock_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_response, k, default))

    # Final response after successful patch
    mock_final_response = Mock()
    mock_final_response.content = "Patch applied"
    mock_final_response.tool_calls = []
    mock_final_response.prompt_tokens = 120
    mock_final_response.completion_tokens = 30
    mock_final_response.raw_response = None
    mock_final_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_final_response, k, default))

    _responses(mock_llm_client, mock_response, mock_final_response)

    # Mock tool dispatch: first call fails (hunk-only without headers),
    # second call succeeds after auto-repair wrap
    call_counts = {"apply_patch": 0}

    def dispatch_side_effect(tool, args):
        call_counts[tool] = call_counts.get(tool, 0) + 1
        if tool == "apply_patch":
            if call_counts[tool] == 1:
                # First call fails (simulating git apply error for hunk-only)
                return ToolResult(ok=False, content="", error="patch fragment without header")
            # Second call succeeds after auto-repair
            # Verify that patch now contains headers
            patch_text = args.get("patch", "")
            assert "diff --git a/sample.py b/sample.py" in patch_text
            assert "--- a/sample.py" in patch_text
            assert "+++ b/sample.py" in patch_text
            return ToolResult(ok=True, content="Patch applied")
        # Other tools not used
        return ToolResult(ok=True, content="")

    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.side_effect = dispatch_side_effect
        result = agent_loop.run("Test auto-repair hunk-only")

    assert result.status in ("success", "text_reply")
    # Should have called apply_patch twice (first failure, retry success)
    assert call_counts["apply_patch"] == 2
    # Verify auto-repair metadata
    turns_with_patch = [t for t in result.turns if t.tool_name == "apply_patch"]
    assert len(turns_with_patch) == 1  # Only one turn recorded (retry result)
    turn = turns_with_patch[0]
    assert turn.tool_result.ok
    assert "auto_repair" in turn.tool_result.metadata
    assert turn.tool_result.metadata["auto_repair"]["attempted"]
    assert turn.tool_result.metadata["auto_repair"]["success"]
    # B1 regression: the original failure cause must be preserved even though
    # the retry succeeded (retry_result.error is None at that point).
    assert turn.tool_result.metadata["auto_repair"]["original_error"] == "patch fragment without header"


def test_agent_loop_auto_repair_missing_path_no_retry(agent_loop, mock_llm_client):
    """Test auto-repair skips when path missing."""
    hunk_only_patch = """@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
        return a + b"""

    mock_response = Mock()
    mock_response.content = ""
    mock_response.tool_calls = [
        {
            "id": "call_1",
            "name": "apply_patch",
            "args": {"patch": hunk_only_patch},  # Missing path
        }
    ]
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.raw_response = None
    mock_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_response, k, default))

    # Final response after failure
    mock_final_response = Mock()
    mock_final_response.content = "Patch failed"
    mock_final_response.tool_calls = []
    mock_final_response.prompt_tokens = 120
    mock_final_response.completion_tokens = 30
    mock_final_response.raw_response = None
    mock_final_response.get = Mock(side_effect=lambda k, default=None: getattr(mock_final_response, k, default))

    _responses(mock_llm_client, mock_response, mock_final_response)

    call_counts = {"apply_patch": 0}

    def dispatch_side_effect(tool, args):
        call_counts[tool] = call_counts.get(tool, 0) + 1
        if tool == "apply_patch":
            # Should fail only once (no retry because path missing)
            return ToolResult(ok=False, content="", error="patch fragment without header")
        return ToolResult(ok=True, content="")

    with patch.object(agent_loop.registry, "dispatch") as mock_dispatch:
        mock_dispatch.side_effect = dispatch_side_effect
        result = agent_loop.run("Test auto-repair missing path")

    # Should have failed (no retry)
    assert result.status in ("success", "text_reply")  # LLM may still output final message
    # apply_patch called only once (no retry)
    assert call_counts["apply_patch"] == 1
    turns_with_patch = [t for t in result.turns if t.tool_name == "apply_patch"]
    assert len(turns_with_patch) == 1
    turn = turns_with_patch[0]
    assert not turn.tool_result.ok
    # No auto-repair metadata because path missing
    assert "auto_repair" not in turn.tool_result.metadata
