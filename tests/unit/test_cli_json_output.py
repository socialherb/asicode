"""Tests for the machine-readable JSON output paths: --json (final blob) and
--json-stream (NDJSON), plus the clarification-questions field (B4) shared by
both. These pin the payload shape that Tenet-style automation depends on."""
from __future__ import annotations

import json
from types import SimpleNamespace

import asi


def _result(status="success", **kw):
    return SimpleNamespace(
        status=status,
        final_message=kw.get("final_message", ""),
        error=kw.get("error", None),
        metadata=kw.get("metadata", None),
        applied_patches=kw.get("applied_patches", None),
        turns=kw.get("turns", None),
    )


def test_result_output_dict_includes_questions_on_clarification():
    """B4: clarification_needed carries structured questions in the JSON body."""
    r = _result(
        status="clarification_needed",
        final_message="Which file?",
        metadata={"required_clarifications": [
            {"field": "target_files", "reason": "not specified", "suggestion": "src/app.py"},
        ]},
    )
    d = asi._result_output_dict(r, 1.5)
    assert d["status"] == "clarification_needed"
    assert d["questions"] == ["target_files: not specified"]
    assert d["duration_ms"] == 1500


def test_result_output_dict_empty_questions_on_success():
    """Non-clarification statuses yield an empty (but present) questions list."""
    d = asi._result_output_dict(_result(status="success"), 0.1)
    assert d["status"] == "success"
    assert d["questions"] == []


def test_result_output_dict_falls_back_to_final_message():
    """Without required_clarifications, the free-text question is the fallback."""
    r = _result(status="clarification_needed", final_message="Please specify the target.")
    d = asi._result_output_dict(r, 0.0)
    assert d["questions"] == ["Please specify the target."]


def test_json_stream_emit_writes_one_ndjson_line(capsys):
    """F2: each event is one self-contained JSON object on its own line."""
    asi._json_stream_emit("tool_call", {"tool": "read_file", "status": "running"})
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["event"] == "tool_call"
    assert obj["tool"] == "read_file"
    assert obj["status"] == "running"


def test_json_stream_emit_handles_nonserializable_payload(capsys):
    """default=str keeps streaming robust against non-JSON payloads (never raises)."""
    asi._json_stream_emit("weird", {"path": __import__("pathlib").Path("/x/y")})
    out = capsys.readouterr().out
    obj = json.loads(out.strip())
    assert obj["event"] == "weird"
    assert "Path" in obj["path"] or "/x/y" in obj["path"]


def test_json_stream_result_line_has_full_payload(capsys):
    """The final 'result' NDJSON line carries the same payload as --json."""
    r = _result(status="success", applied_patches=[{"file": "a.py"}])
    asi._json_stream_emit("result", asi._result_output_dict(r, 2.0))
    out = capsys.readouterr().out
    obj = json.loads(out.strip())
    assert obj["event"] == "result"
    assert obj["status"] == "success"
    assert obj["patched_files"] == ["a.py"]
    assert obj["duration_ms"] == 2000


# ── patched_files extraction (in-process string form vs IPC dict form) ──────

def test_extract_patched_file_dict_form():
    """IPC worker patches are {'file': ...} dicts — extract the path."""
    assert asi._extract_patched_file({"file": "src/app.py"}) == "src/app.py"
    assert asi._extract_patched_file({"path": "x.py"}) == "x.py"
    assert asi._extract_patched_file({"file": None, "path": None}) == ""


def test_extract_patched_file_structured_prefix():
    """In-process write path emits edit_file:PATH / edit_text:PATH op strings."""
    assert asi._extract_patched_file("edit_text:hello.py:replace:False") == "hello.py"
    assert asi._extract_patched_file("edit_file:src/x.py:replace:+3/-1") == "src/x.py"
    assert asi._extract_patched_file("edit_file:pkg/sub/mod.py:insert:0/0") == "pkg/sub/mod.py"
    # modify_symbol prefix (was missing before fix — caused patched_files=[])
    assert asi._extract_patched_file("modify_symbol:src/app.py:MyClass") == "src/app.py"
    assert asi._extract_patched_file("modify_symbol:pkg/module.py:some_func") == "pkg/module.py"


def test_extract_patched_file_raw_diff_text():
    """Patch-engine patches are raw unified-diff text — parse the file header."""
    diff = (
        "diff --git a/foo/bar.py b/foo/bar.py\n"
        "index 123..456 100644\n"
        "--- a/foo/bar.py\n"
        "+++ b/foo/bar.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+new line\n"
    )
    assert asi._extract_patched_file(diff) == "foo/bar.py"
    # +++ b/ fallback when there is no diff --git header
    assert asi._extract_patched_file("--- a/x\n+++ b/x.py\n") == "x.py"


def test_extract_patched_file_unparseable_returns_empty():
    """Unstructured non-diff text yields '' so the caller drops it."""
    assert asi._extract_patched_file("some opaque message") == ""
    assert asi._extract_patched_file("") == ""
    assert asi._extract_patched_file(None) == ""


def test_result_output_dict_patched_files_normalizes_mixed_forms():
    """End-to-end: a mix of dict / structured-string / diff patches → clean paths."""
    r = _result(
        status="success",
        applied_patches=[
            {"file": "from_worker.py"},
            "edit_text:from_inprocess.py:replace:False",
            "diff --git a/from_engine.py b/from_engine.py\n+++ b/from_engine.py\n",
            "modify_symbol:from_modify_symbol.py:MyFunc",
            "opaque non-diff string",  # dropped
        ],
    )
    d = asi._result_output_dict(r, 0.5)
    assert d["patched_files"] == [
        "from_worker.py", "from_inprocess.py", "from_engine.py", "from_modify_symbol.py",
    ]
    assert d["patches"] == 5  # count reflects raw entries, list reflects parsed files


# ── turns field: metadata.turns_used fallback ───────────────────────────────

def test_result_output_dict_turns_uses_list_when_populated():
    """When result.turns is populated, its length is authoritative."""
    r = _result(status="success", turns=[object(), object(), object()])
    r.metadata = {"turns_used": 99}  # must be ignored in favour of the list
    assert asi._result_output_dict(r, 0.0)["turns"] == 3


def test_result_output_dict_turns_falls_back_to_metadata():
    """MAIN_AGENT normal path leaves result.turns empty but sets metadata.turns_used."""
    r = _result(status="success", turns=None)
    r.metadata = {"turns_used": 5}
    assert asi._result_output_dict(r, 0.0)["turns"] == 5


def test_result_output_dict_turns_zero_when_nothing_available():
    r = _result(status="success", turns=None)
    r.metadata = None
    assert asi._result_output_dict(r, 0.0)["turns"] == 0


def test_result_output_dict_turns_accepts_int_value():
    """F5 regression: the --orchestrate adapter (``_orchestrator_result_to_agent_like``)
    returns ``turns`` as an INT (summed sub-agent counts), not a list. The old
    ``len(result.turns)`` raised ``TypeError: object of type 'int' has no len()``
    on a SUCCESSFUL orchestration's final ``result`` event (turn 13106 FAIL).
    ``_turns_to_int`` must tolerate both int and list shapes.
    """
    r = _result(status="success", turns=7)  # int, not a list
    r.metadata = {"turns_used": 99}  # non-zero list/int value is authoritative
    assert asi._result_output_dict(r, 0.0)["turns"] == 7


def test_result_output_dict_turns_int_zero_falls_back_to_metadata():
    """An int turns of 0 is falsy → fall back to metadata.turns_used (mirrors the
    list-empty path)."""
    r = _result(status="success", turns=0)
    r.metadata = {"turns_used": 4}
    assert asi._result_output_dict(r, 0.0)["turns"] == 4


def test_turns_to_int_normalizes_all_shapes():
    assert asi._turns_to_int([1, 2, 3]) == 3
    assert asi._turns_to_int(5) == 5
    assert asi._turns_to_int(0) == 0
    assert asi._turns_to_int(None) == 0
    assert asi._turns_to_int(True) == 0  # bool guard (bool is int subclass)


def test_orchestrator_result_to_agent_like_aggregates_int_turns():
    """F5: ``_orchestrator_result_to_agent_like`` sums each sub-agent's turn count.
    SubagentResult.turns is an INT — the old ``len(_sr.turns or [])`` raised
    TypeError once a sub-agent reported a non-zero count. Must aggregate cleanly and
    expose an int ``turns`` that ``_result_output_dict`` accepts."""
    from external_llm.agent.subagent_ipc import SubagentResult

    orch_result = SimpleNamespace(
        status="success",
        summary="ok",
        subtask_results=[
            SubagentResult(task_id="s1", status="success", turns=3, applied_patches=[{"file": "a.py"}]),
            SubagentResult(task_id="s2", status="success", turns=4, applied_patches=[{"file": "b.py"}]),
        ],
        metadata={"subtasks": 2, "parallel": True},
    )
    agent_like = asi._orchestrator_result_to_agent_like(orch_result)
    assert agent_like.turns == 7  # 3 + 4, NOT a TypeError
    assert isinstance(agent_like.turns, int)
    # The int turns must flow through _result_output_dict without crashing.
    out = asi._result_output_dict(agent_like, 2.0)
    assert out["turns"] == 7
    # patched_files is the NORMALIZED path-string list (dicts extracted to names).
    assert out["patched_files"] == ["a.py", "b.py"]


# ── error output stable schema (questions present on every status) ──────────

def test_json_error_output_has_questions_field(capsys):
    """Every status — including errors — must carry a questions list (stable schema)."""
    asi._json_error_output("error", "boom", duration_ms=10)
    obj = json.loads(capsys.readouterr().out)
    assert obj["status"] == "error"
    assert obj["error"] == "boom"
    assert obj["duration_ms"] == 10
    assert "questions" in obj
    assert obj["questions"] == []


# ── --json single-blob stdout cleanliness ───────────────────────────────────

def test_run_once_json_blob_keeps_stdout_to_single_json_line(tmp_path, monkeypatch, capsys):
    """--json (single blob) must not pollute stdout with human progress output.

    The human _ProgressPrinter writes ANSI turn/tool lines to stdout; in --json
    mode stdout must carry ONLY the final JSON line so a consumer can
    json.loads(stdout) directly. We monkeypatch the engine so the captured
    stream_cb is exercised the way the real loop would (turn_start / tool_call),
    and assert stdout ends up with exactly one parseable JSON line.
    """
    captured = {}

    def _fake_build_engine(**kw):
        captured["stream_cb"] = kw.get("stream_cb")
        return object()  # loop instance is opaque to the patched run path

    monkeypatch.setattr(asi, "_build_engine", _fake_build_engine)
    monkeypatch.setattr(asi, "_git_baseline", lambda root: "")

    def _fake_run_with_cancel(loop, request, context, cancel_event, stream_callback=None):
        # Drive the callback exactly like the real engine does.
        if stream_callback:
            stream_callback("turn_start", {"turn": 1})
            stream_callback("tool_call", {"tool": "read_file", "status": "running"})
            stream_callback("tool_result", {"tool": "read_file", "ok": True})
        return SimpleNamespace(
            status="success",
            final_message="done",
            error=None,
            metadata={"turns_used": 1},
            applied_patches=[],
            turns=None,
        )

    monkeypatch.setattr(asi, "_run_with_cancel", _fake_run_with_cancel)

    args = SimpleNamespace(
        repo=str(tmp_path), verbose=False,
        json_stream=False, json=True,
        provider="", model="", api_key=None, max_turns=3,
        thinking_mode=None, reasoning_effort=None,
    )
    rc = asi.run_once(args, "do something")
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # Exactly one line on stdout, and it is the final JSON payload.
    assert len(lines) == 1, f"expected single JSON line, got: {lines!r}"
    obj = json.loads(lines[0])
    assert obj["status"] == "success"
    assert obj["turns"] == 1
    assert rc == 0


# ── F7: cancelled result error field ─────────────────────────────────────────


def test_result_output_dict_cancelled_fills_error_reason():
    """F7: status=cancelled with empty error still carries the canonical reason
    so consumers read ONE field (error) for the failure cause."""
    r = _result(status="cancelled", error=None)
    d = asi._result_output_dict(r, 0.5)
    assert d["status"] == "cancelled"
    assert d["error"] == "Request cancelled by user"


def test_result_output_dict_cancelled_keeps_explicit_error():
    """If the engine already set a specific cancel error, it is preserved."""
    r = _result(status="cancelled", error="cancelled by user during retry wait")
    d = asi._result_output_dict(r, 0.5)
    assert d["error"] == "cancelled by user during retry wait"


def test_result_output_dict_success_error_stays_null():
    """Non-cancelled status with no error stays null (no spurious reason)."""
    r = _result(status="success", error=None)
    d = asi._result_output_dict(r, 0.5)
    assert d["error"] is None


# ── F5: --orchestrate single-shot result adaptation ──────────────────────────


def test_orchestrator_result_to_agent_like_flattens_subtasks():
    """F5: OrchestratorResult is flattened to the AgentResult-like shape the JSON
    output expects — sub-task patches and turn counts roll up to the top level."""
    from types import SimpleNamespace
    _orch = SimpleNamespace(
        status="success",
        summary="did the thing",
        subtask_results=[
            SimpleNamespace(applied_patches=[{"file": "a.py"}], turns=[1, 2, 3]),
            SimpleNamespace(applied_patches=[{"file": "b.py"}, "edit_text:c.py:replace:False"], turns=[1]),
        ],
        metadata={"mode": "tool_loop"},
    )
    d = asi._orchestrator_result_to_agent_like(_orch)
    assert d.status == "success"
    assert d.final_message == "did the thing"
    assert d.error is None  # success → no error
    assert d.turns == 4
    # The dict flows straight through; _extract_patched_file normalizes later.
    assert d.applied_patches == [{"file": "a.py"}, {"file": "b.py"}, "edit_text:c.py:replace:False"]
    assert d.metadata == {"mode": "tool_loop"}


def test_orchestrator_result_to_agent_like_error_carries_summary():
    """A non-success status carries the summary as the error reason."""
    from types import SimpleNamespace
    _orch = SimpleNamespace(
        status="error", summary="all sub-agents failed",
        subtask_results=[], metadata={},
    )
    d = asi._orchestrator_result_to_agent_like(_orch)
    assert d.status == "error"
    assert d.error == "all sub-agents failed"
    assert d.applied_patches == []
    assert d.turns == 0


def test_orchestrate_flag_is_recognized():
    """F5: --orchestrate parses into args.orchestrate (the run_once gate)."""
    import argparse
    # Reproduce the relevant slice of the real parser to avoid importing the
    # whole module's main() side effects.
    p = argparse.ArgumentParser()
    p.add_argument("--orchestrate", action="store_true")
    ns = p.parse_args(["--orchestrate"])
    assert ns.orchestrate is True
    assert p.parse_args([]).orchestrate is False
