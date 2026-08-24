"""Turn-end coalescing of the post-write semantic check.

``validate_semantics`` spawns a toolchain process (pyright / tsc / go build /
javac / kotlinc / gcc -fsyntax-only) whose cost is dominated by cold start —
measured at 0.35 s for pyright on a two-line file with no imports, and 1.84 s
of pyright across five edits to one file against 1.94 s of total wall. So it is
run once per (turn, file) instead of once per write.

The ordering is the whole point and is what these tests pin. A first-write-wins
cache is the obvious implementation and is wrong: it reports the file as it was
before the later edits, and reports ``semantic_diagnostics: []`` for each of
those edits, which ``agent_loop._append_semantic_diagnostics`` renders exactly
like a clean check. An edit that introduces an undefined name then reaches the
model looking verified. Coalescing must therefore observe the LAST write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin

BROKEN = "undefined_symbol_xyz"


@pytest.fixture
def reg(tool_registry, monkeypatch):
    """Registry whose semantic validator is a fast, deterministic stand-in.

    It reads the file from disk exactly as the real providers do, so "which
    content did the check observe" is a real property of the run rather than
    something the stub decides.

    Two different costs are recorded, because they are no longer the same
    number. ``_sem_calls`` is the files examined; ``_sem_spawns`` is the
    toolchain invocations, one entry per batch. Coalescing collapses repeat
    writes into one entry in the first; batching collapses a turn's whole file
    set into one entry in the second. A test that only ever counted files could
    not tell a batched run from N separate ones.
    """
    from external_llm.languages import python_provider as pp

    calls: list[str] = []
    spawns: list[list[str]] = []

    class _Err:
        def __init__(self, msg):
            self.line, self.col, self.message = 1, 1, msg
            self.severity, self.code = "error", "reportUndefinedVariable"

    class _Res:
        def __init__(self, errors):
            self.errors = errors
            self.ok = not errors

    def _fake(self, file_path):
        calls.append(file_path)
        text = Path(file_path).read_text(encoding="utf-8")
        return _Res([_Err(f'"{BROKEN}" is not defined')] if BROKEN in text else [])

    def _fake_batch(self, file_paths):
        # One entry per invocation — this is what a real cold start costs.
        spawns.append(list(file_paths))
        return {p: _fake(self, p) for p in file_paths}

    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_semantics", _fake)
    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_semantics_batch", _fake_batch)
    tool_registry._sem_calls = calls
    tool_registry._sem_spawns = spawns
    return tool_registry


def _edit(reg, old, new):
    return reg.dispatch("edit_text", {"file_path": "sample.py", "old_string": old, "new_string": new})


def _clean_syntax(self, file_path, content):
    """Stub for validate_syntax: always a clean verdict, never a false one.

    The drain-concurrency tests exercise the semantic batching/overlap path,
    not the syntax toolchain — without this the dispatch path would spawn a
    real tsc per edit and the gate would pay its contention for a property the
    test does not measure.
    """
    return type("R", (), {"ok": True, "errors": [], "language": None})()


def _syntax_check(result) -> dict:
    return (result.metadata or {}).get("syntax_check") or {}


# ── the regression: a later edit in the same turn must not go unreported ──


def test_the_last_edit_of_a_turn_is_the_one_reported(reg):
    """Edit 1 clean, edit 2 broken, same turn → the breakage must surface.

    Under first-write-wins this returned ``[]`` for edit 2 while pyright, run
    directly against the same file, reported the undefined name.
    """
    reg.begin_semantic_turn()
    assert _edit(reg, '    return "world"', '    return "world"  # ok').ok
    assert _edit(reg, "        return a + b", f"        return {BROKEN}(a, b)").ok

    drained = reg.drain_pending_semantic_checks()

    assert len(drained) == 1, "one file was written, so one check should have run"
    diags = next(iter(drained.values())).diagnostics
    assert [d["message"] for d in diags] == [f'"{BROKEN}" is not defined'], (
        "the check observed the file before the breaking edit"
    )


def test_a_turn_that_repairs_its_own_breakage_reports_clean(reg):
    """The mirror case: broken then fixed inside one turn → nothing to report.

    A last-write-wins rule has to be right in both directions; a rule that only
    ever reported the newest error would fail here.
    """
    reg.begin_semantic_turn()
    _edit(reg, "        return a + b", f"        return {BROKEN}(a, b)")
    _edit(reg, f"        return {BROKEN}(a, b)", "        return a + b")

    drained = reg.drain_pending_semantic_checks()
    assert next(iter(drained.values())).diagnostics == []


# ── the cost the coalescing exists to remove ──────────────────────────────


def test_many_writes_to_one_file_cost_one_check(reg):
    reg.begin_semantic_turn()
    for i in range(5):
        _edit(reg, '    return "world"' + "  #" * i, '    return "world"' + "  #" * (i + 1))
    assert reg._sem_calls == [], "checks must not run inline while a turn is open"

    reg.drain_pending_semantic_checks()
    assert len(reg._sem_calls) == 1, f"expected 1 toolchain spawn, got {len(reg._sem_calls)}"


def test_two_files_are_checked_in_one_spawn(reg, temp_repo_root):
    """Both files get checked, and the toolchain starts ONCE for the pair.

    Coalescing removed the repeat-write cost but left a turn touching N files
    paying N cold starts, which is most of the cost of a short check (pyright
    over 4 files: 2.167 s one-at-a-time vs 0.391 s batched). The per-file
    results must still be separate — a batched tool reports the whole set, so a
    naive split hands every file the same diagnostics.
    """
    Path(temp_repo_root, "other.py").write_text("def g():\n    return 1\n", encoding="utf-8")
    reg.begin_semantic_turn()
    _edit(reg, '    return "world"', '    return "world"  # a')
    reg.dispatch("edit_text", {"file_path": "other.py", "old_string": "    return 1", "new_string": "    return 2"})
    drained = reg.drain_pending_semantic_checks()
    assert len(drained) == 2
    assert len(reg._sem_calls) == 2, "each file must still be examined"
    assert len(reg._sem_spawns) == 1, f"expected the pair to share one toolchain spawn, got {reg._sem_spawns}"


def test_a_broken_file_does_not_taint_its_batch_partner(reg, temp_repo_root):
    """Batched diagnostics must be split per file, not copied across the batch.

    pyright returns one flat array for the whole invocation; attributing it
    wholesale would report the broken file's undefined name against the clean
    one too, and the model would go chasing an error that is not there.
    """
    Path(temp_repo_root, "other.py").write_text("def g():\n    return 1\n", encoding="utf-8")
    reg.begin_semantic_turn()
    _edit(reg, "        return a + b", f"        return {BROKEN}(a, b)")
    reg.dispatch("edit_text", {"file_path": "other.py", "old_string": "    return 1", "new_string": "    return 2"})

    drained = reg.drain_pending_semantic_checks()

    assert len(reg._sem_spawns) == 1, "precondition: the two files shared a spawn"
    broken = next(p for p in drained if p.endswith("sample.py"))
    clean = next(p for p in drained if p.endswith("other.py"))
    assert [d["message"] for d in drained[broken].diagnostics] == [f'"{BROKEN}" is not defined']
    assert drained[clean].diagnostics == [], "the clean file inherited its partner's diagnostics"


# ── deferral must never look like a clean check ───────────────────────────


def test_a_deferred_write_carries_no_diagnostics_key(reg):
    """``semantic_diagnostics: []`` on a deferred write reads as "checked, clean".

    ``_append_semantic_diagnostics`` treats an empty list and a missing key
    identically when rendering, so the only way to keep a deferred write from
    impersonating a verified one is to omit the key entirely.
    """
    reg.begin_semantic_turn()
    syn = _syntax_check(_edit(reg, '    return "world"', '    return "world"  # x'))
    assert syn.get("semantic_deferred") is True
    assert "semantic_diagnostics" not in syn


# ── callers outside the agent turn loop keep the inline behaviour ─────────


def test_without_an_open_turn_the_check_runs_inline(reg):
    """MCP server, design chat and direct dispatch never drain the queue.

    If deferral applied to them, their diagnostics would be queued and silently
    dropped, so no active turn must mean "check now".
    """
    syn = _syntax_check(_edit(reg, "        return a + b", f"        return {BROKEN}(a, b)"))
    assert syn.get("semantic_deferred") is None
    assert [d["message"] for d in syn.get("semantic_diagnostics") or []] == [f'"{BROKEN}" is not defined']
    assert len(reg._sem_calls) == 1


def test_drain_ends_the_turn_so_later_writes_run_inline(reg):
    reg.begin_semantic_turn()
    _edit(reg, '    return "world"', '    return "world"  # a')
    reg.drain_pending_semantic_checks()
    syn = _syntax_check(_edit(reg, '    return "world"  # a', '    return "world"  # b'))
    assert "semantic_diagnostics" in syn, "the turn ended, so this write is not deferred"


def test_end_semantic_turn_restores_inline_behaviour(reg):
    """An abandoned turn must not leave subsequent dispatches deferring.

    begin_semantic_turn opens coalescing. If the turn body is abandoned
    (uncaught exception / cancellation) the loop's ``finally`` calls
    end_semantic_turn so the "no active turn" invariant holds. Without it a
    later out-of-turn dispatch would defer into a queue nothing drains and
    silently drop its diagnostics.
    """
    reg.begin_semantic_turn()
    assert reg.defer_semantic_check("/x.py") is True, "defers while a turn is open"
    reg.end_semantic_turn()
    assert reg.defer_semantic_check("/y.py") is False, "after end_semantic_turn a dispatch must run inline again"
    # Pending entries from the abandoned turn are discarded, not carried over.
    reg.begin_semantic_turn()
    assert reg._semantic_pending == {}


# ── injection into the tool-result messages ───────────────────────────────


class _Pipeline(TurnPipelineMixin):
    def __init__(self, registry):
        self.registry = registry


class _Msg:
    def __init__(self, content):
        self.content = content


def _msg_for(result) -> _Msg:
    """The shape ``_build_tool_result_message`` produces: a JSON payload."""
    return _Msg(
        json.dumps(
            {
                "ok": result.ok,
                "content": result.content,
                "error": result.error,
                "metadata": dict(result.metadata or {}),
            },
            ensure_ascii=False,
        )
    )


def test_settle_fills_the_last_message_for_the_file(reg):
    reg.begin_semantic_turn()
    m1 = _msg_for(_edit(reg, '    return "world"', '    return "world"  # ok'))
    m2 = _msg_for(_edit(reg, "        return a + b", f"        return {BROKEN}(a, b)"))
    msgs = [m1, m2]

    _Pipeline(reg)._settle_deferred_semantics(msgs)

    syn2 = json.loads(m2.content)["metadata"]["syntax_check"]
    assert [d["message"] for d in syn2["semantic_diagnostics"]] == [f'"{BROKEN}" is not defined']
    assert "semantic_deferred" not in syn2

    # The superseded write must not gain an empty list — that is the false
    # "clean" signal again, one message earlier.
    syn1 = json.loads(m1.content)["metadata"]["syntax_check"]
    assert "semantic_diagnostics" not in syn1
    assert "semantic_deferred_path" not in syn1, "internal key must not reach the model"


def test_settle_renders_the_diagnostics_block_for_the_model(reg):
    """A coalesced check must reach the model through the SAME formatted
    ``<file_diagnostics>`` channel as an inline one.

    The inline path appends the block during ``_build_tool_result_message``,
    which runs BEFORE settle. Without a second render in settle the only place
    a deferred check's diagnostics appear is raw JSON in ``metadata`` — and the
    surrounding code documents the block as the channel the LLM parses
    reliably. Since EVERY in-turn write is deferred, omitting the block here
    would degrade the whole advisory channel for the common path.
    """
    reg.begin_semantic_turn()
    m = _msg_for(_edit(reg, "        return a + b", f"        return {BROKEN}(a, b)"))
    _Pipeline(reg)._settle_deferred_semantics([m])

    _content = json.loads(m.content)["content"]
    assert "<file_diagnostics>" in _content
    assert BROKEN in _content, "the diagnostic message must be in the rendered block"


def test_settle_adds_no_block_when_the_check_is_clean(reg):
    """A deferred check that comes back clean must not grow an empty block."""
    reg.begin_semantic_turn()
    m = _msg_for(_edit(reg, '    return "world"', '    return "world"  # ok'))
    _Pipeline(reg)._settle_deferred_semantics([m])

    _content = json.loads(m.content)["content"]
    assert "<file_diagnostics>" not in _content


def test_settle_is_a_noop_when_nothing_was_deferred(reg):
    msgs = [_Msg(json.dumps({"ok": True, "content": "x", "metadata": {}}))]
    before = msgs[0].content
    _Pipeline(reg)._settle_deferred_semantics(msgs)
    assert msgs[0].content == before


def test_settle_survives_a_malformed_message(reg):
    """A non-JSON tool message must not break the turn."""
    reg.begin_semantic_turn()
    good = _msg_for(_edit(reg, '    return "world"', '    return "world"  # ok'))
    msgs = [_Msg("not json at all: semantic_deferred"), good]
    _Pipeline(reg)._settle_deferred_semantics(msgs)
    assert "semantic_diagnostics" in json.loads(good.content)["metadata"]["syntax_check"]


# ── two languages in one turn: two toolchains, not two waits ───────────────


def test_provider_groups_run_concurrently(tool_registry, temp_repo_root, monkeypatch):
    """A .py and a .ts edited in one turn must not pay the SUM of both checks.

    Each group is its own process — pyright, npx tsc, go build — sharing
    nothing but the drain that waits on them. Measured with the real
    toolchains on one file each: 0.424 s + 0.804 s = 1.228 s serial against a
    0.804 s parallel bound.

    Asserted as OVERLAP, not as wall time: each stub announces its arrival and
    then waits for the other's. Only a concurrent drain lets both waits
    succeed — run serially, the first group waits out its timeout alone and
    records nothing, and only the second finds a partner already there. A pure
    timing assertion would be a flake generator on a loaded machine.

    "Both groups ran" is NOT the discriminator and was the first version's
    mistake: a serial drain also enters both, one after the other.
    """
    import threading

    from external_llm.languages import python_provider as pp
    from external_llm.languages import typescript_provider as tp

    arrived = {"py": threading.Event(), "ts": threading.Event()}
    entered: list[str] = []
    overlapped: list[str] = []

    class _Res:
        errors: ClassVar[list] = []
        ok = True

    def _make(tag):
        other = "ts" if tag == "py" else "py"

        def _batch(self, file_paths):
            entered.append(tag)
            arrived[tag].set()
            if arrived[other].wait(timeout=3):
                overlapped.append(tag)
            return {p: _Res() for p in file_paths}

        return _batch

    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_semantics_batch", _make("py"))
    monkeypatch.setattr(tp.TypeScriptSyntaxProvider, "validate_semantics_batch", _make("ts"))

    # The dispatch path also runs an immediate SYNTAX check (validate_syntax),
    # which would spawn a real tsc per edit. This test is about the drain
    # overlapping its semantic groups, not about the syntax toolchain, so stub
    # that too — otherwise the gate pays a real tsc spawn (and its contention)
    # for a property this test does not exercise. Same shape as the `reg`
    # fixture's stub: never a false verdict.
    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_syntax", _clean_syntax)
    monkeypatch.setattr(tp.TypeScriptSyntaxProvider, "validate_syntax", _clean_syntax)

    Path(temp_repo_root, "app.ts").write_text("export const a = 1;\n", encoding="utf-8")
    tool_registry.begin_semantic_turn()
    _edit(tool_registry, '    return "world"', '    return "world"  # a')
    tool_registry.dispatch(
        "edit_text",
        {
            "file_path": "app.ts",
            "old_string": "export const a = 1;",
            "new_string": "export const a = 2;",
        },
    )

    drained = tool_registry.drain_pending_semantic_checks()

    assert sorted(entered) == ["py", "ts"], f"both provider groups must run; entered={entered}"
    assert sorted(overlapped) == ["py", "ts"], (
        f"the two toolchains did not overlap — drain ran them serially; overlapped={overlapped}"
    )
    assert len(drained) == 2
    assert all(v.diagnostics == [] and v.checked for v in drained.values())


def test_one_provider_group_never_touches_the_pool(reg, monkeypatch):
    """The common single-language turn must not pay scheduling overhead."""
    from external_llm.agent import tool_registry as tr_mod

    submitted: list = []
    _real_submit = tr_mod.shared_pool.submit

    def _spy(fn, *a, **kw):
        submitted.append(fn)
        return _real_submit(fn, *a, **kw)

    monkeypatch.setattr(tr_mod.shared_pool, "submit", _spy, raising=True)

    reg.begin_semantic_turn()
    _edit(reg, '    return "world"', '    return "world"  # a')
    reg.drain_pending_semantic_checks()

    assert submitted == [], "a single provider group must run inline"


def test_one_group_failing_does_not_cost_the_other_its_diagnostics(
    tool_registry,
    temp_repo_root,
    monkeypatch,
):
    """A provider that raises is caught per group — including in a future."""
    from external_llm.languages import python_provider as pp
    from external_llm.languages import typescript_provider as tp

    class _Err:
        line, col = 1, 1
        message = "boom"
        severity, code = "error", "x"

    class _Res:
        def __init__(self):
            self.errors = [_Err()]
            self.ok = False

    def _ok(self, file_paths):
        return {p: _Res() for p in file_paths}

    def _raises(self, file_paths):
        raise RuntimeError("toolchain exploded")

    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_semantics_batch", _ok)
    monkeypatch.setattr(tp.TypeScriptSyntaxProvider, "validate_semantics_batch", _raises)

    # Same as test_provider_groups_run_concurrently: the dispatch path's
    # immediate syntax check would spawn a real tsc per edit. This test is
    # about per-group failure isolation in the drain, not the syntax toolchain.
    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_syntax", _clean_syntax)
    monkeypatch.setattr(tp.TypeScriptSyntaxProvider, "validate_syntax", _clean_syntax)

    Path(temp_repo_root, "app.ts").write_text("export const a = 1;\n", encoding="utf-8")
    tool_registry.begin_semantic_turn()
    _edit(tool_registry, '    return "world"', '    return "world"  # a')
    tool_registry.dispatch(
        "edit_text",
        {
            "file_path": "app.ts",
            "old_string": "export const a = 1;",
            "new_string": "export const a = 2;",
        },
    )

    drained = tool_registry.drain_pending_semantic_checks()

    py_entry = next(v for k, v in drained.items() if k.endswith("sample.py"))
    ts_entry = next(v for k, v in drained.items() if k.endswith("app.ts"))
    assert [d["message"] for d in py_entry.diagnostics] == ["boom"], (
        "the healthy group lost its diagnostics to the failing one"
    )
    assert not ts_entry.checked, "a failed group reports a skip, never raises"
    assert ts_entry.diagnostics == []


# ── "nothing checked it" must not arrive looking like "checked, clean" ──────


@pytest.fixture
def unavailable(tool_registry, monkeypatch):
    """Registry whose Python semantic checker is simply not installed."""
    from external_llm.languages import python_provider as pp
    from external_llm.languages.models import LanguageId, SyntaxValidationResult

    def _absent(self, file_paths):
        return {
            p: SyntaxValidationResult.unchecked(
                LanguageId.PYTHON,
                "pyright is not installed",
            )
            for p in file_paths
        }

    monkeypatch.setattr(pp.PythonSyntaxProvider, "validate_semantics_batch", _absent)
    return tool_registry


def test_a_missing_toolchain_drains_as_a_skip_not_as_clean(unavailable):
    """The whole point: no pyright must not read as a passing check.

    A `pip install asicode` on a machine with no node has exactly this shape,
    so this is the default experience rather than an edge case.
    """
    unavailable.begin_semantic_turn()
    _edit(unavailable, '    return "world"', '    return "world"  # a')

    outcome = next(iter(unavailable.drain_pending_semantic_checks().values()))

    assert outcome.checked is False
    assert outcome.skip_reason == "pyright is not installed"
    assert outcome.diagnostics == []


def test_settle_tells_the_model_the_check_was_skipped(unavailable):
    unavailable.begin_semantic_turn()
    m = _msg_for(_edit(unavailable, '    return "world"', '    return "world"  # a'))

    _Pipeline(unavailable)._settle_deferred_semantics([m])

    payload = json.loads(m.content)
    syn = payload["metadata"]["syntax_check"]
    assert syn.get("semantic_check_skipped") == "pyright is not installed"
    assert "semantic_diagnostics" not in syn, "an empty diagnostics list is exactly the false 'clean' signal"
    # The internal markers must never ship, skipped or not.
    assert "semantic_deferred" not in syn
    assert "semantic_deferred_path" not in syn
    assert "<file_diagnostics>" not in (payload.get("content") or ""), "a skipped check has nothing to render"


def test_a_genuinely_clean_check_is_still_reported_as_checked(reg):
    """The mirror: a check that RAN and found nothing keeps saying so."""
    reg.begin_semantic_turn()
    _edit(reg, '    return "world"', '    return "world"  # a')
    outcome = next(iter(reg.drain_pending_semantic_checks().values()))
    assert outcome.checked is True
    assert outcome.diagnostics == []
    assert outcome.skip_reason == ""


def test_a_file_with_no_semantic_provider_is_a_skip(tool_registry, temp_repo_root):
    """A .md write is not "clean" either — nothing was ever going to check it."""
    Path(temp_repo_root, "notes.md").write_text("# hi\n", encoding="utf-8")
    tool_registry.begin_semantic_turn()
    # Deferral only happens for files whose provider HAS a semantic validator,
    # so reach the queue directly — the seeded default is what is under test.
    tool_registry.defer_semantic_check(str(Path(temp_repo_root, "notes.md")))
    outcome = next(iter(tool_registry.drain_pending_semantic_checks().values()))
    assert outcome.checked is False
    assert outcome.skip_reason


def test_drain_cancel_event_skips_pending_groups(reg, monkeypatch, tmp_path):
    """ESC during the deferred drain must stop waiting on the in-flight
    toolchain: the still-pending group is marked skipped and the drain returns
    promptly instead of blocking the turn end on the slow provider.

    Regression: future.result() had no timeout, so a slow second toolchain
    (tsc/go cold start) blocked turn end even after ESC — the drain is
    advisory and must never delay a cancelled turn."""
    import threading
    import time as _time

    from external_llm.languages.registry import LanguageRegistry

    cancel_event = threading.Event()
    reg.config.cancel_event = cancel_event

    # Two providers: the FIRST group always runs inline on the drain thread
    # (fast), the second is scheduled on the pool and blocks — the cancel must
    # cut that wait short, not outlast it.
    blocked = threading.Event()
    calls: list[str] = []

    class _Cap:
        has_semantic_validator = True

    class _Res:
        def __init__(self, errors):
            self.errors = errors

    class _FastProvider:
        def capabilities(self):
            return _Cap()

        def validate_semantics_batch(self, paths):
            calls.append("fast")
            return {p: _Res([]) for p in paths}

    class _SlowProvider:
        def capabilities(self):
            return _Cap()

        def validate_semantics_batch(self, paths):
            calls.append("slow")
            blocked.wait(10)  # must be abandoned, not waited out
            return {p: _Res([]) for p in paths}

    fast_file = tmp_path / "fast.py"
    slow_file = tmp_path / "slow.ts"
    fast_file.write_text("x = 1")
    slow_file.write_text("const x = 1;")

    def _fake_get(abs_path):
        return _FastProvider() if abs_path == str(fast_file) else _SlowProvider()

    monkeypatch.setattr(LanguageRegistry.instance(), "get", _fake_get)

    reg.begin_semantic_turn()
    reg.defer_semantic_check(str(fast_file))
    reg.defer_semantic_check(str(slow_file))

    drained: dict = {}

    def _drain():
        drained.update(reg.drain_pending_semantic_checks())

    t = threading.Thread(target=_drain)
    t.start()
    _time.sleep(0.3)  # let the slow provider start and block in the pool
    cancel_event.set()  # user presses ESC
    t.join(timeout=5)
    assert not t.is_alive(), "drain did not return after cancel"
    assert "fast" in calls, "first (inline) group should still have run"
    assert drained[str(fast_file)].checked
    assert drained[str(slow_file)].skip_reason == ("cancelled before the semantic check ran"), (
        f"pending group must be skipped on cancel, got {drained[str(slow_file)].skip_reason!r}"
    )
    # Release the still-blocked pool worker so the shared pool can wind down
    # without keeping the interpreter alive for the full wait.
    blocked.set()
