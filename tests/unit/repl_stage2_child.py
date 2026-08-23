"""Child driver for the stage-2 REPL pty tests (spawned via SpawnPtySession).

Runs ``repl_impl._run_repl_impl`` in a dedicated main thread with a real pty
as stdin/stdout/stderr, re-applying the same fakes the in-process worker tests
use (exec does not inherit monkeypatches). Writes its own coverage data file
before ``os._exit`` (bypassing atexit), so ``coverage combine`` merges it.

Exit contract: 0 on a clean ``exit``/Ctrl+C session end, 1 with a printed
``CHILD-CRASH`` marker on any exception (the parent surfaces the pty tail).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # repo root: import asi / external_llm / tests.unit


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="repl", choices=("repl",))
    p.add_argument("--repo", required=True)
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--collab-sdk", default="installed", choices=("installed", "missing"))
    p.add_argument("--collab-orch", default="none", choices=("none", "raise", "keyboard"))
    p.add_argument(
        "--collab-result-error", action="store_true", help="collab run returns a result whose .error is truthy"
    )
    p.add_argument("--collab-handoff-raise", action="store_true")
    p.add_argument("--collab-verdict-raise", action="store_true")
    p.add_argument(
        "--collab-install",
        default="ok",
        choices=("ok", "fail", "kb"),
        help="_pip_install outcome when SDK is missing and user says y",
    )
    p.add_argument(
        "--next-suggest-off",
        action="store_true",
        help="simulate display.NEXT_SUGGEST disabled (ASICODE_NEXT_SUGGEST=0)",
    )
    p.add_argument("--auto-suggest-text", default="", help="make the next-suggestion LLM reply this text (non-NONE)")
    p.add_argument("--error-turn", action="store_true", help="DesignChatLoop fail_mode=error -> chat_result.is_error")
    p.add_argument(
        "--checkpoint-fail",
        action="store_true",
        help="_newest_checkpoint_id raises on the TURN-END call -> change-summary except",
    )
    p.add_argument(
        "--checkpoint-seq",
        action="store_true",
        help="_newest_checkpoint_id returns cp0 at turn start, cp1 at turn end + changed files",
    )
    p.add_argument(
        "--suggest-kick-fail",
        action="store_true",
        help="_kick_next_prompt_suggestion raises -> next-suggestion kick except branch",
    )
    p.add_argument(
        "--clipboard-image", action="store_true", help="_check_clipboard_image returns an image -> clip path"
    )
    p.add_argument(
        "--force-underline", action="store_true", help="_input_underline stays True -> auto-submit countdown can fire"
    )
    ns = p.parse_args()

    import tempfile

    import coverage

    # The child may already be under coverage: `coverage run -m pytest`
    # sets COVERAGE_PROCESS_START, and coverage's multiprocessing/sitecustomize
    # hook starts an instance before our code runs. Starting a SECOND instance
    # here double-instruments the process, and its data file can end up
    # malformed (observed: "database disk image is malformed" for the child's
    # suffixed file, which then fails to combine). Reuse the active instance
    # when present; only start our own otherwise.
    data_file = os.environ.get("COVERAGE_FILE") or os.path.join(tempfile.gettempdir(), f"covstage2-{os.getpid()}")
    cov = coverage.Coverage.current()
    if cov is None:
        cov = coverage.Coverage(data_file=data_file)
        cov.start()
    try:
        import asi
        import external_llm.agent.design_chat_loop as dcl_mod
        import external_llm.agent.orchestrator as orch_mod
        import external_llm.agent.tool_registry as tool_registry_mod
        import external_llm.client as client_mod
        import external_llm.design_session as design_session_mod
        import external_llm.intelligent_service as isvc_mod
        from external_llm.repl import repl_impl
        from tests.unit.repl_stage2_fakes import (
            FakeAgentConfig,
            FakeClient,
            FakeDesignChatLoop,
            FakeDSM,
            FakeOrchAgent,
            FakeSvc,
            FakeToolRegistry,
        )

        # Plain output paths: rich console stays uninitialized, _print/_prompt
        # fall back to print(), banner uses the non-rich branch ("  ▌ asicode").
        repl_impl._RICH = False
        asi._out_console = None
        asi._margin_stderr = None
        repl_impl._ensure_out_console_imported = lambda: None
        asi._ensure_out_console_imported = lambda: None
        # patch_stdout shows a "Press ENTER to continue..." overlay whenever a
        # background write (e.g. a one-time INFO/WARNING) lands during the
        # prompt, and the overlay SWALLOWS the next Enter (the driver's submit
        # keystroke) — a startup-timing race. With no overlay, background
        # writes just interleave into the pty stream (drained by the parent).
        import contextlib

        asi.patch_stdout = contextlib.nullcontext

        # No-op gates that would otherwise hit the network / embedding model.
        repl_impl._print_dep_status = lambda root, **k: None
        repl_impl._kick_embedding_model_warmup = lambda: None
        repl_impl._get_ollama_models = lambda timeout=5: []
        repl_impl._copy_to_clipboard = lambda text: "pbcopy"  # no real clipboard
        if ns.clipboard_image:
            repl_impl._check_clipboard_image = lambda: [{"media_type": "image/png", "data": "aGVsbG8="}]
        else:
            repl_impl._check_clipboard_image = lambda: []

        # ── /claude collaboration: replace the whole collaborate module ──
        import types as _types

        from tests.unit.repl_stage2_fakes import (
            FakeCollabInstallState,
            FakeCollabModule,
            FakeStreamingDisplayModule,
        )

        _collab_state = FakeCollabInstallState(sdk_installed=(ns.collab_sdk == "installed"))
        _collab_mod = FakeCollabModule(
            _collab_state,
            orch_fail_mode=ns.collab_orch,
            result_error=(RuntimeError("fake err") if ns.collab_result_error else None),
        )
        _collab_mod.handoff_raise = ns.collab_handoff_raise
        _collab_mod.verdict_raise = ns.collab_verdict_raise
        _display_mod = FakeStreamingDisplayModule()

        _collab_holder = _types.ModuleType("external_llm.repl.collaborate")
        for _attr in (
            "is_claude_sdk_installed",
            "build_collaborate_install_spec",
            "build_session_handoff",
            "format_verdict_for_session",
            "CollaborationOrchestratorConfig",
            "CollaborationOrchestrator",
        ):
            setattr(_collab_holder, _attr, getattr(_collab_mod, _attr))
        _collab_holder.DEFAULT_COLLAB_MODEL = _collab_mod.DEFAULT_COLLAB_MODEL
        sys.modules["external_llm.repl.collaborate"] = _collab_holder

        _disp_holder = _types.ModuleType("external_llm.repl.collaborate.streaming_display")
        _disp_holder.StreamingDisplay = _display_mod.StreamingDisplay
        sys.modules["external_llm.repl.collaborate.streaming_display"] = _disp_holder

        # _pip_install must never touch the real environment in tests.
        if ns.collab_install == "kb":

            def _pip_kb(spec, label=None):
                raise KeyboardInterrupt()

            repl_impl._pip_install = _pip_kb
        elif ns.collab_install == "fail":
            repl_impl._pip_install = lambda spec, label=None: False
        else:

            def _pip_ok(spec, label=None):
                # Successful install flips the SDK availability gate.
                _collab_state.sdk_installed = True
                return True

            repl_impl._pip_install = _pip_ok

        if ns.next_suggest_off:
            # NOTE: display is a frozen dataclass — setattr raises
            # FrozenInstanceError. Tests instead pass ASICODE_NEXT_SUGGEST=0
            # via the spawn env so thresholds parses it at import time.
            pass

        if ns.auto_suggest_text:
            # _kick_next_prompt_suggestion resolves the reply via
            # _validate_next_suggestion (same-language guard): a Hangul
            # request needs a Hangul suggestion. Return the flag text.
            _sug_text = ns.auto_suggest_text

            def _fake_suggestion(llm_client, model, user_request, final_message, digest, auto_mode=False):
                # Deliver on a slight delay so the next prompt is live when the
                # suggestion lands (matches the real background-thread timing) —
                # _deliver_next_suggestion only arms the countdown while the app
                # is running.
                import threading as _th
                import time as _tm

                def _deliver():
                    _tm.sleep(0.6)
                    repl_impl._deliver_next_suggestion(_sug_text, repl_impl._next_suggestion_gen)

                _th.Thread(target=_deliver, daemon=True).start()

            repl_impl._kick_next_prompt_suggestion = _fake_suggestion

        if ns.suggest_kick_fail:

            def _kick_fail(*a, **k):
                raise RuntimeError("fake suggestion kick crash")

            repl_impl._kick_next_prompt_suggestion = _kick_fail

        if ns.checkpoint_seq:
            _cp_calls = {"n": 0}

            def _cp_seq(repo_root):
                _cp_calls["n"] += 1
                return "cp0" if _cp_calls["n"] == 1 else "cp1"

            repl_impl._newest_checkpoint_id = _cp_seq
            repl_impl._checkpoint_changed_files = lambda repo_root, checkpoint_id: ["a.txt", "b.txt"]

        if ns.checkpoint_fail:
            _cpf_calls = {"n": 0}

            def _cp_fail(repo_root):
                # Turn-start call (baseline) must succeed; the turn-end call
                # raises so the change-summary except branch is taken.
                _cpf_calls["n"] += 1
                if _cpf_calls["n"] == 1:
                    return "cp0"
                raise OSError("fake checkpoint read failure")

            repl_impl._newest_checkpoint_id = _cp_fail

        if ns.force_underline:
            # _collect_input resets the global to bool(bottom_toolbar) on every
            # prompt (L2943), so a one-time set is useless. Wrap it to force the
            # underline path: _input_underline=True enables the ghost seeding
            # (pre_run -> _maybe_arm_auto_submit) that arms the countdown Timer.
            _orig_collect_input = repl_impl._collect_input

            def _collect_input_underline(prompt, bottom_toolbar=False):
                return _orig_collect_input(prompt, bottom_toolbar=True)

            repl_impl._collect_input = _collect_input_underline

        # Disable the slash-command completer: with complete_while_typing the
        # completion-menu float renders "/clear" (etc.) as soon as "/c" is
        # typed, racing the driver's echo-wait, and Enter with the menu open
        # can accept the completion instead of submitting the line. The
        # completer itself is already covered by direct-call tests.
        # Must subclass Completer: the async path calls
        # get_completions_async (default impl yields from get_completions);
        # a plain duck-typed class raises AttributeError inside the event
        # loop, whose handler then shows a "Press ENTER to continue..."
        # overlay that swallows the driver's submit keystroke.
        from prompt_toolkit.completion import Completer as _PtCompleter

        class _NoCompleter(_PtCompleter):
            def __init__(self, **kw):
                super().__init__()

            def get_completions(self, document, complete_event):
                return iter(())

        repl_impl._SlashCommandCompleter = _NoCompleter

        # Service layer: fake LLM svc + model resolution (no interactive prompt).
        def _make_svc(*a, **k):
            return FakeSvc(provider=ns.provider, model=ns.model)

        repl_impl._retry_create_svc_with_api_key_prompt = _make_svc
        isvc_mod.create_intelligent_service_from_env = _make_svc
        repl_impl._resolve_model_interactive = lambda arg, usage_hint="": (ns.provider, arg.split("/")[-1])

        # Structural deps: registry/session/loops all faked (no RAG/embedding).
        tool_registry_mod.ToolRegistry = FakeToolRegistry
        tool_registry_mod.AgentConfig = FakeAgentConfig
        design_session_mod.DesignSessionManager = FakeDSM
        dcl_mod.DesignChatLoop = lambda client, registry, model: (
            FakeDesignChatLoop(client, registry, model, fail_mode="error_result")
            if ns.error_turn
            else FakeDesignChatLoop(client, registry, model)
        )
        orch_mod.OrchestratorAgent = FakeOrchAgent
        client_mod.create_llm_client = lambda *a, **k: FakeClient()

        # The orchestrator-turn setup calls dataclasses.replace(design_config,
        # cancel_event=..., stream_callback=..., _user_checkpoint_count=0) on
        # the AgentConfig FAKE (a plain class — replace() would raise
        # TypeError). Fall back to shallow copy + setattr for non-dataclasses.
        import copy as _copy_mod
        import dataclasses as _dc_mod

        def _replace_fallback(inst, **changes):
            if _dc_mod.is_dataclass(inst):
                return _dc_mod.replace(inst, **changes)
            out = _copy_mod.copy(inst)
            for _k, _v in changes.items():
                setattr(out, _k, _v)
            return out

        _dc_mod.replace = _replace_fallback

        args = argparse.Namespace(
            repo=ns.repo,
            provider=ns.provider,
            model=ns.model,
            api_key=None,
            no_deps_check=True,
            verbose=False,
        )
        repl_impl._run_repl_impl(args)
    except SystemExit as e:
        rc = int(getattr(e, "code", 0) or 0)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.write("\nCHILD-CRASH\n")
        sys.stdout.flush()
        rc = 1
    else:
        rc = 0
    finally:
        # Coverage data must be flushed BEFORE os._exit (which bypasses
        # atexit) — saved on every exit path, exceptions included.
        try:
            cov.save()
        except BaseException:
            import traceback as _tb

            sys.stderr.write("\nCOV-SAVE-FAIL\n")
            _tb.print_exc()
            sys.stderr.flush()
    return rc


if __name__ == "__main__":
    os._exit(_main())
