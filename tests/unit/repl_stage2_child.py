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
    ns = p.parse_args()

    import tempfile

    import coverage

    data_file = os.environ.get("COVERAGE_FILE") or os.path.join(
        tempfile.gettempdir(), f"covstage2-{os.getpid()}")
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
        repl_impl._check_clipboard_image = lambda: []
        repl_impl._copy_to_clipboard = lambda text: "pbcopy"  # no real clipboard

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
        repl_impl._resolve_model_interactive = (
            lambda arg, usage_hint="": (ns.provider, arg.split("/")[-1]))

        # Structural deps: registry/session/loops all faked (no RAG/embedding).
        tool_registry_mod.ToolRegistry = FakeToolRegistry
        tool_registry_mod.AgentConfig = FakeAgentConfig
        design_session_mod.DesignSessionManager = FakeDSM
        dcl_mod.DesignChatLoop = FakeDesignChatLoop
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
            repo=ns.repo, provider=ns.provider, model=ns.model,
            api_key=None, no_deps_check=True, verbose=False,
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
        with contextlib.suppress(Exception):
            cov.save()
    return rc


if __name__ == "__main__":
    os._exit(_main())
