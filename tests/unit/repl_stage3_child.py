"""Stage-3 child driver for the REPL pty tests (spawned via SpawnPtySession).

Same contract as ``repl_stage2_child.py`` (runs ``_run_repl_impl`` in a real
pty, re-applies the fakes, saves its own coverage data before ``os._exit``)
but lives in its OWN file so the parallel session's in-flight edits to the
stage-2 files are never touched. Adds stage-3 scenario flags that drive the
residual interactive-command branches:

  startup:  --sweep-ok / --sweep-raise / --init-none
  helper:   --helper-client-fail
  models:   --ollama-models / --api-key-env / --no-api-key
  undo:     --undo-cp
  chat:     --ocr-text / --empty-response / --verify-fail
  compact:  --compact-fail / --insights-big / --nudge
  patterns: --fp-data / --fp-empty
  misc:     --preamble-only / --archive-data / --git-repo / --auth-error-turn
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import ClassVar

sys.path.insert(0, os.getcwd())  # repo root: import asi / external_llm / tests.unit


def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="repl", choices=("repl",))
    p.add_argument("--repo", required=True)
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-sonnet-4-6")
    # startup
    p.add_argument("--sweep-ok", action="store_true", help="stale-lock sweep returns 1 -> success log branch")
    p.add_argument("--sweep-raise", action="store_true", help="stale-lock sweep raises -> except branch")
    p.add_argument("--init-none", action="store_true", help="_init_repl_engine returns None -> early return")
    # helper / models
    p.add_argument(
        "--helper-client-fail",
        action="store_true",
        help="_create_llm_client_for returns None -> HELPER_CREATION_FAILED",
    )
    p.add_argument("--ollama-models", action="store_true", help="_get_ollama_models returns a non-empty list")
    p.add_argument(
        "--api-key-env", action="store_true", help="pre-set OPENAI_API_KEY so /model openai/... skips the prompt"
    )
    p.add_argument(
        "--no-api-key", action="store_true", help="leave OPENAI_API_KEY unset -> /model openai/... hits the key prompt"
    )
    # undo / chat
    p.add_argument("--undo-cp", action="store_true", help="checkpoint undo path: newest cp + changed files + revert ok")
    p.add_argument("--ocr-text", action="store_true", help="_images_to_text returns text -> OCR enrichment branch")
    p.add_argument(
        "--empty-response", action="store_true", help="DesignChatLoop returns empty content -> empty-response branch"
    )
    p.add_argument("--verify-fail", action="store_true", help="verify DesignChatLoop returns is_error result")
    p.add_argument(
        "--auth-error-turn", action="store_true", help="turn result is_error + error_type=auth -> API-key retry path"
    )
    # compact / insights
    p.add_argument(
        "--compact-fail", action="store_true", help="insights-compact LLM call raises -> failure notice branch"
    )
    p.add_argument(
        "--insights-big", action="store_true", help="write a >COMPACT_BUDGET_BYTES insights file (noop over-budget)"
    )
    p.add_argument(
        "--nudge", action="store_true", help="write a >NUDGE_BYTES_THRESHOLD insights file -> session-end nudge"
    )
    # failure-patterns
    p.add_argument("--fp-data", action="store_true", help="failure-pattern store starts non-empty")
    p.add_argument("--fp-empty", action="store_true", help="failure-pattern store starts empty")
    # misc
    p.add_argument("--preamble-only", action="store_true", help="insights file has preamble but no entries")
    p.add_argument(
        "--no-insights", action="store_true", help="do NOT create a design_insights.md fixture (no-file branch)"
    )
    p.add_argument("--archive-data", action="store_true", help="create design_insights_archive.md with entries")
    p.add_argument("--git-repo", action="store_true", help="git init + initial commit -> baseline /undo path")
    # stage-3 second wave
    p.add_argument("--helper-persisted", action="store_true", help="pre-seed config.json helper_provider/helper_model")
    p.add_argument("--old-entries", action="store_true", help="insights file includes entries older than 21 days")
    p.add_argument("--empty-insights", action="store_true", help="insights file exists but is empty")
    p.add_argument(
        "--suggest-kick-fail", action="store_true", help="_kick_next_prompt_suggestion raises -> kick except branch"
    )
    p.add_argument("--resolve-none", action="store_true", help="_resolve_model_interactive returns None")
    p.add_argument(
        "--clipboard-image", action="store_true", help="_check_clipboard_image returns an image -> clip path"
    )
    p.add_argument("--dc-crash", action="store_true", help="DesignChatLoop raises -> design chat error branch")
    p.add_argument("--no-result", action="store_true", help="DesignChatLoop returns None -> no-result branch")
    p.add_argument(
        "--cache-tokens", action="store_true", help="DesignChatResult carries cache_read_tokens -> cache hit pct group"
    )
    p.add_argument("--notify", action="store_true", help="background-compress notify callback fires -> deferred notify")
    p.add_argument("--undo-baseline", action="store_true", help="force baseline undo path (git-style change stats)")
    # stage-3 third wave: compact edge modes + misc
    p.add_argument(
        "--compact-mode",
        default="",
        choices=(
            "",
            "truncated",
            "truncated-final",
            "empty",
            "drops-all",
            "noop",
            "big",
            "tokens",
            "fr",
            "kb",
            "partial",
        ),
        help="insights-compact LLM reply mode",
    )
    p.add_argument(
        "--reasoning-model", action="store_true", help="use a reasoning-class model name -> max_tokens raise"
    )
    p.add_argument("--bak-dir", action="store_true", help="pre-create insights .bak as a DIRECTORY -> backup OSError")
    p.add_argument("--notice-fail", action="store_true", help="_compress_failure_notice raises -> last-resort notice")
    p.add_argument(
        "--save-insight", action="store_true", help="turn records a save_insight tool call -> auto-compact gate"
    )
    p.add_argument(
        "--ctx-low", action="store_true", help="_resolve_context_limit returns 100 -> /general compress gate"
    )
    p.add_argument("--verbose", action="store_true", help="args.verbose=True -> traceback print branches")
    p.add_argument("--copy-fail", action="store_true", help="_copy_to_clipboard returns None -> copy failure branch")
    p.add_argument("--dsm-crash", action="store_true", help="DesignSessionManager.get_or_create raises")
    p.add_argument("--arch-fail", action="store_true", help="load_archive_file raises -> archive count except")
    p.add_argument("--fp-race", action="store_true", help="drop_key returns None -> disappeared-before-drop branch")
    p.add_argument("--fp-error", action="store_true", help="get_store raises -> failure-pattern store error branch")
    p.add_argument("--undo-cp-empty", action="store_true", help="checkpoint changed files is EMPTY -> nothing-to-undo")
    p.add_argument("--undo-cp-fail", action="store_true", help="_undo_via_checkpoint returns False -> revert-failed")
    p.add_argument("--undo-base-fail", action="store_true", help="_undo_run_changes returns failures -> partial revert")
    p.add_argument(
        "--llm-client-fail", action="store_true", help="create_llm_client raises -> /model client failure rollback"
    )
    p.add_argument("--ctx-fail", action="store_true", help="_resolve_context_limit raises -> context-limit except")
    p.add_argument("--ocr-fail", action="store_true", help="_images_to_text raises -> OCR enrichment except")
    p.add_argument(
        "--print-summary-fail", action="store_true", help="_print_run_change_summary raises -> turn summary except"
    )
    p.add_argument(
        "--cp-summary-fail", action="store_true", help="_newest_checkpoint_id raises at turn end -> cp summary except"
    )
    p.add_argument("--orch-fail", action="store_true", help="OrchestratorAgent.run raises -> orchestrator error branch")
    p.add_argument(
        "--orch-none", action="store_true", help="OrchestratorAgent.run returns None -> no-result turn close"
    )
    p.add_argument("--nudge-fail", action="store_true", help="should_nudge raises -> session-end nudge except")
    p.add_argument("--verify-crash", action="store_true", help="verify DesignChatLoop raises -> verify except")
    p.add_argument(
        "--compact-echo", action="store_true", help="build_compact_messages returns the raw file -> true noop"
    )
    p.add_argument(
        "--margin-stderr", action="store_true", help="give asi._margin_stderr a fake margin -> /clear reset_bol"
    )
    p.add_argument("--entries-3", action="store_true", help="insights fixture has 3 entries (partial-loss guard)")
    p.add_argument("--ebd-fail", action="store_true", help="enforce_budget_by_demotion raises -> backstop except")
    p.add_argument("--ebd-zero", action="store_true", help="enforce_budget_by_demotion returns (0, size) -> warn")
    p.add_argument("--stats-fail", action="store_true", help="compute_stats raises -> auto-compact stats except")
    p.add_argument(
        "--dsm-turn-crash", action="store_true", help="DesignSessionManager.add_turn raises -> turn close except"
    )
    p.add_argument("--orch-kb", action="store_true", help="OrchestratorAgent.run raises KeyboardInterrupt -> cancelled")
    p.add_argument("--auth-retry-ok", action="store_true", help="auth-error turn, retry succeeds -> commit key")
    p.add_argument(
        "--auth-retry-crash", action="store_true", help="auth-error turn, retry raises -> retry-failed result"
    )
    p.add_argument(
        "--nudge-plain", action="store_true", help="should_nudge returns a message without '/insights' split"
    )
    p.add_argument("--input-eof", action="store_true", help="builtins.input raises EOFError -> EOF branches")
    p.add_argument("--verify-kb", action="store_true", help="verify DesignChatLoop raises KeyboardInterrupt")
    p.add_argument("--rich", action="store_true", help="enable the rich console render paths")
    ns = p.parse_args()

    import tempfile

    import coverage

    data_file = os.environ.get("COVERAGE_FILE") or os.path.join(tempfile.gettempdir(), f"covstage3-{os.getpid()}")
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
            COMPACTED_INSIGHTS,
            FakeAgentConfig,
            FakeClient,
            FakeDesignChatLoop,
            FakeDSM,
            FakeOrchAgent,
            FakeSvc,
            FakeToolRegistry,
        )

        repl_impl._RICH = False
        asi._out_console = None
        asi._margin_stderr = None
        repl_impl._ensure_out_console_imported = lambda: None
        asi._ensure_out_console_imported = lambda: None
        import contextlib

        asi.patch_stdout = contextlib.nullcontext

        # No-op gates that would otherwise hit the network / embedding model.
        repl_impl._print_dep_status = lambda root, **k: None
        repl_impl._kick_embedding_model_warmup = lambda: None
        repl_impl._copy_to_clipboard = (lambda text: None) if ns.copy_fail else (lambda text: "pbcopy")
        repl_impl._check_clipboard_image = lambda: []

        if ns.margin_stderr:

            class _FakeMargin:
                def reset_bol(self):
                    pass

            asi._margin_stderr = _FakeMargin()

        if ns.input_eof:
            import builtins as _builtins

            _builtins.input = lambda *a, **k: (_ for _ in ()).throw(EOFError())

        if ns.rich:
            from rich.console import Console as _RichConsole

            class _RichMargin:
                def __init__(self):
                    self.console = _RichConsole(file=sys.stderr)

                def reset_bol(self):
                    pass

            repl_impl._RICH = True
            asi._out_console = _RichConsole(file=sys.stdout)
            asi._margin_stderr = _RichMargin()
            repl_impl._ensure_out_console_imported = lambda: None

        # ── startup branches ──
        if ns.sweep_ok or ns.sweep_raise:
            import external_llm.common.file_lock as _fl_mod

            if ns.sweep_ok:
                _fl_mod.sweep_stale_lock_files = lambda p: 1
            else:

                def _sweep_raise(p):
                    raise OSError("fake sweep crash")

                _fl_mod.sweep_stale_lock_files = _sweep_raise

        if ns.init_none:
            repl_impl._init_repl_engine = lambda *a, **k: None

        # ── helper / model branches ──
        if ns.helper_client_fail:
            repl_impl._create_llm_client_for = lambda provider, api_key="": None

        if ns.suggest_kick_fail:

            def _kick_fail(*a, **k):
                raise RuntimeError("fake suggestion kick crash")

            repl_impl._kick_next_prompt_suggestion = _kick_fail

        if ns.clipboard_image:
            repl_impl._check_clipboard_image = lambda: [{"media_type": "image/png", "data": "aGVsbG8="}]

        if ns.ollama_models:
            repl_impl._get_ollama_models = lambda timeout=5: ["qwen2.5-coder:3b", "llama3.1:8b"]

        if ns.api_key_env:
            os.environ["OPENAI_API_KEY"] = "sk-test-env"
        if ns.reasoning_model:
            ns.model = "deepseek-reasoner"
        if ns.no_api_key:
            # Force the prompt path even when the parent shell exports a key.
            # Remove ALL per-provider API-key env vars (not just the two below):
            # a dev shell may export any of them, and .env's keys are loaded the
            # same way. The "not set in environment" prompt must be deterministic
            # regardless of ambient environment.
            for _env_var in asi._API_KEY_ENV_MAP.values():
                os.environ.pop(_env_var, None)

        # ── undo checkpoint path ──
        if ns.undo_cp:
            _cp_calls = {"n": 0}

            def _cp_seq(repo_root):
                _cp_calls["n"] += 1
                return "cp0" if _cp_calls["n"] == 1 else "cp1"

            repl_impl._newest_checkpoint_id = _cp_seq
            repl_impl._checkpoint_changed_files = lambda repo_root, checkpoint_id: ["a.txt"]
            repl_impl._undo_via_checkpoint = lambda repo_root, cp: True

        if ns.undo_cp_empty:
            _cpe_calls = {"n": 0}
            _cpe_file_calls = {"n": 0}

            def _cpe_cp(repo_root):
                _cpe_calls["n"] += 1
                return "cp0" if _cpe_calls["n"] == 1 else "cp1"

            def _cpe_files(repo_root, checkpoint_id):
                # turn-end call lists a.txt (so the checkpoint is recorded),
                # /undo call lists nothing -> nothing-to-undo branch
                _cpe_file_calls["n"] += 1
                return ["a.txt"] if _cpe_file_calls["n"] == 1 else []

            repl_impl._newest_checkpoint_id = _cpe_cp
            repl_impl._checkpoint_changed_files = _cpe_files

        if ns.undo_cp_fail:
            _cpf_calls = {"n": 0}

            def _cpf_cp(repo_root):
                _cpf_calls["n"] += 1
                return "cp0" if _cpf_calls["n"] == 1 else "cp1"

            repl_impl._newest_checkpoint_id = _cpf_cp
            repl_impl._checkpoint_changed_files = lambda repo_root, checkpoint_id: ["a.txt"]
            repl_impl._undo_via_checkpoint = lambda repo_root, cp: False

        if ns.undo_baseline:
            repl_impl._run_changed_stats = lambda repo_root, base: [("a.txt", 1, 0)]
            repl_impl._changed_files_since = lambda repo_root, base: ["a.txt"]
            repl_impl._undo_run_changes = lambda repo_root, base: (["a.txt"], [])
            repl_impl._print_run_change_summary = lambda repo_root, base: True
            repl_impl._render_run_diff = lambda repo_root, base: True

        if ns.undo_base_fail:
            repl_impl._run_changed_stats = lambda repo_root, base: [("a.txt", 1, 0)]
            repl_impl._changed_files_since = lambda repo_root, base: ["a.txt"]
            repl_impl._undo_run_changes = lambda repo_root, base: ([], ["a.txt"])
            repl_impl._print_run_change_summary = lambda repo_root, base: True
            repl_impl._render_run_diff = lambda repo_root, base: True

        if ns.print_summary_fail:

            def _summary_boom(repo_root, baseline):
                raise OSError("fake summary crash")

            repl_impl._print_run_change_summary = _summary_boom

        if ns.cp_summary_fail:
            _cps_calls = {"n": 0}

            def _cp_summary_boom(repo_root):
                _cps_calls["n"] += 1
                if _cps_calls["n"] == 1:
                    return "cp0"
                raise OSError("fake cp summary crash")

            repl_impl._newest_checkpoint_id = _cp_summary_boom
            repl_impl._checkpoint_changed_files = lambda repo_root, checkpoint_id: ["a.txt"]

        # ── OCR enrichment ──
        if ns.ocr_text:
            from external_llm import providers as _providers_mod

            _providers_mod._images_to_text = lambda images: "fake OCR text"
        if ns.ocr_fail:
            from external_llm import providers as _providers_mod

            def _ocr_boom(images):
                raise OSError("fake OCR crash")

            _providers_mod._images_to_text = _ocr_boom

        # ── git baseline repo ──
        if ns.git_repo:
            import subprocess as _sp

            _sp.run(["git", "init", "-q", "-b", "main"], cwd=ns.repo, check=True)
            _sp.run(["git", "config", "user.email", "t@t"], cwd=ns.repo, check=True)
            _sp.run(["git", "config", "user.name", "t"], cwd=ns.repo, check=True)
            with open(os.path.join(ns.repo, ".gitignore"), "w") as f:
                f.write(".asicode/\n")
            with open(os.path.join(ns.repo, "seed.txt"), "w") as f:
                f.write("seed\n")
            _sp.run(["git", "add", "-A"], cwd=ns.repo, check=True)
            _sp.run(["git", "commit", "-q", "-m", "init"], cwd=ns.repo, check=True)

        if ns.nudge_fail:
            import external_llm.agent.insights_manager as _im_mod

            def _nudge_boom(stats):
                raise OSError("fake nudge crash")

            _im_mod.should_nudge = _nudge_boom

        if ns.nudge_plain:
            import external_llm.agent.insights_manager as _im_mod

            _im_mod.should_nudge = lambda stats: (True, "plain accumulation notice")

        # ── helper persistence seed (shared config.json -> seeded to terminal cfg) ──
        if ns.helper_persisted:
            _shared_cfg = os.path.join(ns.repo, ".asicode", "config.json")
            os.makedirs(os.path.dirname(_shared_cfg), exist_ok=True)
            import json as _json

            with open(_shared_cfg, "w") as f:
                _json.dump({"helper_provider": "anthropic", "helper_model": "claude-sonnet-4-6"}, f)

        # ── insights fixture files ──
        _ins_dir = os.path.join(ns.repo, ".asicode")
        os.makedirs(_ins_dir, exist_ok=True)
        _ins_path = os.path.join(_ins_dir, "design_insights.md")
        from datetime import datetime as _dt

        _today = _dt.now().strftime("%Y-%m-%d %H:%M +0900")
        _ENTRY = (
            "## \u2550\u2550\u2550 DESIGN INSIGHTS \u2550\u2550\u2550\n"
            "\n"
            "### [pattern] {ts}\n"
            "Stage-3 durable insight line.\n"
            "\n"
            "### [design_decision] {ts}\n"
            "Stage-3 second insight line.\n"
        )
        if ns.no_insights:
            pass  # leave the file absent -> "no design_insights file yet"
        elif ns.entries_3:
            _e3 = _ENTRY.format(ts=_today)
            _e3 += f"\n### [issue] {_today}\nStage-3 third insight line.\n"
            with open(_ins_path, "w") as f:
                f.write(_e3)
        elif ns.empty_insights:
            with open(_ins_path, "w") as f:
                f.write("")
        elif ns.old_entries:
            _old_entry = _ENTRY.format(ts="2026-01-05 10:00 +0900")
            with open(_ins_path, "w") as f:
                f.write(_old_entry)
        elif ns.preamble_only:
            with open(_ins_path, "w") as f:
                f.write("## \u2550\u2550\u2550 DESIGN INSIGHTS \u2550\u2550\u2550\n\nNo entries yet.\n")
        elif ns.insights_big or ns.nudge:
            big = _ENTRY.format(ts=_today)
            while len(big.encode("utf-8")) < 12000:
                big += "        padding line to exceed the budget threshold.\n"
            with open(_ins_path, "w") as f:
                f.write(big)
        else:
            with open(_ins_path, "w") as f:
                f.write(_ENTRY.format(ts=_today))
        if ns.archive_data:
            with open(os.path.join(_ins_dir, "design_insights_archive.md"), "w") as f:
                f.write(
                    f"## \u2550\u2550\u2550 ARCHIVE \u2550\u2550\u2550\n\n### [pattern] {_today}\nArchived entry one.\n"
                )
        if ns.bak_dir:
            with open(_ins_path + ".bak", "w") as f:
                f.write("old backup")
            os.chmod(_ins_path + ".bak", 0o000)

        # ── failure-patterns store ──
        class _FakeFPStore:
            def __init__(self, entries):
                self.entries = list(entries)

            def store_size(self):
                return len(self.entries)

            def top_patterns(self, limit=20):
                return self.entries[:limit]

            def clear(self):
                self.entries = []

            def prune(self, threshold=1.0):
                before = len(self.entries)
                self.entries = [e for e in self.entries if (e.get("effective") or 0) >= threshold]
                return before - len(self.entries)

            def drop(self, idx):
                if 1 <= idx <= len(self.entries):
                    e = self.entries.pop(idx - 1)
                    return (e.get("tool") or "?", e.get("reason") or "")
                return None

            def drop_key(self, key):
                for i, e in enumerate(self.entries):
                    if e.get("key") == key:
                        self.entries.pop(i)
                        return (e.get("tool") or "?", e.get("reason") or "")
                return None

        if ns.arch_fail:
            import external_llm.agent.insights_manager as _im_mod

            _im_mod.load_archive_file = lambda repo_root: (_ for _ in ()).throw(OSError("archive read crash"))

        if ns.fp_data:
            _fp_store = _FakeFPStore(
                [
                    {"key": "k1", "tool": "edit_text", "reason": "syntax error", "count": 3, "effective": 2.5},
                    {"key": "k2", "tool": "bash", "reason": "timeout", "count": 1, "effective": 0.3},
                    {"key": "k3", "tool": "edit_text", "reason": "race condition", "count": 2, "effective": 2.0},
                ]
            )
        else:
            _fp_store = _FakeFPStore([])
        import external_llm.agent.failure_pattern_store as _fps_mod

        if ns.fp_race:
            _fp_store = _FakeFPStore(
                [
                    {"key": "k1", "tool": "edit_text", "reason": "syntax error", "count": 3, "effective": 2.5},
                ]
            )
            _fp_store.drop_key = lambda key: None
        if ns.fp_error:
            _fps_mod.get_store = lambda repo_root: (_ for _ in ()).throw(RuntimeError("store read crash"))
        else:
            _fps_mod.get_store = lambda repo_root: _fp_store

        # ── svc / chat-loop fakes ──
        class _ModeLLMMessage:
            def __init__(
                self, content, *, finish_reason=None, prompt_tokens=0, completion_tokens=0, cache_read_input_tokens=0
            ):
                self.content = content
                self.finish_reason = finish_reason
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
                self.cache_read_input_tokens = cache_read_input_tokens

        class _CompactFailClient(FakeClient):
            def chat(self, messages=None, **kw):
                joined = "\n".join(getattr(m, "content", "") or "" for m in (messages or []))
                if "### [pattern]" in joined or "### [design_decision]" in joined:
                    raise RuntimeError("fake compact LLM crash")
                return super().chat(messages=messages, **kw)

        _compact_calls = {"n": 0}

        class _CompactModeClient(FakeClient):
            """Drives the compact edge branches via --compact-mode."""

            def chat(self, messages=None, **kw):
                joined = "\n".join(getattr(m, "content", "") or "" for m in (messages or []))
                if "### [pattern]" not in joined and "### [design_decision]" not in joined:
                    return super().chat(messages=messages, **kw)
                _compact_calls["n"] += 1
                mode = ns.compact_mode
                if mode == "kb":
                    raise KeyboardInterrupt()
                if mode == "truncated" and _compact_calls["n"] == 1:
                    return _ModeLLMMessage("partial", finish_reason="length", prompt_tokens=5, completion_tokens=3)
                if mode == "truncated-final":
                    return _ModeLLMMessage("partial", finish_reason="length", prompt_tokens=5, completion_tokens=3)
                if mode == "empty":
                    return _ModeLLMMessage("")
                if mode == "drops-all":
                    return _ModeLLMMessage(
                        "## \u2550\u2550\u2550 DESIGN INSIGHTS \u2550\u2550\u2550\n\nno entries at all\n"
                    )
                if mode == "partial":
                    return _ModeLLMMessage(
                        "## \u2550\u2550\u2550 DESIGN INSIGHTS \u2550\u2550\u2550\n"
                        "\n### [pattern] 2026-01-01 00:00 +0900\nsurvivor.\n"
                    )
                if mode == "noop":
                    return _ModeLLMMessage(joined)
                if mode == "big":
                    big = COMPACTED_INSIGHTS
                    while len(big.encode("utf-8")) < 12000:
                        big += "        backstop padding line.\n"
                    return _ModeLLMMessage(big)
                if mode == "tokens":
                    return _ModeLLMMessage(
                        COMPACTED_INSIGHTS, prompt_tokens=10, completion_tokens=5, cache_read_input_tokens=2
                    )
                if mode == "fr":
                    return _ModeLLMMessage(COMPACTED_INSIGHTS, finish_reason="stop")
                return _ModeLLMMessage(COMPACTED_INSIGHTS, finish_reason="stop")

        def _make_svc(*a, **k):
            svc = FakeSvc(provider=ns.provider, model=ns.model)
            if ns.compact_fail:
                svc.llm_service.client = _CompactFailClient()
            elif ns.compact_mode:
                svc.llm_service.client = _CompactModeClient()
            return svc

        if ns.notice_fail:
            import external_llm.agent.context_manager as _cm_mod

            def _notice_boom(*a, **k):
                raise RuntimeError("fake notice crash")

            _cm_mod._compress_failure_notice = _notice_boom

        if ns.compact_echo:
            import external_llm.agent.insights_manager as _im_mod

            _im_mod.build_compact_messages = lambda content, budget_bytes=None: [{"role": "user", "content": content}]

        if ns.ebd_fail or ns.ebd_zero or ns.stats_fail:
            import external_llm.agent.insights_manager as _im_mod

            if ns.ebd_fail:

                def _ebd_boom(repo_root, budget_bytes):
                    raise OSError("fake demotion crash")

                _im_mod.enforce_budget_by_demotion = _ebd_boom
            elif ns.ebd_zero:
                _im_mod.enforce_budget_by_demotion = lambda repo_root, budget_bytes: (0, 10**9)
            if ns.stats_fail:

                def _stats_boom(repo_root):
                    raise OSError("fake stats crash")

                _im_mod.compute_stats = _stats_boom

        repl_impl._retry_create_svc_with_api_key_prompt = _make_svc
        isvc_mod.create_intelligent_service_from_env = _make_svc

        def _resolve_model(arg, usage_hint=""):
            if "/" in arg:
                _prov, _, _name = arg.partition("/")
                return (_prov, _name)
            return (ns.provider, arg)

        if ns.resolve_none:
            repl_impl._resolve_model_interactive = lambda arg, usage_hint="": None
        else:
            repl_impl._resolve_model_interactive = _resolve_model

        if ns.ctx_low or ns.ctx_fail:
            import external_llm.agent.context_budget as _cb_mod

            if ns.ctx_low:
                _cb_mod._resolve_context_limit = lambda model, base_url=None: 7
            else:

                def _ctx_boom(model, base_url=None):
                    raise OSError("fake ctx crash")

                _cb_mod._resolve_context_limit = _ctx_boom

        class _EmptyResponseLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                return DesignChatResult(
                    content="",
                    tool_calls_made=[],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    last_call_prompt_tokens=0,
                    last_call_completion_tokens=0,
                    last_call_cache_read_tokens=0,
                    last_call_cache_creation_tokens=0,
                    provider="anthropic",
                    is_error=False,
                    error_type=None,
                    hit_max_iterations=False,
                    total_llm_calls=0,
                )

        class _AuthErrorLoop(FakeDesignChatLoop):
            # ClassVar: the auth-retry path constructs a FRESH loop instance, so
            # the retry counter must survive across instances.
            _auth_calls: ClassVar[dict] = {"n": 0}

            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                _AuthErrorLoop._auth_calls["n"] += 1
                if ns.auth_retry_ok and _AuthErrorLoop._auth_calls["n"] >= 2:
                    return super().respond(messages, stream_callback=stream_callback)
                if ns.auth_retry_crash and _AuthErrorLoop._auth_calls["n"] >= 2:
                    raise RuntimeError("fake retry crash")
                return DesignChatResult(
                    content="\u26a0\ufe0f auth failed 401",
                    tool_calls_made=[],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    last_call_prompt_tokens=0,
                    last_call_completion_tokens=0,
                    last_call_cache_read_tokens=0,
                    last_call_cache_creation_tokens=0,
                    provider="anthropic",
                    is_error=True,
                    error_type="auth",
                    hit_max_iterations=False,
                    total_llm_calls=0,
                )

        class _VerifyFailLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                return DesignChatResult(
                    content="verify failed",
                    tool_calls_made=[],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    last_call_prompt_tokens=0,
                    last_call_completion_tokens=0,
                    last_call_cache_read_tokens=0,
                    last_call_cache_creation_tokens=0,
                    provider="anthropic",
                    is_error=True,
                    error_type="general",
                    hit_max_iterations=False,
                    total_llm_calls=0,
                )

        class _CrashLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                raise RuntimeError("fake design chat crash")

        class _NoResultLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                return None

        class _CacheTokensLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                r = super().respond(messages, stream_callback=stream_callback)
                return DesignChatResult(
                    content=r.content,
                    tool_calls_made=[],
                    tokens_used=r.tokens_used,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    cache_read_tokens=10,
                    cache_creation_tokens=2,
                    last_call_prompt_tokens=6,
                    last_call_completion_tokens=6,
                    last_call_cache_read_tokens=8,
                    last_call_cache_creation_tokens=2,
                    provider=r.provider,
                    is_error=r.is_error,
                    error_type=r.error_type,
                    hit_max_iterations=r.hit_max_iterations,
                    total_llm_calls=r.total_llm_calls,
                )

        class _SaveInsightLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                from external_llm.agent.design_chat_loop import DesignChatResult

                r = super().respond(messages, stream_callback=stream_callback)
                return DesignChatResult(
                    content=r.content,
                    tool_calls_made=[{"tool": "save_insight"}],
                    tokens_used=r.tokens_used,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    cache_read_tokens=r.cache_read_tokens,
                    cache_creation_tokens=r.cache_creation_tokens,
                    last_call_prompt_tokens=r.last_call_prompt_tokens,
                    last_call_completion_tokens=r.last_call_completion_tokens,
                    last_call_cache_read_tokens=r.last_call_cache_read_tokens,
                    last_call_cache_creation_tokens=r.last_call_cache_creation_tokens,
                    provider=r.provider,
                    is_error=r.is_error,
                    error_type=r.error_type,
                    hit_max_iterations=r.hit_max_iterations,
                    total_llm_calls=r.total_llm_calls,
                )

        class _VerifyCrashLoop(FakeDesignChatLoop):
            def respond(self, messages, stream_callback=None, **kw):
                raise RuntimeError("fake verify crash")

        if ns.empty_response:
            _loop_cls = _EmptyResponseLoop
        elif ns.auth_error_turn or ns.auth_retry_ok or ns.auth_retry_crash:
            _loop_cls = _AuthErrorLoop
        elif ns.verify_fail:
            _loop_cls = _VerifyFailLoop
        elif ns.verify_crash:
            _loop_cls = _VerifyCrashLoop
        elif ns.verify_kb:

            class _VerifyKBLoop(FakeDesignChatLoop):
                def respond(self, messages, stream_callback=None, **kw):
                    raise KeyboardInterrupt()

            _loop_cls = _VerifyKBLoop
        elif ns.dc_crash:
            _loop_cls = _CrashLoop
        elif ns.no_result:
            _loop_cls = _NoResultLoop
        elif ns.cache_tokens:
            _loop_cls = _CacheTokensLoop
        elif ns.save_insight:
            _loop_cls = _SaveInsightLoop
        else:
            _loop_cls = FakeDesignChatLoop

        class _NotifyingDSM(FakeDSM):
            def schedule_background_compress(self, session, model, client, **kw):
                notify = kw.pop("notify", None)
                if notify is not None:
                    notify("background compress complete")
                return super().schedule_background_compress(session, model, client, **kw)

        if ns.dsm_crash:

            class _CrashDSM(FakeDSM):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    self._calls = 0

                def get_or_create(self, sid):
                    # init (_init_repl_engine) succeeds once; the /clear call raises
                    self._calls += 1
                    if self._calls >= 2:
                        raise OSError("fake dsm crash")
                    return super().get_or_create(sid)

            design_session_mod.DesignSessionManager = _CrashDSM
        elif ns.dsm_turn_crash:
            _tc_calls = {"n": 0}

            class _TurnCrashDSM(FakeDSM):
                def add_turn(self, *a, **k):
                    _tc_calls["n"] += 1
                    if _tc_calls["n"] >= 2:
                        raise OSError("fake turn persist crash")
                    return super().add_turn(*a, **k)

            design_session_mod.DesignSessionManager = _TurnCrashDSM
        elif ns.notify:
            design_session_mod.DesignSessionManager = _NotifyingDSM
        else:
            design_session_mod.DesignSessionManager = FakeDSM

        if ns.orch_fail:

            class _OrchFail(FakeOrchAgent):
                def run(self, task, session_id=None, **kw):
                    raise RuntimeError("fake orchestrator crash")

            orch_mod.OrchestratorAgent = _OrchFail
        elif ns.orch_kb:

            class _OrchKB(FakeOrchAgent):
                def run(self, task, session_id=None, **kw):
                    raise KeyboardInterrupt()

            orch_mod.OrchestratorAgent = _OrchKB
        elif ns.orch_none:

            class _OrchNone(FakeOrchAgent):
                def run(self, task, session_id=None, **kw):
                    return None

            orch_mod.OrchestratorAgent = _OrchNone
        else:
            orch_mod.OrchestratorAgent = FakeOrchAgent

        tool_registry_mod.ToolRegistry = FakeToolRegistry
        tool_registry_mod.AgentConfig = FakeAgentConfig

        dcl_mod.DesignChatLoop = lambda client, registry, model: _loop_cls(client, registry, model)
        if ns.llm_client_fail:

            def _client_boom(*a, **k):
                raise RuntimeError("fake client crash")

            client_mod.create_llm_client = _client_boom
        else:
            client_mod.create_llm_client = lambda *a, **k: FakeClient()

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

        from prompt_toolkit.completion import Completer as _PtCompleter

        class _NoCompleter(_PtCompleter):
            def __init__(self, **kw):
                super().__init__()

            def get_completions(self, document, complete_event):
                return iter(())

        repl_impl._SlashCommandCompleter = _NoCompleter

        args = argparse.Namespace(
            repo=ns.repo,
            provider=ns.provider,
            model=ns.model,
            api_key=None,
            no_deps_check=True,
            verbose=ns.verbose,
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
        with contextlib.suppress(Exception):
            cov.save()
    return rc


if __name__ == "__main__":
    os._exit(_main())
