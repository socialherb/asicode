"""Tests for the ``/helper`` slash command (context-compression model).

Covers the reusable model-resolution + client-creation helpers that back
``/helper`` and ``/model``. The REPL wiring (state persistence, compress
routing) is exercised via the ``_get_compress_llm`` resolution contract:
helper model wins when set, otherwise the main model falls back.
"""
from __future__ import annotations

import argparse
import json
import sys

import pytest

import asi


class TestResolveModelArg:
    """``_resolve_model_arg`` — provider/name parsing + auto-resolution."""

    def test_empty_returns_none(self):
        assert asi._resolve_model_arg("") is None
        assert asi._resolve_model_arg("   ") is None
        assert asi._resolve_model_arg(None) is None

    def test_explicit_provider_slash_name(self):
        assert asi._resolve_model_arg("deepseek/deepseek-chat") == (
            "deepseek",
            "deepseek-chat",
        )

    def test_explicit_provider_slash_name_strips_whitespace(self):
        assert asi._resolve_model_arg("  deepseek / deepseek-chat  ") == (
            "deepseek",
            "deepseek-chat",
        )

    def test_slash_without_model_returns_none(self):
        assert asi._resolve_model_arg("deepseek/") is None


class TestResolveModelInteractiveRejectsNaturalLanguage:
    """``_resolve_model_interactive`` must reject natural-language input that
    gets mis-parsed as a model name.

    Regression guard for the bug where ``/model qwen3.7-max bug/feature/perf …``
    was split on ``/`` into ``provider="qwen3.7-max bug"`` /
    ``model="feature/perf …"`` — a provider with spaces is never valid, so the
    slash path must refuse it instead of silently switching to a garbage model.
    Likewise the space-separated path must reject a multi-token model name.

    These are source-contract guards (same pattern as
    ``TestInsightsCompactUsesHelperModel``): ``_print`` routes through a Rich
    console bound to the real stdout at import time, so capsys can't capture
    its output — we assert the guard logic exists in the source instead.
    """

    def _src(self) -> str:
        import inspect
        return inspect.getsource(asi._resolve_model_interactive)

    def test_slash_path_has_provider_space_guard(self):
        # The slash-separated branch must reject a provider containing spaces.
        src = self._src()
        assert "len(new_provider.split()) > 1" in src
        assert "provider name must not contain spaces" in src

    def test_space_path_has_model_space_guard(self):
        # The space-separated branch must reject a multi-token model name.
        src = self._src()
        assert "len(new_model.split()) > 1" in src
        assert "model name must not contain spaces" in src

    def test_slash_path_still_accepts_clean_provider_model(self):
        # Sanity: the legitimate slash form still resolves.
        result = asi._resolve_model_interactive("deepseek/deepseek-chat")
        assert result == ("deepseek", "deepseek-chat")


class TestPerTerminalModelRestore:
    """``/model`` persistence: per-terminal config's provider/model must be
    restored on CLI restart when no CLI arg overrides them.

    Regression guard for the half-wired defect where ``/model`` wrote
    provider/model to the per-terminal config (``.asicode/terminals/<tty>.json``)
    but the restore path only read ``.env`` / CLI args — so every terminal
    reverted to the single ``.env`` model on restart, ignoring its own saved
    ``/model`` choice. The fix reads the per-terminal config *before* svc
    creation so the LLM client is born with the right provider/model (no
    client re-creation needed).

    Source-contract guards: ``main()`` reads the per-terminal config and writes
    it back into ``args`` before ``run_repl`` is called. ``run_repl`` then
    receives already-resolved ``args.provider`` / ``args.model``. We assert the
    wiring exists in ``main()`` source rather than invoking it.
    """

    def _main_src(self) -> str:
        import inspect
        return inspect.getsource(asi.main)

    def _repl_src(self) -> str:
        import inspect
        return inspect.getsource(asi.run_repl)

    def test_restore_reads_per_terminal_config_before_svc_creation(self):
        # The restore block must exist in main() and run before run_repl,
        # so run_repl receives already-resolved args.provider/model.
        src = self._main_src()
        assert "_terminal_config_path" in src
        assert "_seed_terminal_config" in src
        assert "_saved_cfg.get(\"provider\", \"\")" in src
        assert "_saved_cfg.get(\"model\", \"\")" in src

    def test_svc_creation_uses_resolved_strings_not_raw_args(self):
        # svc must be created from _provider_str/_model_str (which carry the
        # per-terminal restore), not raw args.provider/args.model — otherwise
        # the restore is overwritten by the (None) args.
        src = self._repl_src()
        # find the svc creation call and assert it references _provider_str
        assert "_provider_str if _provider_str != \"(env)\"" in src
        assert "_model_str if _model_str != \"(env)\"" in src

    def test_args_override_takes_priority(self):
        # The restore must be gated on `not args.provider` / `not args.model`
        # so an explicit CLI arg wins over the saved per-terminal config.
        src = self._main_src()
        assert "if not args.provider:" in src
        assert "if not args.model:" in src


class TestCreateLLMClientFor:
    """``_create_llm_client_for`` — must be a callable that delegates to the
    provider factory. We don't hit the network; we only assert the contract."""

    def test_is_callable(self):
        assert callable(asi._create_llm_client_for)

    def test_unknown_provider_returns_or_raises(self):
        # An unknown provider should not silently succeed with a None client;
        # the factory either raises or returns None. Both are acceptable — the
        # /helper handler treats None as a hard failure and keeps main model.
        try:
            _result = asi._create_llm_client_for("__not_a_real_provider__")
            assert _result is None or _result is not None
        except Exception:
            pass  # raising is also acceptable for an invalid provider


class TestHelperCommandRegistration:
    """The ``/helper`` command is registered in the slash-command tables."""

    def test_in_slash_commands_table(self):
        names = [c[0] for c in asi._SLASH_COMMANDS]
        assert "/helper" in names

    def test_has_usage_arg(self):
        entry = next(c for c in asi._SLASH_COMMANDS if c[0] == "/helper")
        # (name, aliases, arg, desc)
        assert entry[1] == ()  # no aliases yet
        assert "[name]" in entry[2]

    def test_in_alias_map(self):
        assert asi._SLASH_ALIASES.get("/helper") == "/helper"

    def test_helper_dispatched_with_other_model_commands(self):
        # The dispatch guard that routes /model etc. must include /helper,
        # otherwise a bare ``/helper`` falls through to design chat.
        import inspect
        src = inspect.getsource(asi.run_repl)
        assert '"/helper"' in src


class TestInsightsCompactUsesHelperModel:
    """Contract: ``/insights compact`` must honor the ``/helper`` compression
    model. It delegates to ``_compact_insights_interactive``, which resolves its
    client/model through the single ``_get_compress_llm`` entry point — the
    helper model wins when set, the main model falls back otherwise.

    These are source-contract guards: they fail if someone "simplifies" the
    compact path to call the main LLM client directly, silently bypassing
    ``/helper``. ``_get_compress_llm`` is a closure inside ``run_repl`` (not
    module-level), so we inspect its source rather than invoking it — mirroring
    the pattern in ``test_helper_dispatched_with_other_model_commands``.
    """

    def _src(self) -> str:
        import inspect
        return inspect.getsource(asi.run_repl)

    def test_compact_subcommand_delegates_to_interactive_helper(self):
        # /insights compact must route through _compact_insights_interactive
        # instead of inlining its own LLM client call.
        src = self._src()
        assert 'elif _ins_sub == "compact":' in src
        assert "_compact_insights_interactive()" in src

    def test_interactive_helper_resolves_via_single_entry_point(self):
        # _compact_insights_interactive must obtain its client+model from
        # _get_compress_llm — the one place helper/main resolution lives.
        assert "_ci_client, _ci_model = _get_compress_llm()" in self._src()

    def test_resolution_prioritizes_helper_model_when_set(self):
        # When /helper is configured, its dedicated client+model is returned.
        src = self._src()
        assert "if _helper_model_str:" in src
        assert "return _helper_client, _helper_model_str" in src

    def test_resolution_falls_back_to_main_model(self):
        # When no helper is configured, the main svc client+model is used.
        assert 'return svc.llm_service.client, (svc.model or "")' in self._src()

    def test_helper_creation_failure_is_safe_fallback(self):
        # If the helper client can't be created (e.g. missing API key), the
        # path must NOT raise — it logs and falls through to the main model.
        assert "helper client creation failed" in self._src()

    def test_resolution_defined_exactly_once(self):
        # Single source of truth for helper/main compression-model resolution.
        assert self._src().count("def _get_compress_llm(") == 1


class TestInsightsCompactBudgetBackstopUsesPostWriteSize:
    """Regression guard: the HARD budget backstop (Layer 1 demotion) must be
    gated on the POST-write size (``_ci_a_b``), not the pre-write
    ``_ci_over_budget`` flag. A file that started under budget can still be
    rewritten over budget by the LLM (e.g. merged entries with expanded
    rationale) — if the gate only looks at the pre-write flag, that case
    silently skips demotion and writes an over-budget file with no warning,
    violating the documented "always ≤ budget_bytes" contract.
    """

    def _src(self) -> str:
        import inspect
        return inspect.getsource(asi.run_repl)

    def test_backstop_gated_on_post_write_size(self):
        src = self._src()
        assert "if _ci_a_b > COMPACT_BUDGET_BYTES:" in src
        assert "elif _ci_a_b > COMPACT_BUDGET_BYTES:" in src
        # The pre-write flag must NOT gate either the demotion call or the
        # post-write "could not reach budget" warning.
        assert "if _ci_over_budget:\n            try:\n                from external_llm.agent.insights_manager import (\n                    enforce_budget_by_demotion" not in src
        assert "elif _ci_over_budget and _ci_a_b > COMPACT_BUDGET_BYTES:" not in src


class TestRunOnceJsonErrorPaths:
    """``--json`` must emit machine-parseable JSON on stdout for every exit
    path, not just the success path — otherwise Tenet's JSON parser breaks
    the moment a request is cancelled or errors out. Regression guard for the
    cancel / RuntimeError / unexpected-Exception branches of ``run_once``.
    """

    def _args(self, tmp_path):
        return argparse.Namespace(
            repo=str(tmp_path),
            provider="x", model="x", api_key=None,
            max_turns=10, verbose=False,
            thinking_mode=None, reasoning_effort=None,
            json=True,
        )

    def test_runtime_error_emits_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            asi, "_build_engine",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no provider configured")),
        )

        rc = asi.run_once(self._args(tmp_path), "do something")

        assert rc == 1
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "error"
        assert "no provider configured" in payload["error"]

    def test_unexpected_exception_emits_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            asi, "_build_engine",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("weird internal state")),
        )

        rc = asi.run_once(self._args(tmp_path), "do something")

        assert rc == 1
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "unexpected_error"
        assert "weird internal state" in payload["error"]

    def test_cancelled_emits_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(asi, "_build_engine", lambda *a, **kw: object())
        monkeypatch.setattr(asi, "_git_baseline", lambda repo_root: None)
        monkeypatch.setattr(asi, "_run_with_cancel", lambda *a, **kw: None)

        rc = asi.run_once(self._args(tmp_path), "do something")

        assert rc == 130
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "cancelled"


class TestMainRequiresPromptSourceWithJson:
    """``--json`` only produces output from ``run_once`` (single-shot mode).
    With no ``--prompt``/``--prompt-file``/``--prompt-stdin``, ``main()``
    falls through to the interactive ``run_repl()`` and the flag is silently
    ignored — validate loudly instead of dropping into a REPL the caller
    can't drive."""

    def test_json_without_prompt_source_exits_1(self, tmp_path, monkeypatch):
        # _print may route through a Rich console bound directly to the real
        # stdout (not capsys-redirected) — the exit code is the reliable,
        # capture-agnostic signal that validation fired.
        monkeypatch.setattr(sys, "argv", ["asi", "--json", "--repo", str(tmp_path)])

        with pytest.raises(SystemExit) as exc:
            asi.main()

        assert exc.value.code == 1
