"""RED→GREEN: ToolRegistry 76% → 100% coverage (275 missing lines).

Covers the remaining branches of tool_registry.py that no existing test
reaches — module-level gitignore/scratch helpers, checkpoint gates, semantic
turn drain edge paths, cache-invalidation exception branches, bash classifier
fallbacks, write-safety repair/soft-fail/indent-repair machinery, dispatch
edge paths (non-dict args / cancel / profile / gate / locks / handler crash),
the write-verify cascade, parallel dispatch fallbacks and bias-path extras.

No source changes were needed; every branch below is reachable through a
legitimate configuration or a patched collaborator (same technique the
existing tool_registry tests use for sub-classifiers).
"""
from __future__ import annotations

import builtins
import os
import subprocess
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace

import pytest

import external_llm.agent.tool_registry as tr_mod
from external_llm.agent.tool_registry import (
    AgentConfig,
    SemanticOutcome,
    ToolRegistry,
    ToolResult,
    _ensure_asicode_gitignored,
    _under_scratch_root,
)
from external_llm.agent.tool_safety import _MISSING_SNAP
from external_llm.editor._editor_core.vm.failure_classifier import FailureType
from external_llm.languages import LanguageId, LanguageRegistry


def _raise(exc):
    def _fn(*_a, **_k):
        raise exc
    return _fn


@pytest.fixture(autouse=True)
def _clear_bash_cache():
    """_bash_command_mutates_files is lru_cached; a classification must not
    leak across tests (several tests pin the fallback path via patches)."""
    ToolRegistry._bash_command_mutates_files.cache_clear()
    yield
    ToolRegistry._bash_command_mutates_files.cache_clear()


@pytest.fixture
def cfg():
    return AgentConfig(
        max_turns=5, run_tests=False, run_lint=False, auto_test_on_patch=False,
        self_review_enabled=False, rag_enabled=False, vector_cache_enabled=False,
        parallel_tool_execution_enabled=True,
        tool_result_cache_enabled=False,
    )


@pytest.fixture
def reg(cfg, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    return ToolRegistry(str(tmp_path), cfg)


# ── module-level helpers ────────────────────────────────────────────────────

class TestModuleLevelHelpers:
    def test_semantic_outcome_checked_property(self):
        assert SemanticOutcome().checked is True
        assert SemanticOutcome(skip_reason="x").checked is False

    def test_gitignore_entry_present_returns_early(self, tmp_path):
        (tmp_path / ".gitignore").write_text(".asicode/\n", encoding="utf-8")
        _ensure_asicode_gitignored(str(tmp_path))
        assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == ".asicode/\n"

    def test_gitignore_appends_without_trailing_newline(self, tmp_path):
        p = tmp_path / ".gitignore"
        p.write_text("x=1", encoding="utf-8")  # no trailing newline
        _ensure_asicode_gitignored(str(tmp_path))
        assert p.read_text(encoding="utf-8") == "x=1\n.asicode/\n"

    def test_gitignore_appends_with_trailing_newline(self, tmp_path):
        p = tmp_path / ".gitignore"
        p.write_text("x=1\n", encoding="utf-8")
        _ensure_asicode_gitignored(str(tmp_path))
        assert p.read_text(encoding="utf-8") == "x=1\n.asicode/\n"

    def test_gitignore_io_exception_warns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(builtins, "open", _raise(OSError("denied")))
        _ensure_asicode_gitignored(str(tmp_path))  # must not raise

    def test_under_scratch_root_empty_is_false(self):
        assert _under_scratch_root("") is False

    def test_detect_repo_language_subprocess_raises(self, tmp_path, monkeypatch):
        ToolRegistry._LANGUAGE_DETECTION_CACHE.clear()
        try:
            monkeypatch.setattr(subprocess, "run", _raise(FileNotFoundError("git missing")))
            assert ToolRegistry._detect_repo_language(str(tmp_path)) is None
        finally:
            ToolRegistry._LANGUAGE_DETECTION_CACHE.clear()


# ── construction ────────────────────────────────────────────────────────────

class TestConstruction:
    def test_init_with_agent_profile_logs(self, cfg, tmp_path):
        cfg.agent_profile = SimpleNamespace(name="prof")
        ToolRegistry(str(tmp_path), cfg)  # debug log branch

    def test_make_tool_result_cache_constructor_raises(self, reg, monkeypatch):
        import external_llm.agent.tool_result_cache as trc
        monkeypatch.setattr(trc, "ToolResultCache", lambda *a, **k: _raise(RuntimeError("boom"))())
        assert reg._make_tool_result_cache(AgentConfig(tool_result_cache_enabled=True)) is None


# ── run-checkpoint gates ────────────────────────────────────────────────────

class TestCheckpointGates:
    def test_checkpoint_before_write_missing_gate_rebuilds(self, reg):
        reg._run_checkpoint_gate = None
        reg._checkpoint_before_write("edit_text", {"file_path": "a.py"})
        assert reg._run_checkpoint_gate is not None  # detached gate rebuilt

    def test_checkpoint_before_write_disabled_gate(self, reg):
        reg._run_checkpoint_gate = SimpleNamespace(enabled=False)
        reg._checkpoint_before_write("edit_text", {"file_path": "a.py"})  # early return

    def test_checkpoint_before_write_targets_raise(self, reg, monkeypatch):
        reg._run_checkpoint_gate = SimpleNamespace(enabled=True, before_write=lambda p: None)
        monkeypatch.setattr(reg, "_extract_write_target_paths", _raise(ValueError("bad targets")))
        reg._checkpoint_before_write("edit_file", {})  # warning + return

    def test_checkpoint_after_write_disabled_or_missing_gate(self, reg):
        reg._run_checkpoint_gate = SimpleNamespace(enabled=False)
        reg._checkpoint_after_write("edit_text", {"file_path": "a.py"})
        reg._run_checkpoint_gate = None
        reg._checkpoint_after_write("edit_text", {"file_path": "a.py"})

    def test_checkpoint_after_write_targets_raise(self, reg, monkeypatch):
        reg._run_checkpoint_gate = SimpleNamespace(enabled=True, confirm_writes=lambda p: None)
        monkeypatch.setattr(reg, "_extract_write_target_paths", _raise(ValueError("bad targets")))
        reg._checkpoint_after_write("edit_file", {})  # warning + return


# ── semantic turn coalescing drain ──────────────────────────────────────────

def _patch_lang_registry(monkeypatch, fake_reg):
    monkeypatch.setattr(LanguageRegistry, "instance", staticmethod(lambda: fake_reg))


class TestSemanticTurnDrain:
    def test_begin_end_defer(self, reg):
        reg.begin_semantic_turn()
        assert reg._semantic_turn_active
        assert reg.defer_semantic_check("/abs/f.py", "rel.py") is True
        assert reg._semantic_pending == {"/abs/f.py": "rel.py"}
        assert reg.defer_semantic_check("/abs/g.py") is True
        assert reg._semantic_pending["/abs/g.py"] == "/abs/g.py"
        reg.end_semantic_turn()
        assert not reg._semantic_turn_active and reg._semantic_pending == {}
        assert reg.defer_semantic_check("/abs/h.py") is False

    def test_drain_provider_lookup_raises(self, reg, monkeypatch):
        _patch_lang_registry(monkeypatch, SimpleNamespace(get=_raise(RuntimeError("registry boom"))))
        reg.begin_semantic_turn()
        reg.defer_semantic_check("/tmp/x.py")
        out = reg.drain_pending_semantic_checks()
        assert out["/tmp/x.py"].skip_reason == "the language provider could not be loaded"

    def test_drain_pool_submit_runtime_error_falls_back_inline(self, reg, monkeypatch):
        def ok_batch(paths):
            return {p: SimpleNamespace(checked=True, skip_reason="", errors=[]) for p in paths}
        prov_a = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=ok_batch,
        )
        prov_b = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=ok_batch,
        )
        fake_reg = SimpleNamespace(get=lambda p: {"/a.py": prov_a, "/b.ts": prov_b}[p])
        _patch_lang_registry(monkeypatch, fake_reg)
        monkeypatch.setattr(
            tr_mod, "shared_pool", SimpleNamespace(submit=_raise(RuntimeError("pool down")))
        )
        reg.begin_semantic_turn()
        reg.defer_semantic_check("/a.py")
        reg.defer_semantic_check("/b.ts")
        out = reg.drain_pending_semantic_checks()
        assert out["/a.py"].diagnostics == []
        assert out["/b.ts"].diagnostics == []

    def test_drain_cancel_polls_and_marks_remaining(self, reg, monkeypatch):
        ce = threading.Event()
        ce.set()
        reg.config.cancel_event = ce

        class _TimeoutFuture:
            def result(self, timeout=None):
                raise FutureTimeoutError("poll timeout")

        monkeypatch.setattr(
            tr_mod, "shared_pool", SimpleNamespace(submit=lambda fn, *a, **k: _TimeoutFuture())
        )
        def ok_batch(paths):
            return {p: SimpleNamespace(checked=True, skip_reason="", errors=[]) for p in paths}
        prov_a = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=ok_batch,
        )
        prov_b = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=None,
        )
        prov_c = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=None,
        )
        fake_reg = SimpleNamespace(get=lambda p: {"/a.py": prov_a, "/b.ts": prov_b, "/c.go": prov_c}[p])
        _patch_lang_registry(monkeypatch, fake_reg)
        reg.begin_semantic_turn()
        reg.defer_semantic_check("/a.py")
        reg.defer_semantic_check("/b.ts")
        reg.defer_semantic_check("/c.go")
        out = reg.drain_pending_semantic_checks()
        assert out["/a.py"].diagnostics == []  # inline group still ran
        assert out["/b.ts"].skip_reason == "cancelled before the semantic check ran"
        assert out["/c.go"].skip_reason == "cancelled before the semantic check ran"

    def test_drain_batch_missing_result_entry(self, reg, monkeypatch):
        prov = SimpleNamespace(
            capabilities=lambda: SimpleNamespace(has_semantic_validator=True),
            validate_semantics_batch=lambda paths: {},
        )
        _patch_lang_registry(monkeypatch, SimpleNamespace(get=lambda p: prov))
        reg.begin_semantic_turn()
        reg.defer_semantic_check("/a.py")
        out = reg.drain_pending_semantic_checks()
        assert out["/a.py"].skip_reason == "no semantic checker for this file type"


# ── cache invalidation exception paths ──────────────────────────────────────

def _patch_invalidation_modules(monkeypatch):
    import external_llm.agent._shared_utils as su
    import external_llm.agent.symbol_search as ss
    import external_llm.agent.tool_handlers.write_tools as wt
    import external_llm.common.repo_files as rf
    return su, ss, rf, wt


class TestInvalidationExceptions:
    def test_invalidate_after_write_blank_path_skipped(self, reg):
        reg._invalidate_cache_after_write([""])

    def test_invalidate_after_write_relpath_valueerror(self, reg, monkeypatch):
        monkeypatch.setattr(os.path, "relpath", _raise(ValueError("different drive")))
        reg._invalidate_cache_after_write(["/abs/path.py"])  # kept as-is

    def test_invalidate_after_write_all_exception_branches(self, reg, monkeypatch):
        su, ss, rf, _ = _patch_invalidation_modules(monkeypatch)
        monkeypatch.setattr(reg._rag_searcher, "invalidate_files", _raise(RuntimeError("rag")))
        monkeypatch.setattr(reg._call_graph, "invalidate_files", _raise(AttributeError("gsg")))
        monkeypatch.setattr(su, "invalidate_walk_caches", _raise(RuntimeError("walk")))
        monkeypatch.setattr(rf, "invalidate_for_written_path", _raise(RuntimeError("idx")))
        monkeypatch.setattr(reg._symbol_searcher, "invalidate_nonpy_caches", _raise(RuntimeError("nonpy")))
        monkeypatch.setattr(ss, "invalidate_py_prefilter_cache", _raise(RuntimeError("prefilter")))
        monkeypatch.setattr(reg._symbol_searcher, "invalidate_file_caches", _raise(RuntimeError("filemaps")))
        reg._invalidate_cache_after_write(["a.py", "b.go"])  # each degrades to a debug log

    def test_invalidate_unknown_scope_all_exception_branches(self, reg, monkeypatch):
        su, ss, _, wt = _patch_invalidation_modules(monkeypatch)
        monkeypatch.setattr(reg._call_graph, "invalidate", _raise(RuntimeError("facade")))
        monkeypatch.setattr(reg._call_graph.call_graph_indexer, "invalidate", _raise(RuntimeError("cgi")))
        monkeypatch.setattr(su, "invalidate_walk_caches", _raise(RuntimeError("walk")))
        monkeypatch.setattr(reg._symbol_searcher, "invalidate_nonpy_caches", _raise(RuntimeError("nonpy")))
        monkeypatch.setattr(ss, "invalidate_py_prefilter_cache", _raise(RuntimeError("prefilter")))
        monkeypatch.setattr(reg._symbol_searcher, "invalidate_file_caches", _raise(RuntimeError("filemaps")))
        monkeypatch.setattr(wt, "invalidate_repo_file_index", _raise(RuntimeError("idx")))
        reg._invalidate_caches_unknown_scope()

    def test_ensure_asicode_gitignored_method(self, reg):
        reg._ensure_asicode_gitignored()  # bound-method delegation


# ── bash classifiers ────────────────────────────────────────────────────────

class TestBashClassifierExtras:
    def test_has_redirect_quoted_redirect_is_not_mutation(self):
        assert ToolRegistry._has_redirect_outside_quotes('echo "a>b"') is False
        assert ToolRegistry._has_redirect_outside_quotes("echo 'a>b'") is False
        assert ToolRegistry._has_redirect_outside_quotes("echo hi") is False

    def test_has_redirect_backslash_escape(self):
        assert ToolRegistry._has_redirect_outside_quotes(r"echo \> out") is False

    def test_redirect_is_fd_dup_bare_gt(self):
        assert ToolRegistry._redirect_is_fd_dup(">") is False
        assert ToolRegistry._redirect_is_fd_dup("") is False

    def test_parse_bash_tree_bootstrap_exception(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as tsu
        monkeypatch.setattr(tsu, "is_available", _raise(RuntimeError("ts broken")))
        assert ToolRegistry._parse_bash_tree("echo hi") is None

    def test_parse_bash_tree_parser_none(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as tsu
        monkeypatch.setattr(tsu, "is_available", lambda: True)
        monkeypatch.setattr(tsu, "get_parser", lambda lang: None)
        assert ToolRegistry._parse_bash_tree("echo hi") is None

    def test_bash_mutates_empty_command(self):
        assert ToolRegistry._bash_command_mutates_files("") is False

    def test_bash_mutates_fallback_single_segment(self, monkeypatch):
        monkeypatch.setattr(ToolRegistry, "_parse_bash_tree", classmethod(lambda cls, command: None))
        assert ToolRegistry._bash_command_mutates_files("echo hi") is False


# ── mutation / serial classification ────────────────────────────────────────

class TestMutationSerial:
    def test_tool_call_is_serial_job_kill(self, reg):
        assert reg._tool_call_is_serial("job", {"action": "kill"}) is True
        assert reg._tool_call_is_serial("job", {"action": "output"}) is False

    def test_extract_read_scope_blank_path_none(self, reg):
        assert reg._extract_read_scope_paths("read_file", {"path": ""}) is None
        assert reg._extract_read_scope_paths("read_file", {}) is None


# ── _repair_verify_failure (argument-mismatch auto-repair) ──────────────────

class _RepairEnv:
    """Configurable stand-ins for _repair_verify_failure's collaborators.

    The real ``self._verify_after_write`` (WriteSafetyManager) is kept: its
    provider lookup goes through the same patched LanguageRegistry, so the
    all-files re-verify gate at the tail behaves deterministically.
    """

    def __init__(self, monkeypatch, tmp_path, *, strategy_ops=None, val_ok_after=False,
                 classifier_raises=False, atomic_raises=False, open_raises=False,
                 provider=None, classify=FailureType.ARGUMENT_MISMATCH, file_exists=True):
        self.monkeypatch = monkeypatch
        self.writes: list[tuple[str, str]] = []
        self.val_ok_after = val_ok_after
        self._calls = 0
        if file_exists:
            (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
        self.path = str(tmp_path / "m.py")

        if provider is None:
            provider = SimpleNamespace(
                capabilities=lambda: SimpleNamespace(has_syntax_validator=True),
                language_id=lambda: SimpleNamespace(value="python"),
                validate_syntax=self._validate,
            )
        elif provider is not None:
            provider = SimpleNamespace(
                capabilities=lambda: SimpleNamespace(has_syntax_validator=False),
                language_id=lambda: SimpleNamespace(value="python"),
            )
        self.provider = provider
        fake_reg = SimpleNamespace(get=lambda path: self.provider)
        _patch_lang_registry(monkeypatch, fake_reg)

        import external_llm.editor._editor_core.vm.failure_classifier as fc_mod
        import external_llm.editor._editor_core.vm.repair_registry as rr_mod

        def _cc(lang):
            if classifier_raises:
                raise ValueError("no classifier")
            return SimpleNamespace(classify=lambda errors: classify)

        monkeypatch.setattr(fc_mod, "create_failure_classifier", _cc)

        def _rr(lang):
            if callable(strategy_ops):
                return SimpleNamespace(get=lambda ftype: strategy_ops)
            if strategy_ops is None:
                return SimpleNamespace(get=lambda ftype: None)
            ops = [SimpleNamespace(payload=op) for op in strategy_ops]
            return SimpleNamespace(get=lambda ftype: (lambda code, verr, clf: ops))

        monkeypatch.setattr(rr_mod, "RepairRegistry", _rr)

        def _aw(path, content):
            if atomic_raises:
                raise OSError("disk full")
            self.writes.append((str(path), content))

        monkeypatch.setattr(tr_mod, "atomic_write_text", _aw)

        if open_raises:
            real_open = builtins.open
            target = self.path

            def _open_boom(name, *a, **k):
                if str(name) == target:
                    raise OSError("cannot read")
                return real_open(name, *a, **k)

            monkeypatch.setattr(builtins, "open", _open_boom)

    def _validate(self, path, code):
        self._calls += 1
        if self.val_ok_after and self.writes:
            return SimpleNamespace(ok=True, errors=[])
        return SimpleNamespace(
            ok=False,
            errors=[SimpleNamespace(message="arg mismatch", line=1, col=1)],
        )


class TestRepairVerifyFailure:
    def _call(self, reg, env, snapshots=None):
        snaps = snapshots if snapshots is not None else {env.path: "x = 1\n"}
        return reg._repair_verify_failure(snaps)

    def test_repair_verify_missing_file_skipped(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, file_exists=False)
        assert self._call(reg, env, {env.path: "code"}) is False

    def test_repair_verify_no_provider(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, provider=None)
        env.provider = None
        assert self._call(reg, env) is False

    def test_repair_verify_provider_without_validator(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, provider=SimpleNamespace())
        assert self._call(reg, env) is False

    def test_repair_verify_read_oserror(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, open_raises=True)
        assert self._call(reg, env) is False

    def test_repair_verify_valid_file_skipped(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path)
        # Make validation succeed on every call → the file is clean → no repair.
        env.val_ok_after = True
        env.writes.append(("sentinel", ""))  # val_ok_after gate is `and self.writes`
        assert self._call(reg, env) is False

    def test_repair_verify_classifier_valueerror(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, classifier_raises=True)
        assert self._call(reg, env) is False

    def test_repair_verify_no_strategy(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path)  # strategy_ops=None → registry.get → None
        assert self._call(reg, env) is False

    def test_repair_verify_strategy_returns_none(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, strategy_ops=lambda code, verr, clf: None)
        assert self._call(reg, env) is False

    def test_repair_verify_ops_without_raw_code(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(monkeypatch, tmp_path, strategy_ops=[{"other": 1}])
        assert self._call(reg, env) is False

    def test_repair_verify_atomic_write_oserror(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(
            monkeypatch, tmp_path,
            strategy_ops=[{"__raw_code__": "x = 1\n"}],
            atomic_raises=True,
        )
        assert self._call(reg, env) is False
        assert env.writes == []

    def test_repair_verify_success_reverifies_all(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(
            monkeypatch, tmp_path,
            strategy_ops=[{"__raw_code__": "x = 1\n"}],
            val_ok_after=True,
        )
        assert self._call(reg, env) is True
        assert len(env.writes) == 1

    def test_repair_verify_reverify_fail_restores(self, reg, tmp_path, monkeypatch):
        env = _RepairEnv(
            monkeypatch, tmp_path,
            strategy_ops=[{"__raw_code__": "x = 1\n"}],
        )  # val_ok_after=False → re-verify keeps failing
        assert self._call(reg, env) is False
        assert len(env.writes) == 2  # repair write + restore write


# ── _should_soft_fail_verify ────────────────────────────────────────────────

class _SoftFailEnv:
    def __init__(self, monkeypatch, tmp_path, *, classify=FailureType.SYNTAX_ERROR,
                 classifier_raises=False, orig_valid=True, validator_raises=False,
                 with_provider=True, detail_path=True):
        self.orig_valid = orig_valid
        self.validator_raises = validator_raises
        if with_provider:
            provider = SimpleNamespace(
                capabilities=lambda: SimpleNamespace(has_syntax_validator=True),
                language_id=lambda: SimpleNamespace(value="python"),
                validate_syntax=self._validate,
            )
        else:
            provider = None
        fake_reg = SimpleNamespace(get=lambda path: provider)
        _patch_lang_registry(monkeypatch, fake_reg)

        import external_llm.editor._editor_core.vm.failure_classifier as fc_mod

        def _cc(lang):
            if classifier_raises:
                raise ValueError("no classifier")
            return SimpleNamespace(classify=lambda errors: classify)

        monkeypatch.setattr(fc_mod, "create_failure_classifier", _cc)
        self.path = str(tmp_path / "m.py")
        self.detail = f"{self.path}:2:3: boom" if detail_path else "no colon pattern here"

    def _validate(self, path, code):
        if self.validator_raises:
            raise RuntimeError("validator crash")
        return SimpleNamespace(ok=self.orig_valid, errors=[])


class TestShouldSoftFailVerify:
    def test_soft_fail_empty_snapshots(self):
        assert ToolRegistry._should_soft_fail_verify("m.py:1:1: x", {}) is False

    def test_soft_fail_no_lang_provider(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, with_provider=False)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is False

    def test_soft_fail_syntax_error_is_hard(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, classify=FailureType.SYNTAX_ERROR)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is False

    def test_soft_fail_origin_broken_keeps_edit(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, orig_valid=False)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is True

    def test_soft_fail_validator_crash_orig_ok(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, validator_raises=True)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is False

    def test_soft_fail_classifier_valueerror(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, classifier_raises=True)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is False

    def test_soft_fail_argument_mismatch_is_soft(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, classify=FailureType.ARGUMENT_MISMATCH)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is True

    def test_soft_fail_unknown_is_hard(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, classify=FailureType.UNKNOWN)
        assert ToolRegistry._should_soft_fail_verify(env.detail, {env.path: "code"}) is False

    def test_soft_fail_detail_path_fallback_first_snapshot(self, tmp_path, monkeypatch):
        env = _SoftFailEnv(monkeypatch, tmp_path, detail_path=False)
        snaps = {str(tmp_path / "other.py"): "orig", env.path: "code"}
        assert ToolRegistry._should_soft_fail_verify(env.detail, snaps) is False


# ── _auto_repair_indent ─────────────────────────────────────────────────────

class TestAutoRepairIndent:
    def test_auto_repair_skips_unsupported_op_types(self):
        out = ToolRegistry._auto_repair_indent(
            "a = 1\n", [{"type": "insert_before", "anchor": "a", "content": "b"}]
        )
        assert out is None

    def test_auto_repair_anchor_missing(self):
        assert ToolRegistry._auto_repair_indent(
            "a = 1\n", [{"type": "replace", "anchor": "zzz", "content": "x"}]
        ) is None

    def test_auto_repair_midline_anchor_skipped(self):
        assert ToolRegistry._auto_repair_indent(
            "def f():\n    pass\n",
            [{"type": "replace", "anchor": "f():", "content": "g():\n    pass"}],
        ) is None

    def test_auto_repair_reindent_none_skipped(self, monkeypatch):
        monkeypatch.setattr(tr_mod, "reindent_text", lambda *a, **k: None)
        assert ToolRegistry._auto_repair_indent(
            "def f():\n    pass\n",
            [{"type": "replace", "anchor": "pass", "content": "x = 1"}],
        ) is None

    def test_auto_repair_insert_after_reindent_none_skipped(self, monkeypatch):
        monkeypatch.setattr(tr_mod, "reindent_text", lambda *a, **k: None)
        assert ToolRegistry._auto_repair_indent(
            "x = 1\n",
            [{"type": "insert_after", "anchor": "x = 1", "content": "y = 2"}],
        ) is None

    def test_auto_repair_replace_success(self):
        content = "class A:\n    def m(self):\n        pass\n"
        ops = [{"type": "replace", "anchor": "def m(self):\n        pass", "content": "def m2(self):\n    pass"}]
        out = ToolRegistry._auto_repair_indent(content, ops)
        assert out is not None
        assert "def m2(self):" in out and "class A:" in out

    def test_auto_repair_insert_after_with_trailing_newline(self):
        out = ToolRegistry._auto_repair_indent(
            "x = 1\ny = 2\n",
            [{"type": "insert_after", "anchor": "x = 1", "content": "z = 3"}],
        )
        assert out is not None and "z = 3" in out

    def test_auto_repair_insert_after_no_trailing_newline(self):
        out = ToolRegistry._auto_repair_indent(
            "x = 1\ny = 2",  # last line has no trailing newline
            [{"type": "insert_after", "anchor": "y = 2", "content": "z = 3"}],
        )
        assert out is not None and "z = 3" in out


# ── _after_write_success exceptional paths ──────────────────────────────────

class TestAfterWriteSuccess:
    def test_after_write_success_checkpoint_metadata_raises(self, reg, monkeypatch):
        monkeypatch.setattr(
            ToolRegistry, "run_checkpoint_id", property(lambda self: _raise(RuntimeError("cp boom"))())
        )
        reg._after_write_success("bash", {"command": "rm f"}, ToolResult(ok=True), {})

    def test_after_write_success_snapshot_raises(self, reg, monkeypatch):
        monkeypatch.setattr(reg, "_snapshot_target_files", _raise(RuntimeError("snap boom")))
        reg._after_write_success("edit_text", {"file_path": "a.py"}, ToolResult(ok=True), {})

    def test_after_write_success_semantic_repaired_metadata(self, reg, monkeypatch):
        monkeypatch.setattr(reg._safety_manager, "auto_repair_semantic", lambda snaps: 2)
        res = ToolResult(ok=True)
        snap = {os.path.join(reg.repo_root, "a.py"): "x = 1\n"}
        reg._after_write_success("edit_text", {"file_path": "a.py"}, res, snap)
        assert res.metadata["semantic_repaired"] == 2

    def test_after_write_success_semantic_repair_raises(self, reg, monkeypatch):
        monkeypatch.setattr(
            reg._safety_manager, "auto_repair_semantic", _raise(RuntimeError("sem boom"))
        )
        snap = {os.path.join(reg.repo_root, "a.py"): "x = 1\n"}
        reg._after_write_success(
            "edit_text", {"file_path": "a.py"}, ToolResult(ok=True), snap
        )  # degrades to a debug log

    def test_after_write_success_invalidate_raises(self, reg, monkeypatch):
        monkeypatch.setattr(reg, "_invalidate_cache_after_write", _raise(RuntimeError("inv boom")))
        monkeypatch.setattr(reg._safety_manager, "auto_repair_semantic", lambda snaps: 0)
        snap = {os.path.join(reg.repo_root, "a.py"): "x = 1\n"}
        reg._after_write_success("edit_text", {"file_path": "a.py"}, ToolResult(ok=True), snap)

    def test_after_write_success_git_cache_raises(self, reg, monkeypatch):
        import external_llm.agent.agent_context_manager as acm
        monkeypatch.setattr(acm, "_clear_git_cache", _raise(RuntimeError("git boom")))
        reg._after_write_success("bash", {"command": "rm f"}, ToolResult(ok=True), {})


# ── dispatch edge paths ─────────────────────────────────────────────────────

class TestDispatchEdges:
    def test_dispatch_non_dict_args(self, reg):
        res = reg.dispatch("read_file", "nope.py")
        assert not res.ok

    def test_dispatch_cancel_event(self, reg):
        ev = threading.Event()
        ev.set()
        reg.config.cancel_event = ev
        res = reg.dispatch("read_file", {"path": "a.py"})
        assert not res.ok and "cancelled" in (res.error or "").lower()

    def test_dispatch_profile_blocked_tool(self, reg):
        reg._agent_profile = SimpleNamespace(
            name="restricted", blocked_tools={"read_file"}, allowed_tools=None
        )
        res = reg.dispatch("read_file", {"path": "a.py"})
        assert not res.ok and "blocked" in (res.error or "")
        assert res.metadata.get("blocked") == "agent_profile"

    def test_dispatch_gate_rejection(self, reg):
        reg.config.approval_callback = lambda tool, args, preview: False
        res = reg.dispatch("write_plan", {"plan": {"kind": "ASICODE_PLAN_V1", "ops": []}})
        assert not res.ok and "rejected" in (res.error or "")

    def test_dispatch_file_lock_acquire_release(self, reg):
        released = []
        reg.config.file_lock_manager = SimpleNamespace(
            acquire_relevant=lambda args, tool: ["a.py"],
            release_all=lambda paths: released.append(paths),
        )
        res = reg.dispatch(
            "edit_text", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}
        )
        assert res.ok
        assert released == [["a.py"]]

    def test_dispatch_handler_raises(self, reg):
        reg._tool_read_file = lambda args: _raise(RuntimeError("handler boom"))()
        res = reg.dispatch("read_file", {"path": "a.py"})
        assert not res.ok and "RuntimeError" in (res.error or "")


# ── write-verify cascade (dispatch-level rollback) ──────────────────────────

def _broken_edit_args():
    return {"path": "m.py", "operations": [{"type": "replace", "anchor": "return 1", "content": "return 1 +"}]}


class TestDispatchWriteSafetyCascade:
    @staticmethod
    def _skip_syntax_gate(reg, monkeypatch):
        """edit_file's handler rolls back on syntax errors itself; bypass that
        gate so the DISPATCH-level verify (the layer under test) sees the
        broken file and drives the rollback cascade."""
        monkeypatch.setattr(reg, "_run_syntax_check_for_file", lambda path: {"skipped": True})

    def _broken_file(self, tmp_path):
        p = tmp_path / "m.py"
        p.write_text("def foo():\n    return 1\n", encoding="utf-8")
        return p

    def test_dispatch_edit_file_rollback_new_file_sentinel(self, reg, tmp_path, monkeypatch):
        p = self._broken_file(tmp_path)
        self._skip_syntax_gate(reg, monkeypatch)
        monkeypatch.setattr(reg, "_snapshot_target_files", lambda t, a: {str(p): _MISSING_SNAP})
        res = reg.dispatch("edit_file", _broken_edit_args())
        assert not res.ok and "ROLLBACK" in (res.error or "")
        assert "removed by rollback" in (res.error or "")
        assert not p.exists()

    def test_dispatch_edit_file_rollback_repair_crash(self, reg, tmp_path, monkeypatch):
        p = self._broken_file(tmp_path)
        self._skip_syntax_gate(reg, monkeypatch)
        monkeypatch.setattr(reg, "_repair_verify_failure", _raise(RuntimeError("repair boom")))
        res = reg.dispatch("edit_file", _broken_edit_args())
        assert not res.ok and "ROLLBACK" in (res.error or "")
        assert p.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_dispatch_edit_file_rollback_softfail_crash(self, reg, tmp_path, monkeypatch):
        p = self._broken_file(tmp_path)
        self._skip_syntax_gate(reg, monkeypatch)
        monkeypatch.setattr(reg, "_repair_verify_failure", lambda snaps: False)
        monkeypatch.setattr(reg, "_should_soft_fail_verify", _raise(RuntimeError("soft boom")))
        res = reg.dispatch("edit_file", _broken_edit_args())
        assert not res.ok and "ROLLBACK" in (res.error or "")
        assert p.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_dispatch_edit_file_rollback_detail_context(self, reg, tmp_path, monkeypatch):
        p = self._broken_file(tmp_path)
        self._skip_syntax_gate(reg, monkeypatch)
        res = reg.dispatch("edit_file", _broken_edit_args())
        assert not res.ok and "ROLLBACK" in (res.error or "")
        assert "AFTER ROLLBACK" in (res.error or "")
        assert "2|" in (res.error or "")  # line-numbered restored context
        assert p.read_text(encoding="utf-8") == "def foo():\n    return 1\n"

    def test_dispatch_edit_file_indent_repair_atomic_write_fails(self, reg, tmp_path, monkeypatch):
        p = self._broken_file(tmp_path)
        self._skip_syntax_gate(reg, monkeypatch)
        monkeypatch.setattr(tr_mod, "atomic_write_text", _raise(OSError("disk full")))
        res = reg.dispatch("edit_file", _broken_edit_args())
        assert not res.ok and "ROLLBACK" in (res.error or "")
        assert p.read_text(encoding="utf-8") == "def foo():\n    return 1\n"


# ── dispatch_parallel ───────────────────────────────────────────────────────

class TestDispatchParallelFallbacks:
    def test_parallel_empty_or_single_batch(self, reg):
        assert reg.dispatch_parallel([]) == []
        single = reg.dispatch_parallel([{"tool": "read_file", "args": {"path": "nope.py"}}])
        assert len(single) == 1 and not single[0].ok

    def test_parallel_disabled_falls_back_sequential(self, reg):
        reg.config.parallel_tool_execution_enabled = False
        res = reg.dispatch_parallel([
            {"tool": "read_file", "args": {"path": "nope1.py"}},
            {"tool": "read_file", "args": {"path": "nope2.py"}},
        ])
        assert len(res) == 2

    def test_parallel_write_tool_falls_back_sequential(self, reg):
        res = reg.dispatch_parallel([
            {"tool": "read_file", "args": {"path": "nope.py"}},
            {"tool": "edit_text", "args": {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}},
        ])
        assert len(res) == 2 and res[0].ok is False

    def test_parallel_serial_tool_falls_back_sequential(self, reg):
        res = reg.dispatch_parallel([
            {"tool": "read_file", "args": {"path": "nope.py"}},
            {"tool": "job", "args": {"action": "kill", "job_id": "nope"}},
        ])
        assert len(res) == 2

    def test_parallel_real_pool_success(self, reg):
        res = reg.dispatch_parallel([
            {"tool": "read_file", "args": {"path": "nope1.py"}},
            {"tool": "read_file", "args": {"path": "nope2.py"}},
        ])
        assert len(res) == 2 and not res[0].ok and not res[1].ok

    def test_parallel_exception_wrapped(self, reg, monkeypatch):
        monkeypatch.setattr(reg, "dispatch", lambda tool_name, args: _raise(RuntimeError("boom"))())
        res = reg.dispatch_parallel([
            {"tool": "read_file", "args": {"path": "nope1.py"}},
            {"tool": "grep", "args": {"pattern": "x"}},
        ])
        assert len(res) == 2
        assert all("Parallel execution error" in (r.error or "") for r in res)


# ── schema/name variants ────────────────────────────────────────────────────

class TestToolNameVariants:
    def test_get_tool_names_variants(self, reg):
        names = reg.get_tool_names()
        assert isinstance(names, frozenset) and names
        go_names = reg.get_tool_names(LanguageId.GO)  # python-only tools masked
        assert go_names <= names
        assert reg.get_tool_names(LanguageId.GO, design_chat=True)  # design-chat variant
        schemas = reg.get_tool_schemas(LanguageId.GO, design_chat=True)
        assert isinstance(schemas, list) and schemas


# ── bias-path extras ────────────────────────────────────────────────────────

REPO_ROOT = "/opt/work/myproj"


@pytest.fixture
def bias():
    """A bound _correct_bias_path with a fixed repo_root (no registry needed)."""
    stub = SimpleNamespace(repo_root=REPO_ROOT)
    return ToolRegistry._correct_bias_path.__get__(stub, type(stub))


class TestBiasPathExtras:
    def test_bias_scratch_root_not_rewritten(self, bias, monkeypatch):
        monkeypatch.setattr(tr_mod, "_under_scratch_root", lambda c: True)
        assert bias("cat /workspace/x") == "cat /workspace/x"

    def test_bias_basename_inside_quotes_protected(self, bias):
        text = 'echo "x /myproj "'
        assert bias(text) == text

    def test_bias_dedup_double_path(self, bias):
        assert bias(f"{REPO_ROOT}/myproj/tests") == f"{REPO_ROOT}/tests"

    def test_normalize_args_for_display_non_string(self, reg):
        out = reg.normalize_args_for_display({"path": "/workspace/x", "n": 42})
        assert out["n"] == 42
        assert out["path"] == os.path.join(reg.repo_root, "x")


# ── _secure_path ────────────────────────────────────────────────────────────

class TestSecurePathExtras:
    def test_secure_path_resolve_failure_none(self, reg, monkeypatch):
        monkeypatch.setattr(Path, "resolve", lambda self, *a, **k: _raise(ValueError("boom"))())
        assert reg._secure_path("a.py") is None
