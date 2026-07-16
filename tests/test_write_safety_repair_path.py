"""
Regression tests for the write-safety verify-failure path.

Background: commit a6c2dc15 moved external_llm/vm/* to external_llm/planner/vm/
but tool_registry.py's lazy imports (`from ..vm.failure_classifier import ...`)
were not updated. Any write tool that broke syntax then crashed the repair path
with ModuleNotFoundError — and because the exception escaped dispatch() before
reaching _restore_snapshots(), the syntax-broken file was left on disk and the
LLM received a misleading infra error instead of the verify detail.

Covers:
  1. The lazy imports in _repair_verify_failure / _should_soft_fail_verify
     resolve against the real (moved) modules.
  2. dispatch() rolls back the file even when the repair path crashes.
"""
from __future__ import annotations

import os
import shutil

import pytest

from external_llm.agent.tool_registry import ToolRegistry, ToolResult

# ── 1. Lazy import resolution (the original ModuleNotFoundError) ────────────

class TestRepairPathImports:
    def test_repair_verify_failure_imports_resolve(self):
        """With empty snapshots the method only runs its lazy imports —
        before the fix this raised ModuleNotFoundError: external_llm.vm."""
        assert ToolRegistry._repair_verify_failure(None, {}) is False

    def test_should_soft_fail_verify_imports_resolve(self):
        """Staticmethod variant of the same stale-import regression."""
        assert ToolRegistry._should_soft_fail_verify("some error", {}) is False

    def test_planner_vm_symbols_exist(self):
        """The symbols tool_registry now imports must exist at the new path."""
        from external_llm.editor._editor_core.vm.failure_classifier import (
            FailureType,
            create_failure_classifier,
        )
        from external_llm.editor._editor_core.vm.models import VerifyError
        from external_llm.editor._editor_core.vm.repair_registry import RepairRegistry

        assert FailureType.ARGUMENT_MISMATCH
        assert callable(create_failure_classifier)
        assert callable(RepairRegistry)
        assert callable(VerifyError)


# ── 2. Rollback guarantee when the repair path crashes ──────────────────────

class TestDispatchRollbackOnRepairCrash:
    def test_rollback_runs_when_repair_path_raises(
        self, tool_registry: ToolRegistry, temp_repo_root: str, monkeypatch,
    ):
        """A crash inside _repair_verify_failure must not escape dispatch():
        the snapshot rollback below it is the last line of defense keeping a
        syntax-broken file off the disk."""
        original = "def ok():\n    return 1\n"
        broken = "def ok():\n        bad indent\n"
        path = os.path.join(temp_repo_root, "mod.py")
        with open(path, "w") as f:
            f.write(original)

        # Fake write tool: actually corrupts the file, reports success
        def _fake_edit_file(args):
            with open(path, "w") as f:
                f.write(broken)
            return ToolResult(ok=True, content="written")

        monkeypatch.setattr(tool_registry, "_tool_edit_file", _fake_edit_file)
        monkeypatch.setattr(
            tool_registry, "_verify_after_write",
            lambda snaps: (False, f"{path}:2:9: Syntax error: unexpected indent"),
        )

        def _crashing_repair(snapshots):
            raise RuntimeError("simulated repair-path crash (e.g. broken lazy import)")

        monkeypatch.setattr(tool_registry, "_repair_verify_failure", _crashing_repair)

        # Must not raise — the verify-failure handling catches internal crashes
        result = tool_registry.dispatch("edit_file", {"path": "mod.py"})

        assert result.ok is False
        with open(path) as f:
            assert f.read() == original  # rolled back, not left broken

    def test_rollback_runs_when_soft_fail_classifier_raises(
        self, tool_registry: ToolRegistry, temp_repo_root: str, monkeypatch,
    ):
        """Same guarantee for the soft-fail classification step: a crash there
        must be treated as hard fail (rollback), never propagate."""
        original = "x = 1\n"
        path = os.path.join(temp_repo_root, "mod2.py")
        with open(path, "w") as f:
            f.write(original)

        def _fake_edit_file(args):
            with open(path, "w") as f:
                f.write("x ===== broken\n")
            return ToolResult(ok=True, content="written")

        monkeypatch.setattr(tool_registry, "_tool_edit_file", _fake_edit_file)
        monkeypatch.setattr(
            tool_registry, "_verify_after_write",
            lambda snaps: (False, f"{path}:1:3: Syntax error: invalid syntax"),
        )
        monkeypatch.setattr(
            tool_registry, "_repair_verify_failure", lambda snapshots: False,
        )

        def _crashing_classifier(verify_detail, snapshots):
            raise RuntimeError("simulated classifier crash")

        monkeypatch.setattr(
            tool_registry, "_should_soft_fail_verify", _crashing_classifier,
        )

        result = tool_registry.dispatch("edit_file", {"path": "mod2.py"})

        assert result.ok is False
        with open(path) as f:
            assert f.read() == original


# ── 3. All-files contract: partial repair must not ship a multi-file write ─
#
# Regression: _repair_verify_failure's docstring promised "all files re-verify
# clean", but its body returned True after repairing a SINGLE file in the
# snapshot dict. Because dispatch() trusts a True return as a final green light
# (it returns ``result`` without re-verifying), a multi-file write_plan /
# apply_patch could succeed on one file while leaving another file's genuine
# syntax error silently on disk. The fix adds a full _verify_after_write gate
# before returning True.

class _SyntaxRes:
    """Lightweight validation result used by the fake providers below."""
    def __init__(self, ok, errors=None):
        self.ok = ok
        self.errors = errors or []


class TestAllFilesRepairContract:
    def test_partial_repair_does_not_leave_other_file_broken(
        self, tool_registry: ToolRegistry, temp_repo_root: str, monkeypatch,
    ):
        """Multi-file snapshot: file_a is ARGUMENT_MISMATCH (repairable),
        file_b is a genuine SYNTAX_ERROR (unrepairable). Repair fixes file_a
        but must NOT return True — the caller would then ship the write and
        leave file_b corrupted on disk."""
        import os

        file_a = os.path.join(temp_repo_root, "a.py")
        file_b = os.path.join(temp_repo_root, "b.py")
        orig_a, orig_b = "# original a\n", "# original b\n"
        with open(file_a, "w") as f:
            f.write(orig_a)
        with open(file_b, "w") as f:
            f.write(orig_b)

        # Snapshots captured *before* the (simulated) write.
        snapshots = {file_a: orig_a, file_b: orig_b}

        # Simulate the write having corrupted both files on disk.
        broken_a = "def a(\n    pass\n"          # repairable-looking break
        broken_b = "def b(:\n    pass\n"         # genuine syntax error
        with open(file_a, "w") as f:
            f.write(broken_a)
        with open(file_b, "w") as f:
            f.write(broken_b)

        # _verify_after_write is the public re-verify entry point used by both
        # dispatch() and (post-fix) the repair gate. We make it reflect the
        # on-disk truth: failing while EITHER file is broken, passing only once
        # BOTH are restored to valid Python.
        def _real_verify(snaps):
            import ast as _ast
            for p in snaps:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                try:
                    _ast.parse(src, filename=p)
                except SyntaxError as exc:
                    return False, f"{p}:{exc.lineno or 1}:{exc.offset or 1}: {exc.msg}"
            return True, ""

        monkeypatch.setattr(tool_registry, "_verify_after_write", _real_verify)

        # Repair fixes ONLY file_a (writes valid code). file_b stays broken.
        from external_llm.editor._editor_core.vm.failure_classifier import (
            FailureType,
        )

        class _FakeClassifier:
            def classify(self, verrs):
                msg = verrs[0].message.lower()
                return (
                    FailureType.ARGUMENT_MISMATCH
                    if "positional argument" in msg
                    else FailureType.SYNTAX_ERROR
                )

        class _FakeProvider:
            def language_id(self):
                class _L:
                    value = "python"
                return _L()

            def capabilities(self):
                class _C:
                    has_syntax_validator = True
                return _C()

            def validate_syntax(self, path, content):
                import ast as _ast

                from external_llm.languages.models import SyntaxError_ as _SE
                try:
                    _ast.parse(content, filename=path)
                    compile(content, path, "exec")
                except SyntaxError as _e:
                    msg = (
                        "missing 1 required positional argument"
                        if path == file_a
                        else "Syntax error: unexpected EOF"
                    )
                    return _SyntaxRes(ok=False, errors=[_SE(
                        file=path, line=_e.lineno or 1, col=_e.offset or 1,
                        message=msg,
                    )])
                return _SyntaxRes(ok=True, errors=[])

        _fake_provider = _FakeProvider()

        import external_llm.agent.tool_safety as _ts

        class _FakeLR:
            @staticmethod
            def instance():
                return _FakeLR()

            def get(self, path):
                return _fake_provider

        # Patch the LanguageRegistry lookup used inside _repair_verify_failure.
        monkeypatch.setattr(_ts, "LanguageRegistry", _FakeLR, raising=False)
        # The lazy import inside the method is `from ..languages import
        # LanguageRegistry as _LR`; patch at the languages package too.
        import external_llm.languages as _langpkg

        monkeypatch.setattr(_langpkg, "LanguageRegistry", _FakeLR, raising=False)
        monkeypatch.setattr(
            "external_llm.languages.LanguageRegistry", _FakeLR, raising=False,
        )

        # Stub the registry/classifier so file_a is repairable, file_b is not.
        import external_llm.editor._editor_core.vm.failure_classifier as _fc
        import external_llm.editor._editor_core.vm.repair_registry as _rr

        monkeypatch.setattr(_fc, "create_failure_classifier", lambda lang: _FakeClassifier())

        class _FakeOp:
            def __init__(self, payload):
                self.payload = payload
                self.kind = type("K", (), {"value": "RAW_REPLACE"})

        class _FakeRegistry:
            def __init__(self, lang):
                pass

            def get(self, ftype):
                if ftype == FailureType.ARGUMENT_MISMATCH:
                    return lambda code, verr, clf: [_FakeOp(
                        {"__raw_code__": "def a():\n    return 1\n"}
                    )]
                return None

        monkeypatch.setattr(_rr, "RepairRegistry", _FakeRegistry)

        # --- Act: repair must report FAILURE (False) because file_b is broken ---
        repaired = tool_registry._repair_verify_failure(snapshots)

        # Before the fix this returned True (only file_a was checked). The
        # all-files gate now catches the still-broken file_b.
        assert repaired is False, (
            "partial repair must not claim success while another file is broken"
        )

        # And the caller (dispatch) would then roll back — here we assert the
        # contract directly: a False return is the signal to restore snapshots.
        tool_registry._restore_snapshots(snapshots)
        with open(file_a) as f:
            assert f.read() == orig_a
        with open(file_b) as f:
            assert f.read() == orig_b

    def test_full_repair_returns_true_when_all_files_clean(
        self, tool_registry: ToolRegistry, temp_repo_root: str, monkeypatch,
    ):
        """Positive control: when ALL files in the snapshot are repairable,
        _repair_verify_failure still returns True (the gate must not be
        over-conservative and reject genuinely-clean writes)."""
        import os

        file_a = os.path.join(temp_repo_root, "a.py")
        file_b = os.path.join(temp_repo_root, "b.py")
        orig_a, orig_b = "# original a\n", "# original b\n"
        with open(file_a, "w") as f:
            f.write(orig_a)
        with open(file_b, "w") as f:
            f.write(orig_b)

        snapshots = {file_a: orig_a, file_b: orig_b}

        # Both files "broken" on disk but both repairable.
        with open(file_a, "w") as f:
            f.write("def a(\n    pass\n")
        with open(file_b, "w") as f:
            f.write("def b(\n    pass\n")

        def _real_verify(snaps):
            import ast as _ast
            for p in snaps:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                try:
                    _ast.parse(src, filename=p)
                except SyntaxError as exc:
                    return False, f"{p}:{exc.lineno or 1}:{exc.offset or 1}: {exc.msg}"
            return True, ""

        monkeypatch.setattr(tool_registry, "_verify_after_write", _real_verify)

        from external_llm.editor._editor_core.vm.failure_classifier import (
            FailureType,
        )

        class _FakeClassifier:
            def classify(self, verrs):
                return FailureType.ARGUMENT_MISMATCH

        class _FakeProvider:
            def language_id(self):
                class _L:
                    value = "python"
                return _L()

            def capabilities(self):
                class _C:
                    has_syntax_validator = True
                return _C()

            def validate_syntax(self, path, content):
                import ast as _ast

                from external_llm.languages.models import SyntaxError_ as _SE
                try:
                    _ast.parse(content, filename=path)
                    compile(content, path, "exec")
                except SyntaxError as _e:
                    return _SyntaxRes(ok=False, errors=[_SE(
                        file=path, line=_e.lineno or 1, col=_e.offset or 1,
                        message="missing 1 required positional argument",
                    )])
                return _SyntaxRes(ok=True, errors=[])

        _fake_provider = _FakeProvider()
        import external_llm.agent.tool_safety as _ts
        import external_llm.languages as _langpkg

        class _FakeLR:
            @staticmethod
            def instance():
                return _FakeLR()

            def get(self, path):
                return _fake_provider

        monkeypatch.setattr(_ts, "LanguageRegistry", _FakeLR, raising=False)
        monkeypatch.setattr(_langpkg, "LanguageRegistry", _FakeLR, raising=False)
        monkeypatch.setattr(
            "external_llm.languages.LanguageRegistry", _FakeLR, raising=False,
        )

        import external_llm.editor._editor_core.vm.failure_classifier as _fc
        import external_llm.editor._editor_core.vm.repair_registry as _rr

        monkeypatch.setattr(_fc, "create_failure_classifier", lambda lang: _FakeClassifier())

        class _FakeOp:
            def __init__(self, payload):
                self.payload = payload
                self.kind = type("K", (), {"value": "RAW_REPLACE"})

        class _FakeRegistry:
            def __init__(self, lang):
                pass

            def get(self, ftype):
                if ftype == FailureType.ARGUMENT_MISMATCH:
                    # Repair writes valid code for whichever file.
                    def _strategy(code, verr, clf):
                        # Infer the file from the stored path on the verr-like;
                        # fallback to a.py.
                        return [_FakeOp({"__raw_code__": "def x():\n    return 1\n"})]
                    return _strategy
                return None

        monkeypatch.setattr(_rr, "RepairRegistry", _FakeRegistry)

        repaired = tool_registry._repair_verify_failure(snapshots)
        assert repaired is True, (
            "all-files-clean write must still pass the repair gate"
        )


# ── 4. edit_text language-neutral syntax gate ──────────────────────────────
#
# Regression: edit_text is excluded from dispatch's snapshot+verify+rollback
# cycle (tool_registry.py:1264/1271) and its own syntax gate was Python-only
# (write_tools.py ``if LanguageId.from_path(...) is PYTHON``). Non-Python files
# (TS/JS/Go/JSON…) edited via edit_text therefore had NO syntax safety net: a
# broken new_string went straight to disk. The fix runs the SAME
# provider.validate_syntax the dispatch path uses, mirroring soft-fail so
# edit_text is neither stricter nor looser than apply_patch/edit_file.

class TestEditTextLanguageNeutralGate:
    """edit_text must refuse broken non-Python edits in-memory (no rollback)."""

    def _write(self, repo_root, name, content):
        path = os.path.join(repo_root, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_refuses_broken_typescript(self, tool_registry, temp_repo_root):
        """Core defect: a .ts edit that breaks parsing must be refused and the
        file left untouched (edit_text has no rollback, so the in-memory gate is
        the only thing keeping broken TS off the disk)."""
        original = "function add(a: number, b: number): number {\n  return a + b;\n}\n"
        path = self._write(temp_repo_root, "app.ts", original)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "app.ts",
            "old_string": "  return a + b;",
            "new_string": "  return a + ;",
        })
        assert result.ok is False
        assert "syntax error" in (result.error or "").lower()
        with open(path) as f:
            assert f.read() == original  # disk preserved

    def test_applies_valid_typescript(self, tool_registry, temp_repo_root):
        """A valid .ts edit must still apply — the gate must not over-fire."""
        original = "function add(a: number, b: number): number {\n  return a + b;\n}\n"
        path = self._write(temp_repo_root, "app.ts", original)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "app.ts",
            "old_string": "  return a + b;",
            "new_string": "  return a - b;",
        })
        assert result.ok
        with open(path) as f:
            assert "return a - b" in f.read()

    def test_refuses_broken_json(self, tool_registry, temp_repo_root):
        original = '{"a": 1, "b": 2}'
        path = self._write(temp_repo_root, "cfg.json", original)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "cfg.json",
            "old_string": '"b": 2',
            "new_string": '"b": ,',
        })
        assert result.ok is False
        with open(path) as f:
            assert f.read() == original

    def test_skips_gate_when_original_already_broken(self, tool_registry, temp_repo_root):
        """When the ORIGINAL file already fails to parse, the gate must not
        block an edit (we never refuse an edit fixing a pre-existing error) —
        matching the Python branch's ``_orig_parses`` skip."""
        pre_broken = "function f() {\n  return 1 + ;\n}\n"
        path = self._write(temp_repo_root, "broken.ts", pre_broken)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "broken.ts",
            "old_string": "  return 1 + ;",
            "new_string": "  return 2 + ;",
        })
        assert result.ok
        with open(path) as f:
            assert "return 2 +" in f.read()

    def test_unknown_language_skips_gate(self, tool_registry, temp_repo_root):
        """Files with no language provider (e.g. .txt) must apply normally."""
        path = self._write(temp_repo_root, "notes.txt", "hello world\n")
        result = tool_registry.dispatch("edit_text", {
            "file_path": "notes.txt",
            "old_string": "hello",
            "new_string": "goodbye",
        })
        assert result.ok
        with open(path) as f:
            assert "goodbye" in f.read()

    def test_python_gate_unaffected(self, tool_registry, temp_repo_root):
        """Regression guard: the pre-existing Python compile() gate still
        refuses broken .py edits (the non-Python gate must not disturb it)."""
        original = "def f():\n    return 1\n"
        path = self._write(temp_repo_root, "mod.py", original)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "mod.py",
            "old_string": "    return 1",
            "new_string": "    return 1 +",
        })
        assert result.ok is False
        with open(path) as f:
            assert f.read() == original

    @pytest.mark.skipif(
        shutil.which("go") is None, reason="go toolchain not installed"
    )
    def test_go_soft_fail_undefined_is_kept(
        self, tool_registry, temp_repo_root
    ):
        """Consistency with dispatch: a Go edit introducing ``undefined: foo``
        (a cross-file resolvable reference) must be KEPT, not refused — exactly
        as apply_patch/edit_file keep soft-fail errors. A naive gate that
        refuses on ``not ok`` would make edit_text stricter than its siblings."""
        original = "package main\n\nfunc work() int {\n\treturn 1\n}\n"
        path = self._write(temp_repo_root, "f.go", original)
        result = tool_registry.dispatch("edit_text", {
            "file_path": "f.go",
            "old_string": "\treturn 1",
            "new_string": "\treturn helperFunc()",
        })
        assert result.ok, "Go soft-fail (undefined) must be kept, not refused"
        with open(path) as f:
            assert "helperFunc()" in f.read()


# ── 5. Origin-skip guard + Kotlin "cannot infer type" + Bug B ────────────────
#
# Regression: modify_symbol on a non-Python file whose pre-edit content ALSO
# fails isolated-compile (e.g. an Android ViewModel without the SDK, Kotlin
# without coroutines) triggered a spurious rollback. The isolated-compile
# errors are environmental cascade noise, not caused by the edit. The fix
# mirrors edit_text's ``_et_orig_ok`` gate: when the pre-edit origin also
# fails validate_syntax, soft-fail (keep the edit).
#
# Also covers:
#   - Kotlin "cannot infer type" classified as TYPE_MISMATCH (was UNKNOWN →
#     hard fail-rollback) so it soft-fails even when origin parses OK.
#   - Bug B: on a genuine rollback, dispatch must undo the handler's
#     _text_edited_files recording, else a later apply_patch to the file is
#     wrongly refused with "already edited this session".


class _FakeValResult:
    def __init__(self, ok, message="err"):
        self.ok = ok
        self.errors = [type("_E", (), {"file": "", "line": 1, "col": 1, "message": message})()]


class _FakeCaps:
    has_syntax_validator = True


class _FakeProvider:
    """validate_syntax fails iff content contains the marker 'MISSING_DEP'."""
    _lang = "kotlin"

    def language_id(self):
        return type("_L", (), {"value": self._lang})()

    def capabilities(self):
        return _FakeCaps()

    def validate_syntax(self, path, content):
        return _FakeValResult(ok="MISSING_DEP" not in content, message="cannot infer type")


class _FakeLangRegistry:
    """Stand-in for LanguageRegistry.instance().get() during the unit test."""
    @classmethod
    def instance(cls):
        class _Inst:
            def get(self, path):
                return _FakeProvider() if str(path).endswith(".kt") else None
        return _Inst()


class TestOriginSkipGuard:
    def test_kotlin_cannot_infer_type_is_type_mismatch(self):
        """Keyword fix: 'cannot infer type' → TYPE_MISMATCH (was UNKNOWN → hard
        fail-rollback). TYPE_MISMATCH soft-fails so the edit is kept."""
        from external_llm.editor._editor_core.vm.failure_classifier import (
            FailureType,
            create_failure_classifier,
        )
        from external_llm.editor._editor_core.vm.models import VerifyError
        c = create_failure_classifier("kotlin")
        ft = c.classify([VerifyError(
            message="cannot infer type for type parameter 'T'. Specify it explicitly.",
            line=0, column=0)])
        assert ft == FailureType.TYPE_MISMATCH

    def test_origin_skip_keeps_edit_when_origin_also_broken(self, monkeypatch):
        """Origin-skip (the general root-cause guard): when the pre-edit content
        ALSO fails isolated-compile, the verify errors are environmental cascade
        noise → soft-fail (keep the edit). Before the fix this hard-fail-rolled
        back a correct modify_symbol on an Android/Kotlin file without its deps."""
        monkeypatch.setattr("external_llm.languages.LanguageRegistry", _FakeLangRegistry)
        snapshots = {"/repo/Foo.kt": "class Foo { val x: MISSING_DEP = TODO() }"}
        assert ToolRegistry._should_soft_fail_verify(
            "/repo/Foo.kt:1:1: cannot infer type for type parameter 'T'", snapshots,
        ) is True

    def test_origin_skip_does_not_mask_genuine_syntax_error(self, monkeypatch):
        """Negative case: origin parses OK, edit introduces a genuine syntax
        error → hard fail (return False, roll back). The origin-skip must NOT
        fire here because the origin validated clean."""
        monkeypatch.setattr("external_llm.languages.LanguageRegistry", _FakeLangRegistry)
        snapshots = {"/repo/Foo.kt": "class Foo { fun bar() {} }"}  # origin OK
        assert ToolRegistry._should_soft_fail_verify(
            "/repo/Foo.kt:1:1: expecting '}'", snapshots,
        ) is False

    def test_origin_skip_skipped_for_new_file_snapshot(self, monkeypatch):
        """A new-file snapshot holds the _MISSING_SNAP sentinel (not str) — no
        origin to validate. The origin-skip must be skipped, falling through to
        the normal classifier so a genuine syntax error in a NEW file still
        hard-fails."""
        from external_llm.agent.tool_safety import _MISSING_SNAP
        monkeypatch.setattr("external_llm.languages.LanguageRegistry", _FakeLangRegistry)
        snapshots = {"/repo/New.kt": _MISSING_SNAP}
        assert ToolRegistry._should_soft_fail_verify(
            "/repo/New.kt:1:1: expecting '}'", snapshots,
        ) is False

    # ── Bug #1: multi-file snapshot origin-skip ────────────────────────────────
    # _should_soft_fail_verify must use the FILE THAT PRODUCED THE ERROR for
    # origin-skip + classification, not the FIRST snapshot file. These tests
    # prove the fix by constructing a multi-file snapshot where first-file
    # origin differs from the error's-file origin.

    def test_multi_file_first_file_broken_uses_detail_path_origin(self, monkeypatch):
        """Multi-file snapshot: first file origin broken, verify error from
        second file (clean origin). MUST NOT origin-skip — the second file's
        origin is clean, so the error is a genuine syntax error → hard fail.
        Before the fix the origin-skip fired on the first file's broken origin
        and incorrectly returned True (soft-fail)."""
        monkeypatch.setattr("external_llm.languages.LanguageRegistry", _FakeLangRegistry)
        snapshots = {
            "/repo/A.kt": "class Foo { val x: MISSING_DEP = TODO() }",  # broken
            "/repo/B.kt": "class Bar {}",  # clean
        }
        result = ToolRegistry._should_soft_fail_verify(
            "/repo/B.kt:1:1: expecting '}'", snapshots,
        )
        assert result is False, (
            "origin from B.kt (clean) must not trigger origin-skip; "
            "expecting '}' is SYNTAX_ERROR → hard fail"
        )

    def test_multi_file_second_file_origin_broken_triggers_origin_skip(self, monkeypatch):
        """Multi-file snapshot: first file origin clean, verify error from
        second file (broken origin). MUST origin-skip even though the first
        file's origin is clean — the error is from a file whose baseline was
        already broken (cascade noise)."""
        monkeypatch.setattr("external_llm.languages.LanguageRegistry", _FakeLangRegistry)
        snapshots = {
            "/repo/A.kt": "class Foo {}",  # clean
            "/repo/B.kt": "class Bar { val x: MISSING_DEP = TODO() }",  # broken
        }
        result = ToolRegistry._should_soft_fail_verify(
            "/repo/B.kt:1:1: expecting '}'", snapshots,
        )
        assert result is True, (
            "origin from B.kt (broken) must trigger origin-skip → soft fail"
        )


class TestRollbackRestoresTextEditedFiles:
    """Bug B: on a genuine rollback, dispatch must restore _text_edited_files to
    its pre-handler state. A non-excluded write tool (modify_symbol) records the
    edited path from INSIDE its handler (before dispatch's verify); without the
    restore, the stale entry makes a later apply_patch refuse with 'already
    edited this session'."""

    def test_rollback_undoes_handler_recording(
        self, tool_registry: ToolRegistry, temp_repo_root: str, monkeypatch,
    ):
        original = "x = 1\n"
        path = os.path.join(temp_repo_root, "mod_b.py")
        with open(path, "w") as f:
            f.write(original)

        recorded = {"path": os.path.join(temp_repo_root, "mod_b.py")}
        _edited_before = set(tool_registry._text_edited_files)

        def _fake_modify_symbol(args):
            # Mirror write_tools.py:5078 — records the path BEFORE dispatch verify
            with open(path, "w") as f:
                f.write("x === broken\n")
            tool_registry._record_text_edit(os.path.join(temp_repo_root, "mod_b.py"))
            recorded["path"] = os.path.join(temp_repo_root, "mod_b.py")
            return ToolResult(ok=True, content="written")

        monkeypatch.setattr(tool_registry, "_tool_modify_symbol", _fake_modify_symbol)
        monkeypatch.setattr(
            tool_registry, "_verify_after_write",
            lambda snaps: (False, f"{path}:1:3: Syntax error: invalid syntax"),
        )
        monkeypatch.setattr(
            tool_registry, "_repair_verify_failure", lambda snapshots: False,
        )
        # Force hard fail: no soft-fail, no origin-skip
        monkeypatch.setattr(
            ToolRegistry, "_should_soft_fail_verify", staticmethod(lambda d, s: False),
        )

        result = tool_registry.dispatch("modify_symbol", {"path": "mod_b.py"})

        # (a) dispatch rolled back the working tree
        assert result.ok is False
        with open(path) as f:
            assert f.read() == original
        # (b) Bug B: the handler's _text_edited_files recording was undone
        assert tool_registry._text_edited_files == _edited_before, (
            "rollback must restore _text_edited_files so a later apply_patch "
            "is not refused with 'already edited this session'"
        )
        assert recorded["path"] not in tool_registry._text_edited_files


# ── 6. anchor_edit language-neutral gate: soft-fail + origin-skip ─────────────
# Regression (Finding #1): anchor_edit's syntax gate was a hardcoded
# ``if not _sv.ok: refuse`` with NO origin-skip and NO soft-fail classification
# — unlike edit_text/dispatch/apply_patch/modify_symbol which all keep edits
# whose pre-edit content also fails isolated-compile (cascade noise) or whose
# errors are cross-file-resolvable. For a non-Python file whose original already
# fails parse (e.g. a pre-broken .ts, or an Android Kotlin file without the SDK),
# anchor_edit refused EVERY edit. The fix mirrors edit_text: soft-fail via
# _should_soft_fail_verify, and keep the strict refuse for Python (compile() is
# self-contained) + genuine syntax errors on a valid baseline.

class TestAnchorEditLanguageNeutralGate:
    """anchor_edit must refuse genuine syntax errors but keep origin-skip /
    soft-fail edits — matching edit_text/dispatch, not the old hardcoded gate."""

    def _write(self, repo_root, name, content):
        path = os.path.join(repo_root, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_refuses_broken_typescript_insert(self, tool_registry, temp_repo_root):
        """A .ts anchor_edit that introduces a genuine parse error must be
        refused and the file left untouched (anchor_edit has no rollback, so
        the in-handler gate is the only safety net — see insight A68)."""
        original = (
            "function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
        )
        path = self._write(temp_repo_root, "app.ts", original)
        result = tool_registry.dispatch("anchor_edit", {
            "file_path": "app.ts",
            "anchor_pattern": "  return a + b;",
            "edit_mode": "insert_after",
            "code_snippet": "  return a + ;",
        })
        assert result.ok is False
        with open(path) as f:
            assert f.read() == original  # disk preserved

    def test_skips_gate_when_original_already_broken(self, tool_registry, temp_repo_root):
        """The core Finding #1 fix: when the ORIGINAL file already fails to
        parse, anchor_edit must NOT refuse the edit (we never block an edit on a
        pre-broken baseline). The old hardcoded gate refused every such edit."""
        pre_broken = "function f() {\n  return 1 + ;\n}\n"
        path = self._write(temp_repo_root, "broken.ts", pre_broken)
        result = tool_registry.dispatch("anchor_edit", {
            "file_path": "broken.ts",
            "anchor_pattern": "  return 1 + ;",
            "edit_mode": "replace_line",
            "code_snippet": "  return 2 + ;",
        })
        assert result.ok, "origin-skip must keep edits on a pre-broken baseline"
        with open(path) as f:
            assert "return 2 +" in f.read()

    def test_applies_valid_typescript_insert(self, tool_registry, temp_repo_root):
        """A valid .ts anchor_edit must still apply — the gate must not over-fire
        on a clean baseline."""
        original = (
            "function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
        )
        path = self._write(temp_repo_root, "app.ts", original)
        result = tool_registry.dispatch("anchor_edit", {
            "file_path": "app.ts",
            "anchor_pattern": "  return a + b;",
            "edit_mode": "insert_after",
            "code_snippet": "  console.log(a);",
        })
        assert result.ok
        with open(path) as f:
            assert "console.log(a)" in f.read()

    def test_python_gate_still_refuses_broken(self, tool_registry, temp_repo_root):
        """Regression guard: Python keeps the strict refuse (compile() is
        self-contained — no cascade noise). The non-Python soft-fail path must
        not weaken the Python gate."""
        original = "def f():\n    return 1\n"
        path = self._write(temp_repo_root, "mod.py", original)
        result = tool_registry.dispatch("anchor_edit", {
            "file_path": "mod.py",
            "anchor_pattern": "    return 1",
            "edit_mode": "replace_line",
            "code_snippet": "    return 1 +",
        })
        assert result.ok is False
        with open(path) as f:
            assert f.read() == original
