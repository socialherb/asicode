"""Regression test: TSVerifier._check_tsc ignores config/environment-dependent
tsc diagnostics, keeping only genuine parser syntax errors.

Surfaced by run_20260608_055022: a valid handleLeave refactor on server.ts was
rolled back because the isolated VM verifier runs tsc on a /tmp temp file with
--ignoreConfig, so the project's esModuleInterop / @types / node_modules are
invisible. That produced false errors (TS1259 esModuleInterop, TS2503 'NodeJS',
TS2339 'readyState', TS2307 'ws') and failed the whole run, even though
`tsc -p tsconfig.json` compiles the project cleanly.

Previously only TS2307-relative-import was filtered; now all config/env-
dependent diagnostics are, leaving only real syntax errors as blocking.
"""
from types import SimpleNamespace
from unittest.mock import patch

from external_llm.editor._editor_core.ts_vm.execution_vm.verifier import TSVerifier


def _check(stdout, returncode=2):
    v = TSVerifier(language="typescript")
    proc = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    with patch("external_llm.editor._editor_core.ts_vm.execution_vm.verifier.subprocess.run",
               return_value=proc):
        return v._check_tsc("irrelevant — subprocess mocked")


# The exact diagnostics that rolled back the valid edit in run_20260608_055022.
_MIGRATION_FALSE_POSITIVES = (
    "/tmp/x.ts(1,23): error TS1259: Module can only be default-imported "
    "using the 'esModuleInterop' flag.\n"
    "/tmp/x.ts(7,23): error TS2503: Cannot find namespace 'NodeJS'.\n"
    "/tmp/x.ts(36,58): error TS2339: Property 'readyState' does not exist "
    "on type 'CustomWebSocket'.\n"
    "/tmp/x.ts(1,23): error TS2307: Cannot find module 'ws' or its "
    "corresponding type declarations.\n"
)


class TestVerifierConfigBlind:
    def test_config_env_errors_do_not_block(self):
        ok, errors = _check(_MIGRATION_FALSE_POSITIVES)
        assert ok is True
        assert errors == []

    def test_relative_import_still_ignored(self):
        ok, _ = _check("/tmp/x.ts(2,1): error TS2307: Cannot find module './types'.\n")
        assert ok is True

    def test_genuine_syntax_error_still_blocks(self):
        ok, errors = _check("/tmp/x.ts(2,5): error TS1005: ';' expected.\n")
        assert ok is False
        assert len(errors) == 1
        assert errors[0].code == "TS1005"

    def test_mixed_keeps_only_syntax(self):
        ok, errors = _check(
            _MIGRATION_FALSE_POSITIVES + "/tmp/x.ts(2,5): error TS1005: ';' expected.\n"
        )
        assert ok is False
        assert len(errors) == 1
        assert errors[0].code == "TS1005"

    def test_clean_compile_ok(self):
        ok, errors = _check("", returncode=0)
        assert ok is True
        assert errors == []
