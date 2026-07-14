"""Regression tests: TypeScriptSyntaxProvider.validate_syntax reports genuine
SYNTAX errors only (TS1xxx), not environment-dependent type/semantic ones.

Surfaced by run_20260608_052739: a JS→TS migration's server.ts produced only
type errors (TS2591 'process', TS2503 'NodeJS', TS7016 'ws') because the repo
had no @types/node / node_modules. Once create_file was wired into the quality
gate, those env-dependent errors would have wrongly failed the create. A syntax
validator must ignore them and only flag real parser errors (unbalanced braces,
missing tokens, …).

subprocess.run is mocked so these tests are fast and don't require tsc.
"""
from types import SimpleNamespace
from unittest.mock import patch

from external_llm.languages.typescript_provider import (
    TypeScriptSyntaxProvider,
    is_genuine_syntax_error,
)


def _fake_proc(returncode, stdout):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _validate(stdout, returncode=2):
    p = TypeScriptSyntaxProvider()
    with patch("external_llm.languages.typescript_provider.subprocess.run",
               return_value=_fake_proc(returncode, stdout)):
        return p.validate_syntax("server.ts", "irrelevant — subprocess mocked")


class TestTypeScriptSyntaxOnly:
    def test_type_and_env_errors_are_ignored(self):
        out = (
            "server.ts(7,23): error TS2503: Cannot find namespace 'NodeJS'.\n"
            "server.ts(122,66): error TS2591: Cannot find name 'process'.\n"
            "server.ts(1,23): error TS7016: Could not find a declaration file for module 'ws'.\n"
            "client.ts(293,11): error TS2339: Property 'disabled' does not exist on type 'HTMLElement'.\n"
        )
        r = _validate(out)
        assert r.ok is True
        assert not r.errors

    def test_genuine_syntax_errors_still_fail(self):
        out = (
            "bad.ts(1,12): error TS1005: ':' expected.\n"
            "bad.ts(2,13): error TS1109: Expression expected.\n"
        )
        r = _validate(out)
        assert r.ok is False
        assert len(r.errors) == 2
        assert all(e.message.startswith("TS1") for e in r.errors)

    def test_mixed_keeps_only_syntax(self):
        out = (
            "f.ts(1,1): error TS2591: Cannot find name 'process'.\n"   # ignored
            "f.ts(2,5): error TS1005: ';' expected.\n"                 # kept
        )
        r = _validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert r.errors[0].message.startswith("TS1005")

    def test_clean_compile_is_ok(self):
        r = _validate("", returncode=0)
        assert r.ok is True

    def test_interop_1xxx_codes_are_not_syntax_errors(self):
        # TS1259 lives in the 1xxx band but is an esModuleInterop CONFIG
        # diagnostic — it fired on the migrated server.ts and rolled back a
        # valid edit. It must be ignored by a config-blind syntax check.
        out = (
            "server.ts(1,23): error TS1259: Module 'ws' can only be "
            "default-imported using the 'esModuleInterop' flag.\n"
        )
        r = _validate(out)
        assert r.ok is True
        assert not r.errors


class TestIsGenuineSyntaxError:
    def test_parser_syntax_codes_pass(self):
        for c in ("TS1005", "TS1109", "TS1128", "TS1003"):
            assert is_genuine_syntax_error(c) is True

    def test_config_interop_1xxx_excluded(self):
        for c in ("TS1259", "TS1192", "TS1208", "TS1371", "TS1479"):
            assert is_genuine_syntax_error(c) is False

    def test_type_semantic_env_codes_excluded(self):
        for c in ("TS2307", "TS2503", "TS2591", "TS2339", "TS7016", "TS5112"):
            assert is_genuine_syntax_error(c) is False

    def test_malformed_input(self):
        for c in ("", None, "1005", "TSabc"):
            assert is_genuine_syntax_error(c) is False
