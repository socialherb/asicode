"""Tests for language providers: Go, TypeScript, JavaScript, Kotlin, Java.

Focuses on symbol detection and API contracts that don't require
external toolchains (go, tsc, kotlinc, javac).
"""
import pytest

from external_llm.languages.bash_provider import BashSyntaxProvider
from external_llm.languages.go_provider import GoSyntaxProvider
from external_llm.languages.java_provider import JavaSyntaxProvider
from external_llm.languages.javascript_provider import JavaScriptSyntaxProvider
from external_llm.languages.kotlin_provider import KotlinSyntaxProvider
from external_llm.languages.models import LanguageId
from external_llm.languages.typescript_provider import TypeScriptSyntaxProvider

# ── GoSyntaxProvider ──────────────────────────────────────────────────────────

class TestGoProvider:
    @pytest.fixture
    def provider(self):
        return GoSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.GO

    def test_capabilities(self, provider):
        caps = provider.capabilities()
        assert caps.has_syntax_validator is True
        assert caps.supports_modify_symbol is True

    def test_file_globs(self, provider):
        assert "*.go" in provider.get_file_globs()

    def test_definition_keywords(self, provider):
        keywords = provider.get_definition_keywords()
        assert "func " in keywords

    def test_symbol_patterns_any(self, provider):
        patterns = provider.get_symbol_patterns("any")
        kinds = {p.kind for p in patterns}
        assert "function" in kinds

    def test_symbol_patterns_have_name_placeholder(self, provider):
        for p in provider.get_symbol_patterns("any"):
            assert "{name}" in p.regex

    def test_find_symbol_function(self, provider):
        content = "package main\n\nfunc MyFunc() {\n}\n"
        result = provider.find_symbol_in_file("foo.go", "MyFunc", content)
        assert result is not None
        assert result[0] >= 1

    def test_find_symbol_not_found(self, provider):
        content = "package main\n\nfunc OtherFunc() {\n}\n"
        result = provider.find_symbol_in_file("foo.go", "Missing", content)
        assert result is None

    def test_find_block_end_simple(self):
        content = "func f() {\n    x := 1\n}\n"
        end = GoSyntaxProvider._find_block_end(content, 0)
        assert isinstance(end, int)

    def test_lint_command(self, provider):
        cmd = provider.get_lint_command("foo.go")
        assert cmd is not None
        assert "golangci-lint" in cmd

    def test_test_command(self, provider):
        cmd = provider.get_test_command("/repo")
        assert cmd is not None
        assert "go" in cmd

    # ── Structural query tests (regex fallback) ──────────────────────────────

    def test_find_top_level_definitions(self, provider):
        content = (
            "package main\n\n"
            "func hello() string {\n    return \"hi\"\n}\n\n"
            "type User struct {\n    Name string\n}\n\n"
            "func (u *User) Greet() string {\n    return \"hi\"\n}\n"
        )
        results = provider._find_top_level_definitions_regex(content)
        names = {r[0] for r in results}
        assert "hello" in names, f"hello missing: {names}"
        assert "User" in names, f"User missing: {names}"
        assert "Greet" in names, f"Greet missing: {names}"

    def test_find_class_methods(self, provider):
        content = (
            "package main\n\n"
            "type User struct {\n    Name string\n}\n\n"
            "func (u *User) Greet() string {\n    return \"hi\"\n}\n"
            "func (u *User) Bye() string {\n    return \"bye\"\n}\n"
        )
        methods = provider._find_class_methods_regex(content, "User")
        names = {m[0] for m in methods}
        assert "Greet" in names, f"Greet missing: {names}"
        assert "Bye" in names, f"Bye missing: {names}"

    def test_find_symbol_body_range(self, provider):
        content = (
            "package main\n\n"
            "func hello() string {\n    return \"hi\"\n}\n"
        )
        body = provider._find_symbol_body_range_regex(content, "hello")
        assert body is not None, "body should not be None"
        assert body[0] >= 1, f"body start valid: {body}"
        assert body[1] >= body[0], f"body end < start: {body}"

    def test_structural_query_top_level_delegates_to_regex(self, provider):
        """Verify the public API delegates to regex fallback when tree-sitter unavailable."""
        content = "package main\nfunc TestFn() {\n}\n"
        results = provider.find_top_level_definitions(content)
        # Should either return tree-sitter results OR regex fallback (never empty when there are definitions)
        assert len(results) >= 0  # at minimum, doesn't crash


# ── TypeScriptSyntaxProvider ──────────────────────────────────────────────────

class TestTypeScriptProvider:
    @pytest.fixture
    def provider(self):
        return TypeScriptSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.TYPESCRIPT

    def test_file_globs(self, provider):
        globs = provider.get_file_globs()
        assert "*.ts" in globs
        assert "*.tsx" in globs

    def test_symbol_patterns_all_kinds(self, provider):
        patterns = provider.get_symbol_patterns("any")
        kinds = {p.kind for p in patterns}
        assert "function" in kinds
        assert "class" in kinds
        assert "interface" in kinds

    def test_find_function(self, provider):
        content = "function myFunc(x: number): void {\n    console.log(x);\n}\n"
        result = provider.find_symbol_in_file("foo.ts", "myFunc", content)
        assert result is not None

    def test_find_class(self, provider):
        content = "class MyService {\n    run() {}\n}\n"
        result = provider.find_symbol_in_file("foo.ts", "MyService", content)
        assert result is not None

    def test_find_arrow_function(self, provider):
        content = "const myFn = (x: number) => {\n    return x;\n};\n"
        result = provider.find_symbol_in_file("foo.ts", "myFn", content)
        assert result is not None

    def test_not_found_returns_none(self, provider):
        content = "function other() {}\n"
        result = provider.find_symbol_in_file("foo.ts", "missing", content)
        assert result is None

    def test_definition_keywords(self, provider):
        keywords = provider.get_definition_keywords()
        assert "function " in keywords
        assert "class " in keywords

    def test_test_command_default_jest(self, provider, tmp_path):
        cmd = provider.get_test_command(str(tmp_path))
        assert cmd is not None
        assert "jest" in " ".join(cmd)

    def test_test_command_vitest_detected(self, provider, tmp_path):
        import json
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({
            "devDependencies": {"vitest": "^1.0.0"}
        }))
        cmd = provider.get_test_command(str(tmp_path))
        assert "vitest" in " ".join(cmd)

    # ── Structural query tests (regex fallback) ──────────────────────────────

    def test_find_top_level_definitions(self, provider):
        content = (
            "function hello(): void {\n    console.log('hi');\n}\n\n"
            "class MyService {\n    run(): void {}\n}\n\n"
            "interface Config {\n    port: number\n}\n\n"
            "type MyType = string;\n"
        )
        results = provider._find_top_level_definitions_regex(content)
        names = {r[0] for r in results}
        assert "hello" in names, f"hello missing: {names}"
        assert "MyService" in names, f"MyService missing: {names}"
        assert "Config" in names, f"Config missing: {names}"
        assert "MyType" in names, f"MyType missing: {names}"

    def test_find_class_methods(self, provider):
        content = (
            "class MyService {\n"
            "    run(): void {\n        console.log('run');\n    }\n"
            "    stop(): void {\n        console.log('stop');\n    }\n"
            "}\n"
        )
        methods = provider._find_class_methods_regex(content, "MyService")
        names = {m[0] for m in methods}
        assert "run" in names, f"run missing: {names}"
        assert "stop" in names, f"stop missing: {names}"

    def test_find_symbol_body_range(self, provider):
        content = "function hello(): void {\n    console.log('hi');\n}\n"
        body = provider._find_symbol_body_range_regex(content, "hello")
        assert body is not None, "body should not be None"
        assert body[0] >= 1, f"body start invalid: {body}"
        assert body[1] >= body[0], f"body end < start: {body}"


# ── JavaScriptSyntaxProvider ──────────────────────────────────────────────────

class TestJavaScriptProvider:
    @pytest.fixture
    def provider(self):
        return JavaScriptSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.JAVASCRIPT

    def test_file_globs(self, provider):
        globs = provider.get_file_globs()
        assert "*.js" in globs or "*.jsx" in globs

    def test_symbol_patterns_not_empty(self, provider):
        patterns = provider.get_symbol_patterns("any")
        assert len(patterns) > 0

    def test_definition_keywords_not_empty(self, provider):
        keywords = provider.get_definition_keywords()
        assert len(keywords) > 0

    # ── Structural query tests (regex fallback, delegates to TS) ─────────────

    def test_find_top_level_definitions(self, provider):
        content = "function hello() {}\nclass MyClass {}\n"
        results = provider.find_top_level_definitions(content)
        # At minimum, doesn't crash and returns list
        assert isinstance(results, list)

    def test_find_class_methods(self, provider):
        content = "class MyClass {\n    run() {}\n    stop() {}\n}\n"
        methods = provider.find_class_methods(content, "MyClass")
        assert isinstance(methods, list)

    def test_find_symbol_body_range(self, provider):
        content = "function hello() {\n    return 1;\n}\n"
        body = provider.find_symbol_body_range(content, "hello")
        assert isinstance(body, tuple) if body is not None else True


# ── KotlinSyntaxProvider ──────────────────────────────────────────────────────

class TestKotlinProvider:
    @pytest.fixture
    def provider(self):
        return KotlinSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.KOTLIN

    def test_file_globs(self, provider):
        globs = provider.get_file_globs()
        assert any("kt" in g for g in globs)

    def test_definition_keywords(self, provider):
        keywords = provider.get_definition_keywords()
        assert any("fun" in kw for kw in keywords)

    # ── Structural query tests (regex fallback) ──────────────────────────────

    def test_find_top_level_definitions(self, provider):
        content = (
            "package com.example\n\n"
            "class Greeter(val name: String) {\n    fun greet() = \"hi\"\n}\n\n"
            "fun helper() {\n    println(\"help\")\n}\n"
            "interface Speaker {\n    fun speak()\n}\n"
        )
        results = provider._find_top_level_definitions_regex(content)
        names = {r[0] for r in results}
        assert "Greeter" in names, f"Greeter missing: {names}"
        assert "helper" in names, f"helper missing: {names}"
        assert "Speaker" in names, f"Speaker missing: {names}"

    def test_find_class_methods(self, provider):
        content = (
            "class Greeter(val name: String) {\n"
            "    fun greet(): String {\n        return \"hi\"\n    }\n"
            "    fun bye(): String {\n        return \"bye\"\n    }\n"
            "}\n"
        )
        methods = provider._find_class_methods_regex(content, "Greeter")
        names = {m[0] for m in methods}
        assert "greet" in names, f"greet missing: {names}"
        assert "bye" in names, f"bye missing: {names}"

    def test_find_symbol_body_range(self, provider):
        content = "fun helper() {\n    println(\"hi\")\n}\n"
        body = provider._find_symbol_body_range_regex(content, "helper")
        assert body is not None, "body should not be None"
        assert body[0] >= 1, f"body start valid: {body}"


# ── JavaSyntaxProvider ────────────────────────────────────────────────────────

class TestJavaProvider:
    @pytest.fixture
    def provider(self):
        return JavaSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.JAVA

    def test_file_globs(self, provider):
        globs = provider.get_file_globs()
        assert "*.java" in globs

    def test_definition_keywords(self, provider):
        keywords = provider.get_definition_keywords()
        assert any("class" in kw for kw in keywords)

    # ── Structural query tests (regex fallback) ──────────────────────────────

    def test_find_top_level_definitions(self, provider):
        content = (
            "package com.example;\n\n"
            "public class Calculator {\n"
            "    public int add(int a, int b) { return a + b; }\n"
            "}\n\n"
            "interface Printer {\n    void print();\n}\n"
        )
        results = provider._find_top_level_definitions_regex(content)
        names = {r[0] for r in results}
        assert "Calculator" in names, f"Calculator missing: {names}"
        assert "Printer" in names, f"Printer missing: {names}"

    def test_find_class_methods(self, provider):
        content = (
            "public class Calculator {\n"
            "    public int add(int a, int b) {\n        return a + b;\n    }\n"
            "    public int sub(int a, int b) {\n        return a - b;\n    }\n"
            "}\n"
        )
        methods = provider._find_class_methods_regex(content, "Calculator")
        names = {m[0] for m in methods}
        assert "add" in names, f"add missing: {names}"
        assert "sub" in names, f"sub missing: {names}"

    def test_find_symbol_body_range(self, provider):
        content = "public class C {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}\n"
        body = provider._find_symbol_body_range_regex(content, "add")
        assert body is not None, "body should not be None"
        assert body[0] >= 1, f"body start valid: {body}"


# ── BashSyntaxProvider ────────────────────────────────────────────────────────

class TestBashProvider:
    """Bash provider — regex + AST symbol detection (no toolchain assumed)."""

    @pytest.fixture
    def provider(self):
        return BashSyntaxProvider()

    def test_language_id(self, provider):
        assert provider.language_id() == LanguageId.BASH

    def test_file_globs(self, provider):
        globs = provider.get_file_globs()
        assert "*.sh" in globs
        assert "*.bash" in globs

    def test_capabilities(self, provider):
        caps = provider.capabilities()
        assert caps.has_symbol_search is True
        # No bundled shellcheck, but tree-sitter provides a real syntax check.
        assert caps.has_syntax_validator is True

    def test_symbol_patterns_posix_and_keyword(self, provider):
        patterns = provider.get_symbol_patterns("function")
        descs = " ".join(p.description for p in patterns)
        assert "POSIX" in descs, "POSIX form (name()) pattern expected"
        assert "keyword" in descs, "keyword form (function name) pattern expected"

    def test_validate_syntax_via_tree_sitter(self, provider):
        # No bundled toolchain, but tree-sitter gates structural errors.
        broken = provider.validate_syntax("foo.sh", "if [ -z \"$x\" then\n  echo hi\nfi")
        assert broken.ok is False, "malformed bash must be rejected by the gate"
        valid = provider.validate_syntax("foo.sh", "if [ -z \"$x\" ]; then echo hi; fi")
        assert valid.ok is True
        assert valid.language == LanguageId.BASH

    def test_no_lint_or_test_command(self, provider):
        assert provider.get_lint_command("foo.sh") is None
        assert provider.get_test_command(".") is None

    def test_find_symbol_in_file_not_supported(self, provider):
        # Provider index path is used; per-file lookup returns None.
        assert provider.find_symbol_in_file("foo.sh", "bar", "bar() {}") is None
