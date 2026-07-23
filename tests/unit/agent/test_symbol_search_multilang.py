"""Tests for multi-language symbol search (TS/JS via TSSemanticTracer)."""
import textwrap

import pytest

from external_llm.agent.symbol_search import SymbolSearcher, _is_definition_line
from pathlib import Path
import subprocess

# ── Fixture: temp repo with TS files ────────────────────────────────────────

TS_SERVICE_CODE = textwrap.dedent("""\
    import express from 'express';

    interface Todo {
      id: string;
      title: string;
      completed: boolean;
    }

    const todos: Todo[] = [];

    export function getTodos(): Todo[] {
      return todos;
    }

    export function createTodo(title: string): Todo {
      const todo: Todo = { id: String(Date.now()), title, completed: false };
      todos.push(todo);
      return todo;
    }

    export async function deleteTodo(id: string): Promise<boolean> {
      const index = todos.findIndex(t => t.id === id);
      if (index >= 0) {
        todos.splice(index, 1);
        return true;
      }
      return false;
    }

    export class TodoService {
      private items: Todo[] = [];

      add(title: string): Todo {
        const t: Todo = { id: String(this.items.length + 1), title, completed: false };
        this.items.push(t);
        return t;
      }

      delete(id: string): boolean {
        const idx = this.items.findIndex(t => t.id === id);
        if (idx < 0) return false;
        this.items.splice(idx, 1);
        return true;
      }
    }
""")


@pytest.fixture
def ts_repo(tmp_path):
    """Create a temp repo with TS files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "todoService.ts").write_text(TS_SERVICE_CODE, encoding="utf-8")
    (src_dir / "index.ts").write_text(
        "import { getTodos } from './todoService';\n"
        "console.log(getTodos());\n",
        encoding="utf-8",
    )
    return tmp_path


# ── find_symbol tests ───────────────────────────────────────────────────────


class TestFindSymbolTS:
    def test_find_function(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("getTodos", kind="function")
        assert len(results) >= 1
        assert results[0].kind in ("function", "async_function")
        assert results[0].name == "getTodos"
        assert "todoService.ts" in results[0].file

    def test_find_async_function(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("deleteTodo", kind="function")
        assert len(results) >= 1
        assert results[0].kind == "async_function"
        assert results[0].name == "deleteTodo"

    def test_find_class(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("TodoService", kind="class")
        assert len(results) >= 1
        assert results[0].kind == "class"
        assert results[0].name == "TodoService"
        assert results[0].methods is not None
        assert "add" in results[0].methods
        assert "delete" in results[0].methods

    def test_find_interface(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("Todo", kind="any")
        assert len(results) >= 1
        # Should find the interface
        found = [r for r in results if r.kind == "interface"]
        assert len(found) >= 1
        assert found[0].name == "Todo"

    def test_find_variable(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("todos", kind="variable")
        assert len(results) >= 1
        assert results[0].kind == "variable"
        assert results[0].name == "todos"

    def test_find_with_search_path(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol(
            "getTodos", kind="function",
            search_path=str(ts_repo / "src" / "todoService.ts"),
        )
        assert len(results) >= 1
        assert results[0].name == "getTodos"

    def test_function_has_signature(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("createTodo", kind="function")
        assert len(results) >= 1
        sig = results[0].signature
        assert sig is not None
        assert "title" in sig
        assert "createTodo" in sig

    def test_not_found(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        results = searcher.find_symbol("nonExistentSymbol", kind="any")
        assert len(results) == 0


# ── get_file_outline tests ──────────────────────────────────────────────────


class TestFileOutlineTS:
    def test_outline_returns_symbols(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        outline = searcher.get_file_outline("src/todoService.ts")
        assert len(outline) >= 4  # getTodos, createTodo, deleteTodo, TodoService, Todo, todos
        names = [s.name for s in outline]
        assert "getTodos" in names
        assert "createTodo" in names
        assert "TodoService" in names

    def test_outline_sorted_by_line(self, ts_repo):
        searcher = SymbolSearcher(str(ts_repo))
        outline = searcher.get_file_outline("src/todoService.ts")
        lines = [s.line for s in outline]
        assert lines == sorted(lines)


# ── _build_ts_function_signature tests ──────────────────────────────────────


class TestBuildTsSignature:
    def test_simple_function(self):
        from external_llm.agent.symbol_search import _build_ts_function_signature
        from external_llm.editor.semantic.ts_ir_models import IRFunction, TSParam, TSTypeRef

        fn = IRFunction(
            name="greet",
            params=[TSParam(name="name", type_ref=TSTypeRef(name="string"))],
            return_type=TSTypeRef(name="string"),
        )
        sig = _build_ts_function_signature(fn)
        assert "greet" in sig
        assert "name: string" in sig
        assert ": string" in sig

    def test_async_function(self):
        from external_llm.agent.symbol_search import _build_ts_function_signature
        from external_llm.editor.semantic.ts_ir_models import IRFunction

        fn = IRFunction(name="fetch", is_async=True)
        sig = _build_ts_function_signature(fn)
        assert "async function" in sig


# ── _live_symbol_size multi-language tests ──────────────────────────────────


class TestLiveSymbolSizeMultilang:
    def test_ts_function_returns_nonzero(self, ts_repo):
        """_live_symbol_size should return >0 for TS functions via SyntaxProvider."""
        from external_llm.languages import LanguageRegistry
        registry = LanguageRegistry.instance()
        provider = registry.get("test.ts")
        assert provider is not None

        ts_file = ts_repo / "src" / "todoService.ts"
        content = ts_file.read_text()
        result = provider.find_symbol_in_file(str(ts_file), "deleteTodo", content)
        assert result is not None
        start, end = result
        assert end > start
        assert end - start >= 3  # deleteTodo is at least 3 lines


# ── get_file_outline: ripgrep fallback for non-Python/non-TS-JS ──────────────
# Regression: _outline_ripgrep invoked rg on a SINGLE file without --with-filename,
# so rg emitted "lineno:content" (no path prefix). The 3-part split (path:lineno:content)
# then collapsed — every match was dropped, yielding 0 symbols for Kotlin/Go/Java/etc.

KOTLIN_CODE = textwrap.dedent("""\
    package com.example

    interface Playable {
        fun play()
    }

    object Constants {
        val MAX: Int = 100
    }

    enum class State {
        IDLE, RUNNING
    }

    class AudioRecorderEngine(
        private val dir: String,
    ) {
        fun startRecording(): Boolean = true

        private fun allocateNames(): String = "x"

        override fun toString(): String = "Engine"
    }
""")

GO_CODE = textwrap.dedent("""\
    package main

    type Server struct {
        Port int
    }

    func (s *Server) Start() error {
        return nil
    }

    func NewServer(port int) *Server {
        return &Server{Port: port}
    }
""")


@pytest.fixture
def ripgrep_repo(tmp_path):
    """Temp repo with Kotlin + Go files (both served by _outline_ripgrep)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "Engine.kt").write_text(KOTLIN_CODE, encoding="utf-8")
    (src / "server.go").write_text(GO_CODE, encoding="utf-8")
    return tmp_path


class TestFileOutlineRipgrepFallback:
    """get_file_outline must surface symbols for languages routed to _outline_ripgrep
    (Kotlin, Go, Java, Rust, ...). Pre-fix these all returned 0 symbols."""

    def test_kotlin_outline_returns_symbols(self, ripgrep_repo):
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        names = [s.name for s in outline]
        # At minimum the class and a couple of fun declarations must be detected.
        assert "AudioRecorderEngine" in names
        assert "startRecording" in names
        assert len(outline) >= 3

    def test_kotlin_outline_includes_interface_and_object(self, ripgrep_repo):
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        names = [s.name for s in outline]
        assert "Playable" in names
        assert "Constants" in names

    def test_kotlin_outline_not_empty_regression(self, ripgrep_repo):
        """The exact regression: Kotlin outline must NOT be empty."""
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        assert len(outline) > 0, "Kotlin outline returned 0 symbols (parse regression)"

    def test_kotlin_outline_sorted_by_line(self, ripgrep_repo):
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        lines = [s.line for s in outline]
        assert lines == sorted(lines)

    def test_kotlin_private_function_detected(self, ripgrep_repo):
        """Private/override modifiers must not prevent detection."""
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        names = [s.name for s in outline]
        assert "allocateNames" in names

    def test_go_outline_returns_symbols(self, ripgrep_repo):
        """Same _outline_ripgrep code path — Go must also be recovered."""
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/server.go")
        names = [s.name for s in outline]
        assert "Server" in names
        assert "NewServer" in names
        assert len(outline) >= 2

    def test_class_symbol_has_kind(self, ripgrep_repo):
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        kinds_by_name = {s.name: s.kind for s in outline}
        assert kinds_by_name.get("AudioRecorderEngine") == "class"


def _ts_grammar_available(lang: str) -> bool:
    """True when the tree-sitter binding for ``lang`` is installed."""
    try:
        from external_llm.languages.tree_sitter_utils import get_available_languages
        return lang in get_available_languages()
    except Exception:
        return False


# ── get_file_outline: tree-sitter AST path for non-Python/non-TS-JS ─────────
# When a language's tree-sitter grammar is installed, the outline is served by
# the AST path (find_all_symbols) — the same extractor the cross-file index
# uses. Unlike the rg path, it populates SymbolDef.end_line from the parse, so
# callers get an exact symbol extent. Installing a grammar (e.g.
# tree_sitter_kotlin) enables this with no code change.

GO_OUTLINE_CODE = textwrap.dedent("""\
    package main

    type Server struct {
        Port int
    }

    func (s *Server) Start() error {
        return nil
    }

    func NewServer(port int) *Server {
        return &Server{Port: port}
    }
""")


@pytest.fixture
def go_outline_repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "server.go").write_text(GO_OUTLINE_CODE, encoding="utf-8")
    return tmp_path


class TestFileOutlineTreeSitter:
    """get_file_outline routes Go (and any grammar-installed non-Python lang)
    through tree-sitter, yielding accurate end_line values from the AST.
    """

    def test_go_outline_returns_end_line(self, go_outline_repo):
        if not _ts_grammar_available("go"):
            pytest.skip("tree-sitter-go not installed")
        searcher = SymbolSearcher(str(go_outline_repo))
        outline = searcher.get_file_outline("src/server.go")
        # The rg path never sets end_line; a populated end_line proves the AST
        # path was taken.
        by_name = {s.name: s for s in outline}
        assert "Server" in by_name and "Start" in by_name
        assert by_name["Server"].end_line == 5      # type Server struct { … } → lines 3-5
        assert by_name["Start"].end_line == 9       # receiver method → lines 7-9
        assert by_name["NewServer"].end_line == 13

    def test_go_outline_matches_find_all_symbols(self, go_outline_repo):
        """Outline and the cross-file index must agree (single source of truth)."""
        if not _ts_grammar_available("go"):
            pytest.skip("tree-sitter-go not installed")
        from external_llm.languages.tree_sitter_utils import find_all_symbols
        searcher = SymbolSearcher(str(go_outline_repo))
        outline = searcher.get_file_outline("src/server.go")
        ast_names = {n for n, *_ in find_all_symbols(GO_OUTLINE_CODE, "go")}
        outline_names = {s.name for s in outline}
        assert outline_names == ast_names

    def test_kotlin_outline_unchanged_when_grammar_missing(self, ripgrep_repo):
        """Kotlin (no grammar by default) must still return symbols via rg."""
        if _ts_grammar_available("kotlin"):
            pytest.skip("tree-sitter-kotlin IS installed — rg fallback N/A")
        searcher = SymbolSearcher(str(ripgrep_repo))
        outline = searcher.get_file_outline("src/Engine.kt")
        names = [s.name for s in outline]
        assert "AudioRecorderEngine" in names
        assert "allocateNames" in names
# ─────────────────────────────────────────────────────────────────────────────
# Regression: _is_definition_line language-agnostic detection
# ─────────────────────────────────────────────────────────────────────────────
class TestIsDefinitionLine:
    """Tests for language-aware definition-line detection across multiple languages.
    Covers every language that has a registered SyntaxProvider — Python, JS/TS, Go,
    Rust, Kotlin, Java — plus the generic fallback for unrecognised extensions.
    """
    # ── Python (PythonSyntaxProvider) ──────────────────────────────────────
    def test_python_def(self):
        assert _is_definition_line("foo.py", "def my_func():", "my_func") is True
    def test_python_async_def(self):
        assert _is_definition_line("foo.py", "async def my_func():", "my_func") is True
    def test_python_class(self):
        assert _is_definition_line("foo.py", "class MyClass:", "MyClass") is True
    def test_python_non_definition_call(self):
        assert _is_definition_line("foo.py", "    my_func()", "my_func") is False
    def test_python_non_definition_assign(self):
        assert _is_definition_line("foo.py", "x = my_func", "my_func") is False
    # ── JS/TS (TypeScriptSyntaxProvider / JavaScriptSyntaxProvider) ────────
    def test_ts_function(self, ts_repo):
        """export function name() — detected as definition."""
        file = str(ts_repo / "src" / "todoService.ts")
        assert _is_definition_line(file, "export function getTodos(): Todo[] {", "getTodos") is True
    def test_ts_async_function(self, ts_repo):
        """export async function name() — detected as definition."""
        file = str(ts_repo / "src" / "todoService.ts")
        line = "export async function deleteTodo(id: string): Promise<boolean> {"
        assert _is_definition_line(file, line, "deleteTodo") is True
    def test_ts_class(self, ts_repo):
        """export class name { — detected as definition."""
        file = str(ts_repo / "src" / "todoService.ts")
        assert _is_definition_line(file, "export class TodoService {", "TodoService") is True
    def test_ts_const_arrow_function(self, ts_repo):
        """const name = (...) => ... — detected as definition."""
        file = str(ts_repo / "src" / "todoService.ts")
        assert _is_definition_line(file, "const getCount = () => todos.length;", "getCount") is True
    def test_ts_reference_not_definition(self, ts_repo):
        """import { name } from ... — NOT a definition."""
        file = str(ts_repo / "src" / "index.ts")
        assert _is_definition_line(file, "import { getTodos } from './todoService';", "getTodos") is False
    # ── Go (GoSyntaxProvider) ──────────────────────────────────────────────
    def test_go_func_def(self):
        assert _is_definition_line("main.go", "func main() {", "main") is True
    def test_go_func_receiver(self):
        """Method on a type: func (r *T) Name(...)."""
        line = 'func (s *Server) ServeHTTP(w ResponseWriter, r *Request) {'
        assert _is_definition_line("handler.go", line, "ServeHTTP") is True
    def test_go_struct_def(self):
        assert _is_definition_line("server.go", "type Server struct {", "Server") is True
    def test_go_non_definition_call(self):
        assert _is_definition_line("main.go", '    fmt.Println("hello")', "fmt") is False
    # ── Rust (RustSyntaxProvider) ──────────────────────────────────────────
    def test_rust_fn_def(self):
        assert _is_definition_line("lib.rs", "fn process() -> Result<()> {", "process") is True
    def test_rust_struct_def(self):
        assert _is_definition_line("lib.rs", "struct Config {", "Config") is True
    def test_rust_non_definition_call(self):
        assert _is_definition_line("lib.rs", "    process()?;", "process") is False
    # ── Kotlin (KotlinSyntaxProvider) ──────────────────────────────────────
    def test_kotlin_fun_def(self):
        assert _is_definition_line("Main.kt", "fun main() {", "main") is True
    def test_kotlin_class_def(self):
        assert _is_definition_line("Main.kt", "class AudioRecorderEngine(", "AudioRecorderEngine") is True
    def test_kotlin_non_definition_call(self):
        assert _is_definition_line("Main.kt", "    main()", "main") is False
    # ── Java (JavaSyntaxProvider) ──────────────────────────────────────────
    def test_java_method_def(self):
        assert _is_definition_line("Main.java", "    public void handle() {", "handle") is True
    def test_java_non_definition_call(self):
        assert _is_definition_line("Main.java", "        handle();", "handle") is False
    # ── Generic fallback (unrecognised extension) ──────────────────────────
    def test_generic_fallback_def(self):
        """Unrecognised .xyz — fallback to Python/JS heuristic."""
        assert _is_definition_line("module.xyz", "def process():", "process") is True
    def test_generic_fallback_non_def(self):
        assert _is_definition_line("module.xyz", "    process()  # call", "process") is False
    def test_generic_fallback_class(self):
        assert _is_definition_line("module.xyz", "class Manager:", "Manager") is True
    def test_generic_fallback_const(self):
        assert _is_definition_line("module.xyz", "const PI = 3.14", "PI") is True
# ─────────────────────────────────────────────────────────────────────────────
# Regression: _resolve_search_root sibling-directory rejection
# ─────────────────────────────────────────────────────────────────────────────
class TestResolveSearchRoot:
    """Tests that _resolve_search_root rejects paths outside repo_root.
    Pre-fix the comparison used ``str(p).startswith(str(self.repo_root))`` which
    allowed sibling directories (``asicode-evil``) to pass through — classic prefix
    trap. Fixed with ``p.is_relative_to(self.repo_root)``.
    """
    def test_inside_repo_returns_path(self, tmp_path):
        (tmp_path / "src").mkdir()
        searcher = SymbolSearcher(str(tmp_path))
        result = searcher._resolve_search_root("src")
        assert result is not None
        assert result == (tmp_path / "src").resolve()
    def test_none_returns_repo_root(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        result = searcher._resolve_search_root(None)
        assert result == searcher.repo_root
    def test_rejects_sibling_directory(self, tmp_path):
        """Sibling directory (../evil) must be rejected — the original prefix-trap bug."""
        searcher = SymbolSearcher(str(tmp_path))
        result = searcher._resolve_search_root("../evil")
        assert result is None
    def test_rejects_absolute_outside_path(self, tmp_path):
        """Absolute path outside repo_root must be rejected."""
        searcher = SymbolSearcher(str(tmp_path))
        result = searcher._resolve_search_root(str(Path("/tmp/evil")))
        assert result is None
    def test_rejects_deep_sibling_symlink_name(self, tmp_path):
        """e.g. /Users/.../asicode-evil — the original repro case.

        The sibling MUST share the repo_root's own name as a string prefix
        (``<repo>-evil``): a raw ``str.startswith`` check passes it (the
        prefix trap) while ``is_relative_to`` rejects it. A fixed name like
        ``asicode-evil`` is rejected by BOTH old and new code (no shared
        prefix with the tmp repo), so it cannot detect a regression.
        """
        searcher = SymbolSearcher(str(tmp_path))
        _repo = Path(str(searcher.repo_root)).resolve()
        result = searcher._resolve_search_root(str(_repo.parent / (_repo.name + "-evil")))
        assert result is None
# ─────────────────────────────────────────────────────────────────────────────
# Regression: find_references ValueError resilience
# ─────────────────────────────────────────────────────────────────────────────
class TestFindReferencesValueErrorResilience:
    """Tests that one corrupted rg line doesn't discard all valid results.
    Pre-fix the per-line ``except (AttributeError, TypeError)`` didn't catch
    ``ValueError`` from ``Path.relative_to()`` (symlink-outside-repo) or from
    ``int(parts[1])`` (Windows drive-letter corruption). Any single bad line
    escaped the per-line handler, hit the outer ``except Exception``, and
    returned an empty list — eating all valid results.
    """
    def test_bad_path_does_not_eat_valid_results(self, tmp_path, monkeypatch):
        """A line with a path outside the repo (ValueError from relative_to)
        should be skipped; remaining valid lines still produce results."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text(textwrap.dedent('''\
            import logging
            logger = logging.getLogger(__name__)
            logger.info("hello")
        '''), encoding="utf-8")
        # Inject fake rg output: first line has a path outside the repo
        # (triggers ValueError on relative_to), remaining lines are valid.
        fake_stdout = (
            "/nonexistent/outside.py:3:  logger = None\n"
            f"{tmp_path}/src/main.py:2:  logger = logging.getLogger(__name__)\n"
            f"{tmp_path}/src/main.py:3:  logger.info(\"hello\")\n"
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout=fake_stdout, stderr="",
            ),
        )
        searcher = SymbolSearcher(str(tmp_path))
        refs = searcher.find_references("logger")
        assert len(refs) >= 1, "One bad line must not discard valid results"
        assert any("main.py" in r.file for r in refs)
    def test_bad_lineno_does_not_eat_valid_results(self, tmp_path, monkeypatch):
        """A line with a non-integer "line number" (ValueError from int())
        should be skipped; remaining valid lines survive."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "utils.py").write_text("logger = 'test'\n", encoding="utf-8")
        fake_stdout = (
            f"{tmp_path}/src/utils.py:abc:  logger = 'test'\n"  # int('abc') → ValueError
            f"{tmp_path}/src/utils.py:1:  logger = 'test'\n"
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout=fake_stdout, stderr="",
            ),
        )
        searcher = SymbolSearcher(str(tmp_path))
        refs = searcher.find_references("logger")
        assert len(refs) >= 1, "One bad line must not discard valid results"


class TestLuaScalaHalfWiredRegression:
    """LUA/SCALA were "half-wired": LanguageId, _EXT_MAP, grammar queries and
    comment_syntax all existed, but no provider was registered.  _nonpy_index_for
    iterates *registered* providers, so .lua/.scala files were never reached and
    find_symbol / get_file_outline returned empty with no signal.  Registering
    LuaSyntaxProvider / ScalaSyntaxProvider closes that silent-empty gap.
    """

    LUA_CODE = textwrap.dedent("""\
        local M = {}
        function M.greet(name)
            print(name)
        end
        local function configure()
            return true
        end
        return M
    """)

    SCALA_CODE = textwrap.dedent("""\
        object App {
          def main(args: Array[String]): Unit = ()
          case class Config(path: String)
        }
    """)

    def test_find_symbol_lua(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "m.lua").write_text(self.LUA_CODE, encoding="utf-8")
        res = searcher.find_symbol("configure")
        assert res and res[0].file == "m.lua", (
            ".lua symbol not found — LUA provider not registered (half-wired regression)"
        )

    def test_find_symbol_scala(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "App.scala").write_text(self.SCALA_CODE, encoding="utf-8")
        res = searcher.find_symbol("main")
        assert res and res[0].file == "App.scala", (
            ".scala symbol not found — SCALA provider not registered (half-wired regression)"
        )

    def test_outline_lua_not_empty(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "m.lua").write_text(self.LUA_CODE, encoding="utf-8")
        outline = searcher.get_file_outline("m.lua")
        names = [s.name for s in outline]
        assert "configure" in names
        assert len(outline) >= 2

    def test_outline_scala_not_empty(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "App.scala").write_text(self.SCALA_CODE, encoding="utf-8")
        outline = searcher.get_file_outline("App.scala")
        names = [s.name for s in outline]
        assert "main" in names
        assert "App" in names


class TestExtensionWalkerRegression:
    """Regression: .mts/.cts/.mjs/.cjs/.pyi were first-class in the 5-way SSOT
    (_EXT_MAP + family groups + grammar key + provider globs + provider) but
    find_symbol silently returned nothing for symbols defined in them — the
    repo walkers (_walk_ts_js_files / _walk_py_files) and single-file dispatches
    used hardcoded extension literals that drifted from the SSOT. These pin the
    fix end-to-end via find_symbol + get_file_outline."""

    @pytest.mark.parametrize("ext,code,name", [
        (".mts", "export function mtsFn(x: number): string { return String(x); }\n", "mtsFn"),
        (".cts", "export function ctsFn(): number { return 1; }\n", "ctsFn"),
        (".mjs", "export function mjsFn() { return 2; }\n", "mjsFn"),
        (".cjs", "function cjsFn() { return 3; }\nmodule.exports = { cjsFn };\n", "cjsFn"),
    ])
    def test_find_symbol_modern_js_ts_extensions(self, tmp_path, ext, code, name):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / f"mod{ext}").write_text(code, encoding="utf-8")
        hits = searcher.find_symbol(name, search_path=str(tmp_path))
        assert any(h.name == name for h in hits), f"{name} in {ext} must be findable"

    def test_find_symbol_pyi_type_stub(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "stub.pyi").write_text(
            "class StubClass:\n    def method(self) -> int: ...\n", encoding="utf-8"
        )
        # directory search
        hits = searcher.find_symbol("StubClass", search_path=str(tmp_path))
        assert any(h.name == "StubClass" for h in hits)
        # single-file dispatch (search_path points at the .pyi file itself)
        single = searcher.find_symbol("StubClass", search_path=str(tmp_path / "stub.pyi"))
        assert any(h.name == "StubClass" for h in single)

    def test_outline_pyi_type_stub(self, tmp_path):
        searcher = SymbolSearcher(str(tmp_path))
        (tmp_path / "stub.pyi").write_text(
            "class StubClass:\n    def method(self) -> int: ...\n", encoding="utf-8"
        )
        outline = searcher.get_file_outline("stub.pyi")
        names = [s.name for s in outline]
        assert "StubClass" in names


# ── Provider fallback-pattern accuracy (Lua/Scala) ──────────────────────────
# These providers exist to close "silent-empty result" gaps when the tree-sitter
# grammar is absent. Their fallback regex + name_capture must cover idiomatic
# forms or the gap reopens. These regex-level checks are grammar-agnostic (they
# test the substitution that _outline_ripgrep / _nonpy_index_for perform), so
# they guard the fix regardless of whether the lua/scala grammar is installed.

class TestProviderFallbackPatterns:
    def test_lua_dotted_function_uses_name_capture(self):
        """Outline substitution must use sp.name_capture, not a hardcoded \\w+.

        Lua's ``function M.foo()`` requires name_capture=[\\w.:]+. With the old
        hardcoded \\w+ the whole regex aborted at the '.' (\\w+ matched 'M', then
        \\s*\\( saw '.'), dropping the symbol entirely from the outline.
        """
        import re
        from external_llm.languages.lua_provider import LuaSyntaxProvider
        sp = LuaSyntaxProvider().get_symbol_patterns(kind="any")[0]
        pat = sp.regex.replace("{name}", f"({sp.name_capture})")
        m = re.search(pat, "function M.foo()")
        assert m is not None, "dotted Lua function must match via name_capture"
        assert m.group(1) == "M.foo"

    def test_lua_colon_method_captured(self):
        """Lua OOP colon form ``function Obj:method()`` must index under its full
        qualified name — name_capture includes ':'."""
        import re
        from external_llm.languages.lua_provider import LuaSyntaxProvider
        sp = LuaSyntaxProvider().get_symbol_patterns(kind="any")[0]
        pat = sp.regex.replace("{name}", f"({sp.name_capture})")
        m = re.search(pat, "function Account:withdraw(v)")
        assert m is not None
        assert m.group(1) == "Account:withdraw"

    def test_scala_parameterless_def_matched(self):
        """Scala parameterless defs are idiomatic; the fallback regex must not
        require ``(`` or ``[`` after the name (the previous ``[\\[(]``-requiring
        form silently dropped them)."""
        import re
        from external_llm.languages.scala_provider import ScalaSyntaxProvider
        sp = ScalaSyntaxProvider().get_symbol_patterns(kind="any")[0]
        cases = [
            ("size", "def size = xs.length"),        # parameterless, no type
            ("greet", "def greet(): Unit = ()"),     # empty param list
            ("name", "def name: Int = 1"),           # parameterless with type
            ("apply", "def apply[T](xs: T): T = xs"),  # generic
        ]
        for name, src in cases:
            pat = sp.regex.replace("{name}", re.escape(name))
            assert re.search(pat, src), f"Scala def {name!r} must match: {src!r}"
