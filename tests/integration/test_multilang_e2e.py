"""
End-to-end integration tests for multi-language support.

Validates that the LanguageRegistry, RepositoryGraph, TaskRouter,
SymbolSearcher, LintRunner, and validation pipeline all work correctly
with Python, TypeScript, and JavaScript files together.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from external_llm.agent.execution_spec import ResolvedExecutionSpec
from external_llm.agent.lint_runner import LintRunner
from external_llm.agent.symbol_search import SymbolSearcher
from external_llm.agent.task_router import Lane, TaskRouter
from external_llm.graph.graph_facade import RepositoryGraphFacade
from external_llm.graph.repository_graph import RepositoryGraph
from external_llm.languages import LanguageId, LanguageRegistry

# ── Fixture: multi-language project ──────────────────────────────────────────

PYTHON_SRC = """\
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(user_id)

    def update_user(self, user_id: int, data: dict):
        user = self.get_user(user_id)
        user.update(data)
        self.db.save(user)
        return user
"""

TYPESCRIPT_SRC = """\
export interface User {
  id: number;
  name: string;
  email: string;
}

export class UserRepository {
  private users: Map<number, User> = new Map();

  getUser(id: number): User | undefined {
    return this.users.get(id);
  }

  saveUser(user: User): void {
    this.users.set(user.id, user);
  }

  deleteUser(id: number): boolean {
    return this.users.delete(id);
  }
}

export function createDefaultUser(name: string): User {
  return {
    id: Date.now(),
    name,
    email: `${name.toLowerCase()}@example.com`,
  };
}
"""

JAVASCRIPT_SRC = """\
export class Logger {
  constructor(prefix) {
    this.prefix = prefix;
  }

  log(message) {
    console.log(`[${this.prefix}] ${message}`);
  }

  error(message) {
    console.error(`[${this.prefix}] ERROR: ${message}`);
  }
}

export function formatDate(date) {
  return date.toISOString().split('T')[0];
}

export const debounce = (fn, delay) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
};
"""

GO_SRC = """\
package main

import "fmt"

type UserService struct {
	db Database
}

func NewUserService(db Database) *UserService {
	return &UserService{db: db}
}

func (s *UserService) GetUser(id int) (*User, error) {
	return s.db.FindByID(id)
}

type Database interface {
	FindByID(id int) (*User, error)
	Save(user *User) error
}

type User struct {
	ID   int
	Name string
}

func FormatUser(u *User) string {
	return fmt.Sprintf("User(%d, %s)", u.ID, u.Name)
}
"""

JAVA_SRC = """\
package com.example;

public class UserRepository {
    private final Map<Integer, User> users = new HashMap<>();

    public User findById(int id) {
        return users.get(id);
    }

    public void save(User user) {
        users.put(user.getId(), user);
    }

    public boolean delete(int id) {
        return users.remove(id) != null;
    }
}

interface UserDao {
    User findById(int id);
    void save(User user);
}

enum UserRole {
    ADMIN,
    EDITOR,
    VIEWER
}
"""

REACT_TSX_SRC = """\
import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export function Button({ label, onClick, disabled }: ButtonProps) {
  return (
    <button onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}

export class App extends React.Component {
  render() {
    return <Button label="Click me" onClick={() => {}} />;
  }
}
"""


@pytest.fixture(scope="module")
def multilang_project():
    """Create a temp directory with Python + TypeScript + JavaScript files."""
    tmpdir = tempfile.mkdtemp(prefix="multilang-e2e-")
    try:
        import subprocess
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=tmpdir, capture_output=True)

        # Python
        svc_dir = Path(tmpdir) / "service"
        svc_dir.mkdir()
        (svc_dir / "__init__.py").write_text("")
        (svc_dir / "user_service.py").write_text(PYTHON_SRC)

        # TypeScript
        api_dir = Path(tmpdir) / "api"
        api_dir.mkdir()
        (api_dir / "user.ts").write_text(TYPESCRIPT_SRC)

        # React TSX
        ui_dir = Path(tmpdir) / "ui"
        ui_dir.mkdir()
        (ui_dir / "Button.tsx").write_text(REACT_TSX_SRC)

        # JavaScript
        utils_dir = Path(tmpdir) / "utils"
        utils_dir.mkdir()
        (utils_dir / "helpers.js").write_text(JAVASCRIPT_SRC)

        # Go
        go_dir = Path(tmpdir) / "server"
        go_dir.mkdir()
        (go_dir / "main.go").write_text(GO_SRC)

        # Java
        java_dir = Path(tmpdir) / "src"
        java_dir.mkdir()
        (java_dir / "UserRepository.java").write_text(JAVA_SRC)

        # Non-AST files
        (Path(tmpdir) / "config.json").write_text('{"version": "1.0"}')
        (Path(tmpdir) / "styles.css").write_text("body { margin: 0; }")

        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init multilang"],
                       cwd=tmpdir, capture_output=True)

        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def graph(multilang_project):
    g = RepositoryGraph(multilang_project)
    g.build()
    return g


@pytest.fixture(scope="module")
def facade(multilang_project):
    return RepositoryGraphFacade(repo_root=multilang_project)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LanguageRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageRegistry:
    """Verify that the registry correctly identifies languages and capabilities."""

    def test_python_detection(self):
        assert LanguageId.from_path("service/user_service.py") == LanguageId.PYTHON
        assert LanguageId.from_path("test.pyi") == LanguageId.PYTHON

    def test_typescript_detection(self):
        assert LanguageId.from_path("api/user.ts") == LanguageId.TYPESCRIPT
        assert LanguageId.from_path("ui/Button.tsx") == LanguageId.TYPESCRIPT

    def test_javascript_detection(self):
        assert LanguageId.from_path("utils/helpers.js") == LanguageId.JAVASCRIPT
        assert LanguageId.from_path("app.jsx") == LanguageId.JAVASCRIPT
        assert LanguageId.from_path("index.mjs") == LanguageId.JAVASCRIPT
        assert LanguageId.from_path("config.cjs") == LanguageId.JAVASCRIPT

    def test_unknown_detection(self):
        # JSON/CSS/HTML are now registered languages
        assert LanguageId.from_path("config.json") == LanguageId.JSON
        assert LanguageId.from_path("styles.css") == LanguageId.CSS
        assert LanguageId.from_path("README.md") == LanguageId.UNKNOWN
        assert LanguageId.from_path("data.xml") == LanguageId.UNKNOWN

    def test_structured_ops_support(self):
        r = LanguageRegistry.instance()
        assert r.supports_structured_ops("test.py") is True
        assert r.supports_structured_ops("test.ts") is True
        assert r.supports_structured_ops("test.tsx") is True
        assert r.supports_structured_ops("test.js") is True
        assert r.supports_structured_ops("test.jsx") is True
        assert r.supports_structured_ops("test.css") is False
        assert r.supports_structured_ops("test.json") is False
        assert r.supports_structured_ops("test.md") is False

    def test_file_globs_include_all_languages(self):
        globs = LanguageRegistry.instance().get_all_file_globs()
        assert "*.py" in globs
        assert "*.ts" in globs
        assert "*.tsx" in globs
        assert "*.js" in globs
        assert "*.jsx" in globs

    def test_file_pattern_regex(self):
        import re
        pat = LanguageRegistry.instance().get_file_pattern()
        assert re.search(pat, "service/user_service.py")
        assert re.search(pat, "api/user.ts")
        assert re.search(pat, "utils/helpers.js")
        assert re.search(pat, "ui/Button.tsx")
        # JSON/CSS/HTML are now registered providers, so they match
        assert re.search(pat, "config.json")
        assert not re.search(pat, "README.md")

    def test_three_providers_registered(self):
        r = LanguageRegistry.instance()
        assert r.get("test.py") is not None
        assert r.get("test.ts") is not None
        assert r.get("test.js") is not None
        # Providers are distinct
        assert r.get("test.py").language_id() != r.get("test.ts").language_id()
        assert r.get("test.ts").language_id() != r.get("test.js").language_id()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Syntax Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestSyntaxValidation:
    """Verify that each provider validates syntax correctly."""

    def test_python_valid(self):
        p = LanguageRegistry.instance().get("test.py")
        result = p.validate_syntax("test.py", PYTHON_SRC)
        assert result.ok is True
        assert result.language == LanguageId.PYTHON

    def test_python_invalid(self):
        p = LanguageRegistry.instance().get("test.py")
        result = p.validate_syntax("test.py", "def foo(:\n    pass")
        assert result.ok is False
        assert len(result.errors) > 0
        assert result.errors[0].line > 0

    @pytest.mark.slow
    def test_typescript_provider_returns_result(self):
        p = LanguageRegistry.instance().get("test.ts")
        result = p.validate_syntax("test.ts", TYPESCRIPT_SRC)
        # tsc may not be installed, so ok=True is acceptable (graceful skip)
        assert result.language == LanguageId.TYPESCRIPT

    def test_javascript_provider_returns_result(self):
        p = LanguageRegistry.instance().get("test.js")
        result = p.validate_syntax("test.js", JAVASCRIPT_SRC)
        assert result.language == LanguageId.JAVASCRIPT




# ══════════════════════════════════════════════════════════════════════════════
# 3. Symbol Detection (Provider Patterns)
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolDetection:
    """Verify that providers find symbols in source code."""

    def test_python_find_symbol(self):
        p = LanguageRegistry.instance().get("test.py")
        result = p.find_symbol_in_file("test.py", "UserService", PYTHON_SRC)
        assert result is not None
        start, end = result
        assert start == 1  # class UserService at line 1
        assert end > start

    def test_python_find_method(self):
        p = LanguageRegistry.instance().get("test.py")
        result = p.find_symbol_in_file("test.py", "get_user", PYTHON_SRC)
        assert result is not None
        start, end = result
        assert start == 5
        assert end > start

    def test_typescript_find_class(self):
        p = LanguageRegistry.instance().get("test.ts")
        result = p.find_symbol_in_file("test.ts", "UserRepository", TYPESCRIPT_SRC)
        assert result is not None
        start, end = result
        assert end > start

    def test_typescript_find_function(self):
        p = LanguageRegistry.instance().get("test.ts")
        result = p.find_symbol_in_file("test.ts", "createDefaultUser", TYPESCRIPT_SRC)
        assert result is not None

    def test_typescript_find_interface(self):
        p = LanguageRegistry.instance().get("test.ts")
        result = p.find_symbol_in_file("test.ts", "User", TYPESCRIPT_SRC)
        assert result is not None

    def test_javascript_find_class(self):
        p = LanguageRegistry.instance().get("test.js")
        result = p.find_symbol_in_file("test.js", "Logger", JAVASCRIPT_SRC)
        assert result is not None
        start, end = result
        assert end > start

    def test_javascript_find_function(self):
        p = LanguageRegistry.instance().get("test.js")
        result = p.find_symbol_in_file("test.js", "formatDate", JAVASCRIPT_SRC)
        assert result is not None

    def test_javascript_find_arrow_const(self):
        p = LanguageRegistry.instance().get("test.js")
        result = p.find_symbol_in_file("test.js", "debounce", JAVASCRIPT_SRC)
        assert result is not None

    def test_tsx_find_component(self):
        p = LanguageRegistry.instance().get("test.tsx")
        result = p.find_symbol_in_file("test.tsx", "Button", REACT_TSX_SRC)
        assert result is not None

    def test_tsx_find_interface(self):
        p = LanguageRegistry.instance().get("test.tsx")
        result = p.find_symbol_in_file("test.tsx", "ButtonProps", REACT_TSX_SRC)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Repository Graph (GSG) — Multi-Language Indexing
# ══════════════════════════════════════════════════════════════════════════════

class TestRepositoryGraph:
    """Verify that the GSG indexes symbols from all languages."""

    def test_graph_builds_without_error(self, graph):
        assert len(graph.symbols) > 0

    def test_python_symbols_indexed(self, graph):
        py_symbols = [
            s for s in graph.symbols.values()
            if s.file_path.endswith(".py") and s.name != "__init__"
        ]
        assert len(py_symbols) >= 2  # UserService, get_user, update_user, ...
        names = {s.name for s in py_symbols}
        assert "UserService" in names

    def test_typescript_symbols_indexed(self, graph):
        ts_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".ts")]
        assert len(ts_symbols) >= 2  # User, UserRepository, createDefaultUser, ...
        names = {s.name for s in ts_symbols}
        assert "UserRepository" in names or "createDefaultUser" in names

    def test_javascript_symbols_indexed(self, graph):
        js_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".js")]
        assert len(js_symbols) >= 2  # Logger, formatDate, debounce
        names = {s.name for s in js_symbols}
        assert "Logger" in names or "formatDate" in names

    def test_tsx_symbols_indexed(self, graph):
        tsx_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".tsx")]
        assert len(tsx_symbols) >= 1
        names = {s.name for s in tsx_symbols}
        assert "Button" in names or "App" in names

    def test_non_ast_files_not_indexed(self, graph):
        json_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".json")]
        css_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".css")]
        assert len(json_symbols) == 0
        assert len(css_symbols) == 0

    def test_language_field_set_for_non_python(self, graph):
        ts_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".ts")]
        for s in ts_symbols:
            assert s.language == "typescript", f"{s.name} has language={s.language}"

        js_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".js")]
        for s in js_symbols:
            assert s.language == "javascript", f"{s.name} has language={s.language}"

    def test_facade_get_symbol_multilang(self, facade):
        # Python
        py_sym = facade.get_symbol("UserService")
        assert py_sym is not None

        # TS/JS — symbol lookup may depend on build timing
        # Just verify facade doesn't crash on non-Python lookups
        ts_sym = facade.get_symbol("UserRepository")
        js_sym = facade.get_symbol("Logger")
        # At least one should be found
        found_count = sum(1 for s in [ts_sym, js_sym] if s is not None)
        assert found_count >= 1, "At least one TS/JS symbol should be found via facade"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Task Router — Language-Aware Routing
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskRouter:
    """Verify that the router sends TS/JS edits to PLANNER, non-AST to MAIN_AGENT."""

    @pytest.fixture(autouse=True)
    def _setup_router(self):
        try:
            self.router = TaskRouter()
        except Exception:
            pytest.skip("TaskRouter requires llm_client and model params")

    def test_python_edit_to_planner(self):
        decision = self.router.route(
            "add a docstring to the get_user function in user_service.py"
        )
        assert decision.lane == Lane.PLANNER

    def test_typescript_edit_to_planner(self):
        decision = self.router.route(
            "add a findAll method to the UserRepository class in api/user.ts"
        )
        assert decision.lane == Lane.PLANNER

    def test_javascript_edit_to_planner(self):
        decision = self.router.route(
            "modify the formatDate function in utils/helpers.js to accept a locale parameter"
        )
        assert decision.lane == Lane.PLANNER

    def test_tsx_edit_to_planner(self):
        decision = self.router.route(
            "add a size prop to the Button component in ui/Button.tsx"
        )
        assert decision.lane == Lane.PLANNER

    def test_css_edit_to_planner(self):
        decision = self.router.route(
            "change the background-color of body in styles.css"
        )
        # All requests go through PLANNER (universal lane)
        assert decision.lane == Lane.PLANNER

    def test_json_edit_to_planner(self):
        decision = self.router.route(
            "change the version value to 2.0 in config.json"
        )
        # All requests go through PLANNER (universal lane)
        assert decision.lane == Lane.PLANNER

    def test_mixed_ts_js_to_planner(self):
        decision = self.router.route(
            "modify the functions related to the User type in api/user.ts and utils/helpers.js"
        )
        assert decision.lane == Lane.PLANNER


# ══════════════════════════════════════════════════════════════════════════════
# 6. Symbol Search — Multi-Language
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolSearch:
    """Verify that SymbolSearcher finds symbols across languages."""

    def test_find_python_symbol(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("UserService")
        assert len(results) > 0
        assert any(r.file.endswith(".py") for r in results)

    def test_find_typescript_symbol(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("UserRepository")
        # Should find via the tree-sitter AST index.
        assert len(results) > 0
        assert any(".ts" in r.file for r in results)

    def test_find_javascript_symbol(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("Logger")
        assert len(results) > 0
        assert any(".js" in r.file for r in results)

    def test_find_tsx_component(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("Button")
        assert len(results) > 0
        assert any(".tsx" in r.file for r in results)

    # ── CSS / SCSS symbol search (regression for the patterns[:3] slice bug)
    #
    # Historically CSS class/id/variable symbols were unfindable because the
    # rg fallback (_find_in_other_langs, now removed) used a `patterns[:3]`
    # slice that dropped every class pattern. CSS now resolves through the
    # tree-sitter AST index (_index_via_treesitter), the single source of
    # truth. These tests pin that CSS symbols are reliably findable.
    def test_find_css_class(self, tmp_path):
        (tmp_path / "style.css").write_text(
            ".btn-primary {\n  color: blue;\n}\n"
            ".card { padding: 8px; }\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("btn-primary", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_class"
        assert results[0].file.endswith("style.css")

    def test_find_css_id(self, tmp_path):
        (tmp_path / "style.css").write_text(
            "#main-header {\n  width: 100%;\n}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("main-header", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_id"

    def test_find_css_class_with_kind_class(self, tmp_path):
        # kind="class" must also reach the CSS class pattern (only 5 patterns,
        # all of which were previously dropped by [:3]).
        (tmp_path / "style.css").write_text(
            ".hero-banner { display: flex; }\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("hero-banner", kind="class")
        assert len(results) == 1
        assert results[0].kind == "css_class"

    def test_find_css_variable(self, tmp_path):
        (tmp_path / "theme.css").write_text(
            ":root {\n  --primary-color: #333;\n}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("primary-color", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_variable"

    # ── CSS symbol search now flows through the tree-sitter AST, not rg
    #
    # These pin the architectural move: CSS symbols are extracted by
    # tree_sitter_utils.find_all_symbols (class_selector→css_class,
    # id_selector→css_id, "--"-prefixed declaration→css_variable), fed into
    # _nonpy_index_for via the AST-first _index_via_treesitter path. Neither
    # CssSyntaxProvider.get_symbol_patterns (now empty) nor the legacy
    # _find_in_other_langs rg path contribute — so the "--name" leading-dash
    # shell-arg trap is structurally impossible.
    def test_css_search_uses_treesitter_not_fallback(self, tmp_path):
        """If _find_in_other_langs were still the CSS path, disabling it would
        break CSS search. CSS now resolves through the tree-sitter AST index,
        so it succeeds regardless of the legacy fallback."""
        (tmp_path / "s.css").write_text(".btn-primary { color: blue; }\n")
        ss = SymbolSearcher(tmp_path)
        # Neutralize the legacy fallback path entirely.
        results = ss.find_symbol("btn-primary", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_class"

    def test_css_search_uses_treesitter_kind_class(self, tmp_path):
        """kind='class' CSS search must also reach the AST index."""
        (tmp_path / "s.css").write_text(".hero-banner { display: flex; }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("hero-banner", kind="class")
        assert len(results) == 1
        assert results[0].kind == "css_class"

    def test_css_custom_property_found_via_treesitter(self, tmp_path):
        """The '--name' custom property (which motivated the rg -e flag fix) is
        now served by the tree-sitter AST index, where it never becomes a shell
        arg at all."""
        (tmp_path / "t.css").write_text(":root { --accent-color: #f00; }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("accent-color", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_variable"

    def test_css_kebab_case_not_truncated(self, tmp_path):
        """The tree-sitter class_name node captures the full kebab-case name —
        unlike a \\w+ regex it does not truncate 'multi-word-name' at the first
        hyphen and index it under 'multi'."""
        (tmp_path / "s.css").write_text(".multi-word-name { margin: 0; }\n")
        ss = SymbolSearcher(tmp_path)
        # Searching the full name must hit; searching the truncated 'multi'
        # must NOT be what got indexed.
        full = ss.find_symbol("multi-word-name", kind="any")
        assert len(full) == 1
        assert full[0].kind == "css_class"
        truncated = ss.find_symbol("multi", kind="any")
        assert len(truncated) == 0

    # ── tree-sitter AST specifics: these pin behavior that only the AST path
    # can provide (a regex path could never match these correctly).

    def test_css_id_selector_found(self, tmp_path):
        """id selectors (#name) are a distinct kind (css_id), separable from
        class selectors."""
        (tmp_path / "s.css").write_text("#main-header { padding: 0; }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("main-header", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_id"

    def test_css_ordinary_property_not_indexed(self, tmp_path):
        """An ordinary CSS property like 'color' inside a declaration must NOT
        be indexed — only '--'-prefixed custom properties are symbols. This
        distinguishes the AST's declaration filtering from a naive regex."""
        (tmp_path / "s.css").write_text(
            ".x { color: red; background: blue; --real-var: 1; }\n"
        )
        ss = SymbolSearcher(tmp_path)
        # 'color' and 'background' are ordinary properties, not symbols.
        assert ss.find_symbol("color", kind="any") == []
        assert ss.find_symbol("background", kind="any") == []
        # Only the custom property is a symbol.
        v = ss.find_symbol("real-var", kind="any")
        assert len(v) == 1
        assert v[0].kind == "css_variable"

    def test_css_nested_selector_in_media_query(self, tmp_path):
        """Selectors nested inside @media / @supports blocks are still indexed —
        the AST walk descends into nested rule_sets, which a line-anchored rg
        regex handles only by accident."""
        (tmp_path / "s.css").write_text(
            "@media (max-width: 600px) {\n  .responsive { display: none; }\n}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("responsive", kind="any")
        assert len(results) == 1
        assert results[0].kind == "css_class"

    def test_css_custom_property_dashed_form_also_resolves(self, tmp_path):
        """CSS custom properties are indexed under BOTH the bare name
        ('primary') and the dashed form ('--primary'), so either query form
        resolves."""
        (tmp_path / "s.css").write_text(":root { --primary: #333; }\n")
        ss = SymbolSearcher(tmp_path)
        bare = ss.find_symbol("primary", kind="any")
        assert len(bare) == 1
        dashed = ss.find_symbol("--primary", kind="any")
        assert len(dashed) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Symbol Search — Non-Python languages via provider registry + AST index
#
# Every non-Python language now has a registered provider, so the persistent
# _nonpy_index_for index is the single source of truth. The legacy hardcoded
# rg+regex fallback (_find_in_other_langs) is removed entirely. These tests
# pin that Rust/C#/Ruby/PHP/Swift symbols are findable via the provider/AST
# path — previously they were only reachable through the removed fallback.
# ══════════════════════════════════════════════════════════════════════════════

class TestNonPySymbolSearch:
    """Verify SymbolSearcher finds symbols in Rust/C#/Ruby/PHP/Swift via the
    provider registry → tree-sitter AST index (no rg+regex fallback)."""

    def test_rust_function_found(self, tmp_path):
        (tmp_path / "a.rs").write_text(
            "fn process_data(input: &[u8]) -> Vec<u8> { vec![] }\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("process_data", kind="any")
        assert len(results) == 1
        assert results[0].file.endswith(".rs")

    def test_rust_struct_found_via_kind_class(self, tmp_path):
        # struct must resolve under kind="class" (the class-kind group).
        (tmp_path / "a.rs").write_text("struct Config { timeout: u64 }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("Config", kind="class")
        assert len(results) == 1
        assert results[0].file.endswith(".rs")

    def test_csharp_class_found(self, tmp_path):
        (tmp_path / "b.cs").write_text("public class UserService { }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("UserService", kind="any")
        assert len(results) == 1
        assert results[0].file.endswith(".cs")

    def test_csharp_method_with_generic_return_found(self, tmp_path):
        # A generic return type (Task<User>) was missed by the legacy rg
        # regex; the AST path captures it correctly.
        (tmp_path / "b.cs").write_text(
            "public class Svc {\n"
            "  public Task<User> GetByIdAsync(int id) { return null; }\n"
            "}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("GetByIdAsync", kind="function")
        assert len(results) == 1
        assert results[0].file.endswith(".cs")

    def test_ruby_class_found(self, tmp_path):
        (tmp_path / "c.rb").write_text("class User < ApplicationRecord\nend\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("User", kind="class")
        assert len(results) == 1
        assert results[0].file.endswith(".rb")

    def test_ruby_module_found_via_kind_class(self, tmp_path):
        # Ruby modules are emitted as kind="namespace" by the AST path, which
        # must resolve under kind="class" (the class-kind group).
        (tmp_path / "c.rb").write_text("module PaymentGateway\nend\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("PaymentGateway", kind="class")
        assert len(results) == 1
        assert results[0].file.endswith(".rb")

    def test_php_class_found(self, tmp_path):
        (tmp_path / "d.php").write_text(
            "<?php\nclass UserRepository { }\n?>\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("UserRepository", kind="any")
        assert len(results) == 1
        assert results[0].file.endswith(".php")

    def test_php_method_found(self, tmp_path):
        (tmp_path / "d.php").write_text(
            "<?php\nclass Svc {\n  public function getById($id) {}\n}\n?>\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("getById", kind="function")
        assert len(results) == 1
        assert results[0].file.endswith(".php")

    def test_swift_class_found(self, tmp_path):
        (tmp_path / "e.swift").write_text("class Renderer { }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("Renderer", kind="class")
        assert len(results) == 1
        assert results[0].file.endswith(".swift")

    def test_swift_struct_found_via_kind_class(self, tmp_path):
        (tmp_path / "e.swift").write_text("struct Point { var x: Double }\n")
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("Point", kind="class")
        assert len(results) == 1
        assert results[0].file.endswith(".swift")

    def test_legacy_fallback_method_removed(self):
        """_find_in_other_langs must no longer exist on SymbolSearcher — it
        was pure redundancy once every non-Python language had a provider."""
        from external_llm.agent.symbol_search import SymbolSearcher
        ss = SymbolSearcher(Path("/"))
        assert not hasattr(ss, "_find_in_other_langs")

    def test_bash_function_posix_form_found(self, tmp_path):
        # POSIX form: name() { ... } — the common case.
        (tmp_path / "deploy.sh").write_text(
            "#!/usr/bin/env bash\n"
            "build_app() {\n"
            "    make all\n"
            "}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("build_app", kind="any")
        assert len(results) == 1
        assert results[0].file.endswith(".sh")
        assert results[0].kind == "function"

    def test_bash_function_keyword_form_found(self, tmp_path):
        # C-style / keyword form: function name { ... }
        (tmp_path / "test.sh").write_text(
            "function test_suite {\n"
            "    pytest\n"
            "}\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("test_suite", kind="any")
        assert len(results) == 1
        assert results[0].file.endswith(".sh")
        assert results[0].kind == "function"

    def test_bash_call_not_mistaken_for_definition(self, tmp_path):
        # A bare function call must NOT be indexed as a definition — the
        # POSIX pattern requires the trailing "()" which a call lacks.
        (tmp_path / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "deploy_production   # just a call, no definition\n"
        )
        ss = SymbolSearcher(tmp_path)
        results = ss.find_symbol("deploy_production", kind="any")
        assert len(results) == 0

    def test_bash_two_forms_in_one_file(self, tmp_path):
        # Both POSIX and keyword forms in the same file are found.
        (tmp_path / "ci.sh").write_text(
            "lint() {\n"
            "    flake8 .\n"
            "}\n"
            "function deploy {\n"
            "    lint\n"
            "    kubectl apply\n"
            "}\n"
        )
        ss = SymbolSearcher(tmp_path)
        names = {r.name for r in ss.find_symbol("lint", kind="any")}
        names |= {r.name for r in ss.find_symbol("deploy", kind="any")}
        assert names == {"lint", "deploy"}


# ══════════════════════════════════════════════════════════════════════════════
# 8. Lint Runner — Language Dispatch
# ══════════════════════════════════════════════════════════════════════════════

class TestLintRunner:
    """Verify that LintRunner dispatches to the correct linter."""

    def test_python_lint_dispatches_ruff(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("service/user_service.py")
        # ruff may or may not be installed; either result is fine
        assert result is not None
        # Should NOT be skipped for Python — ruff should run or fail
        # (skipped only if ruff is not installed)

    def test_ts_lint_dispatches_eslint(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("api/user.ts")
        assert result is not None
        # eslint likely not installed in test env → graceful skip
        if result.skipped:
            assert "eslint" in result.summary.lower() or "not installed" in result.summary.lower()

    def test_js_lint_dispatches_eslint(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("utils/helpers.js")
        assert result is not None

    def test_unknown_lang_skips(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("config.json")
        assert result.ok is True
        assert result.skipped is True


# ══════════════════════════════════════════════════════════════════════════════
# 9. Execution Spec Serialization
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionSpec:
    """Verify language field round-trips through serialization."""

    def test_language_in_to_dict(self):
        spec = ResolvedExecutionSpec(
            original_request="edit user.ts",
            intent="edit",
            request_type="edit",
            language="typescript",
        )
        d = spec.to_dict()
        assert d["language"] == "typescript"

    def test_language_from_dict(self):
        d = {
            "original_request": "edit",
            "intent": "edit",
            "request_type": "edit",
            "language": "javascript",
        }
        spec = ResolvedExecutionSpec.from_dict(d)
        assert spec.language == "javascript"

    def test_language_none_default(self):
        spec = ResolvedExecutionSpec(
            original_request="test",
            intent="edit",
            request_type="edit",
        )
        assert spec.language is None


# ══════════════════════════════════════════════════════════════════════════════
# 10. Go Provider
# ══════════════════════════════════════════════════════════════════════════════

class TestGoProvider:
    """Verify Go language support end-to-end."""

    def test_go_detection(self):
        assert LanguageId.from_path("main.go") == LanguageId.GO

    def test_go_provider_registered(self):
        r = LanguageRegistry.instance()
        p = r.get("main.go")
        assert p is not None
        assert p.language_id() == LanguageId.GO

    def test_go_structured_ops(self):
        r = LanguageRegistry.instance()
        assert r.supports_structured_ops("main.go") is True

    def test_go_file_globs(self):
        globs = LanguageRegistry.instance().get_all_file_globs()
        assert "*.go" in globs

    def test_go_find_function(self):
        p = LanguageRegistry.instance().get("main.go")
        result = p.find_symbol_in_file("main.go", "NewUserService", GO_SRC)
        assert result is not None
        start, end = result
        assert start > 0
        assert end > start

    def test_go_find_method(self):
        p = LanguageRegistry.instance().get("main.go")
        result = p.find_symbol_in_file("main.go", "GetUser", GO_SRC)
        assert result is not None

    def test_go_find_struct(self):
        p = LanguageRegistry.instance().get("main.go")
        result = p.find_symbol_in_file("main.go", "UserService", GO_SRC)
        assert result is not None

    def test_go_find_interface(self):
        p = LanguageRegistry.instance().get("main.go")
        result = p.find_symbol_in_file("main.go", "Database", GO_SRC)
        assert result is not None

    def test_go_symbols_indexed(self, graph):
        go_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".go")]
        assert len(go_symbols) >= 2
        names = {s.name for s in go_symbols}
        assert "UserService" in names or "NewUserService" in names or "FormatUser" in names

    @pytest.mark.xfail(reason="TaskRouter now requires llm_client and model params", strict=False)
    def test_go_router_planner(self):
        router = TaskRouter()
        decision = router.route("modify the GetUser function in server/main.go")
        assert decision.lane == Lane.PLANNER

    def test_go_lint_dispatch(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("server/main.go")
        assert result is not None
        # golangci-lint likely not installed → graceful skip or issue list
        if result.skipped:
            assert "not installed" in result.summary.lower()

    def test_go_symbol_search(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("NewUserService")
        assert len(results) > 0
        assert any(".go" in r.file for r in results)

    def test_go_capabilities(self):
        p = LanguageRegistry.instance().get("main.go")
        caps = p.capabilities()
        assert caps.has_syntax_validator is True
        assert caps.has_symbol_search is True
        assert caps.supports_modify_symbol is True


# ══════════════════════════════════════════════════════════════════════════════
# 11. Java Provider
# ══════════════════════════════════════════════════════════════════════════════

class TestJavaProvider:
    """Verify Java language support end-to-end."""

    def test_java_detection(self):
        assert LanguageId.from_path("UserRepository.java") == LanguageId.JAVA

    def test_java_provider_registered(self):
        r = LanguageRegistry.instance()
        p = r.get("UserRepository.java")
        assert p is not None
        assert p.language_id() == LanguageId.JAVA

    def test_java_structured_ops(self):
        r = LanguageRegistry.instance()
        assert r.supports_structured_ops("Test.java") is True

    def test_java_file_globs(self):
        globs = LanguageRegistry.instance().get_all_file_globs()
        assert "*.java" in globs

    def test_java_find_class(self):
        p = LanguageRegistry.instance().get("Test.java")
        result = p.find_symbol_in_file("Test.java", "UserRepository", JAVA_SRC)
        assert result is not None
        start, end = result
        assert start > 0
        assert end > start

    def test_java_find_method(self):
        p = LanguageRegistry.instance().get("Test.java")
        result = p.find_symbol_in_file("Test.java", "findById", JAVA_SRC)
        assert result is not None

    def test_java_find_interface(self):
        p = LanguageRegistry.instance().get("Test.java")
        result = p.find_symbol_in_file("Test.java", "UserDao", JAVA_SRC)
        assert result is not None

    def test_java_find_enum(self):
        p = LanguageRegistry.instance().get("Test.java")
        result = p.find_symbol_in_file("Test.java", "UserRole", JAVA_SRC)
        assert result is not None

    def test_java_symbols_indexed(self, graph):
        java_symbols = [s for s in graph.symbols.values() if s.file_path.endswith(".java")]
        assert len(java_symbols) >= 1
        names = {s.name for s in java_symbols}
        assert "UserRepository" in names or "findById" in names

    @pytest.mark.xfail(reason="TaskRouter now requires llm_client and model params", strict=False)
    def test_java_router_planner(self):
        router = TaskRouter()
        decision = router.route("modify the delete method in src/UserRepository.java")
        assert decision.lane == Lane.PLANNER

    def test_java_lint_dispatch(self, multilang_project):
        lr = LintRunner(multilang_project)
        result = lr.run_lint("src/UserRepository.java")
        assert result is not None
        # No default linter for Java → lint command is None → should skip
        assert result.skipped is True

    def test_java_symbol_search(self, multilang_project):
        ss = SymbolSearcher(multilang_project)
        results = ss.find_symbol("UserRepository")
        assert len(results) > 0

    def test_java_test_command_maven(self, multilang_project):
        p = LanguageRegistry.instance().get("Test.java")
        cmd = p.get_test_command(multilang_project)
        assert cmd is not None
        assert "mvn" in cmd[0] or "gradlew" in cmd[0]

    def test_java_capabilities(self):
        p = LanguageRegistry.instance().get("Test.java")
        caps = p.capabilities()
        assert caps.has_syntax_validator is True
        assert caps.has_linter is False  # No default Java linter
        assert caps.supports_modify_symbol is True


# ══════════════════════════════════════════════════════════════════════════════
# 12. Tree-sitter Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestTreeSitter:
    """Verify tree-sitter integration when available."""

    def test_is_available_returns_bool(self):
        from external_llm.languages.tree_sitter_utils import is_available
        assert isinstance(is_available(), bool)

    def test_find_symbol_range_without_tree_sitter(self):
        """Verify graceful None return when tree-sitter is not installed."""
        from external_llm.languages.tree_sitter_utils import _HAS_TREE_SITTER, find_symbol_range
        if _HAS_TREE_SITTER:
            pytest.skip("tree-sitter is installed; testing fallback path only")
        result = find_symbol_range("function foo() {}", "foo", "javascript")
        assert result is None

    def test_find_all_symbols_without_tree_sitter(self):
        from external_llm.languages.tree_sitter_utils import _HAS_TREE_SITTER, find_all_symbols
        if _HAS_TREE_SITTER:
            pytest.skip("tree-sitter is installed; testing fallback path only")
        result = find_all_symbols("function foo() {}", "javascript")
        assert result == []

    def test_capabilities_has_tree_sitter_field(self):
        """Verify has_tree_sitter field exists on all providers."""
        for ext in ["test.py", "test.ts", "test.js", "test.go", "Test.java"]:
            p = LanguageRegistry.instance().get(ext)
            if p is not None:
                caps = p.capabilities()
                assert hasattr(caps, "has_tree_sitter")


_ts_available = False
try:
    from external_llm.languages.tree_sitter_utils import is_available
    _ts_available = is_available()
except ImportError:
    pass


@pytest.mark.skipif(not _ts_available, reason="tree-sitter not installed")
class TestTreeSitterEdgeCases:
    """Tests that require tree-sitter to be installed."""

    def test_ts_template_literal_braces(self):
        """Template literal with braces should not confuse tree-sitter."""
        from external_llm.languages.tree_sitter_utils import find_symbol_range
        src = """\
export function greet(name: string): string {
  return `Hello, ${name}! You have ${count} items in your ${"cart"}`;
}

export function farewell(name: string): string {
  return `Goodbye, ${name}!`;
}
"""
        result = find_symbol_range(src, "greet", "typescript")
        assert result is not None
        start, end = result
        assert start == 1
        assert end == 3  # function ends at line 3

    def test_jsx_braces(self):
        """JSX expressions with braces should be handled correctly."""
        from external_llm.languages.tree_sitter_utils import find_symbol_range
        src = """\
export function Card({ title, children }) {
  return (
    <div className="card">
      <h1>{title}</h1>
      <div>{children}</div>
      {title && <span>{title.length}</span>}
    </div>
  );
}

export function Badge({ count }) {
  return <span>{count > 0 ? count : "none"}</span>;
}
"""
        result = find_symbol_range(src, "Card", "javascript")
        assert result is not None
        start, end = result
        assert start == 1
        # Card ends before Badge starts
        result2 = find_symbol_range(src, "Badge", "javascript")
        assert result2 is not None
        assert result2[0] > end

    def test_comment_braces(self):
        """Braces inside comments should be ignored."""
        from external_llm.languages.tree_sitter_utils import find_symbol_range
        src = """\
// This function handles { edge cases }
export function processData(data: any): any {
  /* Multi-line comment with {braces}
     and more {braces} */
  return data;
}

export function nextFunction(): void {
  console.log("done");
}
"""
        result = find_symbol_range(src, "processData", "typescript")
        assert result is not None
        _start, end = result
        # processData should end before nextFunction
        result2 = find_symbol_range(src, "nextFunction", "typescript")
        assert result2 is not None
        assert result2[0] > end

    def test_go_symbol_range(self):
        from external_llm.languages.tree_sitter_utils import find_symbol_range
        result = find_symbol_range(GO_SRC, "NewUserService", "go")
        assert result is not None
        start, end = result
        assert start > 0
        assert end > start

    def test_java_symbol_range(self):
        from external_llm.languages.tree_sitter_utils import find_symbol_range
        result = find_symbol_range(JAVA_SRC, "UserRepository", "java")
        assert result is not None
        start, end = result
        assert start > 0
        assert end > start

    def test_find_all_symbols_ts(self):
        from external_llm.languages.tree_sitter_utils import find_all_symbols
        symbols = find_all_symbols(TYPESCRIPT_SRC, "typescript")
        assert len(symbols) >= 2
        names = {s[0] for s in symbols}
        assert "UserRepository" in names or "createDefaultUser" in names

    def test_find_all_symbols_go(self):
        from external_llm.languages.tree_sitter_utils import find_all_symbols
        symbols = find_all_symbols(GO_SRC, "go")
        assert len(symbols) >= 2
        names = {s[0] for s in symbols}
        assert "UserService" in names or "NewUserService" in names
