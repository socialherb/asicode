"""RED→GREEN coverage for ProjectAnalyzer: walk/error paths, manifest parsing
fallbacks, framework marker types, project-type branches, naming styles,
import/entry-point scanners, and example-file discovery.

Companion to test_project_analyzer.py (language/framework/type regressions).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import ClassVar

from external_llm.project_analyzer import ProjectAnalyzer


def _write(root: Path, rel: str, content: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


# ---------------------------------------------------------------------------
# _walk_source_files / _iter_source_files — error & edge paths
# ---------------------------------------------------------------------------


def test_walk_skips_unlistable_dir(tmp_path: Path, monkeypatch):
    _write(tmp_path, "locked/x.py", "x = 1")
    _write(tmp_path, "real.py", "x = 1")
    orig = Path.iterdir

    def fake_iterdir(self):
        if self.name == "locked":
            raise OSError("permission denied")
        return orig(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    s = ProjectAnalyzer(str(tmp_path))
    assert [f.name for f in s._iter_source_files()] == ["real.py"]


def test_walk_skips_broken_symlink(tmp_path: Path, monkeypatch):
    os.symlink(tmp_path / "missing_target", tmp_path / "broken_link")
    s = ProjectAnalyzer(str(tmp_path))
    assert list(s._iter_source_files()) == []


def test_walk_outer_exception_yields_nothing(tmp_path: Path, monkeypatch):
    _write(tmp_path, "a.py", "x = 1")

    def boom(self):
        raise RuntimeError("walk exploded")

    monkeypatch.setattr(Path, "iterdir", boom)
    s = ProjectAnalyzer(str(tmp_path))
    assert list(s._iter_source_files()) == []


def test_iter_source_files_cached_identity(tmp_path: Path):
    _write(tmp_path, "a.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))
    first = s._iter_source_files()
    second = s._iter_source_files()
    assert first is second  # shared immutable cache, not a re-walk


def test_sample_files_str_suffix(tmp_path: Path):
    _write(tmp_path, "a.py", "x = 1")
    _write(tmp_path, "b.txt", "x")
    s = ProjectAnalyzer(str(tmp_path))
    assert [f.name for f in s._sample_files(".py", 5)] == ["a.py"]


# ---------------------------------------------------------------------------
# _dir_exists_pruned — match/skip/error paths
# ---------------------------------------------------------------------------


def test_dir_exists_pruned_match_before_skip(tmp_path: Path):
    """A vendored dir itself matches the target (test-then-prune: the entry is
    compared to *name* before the descent-prune check)."""
    _write(tmp_path, "node_modules/migrations/0001.sql", "-- x")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._dir_exists_pruned("node_modules") is True


def test_dir_exists_pruned_does_not_descend_vendored(tmp_path: Path):
    _write(tmp_path, ".venv/x/migrations/0001.sql", "-- x")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._dir_exists_pruned("migrations") is False


def test_dir_exists_pruned_finds_nested_source_dir(tmp_path: Path):
    _write(tmp_path, ".venv/x/migrations/0001.sql", "-- x")
    _write(tmp_path, "src/migrations/0001.sql", "-- x")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._dir_exists_pruned("migrations") is True


def test_dir_exists_pruned_oserror_skips(tmp_path: Path, monkeypatch):
    _write(tmp_path, "locked/migrations/0001.sql", "-- x")
    orig = Path.iterdir

    def fake_iterdir(self):
        if self.name == "locked":
            raise OSError("denied")
        return orig(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._dir_exists_pruned("migrations") is False


def test_dir_exists_pruned_outer_exception(tmp_path: Path, monkeypatch):
    def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "iterdir", boom)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._dir_exists_pruned("migrations") is False


# ---------------------------------------------------------------------------
# _detect_languages — exclusions, share gate, fallback
# ---------------------------------------------------------------------------


def test_languages_excludes_json_css_html(tmp_path: Path):
    _write(tmp_path, "data.json", "{}")
    _write(tmp_path, "style.css", "body {}")
    _write(tmp_path, "index.html", "<html></html>")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_languages() == []


def test_languages_share_gate_filters_stray_files(tmp_path: Path):
    for i in range(20):
        _write(tmp_path, f"m{i:02d}.js", "x = 1")
    _write(tmp_path, "build_script.py", "x = 1")  # 1/21 ≈ 4.8% < 5%
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_languages() == ["javascript"]


def test_languages_single_file_fallback(tmp_path: Path):
    _write(tmp_path, "main.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_languages() == ["python"]


def test_languages_excludes_gradle_scripts(tmp_path: Path):
    _write(tmp_path, "build.gradle.kts", "plugins {}")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_languages() == []


def test_detect_frameworks_languages_none(tmp_path: Path):
    _write(tmp_path, "app.py", "from flask import Flask\napp = Flask(__name__)\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_frameworks() == ["flask"]  # None → self-detected languages


def test_detect_frameworks_language_gate(tmp_path: Path):
    """A stray Python file in a Go repo must not unlock Python frameworks."""
    _write(tmp_path, "go.mod", "module example.com/tool\n")
    _write(tmp_path, "main.go", "package main\n")
    _write(tmp_path, "scripts/oneoff.py", "from flask import Flask\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._detect_languages() == ["go"]
    assert "flask" not in s._detect_frameworks()


# ---------------------------------------------------------------------------
# Framework markers: import / pkg_dep / file_ext / jvm_import error paths
# ---------------------------------------------------------------------------


def test_import_marker_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "app.py", "from flask import Flask\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.suffix == ".py":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "flask" not in s.frameworks
    assert s.languages == ["python"]  # language counting never reads files


def test_pkg_dep_merges_deps_and_devdeps(tmp_path: Path):
    _write(tmp_path, "package.json", '{"dependencies": {"react": "^18"}, "devDependencies": {"vue": "^3"}}')
    _write(tmp_path, "x.ts", "export const x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "react" in s.frameworks
    assert "vue" in s.frameworks


def test_pkg_dep_invalid_json_ignored(tmp_path: Path):
    _write(tmp_path, "package.json", "{broken json")
    _write(tmp_path, "x.js", "const x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "react" not in s.frameworks
    assert s.frameworks == []


def test_file_ext_marker_detects_astro(tmp_path: Path):
    _write(tmp_path, "x.ts", "export const x = 1")
    _write(tmp_path, "pages/index.astro", "---\nconst a = 1\n---")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "astro" in s.frameworks  # file_ext .astro → weight 2


def test_jvm_import_marker_detects_jetpack(tmp_path: Path):
    _write(
        tmp_path,
        "app/src/main/java/com/x/Main.kt",
        "package com.x\n\nimport androidx.compose.runtime.Composable\n\nclass Main\n",
    )
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "kotlin" in s.languages
    assert "jetpack-compose" in s.frameworks


def test_jvm_import_marker_no_match(tmp_path: Path):
    _write(tmp_path, "Main.kt", "package com.x\n\nclass Main\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "jetpack-compose" not in s.frameworks


def test_jvm_import_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "Main.kt", "import androidx.compose.runtime.Composable\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.suffix == ".kt":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "jetpack-compose" not in s.frameworks


# ---------------------------------------------------------------------------
# _go_mod_requires — missing / unreadable / prefix match
# ---------------------------------------------------------------------------


def test_go_mod_requires_missing_file(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_mod_requires("github.com/spf13/cobra") is False


def test_go_mod_requires_unreadable(tmp_path: Path):
    _write(tmp_path, "go.mod", b"\xff\xfe\x00\x01invalid")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_mod_requires("github.com/spf13/cobra") is False


def test_go_mod_requires_prefix_match(tmp_path: Path):
    _write(tmp_path, "go.mod", "module example.com/tool\n\nrequire (\n\tgithub.com/spf13/cobra v1.10.2\n)\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_mod_requires("github.com/spf13") is True


# ---------------------------------------------------------------------------
# _read_gradle_text — bounds, read failures, outer exception
# ---------------------------------------------------------------------------


def test_gradle_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "build.gradle.kts", 'plugins { id("com.android.application") }\n')
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.name == "build.gradle.kts":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_gradle_text() == ""


def test_gradle_toml_total_bound_breaks(tmp_path: Path, monkeypatch):
    for i in range(10):
        _write(tmp_path, f"m{i}.gradle", "x\n" * 500_000)  # ~1 MB each
    _write(tmp_path, "gradle/libs.versions.toml", "android = 'com.android.application'")
    s = ProjectAnalyzer(str(tmp_path))
    monkeypatch.setattr(s, "_GRADLE_TOTAL_MAX_BYTES", 8 * 1024 * 1024)
    text = s._read_gradle_text()
    # total already at cap when the toml loop starts → break before reading it
    assert "com.android.application" not in text


def test_gradle_toml_oversized_file_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "gradle/libs.versions.toml", "x" * 2_000_000)  # > 1 MiB
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_gradle_text() == ""


def test_gradle_toml_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "gradle/libs.versions.toml", "android = 'com.android.application'")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.name == "libs.versions.toml":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_gradle_text() == ""


def test_gradle_toml_truncated_at_total_cap(tmp_path: Path, monkeypatch):
    _write(tmp_path, "a.gradle", "hello")  # 5 chars
    _write(tmp_path, "gradle/libs.versions.toml", "world")  # 5 chars
    s = ProjectAnalyzer(str(tmp_path))
    monkeypatch.setattr(s, "_GRADLE_TOTAL_MAX_BYTES", 8)
    text = s._read_gradle_text()
    assert text == "hello\nwor"  # second chunk truncated to fit 8-total


def test_gradle_outer_exception_returns_empty(tmp_path: Path, monkeypatch):
    _write(tmp_path, "a.gradle", "x")
    s = ProjectAnalyzer(str(tmp_path))
    monkeypatch.setattr(s, "_iter_source_files", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert s._read_gradle_text() == ""


# ---------------------------------------------------------------------------
# _read_pyproject_deps — failure / non-dict / poetry python skip / setup.py
# ---------------------------------------------------------------------------


def test_pyproject_parse_failure_yields_empty_deps(tmp_path: Path):
    _write(tmp_path, "pyproject.toml", "[project\ndependencies = [\n")  # invalid TOML
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_pyproject_deps() == set()


def test_pyproject_poetry_python_constraint_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "pyproject.toml",
        ('[tool.poetry]\nname = "demo"\n[tool.poetry.dependencies]\npython = "^3.9"\nflask = "^3.0"\nrich = "^13.0"\n'),
    )
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_pyproject_deps() == {"flask", "rich"}


def test_setup_py_non_string_install_requires_elt_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "setup.py",
        ('from setuptools import setup\nsetup(name="demo", install_requires=["flask>=3.0", 123, "rich>=13"])\n'),
    )
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_pyproject_deps() == {"flask", "rich"}


def test_setup_py_parse_error_ignored(tmp_path: Path):
    _write(tmp_path, "setup.py", "def setup(:\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_pyproject_deps() == set()


# ---------------------------------------------------------------------------
# _parse_toml / _parse_toml_fallback
# ---------------------------------------------------------------------------


def test_parse_toml_uses_stdlib(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml('[project]\nname = "demo"\n')
    assert data == {"project": {"name": "demo"}}


def test_parse_toml_falls_back_when_tomllib_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml('[project]\nname = "demo"\n')
    assert data == {"project": {"name": "demo"}}


def test_parse_toml_fallback_full(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml_fallback(
        "# comment\n"
        "\n"
        "[project]\n"
        'name = "demo"\n'
        'dependencies = ["fastapi>=0.110", "pydantic>=2.0"]\n'
        "[tool.poetry.dependencies]\n"
        'flask = "^3.0"\n'
        "not a key value line\n"
    )
    assert data == {
        "project": {
            "name": "demo",
            "dependencies": ["fastapi>=0.110", "pydantic>=2.0"],
        },
        "tool": {"poetry": {"dependencies": {"flask": "^3.0"}}},
    }


def test_parse_toml_fallback_quotes_with_commas(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml_fallback('deps = ["a>=1,<2", "b", 3]\n')
    assert data == {"deps": ["a>=1,<2", "b", "3"]}


def test_read_pyproject_deps_via_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    _write(
        tmp_path,
        "pyproject.toml",
        (
            "[project]\n"
            'dependencies = ["fastapi>=0.110", "PyYAML"]\n'
            "[tool.poetry.dependencies]\n"
            'flask = "^3.0"\n'
            'python = "^3.9"\n'
        ),
    )
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_pyproject_deps() == {"fastapi", "pyyaml", "flask"}


# ---------------------------------------------------------------------------
# _detect_project_types — CLI fallbacks, library, default package
# ---------------------------------------------------------------------------


def test_project_types_cli_via_main_guard(tmp_path: Path):
    _write(tmp_path, "tool.py", "import sys\n\nif __name__ == '__main__':\n    sys.argv\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "cli" in s.project_types


def test_project_types_cli_via_stdlib_hit(tmp_path: Path):
    """A bare argparse import (no __main__ guard) still marks a CLI."""
    _write(tmp_path, "tool.py", "import argparse\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "cli" in s.project_types


def test_project_types_cli_read_failure_falls_through(tmp_path: Path, monkeypatch):
    _write(tmp_path, "tool.py", "import argparse\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.suffix == ".py":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "cli" not in s.project_types
    assert s.project_types == ["package"]


def test_project_types_go_cli_fallback(tmp_path: Path):
    """Go repo, no CLI framework: `package main` alone ⇒ cli (not library)."""
    _write(tmp_path, "go.mod", "module example.com/tool\n")
    _write(tmp_path, "main.go", "package main\n\nfunc main() {}\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.project_types == ["cli"]


def test_project_types_library_via_setup_cfg(tmp_path: Path):
    _write(tmp_path, "setup.cfg", "[metadata]\nname = demo\n")
    _write(tmp_path, "demo/__init__.py", "")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.project_types == ["library"]


def test_project_types_library_go_module_without_main(tmp_path: Path):
    _write(tmp_path, "go.mod", "module example.com/lib\n")
    _write(tmp_path, "pkg/util.go", "package util\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.project_types == ["library"]


def test_project_types_default_package(tmp_path: Path):
    _write(tmp_path, "main.py", "x = 1\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.project_types == ["package"]


# ---------------------------------------------------------------------------
# _go_has_main_package — scanned / break / read failure / empty
# ---------------------------------------------------------------------------


def test_go_has_main_via_main_go(tmp_path: Path):
    _write(tmp_path, "main.go", "package main\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is True


def test_go_has_main_scanned(tmp_path: Path):
    _write(tmp_path, "cmd/run.go", "package main\n\nfunc main() {}\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is True


def test_go_has_main_non_main_breaks_scan(tmp_path: Path):
    _write(tmp_path, "cmd/root.go", "package cmd\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is False


def test_go_has_main_read_failure(tmp_path: Path, monkeypatch):
    _write(tmp_path, "cmd/run.go", "package main\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.suffix == ".go":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is False


def test_go_has_main_no_go_files(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is False


# ---------------------------------------------------------------------------
# _analyze_directories — purpose mapping, dotdirs, failure
# ---------------------------------------------------------------------------


def test_analyze_directories_purposes(tmp_path: Path):
    for d in (
        "models",
        "views",
        "api",
        "services",
        "agents",
        "utils",
        "tests",
        "static",
        "templates",
        "config",
        "misc",
        ".hidden",
    ):
        _write(tmp_path, d + "/keep.txt", "x")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._analyze_directories() == {
        "models": ["models"],
        "views": ["views"],
        "routes": ["api"],
        "services": ["services"],
        "agents": ["agents"],
        "utils": ["utils"],
        "tests": ["tests"],
        "static": ["static"],
        "templates": ["templates"],
        "config": ["config"],
        "other": ["misc"],
    }


def test_analyze_directories_iterdir_failure(tmp_path: Path, monkeypatch):
    _write(tmp_path, "models/keep.txt", "x")

    def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "iterdir", boom)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._analyze_directories() == {}


# ---------------------------------------------------------------------------
# _detect_naming_style — camel/pascal/skips/exception
# ---------------------------------------------------------------------------


def test_naming_style_camel_case(tmp_path: Path):
    _write(tmp_path, "myFile.ts", "export const x = 1")
    _write(tmp_path, "otherFile.ts", "export const y = 2")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.naming_style == "camelCase"


def test_naming_style_pascal_majority(tmp_path: Path):
    _write(tmp_path, "DataModel.ts", "export const x = 1")
    _write(tmp_path, "UserService.ts", "export const x = 1")
    _write(tmp_path, "util_fn.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.naming_style == "PascalCase"


def test_naming_style_skips_framework_names(tmp_path: Path):
    for name in (
        "index.ts",
        "next.ts",
        "vite.ts",
        "nuxt.ts",
        "astro.ts",
        "tailwind.ts",
        "postcss.ts",
        "package.ts",
        "tsconfig.ts",
        "eslint.ts",
        "prettier.ts",
    ):
        _write(tmp_path, name, "export const x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.naming_style == "unknown"  # all skipped → no evidence


def test_naming_style_scan_exception(tmp_path: Path, monkeypatch):
    _write(tmp_path, "a.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(s, "_sample_files", boom)
    assert s._detect_naming_style() is None


# ---------------------------------------------------------------------------
# _find_common_imports — python/js/relative/read-failure
# ---------------------------------------------------------------------------


def test_common_imports_python_and_js(tmp_path: Path):
    _write(
        tmp_path, "app.py", "import os\nimport os\nfrom django.db import models\nfrom .relative import x\nfrom\n"
    )  # 'from' alone → no module token
    _write(
        tmp_path,
        "client.ts",
        "import { a } from 'lodash'\n"
        "import c from './local'\n"
        'import d from "@scope/pkg"\n'
        "const b = require('react')\n"
        'const e = require("./relative2")\n',
    )
    s = ProjectAnalyzer(str(tmp_path))
    imports = s._find_common_imports()
    assert "os" in imports
    assert "django" in imports
    assert "lodash" in imports
    assert "react" in imports
    assert "@scope" in imports
    assert "local" not in imports
    assert "relative" not in imports
    assert "relative2" not in imports


def test_common_imports_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "bad.py", "import os\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.name == "bad.py":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_common_imports() == []


def test_common_imports_scan_exception(tmp_path: Path, monkeypatch):
    _write(tmp_path, "a.py", "import os\n")
    s = ProjectAnalyzer(str(tmp_path))

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(s, "_sample_files", boom)
    assert s._find_common_imports() == []


# ---------------------------------------------------------------------------
# _find_entry_points — all manifest sources + fallbacks
# ---------------------------------------------------------------------------


def test_entry_points_pyproject_scripts(tmp_path: Path):
    _write(tmp_path, "pyproject.toml", ("[project]\nname = \"demo\"\n[project.scripts]\nascii = 'pkg.cli:main'\n"))
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "ascii (pkg.cli:main)" in s.entry_points


def test_entry_points_poetry_scripts(tmp_path: Path):
    _write(tmp_path, "pyproject.toml", ("[tool.poetry]\nname = \"demo\"\n[tool.poetry.scripts]\nxcli = 'pkg.x:main'\n"))
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "xcli (pkg.x:main)" in s.entry_points


def test_entry_points_tomli_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    fake_tomli = types.SimpleNamespace(load=lambda f: {"project": {"scripts": {"ascii": "pkg.cli:main"}}})
    monkeypatch.setitem(sys.modules, "tomli", fake_tomli)
    _write(tmp_path, "pyproject.toml", "[project]\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "ascii (pkg.cli:main)" in s.entry_points


def test_entry_points_toml_both_fail(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.delitem(sys.modules, "tomli", raising=False)
    _write(tmp_path, "pyproject.toml", "[project]\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []


def test_entry_points_setup_py_console_scripts(tmp_path: Path):
    _write(
        tmp_path,
        "setup.py",
        (
            "from setuptools import setup\n"
            "setup(\n"
            '    name="demo",\n'
            "    entry_points={\n"
            "        'console_scripts': ['ascii=pkg.cli:main', 'xcli=pkg.x:main'],\n"
            "    },\n"
            ")\n"
        ),
    )
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "ascii=pkg.cli:main" in s.entry_points
    assert "xcli=pkg.x:main" in s.entry_points


def test_entry_points_setup_py_literal_eval_failure(tmp_path: Path):
    _write(
        tmp_path,
        "setup.py",
        (
            "from setuptools import setup\n"
            "CONFIG = {'console_scripts': ['x=1']}\n"
            "setup(name='demo', entry_points=UNRESOLVED)\n"
        ),
    )
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []


def test_entry_points_setup_py_parse_failure(tmp_path: Path):
    _write(tmp_path, "setup.py", "def setup(:\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []


def test_entry_points_setup_cfg(tmp_path: Path):
    _write(
        tmp_path,
        "setup.cfg",
        ("[options.entry_points]\nconsole_scripts =\n    ascii = pkg.cli:main\n    xcli = pkg.x:main\n"),
    )
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "ascii = pkg.cli:main" in s.entry_points
    assert "xcli = pkg.x:main" in s.entry_points


def test_entry_points_package_json_bin_str_and_main(tmp_path: Path):
    _write(tmp_path, "package.json", '{"bin": "bin/cli.js", "main": "lib/index.js"}')
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "bin/cli.js" in s.entry_points
    assert "lib/index.js" in s.entry_points


def test_entry_points_package_json_bin_dict(tmp_path: Path):
    _write(tmp_path, "package.json", '{"bin": {"ascii": "bin/cli.js", "xcli": "bin/x.js"}}')
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "bin/cli.js" in s.entry_points
    assert "bin/x.js" in s.entry_points


def test_entry_points_package_json_invalid(tmp_path: Path):
    _write(tmp_path, "package.json", "{invalid")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []


def test_entry_points_strong_candidates(tmp_path: Path):
    candidates = [
        "main.py",
        "app.py",
        "manage.py",
        "wsgi.py",
        "asgi.py",
        "asi.py",
        "__main__.py",
        "index.ts",
        "index.tsx",
        "index.js",
        "index.jsx",
        "main.go",
    ]
    for c in candidates:
        _write(tmp_path, c, "x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    for c in candidates:
        assert c in s.entry_points, f"{c} should be an entry point"


def test_entry_points_weak_cli_with_entry_pattern(tmp_path: Path):
    _write(tmp_path, "cli.py", "def main():\n    pass\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "cli.py" in s.entry_points


def test_entry_points_weak_cli_without_pattern(tmp_path: Path):
    _write(tmp_path, "cli.py", "x = 1\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "cli.py" not in s.entry_points


def test_has_entry_pattern_oserror(tmp_path: Path, monkeypatch):
    _write(tmp_path, "cli.py", "def main():\n    pass\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.name == "cli.py":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    assert ProjectAnalyzer._has_entry_pattern(tmp_path / "cli.py") is False


# ---------------------------------------------------------------------------
# _find_test_dir
# ---------------------------------------------------------------------------


def test_find_test_dir_found(tmp_path: Path):
    _write(tmp_path, "tests/test_x.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_test_dir() == "tests"


def test_find_test_dir_priority(tmp_path: Path):
    _write(tmp_path, "test/test_x.py", "x = 1")
    _write(tmp_path, "spec/spec_x.js", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_test_dir() == "test"


# ---------------------------------------------------------------------------
# _first_glob / _find_example_files
# ---------------------------------------------------------------------------


def test_first_glob_match_and_none(tmp_path: Path):
    _write(tmp_path, "a.py", "x = 1")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._first_glob("*.py") == tmp_path / "a.py"
    assert s._first_glob("*.rs") is None


def test_example_files_django(tmp_path: Path):
    _write(tmp_path, "app/views.py", "")
    _write(tmp_path, "app/models.py", "")
    _write(tmp_path, "app/urls.py", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("django") == {
        "views": "app/views.py",
        "models": "app/models.py",
        "urls": "app/urls.py",
    }


def test_example_files_fastapi(tmp_path: Path):
    _write(tmp_path, "routers/users.py", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("fastapi") == {"routers": "routers/users.py"}


def test_example_files_flask(tmp_path: Path):
    _write(tmp_path, "routes/auth.py", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("flask") == {"routes": "routes/auth.py"}


def test_example_files_react(tmp_path: Path):
    _write(tmp_path, "components/Button.tsx", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("react") == {"components": "components/Button.tsx"}


def test_example_files_nextjs_pages(tmp_path: Path):
    _write(tmp_path, "pages/index.tsx", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("nextjs") == {"pages": "pages/index.tsx"}


def test_example_files_vue(tmp_path: Path):
    _write(tmp_path, "components/App.vue", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("vue") == {"components": "components/App.vue"}


def test_example_files_nuxt_layouts(tmp_path: Path):
    _write(tmp_path, "layouts/Default.vue", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("nuxt") == {"layouts": "layouts/Default.vue"}


def test_example_files_no_match(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_example_files("django") == {}


def test_example_files_scan_exception(tmp_path: Path, monkeypatch):
    s = ProjectAnalyzer(str(tmp_path))

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(s, "_first_glob", boom)
    assert s._find_example_files("django") == {}


# ---------------------------------------------------------------------------
# Full analyze() pipeline sanity
# ---------------------------------------------------------------------------


def test_analyze_full_pipeline(tmp_path: Path):
    _write(tmp_path, "pyproject.toml", ('[project]\nname = "demo"\ndependencies = ["fastapi>=0.110"]\n'))
    _write(tmp_path, "main.py", "x = 1\n")
    _write(tmp_path, "models/User.py", "x = 1\n")
    _write(tmp_path, "tests/test_x.py", "x = 1\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.languages == ["python"]
    assert s.primary_language == "python"
    assert s.frameworks == ["fastapi"]
    assert s.framework == "fastapi"
    assert s.project_types == ["web"]
    assert s.entry_points == ["main.py"]
    assert s.test_dir == "tests"
    assert s.directories == {"models": ["models"], "tests": ["tests"]}


# ---------------------------------------------------------------------------
# Remaining branch arcs (post-GREEN): marker-absent, unknown marker type,
# fallback comma/raw values, naming tie, unterminated quotes, TS read failure,
# setup.cfg extra key + parse failure
# ---------------------------------------------------------------------------


def test_pkg_dep_marker_absent(tmp_path: Path):
    """package.json present but without the marker dep → no false positive."""
    _write(tmp_path, "package.json", '{"dependencies": {"lodash": "^4.17"}}')
    _write(tmp_path, "x.js", "const x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "react" not in s.frameworks
    assert s.frameworks == []


class _WeirdAnalyzer(ProjectAnalyzer):
    FRAMEWORK_MARKERS: ClassVar[dict] = {
        **ProjectAnalyzer.FRAMEWORK_MARKERS,
        "weird": [("bogus_type", "anything", 3)],
    }


def test_unknown_marker_type_ignored_with_log(tmp_path: Path, caplog):
    """A typo'd marker type falls through the whole chain — logged, not fatal."""
    _write(tmp_path, "app.py", "x = 1\n")
    s = _WeirdAnalyzer(str(tmp_path))
    with caplog.at_level("DEBUG", logger="external_llm.project_analyzer"):
        frameworks = s._detect_frameworks(["python"])
    assert "weird" not in frameworks
    assert "unknown marker type 'bogus_type'" in caplog.text


def test_parse_toml_fallback_comma_and_raw_values(tmp_path: Path):
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml_fallback('deps = [1, 2, "c"]\ncount = 3\n')
    assert data == {"deps": ["1", "2", "c"], "count": "3"}


def test_naming_style_snake_pascal_tie_defaults_snake(tmp_path: Path):
    _write(tmp_path, "util_fn.py", "x = 1")
    _write(tmp_path, "DataModel.ts", "export const x = 1")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.naming_style == "snake_case"


def test_common_imports_unterminated_quotes(tmp_path: Path):
    _write(
        tmp_path, "weird.ts", "import a from 'unterminated\nconst b = require('unterminated\nimport c from 'lodash'\n"
    )
    s = ProjectAnalyzer(str(tmp_path))
    imports = s._find_common_imports()
    assert "lodash" in imports  # unterminated lines are simply skipped


def test_common_imports_ts_read_failure_skipped(tmp_path: Path, monkeypatch):
    _write(tmp_path, "bad.ts", "import a from 'lodash'\n")
    orig = Path.read_text

    def fake_read(self, *a, **k):
        if self.suffix == ".ts":
            raise OSError("denied")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    s = ProjectAnalyzer(str(tmp_path))
    assert s._find_common_imports() == []


def test_entry_points_setup_cfg_extra_key_ignored(tmp_path: Path):
    _write(tmp_path, "setup.cfg", ("[options.entry_points]\ngui_scripts =\n    gx = pkg.gx:main\n"))
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []  # gui_scripts is not a console script


def test_entry_points_setup_cfg_parse_failure(tmp_path: Path):
    _write(tmp_path, "setup.cfg", "this is not a config section\n")
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert s.entry_points == []


def test_go_has_main_leading_comment_line(tmp_path: Path):
    """A non-package line before the package clause exercises the scan loop."""
    _write(tmp_path, "cmd/run.go", "// generated\npackage main\n\nfunc main() {}\n")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is True


def test_go_has_main_empty_file(tmp_path: Path):
    _write(tmp_path, "cmd/empty.go", "")
    s = ProjectAnalyzer(str(tmp_path))
    assert s._go_has_main_package() is False


def test_parse_toml_fallback_nested_table_revisit(tmp_path: Path):
    """[a] then [a.b] — descend revisits the existing 'a' dict."""
    s = ProjectAnalyzer(str(tmp_path))
    data = s._parse_toml_fallback("[a]\nx = '1'\n[a.b]\ny = '2'\n")
    assert data == {"a": {"x": "1", "b": {"y": "2"}}}
