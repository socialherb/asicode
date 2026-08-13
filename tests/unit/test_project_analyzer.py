"""Tests for ProjectAnalyzer language/framework/type detection.

Regression coverage for the bug where a Go CLI repo was misreported as a
Django web project (a bare ``migrations/`` dir matched Django, project type was
derived from it, and ``snake_case`` was returned with zero evidence). The fixes
are: language detection + language-gated framework scoring, a minimum-evidence
score floor, and honest "unknown" naming when nothing recognizable is scanned.
"""
from __future__ import annotations

from pathlib import Path

from external_llm.project_analyzer import ProjectAnalyzer


def _write(root: Path, rel: str, content: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_go_cli_repo_detected_as_go_cli(tmp_path: Path):
    """A Go repo using cobra is go/cli/cobra — not django/web."""
    _write(tmp_path, "go.mod", "module example.com/tool\n\nrequire (\n\tgithub.com/spf13/cobra v1.10.2\n)\n")
    _write(tmp_path, "main.go", "package main\n\nfunc main() {}\n")
    _write(tmp_path, "internal/cmd/root.go", "package cmd\n")
    # A generic migrations dir must NOT imply Django for a Go repo.
    _write(tmp_path, "internal/db/migrations/0001.sql", "-- migration")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == ["go"]
    assert s.primary_language == "go"
    assert "cobra" in s.frameworks
    assert "django" not in s.frameworks
    assert s.project_types == ["cli"]
    assert "web" not in s.project_types
    # No Python/JS files scanned → no basis for a naming convention.
    assert s.naming_style == "unknown"
    assert "main.go" in s.entry_points


def test_go_dep_ignores_indirect(tmp_path: Path):
    """Transitive (// indirect) deps must not be reported as project frameworks."""
    _write(
        tmp_path,
        "go.mod",
        "module example.com/tool\n\nrequire (\n"
        "\tgithub.com/spf13/cobra v1.10.2\n"
        "\tgithub.com/gin-gonic/gin v1.0.0 // indirect\n)\n",
    )
    _write(tmp_path, "main.go", "package main\n\nfunc main() {}\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "cobra" in s.frameworks
    assert "gin" not in s.frameworks  # indirect → not a direct framework
    assert "web" not in s.project_types


def test_migrations_dir_alone_does_not_imply_django(tmp_path: Path):
    """Within Python, a lone generic ``migrations`` dir can't confirm Django.

    A Flask app using Alembic has a migrations/ dir but is not Django.
    """
    _write(tmp_path, "app.py", "from flask import Flask\n\napp = Flask(__name__)\n")
    _write(tmp_path, "migrations/env.py", "# alembic env\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == ["python"]
    assert "flask" in s.frameworks
    assert "django" not in s.frameworks


def test_real_django_still_detected(tmp_path: Path):
    """Definitive Django signals (manage.py + import) are still detected."""
    _write(tmp_path, "manage.py", "import os\n")
    _write(tmp_path, "myapp/models.py", "from django.db import models\n")
    _write(tmp_path, "myapp/migrations/0001_initial.py", "# generated\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "django" in s.frameworks
    assert "web" in s.project_types


def test_python_cli_naming_and_type(tmp_path: Path):
    """A Python argparse CLI reports snake_case naming and cli type."""
    _write(
        tmp_path,
        "my_tool.py",
        "import argparse\n\nif __name__ == '__main__':\n    argparse.ArgumentParser()\n",
    )
    _write(tmp_path, "helper_utils.py", "x = 1\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == ["python"]
    # argparse is stdlib → never reported as a framework.
    assert "argparse" not in s.frameworks
    assert s.frameworks == []
    # …but it still marks the project as a CLI.
    assert "cli" in s.project_types
    assert s.naming_style == "snake_case"


def test_marker_literal_does_not_self_trigger(tmp_path: Path):
    """A quoted framework name in source must not be matched as a real import.

    Import markers are matched as line-prefixes, so ``"from django."`` inside a
    string literal does not score Django.
    """
    _write(
        tmp_path,
        "scanner.py",
        'MARKERS = ["from django.", "import flask"]\n'
        "import argparse\n"
        "if __name__ == '__main__':\n    argparse.ArgumentParser()\n",
    )

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "django" not in s.frameworks
    assert "flask" not in s.frameworks
    # argparse is stdlib → never a framework (only django/flask literals were
    # present, and those are correctly rejected).
    assert s.frameworks == []


def test_empty_repo_is_unknown_not_defaulted(tmp_path: Path):
    """No source files → no languages, no false naming convention."""
    _write(tmp_path, "README.md", "# docs only\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == []
    assert s.frameworks == []
    assert s.naming_style == "unknown"


# ---------------------------------------------------------------------------
# Manifest-based (py_dep) framework detection — the alias/sampling-immune path.
# ---------------------------------------------------------------------------


def test_pyproject_deps_drive_framework_detection(tmp_path: Path):
    """A framework declared in pyproject.toml is detected even with no matching
    import statement in the (small) source sample.

    Regression for the FastAPI/tree-sitter miss: their import usage lived deep
    under external_llm/ and the fixed 50-file sample never reached it, while the
    manifest always declared them. pyproject.toml is the authoritative signal.
    """
    _write(tmp_path, "pyproject.toml", (
        "[project]\n"
        'name = "demo"\n'
        "dependencies = [\n"
        '  "fastapi>=0.110",\n'
        '  "pydantic>=2.0",\n'
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        'tree-sitter = [\n'
        '  "tree-sitter>=0.23",\n'
        "]\n"
    ))
    # NOTE: no `from fastapi` / `import tree_sitter` anywhere — only manifest.
    _write(tmp_path, "main.py", "x = 1\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "fastapi" in s.frameworks
    assert "pydantic" in s.frameworks
    assert "tree-sitter" in s.frameworks
    assert "web" in s.project_types


def test_py_dep_detects_fastapi_without_import_in_sample(tmp_path: Path):
    """FastAPI is detected from pyproject.toml alone, even though the only
    FastAPI import lives beyond the source sample window.

    Direct regression test for the reported bug: a repo depending on FastAPI in
    pyproject.toml was reported as `argparse`-only because the import usage sat
    beyond the 50-file sample.
    """
    _write(tmp_path, "pyproject.toml", (
        "[project]\n"
        'name = "demo"\n'
        'dependencies = ["fastapi>=0.110"]\n'
    ))
    # Many root-level files so the import-sample window is "used up" before…
    for i in range(60):
        _write(tmp_path, f"mod_{i:02d}.py", "z = 1\n")
    # …this FastAPI import, which must NOT be the (only) detection signal.
    _write(tmp_path, "deep/app.py", "from fastapi import FastAPI\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "fastapi" in s.frameworks
    assert "web" in s.project_types


def test_tree_sitter_detected_via_manifest_not_self_wrapped_import(tmp_path: Path):
    """tree-sitter wrapped behind a local module (e.g. ``tree_sitter_utils``)
    is detected from the manifest, not from the wrapped import name.
    """
    _write(tmp_path, "pyproject.toml", (
        "[project]\n"
        'name = "demo"\n'
        'dependencies = ["tree-sitter>=0.23", "tree-sitter-python>=0.23"]\n'
    ))
    # The package is only ever imported via a self-wrapping helper — a bare
    # `import tree_sitter` never appears. Manifest detection must still fire.
    _write(tmp_path, "ts_utils.py",
           "from external_llm.languages.tree_sitter_utils import parse\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "tree-sitter" in s.frameworks


def test_setup_py_install_requires_detected(tmp_path: Path):
    """setup.py install_requires is parsed for manifest-declared frameworks."""
    _write(tmp_path, "setup.py", (
        "from setuptools import setup\n"
        "setup(\n"
        '    name="demo",\n'
        '    install_requires=["flask>=3.0", "rich>=13"],\n'
        ")\n"
    ))
    _write(tmp_path, "app.py", "x = 1\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "flask" in s.frameworks
    assert "rich" in s.frameworks
    assert "web" in s.project_types


def test_normalize_dep_name_strips_version_and_extras():
    """PEP 508 specs collapse to a bare, case-folded distribution name."""
    norm = ProjectAnalyzer._normalize_dep_name
    assert norm("tree-sitter>=0.23") == "tree-sitter"
    assert norm("fastapi[all]>=0.110,<1") == "fastapi"
    assert norm("PyYAML") == "pyyaml"
    assert norm("typing_extensions ; python_version<'3.12'") == "typing-extensions"


def test_normalize_dep_name_is_alias_immune():
    """Alias imports can't fool manifest detection — the declared name wins."""
    norm = ProjectAnalyzer._normalize_dep_name
    assert norm("tree-sitter") == "tree-sitter"


def test_argparse_is_stdlib_not_framework(tmp_path: Path):
    """argparse (Python stdlib) must never appear in the framework list, even
    when directly imported — but it still classifies the project as a CLI."""
    _write(tmp_path, "cli.py",
           "import argparse\n"
           "if __name__ == '__main__':\n    argparse.ArgumentParser()\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "argparse" not in s.frameworks
    assert "cli" in s.project_types


def test_multi_framework_polyglot_project(tmp_path: Path):
    """A project using several distinct declared frameworks surfaces all of
    them (multi-framework support, not just the top scorer)."""
    _write(tmp_path, "pyproject.toml", (
        "[project]\n"
        'name = "demo"\n'
        'dependencies = ["fastapi>=0.110", "typer>=0.12", "rich>=13", "libcst>=1.4"]\n'
    ))
    _write(tmp_path, "main.py", "x = 1\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    for fw in ("fastapi", "typer", "rich", "libcst"):
        assert fw in s.frameworks, f"{fw} should be detected"
    assert "web" in s.project_types
    assert "cli" in s.project_types  # typer → cli


# ---------------------------------------------------------------------------
# JVM / Android detection (gradle_text + jvm_import markers).
# Regression for the AsRecord miss: an Android app was reported as
# 'package' / 'unknown' because (a) the analyzer never scanned .kt files for
# naming, (b) it had no Android framework markers, and (c) the Android plugin
# id lived in a Gradle version catalog (libs.versions.toml), not the .gradle.kts.
# ---------------------------------------------------------------------------


def test_android_app_with_version_catalog(tmp_path: Path):
    """An Android app declaring plugins via a Gradle version catalog is
    detected as kotlin / android / jetpack-compose, project type 'mobile',
    and PascalCase naming."""
    _write(tmp_path, "settings.gradle.kts",
           'pluginManagement {\n  repositories { google(); mavenCentral() }\n}\n'
           'rootProject.name = "AsRecord"\n')
    _write(tmp_path, "build.gradle.kts",
           "plugins {\n"
           "    alias(libs.plugins.android.application) apply false\n"
           "    alias(libs.plugins.kotlin.android) apply false\n"
           "}\n")
    _write(tmp_path, "app/build.gradle.kts",
           "plugins {\n"
           "    alias(libs.plugins.android.application)\n"
           "    alias(libs.plugins.kotlin.compose)\n"
           "}\n"
           "android {\n"
           '    namespace = "com.asrecord"\n'
           '    defaultConfig { applicationId = "com.asrecord.app" }\n'
           "}\n")
    _write(tmp_path, "gradle/libs.versions.toml",
           '[versions]\nagp = "8.5.0"\nkotlin = "2.0.21"\n'
           "[plugins]\n"
           'android-application = { id = "com.android.application", version.ref = "agp" }\n'
           'kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }\n'
           'kotlin-compose = { id = "org.jetbrains.kotlin.plugin.compose", version.ref = "kotlin" }\n')
    _write(tmp_path, "app/src/main/java/com/asrecord/ProofManager.kt",
           "package com.asrecord\n\n"
           "import androidx.compose.runtime.Composable\n\n"
           "class ProofManager\n")
    _write(tmp_path, "app/src/main/java/com/asrecord/ProofManagerSha256HexTest.kt",
           "package com.asrecord\n\nclass ProofManagerSha256HexTest\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == ["kotlin"]
    assert "android" in s.frameworks
    assert "jetpack-compose" in s.frameworks
    assert s.project_types == ["mobile"]  # not 'package'
    assert s.naming_style == "PascalCase"  # not 'unknown'


def test_android_not_detected_in_non_jvm_repo(tmp_path: Path):
    """Language gate: a Python repo that happens to ship a build.gradle.kts
    mentioning com.android.application must NOT report android (no kotlin/java
    present), so a stray/vendor gradle file can't cross-contaminate."""
    _write(tmp_path, "app.py", "from flask import Flask\napp = Flask(__name__)\n")
    _write(tmp_path, "build.gradle.kts",
           'plugins { id("com.android.application") }\n')

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "python" in s.languages
    assert "kotlin" not in s.languages
    assert "android" not in s.frameworks


def test_neutral_filenames_reported_unknown(tmp_path: Path):
    """Single-word lowercase filenames (main.go, root.go) are genuinely
    ambiguous — no convention's distinctive feature is present — so the result
    is 'unknown', not a silent 'snake_case' default. (Guards the naming-style
    honesty contract while still scanning .go.)"""
    _write(tmp_path, "go.mod", "module example.com/tool\n")
    _write(tmp_path, "main.go", "package main\n\nfunc main() {}\n")
    _write(tmp_path, "internal/cmd/root.go", "package cmd\n")

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert s.languages == ["go"]
    assert s.naming_style == "unknown"


# ---------------------------------------------------------------------------
# P23-3: bounded gradle reads (per-file 1 MiB / total 8 MiB)
# ---------------------------------------------------------------------------

def test_read_gradle_text_skips_oversized_single_file(tmp_path: Path):
    """A single gradle file larger than the per-file cap is skipped entirely
    (framework detection only needs marker substrings)."""
    _write(tmp_path, "build.gradle.kts", "x\n" * 600_000)  # 1.2 MB > 1 MiB
    s = ProjectAnalyzer(str(tmp_path))
    assert s._read_gradle_text() == ""


def test_read_gradle_text_total_capped(tmp_path: Path):
    """Many gradle files: combined text is capped at 8 MiB instead of being
    cached unbounded for the instance lifetime."""
    for i in range(10):
        _write(tmp_path, f"m{i}.gradle", "x\n" * 500_000)  # ~1 MB each, < 1 MiB
    s = ProjectAnalyzer(str(tmp_path))
    text = s._read_gradle_text()
    # 8 full files (8 MB) + a partial 9th; the +16 covers the "\n".join
    # separators between chunks, which are not counted in the per-file total.
    assert len(text) <= 8 * 1024 * 1024 + 16
    assert "x" in text  # at least the first files were read


def test_source_file_walk_capped(tmp_path: Path, monkeypatch):
    """P28-2: the cached source-file walk stops at _WALK_MAX_FILES instead of
    materializing every path in the repo (unbounded memory on huge trees);
    framework markers live near the tree top, so a prefix walk is enough."""
    for i in range(25):
        _write(tmp_path, f"f{i}.txt", "x")
    s = ProjectAnalyzer(str(tmp_path))
    monkeypatch.setattr(s, "_WALK_MAX_FILES", 10)
    files = s._iter_source_files()
    assert len(files) == 10  # capped prefix, not all 25
    assert all(f.is_file() for f in files)


def test_source_file_walk_uncapped_below_limit(tmp_path):
    """Under the cap the walk yields every file (behavior unchanged)."""
    for i in range(5):
        _write(tmp_path, f"g{i}.txt", "x")
    s = ProjectAnalyzer(str(tmp_path))
    assert len(s._iter_source_files()) == 5


def test_source_walk_skips_vendored_via_walk_policy(tmp_path: Path):
    """F7 follow-up: the ProjectAnalyzer source walk delegates directory
    pruning to walk_policy._walk_should_skip_dir, so venv* prefixes,
    *.egg-info and site-packages dirs are excluded from language/framework
    detection — the former exact-match SKIP_DIRS + startswith('.') check
    missed venv310, dep.egg-info and .tox (vendored trees could out-vote
    the real source language).  NOTE: myvenv (venv as suffix, not prefix)
    is intentionally NOT skipped — walk_policy only prunes venv* prefixes
    (pinned by test_shared_utils)."""
    for d in ("venv310", "pkg.egg-info", ".tox"):
        _write(tmp_path, d + "/v.py", "def v():\n    return 1\n")
    _write(tmp_path, "site-packages/s.py", "def s():\n    return 1\n")
    _write(tmp_path, "real.py", "def real():\n    return 1\n")

    s = ProjectAnalyzer(str(tmp_path))
    files = s._iter_source_files()

    assert [f.name for f in files] == ["real.py"]


def test_analyzer_keeps_target_dir_extra(tmp_path: Path):
    """F7 follow-up: ``target/`` (JVM build output) is an analyzer-specific
    skip that walk_policy does not include — it must stay pruned via the
    private extra set."""
    _write(tmp_path, "src/main/java/App.java", "class App {}\n")
    _write(tmp_path, "target/classes/Gen.java", "class Gen {}\n")

    s = ProjectAnalyzer(str(tmp_path))
    files = list(s._iter_source_files())

    assert [f.name for f in files] == ["App.java"]
