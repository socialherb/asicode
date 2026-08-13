"""CGI incremental admission must equal build()'s (P1 + F1, 2026-08-12).

build() indexes only .py/.pyi files (plus ts/js when MULTILANG_CALLGRAPH is
on, minus *.min.js bundles).  invalidate_files() must route exactly the same
set, or the graph's contents depend on the write history rather than the
tree.  Pre-fix: a touched notes.md with valid-Python content or a data.json
was ast.parsed as Python (P1 — json is often valid Python syntax, so bogus
symbols could be injected), and a touched lib.min.js was tree-sitter-indexed
when the opt-in flag was on (F1) — files a fresh build() never visits.
"""
import textwrap

from external_llm.agent.call_graph import CallGraphIndexer

MD_WITH_PYTHON = textwrap.dedent("""\
    # Notes
    def phantom():
        pass
""")

JSON_VALID_PY = '{"items": [1, 2], "nested": {"ok": true}}'

MIN_JS = "function minifiedBundle(){return 1;}\n"


def _build_repo(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


class TestIncrementalLanguageGate:
    def test_non_python_files_are_not_parsed_on_invalidate(self, tmp_path, monkeypatch):
        _build_repo(tmp_path, {
            "mod.py": "def real():\n    pass\n",
            "notes.md": MD_WITH_PYTHON,
            "data.json": JSON_VALID_PY,
        })
        idx = CallGraphIndexer(str(tmp_path))
        idx.build()
        # record which files the incremental path hands to the Python parser
        parsed: list[str] = []
        orig = idx._index_file
        monkeypatch.setattr(idx, "_index_file", lambda p: (parsed.append(str(p)), orig(p))[1])
        idx.invalidate_files(["mod.py", "notes.md", "data.json"])
        # P1 gate: json/md never reach _index_file; mod.py still re-indexes
        assert parsed == [str(tmp_path / "mod.py")]
        assert "phantom" not in idx._nodes
        assert "notes.md" not in idx._file_nodes
        assert "data.json" not in idx._file_nodes
        assert "real" in idx._nodes

    def test_build_and_incremental_index_the_same_file_set(self, tmp_path, monkeypatch):
        _build_repo(tmp_path, {
            "mod.py": "def real():\n    pass\n",
            "notes.md": MD_WITH_PYTHON,
            "data.json": JSON_VALID_PY,
            "src/app.js": "export function app() {}\n",
            "src/lib.min.js": MIN_JS,
        })
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(tmp_path))
        idx.build()
        build_files = set(idx._file_nodes)
        assert {"mod.py", "src/app.js"} <= build_files  # non-vacuous
        idx.invalidate_files(
            ["mod.py", "notes.md", "data.json", "src/app.js", "src/lib.min.js"]
        )
        assert set(idx._file_nodes) == build_files


class TestIncrementalSuffixGate:
    def test_min_js_not_indexed_on_invalidate(self, tmp_path, monkeypatch):
        _build_repo(tmp_path, {
            "src/app.js": "export function app() {}\n",
            "src/lib.min.js": MIN_JS,
        })
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(tmp_path))
        idx.build()
        assert "src/lib.min.js" not in idx._file_nodes
        idx.invalidate_files(["src/lib.min.js"])
        # F1: the incremental path must not admit what build()'s walk drops
        assert "src/lib.min.js" not in idx._file_nodes
        assert "minifiedBundle" not in idx._nodes
        assert "app" in idx._nodes
