"""Tests for multi-language call graph indexing (TS/JS via TSSemanticTracer)."""
import textwrap

import pytest

from external_llm.agent.call_graph import CallGraphIndexer

TS_CODE_A = textwrap.dedent("""\
    import { validate } from './utils';

    export function createUser(name: string, email: string) {
      if (!validate(email)) {
        throw new Error('Invalid email');
      }
      return { name, email };
    }

    export function deleteUser(id: string) {
      console.log('Deleting user', id);
    }
""")

TS_CODE_B = textwrap.dedent("""\
    export function validate(email: string): boolean {
      return email.includes('@');
    }

    export function sanitize(input: string): string {
      return input.trim();
    }
""")


@pytest.fixture
def ts_repo(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "users.ts").write_text(TS_CODE_A, encoding="utf-8")
    (src / "utils.ts").write_text(TS_CODE_B, encoding="utf-8")
    return tmp_path


class TestCallGraphTS:
    def test_build_indexes_ts_nodes(self, ts_repo, monkeypatch):
        """When MULTILANG_CALLGRAPH=True, TS functions are indexed."""
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        assert "createUser" in idx._nodes
        assert "deleteUser" in idx._nodes
        assert "validate" in idx._nodes
        assert "sanitize" in idx._nodes

    def test_build_indexes_call_edges(self, ts_repo, monkeypatch):
        """Call edges from TSModule.call_sites are registered."""
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        callees = idx.get_callees("createUser")
        callee_names = [e.callee_symbol for e in callees]
        assert "validate" in callee_names

    def test_callers_reverse_index(self, ts_repo, monkeypatch):
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        callers = idx.get_callers("validate")
        caller_names = [e.caller_symbol for e in callers]
        assert "createUser" in caller_names

    def test_enabled_by_default(self, ts_repo):
        """MULTILANG_CALLGRAPH=True by default — TS files are indexed."""
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        # TS files indexed by default (no monkeypatch needed)
        assert "createUser" in idx._nodes

    def test_mixed_python_ts(self, ts_repo, monkeypatch):
        """Python and TS symbols coexist in the same graph."""
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        # Add a Python file
        (ts_repo / "helper.py").write_text(
            "def format_name(name):\n    return name.strip()\n",
            encoding="utf-8",
        )
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        assert "createUser" in idx._nodes  # TS
        assert "format_name" in idx._nodes  # Python

    def test_get_related_symbols(self, ts_repo, monkeypatch):
        monkeypatch.setattr("config.MULTILANG_CALLGRAPH", True)
        idx = CallGraphIndexer(str(ts_repo))
        idx.build()
        result = idx.get_related_symbols("createUser")
        assert "callees" in result
        callee_names = [e["symbol"] for e in result["callees"]]
        assert "validate" in callee_names
