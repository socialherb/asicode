"""Unit tests for broken_contract_scanner.

Covers the pure AST state-access helpers, the pairing/shared-state logic, and
the end-to-end ``scan_broken_contracts`` against an in-memory temp file with a
fake repo graph (caller-asymmetry is what distinguishes a broken contract from
a healthy writer/reader pair).
"""
from __future__ import annotations

import ast
import os
import tempfile

import pytest

from external_llm.analysis.broken_contract_scanner import (
    _classify_dynamic_attr_call,
    _classify_mutator_call,
    _classify_open_call,
    _dotted_target,
    _literal_path,
    _literal_str,
    _loc_overlaps,
    _member_caller_count,
    _member_summary,
    _role_of,
    _shared_state,
    _state_accesses,
    _strip_access_verb,
    scan_broken_contracts,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _expr(src: str) -> ast.expr:
    """Parse a single expression and return its AST node."""
    return ast.parse(src, mode="eval").body


def _stmts(src: str) -> list:
    """Parse statements (a function body) and return the stmt list."""
    return ast.parse(src).body


def _first_call(src: str) -> ast.Call:
    """Parse an expression statement and return its top-level Call node."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            return node
    raise AssertionError("no Call node found")


class _FakeGraph:
    """Minimal repo-graph stub: returns canned caller lists per name."""

    def __init__(self, callers: dict):
        self._callers = callers

    def get_callers(self, name):
        return self._callers.get(name, [])

    def get_symbols_in_file(self, file_path):
        return []


# ── _strip_access_verb ──────────────────────────────────────────────────────

class TestStripAccessVerb:
    def test_set_prefix(self):
        assert _strip_access_verb("set_pending_impl_spec") == "pending_impl_spec"

    def test_mark_prefix(self):
        assert _strip_access_verb("mark_f821_protected") == "f821_protected"

    def test_multiword_verb(self):
        # reset_ is a registered verb; the core must itself contain an underscore.
        assert _strip_access_verb("reset_all_stats") == "all_stats"

    def test_single_token_core_rejected(self):
        # get_x → core "x" has no underscore → too generic → "".
        assert _strip_access_verb("get_x") == ""

    def test_no_verb_returns_empty(self):
        assert _strip_access_verb("compute") == ""

    def test_no_underscore_returns_empty(self):
        assert _strip_access_verb("get") == ""

    def test_empty(self):
        assert _strip_access_verb("") == ""


# ── _dotted_target ──────────────────────────────────────────────────────────

class TestDottedTarget:
    def test_attribute_chain(self):
        assert _dotted_target(_expr("self.a.b")) == "self.a.b"

    def test_subscript_drops_key(self):
        assert _dotted_target(_expr("self._d['k']")) == "self._d"

    def test_bare_name(self):
        assert _dotted_target(_expr("counter")) == "counter"

    def test_constant_returns_none(self):
        assert _dotted_target(_expr("42")) is None


# ── _literal_str / _literal_path ────────────────────────────────────────────

class TestLiteralAccessors:
    def test_literal_str_string(self):
        assert _literal_str(_expr("'hello'")) == "hello"

    def test_literal_str_non_string(self):
        assert _literal_str(_expr("42")) == ""

    def test_literal_path_string(self):
        assert _literal_path(_expr("'/tmp/f'")) == "/tmp/f"

    def test_literal_path_subscript_returns_none(self):
        assert _literal_path(_expr("paths['x']")) is None

    def test_literal_path_attribute(self):
        assert _literal_path(_expr("self._path")) == "self._path"


# ── _classify_*_call ────────────────────────────────────────────────────────

class TestClassifyMutatorCall:
    def test_append_writes_base(self):
        w = set()
        _classify_mutator_call(_first_call("self._buf.append(1)"), w)
        assert w == {"self._buf"}

    def test_non_mutator_no_write(self):
        w = set()
        _classify_mutator_call(_first_call("self._buf.len()"), w)
        assert w == set()


class TestClassifyDynamicAttrCall:
    def test_getattr_reads(self):
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("getattr(self, '_y', None)"), w, r)
        assert r == {"self._y"} and not w

    def test_hasattr_reads(self):
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("hasattr(self, '_h')"), w, r)
        assert r == {"self._h"}

    def test_setattr_writes(self):
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("setattr(self, '_z', 1)"), w, r)
        assert w == {"self._z"} and not r

    def test_delattr_writes(self):
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("delattr(self, '_d')"), w, r)
        assert w == {"self._d"}

    def test_non_literal_name_dropped(self):
        # A dynamic attribute name (not a literal string) must not be paired.
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("getattr(self, name)"), w, r)
        assert not w and not r

    def test_not_a_dynamic_accessor(self):
        w, r = set(), set()
        _classify_dynamic_attr_call(_first_call("print('x')"), w, r)
        assert not w and not r


class TestClassifyOpenCall:
    def test_write_mode(self):
        w, r = set(), set()
        _classify_open_call(_first_call("open('/tmp/f', 'w')"), w, r)
        assert w == {"open(/tmp/f)"} and not r

    def test_append_mode_is_write(self):
        w, r = set(), set()
        _classify_open_call(_first_call("open('/tmp/f', 'a')"), w, r)
        assert w == {"open(/tmp/f)"}

    def test_read_mode(self):
        w, r = set(), set()
        _classify_open_call(_first_call("open('/tmp/f', 'r')"), w, r)
        assert r == {"open(/tmp/f)"} and not w

    def test_no_mode_defaults_read(self):
        w, r = set(), set()
        _classify_open_call(_first_call("open('/tmp/f')"), w, r)
        assert r == {"open(/tmp/f)"}

    def test_not_open(self):
        w, r = set(), set()
        _classify_open_call(_first_call("close()"), w, r)
        assert not w and not r


# ── _state_accesses ─────────────────────────────────────────────────────────

class TestStateAccesses:
    def _accesses(self, src):
        return _state_accesses(_stmts(src))

    def test_assignment_write(self):
        w, _r = self._accesses("self._x = 1\n")
        assert "self._x" in w

    def test_augassign_write(self):
        w, _r = self._accesses("self._c += 1\n")
        assert "self._c" in w

    def test_mutator_call_write(self):
        w, _r = self._accesses("self._buf.append(1)\n")
        assert "self._buf" in w

    def test_getattr_read(self):
        _w, r = self._accesses("return getattr(self, '_y', None)\n")
        assert "self._y" in r

    def test_open_write(self):
        w, _r = self._accesses("open('/tmp/f', 'w')\n")
        assert "open(/tmp/f)" in w

    def test_attribute_read_not_double_counted_as_write(self):
        # A plain read (return self._r) must land in reads, not writes.
        w, r = self._accesses("return self._r\n")
        assert "self._r" in r and "self._r" not in w


# ── _shared_state / _loc_overlaps ───────────────────────────────────────────

class TestLocOverlaps:
    def test_equal(self):
        assert _loc_overlaps("x", {"x"}) is True

    def test_loc_prefix_of_other(self):
        assert _loc_overlaps("self._cache", {"self._cache.get"}) is True

    def test_other_prefix_of_loc(self):
        assert _loc_overlaps("self._cache.get", {"self._cache"}) is True

    def test_disjoint(self):
        assert _loc_overlaps("x", {"y"}) is False

    def test_empty(self):
        assert _loc_overlaps("x", set()) is False


class TestSharedState:
    def test_write_read_overlap(self):
        shared = _shared_state({"self._x"}, set(), set(), {"self._x"})
        assert shared == {"self._x"}

    def test_base_prefix_overlap_cache_pattern(self):
        # The mark_f821_protected regression: write _cache[..], read _cache.get(..).
        shared = _shared_state({"self._cache"}, set(), set(), {"self._cache.get"})
        assert shared == {"self._cache"}

    def test_two_readers_no_contract(self):
        shared = _shared_state(set(), {"x"}, set(), {"x"})
        assert shared == set()

    def test_both_writers_overlap(self):
        shared = _shared_state({"x"}, set(), {"x"}, set())
        assert shared == {"x"}


# ── _role_of / _member_summary ──────────────────────────────────────────────

class TestRoleOf:
    def test_writer(self):
        assert _role_of({"writes": {"x"}}) == "writer"

    def test_reader(self):
        assert _role_of({"writes": set()}) == "reader"


class TestMemberSummary:
    def test_summary_shape(self):
        m = {"name": "foo", "lineno": 1, "end_lineno": 3,
             "writes": {"x"}, "reads": set(), "caller_count": 2}
        assert _member_summary(m) == {
            "name": "foo", "lineno": 1, "end_lineno": 3,
            "role": "writer", "caller_count": 2,
        }


# ── scan_broken_contracts (end-to-end) ──────────────────────────────────────

CONTRACT_SRC = """\
class C:
    def set_pending_impl_spec(self, v):
        self._pending_impl_spec = v

    def pop_pending_impl_spec(self):
        return getattr(self, '_pending_impl_spec', None)
"""


def _scan(src, callers, fname="c.py"):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, fname)
        with open(p, "w") as f:
            f.write(src)
        graph = _FakeGraph(callers) if callers is not None else None
        return scan_broken_contracts(repo_root=d, file_paths=[p], repo_graph=graph)


class TestScanBrokenContracts:
    def test_orphan_reader_detected(self):
        # set_ has callers (live), pop_ has none (orphan) → broken contract.
        cands = _scan(CONTRACT_SRC, {
            "set_pending_impl_spec": ["main"],
            "pop_pending_impl_spec": [],
        })
        assert len(cands) == 1
        c = cands[0]
        assert c.core_name == "pending_impl_spec"
        assert c.orphan_role == "reader"
        assert c.orphan_name == "pop_pending_impl_spec"
        assert "self._pending_impl_spec" in c.shared_state

    def test_orphan_writer_detected(self):
        # Inverted: pop_ live, set_ orphan.
        cands = _scan(CONTRACT_SRC, {
            "set_pending_impl_spec": [],
            "pop_pending_impl_spec": ["main"],
        })
        assert len(cands) == 1
        assert cands[0].orphan_role == "writer"
        assert cands[0].orphan_name == "set_pending_impl_spec"

    def test_both_live_no_candidate(self):
        cands = _scan(CONTRACT_SRC, {
            "set_pending_impl_spec": ["a"], "pop_pending_impl_spec": ["b"],
        })
        assert cands == []

    def test_both_dead_defers_to_public_dead_code(self):
        cands = _scan(CONTRACT_SRC, {
            "set_pending_impl_spec": [], "pop_pending_impl_spec": [],
        })
        assert cands == []

    def test_no_graph_returns_empty(self):
        assert _scan(CONTRACT_SRC, None) == []

    def test_graph_missing_api_returns_empty(self):
        # A graph without get_callers / get_symbols_in_file is skipped.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.py")
            with open(p, "w") as f:
                f.write(CONTRACT_SRC)
            cands = scan_broken_contracts(repo_root=d, file_paths=[p], repo_graph=object())
        assert cands == []

    def test_non_python_file_skipped(self):
        # .go / .js files are out of scope for this Python-AST scanner.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.go")
            with open(p, "w") as f:
                f.write("package main\n")
            cands = scan_broken_contracts(
                repo_root=d, file_paths=[p],
                repo_graph=_FakeGraph({}),
            )
        assert cands == []

    def test_no_shared_state_no_candidate(self):
        # Two methods with the same core name but disjoint state → no contract.
        src = (
            "def set_alpha_beta(self):\n    self._alpha = 1\n"
            "def pop_alpha_beta(self):\n    return self._beta\n"
        )
        cands = _scan(src, {"set_alpha_beta": ["x"], "pop_alpha_beta": []})
        assert cands == []

    def test_candidate_to_dict_roundtrip(self):
        cands = _scan(CONTRACT_SRC, {
            "set_pending_impl_spec": ["main"], "pop_pending_impl_spec": [],
        })
        d = cands[0].to_dict()
        assert d["core_name"] == "pending_impl_spec"
        assert d["orphan_role"] == "reader"
        assert len(d["members"]) == 2
        assert isinstance(d["shared_state"], list)


# ── _member_caller_count: suffix-fallback + fail-fast contract ────────────────

class _Sym:
    def __init__(self, name):
        self.name = name


class _SuffixGraph(_FakeGraph):
    """get_callers misses the bare name; get_symbols_in_file offers the qualified one."""

    def __init__(self, callers: dict, syms: list):
        super().__init__(callers)
        self._syms = syms

    def get_symbols_in_file(self, file_path):
        return self._syms


def test_member_caller_count_suffix_fallback():
    # Bare name misses the index; the file-scoped qualified name hits.
    graph = _SuffixGraph({"Klass.method": ["a.py"]}, [_Sym("Klass.method")])
    assert _member_caller_count(graph, "method", "mod.py") == 1


def test_member_caller_count_returns_zero_when_no_callers():
    graph = _FakeGraph({})
    assert _member_caller_count(graph, "method", "mod.py") == 0


def test_member_caller_count_fails_fast_on_graph_error():
    # B3 contract: a graph-access bug must propagate loudly instead of being
    # swallowed into a silent ``0`` — a 0 would be read as "no callers" and
    # report a false orphan in the scan output.
    class _RaisingGraph:
        def get_callers(self, name):
            return []

        def get_symbols_in_file(self, file_path):
            raise RuntimeError("graph index bug")

    with pytest.raises(RuntimeError, match="graph index bug"):
        _member_caller_count(_RaisingGraph(), "method", "mod.py")
