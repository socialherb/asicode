"""Unit tests for container_reachability_scanner.

Covers the pure AST helpers (name privacy, key classification, dict-literal
collection, read-site collection, constant-domain inference, reachability
grading) and the end-to-end ``scan_container_reachability`` against in-memory
temp files.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile

import pytest

from external_llm.analysis.container_reachability_scanner import (
    ContainerReachabilityCandidate,
    ContainerReadSite,
    _binds_name,
    _build_lineno_to_method,
    _classify_key_expr,
    _collect_constant_returns,
    _collect_dict_literals,
    _collect_read_sites,
    _compute_reachability,
    _crx_cache_path,
    _domain_of_rhs,
    _extract_string_keys,
    _find_class_node,
    _find_method_in_class,
    _infer_key_domain_for_var,
    _is_container_ref,
    _is_private_name,
    _reachability_reason,
    _resolve_class_constant,
    scan_container_reachability,
)

# ── helpers ─────────────────────────────────────────────────────────────────

def _mod(src: str) -> ast.Module:
    return ast.parse(src)


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def _first_assign_dict(src: str) -> ast.Dict:
    for node in ast.walk(_mod(src)):
        if isinstance(node, ast.Dict):
            return node
    raise AssertionError("no Dict node")


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no func {name}")


def _cand(**overrides) -> ContainerReachabilityCandidate:
    base = {
        "file": "f.py",
        "container_symbol": "_D",
        "qualified_name": "_D",
        "enclosing_class": None,
        "container_kind": "dict_literal",
        "lineno": 1,
        "end_lineno": 1,
        "all_keys": [],
        "keys_unreachable": [],
        "keys_possibly_unreachable": [],
        "keys_reachable": [],
        "read_sites": [],
        "key_domain": None,
        "confidence": 0.9,
        "evidence": [],
    }
    base.update(overrides)
    return ContainerReachabilityCandidate(**base)


def _scan(src: str, **kw) -> list:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.py")
        with open(p, "w") as f:
            f.write(src)
        return scan_container_reachability(repo_root=d, file_paths=[p], **kw)


# ── _is_private_name ────────────────────────────────────────────────────────

class TestIsPrivateName:
    @pytest.mark.parametrize("name, expected", [
        ("_foo", True), ("_", True), ("_D", True),
        ("__foo__", False), ("__init__", False),
        ("foo", False), ("Bar", False),
    ])
    def test_privacy(self, name, expected):
        assert _is_private_name(name) is expected


# ── _extract_string_keys ────────────────────────────────────────────────────

class TestExtractStringKeys:
    def test_all_strings(self):
        assert _extract_string_keys(_first_assign_dict('_D = {"a": 1, "b": 2}')) == ["a", "b"]

    def test_non_string_key_returns_none(self):
        assert _extract_string_keys(_first_assign_dict('_D = {"a": 1, 2: 3}')) is None

    def test_unpacking_returns_none(self):
        assert _extract_string_keys(_first_assign_dict("_D = {**other}")) is None

    def test_empty_dict(self):
        assert _extract_string_keys(_first_assign_dict("_D = {}")) == []


# ── _classify_key_expr ──────────────────────────────────────────────────────

class TestClassifyKeyExpr:
    def test_literal(self):
        assert _classify_key_expr(_expr('"foo"')) == ("literal", "'foo'")

    def test_name(self):
        assert _classify_key_expr(_expr("intent")) == ("name", "intent")

    def test_self_attr_const(self):
        assert _classify_key_expr(_expr("self.TYPE_A")) == ("attr_const", "TYPE_A")

    def test_cls_attr_const(self):
        assert _classify_key_expr(_expr("cls.TYPE_A")) == ("attr_const", "TYPE_A")

    def test_call(self):
        assert _classify_key_expr(_expr("f()")) == ("call", "f()")

    def test_other(self):
        assert _classify_key_expr(_expr("x + 1")) == ("other", "x + 1")


# ── _is_container_ref ───────────────────────────────────────────────────────

class TestIsContainerRef:
    def test_bare_name(self):
        assert _is_container_ref(_expr("_FOO"), "_FOO", None) is True

    def test_self_attr(self):
        assert _is_container_ref(_expr("self._FOO"), "_FOO", None) is True

    def test_cls_attr(self):
        assert _is_container_ref(_expr("cls._FOO"), "_FOO", None) is True

    def test_class_qualified_attr(self):
        assert _is_container_ref(_expr("MyClass._FOO"), "_FOO", "MyClass") is True

    def test_wrong_class_attr(self):
        assert _is_container_ref(_expr("Other._FOO"), "_FOO", "MyClass") is False

    def test_unrelated_name(self):
        assert _is_container_ref(_expr("_BAR"), "_FOO", None) is False

    def test_unrelated_self_attr(self):
        assert _is_container_ref(_expr("self._BAR"), "_FOO", None) is False


# ── _build_lineno_to_method ─────────────────────────────────────────────────

class TestBuildLinenoToMethod:
    def test_qualified_names(self):
        tree = _mod(
            "class C:\n"
            "    def foo(self):\n"
            "        return 1\n"
            "    async def bar(self):\n"
            "        return 2\n"
        )
        m = _build_lineno_to_method(tree)
        assert m[3] == "C.foo"      # foo body line
        assert m[5] == "C.bar"      # bar body line


# ── _collect_dict_literals ──────────────────────────────────────────────────

class TestCollectDictLiterals:
    def test_private_module_dict(self):
        r = _collect_dict_literals(_mod('_D = {"a": 1}'))
        assert r == [("_D", None, 1, 1, ["a"])]

    def test_non_private_excluded_without_graph(self):
        assert _collect_dict_literals(_mod('D = {"a": 1}')) == []

    def test_non_private_included_with_graph_when_not_referenced(self):
        r = _collect_dict_literals(_mod('D = {"a": 1}'), cross_file_referenced_names=set())
        assert r and r[0][0] == "D"

    def test_non_private_excluded_when_cross_referenced(self):
        r = _collect_dict_literals(_mod('D = {"a": 1}'), cross_file_referenced_names={"D"})
        assert r == []

    def test_private_excluded_when_cross_referenced(self):
        r = _collect_dict_literals(_mod('_D = {"a": 1}'), cross_file_referenced_names={"_D"})
        assert r == []

    def test_annassign_dict(self):
        r = _collect_dict_literals(_mod('_D: dict = {"a": 1}'))
        assert r and r[0][0] == "_D"

    def test_class_level_dict_carries_enclosing_class(self):
        r = _collect_dict_literals(_mod("class C:\n    _D = {'a': 1}\n"))
        assert r and r[0][0] == "_D" and r[0][1] == "C"

    def test_unpacking_skipped(self):
        assert _collect_dict_literals(_mod("_D = {**other}")) == []

    def test_non_dict_skipped(self):
        assert _collect_dict_literals(_mod("_D = [1, 2]")) == []


# ── _collect_read_sites ─────────────────────────────────────────────────────

class TestCollectReadSites:
    def test_get_literal(self):
        tree = _mod('_D = {"a": 1}\nx = _D.get("a", None)\n')
        sites, dyn = _collect_read_sites(tree, "_D", {}, None)
        assert dyn is False
        assert len(sites) == 1
        assert sites[0].access_kind == "get"
        assert sites[0].key_expr_kind == "literal"

    def test_subscript(self):
        tree = _mod('_D = {"a": 1}\nx = _D["a"]\n')
        sites, dyn = _collect_read_sites(tree, "_D", {}, None)
        assert dyn is False
        assert sites[0].access_kind == "subscript"

    def test_get_name_key(self):
        tree = _mod('def f(intent):\n    return _D.get(intent)\n_D = {"a": 1}\n')
        sites, _ = _collect_read_sites(tree, "_D", _build_lineno_to_method(tree), None)
        assert sites[0].key_expr_kind == "name"
        assert sites[0].key_expr_text == "intent"

    def test_iteration_is_dynamic(self):
        tree = _mod('_D = {"a": 1}\nfor k in _D:\n    pass\n')
        _sites, dyn = _collect_read_sites(tree, "_D", {}, None)
        assert dyn is True

    def test_keys_call_is_dynamic(self):
        tree = _mod('_D = {"a": 1}\nx = list(_D.keys())\n')
        _, dyn = _collect_read_sites(tree, "_D", {}, None)
        assert dyn is True


# ── _collect_constant_returns ───────────────────────────────────────────────

class TestCollectConstantReturns:
    def test_all_constants(self):
        tree = _mod('def f():\n    if c:\n        return "a"\n    return "b"\n')
        assert _collect_constant_returns(_func(tree, "f")) == {"a", "b"}

    def test_non_constant_returns_none(self):
        tree = _mod('def f():\n    return x\n')
        assert _collect_constant_returns(_func(tree, "f")) is None

    def test_bare_return_returns_none(self):
        tree = _mod('def f():\n    return\n')
        assert _collect_constant_returns(_func(tree, "f")) is None

    def test_no_return_returns_none(self):
        tree = _mod('def f():\n    pass\n')
        assert _collect_constant_returns(_func(tree, "f")) is None


# ── _find_method_in_class / _resolve_class_constant / _find_class_node ──────

class TestClassResolution:
    def test_find_method_in_class(self):
        tree = _mod("class C:\n    def foo(self):\n        pass\n    def bar(self):\n        pass\n")
        cls = _find_class_node(tree, "C")
        assert _find_method_in_class(cls, "bar").name == "bar"
        assert _find_method_in_class(cls, "missing") is None

    def test_resolve_class_constant_assign(self):
        tree = _mod('class C:\n    TYPE_A = "type_a"\n')
        assert _resolve_class_constant(_find_class_node(tree, "C"), "TYPE_A") == "type_a"

    def test_resolve_class_constant_annassign(self):
        tree = _mod('class C:\n    TYPE_A: str = "type_a"\n')
        assert _resolve_class_constant(_find_class_node(tree, "C"), "TYPE_A") == "type_a"

    def test_resolve_class_constant_missing(self):
        tree = _mod('class C:\n    pass\n')
        assert _resolve_class_constant(_find_class_node(tree, "C"), "TYPE_A") is None

    def test_resolve_class_constant_non_string(self):
        tree = _mod("class C:\n    N = 5\n")
        assert _resolve_class_constant(_find_class_node(tree, "C"), "N") is None

    def test_find_class_node_missing(self):
        assert _find_class_node(_mod("x = 1\n"), "Nope") is None


# ── _binds_name ─────────────────────────────────────────────────────────────

class TestBindsName:
    def test_simple_name(self):
        assert _binds_name(_expr("x"), "x") is True

    def test_tuple_unpack(self):
        assert _binds_name(_expr("(a, b)"), "b") is True

    def test_starred(self):
        assert _binds_name(_expr("a"), "x") is False
        # Starred targets appear inside tuple unpacking.
        assert _binds_name(_expr("(a, *xs)"), "xs") is True


# ── _domain_of_rhs ──────────────────────────────────────────────────────────

class TestDomainOfRhs:
    def test_constant(self):
        assert _domain_of_rhs(_expr('"a"'), _mod("x=1\n"), None) == {"a"}

    def test_self_helper_traced(self):
        tree = _mod(
            'class C:\n'
            '    def _kind(self):\n'
            '        return "general"\n'
            '    def run(self):\n'
            '        k = self._kind()\n'
        )
        helper_call = _func(tree, "run").body[0].value
        assert _domain_of_rhs(helper_call, tree, "C") == {"general"}
    def test_self_helper_non_constant(self):
        tree = _mod(
            'class C:\n'
            '    def _kind(self):\n'
            '        return self.x\n'
            '    def run(self):\n'
            '        k = self._kind()\n'
        )
        helper_call = _func(tree, "run").body[0].value
        assert _domain_of_rhs(helper_call, tree, "C") is None


# ── _infer_key_domain_for_var ───────────────────────────────────────────────

class TestInferKeyDomain:
    def test_union_of_branches(self):
        tree = _mod(
            'def f(cond):\n'
            '    if cond:\n'
            '        intent = "a"\n'
            '    else:\n'
            '        intent = "b"\n'
            '    return _D.get(intent)\n'
        )
        fn = _func(tree, "f")
        # target_lineno = the get() line (6); domain is the union of every
        # reaching assignment with lineno <= target.
        assert _infer_key_domain_for_var(tree, fn, "intent", None, 6) == {"a", "b"}

    def test_param_returns_none(self):
        tree = _mod('def f(intent):\n    return _D.get(intent)\n')
        assert _infer_key_domain_for_var(tree, _func(tree, "f"), "intent", None, 2) is None

    def test_non_constant_returns_none(self):
        tree = _mod('def f():\n    intent = g()\n    return _D.get(intent)\n')
        assert _infer_key_domain_for_var(tree, _func(tree, "f"), "intent", None, 2) is None

    def test_tuple_unpack_returns_none(self):
        tree = _mod('def f():\n    a, intent = 1, "x"\n    return _D.get(intent)\n')
        assert _infer_key_domain_for_var(tree, _func(tree, "f"), "intent", None, 2) is None

    def test_augassign_returns_none(self):
        tree = _mod('def f():\n    intent = "a"\n    intent += "b"\n    return _D.get(intent)\n')
        assert _infer_key_domain_for_var(tree, _func(tree, "f"), "intent", None, 3) is None

    def test_no_assignment_returns_none(self):
        tree = _mod('def f():\n    return _D.get(intent)\n')
        assert _infer_key_domain_for_var(tree, _func(tree, "f"), "intent", None, 2) is None


# ── _reachability_reason ────────────────────────────────────────────────────

class TestReachabilityReason:
    def test_structurally_unreachable(self):
        c = _cand(all_keys=["a", "b"], keys_unreachable=["a"])
        assert _reachability_reason(c) == "1 structurally-unreachable key(s) of 2: 'a'"

    def test_more_than_three(self):
        c = _cand(all_keys=["a", "b", "c", "d"], keys_unreachable=["a", "b", "c", "d"])
        assert "(+1 more)" in _reachability_reason(c)

    def test_possibly_unreachable(self):
        c = _cand(all_keys=["a", "b"], keys_possibly_unreachable=["a", "b"])
        assert "2 possibly-unreachable key(s)" in _reachability_reason(c)

    def test_all_reachable(self):
        c = _cand(all_keys=["a"], keys_reachable=["a"])
        assert _reachability_reason(c) == "all 1 key(s) reachable"

    def test_no_keys(self):
        assert _reachability_reason(_cand()) == "no keys"


# ── _compute_reachability ───────────────────────────────────────────────────

class TestComputeReachability:
    def test_literal_key_marks_others_unreachable(self):
        sites = [ContainerReadSite("get", "literal", "'used'", "", 2)]
        u, _p, r, dom, _ev = _compute_reachability(["used", "unused"], sites, _mod("x=1"), None, {})
        assert u == ["unused"]
        assert r == ["used"]
        assert dom == ["used"]

    def test_name_key_full_domain(self):
        tree = _mod(
            'def f():\n    intent = "general"\n    return _D.get(intent)\n'
        )
        l2m = _build_lineno_to_method(tree)
        method_nodes = {l2m[n.lineno]: n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)}
        sites = [ContainerReadSite("get", "name", "intent", "f", 2)]
        u, _p, _r, dom, _ev = _compute_reachability(
            ["general", "removed"], sites, tree, None, method_nodes)
        assert u == ["removed"]
        assert dom == ["general"]

    def test_name_key_unknown_method_suppresses(self):
        # read site references a method not in method_nodes → unknown domain →
        # per-key possibly_unreachable verdicts are suppressed to reachable.
        sites = [ContainerReadSite("get", "name", "intent", "ghost", 2)]
        u, _p, r, _dom, _ev = _compute_reachability(
            ["a", "b"], sites, _mod("x=1"), None,
        {})
        assert u == []
        assert r == ["a", "b"]


# ── scan_container_reachability (end-to-end) ────────────────────────────────

class TestScanContainerReachability:
    def test_unreachable_key_detected(self):
        src = (
            '_D = {"used": 1, "unused": 2}\n'
            'def f():\n'
            '    return _D.get("used")\n'
        )
        cands = _scan(src)
        assert len(cands) == 1
        c = cands[0]
        assert c.container_symbol == "_D"
        assert c.keys_unreachable == ["unused"]
        assert c.keys_reachable == ["used"]
        assert c.confidence == 0.90

    def test_all_reachable_emits_nothing_by_default(self):
        src = (
            '_D = {"a": 1}\n'
            'def f():\n'
            '    return _D.get("a")\n'
        )
        assert _scan(src) == []

    def test_all_reachable_emitted_with_min_zero(self):
        src = (
            '_D = {"a": 1}\n'
            'def f():\n'
            '    return _D.get("a")\n'
        )
        cands = _scan(src, min_unreachable_keys=0)
        assert len(cands) == 1
        assert cands[0].keys_reachable == ["a"]
        assert cands[0].confidence == 0.30

    def test_dynamic_use_skipped(self):
        # Iteration over the dict → every key may be consumed → no candidate.
        src = (
            '_D = {"a": 1, "b": 2}\n'
            'def f():\n'
            '    for k in _D:\n'
            '        pass\n'
        )
        assert _scan(src) == []

    def test_non_private_skipped_without_graph(self):
        src = (
            'D = {"a": 1, "b": 2}\n'
            'def f():\n'
            '    return D.get("a")\n'
        )
        # No cross_file_referenced_names → conservative private-only → D excluded.
        assert _scan(src) == []

    def test_non_private_scanned_with_empty_cross_file_set(self):
        src = (
            'D = {"a": 1, "removed": 2}\n'
            'def f():\n'
            '    return D.get("a")\n'
        )
        cands = _scan(src, cross_file_referenced_names=set())
        assert len(cands) == 1
        assert cands[0].keys_unreachable == ["removed"]

    def test_to_dict_roundtrip(self):
        src = (
            '_D = {"used": 1, "unused": 2}\n'
            'def f():\n'
            '    return _D.get("used")\n'
        )
        d = _scan(src)[0].to_dict()
        assert d["container_symbol"] == "_D"
        assert d["keys_unreachable"] == ["unused"]
        assert "reason" in d
        assert isinstance(d["read_sites"], list)

    def test_empty_file(self):
        assert _scan("# nothing\n") == []


# ── per-file extraction cache (2026-08-16, commit 786ffcdc pattern) ────────

class TestScanCache:
    """(mtime_ns, size)-fingerprint disk cache: hit/miss parity, invalidation,
    fail-open, and cross-file-set independence of the cached superset."""

    def test_cold_equals_hot(self):
        src = (
            '_PRIVATE = {"alpha": 1, "beta": 2}\n'
            'def f():\n'
            '    return _PRIVATE["alpha"]\n'
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as fh:
                fh.write(src)
            first = scan_container_reachability(repo_root=d, file_paths=[p])
            second = scan_container_reachability(repo_root=d, file_paths=[p])  # cache hit
            assert first and second
            assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
            assert first[0].keys_unreachable == ["beta"]

    def test_edit_invalidates_entry(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as fh:
                fh.write('_D = {"a": 1, "b": 2}\n\ndef f():\n    return _D["a"]\n')
            first = scan_container_reachability(repo_root=d, file_paths=[p])
            assert [c.keys_unreachable for c in first] == [["b"]]
            # different size → fingerprint mismatch → re-extraction
            with open(p, "w") as fh:
                fh.write('_D = {"a": 1, "b": 2, "c": 3}\n\ndef f():\n    return _D["a"]\n')
            second = scan_container_reachability(repo_root=d, file_paths=[p])
            assert [c.keys_unreachable for c in second] == [["b", "c"]]

    def test_corrupted_cache_fails_open(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as fh:
                fh.write('_D = {"a": 1, "b": 2}\n\ndef f():\n    return _D["a"]\n')
            os.makedirs(os.path.join(d, ".cache"), exist_ok=True)
            with open(_crx_cache_path(d), "w") as fh:
                fh.write("{definitely not json")
            out = scan_container_reachability(repo_root=d, file_paths=[p])
            assert len(out) == 1
            assert out[0].keys_unreachable == ["b"]

    def test_version_mismatch_discards_cache(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as fh:
                fh.write('_D = {"a": 1, "b": 2}\n\ndef f():\n    return _D["a"]\n')
            scan_container_reachability(repo_root=d, file_paths=[p])
            # rewrite with a future format version → must be discarded
            with open(_crx_cache_path(d), "w") as fh:
                json.dump({"format": 999, "files": {}}, fh)
            out = scan_container_reachability(repo_root=d, file_paths=[p])
            assert len(out) == 1

    def test_empty_repo_root_bypasses_cache(self, tmp_path, monkeypatch):
        # repo_root="" (unit-test convention) must not write .cache in CWD
        monkeypatch.chdir(tmp_path)
        (tmp_path / "m.py").write_text(
            '_D = {"a": 1, "b": 2}\n\ndef f():\n    return _D["a"]\n'
        )
        out = scan_container_reachability(repo_root="", file_paths=["m.py"])
        assert len(out) == 1
        assert not (tmp_path / ".cache").exists()

    def test_cache_serves_different_cross_sets(self):
        # The cached payload is a cross-file-set-independent superset; switching
        # the graph set between runs (all cache hits after the first) must not
        # change per-run semantics vs. the unfiltered original.
        src = (
            '_PRIVATE = {"a": 1, "x": 2}\n'
            'PUBLIC = {"b": 1, "y": 2}\n'
            'def f():\n'
            '    return _PRIVATE["a"] + PUBLIC["b"]\n'
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            with open(p, "w") as fh:
                fh.write(src)
            # warm the cache with graph mode; neither name referenced → both allowed
            r1 = scan_container_reachability(
                repo_root=d, file_paths=[p], cross_file_referenced_names={"OTHER"})
            assert len(r1) == 2
            # both cross-referenced → nothing allowed (cache hit)
            r2 = scan_container_reachability(
                repo_root=d, file_paths=[p],
                cross_file_referenced_names={"PUBLIC", "_PRIVATE"})
            assert r2 == []
            # no graph data → private only (cache hit)
            r3 = scan_container_reachability(repo_root=d, file_paths=[p])
            assert [c.container_symbol for c in r3] == ["_PRIVATE"]
            # empty set (graph mode, nothing referenced) → both (cache hit)
            r4 = scan_container_reachability(
                repo_root=d, file_paths=[p], cross_file_referenced_names=set())
            assert len(r4) == 2

    def test_unparseable_file_skip_is_cached(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.py")
            with open(p, "w") as fh:
                fh.write("def broken(:\n")
            assert scan_container_reachability(repo_root=d, file_paths=[p]) == []
            assert scan_container_reachability(repo_root=d, file_paths=[p]) == []

    def test_cache_key_uses_rel_path_independent_of_repo_root(self):
        # Same relative file reachable via a different repo_root must not
        # share cache entries (paths are the key).
        src = '_D = {"a": 1, "b": 2}\n\ndef f():\n    return _D["a"]\n'
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            p1 = os.path.join(d1, "m.py")
            with open(p1, "w") as fh:
                fh.write(src)
            out1 = scan_container_reachability(repo_root=d1, file_paths=["m.py"])
            assert len(out1) == 1
            # different root, relative path key must miss (no shared cache)
            out2 = scan_container_reachability(repo_root=d2, file_paths=["m.py"])
            assert out2 == []  # m.py does not exist under d2


def test_crx_cache_path_goes_through_path_guard(tmp_path, monkeypatch):
    """P-1 regression: the cache path must route through the fail-closed guard."""
    from external_llm.analysis import parse_cache
    from external_llm.analysis.container_reachability_scanner import (
        _CRX_CACHE_VERSION,
        _crx_cache_path,
    )

    calls = []
    monkeypatch.setattr(
        parse_cache,
        "cache_file_path",
        lambda root, filename: calls.append((root, filename)) or "/guarded",
    )
    assert _crx_cache_path(str(tmp_path)) == "/guarded"
    assert calls == [(str(tmp_path), f"container_reachability_v{_CRX_CACHE_VERSION}.json")]
