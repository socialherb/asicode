"""Tests for the find_symbol rg prefilter (_rg_py_files_containing).

The prefilter's correctness contract is a three-way distinction:
  * rg present + token exists  -> non-empty set of absolute paths
  * rg present + token absent  -> empty set (a REAL answer, NOT None)
  * rg missing / error         -> None (caller must full-scan)

Mixing up "empty set" and "None" would either silently drop symbol
definitions or needlessly parse the whole repo. These tests pin the contract.
"""
from pathlib import Path

import pytest

import external_llm.agent.symbol_search as ss

REPO = Path(__file__).resolve().parents[3]
REAL_RG = __import__("shutil").which("rg")
pytestmark = pytest.mark.skipif(not REAL_RG, reason="ripgrep not installed")


def _use_real_rg():
    """Point the module's shutil.which at the real rg binary; return restore fn."""
    orig = ss.shutil.which
    ss.shutil.which = lambda name: REAL_RG
    return orig


def test_real_token_returns_abs_paths_under_root():
    orig = _use_real_rg()
    try:
        out = ss._rg_py_files_containing(REPO, "SymbolSearcher")
    finally:
        ss.shutil.which = orig
    assert out is not None and len(out) >= 1
    for p in out:
        assert Path(p).is_absolute()
        assert Path(p).is_relative_to(REPO)


def test_missing_token_returns_empty_set_not_none():
    # Assembled at runtime so the literal never appears in this test file (rg
    # --fixed-strings would otherwise match the test file itself).
    bogus = "zzz" + "NoSuch" + "Token_qxz_9988776655"
    orig = _use_real_rg()
    try:
        out = ss._rg_py_files_containing(REPO, bogus)
    finally:
        ss.shutil.which = orig
    # Critical: an empty set is a trustworthy "no matches" answer and must NOT
    # be None, otherwise the caller needlessly scans the whole repo.
    assert out is not None and out == set()


def test_rg_missing_returns_none_for_full_scan_fallback():
    orig = ss.shutil.which
    ss.shutil.which = lambda name: None  # rg absent
    try:
        out = ss._rg_py_files_containing(REPO, "SymbolSearcher")
    finally:
        ss.shutil.which = orig
    # None means "prefilter untrustworthy" — caller falls back to scanning all.
    assert out is None


def test_candidate_set_covers_defining_file():
    """A file that defines the symbol must survive the prefilter."""
    orig = _use_real_rg()
    try:
        out = ss._rg_py_files_containing(REPO, "AgentLoop")
    finally:
        ss.shutil.which = orig
    assert out is not None
    defining = str((REPO / "external_llm/agent/agent_loop.py").resolve())
    assert defining in out, "prefilter dropped the file that defines AgentLoop"


class TestDefPatternPrefilter:
    """`_rg_py_files_defining` narrows candidates to files that can DEFINE the
    name — the whole win for widely-imported names, where the word-match set
    is dominated by importers. Contract: subset of the word-match set; empty
    means "regex saw nothing, fall back to word-match", None means "not
    answerable this way" (rg missing OR non-identifier token)."""

    @staticmethod
    def _repo(tmp_path):
        (tmp_path / "defs.py").write_text(
            "import os\n\n\n"
            "class Widget:\n"
            "    limit: int = 3\n\n"
            "    async def spin(self):\n"
            "        return 1\n\n\n"
            "def make_widget():\n"
            "    return Widget()\n\n\n"
            "FACTORY = make_widget\n"
            "alias = FACTORY = make_widget\n",
            encoding="utf-8",
        )
        (tmp_path / "importer.py").write_text(
            "from defs import Widget, make_widget, FACTORY\n\n"
            "w = Widget()\n"
            "make_widget()\n"
            "print(FACTORY)\n",
            encoding="utf-8",
        )
        return tmp_path

    def _defining(self, root, token, kind):
        orig = _use_real_rg()
        try:
            return ss._rg_py_files_defining(root, token, kind)
        finally:
            ss.shutil.which = orig

    def test_importer_excluded_definer_kept(self, tmp_path):
        root = self._repo(tmp_path)
        for token, kind in [("Widget", "any"), ("Widget", "class"),
                            ("make_widget", "function"), ("spin", "any"),
                            ("FACTORY", "variable"), ("limit", "variable")]:
            out = self._defining(root, token, kind)
            assert out is not None, (token, kind)
            assert str((root / "defs.py").resolve()) in out, (token, kind)
            assert str((root / "importer.py").resolve()) not in out, (token, kind)

    def test_chained_assignment_second_target_matches(self, tmp_path):
        """`alias = FACTORY = ...` — the AST fallback records FACTORY, so the
        pattern must keep the file even when FACTORY is not the first target."""
        root = self._repo(tmp_path)
        out = self._defining(root, "FACTORY", "variable")
        assert out is not None and str((root / "defs.py").resolve()) in out

    def test_kind_narrows_shape(self, tmp_path):
        root = self._repo(tmp_path)
        # kind="class": an assignment-only name yields an empty set (fallback
        # signal), not the defining file.
        out = self._defining(root, "FACTORY", "class")
        assert out == set()
        # kind="variable": function-only names likewise. (A class-only name is
        # NOT asserted empty: the annotation shape `X\s*:` deliberately
        # over-matches `class X:` — widening is allowed, dropping is not.)
        out = self._defining(root, "make_widget", "variable")
        assert out == set()
        out = self._defining(root, "spin", "variable")
        assert out == set()

    def test_subset_of_word_match(self, tmp_path):
        root = self._repo(tmp_path)
        orig = _use_real_rg()
        try:
            defining = ss._rg_py_files_defining(root, "Widget", "any")
            mentioning = ss._rg_py_files_containing(root, "Widget")
        finally:
            ss.shutil.which = orig
        assert defining is not None and mentioning is not None
        assert defining <= mentioning

    def test_non_identifier_token_returns_none(self, tmp_path):
        assert ss._rg_py_files_defining(tmp_path, "a.b", "any") is None
        assert ss._rg_py_files_defining(tmp_path, "x+y", "any") is None
        assert ss._rg_py_files_defining(tmp_path, "", "any") is None

    def test_rg_missing_returns_none(self, tmp_path):
        orig = ss.shutil.which
        ss.shutil.which = lambda name: None
        try:
            assert ss._rg_py_files_defining(tmp_path, "Widget", "any") is None
        finally:
            ss.shutil.which = orig

    def test_find_symbol_skips_importer_parse(self, tmp_path):
        """End-to-end: the importer file must not be parsed at all."""
        root = self._repo(tmp_path)
        orig = _use_real_rg()
        try:
            s = ss.SymbolSearcher(str(root))
            res = s.find_symbol("Widget")
        finally:
            ss.shutil.which = orig
        assert res and res[0].kind == "class"
        assert res[0].file.endswith("defs.py")
        parsed = set(s._py_file_cache)
        assert str((root / "defs.py").resolve()) in parsed
        assert str((root / "importer.py").resolve()) not in parsed, (
            "def-pattern prefilter should have excluded the importer from parsing"
        )

    def test_find_symbol_falls_back_when_regex_cannot_see_the_def(self, tmp_path):
        """A definition split across physical lines (`X \\` + `= 1`) is
        invisible to the line-based regex. The def-set comes back empty and
        the word-match fallback must still find the symbol."""
        (tmp_path / "weird.py").write_text(
            "HIDDEN_CONT \\\n    = 5\n", encoding="utf-8"
        )
        orig = _use_real_rg()
        try:
            s = ss.SymbolSearcher(str(tmp_path))
            res = s.find_symbol("HIDDEN_CONT")
        finally:
            ss.shutil.which = orig
        assert res, "empty def-set must fall back to the word-match prefilter"
        assert res[0].kind == "constant"


class TestNonPyIndexProbe:
    """`find_symbol`'s default kind="any" reaches the non-Python index branch
    unconditionally, so a Python-only lookup used to pay a whole-repo index
    build. `_rg_token_in_nonpy_files` skips that build when no INDEXED
    non-Python file mentions the token. Its scope must track the provider
    registry — probing a wider set reports True for tokens that live only in
    files the index never indexes, triggering a build that cannot match."""

    def test_probe_scope_matches_index_scope(self):
        """Probe globs must equal what _nonpy_index_for can actually index.

        Owning a glob is not enough: a provider contributes symbols only via the
        tree-sitter batch (grammar installed) or the regex loop (non-empty
        get_symbol_patterns). A provider with neither must be OUT of scope, or
        the probe reports True for tokens living only in files the build walks
        past — paying a whole-repo build that cannot match.
        """
        import external_llm.languages.tree_sitter_utils as tsu
        from external_llm.languages import LanguageRegistry

        ts_langs = tsu.get_available_languages() if ss._HAS_TS else set()
        globs = set(ss._nonpy_index_globs())
        assert globs, "probe would be scopeless"
        expected = set()
        for p in set(LanguageRegistry.instance()._providers.values()):
            lang_id = p.language_id().value
            if lang_id in ("python", "typescript", "javascript"):
                continue
            if lang_id not in ts_langs and not p.get_symbol_patterns(kind="any"):
                continue
            expected.update(p.get_file_globs())
        assert globs == expected
        # Python is excluded, else every Python symbol trivially probes True.
        assert "*.py" not in globs

    def test_unindexable_provider_globs_are_out_of_scope(self):
        """Any provider with no grammar AND no patterns must contribute no glob.

        Asserted over the whole registry rather than naming json, so this keeps
        holding if the dead provider changes or its grammar becomes installable.
        """
        import external_llm.languages.tree_sitter_utils as tsu
        from external_llm.languages import LanguageRegistry

        ts_langs = tsu.get_available_languages() if ss._HAS_TS else set()
        globs = set(ss._nonpy_index_globs())
        for p in set(LanguageRegistry.instance()._providers.values()):
            lang_id = p.language_id().value
            if lang_id in ("python", "typescript", "javascript"):
                continue
            if lang_id in ts_langs or p.get_symbol_patterns(kind="any"):
                continue
            assert not (set(p.get_file_globs()) & globs), (
                f"{lang_id} can emit no symbol yet its globs are probed — "
                "tokens mentioned only in its files trigger a useless build"
            )

    def test_token_only_in_unindexable_file_probes_false(self, tmp_path):
        """A token living only in a file no index path reads must probe False.

        This is the regression the scope filter exists for: before it, a name
        mentioned only in package.json cost a whole-repo non-Python index build
        on every lookup.
        """
        import external_llm.languages.tree_sitter_utils as tsu
        from external_llm.languages import LanguageRegistry

        ts_langs = tsu.get_available_languages() if ss._HAS_TS else set()
        dead = [
            p for p in set(LanguageRegistry.instance()._providers.values())
            if p.language_id().value not in ("python", "typescript", "javascript")
            and p.language_id().value not in ts_langs
            and not p.get_symbol_patterns(kind="any")
        ]
        if not dead:
            pytest.skip("every registered provider is indexable in this install")
        ext = dead[0].get_file_globs()[0].lstrip("*")
        (tmp_path / f"data{ext}").write_text('{"OnlyInDeadFile": 1}\n')
        orig = _use_real_rg()
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "OnlyInDeadFile") is False
        finally:
            ss.shutil.which = orig

    def test_token_only_in_python_probes_false(self, tmp_path):
        (tmp_path / "m.py").write_text("def py_only_symbol():\n    return 1\n")
        (tmp_path / "main.go").write_text("package main\n")
        orig = _use_real_rg()
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "py_only_symbol") is False
        finally:
            ss.shutil.which = orig

    def test_token_in_nonpy_source_probes_true(self, tmp_path):
        (tmp_path / "main.go").write_text("package main\n\nfunc GoOnlySymbol() int {\n\treturn 1\n}\n")
        orig = _use_real_rg()
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "GoOnlySymbol") is True
        finally:
            ss.shutil.which = orig

    def test_rg_missing_probes_none_so_index_is_built(self, tmp_path):
        orig = ss.shutil.which
        ss.shutil.which = lambda name: None
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "anything") is None
        finally:
            ss.shutil.which = orig

    def test_gate_returns_true_on_warm_cache_without_probing(self, tmp_path):
        """A warm index is a dict hit — cheaper than probing. Probing there
        would make the fast path slower, so the gate must short-circuit."""
        import external_llm.agent.symbol_search as _ss
        s = ss.SymbolSearcher(str(tmp_path))
        s._nonpy_index_cache[str(tmp_path)] = (_ss._time.monotonic(), {})
        called = []
        orig = _ss._rg_token_in_nonpy_files
        _ss._rg_token_in_nonpy_files = lambda *a: called.append(a) or False
        try:
            assert s._nonpy_index_worth_building(tmp_path, "whatever") is True
        finally:
            _ss._rg_token_in_nonpy_files = orig
        assert called == [], "warm cache must not pay for a probe"


class TestNonPyProbeInProcessFastPath:
    """The probe walks the indexable set ONCE per root (token-independent,
    TTL-cached) and then answers each token in-process, so a repeat lookup
    costs a read rather than an rg spawn. The fast path must be
    indistinguishable from the rg it replaces, and must defer to rg when the
    set is too large for reading to be the cheaper option."""

    @staticmethod
    def _fresh(tmp_path):
        ss._NONPY_FILES_CACHE.pop(str(tmp_path), None)

    def test_agrees_with_rg_including_non_word_boundary_tokens(self, tmp_path):
        """rg's --word-regexp semantics must be reproduced exactly.

        `(?<!\\w)...(?!\\w)` rather than `\\b...\\b`: for a token starting with a
        non-word char (CSS `--primary-color`) `\\b` anchors the wrong way.
        """
        (tmp_path / "s.css").write_text(
            ".btn-primary { --primary-color: red; }\n", encoding="utf-8"
        )
        (tmp_path / "a.go").write_text(
            "package main\nfunc GoThing() {}\n", encoding="utf-8"
        )
        tokens = ["GoThing", "btn-primary", "--primary-color", "primary",
                  "GoThin", "oThing", "Nope", "main"]
        orig = _use_real_rg()
        try:
            for tok in tokens:
                self._fresh(tmp_path)
                fast = ss._rg_token_in_nonpy_files(tmp_path, tok)
                _cap = ss._NONPY_INPROC_MAX_FILES
                ss._NONPY_INPROC_MAX_FILES = -1  # force the rg path
                try:
                    self._fresh(tmp_path)
                    slow = ss._rg_token_in_nonpy_files(tmp_path, tok)
                finally:
                    ss._NONPY_INPROC_MAX_FILES = _cap
                assert fast == slow, f"{tok!r}: in-process={fast} rg={slow}"
        finally:
            ss.shutil.which = orig

    def test_oversized_set_falls_back_to_rg(self, tmp_path, monkeypatch):
        """Above the cap the flat-cost spawn wins; reading must not be used."""
        (tmp_path / "a.go").write_text("package main\nfunc Sentinel() {}\n", encoding="utf-8")
        monkeypatch.setattr(ss, "_NONPY_INPROC_MAX_FILES", 0)
        called = []
        real = ss._word_in_files
        monkeypatch.setattr(ss, "_word_in_files", lambda *a: called.append(a) or real(*a))
        orig = _use_real_rg()
        try:
            self._fresh(tmp_path)
            assert ss._rg_token_in_nonpy_files(tmp_path, "Sentinel") is True
        finally:
            ss.shutil.which = orig
        assert not called, "in-process scan ran despite exceeding the file cap"

    def test_byte_cap_also_forces_rg(self, tmp_path, monkeypatch):
        (tmp_path / "a.go").write_text("package main\nfunc Sentinel() {}\n", encoding="utf-8")
        monkeypatch.setattr(ss, "_NONPY_INPROC_MAX_BYTES", 1)
        called = []
        real = ss._word_in_files
        monkeypatch.setattr(ss, "_word_in_files", lambda *a: called.append(a) or real(*a))
        orig = _use_real_rg()
        try:
            self._fresh(tmp_path)
            assert ss._rg_token_in_nonpy_files(tmp_path, "Sentinel") is True
        finally:
            ss.shutil.which = orig
        assert not called

    def test_file_list_is_token_independent_and_cached(self, tmp_path):
        """One walk serves every token — that is the entire win."""
        (tmp_path / "a.go").write_text("package main\nfunc GoThing() {}\n", encoding="utf-8")
        orig = _use_real_rg()
        try:
            self._fresh(tmp_path)
            first = ss._nonpy_indexable_files(tmp_path)
            assert first is not None and len(first[0]) == 1
            runs = []
            real_run = ss.subprocess.run
            ss.subprocess.run = lambda *a, **k: runs.append(a) or real_run(*a, **k)
            try:
                for tok in ("GoThing", "Other", "Third"):
                    ss._rg_token_in_nonpy_files(tmp_path, tok)
            finally:
                ss.subprocess.run = real_run
        finally:
            ss.shutil.which = orig
        assert not runs, f"{len(runs)} subprocess spawn(s) on the cached fast path"

    def test_rg_missing_still_returns_none(self, tmp_path):
        """The untrustworthy contract survives the fast path."""
        orig = ss.shutil.which
        ss.shutil.which = lambda name: None
        try:
            self._fresh(tmp_path)
            assert ss._rg_token_in_nonpy_files(tmp_path, "anything") is None
        finally:
            ss.shutil.which = orig

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        """Matches _index_via_treesitter_batch: a file the build cannot read
        holds no indexable symbol either."""
        p = tmp_path / "a.go"
        p.write_text("package main\nfunc GoThing() {}\n", encoding="utf-8")
        assert ss._word_in_files([str(p), str(tmp_path / "gone.go")], "GoThing") is True
        assert ss._word_in_files([str(tmp_path / "gone.go")], "GoThing") is False


class TestNonpyFilesCacheCap:
    """``_NONPY_FILES_CACHE`` is the ONLY per-root cache that was not using
    ``_capped_put`` — three plain assignments left it unbounded, growing on
    every visited root. The fix replaced those three assignments with
    ``_capped_put`` calls so the cache is FIFO-bounded at
    ``_WALK_CACHE_MAX_ENTRIES`` (8), matching every sibling cache.
    """

    # Each mode reaches a DIFFERENT one of the three store sites, so a plain
    # `=` reverted at any single site fails here. Driving the real function is
    # the point: an earlier version of this test called ``_capped_put``
    # directly and therefore passed with all three sites reverted — it was
    # re-testing ``_capped_put`` (already covered by
    # ``test_shared_utils.py::test_capped_put_evicts_oldest_when_over_cap``)
    # rather than the fix it was written to guard.
    @pytest.mark.parametrize("mode", ["success", "rg_bad_returncode", "rg_raises"])
    def test_every_store_site_stays_capped(self, monkeypatch, mode):
        """15 roots through ``_nonpy_indexable_files`` itself leave 8 entries."""
        from types import SimpleNamespace

        from external_llm.agent._shared_utils import _WALK_CACHE_MAX_ENTRIES

        monkeypatch.setattr(ss.shutil, "which", lambda _name: REAL_RG)
        monkeypatch.setattr(ss, "_nonpy_index_globs", lambda: ["*.go"])

        def fake_run(cmd, **kwargs):
            if mode == "rg_raises":
                raise OSError("rg exploded")
            # returncode 2 = rg error -> the "unanswerable" store site.
            rc = 2 if mode == "rg_bad_returncode" else 0
            # getsize() on this fake path raises OSError, which the walker logs
            # and skips — the entry is still stored, which is what we assert.
            return SimpleNamespace(returncode=rc, stdout="a.go\n", stderr="")

        monkeypatch.setattr(ss.subprocess, "run", fake_run)

        orig = dict(ss._NONPY_FILES_CACHE)
        try:
            ss._NONPY_FILES_CACHE.clear()
            for i in range(15):
                ss._nonpy_indexable_files(Path(f"/fake-root-{i:03d}"))

            assert len(ss._NONPY_FILES_CACHE) == _WALK_CACHE_MAX_ENTRIES, (
                f"[{mode}] _NONPY_FILES_CACHE size="
                f"{len(ss._NONPY_FILES_CACHE)} (cap={_WALK_CACHE_MAX_ENTRIES}) — "
                f"this store site is not going through _capped_put"
            )
            # The oldest 7 entries must be gone (15 - 8 = 7 evicted).
            for i in range(7):
                assert f"/fake-root-{i:03d}" not in ss._NONPY_FILES_CACHE, (
                    f"[{mode}] /fake-root-{i:03d} survived FIFO eviction"
                )
            # The newest 8 entries must still be present.
            for i in range(7, 15):
                assert f"/fake-root-{i:03d}" in ss._NONPY_FILES_CACHE, (
                    f"[{mode}] /fake-root-{i:03d} was evicted — "
                    f"cap or eviction order broken"
                )
        finally:
            ss._NONPY_FILES_CACHE.clear()
            ss._NONPY_FILES_CACHE.update(orig)

    def test_all_three_store_sites_use_capped_put(self):
        """Greps the three assignment sites — they must call _capped_put,
        not plain ``=``. A plain ``=`` would leak entries forever.
        """
        import inspect
        import ast

        src = inspect.getsource(ss._nonpy_indexable_files)
        tree = ast.parse(src)
        # Collect every assignment target that is _NONPY_FILES_CACHE[...] = ...
        plain_assignments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        value = ast.get_source_segment(src, target.value)
                        if value and "_NONPY_FILES_CACHE" in value:
                            plain_assignments.append(target.lineno)
        assert not plain_assignments, (
            f"_nonpy_indexable_files has plain `= _NONPY_FILES_CACHE[key]` "
            f"at lines {plain_assignments} — must use _capped_put instead"
        )


# ── The non-Python probe runs concurrently with the Python prefilter ───────
# Both are a function of (root, search_name) alone, but they used to be
# sequential with the whole Python parse between them. find_symbol now starts
# the non-Python probe before the Python scan and collects it after, which took
# 3 cold lookups on this repo from 110 ms to 96 ms (13%, ~4.9 ms per lookup,
# four interleaved A/B alternations with no overlap between the two medians).


def test_nonpy_probe_starts_before_the_python_scan(tmp_path, monkeypatch):
    """The probe must be in flight while the Python prefilter runs.

    Asserted by ordering, not by timing: the probe records when it STARTS, the
    Python prefilter records when it FINISHES, and the probe's start must come
    first. A timing assertion would be flaky on a loaded machine, and asserting
    only that both ran would pass with the old sequential order too.
    """
    (tmp_path / "a.py").write_text("class Widget:\n    pass\n")
    (tmp_path / "b.py").write_text("import Widget\n")

    order: list[str] = []

    real_defining = ss._rg_py_files_defining

    def _slow_defining(root, token, kind):
        out = real_defining(root, token, kind)
        order.append("python-prefilter-done")
        return out

    def _probe(self, root, token):
        order.append("nonpy-probe-start")
        return False

    monkeypatch.setattr(ss, "_rg_py_files_defining", _slow_defining)
    monkeypatch.setattr(
        ss.SymbolSearcher, "_nonpy_index_worth_building", _probe, raising=True
    )

    ss.SymbolSearcher(tmp_path).find_symbol("Widget")

    assert order, "neither the probe nor the prefilter ran"
    assert order[0] == "nonpy-probe-start", (
        f"probe did not start before the Python prefilter finished: {order}"
    )


def test_nonpy_probe_failure_falls_back_inline(tmp_path, monkeypatch):
    """A probe that raises must not lose the non-Python branch.

    The pooled call is an optimisation; its failure has to degrade to the
    inline call, not to "no non-Python results".
    """
    (tmp_path / "a.py").write_text("x = 1\n")
    calls: list[str] = []

    def _boom(self, root, token):
        calls.append("called")
        if len(calls) == 1:
            raise RuntimeError("probe exploded")
        return False

    monkeypatch.setattr(
        ss.SymbolSearcher, "_nonpy_index_worth_building", _boom, raising=True
    )
    # Must not propagate, and must have retried inline.
    ss.SymbolSearcher(tmp_path).find_symbol("Widget")
    assert len(calls) == 2, f"no inline retry after probe failure: {calls}"
