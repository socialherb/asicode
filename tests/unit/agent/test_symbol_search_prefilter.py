"""Tests for the find_symbol rg prefilter (_rg_py_files_containing).

The prefilter's correctness contract is a three-way distinction:
  * rg present + token exists  -> non-empty set of absolute paths
  * rg present + token absent  -> empty set (a REAL answer, NOT None)
  * rg missing / error         -> None (caller must full-scan)

Mixing up "empty set" and "None" would either silently drop symbol
definitions or needlessly parse the whole repo. These tests pin the contract.
"""

import os
import time
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
            "from defs import Widget, make_widget, FACTORY\n\nw = Widget()\nmake_widget()\nprint(FACTORY)\n",
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
        for token, kind in [
            ("Widget", "any"),
            ("Widget", "class"),
            ("make_widget", "function"),
            ("spin", "any"),
            ("FACTORY", "variable"),
            ("limit", "variable"),
        ]:
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
        (tmp_path / "weird.py").write_text("HIDDEN_CONT \\\n    = 5\n", encoding="utf-8")
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
            p
            for p in set(LanguageRegistry.instance()._providers.values())
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
        (tmp_path / "s.css").write_text(".btn-primary { --primary-color: red; }\n", encoding="utf-8")
        (tmp_path / "a.go").write_text("package main\nfunc GoThing() {}\n", encoding="utf-8")
        tokens = ["GoThing", "btn-primary", "--primary-color", "primary", "GoThin", "oThing", "Nope", "main"]
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

    def test_token_spanning_chunk_boundary_is_found(self, tmp_path, monkeypatch):
        """A token split by a stream boundary must still match — carrying
        ``len(token) + 2`` trailing bytes across the seam exists for this.
        Without the carry, chunk 1 holds only ``GoTh`` and chunk 2 only
        ``ing zzz``, so a per-chunk search would miss it."""
        monkeypatch.setattr(ss, "_NONPY_SCAN_CHUNK", 8)
        p = tmp_path / "big.go"
        p.write_text("yyy GoThing zzz", encoding="utf-8")
        assert ss._word_in_files([str(p)], "GoThing") is True

    def test_word_boundary_enforced_across_chunk_seam(self, tmp_path, monkeypatch):
        """The (?<!\\w) lookbehind must still see the word char that chunk 1
        ends with — ``GoThing`` flanked by ``y``/``z`` is not a whole word,
        so the seam must not turn it into a false positive."""
        monkeypatch.setattr(ss, "_NONPY_SCAN_CHUNK", 8)
        p = tmp_path / "big.go"
        p.write_text("yyyyyyGoThingzzzzzzzz", encoding="utf-8")
        assert ss._word_in_files([str(p)], "GoThing") is False

    def test_match_early_stops_reading(self, tmp_path, monkeypatch):
        """The point of streaming: a hit in the first chunk must return
        without reading the rest of the file (the old ``fh.read()`` slurped
        the whole file before searching at all)."""
        monkeypatch.setattr(ss, "_NONPY_SCAN_CHUNK", 64)
        p = tmp_path / "big.go"
        p.write_text("GoThing " + "y" * 8192, encoding="utf-8")
        reads = []

        class _Recording:
            def __init__(self, fh):
                self._fh = fh

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

            def read(self, size=-1):
                data = self._fh.read(size)
                reads.append(len(data))
                return data

        def recording_open(*args, **kwargs):
            return _Recording(real_open(*args, **kwargs))

        real_open = open
        monkeypatch.setattr("builtins.open", recording_open)
        assert ss._word_in_files([str(p)], "GoThing") is True
        assert reads, "file was never read"
        assert sum(reads) <= 64, f"read {sum(reads)} bytes for a first-chunk hit"


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
                    f"[{mode}] /fake-root-{i:03d} was evicted — cap or eviction order broken"
                )
        finally:
            ss._NONPY_FILES_CACHE.clear()
            ss._NONPY_FILES_CACHE.update(orig)

    def test_all_three_store_sites_use_capped_put(self):
        """Greps the three assignment sites — they must call _capped_put,
        not plain ``=``. A plain ``=`` would leak entries forever.
        """
        import ast
        import inspect

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
    monkeypatch.setattr(ss.SymbolSearcher, "_nonpy_index_worth_building", _probe, raising=True)

    ss.SymbolSearcher(tmp_path).find_symbol("Widget")

    assert order, "neither the probe nor the prefilter ran"
    assert order[0] == "nonpy-probe-start", f"probe did not start before the Python prefilter finished: {order}"


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

    monkeypatch.setattr(ss.SymbolSearcher, "_nonpy_index_worth_building", _boom, raising=True)
    # Must not propagate, and must have retried inline.
    ss.SymbolSearcher(tmp_path).find_symbol("Widget")
    assert len(calls) == 2, f"no inline retry after probe failure: {calls}"


def test_nonpy_probe_timeout_falls_back_inline(tmp_path, monkeypatch):
    """A probe that does not finish in time must not wedge the caller.

    P1 audit: dispatch (and therefore find_symbol) can run ON
    _thread_pool.shared_pool, and the speculative probe is submitted to the
    SAME pool — if every worker were blocked on a still-queued probe, the pool
    would deadlock. The timeout caps the wait; the inline retry keeps the
    answer. Asserted by wall-clock behavior: with the cap, find_symbol returns
    (and retries inline) long before the blocked probe would have finished.
    """
    (tmp_path / "a.py").write_text("x = 1\n")
    calls: list[str] = []

    def _slow_probe(self, root, token):
        calls.append("called")
        if len(calls) == 1:
            time.sleep(2)  # longer than the monkeypatched 50 ms cap below
        return False

    monkeypatch.setattr(ss.SymbolSearcher, "_nonpy_index_worth_building", _slow_probe, raising=True)
    monkeypatch.setattr(ss, "_NONPY_PROBE_TIMEOUT_SEC", 0.05)
    # The pooled probe is still sleeping; find_symbol must time out, retry
    # inline (2nd call), and return without ever seeing the probe's answer.
    ss.SymbolSearcher(tmp_path).find_symbol("Widget")
    assert len(calls) == 2, f"no inline retry after probe timeout: {calls}"


class TestNonPyBlobMemo:
    """``_word_in_files`` re-reads the whole (capped) file set on every probe;
    after the first MISS the content memo answers later tokens from memory,
    re-verified per probe by (mtime_ns, size) signatures so a just-written
    file is visible without waiting for the TTL."""

    @staticmethod
    def _two_files(tmp_path):
        (tmp_path / "a.go").write_text("package main\nfunc Alpha() {}\n", encoding="utf-8")
        (tmp_path / "b.go").write_text("package main\nvar Beta = 1\n", encoding="utf-8")

    def test_second_probe_answers_from_memory(self, tmp_path, monkeypatch):
        """After a miss builds the blob, later tokens must not reopen any file."""
        self._two_files(tmp_path)
        orig = _use_real_rg()
        try:
            # Hit: early-exit streaming, builds nothing.
            assert ss._rg_token_in_nonpy_files(tmp_path, "Alpha") is True
            # Miss: full scan, then the content memo is cached.
            assert ss._rg_token_in_nonpy_files(tmp_path, "Gamma") is False
            opened: list[str] = []
            real_open = open

            def recording_open(*args, **kwargs):
                opened.append(str(args[0]))
                return real_open(*args, **kwargs)

            monkeypatch.setattr("builtins.open", recording_open)
            assert ss._rg_token_in_nonpy_files(tmp_path, "Delta") is False
            assert ss._rg_token_in_nonpy_files(tmp_path, "Alpha") is True
            assert not opened, f"memo probe reopened files: {opened}"
        finally:
            ss.shutil.which = orig

    def test_edited_file_invalidates_blob_by_signature(self, tmp_path):
        """An mtime/size change must be visible on the very next probe."""
        self._two_files(tmp_path)
        orig = _use_real_rg()
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "Gamma") is False  # miss -> blob
            (tmp_path / "a.go").write_text(
                "package main\nfunc Alpha() {}\nfunc GammaMarker() {}\n",
                encoding="utf-8",
            )
            assert ss._rg_token_in_nonpy_files(tmp_path, "GammaMarker") is True
        finally:
            ss.shutil.which = orig

    def test_new_file_invalidates_blob_via_list_key(self, tmp_path):
        """A newly created file changes the file list — and with it the memo key."""
        self._two_files(tmp_path)
        orig = _use_real_rg()
        try:
            assert ss._rg_token_in_nonpy_files(tmp_path, "Gamma") is False
            (tmp_path / "c.go").write_text("package main\nfunc ZedNewFile() {}\n", encoding="utf-8")
            # Exactly what invalidate_nonpy_caches does to the walk cache.
            ss._NONPY_FILES_CACHE.pop(str(tmp_path), None)
            assert ss._rg_token_in_nonpy_files(tmp_path, "ZedNewFile") is True
        finally:
            ss.shutil.which = orig

    def test_hit_probe_builds_blob_too(self, tmp_path):
        """The cold probe caches on the first call regardless of the answer —
        the early-exit trade is deliberate: the extra read is hidden behind
        the speculative probe and the cold index build a hit triggers."""
        self._two_files(tmp_path)
        orig = _use_real_rg()
        try:
            _files = ss._nonpy_indexable_files(tmp_path)
            assert _files is not None
            key = (str(tmp_path), tuple(_files[0]))
            ss._NONPY_BLOB_CACHE.pop(key, None)
            assert ss._rg_token_in_nonpy_files(tmp_path, "Alpha") is True
            assert key in ss._NONPY_BLOB_CACHE, "the cold probe must build the memo"
        finally:
            ss.shutil.which = orig

    def test_newline_token_falls_back_to_streaming(self, tmp_path, monkeypatch):
        """A newline token must not use the ``\\n``-joined blob (seam semantics)."""
        self._two_files(tmp_path)
        orig = _use_real_rg()
        try:
            called = []
            real = ss._word_in_files
            monkeypatch.setattr(ss, "_word_in_files", lambda *a: called.append(a) or real(*a))
            assert ss._rg_token_in_nonpy_files(tmp_path, "a\nb") is False
            assert called, "newline token must take the streaming path"
        finally:
            ss.shutil.which = orig

    def test_unreadable_file_keeps_memo_effective(self, tmp_path, monkeypatch):
        """A file that cannot be opened must not invalidate the memo on every
        probe — its real signature is stored even though its content is not."""
        self._two_files(tmp_path)
        p = tmp_path / "a.go"
        orig = _use_real_rg()
        try:
            os.chmod(p, 0)
            try:
                assert ss._rg_token_in_nonpy_files(tmp_path, "Gamma") is False
            finally:
                os.chmod(p, 0o644)
            called = []
            real = ss._word_in_files
            monkeypatch.setattr(ss, "_word_in_files", lambda *a: called.append(a) or real(*a))
            assert ss._rg_token_in_nonpy_files(tmp_path, "Delta") is False
            assert not called, "unreadable file forced a rebuild on every probe"
        finally:
            os.chmod(p, 0o644)
            ss.shutil.which = orig

    def test_blob_contains_word_boundary_semantics(self):
        """Literal search + manual boundary must equal the lookarounds —
        including bad-boundary skips, blob edges, and empty tokens."""
        blob = "xxGoThingyy\nGoThing\nzz"
        cases = [
            ("GoThing", True),  # 1st flanked by x/y (bad), 2nd clean
            ("xxGoThingyy", True),  # blob start
            ("zz", True),  # blob end
            ("GoThin", False),  # partial inside GoThing (followed by g)
            ("GoThingy", False),  # literal hit but bad boundary on both sides
            ("Thing", False),  # both occurrences preceded by G
            ("xx", False),  # blob start but followed by G
            ("Nope", False),
            ("", False),  # must terminate, not loop
        ]
        for tok, exp in cases:
            assert ss._blob_contains_word(blob, tok) is exp, tok

    def test_blob_matches_streaming_core_on_joined_files(self, tmp_path):
        """``_blob_contains_word`` on a ``\\n``-joined blob must agree with the
        per-file streaming scan — a seam must behave exactly like a file
        boundary, never creating or destroying a match."""
        (tmp_path / "a.go").write_text("yyy GoThing zzz", encoding="utf-8")
        (tmp_path / "b.go").write_text("GoOther\n", encoding="utf-8")
        files = [str(tmp_path / "a.go"), str(tmp_path / "b.go")]
        blob = "\n".join(p.read_text(encoding="utf-8") for p in (tmp_path / "a.go", tmp_path / "b.go"))
        for tok in ["GoThing", "GoOther", "Thing", "GoTh", "Other", "yyy", "zzz", "Nope"]:
            got = ss._blob_contains_word(blob, tok)
            exp = ss._word_in_files(files, tok)
            assert got is exp, f"{tok!r}: blob={got} streaming={exp}"

    def test_blob_store_site_uses_capped_put(self):
        """AST grep: the memo's only store site must be _capped_put, not ``=``."""
        import ast
        import inspect

        src = inspect.getsource(ss._word_in_files_cached)
        tree = ast.parse(src)
        plain = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        value = ast.get_source_segment(src, target.value)
                        if value and "_NONPY_BLOB_CACHE" in value:
                            plain.append(target.lineno)
        assert not plain, f"_word_in_files_cached stores the memo with plain `=` at {plain} — use _capped_put"

    def test_blob_cache_stays_fifo_capped(self, tmp_path):
        """cap+2 roots through the real probe leave _NONPY_BLOB_MAX_ENTRIES."""
        orig = _use_real_rg()
        saved = dict(ss._NONPY_BLOB_CACHE)
        ss._NONPY_BLOB_CACHE.clear()
        try:
            cap = ss._NONPY_BLOB_MAX_ENTRIES
            for i in range(cap + 2):
                root = tmp_path / f"r{i}"
                root.mkdir()
                (root / "a.go").write_text("package main\n", encoding="utf-8")
                assert ss._rg_token_in_nonpy_files(root, f"MissToken{i}") is False
            assert len(ss._NONPY_BLOB_CACHE) <= cap
            assert str(tmp_path / "r0") not in {k[0] for k in ss._NONPY_BLOB_CACHE}, (
                "oldest root survived FIFO eviction"
            )
        finally:
            ss._NONPY_BLOB_CACHE.clear()
            ss._NONPY_BLOB_CACHE.update(saved)
            ss.shutil.which = orig
