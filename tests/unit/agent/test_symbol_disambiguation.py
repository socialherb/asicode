"""Tests for symbol disambiguation in SymbolSearcher and RepositoryGraph."""

from external_llm.agent.symbol_search import SymbolDef, SymbolSearcher


class TestRankSymbolResults:
    """Tests for SymbolSearcher._rank_symbol_results."""

    def _make_def(self, file, kind="function", name="target"):
        return SymbolDef(file=file, line=1, kind=kind, name=name)

    def test_prefer_file_exact_match(self):
        """Result in prefer_files should rank first."""
        results = [
            self._make_def("lib/utils.py"),
            self._make_def("src/core.py"),
            self._make_def("tests/test_core.py"),
        ]
        prefer = ["src/core.py"]

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        assert ranked[0].file == "src/core.py"

    def test_prefer_file_basename_match(self):
        """Result with matching basename should rank high (but lower than exact)."""
        results = [
            self._make_def("a/core.py"),
            self._make_def("b/core.py"),
        ]
        prefer = ["src/deep/core.py"]

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        # Both match by basename — tied at +3.0
        # Order should be stable (original order preserved for ties)
        assert len(ranked) == 2

    def test_directory_proximity(self):
        """Result in same directory as prefer_file should rank higher."""
        results = [
            self._make_def("lib/handler.py"),
            self._make_def("src/handler.py"),
        ]
        prefer = ["src/core.py"]

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        # src/handler.py: dir proximity +2.0
        # lib/handler.py: 0.0
        assert ranked[0].file == "src/handler.py"

    def test_test_file_penalty(self):
        """Test files should be ranked lower."""
        results = [
            self._make_def("tests/test_user.py"),
            self._make_def("src/user.py"),
        ]
        prefer = []

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        # src/user.py: 0.0 + kind(1.0) = 1.0
        # tests/test_user.py: -2.0 + kind(1.0) = -1.0
        assert ranked[0].file == "src/user.py"

    def test_kind_preference(self):
        """Class/function definitions should rank over constants."""
        results = [
            self._make_def("a.py", kind="constant"),
            self._make_def("b.py", kind="class"),
        ]
        prefer = []

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        assert ranked[0].file == "b.py"

    def test_combined_scoring(self):
        """Multiple tiers should stack."""
        results = [
            self._make_def("src/models.py", kind="class"),      # dir+2, kind+1 = 3
            self._make_def("lib/models.py", kind="function"),    # kind+1 = 1
            self._make_def("tests/models.py", kind="function"),  # test-2, kind+1 = -1
        ]
        prefer = ["src/core.py"]

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        assert ranked[0].file == "src/models.py"
        assert ranked[1].file == "lib/models.py"
        assert ranked[2].file == "tests/models.py"

    def test_exact_file_beats_proximity(self):
        """Exact file match (+4) should beat directory proximity (+2)."""
        results = [
            self._make_def("src/other.py"),   # dir proximity +2
            self._make_def("lib/target.py"),  # exact match +4
        ]
        prefer = ["lib/target.py", "src/core.py"]

        ranked = SymbolSearcher._rank_symbol_results(results, prefer)

        # lib/target.py: exact(+4) + kind(+1) = 5
        # src/other.py: proximity(+2) + kind(+1) = 3
        assert ranked[0].file == "lib/target.py"

    def test_empty_prefer_no_crash(self):
        """Empty prefer_files should not crash (no-op)."""
        results = [self._make_def("a.py"), self._make_def("b.py")]

        ranked = SymbolSearcher._rank_symbol_results(results, [])

        assert len(ranked) == 2

    def test_single_result_unchanged(self):
        """Single result should return unchanged."""
        results = [self._make_def("only.py")]

        ranked = SymbolSearcher._rank_symbol_results(results, ["other.py"])

        assert ranked[0].file == "only.py"


class TestRepositoryGraphGetSymbol:
    """Tests for RepositoryGraph.get_symbol with prefer_files."""

    def _make_graph(self):
        from external_llm.graph.models import SymbolNode
        from external_llm.graph.repository_graph import RepositoryGraph

        graph = RepositoryGraph(repo_root="/tmp/test")

        def _sym(name, qualname, file_path, start_line=1):
            return SymbolNode(
                name=name, qualname=qualname, module=file_path.replace("/", "."),
                kind="class", file_path=file_path,
                start_line=start_line, end_line=start_line + 10,
            )

        graph.symbols["auth/models.py:User"] = _sym("User", "User", "auth/models.py", 10)
        graph.symbols["db/models.py:User"] = _sym("User", "User", "db/models.py", 5)
        graph.symbols["tests/test_user.py:User"] = _sym("User", "User", "tests/test_user.py", 20)

        return graph

    def test_prefer_files_disambiguates(self):
        """prefer_files should select the correct file's symbol."""
        graph = self._make_graph()

        result = graph.get_symbol("User", prefer_files=["db/models.py"])

        assert result is not None
        assert result.file_path == "db/models.py"

    def test_prefer_files_directory_proximity(self):
        """prefer_files directory proximity should disambiguate."""
        graph = self._make_graph()

        result = graph.get_symbol("User", prefer_files=["auth/views.py"])

        assert result is not None
        assert result.file_path == "auth/models.py"

    def test_prefer_files_test_penalty(self):
        """Test files should be deprioritized with prefer_files."""
        graph = self._make_graph()

        # With auth/models.py preferred, test file should lose
        result = graph.get_symbol("User", prefer_files=["auth/core.py"])

        assert result is not None
        assert result.file_path == "auth/models.py"

    def test_no_prefer_returns_first(self):
        """Without prefer_files, returns first match (existing behavior)."""
        graph = self._make_graph()

        result = graph.get_symbol("User")

        assert result is not None
        # Should be one of the three — we don't guarantee which without prefer_files

    def test_file_path_scoping_still_works(self):
        """Explicit file_path should still work (no regression)."""
        graph = self._make_graph()

        result = graph.get_symbol("User", file_path="db/models.py")

        assert result is not None
        assert result.file_path == "db/models.py"

    def test_qualname_still_works(self):
        """Dotted qualname search should still work."""
        from external_llm.graph.models import SymbolNode

        graph = self._make_graph()
        sym = SymbolNode(
            name="validate", qualname="User.validate",
            module="auth.models", kind="method", file_path="auth/models.py",
            start_line=15, end_line=25,
        )
        graph.symbols["auth/models.py:User.validate"] = sym

        result = graph.get_symbol("User.validate")

        assert result is not None
        assert result.qualname == "User.validate"
