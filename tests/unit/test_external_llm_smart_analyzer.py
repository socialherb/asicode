"""
Regression + coverage tests for SmartRequestAnalyzer (SA-B1 fix).

SA-B1: intent/feature keyword matching used naive substring (``kw in text``)
which misclassified requests because short keywords matched inside larger
words (``'ui'`` ⊂ ``'fluid'``/``'guide'``, ``'fix'`` ⊂ ``'suffix'``). The fix
anchors every keyword with word boundaries (``\\b...\\b``). These tests lock
the word-boundary contract and bring ``smart_analyzer.py`` from 0% coverage.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from external_llm.smart_analyzer import (
    FeaturePattern,
    IntentClassifierRule,
    RequestAnalysis,
    SmartRequestAnalyzer,
    TechDetector,
    _keyword_in,
    _keyword_pattern,
)

# ============================================================
# Word-boundary keyword matching (SA-B1 core)
# ============================================================


class TestKeywordBoundary:
    """The ``kw in text`` substring bug — every case here is a real regression."""

    @pytest.mark.parametrize(
        "kw,text",
        [
            # short keywords must NOT match as substrings of larger words
            ("ui", "please build the guide for fluid layout"),
            ("api", "this is a rapid capital inquiry"),
            ("fix", "rename the suffix and prefix tokens"),
            ("bug", "help me debug the buggy parser"),
            ("test", "show the latest contest results"),
            ("add", "update the address padding field"),
            ("db", "adlib the robin hood story"),  # 'db' ⊂ 'adlib'? no — sanity
            ("vue", "review the avenue and value"),
            ("rest", "restore the forest and arrest"),
        ],
    )
    def test_short_keyword_not_in_larger_word(self, kw: str, text: str) -> None:
        assert _keyword_in(kw, text.lower()) is False

    @pytest.mark.parametrize(
        "kw,text",
        [
            ("ui", "improve the ui of the dashboard"),
            ("api", "add a new api endpoint"),
            ("fix", "fix the login bug"),
            ("bug", "there is a bug in the parser"),
            ("test", "write a test for the parser"),
            ("db", "connect the db to the backend"),
            ("pwd", "reset the pwd for the user"),
            ("vue", "migrate to vue framework"),
            ("rest", "design a rest service"),
        ],
    )
    def test_whole_word_matches(self, kw: str, text: str) -> None:
        assert _keyword_in(kw, text.lower()) is True

    def test_multiword_keyword_phrase_boundary(self) -> None:
        # 'sign in' must match the phrase but not 'signin' (separate keyword)
        assert _keyword_in("sign in", "please sign in to your account") is True
        assert _keyword_in("sign in", "use the signin button") is False

    def test_keyword_at_string_boundaries(self) -> None:
        # \b matches at start/end of string too
        assert _keyword_in("fix", "fix") is True
        assert _keyword_in("fix", "fix the bug") is True
        assert _keyword_in("fix", "the fix") is True

    def test_keyword_punctuation_boundary(self) -> None:
        # \b treats punctuation as a boundary
        assert _keyword_in("api", "call the api.") is True
        assert _keyword_in("api", "(api) endpoint") is True
        assert _keyword_in("api", "api,endpoint") is True

    def test_pattern_cache_returns_same_object(self) -> None:
        # module-level cache must dedupe compilations
        a = _keyword_pattern("login")
        b = _keyword_pattern("login")
        assert a is b


# ============================================================
# Intent detection
# ============================================================


class TestDetectIntent:
    def _a(self, repo_root: str = ".") -> SmartRequestAnalyzer:
        return SmartRequestAnalyzer(repo_root)

    def test_create_feature(self) -> None:
        assert self._a()._detect_intent("create a new module") == "create_feature"

    def test_create_feature_via_add(self) -> None:
        assert self._a()._detect_intent("add a payment service") == "create_feature"

    def test_fix_bug(self) -> None:
        assert self._a()._detect_intent("fix the crash on startup") == "fix_bug"

    def test_refactor(self) -> None:
        assert self._a()._detect_intent("refactor the utils module") == "refactor"

    def test_modify_feature(self) -> None:
        assert self._a()._detect_intent("update the config loader") == "modify_feature"

    def test_add_test(self) -> None:
        # only test-intent keywords present → add_test (no higher-priority match)
        assert self._a()._detect_intent("improve the test coverage") == "add_test"

    def test_general_fallback(self) -> None:
        assert self._a()._detect_intent("hello world nothing here") == "general"

    def test_priority_create_beats_test(self) -> None:
        # "add a test" → create_feature (0.9) outranks add_test (0.5)
        assert self._a()._detect_intent("add a unit test") == "create_feature"

    def test_sa_b1_address_not_create(self) -> None:
        # 'add' must NOT match inside 'address' → no create_feature
        assert self._a()._detect_intent("update the address padding field") != "create_feature"

    def test_sa_b1_suffix_not_fix(self) -> None:
        # 'fix' must NOT match inside 'suffix'/'prefix'
        assert self._a()._detect_intent("rename the suffix and prefix tokens") != "fix_bug"

    def test_sa_b1_latest_not_test(self) -> None:
        # 'test' must NOT match inside 'latest'/'contest'
        assert self._a()._detect_intent("show the latest contest results") != "add_test"

    def test_sa_b1_debug_not_bug(self) -> None:
        # 'bug' must NOT match inside 'debug'/'buggy'
        assert self._a()._detect_intent("help me debug the buggy parser") != "fix_bug"

    def test_cumulative_score_picks_highest(self) -> None:
        # multiple fix_bug keywords accumulate score
        assert self._a()._detect_intent("fix the broken regression bug") == "fix_bug"


# ============================================================
# Feature detection
# ============================================================


class TestDetectFeature:
    def _a(self, repo_root: str = ".") -> SmartRequestAnalyzer:
        return SmartRequestAnalyzer(repo_root)

    @pytest.mark.parametrize(
        "text,feature",
        [
            ("add a login screen", "login"),
            ("build the signup form", "signup"),
            ("implement logout", "logout"),
            ("reset my password", "password"),
            ("manage the user account", "user"),
            ("edit the profile page", "profile"),
            ("create a dashboard", "dashboard"),
            ("admin panel access", "admin"),
            ("new api endpoint", "api"),
            ("connect the database", "database"),
            ("add authorization", "auth"),
            ("show line numbers in editor", "editor"),
            ("redesign the ui", "ui"),
        ],
    )
    def test_feature_detected(self, text: str, feature: str) -> None:
        assert self._a()._detect_feature(text.lower()) == feature

    def test_no_feature_returns_none(self) -> None:
        assert self._a()._detect_feature("hello world") is None

    def test_sa_b1_guide_fluid_not_ui(self) -> None:
        # 'ui' must NOT match inside 'guide'/'fluid'/'build'
        assert self._a()._detect_feature("please build the guide for fluid layout") is None

    def test_sa_b1_rapid_capital_not_api(self) -> None:
        # 'api' must NOT match inside 'rapid'/'capital'
        assert self._a()._detect_feature("this is a rapid capital inquiry") is None

    def test_signin_separate_from_sign_in(self) -> None:
        # 'signin' keyword matches the single word, not 'sign in' phrase
        assert self._a()._detect_feature("use the signin button") == "login"
        assert self._a()._detect_feature("please sign in now") == "login"

    def test_first_match_wins(self) -> None:
        # iteration order: first matching feature pattern returns
        result = self._a()._detect_feature("login and signup")
        assert result in ("login", "signup")


# ============================================================
# Tech stack detection (file-system based — use tmp_path)
# ============================================================


class TestTechStack:
    def test_django_via_manage_py(self, tmp_path: Path) -> None:
        (tmp_path / "manage.py").write_text("")
        assert "django" in SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack()

    def test_react_via_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text('{"dependencies": {"react": "17"}}')
        ts = SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack()
        assert "react" in ts

    def test_typescript_via_tsconfig(self, tmp_path: Path) -> None:
        (tmp_path / "tsconfig.json").write_text("{}")
        assert "typescript" in SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack()

    def test_content_pattern_in_main_py(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("import flask\napp = flask.Flask(__name__)")
        ts = SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack()
        assert "flask" in ts

    def test_empty_repo_no_stack(self, tmp_path: Path) -> None:
        assert SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack() == []

    def test_unreadable_file_is_skipped(self, tmp_path: Path) -> None:
        # a content-pattern probe that raises is swallowed (best-effort)
        (tmp_path / "main.py").write_text("import flask\n")
        analyzer = SmartRequestAnalyzer(str(tmp_path))
        # force read_text to raise -> detector skips without crashing
        with patch.object(Path, "read_text", side_effect=OSError("boom")):
            ts = analyzer._detect_tech_stack()
        assert ts == []  # nothing detected, no exception raised

    def test_outer_exception_swallowed(self, tmp_path: Path) -> None:
        # the outer try/except wraps the file-existence loop too; if
        # Path.exists raises, each detector is skipped without crashing.
        analyzer = SmartRequestAnalyzer(str(tmp_path))
        with patch.object(Path, "exists", side_effect=OSError("boom")):
            assert analyzer._detect_tech_stack() == []

    def test_key_files_read_at_most_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # SA-P1 regression: the content-pattern loop nested the file loop
        # INSIDE the pattern loop, re-reading all 4 probe files for every
        # pattern (worst case 14 patterns x 4 files = 56 read_text calls).
        # Each probe file must be read at most once per detection call.
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "package.json").write_text('{"name": "x"}')
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'")

        original_read_text = Path.read_text
        reads: dict[str, int] = {}

        def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
            reads[str(self)] = reads.get(str(self), 0) + 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)

        SmartRequestAnalyzer(str(tmp_path))._detect_tech_stack()

        for fname in ("main.py", "package.json", "setup.py", "pyproject.toml"):
            count = reads.get(str(tmp_path / fname), 0)
            assert count <= 1, f"{fname} read {count} times (expected at most 1)"
        assert sum(reads.values()) <= 4  # at most the 4 probe files, once each


# ============================================================
# File suggestion
# ============================================================


class TestSuggestFiles:
    def test_create_feature_ui_routes_to_generic_web(self) -> None:
        files, ops = SmartRequestAnalyzer(".")._suggest_files("create_feature", "ui", [])
        assert files  # non-empty
        assert all(op == "create_or_modify" for op in ops.values())

    def test_create_feature_react_routes_to_jsx(self) -> None:
        files, _ = SmartRequestAnalyzer(".")._suggest_files("create_feature", "ui", ["react"])
        assert any(f.endswith(".jsx") for f in files)

    def test_create_feature_with_web_framework_infers_ui(self) -> None:
        # feature None + django tech stack → infers 'ui'
        files, _ = SmartRequestAnalyzer(".")._suggest_files("create_feature", None, ["django"])
        assert files

    def test_create_feature_no_web_framework_no_feature_empty(self) -> None:
        files, ops = SmartRequestAnalyzer(".")._suggest_files("create_feature", None, [])
        assert files == []
        assert ops == {}

    def test_non_create_intent_empty(self) -> None:
        files, ops = SmartRequestAnalyzer(".")._suggest_files("fix_bug", "ui", ["react"])
        assert files == []
        assert ops == {}

    def test_detect_feature_from_context(self) -> None:
        a = SmartRequestAnalyzer(".")
        assert a._detect_feature_from_context("create_feature", ["flask"]) == "ui"
        assert a._detect_feature_from_context("create_feature", []) is None
        assert a._detect_feature_from_context("fix_bug", ["flask"]) is None

    def test_has_web_framework(self) -> None:
        assert SmartRequestAnalyzer._has_web_framework(["django"]) is True
        assert SmartRequestAnalyzer._has_web_framework(["react"]) is True
        assert SmartRequestAnalyzer._has_web_framework(["python"]) is False

    def test_suggest_ui_files_react_vs_generic(self) -> None:
        react = SmartRequestAnalyzer._suggest_ui_files(["react"])
        generic = SmartRequestAnalyzer._suggest_ui_files(["flask"])
        assert any(".jsx" in f for f in react)
        assert any(".css" in f for f in generic)


# ============================================================
# Request enhancement, planning, confidence
# ============================================================


class TestEnhanceRequest:
    def test_appends_feature_and_stack(self) -> None:
        out = SmartRequestAnalyzer(".")._enhance_request("do x", "modify_feature", "login", ["django"])
        assert "do x" in out
        assert "login" in out
        assert "django" in out

    def test_create_feature_adds_requirements(self) -> None:
        out = SmartRequestAnalyzer(".")._enhance_request("do x", "create_feature", None, [])
        assert "Requirements" in out
        assert "error handling" in out

    def test_create_feature_web_adds_mvc(self) -> None:
        out = SmartRequestAnalyzer(".")._enhance_request("do x", "create_feature", None, ["django"])
        assert "MVC" in out

    def test_no_feature_no_stack(self) -> None:
        out = SmartRequestAnalyzer(".")._enhance_request("do x", "general", None, [])
        assert out == "do x"


class TestNeedsPlanning:
    def test_more_than_one_file(self) -> None:
        assert SmartRequestAnalyzer._needs_planning("create_feature", ["a", "b"]) is True

    def test_one_file_no_planning(self) -> None:
        assert SmartRequestAnalyzer._needs_planning("create_feature", ["a"]) is False

    def test_zero_files_no_planning(self) -> None:
        assert SmartRequestAnalyzer._needs_planning("general", []) is False


class TestCalculateConfidence:
    def _a(self) -> SmartRequestAnalyzer:
        return SmartRequestAnalyzer(".")

    def test_general_no_signal(self) -> None:
        assert self._a()._calculate_confidence("general", None, []) == 0.0

    def test_intent_only(self) -> None:
        # non-general intent +0.3
        assert self._a()._calculate_confidence("refactor", None, []) == pytest.approx(0.3)

    def test_intent_and_feature(self) -> None:
        assert self._a()._calculate_confidence("refactor", "login", []) == pytest.approx(0.6)

    def test_create_feature_full(self) -> None:
        # create_feature: 0.3 (intent) + 0.3 (feature) + 0.2 (files) + 0.2 (specific) = 1.0
        assert self._a()._calculate_confidence("create_feature", "ui", ["a.js"]) == pytest.approx(1.0)

    def test_capped_at_one(self) -> None:
        assert self._a()._calculate_confidence("fix_bug", "login", ["a", "b"]) == pytest.approx(1.0)


# ============================================================
# Full analyze() integration
# ============================================================


class TestAnalyzeIntegration:
    def test_create_ui_feature_full_pipeline(self, tmp_path: Path) -> None:
        (tmp_path / "manage.py").write_text("")
        a = SmartRequestAnalyzer(str(tmp_path))
        result = a.analyze("Create a new login ui")
        assert isinstance(result, RequestAnalysis)
        assert result.intent == "create_feature"
        assert result.feature_name == "login"
        assert "django" in result.tech_stack
        assert result.suggested_files  # ui → web files
        assert result.needs_planning is True  # multiple files
        assert result.confidence == pytest.approx(1.0)

    def test_fix_bug_pipeline(self) -> None:
        result = SmartRequestAnalyzer(".").analyze("Fix the login bug")
        assert result.intent == "fix_bug"
        assert result.feature_name == "login"
        assert result.suggested_files == []  # not create_feature
        assert result.needs_planning is False

    def test_sa_b1_regression_guide_not_ui(self) -> None:
        result = SmartRequestAnalyzer(".").analyze("please build the guide for fluid layout")
        # 'build' still triggers create_feature, but feature must NOT be 'ui'
        assert result.feature_name != "ui"

    def test_sa_b1_regression_no_false_intent(self) -> None:
        result = SmartRequestAnalyzer(".").analyze("rename the suffix and prefix tokens")
        # must not be classified as fix_bug
        assert result.intent != "fix_bug"


# ============================================================
# Typed dataclass construction (coverage of dataclass fields)
# ============================================================


class TestDataclasses:
    def test_intent_rule_defaults(self) -> None:
        r = IntentClassifierRule(intent="x", keywords={"a"})
        assert r.priority == 1.0
        assert r.description == ""

    def test_feature_pattern_defaults(self) -> None:
        f = FeaturePattern(feature="x", keywords={"a"})
        assert f.weight == 1.0

    def test_tech_detector_defaults(self) -> None:
        t = TechDetector(tech="x")
        assert t.files == ()
        assert t.content_patterns == ()

    def test_request_analysis_defaults(self) -> None:
        r = RequestAnalysis(original_request="hi")
        assert r.intent == "general"
        assert r.feature_name is None
        assert r.suggested_files == []
        assert r.confidence == 0.0
        assert r.needs_planning is False
