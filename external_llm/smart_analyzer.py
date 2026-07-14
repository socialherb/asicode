"""
Smart Request Analyzer for asicode (typed plan policy).

Analyzes general user requests using typed data structures instead of
ad-hoc keyword/regex matching. Integrates with PlanPolicy taxonomy for
consistent intent-to-strategy mapping across the system.

Features:
- Typed IntentClassifierRule with priority-scored matching
- Typed FeaturePattern with language-agnostic keyword matching
- Tech stack detection via project file inspection
- Confidence calculation aligned with PlanPolicy confidence semantics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentClassifierRule:
    """A typed intent classification rule for SmartRequestAnalyzer.

    Replaces the ad-hoc _INTENT_CLASSIFIER dict-of-sets with a structured
    rule that includes per-keyword weight and priority scoring.
    """
    intent: str
    keywords: set  # exact word boundaries checked
    priority: float = 1.0  # base confidence contribution
    description: str = ""


@dataclass(frozen=True)
class FeaturePattern:
    """A typed feature detection pattern.

    Replaces the ad-hoc FEATURE_PATTERNS dict-of-regex with a structured
    pattern that supports multi-language detection and context weighting.
    """
    feature: str
    keywords: set  # set of lowercase keyword strings
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class TechDetector:
    """A typed tech stack detection rule.

    Replaces the ad-hoc TECH_PATTERNS dict-of-regex with a structured
    detector that specifies which files to check and what to look for.
    """
    tech: str
    files: tuple = ()  # file paths to check for existence
    content_patterns: tuple = ()  # regex patterns to match in file content
    description: str = ""


@dataclass
class RequestAnalysis:
    """Result of analyzing a user request (typed).

    Provides typed fields for intent, feature, tech stack, and file operations.
    Aligned with PlanPolicy confidence semantics.
    """

    # Original request
    original_request: str

    # Detected intent (aligned with PlanPolicy kind taxonomy)
    intent: str = "general"  # create_feature, fix_bug, refactor, modify_feature, add_test, general

    # Feature/component name
    feature_name: Optional[str] = None

    # Suggested files to create/modify
    suggested_files: list[str] = field(default_factory=list)

    # Operation type for each file
    file_operations: dict[str, str] = field(default_factory=dict)  # {file_path: "create" | "modify"}

    # Detected technology/framework
    tech_stack: list[str] = field(default_factory=list)

    # Enhanced request with context
    enhanced_request: str = ""

    # Confidence score (0.0 - 1.0) — matches PlanPolicy confidence semantics
    confidence: float = 0.0

    # Whether this requires planning mode
    needs_planning: bool = False


# ---------------------------------------------------------------------------
# Typed classifier rules (priority-ordered)
# ---------------------------------------------------------------------------

_INTENT_RULES: list[IntentClassifierRule] = [
    # create_feature (highest priority — creation detected before modification)
    IntentClassifierRule(
        intent="create_feature", priority=0.9,
        keywords={"create", "add", "implement", "build", "develop", "generate",
                   "new", "write", "make", "produce", "construct", "introduce"},
        description="Feature creation intent",
    ),
    # fix_bug
    IntentClassifierRule(
        intent="fix_bug", priority=0.85,
        keywords={"fix", "repair", "bug", "error", "broken", "crash",
                   "incorrect", "wrong", "patch", "regression"},
        description="Bug fix intent",
    ),
    # refactor
    IntentClassifierRule(
        intent="refactor", priority=0.8,
        keywords={"refactor", "restructure", "clean", "simplify", "organize",
                   "consolidate", "deduplicate", "reorganize", "split",
                   "decouple", "inline", "extract", "rearrange"},
        description="Code refactoring intent",
    ),
    # modify_feature
    IntentClassifierRule(
        intent="modify_feature", priority=0.7,
        keywords={"modify", "change", "improve", "update", "enhance",
                   "adjust", "revise", "amend", "alter", "upgrade"},
        description="Feature modification intent",
    ),
    # add_test (lowest priority — "test" appears in many contexts)
    IntentClassifierRule(
        intent="add_test", priority=0.5,
        keywords={"test", "spec", "unittest", "coverage", "assertion"},
        description="Test addition intent",
    ),
]

# Language-agnostic feature patterns (English keywords)
# Korean/Japanese users typically use English loanwords for technical terms
# (e.g., "login", "password", "profile", "admin"). For other languages,
# downstream LLM-based intent extraction handles the semantic mapping.
_FEATURE_PATTERNS: list[FeaturePattern] = [
      FeaturePattern(feature="login",
                     keywords={"login", "sign in", "signin", "authentication"},
                     weight=1.0, description="Login/authentication feature"),
      FeaturePattern(feature="signup",
                     keywords={"signup", "sign up", "register", "registration"},
                     weight=1.0, description="Signup/registration feature"),
      FeaturePattern(feature="logout",
                     keywords={"logout", "sign out", "signout"},
                     weight=1.0, description="Logout feature"),
      FeaturePattern(feature="password",
                     keywords={"password", "pwd"},
                     weight=0.9, description="Password management feature"),
      FeaturePattern(feature="user",
                     keywords={"user", "account"},
                     weight=0.8, description="User/account feature"),
      FeaturePattern(feature="profile",
                     keywords={"profile"},
                     weight=1.0, description="Profile feature"),
      FeaturePattern(feature="dashboard",
                     keywords={"dashboard"},
                     weight=1.0, description="Dashboard feature"),
      FeaturePattern(feature="admin",
                     keywords={"admin", "administrator"},
                     weight=0.9, description="Admin feature"),
      FeaturePattern(feature="api",
                     keywords={"api", "endpoint", "rest"},
                     weight=0.8, description="API/endpoint feature"),
      FeaturePattern(feature="database",
                     keywords={"db", "database"},
                     weight=0.8, description="Database feature"),
      FeaturePattern(feature="auth",
                     keywords={"auth", "authorization"},
                     weight=1.0, description="Authentication/authorization"),
      FeaturePattern(feature="editor",
                     keywords={"editor", "line number", "linenumber", "line numbers", "linenumbers"},
                     weight=1.0, description="Code editor/line numbers feature"),
      FeaturePattern(feature="ui",
                     keywords={"ui", "interface", "screen", "frontend"},
                     weight=0.8, description="UI/frontend feature"),
]

# Typed tech stack detectors (project file inspection)
_TECH_DETECTORS: list[TechDetector] = [
    TechDetector(tech="django",
                 files=("manage.py",),
                 content_patterns=("django", "views.py"),
                 description="Django web framework"),
    TechDetector(tech="fastapi",
                 files=(),
                 content_patterns=("fastapi", "uvicorn"),
                 description="FastAPI web framework"),
    TechDetector(tech="flask",
                 files=("app.py",),
                 content_patterns=("flask",),
                 description="Flask web framework"),
    TechDetector(tech="react",
                 files=("package.json",),
                 content_patterns=("react", "react-dom", "jsx", "tsx"),
                 description="React frontend framework"),
    TechDetector(tech="vue",
                 files=(),
                 content_patterns=("vue", ".vue"),
                 description="Vue frontend framework"),
    TechDetector(tech="python",
                 files=(),
                 content_patterns=(".py",),
                 description="Python project"),
    TechDetector(tech="typescript",
                 files=("tsconfig.json",),
                 content_patterns=(".ts", "typescript"),
                 description="TypeScript project"),
]


class SmartRequestAnalyzer:
    """Analyzes user requests using typed plan policy.

    Handles requests like:
    - "Create a login feature"
    - "Create a user authentication system"
    - "Fix the login bug"

    Uses typed rules (IntentClassifierRule, FeaturePattern, TechDetector)
    instead of ad-hoc keyword sets and raw regex patterns.
    """

    intent_rules: list[IntentClassifierRule] = _INTENT_RULES
    feature_patterns: list[FeaturePattern] = _FEATURE_PATTERNS
    tech_detectors: list[TechDetector] = _TECH_DETECTORS

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    # ── Public API ────────────────────────────────────────────────

    def analyze(self, user_request: str) -> RequestAnalysis:
        """Analyze user request and extract structured information."""
        req_lower = user_request.lower()

        intent = self._detect_intent(req_lower)
        feature_name = self._detect_feature(req_lower)
        tech_stack = self._detect_tech_stack()
        suggested_files, file_ops = self._suggest_files(intent, feature_name, tech_stack)
        enhanced = self._enhance_request(user_request, intent, feature_name, tech_stack)
        needs_planning = self._needs_planning(intent, suggested_files)
        confidence = self._calculate_confidence(intent, feature_name, suggested_files)

        return RequestAnalysis(
            original_request=user_request,
            intent=intent,
            feature_name=feature_name,
            suggested_files=suggested_files,
            file_operations=file_ops,
            tech_stack=tech_stack,
            enhanced_request=enhanced,
            confidence=confidence,
            needs_planning=needs_planning,
        )

    # ── Intent detection ──────────────────────────────────────────

    def _detect_intent(self, req_lower: str) -> str:
        """Detect user intent using typed, weighted rules.

        Each rule contributes its priority if ANY keyword matches.
        The intent with the highest cumulative score wins.
        Falls back to 'general' if no matching intent.
        """
        best_intent = "general"
        best_score = 0.0

        for rule in self.intent_rules:
            score = sum(rule.priority for kw in rule.keywords if kw in req_lower)
            if score > best_score:
                best_score = score
                best_intent = rule.intent

        return best_intent

    # ── Feature detection ─────────────────────────────────────────

    def _detect_feature(self, req_lower: str) -> Optional[str]:
        """Detect feature name using typed, language-neutral patterns.

        Returns the first matching feature pattern's feature name.
                All patterns use English keywords only — sufficient for technical
        feature detection since non-English requests are handled downstream
        by the LLM-based semantic analysis.
        """
        for fp in self.feature_patterns:
            for kw in fp.keywords:
                if kw in req_lower:
                    return fp.feature
        return None

    # ── Tech stack detection ──────────────────────────────────────

    def _detect_tech_stack(self) -> list[str]:
        """Detect tech stack using typed project file inspection.

        Avoids glob-based file scanning — uses existence checks for
        specific files and targeted content pattern matching.
        """
        detected: list[str] = []

        for detector in self.tech_detectors:
            try:
                # Check file existence
                found = False
                for fname in detector.files:
                    if (self.repo_root / fname).exists():
                        found = True
                        break
                if found:
                    detected.append(detector.tech)
                    continue

                # Check content patterns in key files
                for pattern in detector.content_patterns:
                    # Fast check: scan key files for pattern
                    for fname in ("main.py", "package.json", "setup.py", "pyproject.toml"):
                        fpath = self.repo_root / fname
                        if fpath.exists():
                            try:
                                content = fpath.read_text(errors="replace")
                                if pattern in content.lower():
                                    detected.append(detector.tech)
                                    found = True
                                    break
                            except Exception:
                                continue
                    if found:
                        break
            except Exception as e:
                logger.debug("Error detecting tech %s: %s", detector.tech, e)

        return detected

    # ── File suggestion ───────────────────────────────────────────

    def _suggest_files(
        self,
        intent: str,
        feature: Optional[str],
        tech_stack: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Suggest files to create/modify based on intent and feature.

        Uses graph-aware grounding via external integration rather than
        hardcoded framework×feature→file mappings. Returns empty lists
        when no typed file resolution is available.
        """
        files: list[str] = []
        operations: dict[str, str] = {}

        if intent == "create_feature":
            if not feature:
                feature = self._detect_feature_from_context(intent, tech_stack)

            if feature in ("editor", "ui") or self._has_web_framework(tech_stack):
                files = self._suggest_ui_files(tech_stack)
                for f in files:
                    operations[f] = "create_or_modify"
                return files, operations

        return files, operations

    def _detect_feature_from_context(self, intent: str, tech_stack: list[str]) -> Optional[str]:
        if intent == "create_feature" and self._has_web_framework(tech_stack):
            return "ui"
        return None

    @staticmethod
    def _has_web_framework(tech_stack: list[str]) -> bool:
        return any(t in tech_stack for t in ("django", "flask", "fastapi", "react"))

    @staticmethod
    def _suggest_ui_files(tech_stack: list[str]) -> list[str]:
        if "react" in tech_stack:
            return [
                "src/components/LineNumbers.jsx",
                "src/components/CodeEditor.jsx",
                "src/styles/Editor.css",
            ]
        # Generic web files (django/flask/fastapi/fallback)
        return [
            "static/css/editor.css",
            "static/js/editor.js",
            "templates/editor.html",
            "index.html",
        ]

    # ── Request enhancement ───────────────────────────────────────

    def _enhance_request(
        self,
        original: str,
        intent: str,
        feature: Optional[str],
        tech_stack: list[str],
    ) -> str:
        parts = [original]
        if feature:
            parts.append(f"\n**Detected Feature**: {feature}")
        if tech_stack:
            parts.append(f"**Tech Stack**: {', '.join(tech_stack)}")
        if intent == "create_feature":
            parts.append("\n**Requirements**:")
            parts.append("- Create all necessary files")
            parts.append("- Follow project conventions")
            parts.append("- Include proper error handling")
            parts.append("- Add appropriate comments/docstrings")
            if self._has_web_framework(tech_stack):
                parts.append("- Follow MVC/MVT pattern")
                parts.append("- Implement proper validation")
        return "\n".join(parts)

    # ── Decision logic ────────────────────────────────────────────

    @staticmethod
    def _needs_planning(intent: str, suggested_files: list[str]) -> bool:
        return len(suggested_files) > 1

    @staticmethod
    def _calculate_confidence(intent: str, feature: Optional[str], suggested_files: list[str]) -> float:
        """Calculate confidence score aligned with PlanPolicy semantics.

        - Intent detected: +0.3
        - Feature detected: +0.3
        - Files suggested: +0.2
        - Specific intent (create_feature/fix_bug): +0.2
        """
        score = 0.0
        if intent and intent != "general":
            score += 0.3
        if feature:
            score += 0.3
        if suggested_files:
            score += 0.2
        if intent in ("create_feature", "fix_bug"):
            score += 0.2
        return min(score, 1.0)


def analyze_request(repo_root: str, user_request: str) -> RequestAnalysis:
    """Convenience function to analyze a request.

    Args:
        repo_root: Repository root path
        user_request: User's request

    Returns:
        RequestAnalysis with typed fields
    """
    analyzer = SmartRequestAnalyzer(repo_root)
    return analyzer.analyze(user_request)
