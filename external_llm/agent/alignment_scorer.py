"""P7 full: Alignment scorer — deterministic, LLM-free.

Computes how well the execution result aligns with the original intent.
Sits above verification: verification = "code isn't broken",
alignment = "code matches what was requested".

Three scoring dimensions:
  structural (0.3) — syntax, forbidden tokens, required symbols
  semantic   (0.4) — semantic verification, caller safety, signature
  intent     (0.3) — plan spec vs actual changes (symbols, tokens, diff)
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from typing import Any

from ..languages import LanguageId

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    score: float  # 0.0 ~ 1.0
    breakdown: dict[str, float]  # structural, semantic, intent
    issues: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "issues": self.issues,
        }


# ── Weights ────────────────────────────────────────────────────────────

W_STRUCTURAL = 0.3
W_SEMANTIC = 0.4
W_INTENT = 0.3


class AlignmentScorer:
    """Deterministic alignment scoring — no LLM calls."""

    def compute(
        self,
        plan,  # OperationPlan
        exec_info: dict[str, Any],
        verification_result,  # VerificationResult (dataclass)
        repo_root: str = "",
    ) -> AlignmentResult:
        issues: list[str] = []

        structural = self._score_structural(verification_result, issues)
        semantic = self._score_semantic(verification_result, exec_info, issues)
        intent = self._score_intent(plan, exec_info, verification_result, repo_root, issues)

        score = (
            W_STRUCTURAL * structural
            + W_SEMANTIC * semantic
            + W_INTENT * intent
        )

        result = AlignmentResult(
            score=score,
            breakdown={
                "structural": structural,
                "semantic": semantic,
                "intent": intent,
            },
            issues=issues,
        )

        logger.info(
            "alignment: score=%.3f (structural=%.2f semantic=%.2f intent=%.2f) issues=%d",
            score, structural, semantic, intent, len(issues),
        )
        return result

    # ── Structural (0.3) ─────────────────────────────────────────────

    def _score_structural(self, vr, issues: list[str]) -> float:
        """Verification-result-based structural checks."""
        score = 1.0
        penalties = 0.0

        # syntax_ok is binary hard gate
        if not getattr(vr, 'syntax_ok', True):
            penalties += 0.5
            issues.append("structural: syntax error")

        # blocking_reasons severity
        blocking = getattr(vr, 'blocking_reasons', []) or []
        if blocking:
            # Each blocking reason costs 0.15, capped at 0.5
            _pen = min(len(blocking) * 0.15, 0.5)
            penalties += _pen
            issues.append(f"structural: {len(blocking)} blocking reason(s)")

        # warnings are minor
        warnings = getattr(vr, 'warnings', []) or []
        if warnings:
            penalties += min(len(warnings) * 0.03, 0.1)

        return max(0.0, score - penalties)

    # ── Semantic (0.4) ───────────────────────────────────────────────

    def _score_semantic(self, vr, exec_info: dict, issues: list[str]) -> float:
        """Semantic verification + caller safety."""
        score = 1.0
        penalties = 0.0

        if not getattr(vr, 'semantic_ok', True):
            # Reduced penalty if the only semantic issue is a safe class shape change
            _vr_details = getattr(vr, 'details', {}) or {}
            _all_codes = []
            for _fp, _fd in _vr_details.items():
                if isinstance(_fd, dict):
                    _all_codes.extend(_fd.get("issue_codes", []))
            _hard_blocking = {'ast_parse_failed', 'symbol_removed', 'signature_changed', 'class_shape_changed'}
            _only_safe = 'class_shape_changed_safe' in _all_codes and not any(c in _hard_blocking for c in _all_codes)
            if _only_safe:
                penalties += 0.05
                issues.append("semantic: class_shape_changed_safe (intent-allowed)")
            else:
                penalties += 0.3
                issues.append("semantic: semantic verification failed")

        if not getattr(vr, 'acceptance_ok', True):
            penalties += 0.2
            issues.append("semantic: acceptance check failed")

        # Per-file semantic details
        details = getattr(vr, 'details', {}) or {}
        for fpath, fdetail in details.items():
            if not isinstance(fdetail, dict):
                continue
            # signature_preserved check
            if fdetail.get("signature_preserved") is False:
                penalties += 0.15
                issues.append(f"semantic: signature mismatch in {fpath}")
            # caller impact
            if fdetail.get("caller_impact_detected"):
                penalties += 0.1
                issues.append(f"semantic: caller impact in {fpath}")
            # semantic blocking at file level
            file_blocking = fdetail.get("blocking_reasons", [])
            if file_blocking:
                penalties += min(len(file_blocking) * 0.1, 0.2)

        return max(0.0, score - penalties)

    # ── Intent (0.3) ─────────────────────────────────────────────────

    def _score_intent(
        self, plan, exec_info: dict, vr, repo_root: str, issues: list[str],
    ) -> float:
        """Plan spec vs actual result alignment.

        Two scoring tiers:
          Mechanistic (checks 1, 5): "did ops execute correctly?"
          Spec-driven (checks 2-4, 6): "does the result match the spec?"

        Mechanistic checks are excluded when their outcome is explained by
        already_satisfied / no-failure states — penalizing correct idempotent
        behaviour is noise, not signal.
        """
        checks_total = 0
        checks_passed = 0

        metadata = plan.metadata if isinstance(getattr(plan, 'metadata', None), dict) else {}
        modified_files = exec_info.get("modified_files", [])

        # ── Pre-compute execution state for mechanistic checks ──────────
        _raw = exec_info.get("raw", {})
        _state = _raw.get("state") if isinstance(_raw, dict) else None
        _completed_ops_dict = getattr(_state, 'completed_ops', {}) if _state else {}
        _failed_ops = getattr(_state, 'failed_ops', {}) if _state else {}
        _n_failed = len(_failed_ops)

        # Build set of file paths whose ops were already_satisfied
        _already_satisfied_paths: set = set()
        if _completed_ops_dict:
            _op_id_to_path = {op.id: getattr(op, 'path', '') for op in plan.operations}
            for _oid, _result in _completed_ops_dict.items():
                if isinstance(_result, dict) and _result.get("already_satisfied"):
                    _path = _op_id_to_path.get(_oid, '')
                    if _path:
                        _already_satisfied_paths.add(_path)

        # ── 1. Were the target files actually modified? ─────────────────
        # Skip already_satisfied paths — they intentionally produce no diff.
        _op_paths = set()
        for op in plan.operations:
            p = getattr(op, 'path', None)
            if p:
                _op_paths.add(p)
        _check_paths = _op_paths - _already_satisfied_paths
        if _check_paths:
            _mod_set = set()
            for mf in modified_files:
                _mod_set.add(mf)
                if repo_root and os.path.isabs(mf):
                    _mod_set.add(os.path.relpath(mf, repo_root))
            for op_path in _check_paths:
                checks_total += 1
                _abs = os.path.join(repo_root, op_path) if repo_root and not os.path.isabs(op_path) else op_path
                if op_path in _mod_set or _abs in _mod_set:
                    checks_passed += 1
                else:
                    issues.append(f"intent: target file not modified: {op_path}")

        # ── 2. Required symbols created? ────────────────────────────────
        _verif = metadata.get("verification", {})
        if isinstance(_verif, dict):
            for sym in (_verif.get("required_symbols") or []):
                checks_total += 1
                if self._symbol_exists_in_files(sym, modified_files, repo_root):
                    checks_passed += 1
                else:
                    issues.append(f"intent: required symbol missing: {sym}")

        # ── 3. Required tokens present? ─────────────────────────────────
        if isinstance(_verif, dict):
            for token in (_verif.get("required_tokens") or []):
                checks_total += 1
                if self._token_in_files(token, modified_files, repo_root):
                    checks_passed += 1
                else:
                    issues.append(f"intent: required token missing: {token}")

        # ── 4. Forbidden tokens absent? ─────────────────────────────────
        if isinstance(_verif, dict):
            for token in (_verif.get("forbidden_tokens") or []):
                checks_total += 1
                if not self._token_in_files(token, modified_files, repo_root):
                    checks_passed += 1
                else:
                    issues.append(f"intent: forbidden token present: {token}")

        # ── 5. Operation completion ─────────────────────────────────────
        # Simplified: if no ops failed, completion is perfect.
        # Only penalize when actual failures exist.
        if _state and len(plan.operations) > 0:
            checks_total += 1
            if _n_failed == 0:
                checks_passed += 1
            else:
                _n_completed = len(_completed_ops_dict)
                _n_ops = len(plan.operations)
                if _n_completed / _n_ops >= 0.8:
                    checks_passed += 1
                else:
                    issues.append(
                        f"intent: low op completion {_n_completed}/{_n_ops} ({_n_failed} failed)"
                    )

        # ── 6. Min diff lines ──────────────────────────────────────────
        _min_diff = _verif.get("min_diff_lines", 0) if isinstance(_verif, dict) else 0
        if _min_diff > 0:
            checks_total += 1
            _diff_lines = self._count_diff_lines(modified_files, repo_root)
            if _diff_lines >= _min_diff:
                checks_passed += 1
            else:
                issues.append(f"intent: diff too small {_diff_lines}<{_min_diff}")

        if checks_total == 0:
            return 1.0  # No spec to check against → assume aligned

        _score = checks_passed / checks_total
        logger.debug(
            "intent detail: %d/%d passed (files=%d already_satisfied=%d failed_ops=%d)",
            checks_passed, checks_total, len(_check_paths),
            len(_already_satisfied_paths), _n_failed,
        )
        return _score

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _symbol_exists_in_files(
        symbol_name: str, files: list, repo_root: str,
    ) -> bool:
        for fp in files:
            abs_fp = os.path.join(repo_root, fp) if repo_root and not os.path.isabs(fp) else fp
            if not os.path.isfile(abs_fp) or LanguageId.from_path(abs_fp) is not LanguageId.PYTHON:
                continue
            try:
                with open(abs_fp, encoding='utf-8', errors='replace') as f:
                    tree = ast.parse(f.read())
                for node in tree.body:
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == symbol_name:
                            return True
            except (OSError, SyntaxError, ValueError):
                continue
        return False

    @staticmethod
    def _token_in_files(token: str, files: list, repo_root: str) -> bool:
        for fp in files:
            abs_fp = os.path.join(repo_root, fp) if repo_root and not os.path.isabs(fp) else fp
            if not os.path.isfile(abs_fp):
                continue
            try:
                with open(abs_fp, encoding='utf-8', errors='replace') as f:
                    if token in f.read():
                        return True
            except OSError:
                continue
        return False

    @staticmethod
    def _count_diff_lines(files: list, repo_root: str) -> int:
        """Count total lines changed across modified files."""
        total = 0
        for fp in files:
            abs_fp = os.path.join(repo_root, fp) if repo_root and not os.path.isabs(fp) else fp
            if os.path.isfile(abs_fp):
                try:
                    with open(abs_fp, encoding='utf-8', errors='replace') as f:
                        # Generator count avoids materializing the whole file as a
                        # list of strings (O(1) memory vs O(file) for readlines()).
                        total += sum(1 for _ in f)
                except OSError:
                    pass
        return total
