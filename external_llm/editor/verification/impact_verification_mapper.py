"""
impact_verification_mapper.py — Maps PatchRiskEstimate into verification decisions.

P9: connects patch risk estimation to verification scope and test discovery.
All functions accept Optional inputs — return safe defaults when None.
"""
from __future__ import annotations
from typing import Any, Optional
def map_verification_scope(patch_risk: Optional[object]) -> str:
    """Map patch risk to a verification scope level.

    Returns: "narrow" | "standard" | "broad"
    """
    if patch_risk is None:
        return "standard"

    level = getattr(patch_risk, "overall_risk", "low")
    if level in ("critical", "high"):
        return "broad"
    if level == "medium":
        return "standard"
    return "narrow"


def map_impacted_tests(
    patch_risk: Optional[object],
    discovered_tests: list[Any],
) -> list[Any]:
    """Boost priority scores of discovered tests that match impacted areas.

    Modifies test targets in-place for efficiency. Returns the same list.

    Boosting rules:
      +0.15 if test matches an impacted file
      +0.20 if test matches an impacted symbol
      +0.10 if test matches a caution symbol
    Cap at 1.0.
    """
    if patch_risk is None or not discovered_tests:
        return discovered_tests

    impacted_files = set(getattr(patch_risk, "impacted_files", []))
    impacted_symbols = set(getattr(patch_risk, "impacted_symbols", []))
    caution_symbols = set(getattr(patch_risk, "caution_symbols", []))

    if not impacted_files and not impacted_symbols and not caution_symbols:
        return discovered_tests

    import os

    for t in discovered_tests:
        boosted = False
        test_path = getattr(t, "test_path", "")
        matched_symbols = getattr(t, "matched_symbols", [])
        matched_files = getattr(t, "matched_files", [])
        reason_codes = getattr(t, "reason_codes", [])
        score = getattr(t, "priority_score", 0.0)

        # Impacted file match (by basename)
        test_base = os.path.basename(test_path) if test_path else ""
        for imp_f in impacted_files:
            imp_base = os.path.basename(imp_f) if imp_f else ""
            if imp_base and (imp_base in test_base or test_base in imp_base):
                score = min(1.0, score + 0.15)
                if hasattr(t, "reason_codes") and "IMPACTED_FILE_MATCH" not in reason_codes:
                    reason_codes.append("IMPACTED_FILE_MATCH")
                boosted = True
                break
        # Also check matched_files
        if not boosted and matched_files:
            for mf in matched_files:
                if mf in impacted_files:
                    score = min(1.0, score + 0.15)
                    if hasattr(t, "reason_codes") and "IMPACTED_FILE_MATCH" not in reason_codes:
                        reason_codes.append("IMPACTED_FILE_MATCH")
                    boosted = True
                    break

        # Impacted symbol match
        for ms in matched_symbols:
            if ms in impacted_symbols:
                score = min(1.0, score + 0.20)
                if hasattr(t, "reason_codes") and "IMPACTED_SYMBOL_MATCH" not in reason_codes:
                    reason_codes.append("IMPACTED_SYMBOL_MATCH")
                break

        # Caution symbol proximity
        for ms in matched_symbols:
            if ms in caution_symbols:
                score = min(1.0, score + 0.10)
                if hasattr(t, "reason_codes") and "CAUTION_SYMBOL_PROXIMITY" not in reason_codes:
                    reason_codes.append("CAUTION_SYMBOL_PROXIMITY")
                break

        # Apply boosted score
        if hasattr(t, "priority_score"):
            t.priority_score = min(1.0, score)

    return discovered_tests


def expand_verification_targets(patch_risk: Optional[object]) -> dict[str, Any]:
    """Extract expanded verification target hints from patch risk.

    Returns a dict suitable for passing to VerificationSetBuilder or test finder:
      - extra_symbols: additional symbols to search for tests
      - extra_files: additional files to consider
      - caution_symbols: symbols needing careful verification
      - scope_hint: recommended scope level
    """
    if patch_risk is None:
        return {}

    result: dict[str, Any] = {}

    impacted_symbols = getattr(patch_risk, "impacted_symbols", [])
    if impacted_symbols:
        result["extra_symbols"] = list(impacted_symbols[:10])

    impacted_files = getattr(patch_risk, "impacted_files", [])
    if impacted_files:
        result["extra_files"] = list(impacted_files[:10])

    caution = getattr(patch_risk, "caution_symbols", [])
    if caution:
        result["caution_symbols"] = list(caution[:5])

    result["scope_hint"] = map_verification_scope(patch_risk)

    return result


def risk_to_verification_metadata(patch_risk: Optional[object]) -> dict[str, Any]:
    """Convert patch risk estimate into verification metadata dict.

    Suitable for vset.metadata["patch_risk"] = ...
    """
    if patch_risk is None:
        return {}

    return {
        "level": getattr(patch_risk, "overall_risk", "unknown"),
        "score": getattr(patch_risk, "risk_score", 0.0),
        "impacted_files": list(getattr(patch_risk, "impacted_files", []))[:10],
        "impacted_symbols": list(getattr(patch_risk, "impacted_symbols", []))[:10],
        "caution_symbols": list(getattr(patch_risk, "caution_symbols", []))[:5],
        "verification_scope": getattr(patch_risk, "verification_scope_hint", "standard"),
    }
