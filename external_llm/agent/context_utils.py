"""P11.1: Shared context key builder — used by executor, selector, distiller."""

from .config.thresholds import config as _cfg


def build_context_key(failure_class: str = "", mode: str = "") -> str:
    """Build a canonical context key for learning lookups.

    Must be called identically in operation_executor (recording) and
    strategy_selector (lookup) to ensure distilled rules match.
    """
    if failure_class:
        return f"{failure_class}:{mode}" if mode else failure_class
    return mode or "unknown"


def adaptive_distill_threshold(sample_count: int) -> float:
    """P11.1: Higher threshold when data is scarce, lower when confident.

    Thresholds are defined in config/thresholds.py (ScoreThresholds) as:
      DISTILL_SPARSE_LIMIT, DISTILL_MODERATE_LIMIT,
      DISTILL_THRESHOLD_SPARSE/MODERATE/CONFIDENT
    """
    _sc = _cfg.scores
    if sample_count < _sc.DISTILL_SPARSE_LIMIT:
        return _sc.DISTILL_THRESHOLD_SPARSE
    if sample_count < _sc.DISTILL_MODERATE_LIMIT:
        return _sc.DISTILL_THRESHOLD_MODERATE
    return _sc.DISTILL_THRESHOLD_CONFIDENT
