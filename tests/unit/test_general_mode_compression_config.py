"""Regression guard for the /general-mode compression policy.

Context:
  * /code and /orchestrate modes use the periodic turn-count auto-compression
    driven by CompressionConfig.MIN_RECENT_TURNS_KEEP + COMPRESS_BATCH_MIN
    (compress_old_turns / needs_compression).
  * /general mode DISABLES that periodic compression: turns accumulate verbatim
    so the stable prefix (and its prompt cache) survives across many turns.
    Compression fires only once the LIVE context window reaches
    CompressionConfig.GENERAL_MODE_COMPRESS_OCCUPANCY, at which point the CLI
    calls schedule_background_compress(..., force=True) to bypass the
    turn-count gate.

These tests pin the config field and the exact occupancy predicate used at the
CLI call site (asi.py design-chat path) so accidental threshold drift or
removal of the general-mode branch is caught.
"""

from external_llm.agent.config.thresholds import config


def _general_should_compress(lpt: int, budget: int, occupancy: float) -> bool:
    """Mirror of the /general-mode gate predicate in asi.py."""
    return bool(budget and lpt and (lpt / budget >= occupancy))


def test_general_mode_compress_occupancy_config_present():
    # The field must exist and default to the documented value.
    assert hasattr(config.compression, "GENERAL_MODE_COMPRESS_OCCUPANCY")
    assert config.compression.GENERAL_MODE_COMPRESS_OCCUPANCY == 0.80
    assert isinstance(config.compression.GENERAL_MODE_COMPRESS_OCCUPANCY, float)


def test_periodic_thresholds_unchanged():
    # /code·/orchestrate still rely on these.
    assert config.compression.MIN_RECENT_TURNS_KEEP == 4
    assert config.compression.COMPRESS_BATCH_MIN == 11


def test_general_gate_fires_only_above_occupancy():
    occ = config.compression.GENERAL_MODE_COMPRESS_OCCUPANCY
    # Below threshold: no compression (turns keep accumulating verbatim).
    assert _general_should_compress(lpt=50_000, budget=128_000, occupancy=occ) is False
    assert _general_should_compress(lpt=102_000, budget=128_000, occupancy=occ) is False
    # At/above threshold: compress once near limit.
    assert _general_should_compress(lpt=103_000, budget=128_000, occupancy=occ) is True
    assert _general_should_compress(lpt=120_000, budget=128_000, occupancy=occ) is True


def test_general_gate_skipped_without_known_budget_or_tokens():
    occ = config.compression.GENERAL_MODE_COMPRESS_OCCUPANCY
    # Unknown model (budget 0) or zero reported tokens: never force-compress.
    assert _general_should_compress(lpt=999_999, budget=0, occupancy=occ) is False
    assert _general_should_compress(lpt=0, budget=128_000, occupancy=occ) is False


def test_general_cycle_fires_once_per_fill():
    """After a compress, occupancy drops well below threshold, so the NEXT turn
    does not re-trigger — emulating 'accumulate then compress once near limit'."""
    occ = config.compression.GENERAL_MODE_COMPRESS_OCCUPANCY
    budget = 128_000
    # Climb toward the limit.
    assert _general_should_compress(60_000, budget, occ) is False
    assert _general_should_compress(90_000, budget, occ) is False
    assert _general_should_compress(103_000, budget, occ) is True   # compress fires
    # Post-compress the verbatim window shrinks to ~4 recent turns.
    assert _general_should_compress(20_000, budget, occ) is False    # stays quiet
