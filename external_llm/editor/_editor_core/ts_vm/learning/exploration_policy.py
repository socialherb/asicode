"""exploration_policy.py — Exploration strategies for strategy selection.

Prevents the system from getting stuck in local optima by occasionally
trying non-best strategies via adaptive ε-greedy: ε decays as total runs
increase (explore less over time).
"""
from __future__ import annotations

import random


class AdaptiveEpsilon:
    """ε decays as experience grows: explore a lot early, exploit later.

    ε(n) = max(min_eps, base / (1 + n / decay_rate))

    With defaults: ε starts at 0.2, halves after 50 runs, floors at 0.05.
    """

    def __init__(
        self,
        base: float = 0.2,
        min_eps: float = 0.05,
        decay_rate: float = 50.0,
        seed: int | None = None,
    ):
        self._base = base
        self._min_eps = min_eps
        self._decay_rate = decay_rate
        self._rng = random.Random(seed)

    def epsilon(self, total_runs: int) -> float:
        return max(self._min_eps, self._base / (1 + total_runs / self._decay_rate))

    def select(self, ranked: list[str], total_runs: int) -> str:
        if not ranked:
            return ""
        if len(ranked) == 1:
            return ranked[0]
        eps = self.epsilon(total_runs)
        if self._rng.random() < eps:
            return self._rng.choice(ranked)
        return ranked[0]
