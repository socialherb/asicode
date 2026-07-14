"""Cold-start strategy priors — shared by the PLANNER lane strategy selector
and the live weight-learning cold-start bias (agent/weight_learning.py).

Lives in _editor_core/common (retained in the CLI-only distribution) so the
live consumer keeps working when the lane/ package is excluded."""
from __future__ import annotations

# Repair_burden is on a 0-3 scale (none=0, low=1, medium=2, high=3).
_COLD_PRIORS: dict[str, dict[str, float]] = {
    "generic_create": {
        "repair_burden": 0.50,
        "success_rate":  0.80,
        "contract_risk": 0.30,
        "complexity":    0.30,
        "cost":          0.20,
    },
    "reference_bound_create": {
        "repair_burden": 0.30,
        "success_rate":  0.85,
        "contract_risk": 0.10,
        "complexity":    0.40,
        "cost":          0.25,
    },
    "symbol_guided_create": {
        "repair_burden": 0.35,
        "success_rate":  0.85,
        "contract_risk": 0.15,
        "complexity":    0.35,
        "cost":          0.20,
    },
    "test_aware_create": {
        "repair_burden": 0.40,
        "success_rate":  0.82,
        "contract_risk": 0.20,
        "complexity":    0.35,
        "cost":          0.25,
    },
    # P11: Modify/extend strategies
    "symbol_edit": {
        "repair_burden": 0.25,
        "success_rate":  0.85,
        "contract_risk": 0.15,
        "complexity":    0.20,
        "cost":          0.15,
    },
    "minimal_patch": {
        "repair_burden": 0.15,
        "success_rate":  0.88,
        "contract_risk": 0.10,
        "complexity":    0.10,
        "cost":          0.10,
    },
    "refactor": {
        "repair_burden": 0.60,
        "success_rate":  0.70,
        "contract_risk": 0.35,
        "complexity":    0.55,
        "cost":          0.30,
    },
    "test_first": {
        "repair_burden": 0.35,
        "success_rate":  0.80,
        "contract_risk": 0.20,
        "complexity":    0.30,
        "cost":          0.25,
    },
}
