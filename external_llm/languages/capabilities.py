from __future__ import annotations

from enum import Enum

from .models import LanguageId


class AnalysisCapability(str, Enum):
    """Analysis capabilities that languages may or may not support.

    Each value represents a specific analysis subsystem.  Languages declare
    which capabilities they support via ``_LANGUAGE_CAPABILITIES``.
    Subsystems use ``is_supported()`` instead of scattering
    ``LanguageId.PYTHON`` checks throughout the codebase.
    """

    # ── Non-scanner subsystems ──────────────────────────────────────────────
    CONTEXT_BUILDING = "context_building"


# ── Language → capability support matrix ────────────────────────────────
# Single source of truth: add a language here once, and all subsystems
# that use is_supported() automatically include/exclude it.

_LANGUAGE_CAPABILITIES: dict[LanguageId, set[AnalysisCapability]] = {
    LanguageId.PYTHON: {
        AnalysisCapability.CONTEXT_BUILDING,
    },
    # Other languages: capabilities can be added as tree-sitter support grows
}


def is_supported(file_path: str, capability: AnalysisCapability) -> bool:
    """Return True when *file_path*'s language supports *capability*.

    Example::

        if not is_supported(op.path, AnalysisCapability.CONTEXT_BUILDING):
            continue
    """
    return capability in _LANGUAGE_CAPABILITIES.get(
        LanguageId.from_path(file_path),
        set(),
    )
