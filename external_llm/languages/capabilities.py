from __future__ import annotations

from enum import Enum

from .models import LanguageId


class AnalysisCapability(str, Enum):
    """Analysis capabilities that languages may or may not support.

    Each value represents a specific analysis subsystem.  Languages declare
    which capabilities they support via ``_LANGUAGE_CAPABILITIES``.
    Subsystems use ``filter_by_capability()`` or ``is_supported()`` instead
    of scattering ``LanguageId.PYTHON`` checks throughout the codebase.
    """

    # ── Scanners ────────────────────────────────────────────────────────────
    DEAD_CODE = "dead_code"
    DUPLICATE_DEFINITION = "duplicate_definition"
    UNUSED_IMPORT = "unused_import"
    CONTRADICTORY_LOGIC = "contradictory_logic"
    AST_SIMILARITY = "ast_similarity"
    CONTAINER_REACHABILITY = "container_reachability"
    PUBLIC_DEAD_CODE = "public_dead_code"

    # ── Non-scanner subsystems ──────────────────────────────────────────────
    IMPORT_RESOLUTION = "import_resolution"
    IMPORT_INSERTION = "import_insertion"
    IMPORT_NORMALIZATION = "import_normalization"
    F821_IMPORT_RESOLUTION = "f821_import_resolution"
    LINEAGE = "lineage"
    CONTEXT_BUILDING = "context_building"
    OUTPUT_NORMALIZATION = "output_normalization"
    SYMBOL_INDEX = "symbol_index"
    REPAIR = "repair"
    RUNTIME_GATE = "runtime_gate"
    VERIFICATION = "verification"
    EXECUTOR_LINEAGE = "executor_lineage"
    AGENT_CONTEXT = "agent_context"
    TARGET_RESOLUTION = "target_resolution"
    PLANNER_SPEC = "planner_spec"
    EXECUTOR = "executor"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    GRAPH_ANALYSIS = "graph_analysis"
    LEARNING = "learning"
    INTELLIGENT_SERVICE = "intelligent_service"


# ── Language → capability support matrix ────────────────────────────────
# Single source of truth: add a language here once, and all subsystems
# that use filter_by_capability() automatically include/exclude it.

_LANGUAGE_CAPABILITIES: dict[LanguageId, set[AnalysisCapability]] = {
    LanguageId.PYTHON: {
        # All analysis subsystems support Python
        cap for cap in AnalysisCapability
    },
    LanguageId.TYPESCRIPT: {
        # TS/JS support tree-sitter based analysis
        AnalysisCapability.CONTRADICTORY_LOGIC,
        AnalysisCapability.AST_SIMILARITY,
        AnalysisCapability.EXECUTOR,
        AnalysisCapability.OUTPUT_NORMALIZATION,
        AnalysisCapability.REPAIR,
        AnalysisCapability.RUNTIME_GATE,
        AnalysisCapability.VERIFICATION,
        AnalysisCapability.AGENT_CONTEXT,
        AnalysisCapability.TARGET_RESOLUTION,
        AnalysisCapability.PLANNER_SPEC,
        AnalysisCapability.GRAPH_ANALYSIS,
        AnalysisCapability.SEMANTIC_ANALYSIS,
        AnalysisCapability.LEARNING,
        AnalysisCapability.INTELLIGENT_SERVICE,
        AnalysisCapability.LINEAGE,
        AnalysisCapability.DEAD_CODE,
        AnalysisCapability.DUPLICATE_DEFINITION,
        AnalysisCapability.PUBLIC_DEAD_CODE,
    },
    LanguageId.JAVASCRIPT: {
        AnalysisCapability.CONTRADICTORY_LOGIC,
        AnalysisCapability.AST_SIMILARITY,
        AnalysisCapability.EXECUTOR,
        AnalysisCapability.OUTPUT_NORMALIZATION,
        AnalysisCapability.REPAIR,
        AnalysisCapability.RUNTIME_GATE,
        AnalysisCapability.VERIFICATION,
        AnalysisCapability.AGENT_CONTEXT,
        AnalysisCapability.TARGET_RESOLUTION,
        AnalysisCapability.PLANNER_SPEC,
        AnalysisCapability.GRAPH_ANALYSIS,
        AnalysisCapability.SEMANTIC_ANALYSIS,
        AnalysisCapability.LEARNING,
        AnalysisCapability.INTELLIGENT_SERVICE,
        AnalysisCapability.LINEAGE,
        AnalysisCapability.DEAD_CODE,
        AnalysisCapability.DUPLICATE_DEFINITION,
        AnalysisCapability.PUBLIC_DEAD_CODE,
    },
    # Go/Java/Kotlin: capabilities can be added as tree-sitter support grows
}


def filter_by_capability(
    file_paths: list[str],
    capability: AnalysisCapability,
) -> list[str]:
    """Return only file paths whose language supports *capability*.

    Example::

        py_files = filter_by_capability(all_files, AnalysisCapability.IMPORT_RESOLUTION)
    """
    return [
        p for p in file_paths
        if capability in _LANGUAGE_CAPABILITIES.get(LanguageId.from_path(p), set())
    ]


def is_supported(file_path: str, capability: AnalysisCapability) -> bool:
    """Return True when *file_path*'s language supports *capability*.

    Example::

        if not is_supported(op.path, AnalysisCapability.IMPORT_RESOLUTION):
            continue
    """
    return capability in _LANGUAGE_CAPABILITIES.get(
        LanguageId.from_path(file_path), set(),
    )
