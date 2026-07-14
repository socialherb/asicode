"""strategy_abstraction.py — Language-specific → abstract strategy mapping.

Maps Python and TS/JS strategy names to language-neutral abstract
strategies, enabling cross-language knowledge transfer.

Two mapping tables:
    1. Strategy mapping: language-specific strategy → AbstractStrategy
    2. Intent mapping: language-specific intent → AbstractIntent

Both are static, deterministic, and extensible.
"""
from __future__ import annotations

from typing import Optional

from external_llm.editor.cross_language.models import (
    AbstractIntent,
    AbstractStrategy,
    CrossLanguageRecord,
    Language,
)

# ── Strategy Mappings ──────────────────────────────────────────────

# Python strategies → abstract
_PYTHON_STRATEGY_MAP: dict[str, AbstractStrategy] = {
    # ExperienceStore / StrategyExecutionStats strategies
    "generic_create": AbstractStrategy.TEMPLATE_BASED,
    "reference_bound_create": AbstractStrategy.TEMPLATE_BASED,
    "symbol_guided_create": AbstractStrategy.STRUCTURED_EDIT,
    "test_aware_create": AbstractStrategy.TEMPLATE_BASED,
    # Primitive-level strategies (from primitive_learning_updater)
    "c2_insert_verify_call": AbstractStrategy.STRUCTURED_EDIT,
    "c2_insert_user_lookup": AbstractStrategy.STRUCTURED_EDIT,
    "c2_insert_error_branch": AbstractStrategy.STRUCTURED_EDIT,
    "c2_insert_persistence": AbstractStrategy.STRUCTURED_EDIT,
    "d_fragment_generation": AbstractStrategy.TEMPLATE_BASED,
    "c2_fix_return_entity": AbstractStrategy.STRUCTURED_EDIT,
    "c2_insert_token": AbstractStrategy.STRUCTURED_EDIT,
    # Operation-level strategies
    "modify_symbol": AbstractStrategy.STRUCTURED_EDIT,
    "create_symbol": AbstractStrategy.TEMPLATE_BASED,
    "delete_symbol": AbstractStrategy.MINIMAL_CHANGE,
    "update_imports": AbstractStrategy.MINIMAL_CHANGE,
    "rename_symbol": AbstractStrategy.STRUCTURED_EDIT,
    "move_symbol": AbstractStrategy.CROSS_FILE,
    # Planner-level strategy names (used by strategy_execution_bias, unified_prioritize)
    "symbol_edit": AbstractStrategy.STRUCTURED_EDIT,
    "minimal_patch": AbstractStrategy.MINIMAL_CHANGE,
    "refactor": AbstractStrategy.CROSS_FILE,
    "test_first": AbstractStrategy.TEST_DRIVEN,
}

# TS strategies → abstract
_TS_STRATEGY_MAP: dict[str, AbstractStrategy] = {
    "minimal_patch": AbstractStrategy.MINIMAL_CHANGE,
    "symbol_edit": AbstractStrategy.STRUCTURED_EDIT,
    "cross_file": AbstractStrategy.CROSS_FILE,
    "graph_repair": AbstractStrategy.ERROR_DRIVEN_REPAIR,
    "body_replace": AbstractStrategy.BODY_REPLACEMENT,
}

# Combined lookup: (language, strategy) → abstract
_STRATEGY_MAP: dict[tuple[str, str], AbstractStrategy] = {}
for _k, _v in _PYTHON_STRATEGY_MAP.items():
    _STRATEGY_MAP[(Language.PYTHON, _k)] = _v
for _k, _v in _TS_STRATEGY_MAP.items():
    _STRATEGY_MAP[(Language.TYPESCRIPT, _k)] = _v

# ── Intent Mappings ────────────────────────────────────────────────

# Python OperationKind → abstract
_PYTHON_INTENT_MAP: dict[str, AbstractIntent] = {
    "MODIFY_SYMBOL": AbstractIntent.MODIFY_SYMBOL,
    "CREATE_SYMBOL": AbstractIntent.ADD_SYMBOL,
    "DELETE_SYMBOL": AbstractIntent.DELETE_SYMBOL,
    "UPDATE_IMPORTS": AbstractIntent.ADD_IMPORT,
    "RENAME_SYMBOL": AbstractIntent.RENAME_SYMBOL,
    "CREATE_FILE": AbstractIntent.CREATE_FILE,
    # Lowercase variants
    "modify_symbol": AbstractIntent.MODIFY_SYMBOL,
    "create_symbol": AbstractIntent.ADD_SYMBOL,
    "delete_symbol": AbstractIntent.DELETE_SYMBOL,
    "update_imports": AbstractIntent.ADD_IMPORT,
    "rename_symbol": AbstractIntent.RENAME_SYMBOL,
    "create_file": AbstractIntent.CREATE_FILE,
}

# TS IntentKind → abstract
_TS_INTENT_MAP: dict[str, AbstractIntent] = {
    "add_function": AbstractIntent.ADD_SYMBOL,
    "modify_function": AbstractIntent.MODIFY_SYMBOL,
    "rename_symbol": AbstractIntent.RENAME_SYMBOL,
    "delete_symbol": AbstractIntent.DELETE_SYMBOL,
    "move_symbol": AbstractIntent.MOVE_SYMBOL,
    "add_import": AbstractIntent.ADD_IMPORT,
    "refactor": AbstractIntent.REFACTOR,
    "fix_error": AbstractIntent.FIX_ERROR,
    "add_file": AbstractIntent.CREATE_FILE,
    "unknown": AbstractIntent.UNKNOWN,
}

_INTENT_MAP: dict[tuple[str, str], AbstractIntent] = {}
for _k, _v in _PYTHON_INTENT_MAP.items():
    _INTENT_MAP[(Language.PYTHON, _k)] = _v
for _k, _v in _TS_INTENT_MAP.items():
    _INTENT_MAP[(Language.TYPESCRIPT, _k)] = _v

# ── Reverse Mappings (abstract → language-specific) ────────────────

# For each (language, abstract_strategy), list candidate local strategies
_REVERSE_STRATEGY: dict[tuple[str, str], list[str]] = {}
for (_lang, _local), _abstract in _STRATEGY_MAP.items():
    key = (_lang, _abstract.value)
    _REVERSE_STRATEGY.setdefault(key, []).append(_local)


# ── Public API ─────────────────────────────────────────────────────

def abstract_strategy(
    language: str | Language,
    strategy: str,
) -> AbstractStrategy:
    """Map a language-specific strategy to its abstract equivalent."""
    lang = Language(language) if isinstance(language, str) else language
    return _STRATEGY_MAP.get(
        (lang, strategy), AbstractStrategy.UNKNOWN)


def abstract_intent(
    language: str | Language,
    intent: str,
) -> AbstractIntent:
    """Map a language-specific intent to its abstract equivalent."""
    lang = Language(language) if isinstance(language, str) else language
    return _INTENT_MAP.get(
        (lang, intent), AbstractIntent.UNKNOWN)


def local_strategies(
    language: str | Language,
    abs_strategy: str | AbstractStrategy,
) -> list[str]:
    """Get language-specific strategies for an abstract strategy."""
    lang = Language(language) if isinstance(language, str) else language
    abs_val = abs_strategy.value if isinstance(
        abs_strategy, AbstractStrategy) else abs_strategy
    return list(_REVERSE_STRATEGY.get((lang, abs_val), []))


def to_cross_language_record(
    language: str | Language,
    intent: str,
    strategy: str,
    success: bool,
    reward: float,
    repair_rounds: int = 0,
    affected_files: int = 1,
    error_types: Optional[list[str]] = None,
) -> CrossLanguageRecord:
    """Convert a language-specific execution result to a cross-language record."""
    lang = Language(language) if isinstance(language, str) else language
    abs_intent = abstract_intent(lang, intent)
    abs_strategy = abstract_strategy(lang, strategy)

    scope = "single" if affected_files <= 1 else "multi"
    context_key = f"{abs_intent.value}:{scope}"

    return CrossLanguageRecord(
        language=lang.value,
        abstract_intent=abs_intent.value,
        abstract_strategy=abs_strategy.value,
        original_strategy=strategy,
        original_intent=intent,
        success=success,
        reward=reward,
        repair_rounds=repair_rounds,
        affected_files=affected_files,
        error_types=error_types or [],
        context_key=context_key,
    )


def peer_strategies(
    language: str | Language,
    strategy: str,
) -> dict[str, list[str]]:
    """Find equivalent strategies in other languages.

    Returns: {other_language: [strategy_names]}
    """
    lang = Language(language) if isinstance(language, str) else language
    abs_strat = abstract_strategy(lang, strategy)
    if abs_strat == AbstractStrategy.UNKNOWN:
        return {}

    result: dict[str, list[str]] = {}
    for other_lang in Language:
        if other_lang == lang:
            continue
        peers = local_strategies(other_lang, abs_strat)
        if peers:
            result[other_lang.value] = peers
    return result
