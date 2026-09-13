"""SSOT drift gate for agent/config/thresholds.py.

``thresholds.py`` is the single source of truth for 70+ hardcoded numeric
thresholds, but nothing until now verified that the SSOT itself stays intact.
This gate runs the *catalog-first* direction that the parity tests
(``test_model_catalog_context_parity``) established:

* **Every SSOT constant must live in exactly one category class.** A rename
  or a move between ``tokens``/``lines``/``scores``/``counts`` breaks the
  consumer contract silently (callers reference ``config.<cat>.NAME``), so
  a constant that is absent from its expected class fails here.
* **Every constant must be consumed by at least one non-test module.** A dead
  constant is a drift you cannot see: nothing reads it, so its value can rot
  without any caller noticing. Consumers are discovered by AST (files
  importing the thresholds module) — not hardcoded, so new consumers are
  included automatically.
* **Consumer category usage must match the SSOT category.** ``config.tokens.
  SUBAGENT_SHORT`` resolves fine even if the constant is actually in
  ``counts``; only this gate catches that the access path is wrong.

The value pins (``KNOWN_SSOT_VALUES``) make every intentional value change an
explicit, reviewed act — the same contract as the provider DEFAULT_MODEL sync
gate in ``test_service_red_green_helpers``.
"""

import ast
import os
import re
from pathlib import Path

import pytest

from external_llm.agent.config.thresholds import config

REPO_ROOT = Path(__file__).resolve().parents[4]
SSOT_PATH = REPO_ROOT / "external_llm" / "agent" / "config" / "thresholds.py"


# Category class name -> config attribute name
CATEGORY_TO_ATTR = {
    "TokenLimits": "tokens",
    "LineLimits": "lines",
    "ScoreThresholds": "scores",
    "CountLimits": "counts",
    "CompressionConfig": "compression",
    "DisplayConfig": "display",
}

# Value pins: changing a threshold intentionally is a deliberate act. A drift
# (someone editing the wrong number) fails the gate with the intended value in
# the failure message, so the fix is explicit, not silent.
# fmt: off
KNOWN_SSOT_VALUES = {
    "INSTRUCTION_JSON": 4096,
    "INSTRUCTION_REPAIR": 8192,
    "PLAN_JSON": 8192,
    "PLAN_REPAIR": 16384,
    "INTENT_CLASSIFY": 4096,
    "INTENT_RESOLVER_DEFAULT": 8192,
    "SERVICE_DEFAULT": 4096,
    "SERVICE_REPAIR": 8192,
    "SUBAGENT_SHORT": 2000,
    "LOCAL_ASSISTANT_SHORT": 512,
    "LOCAL_MODEL_CONTEXT_CHARS": 4000,
    "INTELLIGENT_SERVICE_DEFAULT": 4096,
    "AGENT_STREAM": 4096,
    "ANTHROPIC_DEFAULT": 65536,
    "AGENT_TOOL_CALL": 32768,
    "CONTEXT_HARD_CAP_SAFETY_MARGIN": 1024,
    "BASH_OUTPUT_MAX_CHARS": 60000,
    "DESIGN_TURN_MAX_CHARS": 100000,
    "RAG_FILE_CHARS": 200000,
    "UI_FULL_MAX_LINES": 1000,
    "READ_FILE_FULL_LINES": 800,
    "READ_FILE_MAX_CHARS": 60000,
    "SEARCH_MAX_LINE_CHARS": 2000,
    "READ_FILE_OUTLINE_MAX_SYMBOLS": 60,
    "CALLGRAPH_PY_MAX_BYTES": 1 << 20,
    "CALLGRAPH_TS_MAX_BYTES": 8 * 1024 * 1024,
    "SEMANTIC_INTENT_MIN": 0.10,
    "SEMANTIC_INTENT_MARGIN": 0.08,
    "TOOL_FAILURE_RATE_WARN": 0.50,
    "TOOL_FAILURE_WARN_MIN_CALLS": 3,
    "TOOL_FAILURE_RATE_WINDOW": 30,
    "LATENCY_SAMPLE_WINDOW": 128,
    "TOOL_LATENCY_P95_WARN_MS": 5000.0,
    "LLM_LATENCY_P95_WARN_MS": 30000.0,
    "LATENCY_P95_MIN_SAMPLES": 5,
    "AGENT_NO_TOOL_NUDGE_MAX": 3,
    "AGENT_NO_PROGRESS_THRESHOLD": 5,
    "AGENT_FAIL_LOOP_LARGE": 3,
    "SYMBOL_MAX_PY_FILES": 3000,
    "SYMBOL_MAX_TS_FILES": 1500,
    "RAG_MAX_FILES": 3000,
    "PUSH_CLIENT_QUEUE_SIZE": 200,
    "PROACTIVE_DRAIN_INTERVAL_S": 1.0,
    "AUTONOMOUS_TASK_QUEUE_MAX": 256,
    "AUTONOMOUS_RUNNER_MAX": 8,
    "VULTURE_HUB_IMPORTER_THRESHOLD": 5,
    "SCANNER_DEAD_BLOCK_MAX": 5,
    "SCANNER_PUBLIC_DEAD_BLOCK_MAX": 5,
    "SCANNER_VULTURE_MAX": 10,
    "SCANNER_VULTURE_MIN_CONFIDENCE": 60,
    "SCANNER_DUP_DEF_MAX": 10,
    "SCANNER_UNUSED_IMPORT_MAX": 10,
    "SCANNER_CONTAINER_REACH_MAX": 5,
    "SCANNER_CONTRADICTORY_MAX": 10,
    "SCANNER_CONTRADICTORY_DUP_DISTANCE": 100,
    "SEARCH_RESULTS_CAP": 30,
    "AGENT_TOOL_RETRY_LIMIT": 5,
    "AGENT_MAX_TURNS_DEFAULT": 500,
    "AGENT_MAX_TURNS_WEBAPP_MAX": 200,
    "DESIGN_CHAT_MAX_TOOL_ITERATIONS": 500,
    "DESIGN_CHAT_LLM_MAX_RETRIES": 2,
    "MIN_RECENT_TURNS_KEEP": 4,
    "COMPRESS_BATCH_MIN": 11,
    "GENERAL_MODE_COMPRESS_OCCUPANCY": 0.80,
    "FORCE_COMPRESS_MIN_TURNS": 3,
}
# fmt: on

# DisplayConfig env-driven fields are non-literal: excluded from value pins and
# from the "redefined outside SSOT" scan (their values live in env vars).
NON_LITERAL = {"RUN_DIFF", "NEXT_SUGGEST"}

# Constants whose ONLY consumers live inside webapp/ (or that are referenced
# solely as a comment/literal on both trees — PLAN_JSON). The public CLI
# snapshot (scripts/export_public.py) ships WITHOUT webapp/, so on that tree
# these are intentionally dead: nothing can consume them. The gate must not
# flag them there, or the snapshot release build (release.yml → ssh test job)
# fails on an architectural exclusion, not on a real drift. On the private
# tree (webapp/ present) these ARE consumed and the normal dead check applies.
WEBAPP_ONLY_CONSTANTS = {
    "AGENT_MAX_TURNS_WEBAPP_MAX",  # webapp/routes/agent_stream.py (webapp turn ceiling)
    "AGENT_STREAM",  # webapp/routes/agent_stream.py
    "INSTRUCTION_JSON",  # webapp/llm_execution.py
    "INSTRUCTION_REPAIR",  # webapp/llm_execution.py
    "PLAN_JSON",  # hybrid_parser comment/literal only — no code consumer on either tree
    "PLAN_REPAIR",  # webapp/llm_execution.py
    "UI_FULL_MAX_LINES",  # webapp/ui/ui_tools.py
}

# True only on the public snapshot tree (webapp/ absent). On the private tree
# (webapp/ present) this is empty and the dead check is unchanged.
WEBAPP_EXEMPT_DEAD = WEBAPP_ONLY_CONSTANTS if not (REPO_ROOT / "webapp").is_dir() else set()


def _ssot_constants() -> dict[str, str]:
    """AST-extract name -> category class from the SSOT file."""
    tree = ast.parse(SSOT_PATH.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id.isupper():
                    result[stmt.target.id] = node.name
    return result


def _consumer_files() -> list[Path]:
    """All repo .py files (non-test, excluding the SSOT itself) importing the
    thresholds module."""
    skip_dirs = {".venv", "node_modules", "__pycache__", ".git", "build"}
    # Local name deliberately distinct from the module-level ``consumers``
    # cache — check_missing_global flags functions that assign a name also
    # bound at module scope (undeclared-global policy, zero tolerance).
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            if p == SSOT_PATH:
                continue
            if "tests" in p.parts:
                continue
            if "thresholds" in p.read_text(encoding="utf-8", errors="ignore"):
                found.append(p)
    return found


def _consumed_constants(p: Path) -> set[str]:
    """Constants referenced as ``config.<cat>.<CONST>`` in a consumer file.

    The constant name may contain digits (e.g. ``LATENCY_P95_MIN_SAMPLES``), so
    the token class is ``[A-Za-z0-9_]+``; only names that are actual SSOT
    constants survive the ``ssot`` membership filter.
    """
    src = p.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r"(?:config|_cfg|_threshold_config|_thresholds)\.\w+\.([A-Za-z0-9_]+)")
    return {c for c in pat.findall(src) if c in ssot}


ssot = _ssot_constants()
consumers = _consumer_files()


def test_every_ssot_constant_belongs_to_expected_category():
    """No constant may move between category classes silently."""
    moved = []
    for const, actual_class in ssot.items():
        expected_attr = CATEGORY_TO_ATTR.get(actual_class)
        if expected_attr is None:
            moved.append((const, actual_class, "<unknown>"))
            continue
        # frozen dataclass fields are instance attributes, not class attrs
        if not hasattr(config, expected_attr) or not hasattr(getattr(config, expected_attr), const):
            moved.append((const, actual_class, "not-an-attr"))
    assert not moved, (
        "SSOT constants moved between category classes or missing — "
        "consumers reference config.<category>.<NAME>:\n  " + "\n  ".join(f"{c}: {a}" for c, a, _ in moved)
    )


def test_every_ssot_constant_is_consumed():
    """No dead constants: each SSOT constant must be used by a consumer."""
    consumed = set()
    for p in consumers:
        consumed |= _consumed_constants(p)
    dead = sorted(set(ssot) - consumed - WEBAPP_EXEMPT_DEAD)
    assert not dead, "SSOT constants with no consumer (dead — value drift would be silent):\n  " + "\n  ".join(dead)


def test_consumer_category_access_matches_ssot():
    """Consumer access path (config.<cat>.<C>) must match SSOT category."""
    wrong = []
    for p in consumers:
        src = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"(?:config|_cfg|_threshold_config|_thresholds)\.(\w+)\.([A-Za-z0-9_]+)", src):
            cat, const = m.groups()
            if const not in ssot:
                continue
            expected_attr = CATEGORY_TO_ATTR[ssot[const]]
            if cat != expected_attr:
                wrong.append((p, cat, const, expected_attr))
    assert not wrong, "consumer uses wrong category for SSOT constant:\n  " + "\n  ".join(
        f"{p}: config.{cat}.{const} (expected {exp})" for p, cat, const, exp in wrong
    )


def test_no_ssot_constant_redefined_outside_ssot():
    """No module may shadow an SSOT constant name with a different value.

    The exception: a same-named constant with a *different type* (a string,
    e.g. ``PLAN_JSON = "plan_json"`` in output_modes / execution_mode_classifier)
    is a different-domain constant that merely shares a name with the SSOT
    numeric — it is not a bypass and must not be flagged.
    """
    shadowed = []
    # SSOT literal types: numeric constants are the SSOT's real subject matter
    for p in consumers:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id in ssot and t.id not in NON_LITERAL:
                        # skip if the shadowing value is a string (different domain)
                        if (
                            isinstance(node, ast.Assign)
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, str)
                        ):
                            continue
                        shadowed.append((p, t.id))
    assert not shadowed, "SSOT constant names redefined outside thresholds.py:\n  " + "\n  ".join(
        f"{p}: {c}" for p, c in shadowed
    )


@pytest.mark.parametrize("name", sorted(KNOWN_SSOT_VALUES))
def test_ssot_value_pin(name):
    """Each pinned value must match the live SSOT constant exactly."""
    expected = KNOWN_SSOT_VALUES[name]
    cat = CATEGORY_TO_ATTR[ssot[name]]
    live = getattr(getattr(config, cat), name)
    assert live == expected, (
        f"SSOT {name} drifted: live={live} expected={expected}. If the change is intentional, update KNOWN_SSOT_VALUES."
    )


def test_ssot_constant_count_stable():
    """Guard against accidental mass-addition/removal of constants."""
    assert len(ssot) == 67
