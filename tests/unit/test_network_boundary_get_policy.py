"""AST policy gate: network-response parsers must survive explicit nulls.

Chained access ``x.get(K, {}).get(K2)`` applies the ``{}`` default only on
key ABSENCE. On the wire an explicit ``"K": null`` is a legal value
(gateway shims, proxies, documented nullable fields such as Gemini's
``candidate.content`` or Brave's ``"web"``) and turns the chain into
``None.get(...)`` → AttributeError, which SSE guards convert into a
whole-turn failure.

Policy (structural, no keyword matching of the vulnerable code):

* Scope — every ``external_llm`` module that performs network I/O,
  detected structurally by importing ``requests`` or ``httpx``.
* Rule — the receiver of an outer ``.get(...)`` must not be an inner
  ``.get(...)`` carrying a mutable-literal default (``{}`` / ``[]`` /
  ``""``). Use ``(x.get(K) or {})`` instead, which covers both key
  absence and explicit null.
"""
from __future__ import annotations

import ast
import pathlib

_LLM_ROOT = pathlib.Path(__file__).resolve().parents[2] / "external_llm"
_NETWORK_LIBS = {"requests", "httpx"}


def _imports_network_library(tree: ast.Module) -> bool:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return bool(_NETWORK_LIBS & roots)


def _receiver_is_unsafe_inner_get(recv: ast.AST) -> bool:
    if not (
        isinstance(recv, ast.Call)
        and isinstance(recv.func, ast.Attribute)
        and recv.func.attr == "get"
    ):
        return False
    for arg in recv.args[1:]:  # defaults only — args[0] is the key
        if isinstance(arg, (ast.Dict, ast.List, ast.Set)):
            return True
        if isinstance(arg, ast.Constant) and arg.value == "":
            return True
    return False


def _unsafe_chained_get_lines(tree: ast.Module) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _receiver_is_unsafe_inner_get(node.func.value)
        )
    ]


def test_no_unsafe_chained_gets_in_network_modules() -> None:
    offenders: list[str] = []
    for path in sorted(_LLM_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if not _imports_network_library(tree):
            continue  # static config readers are out of policy scope
        for lineno in _unsafe_chained_get_lines(tree):
            offenders.append(f"{path.relative_to(_LLM_ROOT.parent)}:{lineno}")
    assert not offenders, (
        "network-response parsers must use (x.get(K) or {}) for chained access — "
        "explicit nulls are legal on the wire:\n  " + "\n  ".join(offenders)
    )


def test_policy_selfcheck_flags_unsafe_and_passes_safe_forms() -> None:
    # The detector itself must distinguish the two shapes (guards against a
    # gate that trivially passes everything).
    unsafe = ast.parse('y = x.get("a", {}).get("b")\nz = x.get("a", []).get("b")\n')
    safe = ast.parse('y = (x.get("a") or {}).get("b")\nz = x.get("a", "static-default").get("b")\n')
    assert _unsafe_chained_get_lines(unsafe) == [1, 2]
    assert _unsafe_chained_get_lines(safe) == []
