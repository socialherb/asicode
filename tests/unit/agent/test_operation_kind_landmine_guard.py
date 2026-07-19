"""Structural recurrence guard for the OperationKind str-Enum value-vs-name landmine.

Background
----------
``OperationKind(str, enum.Enum)`` members compare equal to their LOWERCASE *value*
(``K.READ_SYMBOL == "read_symbol"``), NOT their uppercase *name*
(``K.READ_SYMBOL == "READ_SYMBOL"`` -> False). Comparing the raw ``op.kind``
attribute against an uppercase NAME string or an uppercase-only collection is
therefore always-False — silently dead validation code.

This was fixed in c10897a2 (12 sites) by migrating to SSOT frozensets
(``READ_ONLY_KINDS`` etc.). The per-module contract tests in
``test_operation_models.py`` only assert the frozensets' *membership semantics* —
they do NOT detect a future caller re-introducing the bug. A mutation that
restores ``_SAFE_KINDS = {"READ_SYMBOL", ...}`` passes the whole suite (verified).

This test is the real recurrence guard: it walks the ``external_llm/`` source tree
and fails if any **bare** ``<obj>.kind`` attribute is compared against an uppercase
member-name string or an uppercase-only collection.

Detection signature (calibrated to zero false-positives on the fixed tree):

  FLAGGED
    op.kind == "READ_SYMBOL"
    op.kind in ("READ_SYMBOL", "MODIFY_SYMBOL")
    op.kind not in {"READ_SYMBOL", ...}
    op.kind in _SAFE   where _SAFE = {"READ_SYMBOL", ...}   (named module/class const)

  ALLOWED
    op.kind == "read_symbol"            # lowercase value — correct
    op.kind.name == "READ_SYMBOL"       # explicit .name access — correct
    op.kind.value == "read_symbol"      # explicit .value access — correct
    op.kind == OperationKind.READ_SYMBOL  # enum-to-enum — correct
    op.kind in READ_ONLY_KINDS          # SSOT frozenset — correct
    k = op.kind.upper(); k == "..."     # normalized via .upper() — correct
    {"READ_SYMBOL", "read_symbol"}      # defensive dual-case set — correct
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from external_llm.agent.operation_models import OperationKind

_MEMBER_NAMES = {m.name for m in OperationKind}      # e.g. "READ_SYMBOL"  (uppercase)
_MEMBER_VALUES = {m.value for m in OperationKind}    # e.g. "read_symbol"  (lowercase)
_NAME_TO_VALUE = {m.name: m.value for m in OperationKind}

# Enum-defining module owns the member names legitimately (OP_KIND_POLICY keys etc.).
_EXACT_SKIP = {"operation_models.py"}

REPO_ROOT = Path(__file__).resolve().parents[3]   # tests/unit/agent/x.py -> repo root
_SCAN_ROOT = REPO_ROOT / "external_llm"


def _is_bare_kind_attr(node: ast.AST) -> bool:
    """True for ``X.kind`` where the Attribute's attr is exactly 'kind'.

    This excludes ``X.kind.name`` / ``X.kind.value`` (those are Attribute nodes
    whose own attr is 'name'/'value') and ``X.kind.upper()`` (a Call node).
    """
    return isinstance(node, ast.Attribute) and node.attr == "kind"


def _collection_strs(node: ast.AST) -> list[str]:
    """Flatten string constants from a Set/Tuple/List literal or ``set()``/``frozenset()`` call."""
    out: list[str] = []
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        out += [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("set", "frozenset"):
        for arg in node.args:
            if isinstance(arg, (ast.Set, ast.Tuple, ast.List)):
                out += [e.value for e in arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return out


def _collection_is_unsafe(strs: list[str]) -> tuple[bool, str | None]:
    """A collection is a (partial) landmine if it contains an uppercase member NAME
    whose lowercase *value* is NOT also present (the enum member would never match
    that entry). Defensive dual-case sets (``{"X", "x"}``) are therefore safe."""
    sset = set(strs)
    for name in sset & _MEMBER_NAMES:
        val = _NAME_TO_VALUE.get(name)
        if val and val not in sset:
            return True, name
    return False, None


def _named_const_key(node: ast.AST) -> str | None:
    """Return the name a named-constant operand refers to, or None.

    Handles both a bare module/class name (``_SAFE``) and attribute access
    (``self._SAFE`` / ``cls._SAFE``). Class attributes are typically read via
    ``self.<name>`` — that was the exact shape of the original _SAFE_KINDS bug,
    so attribute access MUST resolve here.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, reason), ...] for each landmine Compare in ``path``."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []

    # File-scoped map of name -> set-of-strings for any Assign whose value is a
    # string collection containing at least one member name (module/class/function
    # level; last-wins). Lets us resolve ``op.kind in _SAFE``.
    name_map: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
            continue
        strs = _collection_strs(n.value)
        if not strs or not any(s in _MEMBER_NAMES for s in strs):
            continue
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in targets:
            if isinstance(t, ast.Name):
                name_map[t.id] = set(strs)

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(_is_bare_kind_attr(o) for o in operands):
            continue
        for o in operands:
            if _is_bare_kind_attr(o):
                continue
            # inline member-name string: op.kind == "READ_SYMBOL"
            if isinstance(o, ast.Constant) and isinstance(o.value, str) and o.value in _MEMBER_NAMES:
                hits.append((node.lineno, f"bare .kind compared to uppercase name string {o.value!r}"))
                continue
            # inline collection: op.kind in ("READ_SYMBOL", ...)
            strs = _collection_strs(o)
            if strs:
                unsafe, which = _collection_is_unsafe(strs)
                if unsafe:
                    hits.append((node.lineno, f"bare .kind in uppercase-only collection (missing lowercase value for {which!r})"))
                continue
            # named constant: op.kind in _SAFE  where _SAFE = {...uppercase...}
            # (covers both bare _SAFE and self._SAFE / cls._SAFE attribute access)
            key = _named_const_key(o)
            if key is not None and key in name_map:
                unsafe, which = _collection_is_unsafe(name_map[key])
                if unsafe:
                    hits.append((node.lineno, f"bare .kind in named const {key!r} (missing lowercase value for {which!r})"))
    return hits


def _iter_scan_files() -> list[Path]:
    files = []
    for p in sorted(_SCAN_ROOT.rglob("*.py")):
        sp = str(p)
        if "__pycache__" in sp:
            continue
        if "/test_" in sp or p.name.startswith("test_"):
            continue
        if p.name in _EXACT_SKIP:
            continue
        files.append(p)
    return files


def test_no_operationkind_value_vs_name_landmine_in_source():
    """The value-vs-name landmine must not recur anywhere in external_llm/.

    A failure names every file:line where a bare ``<obj>.kind`` is compared
    against an uppercase member-name string / uppercase-only collection. Fix by
    comparing against the SSOT frozensets (READ_ONLY_KINDS / FILE_WRITING_KINDS /
    CREATES_OR_DELETES_KINDS / DELETE_KINDS) or against the lowercase value / an
    explicit ``.kind.name``|``.kind.value`` access.
    """
    offenders: list[str] = []
    for path in _iter_scan_files():
        for lineno, reason in _scan_file(path):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"  {rel}:{lineno}  {reason}")
    assert not offenders, (
        "OperationKind value-vs-name landmine re-introduced. A bare `op.kind` is "
        "compared against an uppercase NAME string/collection (always-False for a "
        "str-Enum whose members equal their lowercase VALUE). Use the SSOT "
        "frozensets or an explicit .kind.name/.kind.value access:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "src, expect_hit",
    [
        # ── regression shapes that MUST be caught ──────────────────────────────
        ('def f(op):\n  if op.kind == "READ_SYMBOL":\n    pass\n', True),
        ('_SAFE = {"READ_SYMBOL", "READ_FILE_SEGMENT"}\n'
         'def f(op):\n  if op.kind in _SAFE:\n    pass\n', True),
        # class attribute read via self. (the EXACT original _SAFE_KINDS shape)
        ('class V:\n  _SAFE = {"READ_SYMBOL", "READ_FILE_SEGMENT"}\n'
         '  def f(self, op):\n    if op.kind in self._SAFE:\n      pass\n', True),
        ('def f(op):\n  if op.kind not in ("READ_SYMBOL", "MODIFY_SYMBOL"):\n    pass\n', True),
        # ── legitimate shapes that must NOT be flagged ─────────────────────────
        ('def f(op):\n  if op.kind == "read_symbol":\n    pass\n', False),     # lowercase value
        ('def f(op):\n  if op.kind.name == "READ_SYMBOL":\n    pass\n', False),  # explicit .name
        ('def f(op):\n  if op.kind.value == "read_symbol":\n    pass\n', False),  # explicit .value
        ('def f(op, K):\n  if op.kind == K.READ_SYMBOL:\n    pass\n', False),     # enum-to-enum
        ('from om import READ_ONLY_KINDS\n'
         'def f(op):\n  if op.kind in READ_ONLY_KINDS:\n    pass\n', False),      # SSOT frozenset
        ('_SAFE = {"READ_SYMBOL", "read_symbol"}\n'
         'def f(op):\n  if op.kind in _SAFE:\n    pass\n', False),               # defensive dual-case
        ('def f(op):\n  k = (op.kind or "").upper()\n'
         '  if k == "READ_SYMBOL":\n    pass\n', False),                         # .upper() normalized
    ],
)
def test_landmine_detector_signature(src, expect_hit):
    """Parametric proof of the detector's precision (regression cases hit; legit don't)."""
    tree = ast.parse(src)
    name_map: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            strs = _collection_strs(n.value)
            if strs and any(s in _MEMBER_NAMES for s in strs):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    if isinstance(t, ast.Name):
                        name_map[t.id] = set(strs)
    hit = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(_is_bare_kind_attr(o) for o in operands):
            continue
        for o in operands:
            if _is_bare_kind_attr(o):
                continue
            if isinstance(o, ast.Constant) and isinstance(o.value, str) and o.value in _MEMBER_NAMES:
                hit = True
                break
            strs = _collection_strs(o)
            if strs and _collection_is_unsafe(strs)[0]:
                hit = True
                break
            key = _named_const_key(o)
            if key is not None and key in name_map and _collection_is_unsafe(name_map[key])[0]:
                hit = True
                break
    assert hit is expect_hit
