"""Gate: every ``_SLASH_COMMANDS`` entry must be dispatched in ``_dispatch_command``.

The 28th audit round found the REPL slash-command registry
(``asi.py::_SLASH_COMMANDS``, 17 commands) and its dispatcher
(``external_llm/repl/repl_impl.py::_run_repl_impl::_dispatch_command``) live in
different modules and use TWO structurally different dispatch styles:

  1. a ``_cmd_name == "/x"`` handler chain, gated by an
     ``if _cmd_name in ("/help", "/diff", ...)`` tuple (utility commands), plus
     standalone ``if _cmd_name == "/x"`` blocks (``/claude``, ``/think``,
     ``/auto``, ``/quit``);
  2. raw-string mode-switch blocks keyed on ``_stripped.startswith("/code ")``
     etc. (``/code``, ``/general``, ``/orchestrate`` + alias ``/orch``).

The past ``/failure-patterns`` dispatch-omission bug happened exactly because
of this split: a command was defined in the registry but never reached by
either style, silently routing to the chat turn. No test cross-referenced the
definition against the dispatch, so the omission went unnoticed. This gate
closes that gap with static AST analysis (no runtime import — asi.py pulls in
the whole agent stack).

Rules (violations are listed, never raised by helpers):

  R1  every canonical name must appear in a real handler branch
      (``_cmd_name ==`` comparison or a raw dispatch condition);
  R2  every ``_cmd_name ==`` handler nested under the ``_cmd_name in (...)
      gate`` must be listed in that gate tuple (otherwise the handler is
      unreachable — the historical bug class);
  R3  every ``_cmd_name ==``/``in`` dispatch reference must be a registered
      canonical or alias (dead/typo branches);
  R4  every alias must be handled: either directly in a handler branch, or its
      canonical is dispatched via the ``_cmd_name`` path (the alias map
      resolves it); non-slash aliases (``:q``) must appear in the
      ``user_input.lower() in (...)`` quit tuple;
  R5  registry hygiene: unique canonicals, unique aliases, no alias colliding
      with a canonical, slash-token names; the ``_SLASH_ALIASES`` auto-build
      loop over ``_SLASH_COMMANDS`` must still exist (guards a stale hardcoded
      map).

Detection evidence (what counts as a "handler branch"):

  * ``_cmd_name == "/x"`` comparisons and ``_cmd_name in ("/a", ...)`` tuples;
  * raw dispatch conditions: ``startswith("/x")`` args, set literals in
    ``in {...}`` comparisons, ``len("/x")`` slice bounds, and
    ``_stripped == "/x"`` — i.e. the mode-switch block patterns.

Deliberately NOT counted: strings inside ``_print(...)`` calls (display text
such as the "(/code to exit)" hint) and handler-internal tuples such as
``_tok0 in ("/orchestrate", "/orch")`` (inline-task extraction, not dispatch).
The latter means a partial alias removal that keeps only the internal tuple is
still detected — the internal ref is not evidence.

Documented limitations (same stance as the subprocess gate): static analysis
performs no control-flow reasoning, so a handler whose *only* surviving
reference sits in a never-executed branch can evade R1; the reverse check is
scoped to ``_cmd_name``-path refs because raw conditions may legitimately
match path prefixes (e.g. ``startswith("/tmp")``).

Three-layer verification (the gate must not pass vacuously):

  1. compliance — the real files are analyzed and every rule asserted;
  2. detector self-tests — synthetic compliant/violating snippets pin the
     detector's sensitivity;
  3. real-file mutation demos — the actual sources are mutated (dispatch tuple
     entry removed, handler removed, alias lost, build loop replaced, ...) and
     each mutation MUST be detected.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ASI_PY = REPO_ROOT / "asi.py"
REPL_IMPL_PY = REPO_ROOT / "external_llm" / "repl" / "repl_impl.py"

_CMD_TOKEN_RE = re.compile(r"^/[A-Za-z][A-Za-z0-9-]*$")


# ─────────────────────────── static extraction ───────────────────────────


def _assign_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


def _require_str(node: ast.AST) -> str:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise TypeError(f"expected string literal, got {type(node).__name__}")
    return node.value


def extract_slash_commands(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """[(canonical, aliases)] parsed from the ``_SLASH_COMMANDS`` list literal."""
    tree = ast.parse(source)
    for node in tree.body:
        if any(
            isinstance(t, ast.Name) and t.id == "_SLASH_COMMANDS" for t in _assign_targets(node)
        ):
            value = node.value
            if not isinstance(value, ast.List):
                raise AssertionError("_SLASH_COMMANDS must be a list literal for static extraction")
            out: list[tuple[str, tuple[str, ...]]] = []
            for elt in value.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) < 4:
                    raise AssertionError(f"bad _SLASH_COMMANDS entry: {ast.unparse(elt)[:60]!r}")
                name = _require_str(elt.elts[0])
                aliases_elt = elt.elts[1]
                if not isinstance(aliases_elt, ast.Tuple):
                    raise TypeError(f"aliases not a tuple for {name!r}")
                out.append((name, tuple(_require_str(a) for a in aliases_elt.elts)))
            return out
    raise AssertionError("_SLASH_COMMANDS assignment not found")


def _find_def(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )


def _build_parent_map(fn: ast.FunctionDef) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_cmd_name_eq(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "_cmd_name"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and isinstance(node.comparators[0].value, str)
    )


def _is_startswith_call(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr == "startswith") or (
        isinstance(f, ast.Name) and f.id == "startswith"
    )


def _classify(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Kind of dispatch reference: equals / tuple / raw_cond / internal."""
    parent = parents.get(node)
    if isinstance(parent, ast.Compare):
        left = parent.left
        if isinstance(left, ast.Name) and left.id == "_cmd_name":
            return "equals"
        if isinstance(left, ast.Name) and left.id == "_stripped":
            return "raw_cond"
        return None
    if isinstance(parent, ast.Tuple):
        gp = parents.get(parent)
        if isinstance(gp, ast.Compare):
            if isinstance(gp.left, ast.Name) and gp.left.id == "_cmd_name":
                return "tuple"
            return "internal"  # handler-internal ref (e.g. _tok0 in (...)) — not evidence
        if isinstance(gp, ast.Call) and _is_startswith_call(gp):
            return "raw_cond"
        return None
    if isinstance(parent, ast.Set) and isinstance(parents.get(parent), ast.Compare):
        return "raw_cond"
    if isinstance(parent, ast.Call):
        if _is_startswith_call(parent):
            return "raw_cond"
        if (
            isinstance(parent.func, ast.Name)
            and parent.func.id == "len"
            and isinstance(parents.get(parent), ast.Slice)
        ):
            return "raw_cond"
    return None


def _in_user_input_tuple(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if not isinstance(parent, ast.Tuple):
        return False
    gp = parents.get(parent)
    if not isinstance(gp, ast.Compare):
        return False
    return any(isinstance(n, ast.Name) and n.id == "user_input" for n in ast.walk(gp.left))


def _inside_print(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id == "_print":
            return True
        cur = parents.get(cur)
    return False


def _gated_equals(fn: ast.FunctionDef) -> tuple[set[str], bool]:
    """Tokens of ``_cmd_name == "/x"`` handlers nested under the tuple gate."""
    gated: set[str] = set()
    gate: ast.If | None = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            cmp = node.test
            if (
                isinstance(cmp.left, ast.Name)
                and cmp.left.id == "_cmd_name"
                and len(cmp.comparators) == 1
                and isinstance(cmp.comparators[0], ast.Tuple)
            ):
                gate = node
                break
    if gate is None:
        return gated, False

    def _walk_if(n: ast.If) -> None:
        if _is_cmd_name_eq(n.test):
            gated.add(n.test.comparators[0].value)
        for child in list(n.body) + list(n.orelse):
            if isinstance(child, ast.If):
                _walk_if(child)

    for stmt in list(gate.body) + list(gate.orelse):
        if isinstance(stmt, ast.If):
            _walk_if(stmt)
    return gated, True


def extract_dispatch_refs(source: str) -> dict[str, set[str] | bool]:
    """Dispatch evidence extracted from ``_dispatch_command``.

    Keys: equals (``_cmd_name ==``), tuple (``_cmd_name in (... )``),
    raw_cond (startswith / set / len-slice / ``_stripped ==``), internal
    (handler-internal tuples, not evidence), non_slash (``:q``/``quit``/
    ``exit`` in the ``user_input`` tuple), gated + has_gate (tuple-gated
    handler chain).
    """
    tree = ast.parse(source)
    fn = _find_def(tree, "_dispatch_command")
    if fn is None:
        raise AssertionError("_dispatch_command not found")
    parents = _build_parent_map(fn)
    refs: dict[str, set[str] | bool] = {
        "equals": set(),
        "tuple": set(),
        "raw_cond": set(),
        "internal": set(),
        "non_slash": set(),
    }
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        val = node.value.strip()
        if not val or _inside_print(node, parents):
            continue
        if val.startswith("/"):
            tok = val.split()[0]
            if not _CMD_TOKEN_RE.match(tok):
                continue
            kind = _classify(node, parents)
            if kind:
                refs[kind].add(tok)  # type: ignore[union-attr]
        elif _in_user_input_tuple(node, parents):
            refs["non_slash"].add(val)  # type: ignore[union-attr]
    gated, has_gate = _gated_equals(fn)
    refs["gated"] = gated
    refs["has_gate"] = has_gate
    return refs


# ─────────────────────────────── checks ───────────────────────────────


def check_registry(commands: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    """R5: registry hygiene."""
    violations: list[str] = []
    canon = [c for c, _ in commands]
    aliases = [a for _, als in commands for a in als]
    for c in canon:
        if not _CMD_TOKEN_RE.match(c):
            violations.append(f"canonical not a slash token: {c!r}")
    for d in sorted({c for c in canon if canon.count(c) > 1}):
        violations.append(f"duplicate canonical: {d}")
    for d in sorted({a for a in aliases if aliases.count(a) > 1}):
        violations.append(f"duplicate alias: {d}")
    canon_set = set(canon)
    for c, als in commands:
        for a in als:
            if a in canon_set:
                violations.append(f"alias {a!r} collides with canonical (command {c!r})")
            if a == c:
                violations.append(f"alias identical to its canonical: {a!r}")
    return violations


def check_dispatch(commands: list[tuple[str, tuple[str, ...]]], refs: dict[str, set[str] | bool]) -> list[str]:
    """R1-R4: every registry entry must be reachable in ``_dispatch_command``."""
    violations: list[str] = []
    canon = [c for c, _ in commands]
    aliases = [a for _, als in commands for a in als]
    equals: set[str] = refs["equals"]  # type: ignore[assignment]
    raw_cond: set[str] = refs["raw_cond"]  # type: ignore[assignment]
    handlers = equals | raw_cond
    for c in canon:
        if c not in handlers:
            violations.append(f"canonical {c} has no dispatch branch in _dispatch_command")
    if refs.get("has_gate"):
        gated: set[str] = refs["gated"]  # type: ignore[assignment]
        tuple_toks: set[str] = refs["tuple"]  # type: ignore[assignment]
        missing = gated - tuple_toks
        if missing:
            violations.append(f"gated handler(s) missing from dispatch tuple: {sorted(missing)}")
    for c, als in commands:
        for a in als:
            if a.startswith("/"):
                if a not in handlers and c not in equals:
                    violations.append(
                        f"alias {a} of {c} not dispatched (canonical {c} raw-only or absent)"
                    )
            elif a not in refs["non_slash"]:  # type: ignore[operator]
                violations.append(f"non-slash alias {a!r} of {c} not handled")
    unknown = (equals | refs["tuple"]) - set(canon) - set(aliases)  # type: ignore[operator]
    if unknown:
        violations.append(f"dispatch refs without registry entry: {sorted(unknown)}")
    return violations


def check_alias_build_loop(source: str) -> list[str]:
    """R5: the auto-build loop over ``_SLASH_COMMANDS`` must still exist."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "_SLASH_COMMANDS":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "_SLASH_ALIASES":
                            return []
    return ["no for-loop over _SLASH_COMMANDS building _SLASH_ALIASES found"]


def run_gate(asi_source: str, repl_source: str) -> list[str]:
    commands = extract_slash_commands(asi_source)
    violations = check_registry(commands)
    violations += check_dispatch(commands, extract_dispatch_refs(repl_source))
    violations += check_alias_build_loop(asi_source)
    return violations


# ───────────────── layer 1: real-file compliance ─────────────────

ASI_SRC = ASI_PY.read_text(encoding="utf-8")
REPL_SRC = REPL_IMPL_PY.read_text(encoding="utf-8")
COMMANDS = extract_slash_commands(ASI_SRC)
REFS = extract_dispatch_refs(REPL_SRC)
ALIAS_ITEMS = [(a, c) for c, als in COMMANDS for a in als]


def test_registry_has_full_scope() -> None:
    # sanity floor — a truncated registry would silently drop commands
    assert len(COMMANDS) >= 10


def test_registry_hygiene() -> None:
    assert check_registry(COMMANDS) == []


def test_alias_build_loop_present() -> None:
    assert check_alias_build_loop(ASI_SRC) == []


@pytest.mark.parametrize("cmd", [c for c, _ in COMMANDS], ids=[c for c, _ in COMMANDS])
def test_canonical_command_dispatched(cmd: str) -> None:
    handlers = REFS["equals"] | REFS["raw_cond"]  # type: ignore[operator]
    assert cmd in handlers, (
        f"{cmd} is defined in _SLASH_COMMANDS but has no dispatch branch in _dispatch_command"
    )


@pytest.mark.parametrize("alias,canonical", ALIAS_ITEMS, ids=[a for a, _ in ALIAS_ITEMS])
def test_alias_handled(alias: str, canonical: str) -> None:
    handlers = REFS["equals"] | REFS["raw_cond"]  # type: ignore[operator]
    if alias.startswith("/"):
        assert alias in handlers or canonical in REFS["equals"], (  # type: ignore[operator]
            f"alias {alias} of {canonical} is not reachable in _dispatch_command"
        )
    else:
        assert alias in REFS["non_slash"], (  # type: ignore[operator]
            f"non-slash alias {alias!r} of {canonical} is not handled"
        )


def test_no_unknown_refs_and_gate_consistency() -> None:
    # reverse check (R3) + tuple-gate reachability (R2) — covered by check_dispatch
    assert check_dispatch(COMMANDS, REFS) == []


# ─────────────── layer 2: detector self-tests (synthetic) ───────────────

_COMPLIANT_ASI = '''
_SLASH_COMMANDS: list[tuple[str, tuple[str, ...], str, str]] = [
    ("/quit",        (":q", "/exit"), "", "end"),
    ("/help",        ("/?",),         "", "help"),
    ("/status",      ("/info",),      "", "status"),
    ("/code",        (),              "", "code"),
    ("/orchestrate", ("/orch",),      "", "orch"),
]
_SLASH_ALIASES: dict[str, str] = {}
for _name, _aliases, _arg, _desc in _SLASH_COMMANDS:
    _SLASH_ALIASES[_name] = _name
    for _al in _aliases:
        _SLASH_ALIASES[_al] = _name
'''

_COMPLIANT_REPL = '''
def _run_repl_impl():
    def _dispatch_command(user_input: str):
        _cmd_tok = user_input.strip().split(None, 1)
        _cmd_name = _SLASH_ALIASES.get(_cmd_tok[0].lower()) if _cmd_tok else None
        if user_input.lower() in (":q", "quit", "exit") or _cmd_name == "/quit":
            return ("break", user_input)
        if _cmd_name in ("/help", "/status"):
            if _cmd_name == "/help":
                _print("  help  (hint: /code to exit)")
            elif _cmd_name == "/status":
                _print("  status")
            return ("continue", user_input)
        _stripped = user_input.strip()
        if _stripped.startswith("/code ") or _stripped == "/code":
            _print("  switched to Code Chat")
            user_input = _stripped[len("/code"):].lstrip()
        elif _stripped.startswith(("/orchestrate ", "/orch ")) or _stripped in {"/orchestrate", "/orch"}:
            _tok0 = _cmd_tok[0].lower() if _cmd_tok else ""
            _orch_inline = _stripped[len(_tok0):].lstrip() if _tok0 in ("/orchestrate", "/orch") else ""
            user_input = _orch_inline
        return ("chat", user_input)
'''


def _v_asi(replacer: str, replacement: str) -> str:
    assert _COMPLIANT_ASI.count(replacer) == 1, f"self-test asi anchor: {replacer!r}"
    return _COMPLIANT_ASI.replace(replacer, replacement)


def _v_repl(replacer: str, replacement: str) -> str:
    assert _COMPLIANT_REPL.count(replacer) == 1, f"self-test repl anchor: {replacer!r}"
    return _COMPLIANT_REPL.replace(replacer, replacement)


_SELF_TEST_CASES = [
    (
        "compliant",
        _COMPLIANT_ASI,
        _COMPLIANT_REPL,
        None,
    ),
    (
        "canonical_undispatched",
        _COMPLIANT_ASI,
        _v_repl(
            '        if _stripped.startswith("/code ") or _stripped == "/code":\n'
            '            _print("  switched to Code Chat")\n'
            '            user_input = _stripped[len("/code"):].lstrip()\n',
            '        if False:  # /code removed\n            pass\n',
        ),
        "has no dispatch branch",
    ),
    (
        "gated_handler_missing_from_tuple",
        _COMPLIANT_ASI,
        _v_repl('if _cmd_name in ("/help", "/status"):', 'if _cmd_name in ("/help",):'),
        "missing from dispatch tuple",
    ),
    (
        "alias_lost_from_raw_conditions",
        _COMPLIANT_ASI,
        _v_repl(
            '        elif _stripped.startswith(("/orchestrate ", "/orch ")) or _stripped in {"/orchestrate", "/orch"}:\n',
            '        elif _stripped.startswith("/orchestrate "):\n',
        ),
        "alias /orch",
    ),
    (
        "unknown_dispatch_ref",
        _COMPLIANT_ASI,
        _v_repl(
            '        if _cmd_name in ("/help", "/status"):\n',
            '        if _cmd_name == "/ghost-x":\n            return ("continue", user_input)\n        if _cmd_name in ("/help", "/status"):\n',
        ),
        "without registry entry",
    ),
    (
        "non_slash_alias_lost",
        _COMPLIANT_ASI,
        _v_repl('in (":q", "quit", "exit")', 'in ("quit", "exit")'),
        "non-slash alias",
    ),
    (
        "duplicate_alias",
        _v_asi('("/status",      ("/info",),', '("/status",      ("/?",),'),
        _COMPLIANT_REPL,
        "duplicate alias",
    ),
    (
        "alias_collides_with_canonical",
        _v_asi('("/status",      ("/info",),', '("/status",      ("/code",),'),
        _COMPLIANT_REPL,
        "collides with canonical",
    ),
    (
        "alias_build_loop_replaced",
        _v_asi(
            'for _name, _aliases, _arg, _desc in _SLASH_COMMANDS:\n    _SLASH_ALIASES[_name] = _name\n    for _al in _aliases:\n        _SLASH_ALIASES[_al] = _name\n',
            '_SLASH_ALIASES = {"/help": "/help"}  # stale\n',
        ),
        _COMPLIANT_REPL,
        "no for-loop over _SLASH_COMMANDS",
    ),
]


@pytest.mark.parametrize(
    "label,asi_src,repl_src,expected",
    list(_SELF_TEST_CASES),    ids=[c[0] for c in _SELF_TEST_CASES],
)
def test_detector_self_tests(label: str, asi_src: str, repl_src: str, expected: str | None) -> None:
    violations = run_gate(asi_src, repl_src)
    if expected is None:
        assert violations == [], f"compliant synthetic code must pass, got: {violations}"
    else:
        assert any(expected in v for v in violations), (
            f"[{label}] expected {expected!r} in violations, got: {violations}"
        )


# ─────────── layer 3: real-file mutation demos (must be caught) ───────────


def test_mutation_dispatch_tuple_entry_removed() -> None:
    """Historical bug class: command gated by the tuple but handler unreachable."""
    needle = '            "/failure-patterns",\n'
    assert REPL_SRC.count(needle) == 1, "anchor — dispatch tuple changed?"
    mutated = REPL_SRC.replace(needle, "")
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("missing from dispatch tuple" in v for v in violations), violations


def test_mutation_handler_removed_from_gate_chain() -> None:
    """Full historical bug shape: neither tuple entry nor handler remains."""
    needle = '            elif _cmd_name == "/failure-patterns":\n'
    assert REPL_SRC.count(needle) == 1, "anchor — elif chain changed?"
    lines = REPL_SRC.splitlines()
    idx = next(i for i, line in enumerate(lines) if line == needle.rstrip("\n"))
    j = idx + 1
    while j < len(lines) and (lines[j].startswith("            ") or lines[j] == ""):
        j += 1
    mutated = "\n".join(lines[:idx] + lines[j:])
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("/failure-patterns" in v for v in violations), violations


def test_mutation_standalone_handler_removed() -> None:
    needle = '        if _cmd_name == "/think":\n'
    assert REPL_SRC.count(needle) == 1, "anchor — /think block changed?"
    mutated = REPL_SRC.replace(needle, "        if False:  # /think handler removed\n")
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("/think" in v for v in violations), violations


def test_mutation_new_command_undispatched() -> None:
    needle = '    ("/quit",    (":q", "/exit"), "",       "end the session"),\n'
    assert ASI_SRC.count(needle) == 1, "anchor — registry tail changed?"
    mutated = ASI_SRC.replace(needle, needle + '    ("/bogus-x", (), "", "gate self-test"),\n')
    commands = extract_slash_commands(mutated)
    violations = check_dispatch(commands, REFS)
    assert any("/bogus-x" in v for v in violations), violations


def test_mutation_alias_lost_from_raw_dispatch() -> None:
    """/orch removed from the raw dispatch conditions; internal _tok0 ref remains."""
    needle = (
        '        elif _stripped.startswith(("/orchestrate ", "/orch ")) '
        'or _stripped in {"/orchestrate", "/orch"}:\n'
    )
    assert REPL_SRC.count(needle) == 1, "anchor — orchestrator block changed?"
    mutated = REPL_SRC.replace(needle, '        elif _stripped.startswith("/orchestrate "):\n')
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("alias /orch" in v for v in violations), violations


def test_mutation_unknown_ref_added() -> None:
    needle = '        if _cmd_name == "/auto":\n'
    assert REPL_SRC.count(needle) == 1, "anchor — /auto block changed?"
    mutated = REPL_SRC.replace(
        needle,
        '        if _cmd_name == "/ghost-x":\n            return ("continue", user_input)\n' + needle,
    )
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("/ghost-x" in v for v in violations), violations


def test_mutation_non_slash_alias_lost() -> None:
    needle = '        if user_input.lower() in (":q", "quit", "exit") or _cmd_name == "/quit":\n'
    assert REPL_SRC.count(needle) == 1, "anchor — quit dispatch changed?"
    mutated = REPL_SRC.replace(needle, '        if user_input.lower() in ("quit", "exit") or _cmd_name == "/quit":\n')
    violations = check_dispatch(COMMANDS, extract_dispatch_refs(mutated))
    assert any("':q'" in v for v in violations), violations


def test_mutation_alias_build_loop_replaced() -> None:
    loop = (
        "for _name, _aliases, _arg, _desc in _SLASH_COMMANDS:\n"
        "    _SLASH_ALIASES[_name] = _name\n"
        "    for _al in _aliases:\n"
        "        _SLASH_ALIASES[_al] = _name\n"
    )
    assert ASI_SRC.count(loop) == 1, "anchor — alias build loop changed?"
    mutated = ASI_SRC.replace(loop, '_SLASH_ALIASES = {"/help": "/help"}  # stale\n')
    violations = check_alias_build_loop(mutated)
    assert any("no for-loop" in v for v in violations), violations


def test_mutation_duplicate_alias() -> None:
    needle = '    ("/diff",    (),              "",       "re-show the last run\'s file changes"),\n'
    assert ASI_SRC.count(needle) == 1, "anchor — /diff entry changed?"
    mutated = ASI_SRC.replace(needle, '    ("/diff",    ("/cls",),        "",       "re-show the last run\'s file changes"),\n')
    violations = check_registry(extract_slash_commands(mutated))
    assert any("/cls" in v for v in violations), violations
