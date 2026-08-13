#!/usr/bin/env python3
"""Reject redundant per-rule entries in [tool.ruff.lint] select.

A per-rule entry (e.g. ``"B904"``) that is already covered by a broader
prefix already in ``select`` (e.g. ``"B"``) changes NOTHING: the resolved
rule set is byte-identical with or without it.  Such entries are how
no-op "activation" rounds happen — the P-9 B904 round added ``"B904"``
claiming a new gate while the rule had been live under the ``"B"`` prefix
all along.  Every select round must prove a real behavior change, so any
redundant entry is a hard error here.

Prefix semantics match ruff's selector model: selector S covers rule R
iff R.startswith(S).  Entries listed in ``ignore`` are allowed to stay in
``select`` as documentation (the rule is explicitly disabled there).

Usage:
    python scripts/check_select_not_redundant.py
    python scripts/check_select_not_redundant.py <files>...  # args ignored

Any extra args (pre-commit passes filenames) are ignored — the check is
config-only.
"""

import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def _load_rule_lists() -> tuple[list[str], set[str]]:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    lint = data.get("tool", {}).get("ruff", {}).get("lint", {})
    select = list(lint.get("select", []))
    ignore = set(lint.get("ignore", []))
    return select, ignore


def main(argv: list[str]) -> int:
    del argv  # config-only check; pre-commit file args are irrelevant
    select, ignore = _load_rule_lists()
    redundant: list[tuple[str, list[str]]] = []
    for entry in select:
        if entry in ignore:
            continue  # explicitly disabled — select entry is documentation
        covering = sorted(s for s in select if s != entry and entry.startswith(s))
        if covering:
            redundant.append((entry, covering))
    if redundant:
        print(
            "check_select_not_redundant: redundant entries in "
            "[tool.ruff.lint] select (already covered by a broader prefix):"
        )
        for entry, covering in redundant:
            print(f"  {entry!r} is covered by: {', '.join(repr(c) for c in covering)}")
        print(
            "These entries change nothing — remove them (a rule already active "
            "under its prefix must not be re-added as a per-rule 'activation')."
        )
        return 1
    print("check_select_not_redundant: OK — no redundant select entries")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
