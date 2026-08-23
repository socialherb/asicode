"""Single source of truth for "which files does this write-tool call touch?".

Four independent consumers need that answer before the handler runs:

* ``ToolRegistry._extract_write_target_paths`` — the pre-write Undo checkpoint
  and the scoped tool-result cache invalidation,
* ``WriteSafetyManager.snapshot_target_files`` — the dispatch-level rollback
  snapshot,
* ``WriteSafetyManager.count_patch_files`` — the multi-file approval gate,
* ``FileLockManager.acquire_relevant`` — per-file locking in multi-agent runs.

Each used to carry its own copy of the extraction, and the copies had already
drifted apart: only ``snapshot_target_files`` stripped the ``\\t``-separated
timestamp that plain ``diff -u`` puts in its headers, only
``_patch_target_paths`` ignored ``--- b/``, and only ``_plan_target_paths``
guarded against a non-dict op (the other two raised ``AttributeError`` from
inside the checkpoint gate, turning a repairable bad plan into a raw traceback
in front of the model).

The deeper problem the copies shared is one of ORDER. Every write handler
normalises its own arguments — ``__raw_arguments`` recovery for tool calls the
provider truncated mid-stream, and for ``write_plan`` a plan that arrived as a
JSON string, a ```` ```json ```` block, a bare op list, or top-level
``ops``/``operations``. All four consumers ran *before* the handler, on the raw
arguments, so every shape the handler repairs was invisible to them: the write
landed normally while the run silently got no Undo point, no rollback snapshot
and no file lock (reproduced for five accepted ``write_plan`` shapes,
``edit_file`` recovered from ``__raw_arguments``, and ``apply_patch`` with a
no-prefix diff).

So the normalisation lives here, ahead of the extraction, and the handlers
delegate to the same functions — the handler and the gates cannot see different
files because they run the same code. :func:`write_target_paths` is the entry
point; it never raises, because three of its four callers sit on the write path
where failing to answer must not fail the user's edit.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Tools whose targets live inside a ``patch`` / ``plan`` payload rather than in
# a scalar path argument.
PAYLOAD_TOOLS = frozenset({"apply_patch", "write_plan"})

# Scalar target keys, in precedence order. ``file_path`` is the schema name for
# most write tools; ``edit_file`` uses ``path``; ``apply_patch`` accepts either.
_SCALAR_KEYS = ("file_path", "path")

# ``@@ -old[,count] +new[,count] @@`` — the hunk header. The counts are what
# make header detection unambiguous below.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


# ── unified diff ────────────────────────────────────────────────────────────


def _clean_header_path(raw: str) -> str | None:
    """Normalise the path half of a ``---`` / ``+++`` header line.

    Strips the ``a/`` / ``b/`` prefix, and — this is the part every copy but
    ``snapshot_target_files`` was missing — the tab-separated timestamp that
    ``diff -u`` and ``git diff --no-prefix`` emit::

        --- app.py\t2020-01-01 00:00:00.000000000 +0900

    Without the strip the target resolves to ``app.py\\t2020-01-01 …``, which
    is why ``apply_patch`` used to reject the output of plain ``diff -u`` with
    "Target file does not exist".

    Returns None for ``/dev/null`` and for an empty path — both mean "this side
    of the diff names no file".
    """
    path = raw.split("\t", 1)[0].strip()
    if not path or path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path or None


def parse_patch_targets(patch: str) -> list[str]:
    """File paths named by a unified diff's headers, in order of appearance.

    Hunk-aware on purpose. Accepting a bare ``--- foo`` as a header (needed for
    ``diff -u`` / ``git diff --no-prefix`` output, which carries no ``a/``
    ``b/`` prefix) is ambiguous against a diff BODY line: deleting the text
    ``-- comment`` produces the line ``--- comment``, and deleting ``++x``
    produces ``+++x``. The ``a/``-prefix requirement the old copies used dodged
    that ambiguity by refusing the no-prefix form altogether.

    Instead the hunk header's line counts are used to know exactly where each
    hunk body ends, so a ``---`` inside a body is never read as a header. That
    admits the no-prefix form without the false positives.

    Duplicates are kept (a file named by both ``--- a/x`` and ``+++ b/x``
    appears twice); every caller dedupes after normalising to an absolute path,
    and the appearance order is worth preserving for the ones that log it.
    """
    targets: list[str] = []
    if not patch:
        return targets
    lines = patch.splitlines()
    old_left = new_left = 0
    in_hunk = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if in_hunk:
            if old_left <= 0 and new_left <= 0:
                # Hunk consumed — re-dispatch this same line as header context.
                in_hunk = False
                continue
            if line.startswith("\\"):
                pass  # "\ No newline at end of file" counts against neither side
            elif line.startswith("-"):
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            else:
                # Context line (" " prefix, or an empty line standing in for a
                # blank context line — trailing whitespace is routinely eaten
                # in transit, so an empty string must count as context).
                old_left -= 1
                new_left -= 1
            i += 1
            continue

        m = _HUNK_RE.match(line)
        if m:
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            in_hunk = old_left > 0 or new_left > 0
            i += 1
            continue

        if line.startswith("diff --git "):
            # "diff --git a/foo.py b/foo.py" — take the b/ side. Paths with
            # spaces are not recoverable from this line, but the ---/+++ pair
            # that follows names the same file, so nothing is lost.
            parts = line.split()
            if len(parts) >= 4:
                cleaned = _clean_header_path(parts[3])
                if cleaned:
                    targets.append(cleaned)
        elif line.startswith(("--- ", "+++ ")):
            cleaned = _clean_header_path(line[4:])
            if cleaned:
                targets.append(cleaned)
        i += 1
    return targets


# ── truncated / raw tool arguments ──────────────────────────────────────────


def try_repair_truncated_json(raw: str) -> dict[str, Any] | None:
    """Attempt to repair and parse a truncated JSON object string.

    Streaming truncation can cut off the end of a JSON object, e.g.::

        {"path": "example.py", "content": "line1\\nline2

    This tries adding ``"}`` (close string + object) or just ``}``.

    Moved here from ``WriteToolsEditMixin`` so the write handlers and the write
    gates recover the same arguments; the mixin method now delegates.
    """
    if not raw.startswith("{"):
        return None
    _open_b = raw.count("{")
    _close_b = raw.count("}")
    if _open_b <= _close_b:
        return None  # braces balanced or closed more than opened — not a truncation case

    # Only ever close the OBJECT — never close an unterminated string. If the
    # raw ends mid-string, the last value was cut off by the truncation;
    # appending ``"}`` would make the partial value parse as if it were
    # complete, and a tool would then silently write half a file
    # (create_file/write_plan content is the most common victim). Refusing the
    # repair lets the caller surface a clean "truncated arguments" error so the
    # LLM can retry instead.
    try:
        _parsed = json.loads(raw + "}")
        if isinstance(_parsed, dict):
            return _parsed
    except json.JSONDecodeError:
        logger.debug("truncated-JSON repair did not parse", exc_info=True)
    return None


def recover_args_from_raw(args: dict[str, Any], required_keys: tuple[str, ...]) -> dict[str, Any]:
    """Recover tool arguments from ``__raw_arguments`` when *required_keys* are absent.

    Several provider paths preserve the raw tool-call arguments as
    ``__raw_arguments`` when JSON parsing fails (stream truncation, model
    error). This re-parses that raw string and substitutes the result.

    Moved here from ``WriteToolsEditMixin`` (which now delegates) so the gates
    that run BEFORE the handler resolve targets from the same recovered
    arguments the handler will act on. While it lived only on the mixin, a call
    whose ``path`` existed solely inside ``__raw_arguments`` wrote normally but
    was invisible to the checkpoint, the rollback snapshot and the file lock.
    """
    if "__raw_arguments" not in args:
        return args
    raw = args["__raw_arguments"]
    if not isinstance(raw, str):
        return args

    # ── Full json.loads (handles complete JSON) ──
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            logger.debug("recovered args from __raw_arguments for %s", required_keys)
            return parsed
    except (ValueError, json.JSONDecodeError):
        logger.debug("__raw_arguments is not complete JSON; trying regex/repair")

    # ── Regex fallback: extract simple string keys from truncated JSON ──
    try:
        _result = dict(args)  # preserve __raw_arguments
        for _key in required_keys:
            if _key not in _result or not _result.get(_key):
                # Escape-aware value pattern: ([^"\\]|\\.)* steps over \" and \\
                # instead of truncating at the first escaped quote.
                _m = re.search(r'"' + re.escape(_key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                if _m is not None:
                    # JSON-unescape the captured value — the raw text contains
                    # literal \n / \" / \\ sequences, and writing them through
                    # un-decoded would bake a literal backslash-n into a file.
                    try:
                        _result[_key] = json.loads('"' + _m.group(1) + '"')
                    except (ValueError, json.JSONDecodeError):
                        # Leave the key missing → the tool returns a clean error
                        # rather than acting on a half-decoded value.
                        logger.debug("could not JSON-unescape recovered %r", _key, exc_info=True)
                        continue
        if all(_result.get(k) for k in required_keys):
            logger.debug(
                "recovered args from __raw_arguments (regex fallback) for %s",
                required_keys,
            )
            return _result
    except Exception:
        logger.debug("raw-argument regex recovery failed", exc_info=True)

    # ── JSON repair: close unterminated objects ──
    _repaired = try_repair_truncated_json(raw)
    if _repaired is not None:
        _result = dict(args)
        _result.update(_repaired)
        _result.pop("__raw_arguments", None)
        logger.debug("recovered args from __raw_arguments (JSON repair) for %s", required_keys)
        return _result

    return args


# ── write_plan ──────────────────────────────────────────────────────────────


def repair_plan_json(text: str) -> str:
    """Thin re-export of the handler's LLM-JSON repair, imported lazily.

    Lives in ``tool_handlers.write_tools_core``, which pulls in tree-sitter; the
    import is deferred so ``orchestrator`` and ``tool_safety`` do not pay for it
    just by importing this module.
    """
    from .tool_handlers.write_tools_core import _repair_plan_json

    return _repair_plan_json(text)


def normalize_plan(args: dict[str, Any]) -> Any:
    """Return ``write_plan``'s plan in the dict form the handler will act on.

    Mirrors ``_tool_write_plan``'s own normalisation, which accepts far more
    than the documented ``{"plan": {...}}``: a JSON string, a ```` ```json ````
    fenced block, a bare list of ops, or the ops lifted to the top level as
    ``ops`` / ``operations``. Returns the raw value unchanged when it cannot be
    made into a dict — callers treat a non-dict result as "no structured plan"
    and fall back to reading it as a unified diff.

    An unparseable string is returned FENCE-STRIPPED rather than verbatim: it
    is both what ``_tool_write_plan`` quotes back in its "plan must be a valid
    JSON object. Received: …" rejection, and the right text to hand the diff
    parser when the model wrapped a patch in a ```` ``` ```` block.
    """
    plan = args.get("plan")
    if not plan:
        # Ops lifted to the top level — the handler wraps these into a plan.
        ops = args.get("ops")
        if ops is None:
            ops = args.get("operations")
        if ops is None:
            return None
        return {"kind": "ASICODE_PLAN_V1", "ops": ops}

    if isinstance(plan, str):
        stripped = plan.strip()
        md_m = _FENCE_RE.search(stripped)
        if md_m:
            stripped = md_m.group(1).strip()
        try:
            plan = json.loads(stripped)
        except (ValueError, json.JSONDecodeError):
            try:
                plan = json.loads(repair_plan_json(stripped))
            except Exception:
                return stripped  # not JSON — the caller reads it as a diff

    if isinstance(plan, list):
        return {"kind": "ASICODE_PLAN_V1", "ops": plan}
    return plan


def plan_target_paths(plan: Any) -> list[str]:
    """Target paths named by a normalised plan.

    A string plan is read as a unified diff (``write_plan`` accepts one). A
    dict plan yields its ops' ``path`` fields; a bare ``{"path": ...}`` plan
    with no ops list targets that single file.

    Non-dict ops are SKIPPED rather than dereferenced. Two of the three former
    copies did ``op.get("path")`` unguarded, so a plan whose ops were strings
    raised ``AttributeError`` out of the pre-write checkpoint gate and aborted
    the dispatch — replacing the handler's "each op must be a JSON object with
    'op', 'path', …" guidance with a raw traceback the model cannot act on.
    """
    if isinstance(plan, str):
        return parse_patch_targets(plan)
    if not isinstance(plan, dict):
        return []
    plan_ops = plan.get("ops") or plan.get("operations") or []
    if not plan_ops and "path" in plan:
        plan_ops = [plan]
    if not isinstance(plan_ops, list):
        return []
    targets: list[str] = []
    for op in plan_ops:
        if not isinstance(op, dict):
            continue
        path = op.get("path")
        if path:
            targets.append(str(path))
    return targets


# ── entry point ─────────────────────────────────────────────────────────────


def _infer_tool_name(args: dict[str, Any]) -> str | None:
    """Guess the payload tool from the argument keys.

    ``FileLockManager.acquire_relevant`` is also called from the webapp with a
    bare ``{"patch": …}`` and no tool name, so the payload shape has to be able
    to speak for itself.
    """
    if "patch" in args:
        return "apply_patch"
    if "plan" in args or "ops" in args or "operations" in args:
        return "write_plan"
    return None


def write_target_paths(tool_name: str | None, args: dict[str, Any] | None) -> list[str]:
    """Repo-relative (or absolute, as given) target paths for a write-tool call.

    Returns ``[]`` when the targets cannot be determined — callers distinguish
    "no targets" from "unknown scope" themselves, since for cache invalidation
    the safe fallback is a full clear while for the checkpoint it is to capture
    nothing and let a later write extend the snapshot.

    Never raises. Three of the four callers run on the write path, where the
    contract is that a bookkeeping failure must not fail the user's edit; the
    checkpoint gate documents exactly that and used to have it broken by this
    function's predecessor raising before the gate's own try block was entered.
    """
    try:
        args = args or {}
        if not isinstance(args, dict):
            return []
        name = tool_name or _infer_tool_name(args)

        if name in PAYLOAD_TOOLS or name is None:
            args = recover_args_from_raw(args, ("patch", "path", "file_path"))
        else:
            args = recover_args_from_raw(args, ("file_path", "path"))

        targets: list[str] = []
        if name in PAYLOAD_TOOLS or name is None:
            patch = args.get("patch")
            if isinstance(patch, str) and patch:
                targets.extend(parse_patch_targets(patch))
            if not targets:
                targets.extend(plan_target_paths(normalize_plan(args)))

        if not targets:
            for key in _SCALAR_KEYS:
                value = args.get(key)
                if value and isinstance(value, str):
                    targets.append(value)
                    break
    except Exception:
        logger.debug(
            "write target extraction failed for %s — treating scope as unknown",
            tool_name,
            exc_info=True,
        )
        return []
    else:
        return targets
