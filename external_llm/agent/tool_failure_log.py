"""Write-tool failure logging for post-hoc analysis.

Design-chat write tools (apply_patch, edit_text, write_plan, modify_symbol,
anchor_edit, edit_ast, edit_file) surface structured failure metadata via
``ToolResult.metadata["failure_class"]``. This module persists every FAILED
write-tool invocation to a JSONL log so we can later answer:

    * which tools fail most?
    * which failure_class dominates per tool?
    * what files / args shapes correlate with failure?
    * is a given failure_class trending up or down across git SHAs?

Design constraints
------------------
* **Pure / dependency-free** — no run_store, no DB. The design-chat loop has no
  run_store handle, so this module is callable from any chokepoint.
* **Never block the main flow** — every error is swallowed with a debug log.
  Logging must be strictly side-effect-only.
* **One JSON line per failure** — easy to ``grep`` / ``jq`` / pandas-load.
* **Tool-level failure forensics** — distinct from lane-level learning, which
  is aggregated in-memory and persisted to strategy_state.json (not a raw log).

Schema (one JSON object per line)::

    {
      "timestamp":     1719100000.0,
      "timestamp_iso": "2026-06-23T13:05:00",
      "tool":          "anchor_edit",
      "failure_class": "anchor_not_unique",
      "ok":            false,
      "partial":       false,
      "file_path":     "src/foo.py",
      "error":         "anchor_pattern matched 3 times...",
      "args_summary":  {"anchor_pattern": "def handle", "edit_mode": "insert_after"},
      "model":         "claude-...",
      "git_sha":       "abc1234",
      "repo":          "/abs/path/to/repo"
    }

Only the first ``_MAX_ERROR_CHARS`` of the error string are kept to bound log
size; full errors live in the tool result returned to the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections import deque
from typing import Any, Optional

from ._shared_utils import _capped_put

_logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 1200
_MAX_ARGS_SUMMARY_CHARS = 600
# Large payload fields whose full content adds no diagnostic value (they're
# huge and the failure *reason* lives in the error string, not the payload).
# Replaced by a size hint ``<N chars>``.
_ARG_KEYS_TO_DROP = {"patch", "content", "code", "after", "before", "lines"}
# Search/replace snippets: kept in TRUNCATED form (not dropped) because the
# exact indentation / prefix of old_string & new_string is usually what
# *caused* the failure (e.g. indentation mismatch in edit_text). Keeping the
# first ~80 chars preserves the leading indentation and enough context for
# post-hoc diagnosis without bloating the log.
_ARG_KEYS_TO_TRUNCATE = {"old_string", "new_string"}
_TRUNCATE_KEEP_CHARS = 80

# Write tools whose failures we log. Mirrors ToolRegistry._WRITE_TOOLS but kept
# local so this module has zero coupling to the registry.
WRITE_TOOLS = frozenset({
    "apply_patch", "write_plan", "edit_ast", "edit_file",
    "edit_text", "modify_symbol", "anchor_edit",
})

# ── Error-text fallback classifier ───────────────────────────────────────────
# Handlers set ``metadata["failure_class"]`` directly, but several failure paths
# (edit_text "old_string not found", apply_patch 3-way merge, modify_symbol
# validation, write_plan/apply_patch syntax error, apply_patch empty diff,
# anchor_edit pattern miss) return without it. Without this fallback ~80% of
# records landed in "unclassified", making the log useless for analysis.
#
# Each entry is a 4-tuple ``(substring, failure_class, only_tool, co_substring)``:
#   * ``substring``    — primary phrase to match (case-insensitive); ``None`` = any.
#   * ``failure_class``— the class assigned on match (mirrors FailureClass enum).
#   * ``only_tool``    — restrict to a tool name (e.g. "write_plan"); ``None`` = any.
#   * ``co_substring`` — an *additional* required phrase; ``None`` = none.
# The first match wins; order specific→general.
#
# NOTE: the second and third fields have distinct meanings. An earlier schema
# conflated them into one "only_tool" slot, which (a) made tool-scoped entries
# dead code (``"single line" != "anchor_edit"`` always skipped) and (b) left
# anchor_edit's "pattern ... not found in" uncaught because edit_text's
# "old_string not found in" would steal it. Splitting them fixes both.
_ERROR_PATTERNS: tuple = (
    # ── Anchor-edit failures (most specific first) ──
    ("anchor_pattern contains", "anchor_multiline_pattern", None, None),
    ("anchor_not_unique", "anchor_not_unique", None, None),
    ("matched 3 times", "anchor_not_unique", None, None),
    ("code_snippet duplicates", "fragment_duplication", None, None),
    # ── Multiline anchor resolution (anchor_edit). Two failure modes share the
    #    "multiline anchor:" prefix emitted by anchor_shared.resolve_multiline_anchor:
    #      * first line absent  → failure_class anchor_miss  (text: "first line ... not found")
    #      * later line ≠ file  → failure_class multiline_mismatch (text: "pattern line N ... does not match" / "extends past end of file")
    #    Order matters: the anchor_miss variant ("... first line ... not found")
    #    must be matched FIRST, otherwise the bare "multiline anchor" rule below
    #    would steal it into multiline_mismatch. The co_substring "first line"
    #    restricts the first rule to that variant only. ──
    ("multiline anchor", "anchor_miss", None, "first line"),
    ("multiline anchor", "multiline_mismatch", None, None),
    # ── Search-string / old_string mismatch (edit_text). MUST come BEFORE the
    #    syntax block: edit_text batch failures are reported with the wrapper
    #    "edit_text refused ... Found N occurrences of old_string ... Make
    #    old_string more unique", which ALSO contains the generic "edit_text
    #    refused" token. The match step runs before any syntax check, so the
    #    two error families never genuinely co-occur — ordering by the more
    #    specific match-failure keeps batch occurrence errors correctly classed
    #    instead of being stolen by the generic syntax wrapper. ──
    ("old_string not found", "search_string_mismatch", None, None),
    ("make old_string more unique", "search_string_mismatch", None, None),
    ("occurrences of old_string", "search_string_mismatch", None, None),
    ("closest match", "search_string_mismatch", None, None),
    # ── Syntax introduced by the edit ──
    ("would introduce a python syntax error", "syntax_invalid_after_edit", None, None),
    ("edit_text refused", "syntax_invalid_after_edit", None, None),
    ("anchor_edit introduced syntax error", "syntax_invalid_after_edit", None, None),
    ("introduced syntax errors", "syntax_invalid_after_edit", None, None),
    ("introduce a syntax error", "syntax_invalid_after_edit", None, None),
    # ── Patch application / 3-way merge ──
    ("repository lacks the necessary blob", "patch_apply_failed", None, None),
    ("3-way merge", "patch_apply_failed", None, None),
    ("patch application failed", "patch_apply_failed", None, None),
    ("repair attempts exhausted", "patch_apply_failed", None, None),
    ("does not apply", "patch_apply_failed", None, None),
    ("patch failed", "patch_apply_failed", None, None),
    # apply_patch rolled back (its own patch broke the target so badly the
    # apply pipeline crashed and restored the pre-image). Distinct signature
    # ("ROLLBACK: apply_patch") but the underlying cause is still a failed
    # patch, so it buckets with patch_apply_failed rather than "unclassified".
    ("rollback: apply_patch", "patch_apply_failed", None, None),
    # ── Empty / no-op diff (apply_patch salvage produced nothing) ──
    ("empty diff after cleaning", "no_diff_generated", None, None),
    # ── Anchor miss: "pattern ... not found in" (anchor_edit). The
    #    co_substring "pattern" excludes edit_text's "old_string not found in"
    #    (which has no "pattern" token), so the two tools disambiguate cleanly.
    #    Must come AFTER the search_string_mismatch block above. ──
    ("not found in", "anchor_miss", None, "pattern"),
    # ── modify_symbol resolution failures (symbol not found / all strategies
    #    exhausted). modify_symbol's inner error text is wrapped by write_tools
    #    as "modify_symbol failed for {path}@{symbol}: {detail}", so the wrapper
    #    substring catches every modify_symbol outcome not already classed above
    #    (arg errors like "'code' is required" are returned unwrapped by the
    #    handler's own validation, so they still hit "is required" → invalid_args
    #    below). The specific "all strategies failed" pattern also covers direct
    #    callers of symbol_modify_tool that emit it unwrapped. ──
    ("modify_symbol failed for", "modify_failed", None, None),
    ("all strategies failed", "modify_failed", None, None),
    # ── Validation: missing/required args ──
    ("is required", "invalid_args", None, None),
)


def _classify_from_error(tool: str, error: Optional[str]) -> str:
    """Best-effort failure_class from the error text.

    Used only when the handler did not set ``metadata["failure_class"]``.
    Returns ``"unclassified"`` when no pattern matches. Each pattern may carry
    an ``only_tool`` (restrict to a tool name) and/or a ``co_substring`` (an
    additional phrase that must also be present); both default to None.
    """
    if not error:
        return "unclassified"
    e = error.lower()
    for sub, fc, only_tool, co_sub in _ERROR_PATTERNS:
        if only_tool is not None and only_tool != tool:
            continue
        if co_sub is not None and co_sub not in e:
            continue
        if sub is None or sub in e:
            return fc
    return "unclassified"


def _log_path() -> str:
    """Consolidated write-tool failure log under ~/.asicode/learning/.

    Overridable via ``ASICODE_WRITE_TOOL_FAILURE_LOG`` env var (used by tests to
    redirect to a temp file without touching the real log).
    """
    override = os.environ.get("ASICODE_WRITE_TOOL_FAILURE_LOG")
    if override:
        return override
    return os.path.join(
        os.path.expanduser("~"), ".asicode", "learning", "write_tool_failures.jsonl",
    )


# Cache the short git SHA for a short window. write-tool failures often come in
# bursts (a bad plan → several failing edits in a row), so caching avoids one
# ``git rev-parse`` subprocess per record. 5s TTL keeps it fresh across turns
# while collapsing intra-burst cost to a single subprocess.
_GIT_SHA_TTL_S: float = 5.0
# Bounded entry cap (same FIFO pattern as _shared_utils._capped_put /
# _PY_WALK_CACHE): a long-lived server handling many distinct repo_roots would
# otherwise grow this path-keyed dict without bound — the TTL only refreshes
# values, it never evicts keys.
_GIT_SHA_CACHE_MAX_ENTRIES: int = 8
_git_sha_cache: dict[str, tuple] = {}  # repo_root -> (timestamp, sha)


def _git_sha(repo_root: Optional[str]) -> str:
    key = os.path.abspath(repo_root) if repo_root else os.getcwd()
    now = time.time()
    cached = _git_sha_cache.get(key)
    if cached is not None and (now - cached[0]) < _GIT_SHA_TTL_S:
        return cached[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=repo_root or os.getcwd(),
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            _capped_put(_git_sha_cache, key, (now, sha), cap=_GIT_SHA_CACHE_MAX_ENTRIES)
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _summarize_args(args: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact, redacted view of tool args for log analysis.

    Large payload fields (patch/content/code/lines/...) are replaced by a
    size hint so the log stays greppable while still showing the *shape* of
    the failing call (file_path, symbol, anchor_pattern, edit_mode, op, …).
    """
    if not isinstance(args, dict):
        return {}
    summary: dict[str, Any] = {}
    for k, v in args.items():
        if k in _ARG_KEYS_TO_TRUNCATE:
            # Keep a bounded prefix — the leading indentation/structure of
            # old_string & new_string is the single most diagnostic fact for
            # edit_text failures, so we preserve it rather than dropping.
            try:
                _sv = str(v)
            except Exception:
                _sv = "<?>"
            if len(_sv) <= _TRUNCATE_KEEP_CHARS:
                summary[k] = _sv
            else:
                summary[k] = _sv[:_TRUNCATE_KEEP_CHARS] + f"… <+{len(_sv) - _TRUNCATE_KEEP_CHARS} chars>"
            continue
        if k in _ARG_KEYS_TO_DROP:
            try:
                summary[k] = f"<{len(str(v))} chars>"
            except Exception:
                summary[k] = "<?>"
            continue
        # Keep scalar values verbatim; truncate long strings.
        if isinstance(v, str):
            summary[k] = v if len(v) <= 120 else v[:120] + "…"
        elif isinstance(v, (int, float, bool)) or v is None:
            summary[k] = v
        else:
            try:
                rendered = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                rendered = str(v)
            summary[k] = rendered if len(rendered) <= 200 else rendered[:200] + "…"
    # Bound the whole summary.
    try:
        blob = json.dumps(summary, ensure_ascii=False, default=str)
    except Exception:
        return {"_summary_error": "unserializable"}
    if len(blob) > _MAX_ARGS_SUMMARY_CHARS:
        return {"_truncated": True, "keys": list(summary.keys())}
    return summary


def _truncate(text: Optional[str], limit: int = _MAX_ERROR_CHARS) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


# ── Bounded log size ──────────────────────────────────────────────────────────
# Mirror of UnifiedStore._maybe_compact: the append-only failure log would grow
# without bound (no in-memory record list to evict from). We cap it at the same
# default as UnifiedStore (DEFAULT_MAX_RECORDS=5000) and compact by keeping the
# most-recent records. The line-count read is O(n), so it is amortised — we only
# check every ``_COMPACT_CHECK_EVERY`` appends, keeping the per-append cost O(1)
# amortised. The rewrite uses ``atomic_write_jsonl`` (same primitive as
# UnifiedStore._rewrite_all / _heal_file) so a crash mid-compaction never leaves
# a truncated log; unparseable lines are dropped during the rewrite (self-heal).
_MAX_FAILURE_LOG_RECORDS = 5000
_COMPACT_CHECK_EVERY = 64
_append_counter = 0


def _maybe_compact_log(path: str) -> None:
    """Bound the failure-log size by keeping the most recent records.

    Triggered (amortised) after each append. When the live line count exceeds
    ``_MAX_FAILURE_LOG_RECORDS``, the file is atomically rewritten keeping only
    the newest records. Corrupt/unparseable lines are dropped in the process
    (self-heal, same behaviour as UnifiedStore._heal_file). Never raises —
    compaction is strictly best-effort, like the append itself.
    """
    global _append_counter
    _append_counter += 1
    if _append_counter % _COMPACT_CHECK_EVERY != 0:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        if len(lines) <= _MAX_FAILURE_LOG_RECORDS:
            return
        from external_llm.common.atomic_io import atomic_write_jsonl
        records: list = []
        for ln in lines[-_MAX_FAILURE_LOG_RECORDS:]:
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # drop corrupt line (self-heal)
        atomic_write_jsonl(path, records)
        _logger.debug(
            "tool_failure_log: compacted to %d records (dropped %d)",
            len(records), len(lines) - len(records),
        )
    except Exception:
        _logger.debug("tool_failure_log: compaction failed", exc_info=True)
def record_write_tool_failure(
    *,
    tool: str,
    ok: bool,
    error: Optional[str],
    metadata: Optional[dict[str, Any]],
    args: Optional[dict[str, Any]],
    partial_failure: bool = False,
    model: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> None:
    """Append one failure record to the JSONL log. Never raises.

    Only records when ``ok`` is False (or ``partial_failure`` is True) AND
    ``tool`` is a recognised write tool. A success call is a no-op so callers
    can invoke this unconditionally at the chokepoint without a branch.
    """
    # Success of a non-partial write tool → nothing to learn here.
    if ok and not partial_failure:
        return
    if tool not in WRITE_TOOLS:
        return

    md = metadata if isinstance(metadata, dict) else {}
    _fc = md.get("failure_class") or md.get("final_failure_class")
    if not _fc:
        # Handler did not classify — derive from the error text so the log
        # stays useful instead of bucketing everything as "unclassified".
        _fc = _classify_from_error(tool, error)
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": tool,
        "failure_class": _fc,
        "ok": bool(ok),
        "partial": bool(partial_failure),
        "file_path": md.get("file_path") or md.get("path") or args.get("file_path") or args.get("path") if args else None,
        "error": _truncate(error),
        "args_summary": _summarize_args(args),
        "model": model or "",
        "git_sha": _git_sha(repo_root),
        "repo": os.path.abspath(repo_root) if repo_root else "",
    }
    # Attach any verify_warning / semantic_repaired hints when present.
    for hint_key in ("verify_warning", "semantic_repaired", "match_count", "attempt"):
        if hint_key in md:
            record[hint_key] = md[hint_key]

    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        _maybe_compact_log(path)
    except Exception:
        # Logging must never break the main flow.
        _logger.debug("tool_failure_log: failed to append record", exc_info=True)


def record_write_tool_failure_from_tr(
    *,
    tool: str,
    tr: Any,
    args: Optional[dict[str, Any]],
    model: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> None:
    """Convenience wrapper: extract fields from a ToolResult and record.

    ``tr`` is expected to have ``.ok``, ``.error``, ``.metadata`` and
    ``.partial_failure`` attributes (the :class:`ToolResult` shape). Falls back
    gracefully when attributes are missing.
    """
    try:
        ok = getattr(tr, "ok", True)
        error = getattr(tr, "error", None) or getattr(tr, "content", None)
        metadata = getattr(tr, "metadata", None)
        partial = getattr(tr, "partial_failure", False)
    except Exception:
        return
    record_write_tool_failure(
        tool=tool, ok=ok, error=error, metadata=metadata,
        args=args, partial_failure=partial,
        model=model, repo_root=repo_root,
    )


# ── Analysis helpers ──────────────────────────────────────────────────────────
# These let us answer "which tools fail most, in which failure_class" without
# pulling in pandas/jq. Run as a module: ``python -m external_llm.agent.tool_failure_log``


def summarize_log(path: Optional[str] = None) -> dict[str, Any]:
    """Return a breakdown of recorded write-tool failures.

    Useful for a quick "what's been failing?" overview. Returns a dict with
    per-tool counts, per-failure_class counts, and the most-recent N errors.

    The log is bounded at ``_MAX_FAILURE_LOG_RECORDS`` records (oldest evicted
    on overflow), so ``total`` reflects the post-compaction count, not the
    lifetime failure count.
    """
    path = path or _log_path()
    if not os.path.exists(path):
        return {"total": 0, "by_tool": {}, "by_failure_class": {}, "recent": []}
    by_tool: dict[str, int] = {}
    by_fc: dict[str, int] = {}
    # deque(maxlen=N) keeps only the most-recent N records in O(1) memory.
    # Previously the full list was accumulated and then sliced ([-10:]) at the
    # end — fine for small logs, but it held the entire log in memory for no
    # reason. The deque also preserves insertion order for the return value.
    recent: deque = deque(maxlen=10)
    total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            t = r.get("tool", "?")
            fc = r.get("failure_class", "?")
            by_tool[t] = by_tool.get(t, 0) + 1
            by_fc[fc] = by_fc.get(fc, 0) + 1
            recent.append(r)
    return {
        "total": total,
        "by_tool": dict(sorted(by_tool.items(), key=lambda kv: -kv[1])),
        "by_failure_class": dict(sorted(by_fc.items(), key=lambda kv: -kv[1])),
        "recent": list(recent),
    }


def _main() -> None:
    """CLI entry: ``python -m external_llm.agent.tool_failure_log``."""
    summary = summarize_log()
    print(f"Write-tool failure log: {_log_path()}")
    print(f"Total failures: {summary['total']}\n")
    print("By tool:")
    for tool, n in summary["by_tool"].items():
        print(f"  {tool:20s} {n}")
    print("\nBy failure_class:")
    for fc, n in summary["by_failure_class"].items():
        print(f"  {fc:30s} {n}")
    if summary["recent"]:
        print("\nMost recent failure:")
        r = summary["recent"][-1]
        print(f"  {r.get('timestamp_iso', '?')}  {r.get('tool')}  "
              f"[{r.get('failure_class')}]  {r.get('file_path', '')}")
        err = r.get("error", "")
        if err:
            print(f"  error: {err[:200]}")


if __name__ == "__main__":
    _main()
