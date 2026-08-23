"""
Context management mixin for AgentLoop.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits ContextManagerMixin, so all methods have full access to
self.config, self.registry, self._check_small_model(), etc.

Moved here:
  - Module-level git cache (_git_cache, _git_cache_gen, _GIT_CACHE_TTL, _clear_git_cache)
  - ContextTier class
  - _SMALL_MODEL_PATTERNS constant
  - System prompt template (_SYSTEM_PROMPT_TEMPLATE)
  - ContextManagerMixin class with all context build/trim/compress methods
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import suppress
from typing import Any  # f821-protected

from external_llm.common.repo_files import canonical_repo_key

from ._shared_utils import _capped_put

# ---------------------------------------------------------------------------
# Module-level git result cache (10 s TTL, per-root keyed)
# ---------------------------------------------------------------------------
# {repo_root: (monotonic_ts, {"branch": ..., "status": ..., ...})}
# Per-root so a request for repo B cannot receive repo A's snapshot — the
# webapp is a long-lived server process where ToolRegistry is created per-request
# but the module-level cache spans requests / repos.

_git_cache: dict[str, tuple[float, dict[str, str]]] = {}
_git_cache_gen: int = 0  # incremented on invalidation; stale writes skip cache store
_GIT_CACHE_TTL: float = 10.0
_GIT_CACHE_MAX_ENTRIES = 8
# Prompt-injection bound for the snapshot's DISPLAY status (system-prompt
# block, code-review git context, rollback metadata). A large worktree
# (thousands of dirty files) previously injected the FULL `git status --short`
# output into the token budget. Parsing consumers (orchestrator's `-z`
# changed-path detection, diff_apply's porcelain) run their own git calls and
# are deliberately NOT bound by this.
GIT_STATUS_MAX_CHARS: int = 5000
# Coalesced-invalidation window (P3): a read that arrives within this long
# after a write is served the pre-write entry instead of paying a full
# ~40 ms rebuild (3 parallel git subprocesses). Writes and the reads that
# must see them are separated by LLM calls (seconds) in the agent loop, so a
# <1 s write→read gap is the rebuild-burst case (parallel subagents, webapp
# requests), not the freshness case; every snapshot consumer (system-prompt
# status block, rollback metadata, failure-log SHA, service display) is
# display/metadata — no decision input.
_GIT_REBUILD_COALESCE_S: float = 1.0
# {repo_root: monotonic ts of last invalidation} — per-root so a write to
# repo A does not force repo B's entry (or its own, past the window) to be
# treated as permanently dirty. Entries are only added for roots that have a
# cache entry at clear time and are popped when a post-invalidation rebuild
# stores, so this stays bounded by _GIT_CACHE_MAX_ENTRIES.
_git_dirty_since: dict[str, float] = {}
# Guards _git_cache + _git_dirty_since. Held only for the fast cache-check and
# the final store — NOT while running git subprocesses (which can be slow).
_git_cache_lock = threading.Lock()


def _clear_git_cache(repo_root: str | None = None) -> None:
    """Coalesced git-cache invalidation (call after any write operation).

    Pre-P3 this emptied the whole dict, so the very next read — even
    milliseconds after the write — paid a full ~40 ms rebuild. Now entries are
    kept and stamped dirty in ``_git_dirty_since``; ``get_git_snapshot`` serves
    the pre-write entry for reads within ``_GIT_REBUILD_COALESCE_S`` and
    rebuilds afterwards, so a stale snapshot never outlives the window.

    ``repo_root`` may be omitted (legacy/global call sites): then every root
    currently in the cache is stamped — safe, just less precise. The
    generation is bumped either way so an in-flight collector that started
    before the invalidation never stores pre-write data (see
    ``get_git_snapshot``).
    """
    global _git_cache_gen
    with _git_cache_lock:
        _git_cache_gen += 1
        _now = time.monotonic()
        if repo_root:
            _key = canonical_repo_key(repo_root)
            if _key in _git_cache:
                _git_dirty_since[_key] = _now
        else:
            for _root in _git_cache:
                _git_dirty_since[_root] = _now


def _run_git_raw(repo_root: str, *args: str) -> str:
    """Run a git command in repo_root, return trimmed stdout ('' on error).

    Module-level primitive used by get_git_snapshot so the snapshot fetch is
    parallelisable via a thread pool (it cannot be an instance method: the pool
    needs a picklable / plain callable, not a bound ``self._run_git``).

    ``-c core.quotePath=false`` because this output is shown to the MODEL:
    ``status --short`` C-quotes non-ASCII paths by default, so a repo with
    Korean/CJK filenames put ``M "\\355\\225\\234\\352\\270\\200...py"`` into the
    system prompt (see ``_build_session_context``'s "Modified files (git
    status)" block). The model cannot use that as a path, and copying it into a
    tool call produces a file-not-found on a name nobody has. Set here rather
    than at each call site so every snapshot field is covered at once.

    ``quotePath=false``, not ``-z``: these results are displayed as text, so
    newline-separated output is what the callers want; ``-z`` is the right tool
    only where the output is PARSED into paths (see
    ``common.repo_files.git_list_repo_files``).
    """
    try:
        r = subprocess.run(
            ["git", "-c", "core.quotePath=false", *list(args)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):  # git best-effort
        return ""  # non-critical — never block execution


def get_git_snapshot(repo_root: str) -> dict[str, str]:
    """Return a TTL-cached git snapshot, per-root, shared across the agent.

    Single source of truth for per-run git state, consumed by:
      - _collect_git_info (rollback snapshot: head_hash, has_changes)
      - _build_session_context (system-prompt injection: branch, status)
      - EnhancedContextBuilder / SuperContextBuilder (code-review git context)

    ``status`` is truncated to GIT_STATUS_MAX_CHARS HERE, at the SSOT, so every
    prompt/metadata consumer shares one bound instead of each injection site
    (re)discovering truncation. Truthiness is preserved, so ``has_changes`` /
    ``if status:`` callers keep their semantics.

    Both used to fetch branch + status independently (5 git subprocesses per
    run: 3 here + 2 there); now a single shared snapshot fetches branch, status
    and last-commit log in PARALLEL once, cached for _GIT_CACHE_TTL seconds.

    Cache is keyed by *repo_root* so a long-lived server process (webapp) that
    serves multiple repos cannot leak repo A's snapshot to repo B's request.
    Entries are FIFO-bounded via _capped_put (cap _GIT_CACHE_MAX_ENTRIES).

    The cache is cleared after any successful MUTATING tool call —
    ``ToolRegistry._dispatch_impl`` calls _clear_git_cache at the same central
    post-success point it invalidates its other caches — so a stale snapshot
    never follows a successful edit. It is NOT a write-success *callback*: both
    ToolRegistry clone paths reset that list, and this cache is module-global
    and shared across clones, so a subagent's write has to invalidate it too.
    (It was documented as a callback and registered as none, which left every
    caller a test and the snapshot stale for the full TTL after each write.)
    Invalidation is coalesced (_clear_git_cache stamps the root dirty instead
    of emptying the dict): reads within _GIT_REBUILD_COALESCE_S of a write
    serve the pre-write entry, reads after it rebuild — so the "no stale
    snapshot after an edit" guarantee holds for every read that is not
    millisecond-adjacent to the write.
    Double-checked locking keeps the slow git
    subprocess OUTSIDE the lock while the fast cache read / final store run
    INSIDE it (preventing a torn / duplicate-populated cache across threads).

    Returns {branch, status, head_hash, last_commit}; missing repo_root -> {}.
    """
    if not repo_root:
        return {}
    # Canonical key shared with the file-index cache: callers spell the same
    # repo differently (resolved registry.repo_root vs raw request strings from
    # service.py; macOS /var vs /private/var), and an uncanonicalized key let
    # one repo occupy 2+ entries of the 8-entry cache.
    repo_root = canonical_repo_key(repo_root)
    _now = time.monotonic()
    with _git_cache_lock:
        _entry = _git_cache.get(repo_root)
        _dirty_ts = _git_dirty_since.get(repo_root)
        if (
            _entry is not None
            and (_now - _entry[0]) < _GIT_CACHE_TTL
            # Clean, or dirty but inside the coalesce window: serve the entry.
            # Dirty AND past the window falls through to a rebuild below so a
            # pre-write snapshot never outlives _GIT_REBUILD_COALESCE_S.
            and (_dirty_ts is None or (_now - _dirty_ts) < _GIT_REBUILD_COALESCE_S)
        ):
            return dict(_entry[1])
    # Read generation BEFORE the slow git subprocesses — if invalidation bumps
    # the generation while we're collecting, the result is stale and must NOT be
    # cached (it would resurrect the very state _clear_git_cache just killed).
    _gen_before = _git_cache_gen
    # Cache miss: fetch all needed git data in parallel OUTSIDE the lock
    # (git subprocesses are slow; concurrent callers must not serialise on
    # the lock while git runs).
    _cmds: dict[str, tuple] = {
        "branch": ("rev-parse", "--abbrev-ref", "HEAD"),
        "status": ("status", "--short"),
        # Full hash (rollback head_hash) + oneline "%h %s" (display) in ONE
        # git log call instead of separate `rev-parse HEAD` + `log -1`.
        "log": ("log", "-1", "--format=%H%x09%h %s"),
    }
    _fresh: dict[str, str] = {}
    try:
        from ._thread_pool import shared_pool as _pool

        _futures = {key: _pool.submit(_run_git_raw, repo_root, *args) for key, args in _cmds.items()}
        for key, fut in _futures.items():
            try:
                _fresh[key] = fut.result(timeout=5)
            except Exception:
                _fresh[key] = ""
    except Exception:
        for key in _cmds:  # non-critical — never block execution
            _fresh.setdefault(key, "")
    # Bound the display status at the SSOT: every prompt consumer
    # (_build_session_context, context_builder, super_context_builder) and the
    # rollback metadata share this one truncation contract — a consumer-side
    # slice would silently miss the next injection site.
    _fresh["status"] = (_fresh.get("status") or "")[:GIT_STATUS_MAX_CHARS]
    # Decompose the combined log line into head_hash + last_commit.
    _log_line = _fresh.pop("log", "")
    if "\t" in _log_line:
        _fresh["head_hash"], _fresh["last_commit"] = _log_line.split("\t", 1)
    else:
        _fresh["head_hash"], _fresh["last_commit"] = _log_line, ""
    # Store under lock; re-check in case another thread populated meanwhile.
    with _git_cache_lock:
        _entry = _git_cache.get(repo_root)
        _dirty_ts2 = _git_dirty_since.get(repo_root)
        if (
            _entry is not None
            and (_now - _entry[0]) < _GIT_CACHE_TTL
            # P3: only serve the re-check hit when the entry is not a pre-write
            # entry we are currently replacing. A concurrent thread's store
            # already popped the dirty stamp (its entry is post-write); this
            # root's own pre-write entry still carries a stamp past the window
            # and must NOT be served — we just collected fresh data for it.
            and (_dirty_ts2 is None or (_now - _dirty_ts2) < _GIT_REBUILD_COALESCE_S)
        ):
            return dict(_entry[1])
        # Generation changed while we were collecting — invalidation ran
        # mid-collection; don't cache stale data, just return it fresh.
        if _git_cache_gen != _gen_before:
            return _fresh
        _capped_put(_git_cache, repo_root, (_now, _fresh), _GIT_CACHE_MAX_ENTRIES)
        # The entry just stored was collected AFTER the last invalidation
        # (generation matched), so this root is no longer dirty — otherwise
        # every read past the coalesce window would rebuild forever.
        _git_dirty_since.pop(repo_root, None)
        return _fresh


# ---------------------------------------------------------------------------
# ContextTier
# ---------------------------------------------------------------------------


class ContextTier(str):
    """Context injection tier — controls how much startup context is loaded."""

    MAIN_AGENT = "main_agent"  # ~2,500 tokens: lean start, tool-driven exploration
    COMPACT = "compact"  # ~1,000 tokens: small model / subagent


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

# System prompt template
_SYSTEM_PROMPT_TEMPLATE = """\
You are asicode, an expert coding assistant. ("asicode" is your own name as a tool — it is NOT the name of the repository you are working on; never assume the user wants you to operate on a directory called "asicode".) \
You operate inside the user's current repository (see "Working directory" below) and have direct access to read, search, and modify files. Always act on that working directory unless the user explicitly names another path.

## Rules

0. Understand the context of the user's request or conversation. Distinguish whether a tool needs to be used or if the answer can be provided using your existing information or knowledge. If a tool is required, briefly explain the purpose and what needs to be done, then invoke the tool to obtain the necessary information or complete the task before responding; if a tool is not needed, respond immediately.
1. When coding, always read before you write. For unfamiliar tasks, start with get_project_info (project overview) or find_relevant_files (concept/keyword search) before falling back to read_file or find_symbol. read_file/find_symbol are best when you already know the exact target. Before using any write tool (apply_patch, edit_text, edit_ast), you MUST first read the target file/content using read_file, read_symbol, or grep. When a change alters something callers depend on (a symbol's signature, name, or its removal), run `analyze_change_impact` on that symbol FIRST: it enumerates callers/importers from the dependency graph, which is the only reliable way to find ALL affected sites — grep/find_references silently miss transitive and cross-language (TS/JS) references.
2. If it is difficult to provide an accurate answer based on recent and summarized conversation history (or if the user asks about old conversations), use the `search_design_history` tool to search past conversations and provide an answer. **Note: results are from past conversation turns — code state, file contents, and decisions may have changed since then. Verify against the current codebase before acting on retrieved information.**
3. Language — Respond in the same language as the request. Tool results may be in any language.
4. Suspect systemic patterns — A bad pattern in one place likely exists elsewhere. Check with `find_symbol`/`grep` instead of fixing in isolation.
5. If you can answer the user's question based on code you have already read, do so. If you need to read the code to provide an accurate answer, use a tool to read the code again before answering.
6. For patch requests involving implementations or modifications, handle them directly using the available editing tools (apply_patch, modify_symbol, edit_text, anchor_edit, edit_ast, write_plan).
7. Propose and implement a general solution, rather than a narrow solution that addresses only specific keyword matching or specific cases.
8. Persistence contract — your final response is the ONLY thing that survives into the next turn. All tool calls and tool outputs from this turn are discarded when the turn ends (only a compact machine-generated digest of file paths/commands is kept — no content). Therefore, whenever you did real work, your final response MUST include: (a) the exact file paths and symbols you read or changed, (b) what you changed and why, (c) key code snippets (function signatures, class definitions, critical logic) in markdown code blocks, and (d) decisions made or constraints discovered. Never end a working turn with just "done" or a vague summary — anything you omit here is permanently lost and must be re-derived by re-reading files next turn.
9. Decision thresholds (e.g., when to flag uncertainty) are adaptive. If you observe repeated false positives or incorrect routing decisions, suggest a threshold adjustment.
10. Out-of-domain detection — Before invoking any tool, check whether the user's question is actually about this codebase. If the question is a clear real-world factual query (e.g., stock price, news, weather, general knowledge, current events), use `search_web` directly. If the intent is ambiguous (could be code-related or real-world), ask for clarification before proceeding.
11. If there is a user's next request while the most recent conversation turn has not been answered, prioritize the request from the most recent conversation turn.
12. Work plan for large goals — When the request is a large or open-ended goal needing many steps (multi-file feature, broad refactor, "build X"), FIRST call `update_plan` to break it into concrete verifiable items, keep statuses updated as you work (one in_progress at a time), and re-plan freely when reality diverges. Verify each item before marking it done (run tests, check behavior). For small requests (1-3 steps), do NOT create a plan — just do the work.
13. Test-file inclusion — For bug-fix or behavior-change requests, the fix MUST be accompanied by test changes: extend the existing tests that exercise the affected code, or add a focused new test matching the project's test framework and conventions (e.g. pytest files under tests/) when none exists. A fix that only modifies source files is incomplete.

## ═══ CURRENT REPOSITORY STATE ═══
{session_context}

## Project Context (Auto-RAG + Prior Session)

{project_context}
"""


# ---------------------------------------------------------------------------
# ContextManagerMixin
# ---------------------------------------------------------------------------


class ContextManagerMixin:
    """Context building, trimming, and compression methods for AgentLoop.

    Requires the host class to expose:
      - self.config       (AgentConfig)
      - self.registry     (ToolRegistry)
      - self._cb(event, data) — stream callback helper
      - self._check_small_model() -> bool
      - self._run_git(*args) -> str               (defined in this mixin)
      - self._build_session_context(tier) -> str  (defined in this mixin)
    Context trimming/compression/eviction is delegated to a
    ``SlidingWindowContext`` instance created by ``_init_context_manager()``.
    """

    # Host-class attributes (provided by AgentLoop, not set here). Pure typing
    # scaffolding — AgentLoop.__init__ owns the runtime values.
    config: Any
    registry: Any
    llm_client: Any
    _cb: Any
    _check_small_model: Any

    def _init_context_manager(self) -> None:
        """Create the SlidingWindowContext used for trim/compress/evict.

        Call this during host-class initialization (after ``self.config``
        and ``self._cb`` are available).
        """
        from .context_manager import SlidingWindowConfig, SlidingWindowContext

        _raw_window = getattr(self.config, "context_window_size", 60)
        _model_name = getattr(self, "model", None) or getattr(self.config, "model_name", None) or ""
        # context_window_size is a MESSAGE COUNT (default 60 in config), NOT
        # tokens. max(_, 300) is a FLOOR (not an upper bound): it raises the
        # effective window so the main agent keeps a large prefix (better cache
        # economy) before the first trim. Hysteresis (SlidingWindowConfig.
        # hysteresis_factor, default 0.6) then trims to ~180 (300*0.6) once the
        # floor is exceeded. If a caller ever wires a token count in here, the
        # window would explode — keep this as a message-count contract.
        _cfg = SlidingWindowConfig(
            context_window_size=max(_raw_window, 300),
        )
        self._context_sliding = SlidingWindowContext(
            config=_cfg,
            stream_callback=self._cb,
        )

    # ------------------------------------------------------------------
    # Context trimming / compression (delegated to SlidingWindowContext)
    # ------------------------------------------------------------------

    def _trim_context(self, messages: list) -> list:
        """Apply sliding window via SlidingWindowContext."""
        mgr = getattr(self, "_context_sliding", None)
        if mgr is None:
            return messages
        return mgr.prepare_before_call(messages)

    def _trajectory_compress(self, turns: list) -> str:
        """Compress trajectory via SlidingWindowContext."""
        mgr = getattr(self, "_context_sliding", None)
        if mgr is None:
            return ""
        return mgr.trajectory_summary(turns)

    # ------------------------------------------------------------------
    # Context tier resolution
    # ------------------------------------------------------------------

    def _resolve_context_tier(self) -> str:
        """Determine context injection tier based on model size, role, and lane."""
        if getattr(self.config, "is_subagent", False):
            return ContextTier.COMPACT
        return ContextTier.MAIN_AGENT

    # ------------------------------------------------------------------
    # Session context enrichment
    # ------------------------------------------------------------------

    def _run_git(self, *args: str, max_lines: int = 0) -> str:
        """Run a git command and return stdout, trimmed. Returns '' on error."""
        try:
            r = subprocess.run(
                ["git", *list(args)],
                cwd=self.registry.repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            out = r.stdout.strip()
            if max_lines and out:
                lines = out.splitlines()
                out = "\n".join(lines[:max_lines])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):  # git best-effort
            return ""  # non-critical — never block execution
        else:
            return out

    def _build_session_context(self, tier: ContextTier | None = None) -> str:
        """Build rich session context block — injected into system prompt at startup.

        Like Claude Code's automatic git-status/branch injection, this forces the
        LLM to be aware of the current repo state before calling any tool. Git
        state is fetched via the shared get_git_snapshot (parallelised, TTL-cached
        and reused by _collect_git_info so branch/status are fetched ONCE per run,
        not twice).

        Args:
            tier: ContextTier value controlling how much context is injected.
                  COMPACT — branch only (small model / subagent).
                  MAIN_AGENT — branch + status (lean start, tool-driven exploration).
                  None defaults to MAIN_AGENT.
        """

        if tier is None:
            tier = ContextTier(ContextTier.MAIN_AGENT)

        # ── 1-2. Git snapshot (shared TTL cache with _collect_git_info) ─
        # branch + status are fetched once per run and reused by the rollback
        # snapshot, eliminating duplicate git subprocess spawns.
        try:
            _repo_root = self.registry.repo_root
        except AttributeError:
            _repo_root = ""
        _git_results: dict[str, str] = get_git_snapshot(_repo_root)

        parts: list[str] = []
        # Working directory must be explicit: "asicode" in the system prompt is
        # the TOOL's name, not the target repo. Without this line the model can
        # confuse the two and operate on the wrong directory.
        with suppress(AttributeError):
            _wd = self.registry.repo_root
            parts.append(f"Working directory: {_wd}")
        if _git_results.get("branch"):
            parts.append(f"Branch: {_git_results['branch']}")

        if tier != ContextTier.COMPACT:
            status = _git_results.get("status", "")
            if status:
                parts.append(f"Modified files (git status):\n{status}")
            else:
                parts.append("Working tree: clean (no uncommitted changes)")

        return "\n\n".join(parts) if parts else "(session context unavailable)"

    # ------------------------------------------------------------------
    # Initial messages builder
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self,
        request: str,
        context: str,
        tier: str | None = None,
    ) -> list:
        """Build the initial message list for the agent."""
        from ..client import LLMMessage  # local import to avoid circular dep

        if tier is None:
            tier = getattr(self, "_context_tier", ContextTier.MAIN_AGENT)

        _tier = tier if isinstance(tier, ContextTier) else ContextTier(ContextTier.MAIN_AGENT)
        system_content = _SYSTEM_PROMPT_TEMPLATE.format(
            session_context=self._build_session_context(_tier),
            project_context=context or "",
        )

        return [
            LLMMessage(role="system", content=system_content.rstrip()),
            LLMMessage(role="user", content=request),
        ]

    def _build_continuation_messages(
        self,
        continuation_data: dict,
        request: str,
    ) -> list:
        """Build message list from design chat continuation data.

        Reuses the system prompt from the design chat phase (built from the same
        ``_SYSTEM_PROMPT_TEMPLATE``), ensuring Chunk 1 (identity + core rules) is
        identical → Anthropic prompt cache hit on the design → agent transition.

        The design chat conversation is preserved as-is (text turns only).
        A mode-transition marker and the implementation request are appended.
        """
        from ..client import LLMMessage

        # 1. System prompt — IDENTICAL to design chat → prompt cache HIT for Chunk 1
        system_content = continuation_data.get("system_prompt", "")
        messages = [LLMMessage(role="system", content=system_content)]

        # 2. Design chat conversation history (user/assistant text turns)
        for turn in continuation_data.get("conversation", []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:  # skip empty turns
                messages.append(LLMMessage(role=role, content=content))

        # 3. Mode transition marker (system message)
        messages.append(
            LLMMessage(
                role="system",
                content=(
                    "=== Transition to Implementation Mode ===\n\n"
                    "The design analysis phase is complete. "
                    "You now have the full agent tool set available. "
                    "The design conversation above is preserved for context. "
                    "Proceed with implementing the request below."
                ),
            )
        )

        # 4. Implementation request (first user turn in agent mode)
        messages.append(LLMMessage(role="user", content=request))

        return messages
