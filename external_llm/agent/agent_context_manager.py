"""
Context management mixin for AgentLoop.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits ContextManagerMixin, so all methods have full access to
self.config, self.registry, self._check_small_model(), etc.

Moved here:
  - Module-level git cache (_git_cache, _git_cache_ts, _GIT_CACHE_TTL, _clear_git_cache)
  - ContextTier class
  - _SMALL_MODEL_PATTERNS constant
  - System prompt template (_SYSTEM_PROMPT_TEMPLATE)
  - ContextManagerMixin class with all context build/trim/compress methods
"""
from __future__ import annotations

import ast
import logging
import subprocess
import threading
import time

from ..languages import LanguageId
from ..languages.capabilities import AnalysisCapability, is_supported
from .config.thresholds import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level git result cache (10 s TTL)
# ---------------------------------------------------------------------------

_git_cache: dict[str, str] = {}
_git_cache_ts: float = 0.0
_GIT_CACHE_TTL: float = 10.0
# Guards _git_cache / _git_cache_ts. Held only for the fast cache-check and
# the final store — NOT while running git subprocesses (which can be slow).
_git_cache_lock = threading.Lock()


def _clear_git_cache() -> None:
    """Reset the git result cache (call after any write operation)."""
    global _git_cache, _git_cache_ts
    with _git_cache_lock:
        _git_cache = {}
        _git_cache_ts = 0.0


def _run_git_raw(repo_root: str, *args: str) -> str:
    """Run a git command in repo_root, return trimmed stdout ('' on error).

    Module-level primitive used by get_git_snapshot so the snapshot fetch is
    parallelisable via a thread pool (it cannot be an instance method: the pool
    needs a picklable / plain callable, not a bound ``self._run_git``).
    """
    try:
        r = subprocess.run(
            ["git", *list(args)],
            cwd=repo_root,
            capture_output=True, text=True, check=False, timeout=8,
        )
        return r.stdout.strip()
    except Exception:
        return ""  # non-critical — never block execution


def get_git_snapshot(repo_root: str) -> dict[str, str]:
    """Return a TTL-cached git snapshot shared across the agent.

    Single source of truth for per-run git state, consumed by:
      - _collect_git_info (rollback snapshot: head_hash, has_changes)
      - _build_session_context (system-prompt injection: branch, status)

    Both used to fetch branch + status independently (5 git subprocesses per
    run: 3 here + 2 there); now a single shared snapshot fetches branch, status
    and last-commit log in PARALLEL once, cached for _GIT_CACHE_TTL seconds.

    The cache is cleared after any successful write operation (_clear_git_cache
    is registered as a write-success callback), so a stale snapshot never
    follows a successful edit. Double-checked locking keeps the slow git
    subprocess OUTSIDE the lock while the fast cache read / final store run
    INSIDE it (preventing a torn / duplicate-populated cache across threads).

    Returns {branch, status, head_hash, last_commit}; missing repo_root -> {}.
    """
    global _git_cache, _git_cache_ts
    if not repo_root:
        return {}
    _now = time.monotonic()
    with _git_cache_lock:
        if _git_cache and (_now - _git_cache_ts) < _GIT_CACHE_TTL:
            return dict(_git_cache)
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
        _futures = {
            key: _pool.submit(_run_git_raw, repo_root, *args)
            for key, args in _cmds.items()
        }
        for key, fut in _futures.items():
            try:
                _fresh[key] = fut.result(timeout=5)
            except Exception:
                _fresh[key] = ""
    except Exception:
        for key in _cmds:  # non-critical — never block execution
            _fresh.setdefault(key, "")
    # Decompose the combined log line into head_hash + last_commit.
    _log_line = _fresh.pop("log", "")
    if "\t" in _log_line:
        _fresh["head_hash"], _fresh["last_commit"] = _log_line.split("\t", 1)
    else:
        _fresh["head_hash"], _fresh["last_commit"] = _log_line, ""
    # Store under lock; re-check in case another thread populated meanwhile.
    with _git_cache_lock:
        if _git_cache and (_now - _git_cache_ts) < _GIT_CACHE_TTL:
            return dict(_git_cache)
        _git_cache = dict(_fresh)
        _git_cache_ts = _now
        return _fresh


# ---------------------------------------------------------------------------
# ContextTier
# ---------------------------------------------------------------------------

class ContextTier(str):
    """Context injection tier — controls how much startup context is loaded."""
    PLANNER = "planner"      # ~5,000 tokens: structural overview + symbol index
    MAIN_AGENT = "main_agent"  # ~2,500 tokens: lean start, tool-driven exploration
    COMPACT = "compact"      # ~1,000 tokens: small model / subagent


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
      - self._build_quick_symbol_index() -> str  (defined in this mixin)
      - self._run_git(*args) -> str               (defined in this mixin)
      - self._build_session_context(tier) -> str  (defined in this mixin)
    Context trimming/compression/eviction is delegated to a
    ``SlidingWindowContext`` instance created by ``_init_context_manager()``.
    """

    def _init_context_manager(self) -> None:
        """Create the SlidingWindowContext used for trim/compress/evict.

        Call this during host-class initialization (after ``self.config``
        and ``self._cb`` are available).
        """
        from .context_manager import SlidingWindowConfig, SlidingWindowContext

        _raw_window = getattr(self.config, "context_window_size", 60)
        _model_name = (
            getattr(self, "model", None)
            or getattr(self.config, "model_name", None)
            or ""
        )
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

    def _resolve_context_tier(self, route=None) -> str:
        """Determine context injection tier based on model size, role, and lane."""
        from .task_router import Lane
        if getattr(self.config, "is_subagent", False):
            return ContextTier.COMPACT
        if route and getattr(route, "lane", None) == Lane.PLANNER:
            return ContextTier.PLANNER
        return ContextTier.MAIN_AGENT

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def _build_rag_context(self, query: str) -> str:
        """Run BM25 relevance search and return a compact context block."""
        try:
            # Use monotonic clock for elapsed-time measurement: wall-clock
            # (time.time()) can jump backwards on NTP sync / DST transitions,
            # which would yield negative durations or never-trigger limits.
            start_time = time.monotonic()
            results = self.registry._rag_searcher.find_relevant_files(
                query, top_k=self.config.rag_top_k
            )
            search_time_ms = (time.monotonic() - start_time) * 1000

            # Record RAG search metrics
            self.performance_collector.record_rag_search(search_time_ms)

            if not results:
                return ""
            lines = [
                f"[Auto-RAG: Top-{len(results)} files relevant to request (BM25 auto-selected)]",
            ]
            for i, r in enumerate(results, 1):
                lines.append(f"  {i}. {r.file}:{r.line}  (score {r.score:.2f})  — {r.snippet[:80]}")
            lines.append("[Use as exploration starting points; verify actual content with bash (cat)]")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("RAG context build failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Session context enrichment
    # ------------------------------------------------------------------

    def _run_git(self, *args: str, max_lines: int = 0) -> str:
        """Run a git command and return stdout, trimmed. Returns '' on error."""
        try:
            r = subprocess.run(
                ["git", *list(args)],
                cwd=self.registry.repo_root,
                capture_output=True, text=True, check=False, timeout=8,
            )
            out = r.stdout.strip()
            if max_lines and out:
                lines = out.splitlines()
                out = "\n".join(lines[:max_lines])
            return out
        except Exception:
            return ""  # non-critical — never block execution

    def _build_session_context(self, tier: ContextTier = None) -> str:
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
                  PLANNER — branch + status + root structure + symbol index.
                  None defaults to MAIN_AGENT.
        """
        import os

        if tier is None:
            tier = ContextTier.MAIN_AGENT

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
        try:
            _wd = self.registry.repo_root
            parts.append(f"Working directory: {_wd}")
        except AttributeError:
            pass
        if _git_results.get("branch"):
            parts.append(f"Branch: {_git_results['branch']}")

        if tier != ContextTier.COMPACT:
            status = _git_results.get("status", "")
            if status:
                parts.append(f"Modified files (git status):\n{status}")
            else:
                parts.append("Working tree: clean (no uncommitted changes)")

        # ── 3. Project root files overview (PLANNER only) ───────────────
        if tier == ContextTier.PLANNER:
            try:
                root = self.registry.repo_root
                entries = sorted(os.listdir(root))
                py_files = [e for e in entries if LanguageId.from_path(e) is LanguageId.PYTHON and not e.startswith("_")]
                dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))
                        and not e.startswith(".") and e not in ("__pycache__", "node_modules", ".git")]
                overview_lines = []
                if py_files:
                    overview_lines.append("  Python entry files: " + ", ".join(py_files))
                if dirs:
                    overview_lines.append("  Subdirectories: " + ", ".join(dirs))
                if overview_lines:
                    parts.append("Root structure:\n" + "\n".join(overview_lines))
            except (AttributeError, TypeError):
                pass

        # ── 4. GSG: compact symbol index (PLANNER only) ─────────────────
        if tier == ContextTier.PLANNER:
            sym_index = self._build_quick_symbol_index()
            if sym_index:
                parts.append(sym_index)

        return "\n\n".join(parts) if parts else "(session context unavailable)"

    def _build_quick_symbol_index(self) -> str:
        """Build a compact top-level symbol index for system prompt injection.

        Scans key directories with a plain AST walk (no full graph build).
        Time-limited to 1.0s to avoid adding startup latency. Performs a
        broader scan across agent, graph, learning, ui, and common source
        directories to give the LLM a wider instant file→symbol mapping so it
        can skip exploratory find_symbol / bash grep turns for well-known
        classes and functions.
        """
        import os

        root = self.registry.repo_root
        # Monotonic clock — see _build_rag_context for rationale.
        start = time.monotonic()
        _TIME_LIMIT = config.counts.AGENT_CTX_BUDGET_TIME_S

        # Dynamically detect source directories (MAIN_AGENT-first: broader scan)
        _CANDIDATE_DIRS = [
            "external_llm/agent",
            "external_llm",
            "external_llm/agent/graph",
            "external_llm/agent/learning",
            "src", "lib", "app", "api",
            "ui",
        ]
        scan_dirs = [
            d for d in _CANDIDATE_DIRS
            if os.path.isdir(os.path.join(root, d))
        ]

        lines: list = []
        seen_dirs: set = set()
        _time_exceeded = False

        for scan_dir in scan_dirs:
            if _time_exceeded:
                break
            dirpath = os.path.join(root, scan_dir)
            if not os.path.isdir(dirpath) or dirpath in seen_dirs:
                continue
            seen_dirs.add(dirpath)

            try:
                fnames = sorted(os.listdir(dirpath))
            except OSError:
                continue

            for fname in fnames:
                if time.monotonic() - start > _TIME_LIMIT:
                    logger.warning(
                        "[SYM_INDEX] time limit (%.1fs) reached after %d dirs / %d files",
                        _TIME_LIMIT, len(seen_dirs), len(lines),
                    )
                    _time_exceeded = True
                    break
                if not is_supported(fname, AnalysisCapability.AGENT_CONTEXT) or fname.startswith("_"):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, root)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=fpath)
                    names = [
                        n.name for n in tree.body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                        and not n.name.startswith("_")
                    ]
                    if names:
                        lines.append(f"  {rel}: {', '.join(names)}")
                except (SyntaxError, TypeError, AttributeError):
                    continue

        if not lines:
            return ""
        return (
            "Symbol index (top-level classes & public functions — skip find_symbol for these):\n"
            + "\n".join(lines)
        )

    # ------------------------------------------------------------------
    # Initial messages builder
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self, request: str, context: str, has_native_tools: bool,
        tier: str | None = None,
    ) -> list:
        """Build the initial message list for the agent."""
        from ..client import LLMMessage  # local import to avoid circular dep

        if tier is None:
            tier = getattr(self, "_context_tier", ContextTier.MAIN_AGENT)

        system_content = _SYSTEM_PROMPT_TEMPLATE.format(
            session_context=self._build_session_context(tier),
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
        has_native_tools: bool = True,
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
        messages.append(LLMMessage(
            role="system",
            content=(
                "=== Transition to Implementation Mode ===\n\n"
                "The design analysis phase is complete. "
                "You now have the full agent tool set available. "
                "The design conversation above is preserved for context. "
                "Proceed with implementing the request below."
            ),
        ))

        # 4. Implementation request (first user turn in agent mode)
        messages.append(LLMMessage(role="user", content=request))

        return messages
