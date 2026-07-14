"""
asicode → MCP Tool Adapter.

Wraps every handler in ToolRegistry.dispatch() as an SDK @tool function
using create_sdk_mcp_server(). All 38+ tools (read, write, search, analysis,
browser, web) become in-process MCP tools that Claude Code Agent can call.

Async bridge: ToolRegistry.dispatch() is sync, SDK @tool handlers must be
async — uses asyncio.get_running_loop().run_in_executor() under the hood.
"""
from __future__ import annotations

import atexit
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from config import CLAUDE_MCP_TOOL_TIMEOUT, CLAUDE_SDK_MAX_TURNS
from external_llm.agent.tool_registry import ToolRegistry

from .verdict import CollaborationVerdict

logger = logging.getLogger(__name__)


def claude_sdk_missing_error(
    original: Optional[BaseException] = None,
) -> ImportError:
    """Build an actionable ImportError for the missing claude_agent_sdk optional dep.

    The SDK is an optional dependency (``pip install '.[collaborate]'``); a bare
    ``ModuleNotFoundError`` gives the user no remediation hint. Mirrors the
    ``vulture_scanner`` guidance pattern ("install with: pip install ...").
    """
    err = ImportError(
        "claude_agent_sdk is required for Claude Code collaboration but is not installed.\n"
        "    Install it with:  pip install '.[collaborate]'"
    )
    if original is not None:
        err.__cause__ = original
    return err


def is_claude_sdk_installed() -> bool:
    """Whether the optional ``claude_agent_sdk`` is importable.

    Uses :func:`importlib.util.find_spec` (no import side-effects) so callers can
    gate ``/claude`` before constructing the orchestrator. This is the typed
    counterpart to the lazy-import gate below — decisions never key off error
    message text.
    """
    import importlib.util

    return importlib.util.find_spec("claude_agent_sdk") is not None


def build_collaborate_install_spec() -> list[str]:
    """Package spec that adds the ``[collaborate]`` extra (no pip prefix).

    Returns just the spec portion — e.g. ``["-e", "/src/asicode[collaborate]"]``
    for an editable install or ``["asicode[collaborate]"]`` for a PyPI wheel — so
    callers feed it to the shared ``_pip_install`` helper, which owns the pip
    invocation, the live status spinner, and the PEP 668
    ``--break-system-packages`` retry. A bespoke ``subprocess.run`` in the caller
    would silently skip that retry (the bug that broke ``/claude`` on
    externally-managed Python).

    Robust to how ``asicode`` was installed — never depends on the current working
    directory (``repo_root`` is the *user's* project, not the asicode source),
    which is the failure mode a naive ``pip install '.[collaborate]'`` would hit.

    Scans *all* ``asicode`` distribution metadata rather than the first match: an
    editable source tree can expose a legacy ``asicode.egg-info`` (which has no
    ``direct_url.json``) alongside the PEP 660 ``dist-info`` in site-packages —
    only the latter records the editable source path.

    * PEP 660 editable install — ``direct_url.json`` records the source path;
      reinstall editable with the extra (``-e <source>[collaborate]``).
    * Otherwise (PyPI wheel / sdist, or metadata unreadable) — reinstall the
      published distribution name with the extra (``asicode[collaborate]``).
    """
    import importlib.metadata as md
    import json

    try:
        for dist in md.distributions():
            if dist.metadata.get("name", "").lower() != "asicode":
                continue
            raw = dist.read_text("direct_url.json")
            if not raw:
                continue
            try:
                info = json.loads(raw)
            except Exception:  # malformed — try the next asicode metadata source
                continue
            url = info.get("url", "")
            editable = info.get("dir_info", {}).get("editable", False)
            if editable and url.startswith("file://"):
                # direct_url.json stores an RFC-8089 file:// URL; percent-decode
                # the path (spaces/non-ASCII → %20 etc.) and let urlparse drop
                # any host part (file://localhost/x). A naive prefix slice would
                # feed pip a still-encoded path it cannot resolve (e.g.
                # /Users/my%20proj → not a real directory), or leak the host.
                from urllib.parse import unquote, urlparse

                source = unquote(urlparse(url).path)
                return ["-e", f"{source}[collaborate]"]
    except Exception:  # distribution scan failed — fall back to PyPI name
        logger.debug("collaborate install: distribution scan failed", exc_info=True)
    return ["asicode[collaborate]"]
def build_collaborate_install_command() -> list[str]:
    """Full ``pip install`` command (interpreter + ``-m pip install`` + spec).

    Thin wrapper over :func:`build_collaborate_install_spec` for display /
    diagnostics. For the actual install, prefer
    ``_pip_install(build_collaborate_install_spec())`` so the PEP 668
    ``--break-system-packages`` retry path is shared with every other in-REPL
    pip install rather than reimplemented (and silently skipped) per call site.
    """
    import sys

    return [sys.executable, "-m", "pip", "install", *build_collaborate_install_spec()]


# Tools excluded from MCP exposure (asicode internal only)
_TOOL_SPECIFIC_TIMEOUTS: dict[str, int] = {
    "run_structural_scan": 300,  # vulture in-process scan needs up to 3min
    # Read tools complete in ~3s normally; a 60s ceiling gives headroom for a
    # cold cache on a large repo while still failing fast on a stall (vs the
    # 120s default that masks the real problem — see _FAST_READ_EXECUTOR).
    # Applies to every _FAST_READ_TOOLS member so the fast-fail ceiling is
    # uniform across the dedicated read pool — not just read_symbol.
    "read_symbol": 60,
    "read_file": 60,
    "find_symbol": 60,
    "get_file_outline": 60,
    "get_project_info": 60,
}

_EXCLUDED_TOOLS: set[str] = {
    "delegate_to_helper",       # internal sub-agent delegation
    "update_memory",            # asicode internal memory
    "query_experience",         # asicode learning system internal
    "read_image",               # LLM sees OCR text via system; schema overhead not worth it
    "grep",                     # overlaps native Grep; low MCP added value (but Bash is now MCP-exposed since native Bash is disallowed)
    "search_web",              # Claude Code has native web search
    "web_fetch",               # Claude Code has native web fetch
    "browser_action",          # Claude Code has native browser automation
    "update_plan",             # asicode internal planner — not useful for Claude Code agent
    "save_insight",            # design-chat-only; handler lives on DesignChatLoop, not ToolRegistry
    "delete_insight",          # design-chat-only; handler lives on DesignChatLoop, not ToolRegistry
    "edit_insight",            # design-chat-only; handler lives on DesignChatLoop, not ToolRegistry
    "search_design_history",   # design-chat-only; handler lives on DesignChatLoop, not ToolRegistry
}

# Categorised tool lists for MCP annotations
_READ_ONLY_TOOLS: set[str] = {
    "read_file", "find_symbol", "find_references",
    "find_relevant_files", "get_file_outline",
    "query_dependency_graph", "analyze_change_impact", "run_structural_scan",
    "read_symbol", "get_project_info",
}

_OPEN_WORLD_TOOLS: set[str] = set()

_DESTRUCTIVE_TOOLS: set[str] = {
    "apply_patch", "modify_symbol", "write_plan", "edit_ast",
    "anchor_edit", "edit_text", "edit_file",
}

# Not strictly read-only, but safe to expose to analysis sessions.
# Read-only sessions are exposed via whitelist (_READ_ONLY_TOOLS ∪ this set) only —
# the blacklist (_DESTRUCTIVE_TOOLS) approach is fail-open: if a new handler is
# misclassified, write tools leak into the analysis session.
_ANALYSIS_SAFE_TOOLS: set[str] = {
    "ask_user",
    "bash",              # shell command execution — needed in analysis mode for file lookup/search/stats etc.
}

# ─── Read-tool executor isolation ─────────────────────────────────────────
# All MCP handlers dispatch via loop.run_in_executor(). Passing executor=None
# uses the process-wide default ThreadPoolExecutor, shared with every other
# concurrent task. When a heavy tool (run_structural_scan, graph build, RAG
# embedding) saturates the default pool, lightweight read tools (read_symbol
# is normally ~3s) starve waiting for a worker and blow past the 120s MCP
# timeout. Reproduced: read_symbol latency 0.01s -> 7.86s under 10 concurrent
# heavy blockers. Routing always-fast read tools to a dedicated pool keeps
# them responsive regardless of background load.
_FAST_READ_TOOLS: frozenset[str] = frozenset({
    "read_symbol", "read_file", "find_symbol",
    "get_file_outline", "get_project_info",
})

_FAST_READ_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="asr-read",
)
atexit.register(_FAST_READ_EXECUTOR.shutdown, wait=False)

# Static collaboration directives — inserted as system prompt append.
# NEVER mix dynamic values (paths/dates/task names) for cache prefix stability.
_STATIC_SYSTEM_APPEND = """\
## asicode Collaboration

You are collaborating with asicode, a local coding agent running on the user's machine.

- Use asicode tools (mcp__asr__*) for ALL file and code operations. \
Your native Read/Grep/Glob/Bash/Edit/Write tools are disabled.
- The task message may include pre-computed context (project info, relevant files). \
Trust it to skip redundant exploration. Fetch anything else yourself via mcp__asr__ tools \
(e.g. run_structural_scan, query_dependency_graph, find_relevant_files).
- Prefer high-level tools (find_relevant_files, read_symbol, get_file_outline) over \
raw file reads to keep context small.
- Write your answer EXACTLY ONCE. While working, stream only brief one-line \
progress notes — never the analysis itself. The complete analysis goes in the \
verdict's details field as plain markdown (no <details> HTML wrapper, do not \
restate the user's question). Restating the answer in streamed text doubles \
cost for zero value.
- Finish with a structured verdict: status, summary, details, \
confidence (0.0-1.0), suggestions (actionable items), and optionally a plan \
dict for execution. status reflects whether YOU completed the task, NOT a \
judgment of the reviewed code: use 'success' whenever you finished the \
analysis — even if you found a bug, a regression, or recommend against the \
change (put that finding in summary/details/suggestions). Use 'failure' ONLY \
if the task itself could not be done (tools blocked, gave up); 'needs_review' \
if partially done; 'insufficient_info' if you lacked context to conclude. \
The verdict is your final output — write NOTHING after calling StructuredOutput \
(no "Done", no recap)."""

# Appended only to read-only (analysis) sessions. Write sessions use _STATIC_SYSTEM_APPEND alone.
# Mode-specific, so cache prefix need only be byte-identical within each mode.
_READ_ONLY_SYSTEM_APPEND = """\
- This is a READ-ONLY analysis session. Do NOT attempt to modify files — \
no write/patch tools (apply_patch, edit_text, modify_symbol, …) are available \
to you — and do NOT spawn sub-agents (Agent/Task are disabled). Deliver every \
change you would make as concrete suggestions in the verdict, not as edits."""


def get_excluded_tools(allow_write: bool = False) -> set[str]:
    """Tools to exclude from MCP exposure for a collaboration session.

    Analysis mode (default): internal tools + destructive tools (apply_patch,
    edit_*, …) are all excluded — the agent gets a read-only view. Bash is
    allowed even in analysis mode for file inspection and utility commands.
    Pass allow_write=True for execution-mode sessions.
    """
    excluded = set(_EXCLUDED_TOOLS)
    if not allow_write:
        excluded |= _DESTRUCTIVE_TOOLS
    return excluded


def _get_tool_annotations(tool_name: str) -> Optional[Any]:
    """Build MCP ToolAnnotations for a given tool name.

    Returns None for neutral tools, or a ToolAnnotations instance.
    """
    try:
        from claude_agent_sdk import ToolAnnotations
        return ToolAnnotations(
            readOnlyHint=tool_name in _READ_ONLY_TOOLS,
            destructiveHint=tool_name in _DESTRUCTIVE_TOOLS,
            openWorldHint=tool_name in _OPEN_WORLD_TOOLS,
        )
    except ImportError:
        return None


def _convert_schema_to_input_type(schema: dict) -> dict:
    """Convert OpenAI-format tool schema to SDK input_schema dict.

    OpenAI format:
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    SDK format:
        {"path": str}  or  {"type": "object", "properties": {...}, "required": [...]}

    We use the full JSON Schema format for maximum compatibility.
    """
    params = schema.get("parameters", {})
    # Copy the full parameters block — SDK accepts JSON Schema format
    result = dict(params)
    return result


def build_asr_mcp_server(
    registry: ToolRegistry,
    server_name: str = "asicode",
    excluded_tools: Optional[set[str]] = None,
    version: str = "1.0.0",
    read_only: bool = False,
) -> Any:
    """Build an in-process MCP server exposing asicode tools.

    Args:
        registry: An initialized ToolRegistry instance.
        server_name: MCP server name (default: "asicode").
        excluded_tools: Tools to skip; defaults to internal-only tools.
        version: Server version string.
        read_only: If True, only whitelist-classified tools
            (_READ_ONLY_TOOLS ∪ _ANALYSIS_SAFE_TOOLS) are exposed —
            fail-closed against unclassified new handlers.

    Returns:
        McpSdkServerConfig — pass to ClaudeAgentOptions(mcp_servers={"asi": result}).
    """
    try:
        from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server
    except ImportError as _sdk_err:
        # ImportError covers both ModuleNotFoundError (genuinely not installed)
        # and "None in sys.modules" (explicitly blocked). Normalize to an
        # actionable message naming the install command, so EVERY caller —
        # CollaborationOrchestrator and the editor agent's mcp/server.py — gets
        # the remediation hint instead of a bare ModuleNotFoundError.
        raise claude_sdk_missing_error(_sdk_err) from _sdk_err

    excluded = _EXCLUDED_TOOLS if excluded_tools is None else excluded_tools

    sdk_tools: list[SdkMcpTool] = []

    # Schema is the source of truth — use registry's lang-filtered schemas
    # so Python-only tools (edit_ast, run_structural_scan) are hidden in
    # non-Python repos.
    for s in registry.get_tool_schemas(lang_filter=registry.repo_language):
        tool_name = s["name"]

        if tool_name in excluded:
            continue

        if (
            read_only
            and tool_name not in _READ_ONLY_TOOLS
            and tool_name not in _ANALYSIS_SAFE_TOOLS
        ):
            logger.info(
                "Read-only session: skipping unclassified tool %s", tool_name,
            )
            continue

        # Verify handler exists — catches schema-before-handler bugs.
        # Uses has_tool_handler() (not hasattr naming-convention) so tools
        # whose handler method name differs from the tool name (e.g. bash
        # → _tool_shell_exec) are correctly accepted.
        if not registry.has_tool_handler(tool_name):
            logger.warning(
                "Tool %s has a schema but no handler on ToolRegistry — skipping",
                tool_name,
            )
            continue

        description = s.get("description", f"asicode tool: {tool_name}")
        input_schema = _convert_schema_to_input_type(s)
        annotations = _get_tool_annotations(tool_name)

        # Create the async handler with closure capture via factory
        handler = _make_async_handler(registry, tool_name)

        sdk_tools.append(SdkMcpTool(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            annotations=annotations,
        ))

    logger.info(
        "Built asicode MCP server '%s' with %d tools (excluded %d)",
        server_name, len(sdk_tools), len(excluded),
    )

    return create_sdk_mcp_server(
        name=server_name,
        version=version,
        tools=sdk_tools,
    )


def _make_async_handler(registry: ToolRegistry, tool_name: str):
    """Factory: create an async handler for a specific tool.

    Uses closure to properly capture tool_name without loop-variable issues.
    """
    import time

    async def handler(args: dict) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        timeout = _TOOL_SPECIFIC_TIMEOUTS.get(tool_name, CLAUDE_MCP_TOOL_TIMEOUT)
        # Dedicated pool for always-fast read tools so they are not queued
        # behind heavy tools saturating the default executor.
        executor = _FAST_READ_EXECUTOR if tool_name in _FAST_READ_TOOLS else None
        t0 = time.monotonic()
        try:
            coro = loop.run_in_executor(
                executor,
                lambda: registry.dispatch(tool_name, args),
            )
            if timeout > 0:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
            elapsed = time.monotonic() - t0
            if result.ok:
                logger.info(
                    "MCP tool %s ok (%.2fs, %d chars)",
                    tool_name, elapsed, len(result.content or ""),
                )
                return {
                    "content": [
                        {"type": "text", "text": result.content or ""},
                    ]
                }
            else:
                error_msg = result.error or "Unknown error"
                logger.warning(
                    "MCP tool %s failed (%.2fs): %s", tool_name, elapsed, error_msg,
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"ERROR: {error_msg}",
                        }
                    ],
                    "isError": True,
                }
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            # CPython cannot forcefully terminate a running thread — wait_for only
            # cancels the asyncio Future; the run_in_executor-backed registry.dispatch
            # worker keeps running and occupies its slot. Since _FAST_READ_TOOLS use a
            # dedicated pool (max_workers=4), if 4 hung reads timeout, the pool is
            # exhausted and subsequent fast-reads immediately cascade into 60s timeouts.
            # The root fix is cooperative cancellation points in registry.dispatch, but
            # that requires registry-level changes — here we only ensure visibility.
            pool_note = (
                " (orphaned worker still running; dedicated read pool may degrade)"
                if tool_name in _FAST_READ_TOOLS
                else " (orphaned worker may still run on default pool)"
            )
            logger.error(
                "MCP tool %s timed out after %.1fs (limit %ss)%s",
                tool_name, elapsed, timeout, pool_note,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"TOOL_TIMEOUT: {tool_name} exceeded "
                            f"{timeout}s limit"
                        ),
                    }
                ],
                "isError": True,
            }
        except Exception as ex:
            elapsed = time.monotonic() - t0
            logger.exception(
                "MCP tool %s raised exception (%.2fs)", tool_name, elapsed,
            )
            return {
                "content": [
                    {"type": "text", "text": f"EXCEPTION: {ex}"},
                ],
                "isError": True,
            }

    return handler


def get_restricted_options(
    mcp_server_config: Any,
    system_prompt: Optional[str] = None,
    max_turns: int = CLAUDE_SDK_MAX_TURNS,
    permission_mode: str = "bypassPermissions",
    model: Optional[str] = None,
    allow_write: bool = False,
) -> Any:
    """Build ClaudeAgentOptions that restrict Claude Code Agent to asicode tools only.

    Claude Code's own Read/Write/Bash/Grep/Glob/Edit/WebFetch/WebSearch
    tools are disallowed — all file access goes through asicode's MCP tools.

    Cache design: the static collaboration instructions live in the system
    prompt (preset + append) with exclude_dynamic_sections=True, so the
    system prefix is byte-identical across sessions and turns — repeated
    /claude runs within the cache TTL hit the prompt cache. Volatile content
    (task, digest) stays in the user message.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    # Disallow Claude Code's native tools to force asicode tool usage.
    # "Task"/"Agent" = sub-agent spawn — block both names (SDK versions differ)
    # so a read-only collaboration session can't bypass restrictions by
    # delegating Write/Edit/Bash to an unconstrained sub-agent.
    # NOTE: advisor is NOT a permission-gated tool — it's a server-side
    # capability driven by the `advisorModel` setting. Listing it here makes
    # the CLI abort ("Permission deny rule 'advisor' matches no known tool").
    # Disable it by not inheriting advisorModel (setting_sources), not here.
    disallowed = [
        "Read", "Write", "Bash", "Grep", "Glob", "Edit",
        "WebFetch", "WebSearch", "TodoWrite",
        "NotebookEdit", "TaskCreate", "TaskUpdate", "TaskGet",
        "KillBash", "ExitPlanMode",
        "Task", "Agent",
    ]

    # Wrap schema in SDK's expected json_schema format
    schema = CollaborationVerdict.output_format_schema()
    output_format = {
        "type": "json_schema",
        "schema": schema,
    }

    # Static directives go in system append (cache prefix), caller additions follow after.
    # Analysis-only append added for read-only sessions (write sessions as-is).
    append_parts = [_STATIC_SYSTEM_APPEND]
    if not allow_write:
        append_parts.append(_READ_ONLY_SYSTEM_APPEND)
    if system_prompt:
        append_parts.append(system_prompt)

    options = ClaudeAgentOptions(
        mcp_servers={"asi": mcp_server_config},
        allowed_tools=["mcp__asr__*"],  # glob pattern for all ASR tools
        disallowed_tools=disallowed,
        max_turns=max_turns,
        permission_mode=permission_mode,
        # SDK isolation: do NOT inherit ~/.claude/settings.json. This drops
        # `advisorModel` (so the advisor server-side tool stays OFF) and
        # `autoCompactEnabled:false` (so the CLI's default auto-compaction is
        # restored — the session compacts near the window instead of dying
        # with "Prompt is too long"). Permissions/model/tools are all set
        # explicitly above, so no needed setting is lost.
        setting_sources=[],
        output_format=output_format,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": "\n\n".join(append_parts),
            # Strip dynamic sections (working dir/git status/auto-memory etc.) —
            # keeps system prompt byte-identical across sessions for cache hits
            "exclude_dynamic_sections": True,
        },
    )

    if model:
        options.model = model
    return options
