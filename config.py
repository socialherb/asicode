"""
Centralized configuration for asicode.

All environment variables are documented and loaded here.
Other modules should import from this module instead of calling os.getenv directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================
def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment variable with consistent true/false handling."""
    v = (os.getenv(name, "") or "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    """Parse integer environment variable with fallback.

    When *allow_zero* is True, ``0`` is accepted as a valid value
    (useful for settings where ``0`` means "disabled").
    """
    try:
        v = int((os.getenv(name, "") or "").strip() or str(default))
        return v if (v >= 0 if allow_zero else v > 0) else default
    except Exception:
        return default


# ============================================================
# API
# ============================================================
API_VERSION = "0.2.6"

# ============================================================
# Ollama
# ============================================================
OLLAMA_BASE: str = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")

# ============================================================
# Patch dump (CI/smoke test)
# ============================================================
PATCH_DUMP: Path = Path(os.getenv("ASICODE_PATCH_DUMP", "/tmp/asicode.last.cleaned.patch"))

# ============================================================
# Run store
# ============================================================
# ASICODE_RUNS_DIR -- directory for run artifacts
#   If set: use as-is (absolute or relative to CWD)
#   If unset: .asicode/runs relative to the config module location (repo root)
ASICODE_RUNS_DIR: str = os.getenv("ASICODE_RUNS_DIR", "").strip() or str(
    Path(__file__).parent / ".asicode" / "runs"
)

# ============================================================
# Mode flags
# ============================================================
BENCH_RAW_LLM: bool = _env_flag("ASICODE_BENCH_LLM_RAW", False)
LEGACY_DIFF_MODE: bool = _env_flag("ASICODE_LEGACY_DIFF_MODE", False)
INSTRUCTION_MODE: bool = _env_flag("ASICODE_INSTRUCTION_MODE", False)
ALLOW_MULTIFILE: bool = _env_flag("ASICODE_ALLOW_MULTIFILE", False)
STRICT_CLEAN: bool = _env_flag("ASICODE_STRICT_CLEAN", True)

# ============================================================
# Repair
# ============================================================
# ============================================================
# Context expansion
# ============================================================
CTX_MAX_FILES: int = _env_int("ASICODE_CTX_MAX_FILES", 3)
KOTLIN_MAX_SYMBOLS: int = _env_int("ASICODE_KOTLIN_MAX_SYMBOLS", 12)
KOTLIN_SYMBOL_SCAN_LIMIT: int = _env_int("ASICODE_KOTLIN_SYMBOL_SCAN_LIMIT", 8)

# ============================================================
# Diff / patch safety limits
# ============================================================
LARGE_FILE_MAX_BYTES: int = _env_int("ASICODE_LARGE_FILE_MAX_BYTES", 10_485_760)  # 10 MB
BINARY_SNIFF_BYTES: int = _env_int("ASICODE_BINARY_SNIFF_BYTES", 4096)
CONFLICT_MARKER_MAX_BYTES: int = _env_int("ASICODE_CONFLICT_MARKER_MAX_BYTES", 10 * 1024 * 1024)

# ============================================================
# Bench snippet
# ============================================================
BENCH_SNIPPET_LINES: int = _env_int("ASICODE_BENCH_SNIPPET_LINES", 120)
BENCH_SNIPPET_MAX_BYTES: int = _env_int("ASICODE_BENCH_SNIPPET_MAX_BYTES", 20000)

# ============================================================
# Repo root allowlist (security)
# ============================================================
# Allowed absolute repo root prefixes (comma-separated, e.g. /home/dev/projects,/var/repos)
_allowed_raw = os.getenv("ASICODE_ALLOWED_REPO_ROOTS", "").strip()
ALLOWED_REPO_ROOTS: list[str] = [
    p.strip() for p in _allowed_raw.split(",") if p.strip()
] if _allowed_raw else []

# ============================================================
# External LLM
# ============================================================
# Global feature toggle for external LLM providers.
# Provider keys are read by provider-specific client code in external_llm/*.

EXTERNAL_LLM_ENABLED: bool = _env_flag("EXTERNAL_LLM_ENABLED", True)

# ============================================================
# Learning system
# ============================================================
# Set ASICODE_LEARNING_ENABLED=0 to disable learning persistence to disk (in-memory computation unaffected).
LEARNING_ENABLED: bool = _env_flag("ASICODE_LEARNING_ENABLED", True)

# ============================================================
# Multi-language analysis (TS/JS symbol search, call graph, etc.)
# ============================================================
MULTILANG_SYMBOL_SEARCH: bool = _env_flag("ASICODE_MULTILANG_SYMBOL_SEARCH", True)
MULTILANG_SYMBOL_SIZE: bool = _env_flag("ASICODE_MULTILANG_SYMBOL_SIZE", True)
MULTILANG_OUTLINE: bool = _env_flag("ASICODE_MULTILANG_OUTLINE", True)
MULTILANG_CALLGRAPH: bool = _env_flag("ASICODE_MULTILANG_CALLGRAPH", True)

# Default provider (openai | anthropic | google | deepseek | zai | openrouter)
EXTERNAL_LLM_PROVIDER: str = (os.getenv("EXTERNAL_LLM_PROVIDER", "") or "deepseek").strip().lower()

# Default model (optional)
EXTERNAL_LLM_MODEL: str = (os.getenv("EXTERNAL_LLM_MODEL", "") or "").strip()

# Optional base URL override (proxy / self-hosted OpenAI-compatible endpoints)
EXTERNAL_LLM_BASE_URL: str = (os.getenv("EXTERNAL_LLM_BASE_URL", "") or "").strip()

# OpenRouter-specific options (only used when provider=openrouter).
# Pin upstream providers to keep the prompt cache warm across requests (cache
# is provider-local; auto-routing scatters requests and breaks it). Example:
#   OPENROUTER_PROVIDER_ORDER=DeepSeek        # raises cache hit rate ~50%→90%
#   OPENROUTER_PROVIDER_ORDER=DeepSeek,Hyperbolic   # ordered fallback list
# OPENROUTER_SITE_URL / OPENROUTER_APP_TITLE: optional app-attribution headers
# (HTTP-Referer / X-Title) for discoverability on OpenRouter's rankings.
# ============================================================
# Claude Agent SDK (Collaboration Mode)
# ============================================================
# Maximum turns per collaboration session
CLAUDE_SDK_MAX_TURNS: int = _env_int("CLAUDE_SDK_MAX_TURNS", 100)
# Timeout (seconds) for each individual MCP tool call.
# Prevents Claude Code from hanging indefinitely (e.g. 390s) when a tool
# deadlocks or takes too long. Use 0 to disable (0 = no timeout).
CLAUDE_MCP_TOOL_TIMEOUT: int = _env_int("CLAUDE_MCP_TOOL_TIMEOUT", 120, allow_zero=True)

# ============================================================
# Web search
# ============================================================
# Set BRAVE_API_KEY for Brave Search API (free tier available)
# Set SEARXNG_BASE_URL for self-hosted SearXNG instance (e.g. http://localhost:8888)
BRAVE_API_KEY: str = (os.getenv("BRAVE_API_KEY", "") or "").strip()
SEARXNG_BASE_URL: str = (os.getenv("SEARXNG_BASE_URL", "") or "").strip()
