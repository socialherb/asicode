"""
Tool Schema Definitions for asicode Agent

Contains all tool schema definitions (OpenAI format) used by ToolRegistry.
Extracted from tool_registry.py to reduce its size and improve SRP.
"""
from __future__ import annotations

from typing import Any

from .config.thresholds import config as _cfg  # rendered into schema text (see C1)
from .tool_handlers.shell_policy import (
    DANGEROUS_SHELL_COMMANDS as _DANGEROUS_SHELL_COMMANDS,
)
from .tool_handlers.shell_policy import (
    SHELL_TIMEOUT_DEFAULT as _SHELL_TIMEOUT_DEFAULT,
)
from .tool_handlers.shell_policy import (
    SHELL_TIMEOUT_MAX as _SHELL_TIMEOUT_MAX,
)

# The bash schema states its approval set and timeout bounds by rendering the
# policy constants, so an added dangerous command or a retuned bound reaches the
# model automatically. Restating the literals let the description drift: it named
# only `rm` after other commands became gated.
_DANGEROUS_SHELL_COMMANDS_TEXT = ", ".join(sorted(_DANGEROUS_SHELL_COMMANDS))

# Tool schemas in OpenAI format (adapted per provider in AgentLoop)


SCHEMA_MODIFY_SYMBOL = {
        "name": "modify_symbol",
        "description": (
            "Modify a symbol (function, class, method) in a file. "
            "Provide the file path, symbol name, and the new code; the system finds the symbol, "
            "handles indentation, and validates syntax. "
            "★ PREFERRED over apply_patch for symbol-level changes: no line numbers, "
            "no diff syntax, automatic indentation correction, AST precision for Python. "
            "Two modes: Full block (with def/class line — replaces the whole symbol) "
            "or Body-only (just the body — preserves signature). "
            "Supports any language with an installed tree-sitter grammar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file containing the symbol",
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name (function, class, or method). Supports 'ClassName.method_name' for methods.",
                },
                "code": {
                    "type": "string",
                    "description": "New code for the symbol. Can be a full definition block (def/class line included) or just the body (indented code without signature).",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview diff without writing (default: false).",
                },
            },
            "required": ["file_path", "symbol", "code"],
        },
    }

SCHEMA_EDIT_AST = {
        "name": "edit_ast",
        "x_python_only": True,
        "description": (
            "[Python only] Apply typed AST operations deterministically. "
            "Handles formatting automatically — no indentation errors, unlike text-based editing. "
            "Use when adding/removing a decorator, guard, or statement inside a Python function where indentation matters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the Python file to edit",
                },
                "ops": {
                    "type": "array",
                    "description": (
                        "List of AST operations to apply sequentially. Each op is a dict with 'type' (required) "
                        "plus type-specific keys:\n"
                        "  • replace_expr {old, new} — replaces 'old' in symbol scope (SINGLE expression, not full body)\n"
                        "  • delete_stmt {pattern} — delete lines matching pattern\n"
                        "  • add_import {import}\n"
                        "  • remove_import_name {name, module?}\n"
                        "  • add_class_field {class_name, field_name, field_type, field_default?}\n"
                        "  • list_append / list_remove {list_name, value}\n"
                        "  • add_guard {statement, insert_scope?, loop_variable?, loop_iterable_src?}\n"
                        "Scoped ops (replace_expr, delete_stmt, add_guard) require the top-level 'symbol'."
                    ),
                    "items": {
                        "type": "object",
                        "description": "Dict with 'type' + type-specific keys (see list above)",
                    },
                },
                "symbol": {
                    "type": "string",
                    "description": "Function/class to scope operations needing context (add_guard, replace_expr, delete_stmt). Supports 'ClassName.method' for methods.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview diff without writing (default: false). Always dry-run first when unsure.",
                },
            },
            "required": ["file_path", "ops"],
        },
    }

SCHEMA_ANCHOR_EDIT = {
        "name": "anchor_edit",
        "description": (
            "Pattern-based file editing for precise sub-symbol insertion/deletion. "
            "Uses an anchor_pattern (substring first, regex fallback) or anchor_ast_lineno "
            "to locate the target line.\n\n"
            "★ Use for: (1) inserting code at a position inside a large function, "
            "(2) disambiguating non-unique anchors by occurrence/context, "
            "(3) deleting lines matching a pattern, (4) replacing an entire line.\n\n"
            "★ Use edit_text instead for an exact unique string substitution; "
            "use apply_patch for multi-line block edits; to APPEND at end-of-file use bash `>>` "
            "or write_plan insert_after/insert_after_line."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file to edit",
                },
                "anchor_pattern": {
                    "type": "string",
                    "description": (
                        "Pattern to locate the target line. Substring match first, "
                        "regex fallback if not found. May span MULTIPLE lines ('\\n'-joined): "
                        "the FIRST non-empty line locates the anchor, subsequent lines must "
                        "strip-match. UNIQUENESS: matching more than one line fails with "
                        "anchor_not_unique unless occurrence/context disambiguates. "
                        "E.g. 'const data = {' or 'def handle_click'."
                    ),
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["insert_before", "insert_after", "replace_line", "delete"],
                    "description": "How to modify the file at the anchor position",
                },
                "code_snippet": {
                    "type": "string",
                    "description": (
                        "Code to insert or replace with. Indentation is auto-adjusted "
                        "to match the anchor line. Not needed for 'delete' mode. "
                        "Provide the exact code as it should appear."
                    ),
                },
                "occurrence": {
                    "type": "integer",
                    "description": (
                        "Which match to target: 1=first, 2=second, ..., -1=last (default: -1). "
                        "REQUIRED when the pattern matches multiple lines with no context — "
                        "else anchor_not_unique. If it exceeds the match count, falls back to "
                        "the LAST match (with a warning) rather than failing."
                    ),
                },
                "context_before": {
                    "type": "string",
                    "description": (
                        "Optional: the line immediately before the anchor must also "
                        "match this pattern (substring or regex). Disambiguates "
                        "anchors in repetitive code blocks."
                    ),
                },
                "context_after": {
                    "type": "string",
                    "description": (
                        "Optional: the line immediately after the anchor must also "
                        "match this pattern (substring or regex). Disambiguates "
                        "anchors in repetitive code blocks."
                    ),
                },
                "anchor_ast_lineno": {
                    "type": "integer",
                    "description": (
                        "Optional: a 1-indexed line number (as shown by read_file/read_symbol) "
                        "to use DIRECTLY as the anchor, bypassing string search. Use right after "
                        "reading the file — avoids anchor_miss/anchor_not_unique. When set, "
                        "anchor_pattern becomes optional. WARNING: numbers go stale — a stale "
                        "number can silently target the wrong line."
                    ),
                },
            },
            "required": ["file_path", "edit_mode"],
        },
    }

SCHEMA_EDIT_TEXT = {
    "name": "edit_text",
    "description": (
        "Replaces an exact old_string with new_string in a single file. "
        "Pure string replacement — no anchor resolution, no fuzzy matching. "
        "Use for a small, unique string substitution where apply_patch feels like overkill. "
        "old_string must be UNIQUE (exactly 1 occurrence); if it repeats, pass "
        "scope_start_line/scope_end_line or use `edits`. "
        "To APPEND a block at end-of-file, use bash `>>` or write_plan."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative or absolute path to the file to edit"
            },
            "old_string": {
                "type": "string",
                "description": "Text to replace (must match exactly and be unique — exactly 1 occurrence in the file — unless replace_all=true). Uniqueness is enforced by occurrence count, not length."
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text"
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences of old_string (default: false)",
            },
            "scope_start_line": {
                    "type": "integer",
                    "description": (
                        "1-indexed line number. Set TOGETHER with scope_end_line to restrict uniqueness "
                        "matching to that range — occurrences OUTSIDE are ignored. Both must be provided "
                        "together. Use when old_string repeats elsewhere but is unique inside the range you read."
                    ),
            },
            "scope_end_line": {
                    "type": "integer",
                    "description": (
                        "1-indexed line number (inclusive). The end of the scope range paired with "
                        "scope_start_line. See scope_start_line for semantics."
                    ),
            },
            "edits": {
                "type": "array",
                "description": (
                    "Batch mode: a list of edits to apply to the SAME file in one call, in order. "
                    "Each item: {old_string, new_string, replace_all?}. "
                    "ATOMIC: any failed match leaves the file untouched. "
                    "Cannot combine with top-level old_string/new_string."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Text to replace (must be unique unless replace_all)"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)"},
                                "scope_start_line": {"type": "integer", "description": "See top-level scope_start_line."},
                                "scope_end_line": {"type": "integer", "description": "See top-level scope_end_line."},
                },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["file_path"],
    },
}

SCHEMA_WRITE_PLAN = {
        "name": "write_plan",
        "description": (
            "Submit an ASICODE_PLAN_V1 plan for multi-file changes. "
            "Use when edits span multiple files, require create_file/replace_file ops, or need atomic execution. "
            "SIZE LIMIT: inline 'content' is for SMALL files only (under ~200 lines). "
            "Writing or rewriting a whole large file inline reliably breaks JSON escaping — "
            "write large files with bash (heredoc with QUOTED delimiter: << 'EOF') or python3 instead, then use write_plan "
            "for the remaining small edits. Its `insert_after`/`insert_after_line` ops are the natural fit for APPENDING at end-of-file (positional — no text matching, no splice-boundary friction)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": (
                        "ASICODE_PLAN_V1 plan. Must have 'kind'='ASICODE_PLAN_V1' and non-empty 'ops'. "
                        "Ops:\n"
                        "- create_file: {op, path, content}\n"
                        "- replace_file: {op, path, content}\n"
                        "- edit_blocks: {op, path, edits:[{before, after}]}\n"
                        "- insert_after: {op, path, anchor, lines[]}\n"
                        "- insert_before: {op, path, anchor, lines[]}\n"
                        "- insert_after_line: {op, path, line, lines[]} (line-based; no text matching)"
                    ),
                },
            },
            "required": ["plan"],
        },
    }

SCHEMA_APPLY_PATCH = {
        "name": "apply_patch",
        "description": (
            "★ PREFERRED write tool for line-level edits. "
            "Apply a unified diff patch to a single file using exact line ranges and context "
            "lines, avoiding ambiguous text matches. "
            "Line numbers must reflect the CURRENT file state — read the target range first. "
            "For replacing a whole function/class use modify_symbol; for a small unique "
            "string substitution use edit_text; to append at EOF use bash `>>` or write_plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch text",
                },
                "path": {
                    "type": "string",
                    "description": "File path for hunk-only patches (omit for unified diffs with ---/+++ headers)",
                },
            },
            "required": ["patch"],
        },
    }

SCHEMA_READ_FILE = {
        "name": "read_file",
        "description": (
            "Read a file by path. 'path' is required — always pass a file path. "
            f"Without start_line/end_line: files up to {_cfg.lines.READ_FILE_FULL_LINES} lines return full content; "
            "larger files return the line count plus a symbol outline, so the follow-up "
            "call can name an exact range instead of guessing. "
            "Use start_line and end_line (1-indexed, inclusive) to read specific sections. "
            "Very large ranges are truncated at an output budget; the notice names the "
            "line to resume from. "
            "Binary files — and text declaring a UTF-16/32 BOM — are reported as such "
            "instead of being returned as replacement characters; the notice names the "
            "tool that can read them (read_image, or bash `file`/`iconv`). "
            "Use when you need to inspect a line range before editing, or confirm context around a symbol. "
            "Each line is prefixed with its number and an indent gutter `│N│` = the leading-whitespace "
            "column count, so the exact indentation is readable without counting spaces. When constructing "
            "edit_text old_string/new_string or modify_symbol code, match the gutter value for the line — "
            "this is the single biggest source of avoidable write-tool retries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a file in the repository, OR an absolute path. REQUIRED ('path' is required). Repo-external absolute paths are accepted only in the trusted local CLI (unrestricted_read); the webapp confines reads to the repo."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-indexed, inclusive). Required together with end_line."
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (1-indexed, inclusive). Required together with start_line."
                }
            },
            "required": ["path"]
        }
    }

SCHEMA_GREP = {
        "name": "grep",
        "description": (
            "Search for a pattern across files in the repository. "
            "Returns matching file:line pairs with the matched line content. "
            "Supports regex patterns, file glob filtering, and context lines. "
            "Use when you know the exact string or pattern and want to find all locations — faster than find_relevant_files for precise matches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported)"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: repo root). May be repo-relative or absolute; repo-external paths are accepted only in the trusted local CLI (unrestricted_read), the webapp confines search to the repo."
                },
                "include": {
                    "type": "string",
                    "description": "File glob pattern (e.g., '*.py', '*.ts'). Omit to search all files."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default: 200, max: 500)"
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context before/after each match (default: 0). WARNING: context+N on log files or other long-line files causes token explosion — prefer `bash grep -n` then `read_file` with exact line range for log analysis."
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                }
            },
            "required": ["pattern"]
        }
    }

SCHEMA_GLOB = {
        "name": "glob",
        "description": (
            "List repository files whose path matches a glob pattern, most recently "
            "modified first. Use to answer 'what files exist' questions — locating files "
            "by name or extension, surveying a directory, finding every test/config file — "
            "instead of `bash ls`/`find`, which returns unbounded output and is not cached. "
            "Searches file PATHS only; use grep to search file CONTENTS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern. A pattern with no '/' matches the file name anywhere "
                        "in the repo ('*.py', '*_test.go'); a pattern with '/' matches the "
                        "full repo-relative path ('src/**/*.ts', 'tests/unit/*.py'). "
                        "'**' spans directories, '*' and '?' do not."
                    )
                },
                "path": {
                    "type": "string",
                    "description": "Directory to restrict the search to (default: repo root)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max paths to return (default: 200, max: 1000)"
                }
            },
            "required": ["pattern"]
        }
    }

SCHEMA_READ_SYMBOL = {
        "name": "read_symbol",
        "description": (
            "Read a symbol definition (function, class, or variable) by name. "
            "Returns the symbol's source code with surrounding context lines. "
            "Use when you need the full body of a function/class without reading the entire file. "
            "Output prefixes each line with its 1-based line number and an indent gutter "
            "`│N│` (leading-whitespace column count) so the exact indentation of every line "
            "is readable at a glance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to look for (function, class, or variable)"
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to narrow the search"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around the symbol (default: 10)"
                }
            },
            "required": ["name"]
        }
    }

SCHEMA_GET_PROJECT_INFO = {
        "name": "get_project_info",
        "description": (
            "Get project structure: frameworks (Python, JS/TS), entry points, "
            "directory organization, naming conventions, common imports. "
            "Use at session start when unfamiliar with the project layout."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }

SCHEMA_BASH = {
        "name": "bash",
        "description": (
            "Execute a shell command under bash (NOT zsh/sh). "
            f"Destructive commands ({_DANGEROUS_SHELL_COMMANDS_TEXT}) need approval — "
            "to reap a process you started, prefer `kill <pid>` over pattern killers, "
            "which match machine-wide and need approval. "
            "Use for git, cat/head/tail, python3 -c, wc, "
            "sed (no -i), and any CLI without a dedicated tool. "
            "Prefer the dedicated tools where they exist: glob over `find`/`ls` "
            "for locating files by path, grep over `grep`/`rg` for searching "
            "contents — both stay inside the repo, cap their output and are cached."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command executed under bash. ALWAYS quote glob patterns "
                        "(use 'find . -name \"*.py\"' not 'find . -name *.py') and pass multi-line "
                        "python3 code via a here-doc (<< 'PYEOF' ... PYEOF) instead of quoted -c. "
                        "For find, always exclude noise dirs (.venv, node_modules, __pycache__): "
                        'find . -name "*.py" -not -path "./.venv/*" -not -path "./node_modules/*" -not -path "./__pycache__/*"'
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Timeout in seconds (default: {_SHELL_TIMEOUT_DEFAULT}, "
                        f"max: {_SHELL_TIMEOUT_MAX}; larger values are clamped). "
                        "On expiry the command moves to a background job rather "
                        "than being killed — poll it instead of re-running."
                    ),
                },
            },
            "required": ["command"],
        },
    }

SCHEMA_FIND_SYMBOL = {
        "name": "find_symbol",
        "description": (
            "Find symbol definition (function, class, variable) by name. "
            "Returns file path, line, signature, docstring, bases, methods, decorators. "
            "Use when you know the exact symbol name and need its file:line location before reading or editing. "
            "include_inheritance=True returns up to 4 sample references (80-char context each); "
            "use find_references for all locations. Slower — triggers cross-file scan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to look for (exact match)",
                },
                "kind": {
                    "type": "string",
                    "enum": ["any", "function", "class", "variable"],
                    "description": "Kind of symbol to find (default: 'any')",
                },
                "search_path": {
                    "type": "string",
                    "description": "Relative path to narrow the search (file or directory, optional)",
                },
                "include_inheritance": {
                    "type": "boolean",
                    "description": "If True, also returns subclasses (for classes), reference count, and up to 4 sample references (80-char context each; use find_references for all locations). Slower — triggers cross-file ripgrep scan.",
                },
            },
            "required": ["name"],
        },
    }

SCHEMA_FIND_REFERENCES = {
        "name": "find_references",
        "description": (
            "Find ALL reference locations with FULL context lines. "
            "Unlike find_symbol(include_inheritance=True) which returns only 4 truncated samples, "
            "this returns every reference with its surrounding code. "
            "Use before renaming, deleting, or changing a signature to enumerate every call site."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name to find references for",
                },
                "symbol": {
                    "type": "string",
                    "description": "Alias for `name` (either may be used).",
                },
                "search_path": {
                    "type": "string",
                    "description": "Relative path to narrow the search (optional)",
                },
                "include_definitions": {
                    "type": "boolean",
                    "description": "Include definition sites in results (default: false)",
                },
            },
            "anyOf": [{"required": ["name"]}, {"required": ["symbol"]}],
        },
    }


SCHEMA_FIND_TESTS_FOR_SYMBOL = {
        "name": "find_tests_for_symbol",
        "description": (
            "Find the test files that cover a symbol or a source file. "
            "Ranks by why each one matched — a test naming the symbol outranks "
            "one that only imports its module — and labels every hit with that "
            "reason, so a weak match is visible as a weak match. "
            "Use before changing a signature (to know what will break) and after "
            "an edit (to know what to run). "
            "Beats grepping for the name: it understands test-file naming across "
            "Python (pytest), TS/JS (jest/vitest) and Go, and it reads imports, "
            "not just occurrences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Function/class/method name to find tests for",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "Repo-relative source file whose tests you want. "
                        "May be given instead of, or together with, `symbol`."
                    ),
                },
            },
            "anyOf": [{"required": ["symbol"]}, {"required": ["file_path"]}],
        },
    }


SCHEMA_FIND_RELEVANT_FILES = {
        "name": "find_relevant_files",
        "description": (
            "Search files by concept or keyword (BM25 + semantic vector search). "
            "Use when you don't know the exact file/symbol name. "
            "Handles CamelCase/snake_case across Python, JS/TS, Go, Rust, Java, and more. "
            "Returns ranked file:line pairs with snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language, code concept, or partial identifier to search for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 15)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional glob pattern to restrict by language/type (e.g., '*.py', '*.ts', '*.go', '*.rs', '*.md')",
                },
            },
            "required": ["query"],
        },
    }

SCHEMA_QUERY_DEPENDENCY_GRAPH = {
        "name": "query_dependency_graph",
        "description": (
            "Trace structural relationships in the repo graph that a single symbol lookup can't answer. "
            "Primary uses: mode=path — find HOW two symbols connect, i.e. the call chain between them (e.g. source='validate_request', target='execute_command'); "
            "mode=subgraph — map all symbols and their edges INSIDE one file (e.g. source='utils/helpers.py'). "
            "Also: mode=importers (transitive importers of a FILE), mode=reachable (downstream callees of a SYMBOL). "
            "For the common 'what breaks if I change X' question (callers + importers of a symbol), prefer analyze_change_impact instead — it bundles that in one call. "
            "NOTE: 'source' is a FILE PATH for subgraph/importers, but a SYMBOL NAME for reachable/path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["subgraph", "importers", "reachable", "path"],
                    "description": "Query mode (default: 'subgraph')",
                },
                "source": {
                    "type": "string",
                    "description": "Source — file path (subgraph/importers) or symbol name (reachable/path)",
                },
                "target": {
                    "type": "string",
                    "description": "Target symbol name — required for path mode only",
                },
                "direction": {
                    "type": "string",
                    "enum": ["downstream", "upstream", "both"],
                    "description": "Traversal direction for reachable/path modes (default: 'downstream')",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "BFS max depth (1-10, default 5)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 50)",
                },
            },
            "required": ["source"],
        },
    }

SCHEMA_ANALYZE_CHANGE_IMPACT = {
        "name": "analyze_change_impact",
        "description": (
            "Analyze impact BEFORE modifying a symbol: shows callers (upstream), callees (downstream), importers, and file dependencies. Language-agnostic. "
            "Use before renaming, deleting, or changing a signature — direction='upstream' lists every call site that must be updated; "
            "this catches transitive and cross-language references that grep/find_references miss."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to analyze (function, class, or variable)",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to disambiguate symbols with the same name",
                },
                "depth": {
                    "type": "integer",
                    "description": "Transitive depth for callee expansion (1-3, default 2)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["downstream", "upstream", "both"],
                    "description": "Impact direction: callers (upstream), callees (downstream), or both (default: 'both')",
                },
                "include_importers": {
                    "type": "boolean",
                    "description": "Include files that import the symbol's module (default: true)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max references to return (default 30)",
                },
            },
            "required": ["symbol"],
        },
    }

SCHEMA_RUN_STRUCTURAL_SCAN = {
        "name": "run_structural_scan",
        "x_python_only": True,
        "description": (
            "Run structural analysis scanners: dead code, duplicates, unused imports, contradictory logic. "
            "[Python only] Non-Python repos should use language-native tools (e.g. staticcheck for Go). "
            "Use before a cleanup or refactor to identify what can be safely removed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scanner": {
                    "type": "string",
                    "description": "Scanner name or 'all' for all scanners (enum populated at module load from scanner registry)",
                },
                "path": {
                    "type": "string",
                    "description": "Optional file or directory path to limit scanning scope",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max candidate results per scanner (default 30)",
                },
            },
            "required": ["scanner"],
        },
    }

SCHEMA_GET_FILE_OUTLINE = {
        "name": "get_file_outline",
        "description": (
            "Show file structure: classes, functions, constants with line numbers. "
            "Accepts a file path only (not directory). "
            "Use to survey a file's structure before deciding which symbol to read or edit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a FILE (not a directory) within the repository",
                },
            },
            "required": ["path"],
        },
    }

# SCHEMA_SWITCH_TO_PLANNER — removed (planner lane deactivated; see git history)

SCHEMA_SAVE_INSIGHT = {
        "name": "save_insight",
        "x_design_chat_only": True,
        "description": (
            "Save a technical insight/design decision from exploration. "
            "Only for non-obvious findings useful in future sessions. "
            "Use when you discover an architectural constraint or pattern that would be hard to re-derive next time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "insight": {
                    "type": "string",
                    "description": "The insight or design decision",
                },
                "category": {
                    "type": "string",
                    "enum": ["architecture", "pattern", "dependency", "issue", "design_decision"],
                    "description": "Category of the insight",
                },
            },
            "required": ["insight", "category"],
        },
    }

SCHEMA_DELETE_INSIGHT = {
        "name": "delete_insight",
        "x_design_chat_only": True,
        "description": (
            "Delete a design insight from .asicode/design_insights.md by matching its header line. "
            "Read the file first with the design-chat context to see available entries. "
            "Pass a substring of the entry's header line (e.g. \"2026-06-26 05:30\" or \"[architecture] 2026\") "
            "that uniquely identifies one entry. Use when an insight is no longer relevant, "
            "was saved by mistake, or has been superseded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry_match": {
                    "type": "string",
                    "description": (
                        "A substring of the insight's header line that uniquely identifies it. "
                        "Headers look like \"### [category] timestamp\" — you can match by timestamp "
                        "(e.g. \"2026-06-26 05:30\"), by category (e.g. \"[architecture]\"), or both. "
                        "Must match exactly one entry; use a more specific string if ambiguous."
                    ),
                },
            },
            "required": ["entry_match"],
        },
    }

SCHEMA_EDIT_INSIGHT = {
        "name": "edit_insight",
        "x_design_chat_only": True,
        "description": (
            "Edit (replace) an existing design insight in .asicode/design_insights.md. "
            "Read the file first with the design-chat context to see available entries. "
            "Pass a substring of the entry's header line (e.g. \"2026-06-26 05:30\" or \"[pattern] 2026\") "
            "that uniquely identifies one entry. This replaces the entire body while preserving the header."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry_match": {
                    "type": "string",
                    "description": (
                        "A substring of the insight's header line that uniquely identifies it. "
                        "Headers look like \"### [category] timestamp\" — you can match by timestamp "
                        "(e.g. \"2026-06-26 05:30\"), by category (e.g. \"[architecture]\"), or both. "
                        "Must match exactly one entry; use a more specific string if ambiguous."
                    ),
                },
                "new_insight": {
                    "type": "string",
                    "description": "The replacement body text for the insight (without the header line).",
                },
                "new_category": {
                    "type": "string",
                    "enum": ["architecture", "pattern", "dependency", "issue", "design_decision"],
                    "description": "Optional new category for the insight. If omitted, keeps the original.",
                },
            },
            "required": ["entry_match", "new_insight"],
        },
    }

SCHEMA_SEARCH_WEB = {
            "name": "search_web",
            "description": (
                "Search the web for external information. "
                "Use when you need: library documentation, latest language features, "
                "API references, external package info, current events, or any info "
                "not available in the local repository. "
                "NOT for: local code questions, repo-internal symbols, "
                "design decisions already discussed in chat history.\n"
                "Results often include a page EXCERPT (real text from the page, not just a "
                "SERP snippet) and a Published date — read those before reaching for "
                "web_fetch, which is only needed when the excerpt is insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Supports site:filter (e.g. 'python3 httpx docs site:python-httpx.org').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-15, default: 5)",
                        "default": 5,
                    },
                    "site_filter": {
                        "type": "string",
                        "description": "Optional: restrict results to a specific domain (e.g. 'docs.python.org'). Equivalent to adding site:domain to query.",
                    },
                },
                "required": ["query"],
            },
        }

SCHEMA_UPDATE_PLAN = {
    "name": "update_plan",
    "description": (
        "Create or update the work plan for a LARGE multi-step goal. "
        "Use ONLY when the request needs many steps across files (e.g. building a feature "
        "end-to-end, a broad refactor, a vague high-level goal). For small requests "
        "(1-3 steps), do NOT create a plan — just do the work. "
        "Send the FULL item list every call (full replacement, not a diff). "
        "Keep exactly one item in_progress while working, and update statuses as you go: "
        "the moment you finish a step (including running its verification), mark it 'done' "
        "in your next update_plan call before moving on — do not leave a finished step as in_progress. "
        "Re-plan freely (add/remove/rewrite items) when reality diverges from the plan. "
        "Ending without finishing everything is legitimate — but mark remaining items "
        "skipped or blocked with a reason instead of silently stopping, and explain "
        "what was not done and why in your final message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "One-line statement of the overall goal (include on the first call).",
            },
            "items": {
                "type": "array",
                "description": "The complete plan. Each item is one concrete, verifiable step.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Concrete step, ideally with how to verify it (e.g. 'Add /upload endpoint — verify with pytest tests/test_upload.py').",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done", "skipped", "blocked"],
                            "default": "pending",
                        },
                        "note": {
                            "type": "string",
                            "description": "REQUIRED for skipped/blocked: why this item was not completed.",
                        },
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["items"],
    },
}

SCHEMA_ASK_USER = {
    "name": "ask_user",
    "description": (
        "Ask the user a clarification question. Blocks until the user responds. "
        "Use when: the request is ambiguous, you need confirmation before a risky edit, "
        "multiple valid interpretations exist, or the user's intent is unclear. "
        "Always provide a 'reason' explaining why you're asking. "
        "For yes/no questions, set type='confirm' with options=['yes','no'] and a default."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user. Be specific and concise.",
            },
            "type": {
                "type": "string",
                "enum": ["free_text", "confirm", "choice"],
                "description": "free_text (default): open answer. confirm: yes/no. choice: pick from options.",
                "default": "free_text",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For type=choice: list of options the user can pick from.",
            },
            "reason": {
                "type": "string",
                "description": "Why you're asking. Helps the user provide a better answer.",
            },
            "default": {
                "type": "string",
                "description": "Default answer if the user doesn't respond or checkpoint is disabled.",
            },
        },
        "required": ["question"],
    },
}

SCHEMA_WEB_FETCH = {
    "name": "web_fetch",
    "description": (
        "Fetch and read content from a URL. Returns the page content as formatted text. "
        "Use when you need the full content of a web page, library documentation, API spec, "
        "or any URL you found via search_web. "
        "NOT for: search (use search_web instead), local files, authenticated/gated content.\n\n"
        "HTML is converted to readable text with paragraph structure preserved. If output is "
        "TRUNCATED, the marker tells you the exact start_index to pass on the next call to "
        "continue reading the rest — nothing is permanently unreachable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (https://...).",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (1000-50000, default 15000).",
                "default": 15000,
            },
            "start_index": {
                "type": "integer",
                "description": (
                    "Character offset to begin reading at (default 0). Use this to continue "
                    "reading past a previous TRUNCATION — the truncation marker reports the "
                    "exact value to pass here."
                ),
                "default": 0,
            },
        },
        "required": ["url"],
    },
}

SCHEMA_READ_IMAGE = {
    "name": "read_image",
    "description": (
        "Read text from an image file using OCR (Optical Character Recognition). "
        "Supports PNG, JPEG, GIF, BMP, TIFF. "
        "Returns extracted text with positional labels (top-left, middle-center, etc.). "
        "Use when the user pastes an image or asks you to look at a screenshot/image file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the image file",
            },
        },
        "required": ["path"],
    },
}
SCHEMA_SEARCH_DESIGN_HISTORY = {
            "name": "search_design_history",
            "x_design_chat_only": True,
            "description": "Search design chat history across sessions using BM25 + optional semantic vector search. Space-separated keywords -> BM25 relevance ranking (CodeTokenizer tokenizes CamelCase/snake_case). Pass target_session_id for other sessions (files in .asicode/design_sessions/). Use when: recalling decisions/file paths from older turns, resuming after interruption, cross-session recall, user asks about old conversations. NOT for: info already visible in current context (recent turns or already-injected summaries).\n\n**The results are from past conversation history — code state, file contents, and decisions may have changed since those turns. Always verify against the current codebase before acting on retrieved information.**\n\nSession listing: query \"list sessions\" or \"세션 목록\" to list all sessions.\nField-specific search: use search_field=decisions for saved decisions, search_field=summary for compressed summaries, search_field=all for all fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or phrase to find in the conversation history. Be specific -- function names, file paths, technical terms, or design decisions work best.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching turns to return (default: 10). Each result includes +/-1 surrounding turn for context (up to 1000 chars per turn excerpt).",
                        "default": 10,
                    },
                    "target_session_id": {
                        "type": "string",
                        "description": "Optional session ID to search. Omit to search the current session. Session IDs are the filenames in .asicode/design_sessions/ (without .json extension).",
                    },
                    "search_field": {
                        "type": "string",
                        "description": "Optional field to search within: 'content' (default, turn messages), 'decisions' (saved decisions), 'summary' (compressed summaries), 'all' (all fields).",
                        "enum": ["content", "decisions", "summary", "all"],
                        "default": "content",
                    },
                },
                "required": ["query"],
            },
        }

SCHEMA_BROWSER_ACTION = {
    "name": "browser_action",
    "description": (
        "Browser automation using Playwright (headless Chromium). "
        "Opens a browser that persists across calls within the same session.\n\n"
        "Actions:\n"
        "  navigate  — Open a URL and return the rendered page text (SPA/JS content included)\n"
        "  click     — Click an element by CSS selector\n"
        "  type      — Type text into an input field (replaces existing content)\n"
        "  extract   — Get the current page's rendered text\n"
        "  screenshot— Take a full-page screenshot (returns file path; use read_image to view)\n"
        "  evaluate  — Execute JavaScript and return the result\n"
        "  wait      — Wait for a CSS selector to appear, or wait N ms\n"
        "  close     — Close the browser and release resources\n\n"
        "★ Use instead of web_fetch when: the page is a JavaScript SPA (React/Vue), "
        "you need to interact (click/type), or you need a screenshot.\n"
        "★ The browser stays open between calls — navigate once, then click/extract repeatedly.\n"
        "★ Call action='close' when done to free memory.\n"
        "★ Heavy SPAs (YouTube/React/ad-heavy): keep default 30s timeout; for faster returns use "
        "wait_until='domcontentloaded', not a shorter timeout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Action to perform: navigate, click, type, extract, screenshot, evaluate, wait, close"
                ),
                "enum": ["navigate", "click", "type", "extract", "screenshot", "evaluate", "wait", "close"],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (required for navigate action).",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for click/type/wait actions (e.g. '#submit-btn', '.search-box input').",
            },
            "text": {
                "type": "string",
                "description": "Text to type into the selected input field (required for type action).",
            },
            "js": {
                "type": "string",
                "description": "JavaScript code to evaluate in the page context (required for evaluate action).",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Timeout in milliseconds for navigation/click/wait (default: 30000, "
                    "clamped at ~115s max — values above this are silently reduced). "
                    "Failure ceiling, not a target — keep the default for heavy SPAs. "
                    "To return faster, use wait_until='domcontentloaded', NOT a shorter timeout."
                ),
                "default": 30000,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return from navigate/extract (1000-50000, default 15000).",
                "default": 15000,
            },
            "wait_until": {
                "type": "string",
                "description": (
                    "Navigate completion condition (default 'load'). Use 'domcontentloaded' "
                    "for faster returns, or 'networkidle' for SPAs that lazy-load data "
                    "(slower; may time out on pages with persistent connections)."
                ),
                "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                "default": "load",
            },
        },
        "required": ["action"],
    },
}


SCHEMA_JOB = {
    "name": "job",
    "description": (
        "Manage background shell jobs. "
        "When a long-running bash command (e.g., a test suite) exceeds the timeout, "
        "it is automatically moved to the background. Use this tool to check its "
        "progress, list all active jobs, or kill a stuck job.\n\n"
        "Actions:\n"
        "  output — Show current stdout/stderr output for a background job. "
        "Use after a command was automatically backgrounded. "
        "Supports optional wait_timeout (seconds) to block until the job finishes.\n"
        "  kill   — Terminate a background job by its job_id.\n"
        "  list   — Show all tracked background jobs with status and elapsed time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform: list, output, or kill.",
                "enum": ["list", "output", "kill"],
            },
            "job_id": {
                "type": "string",
                "description": "Required for output and kill actions. The job_id returned when the command was backgrounded.",
            },
            "wait_timeout": {
                "type": "number",
                "description": "Optional (output action only). Max seconds to wait for the job to finish. "
                "The tool polls internally and returns only when the job completes or the timeout expires. "
                "Values above 300 are clamped to 300; the wait is cancelled early if the user interrupts. "
                "Default: 0 (return immediately with current output).",
            },
        },
        "required": ["action"],
    },
}



AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    SCHEMA_APPLY_PATCH,    # ★ PREFERRED write tool — listed first for visibility (precise line-range diff)
    SCHEMA_MODIFY_SYMBOL,  # ★ Symbol-level write tool — no line numbers, AST precision
    SCHEMA_EDIT_TEXT,      # fallback: token-level text replacement (old_string/new_string)
    SCHEMA_ANCHOR_EDIT,   # pattern-based sub-symbol insert/delete (occurrence, fuzzy, context)
    SCHEMA_EDIT_AST,
    # SCHEMA_WRITE_PLAN — re-enabled: staged-content writes (snapshot +
    # py_compile + rollback) in WriteToolsMixin._write_staged_files_directly.
    # Multi-file atomic edits and file creation.
    # (standalone create_file was removed; use a create_file op here or bash).
    SCHEMA_WRITE_PLAN,
    SCHEMA_READ_FILE,
    SCHEMA_GREP,
    SCHEMA_GLOB,      # path-pattern listing — the read-only alternative to `bash ls`/`find`
    SCHEMA_READ_SYMBOL,
    # SCHEMA_RUN_TESTS — removed: bash("pytest ...") is equivalent and more flexible; kept as internal dispatch only
    SCHEMA_GET_PROJECT_INFO,
    SCHEMA_BASH,
    SCHEMA_JOB,       # ★ Background job management (list/output/kill for long-running bash commands)
    SCHEMA_FIND_SYMBOL,
    SCHEMA_FIND_REFERENCES,
    SCHEMA_FIND_TESTS_FOR_SYMBOL,  # which tests cover this — pairs with scoped_verification
    SCHEMA_FIND_RELEVANT_FILES,
    SCHEMA_QUERY_DEPENDENCY_GRAPH,
    SCHEMA_ANALYZE_CHANGE_IMPACT,
    # SCHEMA_ESTIMATE_CHANGE_SCOPE — removed: was a loop wrapper around analyze_change_impact, replaced by direct calls
    SCHEMA_RUN_STRUCTURAL_SCAN,
    # SCHEMA_FIND_IMPORT_SOURCE — removed: grep("import.*TargetName") is equivalent
    SCHEMA_GET_FILE_OUTLINE,
    # SCHEMA_SUGGEST_EDIT_LOCATION — removed: Python-only, replaced by direct navigation tools
    # SCHEMA_EXPLORE_CODEBASE — removed: graph dependency, TS not supported, no real usage
    SCHEMA_SAVE_INSIGHT,
    SCHEMA_DELETE_INSIGHT,
    SCHEMA_EDIT_INSIGHT,
    SCHEMA_SEARCH_WEB,
    SCHEMA_BROWSER_ACTION,  # ★ New: Playwright browser automation (SPA, click, type, screenshot)
    SCHEMA_UPDATE_PLAN,   # work plan for large goals — drives the design-chat completion gate
    SCHEMA_ASK_USER,
    SCHEMA_WEB_FETCH,     # ★ Re-enabled: structured web page content fetching
    SCHEMA_READ_IMAGE,
    SCHEMA_SEARCH_DESIGN_HISTORY,
]

# Populate scanner enum from runtime registry so LLM sees valid choices.
# Import is here (not at top) to guarantee scanner_registry._auto_register() has run.
from .scanner_registry import get_registry as _get_scanner_registry  # noqa: E402

_scanner_names = [*sorted(_get_scanner_registry().list_names()), "all"]
for _schema in AGENT_TOOL_SCHEMAS:
    if _schema["name"] == "run_structural_scan":
        _schema["parameters"]["properties"]["scanner"]["enum"] = _scanner_names
        break

# Frozen set of tool names for O(1) membership checks (e.g. validating LLM
# tool-call names in agent_turn_pipeline). Computed once at import; avoids the
# per-turn list() copy + set comprehension that get_tool_schemas() would incur.
AGENT_TOOL_NAMES: frozenset = frozenset(s["name"] for s in AGENT_TOOL_SCHEMAS)

# Tools whose handler lives on DesignChatLoop, NOT on ToolRegistry: the design
# chat loop intercepts them by name before dispatch. ToolRegistry.dispatch has
# no entry for them, so advertising them to the coding-agent lane produced a
# tool the model could call but never use — "Unknown tool: save_insight.
# Available tools: [...]" — with no way for it to tell that from a real bug.
#
# Default-excluded rather than default-included: forgetting to opt IN hides a
# working tool (visible, harmless), while forgetting to opt OUT advertises a
# broken one (silent, and only the model pays). Enforced by
# test_every_advertised_tool_has_a_handler.
DESIGN_CHAT_ONLY_TOOL_NAMES: frozenset = frozenset(
    s["name"] for s in AGENT_TOOL_SCHEMAS if s.get("x_design_chat_only")
)


def _schema_variant(*, python_only: bool, design_chat: bool) -> list[dict[str, Any]]:
    return [
        s for s in AGENT_TOOL_SCHEMAS
        if (python_only or not s.get("x_python_only"))
        and (design_chat or not s.get("x_design_chat_only"))
    ]


# The four (lang_filter x surface) variants, built once at import.
# Keyed ``(include_python_only, include_design_chat)``.
#
# Module-level rather than memoized per registry: the filtering depends only on
# module constants, so every registry would compute an identical list. Sharing
# makes the selection a dict lookup and keeps one object per variant instead of
# one per registry.
#
# Note this is NOT what makes the token cache work — despite what the old
# per-instance memo's comment implied, ``estimate_tokens_from_tool_schemas``
# keys on a CONTENT fingerprint, explicitly "not id() so GC address reuse can
# never poison it". Identity stability here saves an allocation, nothing more;
# a fresh equal list per call would hit that cache just as well.
#
# Callers must NOT mutate these lists or their dicts. (Verified at the time of
# writing: every consumer iterates, and the one that extends the list —
# ``orchestrator._obr_base`` — copies with ``list(...)`` first.)
TOOL_SCHEMA_VARIANTS: dict[tuple[bool, bool], list[dict[str, Any]]] = {
    (p, d): (AGENT_TOOL_SCHEMAS if (p and d) else _schema_variant(python_only=p, design_chat=d))
    for p in (True, False) for d in (True, False)
}
TOOL_NAME_VARIANTS: dict[tuple[bool, bool], frozenset] = {
    key: frozenset(s["name"] for s in schemas)
    for key, schemas in TOOL_SCHEMA_VARIANTS.items()
}
