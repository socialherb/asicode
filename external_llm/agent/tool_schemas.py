"""
Tool Schema Definitions for asicode Agent

Contains all tool schema definitions (OpenAI format) used by ToolRegistry.
Extracted from tool_registry.py to reduce its size and improve SRP.
"""
from __future__ import annotations

from typing import Any

# Tool schemas in OpenAI format (adapted per provider in AgentLoop)


SCHEMA_MODIFY_SYMBOL = {
        "name": "modify_symbol",
        "description": (
            "Modify a symbol (function, class, method) in a file. "
            "You provide the file path, symbol name, and the new code. "
            "The system automatically finds the symbol, calculates the correct line range "
            "(including decorators), handles indentation, and validates syntax. "
            "Supports Python, TypeScript, JavaScript, Go, Java, and Kotlin.\n\n"
            "Supports two modes:\n"
            "  • Full block: provide `def foo(...):\\n    ...` (with def/class line) — "
            "replaces entire symbol including decorators\n"
            "  • Body-only: provide just the body lines (without def/class) — "
            "replaces only the body, preserving signature + decorators\n\n"
            "★ PREFERRED over apply_patch for symbol-level changes: no line numbers, "
            "no diff syntax, automatic indentation correction, AST precision for Python. "
            "Falls back to surgical search/replace for non-Python files."
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
            "Use dry_run=true to preview the diff before writing. "
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
                    "description": "List of AST operations to apply sequentially. Each op is a dict with 'type' (required) and type-specific fields (see items description).",
                    "items": {
                        "type": "object",
                        "description": (
                            "Dict with 'type' + type-specific keys:\n"
                              "  • replace_expr {old, new} — replace first 'old' in symbol scope. NOTE: replaces a SINGLE expression/statement (one line) — NOT the full function body.\n"
                            "  • delete_stmt   {pattern} — delete lines matching pattern\n"
                            "  • add_import    {import} — e.g. 'from os import path'\n"
                            "  • remove_import_name {name, module?} — remove name from import\n"
                            "  • add_class_field {class_name, field_name, field_type, field_default?}\n"
                            "  • list_append / list_remove {list_name, value} — for __all__ etc.\n"
                            "  • add_guard {statement, insert_scope?, loop_variable?,\n"
                            "                loop_iterable_src?} — early-return guard\n"
                            "Scoped ops (replace_expr, delete_stmt, add_guard) require the top-level 'symbol'."
                        ),
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
            "Uses an anchor_pattern (substring match first, regex fallback) to locate "
            "the target line — no line numbers needed. Alternatively, supply "
            "`anchor_ast_lineno` (a 1-indexed line from a fresh read_file/read_symbol) "
            "to bypass string search entirely. Supports occurrence selection "
            "(1=first, -1=last), context-before/after disambiguation, and fuzzy fallback "
            "when exact match fails (string path only).\n\n"
            "★ Use for: (1) inserting code at a specific position inside a large "
            "function (sub-symbol edit), (2) when the anchor text is not globally "
            "unique but can be disambiguated by occurrence or context, "
            "(3) deleting lines matching a pattern, (4) replacing an entire line.\n\n"
            "★ Use edit_text instead for an exact unique string substitution; "
            "use apply_patch for multi-line block edits.\n\n★ To APPEND at end-of-file, use bash `>>` or write_plan `insert_after`/`insert_after_line` instead of anchoring on the last line.\n\n"
            "Modes:\n"
            "  • insert_before — insert code_snippet before the matched line\n"
            "  • insert_after  — insert code_snippet after the matched line\n"
            "  • replace_line  — replace the entire matched line with code_snippet\n"
            "  • delete        — delete matched line(s); no code_snippet needed"
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
                        "Pattern to locate the target line. MUST be a SINGLE LINE "
                        "(no '\\n') — multi-line patterns are rejected at runtime. "
                        "Substring match first (simple, safe for code with special chars), "
                        "regex fallback if substring not found. For a multi-line block, use "
                        "the FIRST line as anchor and disambiguate with context_before/context_after. "
                        "E.g. 'const data = {' or 'def handle_click'. "
                        "UNIQUENESS: if the pattern matches MORE THAN ONE line and you leave "
                        "occurrence/context unset, the call fails with anchor_not_unique "
                        "(default occurrence=-1 silently picks the LAST match, which is "
                        "ambiguous). To avoid this entirely, pass `anchor_ast_lineno` "
                        "instead. Otherwise make the pattern unique, or set `occurrence`, "
                        "`context_before`, or `context_after` to disambiguate."
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
                        "REQUIRED when anchor_pattern matches multiple lines — otherwise the "
                        "call fails with anchor_not_unique. Prefer making the pattern unique "
                        "instead, and use this only when uniqueness is impractical."
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
                        "Optional: a 1-indexed line number (as shown by read_file/read_symbol "
                        "line prefixes) to use DIRECTLY as the anchor, bypassing string/regex/"
                        "fuzzy search entirely. Use this right after reading the file — it "
                        "eliminates anchor_miss and anchor_not_unique failures. When set, "
                        "anchor_pattern becomes optional (kept only as a readability hint). "
                        "WARNING: line numbers are fragile — if the file was edited since you "
                        "read it, the number may be stale and silently target the wrong line."
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
        "No anchor resolution, no fuzzy matching — pure string replacement. "
        "A blocking syntax gate refuses edits that would break Python parsing, and "
        "non-blocking semantic diagnostics are surfaced post-write (like apply_patch). "
        "Use when the change is a small, unique string substitution and apply_patch feels like overkill. "
        "IMPORTANT: old_string must be UNIQUE (exactly 1 occurrence) in the file — uniqueness, not length, is enforced. "
        "Include 2-3 lines of surrounding context ONLY when your anchor matches multiple times.\n\n"
        "If old_string legitimately repeats elsewhere (e.g. similar methods) but is unique inside the "
        "line range you read, pass scope_start_line + scope_end_line to restrict matching to that range — "
        "occurrences outside the scope are ignored.\n\n"
        "Two modes (mutually exclusive):\n"
        "• Single: pass old_string + new_string (optionally replace_all).\n"
        "• Batch (MultiEdit): pass `edits` — a list of {old_string, new_string, replace_all?} objects. "
        "Edits apply in order; each later edit sees the result of earlier ones. The batch is ATOMIC: "
        "if any edit fails to match, the file is left untouched and the failing edit's index is reported. "
        "The file is written exactly once. Use batch mode for several unrelated substitutions in one call. To APPEND a whole block at end-of-file, use bash `>>` or write_plan `insert_after`/`insert_after_line` — old_string splicing at EOF just invites uniqueness and syntax-gate friction."
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
                "description": "Text to replace (must match exactly and be unique — exactly 1 occurrence in the file — unless replace_all=true). "
                    "There is NO minimum length: uniqueness is enforced by occurrence count, not length. "
                    "A short anchor is accepted as long as it appears exactly once. Include 2-3 lines of surrounding "
                    "context ONLY when your anchor matches multiple times."
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
                        "1-indexed line number. When set TOGETHER with scope_end_line, match uniqueness "
                        "is enforced WITHIN this line range only — occurrences OUTSIDE the range are "
                        "ignored entirely. Use this to disambiguate when old_string legitimately repeats "
                        "elsewhere in the file (e.g. similar setter methods, repeated boilerplate) but "
                        "is unique inside the range you actually read. Pair it with the exact line range "
                        "from your most recent read_file/read_symbol. Both scope_start_line AND "
                        "scope_end_line must be provided together; one without the other is rejected."
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
                    "Batch (MultiEdit) mode: a list of edits to apply to the SAME file in one call. "
                    "Each item is an object: {old_string: str, new_string: str, replace_all?: bool}. "
                    "Edits apply sequentially; each later edit sees earlier edits' result. "
                    "ATOMIC: if any edit fails to match, NO edits are written (file untouched). "
                    "Cannot be combined with top-level old_string/new_string. "
                    "Use this to make several substitutions with a single tool call instead of N round-trips."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Text to replace (must be unique unless replace_all)"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)"},
                                "scope_start_line": {"type": "integer", "description": "Optional: 1-indexed start line of a match scope (paired with scope_end_line). See top-level scope_start_line."},
                                "scope_end_line": {"type": "integer", "description": "Optional: 1-indexed inclusive end line of a match scope (paired with scope_start_line)."},
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
            "write large files with bash (heredoc or python3) instead, then use write_plan "
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
            "Apply a unified diff patch to a single file. Uses exact line ranges and context "
            "lines, avoiding ambiguous text matches. "
            "Line numbers must reflect the CURRENT file state — read the target range first; "
            "stale line numbers after a previous edit are the most common failure. "
            "IMPORTANT — dirty target files are REJECTED: if a target file already has uncommitted "
            "working-tree edits made by a text-editing tool THIS SESSION (edit_text / modify_symbol / "
            "edit_ast / anchor_edit), apply_patch returns ok=false. Such edits live in the working tree, "
            "but apply_patch / diff_apply reconstructs hunk context from HEAD and, on a freshly-edited "
            "target, reverts the file to HEAD on conflict — silently losing those session edits. Continue "
            "editing such a file with the SAME text-editing tool instead. "
            "(Patches touching OTHER files in the same call still apply; only the session-edited file is refused.) "
            "Separately, for UNTRACKED files (no git blob) --3way fails with 'repository lacks the necessary blob' "
            "— use the same alternatives. "
            "For replacing a whole function/class use modify_symbol; for a small unique "
            "string substitution use edit_text; to APPEND a block at end-of-file use bash `>>` or write_plan `insert_after`/`insert_after_line`."
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
                    "description": "File path for hunk-only patches (omit for unified diffs with ---/+++ headers). If patch starts with '@@', server auto-wraps into unified diff.",
                },
            },
            "required": ["patch"],
        },
    }

SCHEMA_READ_FILE = {
        "name": "read_file",
        "description": (
            "Read a file by path. 'path' is required — always pass a file path. "
            "Without start_line/end_line: files up to 200 lines return full content; "
            "larger files return only the line count (then call again with a range). "
            "Use start_line and end_line (1-indexed, inclusive) to read specific sections. "
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
                    "description": "Relative path to the file within the repository. REQUIRED ('path' is required)."
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
                    "description": "File or directory to search in (default: repo root)"
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
            "Execute a shell command under bash (NOT zsh/sh). Destructive commands (rm) need approval. "
            "Use for git, grep/rg, cat/head/tail, python3 -c, find, ls, wc, "
            "sed (no -i), and any CLI without a dedicated tool."
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
                    "description": "Timeout in seconds (default: 120, max: 300)",
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
                "design decisions already discussed in chat history."
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
        "NOT for: search (use search_web instead), local files, authenticated/gated content."
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
        "★ Call action='close' when done to free memory."
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
                "description": "Timeout in milliseconds for navigation/click/wait (default: 30000).",
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
    SCHEMA_READ_SYMBOL,
    # SCHEMA_RUN_TESTS — removed: bash("pytest ...") is equivalent and more flexible; kept as internal dispatch only
    SCHEMA_GET_PROJECT_INFO,
    SCHEMA_BASH,
    SCHEMA_JOB,       # ★ Background job management (list/output/kill for long-running bash commands)
    SCHEMA_FIND_SYMBOL,
    SCHEMA_FIND_REFERENCES,
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
