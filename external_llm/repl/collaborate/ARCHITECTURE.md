# asicode ↔ Claude Code Agent Collaborative Architecture

## Overview

asicode and Claude Code Agent collaborate as equal partners, each leveraging their strengths:

| Agent | Strength | Role |
|-------|----------|------|
| asicode (Haiku/cache) | Cheap file I/O, BM25 search, call graph, structural scanners | Preprocessing, digest generation, execution |
| Claude Code Agent (Opus) | Deep reasoning, planning, code review, design judgment | Task decomposition, analysis, verification |

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │  CollaborationOrchestrator           │
                        │  ┌────────────┐  ┌───────────────┐  │
                        │  │asicode    │  │ClaudeSession  │  │
                        │  │Engine      │  │(SDKClient)    │  │
                        │  └─────┬──────┘  └───────┬───────┘  │
                        └────────┼──────────────────┼──────────┘
                                 │                  │
                    ┌────────────┴──────────────────┴────────────┐
                    │  In-Process MCP (create_sdk_mcp_server)     │
                    │  asicode Tools: read_file, grep, bash,     │
                    │  apply_patch, modify_symbol, ... (38 tools) │
                    └────────────┬──────────────────┬────────────┘
                                 │                  │
                    ┌────────────┴──────────────────┴────────────┐
                    │  Claude Code Agent (Opus)                   │
                    │  allowed_tools=["mcp__asr__*"]              │
                    │  disallowed=["Read","Bash","Grep","Glob"]   │
                    └─────────────────────────────────────────────┘
```

## Key Decisions

### 1. In-Process MCP (no subprocess)

asicode ToolRegistry handlers wrapped via SDK `@tool` decorator + `create_sdk_mcp_server()`.
No subprocess management, no IPC overhead, no serialization bugs.

### 2. Tool Restriction

`allowed_tools=["mcp__asr__*"]` + `disallowed_tools=["Read","Write","Bash","Grep","Glob","Edit","WebFetch","WebSearch"]`.
Forces Claude Code Agent to use asicode's tools exclusively.

### 3. Hook-Based Observability

- `PreToolUse` — log every tool call
- `PostToolUse` — capture results
- `Stop` — save session summary

### 4. Structured Verdict

Claude returns typed verdict via `output_format` JSON schema.

## Collaboration Flow

### Phase 1: asicode Preprocessing
1. User gives task to asicode
2. asicode generates digest (relevant files, call graph, search results)
3. asicode creates a ClaudeSession with MCP tools

### Phase 2: Claude Code Agent Analysis
4. Claude Code Agent receives digest + task
5. Claude uses asicode tools for additional info
6. Claude returns verdict + plan

### Phase 3: asicode Execution (optional)
7. asicode executes the plan
8. asicode asks Claude to review result
9. Loop until satisfied or max iterations reached

## Integration Points

- Design Chat: read design_insights.md via asicode tools
- Plan Compiler: Claude proposes plans via structured output
- Save Insight: Claude saves design decisions
- Checkpoints: asicode handles git operations
