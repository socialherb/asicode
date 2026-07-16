# asicode

**Autonomous Software Improvement** — local safe patch runner and code editing tool: an AI-powered assistant for reading, analyzing, and modifying codebases with deterministic AST-level operations, transparent shell execution, and multi-language support.

## Features

- **Context economy**: recent turns stay verbatim while older turns are compressed in the background, and superseded tool outputs are dropped from the window — long sessions stay focused and cheap instead of accumulating until a context cliff
- **Autonomous long-run loop (`/auto`)**: after each turn the model drafts the natural next step as ghost text; auto mode countdown-runs it, chaining turns without a human prompt — a required-follow-up-only contract, a consecutive-step cap, and announced stop points keep the loop from wandering, and typing or Esc hands control back instantly
- **Parallel sub-agents, mixed models**: orchestration (`--orchestrate`) dispatches sub-tasks to worker processes — each opens its own terminal window on macOS — and every worker slot can run a different provider/model (`/model dev_1 …`)
- **Multi-terminal, one repo**: run several `asi` sessions on the same repository at once — cross-process file locks, per-turn ownership markers, and stale-worker reaping keep agents from duplicating or clobbering each other's work
- **Claude Code collaboration**: pair with the Claude Agent SDK for division of labor — Claude analyzes the codebase through asicode's MCP tools (read-only in analysis mode), asicode executes the edits, Claude optionally reviews the result (`pip install 'asicode[collaborate]'`)
- **Multi-language code editing**: Python, TypeScript, JavaScript, Go, Java, Kotlin, Rust, and more via tree-sitter AST parsing
- **AST-precise modifications**: Edit symbols by name, insert/delete lines by anchor, apply typed AST operations (Python)
- **Vector search & RAG**: Semantic code search with FAISS + sentence-transformers embedding
- **Structural analysis**: Dead code detection, duplicate finding, unused import scanning, contradictory logic detection
- **Interactive CLI**: Rich terminal interface with prompt-toolkit, command history, completion, and multi-line editing
- **Headless & automation modes**: single-shot runs (`asi -p`), JSON/NDJSON output
- **Self-searchable history**: sessions persist to disk and survive `/clear` and CLI restarts — the model can search its own past conversations (`search_design_history`, cross-session) to recall decisions and file paths instead of re-discovering them
- **Design chat**: Persistent conversation session with insight management
- **Shell shim layer**: macOS/Linux compatibility layer for BSD-style command-line tools

## Quick Start

```bash
# Recommended for CLI use: pipx installs into an isolated env and puts
# `asi` on your PATH, no venv activation needed
pipx install 'asicode @ git+https://github.com/socialherb/asicode.git'

# Or plain pip (inside a venv of your choice):
pip install git+https://github.com/socialherb/asicode.git

# Or with all features:
pip install 'asicode[all] @ git+https://github.com/socialherb/asicode.git'

# Start the interactive CLI
asi
```

## Installation Options

```bash
# Core (includes tree-sitter AST parsing — one package covers 300+ languages)
pip install asicode

# With RAG (vector search for code)
pip install 'asicode[rag]'

# With browser automation
pip install 'asicode[browser]'

# Development tools
pip install 'asicode[dev,lint]'

# Everything
pip install 'asicode[all]'
```

## Architecture

```
asicode/
├── asi.py              # Interactive CLI (REPL)
├── external_llm/           # Core engine
│   ├── agent/              # Agent loop, tool handlers, verification
│   ├── languages/          # Multi-language providers (Python, TS, Go, etc.)
│   ├── editor/             # Code editing (AST, anchor, text, patch)
│   └── repl/               # Design chat session management
├── scripts/                # Lint/reachability guards (CI)
└── tests/                  # Test suite (unit + integration)
```

## Key Concepts

### Agent Tool Loop
Every request runs through a single LLM tool-use loop: the model reads,
searches, and edits the repository through typed tools, and each write is
followed by verification. Headless mode (`asi -p`) and orchestration mode
(`--orchestrate`, which decomposes a request and dispatches it to parallel
sub-agent workers) drive the same loop.

### Auto-Continue: Long-Running Agent Loops (`/auto`)
After every turn asicode already suggests the natural next task as dim ghost
text on the prompt (accept with `→`). `/auto` turns that suggestion into an
unattended loop: the next step is auto-submitted after a short countdown, so a
single instruction ("find and fix a bug, then keep going") can chain many
turns of work — each one starting with a fresh tool loop, so long runs don't
degrade the context window.

Autonomy is bounded by design, because an unattended loop must know when to
stop:

- **Opt-in only** — `/auto [N|on|off]` is the sole trigger; intent is never
  inferred from prompt text. The prompt status line shows `auto n/N` while armed.
- **Required-follow-up contract** — the next step fires only when the previous
  turn left mandatory work (unverified changes, unfinished steps); "nice to
  have" ideas end the loop instead of extending it (`NONE` is the default).
- **Announced stop points** — natural completion, the consecutive-step cap
  (default 5), or an error turn all stop the loop with a visible notice rather
  than going silent.
- **Instant takeover** — typing cancels the pending step and resets the chain;
  `Esc` skips one step but keeps the mode; `Enter` on the empty prompt runs
  the step immediately. Auto-driven turns are tagged in the session record, so
  you can audit afterwards exactly how far the loop went on its own.

### Context Economy
Most agent CLIs let every turn and tool result pile up until the context
window forces a lossy compaction. asicode manages the window continuously:

- The most recent turns are always kept verbatim; older turns are summarized
  by a background pass that never blocks the conversation.
- Tool outputs feed the turn that requested them; once superseded, stale
  results are dropped from the window (originals persist on disk).
- Durable facts are promoted to a session insight store, so they survive
  compression instead of living in the transcript.
- Compressed and `/clear`-ed turns aren't gone — they're archived to disk, and
  the model can search back through them (and past sessions) with
  `search_design_history` whenever it needs a decision or file path that
  fell out of the active window.

The result: the model sees a focused window instead of a scrolling log —
better attention on the task at hand, and materially fewer tokens per turn.

### Parallel & Concurrent by Design
`--orchestrate` decomposes a request into sub-tasks and runs them in parallel
worker processes over a file-based IPC protocol with heartbeats (a hung worker
is distinguishable from a busy one). Worker slots are independently
configurable, so a cheap fast model can handle mechanical edits while a
stronger model plans.

Concurrency is also safe across *your own* terminals: session state is
guarded by cross-process locks, each in-flight turn carries an owner marker so
other sessions see "being handled elsewhere" instead of re-doing the work, and
turns from crashed processes are detected and reaped.

### Division of Labor with Claude Code
`asi collaborate --task "…"` runs a four-phase pipeline with the Claude Agent
SDK, pairing each agent with what it does best:

1. **Preprocess** — asicode's cheap engine generates a codebase digest
   (structure, relevant files) so the expensive model never burns tokens on
   raw discovery.
2. **Analyze** — the Claude agent receives the digest and explores through
   asicode's in-process MCP tools; its own file tools are disabled and, in
   analysis mode, destructive tools are excluded — a read-only investigation
   by construction.
3. **Execute** — asicode applies the planned edits through its verified
   editing pipeline.
4. **Review** — optionally, the Claude agent reviews the execution and
   returns a verdict.

Requires the optional extra: `pip install 'asicode[collaborate]'`.

### Deterministic Editing
All code modifications are validated through multiple layers:
- Syntax validation (AST parse gate)
- Structural analysis (dependency graph, import consistency)
- Verification loop (edit → verify → repair if needed)

## Requirements

### Python
- Python 3.10+
- macOS or Linux (BSD shim layer for macOS)

### Bundled with the install
- **`ruff`** — the linter used for post-edit verification (F821 undefined-name checks). It is a
  core dependency, so `pip install asicode` installs it automatically. No separate step needed.
- **tree-sitter** — powers AST-based symbol/call/import detection for precise multi-language
  editing. The core install ships `tree-sitter-language-pack` (~2 MB) — a single package with
  full prebuilt-wheel platform coverage that covers every supported language out of the box
  (it also pulls in the `tree-sitter` core library automatically).

### Recommended system tools
asicode degrades gracefully when these are missing, but installing them improves results:

| Tool | Used for | Required when | Behavior if absent |
|------|----------|---------------|--------------------|
| `git` | Version control, diff/apply, change-impact analysis | Almost always | Core features expect git |
| `ripgrep` (`rg`) | Fast code search (`grep` tool) | Recommended | Falls back to system `grep` |
| `node` | TypeScript/JavaScript editing & validation | Only when editing JS/TS | JS/TS features disabled |
| `gofmt` / `golangci-lint` | Go formatting & linting | Only when editing Go | Go linting disabled |
| `docker` | Sandboxed web search (SearXNG) | Only for isolated web search | Web search disabled |

Install `git` and `ripgrep` on macOS:
```bash
brew install git ripgrep
```

`ripgrep` can also be installed via the optional `search` extra (`pip install asicode` is unaffected if it fails — the `grep` fallback applies):
```bash
pip install 'asicode[search]'
```
> Prebuilt wheels exist for macOS-arm64 and linux-x86_64; other platforms build from source.

## Development

```bash
# Clone and install in editable mode
git clone <repo-url>
cd asicode
pip install -e '.[dev,lint,rag]'

# Run tests
pytest

# Run linting
ruff check
ruff format --check

# Run type checking
pyright
```
