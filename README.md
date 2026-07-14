# asicode

**Autonomous Software Improvement** — local safe patch runner and code editing tool: an AI-powered assistant for reading, analyzing, and modifying codebases with deterministic AST-level operations, transparent shell execution, and multi-language support.

## Features

- **Multi-language code editing**: Python, TypeScript, JavaScript, Go, Java, Kotlin, Rust, and more via tree-sitter AST parsing
- **AST-precise modifications**: Edit symbols by name, insert/delete lines by anchor, apply typed AST operations (Python)
- **Vector search & RAG**: Semantic code search with FAISS + sentence-transformers embedding
- **Structural analysis**: Dead code detection, duplicate finding, unused import scanning, contradictory logic detection
- **Interactive CLI**: Rich terminal interface with prompt-toolkit, command history, completion, and multi-line editing
- **Headless & automation modes**: single-shot runs (`asi -p`), JSON/NDJSON output, and multi-agent orchestration (`--orchestrate`)
- **Design chat**: Persistent conversation session with auto-compression and insight management
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
