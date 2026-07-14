# Command Line Interface

> **Deprecation Notice**: This document describes the legacy CLI (`external_llm/agent/cli.py`). The current production CLI is `asi.py` at the project root.

## Current CLI: `asi.py`

Interactive CLI providing Design Chat → PLANNER agent pipeline.

### Usage

```bash
python asi.py                         # Start REPL mode
python asi.py --repo /path/to/repo    # Specific repository
python asi.py -p "fix the bug"         # Run single request then exit
python asi.py --provider anthropic --model claude-sonnet-4-6
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--repo, -r PATH` | Repository root path | Current directory |
| `--prompt, -p TEXT` | Single request text (omit for REPL) | — |
| `--prompt-file FILE` | Read request from file | — |
| `--provider NAME` | LLM provider | `EXTERNAL_LLM_PROVIDER` env |
| `--model, -m NAME` | LLM model | `EXTERNAL_LLM_MODEL` env |
| `--api-key KEY` | API key | env vars take priority |
| `--max-turns N` | Max agent turns | Config default (30) |
| `--verbose, -v` | Verbose logging | Off |
| `--log-level LEVEL` | DEBUG / INFO / WARNING / ERROR / NONE | INFO |
| `--log-file PATH` | Log file (`{date}`, `{time}` supported) | `logs/run_{date}_{time}.log` |

### Environment Variables

- `EXTERNAL_LLM_PROVIDER` — `anthropic` / `openai` / `google` / `deepseek` / `ollama`
- `EXTERNAL_LLM_MODEL` — model name
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `GOOGLE_API_KEY`

### Modes

**REPL mode** (no `--prompt`): Interactive session with prompt_toolkit. All requests go through Design Chat for analysis first. When Design Chat decides to act, it switches to the PLANNER lane. Features: ESC to pause/resume, persistent token tracking, session context accumulation.

**Single-shot mode** (`--prompt` or `--prompt-file`): Run one request and exit. Same pipeline (Design Chat → PLANNER → execution).

### Programmatic Usage

For programmatic access to the agent pipeline, use `_build_engine()` from `asi.py`:

```python
from asi import _build_engine
from external_llm.agent.agent_loop import AgentLoop

# See _build_engine() in asi.py for the full reference pattern
```

## Legacy CLI: `external_llm/agent/cli.py`

The old CLI at `python -m external_llm.agent` is **deprecated**. It uses the old `Agent` class directly without Design Chat or PLANNER lane support.

```bash
# Legacy usage (deprecated):
python -m external_llm.agent --prompt 'Hello'
python -m external_llm.agent --config config.json --prompt 'Process this'
```
