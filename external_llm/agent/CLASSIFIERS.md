# Classifier Architecture (Routing & Intent)

> **TL;DR:** There is **no** single "intent classifier" and the 6–7 modules with
> `intent`/`routing`/`classifier` in the name are **not duplicates**. They occupy
> distinct layers with different inputs, outputs, and decision moments. Read this
> before refactoring or adding a new classifier.

## The dependency chain (call order)

```
 User prompt
     │
     ▼
 ┌─────────────────────────────┐
 │ intent_resolver.py          │  L2 — LLM call (planner model)
 │ IntentResolver.resolve()    │  → IntentResult (target_files, edit_kind,
 └──────────────┬──────────────┘     lane_hint, intent_type, scope, ...)
                │
                ▼
 ┌─────────────────────────────┐
 │ task_router.py              │  L1 — deterministic, NO extra LLM call
 │ TaskRouter.route()          │  ← consumes IntentResult.lane_hint (primary signal)
 │  *THE FINAL ROUTER*         │  → RouteDecision{lane, task_kind, features}
 └──────┬───────────────────────────────────────────────┬──────────────┘
        │ lane = PLANNER                                  │ lane = MAIN_AGENT
        ▼                                                  ▼
 ┌─────────────────────────────┐          ┌─────────────────────────────────┐
 │ (planner lane)              │          │ request_intent_classifier.py    │
 │ routing_policy.py           │  L4      │ L3 — pure transform, NO LLM     │
 │ RoutingPolicy → edit mode   │          │ IntentResult → RoutingIntent    │
 │ (surgical vs replace_symbol)│          │ (read_only / clarify / edit)    │
 └─────────────────────────────┘          └─────────────────────────────────┘

 Separately, in the REST API path only (NOT used by CLI / agent_loop):
 ┌─────────────────────────────┐          ┌─────────────────────────────────┐
 │ execution_mode_classifier   │  L5      │ _user_intent.py                 │
 │ analyze_request_for_optimal │          │ yes/no approval parsing         │
 │ _mode() → ExecuteMode       │          │ (git commit confirm, etc.)      │
 │ (LLM response FORMAT)       │          │ utility, unrelated to routing   │
 └─────────────────────────────┘          └─────────────────────────────────┘

 Cross-cutting dependency:
 ┌─────────────────────────────┐
 │ semantic_intent.py          │  utility — SemanticIntentMatcher
 │ cosine vector matcher       │  used BY execution_mode_classifier and
 └─────────────────────────────┘  operation_models; not a classifier itself
```

## Layer-by-layer

| Layer | Module (LOC) | Input | Output | LLM? | Who calls it |
|-------|--------------|-------|--------|------|--------------|
| **L1** | `task_router.py` (882) | prompt + `IntentResult` | `RouteDecision{lane, task_kind}` | No (uses IntentResult's LLM result) | `routes/agent_stream.py:441`, `asi.py:3360` |
| **L2** | `intent_resolver.py` (848) | prompt | `IntentResult` | **Yes** (planner model) | `task_router.py` (via lane_hint), planner lane |
| **L3** | `request_intent_classifier.py` (85) | `IntentResult` | `RoutingIntent` Literal | No (deterministic transform) | 7 planner modules (`routing_intent_from_intent_result`, `normalize_routing_label`) |
| **L4** | `routing_policy.py` (157) | action_hint + sym_lines + nesting | edit mode string | No (learned JSON policy) | `planner_helpers_contract.py:2118` only |
| **L5** | `execution_mode_classifier.py` (317) | prompt + target_file | `ExecuteMode` string | Optional (5s timeout) | `routes/edit_run.py:530` only (REST API) |
| util  | `_user_intent.py` (125) | user yes/no answer | `UserApproval` | No | approval flows |
| util  | `semantic_intent.py` (163) | query + examples | label | No (cosine) | imported BY L5 + operation_models |

## Who is the FINAL routing authority?

**`TaskRouter.route()` (`task_router.py`).**

It is instantiated at `routes/agent_stream.py:421` and invoked at
`routes/agent_stream.py:441` (`router.route(_routing_text, repo_root=repo_root)`),
returning a `RouteDecision` whose `.lane` (`PLANNER` | `MAIN_AGENT`) is the value
that downstream code branches on.

TaskRouter uses `IntentResult.lane_hint` (from IntentResolver's LLM call) as its
**primary** signal (`task_router.py:567-569`), blended with deterministic
`RouteFeatures` (file extensions, keyword counts). So the chain is sequential,
not competing: IntentResolver (LLM) → feeds → TaskRouter (deterministic blend) → lane.

## Common confusions (and why they are wrong)

1. **"intent_resolver and task_router both classify intent → merge them."**
   No. IntentResolver runs the LLM and emits a structured `IntentResult`;
   TaskRouter consumes `IntentResult.lane_hint` and adds deterministic features.
   Merging would couple an LLM call to a deterministic gate.

2. **"request_intent_classifier is a third classifier."**
   No. It is a **pure transform** (`IntentResult` → `RoutingIntent` Literal),
   documented in its own docstring as "Stage 0 already ran before planning."
   It contains 3 functions, no LLM, no state.

3. **"routing_policy.py routes the lane."**
   Misleading name. It selects the **edit mode** (surgical vs `replace_symbol_body`)
   based on symbol size/nesting — a planner-lane-internal concern, called from
   exactly one place. It has nothing to do with PLANNER-vs-MAIN_AGENT routing.

4. **"execution_mode_classifier is the router."**
   No. `ExecuteMode` (strict_json / intelligent / plan_json / normal / legacy)
   selects the **LLM response format** for the REST `/edit/run` endpoint only.
   It is not imported by `asi.py` or `agent_loop`. (It was formerly the root
   `intent_classifier.py`; moved here in 2026-06 to remove the legacy placement.)

## When to add a new classifier

You almost certainly **shouldn't**. Check first:
- If your decision can be derived deterministically from an existing
  `IntentResult` field → put it in `request_intent_classifier.py`.
- If it is about edit granularity (surgical/replace) → extend
  `routing_policy.py`'s learned policy JSON.
- If it needs an LLM to understand prompt semantics → extend
  `intent_resolver.py`'s schema.
- Only if it is a genuinely orthogonal axis (e.g. a new output format for a
  new transport) add a new module, and update the table above.
