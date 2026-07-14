"""
IntentResolver: Universal intent understanding for all user requests.

Key features:
1. Uses planner model for natural language understanding
2. Language-neutral: handles any language, typos, mixed input
3. No keyword mapping: LLM extracts appropriate search terms
4. Single LLM call per request, cached for reuse
5. Provides intent classification, target inference, and lane suggestions
"""

import ast
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from .enums import Complexity, Scope
from .guard_ir import parse_guard as _parse_guard_ir
from .intent_models import IntentResolutionConfig, IntentResult

logger = logging.getLogger(__name__)


class IntentResolver:
    """Resolves user intent using LLM (planner model)."""

    def __init__(self, config: IntentResolutionConfig):
        self.config = config
        # Bounded LRU + TTL, thread-safe (matches ToolResultCache pattern).
        self._cache: "OrderedDict[str, tuple[IntentResult, float]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max = 128  # bound (LRU eviction)
        self._llm_client = config.llm_client
        self._model = config.model

        if not self._llm_client:
            logger.warning("IntentResolver: no LLM client provided, will only do basic extraction")

    def resolve(self, request: str) -> IntentResult:
        """Resolve intent for a user request.

        Returns cached result if available, otherwise calls LLM.
        """
        if not request or not request.strip():
            return self._create_empty_result("")

        request = request.strip()

        # Check cache (thread-safe bounded LRU; matches ToolResultCache pattern)
        cache_key = self.config.get_cache_key(request)
        if self.config.enable_cache:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    result, timestamp = cached
                    if time.monotonic() - timestamp < self.config.cache_ttl_seconds:
                        self._cache.move_to_end(cache_key)  # refresh LRU position
                        logger.debug("IntentResolver cache hit: %s", cache_key[:8])
                        return result
                    else:
                        del self._cache[cache_key]

        # Resolve with LLM
        result = self._resolve_with_llm(request)

        # Cache result (bounded LRU eviction)
        if self.config.enable_cache:
            with self._cache_lock:
                self._cache[cache_key] = (result, time.monotonic())
                if len(self._cache) >= self._cache_max:
                    self._cache.popitem(last=False)
            logger.debug("IntentResolver cached: %s", cache_key[:8])

        return result

    def _resolve_with_llm(self, request: str) -> IntentResult:
        """Call LLM to understand intent, extract search terms, and infer targets."""
        if not self._llm_client:
            logger.warning("IntentResolver: no LLM client, falling back to basic extraction")
            return self._fallback_extraction(request)

        try:
            from ..client import LLMMessage, effective_content

            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(request)

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            logger.info("IntentResolver LLM call: model=%s, request=%r", self._model, request[:100])

            def _call_llm(max_tokens: int):
                if hasattr(self._llm_client, "chat"):
                    return self._llm_client.chat(
                        messages=messages,
                        model=self._model,
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                elif hasattr(self._llm_client, "chat_with_tools"):
                    return self._llm_client.chat_with_tools(
                        messages=messages,
                        tools=[],
                        model=self._model,
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                return None

            # Initial budget from config (default 4096). If the LLM truncates,
            # retry with doubled budget.
            _initial_budget = self.config.max_tokens
            try:
                response = _call_llm(_initial_budget)
            except Exception as exc:
                from ..client import LLMServerUnavailableError
                if isinstance(exc, LLMServerUnavailableError):
                    raise
                logger.warning("IntentResolver LLM call failed: %s", exc)
                return self._fallback_extraction(request)

            if response is None:
                logger.error("IntentResolver: LLM client has no chat method")
                return self._fallback_extraction(request)

            # Extract response content first so we can detect truncation via content too
            raw = ""
            if hasattr(response, "content"):
                # reasoning_content fallback: GLM-5.2 (thinking ON) / DeepSeek
                # Reasoner may emit the intent JSON in reasoning_content with
                # empty content, collapsing intent parsing to the heuristic fallback.
                raw = effective_content(response)
            elif isinstance(response, str):
                raw = response

            # Detect truncation: check finish_reason attribute, raw_response fallback,
            # and structural heuristic (JSON not closed).
            _finish_reason = getattr(response, "finish_reason", None)
            if _finish_reason is None:
                # Some client wrappers don't expose finish_reason directly — check raw_response
                _raw_resp = getattr(response, "raw_response", None) or {}
                if isinstance(_raw_resp, dict):
                    _choices = _raw_resp.get("choices", [])
                    if _choices and isinstance(_choices[0], dict):
                        _finish_reason = _choices[0].get("finish_reason")
            # Structural heuristic: valid JSON always ends with } or ] (after stripping whitespace)
            _json_looks_truncated = bool(raw) and not raw.rstrip().endswith(('}', ']'))

            logger.debug(
                "IntentResolver: finish_reason=%r json_truncated=%s raw_len=%d",
                _finish_reason, _json_looks_truncated, len(raw),
            )

            if _finish_reason == "length" or _json_looks_truncated:
                logger.warning(
                    "IntentResolver: truncated response detected "
                    "(finish_reason=%r, json_truncated=%s) at max_tokens=%d — retrying with %d",
                    _finish_reason, _json_looks_truncated, _initial_budget, _initial_budget * 2,
                )
                try:
                    response = _call_llm(_initial_budget * 2)
                    if hasattr(response, "content"):
                        raw = effective_content(response) or raw  # prefer retry content
                    elif isinstance(response, str):
                        raw = response or raw
                except Exception as exc:
                    from ..client import LLMServerUnavailableError
                    if isinstance(exc, LLMServerUnavailableError):
                        raise
                    logger.warning("IntentResolver retry failed: %s — using partial response", exc)
                    # Fall through with original partial content

            # Parse JSON from response (with truncation recovery)
            result_dict = self._parse_llm_response(raw, request)
            return self._build_intent_result(request, result_dict)

        except Exception as exc:
            from ..client import LLMServerUnavailableError
            if isinstance(exc, LLMServerUnavailableError):
                raise
            logger.error("IntentResolver LLM resolution failed: %s", exc, exc_info=True)
            return self._fallback_extraction(request)

    def _build_system_prompt(self) -> str:
        """Build system prompt for intent understanding."""
        return """You are an intent understanding system for a code editing assistant.

Your task: Analyze the user's request and extract key information for code search and task routing.

Guidelines:
1. Be language-neutral: Handle any language (English, Korean, Japanese, etc.) and mixed input
2. Correct typos automatically: "functoin" → "function", "authenication" → "authentication"
3. Extract SEARCH TERMS: Technical terms for searching code (function names, class names, domain concepts)
4. Infer INTENT TYPE: What kind of task is this?
5. Suggest EXECUTION LANE: Which part of the system should handle this?
6. Identify POTENTIAL TARGETS: Which files/symbols might need changes?

OUTPUT FORMAT (JSON only):
Note: Output fields in this order. Structural fields come first so that
even if the response is truncated, the most critical routing information is preserved.
{
  "intent_type": "bugfix|feature|refactor|exploration|question|modify|extend|create",
  "lane_hint": "planner|main_agent|read_only|clarify",
  "scope_hint": "single_file|multi_file|project_wide",
  "complexity_hint": "trivial|normal|complex",
  "is_test_write": true|false,
  "is_style_fix": true|false,
  "is_filesystem_op": true|false,
  "is_ui_change": true|false,
  "is_interface_preserving": true|false,
  "modify_symbols": ["ExistingClass"],
  "new_symbols": [
    {"name": "new_method_name", "kind": "method", "parent": "ExistingClass"},
    {"name": "new_function_name", "kind": "function", "parent": null}
  ],
  "reference_symbols": ["existing_function", "SomeType"],
  "search_terms": ["list", "of", "technical", "search", "terms"],
  "edit_kind": "guard_add|body_only|signature_change|full_rewrite|extend|",
  "guard_statement": "if not x: return None",
  "target_loop_iterable": "iterable_expr_or_empty_string",
  "new_files": ["path/to/new_file.py"],
  "confidence": 0.0-1.0,
  "metadata": {
    "language_detected": "en|ko|mixed|unknown",
    "typo_corrections": {"original": "corrected", ...}
  },
  "normalized_query": "Cleaned English version of the request (typo corrected, ≤80 chars)",
  "code_concepts": {
    "data_fields": ["field_or_attr_name"],
    "behavioral_kind": "enforcement|creation|fix|query",
    "scope_phase": "planning|execution|verification|exploration"
  }
}

SCOPE: single_file (default) || multi_file (≥2 files) || project_wide (entire codebase).
COMPLEXITY: trivial (1-line, rename, import) || normal (function/method edit) || complex (multi-file, new module).

is_test_write: true ONLY when user explicitly says write/create/generate tests.
is_style_fix: true ONLY for pure formatting/linting (no logic change).
is_filesystem_op: true ONLY for file/directory move/rename/delete (not content edits).
is_ui_change: true ONLY for visual/style properties (CSS, colors, fonts, layout, icons — not template logic/JS behavior).
is_interface_preserving: true ONLY when user explicitly says preserve API/signature/compatibility. Default false.

new_files: Only populate when the user explicitly names a new file path. Leave [] otherwise. Never guess paths.

Note: Avoid outputting "target_files". File discovery is handled by the grounding engine. Guessing paths causes hallucination.

SYMBOL ROLE GUIDE:
- "modify_symbols": existing symbols whose CODE BODY MUST CHANGE — the target of the request
- "reference_symbols": existing symbols referenced/called/used by new code but NOT being changed
- "new_symbols": symbols that DO NOT EXIST YET and need to be CREATED

Rules:
  • A symbol whose body is the target of the change goes in modify_symbols.
  • A symbol only referenced, imported, or passed as a type goes in reference_symbols.
  • Adding a new member inside a symbol → that symbol goes in modify_symbols, the new member in new_symbols.
  • A standalone new function that takes an existing type as parameter → the function in new_symbols, the type in reference_symbols.
  • Symbols explicitly excluded from modification → reference_symbols only.
  • Preserve "ClassName.method_name" qualified form in modify_symbols.

Examples:
  "Fix bug in ConnectionPool.release" →
    modify_symbols: ["ConnectionPool.release"], reference_symbols: [], new_symbols: []
  "Add validate() method to UserModel" →
    modify_symbols: ["UserModel"], new_symbols: [{"name": "validate", "kind": "method", "parent": "UserModel"}], reference_symbols: []

  "Add parse_response(response), reuse existing json_decoder internally" →
    modify_symbols: [], new_symbols: [{"name": "parse_response", "kind": "function", "parent": null}], reference_symbols: ["json_decoder"]

  "Create tests/test_projects/typescript/simple_math.ts" →
    modify_symbols: []  (pure file creation — no existing code body is modified)
    reference_symbols: []
    new_symbols: []

  "Create a utility function parse_input()" →
    modify_symbols: []  (standalone new function — no existing code modified)
    new_symbols: [{"name": "parse_input", "kind": "function", "parent": null}]
    reference_symbols: []

CRITICAL: modify_symbols must contain ONLY actual code identifiers that exist
(or are strongly implied to exist) in the codebase. Natural-language keywords
like "Create", "generate", "TypeScript", "utility", "file", "tests"
are NEVER valid modify_symbols — they are not code identifiers.
If no existing code body is being modified, modify_symbols MUST be empty.

  "Add calculate_score() that uses ScoreBreakdown type" →
    modify_symbols: [], new_symbols: [{"name": "calculate_score", "kind": "function", "parent": null}], reference_symbols: ["ScoreBreakdown"]

Note: A symbol in reference_symbols should never be modified — avoid including it in modify_symbols.

INTENT TYPE GUIDE:
- bugfix: Fixing errors, bugs, incorrect behavior
- feature: Adding new functionality that doesn't exist
- refactor: Restructuring code without changing behavior
- exploration: User asks purely to explore/read/understand — no change intended.
               Use only when the request has no code change implied. Avoid when change intent is present, even if vague.
- question: Asking how something works or why (read-only, no change intended)
- modify: Changing existing behavior or logic
- extend: Adding to existing feature (new param, field, option)
- create: Creating new files/modules from scratch

LANE HINT GUIDE:
- planner: Code logic changes (Python/JS/TS files, symbol-based edits)
- main_agent: UI styling, non-code files (CSS/HTML/JSON), filesystem operations
- read_only: User ONLY wants to read/understand code — no change intended.
             Use only when the request is purely about reading or understanding.
             Avoid when any change intent is present, even if vague or implicit.
- clarify: Change/edit INTENT is present but too vague or ambiguous to act on.
           Use when the user wants something done but has not specified what or where.
           KEY RULE: If a request could reasonably require writing code, use clarify — not read_only.

CODE CONCEPTS (CRITICAL for finding the RIGHT symbol — not the named entry point):
Extract the actual data structures / field names that need to change, and classify
the behavioral role so SpecResolver can find the enforcement point, not just the
most prominent symbol that matches the request's surface words.

- data_fields: Python field/attribute names that will be READ or WRITTEN by the fix.
  These are actual identifiers as they appear in source code — NOT the request's words.
  Rules:
    • Only include names that appear verbatim in source code (not domain words)
    • Prefer attribute names (created_at, error_count) over class names
    • Leave [] when no specific data field can be identified from the request

- behavioral_kind: The role of the change — determines WHERE to enforce it:
  "enforcement" — ensuring/guaranteeing a property holds (ensure, prevent, guard)
                  Target: the convergence point that ALL execution paths pass through
  "creation"    — adding new functionality that doesn't exist (add, create, introduce)
                  Target: the entry point or insertion location
  "fix"         — correcting a bug in a specific known location (bug, error, incorrect)
                  Target: the specific broken function
  "query"       — reading/understanding only (explain, how, describe)
                  Target: most prominent/central symbol

- scope_phase: Where in the execution pipeline the change should happen:
  "planning"     — during plan/operation creation
  "execution"    — during actual operation execution / scheduling
  "verification" — during validation / test / lint
  "exploration"  — during exploration/discovery/grounding

SEARCH TERMS: Extract CODE IDENTIFIER names — actual function, class, or variable names that are
likely to appear verbatim in the codebase and relate to the change.

Rules:
  • Prefer identifiers that name code constructs implementing the described behavior.
  • For bugfix intents: extract names of functions/classes that IMPLEMENT the described behavior,
    NOT words that describe the symptom. "bug", "error", "wrong", "skip", "fail", "incorrect"
    are symptom words — they do not appear in source code and cause grounding to drift to
    unrelated files.
  • For feature/modify intents: extract identifiers that name the affected functionality.
  • Leave generic nouns (e.g. "function", "method", "code") out — they match too broadly.

EDIT KIND GUIDE (classify the nature of the code change — this drives execution strategy):
- "guard_add"        — adding a conditional check + early exit (return/continue/break/raise/skip)
                       anywhere in the function body: at entry, inside a loop, inside a branch.
                       When selected, extract the exact guard statement into "guard_statement".
- "body_only"        — internal logic change, function signature stays the same.
- "signature_change" — changing parameters (add/remove/rename) or return type.
- "full_rewrite"     — replacing the entire function/class body.
- "extend"           — adding new code (new method, new field, new import).
- ""                 — unknown or does not fit the above categories

guard_statement: Only populate when edit_kind == "guard_add".
  Infer the guard structure (condition shape + control flow) from the request.
  VARIABLE NAME RULES:
  1. If the request explicitly names a variable or constant, use that exact name.
  2. If the request describes a condition abstractly, write the structural guard using a placeholder
     like "condition". Do not fabricate specific variable names not mentioned in the request.
  3. The guard_statement is a structural HINT (control flow + scope), not an exact code spec.
     A placeholder is always better than a wrong name.
  Leave "" only if no guard condition can be inferred at all.

target_loop_iterable: Only populate when edit_kind == "guard_add" AND the guard must go
  inside a specific for-loop (not at function entry).
  Extract the iterable expression of the target loop AS IT LIKELY APPEARS IN SOURCE CODE.
  This is the expression after "for VAR in <HERE>:" — extract from the request description
  of which collection/list is being iterated. Use the variable name as described in the request;
  the verifier maps it to the real source variable.
  Leave "" when the guard is at function entry (not inside a loop), or the loop cannot be
  identified from the request.

Be concise and accurate. Return JSON only."""

    def _build_user_prompt(self, request: str) -> str:
        """Build user prompt with the request."""
        return f"User request: {request}"

    def _recover_truncated_json(self, raw_json: str) -> Optional[dict[str, Any]]:
        """Extract complete key-value pairs from a truncated JSON string.

        When finish_reason=length cuts the response mid-string, we lose the tail.
        Strategy: find positions of top-level field separators (commas at depth=1),
        then try closing the object at each separator from the last one backward.
        O(n) scan + O(k) parse attempts where k = number of top-level fields (≤15).
        """
        text = raw_json.strip()
        if not text.startswith("{"):
            return None

        # Single O(n) pass: track nesting depth and record top-level comma positions.
        # Top-level = inside root object (brace_depth==1) but not inside nested array/object.
        top_level_comma_positions: list = []
        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escape_next = False

        for idx, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif ch == "," and brace_depth == 1 and bracket_depth == 0:
                top_level_comma_positions.append(idx)

        # Try closing at each top-level comma from last to first (most fields → fewest).
        for comma_pos in reversed(top_level_comma_positions):
            candidate = text[:comma_pos].rstrip() + "}"
            try:
                recovered = json.loads(candidate)
                if isinstance(recovered, dict) and recovered:
                    return recovered
            except json.JSONDecodeError:
                continue

        return None

    def _parse_llm_response(self, raw_response: str, original_request: str) -> dict[str, Any]:
        """Parse LLM response JSON and validate."""
        raw = raw_response.strip()

        # Extract JSON block (brace matching, no regex)
        brace_start = raw.find('{')
        if brace_start == -1:
            logger.warning("IntentResolver: no JSON found in LLM response (%d chars): %s", len(raw), raw[:2000])
            return self._create_fallback_dict(original_request)
        brace_end = raw.rfind('}')
        if brace_end == -1 or brace_end <= brace_start:
            logger.warning("IntentResolver: no JSON found in LLM response (%d chars): %s", len(raw), raw[:2000])
            return self._create_fallback_dict(original_request)
        raw_json = raw[brace_start:brace_end + 1]

        try:
            result = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.warning("IntentResolver: JSON parse error: %s, raw (%d chars): %s", e, len(raw), raw[:2000])
            # Attempt partial recovery: extract complete fields before truncation point.
            # JSON truncation typically cuts in the middle of a string value or array.
            # We can recover fields that finished serializing before the cut.
            recovered = self._recover_truncated_json(raw_json)
            if recovered:
                logger.info(
                    "IntentResolver: partial JSON recovery succeeded — fields=%s",
                    sorted(recovered.keys()),
                )
                result = recovered
            else:
                return self._create_fallback_dict(original_request)

        # Validate required fields
        if not isinstance(result, dict):
            return self._create_fallback_dict(original_request)

        # Ensure normalized_query exists
        if "normalized_query" not in result:
            result["normalized_query"] = original_request

        # Ensure lists are lists
        for list_field in ["search_terms", "target_files", "target_symbols",
                           "modify_symbols", "reference_symbols", "new_symbols"]:
            if list_field in result and not isinstance(result[list_field], list):
                result[list_field] = []

        # Ensure confidence is reasonable
        if "confidence" in result:
            try:
                conf = float(result["confidence"])
                result["confidence"] = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                result["confidence"] = 0.5
        else:
            result["confidence"] = 0.5

        # Ensure intent_type is valid
        valid_intent_types = {"bugfix", "feature", "refactor", "exploration", "question", "modify", "extend", "create"}
        if "intent_type" not in result or result["intent_type"] not in valid_intent_types:
            result["intent_type"] = "unknown"

        # Ensure lane_hint is valid
        valid_lanes = {"planner", "main_agent", "read_only", "clarify"}
        if "lane_hint" not in result or result["lane_hint"] not in valid_lanes:
            # Default based on intent_type.
            # Use read_only ONLY for pure read/explain intents.
            # "exploration" can include edit intents in disguise (e.g. "fix this")
            # → default to clarify, not read_only, to avoid blocking the edit path.
            if result["intent_type"] == "question":
                result["lane_hint"] = "read_only"
            else:
                result["lane_hint"] = "planner"

        # Ensure metadata exists
        if "metadata" not in result or not isinstance(result["metadata"], dict):
            result["metadata"] = {}

        return result

    def _build_intent_result(self, original_request: str, result_dict: dict[str, Any]) -> IntentResult:
        """Build IntentResult from parsed LLM response."""
        # Apply limits
        search_terms = result_dict.get("search_terms", [])[:self.config.max_search_terms]
        # target_files: prefer LLM output (semantically informed) over regex.
        # The grounding engine validates paths against filesystem later.
        target_files: list[str] = []
        llm_files = result_dict.get("target_files", [])
        if isinstance(llm_files, list):
            target_files = [f for f in llm_files if isinstance(f, str) and f.strip()]

        # Symbol role classification:
        # modify_symbols / reference_symbols are the new structured fields.
        # For backward compat, also accept old-style target_symbols and infer roles.
        modify_syms_raw: list[str] = result_dict.get("modify_symbols", [])
        ref_syms_raw: list[str] = result_dict.get("reference_symbols", [])
        old_target_syms: list[str] = result_dict.get("target_symbols", [])

        if modify_syms_raw or ref_syms_raw:
            # LLM populated role-aware fields — use as-is
            modify_symbols = [s for s in modify_syms_raw if isinstance(s, str) and s]
            reference_symbols = [s for s in ref_syms_raw if isinstance(s, str) and s]
            # target_symbols = ONLY modify_symbols (not the union with reference).
            # Reference symbols can be non-code strings (parameter names, domain concepts)
            target_symbols = list(modify_symbols)
        else:
            # Old-style: LLM returned only target_symbols without explicit role info.
            # Conservative backward compat: treat all target_symbols as modify targets.
            # The builder's existing existence-check heuristic will handle INSERT vs MODIFY.
            target_symbols = [s for s in old_target_syms if isinstance(s, str) and s]
            modify_symbols = list(target_symbols)
            reference_symbols = []


        # spec_hints: target_files is always empty (grounding engine handles file discovery)
        # Exception: "new_files" explicitly named by user are passed through for CREATE fast-path.
        spec_hints: dict[str, Any] = {}
        _raw_new_files = result_dict.get("new_files", [])
        if isinstance(_raw_new_files, list) and _raw_new_files:
            # Only include entries that look like file paths (contain '/' or end in known ext)
            import os as _os_ir
            _valid_new_files = [
                f for f in _raw_new_files
                if isinstance(f, str) and f.strip()
                and ('/' in f or '.' in _os_ir.path.basename(f))
            ]
            if _valid_new_files:
                spec_hints["new_files"] = _valid_new_files

        # new_symbols: only relevant for add/extend/create intent
        raw_new_symbols = result_dict.get("new_symbols", [])
        new_symbols = [
            ns for ns in raw_new_symbols
            if isinstance(ns, dict) and ns.get("name")
        ]

        logger.debug(
            "IntentResolver role classification: modify=%s reference=%s new=%s",
            modify_symbols, reference_symbols, [ns.get("name") for ns in new_symbols],
        )

        # Edit kind + guard statement + target_loop_iterable (Intent → Policy layer)
        _edit_kind_raw = (result_dict.get("edit_kind") or "").strip().lower()
        _VALID_EDIT_KINDS = {"guard_add", "body_only", "signature_change", "full_rewrite", "extend"}
        _edit_kind = _edit_kind_raw if _edit_kind_raw in _VALID_EDIT_KINDS else ""
        _guard_statement = ""
        _guard_spec = None  # typed GuardIR — authoritative downstream
        _target_loop_iterable = ""
        if _edit_kind == "guard_add":
            _guard_statement = (result_dict.get("guard_statement") or "").strip()
            if _guard_statement:
                try:
                    ast.parse(_guard_statement, mode="exec")
                except SyntaxError:
                    logger.warning(
                        "IntentResolver: guard_statement has SyntaxError, discarding: %r",
                        _guard_statement[:120],
                    )
                    _guard_statement = ""
            # Build typed GuardIR immediately so downstream never re-parses.
            # [GUARD_SPEC] source: intent_resolver
            if _guard_statement:
                _guard_spec = _parse_guard_ir(_guard_statement)
                if _guard_spec and _guard_spec.is_parsed:
                    logger.debug(
                        "[GUARD_SPEC] intent_resolver: guard_spec built: compact=%r control=%r",
                        _guard_spec.compact, _guard_spec.control,
                    )
                else:
                    logger.debug(
                        "[GUARD_SPEC] intent_resolver: parse_guard returned unparsed IR for %r",
                        _guard_statement[:80],
                    )
                    _guard_spec = None
            # target_loop_iterable: the iterable expression the user described.
            # Keep only if it looks like a valid Python identifier or simple expression
            # (guards against hallucinated multi-word phrases).
            _tli_raw = (result_dict.get("target_loop_iterable") or "").strip()
            if _tli_raw and _tli_raw not in ("iterable_expr_or_empty_string", ""):
                # Accept simple identifiers, attribute accesses, and subscripts.
                # Reject multi-word phrases (contain spaces) or placeholder strings.
                try:
                    ast.parse(_tli_raw, mode="eval")
                    _target_loop_iterable = _tli_raw
                except SyntaxError:
                    logger.debug(
                        "IntentResolver: target_loop_iterable %r is not valid Python, discarding",
                        _tli_raw[:80],
                    )

        # code_concepts: validate and normalize
        _raw_cc = result_dict.get("code_concepts", {}) or {}
        _code_concepts: dict[str, Any] = {}
        if isinstance(_raw_cc, dict):
            import re as _re_cc
            _df_raw = _raw_cc.get("data_fields", [])
            _data_fields = [
                f for f in (_df_raw if isinstance(_df_raw, list) else [])
                if isinstance(f, str) and _re_cc.match(r'^[\w.]+$', f) and len(f) >= 2
            ][:8]
            _bk = _raw_cc.get("behavioral_kind", "")
            _behavioral_kind = _bk if _bk in ("enforcement", "creation", "fix", "query") else ""
            _sp = _raw_cc.get("scope_phase", "")
            _scope_phase = _sp if _sp in ("planning", "execution", "verification", "exploration") else ""
            if _data_fields or _behavioral_kind or _scope_phase:
                _code_concepts = {
                    "data_fields": _data_fields,
                    "behavioral_kind": _behavioral_kind,
                    "scope_phase": _scope_phase,
                }

        # Generic vocabulary (e.g. "plan", "execution", "order") is filtered
        # downstream in SpecResolver._filter_terms_by_graph via cardinality gate:

        # scope_hint / complexity_hint / is_test_write / is_style_fix
        _scope_hint = Scope(result_dict.get("scope_hint", "single_file"))

        # project_wide scope_hint contradicts a specific scope_phase — clear scope_phase
        # so SpecResolver doesn't constrain grounding to a narrow system domain.
        if _scope_hint == Scope.PROJECT_WIDE and _scope_phase:
            logger.info(
                "IntentResolver: scope_hint=project_wide overrides scope_phase='%s' — "
                "clearing scope_phase to prevent constrained grounding",
                _scope_phase,
            )
            _scope_phase = ""
            if _code_concepts:
                _code_concepts["scope_phase"] = ""
                if not _code_concepts.get("data_fields") and not _code_concepts.get("behavioral_kind"):
                    _code_concepts = {}

        _complexity_map = {"trivial": Complexity.LOW, "normal": Complexity.MEDIUM, "complex": Complexity.HIGH}
        _complexity_hint = _complexity_map.get(result_dict.get("complexity_hint", ""), Complexity.LOW)

        _is_test_write = bool(result_dict.get("is_test_write", False))
        _is_style_fix = bool(result_dict.get("is_style_fix", False))
        _is_filesystem_op = bool(result_dict.get("is_filesystem_op", False))
        _is_ui_change = bool(result_dict.get("is_ui_change", False))
        _is_interface_preserving = bool(result_dict.get("is_interface_preserving", False))

        # Create IntentResult
        return IntentResult(
            original_request=original_request,
            normalized_query=result_dict.get("normalized_query", original_request),
            search_terms=search_terms,
            intent_type=result_dict.get("intent_type", "unknown"),
            edit_kind=_edit_kind,
            guard_statement=_guard_statement,
            guard_spec=_guard_spec,
            target_loop_iterable=_target_loop_iterable,
            target_files=target_files,
            target_symbols=target_symbols,
            new_symbols=new_symbols,
            modify_symbols=modify_symbols,
            reference_symbols=reference_symbols,
            lane_hint=result_dict.get("lane_hint", "planner"),
            scope_hint=_scope_hint,
            complexity_hint=_complexity_hint,
            is_test_write=_is_test_write,
            is_style_fix=_is_style_fix,
            is_filesystem_op=_is_filesystem_op,
            is_ui_change=_is_ui_change,
            is_interface_preserving=_is_interface_preserving,
            confidence=result_dict.get("confidence", 0.5),
            metadata=result_dict.get("metadata", {}),
            spec_hints=spec_hints,
            code_concepts=_code_concepts,
        )

    def _fallback_extraction(self, request: str) -> IntentResult:
        """Minimal fallback extraction when LLM fails.

        No keyword mapping, no intent classification.
        Only extracts words and file patterns.
        """
        logger.info("IntentResolver: using minimal fallback extraction for: %r", request[:100])

        # 1. Extract all words (language-agnostic, space/punctuation delimited)
        def extract_all_words(text: str) -> list[str]:
            """Extract words from any language."""
            words = []
            _cur = []
            for _ch in text:
                if _ch.isalnum() or _ch in ('_', '-'):
                    _cur.append(_ch)
                else:
                    if _cur:
                        words.append(''.join(_cur))
                        _cur = []
            if _cur:
                words.append(''.join(_cur))
            return [w for w in words if len(w) >= 2]  # Keep words with at least 2 chars

        # 2. Minimal stop words filtering (language-agnostic)
        def filter_minimal_stop_words(words: list[str]) -> list[str]:
            """Filter out only the most common stop words."""
            # Minimal set: articles/prepositions in major languages
            minimal_stop = {
                # English
                "the", "and", "for", "with", "this", "that",
                # Korean particles removed — regex split doesn't separate them
                # from adjacent text in mixed-language requests
                # Universal
                "a", "an", "of", "to", "in", "on", "at", "by",
            }
            return [w for w in words if w not in minimal_stop]

        # 3. Extract words
        all_words = extract_all_words(request)
        search_terms = filter_minimal_stop_words(all_words)

        # 4. Extract file patterns (char split, no regex)
        _file_extensions = {'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'md', 'json', 'yaml', 'yml', 'toml', 'txt'}
        file_matches = []
        _delims = ' :()[]{}<>"\'\t\n\r'
        _buf = []
        for _ch in request:
            if _ch in _delims:
                if _buf:
                    _w = ''.join(_buf).strip('.,;!')
                    if '.' in _w:
                        _parts = _w.rsplit('.', 1)
                        if len(_parts) == 2 and _parts[1].lower() in _file_extensions:
                            file_matches.append(_w)
                    _buf = []
            else:
                _buf.append(_ch)
        if _buf:
            _w = ''.join(_buf).strip('.,;!')
            if '.' in _w:
                _parts = _w.rsplit('.', 1)
                if len(_parts) == 2 and _parts[1].lower() in _file_extensions:
                    file_matches.append(_w)

        # 5. Determine lane hint based on file extension pattern only
        lane_hint = "planner"  # Default
        if file_matches:
            # Check if any file has non-AST extension
            non_ast_extensions = {".css", ".html", ".json", ".md", ".yaml", ".yml", ".toml", ".txt"}
            if any(any(ext in f.lower() for ext in non_ast_extensions) for f in file_matches):
                lane_hint = "main_agent"

        # 6. Basic target inference from file matches
        target_files = list(file_matches)[:self.config.max_target_files]

        return IntentResult(
            original_request=request,
            normalized_query=request,  # No normalization in fallback
            search_terms=search_terms[:self.config.max_search_terms],
            intent_type="unknown",  # No intent classification in fallback
            target_files=target_files,
            target_symbols=[],
            lane_hint=lane_hint,
            confidence=0.1,  # Very low confidence for minimal fallback
            metadata={"source": "minimal_fallback"},
            spec_hints={"modify_files": target_files} if target_files else {},
        )

    def _create_empty_result(self, request: str) -> IntentResult:
        """Create empty result for empty request."""
        return IntentResult(
            original_request=request,
            normalized_query=request,
            search_terms=[],
            intent_type="unknown",
            target_files=[],
            target_symbols=[],
            lane_hint="planner",
            confidence=0.0,
            metadata={"source": "empty_request"},
            spec_hints={},
        )

    def _create_fallback_dict(self, original_request: str) -> dict[str, Any]:
        """Create fallback result dictionary."""
        return {
            "normalized_query": original_request,
            "search_terms": [],
            "intent_type": "unknown",
            "lane_hint": "planner",
            "target_files": [],
            "target_symbols": [],
            "confidence": 0.2,
            "metadata": {"source": "llm_parse_failed"},
        }

    def clear_cache(self) -> None:
        """Clear the intent resolution cache."""
        self._cache.clear()
        logger.debug("IntentResolver cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())[:10],  # First 10 keys
        }


def create_intent_resolver(
    llm_client: Any,
    model: str,
    repo_root: Optional[str] = None,
    enable_cache: bool = True,
) -> IntentResolver:
    """Factory function to create IntentResolver with default config."""
    config = IntentResolutionConfig(
        llm_client=llm_client,
        model=model,
        enable_cache=enable_cache,
        cache_ttl_seconds=300,
        max_search_terms=10,
        max_target_files=5,
    )
    return IntentResolver(config)
