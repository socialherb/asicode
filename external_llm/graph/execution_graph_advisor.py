"""
Graph-aware execution advisor for repair, refactor, safety, and verification.

This module provides graph-derived hints and risk assessments for the execution
pipeline. All functions are safe to call without graph context (graceful fallback).

Canonical entry point for P4 graph integration in execution layers.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Thresholds (tunable)
HIGH_CALLER_THRESHOLD = 10
WIDE_IMPACT_FILE_THRESHOLD = 6


@dataclass
class GraphRepairHints:
    """Graph-derived hints for repair strategy selection."""
    prefer_symbol_focused_repair: bool = False
    high_breakage_risk: bool = False
    prefer_conservative_repair: bool = False
    impact_file_count: int = 0
    caller_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefer_symbol_focused_repair": self.prefer_symbol_focused_repair,
            "high_breakage_risk": self.high_breakage_risk,
            "prefer_conservative_repair": self.prefer_conservative_repair,
            "impact_file_count": self.impact_file_count,
            "caller_count": self.caller_count,
            "reason": self.reason,
        }


@dataclass
class GraphSafetyIssue:
    """A graph-derived safety concern."""
    code: str  # e.g., "HIGH_CALLER_SYMBOL_EDIT", "WIDE_IMPACT_REFACTOR"
    severity: str  # "warning" or "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphVerificationScope:
    """Graph-derived verification scope hints."""
    symbols: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    scope_reason: str = ""
    level: str = "standard"  # "narrow" | "standard" | "broad"
    ranked_test_summary: dict[str, Any] = field(default_factory=dict)  # P8-3: symbol-aware summary
    impact_summary: dict[str, Any] = field(default_factory=dict)  # P9-1: impact propagation summary
    dependency_graph_summary: dict[str, Any] = field(default_factory=dict)  # P9-2: test dependency graph summary
    verification_set_summary: dict[str, Any] = field(default_factory=dict)  # P9-3: verification set builder summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "files": self.files,
            "test_targets": self.test_targets,
            "scope_reason": self.scope_reason,
            "level": self.level,
            "ranked_test_summary": self.ranked_test_summary,
            "impact_summary": self.impact_summary,
            "dependency_graph_summary": self.dependency_graph_summary,
            "verification_set_summary": self.verification_set_summary,
        }


@dataclass
class GraphRefactorContext:
    """Graph-derived context for refactoring operations."""
    target_symbol: str = ""
    caller_count: int = 0
    callee_count: int = 0
    related_symbol_count: int = 0
    impact_files: list[str] = field(default_factory=list)
    risk_level: str = "unknown"  # "low", "medium", "high", "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_symbol": self.target_symbol,
            "caller_count": self.caller_count,
            "callee_count": self.callee_count,
            "related_symbol_count": self.related_symbol_count,
            "impact_files": self.impact_files,
            "risk_level": self.risk_level,
        }


class ExecutionGraphAdvisor:
    """
    Provides graph-aware advice for execution pipeline stages.

    Safe to instantiate with graph_facade=None; all methods return
    safe defaults when graph is unavailable.
    """

    def __init__(self, graph_facade=None):
        self._facade = graph_facade
        self._memo: dict[str, Any] = {}  # internal micro-cache
        self._memo_hits = 0
        self._memo_misses = 0

    @property
    def available(self) -> bool:
        return self._facade is not None

    # ── Internal memoization helpers ─────────────────────────────────────────

    def _memo_key(self, method: str, **kwargs) -> str:
        """Generate memo key for internal cache."""
        parts = [method]
        for k, v in sorted(kwargs.items()):
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}={','.join(str(x) for x in v)}")
            elif isinstance(v, dict):
                parts.append(f"{k}=dict({len(v)})")
            else:
                parts.append(f"{k}={v}")
        return "|".join(parts)

    def _memo_get(self, key: str) -> Optional[Any]:
        if key in self._memo:
            self._memo_hits += 1
            return self._memo[key]
        self._memo_misses += 1
        return None

    def _memo_set(self, key: str, value: Any) -> None:
        self._memo[key] = value

    def clear_memo(self) -> None:
        """Clear internal memoization cache."""
        self._memo.clear()

    def get_memo_stats(self) -> dict[str, Any]:
        """Return memoization statistics."""
        total = self._memo_hits + self._memo_misses
        return {
            "size": len(self._memo),
            "hits": self._memo_hits,
            "misses": self._memo_misses,
            "hit_rate": self._memo_hits / total if total > 0 else 0.0,
        }

    def get_repair_hints(
        self,
        target_symbol: Optional[str] = None,
        target_file: Optional[str] = None,
        graph_context: Optional[dict] = None,
    ) -> GraphRepairHints:
        """
        Get graph-aware repair hints for a target symbol/file.

        Uses graph_context from spec.metadata if available, otherwise
        queries facade directly.
        """
        # Memo check
        memo_key = self._memo_key(
            "repair_hints",
            symbol=target_symbol,
            file=target_file,
            has_gc=bool(graph_context),
        )
        cached = self._memo_get(memo_key)
        if cached is not None:
            return cached

        hints = GraphRepairHints()

        try:
            # Try graph_context first (already computed)
            if graph_context:
                resolved = graph_context.get("resolved_symbols", [])
                impact_files = graph_context.get("impact_files", [])
                callers = graph_context.get("callers", {})
                confidence = graph_context.get("graph_confidence", 0.0)

                hints.impact_file_count = len(impact_files)

                # Count callers for target symbol
                if target_symbol and target_symbol in callers:
                    hints.caller_count = len(callers[target_symbol])
                elif callers:
                    # Sum all caller counts
                    hints.caller_count = sum(len(v) for v in callers.values())

                # Determine repair strategy hints
                if confidence >= 0.8 and resolved:
                    hints.prefer_symbol_focused_repair = True
                    hints.reason = f"High confidence ({confidence:.2f}) with {len(resolved)} resolved symbols"

                if hints.caller_count >= HIGH_CALLER_THRESHOLD:
                    hints.high_breakage_risk = True
                    hints.prefer_conservative_repair = True
                    hints.reason = f"High caller count ({hints.caller_count}) - breakage risk"

                if len(impact_files) >= WIDE_IMPACT_FILE_THRESHOLD:
                    hints.prefer_conservative_repair = True
                    if not hints.reason:
                        hints.reason = f"Wide impact ({len(impact_files)} files)"

                self._memo_set(memo_key, hints)
                return hints

            # Fallback: query facade directly
            if not self._facade or not target_symbol:
                self._memo_set(memo_key, hints)
                return hints

            callers = self._facade.get_callers(target_symbol, file_path=target_file)
            callees = self._facade.get_callees(target_symbol, file_path=target_file)
            hints.caller_count = len(callers)

            if hints.caller_count >= HIGH_CALLER_THRESHOLD:
                hints.high_breakage_risk = True
                hints.prefer_conservative_repair = True
                hints.reason = f"High caller count ({hints.caller_count})"
            elif hints.caller_count == 0 and len(callees) <= 2:
                hints.prefer_symbol_focused_repair = True
                hints.reason = "Low-impact symbol (few callers/callees)"

        except Exception as e:
            logger.debug("Graph repair hints unavailable: %s", e)

        self._memo_set(memo_key, hints)
        return hints

    def get_safety_issues(
        self,
        target_symbols: Optional[list[str]] = None,
        target_files: Optional[list[str]] = None,
        operation_kind: Optional[str] = None,
        graph_context: Optional[dict] = None,
    ) -> list[GraphSafetyIssue]:
        """
        Check for graph-derived safety issues before execution.
        Returns warnings (never blocks).
        """
        # Memo check
        memo_key = self._memo_key(
            "safety_issues",
            symbols=target_symbols,
            files=target_files,
            op_kind=operation_kind,
            has_gc=bool(graph_context),
        )
        cached = self._memo_get(memo_key)
        if cached is not None:
            return cached

        issues = []

        try:
            if graph_context:
                confidence = graph_context.get("graph_confidence", 0.0)
                impact_files = graph_context.get("impact_files", [])
                callers = graph_context.get("callers", {})
                unresolved = graph_context.get("unresolved_symbols", [])

                # Check high-caller symbol edit
                total_callers = sum(len(v) for v in callers.values())
                if total_callers >= HIGH_CALLER_THRESHOLD:
                    issues.append(GraphSafetyIssue(
                        code="HIGH_CALLER_SYMBOL_EDIT",
                        severity="warning",
                        message=f"Editing symbol(s) with {total_callers} callers - verify all call sites",
                        details={"caller_count": total_callers},
                    ))

                # Check wide impact
                if len(impact_files) >= WIDE_IMPACT_FILE_THRESHOLD:
                    issues.append(GraphSafetyIssue(
                        code="WIDE_IMPACT_REFACTOR",
                        severity="warning",
                        message=f"Change impacts {len(impact_files)} files",
                        details={"impact_files": impact_files[:10]},
                    ))

                # Check low confidence + structural edit
                structural_ops = {"RENAME_SYMBOL", "MOVE_SYMBOL", "MODIFY_SYMBOL", "replace_symbol_body"}
                if confidence < 0.4 and operation_kind and operation_kind in structural_ops:
                    issues.append(GraphSafetyIssue(
                        code="LOW_CONFIDENCE_STRUCTURAL_EDIT",
                        severity="warning",
                        message=f"Structural edit with low graph confidence ({confidence:.2f})",
                        details={"graph_confidence": confidence, "operation_kind": operation_kind},
                    ))

                # Check unresolved symbols
                if len(unresolved) >= 2 and operation_kind and operation_kind in structural_ops:
                    issues.append(GraphSafetyIssue(
                        code="UNRESOLVED_SYMBOL_STRUCTURAL_EDIT",
                        severity="warning",
                        message=f"Structural edit with {len(unresolved)} unresolved symbols",
                        details={"unresolved_symbols": unresolved[:5]},
                    ))

            elif self._facade and target_symbols:
                for sym in (target_symbols or [])[:5]:
                    try:
                        callers = self._facade.get_callers(sym)
                        if len(callers) >= HIGH_CALLER_THRESHOLD:
                            issues.append(GraphSafetyIssue(
                                code="HIGH_CALLER_SYMBOL_EDIT",
                                severity="warning",
                                message=f"Symbol '{sym}' has {len(callers)} callers",
                                details={"symbol": sym, "caller_count": len(callers)},
                            ))
                    except Exception:
                        pass

        except Exception as e:
            logger.debug("Graph safety check unavailable: %s", e)

        self._memo_set(memo_key, issues)
        return issues

    def get_verification_scope(
        self,
        target_symbols: Optional[list[str]] = None,
        target_files: Optional[list[str]] = None,
        graph_context: Optional[dict] = None,
        composite_risk_level: Optional[str] = None,  # "low"|"medium"|"high"|"critical"
        patch_risk: Optional[Any] = None,  # P9 PatchRisk object from patch_risk_estimator
    ) -> GraphVerificationScope:
        """
        Get graph-aware verification scope (which files/symbols/tests to check after apply).
        """
        # Memo check
        memo_key = self._memo_key(
            "verification_scope",
            symbols=target_symbols,
            files=target_files,
            has_gc=bool(graph_context),
            composite_risk_level=composite_risk_level,
            has_patch_risk=bool(patch_risk),
        )
        cached = self._memo_get(memo_key)
        if cached is not None:
            return cached

        scope = GraphVerificationScope()

        try:
            if graph_context:
                resolved = graph_context.get("resolved_symbols", [])
                impact_files = graph_context.get("impact_files", [])
                primary_files = graph_context.get("primary_files", [])
                callers = graph_context.get("callers", {})

                # Symbols to verify: resolved targets + their direct callers
                scope.symbols = [s.get("name", "") for s in resolved if isinstance(s, dict)]
                for sym, caller_list in callers.items():
                    for c in caller_list[:5]:
                        caller_name = c.get("symbol", "") if isinstance(c, dict) else str(c)
                        if caller_name and caller_name not in scope.symbols:
                            scope.symbols.append(caller_name)

                # Files to verify: primary + impact (deduplicated)
                all_files = list(dict.fromkeys(primary_files + impact_files))
                scope.files = all_files[:20]

                # P8-3: Symbol-aware test targeting
                _symbol_aware_suffix = ""
                try:
                    from external_llm.testing.symbol_aware_test_finder import SymbolAwareTestFinder

                    # Determine repo_root (from facade or fallback)
                    repo_root = getattr(self._facade, 'repo_root', None) if self._facade else None
                    if repo_root:
                        # P9-2: Build test dependency graph
                        dep_graph = None
                        dep_graph_summary: dict[str, Any] = {}
                        try:
                            from external_llm.testing.test_dependency_graph import DependencyGraphBuilder
                            builder = DependencyGraphBuilder(repo_root)
                            dep_graph = builder.build()
                            dep_graph_summary = dep_graph.get_summary()
                        except Exception as _dep_err:
                            logger.debug("Test dependency graph build failed: %s", _dep_err)

                        finder = SymbolAwareTestFinder(repo_root, graph_facade=self._facade, dependency_graph=dep_graph)
                        ranked_targets = finder.discover_test_targets(
                            target_symbols=list(target_symbols or []),
                            target_files=list(target_files or []),
                            impact_files=graph_context.get("impact_files", []) if graph_context else [],
                            graph_context=graph_context,
                            scope_level=scope.level,
                        )
                        if ranked_targets:
                            scope.test_targets = finder.to_path_list(ranked_targets)
                            scope.ranked_test_summary = finder.build_summary(ranked_targets)
                            _symbol_aware_suffix = f" ({len(ranked_targets)} symbol-aware test targets)"
                        scope.dependency_graph_summary = dep_graph_summary
                except Exception as e:
                    logger.debug("Symbol-aware test finder failed, using fallback: %s", e)

                # Fallback: filename heuristic if no symbol-aware targets found
                if not scope.test_targets:
                    scope.test_targets = [
                        f for f in all_files
                        if "test" in f.lower() or f.endswith("_test.py")
                    ]

                scope.scope_reason = (
                    f"Graph-derived: {len(scope.symbols)} symbols, "
                    f"{len(scope.files)} files, {len(scope.test_targets)} test targets"
                    + _symbol_aware_suffix
                )

            elif self._facade and target_symbols:
                for sym in (target_symbols or [])[:3]:
                    try:
                        callers = self._facade.get_callers(sym)
                        scope.symbols.append(sym)
                        for edge in callers[:5]:
                            caller_sym = getattr(edge, 'caller_symbol', None) or (
                                edge.get('caller_symbol') if isinstance(edge, dict) else None
                            )
                            if caller_sym:
                                scope.symbols.append(caller_sym)
                            caller_file = getattr(edge, 'caller_file', None) or (
                                edge.get('caller_file') if isinstance(edge, dict) else None
                            )
                            if caller_file:
                                scope.files.append(caller_file)
                    except Exception:
                        pass

                # Deduplicate
                scope.symbols = list(dict.fromkeys(scope.symbols))
                scope.files = list(dict.fromkeys(scope.files))
                scope.test_targets = [f for f in scope.files if "test" in f.lower()]
                scope.scope_reason = f"Facade-derived: {len(scope.symbols)} symbols"

            if target_files:
                for f in target_files:
                    if f not in scope.files:
                        scope.files.append(f)

        except Exception as e:
            logger.debug("Graph verification scope unavailable: %s", e)

        # P9-1: Impact propagation summary
        try:
            from external_llm.editor.verification.impact_propagation import ImpactPropagationEngine

            if self._facade:
                prop_engine = ImpactPropagationEngine(graph_facade=self._facade)
                impact = prop_engine.propagate(
                    changed_symbols=list(target_symbols or []),
                    changed_files=list(target_files or []),
                    max_depth=2,
                )
                scope.impact_summary = impact.to_summary()

                # Expand files/symbols from impact if broader than current scope
                for f in impact.impacted_files:
                    if f not in scope.files and len(scope.files) < 20:
                        scope.files.append(f)
                for s in impact.impacted_symbols:
                    if s not in scope.symbols:
                        scope.symbols.append(s)
        except Exception as e:
            logger.debug("Impact propagation in verification scope failed: %s", e)

        # P8-2: Set scope level based on composite risk
        if composite_risk_level == "critical":
            scope.level = "broad"
            scope.scope_reason += " (broadened for critical risk)"
        elif composite_risk_level == "high":
            scope.level = "broad"
            scope.scope_reason += " (broadened for high risk)"
        elif composite_risk_level in ("low", None):
            if len(scope.files) <= 3:
                scope.level = "narrow"
            else:
                scope.level = "standard"
        else:
            scope.level = "standard"

        # P9-3: Build structured verification set
        try:
            from external_llm.editor.verification.verification_set_builder import VerificationSetBuilder

            # Collect inputs from earlier steps (variables may not exist if those steps failed)
            _symbol_targets = locals().get('ranked_targets', None)
            _dep_candidates = None
            _impact_set = locals().get('impact', None)

            if _dep_candidates is None and _impact_set is not None:
                # Use dep_graph from locals if available
                _dep_graph = locals().get('dep_graph', None)
                if _dep_graph is not None:
                    try:
                        _dep_candidates = _dep_graph.get_tests_for_impact_set(_impact_set)
                    except Exception:
                        pass

            vs_builder = VerificationSetBuilder()
            verification_set = vs_builder.build(
                symbol_targets=_symbol_targets,
                dep_graph_candidates=_dep_candidates,
                impact_set=_impact_set,
                composite_risk=None,  # We have level string, not full object
                scope_level_hint=scope.level,
                patch_risk=patch_risk,  # P9 integration
            )
            # Pass risk level directly since we have string not object
            verification_set.composite_risk_level = composite_risk_level or "unknown"
            verification_set.selected_scope_level = vs_builder._resolve_scope_level(
                scope.level, composite_risk_level or "unknown"
            )

            # Update scope with verification set results
            flattened = verification_set.flatten_paths()
            if flattened:
                scope.test_targets = flattened
            scope.verification_set_summary = verification_set.to_summary()

        except Exception as e:
            logger.debug("Verification set builder failed: %s", e)

        self._memo_set(memo_key, scope)
        return scope

    def get_refactor_context(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
    ) -> GraphRefactorContext:
        """
        Get graph-derived context for a refactoring operation.
        """
        # Memo check
        memo_key = self._memo_key("refactor_context", symbol=symbol_name, file=file_path)
        cached = self._memo_get(memo_key)
        if cached is not None:
            return cached

        ctx = GraphRefactorContext(target_symbol=symbol_name)

        if not self._facade:
            self._memo_set(memo_key, ctx)
            return ctx

        try:
            callers = self._facade.get_callers(symbol_name, file_path=file_path)
            callees = self._facade.get_callees(symbol_name, file_path=file_path)
            ctx.caller_count = len(callers)
            ctx.callee_count = len(callees)

            # Collect impact files from callers
            impact_files = set()
            for edge in callers:
                f = getattr(edge, 'caller_file', None)
                if f:
                    impact_files.add(f)
            for edge in callees:
                f = getattr(edge, 'callee_file', None)
                if f:
                    impact_files.add(f)
            ctx.impact_files = sorted(impact_files)

            # Related symbols
            try:
                related = self._facade.get_related_symbols(symbol_name, file_path=file_path)
                ctx.related_symbol_count = len(related) if related else 0
            except Exception:
                ctx.related_symbol_count = 0

            # Risk level
            if ctx.caller_count >= HIGH_CALLER_THRESHOLD or len(ctx.impact_files) >= WIDE_IMPACT_FILE_THRESHOLD:
                ctx.risk_level = "high"
            elif ctx.caller_count >= 3 or len(ctx.impact_files) >= 3:
                ctx.risk_level = "medium"
            else:
                ctx.risk_level = "low"

        except Exception as e:
            logger.debug("Graph refactor context unavailable for '%s': %s", symbol_name, e)

        self._memo_set(memo_key, ctx)
        return ctx

    def get_test_targets(
        self,
        target_symbols: Optional[list[str]] = None,
        target_files: Optional[list[str]] = None,
        graph_context: Optional[dict] = None,
    ) -> list[str]:
        """
        Get graph-derived test file recommendations.
        """
        scope = self.get_verification_scope(target_symbols, target_files, graph_context)
        return scope.test_targets

    def get_execution_policy(
        self,
        target_symbols: Optional[list[str]] = None,
        target_files: Optional[list[str]] = None,
        operation_kind: Optional[str] = None,
        graph_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Derive actionable execution policy from graph signals.

        Augments get_safety_issues() with enforceable policy hints.
        Does NOT remove or replace existing warning behavior.

        Policy is derived from graph_context confidence and unresolved symbols.
        When graph_context is None, returns a trusted (no-op) policy so
        non-graph paths are completely unaffected.

        Returns dict with keys:
            mode                  — "trusted"|"guarded"|"conservative"|"blocked"
            requires_anchor_read  — True when anchor verification is required
            force_conservative_mode — True to use conservative repair strategy
            block_structural_edit — True to abort the structural edit
            fallback_reason       — str or None
            graph_confidence      — float
            unresolved_count      — int
        """
        from external_llm.agent.gsg_safety import STRUCTURAL_OP_KINDS, build_gsg_execution_policy
        is_structural = bool(operation_kind and operation_kind in STRUCTURAL_OP_KINDS)
        return build_gsg_execution_policy(graph_context, is_structural_op=is_structural)
