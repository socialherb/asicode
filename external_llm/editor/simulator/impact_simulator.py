"""
Impact simulator for candidate plans.

Analyses the potential impact of a candidate plan using the repository graph.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.agent.operation_models import OperationPlan
from external_llm.editor.simulator.dependency_traversal import (
    collect_direct_callers,
    collect_file_dependencies,
    collect_impacted_files,
    collect_indirect_callers,
)
from external_llm.editor.simulator.impact_models import ImpactConfig, ImpactReport
from external_llm.editor.simulator.risk_estimator import RiskEstimator
from external_llm.graph.graph_facade import RepositoryGraphFacade

logger = logging.getLogger(__name__)


class ImpactSimulator:
    """Simulates the impact of a candidate plan on the repository."""

    def __init__(
        self,
        graph_facade: RepositoryGraphFacade,
        risk_estimator: Optional[RiskEstimator] = None,
        config: Optional[ImpactConfig] = None,
        test_finder: Any = None,
    ):
        self.graph_facade = graph_facade
        self.config = config or ImpactConfig()
        self.risk_estimator = risk_estimator or RiskEstimator(self.config)
        self.test_finder = test_finder

    def simulate_candidate(self, candidate: Any) -> ImpactReport:
        """
        Simulate impact for a PlanCandidate.

        Args:
            candidate: PlanCandidate object (must have operation_plan attribute).

        Returns:
            ImpactReport with risk score.
        """
        if not hasattr(candidate, "operation_plan"):
            logger.warning("Candidate missing operation_plan, returning empty report")
            return self._empty_report()

        plan = candidate.operation_plan
        return self.simulate_plan(plan)

    def simulate_plan(self, plan: OperationPlan) -> ImpactReport:
        """
        Simulate impact for an OperationPlan.

        Args:
            plan: OperationPlan to analyze.

        Returns:
            ImpactReport with risk score.
        """
        # Extract targets
        target_symbols = self._extract_target_symbols_from_plan(plan)
        target_files = self._extract_target_files_from_plan(plan)

        # Early exit if no targets
        if not target_symbols and not target_files:
            logger.debug("No target symbols or files found in plan")
            return self._empty_report()

        # Collect impact data
        direct_callers: set[str] = set()
        indirect_callers: set[str] = set()
        dependencies: set[str] = set()
        impacted_files: set[str] = set()

        # Process each target symbol
        for symbol in target_symbols:
            # Direct callers
            dc = collect_direct_callers(self.graph_facade, symbol)
            direct_callers.update(dc)

            # Indirect callers (within depth limit)
            ic = collect_indirect_callers(
                self.graph_facade,
                symbol,
                depth=self.config.caller_depth,
                max_nodes=self.config.max_nodes,
            )
            indirect_callers.update(ic)

            # File dependencies (if we can determine symbol's file)
            node = self.graph_facade.get_symbol(symbol)
            if node and node.file_path:
                deps = collect_file_dependencies(self.graph_facade, node.file_path)
                dependencies.update(deps)

        # Process each target file
        for file_path in target_files:
            deps = collect_file_dependencies(self.graph_facade, file_path)
            dependencies.update(deps)

        # Collect impacted files
        impacted_files.update(
            collect_impacted_files(self.graph_facade, target_symbols, target_files)
        )

        # Find affected tests
        affected_tests = self._find_affected_tests(target_symbols, target_files)

        # Build report
        report = ImpactReport(
            target_symbols=list(target_symbols),
            target_files=list(target_files),
            direct_callers=list(direct_callers),
            indirect_callers=list(indirect_callers),
            affected_tests=affected_tests,
            dependencies=list(dependencies),
            impacted_files=list(impacted_files),
            risk_score=0.0,  # computed below
        )

        # Compute risk score
        report.risk_score = self.risk_estimator.estimate(report)

        # Store normalized risk in metadata
        report.metadata["normalized_risk"] = self.risk_estimator.normalize(
            report.risk_score
        )

        logger.debug(
            "Impact simulation complete: %d symbols, %d files, risk=%.3f",
            len(target_symbols),
            len(target_files),
            report.risk_score,
        )
        return report

    def _extract_target_symbols_from_plan(self, plan: OperationPlan) -> list[str]:
        """Extract unique target symbols from plan operations."""
        symbols: set[str] = set()
        for op in plan.operations:
            if op.symbol:
                symbols.add(op.symbol)
        return list(symbols)

    def _extract_target_files_from_plan(self, plan: OperationPlan) -> list[str]:
        """Extract unique target files from plan operations."""
        files: set[str] = set()
        for op in plan.operations:
            if op.path:
                files.add(op.path)
        return list(files)

    def _find_affected_tests(
        self, symbols: list[str], files: list[str]
    ) -> list[str]:
        """Find tests that may be affected by changes to symbols/files."""
        if not self.test_finder:
            return []

        affected: set[str] = set()
        # Look for tests targeting each symbol
        for symbol in symbols:
            try:
                tests = self.test_finder.find_tests_for_symbol(symbol)
                affected.update(tests)
            except Exception:
                pass

        # Look for tests in or referencing each file
        for file_path in files:
            # Simple heuristic: if file ends with _test.py or starts with test_
            if "_test.py" in file_path or file_path.startswith("test_"):
                affected.add(file_path)

        return list(affected)

    def _empty_report(self) -> ImpactReport:
        """Return an empty impact report."""
        return ImpactReport()
