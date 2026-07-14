"""Edit Localization — code-centric structural analysis for edit target ranking.

Core principle: "Where is the value actually determined?" not "Does the request
mention a code token?"

Pipeline:
  1. extract_flow_facts(body_source) -> SymbolFlowFacts
  2. score_edit_relevance(request, facts) -> float
"""

from .dataflow_extractor import SymbolFlowFacts, extract_flow_facts, resolve_alias_root
from .graph_propagator import PropagatedScore, expand_candidates, propagate_scores
from .relevance_scorer import score_edit_relevance
from .request_analyzer import RequestSemantics, analyze_request

__all__ = [
    "PropagatedScore",
    "RequestSemantics",
    "SymbolFlowFacts",
    "analyze_request",
    "expand_candidates",
    "extract_flow_facts",
    "propagate_scores",
    "resolve_alias_root",
    "score_edit_relevance",
]
