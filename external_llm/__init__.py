"""External LLM package.

``ExternalLLMService`` is re-exported here for backwards compatibility
(``from external_llm import ExternalLLMService``), but loaded LAZILY via module
``__getattr__`` (PEP 562).

Why lazy: ``external_llm.service`` pulls in the HTTP stack (``requests`` and
provider clients), costing ~250ms to import. Previously this ran eagerly inside
the package ``__init__``, so importing ANY submodule — e.g.
``external_llm.agent.auto_correction``, ``external_llm.agent.operation_models`` —
paid that cost even though it never touches the service. Sub-agent/IPC workers
are spawned per agent, so this tax compounded across processes. With lazy
loading the service module is imported only when ``ExternalLLMService`` is
actually accessed; submodule imports skip it entirely.

Compatibility:
  * ``from external_llm import ExternalLLMService``      → triggers ``__getattr__`` → works
  * ``external_llm.ExternalLLMService``                     → triggers ``__getattr__`` → works
  * ``unittest.mock.patch("external_llm.ExternalLLMService")`` → resolves via ``getattr``
    (lazy load) then ``setattr`` (plain module-dict write) → works
  * ``import external_llm.agent.X``                          → never touches ``service``

Caching: the first access imports the real attribute and writes it into the
module dict (``globals()[name] = attr``), so subsequent accesses bypass
``__getattr__`` entirely — and a ``patch`` restore leaves the real class in
place.
"""
from __future__ import annotations

__all__ = ["ExternalLLMService"]

# Map each lazily-exported public name to the submodule that defines it.
# Relative module paths are resolved against this package via importlib.
_LAZY_EXPORTS: dict[str, str] = {
    "ExternalLLMService": ".service",
}


def __getattr__(name: str):
    """Lazily import and cache a public name on first access (PEP 562)."""
    rel = _LAZY_EXPORTS.get(name)
    if rel is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    attr = getattr(importlib.import_module(rel, __name__), name)
    globals()[name] = attr  # cache: future access skips __getattr__
    return attr


def __dir__() -> list[str]:
    """Expose lazily-exported names so ``dir(external_llm)`` lists them."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
