"""Guards the lazy ``ExternalLLMService`` export in ``external_llm/__init__.py``.

``external_llm.service`` pulls in the HTTP stack (``requests`` + provider
clients), costing ~250ms. The package ``__init__`` MUST NOT import it eagerly,
otherwise importing any submodule (e.g. ``external_llm.agent.operation_models``)
pays that tax. These tests lock in that invariant so a future eager re-import
is caught as a regression, and that the lazy ``__getattr__`` (PEP 562) keeps
``from external_llm import ExternalLLMService`` and
``unittest.mock.patch("external_llm.ExternalLLMService")`` working.
"""
from __future__ import annotations

import importlib
import sys
import unittest.mock as mock


def _purge_external_llm_tree() -> None:
    """Remove every ``external_llm*`` module from ``sys.modules`` so the next
    import re-runs the package ``__init__`` from a clean state."""
    for mod in [m for m in sys.modules if m == "external_llm" or m.startswith("external_llm.")]:
        del sys.modules[mod]


def test_submodule_import_does_not_load_service() -> None:
    """Core invariant: importing a submodule MUST NOT eagerly import
    ``external_llm.service`` (the ~250ms HTTP-stack chain)."""
    _purge_external_llm_tree()
    importlib.import_module("external_llm.agent.operation_models")
    # The package itself and the requested submodule ARE loaded...
    assert "external_llm" in sys.modules
    assert "external_llm.agent.operation_models" in sys.modules
    # ...but the heavy service module is NOT.
    assert "external_llm.service" not in sys.modules, (
        "external_llm/__init__.py eagerly imported external_llm.service — this "
        "adds ~250ms to every submodule import. Keep the export lazy via "
        "__getattr__ (PEP 562)."
    )


def test_lazy_access_loads_service_and_caches() -> None:
    """Accessing ``ExternalLLMService`` triggers the lazy load exactly once and
    caches the real attribute in the module dict."""
    _purge_external_llm_tree()
    import external_llm

    assert "external_llm.service" not in sys.modules  # not loaded yet
    cls = external_llm.ExternalLLMService  # triggers __getattr__
    assert cls.__name__ == "ExternalLLMService"
    assert "external_llm.service" in sys.modules  # now loaded
    assert "ExternalLLMService" in external_llm.__dict__  # cached for next time


def test_from_import_works() -> None:
    """``from external_llm import ExternalLLMService`` still works via __getattr__."""
    _purge_external_llm_tree()
    from external_llm import ExternalLLMService

    assert ExternalLLMService.__name__ == "ExternalLLMService"


def test_patch_compatibility() -> None:
    """``unittest.mock.patch("external_llm.ExternalLLMService")`` resolves the
    attribute via getattr (lazy load) and restores it via setattr."""
    _purge_external_llm_tree()
    import external_llm

    with mock.patch("external_llm.ExternalLLMService") as patched:
        assert external_llm.ExternalLLMService is patched
    # restored to the real class after the context exits
    assert external_llm.ExternalLLMService.__name__ == "ExternalLLMService"


def test_unknown_attribute_raises_attribute_error() -> None:
    """A name not in ``_LAZY_EXPORTS`` must raise AttributeError, not silently
    import something."""
    _purge_external_llm_tree()
    import external_llm

    try:
        external_llm.DefinitelyNotExported  # noqa: B018
    except AttributeError as exc:
        assert "DefinitelyNotExported" in str(exc)
    else:
        raise AssertionError("expected AttributeError for unknown attribute")


def test_dir_lists_lazy_exports() -> None:
    """``dir(external_llm)`` exposes the lazily-exported public name."""
    _purge_external_llm_tree()
    import external_llm

    assert "ExternalLLMService" in dir(external_llm)
