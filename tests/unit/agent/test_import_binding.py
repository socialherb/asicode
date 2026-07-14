"""Tests for _is_import_binding_in_source — CJS require precision.

Regression for B2: the destructuring pattern
``const { x } = require('m')`` lacked the ``= require(`` tail anchor, so ANY
local destructuring (``const { cfg } = loadConfig()``) was misread as an
import binding. That suppressed RULE_D's missing-anchor redirect, risking
misplaced insertions (import-anchored insert_after_symbol redirected to EOF).
"""

from __future__ import annotations

import pytest

from external_llm.agent.auto_correction import _is_import_binding_in_source
from external_llm.languages.tree_sitter_utils import is_available as _ts_available


def test_cjs_destructure_require_detected() -> None:
    src = "const { WebSocket, Server } = require('ws');"
    assert _is_import_binding_in_source("WebSocket", src)
    assert _is_import_binding_in_source("Server", src)


def test_cjs_plain_require_detected() -> None:
    src = "const path = require('path');"
    assert _is_import_binding_in_source("path", src)


def test_local_destructure_not_import() -> None:
    """The B2 regression: a non-require destructuring must NOT be an import."""
    src = "const { cfg } = loadConfig();"
    assert not _is_import_binding_in_source("cfg", src)


def test_local_assign_not_import() -> None:
    """Plain local assignment is also not an import binding."""
    src = "const cfg = loadConfig();"
    assert not _is_import_binding_in_source("cfg", src)


def test_esm_default_import_detected() -> None:
    src = "import WebSocket from 'ws';"
    assert _is_import_binding_in_source("WebSocket", src)


def test_unrelated_symbol_not_import() -> None:
    src = "const { WebSocket } = require('ws');"
    assert not _is_import_binding_in_source("NotImported", src)


@pytest.mark.skipif(not _ts_available(), reason="tree-sitter not installed")
def test_import_like_text_in_comment_not_detected() -> None:
    """Phase 3 ESM regex would false-positive on import-like text inside a
    comment; tree-sitter (Phase 1) is authoritative and must win, so Phase 3
    must be SKIPPED when tree-sitter ran (the ``_ts_ran`` gate).

    The named-import regex ``import\\s*\\{...\\bWebSocket\\b...\\}`` matches the
    substring ``import { WebSocket }`` inside the comment, so without the gate
    this returns True. With tree-sitter authoritative it correctly returns False.
    """
    src = "// see: import { WebSocket } from 'ws' for the API\nconst real = 1\n"
    assert _is_import_binding_in_source("WebSocket", src) is False


@pytest.mark.skipif(not _ts_available(), reason="tree-sitter not installed")
def test_real_esm_import_still_detected_under_treesitter() -> None:
    """Coverage guard: gating Phase 3 must not lose real ESM detection —
    tree-sitter Phase 1 covers every ESM form (here, a namespace import)."""
    src = "import * as WebSocket from 'ws'\n"
    assert _is_import_binding_in_source("WebSocket", src) is True
