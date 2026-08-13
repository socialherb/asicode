"""P21-3: ASTRewriter._load_ast refuses oversized targets.

Bug: the AST-rewrite fallback path read + ast.parse'd the WHOLE file with no
size guard — a multi-hundred-MB target OOM'd the worker. The P19-4 rewrite
guard (webapp) had no counterpart here.

Fix under test: stat-based 64 MiB refusal (same policy as P19-4) before any
read; normal files unchanged.
"""
from __future__ import annotations

import pytest

from external_llm.ast_rewrite import ASTRewriter


def test_load_ast_oversized_refused(tmp_path):
    big = tmp_path / "big.py"
    with open(big, "wb") as f:
        f.truncate(64 * 1024 * 1024 + 1)  # sparse — stat-only guard
    rw = ASTRewriter(str(tmp_path))
    with pytest.raises(ValueError, match="too large"):
        rw._load_ast("big.py")


def test_load_ast_normal_file(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    rw = ASTRewriter(str(tmp_path))
    src, tree = rw._load_ast("a.py")
    assert "def f" in src
    assert tree is not None


def test_load_ast_missing_file(tmp_path):
    rw = ASTRewriter(str(tmp_path))
    with pytest.raises(ValueError, match="cannot stat"):
        rw._load_ast("nope.py")
