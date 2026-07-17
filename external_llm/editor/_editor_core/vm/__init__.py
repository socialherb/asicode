"""vm — Language-agnostic Execution VM infrastructure.

Provides the full apply -> verify -> repair -> rollback cycle
for all supported languages.

Usage::

    from external_llm.editor._editor_core.vm import create_vm

    vm = create_vm("python")
    result = vm.execute(code, "file.py", [op1, op2])
    if result.success:
        final_code = result.code
"""
from __future__ import annotations

from external_llm.editor._editor_core.vm.models import VerifyError, VMResult
from external_llm.editor._editor_core.vm.vm import GenericExecutionVM

__all__ = [
    "GenericExecutionVM",
    "VerifyError",
    "VMResult",
    "create_vm",
]


def create_vm(language: str = "python", **kwargs) -> GenericExecutionVM:
    """Factory: create a VM configured for *language*.

    Supported languages: python, java, kotlin, go, typescript, javascript.

    Args:
        language: Target language identifier.
        **kwargs: Passed through to VM constructor.

    Returns:
        A fully configured GenericExecutionVM (or TSExecutionVM for TS/JS).
    """
    if language in ("typescript", "javascript"):
        from external_llm.editor._editor_core.ts_vm.execution_vm.vm import TSExecutionVM
        return TSExecutionVM(language=language, **kwargs)
    return GenericExecutionVM(language=language, **kwargs)
