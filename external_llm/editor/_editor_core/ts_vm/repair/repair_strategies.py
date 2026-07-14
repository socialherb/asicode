"""repair_strategies.py — Deterministic repair strategies.

Each strategy:
- Takes (code, error, module) → List[PrimitiveOp] or None
- Is fully deterministic (no LLM)
- Returns primitive ops that the VM can execute

Strategies are dispatched by FailureType via the RepairRegistry.
"""
from __future__ import annotations

from typing import Optional

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveKind, PrimitiveOp
from external_llm.editor._editor_core.ts_vm.repair.failure_classifier import TSFailureClassifier
from external_llm.editor.semantic.ts_ir_models import TSModule

_classifier = TSFailureClassifier()
_IMPORT_MAP = {'useState': ('react', True), 'useEffect': ('react', True), 'useContext': ('react', True), 'useReducer': ('react', True), 'useCallback': ('react', True), 'useMemo': ('react', True), 'useRef': ('react', True), 'useLayoutEffect': ('react', True), 'React': ('react', False), 'Component': ('react', True), 'Fragment': ('react', True), 'createContext': ('react', True), 'forwardRef': ('react', True), 'memo': ('react', True), 'Suspense': ('react', True), 'createRoot': ('react-dom/client', True), 'render': ('react-dom', True), 'useRouter': ('next/router', True), 'usePathname': ('next/navigation', True), 'useSearchParams': ('next/navigation', True), 'NextPage': ('next', True), 'GetServerSideProps': ('next', True), 'axios': ('axios', False), 'express': ('express', False), 'Router': ('express', True), 'Request': ('express', True), 'Response': ('express', True)}

def repair_unknown_symbol(code: str, error: VerifyError, module: TSModule) -> Optional[list[PrimitiveOp]]:
    """Try to add a missing import for an unknown symbol."""
    symbol = _classifier.extract_symbol(error)
    if not symbol:
        return None
    for imp in module.imports:
        if symbol in imp.specifiers or imp.default_name == symbol:
            return None
    entry = _IMPORT_MAP.get(symbol)
    if not entry:
        return None
    source, is_named = entry
    if is_named:
        stmt = f"import {{ {symbol} }} from '{source}'"
    else:
        stmt = f"import {symbol} from '{source}'"
    return [PrimitiveOp(kind=PrimitiveKind.INSERT_IMPORT, payload={'statement': stmt})]

def repair_syntax_error(code: str, error: VerifyError, module: TSModule) -> Optional[list[PrimitiveOp]]:
    """Fix common syntax errors (missing semicolons, unclosed braces)."""
    if error.line is None:
        return None
    msg = error.message.lower()
    lines = code.split('\n')
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    if "expected ';'" in msg or 'missing semicolon' in msg:
        line = lines[idx].rstrip()
        if not line.endswith(';') and (not line.endswith('{')) and (not line.endswith('}')):
            lines[idx] = line + ';'
            return _code_replace_ops(code, '\n'.join(lines))
    for char, close in [("'}'", '}'), ("')'", ')'), ("']'", ']')]:
        if char in msg:
            indent = _get_indent(lines[idx])
            lines.insert(idx + 1, indent + close)
            return _code_replace_ops(code, '\n'.join(lines))
    return None

def repair_argument_mismatch(code: str, error: VerifyError, module: TSModule) -> Optional[list[PrimitiveOp]]:
    expected = _classifier.extract_expected_args(error)
    if expected is None:
        return None
    if error.line is None:
        return None
    lines = code.split('\n')
    idx = error.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx]
    paren_idx = line.find('(')
    if paren_idx == -1:
        return None
    close_idx = line.find(')', paren_idx)
    if close_idx == -1:
        return None
    callee = ''
    i = paren_idx - 1
    while i >= 0 and (line[i].isalnum() or line[i] == '_'):
        callee = line[i] + callee
        i -= 1
    if not callee:
        return None
    if expected == 0:
        return [PrimitiveOp(kind=PrimitiveKind.UPDATE_CALL, payload={'callee': callee, 'new_args': ''})]
    return None

def repair_missing_return(code: str, error: VerifyError, module: TSModule) -> Optional[list[PrimitiveOp]]:
    """Add a return statement to a function missing one."""
    if error.line is None:
        return None
    for func in module.functions:
        if func.meta and func.meta.start_line <= error.line <= func.meta.end_line:
            return [PrimitiveOp(kind=PrimitiveKind.INSERT_STATEMENT, payload={'statement': 'return undefined', 'anchor': func.name, 'position': 'end'})]
    return None

def _get_indent(line: str) -> str:
    indent = ''
    for ch in line:
        if ch in (' ', '\t'):
            indent += ch
        else:
            break
    return indent

def _code_replace_ops(old_code: str, new_code: str) -> list[PrimitiveOp]:
    """Wrap a full code replacement as a primitive-compatible return.

    This is a special case: the repair already produced the fixed code.
    We use a sentinel op that the repair planner handles directly.
    """
    return [PrimitiveOp(kind=PrimitiveKind.INSERT_STATEMENT, payload={'__raw_code__': new_code})]
