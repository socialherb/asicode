"""
Hybrid Output Parser

Parses and validates all output modes
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .output_modes import OutputMode

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Parse result"""

    success: bool
    mode: Optional[OutputMode] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    # Per-mode results
    blocks: Optional[list[dict]] = None  # ASICODE_BLOCK
    diff: Optional[str] = None  # UNIFIED_DIFF
    code: Optional[str] = None  # TARGETED_BLOCK
    insert_point: Optional[str] = None
    content: Optional[str] = None  # FULL_FILE
    file_path: Optional[str] = None
    plan: Optional[dict] = None  # PLAN_JSON

    raw_output: Optional[str] = None


class HybridOutputParser:
    """Hybrid output parser"""

    def parse(self, llm_output: str, expected_mode: OutputMode) -> ParseResult:
        """Parse LLM output"""

        logger.info(f"Parsing (expected={expected_mode.value})")

        #NEEDS_DISAMBIGUATION check
        if "NEEDS_DISAMBIGUATION" in llm_output:
            return ParseResult(
                success=True,
                mode=None,
                raw_output=llm_output
            )

        # Parse in expected mode
        parsers = {
            OutputMode.ASICODE_BLOCK: self._parse_asicode,
            OutputMode.UNIFIED_DIFF: self._parse_diff,
            OutputMode.TARGETED_BLOCK: self._parse_targeted,
            OutputMode.FULL_FILE: self._parse_full_file,
            OutputMode.PLAN_JSON: self._parse_plan,
        }

        result = parsers[expected_mode](llm_output)
        if result.success:
            return result

        # Fallback
        for mode in [OutputMode.UNIFIED_DIFF, OutputMode.FULL_FILE]:
            if mode != expected_mode:
                result = parsers[mode](llm_output)
                if result.success:
                    result.warnings.append(f"Parsed as {mode.value} instead")
                    return result

        return ParseResult(
            success=False,
            error="Failed to parse",
            raw_output=llm_output
        )

    def _parse_asicode(self, text: str) -> ParseResult:
        """Parse ASICODE block"""
        pattern = r'ASICODE_BEGIN\s*\n(.*?)\nASICODE_END'
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return ParseResult(success=False, error="No ASICODE blocks")

        blocks = []
        for match in matches:
            before = re.search(r'BEFORE\s*\n(.*?)\nAFTER', match, re.DOTALL)
            after = re.search(r'AFTER\s*\n(.*?)$', match, re.DOTALL)

            if before and after:
                blocks.append({
                    "before": before.group(1).strip(),
                    "after": after.group(1).strip()
                })

        if not blocks:
            return ParseResult(success=False, error="ASICODE blocks missing BEFORE/AFTER")

        return ParseResult(
            success=True,
            mode=OutputMode.ASICODE_BLOCK,
            blocks=blocks,
            raw_output=text
        )

    def _parse_diff(self, text: str) -> ParseResult:
        """Parse unified diff (strict validation: header + @@ hunk required)"""
        pattern = r'```diff\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            diff = matches[0].strip()
        else:
            diff = text.strip()

        if not diff:
            return ParseResult(success=False, error="No diff found")

        # Minimum: must have --- a/ +++ b/ + @@ to qualify as a real diff
        has_git_header = diff.startswith("diff --git ")
        has_file_headers = ("--- a/" in diff) and ("+++ b/" in diff)
        has_hunk = ("@@ " in diff)

        # Some models write explanatory text starting with "diff --git ...", so require a hunk too
        if (has_git_header or has_file_headers) and has_hunk:
            return ParseResult(
                success=True,
                mode=OutputMode.UNIFIED_DIFF,
                diff=diff,
                raw_output=text
            )

        return ParseResult(success=False, error="No valid unified diff found")

    def _parse_targeted(self, text: str) -> ParseResult:
        """Parse TARGETED_BLOCK"""
        func_match = re.search(r'FUNCTION:\s*(\w+)', text)
        if not func_match:
            return ParseResult(success=False, error="No FUNCTION marker")

        insert_match = re.search(r'INSERT_AFTER:\s*(.+)', text)
        if not insert_match:
            return ParseResult(success=False, error="No INSERT_AFTER")

        code_pattern = r'```python\s*\n(.*?)\n```'
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        if not code_matches:
            return ParseResult(success=False, error="No code block")

        return ParseResult(
            success=True,
            mode=OutputMode.TARGETED_BLOCK,
            code=code_matches[0],
            insert_point=insert_match.group(1).strip(),
            raw_output=text
        )

    def _parse_full_file(self, text: str) -> ParseResult:
        """Parse FULL_FILE - supports fenced or unfenced FILE blocks"""
        # Uses the same rules as EnhancedOutputParser.FILE_BLOCK_RE
        # Allows fenced code or unfenced body after FILE: path
        pattern = r'(?ims)(?:^|\n)\s*(?:FILE|Path|Target file)\s*:\s*(?P<path>[^\n\r]+?)\s*\r?\n(?:```[^\n\r]*\r?\n(?P<code1>[\s\S]*?)\r?\n```|(?P<code2>(?:(?!^\s*(?:FILE|Path|Target file)\s*:).*\r?\n)*))?'

        match = re.search(pattern, text)
        if not match:
            return ParseResult(success=False, error="No FILE marker")

        path = match.group("path").strip()
        code = match.group("code1")
        if code is None:
            code = match.group("code2")
        if code is None:
            code = ""

        code = str(code).replace("\r\n", "\n").rstrip("\n") + "\n"

        if not code.strip():
            return ParseResult(success=False, error="No code content")

        return ParseResult(
            success=True,
            mode=OutputMode.FULL_FILE,
            file_path=path,
            content=code,
            raw_output=text
        )

    def _parse_plan(self, text: str) -> ParseResult:
        """Parse PLAN_JSON"""
        json_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if not matches:
            return ParseResult(success=False, error="No JSON block")

        try:
            plan = json.loads(matches[0])
        except json.JSONDecodeError as e:
            return ParseResult(success=False, error=f"Invalid JSON: {e}")

        if "operations" not in plan:
            return ParseResult(success=False, error="Missing operations")

        return ParseResult(
            success=True,
            mode=OutputMode.PLAN_JSON,
            plan=plan,
            raw_output=text
        )
