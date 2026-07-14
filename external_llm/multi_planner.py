"""
Multi-file Planner for asicode

Takes complex requests like "create login functionality" and:
1. Breaks down into multiple file operations
2. Determines file dependencies and order
3. Generates a step-by-step plan
4. Provides context for each file operation

Works with ExternalLLMService to handle multi-file changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agent.config.thresholds import config as _cfg
from .client import effective_content
from .project_analyzer import ProjectAnalyzer, ProjectStructure
from .smart_analyzer import RequestAnalysis, SmartRequestAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class FileOperation:
    """Single file operation in a plan"""

    # File path (relative to repo root)
    file_path: str

    # Operation type
    operation: str  # "create", "modify", "delete"

    # Description of what to do
    description: str

    # Dependencies (files that must be created/modified first)
    dependencies: list[str] = field(default_factory=list)

    # Priority (lower = higher priority)
    priority: int = 0

    # Template/example to follow
    template_file: Optional[str] = None

    # Specific instructions for this file
    instructions: str = ""


@dataclass
class ExecutionPlan:
    """Complete execution plan for a request"""

    # Original request
    original_request: str

    # File operations in execution order
    operations: list[FileOperation] = field(default_factory=list)

    # Overall strategy
    strategy: str = "sequential"  # "sequential", "parallel"

    # Estimated complexity
    complexity: str = "simple"  # "simple", "moderate", "complex"

    # Success criteria
    success_criteria: list[str] = field(default_factory=list)

    # Warnings/notes
    warnings: list[str] = field(default_factory=list)


class MultiFilePlanner:
    """
    Plans multi-file operations for complex requests

    Example:
        Request: "create login functionality"
        Plan:
            1. Create models/user.py (user model)
            2. Create schemas/login.py (request/response schemas)
            3. Create routers/auth.py (login endpoint)
            4. Modify main.py (register router)
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.analyzer = SmartRequestAnalyzer(repo_root)
        self.project_analyzer = ProjectAnalyzer(repo_root)

    def create_plan(self, user_request: str) -> ExecutionPlan:
        """
        Create execution plan for a request

        Args:
            user_request: User's request

        Returns:
            ExecutionPlan with ordered operations
        """
        # Analyze request
        analysis = self.analyzer.analyze(user_request)

        # Analyze project structure
        structure = self.project_analyzer.analyze()

        # Create plan based on analysis
        if analysis.needs_planning:
            plan = self._create_complex_plan(analysis, structure)
        else:
            plan = self._create_simple_plan(analysis, structure)

        # Order operations by dependencies
        plan.operations = self._order_operations(plan.operations)

        # Set complexity
        plan.complexity = self._assess_complexity(plan.operations)

        return plan

    def _create_simple_plan(
        self,
        analysis: RequestAnalysis,
        structure: ProjectStructure
    ) -> ExecutionPlan:
        """Create plan for simple requests (single file)"""

        plan = ExecutionPlan(original_request=analysis.original_request)

        # If files are suggested, use them
        if analysis.suggested_files:
            for file_path in analysis.suggested_files[:1]:  # Just first file for simple
                op_type = analysis.file_operations.get(file_path, "modify")

                operation = FileOperation(
                    file_path=file_path,
                    operation=op_type,
                    description=f"{op_type.title()} {file_path} to {analysis.intent}",
                    priority=0,
                )

                plan.operations.append(operation)

        plan.strategy = "sequential"

        return plan

    def _create_complex_plan(
        self,
        analysis: RequestAnalysis,
        structure: ProjectStructure
    ) -> ExecutionPlan:
        """Create plan for complex requests (multiple files)"""

        plan = ExecutionPlan(
            original_request=analysis.original_request,
            strategy="sequential"
        )

        feature = analysis.feature_name

        # First, check if we have suggested files from analysis
        if analysis.suggested_files:
            # Convert suggested files to operations
            plan.operations.extend(self._suggested_files_to_operations(analysis, structure))
        else:
            # Use generic feature planning — Developer LLM handles
            # framework-specific details based on project context
            plan.operations.extend(
                self._plan_generic_feature(feature or "feature", structure)
            )

        # Add success criteria
        plan.success_criteria = self._define_success_criteria(analysis, plan.operations)

        return plan

    def _plan_generic_feature(self, feature: str, structure: ProjectStructure) -> list[FileOperation]:
        """Plan generic Python feature"""

        ops = []

        # Main module
        ops.append(FileOperation(
            file_path=f"{feature}.py",
            operation="create",
            description=f"Create {feature} module",
            priority=0,
            instructions=f"Implement {feature} functionality with proper structure",
        ))

        # Tests
        if structure.test_dir:
            ops.append(FileOperation(
                file_path=f"{structure.test_dir}/test_{feature}.py",
                operation="create",
                description=f"Create tests for {feature}",
                priority=1,
                dependencies=[f"{feature}.py"],
                instructions=f"Write unit tests for {feature} module",
            ))

        return ops

    def _suggested_files_to_operations(self, analysis: RequestAnalysis, structure: ProjectStructure) -> list[FileOperation]:
        """Convert suggested files from analysis to FileOperation objects"""
        ops = []

        for file_path in analysis.suggested_files:
            # Determine operation type from analysis or by file existence
            op_type = analysis.file_operations.get(file_path, "create")

            # Create appropriate description
            if analysis.feature_name:
                description = f"{op_type.title()} {file_path} for {analysis.feature_name} feature"
            else:
                description = f"{op_type.title()} {file_path} for {analysis.intent}"

            # Determine priority based on file type
            priority = 0
            if file_path.endswith('.css'):
                priority = 2  # CSS after HTML/JS
            elif file_path.endswith('.js') or file_path.endswith('.jsx'):
                priority = 1  # JS after HTML
            elif file_path.endswith('.html'):
                priority = 0  # HTML first

            # Add dependencies for CSS/JS files
            dependencies = []
            if file_path.endswith('.css') or file_path.endswith('.js'):
                # CSS/JS might depend on HTML template
                html_files = [f for f in analysis.suggested_files if f.endswith('.html')]
                if html_files:
                    dependencies.append(html_files[0])

            instructions = f"Implement changes in {file_path} for: {analysis.original_request[:100]}"

            op = FileOperation(
                file_path=file_path,
                operation=op_type,
                description=description,
                dependencies=dependencies,
                priority=priority,
                instructions=instructions
            )
            ops.append(op)

        return ops

    def _order_operations(self, operations: list[FileOperation]) -> list[FileOperation]:
        """Order operations by dependencies and priority"""

        if not operations:
            return []

        # Sort by priority first
        operations.sort(key=lambda op: op.priority)

        # Then reorder based on dependencies (simple topological sort)
        ordered = []
        remaining = operations.copy()
        max_iterations = len(operations) * 2
        iterations = 0

        while remaining and iterations < max_iterations:
            iterations += 1
            made_progress = False

            for op in remaining[:]:
                # Defensive: FileOperation.dependencies may be None if a caller
                # (or a null YAML field) bypassed parse-time coercion; guard the
                # iteration so a None never raises TypeError here.
                deps = op.dependencies or ()
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    any(done.file_path == dep for done in ordered)
                    for dep in deps
                )

                if deps_satisfied or not deps:
                    ordered.append(op)
                    remaining.remove(op)
                    made_progress = True

            if not made_progress:
                # Circular dependency or unsatisfiable
                # Just add remaining in order
                ordered.extend(remaining)
                break

        return ordered

    def _assess_complexity(self, operations: list[FileOperation]) -> str:
        """Assess plan complexity"""

        num_ops = len(operations)

        if num_ops <= 1:
            return "simple"
        elif num_ops <= 3:
            return "moderate"
        else:
            return "complex"

    def _define_success_criteria(
        self,
        analysis: RequestAnalysis,
        operations: list[FileOperation]
    ) -> list[str]:
        """Define success criteria for the plan"""

        criteria = []

        # All files created/modified
        criteria.append(f"All {len(operations)} files created/modified successfully")

        # No syntax errors
        criteria.append("No syntax errors in generated code")

        # Proper imports
        criteria.append("All necessary imports included")

        return criteria

    def get_plan_summary(self, plan: ExecutionPlan) -> str:
        """Get human-readable plan summary"""

        lines = ["# Execution Plan"]
        lines.append("")
        lines.append(f"**Request**: {plan.original_request}")
        lines.append(f"**Complexity**: {plan.complexity}")
        lines.append(f"**Strategy**: {plan.strategy}")
        lines.append("")

        if plan.operations:
            lines.append("## Operations")
            lines.append("")
            for i, op in enumerate(plan.operations, 1):
                lines.append(f"### {i}. {op.operation.title()} `{op.file_path}`")
                lines.append(f"   **Description**: {op.description}")
                if op.dependencies:
                    lines.append(f"   **Dependencies**: {', '.join(op.dependencies)}")
                if op.instructions:
                    lines.append(f"   **Instructions**: {op.instructions}")
                lines.append("")

        if plan.success_criteria:
            lines.append("## Success Criteria")
            lines.append("")
            for criterion in plan.success_criteria:
                lines.append(f"- {criterion}")
            lines.append("")

        if plan.warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in plan.warnings:
                lines.append(f"⚠️  {warning}")
            lines.append("")

        return "\n".join(lines)


class LLMEnhancedMultiFilePlanner(MultiFilePlanner):
    """
    Enhanced MultiFilePlanner that uses LLM for plan generation

    Extends MultiFilePlanner to use LLM for creating execution plans
    while leveraging project analysis for context.
    """

    def __init__(
        self,
        repo_root: str,
        llm_client=None,
        llm_model: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize LLM-enhanced planner

        Args:
            repo_root: Repository root path
            llm_client: Optional LLMClient instance (if None, uses rule-based only)
            llm_model: Model name to use for LLM planning
            temperature: LLM temperature for planning
        """
        super().__init__(repo_root)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.temperature = temperature

    def create_plan(self, user_request: str) -> ExecutionPlan:
        """
        Create execution plan using LLM if available, fallback to rule-based

        Args:
            user_request: User's request

        Returns:
            ExecutionPlan
        """
        # Analyze request and project (base class does this)
        analysis = self.analyzer.analyze(user_request)
        structure = self.project_analyzer.analyze()

        # Try LLM-based planning if client is available
        if self.llm_client and analysis.needs_planning:
            try:
                llm_plan = self._create_llm_based_plan(user_request, analysis, structure)
                if llm_plan:
                    logger.info(f"LLM-based plan created with {len(llm_plan.operations)} operations")
                    return llm_plan
                else:
                    logger.warning("LLM planning failed, falling back to rule-based")
            except Exception as e:
                logger.error(f"LLM planning error: {e}, falling back to rule-based")

        # Fallback to original rule-based planning
        return super().create_plan(user_request)

    def _create_llm_based_plan(
        self,
        user_request: str,
        analysis: RequestAnalysis,
        structure: ProjectStructure
    ) -> Optional[ExecutionPlan]:
        """
        Create execution plan using LLM

        Args:
            user_request: Original user request
            analysis: Request analysis
            structure: Project structure

        Returns:
            ExecutionPlan or None if failed
        """
        if not self.llm_client:
            return None

        # Build project context summary
        project_context = self._build_project_context_summary(structure)

        # Build LLM prompt for planning
        prompt = self._build_llm_planning_prompt(
            user_request, analysis, structure, project_context
        )

        try:
            # Import here to avoid circular dependencies
            from .client import LLMMessage

            # Prepare messages
            messages = [
                LLMMessage(
                    role="system",
                    content="You are an expert software architect who creates detailed execution plans for code changes."
                ),
                LLMMessage(role="user", content=prompt)
            ]

            # Call LLM
            response = self.llm_client.chat(
                messages=messages,
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=_cfg.tokens.SUBAGENT_SHORT,
            )

            # Parse response
            llm_content = effective_content(response)
            logger.debug(f"LLM planning response ({len(llm_content)} chars): {llm_content[:5000]}...")

            # Parse plan from LLM response
            parsed_plan = self._parse_llm_plan_response(llm_content, user_request)

            if parsed_plan:
                # Order operations by dependencies (important!)
                parsed_plan.operations = self._order_operations(parsed_plan.operations)
                parsed_plan.complexity = self._assess_complexity(parsed_plan.operations)
                return parsed_plan
            else:
                return None

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return None

    def _build_project_context_summary(self, structure: ProjectStructure) -> str:
        """Build concise project context summary for LLM planning"""
        lines = []

        if structure.frameworks:
            lines.append(f"- **Frameworks**: {', '.join(structure.frameworks)}")
        elif structure.framework:
            lines.append(f"- **Framework**: {structure.framework}")

        if structure.project_types:
            lines.append(f"- **Project Type**: {', '.join(structure.project_types)}")

        if structure.directories:
            lines.append("- **Directory Structure**:")
            for purpose, dirs in structure.directories.items():
                if purpose != 'other' and dirs:
                    lines.append(f"  - {purpose}: {', '.join(dirs)}")

        if structure.naming_style:
            lines.append(f"- **Naming Convention**: {structure.naming_style}")

        if structure.common_imports:
            lines.append(f"- **Common Imports**: {', '.join(structure.common_imports[:5])}")

        if structure.example_files:
            lines.append("- **Example Files**:")
            for file_type, path in list(structure.example_files.items())[:3]:
                lines.append(f"  - {file_type}: `{path}`")

        return "\n".join(lines) if lines else "No project context available."

    def _build_llm_planning_prompt(
        self,
        user_request: str,
        analysis: RequestAnalysis,
        structure: ProjectStructure,
        project_context: str
    ) -> str:
        """Build LLM prompt for hierarchical, step-by-step planning"""
        prompt = f"""# Architecture-Driven Implementation Plan

## Project Context
{project_context}

## User Request
{user_request}

## Request Analysis
- Intent: {analysis.intent}
- Feature: {analysis.feature_name or 'Not specified'}
- Tech stack: {', '.join(analysis.tech_stack) if analysis.tech_stack else 'Not detected'}

## Task: Think Step by Step
You are an expert software architect. Your task is to design and plan the implementation of this feature in a hierarchical, step-by-step manner.

### Step 1: Architecture Design
First, think about the overall architecture needed for this feature. Consider:
- What components/modules are required?
- How do they interact with each other?
- What is the data flow?
- What are the key design patterns to use?

### Step 2: Component Breakdown
Break down the architecture into concrete components. For each component:
- What is its responsibility?
- What files will it need?
- How does it depend on other components?

### Step 3: File Structure Design
Based on the components, design the file structure:
- Which files need to be created or modified?
- What should each file contain?
- What are the dependencies between files?

### Step 4: Implementation Sequence
Determine the implementation sequence:
- What should be implemented first? (foundational components)
- What depends on what?
- What can be implemented in parallel?

### Step 5: Detailed Plan
Create a detailed execution plan with all the files and their specific tasks.

## Output Format
Return your plan in the following YAML format:

```yaml
architecture:
  summary: "Brief architectural overview"
  components:
    - name: "Component name"
      responsibility: "What this component does"
      files: ["path/to/file1.py", "path/to/file2.js"]
      dependencies: ["other_component"]  # optional

plan:
  complexity: "simple|moderate|complex"
  strategy: "sequential|parallel"
  phases:
    - phase: 1
      description: "Phase description"
      operations:
        - file_path: "path/to/file.py"
          operation: "create|modify|delete"
          description: "Detailed description of what to implement in this file"
          dependencies: ["other_file.py"]  # files that must exist first
          instructions: "Specific implementation instructions for this file"
          priority: 1  # lower = higher priority

  success_criteria:
    - "All components implemented correctly"
    - "No syntax errors in generated code"
    - "Feature works as specified in user request"
```

## Important Guidelines
1. **Think hierarchically**: Start with high-level architecture, then drill down to details
2. **Consider existing patterns**: Follow the project's existing structure and conventions
3. **Be realistic**: Create an achievable plan with clear dependencies
4. **Be specific**: Provide concrete file paths and implementation details
5. **Consider the framework**: {structure.framework or 'Generic Python'} best practices

Now, think through the architecture step by step and generate the plan:"""

        return prompt

    def _parse_llm_plan_response(self, llm_content: str, original_request: str) -> Optional[ExecutionPlan]:
        """
        Parse LLM response to extract execution plan

        Args:
            llm_content: Raw LLM response
            original_request: Original user request

        Returns:
            ExecutionPlan if successful, None otherwise
        """
        import re

        import yaml

        try:
            # Try to extract YAML block from response
            yaml_pattern = r'```(?:yaml|yml)?\s*(.*?)```'
            matches = re.findall(yaml_pattern, llm_content, re.DOTALL)

            if not matches:
                # Try to find YAML-like content without backticks
                # Look for "plan:" or "operations:" patterns
                yaml_section = llm_content
                # Try to find from "plan:" to end
                plan_start = llm_content.find('plan:')
                if plan_start != -1:
                    yaml_section = llm_content[plan_start:]
                else:
                    # Last resort: use entire content
                    yaml_section = llm_content

                matches = [yaml_section]

            for yaml_text in matches:
                try:
                    # Clean up the YAML text
                    yaml_text = yaml_text.strip()

                    # Parse YAML
                    parsed = yaml.safe_load(yaml_text)

                    if not parsed or 'plan' not in parsed:
                        # Maybe the content is the plan directly
                        if isinstance(parsed, dict) and 'operations' in parsed:
                            plan_data = parsed
                        else:
                            continue
                    else:
                        plan_data = parsed['plan']

                    # Create ExecutionPlan - handle both old and new formats
                    operations = []

                    # Check if we have phases (new format) or direct operations (old format)
                    if 'phases' in plan_data:
                        # New format: phases contain operations
                        for phase in plan_data.get('phases', []):
                            for op_data in phase.get('operations', []):
                                # Log op_data for debugging
                                logger.debug(f"Parsing op_data from phase: {op_data}")
                                # Get file path, try 'file_path' first, then 'file'
                                file_path = op_data.get('file_path') or op_data.get('file') or ''
                                operation = FileOperation(
                                    file_path=file_path,
                                    operation=op_data.get('operation', 'modify'),
                                    description=op_data.get('description', ''),
                                    dependencies=op_data.get('dependencies') or [],
                                    priority=op_data.get('priority', 0),
                                    instructions=op_data.get('instructions', ''),
                                    template_file=op_data.get('template_file'),
                                )
                                operations.append(operation)
                    else:
                        # Old format: direct operations list
                        for op_data in plan_data.get('operations', []):
                            # Log op_data for debugging
                            logger.debug(f"Parsing op_data: {op_data}")
                            # Get file path, try 'file_path' first, then 'file'
                            file_path = op_data.get('file_path') or op_data.get('file') or ''
                            operation = FileOperation(
                                file_path=file_path,
                                operation=op_data.get('operation', 'modify'),
                                description=op_data.get('description', ''),
                                dependencies=op_data.get('dependencies') or [],
                                priority=op_data.get('priority', 0),
                                instructions=op_data.get('instructions', ''),
                                template_file=op_data.get('template_file'),
                            )
                            operations.append(operation)

                    # LLMs frequently emit bare keys (`dependencies:`,
                    # `success_criteria:`) that YAML parses to None. `.get(k, d)`
                    # returns None when the key exists-with-null, not the default,
                    # which then crashes downstream iteration (`for dep in None`).
                    # Coerce null -> empty collection so a single null field never
                    # discards an otherwise-valid plan.
                    plan = ExecutionPlan(
                        original_request=original_request,
                        operations=operations,
                        strategy=plan_data.get('strategy') or 'sequential',
                        complexity=plan_data.get('complexity') or 'moderate',
                        success_criteria=plan_data.get('success_criteria') or [],
                        warnings=plan_data.get('warnings') or [],
                    )

                    logger.info(f"Parsed LLM plan with {len(operations)} operations")
                    return plan

                except yaml.YAMLError as e:
                    logger.debug(f"YAML parsing failed for a section: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Failed to parse plan from YAML section: {e}")
                    continue

            # If we get here, no valid plan was found
            logger.warning("Could not extract valid plan from LLM response")
            return None

        except ImportError:
            logger.warning("PyYAML not available, cannot parse LLM plan response")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM plan response: {e}")
            return None
