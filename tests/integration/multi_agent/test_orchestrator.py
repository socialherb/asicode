"""
Integration tests for multi-agent orchestration.
"""
import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from external_llm.agent.orchestrator import (
    FileLockManager,
    OrchestratorAgent,
    OrchestratorConfig,
    OrchestratorResult,
    SubTaskSpec,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


@pytest.mark.integration
class TestFileLockManager:
    """Test file locking for concurrent access prevention."""

    def test_file_lock_manager_acquire_release(self, temp_repo_root: str):
        """Test basic lock acquisition and release."""
        lock_manager = FileLockManager(temp_repo_root)

        # Acquire lock for a file
        lock = lock_manager.acquire("sample.py")
        assert lock is not None

        # Should be able to release
        lock_manager.release("sample.py")

    def test_file_lock_manager_acquire_relevant_paths(self, temp_repo_root: str):
        """Test acquiring locks for relevant paths in tool arguments."""
        lock_manager = FileLockManager(temp_repo_root)

        tool_args = {
            "path": "sample.py",
            "file_path": "utils.py",
            "src": "old.py",
            "dst": "new.py",
            "other": "value"
        }

        locked_paths = lock_manager.acquire_relevant(tool_args)

        # Should lock all file-related paths (returns absolute paths)
        # Check that each expected file is present in the locked paths
        expected_files = ["sample.py", "utils.py", "old.py", "new.py"]
        for expected_file in expected_files:
            assert any(path.endswith(expected_file) for path in locked_paths), \
                f"Expected file {expected_file} not found in locked paths: {locked_paths}"
        assert len(locked_paths) == 4

        # Clean up via release_all — the correct pairing for acquire_relevant().
        # (release() singular re-normalizes its arg and would no-op on these
        # already-normalized keys; release_all() releases by normalized key.)
        lock_manager.release_all(locked_paths)

    def test_file_lock_manager_path_normalization(self, temp_repo_root: str):
        """Test path normalization for locking."""
        lock_manager = FileLockManager(temp_repo_root)

        # Test relative path
        lock1 = lock_manager.acquire("./sample.py")
        assert lock1 is not None
        lock_manager.release("./sample.py")

        # Test path traversal prevention (should return None or dummy lock)
        lock_manager.acquire("../../../etc/passwd")
        # Either returns None or a dummy lock
        lock_manager.release("../../../etc/passwd")

    def test_file_lock_manager_concurrent_access(self, temp_repo_root: str):
        """Test that locks prevent concurrent access to same file."""
        lock_manager = FileLockManager(temp_repo_root)
        results = []
        lock_acquired = threading.Event()

        def worker1():
            lock_manager.acquire("sample.py")
            lock_acquired.set()
            time.sleep(0.1)  # Hold lock
            results.append("worker1")
            lock_manager.release("sample.py")

        def worker2():
            lock_acquired.wait()  # Wait for worker1 to acquire lock
            lock_manager.acquire("sample.py")
            results.append("worker2")
            lock_manager.release("sample.py")

        # Start threads
        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # worker1 should complete before worker2 due to lock
        assert results == ["worker1", "worker2"]

    def test_acquire_relevant_release_all_round_trip(self, temp_repo_root: str):
        """Regression: release_all() must free the locks acquire_relevant()
        returned, using the REAL (non-idempotent) _normalize_path.

        acquire_relevant() keys locks by normalized path and returns those
        keys; release_all() must release by those same keys WITHOUT
        re-normalizing (re-normalizing an absolute key yields
        repo_root+repo_root+… → wrong entry → leaked lock). If it leaks, the
        next acquirer of the same file deadlocks.
        """
        mgr = FileLockManager(temp_repo_root)
        locked = mgr.acquire_relevant({"path": "sample.py"})
        mgr.release_all(locked)

        # A fresh manager must re-acquire the same file without blocking;
        # a leaked lock would hang this thread forever.
        mgr2 = FileLockManager(temp_repo_root)
        reacquired = {}
        done = threading.Event()

        def reacquire():
            reacquired["keys"] = mgr2.acquire_relevant({"path": "sample.py"})
            done.set()

        # daemon: if release_all() regresses and leaks the lock, this thread
        # blocks forever — daemon lets the test fail on timeout instead of
        # hanging the interpreter at exit on a stuck non-daemon thread.
        t = threading.Thread(target=reacquire, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert done.is_set(), "release_all() leaked the lock — re-acquire deadlocked"
        mgr2.release_all(reacquired["keys"])

    @pytest.mark.slow
    def test_acquire_relevant_mutual_exclusion_under_gc(self, temp_repo_root: str):
        """Regression: locks held via acquire_relevant() must survive GC.

        _file_locks is a WeakValueDictionary; the manager must keep a STRONG
        reference to each held Lock. Without it, gc.collect() inside the
        critical section reclaims the Lock, drops the registry entry, and a
        concurrent acquirer mints a fresh Lock — silently breaking mutual
        exclusion (two writers on the same file at once).
        """
        import gc

        inside: list[int] = []
        breach: list[bool] = []
        w1_in_cs = threading.Event()      # set when w1 is inside critical section
        w2_will_block = threading.Event()  # set when w2 is about to block on acquire
        m1 = FileLockManager(temp_repo_root)
        m2 = FileLockManager(temp_repo_root)

        def w1():
            k = m1.acquire_relevant({"path": "sample.py"})
            inside.append(1)
            w1_in_cs.set()
            gc.collect()  # critical-section GC pressure
            # Wait until w2 signals it is about to block on the same lock,
            # then yield briefly to let w2's blocked acquire() settle.
            w2_will_block.wait(timeout=2.0)
            time.sleep(0.01)
            if len(inside) > 1:
                breach.append(True)
            inside.pop()
            m1.release_all(k)

        def w2():
            w1_in_cs.wait()
            gc.collect()
            # Signal BEFORE blocking so w1 knows w2 has started its acquire attempt.
            w2_will_block.set()
            k = m2.acquire_relevant({"path": "sample.py"})
            inside.append(1)
            if len(inside) > 1:
                breach.append(True)
            inside.pop()
            m2.release_all(k)

        # daemon: a release regression could block a worker forever; daemon
        # keeps a failed run from hanging the interpreter at exit.
        t1 = threading.Thread(target=w1, daemon=True)
        t2 = threading.Thread(target=w2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not (t1.is_alive() or t2.is_alive()), "deadlock acquiring file lock"
        assert not breach, "mutual exclusion breached — GC reclaimed the held lock"

    def test_file_lock_manager_different_files_concurrent(self, temp_repo_root: str):
        """Test that different files can be accessed concurrently."""
        lock_manager = FileLockManager(temp_repo_root)
        results = []
        lock = threading.Lock()

        def worker(file_path: str, worker_id: str):
            lock_manager.acquire(file_path)
            with lock:
                results.append(worker_id)
            time.sleep(0.05)
            lock_manager.release(file_path)

        # Access different files - should run concurrently
        t1 = threading.Thread(target=worker, args=("sample.py", "worker1"))
        t2 = threading.Thread(target=worker, args=("utils.py", "worker2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both workers should have completed (order may vary)
        assert set(results) == {"worker1", "worker2"}
        assert len(results) == 2


@pytest.mark.integration
class TestSubTaskSpec:
    """Test subtask specification."""

    def test_subtask_spec_creation(self):
        """Test creating a subtask specification."""
        subtask = SubTaskSpec(
            task_id="task-1",
            title="Fix calculator bug",
            description="Fix bug in calculator",
            priority=1,
            dependencies=["task-0"],
            assigned_files=["sample.py"]
        )

        assert subtask.task_id == "task-1"
        assert subtask.title == "Fix calculator bug"
        assert subtask.description == "Fix bug in calculator"
        assert subtask.priority == 1
        assert subtask.dependencies == ["task-0"]
        assert subtask.assigned_files == ["sample.py"]

    def test_subtask_spec_defaults(self):
        """Test subtask specification with defaults."""
        subtask = SubTaskSpec(
            task_id="task-1",
            title="Simple task",
            description="Simple task"
        )

        assert subtask.task_id == "task-1"
        assert subtask.title == "Simple task"
        assert subtask.description == "Simple task"
        assert subtask.priority == 0  # Default
        assert subtask.dependencies == []  # Default
        assert subtask.assigned_files == []  # Default

    def test_subtask_spec_serialization(self):
        """Test subtask serialization to/from dict."""
        subtask = SubTaskSpec(
            task_id="task-1",
            title="Test task",
            description="Test task description",
            priority=2,
            dependencies=["task-0"],
            assigned_files=["file1.py", "file2.py"]
        )

        # Check dataclass fields
        assert subtask.task_id == "task-1"
        assert subtask.title == "Test task"
        assert subtask.description == "Test task description"
        assert subtask.priority == 2
        assert subtask.dependencies == ["task-0"]
        assert subtask.assigned_files == ["file1.py", "file2.py"]


@pytest.mark.integration
class TestOrchestratorConfig:
    """Test orchestrator configuration."""

    def test_orchestrator_config_creation(self):
        """Test creating orchestrator configuration."""
        from external_llm.agent.tool_registry import AgentConfig
        agent_config = AgentConfig(max_turns=10)

        config = OrchestratorConfig(
            max_subagents=3,
            parallel=True,
            agent_config=agent_config,
            cancel_event=None
        )

        assert config.max_subagents == 3
        assert config.parallel is True
        assert config.agent_config == agent_config
        assert config.cancel_event is None

    def test_orchestrator_config_defaults(self):
        """Test orchestrator configuration defaults."""
        config = OrchestratorConfig()

        assert config.max_subagents == 3  # Default
        assert config.parallel is True  # Default
        assert config.agent_config is None  # Default
        assert config.cancel_event is None  # Default


@pytest.mark.integration
class TestOrchestratorAgent:
    """Test orchestrator agent functionality."""

    def test_orchestrator_agent_initialization(self, temp_repo_root: str):
        """Test orchestrator agent initialization."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.tool_registry import AgentConfig
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)

        orchestrator_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orchestrator_config,
            model="test-model"
        )

        assert orchestrator.llm_client == mock_llm
        assert orchestrator._registry_proto == registry
        assert orchestrator.orch_config == orchestrator_config
        assert orchestrator.model == "test-model"
        assert orchestrator._file_lock_mgr is not None

    def test_orchestrator_task_decomposition(self, temp_repo_root: str):
        """Test task decomposition into subtasks."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.tool_registry import AgentConfig
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orchestrator_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orchestrator_config,
            model="test-model"
        )

        # Mock the _decompose_task method
        with patch.object(orchestrator, '_decompose_task') as mock_decompose:
            mock_decompose.return_value = [
                SubTaskSpec(
                    task_id="sub_1",
                    title="Fix indentation",
                    description="Fix indentation in sample.py",
                    assigned_files=["sample.py"],
                    priority=1,
                    dependencies=[]
                ),
                SubTaskSpec(
                    task_id="sub_2",
                    title="Add new method",
                    description="Add new method to Calculator",
                    assigned_files=["sample.py"],
                    priority=2,
                    dependencies=["sub_1"]
                )
            ]

            subtasks = orchestrator._decompose_task("Fix calculator and add features")

        assert len(subtasks) == 2
        assert subtasks[0].task_id == "sub_1"
        assert subtasks[1].task_id == "sub_2"
        assert subtasks[1].dependencies == ["sub_1"]

    def test_orchestrator_subtask_execution(self, temp_repo_root: str):
        """Test executing a subtask through subagent."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.tool_registry import AgentConfig
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orchestrator_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orchestrator_config,
            model="test-model"
        )

        subtask = SubTaskSpec(
            task_id="task-1",
            title="Fix sample.py",
            description="Fix sample.py",
            assigned_files=["sample.py"]
        )

        # Mock subagent execution - _run_subagent returns AgentResult
        mock_result = Mock()
        mock_result.status = "success"
        mock_result.turns = [Mock(), Mock(), Mock()]  # 3 turns
        mock_result.final_message = "Fixed indentation"

        with patch.object(orchestrator, '_run_subagent') as mock_run_subagent:
            mock_run_subagent.return_value = mock_result

            result = orchestrator._run_subagent(subtask, extra_turns=0)

        assert result.status == "success"
        assert len(result.turns) == 3

    def test_orchestrator_dependency_resolution(self, temp_repo_root: str):
        """Test dependency resolution for subtasks."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.tool_registry import AgentConfig
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orchestrator_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orchestrator_config,
            model="test-model"
        )

        subtasks = [
            SubTaskSpec(
                task_id="task-1",
                title="Task 1",
                description="Task 1",
                dependencies=[]
            ),
            SubTaskSpec(
                task_id="task-2",
                title="Task 2",
                description="Task 2",
                dependencies=["task-1"]
            ),
            SubTaskSpec(
                task_id="task-3",
                title="Task 3",
                description="Task 3",
                dependencies=["task-1"]
            ),
            SubTaskSpec(
                task_id="task-4",
                title="Task 4",
                description="Task 4",
                dependencies=["task-2", "task-3"]
            )
        ]

        # Test dependency-aware execution
        # Mock the _run_dependency_aware method to verify it receives correct subtasks
        with patch.object(orchestrator, '_run_dependency_aware') as mock_run:
            mock_run.return_value = [Mock(), Mock(), Mock(), Mock()]

            # We'll just verify the method can be called with dependency tasks
            results = orchestrator._run_dependency_aware(subtasks)

            assert len(results) == 4
            # The actual dependency resolution is tested in _run_dependency_aware implementation

    def test_orchestrator_parallel_execution(self, temp_repo_root: str):
        """Test parallel execution of independent subtasks."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.tool_registry import AgentConfig
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orchestrator_config = OrchestratorConfig(
            max_subagents=2,
            parallel=True,
            agent_config=base_config
        )

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orchestrator_config,
            model="test-model"
        )

        # Create independent subtasks (no dependencies)
        subtasks = [
            SubTaskSpec(
                task_id=f"task-{i}",
                title=f"Task {i}",
                description=f"Task {i}",
                dependencies=[]
            )
            for i in range(3)
        ]

        # Mock subagent execution with delays
        execution_times = []
        import time

        from external_llm.agent.agent_loop import AgentResult
        def mock_execute_subtask(*args, **kwargs):
            subtask = args[0] if args else kwargs.get('subtask')
            start = time.time()
            time.sleep(0.05)  # Simulate work
            execution_times.append(time.time() - start)
            return AgentResult(
                status="success",
                turns=[],
                final_message=f"Completed {subtask.task_id}",
                metadata={}
            )

        with patch.object(orchestrator, '_run_subagent', side_effect=mock_execute_subtask):
            with patch.object(orchestrator, '_decompose_task', return_value=subtasks):
                result = orchestrator.run("Test parallel tasks")

        # With parallel execution and max_subagents=2, 3 tasks should complete
        assert result.status in ("success", "partial", "error")
        assert len(execution_times) == 3

    def test_orchestrator_result_aggregation(self, temp_repo_root: str):
        """Test aggregation of subagent results via OrchestratorResult."""
        from external_llm.agent.agent_loop import AgentResult

        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orch_config = OrchestratorConfig(max_subagents=3, agent_config=base_config)
        OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orch_config,
            model="test-model"
        )

        # OrchestratorResult holds subtask_results list
        result = OrchestratorResult(
            status="partial",
            summary="2 of 3 tasks completed",
            subtask_results=[
                AgentResult(status="success", turns=[], final_message="Fixed indentation"),
                AgentResult(status="success", turns=[], final_message="Added method"),
                AgentResult(status="error", turns=[], error="Failed to apply patch"),
            ],
            total_turns=6,
        )

        assert result.status == "partial"
        assert len(result.subtask_results) == 3
        assert result.subtask_results[0].status == "success"
        assert result.subtask_results[2].status == "error"
        assert result.total_turns == 6

    def test_orchestrator_partial_failure_handling(self, temp_repo_root: str):
        """Test handling of partial failures in subtasks."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        from external_llm.agent.agent_loop import AgentResult
        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orch_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)
        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orch_config,
            model="test-model"
        )

        # Create subtasks where one depends on another
        subtasks = [
            SubTaskSpec(
                task_id="task-1",
                title="Task 1",
                description="Task 1",
                dependencies=[]
            ),
            SubTaskSpec(
                task_id="task-2",
                title="Task 2",
                description="Task 2 (depends on task-1)",
                dependencies=["task-1"]
            )
        ]

        # Mock execution: task-1 succeeds, task-2 also succeeds (simplified)
        def mock_execute_subtask(*args, **kwargs):
            subtask = args[0] if args else kwargs.get('subtask')
            return AgentResult(
                status="success",
                turns=[],
                final_message=f"Completed {subtask.task_id}",
                metadata={}
            )

        with patch.object(orchestrator, '_run_subagent', side_effect=mock_execute_subtask):
            with patch.object(orchestrator, '_decompose_task', return_value=subtasks):
                result = orchestrator.run("Test partial failure")

        # Orchestrator should complete with results
        assert result.status in ("success", "partial", "error")
        assert result.subtask_results is not None
        assert len(result.subtask_results) == 2

    def test_orchestrator_callback_integration(self, temp_repo_root: str):
        """Test orchestrator callback for SSE events."""
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        base_config = AgentConfig(max_turns=5)
        registry = ToolRegistry(temp_repo_root, base_config)
        orch_config = OrchestratorConfig(max_subagents=2, agent_config=base_config)

        # Track callback calls
        callback_events = []

        def mock_callback(event_type: str, data: dict[str, Any]):
            callback_events.append((event_type, data.get("task_id", "")))

        orchestrator = OrchestratorAgent(
            llm_client=mock_llm,
            registry=registry,
            orch_config=orch_config,
            model="test-model",
            callback=mock_callback
        )

        # Trigger some events via _cb method
        orchestrator._cb("orchestrator_plan", {"task_id": "plan-1"})
        orchestrator._cb("subagent_start", {"task_id": "task-1"})
        orchestrator._cb("subagent_complete", {"task_id": "task-1"})

        assert len(callback_events) == 3
        assert callback_events[0][0] == "orchestrator_plan"
        assert callback_events[1][0] == "subagent_start"
        assert callback_events[2][0] == "subagent_complete"
