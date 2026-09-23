"""
tests/test_canonical_state_and_events.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive tests for Phase 1: Canonical Task State + Event Model.

Covers Part O testing requirements:
1. Valid lifecycle transitions (created -> understanding -> planning -> ready -> executing -> verifying -> completed).
2. Invalid state transitions raise InvalidStateTransitionError.
3. Terminal states cannot transition to non-terminal states.
4. Appending events produces correct sequence ordering.
5. Restart recovery: reconstructing TaskState from stored events.
6. Replaying events produces identical TaskState (pure reducer).
7. Event store persistence across new store instances.
8. Concurrent event appends preserve monotonic ordering.
9. Secrets and tokens are redacted from events.
10. EventBus backwards compatibility with existing subscribers.
11. Integration with AgentRuntime.
"""

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ultron.core.agents.base import BaseAgent
from ultron.core.runtime import (
    AgentRuntime,
    EventBus,
    InMemoryEventStore,
    InvalidStateTransitionError,
    JsonlEventStore,
    SequenceViolationError,
    TaskEvent,
    TaskEventType,
    TaskLifecycleStatus,
    TaskState,
    assert_task_transition,
    project_task_state,
    sanitize_event_payload,
)
from ultron.core.types import (
    ChatMessage,
    Role,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# 1 & 2 & 3. Lifecycle Transitions & State Machine
# ---------------------------------------------------------------------------


def test_valid_lifecycle_transitions():
    task = TaskState(goal="Build microservice")
    assert task.lifecycle_status == TaskLifecycleStatus.CREATED
    assert task.status == TaskStatus.TASK_STARTED

    task.transition_to(TaskLifecycleStatus.UNDERSTANDING)
    assert task.lifecycle_status == TaskLifecycleStatus.UNDERSTANDING

    task.transition_to(TaskLifecycleStatus.PLANNING)
    assert task.lifecycle_status == TaskLifecycleStatus.PLANNING

    task.transition_to(TaskLifecycleStatus.READY)
    assert task.lifecycle_status == TaskLifecycleStatus.READY

    task.transition_to(TaskLifecycleStatus.EXECUTING)
    assert task.lifecycle_status == TaskLifecycleStatus.EXECUTING
    assert task.status == TaskStatus.TASK_RUNNING

    task.transition_to(TaskLifecycleStatus.VERIFYING)
    assert task.lifecycle_status == TaskLifecycleStatus.VERIFYING

    task.transition_to(TaskLifecycleStatus.COMPLETED)
    assert task.lifecycle_status == TaskLifecycleStatus.COMPLETED
    assert task.status == TaskStatus.TASK_COMPLETED


def test_invalid_lifecycle_transition_raises():
    task = TaskState(goal="Build app")
    # Transitions from CREATED directly to REPAIRING or VERIFYING are illegal
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        task.transition_to(TaskLifecycleStatus.REPAIRING)
    assert "Illegal task state transition" in str(exc_info.value)
    assert exc_info.value.current == TaskLifecycleStatus.CREATED
    assert exc_info.value.target == TaskLifecycleStatus.REPAIRING


def test_terminal_states_cannot_transition():
    task = TaskState(goal="Finished app")
    task.transition_to(TaskLifecycleStatus.COMPLETED)
    assert task.lifecycle_status.is_terminal

    for target in (
        TaskLifecycleStatus.CREATED,
        TaskLifecycleStatus.UNDERSTANDING,
        TaskLifecycleStatus.PLANNING,
        TaskLifecycleStatus.READY,
        TaskLifecycleStatus.EXECUTING,
        TaskLifecycleStatus.VERIFYING,
        TaskLifecycleStatus.REPAIRING,
    ):
        with pytest.raises(InvalidStateTransitionError):
            assert_task_transition(task.lifecycle_status, target)
        with pytest.raises(InvalidStateTransitionError):
            task.transition_to(target)


def test_cancelled_terminal_state():
    task = TaskState(goal="Cancelled app")
    task.transition_to(TaskLifecycleStatus.CANCELLED)
    assert task.lifecycle_status.is_terminal
    with pytest.raises(InvalidStateTransitionError):
        task.transition_to(TaskLifecycleStatus.EXECUTING)


# ---------------------------------------------------------------------------
# 4 & 7. Event Store Sequencing & Monotonicity
# ---------------------------------------------------------------------------


def test_in_memory_event_store_sequencing():
    store = InMemoryEventStore()
    e1 = TaskEvent(task_id="t1", sequence=1, event_type=TaskEventType.TASK_CREATED)
    e2 = TaskEvent(task_id="t1", sequence=2, event_type=TaskEventType.TASK_STATE_CHANGED)
    store.append(e1)
    store.append(e2)

    assert store.last_sequence("t1") == 2
    assert len(store.get("t1")) == 2

    # Sequence violation (equal or lower)
    with pytest.raises(SequenceViolationError):
        store.append(TaskEvent(task_id="t1", sequence=2, event_type=TaskEventType.TASK_STATE_CHANGED))
    with pytest.raises(SequenceViolationError):
        store.append(TaskEvent(task_id="t1", sequence=1, event_type=TaskEventType.TASK_STATE_CHANGED))


def test_jsonl_event_store_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        store1 = JsonlEventStore(base_dir=tmpdir)
        e1 = TaskEvent(task_id="t100", sequence=1, event_type=TaskEventType.TASK_CREATED, payload={"goal": "persisted"})
        e2 = TaskEvent(task_id="t100", sequence=2, event_type=TaskEventType.ACTION_PROPOSED, payload={"action": "step1"})
        store1.append(e1)
        store1.append(e2)

        # Reopen with brand new store instance
        store2 = JsonlEventStore(base_dir=tmpdir)
        events = store2.get("t100")
        assert len(events) == 2
        assert events[0].sequence == 1
        assert events[0].payload["goal"] == "persisted"
        assert events[1].sequence == 2
        assert store2.last_sequence("t100") == 2
        assert store2.list_tasks() == ["t100"]


def test_jsonl_event_store_skips_corrupted_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "t_corrupt.jsonl"
        valid_ev1 = TaskEvent(task_id="t_corrupt", sequence=1, event_type=TaskEventType.TASK_CREATED).model_dump_json()
        valid_ev2 = TaskEvent(task_id="t_corrupt", sequence=2, event_type=TaskEventType.TASK_COMPLETED).model_dump_json()

        with open(path, "w") as f:
            f.write(valid_ev1 + "\n")
            f.write("INVALID NON-JSON LINE @@@\n")
            f.write(valid_ev2 + "\n")

        store = JsonlEventStore(base_dir=tmpdir)
        events = store.get("t_corrupt")
        assert len(events) == 2
        assert events[0].sequence == 1
        assert events[1].sequence == 2


# ---------------------------------------------------------------------------
# 5 & 6. State Projection & Deterministic Replay
# ---------------------------------------------------------------------------


def test_state_projection_deterministic_replay():
    now = datetime.now(UTC)
    events = [
        TaskEvent(
            task_id="task_p1",
            sequence=1,
            event_type=TaskEventType.TASK_CREATED,
            timestamp=now,
            payload={"goal": "Build todo app", "workspace_root": "/tmp/todo"},
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=2,
            event_type=TaskEventType.MODEL_SELECTED,
            payload={"model": "qwen2.5-coder-7b", "role": "coding"},
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=3,
            event_type=TaskEventType.TASK_STATE_CHANGED,
            payload={"to_state": "executing", "reason": "agent started"},
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=4,
            event_type=TaskEventType.TOOL_STARTED,
            payload={"tool_name": "write_file", "action_id": "act_1"},
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=5,
            event_type=TaskEventType.TOOL_COMPLETED,
            payload={
                "tool_name": "write_file",
                "action_id": "act_1",
                "target": "todo.py",
                "success": True,
                "detail": "Created file",
            },
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=6,
            event_type=TaskEventType.VERIFICATION_STARTED,
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=7,
            event_type=TaskEventType.VERIFICATION_COMPLETED,
            payload={"passed": True, "verified_criteria": ["app_runs"]},
        ),
        TaskEvent(
            task_id="task_p1",
            sequence=8,
            event_type=TaskEventType.TASK_COMPLETED,
        ),
    ]

    # Project state
    state1 = project_task_state(events)
    state2 = project_task_state(events)

    # Invariant: identical state output
    assert state1.task_id == "task_p1"
    assert state1.goal == "Build todo app"
    assert state1.selected_model == "qwen2.5-coder-7b"
    assert state1.lifecycle_status == TaskLifecycleStatus.COMPLETED
    assert len(state1.execution_history) == 1
    assert state1.execution_history[0].tool_name == "write_file"
    assert state1.execution_history[0].success is True
    assert state1.verification_status == "passed"
    assert "app_runs" in [c.id for c in state1.acceptance_criteria]

    # Pure reducer equality
    assert state1.task_id == state2.task_id
    assert state1.goal == state2.goal
    assert state1.lifecycle_status == state2.lifecycle_status
    assert len(state1.execution_history) == len(state2.execution_history)
    assert len(state1.acceptance_criteria) == len(state2.acceptance_criteria)
    assert state1.summary() == state2.summary()


# ---------------------------------------------------------------------------
# 8. Concurrent Event Appends
# ---------------------------------------------------------------------------


def test_concurrent_event_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlEventStore(base_dir=tmpdir)

        def append_task_events(task_id: str):
            for seq in range(1, 21):
                store.append(
                    TaskEvent(
                        task_id=task_id,
                        sequence=seq,
                        event_type=TaskEventType.TOOL_COMPLETED,
                        payload={"step": seq},
                    )
                )

        # Concurrently append to distinct tasks
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(append_task_events, f"task_conc_{i}")
                for i in range(5)
            ]
            for f in futures:
                f.result()

        for i in range(5):
            evs = store.get(f"task_conc_{i}")
            assert len(evs) == 20
            assert [e.sequence for e in evs] == list(range(1, 21))


# ---------------------------------------------------------------------------
# 9. Secret and Credential Redaction
# ---------------------------------------------------------------------------


def test_secret_scrubbing_in_events():
    payload = {
        "api_key": "sk-1234567890abcdef1234567890abcdef",
        "nested": {
            "password": "supersecretpassword",
            "token": "ghp_1234567890abcdef1234567890abcdef",
        },
        "command": "curl -H 'Authorization: Bearer my-secret-jwt-token-val' https://example.com",
        "safe_data": "normal value",
    }

    sanitized = sanitize_event_payload(payload)
    assert sanitized["api_key"] == "********"
    assert sanitized["nested"]["password"] == "********"
    assert sanitized["nested"]["token"] == "********"
    assert sanitized["safe_data"] == "normal value"

    # Verify TaskEvent automatically sanitizes on creation
    ev = TaskEvent(
        task_id="t_sec",
        sequence=1,
        event_type=TaskEventType.TOOL_STARTED,
        payload=payload,
    )
    assert ev.payload["api_key"] == "********"
    assert ev.payload["nested"]["password"] == "********"


# ---------------------------------------------------------------------------
# 10. EventBus Backwards Compatibility
# ---------------------------------------------------------------------------


def test_event_bus_backwards_compatibility():
    bus = EventBus()
    received = []

    def listener(ev):
        received.append(ev)

    bus.subscribe(listener, TaskEventType.RUN_STARTED)

    # Synchronous emission
    bus.emit_sync(
        TaskEvent(
            task_id="t1",
            sequence=1,
            event_type=TaskEventType.RUN_STARTED,
            payload={"msg": "hello"},
        )
    )

    assert len(received) == 1
    assert len(bus.history) == 1
    assert bus.history[0].event_type == TaskEventType.RUN_STARTED


# ---------------------------------------------------------------------------
# 11. AgentRuntime Integration with EventStore
# ---------------------------------------------------------------------------


class FakeAgent(BaseAgent):
    def __init__(self, response_msg: ChatMessage):
        self.response_msg = response_msg

    async def run(self, user_input: str, history=None, **kwargs) -> ChatMessage:
        return self.response_msg


def test_agent_runtime_with_event_store():
    async def _test():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlEventStore(base_dir=tmpdir)
            bus = EventBus(store=store)
            runtime = AgentRuntime(event_bus=bus)

            task = TaskState(goal="Implement test feature")
            agent = FakeAgent(ChatMessage(role=Role.ASSISTANT, content="Done", task_state=task))

            result = await runtime.execute(agent, "Implement test feature", task=task)
            assert result.is_success

            # Check that events were persisted to the store
            persisted = store.get(task.task_id)
            assert len(persisted) >= 1
            assert any(e.event_type == TaskEventType.TASK_CREATED for e in persisted)

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 12. Regression Tests (Phase 1 Integration Fixes)
# ---------------------------------------------------------------------------


def test_user_prompt_not_equal_task_id():
    prompt = "I am testing the canonical TaskState and durable EventStore integration."

    # Positional invocation
    task1 = TaskState(prompt)
    assert task1.goal == prompt
    assert task1.task_id != prompt
    assert task1.task_id.startswith("task_")

    # Keyword invocation
    task2 = TaskState(goal=prompt)
    assert task2.goal == prompt
    assert task2.task_id != prompt
    assert task2.task_id.startswith("task_")

    # Erroneous attempt to assign task_id == prompt
    task3 = TaskState(task_id=prompt, goal=prompt)
    assert task3.task_id != prompt
    assert task3.task_id.startswith("task_")
    assert task3.goal == prompt


def test_task_id_stability_across_execution():
    async def _test():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlEventStore(base_dir=tmpdir)
            bus = EventBus(store=store)
            runtime = AgentRuntime(event_bus=bus)

            prompt = "Read project status"
            task = TaskState(goal=prompt)
            initial_id = task.task_id

            agent = FakeAgent(ChatMessage(role=Role.ASSISTANT, content="Done", task_state=task))
            result = await runtime.execute(agent, prompt, task=task)

            assert result.task_state.task_id == initial_id
            assert task.task_id == initial_id

    asyncio.run(_test())


def test_production_event_bus_defaults_to_jsonl_store():
    bus = EventBus()
    assert isinstance(bus._store, JsonlEventStore)


def test_events_actually_written_to_disk_in_runtime_execution():
    async def _test():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlEventStore(base_dir=tmpdir)
            bus = EventBus(store=store)
            runtime = AgentRuntime(event_bus=bus)

            prompt = "Inspect repository structure"
            task = TaskState(goal=prompt)
            agent = FakeAgent(ChatMessage(role=Role.ASSISTANT, content="Inspected", task_state=task))

            result = await runtime.execute(agent, prompt, task=task)
            task_id = result.task_state.task_id

            path = Path(tmpdir) / f"{task_id}.jsonl"
            assert path.exists()
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) >= 3

            persisted = store.get(task_id)
            assert len(persisted) == len(lines)
            assert persisted[0].event_type == TaskEventType.TASK_CREATED
            assert any(e.event_type == TaskEventType.TASK_COMPLETED for e in persisted)

    asyncio.run(_test())


def test_prompt_instructing_do_not_create_files_does_not_scaffold():
    from ultron.core.intelligence.task_classification import classify_task_deterministic
    from ultron.core.intelligence.task_planning import fallback_plan
    from ultron.core.types import TaskType, WorkspaceKind

    prompt = (
        "I am testing the canonical TaskState and durable EventStore integration. "
        "Do not create files. Do not run commands. After completing read-only inspection, report the state."
    )
    classification = classify_task_deterministic(prompt)
    assert classification.task_type in (TaskType.RESEARCH, TaskType.INFORMATIONAL)
    assert classification.task_type != TaskType.SOFTWARE_ENGINEERING

    # Fallback plan for MULTI_STEP must not scaffold application files
    multi_step_plan = fallback_plan("General multi step work", TaskType.MULTI_STEP, WorkspaceKind.UNKNOWN)
    assert not any("scaffold" in step.description.lower() for step in multi_step_plan.steps)
    assert not any("implement application files" in step.description.lower() for step in multi_step_plan.steps)


def test_prompt_instructing_do_not_plan_bypasses_planning():
    async def _test():
        from ultron.core.intelligence.task_planning import prepare_task_for_execution

        prompt = (
            "I am testing the canonical TaskState and durable EventStore integration. "
            "Do not create a plan. Do not create files. Report the actual TaskState."
        )
        task = await prepare_task_for_execution(prompt, None)
        assert task is None

    asyncio.run(_test())


def test_read_only_prompt_does_not_fail_with_plan_step_failed():
    async def _test():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlEventStore(base_dir=tmpdir)
            bus = EventBus(store=store)
            runtime = AgentRuntime(event_bus=bus)

            prompt = (
                "I am testing the canonical TaskState and durable EventStore integration. "
                "Do not create a plan. Do not create files. Report the actual TaskState."
            )
            agent = FakeAgent(ChatMessage(role=Role.ASSISTANT, content="TaskState: ready"))
            result = await runtime.execute(agent, prompt)

            task = result.task_state
            assert task.lifecycle_status in (TaskLifecycleStatus.READY, TaskLifecycleStatus.EXECUTING, TaskLifecycleStatus.COMPLETED)
            assert "plan_step_failed" not in task.lifecycle_history
            assert not any(err.message == "plan_step_failed" for err in task.errors)

    asyncio.run(_test())


def test_process_restart_replay_preserves_task_state_exactly():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Process A writes events
        store_a = JsonlEventStore(base_dir=tmpdir)
        tid = "task_restart_test_123"
        ev1 = TaskEvent(
            task_id=tid,
            sequence=1,
            event_type=TaskEventType.TASK_CREATED,
            payload={"goal": "Survive process restart", "workspace_root": "/tmp/testws"},
            source="test",
        )
        ev2 = TaskEvent(
            task_id=tid,
            sequence=2,
            event_type=TaskEventType.TASK_STATE_CHANGED,
            payload={"to_state": TaskLifecycleStatus.READY.value, "reason": "planning complete"},
            source="test",
        )
        ev3 = TaskEvent(
            task_id=tid,
            sequence=3,
            event_type=TaskEventType.TASK_STATE_CHANGED,
            payload={"to_state": TaskLifecycleStatus.EXECUTING.value, "reason": "starting execution"},
            source="test",
        )
        ev4 = TaskEvent(
            task_id=tid,
            sequence=4,
            event_type=TaskEventType.TASK_COMPLETED,
            payload={"goal": "Survive process restart"},
            source="test",
        )
        store_a.append_many([ev1, ev2, ev3, ev4])

        # Drop store_a completely to simulate process exit
        del store_a

        # Phase 2: Process B opens a fresh store instance pointing to same disk dir
        store_b = JsonlEventStore(base_dir=tmpdir)
        loaded_events = store_b.get(tid)
        assert len(loaded_events) == 4

        rebuilt = project_task_state(loaded_events)
        assert rebuilt.task_id == tid
        assert rebuilt.goal == "Survive process restart"
        assert rebuilt.lifecycle_status == TaskLifecycleStatus.COMPLETED
        assert rebuilt.workspace_root == "/tmp/testws"
        assert rebuilt.lifecycle_history == ["created", "ready", "executing", "completed"]

