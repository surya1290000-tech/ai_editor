"""
core/workflow/engine.py

DAG Workflow Execution Engine.
- Resolves dependencies and dispatches ready nodes.
- Executes independent nodes in parallel via ThreadPoolExecutor.
- Propagates typed artifacts between nodes through edges.
- Emits lifecycle events through the EventBus.
- Supports cancellation, failure recovery, and re-runs.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Callable, Optional

from core.workflow.models import (
    WorkflowGraph, NodeDefinition, NodeStatus, WorkflowStatus,
)
from core.workflow.events import EventBus, EventType, WorkflowEvent
from core.logger import get_logger

logger = get_logger(__name__)


class ExecutionContext:
    """
    Context passed to each node executor.
    Provides access to input data, output dir, config, event emission, and progress reporting.
    """

    def __init__(
        self,
        node_id: str,
        inputs: dict[str, Any],
        output_dir: Path,
        video_path: Path,
        config: dict,
        event_bus: EventBus,
        workflow_id: str,
    ):
        self.node_id = node_id
        self.inputs = inputs           # port_name → value (file path, json dict, etc.)
        self.output_dir = output_dir
        self.video_path = video_path
        self.config = config           # hardware profile, model selection, etc.
        self._event_bus = event_bus
        self._workflow_id = workflow_id

    def log(self, message: str, level: str = "INFO"):
        self._event_bus.emit(WorkflowEvent(
            event_type=EventType.NODE_LOG,
            workflow_id=self._workflow_id,
            node_id=self.node_id,
            data={"level": level, "message": message},
        ))

    def set_progress(self, progress: int, message: str = ""):
        self._event_bus.emit(WorkflowEvent(
            event_type=EventType.NODE_PROGRESS,
            workflow_id=self._workflow_id,
            node_id=self.node_id,
            data={"progress": min(max(progress, 0), 100), "message": message},
        ))

    def add_artifact(self, name: str, path: str, artifact_type: str = "json"):
        self._event_bus.emit(WorkflowEvent(
            event_type=EventType.NODE_ARTIFACT,
            workflow_id=self._workflow_id,
            node_id=self.node_id,
            data={"name": name, "path": path, "type": artifact_type},
        ))

    def set_metrics(self, metrics: dict):
        self._event_bus.emit(WorkflowEvent(
            event_type=EventType.NODE_METRICS,
            workflow_id=self._workflow_id,
            node_id=self.node_id,
            data={"metrics": metrics},
        ))


# Node executor signature: (ctx: ExecutionContext) -> dict[str, Any]
# Returns a mapping of output port names → values.
NodeExecutorFn = Callable[[ExecutionContext], dict[str, Any]]


class WorkflowEngine:
    """
    Executes a WorkflowGraph by topologically dispatching ready nodes
    in parallel, propagating outputs through edges.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        event_bus: EventBus,
        executors: dict[str, NodeExecutorFn],
        config: dict | None = None,
        max_workers: int = 3,
    ):
        self.graph = graph
        self.event_bus = event_bus
        self.executors = executors       # node_id → callable
        self.config = config or {}
        self.max_workers = max_workers

        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._port_values: dict[str, dict[str, Any]] = {}  # node_id → {port_name: value}
        self._running_futures: dict[str, Future] = {}

    def run(self, video_path: str | Path, output_dir: str | Path):
        """
        Execute the entire workflow synchronously (blocking).
        Call from a background thread if you don't want to block.
        """
        self.graph.video_path = str(video_path)
        self.graph.output_dir = str(output_dir)
        self.graph.status = WorkflowStatus.RUNNING
        self.graph.started_at = time.time()
        self._cancelled.clear()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.event_bus.emit(WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=self.graph.workflow_id,
            data={"video_path": str(video_path), "output_dir": str(output_dir)},
        ))

        # Mark the raw video input node as completed immediately
        raw_node = self.graph.nodes.get("node_raw_video")
        if raw_node:
            raw_node.execution.mark_running()
            self._emit_status(raw_node)
            raw_node.execution.mark_completed()
            raw_node.execution.metrics = {"file": Path(video_path).name}
            self._emit_status(raw_node)
            self._port_values["node_raw_video"] = {"video_file": str(video_path)}

        try:
            self._execute_loop(Path(video_path), Path(output_dir))
        except Exception as e:
            self.graph.status = WorkflowStatus.FAILED
            self.graph.completed_at = time.time()
            self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                workflow_id=self.graph.workflow_id,
                data={"error": str(e)},
            ))
            return

        if self._cancelled.is_set():
            self.graph.status = WorkflowStatus.CANCELLED
            self.event_bus.emit(WorkflowEvent(
                event_type=EventType.WORKFLOW_CANCELLED,
                workflow_id=self.graph.workflow_id,
            ))
        else:
            # Check if any non-planned node failed
            any_failed = any(
                n.execution.status == NodeStatus.FAILED
                for n in self.graph.nodes.values()
                if not n.is_planned
            )
            if any_failed:
                self.graph.status = WorkflowStatus.FAILED
                self.event_bus.emit(WorkflowEvent(
                    event_type=EventType.WORKFLOW_FAILED,
                    workflow_id=self.graph.workflow_id,
                    data={"error": "One or more nodes failed"},
                ))
            else:
                self.graph.status = WorkflowStatus.COMPLETED
                self.event_bus.emit(WorkflowEvent(
                    event_type=EventType.WORKFLOW_COMPLETED,
                    workflow_id=self.graph.workflow_id,
                ))

        self.graph.completed_at = time.time()

    def cancel(self):
        """Signal cancellation. Running nodes will finish but no new nodes will start."""
        self._cancelled.set()

    def _execute_loop(self, video_path: Path, output_dir: Path):
        """Main dispatch loop: find ready nodes, execute in parallel, repeat."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while not self._cancelled.is_set():
                ready = self.graph.get_ready_nodes()

                if not ready:
                    # Check if anything is still running
                    running = [
                        nid for nid, n in self.graph.nodes.items()
                        if n.execution.status == NodeStatus.RUNNING
                    ]
                    if running:
                        # Wait for running nodes to finish, then re-check
                        time.sleep(0.3)
                        continue
                    else:
                        # Nothing ready, nothing running → done
                        break

                futures: dict[str, Future] = {}

                for node_id in ready:
                    if self._cancelled.is_set():
                        break

                    node = self.graph.nodes[node_id]
                    executor = self.executors.get(node_id)

                    if executor is None:
                        # No executor registered — mark as pending/skip
                        logger.warning(f"No executor for node '{node_id}' — skipping")
                        node.execution.status = NodeStatus.FAILED
                        node.execution.error_message = "No executor registered"
                        self._emit_status(node)
                        continue

                    # Gather inputs from upstream edges
                    inputs = self._gather_inputs(node_id)

                    # Mark as running
                    node.execution.mark_running()
                    self._emit_status(node)

                    # Build context
                    ctx = ExecutionContext(
                        node_id=node_id,
                        inputs=inputs,
                        output_dir=output_dir,
                        video_path=video_path,
                        config=self.config,
                        event_bus=self.event_bus,
                        workflow_id=self.graph.workflow_id,
                    )

                    future = pool.submit(self._run_node, node, executor, ctx)
                    futures[node_id] = future
                    self._running_futures[node_id] = future

                # Wait for this batch to complete
                for node_id, future in futures.items():
                    try:
                        future.result()  # blocks until done
                    except Exception:
                        pass  # Errors handled inside _run_node
                    self._running_futures.pop(node_id, None)

    def _run_node(self, node: NodeDefinition, executor: NodeExecutorFn, ctx: ExecutionContext):
        """Execute a single node in a worker thread."""
        try:
            outputs = executor(ctx)
            if outputs is None:
                outputs = {}

            # Store output port values for downstream propagation
            with self._lock:
                self._port_values[node.node_id] = outputs

            node.execution.mark_completed()
            self._emit_status(node)

        except Exception as e:
            tb = traceback.format_exc()
            node.execution.mark_failed(str(e), tb)
            self._emit_status(node)
            ctx.log(f"FAILED: {e}", level="ERROR")
            logger.error(f"Node '{node.node_id}' failed: {e}\n{tb}")

    def _gather_inputs(self, node_id: str) -> dict[str, Any]:
        """
        Collect input values for a node by following edges from upstream outputs.
        """
        inputs: dict[str, Any] = {}
        for edge in self.graph.edges:
            if edge.target_node != node_id:
                continue
            source_outputs = self._port_values.get(edge.source_node, {})
            if edge.source_port in source_outputs:
                inputs[edge.target_port] = source_outputs[edge.source_port]
        return inputs

    def _emit_status(self, node: NodeDefinition):
        """Emit a node:status_change event."""
        self.event_bus.emit(WorkflowEvent(
            event_type=EventType.NODE_STATUS_CHANGE,
            workflow_id=self.graph.workflow_id,
            node_id=node.node_id,
            data={
                "status": node.execution.status.value,
                "progress": node.execution.progress,
                "progress_message": node.execution.progress_message,
                "started_at": node.execution.started_at,
                "completed_at": node.execution.completed_at,
                "runtime_seconds": node.execution.runtime_seconds,
                "error_message": node.execution.error_message,
                "metrics": node.execution.metrics,
                "artifacts": node.execution.artifacts,
            },
        ))
