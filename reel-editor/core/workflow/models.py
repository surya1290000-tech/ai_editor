"""
core/workflow/models.py

Data models for the workflow DAG: Graphs, Nodes, Ports, Edges, and execution state.
All models are plain dataclasses — no ORM, no external dependencies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─── Enumerations ────────────────────────────────────────────────────────────

class NodeStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PLANNED = "planned"       # Visible on canvas but not executable yet


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ─── Port & Edge Models ─────────────────────────────────────────────────────

@dataclass
class NodePort:
    """A typed input or output slot on a node."""
    name: str                          # e.g. "audio_wav", "transcript_raw"
    direction: PortDirection
    data_type: str                     # e.g. "file:audio/wav", "json:transcript", "dir:frames"
    required: bool = True
    description: str = ""
    value: Any = None                  # Populated at runtime with actual path / data reference


@dataclass
class Edge:
    """A directed connection between an output port and an input port."""
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_node: str = ""              # node_id of upstream node
    source_port: str = ""              # port name on upstream node
    target_node: str = ""              # node_id of downstream node
    target_port: str = ""              # port name on downstream node


# ─── Node Models ─────────────────────────────────────────────────────────────

@dataclass
class NodeExecution:
    """Runtime execution state for a node."""
    status: NodeStatus = NodeStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    runtime_seconds: Optional[float] = None
    progress: int = 0                  # 0-100
    progress_message: str = ""
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metrics: dict = field(default_factory=dict)       # e.g. {"words": 251, "wpm": 192}
    artifacts: list[dict] = field(default_factory=list)  # [{"name": "transcript_raw.json", "path": "...", "type": "json"}]
    logs: list[dict] = field(default_factory=list)       # [{"level": "INFO", "message": "...", "ts": ...}]
    parameters: dict = field(default_factory=dict)       # Runtime config passed to this node
    tool: Optional[str] = None                           # "faster-whisper", "librosa", etc.
    model: Optional[str] = None                          # "base", "gemma3:4b", etc.

    def mark_running(self):
        self.status = NodeStatus.RUNNING
        self.started_at = time.time()
        self.progress = 0

    def mark_completed(self):
        self.status = NodeStatus.COMPLETED
        self.completed_at = time.time()
        self.progress = 100
        if self.started_at:
            self.runtime_seconds = round(self.completed_at - self.started_at, 2)

    def mark_failed(self, error: str, traceback: str = ""):
        self.status = NodeStatus.FAILED
        self.completed_at = time.time()
        self.error_message = error
        self.error_traceback = traceback
        if self.started_at:
            self.runtime_seconds = round(self.completed_at - self.started_at, 2)

    def reset(self):
        """Reset to pending for re-run."""
        self.status = NodeStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.runtime_seconds = None
        self.progress = 0
        self.progress_message = ""
        self.error_message = None
        self.error_traceback = None
        self.metrics = {}
        self.artifacts = []
        self.logs = []


@dataclass
class NodeDefinition:
    """Static definition of a workflow node (registered in the node registry)."""
    node_id: str                       # "node_s2_speech"
    title: str                         # "Speech Extraction"
    description: str = ""
    agent_group: str = "Agent 1"       # "Agent 1", "Agent 2", "Agent 3"
    category: str = ""                 # "extraction", "analysis", "fusion", etc.
    icon: str = "📦"
    version: str = "1.0.0"
    tool: str = ""                     # Primary tool / library
    inputs: list[NodePort] = field(default_factory=list)
    outputs: list[NodePort] = field(default_factory=list)
    is_planned: bool = False           # True for Agent 2/3 placeholder nodes
    execution: NodeExecution = field(default_factory=NodeExecution)

    # Canvas layout hints (optional, frontend can override)
    position_x: float = 0.0
    position_y: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for API / frontend consumption."""
        return {
            "node_id": self.node_id,
            "title": self.title,
            "description": self.description,
            "agent_group": self.agent_group,
            "category": self.category,
            "icon": self.icon,
            "version": self.version,
            "tool": self.tool,
            "is_planned": self.is_planned,
            "inputs": [_port_dict(p) for p in self.inputs],
            "outputs": [_port_dict(p) for p in self.outputs],
            "position": {"x": self.position_x, "y": self.position_y},
            "execution": {
                "status": self.execution.status.value,
                "started_at": self.execution.started_at,
                "completed_at": self.execution.completed_at,
                "runtime_seconds": self.execution.runtime_seconds,
                "progress": self.execution.progress,
                "progress_message": self.execution.progress_message,
                "error_message": self.execution.error_message,
                "metrics": self.execution.metrics,
                "artifacts": self.execution.artifacts,
                "logs": self.execution.logs[-50:],  # Last 50 logs for API
                "parameters": self.execution.parameters,
                "tool": self.execution.tool,
                "model": self.execution.model,
            },
        }


# ─── Graph Model ─────────────────────────────────────────────────────────────

@dataclass
class WorkflowGraph:
    """The complete workflow DAG: nodes + edges + execution state."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "Reel Analysis Pipeline"
    status: WorkflowStatus = WorkflowStatus.IDLE
    nodes: dict[str, NodeDefinition] = field(default_factory=dict)  # node_id → NodeDefinition
    edges: list[Edge] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    video_path: Optional[str] = None
    output_dir: Optional[str] = None

    def add_node(self, node: NodeDefinition):
        self.nodes[node.node_id] = node

    def add_edge(self, source_node: str, source_port: str, target_node: str, target_port: str):
        self.edges.append(Edge(
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
        ))

    def get_upstream_nodes(self, node_id: str) -> list[str]:
        """Return node_ids that feed into this node."""
        return list({e.source_node for e in self.edges if e.target_node == node_id})

    def get_downstream_nodes(self, node_id: str) -> list[str]:
        """Return node_ids that this node feeds into."""
        return list({e.target_node for e in self.edges if e.source_node == node_id})

    def get_ready_nodes(self) -> list[str]:
        """
        Return node_ids whose upstream dependencies are all COMPLETED
        and whose own status is PENDING or QUEUED.
        Planned nodes are never ready.
        """
        ready = []
        for node_id, node in self.nodes.items():
            if node.is_planned:
                continue
            if node.execution.status not in (NodeStatus.PENDING, NodeStatus.QUEUED):
                continue
            upstream = self.get_upstream_nodes(node_id)
            all_done = all(
                self.nodes[uid].execution.status == NodeStatus.COMPLETED
                for uid in upstream
                if uid in self.nodes
            )
            if all_done:
                ready.append(node_id)
        return ready

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_node": e.source_node,
                    "source_port": e.source_port,
                    "target_node": e.target_node,
                    "target_port": e.target_port,
                }
                for e in self.edges
            ],
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _port_dict(p: NodePort) -> dict:
    return {
        "name": p.name,
        "direction": p.direction.value,
        "data_type": p.data_type,
        "required": p.required,
        "description": p.description,
    }
