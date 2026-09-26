"""
core/workflow/events.py

Event schema and event bus for the workflow execution engine.
Events are persisted to a JSONL file and broadcast via SSE to connected frontends.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from core.json_utils import json_default


class EventType(str, Enum):
    WORKFLOW_STARTED = "workflow:started"
    WORKFLOW_COMPLETED = "workflow:completed"
    WORKFLOW_FAILED = "workflow:failed"
    WORKFLOW_CANCELLED = "workflow:cancelled"
    NODE_STATUS_CHANGE = "node:status_change"
    NODE_PROGRESS = "node:progress"
    NODE_ARTIFACT = "node:artifact"
    NODE_LOG = "node:log"
    NODE_METRICS = "node:metrics"


@dataclass
class WorkflowEvent:
    """A single immutable event in the workflow execution lifecycle."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    workflow_id: str = ""
    node_id: Optional[str] = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=json_default)


class EventBus:
    """
    Thread-safe event bus that:
    1. Persists all events to a JSONL file (crash-recoverable).
    2. Notifies in-process subscribers synchronously.
    3. Pushes to async SSE subscribers for frontend consumption.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._subscribers: list[Callable[[WorkflowEvent], None]] = []
        self._async_queues: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._persist_path = persist_path
        self._events: list[WorkflowEvent] = []

        # Load existing events if persist file exists
        if persist_path and persist_path.exists():
            self._load_persisted()

    def _load_persisted(self):
        """Replay persisted events into memory."""
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    evt = WorkflowEvent(
                        event_type=EventType(d["event_type"]),
                        timestamp=d["timestamp"],
                        workflow_id=d.get("workflow_id", ""),
                        node_id=d.get("node_id"),
                        data=d.get("data", {}),
                    )
                    self._events.append(evt)
        except Exception:
            pass  # Corrupted file — start fresh

    def emit(self, event: WorkflowEvent):
        """Emit an event: persist, notify sync subscribers, push to async queues."""
        with self._lock:
            self._events.append(event)

            # Persist to JSONL
            if self._persist_path:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._persist_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")

            # Sync subscribers (node wrappers, logging, etc.)
            for sub in self._subscribers:
                try:
                    sub(event)
                except Exception:
                    pass

            # Async queues (SSE connections)
            for q in self._async_queues:
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    def subscribe(self, callback: Callable[[WorkflowEvent], None]):
        """Register a synchronous subscriber."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[WorkflowEvent], None]):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def create_async_queue(self) -> asyncio.Queue:
        """Create an async queue for SSE streaming. Caller must remove it when done."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._async_queues.append(q)
        return q

    def remove_async_queue(self, q: asyncio.Queue):
        with self._lock:
            self._async_queues = [x for x in self._async_queues if x is not q]

    def get_all_events(self, workflow_id: Optional[str] = None) -> list[dict]:
        """Return all recorded events, optionally filtered by workflow_id."""
        with self._lock:
            if workflow_id:
                return [e.to_dict() for e in self._events if e.workflow_id == workflow_id]
            return [e.to_dict() for e in self._events]

    def clear(self):
        """Clear in-memory events and persistence file."""
        with self._lock:
            self._events.clear()
            if self._persist_path and self._persist_path.exists():
                self._persist_path.write_text("")
