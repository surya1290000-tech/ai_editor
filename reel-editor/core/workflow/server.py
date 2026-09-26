"""
core/workflow/server.py

FastAPI local server for the visual workflow system.
- Serves the n8n-style node canvas frontend
- Exposes REST API for graph state, node details, artifact access
- Broadcasts real-time execution events via SSE (Server-Sent Events)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from core.workflow.models import WorkflowGraph, WorkflowStatus, NodeStatus
from core.workflow.events import EventBus, EventType, WorkflowEvent
from core.workflow.engine import WorkflowEngine
from core.workflow.registry import build_default_graph
from core.workflow.node_executors import get_all_executors
from core.json_utils import json_default

# ─── Global State ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_WEB_DIR = _PROJECT_ROOT / "web"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
_EVENTS_FILE = _OUTPUTS_DIR / "workflow_events.jsonl"

# Singleton workflow state
_graph: Optional[WorkflowGraph] = None
_event_bus: Optional[EventBus] = None
_engine: Optional[WorkflowEngine] = None
_engine_thread: Optional[threading.Thread] = None


def _get_graph() -> WorkflowGraph:
    global _graph
    if _graph is None:
        _graph = build_default_graph()
    return _graph


def _get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        _event_bus = EventBus(persist_path=_EVENTS_FILE)
    return _event_bus


def _load_config() -> dict:
    """Load hardware profile / model config."""
    profile_path = _OUTPUTS_DIR / "hardware_profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            hw_data = json.load(f)
        models = hw_data.get("models", {})
        return {
            "whisper_model": models.get("whisper_model", "base"),
            "whisper_device": models.get("whisper_device", "cpu"),
            "whisper_compute_type": models.get("whisper_compute_type", "int8"),
            "llm_model": models.get("llm_model", "gemma3:4b"),
            "frame_sample_fps": models.get("frame_sample_fps", 1),
            "mediapipe_complexity": models.get("mediapipe_complexity", 0),
        }
    return {
        "whisper_model": "base",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "frame_sample_fps": 1,
        "mediapipe_complexity": 0,
    }


# ─── App Factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="AI Reel Editor — Workflow Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static frontend files
    if _WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    # ── Routes ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = _WEB_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>AI Reel Editor — Workflow Server</h1><p>Frontend not built yet.</p>")

    @app.get("/api/workflow/graph")
    async def get_graph():
        graph = _get_graph()
        return JSONResponse(
            content=json.loads(json.dumps(graph.to_dict(), default=json_default))
        )

    @app.get("/api/workflow/nodes/{node_id}")
    async def get_node(node_id: str):
        graph = _get_graph()
        node = graph.nodes.get(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
        return JSONResponse(
            content=json.loads(json.dumps(node.to_dict(), default=json_default))
        )

    @app.get("/api/workflow/events")
    async def sse_events(request: Request):
        """SSE endpoint — streams real-time workflow events to the frontend."""
        bus = _get_event_bus()
        queue = bus.create_async_queue()

        async def event_generator():
            try:
                # First, send a snapshot event with current graph state
                graph = _get_graph()
                yield {
                    "event": "snapshot",
                    "data": json.dumps(graph.to_dict(), default=json_default),
                }

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield {
                            "event": event.event_type.value,
                            "data": event.to_json(),
                        }
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield {"event": "keepalive", "data": "{}"}
            finally:
                bus.remove_async_queue(queue)

        return EventSourceResponse(event_generator())

    @app.get("/api/workflow/events/history")
    async def get_event_history():
        """Return all persisted events for reconnection."""
        bus = _get_event_bus()
        return JSONResponse(content=bus.get_all_events())

    @app.post("/api/workflow/run")
    async def run_workflow(background_tasks: BackgroundTasks, request: Request):
        """Trigger workflow execution on a video file."""
        global _engine, _engine_thread

        body = await request.json()
        video_path = body.get("video_path", "")

        if not video_path:
            raise HTTPException(status_code=400, detail="video_path is required")

        video_file = Path(video_path)
        if not video_file.is_absolute():
            video_file = _PROJECT_ROOT / video_path

        if not video_file.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_file}")

        graph = _get_graph()
        if graph.status == WorkflowStatus.RUNNING:
            raise HTTPException(status_code=409, detail="Workflow is already running")

        # Reset all non-planned node executions
        for node in graph.nodes.values():
            if not node.is_planned:
                node.execution.reset()

        bus = _get_event_bus()
        bus.clear()

        config = _load_config()
        executors = get_all_executors()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _OUTPUTS_DIR / f"{video_file.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        _engine = WorkflowEngine(
            graph=graph,
            event_bus=bus,
            executors=executors,
            config=config,
            max_workers=2,
        )

        def _run():
            _engine.run(video_path=video_file, output_dir=output_dir)

        _engine_thread = threading.Thread(target=_run, daemon=True)
        _engine_thread.start()

        return JSONResponse(content={
            "status": "started",
            "workflow_id": graph.workflow_id,
            "output_dir": str(output_dir),
        })

    @app.post("/api/workflow/cancel")
    async def cancel_workflow():
        global _engine
        if _engine:
            _engine.cancel()
            return JSONResponse(content={"status": "cancellation_requested"})
        raise HTTPException(status_code=400, detail="No workflow running")

    @app.post("/api/workflow/upload")
    async def upload_video(file: UploadFile = File(...)):
        """Upload a raw video file to process."""
        allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed))}"
            )

        uploads_dir = _PROJECT_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / file.filename

        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        # Update graph source node
        graph = _get_graph()
        if "node_raw_video" in graph.nodes:
            raw_node = graph.nodes["node_raw_video"]
            raw_node.execution.metrics = {
                "file": file.filename,
                "size_mb": f"{dest.stat().st_size / (1024*1024):.1f} MB",
            }

        return JSONResponse(content={
            "status": "uploaded",
            "filename": file.filename,
            "path": str(dest),
            "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        })

    @app.get("/api/workflow/download-video")
    async def download_video():
        """Download the latest rendered final video."""
        videos = sorted(_OUTPUTS_DIR.glob("**/final_reel_edited.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not videos:
            raise HTTPException(status_code=404, detail="No final rendered video found. Run workflow first.")
        video_path = videos[0]
        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            filename="final_reel_edited.mp4"
        )

    @app.get("/api/workflow/download-edl")
    async def download_edl():
        """Download the latest DaVinci Resolve EDL."""
        edls = sorted(_OUTPUTS_DIR.glob("**/timeline.edl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not edls:
            raise HTTPException(status_code=404, detail="No EDL timeline found. Run workflow first.")
        return FileResponse(
            str(edls[0]),
            media_type="text/plain",
            filename="timeline.edl"
        )

    @app.get("/api/artifacts/{path:path}")
    async def get_artifact(path: str):
        """Serve generated artifact files."""
        file_path = _OUTPUTS_DIR / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")

        if file_path.suffix == ".json":
            content = file_path.read_text(encoding="utf-8")
            return JSONResponse(content=json.loads(content))
        elif file_path.suffix in (".jpg", ".jpeg", ".png"):
            return FileResponse(str(file_path), media_type=f"image/{file_path.suffix[1:]}")
        elif file_path.suffix == ".wav":
            return FileResponse(str(file_path), media_type="audio/wav")
        elif file_path.suffix == ".mp4":
            return FileResponse(str(file_path), media_type="video/mp4", filename=file_path.name)
        elif file_path.suffix == ".edl":
            return FileResponse(str(file_path), media_type="text/plain", filename=file_path.name)
        elif file_path.suffix == ".srt":
            return FileResponse(str(file_path), media_type="text/plain", filename=file_path.name)
        elif file_path.suffix == ".html":
            return HTMLResponse(file_path.read_text(encoding="utf-8"))
        else:
            return FileResponse(str(file_path))

    # ── Phase 6: Timeline Editor & Human Review Endpoints ──────────────
    from core.timeline.project import TimelineProjectManager

    def _get_timeline_project(project_name: Optional[str] = None) -> TimelineProjectManager:
        # Locate project directory dynamically
        target_dir = None
        if project_name:
            cand = _OUTPUTS_DIR / project_name
            if cand.exists():
                target_dir = cand

        if not target_dir:
            # Pick latest output directory that has timeline_ir.json or project folder
            valid_dirs = [
                d for d in _OUTPUTS_DIR.iterdir()
                if d.is_dir() and not d.name.startswith(".") and ((d / "timeline_ir.json").exists() or (d / "project").exists())
            ]
            if valid_dirs:
                target_dir = sorted(valid_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
            else:
                target_dir = _OUTPUTS_DIR / "default_project"
                target_dir.mkdir(parents=True, exist_ok=True)

        proj_dir = target_dir / "project"
        base_ir = target_dir / "timeline_ir.json"
        return TimelineProjectManager(
            project_dir=proj_dir,
            base_timeline_path=base_ir if base_ir.exists() else None,
            context_dir=target_dir,
        )

    @app.get("/editor", response_class=HTMLResponse)
    async def get_editor_page():
        """Serves the interactive multi-track timeline editor UI."""
        editor_html = _WEB_DIR / "editor.html"
        if not editor_html.exists():
            raise HTTPException(status_code=404, detail="editor.html not found.")
        return HTMLResponse(editor_html.read_text(encoding="utf-8"))

    @app.get("/review", response_class=HTMLResponse)
    async def get_review_report():
        """Serves the latest publication-grade Human-Quality Editorial Review report."""
        reports = sorted(_OUTPUTS_DIR.glob("**/human_quality_report.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not reports:
            raise HTTPException(status_code=404, detail="No human quality review report found. Run review first.")
        return HTMLResponse(reports[0].read_text(encoding="utf-8"))

    @app.get("/api/editor/project")
    async def get_editor_project():
        """Returns the current timeline IR, project manifest, and available versions."""
        mgr = _get_timeline_project()
        manifest = mgr.get_manifest()
        curr_ir = mgr.get_current_timeline()
        return {
            "manifest": manifest,
            "timeline": curr_ir,
        }

    @app.post("/api/editor/event/update")
    async def update_timeline_event(payload: dict):
        """Updates properties of a timeline event and creates a new version branch."""
        mgr = _get_timeline_project()
        event_id = payload.get("event_id")
        track_type = payload.get("track_type", "TEXT")
        updates = payload.get("updates", {})
        description = payload.get("description", f"Updated {event_id}")

        timeline = mgr.get_current_timeline()
        found = False

        if track_type == "TEXT":
            for track in timeline.get("text_tracks", []):
                for ev in track.get("events", []):
                    if ev.get("event_id") == event_id:
                        ev.update(updates)
                        found = True
                        break
                if found:
                    break
        elif track_type in ("VIDEO", "BROLL"):
            for track in timeline.get("video_tracks", []):
                for cl in track.get("clips", []):
                    if cl.get("clip_id") == event_id:
                        cl.update(updates)
                        found = True
                        break
                if found:
                    break
        elif track_type in ("SFX", "VOICE"):
            for track in timeline.get("audio_tracks", []):
                for cl in track.get("clips", []):
                    if cl.get("clip_id") == event_id:
                        cl.update(updates)
                        found = True
                        break
                if found:
                    break

        if not found:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found in {track_type} tracks.")

        new_ver = mgr.save_new_version(timeline, description=description)
        return {"status": "success", "version": new_ver, "timeline": timeline}

    @app.post("/api/editor/event/delete")
    async def delete_timeline_event(payload: dict):
        """Removes a clip or text event from the timeline and saves a new version."""
        mgr = _get_timeline_project()
        event_id = payload.get("event_id")
        track_type = payload.get("track_type", "TEXT")
        description = payload.get("description", f"Deleted {event_id}")

        timeline = mgr.get_current_timeline()
        if track_type == "TEXT":
            for track in timeline.get("text_tracks", []):
                track["events"] = [ev for ev in track.get("events", []) if ev.get("event_id") != event_id]
        elif track_type in ("VIDEO", "BROLL"):
            for track in timeline.get("video_tracks", []):
                track["clips"] = [cl for cl in track.get("clips", []) if cl.get("clip_id") != event_id]
        elif track_type in ("SFX", "VOICE"):
            for track in timeline.get("audio_tracks", []):
                track["clips"] = [cl for cl in track.get("clips", []) if cl.get("clip_id") != event_id]

        new_ver = mgr.save_new_version(timeline, description=description)
        return {"status": "deleted", "version": new_ver, "timeline": timeline}

    @app.post("/api/editor/version/rollback")
    async def rollback_version(payload: dict):
        """Rollbacks the active project version to a target version."""
        version = payload.get("version")
        if not version:
            raise HTTPException(status_code=400, detail="Missing target version.")
        mgr = _get_timeline_project()
        timeline = mgr.rollback_to_version(version)
        return {"status": "success", "current_version": version, "timeline": timeline}

    @app.post("/api/editor/undo")
    async def undo_timeline_change():
        mgr = _get_timeline_project()
        timeline = mgr.undo()
        if not timeline:
            raise HTTPException(status_code=400, detail="Nothing to undo.")
        return {"status": "success", "timeline": timeline}

    @app.post("/api/editor/redo")
    async def redo_timeline_change():
        mgr = _get_timeline_project()
        timeline = mgr.redo()
        if not timeline:
            raise HTTPException(status_code=400, detail="Nothing to redo.")
        return {"status": "success", "timeline": timeline}

    @app.post("/api/editor/rerender")
    async def selective_rerender(payload: dict):
        """Executes selective re-render based on requested mode: GRAPHICS_ONLY, AUDIO_ONLY, or FULL."""
        mode = payload.get("mode", "GRAPHICS_ONLY")
        mgr = _get_timeline_project()
        if mode == "GRAPHICS_ONLY":
            res = mgr.rerender_graphics_only()
        elif mode == "AUDIO_ONLY":
            res = mgr.rerender_audio_only()
        else:
            res = mgr.rerender_full()
        return res

    @app.get("/api/editor/trace/{event_id}")
    async def get_decision_trace(event_id: str):
        """Returns the full AI decision trace graph for the specified event or clip."""
        mgr = _get_timeline_project()
        trace = mgr.build_traceability_graph(event_id)
        return trace

    @app.get("/api/editor/export/{fmt}")
    async def export_timeline_format(fmt: str):
        mgr = _get_timeline_project()
        if fmt.lower() == "edl":
            content = mgr.export_edl()
            return Response(content=content, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=timeline_export.edl"})
        elif fmt.lower() == "srt":
            content = mgr.export_srt()
            return Response(content=content, media_type="text/plain", headers={"Content-Disposition": "attachment; filename=subtitles_export.srt"})
        elif fmt.lower() == "mp4":
            mp4_path = mgr.project_dir / "final_reel_edited.mp4"
            if not mp4_path.exists():
                mp4_path = mgr.context_dir / "final_reel_edited.mp4"
            if not mp4_path.exists():
                raise HTTPException(status_code=404, detail="Rendered MP4 file not found.")
            return FileResponse(str(mp4_path), media_type="video/mp4", filename="final_reel_edited.mp4")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format '{fmt}'")

    return app

