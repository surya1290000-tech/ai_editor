"""
reel_editor/server.py

Entry point: python -m reel_editor.server

Starts the local FastAPI server serving the visual workflow canvas
and the real-time workflow execution API.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from core.workflow.server import create_app

app = create_app()


def main():
    print()
    print("=" * 60)
    print("  AI REEL EDITOR — Workflow Server")
    print("=" * 60)
    print()
    print("  Open in browser:  http://localhost:8000")
    print()
    print("  API endpoints:")
    print("    GET  /api/workflow/graph       — Current workflow graph")
    print("    GET  /api/workflow/events      — SSE event stream")
    print("    POST /api/workflow/run         — Start workflow execution")
    print("    GET  /api/workflow/nodes/{id}  — Node details")
    print()
    print("=" * 60)
    print()

    uvicorn.run(
        "reel_editor.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
