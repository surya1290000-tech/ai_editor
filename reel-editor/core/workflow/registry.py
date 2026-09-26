"""
core/workflow/registry.py

Node Registry: defines all available nodes and builds the default Reel analysis
pipeline graph with proper edges and port contracts.
"""

from __future__ import annotations

from typing import Callable, Optional, Any

from core.workflow.models import (
    NodeDefinition, NodePort, PortDirection, WorkflowGraph,
)


# ─── Node Executor Type ─────────────────────────────────────────────────────

# An executor is a callable: (ctx: ExecutionContext) -> dict[str, Any]
# where the returned dict maps output port names to their values.
NodeExecutor = Callable  # We'll type this more precisely in engine.py


class NodeRegistry:
    """
    Central registry of all known node types.
    Each registration is: (NodeDefinition template, executor function).
    """

    def __init__(self):
        self._nodes: dict[str, tuple[NodeDefinition, Optional[NodeExecutor]]] = {}

    def register(self, node_def: NodeDefinition, executor: Optional[NodeExecutor] = None):
        self._nodes[node_def.node_id] = (node_def, executor)

    def get_definition(self, node_id: str) -> Optional[NodeDefinition]:
        entry = self._nodes.get(node_id)
        return entry[0] if entry else None

    def get_executor(self, node_id: str) -> Optional[NodeExecutor]:
        entry = self._nodes.get(node_id)
        return entry[1] if entry else None

    def list_nodes(self) -> list[str]:
        return list(self._nodes.keys())


def _in(name, dtype, desc="", required=True):
    return NodePort(name=name, direction=PortDirection.INPUT, data_type=dtype, description=desc, required=required)

def _out(name, dtype, desc=""):
    return NodePort(name=name, direction=PortDirection.OUTPUT, data_type=dtype, description=desc)


def build_default_graph() -> WorkflowGraph:
    """
    Construct the default Reel analysis pipeline DAG.
    Returns a WorkflowGraph with all nodes and edges pre-wired.
    """
    graph = WorkflowGraph(name="Reel Analysis Pipeline")

    # ── Source Node ───────────────────────────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_raw_video",
        title="Raw Video Input",
        description="Source video file uploaded by the user",
        agent_group="Input",
        category="source",
        icon="🎥",
        tool="file_system",
        outputs=[_out("video_file", "file:video/*", "Source video file path")],
        position_x=400, position_y=50,
    ))

    # ── Agent 1: Stage 1 — Media Extraction ──────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s1_media",
        title="Media Demuxing",
        description="Extract audio, frames, and metadata via FFmpeg + PySceneDetect",
        agent_group="Agent 1",
        category="extraction",
        icon="📦",
        tool="FFmpeg + PySceneDetect",
        inputs=[_in("video_file", "file:video/*", "Source video")],
        outputs=[
            _out("audio_wav", "file:audio/wav", "16kHz mono WAV"),
            _out("frames_dir", "dir:frames", "Extracted JPEG frames (1fps)"),
            _out("media_meta", "json:media_meta", "Video metadata & scene info"),
        ],
        position_x=400, position_y=180,
    ))

    # ── Agent 1: Stage 2 — Speech ────────────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s2_speech",
        title="Speech Extraction",
        description="Word-level transcription with timestamps, pauses, filler detection",
        agent_group="Agent 1",
        category="analysis",
        icon="🗣️",
        tool="faster-whisper",
        inputs=[_in("audio_wav", "file:audio/wav", "Audio file")],
        outputs=[_out("transcript_raw", "json:transcript", "Word-level transcript")],
        position_x=150, position_y=330,
    ))

    # ── Agent 1: Stage 3 — Audio ────────────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s3_audio",
        title="Audio Signal Analysis",
        description="RMS energy, pitch, emphasis scoring, speaking rate",
        agent_group="Agent 1",
        category="analysis",
        icon="🎵",
        tool="librosa",
        inputs=[
            _in("audio_wav", "file:audio/wav", "Audio file"),
            _in("transcript_raw", "json:transcript", "Word list for per-word alignment"),
        ],
        outputs=[_out("audio_features", "json:audio_features", "Per-word energy, pitch, emphasis")],
        position_x=400, position_y=330,
    ))

    # ── Agent 1: Stage 4 — Visual ────────────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s4_visual",
        title="Visual CV Analysis",
        description="Face detection, framing, optical flow, quality metrics",
        agent_group="Agent 1",
        category="analysis",
        icon="👁️",
        tool="MediaPipe + OpenCV",
        inputs=[
            _in("frames_dir", "dir:frames", "Extracted frames directory"),
            _in("media_meta", "json:media_meta", "Video metadata"),
        ],
        outputs=[_out("visual_features", "json:visual_features", "Per-frame CV analysis")],
        position_x=650, position_y=330,
    ))

    # ── Agent 1: Stage 5 — Feature Fusion ────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s5_fusion",
        title="Feature Fusion",
        description="Merge speech + audio + visual into unified timeline, classify pauses",
        agent_group="Agent 1",
        category="fusion",
        icon="⚡",
        tool="Python (deterministic)",
        inputs=[
            _in("transcript_raw", "json:transcript"),
            _in("audio_features", "json:audio_features"),
            _in("visual_features", "json:visual_features"),
            _in("media_meta", "json:media_meta"),
        ],
        outputs=[
            _out("fused_timeline", "json:fused_timeline", "Unified multi-modal timeline"),
            _out("llm_input", "json:llm_input", "Humanized briefing for LLM"),
        ],
        position_x=400, position_y=480,
    ))

    # ── Agent 1: Stage 6 — Story Analysis ────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s6_story",
        title="Story & Semantic Arc",
        description="LLM-powered narrative structure, hook, climax, emotional arc analysis",
        agent_group="Agent 1",
        category="analysis",
        icon="🧠",
        tool="Ollama LLM",
        inputs=[
            _in("llm_input", "json:llm_input"),
            _in("fused_timeline", "json:fused_timeline"),
        ],
        outputs=[_out("story_analysis", "json:story_analysis")],
        position_x=400, position_y=600,
    ))

    # ── Agent 1: Stage 7 — Opportunity Engine ────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s7_opportunities",
        title="Opportunity Engine",
        description="Identify punch-in, B-roll, caption, and cut candidates with evidence",
        agent_group="Agent 1",
        category="analysis",
        icon="🎯",
        tool="Rule Engine + LLM tags",
        inputs=[
            _in("fused_timeline", "json:fused_timeline"),
            _in("story_analysis", "json:story_analysis"),
        ],
        outputs=[_out("opportunity_inventory", "json:opportunity_inventory")],
        position_x=400, position_y=720,
    ))

    # ── Agent 1: Stage 8 — Blueprint Generator ───────────────────
    graph.add_node(NodeDefinition(
        node_id="node_s8_blueprint",
        title="Edit Blueprint",
        description="Assemble final edit blueprint and interactive HTML report",
        agent_group="Agent 1",
        category="output",
        icon="📋",
        tool="Jinja2 + JSON",
        inputs=[
            _in("opportunity_inventory", "json:opportunity_inventory"),
            _in("story_analysis", "json:story_analysis"),
            _in("fused_timeline", "json:fused_timeline"),
        ],
        outputs=[
            _out("edit_blueprint", "json:edit_blueprint"),
            _out("report_html", "file:text/html"),
        ],
        position_x=400, position_y=840,
    ))

    # ── Agent 2: Creative Director ──────────────────────────────
    graph.add_node(NodeDefinition(
        node_id="node_agent2",
        title="Creative Director",
        description="Formulate cut schedule, punch-in framing, and subtitle styling",
        agent_group="Agent 2",
        category="creative",
        icon="🎨",
        tool="Creative Policy Engine",
        is_planned=False,
        inputs=[
            _in("edit_blueprint", "json:edit_blueprint"),
            _in("fused_timeline", "json:fused_timeline"),
        ],
        outputs=[_out("creative_decisions", "json:creative_decisions")],
        position_x=400, position_y=970,
    ))

    # ── Agent 3: Timeline Assembler & Video Renderer ────────────
    graph.add_node(NodeDefinition(
        node_id="node_agent3",
        title="Timeline Assembler & Renderer",
        description="Render final edited MP4, subtitle track, and DaVinci Resolve EDL",
        agent_group="Agent 3",
        category="execution",
        icon="🎬",
        tool="FFmpeg + CMX 3600 EDL",
        is_planned=False,
        inputs=[
            _in("creative_decisions", "json:creative_decisions"),
            _in("video_file", "file:video/*"),
            _in("fused_timeline", "json:fused_timeline"),
        ],
        outputs=[
            _out("final_video", "file:video/mp4"),
            _out("edl_file", "file:text/plain"),
            _out("subtitles", "file:text/srt"),
        ],
        position_x=400, position_y=1100,
    ))

    # ── Edges (data flow) ────────────────────────────────────────
    # Raw Video → S1
    graph.add_edge("node_raw_video", "video_file", "node_s1_media", "video_file")

    # S1 → parallel branches
    graph.add_edge("node_s1_media", "audio_wav", "node_s2_speech", "audio_wav")
    graph.add_edge("node_s1_media", "audio_wav", "node_s3_audio", "audio_wav")
    graph.add_edge("node_s1_media", "frames_dir", "node_s4_visual", "frames_dir")
    graph.add_edge("node_s1_media", "media_meta", "node_s4_visual", "media_meta")

    # S2 → S3 (transcript needed for per-word alignment)
    graph.add_edge("node_s2_speech", "transcript_raw", "node_s3_audio", "transcript_raw")

    # S2, S3, S4 → S5
    graph.add_edge("node_s2_speech", "transcript_raw", "node_s5_fusion", "transcript_raw")
    graph.add_edge("node_s3_audio", "audio_features", "node_s5_fusion", "audio_features")
    graph.add_edge("node_s4_visual", "visual_features", "node_s5_fusion", "visual_features")
    graph.add_edge("node_s1_media", "media_meta", "node_s5_fusion", "media_meta")

    # S5 → S6
    graph.add_edge("node_s5_fusion", "llm_input", "node_s6_story", "llm_input")
    graph.add_edge("node_s5_fusion", "fused_timeline", "node_s6_story", "fused_timeline")

    # S5, S6 → S7
    graph.add_edge("node_s5_fusion", "fused_timeline", "node_s7_opportunities", "fused_timeline")
    graph.add_edge("node_s6_story", "story_analysis", "node_s7_opportunities", "story_analysis")

    # S6, S7, S5 → S8
    graph.add_edge("node_s7_opportunities", "opportunity_inventory", "node_s8_blueprint", "opportunity_inventory")
    graph.add_edge("node_s6_story", "story_analysis", "node_s8_blueprint", "story_analysis")
    graph.add_edge("node_s5_fusion", "fused_timeline", "node_s8_blueprint", "fused_timeline")

    # S8, S5 → Agent 2
    graph.add_edge("node_s8_blueprint", "edit_blueprint", "node_agent2", "edit_blueprint")
    graph.add_edge("node_s5_fusion", "fused_timeline", "node_agent2", "fused_timeline")

    # Agent 2, Raw Video, S5 → Agent 3
    graph.add_edge("node_agent2", "creative_decisions", "node_agent3", "creative_decisions")
    graph.add_edge("node_raw_video", "video_file", "node_agent3", "video_file")
    graph.add_edge("node_s5_fusion", "fused_timeline", "node_agent3", "fused_timeline")


    return graph
