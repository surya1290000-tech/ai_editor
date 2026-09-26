"""
core/timeline/__init__.py

Authoritative Timeline IR, Timecode Remapping, Compositing, Render Compiler & Edit Validator.
"""

from core.timeline.models import (
    TimelineIR, VideoTrack, VideoClip, TextTrack, TextEvent,
    AudioTrack, AudioClip, BrollTrack, BrollClip,
    Transition, Transform2D, CropRect,
)
from core.timeline.remapper import TimecodeRemapper
from core.timeline.reframe import auto_reframe, ReframePlan
from core.timeline.captions import CaptionCompiler
from core.timeline.broll import BrollAssetResolver
from core.timeline.audio import AudioTimelineBuilder
from core.timeline.composer import TimelineComposer
from core.timeline.compiler import RenderCompiler
from core.timeline.validator import EditValidator, ValidationReport

__all__ = [
    "TimelineIR",
    "VideoTrack",
    "VideoClip",
    "TextTrack",
    "TextEvent",
    "AudioTrack",
    "AudioClip",
    "BrollTrack",
    "BrollClip",
    "Transition",
    "Transform2D",
    "CropRect",
    "TimecodeRemapper",
    "auto_reframe",
    "ReframePlan",
    "CaptionCompiler",
    "BrollAssetResolver",
    "AudioTimelineBuilder",
    "TimelineComposer",
    "RenderCompiler",
    "EditValidator",
    "ValidationReport",
]
