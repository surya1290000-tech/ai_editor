"""
tests/test_generalization.py

Unit tests for Generalization: Smart Visual Sampling, Dynamic Key Moments, and Decoupled Narrative Partitioning
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from core.evidence.smart_sampler import SmartVisualSampler
from core.review.reviewer import EditorialReviewer
from core.evidence.story_analyzer import StoryObserver
from core.evidence.models import StoryObservation, StoryObservationsInventory


def test_smart_visual_sampler_logic(tmp_path):
    """Verifies that SmartVisualSampler creates base cadence, event drill-downs, and composites."""
    sampler = SmartVisualSampler(base_cadence_sec=5.0, max_reasoning_frames=10)

    # Create mock 1-second black mp4 video
    test_video = tmp_path / "dummy.mp4"
    # Create via ffmpeg
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=720x1280:d=12",
        "-c:v", "libx264", "-t", "12", str(test_video)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    audio_features = {
        "silence_regions": [{"start": 4.0, "end": 5.2}],
        "energy_peaks": [{"timestamp": 8.0, "level_db": -12.0}],
    }
    transcript_raw = {
        "words": [
            {"word": "Hello", "start": 0.5, "end": 1.0, "confidence": 0.95},
            {"word": "uncertain", "start": 6.0, "end": 6.5, "confidence": 0.40},
        ]
    }
    media_meta = {"duration_seconds": 12.0, "width": 720, "height": 1280}

    manifest = sampler.sample_video(
        video_path=test_video,
        output_dir=tmp_path,
        audio_features=audio_features,
        transcript_raw=transcript_raw,
        media_meta=media_meta,
    )

    assert manifest["source_video"] == "dummy.mp4"
    assert manifest["total_sampled_keyframes"] > 0
    assert manifest["total_sampled_keyframes"] <= 10
    assert "HEROTEXT_CANDIDATE" in [k["category"] for k in manifest["sampled_keyframes"]]
    assert (tmp_path / "smart_visual_samples" / "smart_visual_manifest.json").exists()


def test_reviewer_dynamic_moment_discovery():
    """Verifies that EditorialReviewer dynamically extracts moments for a generic new video."""
    reviewer = EditorialReviewer()

    edit_decisions = {
        "accepted_decisions": [
            {
                "decision_id": "new_hero_01",
                "operation": "HERO_TEXT",
                "start": 0.0,
                "end": 2.0,
                "confidence": 0.92,
                "importance": 0.85,
                "text_content": "STARTUP INSIGHTS",
                "reason": "Opening hook for startup discussion"
            },
            {
                "decision_id": "new_broll_01",
                "operation": "BROLL_CONTEXT",
                "start": 15.0,
                "end": 19.5,
                "confidence": 0.88,
                "importance": 0.80,
                "concept_query": "office laptop coding",
                "reason": "Visual cutaway illustrating work environment"
            },
            {
                "decision_id": "new_punch_01",
                "operation": "PUNCH_IN",
                "start": 35.0,
                "end": 37.5,
                "confidence": 0.85,
                "importance": 0.82,
                "strength": "strong",
                "reason": "Payoff focal emphasis"
            },
        ],
        "preserved_no_edit_spans": [
            {"start": 5.0, "end": 12.0, "reason": "Uninterrupted speech segment"}
        ]
    }

    moments = reviewer._discover_key_moments(
        edit_decisions=edit_decisions,
        creative_plan={},
        timeline_ir={},
        duration=45.0,
    )

    assert len(moments) == 4
    # Check that dynamic labels were applied without Surya strings
    ops = [m["operation"] for m in moments]
    assert "HERO_TEXT" in ops
    assert "BROLL_CONTEXT" in ops
    assert "PUNCH_IN" in ops
    assert "NO_EDIT" in ops

    hero_moment = next(m for m in moments if m["operation"] == "HERO_TEXT")
    assert "STARTUP INSIGHTS" in hero_moment["label"]
    assert hero_moment["interval"] == [0.0, 2.0]
    assert hero_moment["raw_timestamp"] == 1.0


def test_story_observer_decoupled_partitioning():
    """Verifies that StoryObserver partitions narrative acts without any hardcoded strings."""
    class DummySentence:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    # Test Case A: Single long take (1 sentence, 60 seconds)
    sentences_single = [DummySentence("This is a continuous sixty second monologue about software engineering.", 0.0, 60.0)]
    observer = StoryObserver(duration=60.0)

    acts_single = observer._deterministic_partition(sentences_single)
    assert len(acts_single) == 5
    assert acts_single[0]["act"] == "HOOK"
    assert acts_single[4]["act"] == "PAYOFF"
    assert "software engineering" in acts_single[0]["description"]
    assert "empathy" not in acts_single[0]["description"].lower()

    # Test Case B: 10 sentences
    sentences_multi = [
        DummySentence(f"Sentence {i} discussing architecture and scalability.", i * 5.0, (i + 1) * 5.0)
        for i in range(10)
    ]
    acts_multi = observer._deterministic_partition(sentences_multi)
    assert len(acts_multi) == 5
    assert acts_multi[0]["act"] == "HOOK"
    assert acts_multi[4]["act"] == "PAYOFF"
    assert "architecture and scalability" in acts_multi[0]["description"]
    assert "empathy" not in str(acts_multi).lower()
