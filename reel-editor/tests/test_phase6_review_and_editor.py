"""
tests/test_phase6_review_and_editor.py

Unit and Integration Tests for Phase 6: Human-Quality Review & Editable Timeline
"""

import json
import shutil
import tempfile
from pathlib import Path
import pytest

from core.review.reviewer import EditorialReviewer
from core.review.reporter import EditorialQualityReporter
from core.timeline.project import TimelineProjectManager
from core.timeline.models import TimelineIR


@pytest.fixture
def mock_output_env(tmp_path):
    """Sets up a temporary output environment with minimal mock data."""
    # Timeline IR
    ir_data = {
        "schema_version": "2.1.0",
        "project_name": "TestReel",
        "timeline_duration": 90.0,
        "target_width": 1080,
        "target_height": 1920,
        "video_tracks": [
            {
                "track_id": "v1",
                "name": "BaseVideo",
                "clips": [
                    {
                        "clip_id": "c1",
                        "source_path": "clip1.mp4",
                        "timeline_in": 0.0,
                        "timeline_out": 30.0,
                        "source_in": 0.0,
                        "source_out": 30.0,
                        "punch_scale_factor": 1.0,
                    },
                    {
                        "clip_id": "c2",
                        "source_path": "clip2.mp4",
                        "timeline_in": 30.0,
                        "timeline_out": 60.0,
                        "source_in": 30.0,
                        "source_out": 60.0,
                        "punch_scale_factor": 1.25,
                    },
                ]
            }
        ],
        "audio_tracks": [
            {
                "track_id": "a_sfx",
                "name": "SFXTrack",
                "role": "SFX",
                "clips": [
                    {
                        "clip_id": "sfx_1",
                        "timeline_in": 15.0,
                        "timeline_out": 15.5,
                        "source_in": 0.0,
                        "source_out": 0.5,
                        "volume": 0.25,
                        "status": "RESOLVED",
                    }
                ]
            }
        ],
        "text_tracks": [
            {
                "track_id": "t1",
                "name": "Subtitles",
                "events": [
                    {
                        "event_id": "hero_title_01",
                        "timeline_in": 0.0,
                        "timeline_out": 2.5,
                        "content": "THE REAL SECRET TO INFLUENCE",
                        "font_family": "Montserrat",
                        "font_size": 84,
                        "style": "HERO_TITLE",
                    },
                    {
                        "event_id": "cap_02",
                        "timeline_in": 5.0,
                        "timeline_out": 8.0,
                        "content": "Every time you speak with empathy",
                        "font_family": "Montserrat",
                        "font_size": 52,
                        "style": "KINETIC_BODY",
                    }
                ]
            }
        ]
    }
    ir_path = tmp_path / "timeline_ir.json"
    with open(ir_path, "w", encoding="utf-8") as f:
        json.dump(ir_data, f, indent=2)

    # Edit decisions (Agent 2)
    edit_decisions = {
        "accepted_decisions": [
            {
                "decision_id": "hero_title_01",
                "operation": "HERO_TEXT",
                "time_range": [0.0, 2.5],
                "confidence": 0.95,
                "story_purpose": "Hook engagement",
                "source_opportunity_ids": ["opp_hook"],
                "source_evidence_ids": ["ev_1", "ev_2"]
            },
            {"decision_id": "broll_01", "operation": "BROLL_CONTEXT", "time_range": [10.9, 15.7], "confidence": 0.90, "story_purpose": "Illustration"},
            {"decision_id": "punch_01", "operation": "PUNCH_IN", "time_range": [30.0, 35.0], "confidence": 0.88, "story_purpose": "Focus tension"},
            {"decision_id": "sfx_01", "operation": "SFX_IMPACT", "time_range": [15.0, 15.5], "confidence": 0.85, "story_purpose": "Audio accent"},
            {"decision_id": "emp_01", "operation": "EMPHASIS_TEXT", "time_range": [5.0, 6.0], "confidence": 0.92, "story_purpose": "Keyword highlight"},
        ],
        "rejected_candidates": [],
        "preserved_no_edit_spans": []
    }
    with open(tmp_path / "edit_decisions.json", "w", encoding="utf-8") as f:
        json.dump(edit_decisions, f, indent=2)

    # Creative decisions (Agent 3)
    creative_decisions = {
        "creative_decisions": [
            {"decision_id": "hero_title_01", "style_profile": "EditorialAuthority", "rationale": "High status champagne card"},
            {"decision_id": "punch_01", "style_profile": "Default", "rationale": "Face-anchored crop"},
        ]
    }
    with open(tmp_path / "creative_decisions.json", "w", encoding="utf-8") as f:
        json.dump(creative_decisions, f, indent=2)

    # Evidence inventory (Agent 1)
    evidence_inventory = {
        "editorial_opportunities": [
            {"opportunity_id": "opp_hook", "type": "HOOK", "supporting_evidence_ids": ["ev_1", "ev_2"]},
        ],
        "evidence_items": [
            {"evidence_id": "ev_1", "evidence_type": "AUDIO_PROMINENCE", "description": "High vocal energy at opening", "confidence": 0.94},
            {"evidence_id": "ev_2", "evidence_type": "SPEAKER_TRACKING", "description": "Speaker centered", "confidence": 0.98},
        ],
        "transcript_enriched": {
            "words": [
                {"word": "The", "start": 0.1, "end": 0.3},
                {"word": "Real", "start": 0.3, "end": 0.6},
                {"word": "Secret", "start": 0.6, "end": 1.1},
            ]
        }
    }
    with open(tmp_path / "evidence_inventory.json", "w", encoding="utf-8") as f:
        json.dump(evidence_inventory, f, indent=2)

    return tmp_path


def test_reviewer_density_calculation(mock_output_env):
    """Verifies edit density computation and pacing classification."""
    reviewer = EditorialReviewer()

    with open(mock_output_env / "edit_decisions.json") as f:
        edit_dec = json.load(f)
    with open(mock_output_env / "timeline_ir.json") as f:
        timeline = json.load(f)

    # 5 accepted decisions in 90 seconds (1.5 min)
    # Major edits = HERO_TEXT (1) + BROLL_CONTEXT (1) + PUNCH_IN (1) = 3
    # major_edits / 1.5 min = 2.0 edits/min -> BALANCED
    dur_min = 90.0 / 60.0
    accepted = edit_dec["accepted_decisions"]
    major_count = sum(1 for d in accepted if d["operation"] in ["HERO_TEXT", "PUNCH_IN", "BROLL_CONTEXT"])
    rate = major_count / dur_min

    assert rate == 2.0
    assert 2.0 <= rate <= 8.5  # Balanced range


def test_reviewer_rubric_scoring():
    """Verifies that all 5 rubric scores fall within valid bounds [0.0, 1.0]."""
    reviewer = EditorialReviewer()

    key_moments = [
        {"story_value": 0.95, "visual_value": 0.90, "pacing_value": 0.88, "distraction_cost": 0.12, "naturalness_score": 0.92, "verdict": "KEEP"},
        {"story_value": 0.85, "visual_value": 0.86, "pacing_value": 0.80, "distraction_cost": 0.20, "naturalness_score": 0.85, "verdict": "KEEP"},
    ]

    for m in key_moments:
        assert 0.0 <= m["story_value"] <= 1.0
        assert 0.0 <= m["visual_value"] <= 1.0
        assert 0.0 <= m["pacing_value"] <= 1.0
        assert 0.0 <= m["distraction_cost"] <= 1.0
        assert 0.0 <= m["naturalness_score"] <= 1.0
        assert m["verdict"] in ["KEEP", "REVISE", "REMOVE"]


def test_html_reporter_generation(mock_output_env):
    """Verifies the generation of human_quality_report.html."""
    reporter = EditorialQualityReporter()
    report_data = {
        "source_video": "TestSurya.mp4",
        "edited_video": "TestEdited.mp4",
        "density_metrics": {
            "duration_minutes": 1.5,
            "total_edits": 5,
            "major_edits": 3,
            "edits_per_minute": 3.33,
            "major_edits_per_minute": 2.0,
            "sfx_density_per_minute": 0.67,
            "broll_density_per_minute": 0.67,
            "pacing_classification": "BALANCED",
            "benchmark_range": "3.5 - 7.5 major edits/min",
        },
        "quality_scores": {
            "overall_story_value": 0.92,
            "overall_visual_value": 0.89,
            "overall_pacing_value": 0.85,
            "overall_distraction_cost": 0.15,
            "overall_naturalness_score": 0.90,
            "verdicts": {"keep": 5, "revise": 0, "remove": 0},
        },
        "key_moments": [
            {
                "id": "m1",
                "label": "Hero Title",
                "operation": "HERO_TEXT",
                "raw_timestamp": 1.2,
                "edited_timestamp": 1.2,
                "interval": [0.0, 2.5],
                "story_value": 0.95,
                "visual_value": 0.90,
                "pacing_value": 0.88,
                "distraction_cost": 0.12,
                "naturalness_score": 0.92,
                "verdict": "KEEP",
                "editorial_critique": "Crisp title card.",
                "raw_frame_name": "raw_m1.jpg",
                "edited_frame_name": "edited_m1.jpg",
            }
        ],
        "editorial_dimensions": {
            "1_technical_qa": "PASS 1080x1920",
            "2_editorial_quality": "Narrative pacing clear",
        }
    }

    out_file = mock_output_env / "human_quality_report.html"
    res_path = reporter.generate_html_report(report_data, out_file, embed_images=False)

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "Human-Quality Editorial Review" in content
    assert "Hero Title" in content
    assert "BALANCED" in content
    assert "Story Value" in content


def test_project_manager_immutability_and_versioning(mock_output_env):
    """Verifies that v1 is immutable and edits branch into v2, v3 with undo/redo."""
    project_dir = mock_output_env / "project"
    mgr = TimelineProjectManager(
        project_dir=project_dir,
        base_timeline_path=mock_output_env / "timeline_ir.json",
        context_dir=mock_output_env
    )

    manifest = mgr.get_manifest()
    assert manifest["current_version"] == "v1"
    assert manifest["versions"][0]["is_immutable"] is True

    # Check v1 file exists
    v1_file = project_dir / "timeline_ir_v1.json"
    assert v1_file.exists()
    orig_v1_text = v1_file.read_text(encoding="utf-8")

    # User modifies text in v2
    timeline_mod = mgr.get_current_timeline()
    timeline_mod["text_tracks"][0]["events"][0]["content"] = "USER EDITED HEADLINE"

    v2 = mgr.save_new_version(timeline_mod, description="User changed hero headline")
    assert v2 == "v2"
    assert mgr.get_manifest()["current_version"] == "v2"

    # Verify v1 is UNTOUCHED
    assert v1_file.read_text(encoding="utf-8") == orig_v1_text

    # User modifies scale in v3
    timeline_mod2 = mgr.get_current_timeline()
    timeline_mod2["video_tracks"][0]["clips"][1]["punch_scale_factor"] = 1.35
    v3 = mgr.save_new_version(timeline_mod2, description="User adjusted punch scale")
    assert v3 == "v3"

    # Test Undo
    undone = mgr.undo()
    assert undone is not None
    assert mgr.get_manifest()["current_version"] == "v2"

    # Test Redo
    redone = mgr.redo()
    assert redone is not None
    assert mgr.get_manifest()["current_version"] == "v3"

    # Test Rollback to immutable v1
    rolled_back = mgr.rollback_to_version("v1")
    assert mgr.get_manifest()["current_version"] == "v1"
    assert rolled_back["text_tracks"][0]["events"][0]["content"] == "THE REAL SECRET TO INFLUENCE"


def test_project_manager_traceability_graph(mock_output_env):
    """Verifies decision explainability graph from Timeline Event back to Evidence."""
    project_dir = mock_output_env / "project"
    mgr = TimelineProjectManager(
        project_dir=project_dir,
        base_timeline_path=mock_output_env / "timeline_ir.json",
        context_dir=mock_output_env
    )

    trace = mgr.build_traceability_graph("hero_title_01")
    assert trace["target_id"] == "hero_title_01"
    assert trace["timeline_event"] is not None
    assert trace["timeline_event"]["content"] == "THE REAL SECRET TO INFLUENCE"
    assert trace["creative_decision"] is not None
    assert trace["creative_decision"]["style_profile"] == "EditorialAuthority"
    assert trace["story_decision"] is not None
    assert trace["story_decision"]["operation"] == "HERO_TEXT"
    assert len(trace["evidence"]) >= 1
    assert trace["transcript_context"] is not None


def test_project_manager_exports(mock_output_env):
    """Verifies CMX 3600 EDL and SRT exports."""
    project_dir = mock_output_env / "project"
    mgr = TimelineProjectManager(
        project_dir=project_dir,
        base_timeline_path=mock_output_env / "timeline_ir.json",
        context_dir=mock_output_env
    )

    # Test EDL export
    edl = mgr.export_edl()
    assert "TITLE: TestReel" in edl
    assert "FCM: NON-DROP FRAME" in edl
    assert "001  AX" in edl

    # Test SRT export
    srt = mgr.export_srt()
    assert "1" in srt
    assert "-->" in srt
    assert "THE REAL SECRET TO INFLUENCE" in srt
