"""
tests/test_agent3_creative_director.py

Acceptance Test Suite for Phase 4: Agent 3 (Creative Director)
──────────────────────────────────────────────────────────────
Verifies all 13 strict acceptance criteria:
1. StyleProfile serialization
2. CreativeDecision serialization
3. Agent 2 -> Agent 3 traceability
4. Typography profile validity
5. Telugu font resolution
6. Safe-zone validation
7. Animation parameter validity
8. Punch-in framing bounds
9. B-roll unresolved asset behavior
10. SFX asset resolution
11. Overlap/conflict resolution
12. Timeline IR conversion
13. Deterministic-equivalent creative output
"""

import json
import pytest
from pathlib import Path

from agents.agent3_creative.styles import (
    StyleProfile, StyleProfileName, get_style_profile
)
from agents.agent3_creative.models import (
    CreativePlan, CreativeDecision, CreativeOperation, CreativeStatus,
    TypographySpec, AnimationSpec, PunchInSpec, BrollSpec, SfxSpec, AudioSpec
)
from agents.agent3_creative.assets import LocalAssetLibraryResolver
from agents.agent3_creative.creative_director import CreativeDirector, is_telugu_text
from agents.agent3_creative.timeline_compiler import CreativeTimelineCompiler
from agents.agent3_creative.validator import CreativeDirectorValidator
from core.timeline.models import TimelineIR


@pytest.fixture
def mock_agent2_decisions():
    """Provides representative Agent 2 decisions matching Surya.mp4 output structure."""
    return {
        "version": "3.0.0",
        "source_video": "Surya.mp4",
        "duration_seconds": 90.833,
        "accepted_decisions": [
            {
                "decision_id": "edit_001",
                "operation": "HERO_TEXT",
                "start": 0.0,
                "end": 2.5,
                "text_content": "WORLD WITHOUT EMPATHY",
                "story_act": "HOOK",
                "confidence": 0.92,
                "source_evidence_ids": ["ev_001", "ev_024"],
                "source_opportunity_ids": ["opp_022"],
                "reason": "Primary narrative title card in HOOK act.",
            },
            {
                "decision_id": "edit_006",
                "operation": "BROLL_CONTEXT",
                "start": 10.9,
                "end": 15.7,
                "concept_query": "daily human interaction conversation",
                "story_act": "CONTEXT",
                "confidence": 0.78,
                "source_evidence_ids": ["ev_052"],
                "source_opportunity_ids": ["opp_012"],
                "reason": "Contextual B-roll cutaway.",
            },
            {
                "decision_id": "edit_012",
                "operation": "EMPHASIS_TEXT",
                "start": 60.18,
                "end": 60.46,
                "text_content": "EMPATHY",
                "story_act": "PROBLEM",
                "confidence": 0.94,
                "source_evidence_ids": ["ev_025"],
                "source_opportunity_ids": ["opp_018"],
                "reason": "Vocal emphasis token.",
            },
            {
                "decision_id": "edit_013",
                "operation": "PUNCH_IN",
                "start": 60.18,
                "end": 62.46,
                "strength": "SUBTLE",
                "story_act": "PROBLEM",
                "confidence": 0.94,
                "source_evidence_ids": ["ev_042"],
                "source_opportunity_ids": ["opp_002"],
                "reason": "Subtle framing jump cut.",
            },
            {
                "decision_id": "edit_024",
                "operation": "SFX_IMPACT",
                "start": 76.86,
                "end": 77.21,
                "sound_cue": "impact_subtle",
                "story_act": "REALIZATION",
                "confidence": 0.99,
                "source_evidence_ids": ["ev_049"],
                "source_opportunity_ids": ["opp_009"],
                "reason": "Acoustic accent on inflection point.",
            },
            {
                "decision_id": "edit_026",
                "operation": "PUNCH_IN",
                "start": 76.86,
                "end": 79.18,
                "strength": "STRONG",
                "story_act": "REALIZATION",
                "confidence": 0.99,
                "source_evidence_ids": ["ev_048"],
                "source_opportunity_ids": ["opp_005"],
                "reason": "Strong framing jump cut at realization payoff.",
            },
            {
                "decision_id": "edit_027",
                "operation": "REMOVE_DEAD_SPACE",
                "start": 77.18,
                "end": 77.96,
                "story_act": "REALIZATION",
                "confidence": 0.92,
                "source_evidence_ids": ["ev_004"],
                "source_opportunity_ids": ["opp_001"],
                "reason": "Remove 0.78s dead space retaining 0.10s buffer.",
            },
        ],
        "no_edit_decisions": [
            {
                "decision_id": "keep_004",
                "operation": "NO_EDIT",
                "start": 0.18,
                "end": 3.28,
                "story_act": "HOOK",
                "confidence": 0.75,
                "source_evidence_ids": ["ev_057"],
                "source_opportunity_ids": ["opp_024"],
                "reason": "Natural fluent delivery; preserved untouched.",
            }
        ]
    }


# ── Test 1: StyleProfile Serialization ──────────────────────────────────────────
def test_style_profile_serialization():
    for profile_name in StyleProfileName:
        profile = get_style_profile(profile_name)
        assert profile is not None
        d = profile.to_dict()
        assert d["name"] == profile_name.value
        json_str = json.dumps(d)
        assert len(json_str) > 50


# ── Test 2: CreativeDecision Serialization ─────────────────────────────────────
def test_creative_decision_serialization():
    dec = CreativeDecision(
        creative_id="creative_001",
        source_decision_id="edit_001",
        operation=CreativeOperation.HERO_TEXT,
        time_range=[0.0, 2.5],
        style_profile="EDITORIAL_CINEMATIC",
        status=CreativeStatus.RESOLVED,
        confidence=0.95,
        reason="Hero title card",
        story_act="HOOK",
        text_content="HERO STATEMENT",
        typography=TypographySpec(
            font="Arial Black",
            font_fallback=["Nirmala UI", "Arial"],
            size=84,
            weight="bold",
        ),
        animation=AnimationSpec(preset="pop", easing="ease_out", duration=0.28),
    )
    plan = CreativePlan(
        source_video="test.mp4",
        duration_seconds=90.0,
        style_profile="EDITORIAL_CINEMATIC",
        decisions=[dec],
    )
    d = plan.to_dict()
    roundtrip = CreativePlan.from_dict(d)
    assert len(roundtrip.decisions) == 1
    assert roundtrip.decisions[0].creative_id == "creative_001"
    assert roundtrip.decisions[0].typography.font == "Arial Black"


# ── Test 3: Agent 2 -> Agent 3 Traceability ────────────────────────────────────
def test_agent2_traceability(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    expected_ids = {d["decision_id"] for d in mock_agent2_decisions["accepted_decisions"]} | \
                   {d["decision_id"] for d in mock_agent2_decisions["no_edit_decisions"]}
    actual_source_ids = {d.source_decision_id for d in plan.decisions}

    assert expected_ids.issubset(actual_source_ids)
    assert len(plan.decisions) >= len(expected_ids)


# ── Test 4: Typography Profile Validity ─────────────────────────────────────────
def test_typography_profile_validity(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    hero_dec = next(d for d in plan.decisions if d.operation == CreativeOperation.HERO_TEXT)
    assert hero_dec.typography is not None
    assert hero_dec.typography.font != ""
    assert hero_dec.typography.size >= 70
    assert len(hero_dec.typography.font_fallback) > 0
    assert hero_dec.typography.safe_zone is True


# ── Test 5: Telugu Font Resolution ─────────────────────────────────────────────
def test_telugu_font_resolution():
    assert is_telugu_text("ధన్యవాదాలు") is True
    assert is_telugu_text("Hello World") is False
    assert is_telugu_text("Hello ధన్యవాదాలు World") is True

    telugu_decisions = {
        "source_video": "test.mp4",
        "duration_seconds": 60.0,
        "accepted_decisions": [
            {
                "decision_id": "edit_tel_001",
                "operation": "HERO_TEXT",
                "start": 0.0,
                "end": 2.5,
                "text_content": "సానుభూతి లేని ప్రపంచం",  # Telugu text
                "story_act": "HOOK",
                "confidence": 0.90,
            }
        ],
        "no_edit_decisions": []
    }
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=telugu_decisions)

    hero = plan.decisions[0]
    assert hero.typography.is_indic_script is True
    assert hero.typography.font == "Nirmala UI"
    assert hero.typography.vertical_line_padding >= 1.05


# ── Test 6: Safe-Zone Validation ───────────────────────────────────────────────
def test_safe_zone_validation(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    for dec in plan.decisions:
        if dec.typography:
            assert dec.typography.safe_zone is True


# ── Test 7: Animation Parameter Validity ───────────────────────────────────────
def test_animation_parameter_validity(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    for dec in plan.decisions:
        if dec.animation:
            assert dec.animation.easing in CreativeDirectorValidator.VALID_EASINGS
            assert 0.0 < dec.animation.duration <= 2.0


# ── Test 8: Punch-In Framing Bounds ────────────────────────────────────────────
def test_punch_in_framing_bounds(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(
        edit_decisions_data=mock_agent2_decisions,
        visual_features={"face_tracking": {"primary_face_center_x": 0.45, "primary_face_center_y": 0.38}}
    )

    punches = [d for d in plan.decisions if d.operation == CreativeOperation.PUNCH_IN]
    assert len(punches) == 2

    subtle_p = next(p for p in punches if p.punch_in.conceptual_strength == "SUBTLE")
    strong_p = next(p for p in punches if p.punch_in.conceptual_strength == "STRONG")

    assert subtle_p.punch_in.scale < strong_p.punch_in.scale
    assert 1.05 <= subtle_p.punch_in.scale <= 1.20
    assert 1.20 <= strong_p.punch_in.scale <= 1.40

    for p in punches:
        assert 0.10 <= p.punch_in.anchor["x"] <= 0.90
        assert 0.10 <= p.punch_in.anchor["y"] <= 0.90
        assert p.punch_in.transition == "JUMP_CUT"


# ── Test 9: B-Roll Unresolved Asset Behavior ───────────────────────────────────
def test_broll_unresolved_asset_behavior(mock_agent2_decisions, tmp_path):
    empty_resolver = LocalAssetLibraryResolver(workspace_root=tmp_path)
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC", asset_resolver=empty_resolver)
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    broll = next(d for d in plan.decisions if d.operation == CreativeOperation.BROLL_CONTEXT)
    assert broll.broll is not None
    # No fake completion: since no file exists in empty directory, must be UNRESOLVED_ASSET
    assert broll.status == CreativeStatus.UNRESOLVED_ASSET
    assert broll.broll.status == "UNRESOLVED_ASSET"
    assert broll.broll.asset_query != ""
    assert broll.broll.required_duration > 0.0


# ── Test 10: SFX Asset Resolution ──────────────────────────────────────────────
def test_sfx_asset_resolution(mock_agent2_decisions):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    sfx = next(d for d in plan.decisions if d.operation == CreativeOperation.SFX_IMPACT)
    assert sfx.sfx is not None
    assert sfx.sfx.category == "impact"
    assert sfx.sfx.gain_db <= 0.0  # Respects voice priority
    assert sfx.status in [CreativeStatus.RESOLVED, CreativeStatus.UNRESOLVED_ASSET]


# ── Test 11: Overlap / Conflict Resolution ─────────────────────────────────────
def test_conflict_resolution_hierarchy():
    conflicting_decisions = {
        "source_video": "test.mp4",
        "duration_seconds": 30.0,
        "accepted_decisions": [
            {
                "decision_id": "hero_01",
                "operation": "HERO_TEXT",
                "start": 0.0,
                "end": 3.0,
                "text_content": "IMPORTANT TITLE",
                "story_act": "HOOK",
                "confidence": 0.95,
            },
            {
                "decision_id": "broll_01",
                "operation": "BROLL_CONTEXT",
                "start": 1.0,
                "end": 4.0,  # Overlaps HeroText!
                "concept_query": "urban street crowd",
                "story_act": "HOOK",
                "confidence": 0.70,
            }
        ],
        "no_edit_decisions": []
    }
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=conflicting_decisions)

    broll_dec = next(d for d in plan.decisions if d.operation == CreativeOperation.BROLL_CONTEXT)
    # Under hierarchy rule 1 (HeroText > B-roll), B-roll is marked CONFLICT
    assert broll_dec.status == CreativeStatus.CONFLICT
    assert "HeroText" in broll_dec.reason


# ── Test 12: Timeline IR Conversion ────────────────────────────────────────────
def test_timeline_ir_conversion(mock_agent2_decisions, tmp_path):
    director = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan = director.direct(edit_decisions_data=mock_agent2_decisions)

    compiler = CreativeTimelineCompiler()
    timeline_ir, reframe_plan = compiler.compile(
        creative_plan=plan,
        source_video_path="Surya.mp4",
        media_meta={"width": 2160, "height": 1192, "duration_seconds": 90.833, "fps": 30.0},
    )

    assert isinstance(timeline_ir, TimelineIR)
    assert timeline_ir.timeline_duration > 0.0
    assert len(timeline_ir.video_tracks[0].clips) > 0
    assert len(timeline_ir.text_tracks[0].events) > 0
    # Check that HeroText event exists
    hero_events = [e for e in timeline_ir.text_tracks[0].events if e.position_align == 5]
    assert len(hero_events) >= 1
    assert hero_events[0].text == "WORLD WITHOUT EMPATHY"


# ── Test 13: Deterministic-Equivalent Creative Output ──────────────────────────
def test_deterministic_equivalent_output(mock_agent2_decisions):
    director1 = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan1 = director1.direct(edit_decisions_data=mock_agent2_decisions)

    director2 = CreativeDirector(style_profile="EDITORIAL_CINEMATIC")
    plan2 = director2.direct(edit_decisions_data=mock_agent2_decisions)

    assert plan1.to_dict() == plan2.to_dict()

    validator = CreativeDirectorValidator()
    report = validator.validate(creative_plan=plan1, edit_decisions=mock_agent2_decisions)
    assert report["passed"] is True
    assert report["total_checks"] == 13
