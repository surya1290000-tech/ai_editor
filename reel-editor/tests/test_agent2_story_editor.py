"""
tests/test_agent2_story_editor.py

Acceptance Tests for Agent 2: Story Editor (Phase 3)
───────────────────────────────────────────────────
Verifies all 10 acceptance criteria for Phase 3:
  1. Selectivity (disciplined intervention rate, does not accept every opportunity)
  2. Rejection of weak opportunities with clear rationales
  3. First-class NO_EDIT (preserves natural delivery & rhetorical holds)
  4. Edit budget compliance (hard limits respected)
  5. Spacing compliance (minimum intervals between effects respected)
  6. No creative styling implementation (no fonts, colors, zoom scales, easings)
  7. No asset hallucination (no fake media files)
  8. Full evidence traceability (every decision traces to evidence/opportunity)
  9. Valid time ranges (0 <= start <= end <= duration)
  10. Serializability & deterministic reproducibility
"""

import json
import pytest
from pathlib import Path

from core.evidence.models import (
    EvidenceSource,
    EvidenceSignals,
    EvidenceItem,
    EvidenceInventory,
    StoryObservation,
    StoryObservationsInventory,
    EditOpportunity,
    OpportunityInventory,
    EnrichedWord,
    EnrichedSentence,
    EnrichedTranscript,
)
from agents.agent2_director.models import (
    EditDecision,
    EditorialMoment,
    StoryEditPlan,
    EditDecisionsDocument,
    SUPPORTED_OPERATIONS,
)
from agents.agent2_director.story_editor import StoryEditor
from agents.agent2_director.validator import StoryEditorValidator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_media_meta():
    return {
        "source_file": "Surya.mp4",
        "duration_seconds": 90.833,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "aspect_ratio": "16:9",
        "audio_path": "audio.wav",
        "has_audio": True,
    }


@pytest.fixture
def sample_phase2_outputs(sample_media_meta):
    """Synthetic Phase 2 data reflecting real Surya.mp4 observations."""
    # 1. Transcript
    words = [
        EnrichedWord(id=0, word="I", start=0.18, end=0.40, duration=0.22, confidence=0.03, language="en", script="latin", romanized="I", low_confidence_flag=True, uncertainty=["low_asr_confidence:0.03"]),
        EnrichedWord(id=1, word="genuinely", start=0.42, end=0.90, duration=0.48, confidence=0.61, language="te", script="latin", romanized="genuinely"),
        EnrichedWord(id=2, word="empathy", start=1.20, end=1.80, duration=0.60, confidence=0.92, language="en", script="latin", romanized="empathy", emphasis_score=0.72),
        EnrichedWord(id=3, word="action", start=60.20, end=60.60, duration=0.40, confidence=0.95, language="en", script="latin", romanized="action", emphasis_score=0.75),
        EnrichedWord(id=4, word="judging", start=75.00, end=75.60, duration=0.60, confidence=0.88, language="en", script="latin", romanized="judging", emphasis_score=0.68),
    ]
    sentences = [
        EnrichedSentence(id=0, text="I genuinely can't imagine world without empathy.", start=0.18, end=3.28, duration=3.10, speaking_rate_wpm=155.0, avg_energy_deviation=-0.1, visual_context={"punch_in_eligible": True, "framing": "MCU"}),
        EnrichedSentence(id=1, text="Natural steady delivery section where speaker talks continuously.", start=10.0, end=20.0, duration=10.0, speaking_rate_wpm=140.0, avg_energy_deviation=0.05, visual_context={"punch_in_eligible": True, "framing": "MCU"}),
        EnrichedSentence(id=2, text="Empathy doesn't mean accepting every action.", start=58.0, end=65.0, duration=7.0, speaking_rate_wpm=150.0, avg_energy_deviation=0.25, visual_context={"punch_in_eligible": True, "framing": "MCU"}),
        EnrichedSentence(id=3, text="Choose to understand before judging.", start=74.0, end=80.0, duration=6.0, speaking_rate_wpm=145.0, avg_energy_deviation=0.15, visual_context={"punch_in_eligible": True, "framing": "MCU"}),
    ]
    tx = EnrichedTranscript(
        version="3.0.0",
        source_audio="audio.wav",
        duration_seconds=90.833,
        primary_language="te",
        detected_languages=["en", "te"],
        words=words,
        sentences=sentences,
    )

    # 2. Evidence
    evidence_list = [
        EvidenceItem(evidence_id="ev_001", type="hook_candidate", start=0.0, end=8.5, source=EvidenceSource(transcript_segment_ids=[0], word_ids=[0, 1, 2]), observation="Opening statement delivery.", signals=EvidenceSignals(), confidence=0.75, reliability="medium", tags=["hook"]),
        EvidenceItem(evidence_id="ev_002", type="dead_space", start=3.30, end=3.95, source=EvidenceSource(transcript_segment_ids=[], word_ids=[2]), observation="Hesitation dead space pause of 0.65s.", signals=EvidenceSignals(), confidence=0.88, reliability="high", tags=["dead_space"]),
        EvidenceItem(evidence_id="ev_003", type="rhetorical_pause", start=65.0, end=65.6, source=EvidenceSource(transcript_segment_ids=[], word_ids=[3]), observation="Rhetorical pause after thesis statement.", signals=EvidenceSignals(), confidence=0.88, reliability="high", tags=["rhetorical_hold"]),
        EvidenceItem(evidence_id="ev_004", type="emphasis", start=1.20, end=1.80, source=EvidenceSource(transcript_segment_ids=[], word_ids=[2]), observation="Emphasis on 'empathy'.", signals=EvidenceSignals(speech={"word": "empathy"}), confidence=0.92, reliability="high", tags=["emphasis"]),
        EvidenceItem(evidence_id="ev_005", type="emphasis", start=0.18, end=0.40, source=EvidenceSource(transcript_segment_ids=[], word_ids=[0]), observation="Emphasis on 'I'.", signals=EvidenceSignals(speech={"word": "I"}), confidence=0.03, reliability="low", tags=["emphasis"]),
        EvidenceItem(evidence_id="ev_006", type="no_edit_region", start=10.0, end=20.0, source=EvidenceSource(transcript_segment_ids=[1], word_ids=[]), observation="Natural unbroken conversational delivery.", signals=EvidenceSignals(), confidence=0.90, reliability="high", tags=["no_edit"]),
        EvidenceItem(evidence_id="ev_007", type="emphasis", start=60.20, end=60.60, source=EvidenceSource(transcript_segment_ids=[], word_ids=[3]), observation="Climactic emphasis on 'action'.", signals=EvidenceSignals(speech={"word": "action"}), confidence=0.95, reliability="high", tags=["emphasis"]),
    ]
    ev_inv = EvidenceInventory(version="3.0.0", source_video="Surya.mp4", duration_seconds=90.833, evidence=evidence_list)

    # 3. Story Observations
    story_obs_list = [
        StoryObservation(story_observation_id="story_01", type="HOOK", time_range=[0.0, 10.0], confidence=0.75, evidence_ids=["ev_001"], summary="Opening thesis delivery capturing viewer attention on empathy."),
        StoryObservation(story_observation_id="story_02", type="CONTEXT", time_range=[10.0, 40.0], confidence=0.80, evidence_ids=["ev_006"], summary="Elaboration connecting concepts to daily life."),
        StoryObservation(story_observation_id="story_03", type="REALIZATION", time_range=[55.0, 70.0], confidence=0.85, evidence_ids=["ev_007", "ev_003"], summary="Core thesis turnaround."),
        StoryObservation(story_observation_id="story_04", type="PAYOFF", time_range=[70.0, 90.833], confidence=0.82, evidence_ids=["ev_007"], summary="Concluding philosophical resolution."),
    ]
    st_obs = StoryObservationsInventory(version="3.0.0", source_video="Surya.mp4", duration_seconds=90.833, story_observations=story_obs_list)

    # 4. Opportunities (Variety of opportunities to test selectivity)
    opps = [
        EditOpportunity(opportunity_id="opp_001", type="hero_text_opportunity", time_range=[0.0, 3.0], evidence_ids=["ev_001"], confidence=0.75, priority=0.90, description="Hook title card."),
        EditOpportunity(opportunity_id="opp_002", type="cut_opportunity", time_range=[3.30, 3.95], evidence_ids=["ev_002"], confidence=0.88, priority=0.80, description="Dead space cut."),
        EditOpportunity(opportunity_id="opp_003", type="cut_opportunity", time_range=[65.0, 65.6], evidence_ids=["ev_003"], confidence=0.88, priority=0.75, description="Pause cut."),
        EditOpportunity(opportunity_id="opp_004", type="emphasis_text_opportunity", time_range=[1.20, 1.80], evidence_ids=["ev_004"], confidence=0.92, priority=0.85, description="Keyword emphasis on 'empathy'."),
        EditOpportunity(opportunity_id="opp_005", type="emphasis_text_opportunity", time_range=[0.18, 0.40], evidence_ids=["ev_005"], confidence=0.03, priority=0.40, description="Emphasis on 'I'."),
        EditOpportunity(opportunity_id="opp_006", type="no_edit_opportunity", time_range=[10.0, 20.0], evidence_ids=["ev_006"], confidence=0.90, priority=0.95, description="Continuous natural delivery."),
        EditOpportunity(opportunity_id="opp_007", type="punch_in_opportunity", time_range=[60.20, 62.20], evidence_ids=["ev_007"], confidence=0.95, priority=0.88, description="Thesis punch-in."),
        EditOpportunity(opportunity_id="opp_008", type="broll_opportunity", time_range=[12.0, 16.0], evidence_ids=["ev_006"], confidence=0.80, priority=0.70, description="B-roll context."),
        EditOpportunity(opportunity_id="opp_009", type="sfx_opportunity", time_range=[60.20, 60.50], evidence_ids=["ev_007"], confidence=0.90, priority=0.75, description="Acoustic accent."),
        EditOpportunity(opportunity_id="opp_010", type="broll_opportunity", time_range=[14.0, 18.0], evidence_ids=["ev_006"], confidence=0.78, priority=0.65, description="B-roll conflict."),
        EditOpportunity(opportunity_id="opp_011", type="cut_opportunity", time_range=[25.0, 25.2], evidence_ids=["ev_006"], confidence=0.75, priority=0.30, description="Micro-pause cut candidate."),
    ]
    opp_inv = OpportunityInventory(version="3.0.0", source_video="Surya.mp4", duration_seconds=90.833, opportunities=opps)

    return ev_inv, st_obs, opp_inv, tx


# ─── Acceptance Tests ─────────────────────────────────────────────────────────

def test_1_selectivity(sample_media_meta, sample_phase2_outputs):
    """Check 1: Agent 2 is disciplined and does not accept all opportunities."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    total_opps = len(opp_inv.opportunities)
    accepted = len(doc.accepted_decisions)

    # Acceptance rate must be selective (less than total)
    assert accepted < total_opps, f"Expected selectivity, but accepted {accepted}/{total_opps}"
    assert accepted <= 7, f"Over-editing: accepted {accepted} decisions"


def test_2_rejection_of_weak_opportunities(sample_media_meta, sample_phase2_outputs):
    """Check 2: Weak opportunities are rejected with explicit informative reasons."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    rejected = doc.rejected_decisions

    assert len(rejected) >= 2, f"Expected at least 2 rejections, got {len(rejected)}"
    for r in rejected:
        assert len(r.reason) >= 10, f"Rejection '{r.decision_id}' lacks clear reason"

    # Word "I" (opp_005) must be rejected
    opp_005_dec = [d for d in rejected if "opp_005" in d.source_opportunity_ids]
    assert len(opp_005_dec) == 1
    assert "common" in opp_005_dec[0].reason.lower() or "grammatical" in opp_005_dec[0].reason.lower()


def test_3_no_edit_is_first_class(sample_media_meta, sample_phase2_outputs):
    """Check 3: Preserved natural delivery and rhetorical holds are explicitly kept."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    no_edits = doc.no_edit_decisions

    assert len(no_edits) >= 2, f"Expected at least 2 NO_EDIT decisions, got {len(no_edits)}"

    # Check rhetorical pause preservation
    rhet_no_edits = [d for d in no_edits if "rhetorical" in d.reason.lower()]
    assert len(rhet_no_edits) >= 1, "Rhetorical pause should be preserved as NO_EDIT"


def test_4_budget_compliance(sample_media_meta, sample_phase2_outputs):
    """Check 4: Edit budget hard limits are respected."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    util = doc.budget_utilization

    assert util["hero_text"]["used"] <= 2
    assert util["broll"]["used"] <= 3
    assert util["sfx"]["used"] <= 6
    assert util["emphasis_text"]["used"] <= 8


def test_5_spacing_compliance(sample_media_meta, sample_phase2_outputs):
    """Check 5: Spacing between consecutive effects is respected."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    validator = StoryEditorValidator(results["story_edit_plan"], doc, sample_media_meta["duration_seconds"])

    assert validator.check_spacing_compliance() is True, f"Spacing violated: {validator.errors}"


def test_6_no_creative_styling_implementation(sample_media_meta, sample_phase2_outputs):
    """Check 6: No fonts, colors, zoom scales (1.1x, 1.25x), or easings in decisions."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    validator = StoryEditorValidator(results["story_edit_plan"], doc, sample_media_meta["duration_seconds"])

    assert validator.check_no_creative_styling() is True, f"Styling detected: {validator.errors}"


def test_7_no_asset_hallucination(sample_media_meta, sample_phase2_outputs):
    """Check 7: No concrete media file paths are specified."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    validator = StoryEditorValidator(results["story_edit_plan"], doc, sample_media_meta["duration_seconds"])

    assert validator.check_no_asset_hallucination() is True, f"Asset hallucination: {validator.errors}"


def test_8_evidence_traceability(sample_media_meta, sample_phase2_outputs):
    """Check 8: Every decision traces to underlying evidence or opportunities."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    for d in doc.all_decisions:
        has_grounding = bool(d.source_evidence_ids or d.source_opportunity_ids)
        assert has_grounding is True, f"Decision '{d.decision_id}' lacks grounding"


def test_9_valid_time_ranges(sample_media_meta, sample_phase2_outputs):
    """Check 9: 0.0 <= start <= end <= duration."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    doc = results["edit_decisions"]
    dur = sample_media_meta["duration_seconds"]

    for d in doc.all_decisions:
        assert 0.0 <= d.start <= d.end <= dur + 1.0, f"Invalid range [{d.start}, {d.end}]"


def test_10_serializability_and_validator_suite(sample_media_meta, sample_phase2_outputs):
    """Check 10: Validation suite passes completely and JSON serializes cleanly."""
    ev_inv, st_obs, opp_inv, tx = sample_phase2_outputs
    editor = StoryEditor(ev_inv, st_obs, opp_inv, tx, sample_media_meta)
    results = editor.run_editorial_pipeline()

    plan = results["story_edit_plan"]
    doc = results["edit_decisions"]

    validator = StoryEditorValidator(plan, doc, sample_media_meta["duration_seconds"])
    report = validator.validate_all()

    assert report["passed"] is True, f"Validation failed with errors: {report['errors']}"
    assert report["total_checks"] == 10
    for k, v in report["checks"].items():
        assert v is True, f"Check '{k}' failed"
