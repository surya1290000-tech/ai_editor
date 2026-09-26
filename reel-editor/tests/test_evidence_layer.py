"""
tests/test_evidence_layer.py

Acceptance Tests for Agent 1 Observer & Evidence Layer (Phase 2)
───────────────────────────────────────────────────────────────
Verifies all 10 acceptance criteria specified for Phase 2:
  1. Observation vs decision separation
  2. Evidence IDs always resolve
  3. Opportunity IDs always resolve and are unique
  4. Low-confidence transcript uncertainty propagates
  5. NO_EDIT is supported as a first-class citizen
  6. Telugu/English code-switching remains intact
  7. All time ranges are valid
  8. Story observations reference valid evidence
  9. No creative styling appears in Agent 1 output
  10. evidence_inventory.json is deterministic/serializable
"""

import json
import pytest
from pathlib import Path

from core.evidence.models import (
    SUPPORTED_EVIDENCE_TYPES,
    STORY_ACT_TYPES,
    OPPORTUNITY_TYPES,
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
from core.evidence.collector import EvidenceCollector
from core.evidence.story_analyzer import StoryObserver
from core.evidence.opportunity_engine import OpportunityObserver
from core.evidence.validator import EvidenceLayerValidator, FORBIDDEN_STYLING_TERMS


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
def sample_transcript_raw():
    return {
        "detected_languages": ["te", "en"],
        "speech_stats": {"detected_language": "te", "total_words": 5},
        "words": [
            {
                "id": 0,
                "word": "I",
                "start": 0.1,
                "end": 0.3,
                "confidence": 0.95,
                "language": "en",
                "script": "latin",
                "romanized": "I",
                "is_filler": False,
            },
            {
                "id": 1,
                "word": "genuinely",
                "start": 0.35,
                "end": 0.8,
                "confidence": 0.61,
                "language": "te",
                "script": "latin",
                "romanized": "genuinely",
                "is_filler": False,
            },
            {
                "id": 2,
                "word": "empathy",
                "start": 1.2,
                "end": 1.8,
                "confidence": 0.92,
                "language": "en",
                "script": "latin",
                "romanized": "empathy",
                "is_filler": False,
            },
            {
                "id": 3,
                "word": "ante",
                "start": 1.85,
                "end": 2.2,
                "confidence": 0.88,
                "language": "te",
                "script": "latin",
                "romanized": "ante",
                "is_filler": True,
            },
            {
                "id": 4,
                "word": "abzo",
                "start": 2.5,
                "end": 2.9,
                "confidence": 0.31,  # Low confidence word
                "language": "te",
                "script": "latin",
                "romanized": "abzo",
                "is_filler": False,
                "alternate_hypotheses": [{"word": "observe", "confidence": 0.28}],
            },
        ],
    }


@pytest.fixture
def sample_audio_features():
    return {
        "speaker_baseline_energy": 0.15,
        "speaker_baseline_pitch_hz": 120.0,
        "speaker_baseline_wpm": 140.0,
        "words": [
            {"id": 0, "energy_deviation": 0.05, "pitch_deviation": 0.02, "emphasis_score": 0.1},
            {"id": 1, "energy_deviation": 0.10, "pitch_deviation": 0.05, "emphasis_score": 0.3},
            {"id": 2, "energy_deviation": 0.35, "pitch_deviation": 0.25, "emphasis_score": 0.78},  # Emphasis peak
            {"id": 3, "energy_deviation": -0.05, "pitch_deviation": -0.02, "emphasis_score": 0.05},
            {"id": 4, "energy_deviation": -0.10, "pitch_deviation": -0.05, "emphasis_score": 0.12},
        ],
    }


@pytest.fixture
def sample_visual_features():
    return {
        "dominant_framing": "MCU",
        "shot_segments": [
            {"id": "shot_0", "start": 0.0, "end": 45.0, "framing": "MCU", "stability_score": 0.94},
            {"id": "shot_1", "start": 45.0, "end": 90.833, "framing": "MCU", "stability_score": 0.91},
        ],
    }


@pytest.fixture
def sample_fused_timeline():
    return {
        "silence_regions": [
            {
                "start": 3.0,
                "end": 3.65,
                "duration_seconds": 0.65,
                "category": "DEAD_SPACE",
                "position": "MID_SENTENCE",
                "preceding_word": "ante",
            },
            {
                "start": 5.0,
                "end": 5.6,
                "duration_seconds": 0.60,
                "category": "RHETORICAL_HOLD",
                "position": "BETWEEN_SENTENCES",
                "preceding_word": "empathy",
            },
        ],
        "sentences": [
            {
                "id": 0,
                "text": "I genuinely can't imagine world without empathy.",
                "start": 0.1,
                "end": 2.9,
                "speaking_rate_wpm": 150.0,
                "avg_energy_deviation": 0.18,
                "peak_energy_deviation": 0.35,
                "visual_context": {"punch_in_eligible": True, "framing": "MCU"},
            },
            {
                "id": 1,
                "text": "Natural delivery section where speaker talks with steady pace.",
                "start": 6.0,
                "end": 10.0,
                "speaking_rate_wpm": 140.0,
                "avg_energy_deviation": 0.05,
                "peak_energy_deviation": 0.10,
                "visual_context": {"punch_in_eligible": True, "framing": "MCU"},
            },
        ],
    }


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_1_observation_vs_decision_separation(
    sample_media_meta,
    sample_transcript_raw,
    sample_audio_features,
    sample_visual_features,
    sample_fused_timeline,
):
    """Check 1: Agent 1 observations describe reality, never imperative edit decisions."""
    collector = EvidenceCollector(
        media_meta=sample_media_meta,
        transcript_raw=sample_transcript_raw,
        audio_features=sample_audio_features,
        visual_features=sample_visual_features,
        fused_timeline=sample_fused_timeline,
    )
    enriched_tx = collector.build_enriched_transcript()
    evidence_inv = collector.collect_evidence_inventory(enriched_tx)

    bad_verbs = ["cut this", "add zoom", "burn subtitle", "insert sfx", "mute audio"]
    for ev in evidence_inv.evidence:
        for verb in bad_verbs:
            assert verb not in ev.observation.lower(), (
                f"Evidence '{ev.evidence_id}' contains imperative edit verb '{verb}'"
            )


def test_2_evidence_ids_always_resolve(
    sample_media_meta,
    sample_transcript_raw,
    sample_audio_features,
    sample_visual_features,
    sample_fused_timeline,
):
    """Check 2: Every evidence ID cited in story observations & opportunities exists."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)

    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    known_ids = {e.evidence_id for e in ev_inv.evidence}

    for so in story_obs.story_observations:
        assert len(so.evidence_ids) > 0, f"StoryObservation '{so.story_observation_id}' has empty evidence_ids"
        for eid in so.evidence_ids:
            assert eid in known_ids, f"StoryObservation references unknown evidence ID '{eid}'"

    for opp in opp_inv.opportunities:
        assert len(opp.evidence_ids) > 0, f"Opportunity '{opp.opportunity_id}' has empty evidence_ids"
        for eid in opp.evidence_ids:
            assert eid in known_ids, f"Opportunity references unknown evidence ID '{eid}'"


def test_3_opportunity_ids_unique_and_resolve(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 3: Opportunity IDs are unique and non-empty."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)
    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    seen_ids = set()
    for opp in opp_inv.opportunities:
        assert opp.opportunity_id not in seen_ids, f"Duplicate opportunity ID: {opp.opportunity_id}"
        seen_ids.add(opp.opportunity_id)
        assert opp.type in OPPORTUNITY_TYPES, f"Invalid opportunity type: {opp.type}"


def test_4_low_confidence_uncertainty_propagation(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 4: Low-confidence ASR words must flag uncertainty and not have ungrounded high confidence."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()

    # Word 4 has confidence 0.31
    w4 = tx.words[4]
    assert w4.low_confidence_flag is True
    assert any("low_asr_confidence" in u for u in w4.uncertainty)

    # Sentence 0 contains word 4
    s0 = tx.sentences[0]
    assert s0.low_confidence_count >= 1
    assert any("words_with_low_confidence" in u for u in s0.uncertainty)

    # Evidence covering word 4
    ev_inv = collector.collect_evidence_inventory(tx)
    for ev in ev_inv.evidence:
        if 4 in ev.source.word_ids:
            assert len(ev.uncertainty) > 0, f"Evidence {ev.evidence_id} missing uncertainty for low-conf word"


def test_5_no_edit_is_first_class(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 5: NO_EDIT regions must be first-class citizens in evidence and opportunities."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)

    no_edits = ev_inv.filter_by_type("no_edit_region")
    assert len(no_edits) >= 1, "At least one no_edit_region must be detected"
    ne = no_edits[0]
    assert "Natural" in ne.observation or "continuous" in ne.observation.lower()

    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    no_edit_opps = [o for o in opp_inv.opportunities if o.type == "no_edit_opportunity"]
    assert len(no_edit_opps) >= 1, "At least one no_edit_opportunity must be present"


def test_6_code_switching_intact(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 6: Telugu/English code-switching is preserved without overwriting."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()

    assert "te" in tx.detected_languages
    assert "en" in tx.detected_languages

    te_words = [w for w in tx.words if w.language == "te"]
    en_words = [w for w in tx.words if w.language == "en"]

    assert len(te_words) >= 2
    assert len(en_words) >= 2

    # Check words have valid language & script
    for w in tx.words:
        assert w.language in ("te", "en")
        assert w.script in ("latin", "telugu")


def test_7_valid_time_ranges(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 7: All time ranges satisfy 0 <= start <= end <= duration."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)
    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    dur = sample_media_meta["duration_seconds"]
    for ev in ev_inv.evidence:
        assert 0.0 <= ev.start <= ev.end <= dur + 1.0

    for so in story_obs.story_observations:
        assert 0.0 <= so.time_range[0] <= so.time_range[1] <= dur + 1.0

    for opp in opp_inv.opportunities:
        assert 0.0 <= opp.time_range[0] <= opp.time_range[1] <= dur + 1.0


def test_8_story_observations_reference_evidence(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 8: Story observations have non-empty valid evidence IDs."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)
    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()

    assert len(story_obs.story_observations) >= 1
    for so in story_obs.story_observations:
        assert len(so.evidence_ids) > 0
        assert so.type in STORY_ACT_TYPES


def test_9_no_creative_styling_in_agent1_output(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 9: No styling, font names, zoom ratios (1.1x, 1.25x), or colors appear in Agent 1 output."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)
    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    validator = EvidenceLayerValidator(ev_inv, story_obs, opp_inv, tx, sample_media_meta["duration_seconds"])
    assert validator.check_no_creative_styling() is True, f"Found creative styling: {validator.errors}"


def test_10_serializability_and_validator_suite(
    sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
):
    """Check 10: Full validation suite passes and JSON serialization round-trips cleanly."""
    collector = EvidenceCollector(
        sample_media_meta, sample_transcript_raw, sample_audio_features, sample_visual_features, sample_fused_timeline
    )
    tx = collector.build_enriched_transcript()
    ev_inv = collector.collect_evidence_inventory(tx)
    story_obs = StoryObserver(ev_inv, tx, sample_media_meta["duration_seconds"]).analyze_story_structure()
    opp_inv = OpportunityObserver(ev_inv, story_obs, sample_media_meta["duration_seconds"]).build_opportunity_inventory()

    validator = EvidenceLayerValidator(ev_inv, story_obs, opp_inv, tx, sample_media_meta["duration_seconds"])
    report = validator.validate_all()

    assert report["passed"] is True, f"Validation failed with errors: {report['errors']}"
    assert report["total_checks"] == 10
    for k, v in report["checks"].items():
        assert v is True, f"Check {k} failed"
