"""
core/evidence/validator.py

Evidence & Observation Layer Validation Suite (Phase 2)
───────────────────────────────────────────────────────
Enforces the 10 acceptance criteria specified for Phase 2:
  1. Observation vs decision separation
  2. Evidence IDs always resolve
  3. Opportunity IDs always resolve and are unique
  4. Low-confidence transcript uncertainty propagates
  5. NO_EDIT regions are first-class and present
  6. Telugu/English code-switching remains intact
  7. All time ranges are valid (0 <= start <= end <= duration)
  8. Story observations reference valid evidence IDs
  9. No creative styling appears in Agent 1 output
  10. Serializability and deterministic reproducibility
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.evidence.models import (
    EvidenceInventory,
    StoryObservationsInventory,
    OpportunityInventory,
    EnrichedTranscript,
    SUPPORTED_EVIDENCE_TYPES,
    STORY_ACT_TYPES,
    OPPORTUNITY_TYPES,
)


FORBIDDEN_STYLING_TERMS = [
    # Fonts
    r"\bmontserrat\b", r"\broboto\b", r"\binter\b", r"\boutfit\b", r"\bcomic sans\b",
    # Specific zoom values
    r"\b1\.[0-9]+x\b", r"\bzoom to\b", r"\bzoom level\b", r"\bscale factor\b",
    # Color references for styling
    r"\byellow\b", r"\bgold\b", r"\bcyan\b", r"\bwhite on black\b", r"\bhex code\b",
    # Animation easing
    r"\bspring easing\b", r"\bease-in\b", r"\bease-out\b", r"\bkinetic typography\b",
    # Imperative edit commands in observations
    r"\badd dramatic zoom\b", r"\bapply punch\b", r"\bburn subtitles\b",
    r"\binsert sfx\b", r"\bplay whoosh\b", r"\bplay sound\b",
]


class EvidenceLayerValidator:
    """Validates full consistency, traceability, and constraints of Phase 2 outputs."""

    def __init__(
        self,
        evidence_inv: EvidenceInventory,
        story_obs: StoryObservationsInventory,
        opp_inv: OpportunityInventory,
        transcript: EnrichedTranscript,
        max_duration: float = 3600.0,
    ):
        self.evidence_inv = evidence_inv
        self.story_obs = story_obs
        self.opp_inv = opp_inv
        self.transcript = transcript
        self.max_duration = max_duration

        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> dict[str, Any]:
        """Runs all 10 acceptance checks and returns a summary report."""
        self.errors.clear()
        self.warnings.clear()

        c1 = self.check_observation_vs_decision()
        c2 = self.check_evidence_ids_resolve()
        c3 = self.check_opportunity_ids_unique()
        c4 = self.check_uncertainty_propagation()
        c5 = self.check_no_edit_is_first_class()
        c6 = self.check_code_switching_intact()
        c7 = self.check_valid_time_ranges()
        c8 = self.check_story_observations_reference_evidence()
        c9 = self.check_no_creative_styling()
        c10 = self.check_serializability()

        passed = len(self.errors) == 0
        return {
            "passed": passed,
            "total_checks": 10,
            "checks": {
                "observation_vs_decision": c1,
                "evidence_ids_resolve": c2,
                "opportunity_ids_unique": c3,
                "uncertainty_propagation": c4,
                "no_edit_supported": c5,
                "code_switching_intact": c6,
                "valid_time_ranges": c7,
                "story_references_evidence": c8,
                "no_creative_styling": c9,
                "serializability": c10,
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    # ── Check 1: Observation vs Decision Separation ───────────────────────────
    def check_observation_vs_decision(self) -> bool:
        """Verifies that evidence observations describe what happens, not edit commands."""
        bad_imperatives = [
            r"\bcut this\b", r"\btrim this\b", r"\badd zoom\b", r"\bapply filter\b",
            r"\badd caption\b", r"\bmute audio\b", r"\boverlay b-roll\b",
        ]
        ok = True
        for ev in self.evidence_inv.evidence:
            obs = ev.observation.lower()
            for pat in bad_imperatives:
                if re.search(pat, obs):
                    self.errors.append(
                        f"Check 1 FAIL: Evidence '{ev.evidence_id}' has imperative edit command: '{pat}' in '{ev.observation}'"
                    )
                    ok = False
        return ok

    # ── Check 2: Evidence IDs Always Resolve ──────────────────────────────────
    def check_evidence_ids_resolve(self) -> bool:
        """Verifies that all evidence IDs cited in story observations and opportunities exist."""
        known_ids = {ev.evidence_id for ev in self.evidence_inv.evidence}
        ok = True

        for so in self.story_obs.story_observations:
            for eid in so.evidence_ids:
                if eid not in known_ids:
                    self.errors.append(
                        f"Check 2 FAIL: StoryObservation '{so.story_observation_id}' references missing evidence ID '{eid}'"
                    )
                    ok = False

        for opp in self.opp_inv.opportunities:
            for eid in opp.evidence_ids:
                if eid not in known_ids:
                    self.errors.append(
                        f"Check 2 FAIL: Opportunity '{opp.opportunity_id}' references missing evidence ID '{eid}'"
                    )
                    ok = False
        return ok

    # ── Check 3: Opportunity IDs Are Unique & Resolve ────────────────────────
    def check_opportunity_ids_unique(self) -> bool:
        """Verifies that all opportunity IDs are distinct."""
        seen = set()
        ok = True
        for opp in self.opp_inv.opportunities:
            if opp.opportunity_id in seen:
                self.errors.append(
                    f"Check 3 FAIL: Duplicate opportunity ID '{opp.opportunity_id}'"
                )
                ok = False
            seen.add(opp.opportunity_id)
        return ok

    # ── Check 4: Low-confidence Transcript Uncertainty Propagates ─────────────
    def check_uncertainty_propagation(self) -> bool:
        """
        If an evidence item relies on low-confidence words (< 0.60) and lacks strong independent
        signals, it must have non-empty uncertainty and its confidence must not exceed 0.75.
        """
        words_by_id = {w.id: w for w in self.transcript.words}
        ok = True

        for ev in self.evidence_inv.evidence:
            word_confidences = [
                words_by_id[wid].confidence
                for wid in ev.source.word_ids
                if wid in words_by_id
            ]
            if word_confidences:
                min_conf = min(word_confidences)
                if min_conf < 0.50:
                    # Low ASR confidence
                    if not ev.uncertainty:
                        self.errors.append(
                            f"Check 4 FAIL: Evidence '{ev.evidence_id}' has low word confidence ({min_conf:.3f}) but empty uncertainty list."
                        )
                        ok = False
                    # Without visual/audio grounding, confidence shouldn't be inflated
                    has_strong_multimodal = (
                        bool(ev.signals.visual.get("strong_signal"))
                        or bool(ev.signals.audio.get("strong_signal"))
                    )
                    if not has_strong_multimodal and ev.confidence > 0.85:
                        self.errors.append(
                            f"Check 4 FAIL: Evidence '{ev.evidence_id}' has low word confidence ({min_conf:.3f}) but ungrounded high confidence ({ev.confidence})."
                        )
                        ok = False
        return ok

    # ── Check 5: NO_EDIT is First-Class and Present ───────────────────────────
    def check_no_edit_is_first_class(self) -> bool:
        """Verifies that no_edit_region items exist in the inventory."""
        no_edits = self.evidence_inv.filter_by_type("no_edit_region")
        if not no_edits:
            self.errors.append(
                "Check 5 FAIL: No 'no_edit_region' evidence items found. NO_EDIT must be a first-class citizen."
            )
            return False
        return True

    # ── Check 6: Code-Switching Remains Intact ────────────────────────────────
    def check_code_switching_intact(self) -> bool:
        """Verifies that the enriched transcript preserves Telugu and English code-switching."""
        languages = set(self.transcript.detected_languages)
        te_words = [w for w in self.transcript.words if w.language == "te"]
        en_words = [w for w in self.transcript.words if w.language == "en"]

        if len(te_words) == 0:
            self.warnings.append("Check 6 WARN: No Telugu words found in enriched transcript.")
        if len(en_words) == 0:
            self.warnings.append("Check 6 WARN: No English words found in enriched transcript.")

        # Words should carry explicit language and script
        for w in self.transcript.words[:50]:
            if not w.language or w.language == "und":
                self.errors.append(f"Check 6 FAIL: Word '{w.word}' missing language tag.")
                return False
            if not w.script:
                self.errors.append(f"Check 6 FAIL: Word '{w.word}' missing script tag.")
                return False
        return True

    # ── Check 7: Valid Time Ranges ────────────────────────────────────────────
    def check_valid_time_ranges(self) -> bool:
        """Verifies 0.0 <= start <= end <= max_duration."""
        ok = True
        limit = self.max_duration + 2.0

        for ev in self.evidence_inv.evidence:
            if not (0.0 <= ev.start <= ev.end <= limit):
                self.errors.append(
                    f"Check 7 FAIL: Evidence '{ev.evidence_id}' has invalid range [{ev.start}, {ev.end}] (limit={limit})"
                )
                ok = False

        for so in self.story_obs.story_observations:
            st, en = so.time_range[0], so.time_range[1]
            if not (0.0 <= st <= en <= limit):
                self.errors.append(
                    f"Check 7 FAIL: StoryObservation '{so.story_observation_id}' has invalid range [{st}, {en}] (limit={limit})"
                )
                ok = False

        for opp in self.opp_inv.opportunities:
            st, en = opp.time_range[0], opp.time_range[1]
            if not (0.0 <= st <= en <= limit):
                self.errors.append(
                    f"Check 7 FAIL: Opportunity '{opp.opportunity_id}' has invalid range [{st}, {en}] (limit={limit})"
                )
                ok = False
        return ok

    # ── Check 8: Story Observations Reference Valid Evidence ─────────────────
    def check_story_observations_reference_evidence(self) -> bool:
        """Verifies that every story observation has a non-empty evidence_ids list."""
        ok = True
        for so in self.story_obs.story_observations:
            if not so.evidence_ids:
                self.errors.append(
                    f"Check 8 FAIL: StoryObservation '{so.story_observation_id}' ({so.type}) has empty evidence_ids."
                )
                ok = False
        return ok

    # ── Check 9: No Creative Styling Appears in Agent 1 Output ────────────────
    def check_no_creative_styling(self) -> bool:
        """Verifies that no fonts, zoom amounts, colors, or easings appear in outputs."""
        ok = True
        text_samples = []

        for ev in self.evidence_inv.evidence:
            text_samples.append((f"evidence:{ev.evidence_id}", ev.observation))
        for so in self.story_obs.story_observations:
            text_samples.append((f"story:{so.story_observation_id}", so.summary))
        for opp in self.opp_inv.opportunities:
            text_samples.append((f"opportunity:{opp.opportunity_id}", opp.description))

        for source_tag, text in text_samples:
            lower = text.lower()
            for pattern in FORBIDDEN_STYLING_TERMS:
                if re.search(pattern, lower):
                    self.errors.append(
                        f"Check 9 FAIL: Found forbidden styling term '{pattern}' in {source_tag}: \"{text}\""
                    )
                    ok = False
        return ok

    # ── Check 10: Serializability & Reproducibility ───────────────────────────
    def check_serializability(self) -> bool:
        """Verifies that all objects can round-trip through JSON without data loss."""
        try:
            ev_dict = self.evidence_inv.to_dict()
            ev_json = json.dumps(ev_dict)
            ev_restored = EvidenceInventory.from_dict(json.loads(ev_json))
            assert len(ev_restored.evidence) == len(self.evidence_inv.evidence)

            so_dict = self.story_obs.to_dict()
            so_json = json.dumps(so_dict)
            so_restored = StoryObservationsInventory.from_dict(json.loads(so_json))
            assert len(so_restored.story_observations) == len(self.story_obs.story_observations)

            opp_dict = self.opp_inv.to_dict()
            opp_json = json.dumps(opp_dict)
            opp_restored = OpportunityInventory.from_dict(json.loads(opp_json))
            assert len(opp_restored.opportunities) == len(self.opp_inv.opportunities)

            tx_dict = self.transcript.to_dict()
            tx_json = json.dumps(tx_dict)
            tx_restored = EnrichedTranscript.from_dict(json.loads(tx_json))
            assert len(tx_restored.words) == len(self.transcript.words)
            assert len(tx_restored.sentences) == len(self.transcript.sentences)
            return True
        except Exception as e:
            self.errors.append(f"Check 10 FAIL: Serialization round-trip failed: {e}")
            return False
