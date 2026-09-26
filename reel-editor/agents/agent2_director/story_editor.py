"""
agents/agent2_director/story_editor.py

Agent 2: True Story Editor (Phase 3)
───────────────────────────────────
Answers: "What should actually change in this reel?"

Core Responsibilities:
  1. Narrative Understanding: Understands reel structure (HOOK, CONTEXT, PROBLEM, REALIZATION, PAYOFF).
  2. Moment Classification: PRIMARY, SECONDARY, PRESERVE.
  3. Selective Intervention: Evaluates all Agent 1 opportunities, accepts a disciplined subset,
     rejects weak/redundant opportunities, and treats NO_EDIT as a first-class citizen.
  4. Conflict Resolution & Budget Enforcement: Enforces limits and spacing without effect stacking.
  5. Traceability: transcript → evidence → story observation → opportunity → decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger
from core.json_utils import save_json
from core.evidence.models import (
    EvidenceInventory,
    StoryObservationsInventory,
    OpportunityInventory,
    EnrichedTranscript,
)
from agents.agent2_director.models import (
    EditDecision,
    EditorialMoment,
    StoryEditPlan,
    EditDecisionsDocument,
)
from agents.agent2_director.scorer import EditBudgetTracker, EditorialScorer

logger = get_logger(__name__)


class StoryEditor:
    """Agent 2: Skilled short-form video story editor."""

    def __init__(
        self,
        evidence_inv: EvidenceInventory,
        story_obs: StoryObservationsInventory,
        opp_inv: OpportunityInventory,
        transcript: EnrichedTranscript,
        media_meta: dict[str, Any],
        pacing_preset: str = "balanced",
        budget_config_path: Optional[Path | str] = None,
    ):
        self.evidence_inv = evidence_inv
        self.story_obs = story_obs
        self.opp_inv = opp_inv
        self.transcript = transcript
        self.media_meta = media_meta
        self.duration = float(media_meta.get("duration_seconds", 0.0))
        self.source_video = str(media_meta.get("source_file", "video.mp4"))
        self.pacing_preset = pacing_preset

        self.budget_tracker = EditBudgetTracker(budget_config_path, duration=self.duration)
        self.scorer = EditorialScorer()
        self._decision_counter = 0

    def _next_id(self, prefix: str = "dec") -> str:
        self._decision_counter += 1
        return f"{prefix}_{self._decision_counter:03d}"

    def run_editorial_pipeline(self) -> dict[str, Any]:
        """
        Executes editorial intelligence pass and produces:
          1. StoryEditPlan (story_edit_plan.json)
          2. EditDecisionsDocument (edit_decisions.json)
        """
        self._decision_counter = 0

        # Step 1: Narrative Understanding & Moment Classification
        editorial_moments = self._classify_editorial_moments()

        # Step 2: Evaluate All Opportunities (Selective Acceptance & Disciplined Rejection)
        accepted_decisions: list[EditDecision] = []
        no_edit_decisions: list[EditDecision] = []
        rejected_decisions: list[EditDecision] = []
        traceability_matrix: list[dict[str, Any]] = []

        # Index story acts by time for lookup
        acts = self.story_obs.story_observations

        # Process each candidate opportunity from Agent 1
        for opp in self.opp_inv.opportunities:
            # Determine overlapping story act
            act_name = self._find_act_for_interval(opp.time_range[0], opp.time_range[1])
            moment = self._find_moment_for_time(opp.time_range[0], editorial_moments)
            classification = moment.classification if moment else "SECONDARY"

            # Check underlying evidence uncertainty
            supporting_ev = [
                self.evidence_inv.get(eid) for eid in opp.evidence_ids
                if self.evidence_inv.get(eid) is not None
            ]
            has_unc = any(len(e.uncertainty) > 0 for e in supporting_ev if e)
            avg_ev_conf = (
                sum(e.confidence for e in supporting_ev) / len(supporting_ev)
                if supporting_ev else opp.confidence
            )

            # Map story observation IDs
            so_ids = [
                so.story_observation_id for so in acts
                if not (so.time_range[1] < opp.time_range[0] or so.time_range[0] > opp.time_range[1])
            ]

            # Evaluate candidate based on type
            decision = self._evaluate_single_opportunity(
                opp=opp,
                act_name=act_name,
                classification=classification,
                supporting_ev=supporting_ev,
                has_unc=has_unc,
                avg_ev_conf=avg_ev_conf,
                so_ids=so_ids,
            )

            # Route decision based on status
            if decision.status == "ACCEPTED":
                accepted_decisions.append(decision)
            elif decision.status == "NO_EDIT":
                no_edit_decisions.append(decision)
            else:
                rejected_decisions.append(decision)

            # Record Traceability Chain
            traceability_matrix.append({
                "decision_id": decision.decision_id,
                "operation": decision.operation,
                "status": decision.status,
                "time_range": [decision.start, decision.end],
                "source_opportunity_id": opp.opportunity_id,
                "source_evidence_ids": decision.source_evidence_ids,
                "story_act": decision.story_act,
                "story_observation_ids": decision.story_observation_ids,
                "confidence": decision.confidence,
                "reason": decision.reason,
            })

        # Step 3: Identify untouched intervals and emit explicit KEEP / NO_EDIT decisions
        additional_no_edits = self._synthesize_coverage_no_edits(
            accepted_decisions, no_edit_decisions, editorial_moments
        )
        no_edit_decisions.extend(additional_no_edits)
        for ned in additional_no_edits:
            traceability_matrix.append({
                "decision_id": ned.decision_id,
                "operation": ned.operation,
                "status": ned.status,
                "time_range": [ned.start, ned.end],
                "source_opportunity_id": None,
                "source_evidence_ids": ned.source_evidence_ids,
                "story_act": ned.story_act,
                "story_observation_ids": ned.story_observation_ids,
                "confidence": ned.confidence,
                "reason": ned.reason,
            })

        # Sort all decision lists chronologically
        accepted_decisions.sort(key=lambda d: (d.start, d.end, d.decision_id))
        no_edit_decisions.sort(key=lambda d: (d.start, d.end, d.decision_id))
        rejected_decisions.sort(key=lambda d: (d.start, d.end, d.decision_id))

        # Step 4: Assemble Edit Decisions Document
        edit_decisions_doc = EditDecisionsDocument(
            version="3.0.0",
            source_video=self.source_video,
            duration_seconds=self.duration,
            accepted_decisions=accepted_decisions,
            no_edit_decisions=no_edit_decisions,
            rejected_decisions=rejected_decisions,
            budget_utilization=self.budget_tracker.get_utilization_report(),
            traceability_matrix=traceability_matrix,
        )

        # Step 5: Assemble Story Edit Plan
        strategy_desc = (
            f"Narrative-driven {self.pacing_preset} edit: Hook attention with clear thesis opening, "
            f"preserve authentic conversational pacing and rhetorical holds, reinforce key turning point "
            f"in {self._find_realization_act()} act with punch-in emphasis and hero text, while rejecting "
            f"decorative or excessive effect clustering."
        )

        story_edit_plan = StoryEditPlan(
            version="3.0.0",
            source_video=self.source_video,
            duration_seconds=self.duration,
            pacing_preset=self.pacing_preset,
            narrative_strategy=strategy_desc,
            moments=editorial_moments,
            budget_limits=self.budget_tracker.limits,
            decisions_summary={
                "total_opportunities_evaluated": len(self.opp_inv.opportunities),
                "total_accepted_interventions": len(accepted_decisions),
                "total_no_edit_preserved": len(no_edit_decisions),
                "total_rejected": len(rejected_decisions),
                "selectivity_ratio": (
                    f"{len(accepted_decisions)}/{len(self.opp_inv.opportunities)} "
                    f"({(len(accepted_decisions)/max(1, len(self.opp_inv.opportunities)))*100:.1f}%)"
                ),
            },
        )

        return {
            "story_edit_plan": story_edit_plan,
            "edit_decisions": edit_decisions_doc,
        }

    # ── Internal Evaluation Logic ───────────────────────────────────────────────

    def _classify_editorial_moments(self) -> list[EditorialMoment]:
        """Understands the reel as a narrative and classifies moments into PRIMARY, SECONDARY, PRESERVE."""
        moments: list[EditorialMoment] = []
        sentences = self.transcript.sentences if self.transcript else []

        if not sentences:
            # Fallback for silent / low-speech videos: partition duration into visual-first narrative moments
            dur = self.duration if self.duration > 0 else 30.0
            act_spans = [
                ("HOOK", 0.0, min(dur, 3.0), "PRIMARY", "Opening visual hook"),
                ("CONTEXT", min(dur, 3.0), dur * 0.5, "SECONDARY", "Visual context and development"),
                ("REALIZATION", dur * 0.5, dur * 0.85, "PRIMARY", "Key visual turning point"),
                ("PAYOFF", dur * 0.85, dur, "SECONDARY", "Concluding resolution"),
            ]
            for idx, (act_name, st, en, cls_type, desc) in enumerate(act_spans):
                if st < en:
                    moments.append(
                        EditorialMoment(
                            moment_id=f"moment_{idx:02d}",
                            story_act=act_name,
                            classification=cls_type,
                            start=round(st, 2),
                            end=round(en, 2),
                            core_idea=desc,
                            strongest_statement="Visual delivery",
                            emotional_importance=0.80 if cls_type == "PRIMARY" else 0.60,
                            rhetorical_importance=0.85 if cls_type == "PRIMARY" else 0.65,
                            pacing_weakness=None,
                            attention_peak=(cls_type == "PRIMARY"),
                            attention_drop=False,
                            evidence_ids=[],
                        )
                    )
            return moments

        for idx, s in enumerate(sentences):
            act_name = self._find_act_for_interval(s.start, s.end)

            # Determine classification
            is_hook = act_name == "HOOK" or s.start < 6.0
            has_peak = getattr(s, "emphasis_peak", False) or getattr(s, "avg_energy_deviation", 0.0) > 0.20
            is_thesis_peak = (act_name == "REALIZATION" and has_peak) or (act_name == "PAYOFF" and has_peak) or getattr(s, "emphasis_peak", False)
            is_steady_flow = s.duration >= 3.0 and getattr(s, "filler_count", 0) == 0 and s.speaking_rate_wpm >= 120 and not has_peak

            if is_hook or is_thesis_peak:
                cls_type = "PRIMARY"
                core_idea = "Core thesis statement or hook opening."
                strongest_stmt = s.text[:80]
                emo_imp = 0.90
                rhet_imp = 0.95
                att_peak = True
                att_drop = False
            elif is_steady_flow:
                cls_type = "PRESERVE"
                core_idea = "Natural fluent narrative progression."
                strongest_stmt = s.text[:80]
                emo_imp = 0.60
                rhet_imp = 0.65
                att_peak = False
                att_drop = False
            else:
                cls_type = "SECONDARY"
                core_idea = "Contextual development and illustrative tension."
                strongest_stmt = s.text[:80]
                emo_imp = 0.65
                rhet_imp = 0.70
                att_peak = False
                att_drop = s.speaking_rate_wpm < 110

            moments.append(
                EditorialMoment(
                    moment_id=f"moment_{idx:02d}",
                    story_act=act_name,
                    classification=cls_type,
                    start=s.start,
                    end=s.end,
                    core_idea=core_idea,
                    strongest_statement=strongest_stmt,
                    emotional_importance=emo_imp,
                    rhetorical_importance=rhet_imp,
                    pacing_weakness="Low speaking pace" if att_drop else None,
                    attention_peak=att_peak,
                    attention_drop=att_drop,
                    evidence_ids=[f"ev_sent_{s.id}"],
                )
            )
        return moments

    def _evaluate_single_opportunity(
        self,
        opp: Any,
        act_name: str,
        classification: str,
        supporting_ev: list[Any],
        has_unc: bool,
        avg_ev_conf: float,
        so_ids: list[str],
    ) -> EditDecision:
        """Disciplined editorial assessment of an Agent 1 opportunity."""
        opp_type = opp.type
        st, en = opp.time_range[0], opp.time_range[1]
        dur = en - st

        # ── 1. NO_EDIT OPPORTUNITY ─────────────────────────────────────
        if opp_type == "no_edit_opportunity":
            return EditDecision(
                decision_id=self._next_id("keep"),
                operation="NO_EDIT",
                status="NO_EDIT",
                start=st,
                end=en,
                source_evidence_ids=list(opp.evidence_ids),
                source_opportunity_ids=[opp.opportunity_id],
                story_observation_ids=so_ids,
                story_act=act_name,
                importance=0.75,
                confidence=min(0.92, avg_ev_conf),
                reason="Natural fluent delivery; intervention would add unnecessary visual density.",
            )

        # ── 2. CUT OPPORTUNITY (DEAD SPACE vs PAUSE) ────────────────────
        if opp_type == "cut_opportunity":
            # Check if this pause is a rhetorical hold
            is_rhetorical = any(
                e.type == "rhetorical_pause" or "rhetorical" in e.tags
                for e in supporting_ev if e
            )
            if is_rhetorical:
                return EditDecision(
                    decision_id=self._next_id("keep"),
                    operation="NO_EDIT",
                    status="NO_EDIT",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.80,
                    confidence=avg_ev_conf,
                    reason="Preserved rhetorical pause: Speaker deliberately pauses for impact after a key statement; cutting would damage emotional cadence.",
                )

            if dur < 0.35:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="CUT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.30,
                    confidence=avg_ev_conf,
                    reason="Rejected: Micro-pause is part of natural cadence; cutting would make speech sound unnatural and rushed.",
                )

            # Dead space cut acceptance
            can_acc, reason = self.budget_tracker.can_accept("REMOVE_DEAD_SPACE", st, en)
            if can_acc:
                trim_dur = round(max(0.12, dur - 0.10), 3)
                self.budget_tracker.consume("REMOVE_DEAD_SPACE", st, en)
                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="REMOVE_DEAD_SPACE",
                    status="ACCEPTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.85,
                    confidence=avg_ev_conf,
                    reason=f"Remove {dur:.2f}s hesitation dead space to tighten delivery rhythm while retaining a 0.10s natural breath buffer.",
                    trim_duration=trim_dur,
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="REMOVE_DEAD_SPACE",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.40,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # ── 3. HERO TEXT OPPORTUNITY ───────────────────────────────────
        if opp_type == "hero_text_opportunity":
            # Hero text strictly for primary turning points in HOOK or REALIZATION
            if act_name not in ("HOOK", "REALIZATION"):
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="HERO_TEXT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.30,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: HeroText reserved exclusively for opening premise and core thesis realization (found in {act_name}).",
                )

            can_acc, reason = self.budget_tracker.can_accept("HERO_TEXT", st, en)
            if can_acc:
                self.budget_tracker.consume("HERO_TEXT", st, en)
                # Dynamically derive HeroText title from transcript keywords in this act/time range
                words_in_span = [
                    w.word.strip().upper() for w in self.transcript.words
                    if not (w.end < st or w.start > min(self.duration, st + 3.5))
                    and len(w.word.strip().strip(".,?!")) > 1
                ]
                clean_words = [
                    w.strip(".,?!") for w in words_in_span
                    if w.strip(".,?!") not in ("THE", "AND", "THAT", "THIS", "WITH", "FROM", "THEY", "HAVE", "WAS", "WERE", "ALSO", "WHEN", "IN", "TO", "OF", "YOU", "ARE", "FOR", "I")
                ]
                if clean_words:
                    text = " ".join(clean_words[:3])
                else:
                    text = " ".join([w.strip(".,?!") for w in words_in_span[:3]]) if words_in_span else "THEME TITLE"

                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="HERO_TEXT",
                    status="ACCEPTED",
                    start=st,
                    end=min(self.duration, st + 2.5),
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.92,
                    confidence=min(0.90, avg_ev_conf),
                    reason=f"Primary narrative title card highlighting core theme in {act_name} act.",
                    text_content=text,
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="HERO_TEXT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.50,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # ── 4. PUNCH-IN OPPORTUNITY ────────────────────────────────────
        if opp_type == "punch_in_opportunity":
            # Decide conceptual strength
            if act_name == "REALIZATION":
                strength = "STRONG"
                imp = 0.90
            elif act_name == "HOOK":
                strength = "MEDIUM"
                imp = 0.85
            else:
                strength = "SUBTLE"
                imp = 0.70

            # Restraint in PRESERVE regions
            if classification == "PRESERVE":
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="PUNCH_IN",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.40,
                    confidence=avg_ev_conf,
                    reason="Rejected: Framing is already natural and steady; artificial jump cut would interrupt conversational fluency.",
                )

            can_acc, reason = self.budget_tracker.can_accept("PUNCH_IN", st, en, strength)
            if can_acc:
                self.budget_tracker.consume("PUNCH_IN", st, en, strength)
                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="PUNCH_IN",
                    status="ACCEPTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=imp,
                    confidence=avg_ev_conf,
                    reason=f"Framing emphasis jump cut ({strength}) to amplify thesis argument and re-anchor viewer attention.",
                    strength=strength,
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="PUNCH_IN",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.45,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # ── 5. EMPHASIS TEXT OPPORTUNITY ───────────────────────────────
        if opp_type == "emphasis_text_opportunity":
            # Extract word from evidence
            kw = "keyword"
            for ev in supporting_ev:
                if ev and ev.signals.speech.get("word"):
                    kw = ev.signals.speech["word"].strip(".,!?:;\"")
                    break

            # Reject common non-semantic words ("I", "the", "at", "and", "or")
            if kw.lower() in ("i", "the", "at", "and", "or", "to", "in", "of", "a", "us"):
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="EMPHASIS_TEXT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.25,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: Word '{kw}' is a common functional grammatical word without core semantic thesis value.",
                )

            can_acc, reason = self.budget_tracker.can_accept("EMPHASIS_TEXT", st, en)
            if can_acc:
                self.budget_tracker.consume("EMPHASIS_TEXT", st, en)
                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="EMPHASIS_TEXT",
                    status="ACCEPTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.88,
                    confidence=avg_ev_conf,
                    reason=f"Highlight semantically critical word '{kw}' aligned with high vocal emphasis.",
                    text_content=kw.upper(),
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="EMPHASIS_TEXT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.45,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # ── 6. B-ROLL OPPORTUNITY ──────────────────────────────────────
        if opp_type == "broll_opportunity":
            # Highly selective: accept at most 2–3 in CONTEXT / PROBLEM
            if act_name not in ("CONTEXT", "PROBLEM"):
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="BROLL_SUPPORT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.35,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: B-roll cutaway not editorially warranted in {act_name} act; speaker's on-camera delivery carries higher authenticity.",
                )

            can_acc, reason = self.budget_tracker.can_accept("BROLL_CONTEXT", st, en)
            if can_acc:
                self.budget_tracker.consume("BROLL_CONTEXT", st, en)
                # Derive concept query dynamically from transcript words in this span
                words_in_span = [
                    w.word.strip() for w in self.transcript.words
                    if not (w.end < st or w.start > en)
                    and len(w.word.strip().strip(".,?!")) > 2
                ]
                clean_words = [
                    w.strip(".,?!").lower() for w in words_in_span
                    if w.strip(".,?!").lower() not in ("the", "and", "that", "this", "with", "from", "they", "have", "was", "were", "also", "about", "there", "their", "when")
                ]
                concept = " ".join(clean_words[:5]) if clean_words else f"{act_name.lower()} narrative context"

                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="BROLL_CONTEXT",
                    status="ACCEPTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.78,
                    confidence=min(0.85, avg_ev_conf),
                    reason=f"Contextual B-roll cutaway illustrating {concept}, grounding the speaker delivery in act {act_name}.",
                    concept_query=concept,
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="BROLL_CONTEXT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.40,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # ── 7. SFX OPPORTUNITY ─────────────────────────────────────────
        if opp_type == "sfx_opportunity":
            # SFX strictly paired with significant narrative inflection
            if act_name not in ("HOOK", "REALIZATION"):
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="SFX_IMPACT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.30,
                    confidence=avg_ev_conf,
                    reason="Rejected: Sound design restraint avoids artificial acoustic clutter on secondary narrative phrases.",
                )

            can_acc, reason = self.budget_tracker.can_accept("SFX_IMPACT", st, en)
            if can_acc:
                self.budget_tracker.consume("SFX_IMPACT", st, en)
                return EditDecision(
                    decision_id=self._next_id("edit"),
                    operation="SFX_IMPACT",
                    status="ACCEPTED",
                    start=st,
                    end=min(self.duration, st + 0.35),
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.75,
                    confidence=avg_ev_conf,
                    reason=f"Subtle acoustic accent to underscore major conceptual inflection in {act_name} act.",
                    sound_cue="impact_subtle",
                )
            else:
                return EditDecision(
                    decision_id=self._next_id("rej"),
                    operation="SFX_IMPACT",
                    status="REJECTED",
                    start=st,
                    end=en,
                    source_evidence_ids=list(opp.evidence_ids),
                    source_opportunity_ids=[opp.opportunity_id],
                    story_observation_ids=so_ids,
                    story_act=act_name,
                    importance=0.40,
                    confidence=avg_ev_conf,
                    reason=f"Rejected: {reason}",
                )

        # Default fallback
        return EditDecision(
            decision_id=self._next_id("rej"),
            operation="KEEP",
            status="REJECTED",
            start=st,
            end=en,
            source_evidence_ids=list(opp.evidence_ids),
            source_opportunity_ids=[opp.opportunity_id],
            story_observation_ids=so_ids,
            story_act=act_name,
            importance=0.20,
            confidence=avg_ev_conf,
            reason="Rejected: Insufficient editorial justification.",
        )

    def _synthesize_coverage_no_edits(
        self,
        accepted: list[EditDecision],
        existing_no_edits: list[EditDecision],
        moments: list[EditorialMoment],
    ) -> list[EditDecision]:
        """Ensures that unedited narrative stretches are explicitly recognized as preserved spans."""
        added: list[EditDecision] = []
        for m in moments:
            if m.classification == "PRESERVE":
                # Check if already covered by an existing no-edit decision
                already_covered = any(
                    abs(ne.start - m.start) < 0.5 for ne in existing_no_edits
                )
                if not already_covered:
                    added.append(
                        EditDecision(
                            decision_id=self._next_id("keep"),
                            operation="NO_EDIT",
                            status="NO_EDIT",
                            start=m.start,
                            end=m.end,
                            source_evidence_ids=list(m.evidence_ids),
                            source_opportunity_ids=[],
                            story_observation_ids=[so.story_observation_id for so in self.story_obs.story_observations if not (so.time_range[1] < m.start or so.time_range[0] > m.end)],
                            story_act=m.story_act,
                            importance=0.75,
                            confidence=0.90,
                            reason="Preserved natural delivery: Conversational cadence and facial framing are strong; editorial intervention would add friction.",
                        )
                    )
        return added

    def _find_act_for_interval(self, start: float, end: float) -> str:
        for so in self.story_obs.story_observations:
            if not (so.time_range[1] < start or so.time_range[0] > end):
                return so.type
        return "CONTEXT"

    def _find_moment_for_time(self, t: float, moments: list[EditorialMoment]) -> Optional[EditorialMoment]:
        for m in moments:
            if m.start <= t <= m.end:
                return m
        return None

    def _find_realization_act(self) -> str:
        for so in self.story_obs.story_observations:
            if so.type in ("REALIZATION", "PAYOFF"):
                return so.type
        return "REALIZATION"
