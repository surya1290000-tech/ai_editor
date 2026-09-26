"""
core/evidence/story_analyzer.py

Narrative Structure & Story Observations (Phase 2)
─────────────────────────────────────────────────
Derives narrative acts and semantic structure:
  HOOK → CONTEXT → PROBLEM → BUILD_UP → EXAMPLE → CONTRAST → REALIZATION → PAYOFF → CTA → ENDING

Key Design Rules:
  - Every story observation MUST reference underlying evidence IDs.
  - Strictly an OBSERVER describing story progression, NOT editorial commands.
  - Confidence reflects underlying evidence uncertainty (no ungrounded 0.99 confidence).
"""

from __future__ import annotations

from typing import Any, Optional

from core.logger import get_logger
from core.evidence.models import (
    EvidenceInventory,
    StoryObservation,
    StoryObservationsInventory,
    EnrichedTranscript,
)

logger = get_logger(__name__)


class StoryObserver:
    """Analyzes narrative structure and produces grounded StoryObservation items."""

    def __init__(
        self,
        evidence_inv: Optional[EvidenceInventory] = None,
        transcript: Optional[EnrichedTranscript] = None,
        duration: float = 0.0,
    ):
        self.evidence_inv = evidence_inv
        self.transcript = transcript
        self.duration = duration
        self._story_counter = 0

    def _next_id(self) -> str:
        self._story_counter += 1
        return f"story_{self._story_counter:02d}"

    def analyze_story_structure(
        self,
        story_analysis_raw: Optional[dict[str, Any]] = None,
    ) -> StoryObservationsInventory:
        """
        Maps video timeline into narrative acts, each strictly linked to supporting evidence IDs.
        Uses Stage 6 LLM/heuristic acts if available, otherwise performs deterministic temporal partitioning.
        """
        self._story_counter = 0
        story_obs_list: list[StoryObservation] = []

        sentences = self.transcript.sentences
        if not sentences:
            return StoryObservationsInventory(
                source_video=self.evidence_inv.source_video,
                duration_seconds=self.duration,
                story_observations=[],
            )

        # Check if raw narrative acts exist from Stage 6 and provide meaningful coverage
        raw_acts = []
        if story_analysis_raw and "narrative_acts" in story_analysis_raw:
            acts_cand = story_analysis_raw["narrative_acts"]
            if len(acts_cand) >= 3:
                max_end = max(float(a.get("end_time", a.get("end", a.get("end_seconds", 0)))) for a in acts_cand)
                if max_end >= self.duration * 0.70:
                    raw_acts = acts_cand

        if not raw_acts:
            raw_acts = self._deterministic_partition(sentences)

        for act in raw_acts:
            raw_name = str(act.get("act", act.get("act_name", act.get("name", "CONTEXT")))).upper()
            # If name has multiple pipe-separated or hyphenated names, extract first valid
            act_name = "CONTEXT"
            for token in raw_name.replace("|", " ").replace("-", " ").replace("&", " ").split():
                if token in ("HOOK", "CONTEXT", "PROBLEM", "BUILD_UP", "EXAMPLE", "CONTRAST", "REALIZATION", "PAYOFF", "CTA", "ENDING"):
                    act_name = token
                    break

            st = float(act.get("start", act.get("start_time", act.get("start_seconds", 0.0))))
            en = float(act.get("end", act.get("end_time", act.get("end_seconds", self.duration))))
            en = min(self.duration, max(st + 0.1, en))

            # Find all evidence items overlapping this act
            supporting_evidence = self.evidence_inv.filter_by_time_range(st, en)
            if not supporting_evidence:
                supporting_evidence = self.evidence_inv.evidence[:2]

            supporting_ids = [e.evidence_id for e in supporting_evidence]

            # Calculate confidence respecting evidence uncertainty
            ev_confidences = [e.confidence for e in supporting_evidence]
            avg_ev_conf = sum(ev_confidences) / len(ev_confidences) if ev_confidences else 0.8
            has_uncertainty = any(len(e.uncertainty) > 0 for e in supporting_evidence)

            act_conf = round(avg_ev_conf * (0.85 if has_uncertainty else 1.0), 3)

            summary = str(
                act.get(
                    "summary",
                    act.get(
                        "editorial_role",
                        act.get("description", f"Narrative section functioning as {act_name}."),
                    ),
                )
            )

            # Ensure summary is an observation, not an edit command
            summary = self._sanitize_observation(summary, act_name)

            story_obs_list.append(
                StoryObservation(
                    story_observation_id=self._next_id(),
                    type=act_name,
                    time_range=[round(st, 3), round(en, 3)],
                    confidence=act_conf,
                    evidence_ids=supporting_ids,
                    summary=summary,
                    signals={
                        "sentences_count": sum(1 for s in sentences if not (s.end < st or s.start > en)),
                        "supporting_evidence_count": len(supporting_ids),
                    },
                )
            )

        return StoryObservationsInventory(
            version="3.0.0",
            source_video=self.evidence_inv.source_video,
            duration_seconds=self.duration,
            story_observations=story_obs_list,
            narrative_health={
                "total_acts": len(story_obs_list),
                "has_hook": any(so.type == "HOOK" for so in story_obs_list),
                "has_realization": any(so.type in ("REALIZATION", "PAYOFF") for so in story_obs_list),
            },
        )

    def _deterministic_partition(self, sentences: list[Any]) -> list[dict[str, Any]]:
        """Sentence-aligned narrative partition based on rhetorical structure and speech timing."""
        total_s = len(sentences)
        dur = self.duration if self.duration > 0 else 60.0

        if total_s == 0 or (total_s == 1 and dur >= 20.0):
            # Partition along time intervals for single-block transcripts or long takes
            t_hook = round(dur * 0.15, 2)
            t_ctx = round(dur * 0.40, 2)
            t_prob = round(dur * 0.68, 2)
            t_real = round(dur * 0.88, 2)

            text_preview = sentences[0].text[:80].strip() if total_s == 1 and hasattr(sentences[0], "text") else "Speaker delivery"

            return [
                {"act": "HOOK", "start": 0.0, "end": t_hook, "description": f"Opening hook establishing premise: '{text_preview}'."},
                {"act": "CONTEXT", "start": t_hook, "end": t_ctx, "description": "Contextual setup and background exposition."},
                {"act": "PROBLEM", "start": t_ctx, "end": t_prob, "description": "Core narrative tension and problem exploration."},
                {"act": "REALIZATION", "start": t_prob, "end": t_real, "description": "Key conceptual turn and realization."},
                {"act": "PAYOFF", "start": t_real, "end": round(dur, 2), "description": "Concluding resolution and takeaway."},
            ]

        if total_s == 1:
            return [{
                "act": "HOOK",
                "start": 0.0,
                "end": round(dur, 2),
                "description": f"Opening short delivery: '{sentences[0].text[:60] if hasattr(sentences[0], 'text') else 'Delivery'}'.",
            }]

        if total_s >= 8:
            # 5-act narrative arc dynamically partitioned by sentence boundaries
            k1 = max(1, min(total_s - 4, int(round(total_s * 0.15))))
            k2 = max(k1 + 1, min(total_s - 3, int(round(total_s * 0.38))))
            k3 = max(k2 + 1, min(total_s - 2, int(round(total_s * 0.68))))
            k4 = max(k3 + 1, min(total_s - 1, int(round(total_s * 0.88))))

            s_hook = sentences[k1]
            s_ctx = sentences[k2]
            s_prob = sentences[k3]
            s_real = sentences[k4]

            p_hook = sentences[0].text[:60] if hasattr(sentences[0], "text") else ""
            p_ctx = sentences[k1].text[:60] if hasattr(sentences[k1], "text") else ""
            p_prob = sentences[k2].text[:60] if hasattr(sentences[k2], "text") else ""
            p_real = sentences[k3].text[:60] if hasattr(sentences[k3], "text") else ""
            p_pay = sentences[k4].text[:60] if hasattr(sentences[k4], "text") else ""

            return [
                {"act": "HOOK", "start": 0.0, "end": round(s_hook.end, 2), "description": f"Opening thesis: '{p_hook}'."},
                {"act": "CONTEXT", "start": round(s_hook.end, 2), "end": round(s_ctx.end, 2), "description": f"Contextual development: '{p_ctx}'."},
                {"act": "PROBLEM", "start": round(s_ctx.end, 2), "end": round(s_prob.end, 2), "description": f"Core tension/problem: '{p_prob}'."},
                {"act": "REALIZATION", "start": round(s_prob.end, 2), "end": round(s_real.end, 2), "description": f"Narrative realization: '{p_real}'."},
                {"act": "PAYOFF", "start": round(s_real.end, 2), "end": round(dur, 2), "description": f"Closing takeaway: '{p_pay}'."},
            ]

        # Standard 3-act partition for 2–7 sentences
        p1 = sentences[max(0, total_s // 3)].end
        p2 = sentences[max(1, (2 * total_s) // 3)].end
        return [
            {"act": "HOOK", "start": 0.0, "end": round(p1, 2), "description": f"Opening segment: '{sentences[0].text[:60] if hasattr(sentences[0], 'text') else 'Hook'}'."},
            {"act": "CONTEXT", "start": round(p1, 2), "end": round(p2, 2), "description": "Core context and narrative elaboration."},
            {"act": "REALIZATION", "start": round(p2, 2), "end": round(dur, 2), "description": "Resolution and final takeaway."},
        ]

    def _sanitize_observation(self, text: str, act_name: str) -> str:
        """Ensures text is descriptive of narrative reality, not an editorial command."""
        text = text.replace("Add ", "Features ")
        text = text.replace("Insert ", "Presents ")
        text = text.replace("Cut to ", "Transitions to ")
        text = text.replace("Zoom in on ", "Visual focus aligns with ")
        if not text:
            text = f"Narrative section functioning as {act_name}."
        return text
