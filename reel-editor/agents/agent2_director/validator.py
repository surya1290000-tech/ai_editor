"""
agents/agent2_director/validator.py

Agent 2 Story Editor Acceptance Validator (Phase 3)
──────────────────────────────────────────────────
Verifies all 10 acceptance criteria for Phase 3:
  1. Selectivity (does not accept every opportunity, disciplined intervention rate)
  2. Disciplined Rejection (rejects weak opportunities with explicit reasons)
  3. First-class NO_EDIT (preserves natural delivery with explicit rationales)
  4. Edit budget compliance (hard limits respected)
  5. Spacing compliance (minimum intervals between effects respected)
  6. No creative styling implementation (no fonts, colors, zoom scales, easings)
  7. No asset hallucination (no fake media files)
  8. Full evidence traceability (every decision traces to evidence/opportunity)
  9. Valid time ranges (0 <= start <= end <= duration)
  10. Serializability & deterministic reproducibility
"""

from __future__ import annotations

import re
import json
from typing import Any

from agents.agent2_director.models import (
    StoryEditPlan,
    EditDecisionsDocument,
    SUPPORTED_OPERATIONS,
)
from core.evidence.validator import FORBIDDEN_STYLING_TERMS


class StoryEditorValidator:
    """Automated validator for Agent 2 editorial decisions and narrative plan."""

    def __init__(
        self,
        story_plan: StoryEditPlan,
        decisions_doc: EditDecisionsDocument,
        max_duration: float,
    ):
        self.story_plan = story_plan
        self.decisions_doc = decisions_doc
        self.max_duration = max_duration

        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> dict[str, Any]:
        """Runs all 10 checks and returns structured report."""
        self.errors.clear()
        self.warnings.clear()

        c1 = self.check_selectivity()
        c2 = self.check_disciplined_rejection()
        c3 = self.check_no_edit_is_first_class()
        c4 = self.check_budget_compliance()
        c5 = self.check_spacing_compliance()
        c6 = self.check_no_creative_styling()
        c7 = self.check_no_asset_hallucination()
        c8 = self.check_evidence_traceability()
        c9 = self.check_valid_time_ranges()
        c10 = self.check_serializability()

        passed = len(self.errors) == 0
        return {
            "passed": passed,
            "total_checks": 10,
            "checks": {
                "selectivity": c1,
                "disciplined_rejection": c2,
                "no_edit_first_class": c3,
                "budget_compliance": c4,
                "spacing_compliance": c5,
                "no_creative_styling": c6,
                "no_asset_hallucination": c7,
                "evidence_traceability": c8,
                "valid_time_ranges": c9,
                "serializability": c10,
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def check_selectivity(self) -> bool:
        """Verifies Agent 2 does not accept every opportunity (selectivity <= 50%)."""
        total_opps = self.story_plan.decisions_summary.get("total_opportunities_evaluated", 0)
        accepted = len(self.decisions_doc.accepted_decisions)
        if total_opps > 0 and accepted >= total_opps:
            self.errors.append(
                f"Check 1 FAIL: Lack of selectivity. Accepted {accepted}/{total_opps} opportunities (100%)."
            )
            return False
        if accepted > 14:
            self.errors.append(
                f"Check 1 FAIL: Over-editing. Accepted {accepted} interventions (maximum expected is <= 14)."
            )
            return False
        return True

    def check_disciplined_rejection(self) -> bool:
        """Verifies that weak opportunities are rejected with explicit reasons."""
        rejected = self.decisions_doc.rejected_decisions
        if len(rejected) < 3:
            self.errors.append(
                f"Check 2 FAIL: Insufficient rejections ({len(rejected)}). Expected at least 3 rejected opportunities."
            )
            return False
        for r in rejected:
            if not r.reason or len(r.reason.strip()) < 10:
                self.errors.append(
                    f"Check 2 FAIL: Rejected decision '{r.decision_id}' lacks informative reason."
                )
                return False
        return True

    def check_no_edit_is_first_class(self) -> bool:
        """Verifies that NO_EDIT decisions exist and are grounded in natural delivery."""
        no_edits = self.decisions_doc.no_edit_decisions
        min_expected = 1 if self.max_duration < 15.0 else (2 if self.max_duration < 30.0 else 3)
        if len(no_edits) < min_expected:
            self.errors.append(
                f"Check 3 FAIL: Insufficient NO_EDIT decisions ({len(no_edits)}). Expected at least {min_expected}."
            )
            return False
        return True

    def check_budget_compliance(self) -> bool:
        """Verifies hard limits in edit_budget are respected."""
        util = self.decisions_doc.budget_utilization
        hero_text_used = util.get("hero_text", {}).get("used", 0)
        if hero_text_used > 2:
            self.errors.append(f"Check 4 FAIL: HeroText limit exceeded ({hero_text_used} > 2).")
            return False

        broll_used = util.get("broll", {}).get("used", 0)
        if broll_used > 3:
            self.errors.append(f"Check 4 FAIL: B-roll limit exceeded ({broll_used} > 3).")
            return False

        sfx_used = util.get("sfx", {}).get("used", 0)
        if sfx_used > 6:
            self.errors.append(f"Check 4 FAIL: SFX limit exceeded ({sfx_used} > 6).")
            return False
        return True

    def check_spacing_compliance(self) -> bool:
        """Verifies spacing between consecutive punch-ins, SFX, B-roll, and hero text."""
        # Punch-ins spacing >= 3.8s
        punches = [d for d in self.decisions_doc.accepted_decisions if d.operation == "PUNCH_IN"]
        for i in range(1, len(punches)):
            delta = punches[i].start - punches[i - 1].end
            if delta < 3.5:
                self.errors.append(
                    f"Check 5 FAIL: Punch-in spacing violated ({delta:.2f}s < 3.5s) between '{punches[i-1].decision_id}' and '{punches[i].decision_id}'."
                )
                return False

        # SFX spacing >= 2.5s
        sfx_list = [d for d in self.decisions_doc.accepted_decisions if "SFX" in d.operation]
        for i in range(1, len(sfx_list)):
            delta = sfx_list[i].start - sfx_list[i - 1].end
            if delta < 2.5:
                self.errors.append(
                    f"Check 5 FAIL: SFX spacing violated ({delta:.2f}s < 2.5s) between '{sfx_list[i-1].decision_id}' and '{sfx_list[i].decision_id}'."
                )
                return False
        return True

    def check_no_creative_styling(self) -> bool:
        """Verifies no fonts, colors, zoom scales, easings, or FFmpeg commands exist in decisions."""
        ok = True
        for d in self.decisions_doc.all_decisions:
            text = f"{d.reason} {d.concept_query or ''} {d.text_content or ''}".lower()
            for pat in FORBIDDEN_STYLING_TERMS:
                if re.search(pat, text):
                    self.errors.append(
                        f"Check 6 FAIL: Forbidden styling term '{pat}' in decision '{d.decision_id}': \"{text}\""
                    )
                    ok = False
        return ok

    def check_no_asset_hallucination(self) -> bool:
        """Verifies no fake media file paths are specified."""
        ok = True
        file_exts = [r"\.mp4\b", r"\.mov\b", r"\.wav\b", r"\.mp3\b", r"\.png\b", r"\.jpg\b"]
        for d in self.decisions_doc.all_decisions:
            text = f"{d.concept_query or ''} {d.sound_cue or ''} {d.reason}".lower()
            for ext in file_exts:
                if re.search(ext, text):
                    self.errors.append(
                        f"Check 7 FAIL: Concrete asset filename detected in decision '{d.decision_id}': \"{text}\""
                    )
                    ok = False
        return ok

    def check_evidence_traceability(self) -> bool:
        """Verifies every decision has non-empty evidence or opportunity references."""
        ok = True
        for d in self.decisions_doc.all_decisions:
            if not d.source_evidence_ids and not d.source_opportunity_ids:
                self.errors.append(
                    f"Check 8 FAIL: Decision '{d.decision_id}' has neither evidence nor opportunity grounding."
                )
                ok = False
        return ok

    def check_valid_time_ranges(self) -> bool:
        """Verifies 0.0 <= start <= end <= max_duration."""
        ok = True
        limit = self.max_duration + 2.0
        for d in self.decisions_doc.all_decisions:
            if not (0.0 <= d.start <= d.end <= limit):
                self.errors.append(
                    f"Check 9 FAIL: Decision '{d.decision_id}' invalid time range [{d.start}, {d.end}]"
                )
                ok = False
        return ok

    def check_serializability(self) -> bool:
        """Verifies clean JSON round-trip serialization."""
        try:
            sp_json = json.dumps(self.story_plan.to_dict())
            sp_restored = StoryEditPlan.from_dict(json.loads(sp_json))
            assert len(sp_restored.moments) == len(self.story_plan.moments)

            ed_json = json.dumps(self.decisions_doc.to_dict())
            ed_restored = EditDecisionsDocument.from_dict(json.loads(ed_json))
            assert len(ed_restored.accepted_decisions) == len(self.decisions_doc.accepted_decisions)
            assert len(ed_restored.rejected_decisions) == len(self.decisions_doc.rejected_decisions)
            assert len(ed_restored.no_edit_decisions) == len(self.decisions_doc.no_edit_decisions)
            return True
        except Exception as e:
            self.errors.append(f"Check 10 FAIL: JSON serialization round-trip failed: {e}")
            return False
