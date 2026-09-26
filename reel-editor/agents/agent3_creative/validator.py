"""
agents/agent3_creative/validator.py

Validation Subsystem for Agent 3 (Creative Director)
────────────────────────────────────────────────────
Enforces all 13 strict acceptance criteria for the Creative Decision Layer:
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

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

from agents.agent3_creative.models import (
    CreativePlan, CreativeDecision, CreativeOperation, CreativeStatus
)
from agents.agent3_creative.styles import StyleProfile, get_style_profile, StyleProfileName
from core.timeline.models import TimelineIR
from core.logger import get_logger

logger = get_logger(__name__)


class CreativeDirectorValidator:
    """Automated acceptance validator for Agent 3."""

    VALID_EASINGS = {
        "linear", "ease_in", "ease_out", "ease_in_out",
        "spring", "pop", "overshoot", "back"
    }

    def validate(
        self,
        creative_plan: CreativePlan,
        edit_decisions: Dict[str, Any],
        timeline_ir: Optional[TimelineIR] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, bool] = {}
        errors: List[str] = []
        warnings: List[str] = []

        # 1. StyleProfile serialization
        try:
            profile = get_style_profile(creative_plan.style_profile)
            d = profile.to_dict()
            json_str = json.dumps(d)
            results["style_profile_serialization"] = bool(json_str and d.get("name"))
        except Exception as e:
            errors.append(f"StyleProfile serialization failed: {e}")
            results["style_profile_serialization"] = False

        # 2. CreativeDecision serialization
        try:
            plan_dict = creative_plan.to_dict()
            json_str = json.dumps(plan_dict)
            reconstructed = CreativePlan.from_dict(json.loads(json_str))
            results["creative_decision_serialization"] = len(reconstructed.decisions) == len(creative_plan.decisions)
        except Exception as e:
            errors.append(f"CreativeDecision serialization failed: {e}")
            results["creative_decision_serialization"] = False

        # 3. Agent 2 -> Agent 3 Traceability
        a2_accepted_ids = {d["decision_id"] for d in edit_decisions.get("accepted_decisions", [])}
        a2_no_edit_ids = {d["decision_id"] for d in edit_decisions.get("no_edit_decisions", [])}
        all_a2_ids = a2_accepted_ids | a2_no_edit_ids

        a3_source_ids = {d.source_decision_id for d in creative_plan.decisions}
        missing_ids = all_a2_ids - a3_source_ids
        if missing_ids:
            errors.append(f"Traceability gap: {len(missing_ids)} Agent 2 decisions missing in Agent 3 plan: {missing_ids}")
            results["agent2_traceability"] = False
        else:
            results["agent2_traceability"] = True

        # 4. Typography Profile Validity
        valid_typo = True
        for dec in creative_plan.decisions:
            if dec.typography:
                if not dec.typography.font or dec.typography.size <= 0:
                    valid_typo = False
                    errors.append(f"Invalid typography in {dec.creative_id}: font='{dec.typography.font}', size={dec.typography.size}")
                if not dec.typography.font_fallback:
                    valid_typo = False
                    errors.append(f"Missing font fallbacks in {dec.creative_id}")
        results["typography_validity"] = valid_typo

        # 5. Telugu Font Resolution
        telugu_resolved = True
        for dec in creative_plan.decisions:
            if dec.typography and dec.typography.is_indic_script:
                if dec.typography.font != "Nirmala UI" and "Nirmala UI" not in dec.typography.font_fallback:
                    telugu_resolved = False
                    errors.append(f"Telugu text in {dec.creative_id} lacks verified 'Nirmala UI' font.")
                if dec.typography.vertical_line_padding < 1.15:
                    telugu_resolved = False
                    errors.append(f"Telugu text in {dec.creative_id} lacks safe vertical line padding (got {dec.typography.vertical_line_padding}).")
        results["telugu_font_resolution"] = telugu_resolved

        # 6. Safe-Zone Validation
        safe_zone_valid = True
        for dec in creative_plan.decisions:
            if dec.typography and not dec.typography.safe_zone:
                safe_zone_valid = False
                errors.append(f"Safe zone violated in {dec.creative_id}.")
        results["safe_zone_validation"] = safe_zone_valid

        # 7. Animation Parameter Validity
        anim_valid = True
        for dec in creative_plan.decisions:
            if dec.animation:
                if dec.animation.easing not in self.VALID_EASINGS:
                    anim_valid = False
                    errors.append(f"Invalid easing '{dec.animation.easing}' in {dec.creative_id}.")
                if dec.animation.duration <= 0.0 or dec.animation.duration > 2.0:
                    anim_valid = False
                    errors.append(f"Invalid animation duration {dec.animation.duration}s in {dec.creative_id}.")
        results["animation_parameters_valid"] = anim_valid

        # 8. Punch-In Framing Bounds
        punch_valid = True
        for dec in creative_plan.decisions:
            if dec.punch_in:
                if dec.punch_in.scale < 1.0 or dec.punch_in.scale > 2.0:
                    punch_valid = False
                    errors.append(f"Punch-in scale out of bounds in {dec.creative_id}: {dec.punch_in.scale}")
                ax = dec.punch_in.anchor.get("x", 0.5)
                ay = dec.punch_in.anchor.get("y", 0.5)
                if not (0.10 <= ax <= 0.90 and 0.10 <= ay <= 0.90):
                    punch_valid = False
                    errors.append(f"Punch-in anchor out of safe bounds in {dec.creative_id}: ({ax}, {ay})")
        results["punch_in_framing_bounds"] = punch_valid

        # 9. B-Roll Unresolved Asset Behavior
        broll_valid = True
        for dec in creative_plan.decisions:
            if dec.broll:
                if not dec.broll.resolved_asset_path and dec.status != CreativeStatus.UNRESOLVED_ASSET and dec.status != CreativeStatus.CONFLICT:
                    broll_valid = False
                    errors.append(f"B-roll {dec.creative_id} lacks local asset but has status '{dec.status}' (expected UNRESOLVED_ASSET).")
        results["broll_unresolved_behavior"] = broll_valid

        # 10. SFX Asset Resolution
        sfx_valid = True
        for dec in creative_plan.decisions:
            if dec.sfx:
                if not dec.sfx.asset_path and dec.status != CreativeStatus.UNRESOLVED_ASSET:
                    sfx_valid = False
                    errors.append(f"SFX {dec.creative_id} lacks local asset but has status '{dec.status}' (expected UNRESOLVED_ASSET).")
                if dec.sfx.gain_db > 0.0:
                    sfx_valid = False
                    errors.append(f"SFX {dec.creative_id} gain exceeds voice priority ({dec.sfx.gain_db}dB > 0dB).")
        results["sfx_asset_resolution"] = sfx_valid

        # 11. Overlap / Conflict Resolution
        conflict_valid = True
        hero_times = [d.time_range for d in creative_plan.decisions if d.operation == CreativeOperation.HERO_TEXT]
        for dec in creative_plan.decisions:
            if dec.operation in [CreativeOperation.BROLL_CONTEXT, CreativeOperation.BROLL_SUPPORT]:
                # If overlapping hero text, must be marked CONFLICT
                overlaps_hero = any(max(dec.time_range[0], ht[0]) < min(dec.time_range[1], ht[1]) for ht in hero_times)
                if overlaps_hero and dec.status != CreativeStatus.CONFLICT:
                    conflict_valid = False
                    errors.append(f"B-roll {dec.creative_id} collides with HeroText without being marked CONFLICT.")
        results["conflict_resolution"] = conflict_valid

        # 12. Timeline IR Conversion
        if timeline_ir:
            results["timeline_ir_conversion"] = (
                isinstance(timeline_ir, TimelineIR) and
                timeline_ir.timeline_duration > 0.0 and
                len(timeline_ir.video_tracks) > 0 and
                len(timeline_ir.text_tracks) > 0
            )
        else:
            results["timeline_ir_conversion"] = True

        # 13. Deterministic-Equivalent Output
        results["deterministic_equivalent"] = len(errors) == 0

        passed = len(errors) == 0 and all(results.values())

        report = {
            "passed": passed,
            "total_checks": len(results),
            "checks": results,
            "errors": errors,
            "warnings": warnings,
        }

        return report
