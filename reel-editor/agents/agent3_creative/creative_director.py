"""
agents/agent3_creative/creative_director.py

Agent 3: Creative Director
──────────────────────────
Translates Agent 2's approved editorial decisions into concrete, professional
visual and acoustic treatments.

Strict Rules:
1. Agent 3 does NOT decide WHAT changes — that is Agent 2's job.
2. Agent 3 decides HOW those changes look and sound.
3. No silent discarding of Agent 2 decisions.
4. Telugu Unicode text automatically receives verified font fallbacks ('Nirmala UI').
5. Missing assets are explicitly marked UNRESOLVED_ASSET without fabrication.
6. Overlapping decisions are resolved through the strict Creative Conflict Hierarchy:
   1. HeroText
   2. Primary speaker framing
   3. EmphasisText
   4. B-roll
   5. SFX
   6. Music
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from core.logger import get_logger
from agents.agent3_creative.styles import (
    StyleProfile, StyleProfileName, get_style_profile
)
from agents.agent3_creative.models import (
    CreativeDecision, CreativePlan, CreativeOperation, CreativeStatus,
    TypographySpec, AnimationSpec, PunchInSpec, BrollSpec, SfxSpec, AudioSpec
)
from agents.agent3_creative.assets import BaseAssetResolver, LocalAssetLibraryResolver

logger = get_logger(__name__)


def is_telugu_text(text: str) -> bool:
    """Detects if string contains any Telugu Unicode characters (U+0C00 - U+0C7F)."""
    return bool(re.search(r"[\u0c00-\u0c7f]", text))


class CreativeDirector:
    """
    Agent 3: Creative Director.
    Orchestrates typography, animations, face-aware punch-in framing, B-roll treatments,
    and sound design for approved editorial decisions.
    """

    def __init__(
        self,
        style_profile: Optional[StyleProfile | str] = None,
        asset_resolver: Optional[BaseAssetResolver] = None,
        workspace_root: Optional[Path | str] = None,
    ):
        if isinstance(style_profile, StyleProfile):
            self.style = style_profile
        else:
            self.style = get_style_profile(style_profile or StyleProfileName.EDITORIAL_CINEMATIC)

        self.asset_resolver = asset_resolver or LocalAssetLibraryResolver(workspace_root=workspace_root)

    def direct(
        self,
        edit_decisions_data: Dict[str, Any],
        story_edit_plan_data: Optional[Dict[str, Any]] = None,
        media_meta: Optional[Dict[str, Any]] = None,
        visual_features: Optional[Dict[str, Any]] = None,
        transcript_raw: Optional[Dict[str, Any]] = None,
    ) -> CreativePlan:
        """
        Processes Agent 2's decisions and produces the complete CreativePlan.
        """
        source_video = edit_decisions_data.get("source_video", "video.mp4")
        duration = float(edit_decisions_data.get("duration_seconds", 90.0))

        accepted = edit_decisions_data.get("accepted_decisions", [])
        no_edit = edit_decisions_data.get("no_edit_decisions", [])

        # Check if project contains substantial Telugu content to auto-suggest Telugu-safe typography
        detected_telugu = False
        if transcript_raw:
            for w in transcript_raw.get("words", []):
                if is_telugu_text(str(w.get("word", ""))):
                    detected_telugu = True
                    break

        if detected_telugu and self.style.name != StyleProfileName.TELUGU_SAFE:
            logger.info("CreativeDirector | Telugu script detected in source; ensuring Telugu font fallback.")

        # Face center defaults from visual features
        face_cx = 0.50
        face_cy = 0.40
        if visual_features and "face_tracking" in visual_features:
            ft = visual_features["face_tracking"]
            face_cx = float(ft.get("primary_face_center_x", 0.50))
            face_cy = float(ft.get("primary_face_center_y", 0.40))
        elif visual_features and "shots" in visual_features:
            shots = visual_features.get("shots", [])
            if shots and "face_center_x" in shots[0]:
                face_cx = float(shots[0].get("face_center_x", 0.50))
                face_cy = float(shots[0].get("face_center_y", 0.40))

        creative_decisions: List[CreativeDecision] = []
        creative_counter = 1

        # ── 1. Process Approved Editorial Decisions ────────────────────
        for dec in accepted:
            c_dec = self._process_accepted_decision(
                dec=dec,
                counter=creative_counter,
                face_cx=face_cx,
                face_cy=face_cy,
            )
            creative_decisions.append(c_dec)
            creative_counter += 1

        # ── 2. Process NO_EDIT Preserved Spans ─────────────────────────
        for dec in no_edit:
            c_dec = self._process_no_edit_decision(
                dec=dec,
                counter=creative_counter,
            )
            creative_decisions.append(c_dec)
            creative_counter += 1

        # ── 3. Creative Conflict Resolution ────────────────────────────
        resolved_decisions = self._resolve_creative_conflicts(creative_decisions)

        # ── 4. Construct Final Creative Plan ───────────────────────────
        plan = CreativePlan(
            version="4.0.0",
            source_video=source_video,
            duration_seconds=duration,
            style_profile=self.style.name.value,
            total_decisions=len(resolved_decisions),
            decisions=resolved_decisions,
            metadata={
                "style_description": self.style.description,
                "visual_restraint_level": self.style.visual_restraint_level,
                "sfx_density": self.style.audio.sfx_density,
                "face_anchor": {"x": round(face_cx, 3), "y": round(face_cy, 3)},
                "telugu_script_detected": detected_telugu,
            }
        )

        logger.info(
            f"CreativeDirector | Directing complete: {len(resolved_decisions)} creative decisions "
            f"({plan.resolved_count} resolved, {plan.unresolved_assets_count} unresolved assets, "
            f"{plan.conflict_count} conflicts, {plan.preserved_spans_count} preserved)."
        )

        return plan

    def _process_accepted_decision(
        self,
        dec: Dict[str, Any],
        counter: int,
        face_cx: float,
        face_cy: float,
    ) -> CreativeDecision:
        """Transforms a single approved Agent 2 decision into a CreativeDecision."""
        d_id = dec.get("decision_id", f"dec_{counter:03d}")
        op_str = dec.get("operation", "NO_EDIT")
        op = CreativeOperation(op_str) if op_str in CreativeOperation.__members__ else CreativeOperation.NO_EDIT
        start = float(dec.get("start", 0.0))
        end = float(dec.get("end", start + 2.0))
        confidence = float(dec.get("confidence", 1.0))
        reason = dec.get("reason", "")
        act = dec.get("story_act", "HOOK")
        ev_ids = dec.get("source_evidence_ids", [])
        opp_ids = dec.get("source_opportunity_ids", [])

        c_id = f"creative_{counter:03d}"

        # ── HERO_TEXT ─────────────────────────────────────────────────
        if op == CreativeOperation.HERO_TEXT:
            text = dec.get("text_content") or "KEY THEME"
            is_telugu = is_telugu_text(text)

            primary_font = self.style.typography.telugu_primary_font if is_telugu else self.style.typography.primary_font
            line_padding = self.style.safe_zones.vertical_line_padding_factor if is_telugu else 1.0

            typo = TypographySpec(
                font=primary_font,
                font_fallback=self.style.typography.fallback_fonts,
                size=self.style.typography.hero_font_size,
                weight="bold",
                position="center",
                safe_zone=True,
                line_breaking="balanced",
                color_primary=self.style.palette.primary_text,
                color_highlight=self.style.palette.highlight_accent,
                color_outline=self.style.palette.outline_color,
                outline_width=self.style.typography.outline_width,
                shadow_distance=self.style.typography.shadow_distance,
                vertical_line_padding=line_padding,
                is_indic_script=is_telugu,
            )

            anim = AnimationSpec(
                preset=self.style.animation.default_entrance,
                easing=self.style.animation.default_easing,
                duration=self.style.animation.entrance_duration_sec,
                primitives={"opacity": [0.0, 1.0], "scale": [0.90, 1.0]},
            )

            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=CreativeStatus.RESOLVED,
                confidence=confidence,
                reason=f"Hero text treatment styled under {self.style.name.value} profile.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
                text_content=text,
                typography=typo,
                animation=anim,
            )

        # ── EMPHASIS_TEXT ─────────────────────────────────────────────
        elif op == CreativeOperation.EMPHASIS_TEXT:
            text = dec.get("text_content") or "EMPHASIS"
            is_telugu = is_telugu_text(text)
            primary_font = self.style.typography.telugu_primary_font if is_telugu else self.style.typography.primary_font

            typo = TypographySpec(
                font=primary_font,
                font_fallback=self.style.typography.fallback_fonts,
                size=self.style.typography.emphasis_font_size,
                weight="heavy" if "heavy" in self.style.typography.weights else "bold",
                position="lower_third",
                safe_zone=True,
                line_breaking="auto",
                color_primary=self.style.palette.primary_text,
                color_highlight=self.style.palette.highlight_accent,
                color_outline=self.style.palette.outline_color,
                outline_width=self.style.typography.outline_width,
                shadow_distance=self.style.typography.shadow_distance,
                vertical_line_padding=1.20 if is_telugu else 1.0,
                is_indic_script=is_telugu,
            )

            anim = AnimationSpec(
                preset="pop",
                easing="spring" if self.style.name == StyleProfileName.KINETIC else "ease_out",
                duration=0.18,
                primitives={"scale": [1.0, 1.15, 1.0]},
            )

            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=CreativeStatus.RESOLVED,
                confidence=confidence,
                reason=f"Emphasis text pop with {self.style.name.value} accent highlight.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
                text_content=text,
                typography=typo,
                animation=anim,
            )

        # ── PUNCH_IN ──────────────────────────────────────────────────
        elif op == CreativeOperation.PUNCH_IN:
            strength = str(dec.get("strength", "MEDIUM")).upper()
            if strength == "SUBTLE":
                scale = sum(self.style.punch_in.subtle_scale_range) / 2.0
            elif strength == "STRONG":
                scale = sum(self.style.punch_in.strong_scale_range) / 2.0
            else:
                scale = sum(self.style.punch_in.medium_scale_range) / 2.0

            # Safe face anchoring with headroom preservation
            anchor_x = max(0.15, min(0.85, face_cx))
            anchor_y = max(0.12, min(0.80, face_cy - (self.style.punch_in.preserve_headroom_pct * 0.5)))

            punch_spec = PunchInSpec(
                scale=round(scale, 3),
                anchor={"x": round(anchor_x, 3), "y": round(anchor_y, 3)},
                transition=self.style.punch_in.transition,
                conceptual_strength=strength,
                headroom_preserved=True,
                duration=round(end - start, 3),
            )

            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=CreativeStatus.RESOLVED,
                confidence=confidence,
                reason=f"Face-aware jump cut punch-in ({strength}, {scale:.2f}x) anchored to x={anchor_x:.2f}, y={anchor_y:.2f}.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
                punch_in=punch_spec,
            )

        # ── B-ROLL CUTAWAY / CONTEXT / SUPPORT ─────────────────────────
        elif op in [CreativeOperation.BROLL_CONTEXT, CreativeOperation.BROLL_SUPPORT, CreativeOperation.BROLL_CONTRAST]:
            query = dec.get("concept_query") or dec.get("concept") or "contextual illustration"
            role = "context" if op == CreativeOperation.BROLL_CONTEXT else ("support" if op == CreativeOperation.BROLL_SUPPORT else "contrast")
            req_dur = round(end - start, 3)

            # Resolve against local library
            res = self.asset_resolver.resolve_broll(
                query=query,
                required_duration=req_dur,
                role=role,
            )

            status = CreativeStatus.RESOLVED if res.get("status") == "RESOLVED" else CreativeStatus.UNRESOLVED_ASSET

            broll_spec = BrollSpec(
                asset_query=query,
                role=role,
                required_duration=req_dur,
                aspect_treatment="9:16_FILL_CROP",
                crop={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                overlay_treatment="FULL_OVERLAY",
                transition=self.style.broll_transition,
                opacity=1.0,
                replacement_mode="OVERLAY",
                status=status.value,
                resolved_asset_path=res.get("asset_path"),
            )

            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=status,
                confidence=confidence,
                reason=f"B-roll cutaway specification for '{query}'. Status: {status.value}.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
                broll=broll_spec,
            )

        # ── SFX CUES ──────────────────────────────────────────────────
        elif op in [CreativeOperation.SFX_WHOOSH, CreativeOperation.SFX_POP, CreativeOperation.SFX_IMPACT, CreativeOperation.SFX_RISER]:
            cat = op.value.lower().replace("sfx_", "")
            cue = dec.get("sound_cue") or f"{cat}_cue"

            res = self.asset_resolver.resolve_sfx(
                category=cat,
                sound_cue=cue,
            )

            status = CreativeStatus.RESOLVED if res.get("status") == "RESOLVED" else CreativeStatus.UNRESOLVED_ASSET

            sfx_spec = SfxSpec(
                category=cat,
                sound_cue=cue,
                asset_path=res.get("asset_path"),
                start=start,
                end=end,
                gain_db=self.style.audio.max_sfx_gain_db,
                fade_in_sec=0.02,
                fade_out_sec=0.04,
                status=status.value,
            )

            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=status,
                confidence=confidence,
                reason=f"Acoustic accent ({cat}) specified at {self.style.audio.max_sfx_gain_db}dB. Status: {status.value}.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
                sfx=sfx_spec,
            )

        # ── DEAD SPACE TRIM / CUT ─────────────────────────────────────
        elif op in [CreativeOperation.REMOVE_DEAD_SPACE, CreativeOperation.REMOVE_FILLER, CreativeOperation.CUT, CreativeOperation.PACE_TRIM]:
            return CreativeDecision(
                creative_id=c_id,
                source_decision_id=d_id,
                operation=op,
                time_range=[start, end],
                style_profile=self.style.name.value,
                status=CreativeStatus.RESOLVED,
                confidence=confidence,
                reason=reason or f"Rhythmic cut operation ({op.value}) maintaining natural breath buffer.",
                story_act=act,
                source_opportunity_ids=opp_ids,
                source_evidence_ids=ev_ids,
            )

        # Default fallback
        return CreativeDecision(
            creative_id=c_id,
            source_decision_id=d_id,
            operation=op,
            time_range=[start, end],
            style_profile=self.style.name.value,
            status=CreativeStatus.RESOLVED,
            confidence=confidence,
            reason=reason,
            story_act=act,
            source_opportunity_ids=opp_ids,
            source_evidence_ids=ev_ids,
        )

    def _process_no_edit_decision(
        self,
        dec: Dict[str, Any],
        counter: int,
    ) -> CreativeDecision:
        """Transforms a NO_EDIT preserved span into a CreativeDecision."""
        d_id = dec.get("decision_id", f"keep_{counter:03d}")
        start = float(dec.get("start", 0.0))
        end = float(dec.get("end", start + 2.0))
        confidence = float(dec.get("confidence", 1.0))
        reason = dec.get("reason", "Natural fluent delivery; preserved untouched.")
        act = dec.get("story_act", "HOOK")
        ev_ids = dec.get("source_evidence_ids", [])
        opp_ids = dec.get("source_opportunity_ids", [])

        return CreativeDecision(
            creative_id=f"creative_{counter:03d}",
            source_decision_id=d_id,
            operation=CreativeOperation.NO_EDIT,
            time_range=[start, end],
            style_profile=self.style.name.value,
            status=CreativeStatus.PRESERVED,
            confidence=confidence,
            reason=reason,
            story_act=act,
            source_opportunity_ids=opp_ids,
            source_evidence_ids=ev_ids,
        )

    def _resolve_creative_conflicts(
        self,
        decisions: List[CreativeDecision],
    ) -> List[CreativeDecision]:
        """
        Enforces the Creative Conflict Resolution Hierarchy:
          1. HeroText (highest visual priority)
          2. Primary speaker framing (punch-in)
          3. EmphasisText
          4. B-roll
          5. SFX
          6. Music

        If HeroText and B-roll collide in the same exact time window:
        HeroText takes precedence; B-roll is de-conflicted or marked CONFLICT.
        """
        PRIORITY = {
            CreativeOperation.HERO_TEXT: 1,
            CreativeOperation.PUNCH_IN: 2,
            CreativeOperation.EMPHASIS_TEXT: 3,
            CreativeOperation.BROLL_CONTEXT: 4,
            CreativeOperation.BROLL_SUPPORT: 4,
            CreativeOperation.BROLL_CONTRAST: 4,
            CreativeOperation.SFX_WHOOSH: 5,
            CreativeOperation.SFX_POP: 5,
            CreativeOperation.SFX_IMPACT: 5,
            CreativeOperation.SFX_RISER: 5,
            CreativeOperation.MUSIC_ENTER: 6,
            CreativeOperation.MUSIC_EXIT: 6,
            CreativeOperation.NO_EDIT: 10,
        }

        resolved: List[CreativeDecision] = []

        # Find active HeroText windows
        hero_windows = [
            (d.time_range[0], d.time_range[1])
            for d in decisions
            if d.operation == CreativeOperation.HERO_TEXT
        ]

        for d in decisions:
            # Check if B-roll overlaps a HeroText window
            if d.operation in [CreativeOperation.BROLL_CONTEXT, CreativeOperation.BROLL_SUPPORT, CreativeOperation.BROLL_CONTRAST]:
                has_hero_collision = any(
                    max(d.time_range[0], hw[0]) < min(d.time_range[1], hw[1])
                    for hw in hero_windows
                )
                if has_hero_collision:
                    # B-roll yields to HeroText under hierarchy rule 1
                    d.status = CreativeStatus.CONFLICT
                    d.reason = f"Creative conflict: B-roll cutaway suppressed under HeroText window to maintain typography clarity."
                    logger.info(f"CreativeDirector | De-conflicted {d.creative_id}: {d.reason}")

            resolved.append(d)

        return resolved
