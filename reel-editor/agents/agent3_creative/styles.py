"""
agents/agent3_creative/styles.py

Style Profile Abstraction for Agent 3 (Creative Director)
──────────────────────────────────────────────────────────
Defines how approved editorial decisions look and sound across 6 distinct aesthetic profiles:
1. EDITORIAL_CINEMATIC: Restrained, elegant serif/modern sans, subtle zoom (1.10x–1.18x), champagne/gold accents, high restraint.
2. BOLD_SOCIAL: High-energy bold typography, vibrant yellow/cyan accents, snappy overshoot/pop, 1.20x–1.30x punch-ins, active sound design.
3. MINIMAL: Clean modern sans, monochrome palette, subtle opacity fades, minimal punch-in (1.08x–1.14x), zero gratuitous motion.
4. CLEAN_EDUCATIONAL: Structured readable hierarchy, card overlays, balanced safe-zone placement, clarity first.
5. KINETIC: Dynamic rhythmic typography, spring/overshoot easing, rhythmic audio accents.
6. TELUGU_SAFE: Primary Indic typography anchored to 'Nirmala UI' with Gautami/Noto fallbacks, expanded vertical line heights (20% extra line-padding) to avoid diacritic clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional


class StyleProfileName(str, Enum):
    EDITORIAL_CINEMATIC = "EDITORIAL_CINEMATIC"
    BOLD_SOCIAL = "BOLD_SOCIAL"
    MINIMAL = "MINIMAL"
    CLEAN_EDUCATIONAL = "CLEAN_EDUCATIONAL"
    KINETIC = "KINETIC"
    TELUGU_SAFE = "TELUGU_SAFE"


@dataclass
class SafeZoneSpec:
    top_margin_pct: float = 0.12        # Safe from status bars / top notch
    bottom_margin_pct: float = 0.20     # Safe from captions, player scrubber, UI icons
    left_margin_pct: float = 0.08       # Horizontal safe margin
    right_margin_pct: float = 0.08
    vertical_line_padding_factor: float = 1.0  # Multiplier for line height (1.20 for Telugu)


@dataclass
class ColorPalette:
    primary_text: str = "&H00FFFFFF&"        # ASS Hex (&HAABBGGRR&): Pure White
    highlight_accent: str = "&H0000D7FF&"    # ASS Hex: Vibrant Gold/Yellow
    secondary_accent: str = "&H00FFFF00&"    # ASS Hex: Cyan/Aqua
    background_overlay: str = "&H80000000&"  # ASS Hex: 50% Semi-transparent Black
    outline_color: str = "&H00000000&"       # ASS Hex: Pure Black
    shadow_color: str = "&H80000000&"        # ASS Hex: Shadow Black


@dataclass
class TypographyFamily:
    primary_font: str = "Arial Black"
    fallback_fonts: list[str] = field(default_factory=lambda: ["Nirmala UI", "Gautami", "Noto Sans Telugu", "Arial"])
    telugu_primary_font: str = "Nirmala UI"
    weights: list[str] = field(default_factory=lambda: ["bold", "heavy"])
    hero_font_size: int = 86
    emphasis_font_size: int = 76
    caption_font_size: int = 70
    outline_width: int = 5
    shadow_distance: int = 2


@dataclass
class AnimationLanguage:
    default_entrance: str = "pop"            # "pop", "fade", "overshoot", "spring"
    default_easing: str = "ease_out"         # "linear", "ease_in", "ease_out", "ease_in_out", "spring", "pop", "overshoot", "back"
    entrance_duration_sec: float = 0.28
    exit_duration_sec: float = 0.20
    text_motion_allowed: bool = True
    camera_motion_allowed: bool = False      # Agent 3 avoids gratuitous camera pan/scan


@dataclass
class PunchInBehavior:
    subtle_scale_range: tuple[float, float] = (1.10, 1.14)   # Conceptual SUBTLE
    medium_scale_range: tuple[float, float] = (1.15, 1.22)   # Conceptual MEDIUM
    strong_scale_range: tuple[float, float] = (1.25, 1.35)   # Conceptual STRONG
    transition: str = "JUMP_CUT"                             # Professional editor jump cut
    preserve_headroom_pct: float = 0.15                      # Keep face upper headroom intact
    min_face_margin_pct: float = 0.10                        # Never crop past 10% from face center


@dataclass
class AudioRestraintSpec:
    sfx_density: str = "RESTRICTED"       # "MINIMAL", "RESTRICTED", "BALANCED", "ACTIVE"
    max_sfx_gain_db: float = -14.0        # Voice priority: SFX never exceeds speech
    music_density: str = "OPTIONAL"       # "NONE", "OPTIONAL", "PERSISTENT"
    default_ducking_db: float = -18.0     # Music attenuation under active voice
    voice_priority_level: float = 1.0     # Voice is authoritative anchor


@dataclass
class StyleProfile:
    name: StyleProfileName
    description: str
    typography: TypographyFamily
    safe_zones: SafeZoneSpec
    palette: ColorPalette
    animation: AnimationLanguage
    punch_in: PunchInBehavior
    audio: AudioRestraintSpec
    broll_transition: str = "CROSSFADE"   # "CROSSFADE" or "HARD_CUT"
    visual_restraint_level: str = "HIGH"  # "HIGH", "BALANCED", "LOW"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name.value if isinstance(self.name, StyleProfileName) else self.name
        return d


# ── Profile Definitions ──────────────────────────────────────────────────────────

_PROFILES: Dict[StyleProfileName, StyleProfile] = {
    StyleProfileName.EDITORIAL_CINEMATIC: StyleProfile(
        name=StyleProfileName.EDITORIAL_CINEMATIC,
        description="Elegant, restrained editorial visual language with subtle zoom accents and refined gold highlights.",
        typography=TypographyFamily(
            primary_font="Montserrat",
            fallback_fonts=["Nirmala UI", "Gautami", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["bold", "medium"],
            hero_font_size=84,
            emphasis_font_size=74,
            caption_font_size=68,
            outline_width=4,
            shadow_distance=3,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.14,
            bottom_margin_pct=0.22,
            left_margin_pct=0.10,
            right_margin_pct=0.10,
            vertical_line_padding_factor=1.05,
        ),
        palette=ColorPalette(
            primary_text="&H00F5F5F5&",       # Soft White
            highlight_accent="&H007BD4F6&",   # Champagne Gold
            secondary_accent="&H00D0B48A&",   # Warm Bronze
            background_overlay="&H990F0F0F&",
            outline_color="&H00000000&",
            shadow_color="&HA0000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="pop",
            default_easing="ease_out",
            entrance_duration_sec=0.25,
            exit_duration_sec=0.20,
            text_motion_allowed=True,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.10, 1.14),
            medium_scale_range=(1.16, 1.20),
            strong_scale_range=(1.24, 1.30),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.16,
            min_face_margin_pct=0.12,
        ),
        audio=AudioRestraintSpec(
            sfx_density="RESTRICTED",
            max_sfx_gain_db=-15.0,
            music_density="OPTIONAL",
            default_ducking_db=-20.0,
        ),
        broll_transition="CROSSFADE",
        visual_restraint_level="HIGH",
    ),

    StyleProfileName.BOLD_SOCIAL: StyleProfile(
        name=StyleProfileName.BOLD_SOCIAL,
        description="High-energy vertical reel style with prominent punchy typography, vibrant yellow accents, and snappy pops.",
        typography=TypographyFamily(
            primary_font="Arial Black",
            fallback_fonts=["Nirmala UI", "Gautami", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["heavy", "black"],
            hero_font_size=92,
            emphasis_font_size=82,
            caption_font_size=74,
            outline_width=6,
            shadow_distance=3,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.12,
            bottom_margin_pct=0.20,
            left_margin_pct=0.08,
            right_margin_pct=0.08,
            vertical_line_padding_factor=1.0,
        ),
        palette=ColorPalette(
            primary_text="&H00FFFFFF&",       # Pure White
            highlight_accent="&H0000D7FF&",   # Electric Gold / Yellow
            secondary_accent="&H00FFFF00&",   # Cyan
            background_overlay="&HB0000000&",
            outline_color="&H00000000&",
            shadow_color="&HC0000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="overshoot",
            default_easing="overshoot",
            entrance_duration_sec=0.22,
            exit_duration_sec=0.18,
            text_motion_allowed=True,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.12, 1.16),
            medium_scale_range=(1.20, 1.25),
            strong_scale_range=(1.28, 1.35),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.14,
            min_face_margin_pct=0.10,
        ),
        audio=AudioRestraintSpec(
            sfx_density="ACTIVE",
            max_sfx_gain_db=-12.0,
            music_density="PERSISTENT",
            default_ducking_db=-16.0,
        ),
        broll_transition="HARD_CUT",
        visual_restraint_level="BALANCED",
    ),

    StyleProfileName.MINIMAL: StyleProfile(
        name=StyleProfileName.MINIMAL,
        description="Clean, distraction-free modern typography with understated monochrome highlights and minimal punch scale.",
        typography=TypographyFamily(
            primary_font="Segoe UI",
            fallback_fonts=["Nirmala UI", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["semibold", "bold"],
            hero_font_size=80,
            emphasis_font_size=70,
            caption_font_size=64,
            outline_width=3,
            shadow_distance=1,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.15,
            bottom_margin_pct=0.22,
            left_margin_pct=0.10,
            right_margin_pct=0.10,
            vertical_line_padding_factor=1.0,
        ),
        palette=ColorPalette(
            primary_text="&H00FFFFFF&",       # Pure White
            highlight_accent="&H00D8D8D8&",   # Crisp Platinum
            secondary_accent="&H00A0A0A0&",   # Muted Silver
            background_overlay="&H80000000&",
            outline_color="&H00111111&",
            shadow_color="&H60000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="fade",
            default_easing="ease_out",
            entrance_duration_sec=0.20,
            exit_duration_sec=0.18,
            text_motion_allowed=False,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.08, 1.12),
            medium_scale_range=(1.12, 1.16),
            strong_scale_range=(1.18, 1.24),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.18,
            min_face_margin_pct=0.12,
        ),
        audio=AudioRestraintSpec(
            sfx_density="MINIMAL",
            max_sfx_gain_db=-18.0,
            music_density="NONE",
            default_ducking_db=-24.0,
        ),
        broll_transition="CROSSFADE",
        visual_restraint_level="HIGH",
    ),

    StyleProfileName.CLEAN_EDUCATIONAL: StyleProfile(
        name=StyleProfileName.CLEAN_EDUCATIONAL,
        description="Structured educational style with high clarity, prominent headline cards, and calm visual pacing.",
        typography=TypographyFamily(
            primary_font="Segoe UI",
            fallback_fonts=["Nirmala UI", "Gautami", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["bold"],
            hero_font_size=84,
            emphasis_font_size=74,
            caption_font_size=68,
            outline_width=4,
            shadow_distance=2,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.14,
            bottom_margin_pct=0.22,
            left_margin_pct=0.08,
            right_margin_pct=0.08,
            vertical_line_padding_factor=1.1,
        ),
        palette=ColorPalette(
            primary_text="&H00FFFFFF&",       # White
            highlight_accent="&H0032CD32&",   # Lime Green / Mint accent
            secondary_accent="&H00FFFF00&",   # Sky Cyan
            background_overlay="&HA0000000&",
            outline_color="&H00000000&",
            shadow_color="&H90000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="pop",
            default_easing="ease_out",
            entrance_duration_sec=0.25,
            exit_duration_sec=0.20,
            text_motion_allowed=True,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.10, 1.15),
            medium_scale_range=(1.16, 1.22),
            strong_scale_range=(1.22, 1.28),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.15,
            min_face_margin_pct=0.10,
        ),
        audio=AudioRestraintSpec(
            sfx_density="RESTRICTED",
            max_sfx_gain_db=-15.0,
            music_density="OPTIONAL",
            default_ducking_db=-18.0,
        ),
        broll_transition="CROSSFADE",
        visual_restraint_level="HIGH",
    ),

    StyleProfileName.KINETIC: StyleProfile(
        name=StyleProfileName.KINETIC,
        description="Dynamic kinetic reel treatment with energetic spring easing, rapid word bursts, and rhythmic accents.",
        typography=TypographyFamily(
            primary_font="Arial Black",
            fallback_fonts=["Nirmala UI", "Gautami", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["black"],
            hero_font_size=96,
            emphasis_font_size=84,
            caption_font_size=76,
            outline_width=6,
            shadow_distance=4,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.12,
            bottom_margin_pct=0.20,
            left_margin_pct=0.08,
            right_margin_pct=0.08,
            vertical_line_padding_factor=1.0,
        ),
        palette=ColorPalette(
            primary_text="&H00FFFFFF&",       # White
            highlight_accent="&H0014F0FF&",   # Neon Yellow
            secondary_accent="&H00FF00FF&",   # Magenta
            background_overlay="&HB0000000&",
            outline_color="&H00000000&",
            shadow_color="&HB0000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="spring",
            default_easing="spring",
            entrance_duration_sec=0.22,
            exit_duration_sec=0.15,
            text_motion_allowed=True,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.12, 1.18),
            medium_scale_range=(1.18, 1.25),
            strong_scale_range=(1.26, 1.35),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.14,
            min_face_margin_pct=0.10,
        ),
        audio=AudioRestraintSpec(
            sfx_density="ACTIVE",
            max_sfx_gain_db=-12.0,
            music_density="PERSISTENT",
            default_ducking_db=-16.0,
        ),
        broll_transition="HARD_CUT",
        visual_restraint_level="BALANCED",
    ),

    StyleProfileName.TELUGU_SAFE: StyleProfile(
        name=StyleProfileName.TELUGU_SAFE,
        description="Indic-specialized typography profile anchored to Nirmala UI with 20% vertical line padding to avoid diacritic clipping.",
        typography=TypographyFamily(
            primary_font="Nirmala UI",
            fallback_fonts=["Gautami", "Noto Sans Telugu", "Vani", "Segoe UI", "Arial"],
            telugu_primary_font="Nirmala UI",
            weights=["bold"],
            hero_font_size=82,
            emphasis_font_size=74,
            caption_font_size=68,
            outline_width=5,
            shadow_distance=2,
        ),
        safe_zones=SafeZoneSpec(
            top_margin_pct=0.14,
            bottom_margin_pct=0.22,
            left_margin_pct=0.09,
            right_margin_pct=0.09,
            vertical_line_padding_factor=1.20,  # CRITICAL: 20% extra line-height for Telugu matras/vowels
        ),
        palette=ColorPalette(
            primary_text="&H00FFFFFF&",       # White
            highlight_accent="&H0000D7FF&",   # Gold
            secondary_accent="&H0000E5FF&",   # Amber
            background_overlay="&HA0000000&",
            outline_color="&H00000000&",
            shadow_color="&HA0000000&",
        ),
        animation=AnimationLanguage(
            default_entrance="pop",
            default_easing="ease_out",
            entrance_duration_sec=0.26,
            exit_duration_sec=0.20,
            text_motion_allowed=True,
            camera_motion_allowed=False,
        ),
        punch_in=PunchInBehavior(
            subtle_scale_range=(1.10, 1.15),
            medium_scale_range=(1.16, 1.22),
            strong_scale_range=(1.24, 1.30),
            transition="JUMP_CUT",
            preserve_headroom_pct=0.16,
            min_face_margin_pct=0.11,
        ),
        audio=AudioRestraintSpec(
            sfx_density="RESTRICTED",
            max_sfx_gain_db=-15.0,
            music_density="OPTIONAL",
            default_ducking_db=-20.0,
        ),
        broll_transition="CROSSFADE",
        visual_restraint_level="HIGH",
    ),
}


def get_style_profile(name_or_enum: StyleProfileName | str = StyleProfileName.EDITORIAL_CINEMATIC) -> StyleProfile:
    """
    Look up a StyleProfile by name or enum. Defaults to EDITORIAL_CINEMATIC.
    """
    if isinstance(name_or_enum, StyleProfileName):
        return _PROFILES.get(name_or_enum, _PROFILES[StyleProfileName.EDITORIAL_CINEMATIC])

    name_str = str(name_or_enum).strip().upper()
    for enum_key, profile in _PROFILES.items():
        if enum_key.value == name_str:
            return profile

    return _PROFILES[StyleProfileName.EDITORIAL_CINEMATIC]
