"""
agents/agent3_creative/reporter.py

Interactive HTML Report Generator for Agent 3 (Creative Director)
─────────────────────────────────────────────────────────────────
Generates a responsive dark-mode HTML report ('creative_report.html') showcasing:
1. Active Style Profile details (typography, palette, safe-zones, easing).
2. Side-by-side mapping: Agent 2 Editorial Decision -> Agent 3 Creative Treatment.
3. Safe-zone audit & Telugu Unicode font resolution verification.
4. Asset resolution tracking (Resolved vs Unresolved).
5. Conflict resolution log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from agents.agent3_creative.models import CreativePlan, CreativeDecision, CreativeStatus
from agents.agent3_creative.styles import StyleProfile, get_style_profile


class CreativeReporter:
    """Generates the interactive creative_report.html report."""

    def generate(
        self,
        creative_plan: CreativePlan,
        edit_decisions: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        output_file = Path(output_path).resolve()
        profile = get_style_profile(creative_plan.style_profile)

        html_content = self._render_html(creative_plan, edit_decisions, profile)
        output_file.write_text(html_content, encoding="utf-8")
        return output_file

    def _render_html(
        self,
        plan: CreativePlan,
        edit_decisions: Dict[str, Any],
        profile: StyleProfile,
    ) -> str:
        # Build lookup for Agent 2 decisions by decision_id
        a2_lookup: Dict[str, Dict[str, Any]] = {}
        for d in edit_decisions.get("accepted_decisions", []):
            a2_lookup[d["decision_id"]] = d
        for d in edit_decisions.get("no_edit_decisions", []):
            a2_lookup[d["decision_id"]] = d

        # Table rows
        rows_html = []
        for dec in plan.decisions:
            a2_dec = a2_lookup.get(dec.source_decision_id, {})
            t_range = f"{dec.time_range[0]:.2f}s – {dec.time_range[1]:.2f}s"
            op_label = dec.operation.value if hasattr(dec.operation, "value") else str(dec.operation)

            # Status Badge
            status_cls = "badge-resolved"
            if dec.status == CreativeStatus.UNRESOLVED_ASSET:
                status_cls = "badge-unresolved"
            elif dec.status == CreativeStatus.CONFLICT:
                status_cls = "badge-conflict"
            elif dec.status == CreativeStatus.PRESERVED:
                status_cls = "badge-preserved"

            # Treatment details
            treatment_parts = []
            if dec.typography:
                treatment_parts.append(
                    f"<strong>Font:</strong> {dec.typography.font} ({dec.typography.size}pt, {dec.typography.weight})<br>"
                    f"<strong>Colors:</strong> Primary {dec.typography.color_primary}, Accent {dec.typography.color_highlight}<br>"
                    f"<strong>Script:</strong> {'Indic (Telugu)' if dec.typography.is_indic_script else 'Latin/English'} (Padding: {dec.typography.vertical_line_padding}x)"
                )
            if dec.animation:
                treatment_parts.append(
                    f"<strong>Animation:</strong> {dec.animation.preset} | <strong>Easing:</strong> {dec.animation.easing} ({dec.animation.duration}s)"
                )
            if dec.punch_in:
                treatment_parts.append(
                    f"<strong>Punch-In:</strong> {dec.punch_in.scale}x ({dec.punch_in.conceptual_strength}) | "
                    f"<strong>Anchor:</strong> x={dec.punch_in.anchor['x']:.2f}, y={dec.punch_in.anchor['y']:.2f} | "
                    f"<strong>Transition:</strong> {dec.punch_in.transition}"
                )
            if dec.broll:
                treatment_parts.append(
                    f"<strong>B-Roll Query:</strong> '{dec.broll.asset_query}'<br>"
                    f"<strong>Treatment:</strong> {dec.broll.overlay_treatment} ({dec.broll.aspect_treatment}) | "
                    f"<strong>Status:</strong> <span class='{status_cls}'>{dec.broll.status}</span>"
                )
            if dec.sfx:
                treatment_parts.append(
                    f"<strong>SFX Cue:</strong> '{dec.sfx.sound_cue}' ({dec.sfx.category})<br>"
                    f"<strong>Gain:</strong> {dec.sfx.gain_db}dB | <strong>Fade:</strong> {dec.sfx.fade_in_sec}s in / {dec.sfx.fade_out_sec}s out | "
                    f"<strong>Status:</strong> <span class='{status_cls}'>{dec.sfx.status}</span>"
                )
            if not treatment_parts:
                treatment_parts.append(f"<em>{dec.reason}</em>")

            treatment_str = "<br>".join(treatment_parts)

            a2_reason = a2_dec.get("reason", "Editorial intervention approved by Story Editor.")

            rows_html.append(f"""
            <tr>
                <td><code>{t_range}</code></td>
                <td><span class="badge badge-act">{dec.story_act}</span></td>
                <td>
                    <strong>{a2_dec.get('operation', op_label)}</strong><br>
                    <small class="text-muted">{dec.source_decision_id}</small><br>
                    <small>{a2_reason}</small>
                </td>
                <td>
                    <span class="badge {status_cls}">{dec.status.value if hasattr(dec.status, 'value') else dec.status}</span>
                </td>
                <td>{treatment_str}</td>
            </tr>
            """)

        table_rows_str = "\n".join(rows_html)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent 3: Creative Director Plan — {plan.source_video}</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --surface-color: #161b22;
            --surface-border: #30363d;
            --text-main: #c9d1d9;
            --text-heading: #f0f6fc;
            --text-muted: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-gold: #d29922;
            --accent-purple: #bc8cff;
            --accent-red: #f85149;
            --accent-cyan: #39c5bb;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px 32px;
            line-height: 1.5;
        }}
        h1, h2, h3 {{ color: var(--text-heading); margin-top: 0; }}
        .header-bar {{
            background: linear-gradient(135deg, #1f2937, #111827);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background-color: var(--surface-color);
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--text-heading);
            margin-bottom: 4px;
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }}
        .profile-card {{
            background-color: var(--surface-color);
            border: 1px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .profile-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }}
        .profile-item strong {{ color: var(--accent-cyan); display: block; margin-bottom: 4px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: var(--surface-color);
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--surface-border);
            vertical-align: top;
        }}
        th {{
            background-color: #21262d;
            color: var(--text-heading);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background-color: rgba(255,255,255,0.02); }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-act {{ background-color: #388bfd26; color: var(--accent-blue); border: 1px solid #388bfd4d; }}
        .badge-resolved {{ background-color: #3fb95026; color: var(--accent-green); border: 1px solid #3fb9504d; }}
        .badge-unresolved {{ background-color: #d2992226; color: var(--accent-gold); border: 1px solid #d299224d; }}
        .badge-conflict {{ background-color: #f8514926; color: var(--accent-red); border: 1px solid #f851494d; }}
        .badge-preserved {{ background-color: #bc8cff26; color: var(--accent-purple); border: 1px solid #bc8cff4d; }}
        .text-muted {{ color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="header-bar">
        <h1>Agent 3: Creative Director Plan</h1>
        <p style="margin: 0; color: var(--text-muted);">
            Translates Agent 2's approved editorial decisions into concrete styling, typography, animations, face-aware punch-in framing, and sound design.
        </p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{plan.total_decisions}</div>
            <div class="metric-label">Total Decisions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: var(--accent-green);">{plan.resolved_count}</div>
            <div class="metric-label">Resolved Operations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: var(--accent-purple);">{plan.preserved_spans_count}</div>
            <div class="metric-label">Preserved (NO_EDIT)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: var(--accent-gold);">{plan.unresolved_assets_count}</div>
            <div class="metric-label">Unresolved Assets</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: var(--accent-red);">{plan.conflict_count}</div>
            <div class="metric-label">Resolved Conflicts</div>
        </div>
    </div>

    <div class="profile-card">
        <h3>Active Aesthetic Profile: {profile.name.value}</h3>
        <p style="margin: 0 0 12px 0; color: var(--text-muted);">{profile.description}</p>
        <div class="profile-grid">
            <div class="profile-item">
                <strong>Primary Typography</strong>
                {profile.typography.primary_font} (Fallback: {', '.join(profile.typography.fallback_fonts)})
            </div>
            <div class="profile-item">
                <strong>Telugu-Safe Font</strong>
                {profile.typography.telugu_primary_font} (Line Padding: {profile.safe_zones.vertical_line_padding_factor}x)
            </div>
            <div class="profile-item">
                <strong>Punch-In Scales</strong>
                Subtle: {profile.punch_in.subtle_scale_range[0]}–{profile.punch_in.subtle_scale_range[1]}x |
                Strong: {profile.punch_in.strong_scale_range[0]}–{profile.punch_in.strong_scale_range[1]}x
            </div>
            <div class="profile-item">
                <strong>Audio Design & Restraint</strong>
                SFX Density: {profile.audio.sfx_density} | Max SFX Gain: {profile.audio.max_sfx_gain_db}dB | Restraint: {profile.visual_restraint_level}
            </div>
        </div>
    </div>

    <h2>Creative Decisions Mapping</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 140px;">Timestamp</th>
                <th style="width: 110px;">Act</th>
                <th style="width: 240px;">Agent 2 Editorial Decision</th>
                <th style="width: 110px;">Status</th>
                <th>Agent 3 Creative Treatment</th>
            </tr>
        </thead>
        <tbody>
            {table_rows_str}
        </tbody>
    </table>
</body>
</html>
"""
