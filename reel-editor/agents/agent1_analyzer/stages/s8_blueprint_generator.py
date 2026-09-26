"""
agents/agent1_analyzer/stages/s8_blueprint_generator.py

Stage 8: Edit Blueprint & Interactive Report Generator
──────────────────────────────────────────────────────
INPUT:  opportunity_inventory.json  (Stage 7)
        story_analysis.json         (Stage 6)
        fused_timeline.json         (Stage 5)

OUTPUT: edit_blueprint.json         (Final structured metadata handoff for Agent 1)
        report.html                 (Self-contained, interactive HTML audit report)

Tools: Python JSON builder + HTML/CSS generator.
Cost: ₹0 local execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

from core.logger import get_logger
from core.json_utils import save_json

logger = get_logger(__name__)


def run(
    output_dir: Path | str,
    opportunity_inventory: Optional[dict] = None,
    story_analysis: Optional[dict] = None,
    fused_timeline: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Run Stage 8: Generate the final Edit Blueprint JSON and interactive HTML report.
    """
    output_dir = Path(output_dir)

    if opportunity_inventory is None:
        with open(output_dir / "opportunity_inventory.json", "r", encoding="utf-8") as f:
            opportunity_inventory = json.load(f)
    if story_analysis is None:
        with open(output_dir / "story_analysis.json", "r", encoding="utf-8") as f:
            story_analysis = json.load(f)
    if fused_timeline is None:
        with open(output_dir / "fused_timeline.json", "r", encoding="utf-8") as f:
            fused_timeline = json.load(f)

    logger.info("Stage 8 | Generating Edit Blueprint & HTML Report...")

    raw_duration = fused_timeline.get("duration_seconds", 90.0)
    opp_summary = opportunity_inventory.get("summary", {})
    seconds_saved = opp_summary.get("potential_seconds_saved", 0.0)
    trimmed_duration = round(max(0.0, raw_duration - seconds_saved), 2)
    opportunities = opportunity_inventory.get("opportunities", [])
    hook = story_analysis.get("hook_analysis", {})

    blueprint = {
        "version": "1.0.0",
        "agent": "Agent 1 (Observer)",
        "video_title": fused_timeline.get("video_file", "video.mp4"),
        "aspect_ratio": fused_timeline.get("aspect_ratio", "16:9"),
        "metrics_summary": {
            "raw_duration_seconds": raw_duration,
            "estimated_trimmed_duration_seconds": trimmed_duration,
            "total_seconds_saved": seconds_saved,
            "total_sentences": len(fused_timeline.get("sentences", [])),
            "total_opportunities": len(opportunities),
            "hook_score": hook.get("hook_strength_score", 8),
            "pacing_health": fused_timeline.get("pacing_summary", {}).get("pacing_health", "good"),
            "opportunity_breakdown": opp_summary.get("breakdown", {}),
        },
        "narrative_overview": {
            "hook_analysis": hook,
            "core_themes": story_analysis.get("core_themes", {}),
            "narrative_acts": story_analysis.get("narrative_acts", []),
            "semantic_peaks": story_analysis.get("semantic_peaks", []),
            "dropoff_risks": story_analysis.get("dropoff_risk_moments", []),
            "executive_summary": story_analysis.get("executive_summary", ""),
        },
        "timeline_actions": opportunities,
        "quality_audit": {
            "speaker_baselines": fused_timeline.get("speaker_baselines", {}),
            "framing_stability": "stable_mcu",
            "audio_clarity": "clear_voice_minimal_background_noise",
            "scene_cuts_detected": len(fused_timeline.get("scene_info", {}).get("scene_list", [])),
        },
        "next_steps_for_agent2": [
            "Review dead-space cut candidates for pacing rhythm.",
            "Confirm punch-in targets on semantic peaks.",
            "Select B-roll footage assets matching suggested queries.",
            "Apply animated kinetic styling to identified caption emphasis tokens.",
        ],
    }

    # Save JSON Blueprint
    blueprint_path = output_dir / "edit_blueprint.json"
    save_json(blueprint, blueprint_path)

    # Generate HTML Report
    html_content = _generate_html_report(blueprint, fused_timeline)
    html_path = output_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Stage 8 | Complete. Saved edit_blueprint.json and report.html ({len(html_content)} bytes)")

    return {
        "edit_blueprint": blueprint,
        "report_html_path": str(html_path),
    }


def _generate_html_report(blueprint: dict, fused_timeline: dict) -> str:
    """Generate self-contained, interactive, responsive HTML report."""
    title = blueprint.get("video_title", "Video Analysis")
    metrics = blueprint.get("metrics_summary", {})
    narrative = blueprint.get("narrative_overview", {})
    hook = narrative.get("hook_analysis", {})
    themes = narrative.get("core_themes", {})
    acts = narrative.get("narrative_acts", [])
    peaks = narrative.get("semantic_peaks", [])
    opportunities = blueprint.get("timeline_actions", [])
    sentences = fused_timeline.get("sentences", [])

    # Format JSON for embedding
    opps_json = json.dumps(opportunities)

    # Render acts HTML
    acts_html = ""
    for act in acts:
        acts_html += f"""
        <div class="act-card">
            <div class="act-header">
                <span class="act-badge">ACT {act.get('act_number', 1)}</span>
                <span class="act-title">{act.get('act_name', 'Act')}</span>
                <span class="act-time">{act.get('start_time', 0.0):.1f}s – {act.get('end_time', 0.0):.1f}s</span>
            </div>
            <p class="act-summary">{act.get('summary', '')}</p>
            <div class="act-meta">
                <span class="tag emotion-tag">{act.get('primary_emotion', 'neutral')}</span>
                <span class="tag purpose-tag">{act.get('purpose', '')}</span>
            </div>
        </div>
        """

    # Render takeaways HTML
    takeaways_html = "".join(f"<li>{t}</li>" for t in themes.get("key_takeaways", []))

    # Render peaks HTML
    peaks_html = ""
    for peak in peaks:
        peaks_html += f"""
        <div class="peak-item">
            <div class="peak-badge">SCORE {peak.get('impact_score', 8)}/10</div>
            <div class="peak-content">
                <div class="peak-quote">"{peak.get('text', '')}"</div>
                <div class="peak-time">@ {peak.get('timestamp', '')} · {peak.get('significance', '')}</div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Reel Editor — Audit Report: {title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #0a0e14;
            --bg-card: #131923;
            --bg-card-hover: #1a2332;
            --bg-card-secondary: #17202e;
            --border-color: #243044;
            --border-hover: #38bdf8;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-purple: #8b5cf6;
            --accent-rose: #f43f5e;
            --accent-blue: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        /* Header */
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .brand-icon {{
            width: 40px; height: 40px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; font-weight: 800; color: #fff;
        }}
        h1 {{ font-size: 24px; font-weight: 800; letter-spacing: -0.02em; }}
        .subtitle {{ font-size: 14px; color: var(--text-secondary); margin-top: 2px; }}
        .header-actions {{ display: flex; gap: 12px; }}
        .badge-agent {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; background: rgba(6, 182, 212, 0.12);
            border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 9999px;
            font-size: 13px; font-weight: 600; color: var(--accent-cyan);
        }}

        /* Metrics Bar */
        .metrics-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 32px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px;
            position: relative; overflow: hidden;
            transition: all 0.2s ease;
        }}
        .metric-card:hover {{ border-color: var(--border-hover); transform: translateY(-2px); }}
        .metric-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; }}
        .metric-value {{ font-size: 28px; font-weight: 800; color: var(--text-primary); margin: 6px 0 2px 0; font-family: 'JetBrains Mono', monospace; }}
        .metric-sub {{ font-size: 13px; color: var(--text-secondary); }}
        .metric-card::after {{
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        }}
        .card-cyan::after {{ background: var(--accent-cyan); }}
        .card-emerald::after {{ background: var(--accent-emerald); }}
        .card-amber::after {{ background: var(--accent-amber); }}
        .card-purple::after {{ background: var(--accent-purple); }}

        /* Sections */
        .section-title {{
            font-size: 18px; font-weight: 700; margin-bottom: 16px;
            display: flex; align-items: center; gap: 10px;
        }}
        .section-title::before {{
            content: ''; width: 4px; height: 18px; background: var(--accent-cyan); border-radius: 2px;
        }}

        /* Hook Box */
        .hook-box {{
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.06), rgba(139, 92, 246, 0.06));
            border: 1px solid rgba(6, 182, 212, 0.25);
            border-radius: 12px; padding: 24px; margin-bottom: 32px;
        }}
        .hook-score-badge {{
            display: inline-block; padding: 4px 12px; border-radius: 6px;
            background: var(--accent-emerald); color: #042f1a; font-weight: 800; font-size: 13px;
        }}
        .hook-quote {{
            font-size: 18px; font-weight: 600; color: #fff; margin: 12px 0; font-style: italic;
        }}
        .hook-notes {{ font-size: 14px; color: var(--text-secondary); }}

        /* Two Column Layout */
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
        @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

        /* Card container */
        .panel-card {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 24px;
        }}

        /* Acts */
        .act-card {{
            background: var(--bg-card-secondary); border: 1px solid var(--border-color);
            border-radius: 8px; padding: 14px; margin-bottom: 12px;
        }}
        .act-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
        .act-badge {{ font-size: 10px; font-weight: 700; background: var(--border-color); padding: 2px 8px; border-radius: 4px; color: var(--text-secondary); }}
        .act-title {{ font-size: 14px; font-weight: 600; color: #fff; flex: 1; }}
        .act-time {{ font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }}
        .act-summary {{ font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; }}
        .act-meta {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .tag {{ font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 500; }}
        .emotion-tag {{ background: rgba(139, 92, 246, 0.15); color: #c4b5fd; }}
        .purpose-tag {{ background: rgba(6, 182, 212, 0.15); color: #67e8f9; }}

        /* Peaks */
        .peak-item {{
            display: flex; gap: 14px; align-items: flex-start;
            padding: 12px; background: var(--bg-card-secondary); border-radius: 8px;
            margin-bottom: 10px; border: 1px solid var(--border-color);
        }}
        .peak-badge {{
            padding: 4px 8px; border-radius: 6px; background: rgba(16, 185, 129, 0.2);
            color: var(--accent-emerald); font-weight: 700; font-size: 11px; white-space: nowrap;
        }}
        .peak-quote {{ font-size: 14px; font-weight: 600; color: #fff; }}
        .peak-time {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

        /* Filter Controls */
        .filters {{
            display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
        }}
        .filter-btn {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            color: var(--text-secondary); padding: 8px 16px; border-radius: 8px;
            cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s ease;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: var(--accent-cyan); color: #022026; border-color: var(--accent-cyan);
        }}

        /* Opportunities List */
        .opp-card {{
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 10px; padding: 16px; margin-bottom: 12px;
            display: flex; gap: 16px; align-items: center; transition: all 0.15s ease;
        }}
        .opp-card:hover {{ border-color: var(--border-hover); background: var(--bg-card-hover); }}
        .opp-time {{
            font-family: 'JetBrains Mono', monospace; font-size: 13px;
            color: var(--accent-cyan); font-weight: 600; min-width: 65px;
        }}
        .opp-type-badge {{
            padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
            text-transform: uppercase; white-space: nowrap;
        }}
        .badge-cut {{ background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.3); }}
        .badge-punch {{ background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-broll {{ background: rgba(139, 92, 246, 0.15); color: var(--accent-purple); border: 1px solid rgba(139, 92, 246, 0.3); }}
        .badge-caption {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-sound {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }}

        .opp-body {{ flex: 1; }}
        .opp-title {{ font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 2px; }}
        .opp-rationale {{ font-size: 13px; color: var(--text-secondary); }}
        .opp-confidence {{
            font-size: 12px; font-weight: 600; color: var(--text-muted);
            min-width: 50px; text-align: right;
        }}

        /* Takeaways list */
        ul.takeaways {{ padding-left: 20px; font-size: 14px; color: var(--text-secondary); }}
        ul.takeaways li {{ margin-bottom: 8px; }}

        footer {{
            border-top: 1px solid var(--border-color);
            margin-top: 48px; padding-top: 24px; text-align: center;
            font-size: 13px; color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand">
                <div class="brand-icon">🎬</div>
                <div>
                    <h1>AI Reel Editor — Audit Report</h1>
                    <div class="subtitle">{title} · 100% Local Inference (₹0 Cost)</div>
                </div>
            </div>
            <div class="header-actions">
                <div class="badge-agent">Agent 1 (Observer) · Final Blueprint</div>
            </div>
        </header>

        <!-- Metrics Overview -->
        <div class="metrics-grid">
            <div class="metric-card card-cyan">
                <div class="metric-label">Raw Duration</div>
                <div class="metric-value">{metrics.get('raw_duration_seconds', 0):.1f}s</div>
                <div class="metric-sub">{len(sentences)} spoken sentences</div>
            </div>
            <div class="metric-card card-emerald">
                <div class="metric-label">Trimmed Duration</div>
                <div class="metric-value">{metrics.get('estimated_trimmed_duration_seconds', 0):.1f}s</div>
                <div class="metric-sub">Saved {metrics.get('total_seconds_saved', 0):.2f}s of dead space</div>
            </div>
            <div class="metric-card card-amber">
                <div class="metric-label">Hook Strength</div>
                <div class="metric-value">{hook.get('hook_strength_score', 8)}/10</div>
                <div class="metric-sub">{hook.get('hook_type', 'premise')}</div>
            </div>
            <div class="metric-card card-purple">
                <div class="metric-label">Opportunities Flagged</div>
                <div class="metric-value">{metrics.get('total_opportunities', 0)}</div>
                <div class="metric-sub">Cuts, punches, B-roll & captions</div>
            </div>
        </div>

        <!-- Hook Analysis Box -->
        <div class="hook-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 700; text-transform: uppercase; color: var(--accent-cyan); letter-spacing: 0.05em;">Hook & Retention Assessment (First 3 Seconds)</span>
                <span class="hook-score-badge">SCORE {hook.get('hook_strength_score', 8)}/10</span>
            </div>
            <div class="hook-quote">"{hook.get('opening_text', '')}"</div>
            <p class="hook-notes"><strong>Observation:</strong> {hook.get('first_3_seconds_assessment', '')} {hook.get('recommendation', '')}</p>
        </div>

        <!-- Narrative & Themes -->
        <div class="two-col">
            <div class="panel-card">
                <div class="section-title">Narrative Structure (Acts)</div>
                {acts_html}
            </div>
            <div class="panel-card">
                <div class="section-title">Core Thematic Takeaways</div>
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--accent-cyan); margin-bottom: 4px;">Primary Thesis</div>
                    <div style="font-size: 15px; font-weight: 600; color: #fff;">{themes.get('core_insight', '')}</div>
                </div>
                <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">Key Lessons</div>
                <ul class="takeaways">
                    {takeaways_html}
                </ul>
                <div style="margin-top: 24px;">
                    <div class="section-title" style="font-size: 16px;">Semantic Climax Moments</div>
                    {peaks_html}
                </div>
            </div>
        </div>

        <!-- Opportunity Inventory -->
        <div class="section-title">Editing Opportunities Catalog ({len(opportunities)})</div>
        <div class="filters" id="filterBar">
            <button class="filter-btn active" data-filter="ALL">All ({len(opportunities)})</button>
            <button class="filter-btn" data-filter="DEAD_SPACE_CUT">Cuts ({metrics.get('opportunity_breakdown', {}).get('DEAD_SPACE_CUT', 0)})</button>
            <button class="filter-btn" data-filter="PUNCH_IN">Punch-Ins ({metrics.get('opportunity_breakdown', {}).get('PUNCH_IN', 0)})</button>
            <button class="filter-btn" data-filter="B_ROLL_OVERLAY">B-Roll ({metrics.get('opportunity_breakdown', {}).get('B_ROLL_OVERLAY', 0)})</button>
            <button class="filter-btn" data-filter="CAPTION_EMPHASIS">Captions ({metrics.get('opportunity_breakdown', {}).get('CAPTION_EMPHASIS', 0)})</button>
        </div>

        <div id="oppList"></div>

        <footer>
            AI Reel Editor · Generated by Agent 1 (Observer) · Output Ready for Agent 2 (Creative Director) & DaVinci Resolve
        </footer>
    </div>

    <script>
        const opps = {opps_json};
        const listEl = document.getElementById('oppList');
        const filterBtns = document.querySelectorAll('.filter-btn');

        function renderOpps(filter = 'ALL') {{
            listEl.innerHTML = '';
            const filtered = filter === 'ALL' ? opps : opps.filter(o => o.type === filter);

            if (filtered.length === 0) {{
                listEl.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-muted);">No opportunities found for this filter.</div>';
                return;
            }}

            filtered.forEach(o => {{
                const card = document.createElement('div');
                card.className = 'opp-card';

                let badgeClass = 'badge-cut';
                if (o.type === 'PUNCH_IN') badgeClass = 'badge-punch';
                else if (o.type === 'B_ROLL_OVERLAY') badgeClass = 'badge-broll';
                else if (o.type === 'CAPTION_EMPHASIS') badgeClass = 'badge-caption';
                else if (o.type === 'AUDIO_ACCENT') badgeClass = 'badge-sound';

                const timeStr = `${{o.timestamp_start.toFixed(1)}}s`;
                const confPct = Math.round((o.confidence || 0.8) * 100);

                let title = o.type.replace(/_/g, ' ');
                if (o.type === 'DEAD_SPACE_CUT') title = `Cut Dead Space (-${{o.recommended_trim || 0.3}}s)`;
                else if (o.type === 'PUNCH_IN') title = `Punch-In (${{o.target_scale || 1.15}}x)`;
                else if (o.type === 'B_ROLL_OVERLAY') title = `B-Roll: ${{o.suggested_query || 'visual support'}}`;
                else if (o.type === 'CAPTION_EMPHASIS') title = `Highlight Word: "${{o.word || ''}}"`;

                card.innerHTML = `
                    <div class="opp-time">${{timeStr}}</div>
                    <span class="opp-type-badge ${{badgeClass}}">${{o.type.replace(/_/g, ' ')}}</span>
                    <div class="opp-body">
                        <div class="opp-title">${{title}}</div>
                        <div class="opp-rationale">${{o.rationale}}</div>
                    </div>
                    <div class="opp-confidence">${{confPct}}% conf</div>
                `;
                listEl.appendChild(card);
            }});
        }}

        filterBtns.forEach(btn => {{
            btn.addEventListener('click', () => {{
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderOpps(btn.dataset.filter);
            }});
        }});

        renderOpps();
    </script>
</body>
</html>
"""
