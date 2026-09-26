"""
core/review/reporter.py

HTML Quality Report Generator for AI Reel Editor (Phase 6)
──────────────────────────────────────────────────────────
Generates a publication-grade interactive HTML report comparing:
- Side-by-side Raw vs Edited frame captures
- Five-dimension quality scoring rubrics
- Edit density analysis and pacing classification
- In-depth evaluation across the 10 professional editorial dimensions
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)


class EditorialQualityReporter:
    """
    Generates human_quality_report.html with rich side-by-side comparisons,
    metric visualizations, and professional editorial commentary.
    """

    @staticmethod
    def _image_to_base64(img_path: Path | str) -> str:
        p = Path(img_path)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                return f"data:{mime};base64,{data}"
            except Exception as e:
                logger.warning(f"Failed to encode image {p}: {e}")
        return ""

    def generate_html_report(
        self,
        report_data: Dict[str, Any],
        output_file: Path | str,
        embed_images: bool = True
    ) -> Path:
        """
        Builds the interactive HTML report file.
        """
        out_path = Path(output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        density = report_data.get("density_metrics", {})
        scores = report_data.get("quality_scores", {})
        moments = report_data.get("key_moments", [])
        dims = report_data.get("editorial_dimensions", {})
        verdicts = scores.get("verdicts", {})

        # Build cards for key moments
        moment_cards_html = []
        for i, m in enumerate(moments, 1):
            raw_img_src = m.get("raw_frame_name", "")
            edited_img_src = m.get("edited_frame_name", "")

            if embed_images:
                raw_b64 = self._image_to_base64(m.get("raw_frame_path", ""))
                edited_b64 = self._image_to_base64(m.get("edited_frame_path", ""))
                if raw_b64:
                    raw_img_src = raw_b64
                if edited_b64:
                    edited_img_src = edited_b64

            verdict_badge_class = "badge-keep" if m["verdict"] == "KEEP" else ("badge-revise" if m["verdict"] == "REVISE" else "badge-remove")

            card = f"""
            <div class="moment-card" id="{m['id']}">
                <div class="moment-header">
                    <div class="moment-title-wrap">
                        <span class="moment-index">#{i}</span>
                        <h3>{m['label']}</h3>
                        <span class="op-tag">{m['operation']}</span>
                    </div>
                    <div class="moment-timing-verdict">
                        <span class="time-range">{m['interval'][0]:.2f}s – {m['interval'][1]:.2f}s</span>
                        <span class="verdict-badge {verdict_badge_class}">{m['verdict']}</span>
                    </div>
                </div>

                <div class="comparison-grid">
                    <div class="frame-col">
                        <div class="frame-header">
                            <span class="badge raw">RAW FOOTAGE</span>
                            <span class="timecode">T = {m['raw_timestamp']:.2f}s</span>
                        </div>
                        <div class="frame-container">
                            <img src="{raw_img_src}" alt="Raw Frame {m['id']}" loading="lazy" />
                        </div>
                    </div>
                    <div class="frame-col">
                        <div class="frame-header">
                            <span class="badge edited">AI-EDITED OUTPUT</span>
                            <span class="timecode">T = {m['edited_timestamp']:.2f}s</span>
                        </div>
                        <div class="frame-container">
                            <img src="{edited_img_src}" alt="Edited Frame {m['id']}" loading="lazy" />
                        </div>
                    </div>
                </div>

                <div class="moment-body">
                    <div class="rubric-bars">
                        <div class="rubric-item">
                            <span class="label">Story Value</span>
                            <div class="bar-track"><div class="bar-fill story" style="width: {m['story_value']*100}%"></div></div>
                            <span class="score">{m['story_value']:.2f}</span>
                        </div>
                        <div class="rubric-item">
                            <span class="label">Visual Value</span>
                            <div class="bar-track"><div class="bar-fill visual" style="width: {m['visual_value']*100}%"></div></div>
                            <span class="score">{m['visual_value']:.2f}</span>
                        </div>
                        <div class="rubric-item">
                            <span class="label">Pacing Value</span>
                            <div class="bar-track"><div class="bar-fill pacing" style="width: {m['pacing_value']*100}%"></div></div>
                            <span class="score">{m['pacing_value']:.2f}</span>
                        </div>
                        <div class="rubric-item">
                            <span class="label">Distraction Cost</span>
                            <div class="bar-track"><div class="bar-fill distraction" style="width: {m['distraction_cost']*100}%"></div></div>
                            <span class="score">{m['distraction_cost']:.2f}</span>
                        </div>
                        <div class="rubric-item">
                            <span class="label">Naturalness</span>
                            <div class="bar-track"><div class="bar-fill naturalness" style="width: {m['naturalness_score']*100}%"></div></div>
                            <span class="score">{m['naturalness_score']:.2f}</span>
                        </div>
                    </div>

                    <div class="critique-box">
                        <strong>Editorial Assessment:</strong> {m.get('editorial_critique', '')}
                    </div>
                </div>
            </div>
            """
            moment_cards_html.append(card)

        # Build 10 dimensions section
        dims_html = []
        dim_titles = {
            "1_technical_qa": "1. Technical QA (Aspect Ratio, Duration, Audio, Sync)",
            "2_editorial_quality": "2. Editorial Quality (Story Arc & Narrative Intent)",
            "3_potentially_unnecessary_edits": "3. Potentially Unnecessary Edits (Optimization Candidates)",
            "4_potentially_missing_edits": "4. Potentially Missing Edits (Creative Opportunities)",
            "5_over_editing_assessment": "5. Over-Editing Assessment (Density & Breathing Room)",
            "6_typography_assessment": "6. Typography Assessment (Hierarchy, Styling & Telugu Rendering)",
            "7_broll_assessment": "7. B-Roll Assessment (Contextual Relevance & Variety)",
            "8_punch_in_assessment": "8. Punch-In Assessment (Crop Framing & Dynamic Focus)",
            "9_sfx_assessment": "9. Sound Effects Assessment (Acoustic Restraint & Ducking)",
            "10_no_edit_assessment": "10. NO_EDIT Preservation (Authentic Speaker Rapport)",
        }
        for k, title in dim_titles.items():
            content = dims.get(k, "Not evaluated.")
            dims_html.append(f"""
            <div class="dim-card">
                <h4>{title}</h4>
                <p>{content}</p>
            </div>
            """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Reel Editor — Human-Quality Editorial Review</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --surface-elevated: #1f2937;
            --surface-border: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #38bdf8;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-purple: #a855f7;
            --accent-gold: #fbbf24;
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background: var(--bg);
            color: var(--text-primary);
            font-family: var(--font-main);
            line-height: 1.6;
            padding: 40px 24px;
        }}

        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--surface-border);
            padding-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .header-title h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 30%, var(--accent-blue) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-title p {{
            color: var(--text-secondary);
            margin-top: 6px;
            font-size: 1.05rem;
        }}

        .header-actions {{
            display: flex;
            gap: 12px;
            align-items: center;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-primary {{
            background: var(--accent-blue);
            color: #0b0f19;
            border: none;
        }}

        .btn-primary:hover {{
            background: #7dd3fc;
            transform: translateY(-1px);
        }}

        .btn-secondary {{
            background: var(--surface-elevated);
            color: var(--text-primary);
            border: 1px solid var(--surface-border);
        }}

        .btn-secondary:hover {{
            background: #2d3748;
        }}

        /* KPI Dashboard */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }}

        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .kpi-title {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
        }}

        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: var(--font-mono);
        }}

        .kpi-subtitle {{
            font-size: 0.85rem;
            color: var(--accent-emerald);
            font-weight: 500;
        }}

        .kpi-subtitle.balanced {{
            color: var(--accent-emerald);
        }}

        /* Overall Score Banner */
        .score-banner {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 24px;
        }}

        .score-group {{
            display: flex;
            gap: 32px;
            flex-wrap: wrap;
        }}

        .score-pill {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .score-pill span.metric-label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-weight: 600;
        }}

        .score-pill span.metric-val {{
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--accent-blue);
            font-family: var(--font-mono);
        }}

        /* Section Titles */
        .section-header {{
            margin: 40px 0 20px 0;
            display: flex;
            align-items: baseline;
            gap: 12px;
        }}

        .section-header h2 {{
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .section-header span {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}

        /* Key Moments Cards */
        .moment-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 28px;
            transition: border-color 0.2s;
        }}

        .moment-card:hover {{
            border-color: rgba(56, 189, 248, 0.4);
        }}

        .moment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .moment-title-wrap {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .moment-index {{
            background: var(--surface-elevated);
            color: var(--accent-blue);
            font-weight: 700;
            font-size: 0.85rem;
            padding: 4px 8px;
            border-radius: 6px;
            font-family: var(--font-mono);
        }}

        .moment-title-wrap h3 {{
            font-size: 1.25rem;
            font-weight: 700;
        }}

        .op-tag {{
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.25);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            font-family: var(--font-mono);
        }}

        .moment-timing-verdict {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .time-range {{
            font-family: var(--font-mono);
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        .verdict-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}

        .badge-keep {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-revise {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .badge-remove {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}

        /* Comparison Frame Grid */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        @media (max-width: 768px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .frame-col {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .frame-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .frame-header .badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
        }}

        .frame-header .badge.raw {{
            background: #374151;
            color: #d1d5db;
        }}

        .frame-header .badge.edited {{
            background: #0369a1;
            color: #e0f2fe;
        }}

        .frame-header .timecode {{
            font-family: var(--font-mono);
            color: var(--text-muted);
        }}

        .frame-container {{
            position: relative;
            background: #000000;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--surface-border);
            aspect-ratio: 9 / 16;
            max-height: 480px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .frame-container img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}

        /* Rubrics */
        .moment-body {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .rubric-bars {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            background: var(--surface-elevated);
            padding: 16px;
            border-radius: 10px;
        }}

        .rubric-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .rubric-item .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
        }}

        .bar-track {{
            height: 6px;
            background: #374151;
            border-radius: 3px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}

        .bar-fill.story {{ background: var(--accent-blue); }}
        .bar-fill.visual {{ background: var(--accent-emerald); }}
        .bar-fill.pacing {{ background: var(--accent-amber); }}
        .bar-fill.distraction {{ background: var(--accent-rose); }}
        .bar-fill.naturalness {{ background: var(--accent-purple); }}

        .rubric-item .score {{
            font-size: 0.85rem;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }}

        .critique-box {{
            background: rgba(56, 189, 248, 0.05);
            border-left: 3px solid var(--accent-blue);
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 0.95rem;
            color: #e2e8f0;
        }}

        /* 10 Dimensions Grid */
        .dimensions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-bottom: 60px;
        }}

        .dim-card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 20px;
        }}

        .dim-card h4 {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 10px;
        }}

        .dim-card p {{
            font-size: 0.92rem;
            color: var(--text-secondary);
            line-height: 1.55;
        }}

        footer {{
            border-top: 1px solid var(--surface-border);
            padding-top: 24px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Human-Quality Editorial Review</h1>
                <p>Comprehensive frame-by-frame quality assessment and pacing analysis for <strong>{report_data.get('source_video', 'video.mp4')}</strong></p>
            </div>
            <div class="header-actions">
                <a href="/editor" class="btn btn-primary">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M10 4v16"/><path d="M2 12h20"/></svg>
                    Open in Timeline Editor
                </a>
                <a href="#moments" class="btn btn-secondary">Jump to Frame Comparisons</a>
            </div>
        </header>

        <!-- KPI Metrics -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-title">Pacing Classification</span>
                <span class="kpi-value" style="color: var(--accent-emerald)">{density.get('pacing_classification', 'BALANCED')}</span>
                <span class="kpi-subtitle">{density.get('benchmark_range', '')}</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-title">Major Edit Density</span>
                <span class="kpi-value">{density.get('major_edits_per_minute', 0)} <span style="font-size: 1rem; color: var(--text-muted)">/min</span></span>
                <span class="kpi-subtitle">{density.get('major_edits', 0)} major edits across {density.get('duration_minutes', 0)}m</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-title">Total Edit Decisions</span>
                <span class="kpi-value">{density.get('total_edits', 0)}</span>
                <span class="kpi-subtitle">{density.get('edits_per_minute', 0)} operations / minute</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-title">SFX / B-Roll Density</span>
                <span class="kpi-value">{density.get('sfx_density_per_minute', 0)} <span style="font-size: 1rem; color: var(--text-muted)">SFX</span> | {density.get('broll_density_per_minute', 0)} <span style="font-size: 1rem; color: var(--text-muted)">B-roll</span></span>
                <span class="kpi-subtitle">Disciplined acoustic & cutaway pacing</span>
            </div>
        </div>

        <!-- Overall Score Banner -->
        <div class="score-banner">
            <div>
                <h3 style="font-size: 1.1rem; margin-bottom: 4px;">Editorial Rubric Averages (0.00 – 1.00)</h3>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Aggregated across all 9 critical narrative moments</p>
            </div>
            <div class="score-group">
                <div class="score-pill">
                    <span class="metric-label">Story Value</span>
                    <span class="metric-val">{scores.get('overall_story_value', 0):.2f}</span>
                </div>
                <div class="score-pill">
                    <span class="metric-label">Visual Value</span>
                    <span class="metric-val" style="color: var(--accent-emerald)">{scores.get('overall_visual_value', 0):.2f}</span>
                </div>
                <div class="score-pill">
                    <span class="metric-label">Pacing Value</span>
                    <span class="metric-val" style="color: var(--accent-amber)">{scores.get('overall_pacing_value', 0):.2f}</span>
                </div>
                <div class="score-pill">
                    <span class="metric-label">Distraction Cost</span>
                    <span class="metric-val" style="color: var(--accent-rose)">{scores.get('overall_distraction_cost', 0):.2f}</span>
                </div>
                <div class="score-pill">
                    <span class="metric-label">Naturalness</span>
                    <span class="metric-val" style="color: var(--accent-purple)">{scores.get('overall_naturalness_score', 0):.2f}</span>
                </div>
            </div>
        </div>

        <!-- Key Moments Frame Comparisons -->
        <div class="section-header" id="moments">
            <h2>Critical Editorial Moments (Raw vs Edited)</h2>
            <span>Matched timestamp frame pairs with editorial critique</span>
        </div>

        <div class="moments-list">
            {"".join(moment_cards_html)}
        </div>

        <!-- 10 In-Depth Editorial Dimensions -->
        <div class="section-header">
            <h2>Deep Professional Editorial Assessment (10 Dimensions)</h2>
            <span>Comprehensive evaluation of technical execution, restraint, and storytelling</span>
        </div>

        <div class="dimensions-grid">
            {"".join(dims_html)}
        </div>

        <footer>
            <p>Antigravity AI Reel Editor — Phase 6 Human-Quality Review & Editable Timeline Architecture</p>
        </footer>
    </div>
</body>
</html>
        """

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated human quality report at {out_path}")
        return out_path
