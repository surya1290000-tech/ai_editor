"""
core/evidence/reporter.py

Phase 2 Multi-Modal Analysis & Evidence HTML Reporter
────────────────────────────────────────────────────
Generates an interactive, inspectable HTML timeline report:
  timestamp → transcript → audio signal → visual signal → observation → confidence & uncertainty
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from core.evidence.models import (
    EvidenceInventory,
    StoryObservationsInventory,
    OpportunityInventory,
    EnrichedTranscript,
)


def generate_analysis_html_report(
    evidence_inv: EvidenceInventory,
    story_obs: StoryObservationsInventory,
    opp_inv: OpportunityInventory,
    transcript: EnrichedTranscript,
    output_path: Path | str,
) -> Path:
    """Generates a comprehensive, modern dark-themed HTML report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_words = len(transcript.words)
    low_conf_words = sum(1 for w in transcript.words if w.low_confidence_flag)
    low_conf_pct = (low_conf_words / max(total_words, 1)) * 100

    no_edit_items = evidence_inv.filter_by_type("no_edit_region")
    no_edit_duration = sum(e.end - e.start for e in no_edit_items)
    no_edit_coverage = (no_edit_duration / max(evidence_inv.duration_seconds, 1.0)) * 100

    # Build story cards HTML
    story_cards_html = []
    for so in story_obs.story_observations:
        story_cards_html.append(f"""
        <div class="story-card act-{so.type.lower()}">
            <div class="story-header">
                <span class="act-badge">{html.escape(so.type)}</span>
                <span class="act-time">{so.time_range[0]:.1f}s – {so.time_range[1]:.1f}s</span>
                <span class="act-conf">Conf: {so.confidence:.2f}</span>
            </div>
            <p class="story-summary">{html.escape(so.summary)}</p>
            <div class="story-meta">Evidence IDs: <code>{', '.join(so.evidence_ids[:5])}</code></div>
        </div>
        """)

    # Build evidence rows HTML
    evidence_rows_html = []
    for ev in evidence_inv.evidence:
        # Badge color class
        type_class = f"badge-{ev.type.replace('_', '-')}"

        # Uncertainty pills
        unc_html = ""
        if ev.uncertainty:
            pills = "".join(f'<span class="unc-pill">⚠️ {html.escape(u)}</span>' for u in ev.uncertainty)
            unc_html = f'<div class="unc-container">{pills}</div>'

        # Signal details
        audio_details = []
        for k, v in ev.signals.audio.items():
            if isinstance(v, float):
                audio_details.append(f"<b>{k}</b>: {v:+.2f}" if "deviation" in k or "energy" in k else f"<b>{k}</b>: {v:.2f}")
            else:
                audio_details.append(f"<b>{k}</b>: {v}")
        audio_str = " &bull; ".join(audio_details) if audio_details else "—"

        visual_details = []
        for k, v in ev.signals.visual.items():
            if isinstance(v, float):
                visual_details.append(f"<b>{k}</b>: {v:.2f}")
            else:
                visual_details.append(f"<b>{k}</b>: {v}")
        visual_str = " &bull; ".join(visual_details) if visual_details else "—"

        speech_details = []
        for k, v in ev.signals.speech.items():
            if k == "text":
                continue
            speech_details.append(f"<b>{k}</b>: {v}")
        speech_str = " &bull; ".join(speech_details) if speech_details else "—"

        # Transcript text if available
        matched_text = ""
        if ev.source.transcript_segment_ids:
            s_texts = [
                s.text for s in transcript.sentences if s.id in ev.source.transcript_segment_ids
            ]
            if s_texts:
                matched_text = f'<div class="transcript-quote">"{html.escape(s_texts[0])}"</div>'
        elif ev.source.word_ids:
            w_objs = [w for w in transcript.words if w.id in ev.source.word_ids]
            if w_objs:
                words_html = []
                for w in w_objs:
                    w_class = "word-low-conf" if w.low_confidence_flag else "word-normal"
                    words_html.append(f'<span class="{w_class}">"{html.escape(w.word)}" [{w.language}:{w.confidence:.2f}]</span>')
                matched_text = f'<div class="transcript-quote">{" ".join(words_html)}</div>'

        rel_class = f"rel-{ev.reliability}"

        evidence_rows_html.append(f"""
        <tr class="evidence-row">
            <td class="col-id"><code>{ev.evidence_id}</code></td>
            <td class="col-time"><span class="time-range">{ev.start:.2f}s – {ev.end:.2f}s</span><br><span class="dur">({ev.duration:.2f}s)</span></td>
            <td class="col-type"><span class="evidence-badge {type_class}">{html.escape(ev.type)}</span></td>
            <td class="col-obs">
                <div class="obs-text">{html.escape(ev.observation)}</div>
                {matched_text}
                {unc_html}
            </td>
            <td class="col-signals">
                <div class="sig-block"><span class="sig-title">Audio:</span> {audio_str}</div>
                <div class="sig-block"><span class="sig-title">Visual:</span> {visual_str}</div>
                <div class="sig-block"><span class="sig-title">Speech:</span> {speech_str}</div>
            </td>
            <td class="col-conf">
                <div class="conf-badge">{ev.confidence:.2f}</div>
                <div class="rel-badge {rel_class}">{ev.reliability.upper()}</div>
            </td>
        </tr>
        """)

    # Build opportunity rows HTML
    opp_rows_html = []
    for opp in opp_inv.opportunities:
        opp_type_class = f"opp-{opp.type.replace('_', '-')}"
        opp_rows_html.append(f"""
        <tr class="opp-row">
            <td><code>{opp.opportunity_id}</code></td>
            <td><span class="evidence-badge {opp_type_class}">{html.escape(opp.type)}</span></td>
            <td><span class="time-range">{opp.time_range[0]:.2f}s – {opp.time_range[1]:.2f}s</span></td>
            <td>{opp.description}</td>
            <td><code>{', '.join(opp.evidence_ids)}</code></td>
            <td><span class="conf-badge">{opp.confidence:.2f}</span></td>
            <td><span class="prio-badge">Prio: {opp.priority:.2f}</span></td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent 1 Observer Evidence Report — {html.escape(evidence_inv.source_video)}</title>
<style>
  :root {{
    --bg-main: #0d1117;
    --bg-card: #161b22;
    --bg-hover: #21262d;
    --border: #30363d;
    --text-primary: #f0f6fc;
    --text-muted: #8b949e;
    --accent-blue: #58a6ff;
    --accent-emerald: #3fb950;
    --accent-amber: #d29922;
    --accent-rose: #f85149;
    --accent-purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px 32px;
  }}
  h1, h2, h3 {{ color: var(--text-primary); font-weight: 600; }}
  h1 {{ font-size: 24px; margin-bottom: 6px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px 0; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .header-meta {{ color: var(--text-muted); font-size: 13px; margin-bottom: 20px; }}
  
  /* Stats Cards */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }}
  .stat-val {{ font-size: 24px; font-weight: 700; color: var(--text-primary); }}
  .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}

  /* Story Cards */
  .story-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-bottom: 28px;
  }}
  .story-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    border-left: 4px solid var(--accent-blue);
  }}
  .story-card.act-hook {{ border-left-color: var(--accent-purple); }}
  .story-card.act-realization {{ border-left-color: var(--accent-emerald); }}
  .story-card.act-problem {{ border-left-color: var(--accent-rose); }}
  .story-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .act-badge {{ background: #1f242c; font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 12px; }}
  .act-time {{ font-size: 12px; color: var(--accent-blue); font-family: monospace; }}
  .act-conf {{ font-size: 11px; color: var(--text-muted); }}
  .story-summary {{ font-size: 13px; color: var(--text-primary); margin-bottom: 8px; }}
  .story-meta {{ font-size: 11px; color: var(--text-muted); }}

  /* Evidence Table */
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 32px;
    font-size: 13px;
  }}
  th, td {{ padding: 10px 14px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }}
  th {{ background: #1b212a; color: var(--text-muted); font-size: 11px; text-transform: uppercase; }}
  tr:hover {{ background: var(--bg-hover); }}
  .time-range {{ font-family: monospace; font-weight: 600; color: var(--accent-blue); }}
  .dur {{ font-size: 11px; color: var(--text-muted); }}
  
  /* Badges */
  .evidence-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    background: #21262d;
    color: var(--text-primary);
  }}
  .badge-no-edit-region, .opp-no-edit-opportunity {{ background: rgba(63, 185, 80, 0.2); color: #3fb950; border: 1px solid #3fb950; }}
  .badge-dead-space, .opp-cut-opportunity {{ background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }}
  .badge-emphasis, .opp-emphasis-text-opportunity {{ background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #d29922; }}
  .badge-hook-candidate, .opp-hero-text-opportunity {{ background: rgba(188, 140, 255, 0.2); color: #bc8cff; border: 1px solid #bc8cff; }}
  .badge-punch-in-opportunity, .opp-punch-in-opportunity {{ background: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid #58a6ff; }}
  .badge-broll-opportunity, .opp-broll-opportunity {{ background: rgba(56, 139, 253, 0.2); color: #79c0ff; border: 1px solid #79c0ff; }}
  .badge-sfx-opportunity, .opp-sfx-opportunity {{ background: rgba(219, 97, 162, 0.2); color: #db61a2; border: 1px solid #db61a2; }}
  .badge-rhetorical-pause {{ background: rgba(86, 211, 100, 0.15); color: #7ee787; }}
  .badge-filler {{ background: rgba(110, 118, 129, 0.2); color: #8b949e; }}

  /* Uncertainty & Transcript */
  .unc-container {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .unc-pill {{ background: rgba(210, 153, 34, 0.15); color: #e3b341; padding: 1px 6px; border-radius: 4px; font-size: 11px; }}
  .transcript-quote {{ font-style: italic; color: #a5d6ff; margin-top: 4px; font-size: 12px; }}
  .word-low-conf {{ color: #e3b341; text-decoration: underline dotted; }}
  
  /* Signals */
  .sig-block {{ margin-bottom: 2px; font-size: 11px; color: var(--text-muted); }}
  .sig-title {{ font-weight: 600; color: #c9d1d9; }}
  
  .conf-badge {{ font-family: monospace; font-weight: 700; font-size: 12px; }}
  .rel-badge {{ font-size: 10px; font-weight: 700; padding: 1px 4px; border-radius: 3px; display: inline-block; margin-top: 2px; }}
  .rel-high {{ background: rgba(63, 185, 80, 0.2); color: #3fb950; }}
  .rel-medium {{ background: rgba(210, 153, 34, 0.2); color: #d29922; }}
  .rel-low {{ background: rgba(248, 81, 73, 0.2); color: #f85149; }}
  .prio-badge {{ font-size: 11px; font-family: monospace; color: var(--accent-amber); }}
  code {{ font-family: monospace; background: #21262d; padding: 2px 4px; border-radius: 4px; font-size: 11px; }}
</style>
</head>
<body>

<h1>Agent 1 Observer Multi-Modal Evidence Report</h1>
<div class="header-meta">
  Source: <b>{html.escape(evidence_inv.source_video)}</b> &bull;
  Duration: <b>{evidence_inv.duration_seconds:.2f}s</b> &bull;
  Languages: <b>{', '.join(transcript.detected_languages)}</b> &bull;
  Schema Version: <b>3.0.0</b> (Phase 2 Evidence Layer)
</div>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-val">{len(evidence_inv.evidence)}</div>
    <div class="stat-label">Evidence Items</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{len(story_obs.story_observations)}</div>
    <div class="stat-label">Story Acts</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{len(opp_inv.opportunities)}</div>
    <div class="stat-label">Opportunities</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{no_edit_coverage:.1f}%</div>
    <div class="stat-label">NO-EDIT Coverage ({no_edit_duration:.1f}s)</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{low_conf_pct:.1f}%</div>
    <div class="stat-label">Low-Confidence ASR Words</div>
  </div>
</div>

<h2>1. Narrative Structure & Story Observations</h2>
<div class="story-grid">
  {''.join(story_cards_html)}
</div>

<h2>2. Multi-Modal Evidence Timeline (Inspectable Ground Truth)</h2>
<table>
  <thead>
    <tr>
      <th style="width:70px">ID</th>
      <th style="width:130px">Timestamp</th>
      <th style="width:140px">Evidence Type</th>
      <th>Observation & Grounded Text</th>
      <th style="width:260px">Signals (Audio &bull; Visual &bull; Speech)</th>
      <th style="width:70px">Conf</th>
    </tr>
  </thead>
  <tbody>
    {''.join(evidence_rows_html)}
  </tbody>
</table>

<h2>3. Candidate Opportunities Inventory (for Agent 2 Review)</h2>
<table>
  <thead>
    <tr>
      <th style="width:80px">ID</th>
      <th style="width:160px">Type</th>
      <th style="width:130px">Time Range</th>
      <th>Observed Opportunity Rationale</th>
      <th style="width:120px">Evidence Grounding</th>
      <th style="width:60px">Conf</th>
      <th style="width:80px">Priority</th>
    </tr>
  </thead>
  <tbody>
    {''.join(opp_rows_html)}
  </tbody>
</table>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
