"""
agents/agent2_director/reporter.py

Phase 3 Editorial Intelligence HTML Reporter
────────────────────────────────────────────
Generates the comprehensive dark-mode editorial report (editorial_report.html):
  TIMESTAMP → TRANSCRIPT → EVIDENCE → STORY ACT → OPPORTUNITY → AGENT 2 DECISION → REASON → CONFIDENCE
Includes:
  - Total Opportunities, Accepted, Rejected, NO_EDIT counts
  - Edit Budget vs Budget Used tracking
  - Full traceability chain inspection
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from core.evidence.models import EnrichedTranscript, EvidenceInventory
from agents.agent2_director.models import StoryEditPlan, EditDecisionsDocument


def generate_editorial_html_report(
    story_plan: StoryEditPlan,
    decisions_doc: EditDecisionsDocument,
    transcript: EnrichedTranscript,
    evidence_inv: EvidenceInventory,
    output_path: Path | str,
) -> Path:
    """Renders comprehensive editorial decision document to HTML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b_util = decisions_doc.budget_utilization
    summary = story_plan.decisions_summary

    # Build Budget Cards HTML
    budget_cards_html = [
        f"""
        <div class="stat-card">
            <div class="stat-val">{b_util.get('hero_text', {}).get('used', 0)} / {b_util.get('hero_text', {}).get('limit', 2)}</div>
            <div class="stat-label">Hero Text Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{b_util.get('punch_ins', {}).get('total_used', 0)} / 16</div>
            <div class="stat-label">Punch-Ins (S:{b_util.get('punch_ins',{}).get('strong',{}).get('used',0)} M:{b_util.get('punch_ins',{}).get('medium',{}).get('used',0)})</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{b_util.get('broll', {}).get('used', 0)} / {b_util.get('broll', {}).get('limit', 3)}</div>
            <div class="stat-label">B-Roll Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{b_util.get('sfx', {}).get('used', 0)} / {b_util.get('sfx', {}).get('limit', 6)}</div>
            <div class="stat-label">SFX Cues Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{b_util.get('emphasis_text', {}).get('used', 0)} / {b_util.get('emphasis_text', {}).get('limit', 8)}</div>
            <div class="stat-label">Emphasis Text</div>
        </div>
        """
    ]

    # Map words and sentences for quick lookup
    sentences_by_time = transcript.sentences

    def get_transcript_for_interval(st: float, en: float) -> str:
        matching = [
            s.text for s in sentences_by_time
            if not (s.end < st - 0.1 or s.start > en + 0.1)
        ]
        if matching:
            return matching[0]
        # fallback to words
        words = [w.word for w in transcript.words if st <= w.start <= en]
        return " ".join(words) if words else "—"

    # Build Decisions Rows HTML
    decision_rows_html = []
    for d in decisions_doc.all_decisions:
        # Status class
        status_class = f"status-{d.status.lower().replace('_', '-')}"
        op_class = f"op-{d.operation.lower().replace('_', '-')}"

        tx_text = get_transcript_for_interval(d.start, d.end)
        opp_str = ", ".join(d.source_opportunity_ids) if d.source_opportunity_ids else "—"
        ev_str = ", ".join(d.source_evidence_ids[:3]) if d.source_evidence_ids else "—"

        # Conceptual params badge
        extra_param_html = ""
        if d.strength:
            extra_param_html = f'<span class="param-pill">Strength: {d.strength}</span>'
        elif d.text_content:
            extra_param_html = f'<span class="param-pill">Text: "{html.escape(d.text_content)}"</span>'
        elif d.concept_query:
            extra_param_html = f'<span class="param-pill">Query: "{html.escape(d.concept_query)}"</span>'
        elif d.sound_cue:
            extra_param_html = f'<span class="param-pill">Cue: {d.sound_cue}</span>'
        elif d.trim_duration:
            extra_param_html = f'<span class="param-pill">Trim: {d.trim_duration:.2f}s</span>'

        decision_rows_html.append(f"""
        <tr class="dec-row {status_class}">
            <td class="col-time"><span class="time-range">{d.start:.2f}s – {d.end:.2f}s</span><br><span class="dur">({d.duration:.2f}s)</span></td>
            <td class="col-act"><span class="act-pill act-{d.story_act.lower()}">{html.escape(d.story_act)}</span></td>
            <td class="col-tx"><div class="tx-snippet">"{html.escape(tx_text)}"</div></td>
            <td class="col-trace">
                <div class="trace-item"><span class="trace-label">Opp:</span> <code>{html.escape(opp_str)}</code></div>
                <div class="trace-item"><span class="trace-label">Ev:</span> <code>{html.escape(ev_str)}</code></div>
            </td>
            <td class="col-dec">
                <span class="status-badge {status_class}">{d.status}</span>
                <span class="op-badge {op_class}">{html.escape(d.operation)}</span>
                {extra_param_html}
            </td>
            <td class="col-reason">{html.escape(d.reason)}</td>
            <td class="col-conf">
                <div class="conf-val">{d.confidence:.2f}</div>
                <div class="imp-val">Imp: {d.importance:.2f}</div>
            </td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent 2 Story Editor Editorial Report — {html.escape(story_plan.source_video)}</title>
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
    --accent-cyan: #39c5bb;
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

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .stat-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }}
  .stat-val {{ font-size: 22px; font-weight: 700; color: var(--text-primary); }}
  .stat-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}

  .strategy-box {{
    background: #1b212a;
    border-left: 4px solid var(--accent-blue);
    padding: 14px 18px;
    border-radius: 4px;
    margin-bottom: 24px;
    font-size: 14px;
  }}

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
  th, td {{ padding: 10px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }}
  th {{ background: #1b212a; color: var(--text-muted); font-size: 11px; text-transform: uppercase; }}
  tr:hover {{ background: var(--bg-hover); }}

  .time-range {{ font-family: monospace; font-weight: 600; color: var(--accent-blue); }}
  .dur {{ font-size: 11px; color: var(--text-muted); }}
  .tx-snippet {{ font-style: italic; color: #a5d6ff; font-size: 12px; max-width: 240px; }}

  .status-badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .status-accepted {{ background: rgba(63, 185, 80, 0.2); color: #3fb950; border: 1px solid #3fb950; }}
  .status-no-edit {{ background: rgba(57, 197, 187, 0.2); color: #39c5bb; border: 1px solid #39c5bb; }}
  .status-rejected {{ background: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid #484f58; }}

  .op-badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    margin-left: 4px;
    background: #21262d;
    color: var(--text-primary);
  }}

  .act-pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
  }}
  .act-hook {{ background: rgba(188, 140, 255, 0.2); color: #bc8cff; }}
  .act-context {{ background: rgba(88, 166, 255, 0.2); color: #58a6ff; }}
  .act-problem {{ background: rgba(248, 81, 73, 0.2); color: #f85149; }}
  .act-realization {{ background: rgba(63, 185, 80, 0.2); color: #3fb950; }}
  .act-payoff {{ background: rgba(210, 153, 34, 0.2); color: #d29922; }}

  .param-pill {{
    display: inline-block;
    margin-top: 4px;
    padding: 1px 6px;
    background: rgba(210, 153, 34, 0.15);
    color: #e3b341;
    border-radius: 3px;
    font-size: 11px;
    font-family: monospace;
  }}

  .trace-item {{ font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }}
  .trace-label {{ font-weight: 600; color: #8b949e; }}
  code {{ font-family: monospace; background: #21262d; padding: 1px 4px; border-radius: 3px; font-size: 11px; }}
  .conf-val {{ font-family: monospace; font-weight: 700; font-size: 12px; }}
  .imp-val {{ font-size: 11px; color: var(--text-muted); }}
</style>
</head>
<body>

<h1>Agent 2 Story Editor — Editorial Intelligence Report</h1>
<div class="header-meta">
  Source Footage: <b>{html.escape(story_plan.source_video)}</b> &bull;
  Duration: <b>{story_plan.duration_seconds:.2f}s</b> &bull;
  Preset: <b>{story_plan.pacing_preset.upper()}</b> &bull;
  Selectivity Ratio: <b>{summary.get('selectivity_ratio', 'N/A')}</b>
</div>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-val">{summary.get('total_opportunities_evaluated', 0)}</div>
    <div class="stat-label">Total Opportunities</div>
  </div>
  <div class="stat-card" style="border-color: var(--accent-emerald);">
    <div class="stat-val" style="color: var(--accent-emerald);">{summary.get('total_accepted_interventions', 0)}</div>
    <div class="stat-label">Accepted Edits</div>
  </div>
  <div class="stat-card" style="border-color: var(--accent-cyan);">
    <div class="stat-val" style="color: var(--accent-cyan);">{summary.get('total_no_edit_preserved', 0)}</div>
    <div class="stat-label">NO_EDIT Preserved</div>
  </div>
  <div class="stat-card">
    <div class="stat-val" style="color: var(--text-muted);">{summary.get('total_rejected', 0)}</div>
    <div class="stat-label">Rejected Opportunities</div>
  </div>
</div>

<h2>Edit Budget vs. Actual Utilization</h2>
<div class="stats-grid">
  {''.join(budget_cards_html)}
</div>

<h2>Editorial Strategy</h2>
<div class="strategy-box">
  <b>Narrative Direction:</b> {html.escape(story_plan.narrative_strategy)}
</div>

<h2>Master Editorial Decisions (Traceable Chain)</h2>
<table>
  <thead>
    <tr>
      <th style="width:115px">Timestamp</th>
      <th style="width:100px">Story Act</th>
      <th style="width:220px">Transcript Context</th>
      <th style="width:130px">Traceability (Opp/Ev)</th>
      <th style="width:190px">Agent 2 Decision</th>
      <th>Editorial Rationale</th>
      <th style="width:75px">Score</th>
    </tr>
  </thead>
  <tbody>
    {''.join(decision_rows_html)}
  </tbody>
</table>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
