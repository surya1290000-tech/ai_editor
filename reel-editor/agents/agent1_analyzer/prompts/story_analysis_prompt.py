"""
agents/agent1_analyzer/prompts/story_analysis_prompt.py

Structured prompt template for Stage 6: Story & Semantic Arc Analyzer.
Enforces strict JSON schema output from local LLM (Ollama).
"""

STORY_ANALYSIS_SYSTEM_PROMPT = """You are an expert video narrative analyst and storytelling observer for short-form video (Reels, TikToks, Shorts).
Your role is to strictly OBSERVE and ANALYZE the spoken narrative, structure, and emotional progression of raw footage.
You do NOT dictate creative styling, color grading, or software edits.
You identify narrative hooks, thematic structure, key semantic peaks, emotional curves, and comprehension or retention drop-off risks.

You must respond ONLY with a valid, parseable JSON object matching the exact schema requested. No prose before or after the JSON.
"""

STORY_ANALYSIS_USER_PROMPT = """Analyze the following video briefing derived from audio transcription, vocal energy analysis, and visual computer vision:

---
VIDEO TITLE: {video_title}
DURATION: {duration_seconds:.1f} seconds
TOTAL SENTENCES: {total_sentences}
PACING HEALTH: {pacing_health}
DEAD SPACE: {dead_space_count} instances ({total_dead_space_seconds:.2f}s total)

SENTENCE FLOW & MULTI-MODAL DELIVERY:
{sentence_flow_formatted}
---

Return a JSON object with EXACTLY this structure:
{{
  "hook_analysis": {{
    "opening_text": "text of sentence 0",
    "hook_type": "philosophical_premise | bold_claim | question | personal_anecdote | counter_intuitive_statement",
    "hook_strength_score": 8,
    "first_3_seconds_assessment": "concise 1-2 sentence evaluation of whether the opening grabs attention",
    "retention_risk_opening": "low | medium | high",
    "recommendation": "objective observation on the hook's clarity and punch"
  }},
  "narrative_acts": [
    {{
      "act_number": 1,
      "act_name": "Hook & Premise | The Conflict | Re-framing | Climax & Core Insight | Resolution",
      "sentence_range": [0, 2],
      "start_time": 0.2,
      "end_time": 10.9,
      "summary": "1 sentence summarizing what happens in this act",
      "purpose": "Why this section exists in the narrative",
      "primary_emotion": "reflective | urgent | empathetic | confrontational | inspirational"
    }}
  ],
  "core_themes": {{
    "primary_topic": "The main concept of the video",
    "core_insight": "The single most important lesson or thesis statement",
    "key_takeaways": [
      "Takeaway 1",
      "Takeaway 2",
      "Takeaway 3"
    ]
  }},
  "semantic_peaks": [
    {{
      "sentence_id": 14,
      "text": "Quote of the sentence",
      "timestamp": "65.4s - 68.4s",
      "peak_type": "thesis_statement | emotional_climax | rhetorical_question | key_twist",
      "impact_score": 9,
      "significance": "Why this is a high-impact moment"
    }}
  ],
  "dropoff_risk_moments": [
    {{
      "sentence_id": 2,
      "timestamp": "10.9s - 15.6s",
      "risk_type": "unclear_phrasing | monotone_delivery | repetitive_argument | prolonged_lull",
      "severity": "low | medium | high",
      "observation": "Objective description of why viewer attention may wander here"
    }}
  ],
  "emotional_trajectory": [
    {{
      "sentence_id": 0,
      "emotion": "reflective",
      "intensity": 6
    }}
  ],
  "executive_summary": "A concise 2-3 sentence overview of the video's narrative power and storytelling effectiveness."
}}
"""
