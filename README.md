# AI Reel Editor

A local-first, free-to-run AI editing decision system for Instagram Reels.

**Current milestone: Agent 1 — Raw Reel Analyzer**

Agent 1 analyzes a raw video and produces a structured `edit_blueprint.json`
containing every observable signal from the footage. It does not edit, render,
or apply creative decisions. It observes, measures, and surfaces opportunities.

---

## Architecture

```
Agent 1: Observer      → edit_blueprint.json
Agent 2: Creative Dir  → style_blueprint.json      (future)
Agent 3: Editor        → .otio timeline             (future, DaVinci Resolve)
Agent 4: Visual Assets → visual_assets_manifest     (future)
Agent 5: Motion Gfx    → rendered graphic clips     (future)
Agent 6: Sound Design  → audio mix / automation     (future)
Agent 7: QA            → review + correction loop   (future)
```

---

## System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Windows 10+ / Ubuntu 20.04+ / macOS 12+ | |
| Python | 3.11+ | |
| RAM | 8 GB | 4 GB works with `base` Whisper model |
| Storage | 10 GB free | For models and analysis outputs |
| GPU | Not required | Intel Iris Xe / integrated supported (CPU-only mode) |
| FFmpeg | Required | Must be on system PATH |
| Ollama | Recommended | For semantic/story analysis (LLM stage) |

---

## Quick Start

### 1. Install FFmpeg (Windows)

Download from: https://www.gyan.dev/ffmpeg/builds/
Get: `ffmpeg-release-essentials.zip`

Extract the zip. Add the `\bin` folder to your system PATH.

Verify: `ffmpeg -version`

### 2. Install Ollama (for LLM analysis)

Download from: https://ollama.com

After install, pull the recommended model:

```bash
ollama pull phi3.5:mini
```

Verify: `ollama list`

### 3. Create a Python virtual environment

```bash
cd reel-editor
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

Note: `faster-whisper` uses CTranslate2 (not PyTorch) — much lighter install.

### 5. Run the diagnostic script

```bash
python scripts/diagnose.py
```

This will:
- Detect your CPU, RAM, GPU
- Verify FFmpeg and Ollama are available
- Select the appropriate Whisper and LLM models
- Show estimated processing time
- Save `outputs/hardware_profile.json`

Run this every time you change hardware or install new models.

### 6. (Coming soon) Run analysis on a Reel

```bash
python scripts/run_analysis.py --video your_reel.mp4
```

---

## Project Structure

```
reel-editor/
├── config/
│   ├── hardware_profiles.yaml    # Model selection tiers
│   ├── editorial_rules.yaml      # Tunable thresholds (pauses, emphasis, etc.)
│   └── default_style_profile.yaml
│
├── core/
│   ├── hardware_detector.py      # CPU / RAM / GPU detection
│   ├── model_selector.py         # Selects Whisper + LLM model for your hardware
│   └── logger.py                 # Centralised loguru setup
│
├── agents/
│   └── agent1_analyzer/
│       ├── pipeline.py           # Orchestrates all 8 stages
│       └── stages/
│           ├── s0_hardware.py    # Stage 0: System diagnostics
│           ├── s1_media.py       # Stage 1: FFmpeg extraction
│           ├── s2_speech.py      # Stage 2: Whisper transcription
│           ├── s3_audio.py       # Stage 3: librosa audio analysis
│           ├── s4_visual.py      # Stage 4: OpenCV + MediaPipe
│           ├── s5_fusion.py      # Stage 5: Feature fusion
│           ├── s6_llm.py         # Stage 6: LLM story analysis (2 calls)
│           ├── s7_inventory.py   # Stage 7: Opportunity inventory
│           └── s8_blueprint.py   # Stage 8: Blueprint assembly + HTML report
│
├── schemas/
│   └── blueprint_v1.py           # Pydantic v2 schemas for edit_blueprint.json
│
├── outputs/                      # Analysis outputs (gitignored except .gitkeep)
├── assets/user_broll/            # Place your own B-roll footage here
├── scripts/
│   ├── diagnose.py               # ← START HERE
│   └── run_analysis.py           # Main CLI entry point (coming Day 2+)
└── tests/
```

---

## What Agent 1 Produces

### `edit_blueprint.json`
A structured, schema-validated JSON file containing:
- Full word-level timestamped transcript
- Audio signal measurements (energy, pitch, emphasis per word)
- Visual measurements (framing, face position, camera stability)
- Story structure (sections, hook, emotional peaks)
- **Opportunity inventory**: 8 typed opportunities, each with full evidence linkage

### `analysis_report.html`
A self-contained human-readable report with:
- Visual timeline
- Per-opportunity evidence display
- Quality warnings
- Explicit list of moments that should NOT be edited

### Opportunity Types

| Type | What it means |
|---|---|
| `CUT_OPPORTUNITY` | Potential filler/dead-space removal |
| `PACING_HOLD` | Pause worth preserving |
| `PUNCH_IN_OPPORTUNITY` | Subtle crop for visual emphasis |
| `BROLL_OPPORTUNITY` | Moment where visual evidence might help |
| `TEXT_EMPHASIS_OPPORTUNITY` | Key word with high emphasis value |
| `CONCEPT_VISUALIZATION_OPPORTUNITY` | Structural relationship with visual potential |
| `NO_EDIT_RECOMMENDATION` | Protect this moment — do not intervene |
| `COMPREHENSION_CONCERN` | High-density/abstract content that may need help |

Every opportunity contains: `reason`, `evidence`, `supporting_signals`,
`opposing_signals`, `confidence`, and `confidence_rationale`.

---

## Design Philosophy

Agent 1 asks:
> "What is happening, and what does this specific moment need?"

Not:
> "What effects can I add?"

The most important capability is deciding: **NO EDIT REQUIRED.**

---

## Cost

₹0 recurring. All analysis runs locally.
- Whisper: free, open-source (MIT)
- Ollama + Phi-3.5-mini: free, local
- FFmpeg: free, open-source (GPL)
- OpenCV, MediaPipe, librosa: free, open-source

---

## Limitations (Honest)

- **Processing time**: ~8–12 minutes for a 90-second Reel on Intel Iris Xe (CPU-only)
- **Hinglish quality**: Phi-3.5-mini has limited Hinglish semantic understanding. Confidence scores reflect this.
- **No internet required**: Agent 1 makes zero external API calls
- **No video output**: Agent 1 is an observer only
