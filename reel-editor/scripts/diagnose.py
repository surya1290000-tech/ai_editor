"""
scripts/diagnose.py

Hardware diagnostic script for the AI Reel Editor.

Run this FIRST before any analysis to:
  1. Detect your hardware and select appropriate models
  2. Verify FFmpeg and Ollama are available
  3. See estimated processing time
  4. Save a hardware_profile.json for the pipeline to use

Usage:
  python scripts/diagnose.py
  python scripts/diagnose.py --verbose
  python scripts/diagnose.py --save-overrides  (interactive model override)
"""

from __future__ import annotations

# Force UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError)
import sys as _sys
import io as _io
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.hardware_detector import detect_hardware
from core.model_selector import select_models
from core.logger import setup_logger, get_logger

setup_logger(verbose=False)
logger = get_logger(__name__)


# ─── ANSI colour helpers (Windows 10+ supports ANSI in terminal) ──────────────
def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")

# Safe ASCII alternatives for Windows terminals
SEP  = "-" * 40   # section separator line
TICK = "[OK]"     # success indicator
CROSS= "[!!]"     # error indicator
WARN = "[??]"     # warning indicator
ARROW= "  +-"     # tree branch indicator


# ─── Dependency checks ────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def check_ollama() -> tuple[bool, list[str]]:
    """
    Return (is_running, list_of_installed_model_names).
    Uses Ollama's local REST API on localhost:11434.
    """
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=4)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []


# ─── Main diagnostic ──────────────────────────────────────────────────────────

def run_diagnostics(verbose: bool = False) -> dict:
    """
    Run all diagnostic checks and return a combined profile dict.
    """
    print()
    print(bold("=" * 62))
    print(bold("  AI REEL EDITOR — SYSTEM DIAGNOSTICS"))
    print(bold("=" * 62))

    # ── Hardware ──────────────────────────────────────────────────
    print(f"\n  {bold('HARDWARE')}")
    print(f"  {SEP}")

    hw = detect_hardware()

    print(f"  OS         : {hw['os']}")
    print(f"  Python     : {hw['python_version']}")
    print(f"  CPU        : {hw['cpu']}")
    print(f"  Cores      : {hw['cpu_cores_logical']} logical / {hw['cpu_cores_physical']} physical")

    ram_avail = hw["ram_available_gb"]
    ram_total = hw["ram_total_gb"]
    ram_color = green if ram_avail >= 8 else (yellow if ram_avail >= 4 else red)
    print(f"  RAM        : {ram_total} GB total, {ram_color(str(ram_avail) + ' GB available')}")

    gpu_name = hw.get("gpu_name") or dim("Not detected")
    vram = hw.get("vram_gb")
    vram_str = f"{vram} GB" if vram else dim("shared/unknown")
    is_int = hw.get("gpu_is_integrated", False)
    gpu_type = dim("(integrated)") if is_int else green("(discrete)")
    print(f"  GPU        : {gpu_name} {gpu_type}")
    print(f"  VRAM       : {vram_str}")
    cuda = hw.get("cuda_available", False)
    print(f"  CUDA       : {green('Available') if cuda else dim('Not available')}")

    storage = hw.get("free_storage_gb")
    storage_str = f"{storage} GB free" if storage else dim("unknown")
    storage_color = green if (storage or 0) >= 10 else (yellow if (storage or 0) >= 5 else red)
    print(f"  Storage    : {storage_color(storage_str)}")

    # ── Dependencies ──────────────────────────────────────────────
    print(f"\n  {bold('DEPENDENCIES')}")
    print(f"  {SEP}")

    ffmpeg_ok = check_ffmpeg()
    ffmpeg_str = green("✓ Found") if ffmpeg_ok else red("✗ Not found — REQUIRED")
    print(f"  FFmpeg     : {ffmpeg_str}")

    ollama_ok, ollama_models = check_ollama()
    ollama_str = green("✓ Running") if ollama_ok else yellow("✗ Not running")
    print(f"  Ollama     : {ollama_str}")

    if ollama_models:
        print(f"  LLM models : {green(str(len(ollama_models)) + ' installed')}")
        for m in ollama_models:
            print(f"               +- {m}")
    elif ollama_ok:
        print(f"  LLM models : {yellow('None installed')} -- run: ollama pull phi3.5:mini")

    deps = {
        "ffmpeg_available": ffmpeg_ok,
        "ollama_running": ollama_ok,
        "ollama_models": ollama_models,
    }

    # ── Model selection ────────────────────────────────────────────
    print(f"\n  {bold('SELECTED MODELS')}")
    print(f"  {SEP}")

    models = select_models(hw, deps)
    profile_desc = models.get("profile_description", "")

    print(f"  Profile    : {cyan(models['profile_key'])}  {dim(profile_desc)}")
    print(f"  Whisper    : {cyan(models['whisper_model'])}  {dim('(' + models['whisper_device'] + ' / ' + models['whisper_compute_type'] + ')')}")

    llm_installed = models.get("llm_installed", False)
    llm_avail = models.get("llm_available", False)
    llm_str = cyan(models["llm_model"])
    if not llm_avail:
        llm_str += f"  {yellow('(Ollama not running)')}"
    elif not llm_installed:
        llm_str += f"  {yellow('(not installed -- run: ollama pull ' + models['llm_model'] + ')')}"
    else:
        llm_str += f"  {green('(installed)')}"
    print(f"  LLM        : {llm_str}")
    print(f"  Frame rate : {models['frame_sample_fps']} fps sample")

    # ── Time estimates ─────────────────────────────────────────────
    print(f"\n  {bold('ESTIMATED PROCESSING TIME')}  {dim('(90-second Reel)')}")
    print(f"  {SEP}")

    t = models["estimated_time"]
    print(f"  Whisper    : ~{t['whisper_min']} min")
    print(f"  LLM        : ~{t['llm_min']} min" + (dim("  (disabled)") if not llm_avail else ""))
    print(f"  CV/Visual  : ~{t['cv_min']} min")
    print(f"  {'-' * 26}")
    print(f"  {bold('Total')}      : ~{int(t['total_min'])}-{int(t['total_max'])} min")

    # ── Warnings ───────────────────────────────────────────────────
    all_warnings = list(models.get("warnings", []))

    if not ffmpeg_ok:
        all_warnings.insert(0,
            "FFmpeg not found. This is required for all video processing.\n"
            "    Download: https://www.gyan.dev/ffmpeg/builds/\n"
            "    Extract the zip and add the /bin folder to your system PATH."
        )

    if storage and storage < 5:
        all_warnings.append(
            f"Low disk space ({storage} GB free). Analysis outputs can be 100–500 MB per run."
        )

    if all_warnings:
        print(f"\n  {bold('WARNINGS')}")
        print(f"  {SEP}")
        for w in all_warnings:
            lines = w.split("\n")
            print(f"  {yellow('[??]')}  {lines[0]}")
            for line in lines[1:]:
                print(f"       {line}")

    # ── Status summary ─────────────────────────────────────────────
    ready = ffmpeg_ok
    llm_ready = ollama_ok and llm_installed

    print(f"\n  {bold('STATUS')}")
    print(f"  {SEP}")
    if ready and llm_ready:
        print(f"  {green('[READY]')}   Full pipeline available (transcript + vision + semantic)")
    elif ready and ollama_ok and not llm_installed:
        print(f"  {yellow('[PARTIAL]')} FFmpeg OK, Ollama running, LLM not installed")
        print(f"     Run: {cyan('ollama pull ' + models['llm_model'])}  then re-run diagnose.py")
    elif ready and not ollama_ok:
        print(f"  {yellow('[PARTIAL]')} FFmpeg OK, Ollama not running")
        print(f"     Semantic analysis disabled. Start Ollama for full pipeline.")
    else:
        print(f"  {red('[NOT READY]')} FFmpeg is required. See warnings above.")

    print(bold("\n" + "=" * 62 + "\n"))

    # ── Assemble full profile ──────────────────────────────────────
    profile = {
        "generated_at": datetime.now().isoformat(),
        "hardware": hw,
        "dependencies": deps,
        "models": models,
        "ready_for_analysis": ready,
        "llm_ready": llm_ready,
    }

    return profile


def save_profile(profile: dict) -> Path:
    """Save the hardware profile to outputs/hardware_profile.json."""
    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hardware_profile.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return output_path


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Reel Editor — Hardware Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug-level output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full profile as JSON to stdout (for scripting)",
    )
    args = parser.parse_args()

    profile = run_diagnostics(verbose=args.verbose)
    path = save_profile(profile)
    print(f"  Profile saved: {dim(str(path))}\n")

    if args.json:
        print(json.dumps(profile, indent=2))

    # Exit code 1 if FFmpeg is missing (blocking dependency)
    return 0 if profile["dependencies"]["ffmpeg_available"] else 1


if __name__ == "__main__":
    sys.exit(main())
