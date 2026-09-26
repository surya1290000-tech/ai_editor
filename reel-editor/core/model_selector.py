"""
core/model_selector.py

Selects the optimal Whisper and LLM models based on detected hardware.
Reads hardware_profiles.yaml for tier definitions and time estimates.

Design principles:
- No model is selected that cannot realistically run on the detected hardware.
- If Ollama is not running, LLM is marked unavailable (pipeline continues without it).
- Estimates are honest — CPU-only is slow, and we say so.
- Users can override selections via config/overrides.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import yaml


_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_profiles() -> dict:
    """Load hardware_profiles.yaml."""
    path = _CONFIG_DIR / "hardware_profiles.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_overrides() -> dict:
    """Load optional user overrides. Returns empty dict if not present."""
    path = _CONFIG_DIR / "overrides.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _select_profile_key(hw: dict) -> str:
    """
    Map hardware properties to a profile key from hardware_profiles.yaml.

    Priority:
    1. GPU with CUDA → gpu_high_vram or gpu_low_vram
    2. Integrated GPU (Intel Iris, AMD Vega, Apple M) → treat as cpu_only
    3. No GPU → cpu_only_*
    """
    cuda = hw.get("cuda_available", False)
    vram = hw.get("vram_gb") or 0.0
    ram_avail = hw.get("ram_available_gb", 4.0)
    is_integrated = hw.get("gpu_is_integrated", True)

    if cuda and not is_integrated:
        if vram >= 8:
            return "gpu_high_vram"
        else:
            return "gpu_low_vram"

    # CPU-only path
    if ram_avail >= 8:
        return "cpu_only_high"
    elif ram_avail >= 4:
        return "cpu_only_standard"
    else:
        return "cpu_only_low"


def _find_installed_llm(ollama_models: list[str], preferred: str) -> tuple[str, bool]:
    """
    Find the best available installed LLM from the Ollama model list.
    Returns (model_name, is_installed).

    Preference order tries to match the preferred model first,
    then falls back through a ranked list.
    """
    fallback_order = [
        preferred,
        "phi3.5:mini",
        "gemma3:4b",       # Strong 4B model — excellent for story analysis
        "phi3.5:mini",
        "phi3:mini",
        "llama3.2:latest",
        "llama3.2:3b",
        "gemma2:2b",
        "mistral:latest",
        "tinyllama:latest",
    ]

    for candidate in fallback_order:
        candidate_base = candidate.split(":")[0].lower()
        for installed in ollama_models:
            if candidate_base in installed.lower():
                return installed, True

    # Nothing installed — return preferred as recommendation (user must pull it)
    return preferred, False


def _estimate_time(
    whisper_model: str,
    llm_model: str,
    llm_available: bool,
    is_gpu: bool,
    profiles_data: dict,
) -> dict:
    """
    Estimate processing time (minutes) for a typical 90-second Reel.
    Returns dict with per-stage and total min/max.
    """
    estimates = profiles_data.get("time_estimates", {})

    # GPU acceleration factor
    mode = "gpu" if is_gpu else "cpu"

    whisper_times = estimates.get("whisper", {})
    wt = (whisper_times.get(whisper_model) or {}).get(mode, 3.5)

    llm_times = estimates.get("llm", {})
    lt = 0.0
    if llm_available:
        llm_key = llm_model  # try exact match first
        lt = (llm_times.get(llm_key) or {}).get(mode, 2.5)

        # If no exact match, try prefix match
        if lt == 2.5:
            for k, v in llm_times.items():
                if llm_model.split(":")[0] in k:
                    lt = v.get(mode, 2.5)
                    break

    cv = estimates.get("cv_fixed", 1.5)
    other = estimates.get("other", 0.5)

    total = wt + lt + cv + other
    return {
        "whisper_min": round(wt, 1),
        "llm_min": round(lt, 1),
        "cv_min": round(cv, 1),
        "total_min": round(total, 0),
        "total_max": round(total * 1.35, 0),  # 35% buffer
    }


def select_models(hw: dict, deps: dict) -> dict:
    """
    Select optimal models and produce processing time estimates.

    Args:
        hw: Output from hardware_detector.detect_hardware()
        deps: Output from diagnose.check_dependencies()

    Returns dict with:
        profile_key, whisper_model, whisper_device, whisper_compute_type,
        llm_model, llm_available, llm_installed, llm_to_pull,
        frame_sample_fps, estimated_time, warnings[]
    """
    profiles_data = _load_profiles()
    overrides = _load_overrides()
    profiles = profiles_data.get("profiles", {})

    profile_key = _select_profile_key(hw)
    profile = profiles.get(profile_key, profiles["cpu_only_standard"])

    # Apply overrides if present
    whisper_model = overrides.get("whisper_model") or profile["whisper_model"]
    whisper_device = overrides.get("whisper_device") or profile["whisper_device"]
    whisper_compute = overrides.get("whisper_compute_type") or profile["whisper_compute_type"]
    preferred_llm = overrides.get("llm_model") or profile["llm_model"]
    frame_fps = overrides.get("frame_sample_fps") or profile["frame_sample_fps"]
    mediapipe_complexity = overrides.get("mediapipe_complexity", profile.get("mediapipe_complexity", 0))
    llm_timeout = profile.get("llm_call_timeout_seconds", 120)

    ollama_running = deps.get("ollama_running", False)
    ollama_models = deps.get("ollama_models", [])

    llm_model, llm_installed = _find_installed_llm(ollama_models, preferred_llm)
    llm_available = ollama_running  # available = Ollama is running; quality = installed

    # Determine if this is GPU-accelerated
    is_gpu = whisper_device in ("cuda", "rocm", "mps") and hw.get("cuda_available", False)

    time_est = _estimate_time(whisper_model, llm_model, llm_available, is_gpu, profiles_data)

    warnings = []

    if hw.get("ram_available_gb", 8) < 4:
        warnings.append(
            f"Low available RAM ({hw.get('ram_available_gb', '?')} GB). "
            "Close other applications before running analysis."
        )

    if not llm_installed and ollama_running:
        warnings.append(
            f"Recommended LLM '{llm_model}' is not installed. "
            f"Run: ollama pull {llm_model}"
        )

    if not ollama_running:
        warnings.append(
            "Ollama is not running. Semantic/story analysis (Stage 6) will be disabled. "
            "The blueprint will contain full measurement data but no narrative understanding."
        )

    if whisper_model in ("tiny", "base") and hw.get("ram_available_gb", 8) >= 6:
        warnings.append(
            f"Whisper '{whisper_model}' selected due to RAM constraints. "
            "Accuracy for Hinglish/mixed-language content will be reduced."
        )

    return {
        "profile_key": profile_key,
        "profile_description": profile.get("description", ""),
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "whisper_compute_type": whisper_compute,
        "llm_model": llm_model,
        "llm_available": llm_available,
        "llm_installed": llm_installed,
        "llm_to_pull": llm_model if not llm_installed else None,
        "llm_call_timeout_seconds": llm_timeout,
        "frame_sample_fps": frame_fps,
        "mediapipe_complexity": mediapipe_complexity,
        "is_gpu_accelerated": is_gpu,
        "estimated_time": time_est,
        "warnings": warnings,
    }
