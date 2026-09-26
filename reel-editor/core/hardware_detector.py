"""
core/hardware_detector.py

Detects available hardware (CPU, RAM, GPU, storage, CUDA).
Returns a structured dict consumed by model_selector.py and diagnose.py.

Design principles:
- Every detection is wrapped in try/except. A failed detection returns None, not a crash.
- On Windows, GPU detection uses PowerShell + WMI (more reliable than wmic on Win 11).
- CUDA is detected via nvcc (available if NVIDIA drivers installed with SDK),
  not via torch (we don't want to import torch just for diagnostics).
- All values are in human-friendly units (GB not bytes, etc.).
"""

from __future__ import annotations

import os
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _get_cpu_name() -> str:
    """Get detailed CPU brand name."""
    try:
        import cpuinfo  # py-cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", platform.processor() or "Unknown CPU")
    except Exception:
        return platform.processor() or "Unknown CPU"


def _get_ram() -> dict:
    """Get total and available system RAM in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "used_percent": mem.percent,
        }
    except Exception:
        return {"total_gb": 0.0, "available_gb": 0.0, "used_percent": 0.0}


def _get_cpu_cores() -> dict:
    """Get logical and physical core counts."""
    try:
        import psutil
        return {
            "logical": psutil.cpu_count(logical=True) or 1,
            "physical": psutil.cpu_count(logical=False) or 1,
        }
    except Exception:
        return {"logical": os.cpu_count() or 1, "physical": os.cpu_count() or 1}


def _get_gpu_windows() -> tuple[Optional[str], Optional[float]]:
    """
    Detect GPU name and VRAM on Windows using PowerShell + WMI.
    Returns (gpu_name, vram_gb). Both can be None on failure.

    We query Win32_VideoController and skip 'Microsoft Basic Display Adapter'
    which is the virtual fallback adapter.
    """
    ps_script = (
        "Get-WmiObject Win32_VideoController "
        "| Where-Object { $_.Name -notlike '*Microsoft Basic*' } "
        "| Select-Object -First 1 -Property Name, AdapterRAM "
        "| ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None, None

        data = json.loads(result.stdout.strip())
        # PowerShell may return a single object or a list
        if isinstance(data, list):
            data = data[0] if data else {}

        gpu_name = (data.get("Name") or "").strip() or None
        adapter_ram = data.get("AdapterRAM") or 0
        # AdapterRAM is in bytes; 0 or negative = shared/unknown
        vram_gb = round(adapter_ram / (1024 ** 3), 1) if adapter_ram > 0 else None

        return gpu_name, vram_gb
    except Exception:
        return None, None


def _get_gpu_linux() -> tuple[Optional[str], Optional[float]]:
    """Detect GPU on Linux via lspci or nvidia-smi."""
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip()) if len(parts) > 1 else 0
            return name, round(vram_mb / 1024, 1)
    except Exception:
        pass

    # Fallback: lspci
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "VGA" in line or "3D" in line or "Display" in line:
                return line.split(":", 2)[-1].strip(), None
    except Exception:
        pass

    return None, None


def _check_cuda() -> bool:
    """Check if CUDA (NVIDIA) is available via nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_free_storage() -> Optional[float]:
    """Get free disk space on the drive where the project lives (GB)."""
    try:
        import psutil
        # Use the drive of the current working directory
        disk = psutil.disk_usage(Path.cwd().anchor)
        return round(disk.free / (1024 ** 3), 1)
    except Exception:
        return None


def detect_hardware() -> dict:
    """
    Detect all relevant hardware properties.

    Returns a dict with:
      os, python_version, cpu, cpu_cores_logical, cpu_cores_physical,
      ram_total_gb, ram_available_gb, gpu_name, vram_gb,
      cuda_available, free_storage_gb
    """
    sys_name = platform.system()

    # CPU
    cpu_name = _get_cpu_name()
    cores = _get_cpu_cores()

    # RAM
    ram = _get_ram()

    # GPU
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    if sys_name == "Windows":
        gpu_name, vram_gb = _get_gpu_windows()
    elif sys_name == "Linux":
        gpu_name, vram_gb = _get_gpu_linux()
    elif sys_name == "Darwin":
        # macOS — Apple Silicon or AMD. For now, report name only.
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            gpus = data.get("SPDisplaysDataType", [{}])
            gpu_name = gpus[0].get("sppci_model") if gpus else None
        except Exception:
            pass

    cuda = _check_cuda()
    storage = _get_free_storage()

    # Classify integrated vs discrete
    is_integrated = False
    if gpu_name:
        integrated_keywords = ["iris", "uhd", "hd graphics", "vega", "radeon vega",
                               "apple m", "llvm", "llvmpipe", "swiftshader"]
        is_integrated = any(kw in gpu_name.lower() for kw in integrated_keywords)

    return {
        "os": f"{sys_name} {platform.release()}",
        "python_version": platform.python_version(),
        "cpu": cpu_name,
        "cpu_cores_logical": cores["logical"],
        "cpu_cores_physical": cores["physical"],
        "ram_total_gb": ram["total_gb"],
        "ram_available_gb": ram["available_gb"],
        "ram_used_percent": ram["used_percent"],
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "gpu_is_integrated": is_integrated,
        "cuda_available": cuda,
        "free_storage_gb": storage,
    }
