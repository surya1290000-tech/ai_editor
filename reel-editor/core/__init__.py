# core package
import os
import shutil
import platform
import glob

def ensure_ffmpeg_on_path():
    """Ensure ffmpeg and ffprobe are discoverable on PATH (especially on Windows after winget install)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    if platform.system() == "Windows":
        candidates = []
        # Check WinGet packages
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.extend(glob.glob(f"{local_app}/Microsoft/WinGet/Packages/**/bin", recursive=True))

        # Check Desktop folders
        desktop = os.path.expanduser("~/Desktop")
        if os.path.exists(desktop):
            candidates.extend(glob.glob(f"{desktop}/**/bin", recursive=True))

        # Check registry PATH if not already in os.environ
        try:
            import winreg
            for hkey, subkey in [
                (winreg.HKEY_CURRENT_USER, r"Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            ]:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        candidates.extend(val.split(";"))
                except Exception:
                    pass
        except Exception:
            pass

        for c in candidates:
            c_str = str(c).strip()
            if c_str and os.path.exists(c_str):
                ffmpeg_exe = os.path.join(c_str, "ffmpeg.exe")
                if os.path.isfile(ffmpeg_exe):
                    os.environ["PATH"] = c_str + os.pathsep + os.environ.get("PATH", "")
                    break

ensure_ffmpeg_on_path()
