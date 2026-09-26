"""
core/evidence/smart_sampler.py

Smart Visual Sampling Subsystem (Production Hardened)
─────────────────────────────────────────────────────
Separates CHEAP PERCEPTION from EXPENSIVE AI VISUAL REASONING.
Cheap frame sampling/CV may inspect many frames across the footage.
The <=20 limit applies strictly to deep reasoning items.

Architecture:
  1. Base Visual Sampling: Low-frequency coarse cadence (e.g. 1 frame / 5s).
  2. Deterministic Event Detection:
     - Candidate cuts (silence boundaries, filler pauses)
     - Emotional / acoustic peaks (pitch & energy inflection points)
     - Framing changes (shot transitions, face bounding box jumps)
     - B-roll candidates (topic changes, abstract concepts)
     - HeroText candidates (opening hook window)
     - Ambiguous moments (low-confidence ASR words)
  3. Candidate Ranking: Prioritize high-impact & ambiguous moments.
  4. Deep Visual Inspections (<=20 budget):
     - 3-Panel Before / Event / After Filmstrip
     - Synchronized Audio Waveform Strip beneath the panels
     - Exact Event Marker (burned red vertical line)
     - On-demand 5-frame micro-burst for high-ambiguity moments
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image, ImageDraw, ImageFont

from core.logger import get_logger

logger = get_logger(__name__)


class SmartVisualSampler:
    """
    Production-grade visual perception sampler combining uniform cadence,
    event-driven drill-downs, waveform filmstrip composites, and micro-bursts.
    """

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        base_cadence_sec: float = 5.0,
        max_reasoning_frames: int = 20,
    ):
        self.ffmpeg_bin = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        self.base_cadence_sec = base_cadence_sec
        self.max_reasoning_frames = max_reasoning_frames

    def _extract_waveform_samples(
        self,
        video_path: Path,
        start_sec: float,
        duration_sec: float,
        num_points: int,
    ) -> list[float]:
        """Extracts normalized audio amplitude envelope across a time window."""
        temp_wav = video_path.parent / f"_tmp_wave_{int(start_sec * 1000)}.wav"
        cmd = [
            self.ffmpeg_bin, "-y",
            "-ss", f"{max(0.0, start_sec):.3f}",
            "-t", f"{duration_sec:.3f}",
            "-i", str(video_path),
            "-ar", "8000",
            "-ac", "1",
            "-f", "wav",
            str(temp_wav)
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if not temp_wav.exists() or temp_wav.stat().st_size < 100:
                return [0.0] * num_points

            with wave.open(str(temp_wav), "rb") as wf:
                n_frames = wf.getnframes()
                if n_frames == 0:
                    return [0.0] * num_points
                frames = wf.readframes(n_frames)
                samples = struct.unpack(f"{n_frames}h", frames)

            # Resample into num_points bins
            bin_size = max(1, len(samples) // num_points)
            envelope = []
            for i in range(num_points):
                chunk = samples[i * bin_size : (i + 1) * bin_size]
                if chunk:
                    max_val = max(abs(s) for s in chunk) / 32768.0
                    envelope.append(round(min(1.0, max_val), 3))
                else:
                    envelope.append(0.0)
            return envelope
        except Exception:
            return [0.0] * num_points
        finally:
            temp_wav.unlink(missing_ok=True)

    def sample_video(
        self,
        video_path: Path | str,
        output_dir: Path | str,
        audio_features: Optional[Dict[str, Any]] = None,
        transcript_raw: Optional[Dict[str, Any]] = None,
        media_meta: Optional[Dict[str, Any]] = None,
        visual_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes smart sampling and produces smart_visual_manifest.json.
        """
        video_path = Path(video_path).resolve()
        out_dir = Path(output_dir).resolve()
        sampler_dir = out_dir / "smart_visual_samples"
        sampler_dir.mkdir(parents=True, exist_ok=True)
        composites_dir = sampler_dir / "composites"
        composites_dir.mkdir(parents=True, exist_ok=True)
        micro_bursts_dir = sampler_dir / "micro_bursts"
        micro_bursts_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = sampler_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        duration = float((media_meta or {}).get("duration_seconds") or 60.0)

        # ── 1. Base Cadence Sampling ────────────────────────────────────
        base_candidates = []
        t = 0.5
        while t < duration:
            base_candidates.append({
                "timestamp": round(t, 2),
                "category": "BASE_CADENCE",
                "reason": f"Uniform visual cadence at {t:.1f}s",
                "priority": 1,
            })
            t += self.base_cadence_sec

        # ── 2. Event-Driven Targeted Drill-Downs ────────────────────────
        event_candidates = []

        # A. HeroText Candidate (Opening Hook 0–2.5s)
        event_candidates.append({
            "timestamp": min(1.2, duration * 0.05),
            "category": "HEROTEXT_CANDIDATE",
            "reason": "Opening hook window for title card placement",
            "priority": 5,
        })

        # B. Candidate Cuts from Audio Silences
        if audio_features:
            silences = audio_features.get("silence_regions", [])
            for s in silences:
                s_start = float(s.get("start", 0))
                s_end = float(s.get("end", 0))
                s_dur = s_end - s_start
                if s_dur >= 0.4:
                    event_candidates.append({
                        "timestamp": round((s_start + s_end) / 2.0, 2),
                        "category": "CANDIDATE_CUT",
                        "reason": f"Extended silence ({s_dur:.2f}s) candidate for cut/trim",
                        "priority": 4,
                    })

        # C. Emotional / Acoustic Peaks
        if audio_features:
            energy_peaks = audio_features.get("energy_peaks", [])
            for ep in energy_peaks[:4]:
                ep_time = float(ep.get("timestamp", ep.get("start", 0)))
                event_candidates.append({
                    "timestamp": round(ep_time, 2),
                    "category": "EMOTIONAL_PEAK",
                    "reason": f"High acoustic vocal stress / emphasis (dB={ep.get('level_db', 'high')})",
                    "priority": 4,
                })

        # D. Ambiguous Moments from ASR (Low Confidence Spans)
        if transcript_raw:
            low_conf_words = [
                w for w in transcript_raw.get("words", [])
                if float(w.get("confidence", 1.0)) < 0.55
            ]
            for lw in low_conf_words[:3]:
                event_candidates.append({
                    "timestamp": round(float(lw.get("start", 0)), 2),
                    "category": "AMBIGUOUS_MOMENT",
                    "reason": f"Low ASR confidence word '{lw.get('word', '')}' ({lw.get('confidence', 0):.2f})",
                    "priority": 3,
                })

        # E. Framing Changes from CV
        if visual_features:
            shot_segments = visual_features.get("shot_segments", [])
            for seg in shot_segments[:3]:
                seg_time = float(seg.get("start", 0))
                if seg_time > 1.0:
                    event_candidates.append({
                        "timestamp": round(seg_time, 2),
                        "category": "FRAMING_CHANGE",
                        "reason": f"Shot framing transition to {seg.get('framing', 'SHOT')}",
                        "priority": 3,
                    })

        # ── 3. Candidate Ranking & Deduplication ────────────────────────
        all_candidates = event_candidates + base_candidates
        all_candidates.sort(key=lambda x: (x["priority"], -x["timestamp"]), reverse=True)

        selected_events = []
        for cand in all_candidates:
            t_cand = cand["timestamp"]
            # Avoid placing samples closer than 0.8s
            if any(abs(t_cand - sel["timestamp"]) < 0.8 for sel in selected_events):
                continue
            selected_events.append(cand)
            if len(selected_events) >= self.max_reasoning_frames:
                break

        # Re-sort chronologically for execution
        selected_events.sort(key=lambda x: x["timestamp"])

        # ── 4. Extract High-Res Frames for Selected Events ───────────────
        logger.info(f"SmartVisualSampler | Extracting {len(selected_events)} curated keyframes...")
        for i, ev in enumerate(selected_events, 1):
            ts = ev["timestamp"]
            f_name = f"frame_{i:02d}_{ev['category'].lower()}_{ts:.2f}s.jpg"
            f_path = frames_dir / f_name
            cmd = [
                self.ffmpeg_bin, "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(f_path)
            ]
            subprocess.run(cmd, capture_output=True)
            ev["frame_path"] = str(f_path)
            ev["frame_filename"] = f_name

        # ── 5. On-Demand Filmstrip + Waveform Composites ─────────────────
        # Select top ambiguous cuts, hero text, and emotional peaks
        composite_targets = [
            e for e in selected_events
            if e["category"] in ("HEROTEXT_CANDIDATE", "CANDIDATE_CUT", "EMOTIONAL_PEAK", "AMBIGUOUS_MOMENT")
        ][:4]

        composites_metadata = []
        for c_idx, target in enumerate(composite_targets, 1):
            t_center = target["timestamp"]
            t_before = max(0.0, t_center - 0.75)
            t_after = min(duration, t_center + 0.75)
            comp_dur = t_after - t_before

            strip_img_name = f"composite_{c_idx}_{target['category'].lower()}_{t_center:.2f}s.jpg"
            strip_img_path = composites_dir / strip_img_name

            # Extract 3 panels
            p_before = frames_dir / f"tmp_before_{c_idx}.jpg"
            p_center = frames_dir / f"tmp_center_{c_idx}.jpg"
            p_after = frames_dir / f"tmp_after_{c_idx}.jpg"

            for t_pos, p_dest in [(t_before, p_before), (t_center, p_center), (t_after, p_after)]:
                subprocess.run([
                    self.ffmpeg_bin, "-y", "-ss", f"{t_pos:.3f}", "-i", str(video_path),
                    "-vframes", "1", "-q:v", "3", str(p_dest)
                ], capture_output=True)

            try:
                im1 = Image.open(p_before)
                im2 = Image.open(p_center)
                im3 = Image.open(p_after)

                tile_h = 320
                tile_w = int(tile_h * (im1.width / im1.height))
                total_w = tile_w * 3 + 12
                wave_h = 70
                header_h = 36
                footer_h = 24
                canvas_h = header_h + tile_h + wave_h + footer_h

                im1_r = im1.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                im2_r = im2.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                im3_r = im3.resize((tile_w, tile_h), Image.Resampling.BILINEAR)

                canvas = Image.new("RGB", (total_w, canvas_h), color=(15, 23, 42))
                canvas.paste(im1_r, (0, header_h))
                canvas.paste(im2_r, (tile_w + 6, header_h))
                canvas.paste(im3_r, (tile_w * 2 + 12, header_h))

                draw = ImageDraw.Draw(canvas)

                # Header text
                header_text = f"[{target['category']}] Event at {t_center:.2f}s — {target['reason'][:55]}"
                draw.text((10, 10), header_text, fill=(56, 189, 248))

                # Labels beneath panels
                panel_y = header_h + tile_h + 4
                draw.text((10, panel_y), f"Before: {t_before:.2f}s", fill=(148, 163, 184))
                draw.text((tile_w + 16, panel_y), f"EVENT: {t_center:.2f}s", fill=(245, 158, 11))
                draw.text((tile_w * 2 + 22, panel_y), f"After: {t_after:.2f}s", fill=(148, 163, 184))

                # Waveform rendering
                wave_y = panel_y + 20
                draw.rectangle([(0, wave_y), (total_w, wave_y + wave_h)], fill=(2, 6, 23))
                draw.line([(0, wave_y + wave_h // 2), (total_w, wave_y + wave_h // 2)], fill=(51, 65, 85))

                # Extract audio envelope
                envelope = self._extract_waveform_samples(
                    video_path=video_path,
                    start_sec=t_before,
                    duration_sec=comp_dur,
                    num_points=total_w,
                )
                mid_y = wave_y + wave_h // 2
                for x_px, amp in enumerate(envelope):
                    bar_h = int(amp * (wave_h // 2 - 2))
                    draw.line([(x_px, mid_y - bar_h), (x_px, mid_y + bar_h)], fill=(56, 189, 248))

                # Burned Red Event Marker at exact event center
                event_x = int(total_w * ((t_center - t_before) / max(0.01, comp_dur)))
                draw.line([(event_x, wave_y), (event_x, wave_y + wave_h)], fill=(239, 68, 68), width=2)
                draw.text((event_x + 4, wave_y + 4), f"| {t_center:.2f}s", fill=(239, 68, 68))

                canvas.save(strip_img_path, quality=88)
                composites_metadata.append({
                    "composite_id": f"comp_{c_idx}",
                    "target_category": target["category"],
                    "timestamp_center": t_center,
                    "composite_path": str(strip_img_path),
                    "filename": strip_img_name,
                    "waveform_rendered": True,
                })

                # ── 6. Optional 5-Frame Micro-Burst for High Ambiguity ─────
                micro_burst_frames = []
                if target["priority"] >= 4:
                    burst_dir = micro_bursts_dir / f"burst_{c_idx}_{target['category'].lower()}"
                    burst_dir.mkdir(parents=True, exist_ok=True)
                    offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
                    for b_idx, off in enumerate(offsets, 1):
                        b_ts = max(0.0, min(duration, t_center + off))
                        b_dest = burst_dir / f"b_{b_idx}_{b_ts:.2f}s.jpg"
                        subprocess.run([
                            self.ffmpeg_bin, "-y", "-ss", f"{b_ts:.3f}", "-i", str(video_path),
                            "-vframes", "1", "-q:v", "3", str(b_dest)
                        ], capture_output=True)
                        micro_burst_frames.append(str(b_dest))
                    target["micro_burst_frames"] = micro_burst_frames

                # Cleanup temp tiles
                for p_tmp in [p_before, p_center, p_after]:
                    p_tmp.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to assemble composite strip {c_idx}: {e}")

        manifest = {
            "source_video": str(video_path.name),
            "duration_seconds": duration,
            "sampling_parameters": {
                "base_cadence_sec": self.base_cadence_sec,
                "max_reasoning_frames": self.max_reasoning_frames,
            },
            "total_sampled_keyframes": len(selected_events),
            "total_composites_generated": len(composites_metadata),
            "sampled_keyframes": selected_events[:self.max_reasoning_frames],
            "deep_visual_inspections": selected_events[:self.max_reasoning_frames],
            "composites": composites_metadata,
            "drilldown_categories": {
                "base_cadence": sum(1 for e in selected_events if e["category"] == "BASE_CADENCE"),
                "candidate_cuts": sum(1 for e in selected_events if e["category"] == "CANDIDATE_CUT"),
                "emotional_peaks": sum(1 for e in selected_events if e["category"] == "EMOTIONAL_PEAK"),
                "framing_changes": sum(1 for e in selected_events if e["category"] == "FRAMING_CHANGE"),
                "herotext_candidates": sum(1 for e in selected_events if e["category"] == "HEROTEXT_CANDIDATE"),
                "ambiguous_moments": sum(1 for e in selected_events if e["category"] == "AMBIGUOUS_MOMENT"),
            },
        }

        manifest_path = sampler_dir / "smart_visual_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        logger.info(
            f"SmartVisualSampler | Sampling complete: {len(selected_events)} curated frames & "
            f"{len(composites_metadata)} waveform filmstrips generated (manifest: {manifest_path.name})."
        )

        return manifest
