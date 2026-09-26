"""
core/timeline/broll.py

B-Roll Asset Resolver Interface
───────────────────────────────
Defines the contract: BrollRequest -> AssetResolver -> BrollAsset -> TimelineTrack.

If no verified local asset is resolved, marks the opportunity as BROLL_UNRESOLVED
and falls back gracefully to talking-head footage without inserting fake media.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from core.timeline.models import BrollTrack, BrollClip, CropRect
from core.timeline.remapper import TimecodeRemapper
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BrollRequest:
    request_id: str
    origin_opportunity_id: str
    concept: str
    search_query: str
    source_start: float
    source_end: float
    duration: float
    confidence: float
    reason: str


@dataclass
class BrollAsset:
    asset_id: str
    file_path: Path
    duration: float
    width: int
    height: int
    license: str = "local"


class BrollAssetResolver:
    """
    Resolves B-roll requests against local library, user uploads, or stock sources.
    """

    def __init__(self, local_asset_dir: Optional[Path] = None):
        self.local_asset_dir = local_asset_dir

    def resolve(self, request: BrollRequest) -> Optional[BrollAsset]:
        """
        Attempts to locate a matching B-roll asset on disk.
        Returns None if no matching asset exists.
        """
        if not self.local_asset_dir or not self.local_asset_dir.exists():
            return None

        # Search for file matching concept in local library
        concept_slug = request.concept.lower().replace(" ", "_")
        candidates = list(self.local_asset_dir.glob(f"*{concept_slug}*.mp4"))
        if candidates and candidates[0].exists():
            p = candidates[0]
            return BrollAsset(
                asset_id=f"asset_{concept_slug}",
                file_path=p,
                duration=request.duration,
                width=1080,
                height=1920,
            )

        return None

    def build_broll_track(
        self,
        broll_requests: List[Dict[str, Any]],
        remapper: TimecodeRemapper,
    ) -> tuple[BrollTrack, List[Dict[str, Any]]]:
        """
        Builds the B-Roll timeline track. If an asset cannot be resolved,
        records it as BROLL_UNRESOLVED in the audit report.
        """
        track_clips: List[BrollClip] = []
        unresolved_reports: List[Dict[str, Any]] = []

        for req_dict in broll_requests:
            req = BrollRequest(
                request_id=req_dict.get("decision_id", "req_broll"),
                origin_opportunity_id=req_dict.get("origin_opportunity_id", "opp_broll"),
                concept=req_dict.get("concept", "general"),
                search_query=req_dict.get("search_query", ""),
                source_start=float(req_dict.get("source_start", 0.0)),
                source_end=float(req_dict.get("source_end", 3.0)),
                duration=float(req_dict.get("duration", 3.0)),
                confidence=float(req_dict.get("confidence", 0.8)),
                reason=req_dict.get("reason", "Visual cutaway"),
            )

            t_interval = remapper.map_interval_to_timeline(req.source_start, req.source_end)
            if not t_interval:
                continue

            asset = self.resolve(req)

            if asset:
                track_clips.append(BrollClip(
                    clip_id=f"broll_{req.request_id}",
                    concept=req.concept,
                    search_query=req.search_query,
                    timeline_in=t_interval[0],
                    timeline_out=t_interval[1],
                    source_reference=str(asset.file_path),
                    source_in=0.0,
                    source_out=round(t_interval[1] - t_interval[0], 3),
                    status="RESOLVED",
                    origin_opportunity_id=req.origin_opportunity_id,
                    reason=req.reason,
                    confidence=req.confidence,
                ))
            else:
                # Graceful fallback: register placeholder without inserting fake media
                track_clips.append(BrollClip(
                    clip_id=f"broll_unresolved_{req.request_id}",
                    concept=req.concept,
                    search_query=req.search_query,
                    timeline_in=t_interval[0],
                    timeline_out=t_interval[1],
                    source_reference=None,
                    status="BROLL_UNRESOLVED",
                    origin_opportunity_id=req.origin_opportunity_id,
                    reason=f"{req.reason} (No local asset matching '{req.concept}' — talking head preserved)",
                    confidence=req.confidence,
                ))
                unresolved_reports.append({
                    "item_type": "B_ROLL",
                    "concept": req.concept,
                    "search_query": req.search_query,
                    "timeline_interval": [t_interval[0], t_interval[1]],
                    "status": "BROLL_UNRESOLVED",
                    "action_taken": "Preserved primary talking-head shot without interruption",
                })

        return BrollTrack(track_id="track_broll_v3", name="B-Roll Overlay Track", clips=track_clips), unresolved_reports
