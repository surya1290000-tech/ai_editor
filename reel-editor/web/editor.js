/**
 * web/editor.js
 * 
 * Interactive Multi-Track Timeline Studio (Phase 6)
 * Synchronized video player, multi-track canvas, property inspector,
 * AI explainability trace graph, and selective re-rendering.
 */

let projectData = null;
let currentTimeline = null;
let selectedEvent = null;
let selectedTrackType = null;
let isPlaying = false;
let timelineDuration = 90.05;
let pxPerSec = 12.0;

// DOM Elements
const video = document.getElementById("videoPreview");
const playhead = document.getElementById("playhead");
const timecodeDisplay = document.getElementById("timecodeDisplay");
const currentTimeText = document.getElementById("currentTimeText");
const totalDurationText = document.getElementById("totalDurationText");
const btnPlayPause = document.getElementById("btnPlayPause");
const playIcon = document.getElementById("playIcon");
const pauseIcon = document.getElementById("pauseIcon");
const btnStepBack = document.getElementById("btnStepBack");
const btnStepForward = document.getElementById("btnStepForward");
const btnUndo = document.getElementById("btnUndo");
const btnRedo = document.getElementById("btnRedo");
const versionBadge = document.getElementById("versionBadge");

const tracksViewport = document.getElementById("tracksViewport");
const rulerCanvas = document.getElementById("timelineRuler");
const laneText = document.getElementById("laneText");
const laneBroll = document.getElementById("laneBroll");
const laneVideo = document.getElementById("laneVideo");
const laneSfx = document.getElementById("laneSfx");
const laneVoice = document.getElementById("laneVoice");

// ── 1. Initialization ────────────────────────────────────────────────────────
async function initStudio() {
    try {
        await loadProjectData();
        setupPlaybackControls();
        setupTabs();
        setupDropdowns();
        setupInspector();
        setupRuler();
        window.addEventListener("resize", onResize);
    } catch (err) {
        console.error("Failed to initialize studio:", err);
        showToast("Error loading project: " + err.message);
    }
}

async function loadProjectData() {
    const res = await fetch("/api/editor/project");
    if (!res.ok) {
        throw new Error("Failed to fetch project timeline.");
    }
    projectData = await res.json();
    currentTimeline = projectData.timeline;
    timelineDuration = currentTimeline.timeline_duration || 90.05;

    // Update UI headers
    document.getElementById("projectName").innerText = currentTimeline.project_name || "Surya.mp4";
    versionBadge.innerText = projectData.manifest.current_version + (projectData.manifest.current_version === "v1" ? " (AI Baseline)" : " (User Modified)");
    totalDurationText.innerText = formatTimecode(timelineDuration);

    btnUndo.disabled = projectData.manifest.undo_stack.length === 0;
    btnRedo.disabled = projectData.manifest.redo_stack.length === 0;

    renderTimeline();
    renderVersionHistory();
}

function onResize() {
    renderTimeline();
}

// ── 2. Time Ruler & Layout ───────────────────────────────────────────────────
function setupRuler() {
    rulerCanvas.parentElement.addEventListener("click", (e) => {
        const rect = rulerCanvas.parentElement.getBoundingClientRect();
        const clickX = e.clientX - rect.left + tracksViewport.scrollLeft;
        const targetTime = Math.max(0, Math.min(timelineDuration, clickX / pxPerSec));
        video.currentTime = targetTime;
        updatePlayhead(targetTime);
    });
}

function drawRuler() {
    const ctx = rulerCanvas.getContext("2d");
    const width = tracksViewport.scrollWidth || 1200;
    const height = 24;

    rulerCanvas.width = width;
    rulerCanvas.height = height;

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#475569";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textBaseline = "top";

    const stepSec = pxPerSec >= 15 ? 1 : (pxPerSec >= 8 ? 5 : 10);
    const totalSecs = Math.ceil(timelineDuration);

    for (let s = 0; s <= totalSecs; s += stepSec) {
        const x = s * pxPerSec;
        ctx.beginPath();
        ctx.moveTo(x, height - 10);
        ctx.lineTo(x, height);
        ctx.stroke();

        if (s % (stepSec * 2) === 0 || stepSec >= 5) {
            ctx.fillText(`${s}s`, x + 4, 4);
        }
    }
}

// ── 3. Multi-Track Rendering ─────────────────────────────────────────────────
function renderTimeline() {
    const minWidth = Math.max(tracksViewport.clientWidth, timelineDuration * pxPerSec);
    pxPerSec = minWidth / timelineDuration;

    drawRuler();

    // Clear lanes
    laneText.innerHTML = "";
    laneBroll.innerHTML = "";
    laneVideo.innerHTML = "";
    laneSfx.innerHTML = "";
    laneVoice.innerHTML = "";

    // 1. Text Track
    const textTrack = currentTimeline.text_tracks?.[0];
    if (textTrack?.events) {
        textTrack.events.forEach(ev => {
            const block = createClipBlock({
                id: ev.event_id,
                title: ev.content || ev.style || "Text",
                tIn: ev.timeline_in,
                tOut: ev.timeline_out,
                cssClass: "clip-text",
                trackType: "TEXT",
                data: ev,
            });
            laneText.appendChild(block);
        });
    }

    // 2. Video Track (Base Clips + Punch-ins + B-Roll)
    const videoTrack = currentTimeline.video_tracks?.[0];
    if (videoTrack?.clips) {
        videoTrack.clips.forEach(clip => {
            const isPunch = (clip.punch_scale_factor || 1.0) > 1.05;
            const isBroll = clip.clip_type === "BROLL" || (clip.source_path && clip.source_path.includes("broll"));

            if (isBroll) {
                const brollBlock = createClipBlock({
                    id: clip.clip_id,
                    title: `B-Roll: ${clip.source_path ? clip.source_path.split(/[\\/]/).pop() : "Overlay"}`,
                    tIn: clip.timeline_in,
                    tOut: clip.timeline_out,
                    cssClass: "clip-broll",
                    trackType: "BROLL",
                    data: clip,
                });
                laneBroll.appendChild(brollBlock);
            } else {
                const videoBlock = createClipBlock({
                    id: clip.clip_id,
                    title: isPunch ? `Punch ${clip.punch_scale_factor.toFixed(2)}x` : `Clip (${(clip.timeline_out - clip.timeline_in).toFixed(1)}s)`,
                    tIn: clip.timeline_in,
                    tOut: clip.timeline_out,
                    cssClass: isPunch ? "clip-video clip-punch" : "clip-video",
                    trackType: "VIDEO",
                    data: clip,
                });
                laneVideo.appendChild(videoBlock);
            }
        });
    }

    // 3. Audio Tracks (Voice & SFX)
    if (currentTimeline.audio_tracks) {
        currentTimeline.audio_tracks.forEach(track => {
            const isSfx = track.role === "SFX" || track.name?.includes("SFX");
            const lane = isSfx ? laneSfx : laneVoice;
            const css = isSfx ? "clip-sfx" : "clip-voice";

            track.clips?.forEach(aClip => {
                const aBlock = createClipBlock({
                    id: aClip.clip_id,
                    title: isSfx ? `SFX: ${aClip.label || "Impact"} (${aClip.volume.toFixed(2)})` : "Master Voice",
                    tIn: aClip.timeline_in,
                    tOut: aClip.timeline_out || (aClip.timeline_in + (aClip.source_out - aClip.source_in)),
                    cssClass: css,
                    trackType: isSfx ? "SFX" : "VOICE",
                    data: aClip,
                });
                lane.appendChild(aBlock);
            });
        });
    }

    updatePlayhead(video.currentTime || 0);
}

function createClipBlock({ id, title, tIn, tOut, cssClass, trackType, data }) {
    const el = document.createElement("div");
    el.className = `timeline-clip ${cssClass}`;
    el.id = `clip_${id}`;

    const left = tIn * pxPerSec;
    const width = Math.max(12, (tOut - tIn) * pxPerSec);

    el.style.left = `${left}px`;
    el.style.width = `${width}px`;
    el.innerText = title;

    el.addEventListener("click", (e) => {
        e.stopPropagation();
        selectEvent(data, trackType, el);
    });

    return el;
}

// ── 4. Playback & Scrubbing ──────────────────────────────────────────────────
function setupPlaybackControls() {
    btnPlayPause.addEventListener("click", togglePlayPause);
    window.addEventListener("keydown", (e) => {
        if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
            e.preventDefault();
            togglePlayPause();
        }
    });

    btnStepBack.addEventListener("click", () => {
        video.currentTime = Math.max(0, video.currentTime - (1 / 30));
    });
    btnStepForward.addEventListener("click", () => {
        video.currentTime = Math.min(timelineDuration, video.currentTime + (1 / 30));
    });

    video.addEventListener("timeupdate", () => {
        updatePlayhead(video.currentTime);
    });

    video.addEventListener("ended", () => {
        isPlaying = false;
        playIcon.style.display = "block";
        pauseIcon.style.display = "none";
    });

    // History controls
    btnUndo.addEventListener("click", executeUndo);
    btnRedo.addEventListener("click", executeRedo);
}

function togglePlayPause() {
    if (video.paused) {
        video.play();
        isPlaying = true;
        playIcon.style.display = "none";
        pauseIcon.style.display = "block";
    } else {
        video.pause();
        isPlaying = false;
        playIcon.style.display = "block";
        pauseIcon.style.display = "none";
    }
}

function updatePlayhead(time) {
    const x = time * pxPerSec;
    playhead.style.transform = `translateX(${x}px)`;

    const frames = Math.floor((time % 1) * 30);
    const secs = Math.floor(time % 60);
    const mins = Math.floor((time / 60) % 60);
    const hrs = Math.floor(time / 3600);

    const tc = `${pad(hrs)}:${pad(mins)}:${pad(secs)}:${pad(frames)}`;
    timecodeDisplay.innerText = tc;
    currentTimeText.innerText = `${pad(mins)}:${pad(secs)}.${Math.floor((time % 1) * 10)}`;
}

function pad(num) {
    return String(num).padStart(2, "0");
}

function formatTimecode(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const dec = Math.floor((seconds % 1) * 10);
    return `${pad(mins)}:${pad(secs)}.${dec}`;
}

// ── 5. Selection & Inspector ─────────────────────────────────────────────────
function selectEvent(data, trackType, element) {
    selectedEvent = data;
    selectedTrackType = trackType;

    // Highlight selected block
    document.querySelectorAll(".timeline-clip").forEach(c => c.classList.remove("selected"));
    if (element) {
        element.classList.add("selected");
    }

    // Populate Inspector
    document.getElementById("inspectorEmpty").style.display = "none";
    const form = document.getElementById("inspectorForm");
    form.style.display = "block";

    document.getElementById("inspTrackType").innerText = trackType;
    document.getElementById("inspEventId").innerText = data.event_id || data.clip_id;
    document.getElementById("inspTimeIn").value = data.timeline_in?.toFixed(2) || 0;
    document.getElementById("inspTimeOut").value = (data.timeline_out || (data.timeline_in + 1.0)).toFixed(2);

    // Toggle track fields
    const fText = document.getElementById("fieldsText");
    const fVideo = document.getElementById("fieldsVideo");
    const fAudio = document.getElementById("fieldsAudio");

    fText.style.display = trackType === "TEXT" ? "block" : "none";
    fVideo.style.display = trackType === "VIDEO" || trackType === "BROLL" ? "block" : "none";
    fAudio.style.display = trackType === "SFX" || trackType === "VOICE" ? "block" : "none";

    if (trackType === "TEXT") {
        document.getElementById("inspTextContent").value = data.content || "";
        document.getElementById("inspFontFamily").value = data.font_family || "Montserrat";
        document.getElementById("inspFontSize").value = data.font_size || 84;
    } else if (trackType === "VIDEO") {
        const scale = data.punch_scale_factor || 1.0;
        document.getElementById("inspPunchScale").value = scale;
        document.getElementById("punchScaleVal").innerText = `${scale.toFixed(2)}x`;
        document.getElementById("inspAnchorX").value = data.face_anchor_x || 0.5;
        document.getElementById("inspAnchorY").value = data.face_anchor_y || 0.35;
    } else if (trackType === "SFX") {
        const vol = data.volume || 0.2;
        document.getElementById("inspAudioVolume").value = vol;
        document.getElementById("audioVolumeVal").innerText = `${vol.toFixed(2)}`;
    }

    // Load Traceability Graph
    loadTraceability(data.event_id || data.clip_id);
}

function setupInspector() {
    const form = document.getElementById("inspectorForm");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedEvent) return;

        const updates = {
            timeline_in: parseFloat(document.getElementById("inspTimeIn").value),
            timeline_out: parseFloat(document.getElementById("inspTimeOut").value),
        };

        if (selectedTrackType === "TEXT") {
            updates.content = document.getElementById("inspTextContent").value;
            updates.font_family = document.getElementById("inspFontFamily").value;
            updates.font_size = parseInt(document.getElementById("inspFontSize").value, 10);
        } else if (selectedTrackType === "VIDEO") {
            updates.punch_scale_factor = parseFloat(document.getElementById("inspPunchScale").value);
            updates.face_anchor_x = parseFloat(document.getElementById("inspAnchorX").value);
            updates.face_anchor_y = parseFloat(document.getElementById("inspAnchorY").value);
        } else if (selectedTrackType === "SFX") {
            updates.volume = parseFloat(document.getElementById("inspAudioVolume").value);
        }

        try {
            const res = await fetch("/api/editor/event/update", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    event_id: selectedEvent.event_id || selectedEvent.clip_id,
                    track_type: selectedTrackType,
                    updates: updates,
                    description: `User adjusted ${selectedTrackType} ${selectedEvent.event_id || selectedEvent.clip_id}`,
                })
            });
            if (!res.ok) throw new Error("Failed to save changes.");
            const data = await res.json();
            showToast(`Updated to ${data.version}! New version created.`);
            await loadProjectData();
        } catch (err) {
            showToast(`Error: ${err.message}`);
        }
    });

    document.getElementById("btnDeleteEvent").addEventListener("click", async () => {
        if (!selectedEvent) return;
        if (!confirm("Remove this clip from the timeline?")) return;

        try {
            const res = await fetch("/api/editor/event/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    event_id: selectedEvent.event_id || selectedEvent.clip_id,
                    track_type: selectedTrackType,
                    description: `User deleted ${selectedEvent.event_id || selectedEvent.clip_id}`,
                })
            });
            if (!res.ok) throw new Error("Failed to delete event.");
            const data = await res.json();
            showToast(`Deleted! Created ${data.version}`);
            await loadProjectData();
        } catch (err) {
            showToast(`Error: ${err.message}`);
        }
    });

    // Slider value sync
    document.getElementById("inspPunchScale").addEventListener("input", (e) => {
        document.getElementById("punchScaleVal").innerText = `${parseFloat(e.target.value).toFixed(2)}x`;
    });
    document.getElementById("inspAudioVolume").addEventListener("input", (e) => {
        document.getElementById("audioVolumeVal").innerText = `${parseFloat(e.target.value).toFixed(2)}`;
    });
}

// ── 6. Decision Traceability ─────────────────────────────────────────────────
async function loadTraceability(eventId) {
    const traceEmpty = document.getElementById("traceEmpty");
    const traceChain = document.getElementById("traceChain");

    traceEmpty.style.display = "block";
    traceChain.style.display = "none";
    traceEmpty.innerHTML = "<p>Loading explainability trace...</p>";

    try {
        const res = await fetch(`/api/editor/trace/${eventId}`);
        if (!res.ok) throw new Error("Failed to fetch trace.");
        const trace = await res.json();

        traceEmpty.style.display = "none";
        traceChain.style.display = "flex";

        // 1. Timeline Event
        document.getElementById("traceEventBody").innerHTML = `
            <strong>${trace.timeline_event?.style || trace.timeline_event?.operation || trace.target_id}</strong><br>
            Interval: ${trace.timeline_event?.timeline_in?.toFixed(2) || 0}s – ${trace.timeline_event?.timeline_out?.toFixed(2) || 0}s
        `;

        // 2. Creative Decision (Agent 3)
        const cd = trace.creative_decision;
        document.getElementById("traceCreativeBody").innerHTML = cd ? `
            <strong>Style Profile: ${cd.style_profile || 'Default'}</strong><br>
            <em>${cd.rationale || 'Applied typography and visual parameters.'}</em>
        ` : '<em>Inherited baseline aesthetic rules.</em>';

        // 3. Story Decision (Agent 2)
        const sd = trace.story_decision;
        document.getElementById("traceStoryBody").innerHTML = sd ? `
            <strong>Operation: ${sd.operation} (${sd.verdict || 'ACCEPTED'})</strong><br>
            Story Purpose: ${sd.story_purpose || 'Structural pacing reinforcement.'}<br>
            Confidence: ${(sd.confidence * 100).toFixed(0)}%
        ` : '<em>Direct structural clip baseline.</em>';

        // 4. Opportunity (Agent 1)
        const opp = trace.opportunity;
        document.getElementById("traceOpportunityBody").innerHTML = opp ? `
            <strong>Opportunity: ${opp.type || opp.opportunity_id}</strong><br>
            Suggested Treatment: ${opp.suggested_treatment || 'Visual highlight'}
        ` : '<em>Story continuity segment.</em>';

        // 5. Evidence & Transcript
        const evList = trace.evidence || [];
        const evHtml = evList.map(ev => `
            <div>• [${ev.evidence_type}] ${ev.description || ev.evidence_id} (conf: ${ev.confidence})</div>
        `).join("");

        const transcript = trace.transcript_context?.text ? `
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.1); color: var(--accent-gold);">
                🗣️ "${trace.transcript_context.text}"
            </div>
        ` : "";

        document.getElementById("traceEvidenceBody").innerHTML = (evHtml || "<em>Evidence grounded in multimodal camera tracking.</em>") + transcript;

    } catch (err) {
        traceEmpty.innerHTML = `<p style="color: var(--accent-rose)">Trace unavailable: ${err.message}</p>`;
    }
}

// ── 7. Version History & Rollback ────────────────────────────────────────────
function renderVersionHistory() {
    const list = document.getElementById("versionHistoryList");
    list.innerHTML = "";

    const manifest = projectData.manifest;
    manifest.versions.forEach(v => {
        const item = document.createElement("div");
        item.className = `version-item ${v.version === manifest.current_version ? "active" : ""}`;
        item.innerHTML = `
            <div class="ver-info">
                <h4>${v.version} ${v.is_immutable ? "★ (Immutable AI Baseline)" : ""}</h4>
                <p>${v.description}</p>
                <small style="color: var(--text-muted); font-size: 0.7rem">${new Date(v.created_at).toLocaleTimeString()}</small>
            </div>
            <div>
                ${v.version !== manifest.current_version ? `
                    <button class="btn btn-secondary" onclick="rollbackTo('${v.version}')">Rollback</button>
                ` : `<span style="color: var(--accent-emerald); font-size: 0.75rem; font-weight: 700;">CURRENT</span>`}
            </div>
        `;
        list.appendChild(item);
    });
}

async function rollbackTo(ver) {
    if (!confirm(`Rollback active timeline to ${ver}?`)) return;
    try {
        const res = await fetch("/api/editor/version/rollback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ version: ver })
        });
        if (!res.ok) throw new Error("Rollback failed.");
        showToast(`Rolled back to ${ver}`);
        await loadProjectData();
    } catch (err) {
        showToast(`Error: ${err.message}`);
    }
}

async function executeUndo() {
    try {
        const res = await fetch("/api/editor/undo", { method: "POST" });
        if (res.ok) {
            showToast("Undo successful");
            await loadProjectData();
        }
    } catch (err) {
        showToast(`Undo failed: ${err.message}`);
    }
}

async function executeRedo() {
    try {
        const res = await fetch("/api/editor/redo", { method: "POST" });
        if (res.ok) {
            showToast("Redo successful");
            await loadProjectData();
        }
    } catch (err) {
        showToast(`Redo failed: ${err.message}`);
    }
}

// ── 8. Selective Re-Rendering ────────────────────────────────────────────────
function setupDropdowns() {
    document.querySelectorAll(".dropdown-toggle").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const parent = btn.closest(".dropdown");
            document.querySelectorAll(".dropdown").forEach(d => {
                if (d !== parent) d.classList.remove("active");
            });
            parent.classList.toggle("active");
        });
    });

    window.addEventListener("click", () => {
        document.querySelectorAll(".dropdown").forEach(d => d.classList.remove("active"));
    });

    document.getElementById("btnRerenderGraphics").addEventListener("click", () => triggerRerender("GRAPHICS_ONLY"));
    document.getElementById("btnRerenderAudio").addEventListener("click", () => triggerRerender("AUDIO_ONLY"));
    document.getElementById("btnRerenderFull").addEventListener("click", () => triggerRerender("FULL"));
}

async function triggerRerender(mode) {
    const spinner = document.getElementById("renderSpinner");
    spinner.style.display = "inline-block";
    showToast(`Initiating selective re-render (${mode})...`);

    try {
        const res = await fetch("/api/editor/rerender", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: mode })
        });
        if (!res.ok) throw new Error("Re-render failed.");
        const data = await res.json();
        showToast(`Render complete in ${data.render_time_sec || 0}s!`);

        // Reload video cache-busted
        const curTime = video.currentTime;
        video.src = `/api/workflow/download-video?t=${Date.now()}`;
        video.currentTime = curTime;
    } catch (err) {
        showToast(`Re-render error: ${err.message}`);
    } finally {
        spinner.style.display = "none";
    }
}

// ── Tabs & Toast ─────────────────────────────────────────────────────────────
function setupTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(`tab-${tabId}`).classList.add("active");
        });
    });
}

function showToast(msg) {
    const toast = document.getElementById("toast");
    toast.innerText = msg;
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3500);
}

// Bootstrap
window.addEventListener("DOMContentLoaded", initStudio);
