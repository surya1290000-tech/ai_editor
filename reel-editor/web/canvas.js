/* ═══════════════════════════════════════════════════════════
   AI Reel Editor — Visual Workflow Canvas Engine
   
   Features:
   • Fetches graph topology from GET /api/workflow/graph
   • Renders nodes at layout positions with status-driven styles
   • Draws cubic bezier edges on an SVG overlay
   • Supports pan (drag canvas) & zoom (scroll wheel)
   • Connects to SSE /api/workflow/events for real-time updates
   • Inspector panel with ports, metrics, logs, artifacts
   • Run/Cancel workflow controls
   • Minimap
   ═══════════════════════════════════════════════════════════ */

(() => {
"use strict";

// ─── DOM refs ───────────────────────────────────────────────
const canvasContainer = document.getElementById("canvas-container");
const edgeLayer       = document.getElementById("edge-layer");
const nodeLayer       = document.getElementById("node-layer");
const zoomDisplay     = document.getElementById("zoom-display");
const statusPill      = document.getElementById("workflow-status");
const statusText      = statusPill.querySelector(".status-text");
const btnRun          = document.getElementById("btn-run");
const btnCancel       = document.getElementById("btn-cancel");
const btnFit          = document.getElementById("btn-fit");
const inspector       = document.getElementById("inspector");
const inspectorIcon   = document.getElementById("inspector-icon");
const inspectorTitle  = document.getElementById("inspector-title");
const inspectorBody   = document.getElementById("inspector-body");
const inspectorClose  = document.getElementById("inspector-close");
const runOverlay      = document.getElementById("run-dialog-overlay");
const runDialog       = document.getElementById("run-dialog");
const videoPathInput  = document.getElementById("video-path-input");
const dialogCancel    = document.getElementById("run-dialog-cancel");
const dialogConfirm   = document.getElementById("run-dialog-confirm");
const pipelineName    = document.getElementById("pipeline-name");
const minimapCanvas   = document.getElementById("minimap-canvas");

// ─── State ──────────────────────────────────────────────────
let graph = null;         // full graph object from API
let transform = { x: 0, y: 0, scale: 1 };
let selectedNodeId = null;
let isPanning = false;
let panStart = { x: 0, y: 0 };
let eventSource = null;
let nodeElements = {};    // node_id → DOM element

// ─── Constants ──────────────────────────────────────────────
const NODE_WIDTH  = 220;
const MIN_ZOOM    = 0.15;
const MAX_ZOOM    = 2.5;
const PAN_MARGIN  = 80;


// ═══════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════

async function init() {
    await fetchGraph();
    if (graph) {
        renderGraph();
        fitToView();
        connectSSE();
    }
    bindEvents();
}

async function fetchGraph() {
    try {
        const res = await fetch("/api/workflow/graph");
        graph = await res.json();
        if (pipelineName) pipelineName.textContent = graph.name || "Workflow";
        updateWorkflowStatus(graph.status);
    } catch (e) {
        console.error("Failed to fetch graph:", e);
    }
}


// ═══════════════════════════════════════════════════════════
// GRAPH RENDERING
// ═══════════════════════════════════════════════════════════

function renderGraph() {
    nodeLayer.innerHTML = "";
    edgeLayer.innerHTML = "";
    nodeElements = {};

    if (!graph) return;

    // Render nodes
    for (const [nodeId, node] of Object.entries(graph.nodes)) {
        const el = createNodeElement(nodeId, node);
        nodeLayer.appendChild(el);
        nodeElements[nodeId] = el;
    }

    // Render edges
    requestAnimationFrame(() => drawAllEdges());
}

function createNodeElement(nodeId, node) {
    const el = document.createElement("div");
    el.className = "wf-node";
    el.id = `node-${nodeId}`;
    el.dataset.nodeId = nodeId;
    el.dataset.status = node.execution?.status || "pending";
    el.dataset.agent = node.agent_group || "";
    
    const pos = node.position || { x: 0, y: 0 };
    el.style.left = `${pos.x}px`;
    el.style.top = `${pos.y}px`;

    const status = node.execution?.status || "pending";
    const runtime = node.execution?.runtime_seconds;
    const metrics = node.execution?.metrics || {};
    const progress = node.execution?.progress || 0;

    el.innerHTML = `
        ${node.inputs?.length ? '<div class="node-port port-input"></div>' : ''}
        ${node.outputs?.length ? '<div class="node-port port-output"></div>' : ''}
        <div class="node-header">
            <div class="node-icon">${node.icon || '📦'}</div>
            <span class="node-title">${node.title}</span>
            <span class="node-status-badge" data-status="${status}">${status}</span>
        </div>
        <div class="node-body">
            <div class="node-agent-tag">${node.agent_group || ''}</div>
            <div class="node-tool">${node.tool || ''}</div>
            ${status === 'running' ? `
                <div class="node-progress-bar">
                    <div class="node-progress-fill" style="width: ${progress}%"></div>
                </div>
            ` : ''}
            ${runtime ? `<div class="node-runtime">${runtime.toFixed(1)}s</div>` : ''}
            ${Object.keys(metrics).length ? `
                <div class="node-metrics">
                    ${Object.entries(metrics).slice(0, 3).map(([k, v]) =>
                        `<span class="metric-chip"><span class="metric-key">${k}:</span> ${v}</span>`
                    ).join('')}
                </div>
            ` : ''}
        </div>
    `;

    el.addEventListener("click", (e) => {
        e.stopPropagation();
        selectNode(nodeId);
    });

    return el;
}

function updateNodeElement(nodeId, nodeData) {
    if (!graph) return;
    graph.nodes[nodeId] = { ...graph.nodes[nodeId], ...nodeData };
    
    const oldEl = nodeElements[nodeId];
    if (!oldEl) return;

    const newEl = createNodeElement(nodeId, graph.nodes[nodeId]);
    oldEl.replaceWith(newEl);
    nodeElements[nodeId] = newEl;

    if (selectedNodeId === nodeId) {
        newEl.classList.add("selected");
    }

    requestAnimationFrame(() => drawAllEdges());
}


// ═══════════════════════════════════════════════════════════
// EDGE RENDERING (Cubic Beziers)
// ═══════════════════════════════════════════════════════════

function drawAllEdges() {
    edgeLayer.innerHTML = "";
    if (!graph || !graph.edges) return;

    // SVG must cover the full transform space
    edgeLayer.setAttribute("viewBox", `0 0 10000 10000`);
    edgeLayer.style.width = "10000px";
    edgeLayer.style.height = "10000px";
    edgeLayer.style.transform = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
    edgeLayer.style.transformOrigin = "0 0";

    for (const edge of graph.edges) {
        const srcEl = nodeElements[edge.source_node];
        const tgtEl = nodeElements[edge.target_node];
        if (!srcEl || !tgtEl) continue;

        const srcNode = graph.nodes[edge.source_node];
        const tgtNode = graph.nodes[edge.target_node];

        const srcPos = srcNode.position || { x: 0, y: 0 };
        const tgtPos = tgtNode.position || { x: 0, y: 0 };

        // Calculate port positions (output port is on right, input port is on left)
        const x1 = srcPos.x + NODE_WIDTH + 5;
        const y1 = srcPos.y + (srcEl.offsetHeight || 80) / 2;
        const x2 = tgtPos.x - 5;
        const y2 = tgtPos.y + (tgtEl.offsetHeight || 80) / 2;

        // Determine edge class based on node status
        const srcStatus = srcNode.execution?.status || "pending";
        const tgtStatus = tgtNode.execution?.status || "pending";
        let edgeClass = "edge-path";
        if (srcStatus === "running" || tgtStatus === "running") {
            edgeClass += " active";
        } else if (srcStatus === "completed" && tgtStatus === "completed") {
            edgeClass += " completed";
        }

        // Cubic bezier with vertical tendency
        const dy = Math.abs(y2 - y1);
        const dx = Math.abs(x2 - x1);
        const curvature = Math.max(40, Math.min(dy * 0.4, 120));

        let d;
        if (Math.abs(x1 - x2) < NODE_WIDTH * 0.5 && y2 > y1) {
            // Mostly vertical: use S-curve going down
            const midY = (y1 + y2) / 2;
            d = `M ${x1} ${y1} C ${x1 + curvature} ${y1}, ${x2 - curvature} ${y2}, ${x2} ${y2}`;
        } else {
            // Horizontal + vertical: classic n8n bezier
            const cx = Math.max(60, dx * 0.3);
            d = `M ${x1} ${y1} C ${x1 + cx} ${y1}, ${x2 - cx} ${y2}, ${x2} ${y2}`;
        }

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        path.setAttribute("class", edgeClass);
        edgeLayer.appendChild(path);
    }
}


// ═══════════════════════════════════════════════════════════
// PAN & ZOOM
// ═══════════════════════════════════════════════════════════

function applyTransform() {
    const t = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
    nodeLayer.style.transform = t;
    edgeLayer.style.transform = t;
    zoomDisplay.textContent = `${Math.round(transform.scale * 100)}%`;
    updateMinimap();
}

function fitToView() {
    if (!graph || !graph.nodes) return;
    const nodes = Object.values(graph.nodes);
    if (nodes.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
        const p = n.position || { x: 0, y: 0 };
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x + NODE_WIDTH);
        maxY = Math.max(maxY, p.y + 120); // approx node height
    }

    const graphW = maxX - minX;
    const graphH = maxY - minY;
    const containerW = canvasContainer.clientWidth;
    const containerH = canvasContainer.clientHeight;

    const scaleX = (containerW - PAN_MARGIN * 2) / graphW;
    const scaleY = (containerH - PAN_MARGIN * 2) / graphH;
    const scale = Math.min(Math.max(Math.min(scaleX, scaleY), MIN_ZOOM), MAX_ZOOM);

    transform.scale = scale;
    transform.x = (containerW - graphW * scale) / 2 - minX * scale;
    transform.y = (containerH - graphH * scale) / 2 - minY * scale;

    applyTransform();
    requestAnimationFrame(() => drawAllEdges());
}

// Pan handlers
canvasContainer.addEventListener("mousedown", (e) => {
    if (e.target === canvasContainer || e.target === edgeLayer) {
        isPanning = true;
        panStart = { x: e.clientX - transform.x, y: e.clientY - transform.y };
    }
});

window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    transform.x = e.clientX - panStart.x;
    transform.y = e.clientY - panStart.y;
    applyTransform();
});

window.addEventListener("mouseup", () => { isPanning = false; });

// Zoom handler
canvasContainer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.min(Math.max(transform.scale * delta, MIN_ZOOM), MAX_ZOOM);

    // Zoom toward cursor
    const rect = canvasContainer.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const factor = newScale / transform.scale;
    transform.x = cx - (cx - transform.x) * factor;
    transform.y = cy - (cy - transform.y) * factor;
    transform.scale = newScale;

    applyTransform();
    requestAnimationFrame(() => drawAllEdges());
}, { passive: false });


// ═══════════════════════════════════════════════════════════
// NODE SELECTION & INSPECTOR
// ═══════════════════════════════════════════════════════════

function selectNode(nodeId) {
    // Deselect old
    if (selectedNodeId && nodeElements[selectedNodeId]) {
        nodeElements[selectedNodeId].classList.remove("selected");
    }

    if (selectedNodeId === nodeId) {
        // Toggle off
        selectedNodeId = null;
        inspector.classList.add("closed");
        return;
    }

    selectedNodeId = nodeId;
    if (nodeElements[nodeId]) {
        nodeElements[nodeId].classList.add("selected");
    }
    openInspector(nodeId);
}

function openInspector(nodeId) {
    const node = graph?.nodes?.[nodeId];
    if (!node) return;

    inspectorIcon.textContent = node.icon || "📦";
    inspectorTitle.textContent = node.title;
    inspector.classList.remove("closed");

    const exec = node.execution || {};
    const status = exec.status || "pending";
    const metrics = exec.metrics || {};
    const artifacts = exec.artifacts || [];
    const logs = exec.logs || [];
    const inputs = node.inputs || [];
    const outputs = node.outputs || [];

    let html = "";

    // Status section
    html += `<div class="insp-section">
        <div class="insp-section-title">Execution</div>
        <div class="insp-row"><span class="insp-key">Status</span><span class="insp-value"><span class="node-status-badge" data-status="${status}">${status}</span></span></div>
        ${exec.runtime_seconds != null ? `<div class="insp-row"><span class="insp-key">Runtime</span><span class="insp-value">${exec.runtime_seconds.toFixed(2)}s</span></div>` : ''}
        ${exec.progress && status === 'running' ? `<div class="insp-row"><span class="insp-key">Progress</span><span class="insp-value">${exec.progress}% — ${exec.progress_message || ''}</span></div>` : ''}
        ${exec.tool ? `<div class="insp-row"><span class="insp-key">Tool</span><span class="insp-value">${exec.tool}</span></div>` : ''}
        ${exec.model ? `<div class="insp-row"><span class="insp-key">Model</span><span class="insp-value">${exec.model}</span></div>` : ''}
        ${exec.error_message ? `<div class="insp-row"><span class="insp-key">Error</span><span class="insp-value" style="color: var(--status-failed)">${exec.error_message}</span></div>` : ''}
    </div>`;

    // Description
    if (node.description) {
        html += `<div class="insp-section">
            <div class="insp-section-title">Description</div>
            <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.6">${node.description}</div>
        </div>`;
    }

    // Node info
    html += `<div class="insp-section">
        <div class="insp-section-title">Details</div>
        <div class="insp-row"><span class="insp-key">Node ID</span><span class="insp-value">${nodeId}</span></div>
        <div class="insp-row"><span class="insp-key">Agent</span><span class="insp-value">${node.agent_group}</span></div>
        <div class="insp-row"><span class="insp-key">Category</span><span class="insp-value">${node.category}</span></div>
        <div class="insp-row"><span class="insp-key">Tool</span><span class="insp-value">${node.tool || '—'}</span></div>
    </div>`;

    // Ports
    if (inputs.length || outputs.length) {
        html += `<div class="insp-section"><div class="insp-section-title">Ports</div>`;
        for (const p of inputs) {
            html += `<div class="insp-port"><div class="insp-port-dot" style="background: #818cf8"></div><span class="insp-port-name">${p.name}</span><span class="insp-port-type">${p.data_type}</span></div>`;
        }
        for (const p of outputs) {
            html += `<div class="insp-port"><div class="insp-port-dot" style="background: #10b981"></div><span class="insp-port-name">${p.name}</span><span class="insp-port-type">${p.data_type}</span></div>`;
        }
        html += `</div>`;
    }

    // Metrics
    if (Object.keys(metrics).length) {
        html += `<div class="insp-section"><div class="insp-section-title">Metrics</div>`;
        for (const [k, v] of Object.entries(metrics)) {
            html += `<div class="insp-row"><span class="insp-key">${k}</span><span class="insp-value">${v}</span></div>`;
        }
        html += `</div>`;
    }

    // Video player & Download group if final video exists or if inspecting node_agent3
    const videoArtifact = artifacts.find(a => a.type === 'video' || (a.name && a.name.endsWith('.mp4')));
    if (videoArtifact || (nodeId === 'node_agent3' && status === 'completed')) {
        const cleanPath = videoArtifact ? (videoArtifact.path.replace(/\\/g, '/').split('/outputs/')[1] || videoArtifact.name) : '';
        const baseSrc = cleanPath ? `/api/artifacts/${cleanPath}` : '/api/workflow/download-video';
        const videoSrc = `${baseSrc}?t=${Date.now()}`;
        html += `<div class="insp-section">
            <div class="insp-section-title">Final Edited Video Preview</div>
            <div class="insp-video-container">
                <video controls class="insp-video-player" src="${videoSrc}"></video>
            </div>
            <div class="insp-download-group">
                <a href="/api/workflow/download-video?t=${Date.now()}" download="final_reel_edited.mp4" class="btn btn-download btn-download-link">
                    ⬇️ Download Final Reel (.mp4)
                </a>
                <a href="/api/workflow/download-edl" download="timeline.edl" class="btn btn-secondary btn-download-link">
                    🎬 Export DaVinci Resolve EDL
                </a>
            </div>
        </div>`;
    }

    // Artifacts
    if (artifacts.length) {
        html += `<div class="insp-section"><div class="insp-section-title">Artifacts</div>`;
        for (const a of artifacts) {
            const icon = a.type === 'json' ? '📄' : a.type === 'audio' ? '🔊' : a.type === 'image' ? '🖼️' : a.type === 'video' ? '🎬' : '📁';
            const cleanPath = (a.path || a.name).replace(/\\/g, '/').split('/outputs/')[1] || a.name;
            html += `<div class="insp-artifact" style="display: flex; align-items: center; justify-content: space-between; padding: 6px 10px;">
                <div style="display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    <span class="insp-artifact-icon">${icon}</span>
                    <a href="/api/artifacts/${cleanPath}" target="_blank" class="insp-artifact-name" style="color:var(--text-primary); text-decoration:none; font-size: 12px;">${a.name}</a>
                </div>
                <a href="/api/artifacts/${cleanPath}" download="${a.name}" class="btn btn-ghost" style="padding: 2px 8px; font-size: 11px; margin-left: 8px;" title="Download Artifact">⬇️</a>
            </div>`;
        }
        html += `</div>`;
    }

    // Logs
    if (logs.length) {
        html += `<div class="insp-section"><div class="insp-section-title">Logs (last ${Math.min(logs.length, 50)})</div>`;
        for (const log of logs.slice(-30)) {
            const level = log.level || "INFO";
            html += `<div class="insp-log level-${level}">${log.message}</div>`;
        }
        html += `</div>`;
    }

    inspectorBody.innerHTML = html;
}


inspectorClose.addEventListener("click", () => {
    inspector.classList.add("closed");
    if (selectedNodeId && nodeElements[selectedNodeId]) {
        nodeElements[selectedNodeId].classList.remove("selected");
    }
    selectedNodeId = null;
});

// Click canvas to deselect
canvasContainer.addEventListener("click", (e) => {
    if (e.target === canvasContainer || e.target === edgeLayer) {
        if (selectedNodeId && nodeElements[selectedNodeId]) {
            nodeElements[selectedNodeId].classList.remove("selected");
        }
        selectedNodeId = null;
        inspector.classList.add("closed");
    }
});


// ═══════════════════════════════════════════════════════════
// SSE — Real-Time Event Stream
// ═══════════════════════════════════════════════════════════

function connectSSE() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("/api/workflow/events");

    eventSource.addEventListener("snapshot", (e) => {
        const data = JSON.parse(e.data);
        graph = data;
        renderGraph();
        applyTransform();
        updateWorkflowStatus(graph.status);
    });

    eventSource.addEventListener("node:status_change", (e) => {
        const evt = JSON.parse(e.data);
        handleNodeStatusChange(evt);
    });

    eventSource.addEventListener("node:progress", (e) => {
        const evt = JSON.parse(e.data);
        handleNodeProgress(evt);
    });

    eventSource.addEventListener("node:log", (e) => {
        const evt = JSON.parse(e.data);
        handleNodeLog(evt);
    });

    eventSource.addEventListener("node:artifact", (e) => {
        const evt = JSON.parse(e.data);
        handleNodeArtifact(evt);
    });

    eventSource.addEventListener("node:metrics", (e) => {
        const evt = JSON.parse(e.data);
        handleNodeMetrics(evt);
    });

    eventSource.addEventListener("workflow:started", () => {
        updateWorkflowStatus("running");
    });

    eventSource.addEventListener("workflow:completed", () => {
        updateWorkflowStatus("completed");
    });

    eventSource.addEventListener("workflow:failed", () => {
        updateWorkflowStatus("failed");
    });

    eventSource.addEventListener("workflow:cancelled", () => {
        updateWorkflowStatus("cancelled");
    });

    eventSource.onerror = () => {
        console.warn("SSE connection lost, reconnecting in 3s...");
        setTimeout(connectSSE, 3000);
    };
}

function handleNodeStatusChange(evt) {
    const nodeId = evt.node_id;
    const data = evt.data;
    if (!graph?.nodes?.[nodeId]) return;

    const node = graph.nodes[nodeId];
    node.execution = { ...node.execution, ...data };
    updateNodeElement(nodeId, node);

    // Refresh inspector if this node is selected
    if (selectedNodeId === nodeId) {
        openInspector(nodeId);
    }
}

function handleNodeProgress(evt) {
    const nodeId = evt.node_id;
    const data = evt.data;
    if (!graph?.nodes?.[nodeId]) return;

    graph.nodes[nodeId].execution.progress = data.progress;
    graph.nodes[nodeId].execution.progress_message = data.message;

    // Update progress bar in node
    const el = nodeElements[nodeId];
    if (el) {
        const fill = el.querySelector(".node-progress-fill");
        if (fill) fill.style.width = `${data.progress}%`;
    }

    if (selectedNodeId === nodeId) {
        openInspector(nodeId);
    }
}

function handleNodeLog(evt) {
    const nodeId = evt.node_id;
    if (!graph?.nodes?.[nodeId]) return;

    if (!graph.nodes[nodeId].execution.logs) {
        graph.nodes[nodeId].execution.logs = [];
    }
    graph.nodes[nodeId].execution.logs.push(evt.data);

    if (selectedNodeId === nodeId) {
        openInspector(nodeId);
    }
}

function handleNodeArtifact(evt) {
    const nodeId = evt.node_id;
    if (!graph?.nodes?.[nodeId]) return;

    if (!graph.nodes[nodeId].execution.artifacts) {
        graph.nodes[nodeId].execution.artifacts = [];
    }
    graph.nodes[nodeId].execution.artifacts.push(evt.data);

    if (selectedNodeId === nodeId) {
        openInspector(nodeId);
    }
}

function handleNodeMetrics(evt) {
    const nodeId = evt.node_id;
    if (!graph?.nodes?.[nodeId]) return;

    graph.nodes[nodeId].execution.metrics = {
        ...graph.nodes[nodeId].execution.metrics,
        ...evt.data.metrics,
    };

    updateNodeElement(nodeId, graph.nodes[nodeId]);

    if (selectedNodeId === nodeId) {
        openInspector(nodeId);
    }
}


// ═══════════════════════════════════════════════════════════
// WORKFLOW STATUS
// ═══════════════════════════════════════════════════════════

function updateWorkflowStatus(status) {
    statusPill.dataset.status = status;
    statusText.textContent = capitalize(status);

    const btnDownload = document.getElementById("btn-download-video");

    if (status === "running") {
        btnRun.classList.add("hidden");
        btnCancel.classList.remove("hidden");
        if (btnDownload) btnDownload.classList.add("hidden");
    } else {
        btnRun.classList.remove("hidden");
        btnCancel.classList.add("hidden");
        if (status === "completed" && btnDownload) {
            btnDownload.classList.remove("hidden");
        }
    }
}



// ═══════════════════════════════════════════════════════════
// RUN / CANCEL
// ═══════════════════════════════════════════════════════════

btnRun.addEventListener("click", () => {
    runOverlay.classList.remove("hidden");
    videoPathInput.focus();
});

dialogCancel.addEventListener("click", () => {
    runOverlay.classList.add("hidden");
});

runOverlay.addEventListener("click", (e) => {
    if (e.target === runOverlay) runOverlay.classList.add("hidden");
});

dialogConfirm.addEventListener("click", async () => {
    const videoPath = videoPathInput.value.trim();
    if (!videoPath) return;

    runOverlay.classList.add("hidden");
    try {
        const res = await fetch("/api/workflow/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_path: videoPath }),
        });
        const result = await res.json();
        if (!res.ok) {
            alert(`Error: ${result.detail || 'Unknown error'}`);
        }
    } catch (e) {
        alert("Failed to start workflow: " + e.message);
    }
});

btnCancel.addEventListener("click", async () => {
    try {
        await fetch("/api/workflow/cancel", { method: "POST" });
    } catch (e) {
        console.error("Failed to cancel:", e);
    }
});

btnFit.addEventListener("click", fitToView);


// ═══════════════════════════════════════════════════════════
// MINIMAP
// ═══════════════════════════════════════════════════════════

function updateMinimap() {
    if (!minimapCanvas || !graph) return;
    const ctx = minimapCanvas.getContext("2d");
    const cw = minimapCanvas.width = minimapCanvas.clientWidth * 2;
    const ch = minimapCanvas.height = minimapCanvas.clientHeight * 2;

    ctx.clearRect(0, 0, cw, ch);

    const nodes = Object.values(graph.nodes);
    if (!nodes.length) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of nodes) {
        const p = n.position || { x: 0, y: 0 };
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x + NODE_WIDTH);
        maxY = Math.max(maxY, p.y + 100);
    }

    const graphW = maxX - minX + 60;
    const graphH = maxY - minY + 60;
    const scale = Math.min(cw / graphW, ch / graphH) * 0.85;
    const offX = (cw - graphW * scale) / 2;
    const offY = (ch - graphH * scale) / 2;

    // Draw edges
    ctx.strokeStyle = "rgba(124, 58, 237, 0.2)";
    ctx.lineWidth = 1;
    for (const edge of (graph.edges || [])) {
        const src = graph.nodes[edge.source_node]?.position;
        const tgt = graph.nodes[edge.target_node]?.position;
        if (!src || !tgt) continue;
        ctx.beginPath();
        ctx.moveTo(offX + (src.x - minX + NODE_WIDTH) * scale, offY + (src.y - minY + 40) * scale);
        ctx.lineTo(offX + (tgt.x - minX) * scale, offY + (tgt.y - minY + 40) * scale);
        ctx.stroke();
    }

    // Draw nodes
    for (const n of nodes) {
        const p = n.position || { x: 0, y: 0 };
        const x = offX + (p.x - minX) * scale;
        const y = offY + (p.y - minY) * scale;
        const w = NODE_WIDTH * scale;
        const h = 18 * scale;

        const status = n.execution?.status || "pending";
        ctx.fillStyle = status === "completed" ? "rgba(16,185,129,0.6)"
                      : status === "running" ? "rgba(245,158,11,0.6)"
                      : status === "failed" ? "rgba(239,68,68,0.6)"
                      : "rgba(124,58,237,0.3)";

        ctx.fillRect(x, y, w, h);
    }

    // Draw viewport rectangle
    const containerW = canvasContainer.clientWidth;
    const containerH = canvasContainer.clientHeight;
    const vx = offX + (-transform.x / transform.scale - minX) * scale;
    const vy = offY + (-transform.y / transform.scale - minY) * scale;
    const vw = (containerW / transform.scale) * scale;
    const vh = (containerH / transform.scale) * scale;

    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vx, vy, vw, vh);
}


// ═══════════════════════════════════════════════════════════
// EVENT BINDINGS
// ═══════════════════════════════════════════════════════════

function bindEvents() {
    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            inspector.classList.add("closed");
            runOverlay.classList.add("hidden");
            if (selectedNodeId && nodeElements[selectedNodeId]) {
                nodeElements[selectedNodeId].classList.remove("selected");
            }
            selectedNodeId = null;
        }
        if (e.key === "f" && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== "INPUT") {
            fitToView();
        }
    });

    // Handle window resize
    window.addEventListener("resize", () => {
        requestAnimationFrame(() => drawAllEdges());
        updateMinimap();
    });

    // ── Video Upload Handling ──────────────────────────────────
    const btnUpload = document.getElementById("btn-upload");
    const fileUploadInput = document.getElementById("file-upload-input");

    if (btnUpload && fileUploadInput) {
        btnUpload.addEventListener("click", () => fileUploadInput.click());

        fileUploadInput.addEventListener("change", async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;

            btnUpload.disabled = true;
            btnUpload.innerHTML = `<span>Uploading ${file.name}...</span>`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch("/api/workflow/upload", {
                    method: "POST",
                    body: formData,
                });
                const data = await res.json();
                if (res.ok) {
                    videoPathInput.value = data.path || data.filename;
                    if (graph?.nodes?.["node_raw_video"]) {
                        graph.nodes["node_raw_video"].execution.metrics = {
                            file: data.filename,
                            size: `${data.size_mb} MB`,
                        };
                        updateNodeElement("node_raw_video", graph.nodes["node_raw_video"]);
                    }
                    // Reset statuses to idle
                    updateWorkflowStatus("idle");
                    alert(`✅ Video "${data.filename}" uploaded successfully (${data.size_mb} MB)!\nClick 'Run' to analyze and edit.`);
                } else {
                    alert(`Upload failed: ${data.detail || "Unknown error"}`);
                }
            } catch (err) {
                alert(`Upload error: ${err.message}`);
            } finally {
                btnUpload.disabled = false;
                btnUpload.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    <span>Upload Video</span>
                `;
                fileUploadInput.value = "";
            }
        });
    }
}



// ═══════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════

function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}


// ─── Boot ───────────────────────────────────────────────────
init();

})();
