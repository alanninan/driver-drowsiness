"use strict";
const $ = id => document.getElementById(id);
let socket = null,
    stream = null,
    running = false,
    mode = null,
    originTime = 0;
let frameTimer, staleTimer, lastReceived = 0,
    lastState = "",
    lastSample = -1;
let history = [],
    sessionStarted = null,
    metadata = null;
const alertSound = new AlertSound({
    enabled: () => $("sound").checked,
    status: text => {
        if ($("sound-status").textContent !== text) $("sound-status").textContent = text;
    },
    createContext: () => {
        const Audio = window.AudioContext || window.webkitAudioContext;
        if (!Audio) throw new Error("Web Audio unavailable");
        return new Audio();
    }
});
let generation = 0;
let paused = false, pausePending = false, calibrated = false, captureGeneration = 0;
const capture = document.createElement("canvas");
const context = capture.getContext("2d");
const stateNames = {
    calibrating: "Learning your baseline",
    warming: "Gathering history",
    unknown: "Tracking unavailable",
    alert: "Alert",
    fatigued: "Fatigued",
    critical: "Critical"
};
const units = {
    "closure-ratio": "% of observed time",
    "closure-seconds": "seconds",
    "blink-seconds": "seconds",
    yawns: "per observed minute",
    nods: "per observed minute",
    coverage: "% of recent time"
};

function scoringLabels(learned) {
    $("model-kind").textContent = learned ? "Personal model" : "Rule engine";
    const levels = document.querySelector(".levels");
    levels.setAttribute("aria-label", learned ? "Predicted vigilance classes" : "Heuristic fatigue score ranges");
    const labels = learned ? ["0", "50", "100"] : ["0–34", "35–64", "65–100"];
    levels.querySelectorAll("small").forEach((el, i) => el.textContent = labels[i]);
    document.querySelector("footer p").textContent = learned
        ? "Experimental model. Levels 0, 50 and 100 mean alert, low vigilance and drowsy; they are not confidence percentages. Do not test while driving."
        : "Experimental fatigue cues, not a validated driving-safety system. The score is a heuristic, not a probability. Do not test while driving.";
}

function measurement(id, value, digits = 1) {
    const el = $(id);
    el.replaceChildren(document.createTextNode(value == null ? "—" : Number(value).toFixed(digits)));
    const small = document.createElement("small");
    small.textContent = units[id];
    el.append(small);
}

function activity(text) {
    if ($("events").querySelector(".muted")) $("events").replaceChildren();
    const li = document.createElement("li"),
        time = document.createElement("time"),
        span = document.createElement("span");
    time.textContent = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
    span.textContent = text;
    li.append(time, span);
    $("events").prepend(li);
    while ($("events").children.length > 10) $("events").lastChild.remove();
}

function showError(message) {
    $("error").textContent = message;
    $("error").hidden = false;
}

function controls(active) {
    $("start").disabled = active;
    $("demo").disabled = active;
    $("stop").disabled = !active;
    $("duration").disabled = active;
    $("clahe").disabled = active;
    $("pause").disabled = !active || mode !== "live" || !calibrated || pausePending;
    $("pause").textContent = paused ? "Resume camera" : "Pause camera";
}

function clearMeasurements() {
    for (const id of Object.keys(units)) measurement(id, null);
    ["ear", "mar", "pitch", "score"].forEach(id => $(id).textContent = "—");
}

function reset() {
    paused = pausePending = calibrated = false;
    captureGeneration++;
    history = [];
    lastSample = -1;
    metadata = null;
    lastState = "";
    alertSound.reset();
    $("error").hidden = true;
    $("export").disabled = true;
    $("trace").setAttribute("d", "");
    $("chart-empty").textContent = "Your fatigue history will appear here";
    $("chart-empty").style.display = "";
    $("calibration").value = 0;
    $("calibration-percent").textContent = "0%";
    $("calibration-title").textContent = "A baseline that belongs to you";
    $("events").replaceChildren();
    clearMeasurements();
    sessionStarted = new Date().toISOString();
}

function stop(note = "Session stopped", record = true) {
    paused = pausePending = false;
    captureGeneration++;
    alertSound.stop();
    generation++;
    running = false;
    clearTimeout(frameTimer);
    clearInterval(staleTimer);
    const previous = socket;
    socket = null;
    if (previous) previous.close();
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    $("video").srcObject = null;
    $("video").hidden = true;
    $("overlay").hidden = true;
    $("empty-preview").hidden = false;
    $("preview-title").textContent = "Ready for another session";
    $("preview-message").textContent = "Start when rested to learn a fresh baseline.";
    $("connection").textContent = "Camera off";
    $("preview-tag").textContent = "Session stopped";
    $("state").textContent = note;
    $("reason").textContent = "No current estimate. Session history remains available to export.";
    document.body.dataset.state = "unknown";
    clearMeasurements();
    controls(false);
    if (record) activity(note);
}

function togglePause() {
    if (!running || mode !== "live" || !calibrated || pausePending || socket?.readyState !== WebSocket.OPEN) return;
    pausePending = true;
    captureGeneration++;
    clearTimeout(frameTimer);
    alertSound.stop();
    if (!paused) {
        // Disable capture immediately; keep the track/session for Resume.
        stream?.getVideoTracks().forEach(track => track.enabled = false);
        $("video").pause();
        $("video").hidden = $("overlay").hidden = true;
        $("empty-preview").hidden = false;
        $("preview-title").textContent = "Camera paused";
        $("preview-message").textContent = "Calibration is saved for this session.";
        $("state").textContent = "Camera paused";
        $("reason").textContent = "No measurements or alerts while paused.";
        document.body.dataset.state = "unknown";
        clearMeasurements();
        history.push({timestamp: (performance.now() - originTime) / 1000, score: null, state: "paused"});
        drawChart();
    } else {
        void alertSound.enable();
    }
    socket.send(JSON.stringify({action: paused ? "resume" : "pause"}));
    lastReceived = performance.now();
    controls(true);
}
async function start(nextMode) {
    if (running) return;
    reset();
    running = true;
    mode = nextMode;
    controls(true);
    const run = ++generation;
    $("demo-banner").hidden = mode !== "demo";
    $("state").textContent = "Starting session";
    $("reason").textContent = "Connecting to the local detector…";
    try {
        // A blocked audio device must not prevent camera or demo startup.
        void alertSound.enable();
        if (mode === "live") {
            if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs localhost in a supported browser.");
            const acquired = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: {
                        ideal: 640
                    },
                    height: {
                        ideal: 480
                    },
                    facingMode: "user"
                },
                audio: false
            });
            if (run !== generation) {
                acquired.getTracks().forEach(t => t.stop());
                return;
            }
            stream = acquired;
            $("video").srcObject = stream;
            await $("video").play();
            if (run !== generation) return;
            $("video").hidden = false;
            $("overlay").hidden = false;
            $("empty-preview").hidden = true;
            stream.getVideoTracks()[0].addEventListener("ended", () => {
                if (running) stop("Camera disconnected");
            });
        } else {
            $("preview-title").textContent = "Simulated measurements";
            $("preview-message").textContent = "Watch the rule engine respond to generated eye closures, yawns, and head movement.";
            $("calibration-title").textContent = "Fixed example baseline";
            $("calibration-detail").textContent = "Synthetic signals · no personal calibration";
        }
        const ws = new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/${mode}`);
        socket = ws;
        originTime = performance.now();
        lastReceived = performance.now();
        ws.onopen = () => {
            if (run === generation) ws.send(JSON.stringify({
                calibration_seconds: Number($("duration").value),
                clahe: $("clahe").checked
            }));
        };
        ws.onmessage = async event => {
            if (run !== generation) return;
            lastReceived = performance.now();
            let data;
            try {
                data = JSON.parse(event.data);
            } catch {
                stop("Invalid server response");
                showError("Restart the local server and try again.");
                return;
            }
            if (data.error) {
                stop("Session interrupted");
                showError(data.error);
                return;
            }
            if (data.pause_error) {
                stop("Pause unavailable");
                showError(data.pause_error);
                return;
            }
            if (typeof data.paused === "boolean") {
                paused = data.paused;
                if (paused) {
                    pausePending = false;
                    $("connection").textContent = "Camera paused";
                    $("preview-tag").textContent = "Calibration saved";
                    activity("Camera paused; calibration saved");
                } else {
                    try {
                        stream?.getVideoTracks().forEach(track => track.enabled = true);
                        await $("video").play();
                        if (run !== generation) return;
                        pausePending = false;
                        $("video").hidden = $("overlay").hidden = false;
                        $("empty-preview").hidden = true;
                        $("state").textContent = "Gathering fresh history";
                        $("reason").textContent = "Calibration retained; waiting for fresh measurements.";
                        lastReceived = performance.now();
                        alertSound.reset();
                        activity("Camera resumed; calibration retained");
                        sendFrame(run);
                    } catch (error) {
                        if (run !== generation) return;
                        stop("Camera unavailable");
                        showError(error.message);
                    }
                }
                controls(running);
                return;
            }
            if (data.ready) {
                scoringLabels(data.scoring === "behavioral_classifier");
                activity(mode === "demo" ? "Simulated session started" : "Camera connected; calibration started");
                if (mode === "live") sendFrame(run);
                return;
            }
            if (paused || pausePending) return;
            calibrated = data.calibration >= 1;
            controls(true);
            render(data);
            if (mode === "live") frameTimer = setTimeout(() => sendFrame(run), 33);
        };
        ws.onerror = () => {
            if (run === generation) {
                stop("Connection failed");
                showError("Cannot reach the local detector. Check the terminal, then restart the session.");
            }
        };
        ws.onclose = () => {
            if (run === generation && running) {
                stop("Connection closed");
                showError("The detector disconnected. Restart the session to recalibrate.");
            }
        };
        staleTimer = setInterval(() => {
            if (paused && !pausePending) return;
            if (performance.now() - lastReceived > 4000 && running) {
                $("state").textContent = "Connection stalled";
                $("reason").textContent = "Waiting for fresh measurements…";
                document.body.dataset.state = "unknown";
                clearMeasurements();
            }
            if (performance.now() - lastReceived > 12000 && running) {
                stop("Stream timed out");
                showError("No frames arrived for 12 seconds. Restart the session.");
            }
        }, 1000);
    } catch (error) {
        if (run !== generation) return;
        stop("Camera unavailable");
        const message = error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access in the browser and try again." : error.name === "NotFoundError" ? "No camera was found. Connect a webcam or try the simulated session." : error.name === "NotReadableError" ? "The camera is busy. Close other camera apps and try again." : error.message;
        showError(message);
    }
}

function sendFrame(run) {
    if (!running || paused || pausePending || run !== generation || socket?.readyState !== WebSocket.OPEN) return;
    const captureRun = captureGeneration;
    const video = $("video");
    if (!video.videoWidth) return;
    capture.width = Math.min(640, video.videoWidth);
    capture.height = Math.round(video.videoHeight * capture.width / video.videoWidth);
    context.drawImage(video, 0, 0, capture.width, capture.height);
    const timestamp = (performance.now() - originTime) / 1000;
    capture.toBlob(async blob => {
        if (!blob) return;
        const jpeg = await blob.arrayBuffer();
        if (!running || paused || pausePending || captureRun !== captureGeneration || run !== generation || socket?.readyState !== WebSocket.OPEN) return;
        const packet = new Uint8Array(8 + jpeg.byteLength);
        new DataView(packet.buffer).setFloat64(0, timestamp, true);
        packet.set(new Uint8Array(jpeg), 8);
        socket.send(packet);
    }, "image/jpeg", .8);
}

function render(data) {
    metadata = data.metadata;
    document.body.dataset.state = data.state;
    const stateLabel = stateNames[data.state] || data.state;
    // Avoid re-announcing the same aria-live status for every camera frame.
    if ($("state").textContent !== stateLabel) $("state").textContent = stateLabel;
    $("reason").textContent = data.reason;
    $("score").textContent = data.score == null ? "—" : Math.round(data.score);
    $("connection").textContent = mode === "live" ? "Camera connected" : "Synthetic input";
    $("preview-tag").textContent = mode === "demo" ? "SIMULATION · no camera" : data.state === "unknown" ? "Tracking unavailable" : "Live · local processing";
    $("latency").textContent = `${data.processing_ms.toFixed(0)} ms / frame`;
    $("calibration").value = data.calibration;
    $("calibration-percent").textContent = `${Math.round(data.calibration*100)}%`;
    if (mode === "live") {
        $("calibration-title").textContent = data.calibration >= 1 ? "Your baseline is ready" : "Learning your baseline";
        $("calibration-detail").textContent = data.calibration >= 1 ? "Fixed for this session · restart to recalibrate" : data.metadata?.scoring === "behavioral_classifier" ? "3 minutes while rested · blink naturally and keep your face visible" : `${$("duration").value} seconds of usable footage · blink naturally`;
    }
    const usable = data.state !== "unknown" && data.state !== "calibrating";
    measurement("closure-ratio", usable && data.closure_ratio != null ? data.closure_ratio * 100 : null);
    measurement("closure-seconds", usable ? data.closure_seconds : null, 2);
    measurement("blink-seconds", usable ? data.blink_seconds : null, 2);
    measurement("yawns", usable ? data.yawns_per_minute : null);
    measurement("nods", usable ? data.nods_per_minute : null);
    measurement("coverage", data.coverage * 100, 0);
    $("ear").textContent = data.ear?.toFixed(3) ?? "—";
    $("mar").textContent = data.mar?.toFixed(3) ?? "—";
    $("pitch").textContent = data.pitch == null ? "—" : `${data.pitch.toFixed(1)}°`;
    const overlay = $("overlay");
    overlay.width = capture.width || 640;
    overlay.height = capture.height || 480;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.fillStyle = "#d8efba";
    for (const [x, y] of data.overlay || []) {
        ctx.beginPath();
        ctx.arc(x * overlay.width, y * overlay.height, 2, 0, Math.PI * 2);
        ctx.fill();
    }
    if (data.state !== lastState) {
        activity(stateNames[data.state] || data.state);
        lastState = data.state;
    }
    if (Math.floor(data.timestamp) > lastSample) {
        lastSample = Math.floor(data.timestamp);
        const {
            overlay,
            metadata,
            ...record
        } = data;
        history.push(record);
        if (history.length > 3600) history.shift();
        $("export").disabled = false;
        drawChart();
    }
    if (running && data.state === "critical") alertSound.play();
    else alertSound.stop();
}

function drawChart() {
    const width = Math.max(280, $("chart").getBoundingClientRect().width);
    $("chart").setAttribute("viewBox", `0 0 ${width} 180`);
    $("chart").querySelector(".grid path").setAttribute("d", `M38 18H${width-12}M38 66H${width-12}M38 113H${width-12}M38 160H${width-12}`);
    $("chart-empty").setAttribute("x", (width + 38) / 2);
    const end = history.at(-1)?.timestamp || 0;
    const points = history.filter(p => p.timestamp >= end - 120);
    let d = "",
        connected = false,
        previous = null;
    for (const point of points) {
        if (point.score == null) {
            connected = false;
            continue;
        }
        if (previous != null && point.timestamp - previous > 2) connected = false;
        const x = 38 + (point.timestamp - (end - 120)) / 120 * (width - 50),
            y = 160 - point.score / 100 * 142;
        d += `${connected?"L":"M"}${x.toFixed(1)},${y.toFixed(1)} `;
        connected = true;
        previous = point.timestamp;
    }
    $("trace").setAttribute("d", d);
    $("chart-empty").style.display = d ? "none" : "";
    $("chart-empty").textContent = history.length ? "Gathering usable history…" : width < 500 ? "Waiting for a session" : "Your fatigue history will appear here";
}
window.addEventListener("resize", drawChart);
drawChart();
$("start").addEventListener("click", () => start("live"));
$("demo").addEventListener("click", () => start("demo"));
$("stop").addEventListener("click", () => stop());
$("pause").addEventListener("click", togglePause);
$("duration").addEventListener("change", () => {
    $("calibration-detail").textContent = `${$("duration").value} seconds of usable footage · blink naturally`;
});
$("sound").addEventListener("change", async () => {
    if ($("sound").checked) {
        await alertSound.enable();
    } else {
        alertSound.stop();
        alertSound.updateStatus();
    }
});
$("test-sound").addEventListener("click", async () => {
    $("sound").checked = true;
    if (await alertSound.enable()) alertSound.play(false);
});
$("export").addEventListener("click", () => {
    const data = {
        schema: 1,
        started_at: sessionStarted,
        exported_at: new Date().toISOString(),
        mode,
        metadata,
        note: "At most the latest 3600 one-second samples. No video. See metadata for model and score meaning; displayed values are not probabilities.",
        samples: history
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json"
    }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `drowsiness-${mode}-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("beforeunload", () => {
    alertSound.stop();
    if (stream) stream.getTracks().forEach(t => t.stop());
    socket?.close();
});
fetch("/api/health").then(r => {
    if (!r.ok) throw new Error("Health check failed");
    return r.json();
}).then(data => {
    $("runtime").textContent = `v${data.version} · ${data.classifier_ready ? "Personal model" : "Rule engine"} · local only`;
    scoringLabels(data.classifier_ready);
    if (data.classifier_ready) {
        const option = document.createElement("option");
        option.value = "180";
        option.textContent = "3 minutes · personal model";
        $("duration").replaceChildren(option);
        $("calibration-detail").textContent = "3 minutes while rested · blink naturally";
    }
    if (!data.model_ready) showError("Face model is missing. In the project terminal, run: uv run drowsiness download-model. The simulated session is still available.");
}).catch(() => {
    $("runtime").textContent = "Local detector unavailable";
    showError("Cannot reach the local detector. Start it with: uv run drowsiness serve");
});
