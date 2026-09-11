"use strict";

const $ = (id) => document.getElementById(id);
const video = $("camera-video");
const promptInput = $("prompt-input");
const saveButton = $("save-prompt");
const saveStatus = $("save-status");
const STARTUP_BUFFER_SECONDS = 0.36;
const TARGET_LIVE_LATENCY_SECONDS = 0.16;
const REBUFFER_RESUME_SECONDS = 0.22;
const MAX_LIVE_LATENCY_SECONDS = 0.40;
const LOW_BUFFER_SECONDS = 0.12;
const RECOVERED_BUFFER_SECONDS = 0.17;
const BUFFER_RECOVERY_PLAYBACK_RATE = 0.985;
const REBUFFER_DEBOUNCE_MS = 180;
const MAX_APPEND_QUEUE_SEGMENTS = 12;
let serverPromptVersion = 0;
let userIsEditing = false;
let mediaSource;
let mediaUrl;
let sourceBuffer;
let appendQueue = [];
let videoSocket;
let reconnectTimer;
let playbackStarted = false;
let rebuffering = true;
let liveSeekPending = false;
let rebufferTimer;
const playbackDiagnostics = {
  waitingEvents: 0,
  confirmedRebuffers: 0,
  rebufferRecoveries: 0,
  liveEdgeSeeks: 0,
  clockRecoveries: 0,
  appendQueueOverflows: 0,
  bufferedAheadSeconds: 0,
  appendQueueSegments: 0,
  playbackRate: 1,
  captureToMediaEdgeSeconds: null,
  captureToDisplaySeconds: null,
};
window.__vlmPlaybackDiagnostics = playbackDiagnostics;

function fixed(value, digits = 1, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

function setText(id, value) {
  $(id).textContent = value;
}

function updatePromptCount() {
  setText("prompt-count", `${promptInput.value.length} / 512`);
}

function updateState(state) {
  const performance = state.performance;
  const prompt = state.prompt;
  const caption = state.caption;
  setText("pipeline-state", state.pipeline.state.toUpperCase());
  setText("camera-fps", fixed(performance.camera_fps));
  setText("yolo-fps", fixed(performance.yolo_fps));
  setText("yolo-ms", fixed(performance.yolo_inference_ms));
  setText("vlm-seconds", Number.isFinite(performance.vlm_latency_ms) ? fixed(performance.vlm_latency_ms / 1000, 2) : "—");
  setText("vlm-tps", fixed(performance.vlm_tokens_per_second));
  setText("vlm-state", performance.vlm_state);
  setText("caption", caption.text);
  setText("caption-frame", caption.request_id ? `Frame ${caption.source_frame_id} · Request ${caption.request_id}` : "No result yet");
  setText("prompt-version-result", caption.prompt_version ? `Prompt v${caption.prompt_version} applied` : "Prompt not applied yet");
  setText("client-count", `${state.video.clients} viewer${state.video.clients === 1 ? "" : "s"}`);
  const timing = state.video.timing;
  if (
    playbackStarted
    && timing
    && Number.isFinite(timing.latest_fragment_capture_monotonic_seconds)
    && Number.isFinite(timing.server_monotonic_seconds)
  ) {
    const range = latestBufferedRange();
    const fragmentAge = Math.max(
      0,
      timing.server_monotonic_seconds
        - timing.latest_fragment_capture_monotonic_seconds,
    );
    playbackDiagnostics.captureToMediaEdgeSeconds = fragmentAge;
    playbackDiagnostics.captureToDisplaySeconds = range
      ? fragmentAge + Math.max(0, range.end - video.currentTime)
      : null;
  }
  if (!userIsEditing && prompt.version !== serverPromptVersion) {
    promptInput.value = prompt.text;
    serverPromptVersion = prompt.version;
    updatePromptCount();
  }
  if (!userIsEditing) {
    if (prompt.applied_version === prompt.version) {
      saveStatus.textContent = `Prompt v${prompt.version} is live`;
    } else if (prompt.scheduled_version === prompt.version) {
      saveStatus.textContent = `Prompt v${prompt.version} is running`;
    } else {
      saveStatus.textContent = `Prompt v${prompt.version} saved · waiting for next request`;
    }
  }
}

async function pollState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    updateState(await response.json());
  } catch (error) {
    setText("pipeline-state", "OFFLINE");
  } finally {
    setTimeout(pollState, 500);
  }
}

function latestBufferedRange() {
  if (!video.buffered.length) return null;
  const index = video.buffered.length - 1;
  return { start: video.buffered.start(index), end: video.buffered.end(index) };
}

function updatePlaybackDiagnostics(range = latestBufferedRange()) {
  playbackDiagnostics.bufferedAheadSeconds = range
    ? Math.max(0, range.end - video.currentTime)
    : 0;
  playbackDiagnostics.appendQueueSegments = appendQueue.length;
  playbackDiagnostics.playbackRate = video.playbackRate;
}

function seekToBufferedLiveTarget(range) {
  const target = Math.max(range.start, range.end - TARGET_LIVE_LATENCY_SECONDS);
  if (Math.abs(video.currentTime - target) > 0.05) {
    liveSeekPending = true;
    video.currentTime = target;
    playbackDiagnostics.liveEdgeSeeks += 1;
    return true;
  }
  return false;
}

function requestVideoPlayback() {
  video.play().catch(() => setText("video-status", "Click video to play"));
}

function maintainLivePlayback() {
  const range = latestBufferedRange();
  updatePlaybackDiagnostics(range);
  if (!range) return;
  if (mediaSource && "setLiveSeekableRange" in mediaSource) {
    try {
      mediaSource.setLiveSeekableRange(range.start, range.end);
    } catch (_) {
      // Some browsers reject updates while the SourceBuffer is changing.
    }
  }

  const bufferedSpan = range.end - range.start;
  if (!playbackStarted) {
    if (bufferedSpan < STARTUP_BUFFER_SECONDS) return;
    playbackStarted = true;
    rebuffering = false;
    video.playbackRate = 1;
    if (!seekToBufferedLiveTarget(range)) requestVideoPlayback();
    return;
  }

  let bufferedAhead = range.end - video.currentTime;
  if (video.currentTime < range.start || bufferedAhead > MAX_LIVE_LATENCY_SECONDS) {
    seekToBufferedLiveTarget(range);
    bufferedAhead = range.end - video.currentTime;
  }
  if (rebuffering && bufferedAhead >= REBUFFER_RESUME_SECONDS) {
    rebuffering = false;
    video.playbackRate = 1;
    playbackDiagnostics.rebufferRecoveries += 1;
    if (!seekToBufferedLiveTarget(range)) requestVideoPlayback();
  }
  if (!rebuffering && bufferedAhead < LOW_BUFFER_SECONDS && video.playbackRate === 1) {
    video.playbackRate = BUFFER_RECOVERY_PLAYBACK_RATE;
    playbackDiagnostics.clockRecoveries += 1;
  } else if (
    !rebuffering
    && video.playbackRate < 1
    && bufferedAhead >= RECOVERED_BUFFER_SECONDS
  ) {
    video.playbackRate = 1;
  }
  updatePlaybackDiagnostics(range);
}

function pumpAppendQueue() {
  if (!sourceBuffer || sourceBuffer.updating) return;
  if (video.buffered.length && video.currentTime > 20) {
    const oldest = video.buffered.start(0);
    const keepFrom = video.currentTime - 10;
    if (oldest < keepFrom - 2) {
      sourceBuffer.remove(oldest, keepFrom);
      return;
    }
  }
  if (appendQueue.length === 0) return;
  try {
    sourceBuffer.appendBuffer(appendQueue.shift());
  } catch (error) {
    appendQueue = [];
    scheduleReconnect();
  }
}

function createMediaSource(mime) {
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  video.pause();
  video.playbackRate = 1;
  clearTimeout(rebufferTimer);
  liveSeekPending = false;
  playbackStarted = false;
  rebuffering = true;
  mediaSource = new MediaSource();
  mediaUrl = URL.createObjectURL(mediaSource);
  video.src = mediaUrl;
  mediaSource.addEventListener("sourceopen", () => {
    if (!MediaSource.isTypeSupported(mime)) {
      setText("video-status", "Unsupported browser codec");
      return;
    }
    sourceBuffer = mediaSource.addSourceBuffer(mime);
    sourceBuffer.mode = "segments";
    sourceBuffer.addEventListener("updateend", () => {
      maintainLivePlayback();
      pumpAppendQueue();
    });
    sourceBuffer.addEventListener("error", scheduleReconnect);
    pumpAppendQueue();
  }, { once: true });
}

function connectVideo() {
  clearTimeout(reconnectTimer);
  appendQueue = [];
  sourceBuffer = null;
  playbackStarted = false;
  rebuffering = true;
  clearTimeout(rebufferTimer);
  liveSeekPending = false;
  updatePlaybackDiagnostics(null);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  videoSocket = new WebSocket(`${protocol}//${location.host}/video`);
  videoSocket.binaryType = "arraybuffer";
  videoSocket.onopen = () => setText("video-status", "Video connected");
  videoSocket.onmessage = (event) => {
    if (typeof event.data === "string") {
      const info = JSON.parse(event.data);
      if (info.type === "stream-info") createMediaSource(info.mime);
      return;
    }
    appendQueue.push(event.data);
    if (appendQueue.length > MAX_APPEND_QUEUE_SEGMENTS) {
      playbackDiagnostics.appendQueueOverflows += 1;
      scheduleReconnect();
      return;
    }
    updatePlaybackDiagnostics();
    pumpAppendQueue();
  };
  videoSocket.onerror = () => setText("video-status", "Video error");
  videoSocket.onclose = () => {
    setText("video-status", "Video reconnecting");
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (videoSocket && videoSocket.readyState < WebSocket.CLOSING) videoSocket.close();
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connectVideo, 1200);
}

video.addEventListener("playing", () => {
  clearTimeout(rebufferTimer);
  $("video-empty").classList.add("hidden");
  $("video-dot").style.background = "#57e09b";
  setText("video-status", "Video live");
});
video.addEventListener("seeked", () => {
  if (!liveSeekPending) return;
  liveSeekPending = false;
  if (playbackStarted && !rebuffering) requestVideoPlayback();
});
video.addEventListener("waiting", () => {
  playbackDiagnostics.waitingEvents += 1;
  if (!playbackStarted || rebuffering || liveSeekPending || video.seeking) return;
  clearTimeout(rebufferTimer);
  rebufferTimer = setTimeout(() => {
    if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA || video.seeking) return;
    playbackDiagnostics.confirmedRebuffers += 1;
    rebuffering = true;
    video.pause();
    setText("video-status", "Video buffering");
  }, REBUFFER_DEBOUNCE_MS);
});
video.addEventListener("click", requestVideoPlayback);
promptInput.addEventListener("input", () => {
  userIsEditing = true;
  updatePromptCount();
  saveStatus.textContent = "Unsaved changes";
});

$("prompt-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  saveButton.disabled = true;
  saveStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/prompt", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptInput.value }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    serverPromptVersion = body.version;
    userIsEditing = false;
    saveStatus.textContent = `Prompt v${body.version} saved · waiting for next request`;
  } catch (error) {
    saveStatus.textContent = `Not saved: ${error.message}`;
  } finally {
    saveButton.disabled = false;
  }
});

updatePromptCount();
pollState();
connectVideo();
