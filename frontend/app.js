// Nav Recorder: full rear-camera video + raw IMU/orientation + push-to-talk clips + tags, uploaded while walking.
// One clock for everything: t_ms = performance.now() - t0 (event.timeStamp shares that time base).
// Format: nav/schemas/recording.py. API: /api/v1 (see /docs).
"use strict";

const API = NAV.API; // frontend/config.js
const VIDEO_TIMESLICE_MS = 2000, BATCH_MS = 1000, VIDEO_BPS = 5_000_000;
// WebM first: Chrome's MP4 muxer only emits data at keyframes (in practice at stop), WebM honours the timeslice.
// H.264 is copied into video.mp4 on the server without re-encoding. iOS Safari only offers MP4.
const VIDEO_MIMES = ["video/webm;codecs=h264", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm",
  "video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1", "video/mp4"];
const AUDIO_MIMES = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus", "audio/webm"];
// Live step detector on |accel incl. gravity|: low-pass, subtract a slow baseline, and fire when the bounce exceeds
// a fraction of its own recent size (a gentle handheld walk bounces ~0.3 m/s², a brisk one several m/s²).
// Display only: speed = steps in the last WINDOW_MS × step length; the real estimate is computed offline.
const STEP = { lpMs: 60, baseMs: 1500, varMs: 3000, frac: 0.4, floor: 0.1, minGapMs: 400, windowMs: 4000, minSteps: 3 };
const TAP_MS = 350;  // Talk pressed shorter than this toggles instead of push-to-talk

const $ = (id) => document.getElementById(id);
const r4 = (v) => (v == null ? null : Math.round(v * 1e4) / 1e4);
const pick = (list) => list.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";

let S = null;  // current session

// --- upload queue: sequential, retried with backoff; every upload is idempotent server-side ---------------
const queue = { tasks: [], running: false, failures: 0, bytes: 0 };

function enqueue(label, bytes, run) {
  queue.tasks.push({ label, bytes, run });
  queue.bytes += bytes;
  showUpload();
  if (!queue.running) drain();
}

async function drain() {
  queue.running = true;
  while (queue.tasks.length) {
    const task = queue.tasks[0];
    try {
      const r = await task.run();
      if (!r.ok && (r.status >= 500 || r.status === 408 || r.status === 429)) throw new Error(`HTTP ${r.status}`);
      if (!r.ok) console.error(task.label, r.status, await r.text());  // client error: retrying won't help
      queue.tasks.shift();
      queue.bytes -= task.bytes;
      queue.failures = 0;
    } catch (err) {
      queue.failures++;
      showUpload();
      await new Promise((ok) => setTimeout(ok, Math.min(1000 * 2 ** queue.failures, 15000)));
    }
    showUpload();
  }
  queue.running = false;
}

function showUpload() {
  const el = $("upload");
  const n = queue.tasks.length;
  const mb = (queue.bytes / 1e6).toFixed(1);
  el.textContent = queue.failures ? `upload retrying · ${n} waiting` : n > 2 ? `uploading · ${mb} MB` : "uploads ok";
  el.classList.toggle("warn", queue.failures > 0 || queue.bytes > 30e6);
}

const idle = () => new Promise((ok) => {
  const tick = () => (queue.tasks.length ? setTimeout(tick, 200) : ok());
  tick();
});

const api = (method, path, body, headers = {}) =>
  NAV.fetch(API + path, { method, body, headers, cache: "no-store" });
const json = (method, path, obj) => api(method, path, JSON.stringify(obj), { "Content-Type": "application/json" });

// --- session helpers ---------------------------------------------------------------------------------------
const now = () => performance.now() - S.t0;
const at = (ts) => r4(ts - S.t0);
function ev(type, data = {}) { S.events.push({ t_ms: r4(now()), type, data }); }

function say(text) { $("status").textContent = text; }

function buzz(ms = 40) { try { navigator.vibrate?.(ms); } catch {} }

// --- sensors -----------------------------------------------------------------------------------------------
function onMotion(e) {
  const a = e.accelerationIncludingGravity || {}, l = e.acceleration || {}, g = e.rotationRate || {};
  if (a.x == null && g.alpha == null) return;
  // rotationRate: alpha = around z, beta = around x, gamma = around y (deg/s)
  S.imu.push({ t_ms: at(e.timeStamp), ax: r4(a.x), ay: r4(a.y), az: r4(a.z), lax: r4(l.x), lay: r4(l.y), laz: r4(l.z),
    gx: r4(g.beta), gy: r4(g.gamma), gz: r4(g.alpha), interval_ms: e.interval ?? null });
  if (a.x != null) detectStep(Math.hypot(a.x, a.y, a.z), e.timeStamp);
}

function detectStep(mag, ts) {
  const st = S.steps;
  if (st.last == null) { st.lp = st.base = mag; st.last = ts; return; }
  const dt = Math.max(ts - st.last, 1), k = (tau) => 1 - Math.exp(-dt / tau);  // time constants, any sensor rate
  st.last = ts;
  st.lp += k(STEP.lpMs) * (mag - st.lp);
  st.base += k(STEP.baseMs) * (st.lp - st.base);
  const d = st.lp - st.base;
  st.var += k(STEP.varMs) * (d * d - st.var);
  const hi = Math.max(STEP.floor, STEP.frac * Math.sqrt(st.var));
  if (st.armed && d > hi) {
    st.armed = false;
    if (!st.times.length || ts - st.times.at(-1) > STEP.minGapMs) {
      st.times.push(ts);
      st.count++;
      const { cadence, speed } = walk(ts);
      S.events.push({ t_ms: at(ts), type: "step", data: { cadence: r4(cadence), speed_mps: r4(speed) } });
    }
  } else if (!st.armed && d < -0.3 * hi) {
    st.armed = true;
  }
}

function walk(ts) {
  const st = S.steps;
  st.times = st.times.filter((t) => ts - t < STEP.windowMs);
  const cadence = st.times.length >= STEP.minSteps ? st.times.length / (STEP.windowMs / 1000) : 0;
  return { cadence, speed: cadence * S.stepLength };
}

// Heading of the rear-camera axis (device -z) from W3C Euler angles (Z-X'-Y'', earth frame x=E, y=N, z=up).
function cameraHeading(alpha, beta, gamma) {
  const r = Math.PI / 180;
  const cA = Math.cos(alpha * r), sA = Math.sin(alpha * r), cB = Math.cos(beta * r), sB = Math.sin(beta * r);
  const cG = Math.cos(gamma * r), sG = Math.sin(gamma * r);
  const east = -cA * sG - sA * sB * cG, north = -sA * sG + cA * sB * cG;
  if (Math.hypot(east, north) < 0.2) return null;  // camera pointing at floor/ceiling
  return (Math.atan2(east, north) / r + 360) % 360;
}

function onOrientation(e) {
  if (e.alpha == null && e.webkitCompassHeading == null) return;
  const absolute = e.type === "deviceorientationabsolute" || !!e.absolute;
  let heading = null;
  if (e.webkitCompassHeading != null) heading = e.webkitCompassHeading;  // iOS: already true-north-ish
  else if (absolute && e.alpha != null) heading = cameraHeading(e.alpha, e.beta, e.gamma);
  S.orientation.push({ t_ms: at(e.timeStamp), alpha: r4(e.alpha), beta: r4(e.beta), gamma: r4(e.gamma), absolute,
    compass_deg: r4(e.webkitCompassHeading), compass_acc: r4(e.webkitCompassAccuracy), heading_deg: r4(heading),
    screen_angle: screen.orientation?.angle ?? window.orientation ?? null });
  if (heading != null) S.heading = heading;
}

function onVisibility() {
  ev(document.hidden ? "hidden" : "visible");
  if (!document.hidden) lockScreen();
}

function onRotate() { ev("orientation_change", { angle: screen.orientation?.angle ?? null }); }

async function lockScreen() {
  try { S.wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
}

const orientationEvent = "ondeviceorientationabsolute" in window ? "deviceorientationabsolute" : "deviceorientation";

function listen(on) {
  const f = on ? "addEventListener" : "removeEventListener";
  window[f]("devicemotion", onMotion);
  window[f](orientationEvent, onOrientation);
  document[f]("visibilitychange", onVisibility);
  screen.orientation?.[f]("change", onRotate);
}

// --- sample batches ----------------------------------------------------------------------------------------
function flush() {
  if (!S.imu.length && !S.orientation.length && !S.events.length) return;
  const batch = { seq: S.batchSeq++, imu: S.imu, orientation: S.orientation, events: S.events };
  S.imu = []; S.orientation = []; S.events = [];
  const body = JSON.stringify(batch);
  enqueue(`samples ${batch.seq}`, body.length, () => api("POST", `/recordings/${S.id}/samples`, body,
    { "Content-Type": "application/json" }));
}

function tickUi() {
  const s = Math.floor(now() / 1000);
  $("clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  $("steps").textContent = S.steps.count;
  $("speed").textContent = walk(performance.now()).speed.toFixed(2);
  $("heading").textContent = S.heading == null ? "—" : `${Math.round(S.heading)}°`;
}

// --- push-to-talk ------------------------------------------------------------------------------------------
function talkDown(e) {
  if (!S?.audio || S.stopping) return;
  e.preventDefault();
  if (S.voice) { S.voiceToggle = false; return; }  // tap while toggled on: stop on release
  S.pressAt = performance.now();
  startVoice();
}

function talkUp(e) {
  if (!S?.voice) return;
  e.preventDefault();
  if (S.voiceToggle === null && performance.now() - S.pressAt < TAP_MS) {
    S.voiceToggle = true;  // short tap: keep talking until the next tap
    say("Talking… tap Talk again to stop");
    return;
  }
  if (S.voiceToggle) return;
  stopVoice();
}

function startVoice() {
  const clip = S.clipSeq++;
  const rec = new MediaRecorder(S.audio, S.audioMime ? { mimeType: S.audioMime } : {});
  const parts = [];
  const t_start = r4(now());
  rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
  rec.onstop = () => {
    const t_end = r4(now());
    const blob = new Blob(parts, { type: rec.mimeType || S.audioMime || "audio/webm" });
    enqueue(`voice ${clip}`, blob.size, () =>
      api("PUT", `/recordings/${S.id}/voice/${clip}?t_start_ms=${t_start}&t_end_ms=${t_end}`, blob,
        { "Content-Type": blob.type }));
    S.events.push({ t_ms: t_end, type: "voice_end", data: { clip } });
  };
  S.voiceDone = new Promise((ok) => rec.addEventListener("stop", ok));
  rec.start();
  S.voice = rec;
  S.voiceToggle = null;
  ev("voice_start", { clip });
  $("talk").setAttribute("aria-pressed", "true");
  $("talk").textContent = "Talking…";
  buzz(30);
  say("Talking…");
}

function stopVoice() {
  if (!S.voice) return;
  S.voice.stop();
  S.voice = null;
  S.voiceToggle = null;
  $("talk").setAttribute("aria-pressed", "false");
  $("talk").textContent = "Talk";
  buzz(30);
  say("Voice note saved");
}

function onTag(e) {
  const btn = e.currentTarget, tag = btn.dataset.tag;
  ev("mark", { tag });
  buzz(60);
  btn.classList.add("flash");
  setTimeout(() => btn.classList.remove("flash"), 400);
  say(`Tagged ${btn.textContent.toLowerCase()}`);
}

// --- start / stop ------------------------------------------------------------------------------------------
async function start() {
  const btn = $("start"), msg = $("setup-msg");
  btn.disabled = true;
  msg.textContent = "";
  let video = null, audio = null;
  const place = $("place-pick").value || $("place").value.trim();
  if (!place) {
    msg.textContent = "Pick the place, or name the new place you are mapping.";
    $("place").focus();
    btn.disabled = false;
    return;
  }
  try {
    // iOS (and newer Chrome) ask for motion permission, requested directly inside the tap
    const warnings = [];
    for (const E of [window.DeviceMotionEvent, window.DeviceOrientationEvent]) {
      const granted = typeof E?.requestPermission !== "function" || (await E.requestPermission().catch(() => "denied")) === "granted";
      if (!granted && !warnings.length) warnings.push("Motion sensors denied: recording video and voice only.");
    }
    if (!window.MediaRecorder) throw new Error("This browser cannot record video (no MediaRecorder)");
    video = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
      facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } } });
    try { audio = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: true, echoCancellation: false } }); }
    catch { warnings.push("No microphone: Talk is disabled, tags still work."); }

    const track = video.getVideoTracks()[0], cs = track.getSettings();
    const mime = pick(VIDEO_MIMES);
    const t0 = performance.now();
    const res = await json("POST", "/recordings", {
      place, notes: $("notes").value.trim(), holding: $("holding").value,
      step_length_m: Number($("step").value) || 0.7, t0_epoch_ms: Math.round(performance.timeOrigin + t0),
      device: { user_agent: navigator.userAgent, platform: navigator.userAgentData?.platform || navigator.platform || null,
        screen_w: screen.width, screen_h: screen.height, pixel_ratio: devicePixelRatio },
      camera: { width: cs.width, height: cs.height, fps: cs.frameRate ?? null, mime: mime || "video/webm",
        facing: cs.facingMode ?? null, label: track.label || null },
    });
    if (!res.ok) throw new Error(`Server refused the recording (HTTP ${res.status})`);
    const manifest = await res.json();

    S = { id: manifest.rec_id, t0, stepLength: manifest.step_length_m, video, audio, audioMime: pick(AUDIO_MIMES),
      imu: [], orientation: [], events: [], batchSeq: 0, videoSeq: 0, clipSeq: 0, heading: null,
      steps: { last: null, lp: 0, base: 0, var: 0.04, armed: true, times: [], count: 0 }, voice: null, voiceToggle: null };
    ev("start", { rec_id: S.id });

    const recorder = new MediaRecorder(video, { ...(mime ? { mimeType: mime } : {}), videoBitsPerSecond: VIDEO_BPS,
      videoKeyFrameIntervalDuration: 1000 });
    recorder.onstart = () => ev("video_start", { mime: recorder.mimeType });
    recorder.ondataavailable = (e) => {
      if (!e.data.size) return;
      const seq = S.videoSeq++;
      ev("video_chunk", { seq, bytes: e.data.size });
      enqueue(`video ${seq}`, e.data.size, () => api("PUT", `/recordings/${S.id}/video/${seq}`, e.data,
        { "Content-Type": "application/octet-stream" }));
    };
    S.stopped = new Promise((ok) => (recorder.onstop = ok));
    S.recorder = recorder;
    recorder.start(VIDEO_TIMESLICE_MS);

    listen(true);
    S.timers = [setInterval(flush, BATCH_MS), setInterval(tickUi, 250)];
    await lockScreen();

    $("preview").srcObject = video;
    $("talk").disabled = !audio;
    $("setup").hidden = true;
    $("rec").hidden = false;
    say(["Recording", ...warnings].join(". "));
    warnings.forEach((w) => ev("error", { message: w }));
    buzz(80);
  } catch (err) {
    console.error(err);
    msg.textContent = err.message || String(err);
    [video, audio].forEach((s) => s?.getTracks().forEach((t) => t.stop()));
    S = null;
  } finally {
    btn.disabled = false;
  }
}

async function stop() {
  if (!S || S.stopping) return;
  S.stopping = true;
  $("stop").disabled = true;
  const voiceDone = S.voice && S.voiceDone;
  stopVoice();
  await voiceDone;  // its upload is queued
  ev("stop");
  S.recorder.stop();
  await S.stopped;  // last video chunk has been queued
  listen(false);
  S.timers.forEach(clearInterval);
  flush();
  S.video.getTracks().forEach((t) => t.stop());
  S.audio?.getTracks().forEach((t) => t.stop());
  S.wakeLock?.release().catch(() => {});

  $("rec").hidden = true;
  $("done").hidden = false;
  const status = $("done-status");
  status.textContent = `Uploading… (${queue.tasks.length} left)`;
  const poll = setInterval(() => (status.textContent = `Uploading… (${queue.tasks.length} left)`), 500);
  await idle();
  clearInterval(poll);
  const fin = { duration_ms: r4(now()), video_chunks: S.videoSeq, sample_batches: S.batchSeq, voice_clips: S.clipSeq };
  let m = null;
  for (let i = 0; !m; i++) {
    const r = await json("POST", `/recordings/${S.id}/finish`, fin).catch(() => null);
    if (r?.ok) m = await r.json();
    else await new Promise((ok) => setTimeout(ok, Math.min(1000 * 2 ** i, 15000)));
  }
  status.textContent = "Processing on the server…";
  while (m.status === "finalizing" || m.status === "recording") {
    await new Promise((ok) => setTimeout(ok, 1500));
    const r = await api("GET", `/recordings/${m.rec_id}`).catch(() => null);
    if (r?.ok) m = await r.json();
  }
  status.textContent = { complete: "Complete ✓", incomplete: "Saved with gaps", failed: "Processing failed" }[m.status];
  const v = m.video;
  $("done-summary").textContent = [
    `place     ${m.place}`,
    `id        ${m.rec_id}`,
    `duration  ${(m.duration_ms / 1000).toFixed(0)} s`,
    v ? `video     ${v.width}×${v.height} ${v.codec}, ${v.duration_s?.toFixed(1)} s` : "video     —",
    `voice     ${m.voice.length} clips`,
    `tags      ${m.counts.marks}`,
    `imu       ${m.counts.imu} samples, orientation ${m.counts.orientation}`,
    ...(m.gaps.length ? [`gaps      ${m.gaps.join(", ")}`] : []),
    ...(m.error ? [`error     ${m.error}`] : []),
  ].join("\n");
  buzz(120);
  S = null;
}

async function loadRecent() {
  const pr = await api("GET", "/places").catch(() => null);
  if (pr?.ok) {
    // Places newest first; the one picked last on this phone (or the newest) is preselected, so a new walk of the
    // same building joins its map without typing (a typo would start a separate map).
    const places = await pr.json(), pick = $("place-pick");
    let last = null;
    try { last = localStorage.getItem("nav.place"); } catch {}
    pick.replaceChildren(...places.map((p) => Object.assign(document.createElement("option"), {
      value: p.place, textContent: `${p.place} (${p.recordings} walk${p.recordings === 1 ? "" : "s"})` })),
      Object.assign(document.createElement("option"), { value: "", textContent: "New place…" }));
    pick.value = places.some((p) => p.place === last) ? last : places[0]?.place ?? "";
    placeChanged();
  }
  const r = await api("GET", "/recordings").catch(() => null);
  if (!r?.ok) return;
  const list = await r.json();
  $("recent").replaceChildren(...list.slice(0, 10).map((s) => {
    const li = document.createElement("li");
    const dur = s.duration_ms ? `${Math.round(s.duration_ms / 1000)} s` : "—";
    li.textContent = `${s.place} · ${dur} · ${s.status}`;
    return li;
  }));
}

function placeChanged() {
  const isNew = !$("place-pick").value;
  $("place").hidden = !isNew;
  try { if (!isNew) localStorage.setItem("nav.place", $("place-pick").value); } catch {}
}

// --- wiring ------------------------------------------------------------------------------------------------
$("place-pick").onchange = () => { placeChanged(); if (!$("place-pick").value) $("place").focus(); };
placeChanged();
$("start").onclick = start;
$("stop").onclick = stop;
$("again").onclick = () => { $("done").hidden = true; $("setup").hidden = false; $("done-summary").textContent = ""; loadRecent(); };
document.querySelectorAll(".tag").forEach((b) => (b.onclick = onTag));
const talk = $("talk");
talk.addEventListener("pointerdown", (e) => { talk.setPointerCapture?.(e.pointerId); talkDown(e); });
talk.addEventListener("pointerup", talkUp);
talk.addEventListener("pointercancel", talkUp);
talk.addEventListener("contextmenu", (e) => e.preventDefault());
talk.addEventListener("keydown", (e) => { if ((e.key === " " || e.key === "Enter") && !e.repeat) talkDown(e); });
talk.addEventListener("keyup", (e) => { if (e.key === " " || e.key === "Enter") talkUp(e); });
window.addEventListener("beforeunload", (e) => { if (S || queue.tasks.length) e.preventDefault(); });
loadRecent();
