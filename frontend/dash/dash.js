// Nav Recordings dashboard: browse walks by place; video, tags, voice and sensors on one timeline.
// Time base everywhere is the recording clock in seconds (t_ms / 1000); video time 0 is the video_start event.
"use strict";

const API = NAV.API; // frontend/config.js
document.getElementById("api-docs").href = NAV.docs;
const TAGS = { door: "Door", junction: "Junction", stairs: "Stairs", lift: "Lift", destination: "Destination" };
const HOLDING = { handheld_portrait: "in hand, upright", handheld_landscape: "in hand, sideways", chest_mount: "chest mount",
  lanyard: "lanyard", other: "other" };

const $ = (sel, el = document) => el.querySelector(sel);
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...kids.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}
async function getJSON(path) {
  const r = await NAV.fetch(API + path, { cache: "no-store" });
  if (!r.ok) throw Object.assign(new Error(`${path}: HTTP ${r.status}`), { status: r.status });
  return r.json();
}
const fileUrl = (id, path) => NAV.url(`${API}/recordings/${id}/files/${path}`);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const pad2 = (n) => String(n).padStart(2, "0");
function fmtDur(ms) {
  if (ms == null) return "–";
  const s = Math.round(ms / 1000);
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}` : `${Math.floor(s / 60)}:${pad2(s % 60)}`;
}
const fmtT = (sec) => (sec == null || !isFinite(sec) ? "–" : `${Math.floor(Math.max(0, sec) / 60)}:${(Math.max(0, sec) % 60).toFixed(1).padStart(4, "0")}`);
const fmtDate = (iso) => new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b >= 1e3 ? `${(b / 1e3).toFixed(0)} kB` : `${b} B`);

// --- list --------------------------------------------------------------------------------------------------
let recordings = [];
let cur = null;  // the open recording: { m, dur, vs, events, marks, voice, charts, video, ... }

async function loadList() {
  try {
    recordings = await getJSON("/recordings");
    renderList();
  } catch (err) {
    $("#list").replaceChildren(h("p", { class: "alert" }, `Could not load recordings: ${err.message}`));
  }
}

function renderList() {
  const q = $("#filter").value.trim().toLowerCase();
  const groups = new Map();
  for (const s of recordings) {
    const key = s.place.toLowerCase();
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  }
  const shown = [...groups.values()].filter((g) => !q || g[0].place.toLowerCase().includes(q));
  $("#list").replaceChildren(...(shown.length ? shown.map((g) => h("section", { class: "place" },
    h("h2", {}, g[0].place, h("span", { class: "muted" },
      `${g.length} walk${g.length > 1 ? "s" : ""} · ${fmtDur(g.reduce((a, s) => a + (s.duration_ms || 0), 0))}`)),
    h("ul", {}, g.map((s) => h("li", {}, h("a", {
      class: `rec-link${s.rec_id === cur?.m.rec_id ? " active" : ""}`, href: `#${s.rec_id}`,
      "aria-current": s.rec_id === cur?.m.rec_id ? "page" : null,
    },
      h("span", {}, fmtDate(s.created_at_utc)),
      h("span", { class: "dur" }, fmtDur(s.duration_ms)),
      h("span", { class: "muted small" }, `${s.counts.marks} tags · ${s.counts.voice_clips} voice`),
      h("span", { class: `badge ${s.status}` }, s.status)))))))
    : [h("p", { class: "muted" }, recordings.length ? "No place matches." : "No recordings yet.")]));
}

// --- one recording -----------------------------------------------------------------------------------------
async function openRecording(id) {
  teardown();
  const main = $("#detail");
  if (!id) {
    main.replaceChildren(h("div", { class: "empty" }, h("div", {},
      h("p", {}, "Pick a recording on the left."),
      h("p", { class: "small" }, "Video, tags, voice notes and sensors are shown on one timeline: click anywhere on it to jump."))));
    renderList();
    return;
  }
  main.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  let m, files, events;
  try {
    [m, files, events] = await Promise.all([getJSON(`/recordings/${id}`), getJSON(`/recordings/${id}/files`), loadEvents(id)]);
  } catch (err) {
    main.replaceChildren(h("p", { class: "alert" }, err.status === 404 ? `Recording ${id} not found.` : err.message));
    return;
  }
  const vs = (events.find((e) => e.type === "video_start")?.t_ms ?? 0) / 1000;
  const lastT = events.reduce((a, e) => Math.max(a, e.t_ms), 0) / 1000;
  const dur = Math.max((m.duration_ms ?? 0) / 1000, lastT, 1);
  cur = {
    m, dur, vs, events, t: 0, charts: [], timers: [],
    marks: events.filter((e) => e.type === "mark").map((e) => ({ t: e.t_ms / 1000, tag: e.data.tag })),
    voice: m.voice.map((c) => ({ ...c, t0: c.t_start_ms / 1000, t1: c.t_end_ms / 1000, el: null, synced: false })),
  };
  renderList();
  main.replaceChildren(header(m), meta(m, events), player(m), timeline(), h("div", { class: "charts", id: "charts" }), filesBox(m, files));
  loadSensors(m.rec_id);
  update();
}

async function loadEvents(id) {
  const r = await NAV.fetch(fileUrl(id, "events.jsonl"), { cache: "no-store" });
  if (r.status === 404) return [];  // not finalized yet
  if (!r.ok) throw new Error(`events.jsonl: HTTP ${r.status}`);
  return (await r.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l)).sort((a, b) => a.t_ms - b.t_ms);
}

function teardown() {
  if (!cur) return;
  cur.charts.forEach((u) => u.destroy());
  cur.timers.forEach(cancelAnimationFrame);
  cur.voice.forEach((c) => c.el?.pause());
  cur.resize?.disconnect();
  cur = null;
}

function header(m) {
  return h("div", { class: "head" },
    h("h1", {}, m.place),
    h("span", { class: "muted" }, fmtDate(m.created_at_utc)),
    h("span", { class: `badge ${m.status}` }, m.status),
    h("span", { class: "mono muted" }, m.rec_id));
}

function meta(m, events) {
  const steps = events.filter((e) => e.type === "step").length;
  const dev = m.device.user_agent.match(/\(([^)]+)\)/)?.[1] ?? m.device.platform ?? "";
  const v = m.video;
  const rows = [
    ["Duration", fmtDur(m.duration_ms)],
    ["Phone", `${HOLDING[m.holding] ?? m.holding} · step ${m.step_length_m} m`],
    ["Video", v ? `${v.width}×${v.height} ${v.codec}, ${v.duration_s?.toFixed(1)} s` : "none"],
    ["Sensors", `${m.counts.imu} motion · ${m.counts.orientation} orientation`],
    ["Tags · voice · steps", `${m.counts.marks} · ${m.voice.length} · ${steps} (live)`],
    ["Device", dev],
    ["By", m.created_by ?? "–"],
    ["Notes", m.notes || "–"],
  ];
  if (m.gaps.length) rows.push(["Gaps", m.gaps.join(", ")]);
  if (m.error) rows.push(["Error", m.error]);
  return h("dl", { class: "meta" }, rows.map(([k, val]) =>
    h("div", {}, h("dt", {}, k), h("dd", { class: k === "Gaps" || k === "Error" ? "alert" : null }, val))));
}

// --- video + notes ---------------------------------------------------------------------------------------
function player(m) {
  let box;
  if (m.video) {
    const video = h("video", { src: fileUrl(m.rec_id, m.video.file), controls: true, playsinline: true, preload: "metadata" });
    cur.video = video;
    const loop = () => { update(); cur && !video.paused && cur.timers.push(requestAnimationFrame(loop)); };
    video.addEventListener("play", loop);
    video.addEventListener("pause", update);
    video.addEventListener("seeked", () => { cur.voice.forEach(stopSynced); update(); });
    video.addEventListener("timeupdate", () => video.paused && update());
    box = h("div", { class: "video-box" }, video);
  } else {
    box = h("div", { class: "video-box novideo" }, "No video in this recording.");
  }
  const voiceToggle = h("input", { type: "checkbox", checked: true, id: "voice-sync" });
  voiceToggle.addEventListener("change", () => cur.voice.forEach(stopSynced));
  const left = h("div", {}, box, h("div", { class: "clock" },
    h("span", { id: "clock" }, "0:00.0"),
    m.voice.length ? h("label", { title: "Play each voice note when the video reaches it" }, voiceToggle, "voice with video") : null));
  return h("div", { class: "player" }, left, notesPanel());
}

function notesPanel() {
  const items = [];
  for (const e of cur.marks) {
    items.push({ t: e.t, kind: "mark", node: [h("b", {}, TAGS[e.tag] ?? e.tag)] });
  }
  for (const c of cur.voice) {
    c.el = h("audio", { src: fileUrl(cur.m.rec_id, c.file), controls: true, preload: "none", onclick: (ev) => ev.stopPropagation() });
    c.el.addEventListener("play", () => { if (!c.synced) seek(c.t0 + c.el.currentTime, { keepVoice: true }); });
    items.push({ t: c.t0, end: c.t1, kind: "voice", node: [h("b", {}, `Voice note ${c.clip + 1}`),
      h("span", { class: "muted small" }, ` ${(c.t1 - c.t0).toFixed(1)} s`), c.el] });
  }
  for (const e of cur.events) {
    if (e.type === "error") items.push({ t: e.t_ms / 1000, kind: "error", node: [h("b", {}, "Warning "), e.data.message ?? ""] });
    if (e.type === "hidden") items.push({ t: e.t_ms / 1000, kind: "hidden", node: [h("b", {}, "App in background"),
      h("span", { class: "muted small" }, " sensors and camera may pause")] });
  }
  items.sort((a, b) => a.t - b.t);
  cur.noteItems = items;
  const list = h("ul", { class: "notes" }, items.map((it) => (it.li = h("li", { class: `kind-${it.kind}`, onclick: () => seek(it.t) },
    h("span", { class: "t" }, fmtT(it.t)), h("span", {}, it.node)))));
  return h("section", { class: "panel" },
    h("h3", {}, "Tags and voice notes", h("span", { class: "muted" }, String(items.length))),
    items.length ? list : h("p", { class: "muted", style: "padding: 0 12px" }, "No tags or voice notes."));
}

// --- timeline strip ---------------------------------------------------------------------------------------
function timeline() {
  const pct = (t) => `${(clamp(t, 0, cur.dur) / cur.dur) * 100}%`;
  const width = (a, b) => `${((clamp(b, 0, cur.dur) - clamp(a, 0, cur.dur)) / cur.dur) * 100}%`;
  const lane = (name, ...kids) => h("div", { class: "lane" }, h("span", { class: "name" }, name), h("div", { class: "track" }, kids));
  const hidden = [];
  let since = null;
  for (const e of cur.events) {
    if (e.type === "hidden") since = e.t_ms / 1000;
    if (e.type === "visible" && since != null) { hidden.push([since, e.t_ms / 1000]); since = null; }
  }
  if (since != null) hidden.push([since, cur.dur]);
  const steps = cur.events.filter((e) => e.type === "step");
  const lanes = [
    lane("Tags", cur.marks.map((e) => h("span", { class: "mk", style: `left:${pct(e.t)}`, title: `${TAGS[e.tag] ?? e.tag} · ${fmtT(e.t)}` },
      (TAGS[e.tag] ?? e.tag)[0]))),
    lane("Voice", cur.voice.map((c) => h("span", { class: "span voice", style: `left:${pct(c.t0)};width:${width(c.t0, c.t1)}`,
      title: `Voice note ${c.clip + 1} · ${fmtT(c.t0)}–${fmtT(c.t1)}` }))),
    lane("Steps", steps.map((e) => h("span", { class: "tick", style: `left:${pct(e.t_ms / 1000)}` }))),
  ];
  if (hidden.length) lanes.push(lane("Background", hidden.map(([a, b]) =>
    h("span", { class: "span hidden-page", style: `left:${pct(a)};width:${width(a, b)}`, title: `${fmtT(a)}–${fmtT(b)}` }))));
  const head = h("div", { class: "playhead", id: "tl-head" });
  const overlay = h("div", { class: "overlay" }, head);
  const box = h("div", { class: "lanes" }, lanes, overlay);
  box.addEventListener("click", (ev) => {
    const r = overlay.getBoundingClientRect();
    if (ev.clientX >= r.left) seek(((ev.clientX - r.left) / r.width) * cur.dur);
  });
  const step = niceStep(cur.dur / 8);
  const ticks = [];
  for (let t = 0; t <= cur.dur + 1e-6; t += step) ticks.push(h("span", { style: `left:${pct(t)}` }, fmtDur(t * 1000)));
  return h("section", { class: "panel timeline" }, box, h("div", { class: "axis" }, ticks));
}

function niceStep(raw) {
  for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]) if (s >= raw) return s;
  return 3600;
}

// --- sensor charts ----------------------------------------------------------------------------------------
async function loadSensors(id) {
  const box = $("#charts");
  const [imu, ori] = await Promise.all(["imu", "orientation"].map((s) => getJSON(`/recordings/${id}/series/${s}`).catch(() => null)));
  if (!cur || cur.m.rec_id !== id) return;  // user moved on
  if (!imu && !ori) {
    box.append(h("p", { class: "muted" }, "No sensor data (the recording is not finalized, or motion sensors were denied)."));
    return;
  }
  const sec = (col) => col.map((v) => v / 1000);
  // pad every chart to the whole recording so all x axes line up: [0 … dur] with empty values at the ends
  const padded = (x, ...ys) => [[0, ...x, cur.dur], ...ys.map((y) => [null, ...y, null])];
  const c1 = css("--c1"), c2 = css("--c2"), c3 = css("--c3"), fg = css("--fg"), muted = css("--muted");
  const line = (label, stroke, extra = {}) => ({ label, stroke, width: 1, ...extra });
  if (imu?.t_ms.length) {
    const x = sec(imu.t_ms);
    const mag = imu.ax.map((v, i) => (v == null ? null : Math.hypot(v, imu.ay[i], imu.az[i])));
    addChart(box, "Acceleration incl. gravity (m/s²)", "Click to jump the video there; drag to zoom, double-click to reset. Click a legend entry to hide it.",
      padded(x, mag, imu.ax, imu.ay, imu.az),
      [line("|a|", fg, { width: 1.25 }), line("x", c1), line("y", c2), line("z", c3)]);
    addChart(box, "Rotation rate (deg/s)", null, padded(x, imu.gx, imu.gy, imu.gz),
      [line("x (beta)", c1), line("y (gamma)", c2), line("z (alpha)", c3)]);
  }
  if (ori?.t_ms.length) {
    const dots = (label, stroke) => ({ label, stroke, fill: stroke, paths: () => null, points: { show: true, size: 2.5, stroke, fill: stroke } });
    addChart(box, "Heading (deg, from the phone's compass)", "Indoors the compass is disturbed by steel and wiring; offline fusion uses the gyro.",
      padded(sec(ori.t_ms), ori.heading_deg), [dots("heading", c3)], { y: { range: [0, 360] } });
  }
  const steps = cur.events.filter((e) => e.type === "step");
  if (steps.length) {
    addChart(box, "Live speed shown on the phone (m/s)", "Rough step-count estimate, display only. The first recordings (up to 9 Oct 2026) used a step detector that missed gentle walks, so they show almost nothing here.",
      padded(steps.map((e) => e.t_ms / 1000), steps.map((e) => e.data.speed_mps ?? null)),
      [{ label: "speed", stroke: muted, width: 1.5, paths: uPlot.paths.stepped({ align: 1 }), points: { show: true, size: 4 } }],
      { y: { range: (u, lo, hi) => [0, Math.max(1.5, hi ?? 0)] } });
  }
  cur.resize = new ResizeObserver(() => cur?.charts.forEach((u) => u.setSize({ width: chartWidth(u.root.parentElement), height: u.height })));
  cur.resize.observe(box);
}

const chartWidth = (el) => Math.max(200, el.clientWidth - 16);

function addChart(box, title, hint, data, series, scales = {}) {
  const wrap = h("section", { class: "panel chart" });
  box.append(wrap);
  const grid = { stroke: css("--line"), width: 1 };
  const axis = { stroke: css("--muted"), grid, ticks: grid };
  const u = new uPlot({
    title, width: chartWidth(wrap), height: 190,
    scales: { x: { time: false }, ...scales },
    axes: [{ ...axis, values: (u, vals) => vals.map((v) => fmtDur(v * 1000)) }, { ...axis, size: 48 }],
    series: [{ label: "t", value: (u, v) => fmtT(v) }, ...series],
    cursor: { sync: { key: "rec" }, drag: { x: true, y: false } },
    hooks: {
      draw: [drawOverlay],
      setScale: [(u, key) => {
        if (key !== "x" || cur.syncingZoom) return;
        cur.syncingZoom = true;
        cur.charts.forEach((o) => o !== u && o.setScale("x", { min: u.scales.x.min, max: u.scales.x.max }));
        cur.syncingZoom = false;
      }],
    },
  }, data, wrap);
  if (hint) wrap.insertBefore(h("p", { class: "hint" }, hint), u.root.nextSibling);
  let down = null;
  u.over.addEventListener("mousedown", (e) => (down = e.clientX));
  u.over.addEventListener("click", (e) => {
    if (down != null && Math.abs(e.clientX - down) > 3) return;  // that was a zoom drag
    const t = u.posToVal(u.cursor.left, "x");
    if (isFinite(t)) seek(t);
  });
  cur.charts.push(u);
}

function drawOverlay(u) {
  if (!cur) return;
  const { ctx, bbox } = u, dpr = devicePixelRatio || 1;
  const vline = (t) => {
    const x = Math.round(u.valToPos(t, "x", true));
    if (x < bbox.left || x > bbox.left + bbox.width) return;
    ctx.beginPath(); ctx.moveTo(x, bbox.top); ctx.lineTo(x, bbox.top + bbox.height); ctx.stroke();
  };
  ctx.save();
  ctx.lineWidth = dpr; ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.strokeStyle = css("--tag");
  cur.marks.forEach((e) => vline(e.t));
  ctx.setLineDash([]); ctx.lineWidth = 2 * dpr; ctx.strokeStyle = css("--accent");
  vline(cur.t);
  ctx.restore();
}

// --- files ------------------------------------------------------------------------------------------------
function filesBox(m, files) {
  const main = files.filter((f) => !f.path.startsWith("raw/"));
  const raw = files.length - main.length;
  return h("details", { class: "panel files" },
    h("summary", {}, `Files (${main.length})`, raw ? h("span", { class: "muted" }, ` + ${raw} raw uploads`) : null),
    h("ul", {}, main.map((f) => h("li", {}, h("a", { href: fileUrl(m.rec_id, f.path), download: f.path.replaceAll("/", "_") }, f.path),
      h("span", { class: "muted" }, fmtBytes(f.bytes))))));
}

// --- playhead -----------------------------------------------------------------------------------------------
function seek(t, { keepVoice = false } = {}) {
  if (!cur) return;
  t = clamp(t, 0, cur.dur);
  if (!keepVoice) cur.voice.forEach((c) => c.el && !c.synced && !c.el.paused && c.el.pause());
  const v = cur.video;
  if (v && isFinite(v.duration || NaN)) v.currentTime = clamp(t - cur.vs, 0, v.duration);
  else if (v) v.currentTime = Math.max(0, t - cur.vs);
  cur.t = t;  // before metadata loads, or for times before the video started
  update();
}

function update() {
  if (!cur) return;
  const v = cur.video;
  if (v && (v.currentTime > 0 || !v.paused)) cur.t = v.currentTime + cur.vs;
  const t = cur.t;
  $("#clock") && ($("#clock").textContent = `${fmtT(t)} / ${fmtT(cur.dur)}`);
  const head = $("#tl-head");
  if (head) head.style.left = `${(clamp(t, 0, cur.dur) / cur.dur) * 100}%`;
  cur.charts.forEach((u) => u.redraw(false));
  for (const it of cur.noteItems ?? []) it.li.classList.toggle("now", t >= it.t - 0.05 && t < (it.end ?? it.t + 1.5));
  syncVoice(t, v && !v.paused);
}

// The video has no sound: voice notes are separate clips, played here when the video reaches them.
function syncVoice(t, playing) {
  const on = $("#voice-sync")?.checked;
  for (const c of cur.voice) {
    if (!c.el) continue;
    const inside = playing && on && t >= c.t0 && t < c.t1;
    if (inside && c.el.paused) {
      c.synced = true;
      c.el.currentTime = t - c.t0;
      c.el.play().catch(() => (c.synced = false));
    } else if (!inside) {
      stopSynced(c);
    }
  }
}

function stopSynced(c) {
  if (c.synced) { c.el.pause(); c.synced = false; }
}

// --- wiring -------------------------------------------------------------------------------------------------
$("#filter").addEventListener("input", renderList);
addEventListener("hashchange", () => openRecording(decodeURIComponent(location.hash.slice(1))));
loadList().then(() => openRecording(decodeURIComponent(location.hash.slice(1))));
setInterval(() => document.visibilityState === "visible" && loadList(), 30000);
