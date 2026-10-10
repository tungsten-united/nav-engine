// Shared by the localization (/loc/) and navigation (/nav/) debuggers: calls to nav-api (NAV.nav, config.js), frame
// sources (the rear camera, a recording's video from map-api, an image file), the compass, and how a localization
// result is drawn. Vanilla JS; these globals are used by loc.js and nav.js.
"use strict";

const NAVAPI = NAV.nav;
const $ = (sel, root = document) => root.querySelector(sel);
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
const fmtDeg = (d) => (d == null ? "?" : `${Math.round(d)}°`);
const fmtM = (m) => (m == null ? "?" : `${Math.round(m)} m`);
const fmtScore = (s) => (s == null ? "–" : s.toFixed(2));

async function call(client, path, opts = {}) {
  const r = await client.fetch(client.API + path, { cache: "no-store", ...opts });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail; } catch {}
    throw Object.assign(new Error(`${client.name} ${path}: HTTP ${r.status}${detail ? ` — ${detail}` : ""}`), { status: r.status });
  }
  return r.json();
}
const navJSON = (path, opts) => call(NAVAPI, path, opts);
const mapJSON = (path, opts) => call(NAV, path, opts);
const navFileUrl = (mapId, rel) => NAVAPI.url(`${NAVAPI.API}/maps/${encodeURIComponent(mapId)}/files/${rel}`);

// --- page frame ----------------------------------------------------------------------------------------------
const PAGES = [["/", "Recorder"], ["/dash/", "Recordings"], ["/viz/", "Maps"], ["/loc/", "Localize"], ["/nav/", "Navigate"]];
function header(title, current) {
  return h("header", { class: "bar" },
    h("a", { class: "brand", href: current }, title),
    h("nav", { class: "links" },
      PAGES.map(([href, label]) => h("a", { href, "aria-current": href === current ? "page" : null }, label)),
      h("a", { href: NAV.docs, title: NAV.base }, "map-api"),
      h("a", { href: NAVAPI.docs, title: NAVAPI.base }, "nav-api")));
}

async function healthLine(el) {
  try {
    const j = await navJSON("/health?deep=1");
    const inf = j.inference || {};
    const where = (inf.where || "").replace(/^https?:\/\//, "");
    el.replaceChildren(h("span", { class: "dot ok" }), ` nav-api ok · ${j.maps} maps · inference `,
      h("span", { class: `dot ${inf.ok ? "ok" : "bad"}` }), ` ${where}`,
      inf.gpu ? ` · ${inf.gpu}` : "", inf.gpu_pending != null ? ` · queue ${inf.gpu_pending} + live ${inf.gpu_live_pending ?? "?"}` : "",
      inf.error ? ` · ${inf.error}` : "");
  } catch (err) {
    el.replaceChildren(h("span", { class: "dot bad" }), ` ${err.message}`);
  }
}

async function loadMaps(select, wanted) {
  const maps = await navJSON("/maps");
  select.replaceChildren(...maps.map((m) => h("option", { value: m.map_id, selected: m.map_id === wanted },
    `${m.place} · ${m.nodes} nodes`)));
  if (!maps.length) select.append(h("option", { value: "" }, "no published maps"));
  return maps;
}

// --- compass -------------------------------------------------------------------------------------------------
// Heading of the rear-camera axis (device -z) from W3C Euler angles, as the recorder computes it (frontend/app.js).
function cameraHeading(alpha, beta, gamma) {
  const r = Math.PI / 180;
  const cA = Math.cos(alpha * r), sA = Math.sin(alpha * r), cB = Math.cos(beta * r), sB = Math.sin(beta * r);
  const cG = Math.cos(gamma * r), sG = Math.sin(gamma * r);
  const east = -cA * sG - sA * sB * cG, north = -sA * sG + cA * sB * cG;
  if (Math.hypot(east, north) < 0.2) return null;
  return (Math.atan2(east, north) / r + 360) % 360;
}
const Compass = {
  heading: null,
  started: false,
  async start() {
    if (this.started) return;
    this.started = true;
    try {
      if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function")
        await DeviceOrientationEvent.requestPermission();
    } catch {}  // denied: localization still works, without heading
    const on = (e) => {
      if (e.webkitCompassHeading != null) this.heading = e.webkitCompassHeading;
      else if ((e.type === "deviceorientationabsolute" || e.absolute) && e.alpha != null)
        this.heading = cameraHeading(e.alpha, e.beta, e.gamma) ?? this.heading;
    };
    addEventListener("deviceorientationabsolute", on);
    addEventListener("deviceorientation", on);
  },
};

// --- frame sources -------------------------------------------------------------------------------------------
// One at a time: the rear camera, a recording's video (map-api; its compass heading follows the video), or an image.
const MAX_SIDE = 768;  // the map's frames are 768 px
const Frames = {
  kind: null,
  video: null,
  img: null,
  rec: null,  // { id, vs: video start on the recording clock (s), t: [ms], hd: [deg] }
  stop() {
    const v = this.video;
    if (v) {
      v.srcObject?.getTracks().forEach((t) => t.stop());
      v.pause();
      v.removeAttribute("src");
      v.srcObject = null;
      v.load();
    }
    if (this.img?.src?.startsWith("blob:")) URL.revokeObjectURL(this.img.src);
    this.kind = this.rec = null;
  },
  async camera(video) {
    this.stop();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("no camera here (needs HTTPS)");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    Object.assign(video, { muted: true, playsInline: true, controls: false });
    video.srcObject = stream;
    await video.play();
    Object.assign(this, { kind: "camera", video });
    Compass.start();
  },
  async recording(video, recId) {
    this.stop();
    Object.assign(video, { crossOrigin: "anonymous", controls: true, muted: true, playsInline: true });
    video.src = NAV.url(`${NAV.API}/recordings/${recId}/files/video.mp4`);
    Object.assign(this, { kind: "recording", video, rec: { id: recId, vs: 0, t: [], hd: [] } });
    try {  // heading along the video: the recording's orientation samples, video time 0 = the video_start event
      const [events, ori] = await Promise.all([
        NAV.fetch(`${NAV.API}/recordings/${recId}/files/events.jsonl`, { cache: "no-store" }).then((r) => (r.ok ? r.text() : "")),
        mapJSON(`/recordings/${recId}/series/orientation?max_points=20000`),
      ]);
      const start = events.split("\n").filter(Boolean).map((l) => JSON.parse(l)).find((e) => e.type === "video_start");
      const keep = ori.t_ms.map((t, i) => [t, ori.heading_deg?.[i]]).filter(([, d]) => d != null);
      if (this.rec?.id === recId) Object.assign(this.rec, { vs: (start?.t_ms ?? 0) / 1000, t: keep.map((x) => x[0]), hd: keep.map((x) => x[1]) });
    } catch {}  // no sensors (an imported video): no heading
  },
  image(img, file) {
    this.stop();
    img.src = URL.createObjectURL(file);
    Object.assign(this, { kind: "image", img });
  },
  ready() {
    if (this.kind === "image") return this.img.complete && this.img.naturalWidth > 0;
    return !!this.video && this.video.readyState >= 2 && this.video.videoWidth > 0;
  },
  heading() {
    if (this.kind === "camera") return Compass.heading;
    const r = this.rec;
    if (this.kind !== "recording" || !r?.t.length) return null;
    const t = (r.vs + this.video.currentTime) * 1000;
    let lo = 0, hi = r.t.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (r.t[mid] < t) lo = mid + 1; else hi = mid; }
    return Math.abs(r.t[lo] - t) < 1000 ? r.hd[lo] : null;
  },
  label() {
    if (this.kind === "recording") return `${this.rec.id} @ ${this.video.currentTime.toFixed(1)} s`;
    return this.kind || "";
  },
  grab() {
    const src = this.kind === "image" ? this.img : this.video;
    const w = this.kind === "image" ? src.naturalWidth : src.videoWidth;
    const hgt = this.kind === "image" ? src.naturalHeight : src.videoHeight;
    const k = Math.min(1, MAX_SIDE / Math.max(w, hgt));
    const c = h("canvas", { width: Math.round(w * k), height: Math.round(hgt * k) });
    c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
    return new Promise((ok, fail) => c.toBlob((b) => (b ? ok(b) : fail(new Error("could not encode the frame"))), "image/jpeg", 0.85));
  },
  // A short burst: n frames gapMs apart (an image source gives the same frame once).
  async burst(n = 1, gapMs = 250) {
    if (!this.ready()) throw new Error("no frame yet: start the camera, load a recording or pick an image");
    const out = [];
    for (let i = 0; i < (this.kind === "image" ? 1 : n); i++) {
      if (i) await new Promise((r) => setTimeout(r, gapMs));
      out.push(await this.grab());
    }
    return out;
  },
};

async function localize(mapId, blobs, { previous, expected, heading, topK = 5 } = {}) {
  const fd = new FormData();
  blobs.forEach((b, i) => fd.append("images", b, `frame${i}.jpg`));
  if (previous) fd.append("previous", previous);
  if (expected) fd.append("expected", expected);
  if (heading != null) fd.append("heading_deg", heading.toFixed(1));
  fd.append("top_k", String(topK));
  const t0 = performance.now();
  const res = await navJSON(`/maps/${encodeURIComponent(mapId)}/localize`, { method: "POST", body: fd });
  res.roundtrip_ms = Math.round(performance.now() - t0);
  return res;
}

// --- drawing a result ----------------------------------------------------------------------------------------
const roleText = (r) => (r.role === "outgoing" ? `leaving${r.toward ? ` → ${r.toward}` : ""}` : r.role);
function scoreBar(score, th) {
  const pct = (x) => `${Math.max(0, Math.min(1, x)) * 100}%`;
  return h("span", { class: "bar-track", title: `lost < ${th.lost_score}, confirm ≥ ${th.confirm_score}` },
    h("span", { class: "bar-fill", style: `width:${pct(score)}` }),
    h("span", { class: "bar-mark", style: `left:${pct(th.lost_score)}` }),
    h("span", { class: "bar-mark", style: `left:${pct(th.confirm_score)}` }));
}
function statusBadge(status) {
  return h("span", { class: `badge st-${status}` }, status);
}
// `names`: node id -> name, for previous / expected. `queries`: the Blobs that were sent.
function resultCard(res, { names = {}, queries = [], compact = false, onZoom } = {}) {
  const best = res.candidates[0];
  const timing = [`round trip ${res.roundtrip_ms} ms`, ...Object.entries(res.took_ms).map(([k, v]) => `${k} ${Math.round(v)}`)].join(" · ");
  const card = h("section", { class: `result ${compact ? "compact" : ""}` },
    h("div", { class: "result-head" }, statusBadge(res.status),
      h("strong", {}, best ? best.name : "nothing"),
      best ? h("span", { class: "muted" }, ` score ${fmtScore(best.score)} · margin ${fmtScore(res.margin)}`) : null),
    h("p", { class: "muted small" }, res.reason),
    h("p", { class: "muted small" }, timing, res.frames > 1 ? ` · ${res.frames} frames` : "",
      res.heading_map_deg != null ? ` · facing ${fmtDeg(res.heading_map_deg)} (map)` : ""),
    res.expected ? h("p", { class: "small" }, `Expected ${names[res.expected] || res.expected}: `,
      res.expected_rank ? `rank ${res.expected_rank}, score ${fmtScore(res.expected_score)}` : "not among the references") : null);
  if (queries.length && !compact) {
    card.append(h("div", { class: "queries" }, queries.map((b) => h("img", { src: URL.createObjectURL(b), alt: "frame sent", class: "thumb" }))));
  }
  const list = h("ol", { class: "cands" });
  for (const [i, n] of res.candidates.entries()) {
    if (compact && i >= 3) break;
    list.append(h("li", { class: i === 0 ? "top" : "" },
      h("div", { class: "cand-head" }, h("span", { class: "cand-name" }, n.name), scoreBar(n.score, res.thresholds),
        h("span", { class: "mono" }, fmtScore(n.score))),
      compact ? null : h("div", { class: "refs" }, n.refs.map((r) => {
        const src = navFileUrl(res.map_id, r.image);
        const cap = `${roleText(r)} · ${fmtScore(r.score)}${r.heading_diff_deg != null ? ` · Δ${fmtDeg(r.heading_diff_deg)}` : ""}`;
        return h("figure", { class: "ref" },
          h("img", { src, alt: `${n.name}, ${cap}`, loading: "lazy", class: "thumb", onclick: () => onZoom?.(src, `${n.name}: ${cap}`) }),
          h("figcaption", {}, cap));
      }))));
  }
  card.append(list);
  return card;
}

// Last results as coloured cells; "steady" when the last 3 confirmed the same node (the temporal check of 4.2).
const History = {
  items: [],
  push(res) {
    this.items.push(res);
    if (this.items.length > 20) this.items.shift();
  },
  steady(n = 3) {
    const last = this.items.slice(-n);
    return last.length === n && last.every((r) => r.status === "confirmed" && r.best === last[0].best) ? last[0].best : null;
  },
  render(el, names = {}) {
    el.replaceChildren(...this.items.map((r) => h("span", {
      class: `cell st-${r.status}`, title: `${r.status}: ${names[r.best] || r.best || "–"} ${fmtScore(r.candidates[0]?.score)}`,
    }, (names[r.best] || r.best || "–").slice(0, 3))));
  },
};

function zoomer() {
  const dlg = h("dialog", { class: "zoom" }, h("img", { alt: "" }), h("p", { class: "caption" }),
    h("form", { method: "dialog" }, h("button", {}, "Close")));
  document.body.append(dlg);
  return (src, caption) => {
    $("img", dlg).src = src;
    $(".caption", dlg).textContent = caption;
    dlg.showModal();
  };
}

// A source picker (camera / recording of the map's place / image file) wired to Frames.
function sourcePicker(video, img, { images = true, onChange } = {}) {
  const kind = h("select", { "aria-label": "Frame source" }, h("option", { value: "" }, "Pick a source…"),
    h("option", { value: "camera" }, "Camera"), h("option", { value: "recording" }, "Recording"),
    images ? h("option", { value: "image" }, "Image file") : null);
  const rec = h("select", { "aria-label": "Recording", hidden: true });
  const file = h("input", { type: "file", accept: "image/*", hidden: true, "aria-label": "Image file" });
  const msg = h("span", { class: "muted small", role: "status" });
  let mapRecs = new Set();
  const show = async () => {
    msg.textContent = "";
    rec.hidden = kind.value !== "recording";
    file.hidden = kind.value !== "image";
    video.hidden = !["camera", "recording"].includes(kind.value);
    img.hidden = kind.value !== "image";
    try {
      if (kind.value === "camera") await Frames.camera(video);
      else if (kind.value === "recording" && rec.value) await Frames.recording(video, rec.value);
      else Frames.stop();
    } catch (err) {
      msg.textContent = err.message;
    }
    onChange?.();
  };
  kind.addEventListener("change", show);
  rec.addEventListener("change", show);
  file.addEventListener("change", () => { if (file.files[0]) { Frames.image(img, file.files[0]); onChange?.(); } });
  return {
    el: h("span", { class: "source" }, kind, rec, file, msg),
    // The recordings of a map's place, newest first; the walks the map was built from are marked (they match
    // themselves: held-out walks are the honest test).
    async setPlace(place, builtFrom = []) {
      mapRecs = new Set(builtFrom);
      try {
        const recs = await mapJSON(`/recordings?place=${encodeURIComponent(place)}`);
        rec.replaceChildren(...recs.filter((r) => r.status !== "recording").map((r) => h("option", { value: r.rec_id },
          `${r.rec_id}${r.duration_ms ? ` · ${Math.round(r.duration_ms / 1000)} s` : ""}${mapRecs.has(r.rec_id) ? " · in the map" : " · held out"}`)));
      } catch (err) {
        rec.replaceChildren(h("option", { value: "" }, `recordings: ${err.message}`));
      }
      if (kind.value === "recording") show();
    },
    start: show,
  };
}
