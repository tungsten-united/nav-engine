// Nav Maps: the navigation graph of a place. Nodes sit at their dead-reckoned positions (metres, recordings aligned
// through shared nodes) and edges follow the walked path between them. Hover previews, click pins the details.
"use strict";

const API = "/api/v1";
const KINDS = {
  entrance: "entrance", door: "door", corridor: "corridor", junction: "junction", room: "room", open_area: "open area",
  stairs: "stairs", lift: "lift", destination: "destination", landmark: "landmark", other: "other",
};

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  el.append(...kids.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}
function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs);
  el.append(...kids.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}
async function getJSON(path) {
  const r = await fetch(API + path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
const kindColor = (k) => `var(--k-${KINDS[k] ? k : "other"})`;
const fmtT = (ms) => (ms == null ? "" : `t=${Math.round(ms / 1000)}s`);
const fmtM = (m) => (m == null ? "?" : `${Math.round(m)} m`);
const fmtDate = (iso) => new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });

// --- state -------------------------------------------------------------------------------------------------
const S = {
  maps: [], map: null, nodes: new Map(), edges: new Map(), pos: new Map(),
  view: { s: 1, tx: 0, ty: 0 }, walked: true, pin: null, route: null, turns: new Map(), order: [],
};

// --- geometry ----------------------------------------------------------------------------------------------
const bearing = (a, b) => (Math.atan2(b.x - a.x, b.y - a.y) * 180) / Math.PI;  // compass-like, world y = north
const angDiff = (a, b) => ((b - a + 540) % 360) - 180;  // + = clockwise = right
function project(p) {
  const v = S.view;
  return { x: p.x * v.s + v.tx, y: -p.y * v.s + v.ty };
}
function edgePoints(e) {
  const a = S.pos.get(e.source), b = S.pos.get(e.target);
  if (!S.walked || !e.path || e.path.length < 2) return [a, b];
  return e.path.map((p) => ({ x: p.x_m, y: p.y_m }));
}
function smoothPath(pts) {
  // screen points -> drop near-duplicates -> Catmull-Rom as cubic Beziers
  const q = [pts[0]];
  for (const p of pts.slice(1, -1)) if (Math.hypot(p.x - q.at(-1).x, p.y - q.at(-1).y) > 8) q.push(p);
  q.push(pts.at(-1));
  if (q.length === 2) return `M${q[0].x},${q[0].y}L${q[1].x},${q[1].y}`;
  let d = `M${q[0].x},${q[0].y}`;
  for (let i = 0; i < q.length - 1; i++) {
    const p0 = q[Math.max(0, i - 1)], p1 = q[i], p2 = q[i + 1], p3 = q[Math.min(q.length - 1, i + 2)];
    d += `C${p1.x + (p2.x - p0.x) / 6},${p1.y + (p2.y - p0.y) / 6} ${p2.x - (p3.x - p1.x) / 6},${p2.y - (p3.y - p1.y) / 6} ${p2.x},${p2.y}`;
  }
  return d;
}
function midpoint(pts) {
  const seg = pts.slice(1).map((p, i) => Math.hypot(p.x - pts[i].x, p.y - pts[i].y));
  let half = seg.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < seg.length; i++) {
    if (half <= seg[i] || i === seg.length - 1) {
      const f = seg[i] ? Math.min(1, half / seg[i]) : 0, a = pts[i], b = pts[i + 1];
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, ang: Math.atan2(b.y - a.y, b.x - a.x) };
    }
    half -= seg[i];
  }
  return { ...pts[0], ang: 0 };
}

function layoutPositions(m) {
  S.pos.clear();
  const placed = m.nodes.filter((n) => n.position);
  for (const n of placed) S.pos.set(n.id, { x: n.position.x_m, y: n.position.y_m });
  const rest = m.nodes.filter((n) => !n.position);
  if (rest.length) {  // no motion data: a circle around what is placed (or around the origin)
    const c = placed.length ? { x: avg(placed.map((n) => n.position.x_m)), y: avg(placed.map((n) => n.position.y_m)) } : { x: 0, y: 0 };
    const r = Math.max(10, rest.length * 3);
    rest.forEach((n, i) => S.pos.set(n.id, { x: c.x + r * Math.cos((2 * Math.PI * i) / rest.length), y: c.y + r * Math.sin((2 * Math.PI * i) / rest.length) }));
  }
}
const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function fit() {
  const el = $("#graph"), W = el.clientWidth, H = el.clientHeight;
  const pts = [...S.pos.values()];
  if (S.walked) for (const e of S.map?.edges || []) for (const p of e.path || []) pts.push({ x: p.x_m, y: p.y_m });
  if (!pts.length) return;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const padX = Math.min(170, W * 0.18), padY = Math.min(50, H * 0.08);  // room for labels
  const sc = Math.min((W - 2 * padX) / Math.max(x1 - x0, 1), (H - 2 * padY) / Math.max(y1 - y0, 1), 40);
  S.view = { s: sc, tx: W / 2 - ((x0 + x1) / 2) * sc, ty: H / 2 + ((y0 + y1) / 2) * sc };
  draw();
}

// --- turns and recording order ------------------------------------------------------------------------------
function computeOrder(m) {
  // edges in the order they were walked; turn at the start of an edge relative to how you arrived at its source
  S.order = [...m.edges].sort((a, b) => (a.walked_in || "").localeCompare(b.walked_in || "") ||
    (a.walked_at_ms ?? Infinity) - (b.walked_at_ms ?? Infinity));
  S.turns.clear();
  const pts = (e) => (e.path?.length >= 2 ? e.path.map((p) => ({ x: p.x_m, y: p.y_m })) : [S.pos.get(e.source), S.pos.get(e.target)]);
  const head = (p, fromEnd) => {  // direction over the first / last ~4 m of a path
    const seq = fromEnd ? [...p].reverse() : p;
    let j = 1;
    while (j < seq.length - 1 && Math.hypot(seq[j].x - seq[0].x, seq[j].y - seq[0].y) < 4) j++;
    return fromEnd ? bearing(seq[j], seq[0]) : bearing(seq[0], seq[j]);
  };
  S.order.forEach((e, i) => {
    const prev = S.order[i - 1];
    if (prev && prev.target === e.source && prev.walked_in === e.walked_in) S.turns.set(e.id, angDiff(head(pts(prev), true), head(pts(e), false)));
  });
}
function turnText(t, withDeg = false) {
  if (t == null) return "";
  const a = Math.abs(t), side = t > 0 ? "right" : "left", deg = withDeg ? ` (${t > 0 ? "+" : "−"}${Math.round(a)}°)` : "";
  if (a < 25) return `↑ straight${deg}`;
  if (a < 60) return `${t > 0 ? "↗" : "↖"} bear ${side}${deg}`;
  if (a < 135) return `${t > 0 ? "→" : "←"} turn ${side}${deg}`;
  return `↓ turn around${deg}`;
}

// --- drawing -----------------------------------------------------------------------------------------------
function draw() {
  const world = $("#world");
  const m = S.map;
  world.replaceChildren();
  if (!m) return;
  const on = highlighted();
  $("#svg").classList.toggle("dim", !!S.route);
  if (S.walked) {
    for (const t of m.tracks || []) {
      const d = t.points.map(([, x, y], i) => { const p = project({ x, y }); return `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join("");
      world.append(s("path", { class: "track", d }));
    }
  }
  for (const e of m.edges) {
    const pts = edgePoints(e).map(project);
    const mid = midpoint(pts), d = smoothPath(pts);
    const cls = ["edge", e.confidence === "low" && "low", on.has(e.id) && "on"].filter(Boolean).join(" ");
    const deg = (mid.ang * 180) / Math.PI;
    const nx = -Math.sin(mid.ang) * 11, ny = Math.cos(mid.ang) * 11;  // hazard badge beside the arrow
    world.append(s("g", { class: cls, "data-id": e.id, onmousemove: (ev) => tip(ev, "edge", e.id), onmouseleave: hideTip,
      onclick: (ev) => { ev.stopPropagation(); pin("edge", e.id); } },
      s("path", { class: "line", d }),
      s("path", { class: "arrow", d: "M5,0 L-4,-4 L-4,4 Z", transform: `translate(${mid.x},${mid.y}) rotate(${deg})` }),
      e.hazards.length && s("g", { class: "badge", transform: `translate(${mid.x + nx},${mid.y + ny})` },
        s("circle", { r: 6 }), s("text", { y: 3.5 }, "!")),
      s("path", { class: "hit", d })));
  }
  // placed label boxes (and every node's dot), to keep labels from overlapping each other and other nodes
  const boxes = m.nodes.map((n) => { const p = project(S.pos.get(n.id)); return { x0: p.x - 9, x1: p.x + 9, y0: p.y - 9, y1: p.y + 9, id: n.id }; });
  for (const n of m.nodes) {
    const p = project(S.pos.get(n.id));
    const big = ["entrance", "destination", "room", "open_area"].includes(n.kind);
    const t0 = n.evidence.length ? Math.min(...n.evidence.map((e) => e.t_ms)) : null;
    const name = n.name.length > 28 ? `${n.name.slice(0, 27)}…` : n.name;
    const w = name.length * 6.4, r = big ? 8 : 6;
    const spots = [[12, -1, "start"], [-12, -1, "end"], [12, 20, "start"], [12, -18, "start"], [-12, 20, "end"], [-12, -18, "end"]];
    const box = (x, y, anchor) => ({ x0: p.x + (anchor === "end" ? x - w : x), x1: p.x + (anchor === "end" ? x : x + w), y0: p.y + y - 11, y1: p.y + y + 13 });
    const free = (b) => !boxes.some((o) => o.id !== n.id && b.x0 < o.x1 && o.x0 < b.x1 && b.y0 < o.y1 && o.y0 < b.y1);
    const [lx, ly, anchor] = spots.find(([x, y, a]) => free(box(x, y, a))) || spots[0];
    boxes.push(box(lx, ly, anchor));
    world.append(s("g", { class: `node${on.has(n.id) ? " on" : ""}`, transform: `translate(${p.x},${p.y})`,
      onmousemove: (ev) => tip(ev, "node", n.id), onmouseleave: hideTip,
      onclick: (ev) => { ev.stopPropagation(); pin("node", n.id); } },
      s("circle", { r, fill: kindColor(n.kind) }),
      s("text", { x: lx, y: ly, "text-anchor": anchor }, name),
      s("text", { x: lx, y: ly + 12, class: "meta", "text-anchor": anchor }, `${n.id} · ${fmtT(t0)}`)));
  }
}
function highlighted() {
  const on = new Set();
  if (S.route) S.route.forEach(({ node, edge }) => { on.add(node); if (edge) on.add(edge.id); });
  if (S.pin) {
    on.add(S.pin.id);
    if (S.pin.type === "edge") { const e = S.edges.get(S.pin.id); on.add(e.source); on.add(e.target); }
  }
  return on;
}

// --- tooltip -----------------------------------------------------------------------------------------------
const frameUrl = (rel) => `${API}/maps/${S.map.map_id}/files/${rel}`;
const candUrl = (cid) => frameUrl(`builds/${S.map.build.build_id}/frames/${cid}.jpg`);
function tip(ev, type, id) {
  const el = $("#tip"), box = $("#graph").getBoundingClientRect();
  let html;
  if (type === "node") {
    const n = S.nodes.get(id);
    el.replaceChildren(h("b", {}, n.name), h("div", { class: "turn" }, `${KINDS[n.kind] || n.kind} · ${n.confidence} confidence`),
      h("p", {}, n.description), n.candidates[0] && h("img", { src: candUrl(n.candidates[0]), alt: "" }));
  } else {
    const e = S.edges.get(id);
    const img = e.candidates.at(Math.floor(e.candidates.length / 2)) || S.nodes.get(e.target).candidates[0];
    el.replaceChildren(h("b", {}, `${S.nodes.get(e.source).name} → ${S.nodes.get(e.target).name}`),
      h("div", { class: "turn" }, [turnText(S.turns.get(e.id), true), fmtM(e.measured_distance_m ?? e.distance_m)].filter(Boolean).join(" · ")),
      h("p", {}, e.instruction), img && h("img", { src: candUrl(img), alt: "" }));
  }
  el.hidden = false;
  const x = ev.clientX - box.left + 16, y = ev.clientY - box.top + 12;
  el.style.left = `${Math.min(x, box.width - el.offsetWidth - 8)}px`;
  el.style.top = `${Math.min(y, box.height - el.offsetHeight - 8)}px`;
}
function hideTip() { $("#tip").hidden = true; }

// --- panel -------------------------------------------------------------------------------------------------
function pin(type, id) {
  S.pin = type ? { type, id } : null;
  draw();
  renderPanel();
  if (type) $("#panel").scrollTo({ top: 0, behavior: "smooth" });
}
function frames(cids) {
  const m = S.map;
  const view = (cid) => m.nodes.flatMap((n) => n.evidence).find((ev) => ev.candidate === cid);
  return h("div", { class: "frames" }, cids.map((cid) => {
    const v = view(cid);
    const cap = v ? `${cid} · ${fmtT(v.t_ms)}` : cid;
    return h("figure", { onclick: () => zoom(candUrl(cid), cap + (v ? ` · ${v.rec_id}` : "")) },
      h("img", { src: candUrl(cid), alt: cap, loading: "lazy" }), h("figcaption", {}, cap));
  }));
}
function items(title, list, cls) {
  return list?.length ? [h("h4", {}, title), h("ul", { class: cls }, list.map((x) => h("li", {}, x)))] : [];
}
function pinCard() {
  if (!S.pin) return null;
  const close = h("button", { class: "close", "aria-label": "Close", onclick: () => pin(null) }, "×");
  if (S.pin.type === "node") {
    const n = S.nodes.get(S.pin.id);
    const recs = [...new Set(n.evidence.map((e) => e.rec_id))];
    return h("div", { class: "pin" }, close, h("h2", {}, n.name),
      h("div", { class: "chips" }, h("span", { class: "chip", style: `color:${kindColor(n.kind)}` }, KINDS[n.kind] || n.kind),
        h("span", { class: `chip ${n.confidence}` }, `${n.confidence} confidence`), n.floor && h("span", { class: "chip" }, `floor ${n.floor}`),
        h("span", { class: "chip" }, n.id)),
      h("p", {}, n.description), ...items("Landmarks", n.landmarks), ...items("Hazards", n.hazards, "hazards"),
      ...items("Visual cues (sighted helpers, camera)", n.visual_cues, "visual"),
      h("h4", {}, `Seen in ${recs.length} walk${recs.length === 1 ? "" : "s"}`),
      h("dl", { class: "kv" }, recs.map((r) => [h("dt", {}, h("a", { href: `/dash/#${r}` }, r)),
        h("dd", {}, n.evidence.filter((e) => e.rec_id === r).map((e) => fmtT(e.t_ms)).join(", "))])),
      frames(n.candidates));
  }
  const e = S.edges.get(S.pin.id), a = S.nodes.get(e.source), b = S.nodes.get(e.target);
  return h("div", { class: "pin" }, close, h("h2", {}, `${a.name} → ${b.name}`),
    h("div", { class: "chips" }, h("span", { class: "chip" }, e.kind), h("span", { class: `chip ${e.confidence}` }, `${e.confidence} confidence`),
      h("span", { class: "chip" }, e.bidirectional ? "both ways" : "one way"), h("span", { class: "chip" }, e.id)),
    h("dl", { class: "kv" }, h("dt", {}, "Distance"), h("dd", {}, `${fmtM(e.distance_m)} (LLM) · ${fmtM(e.measured_distance_m)} (steps)`),
      S.turns.has(e.id) && [h("dt", {}, "Turn"), h("dd", {}, turnText(S.turns.get(e.id), true))]),
    h("div", { class: "instruction" }, e.instruction),
    e.reverse_instruction && [h("h4", {}, `Back: ${b.name} → ${a.name}`), h("div", { class: "instruction" }, e.reverse_instruction)],
    ...items("Along the way", e.landmarks), ...items("Hazards", e.hazards, "hazards"),
    ...items("Visual cues (sighted helpers, camera)", e.visual_cues, "visual"),
    e.candidates.length ? frames(e.candidates) : null);
}

function directions() {
  const opts = (sel) => S.map.nodes.map((n) => h("option", { value: n.id, selected: n.id === sel }, n.name));
  const from = h("select", { "aria-label": "From" }, opts(S.route?.[0]?.node ?? S.map.nodes[0]?.id));
  const to = h("select", { "aria-label": "To" }, opts(S.route?.at(-1)?.node ?? S.map.nodes.at(-1)?.id));
  const form = h("form", { class: "route", onsubmit: (ev) => { ev.preventDefault(); S.route = findRoute(from.value, to.value); draw(); renderPanel(); } },
    from, to, h("button", { type: "submit" }, "Go"));
  const out = [h("h3", {}, "Directions"), form];
  if (S.route) {
    if (S.route.length < 2) out.push(h("p", { class: "hint" }, "No walked connection between these two."));
    else {
      const total = S.route.slice(1).reduce((t, { edge }) => t + (edge.measured_distance_m ?? edge.distance_m ?? 0), 0);
      out.push(h("ol", { class: "steps" }, S.route.slice(1).map(({ node, edge, forward }) =>
        h("li", {}, h("b", {}, `To ${S.nodes.get(node).name} · ${fmtM(edge.measured_distance_m ?? edge.distance_m)}`),
          forward ? edge.instruction : (edge.reverse_instruction || `Reverse of: ${edge.instruction}`)))),
        h("p", { class: "hint" }, `About ${Math.round(total)} m. `, h("button", { class: "link", onclick: () => { S.route = null; draw(); renderPanel(); } }, "Clear")));
    }
  }
  return out;
}
function findRoute(src, dst) {
  // Dijkstra over walked distances; edges that aren't bidirectional only go source -> target
  const dist = new Map([[src, 0]]), prev = new Map(), todo = new Set(S.map.nodes.map((n) => n.id));
  while (todo.size) {
    let u = null;
    for (const v of todo) if (dist.has(v) && (u == null || dist.get(v) < dist.get(u))) u = v;
    if (u == null || u === dst) break;
    todo.delete(u);
    for (const e of S.map.edges) {
      const w = e.measured_distance_m ?? e.distance_m ?? 10;
      for (const [a, b, forward] of [[e.source, e.target, true], [e.target, e.source, false]]) {
        if (a !== u || (!forward && !e.bidirectional)) continue;
        if (!dist.has(b) || dist.get(u) + w < dist.get(b)) { dist.set(b, dist.get(u) + w); prev.set(b, { from: u, edge: e, forward }); }
      }
    }
  }
  if (!dist.has(dst)) return [{ node: src }];
  const steps = [];
  for (let v = dst; prev.has(v); v = prev.get(v).from) steps.unshift({ node: v, edge: prev.get(v).edge, forward: prev.get(v).forward });
  return [{ node: src }, ...steps];
}

function renderPanel() {
  const m = S.map, p = $("#panel");
  if (!m) { p.replaceChildren(h("h2", {}, "Nav Maps"), h("p", { class: "hint" }, "Build a map with `nav map build <place>`.")); return; }
  const b = m.build;
  p.replaceChildren(...[
    h("h2", {}, m.place), h("p", { class: "hint" }, "Hover a node or edge for a preview; click to pin its details here."),
    pinCard(),
    ...directions(),
    h("h3", {}, "Route (recording order)"),
    h("ul", { class: "list" }, S.order.map((e) => h("li", { class: S.pin?.id === e.id ? "on" : null, onclick: () => pin("edge", e.id) },
      `${S.nodes.get(e.source).name} → ${S.nodes.get(e.target).name} `, h("span", { class: "glyph" }, [turnText(S.turns.get(e.id)), fmtM(e.measured_distance_m ?? e.distance_m)].filter(Boolean).join(" · "))))),
    h("h3", {}, "Nodes"),
    h("ul", { class: "list" }, m.nodes.map((n) => h("li", { class: S.pin?.id === n.id ? "on" : null, onclick: () => pin("node", n.id) },
      h("span", { class: "dot", style: `--c:${kindColor(n.kind)}` }, n.name), " ", h("span", { class: "id" }, `· ${n.id}`)))),
    h("h3", {}, "About this place"), h("p", {}, m.summary),
    m.notes.length && [h("h3", {}, "Notes"), h("ul", { class: "notes" }, m.notes.map((n) => h("li", {}, n)))],
    m.warnings.length && [h("h3", {}, "Pipeline warnings"), h("ul", { class: "notes warnings" }, m.warnings.map((n) => h("li", {}, n)))],
    h("h3", {}, "Build"),
    h("dl", { class: "kv" },
      h("dt", {}, "Built"), h("dd", {}, `${fmtDate(b.built_at_utc)} · ${b.build_id}`),
      h("dt", {}, "Walks"), h("dd", {}, m.recordings.map((r) => [h("a", { href: `/dash/#${r}` }, r), " "])),
      h("dt", {}, "Candidates"), h("dd", {}, `${b.candidates} frames shown to the LLM`),
      h("dt", {}, "LLM"), h("dd", {}, `${b.llm.model}${b.llm.effort ? ` (${b.llm.effort} effort)` : ""} via ${b.llm.backend.replace("_", " ")}` + (b.llm.cost_usd != null ? ` · $${b.llm.cost_usd.toFixed(2)} at API list price` : "") + (b.llm.duration_s ? ` · ${Math.round(b.llm.duration_s)} s` : "")),
      h("dt", {}, "Embeddings"), h("dd", {}, `${b.embed_model}${b.embed_dim ? ` · ${b.embed_dim}-d` : ""}`),
      h("dt", {}, "Voice"), h("dd", {}, b.transcribe_model || "not transcribed"),
      h("dt", {}, "Files"), h("dd", {}, ["candidates.json", "job/prompt.md", "job/claude.json"].map((f) => [h("a", { href: frameUrl(`builds/${b.build_id}/${f}`) }, f), " "])))
  ].flat(Infinity).filter(Boolean));
}

function legend() {
  const kinds = [...new Set(S.map?.nodes.map((n) => n.kind) || [])];
  $("#legend").replaceChildren(
    ...kinds.map((k) => h("span", { class: "dot", style: `--c:${kindColor(k)}` }, KINDS[k] || k)),
    h("span", { class: "ln", style: "--c:var(--edge)" }, "edge (arrow: walked direction)"),
    h("span", { class: "haz" }, "hazards on the way"),
    h("span", { class: "ln dash", style: "--c:var(--edge)" }, "low confidence"),
    S.walked && h("span", { class: "ln dash", style: "--c:var(--track)" }, "walked track"));
}

function zoom(src, caption) {
  const d = $("#zoom");
  d.querySelector("img").src = src;
  d.querySelector(".caption").textContent = caption;
  d.showModal();
}

// --- loading -----------------------------------------------------------------------------------------------
async function openMap(id) {
  S.pin = S.route = null;
  hideTip();
  try {
    const m = await getJSON(`/maps/${encodeURIComponent(id)}`);
    S.map = m;
    S.nodes = new Map(m.nodes.map((n) => [n.id, n]));
    S.edges = new Map(m.edges.map((e) => [e.id, e]));
    layoutPositions(m);
    computeOrder(m);
    $("#title").textContent = `${m.place} — navigation graph`;
    $("#subtitle").textContent = `${m.nodes.length} nodes · ${m.edges.length} edges · ${m.recordings.length} walk${m.recordings.length === 1 ? "" : "s"} · ` +
      `layout dead-reckoned from steps and heading (approximate) · built ${fmtDate(m.build.built_at_utc)}`;
    $("#maps").value = id;
    $("#empty").hidden = true;
  } catch (err) {
    S.map = null;
    $("#empty").textContent = `Could not load map ${id}: ${err.message}`;
    $("#empty").hidden = false;
  }
  legend();
  renderPanel();
  fit();
}

async function init() {
  try {
    S.maps = await getJSON("/maps");
  } catch (err) {
    $("#empty").textContent = `Could not load maps: ${err.message}`;
    $("#empty").hidden = false;
    return;
  }
  $("#maps").replaceChildren(...S.maps.map((m) => h("option", { value: m.map_id }, `${m.place} (${m.nodes} nodes)`)));
  if (!S.maps.length) {
    $("#empty").textContent = "No maps yet. Build one with: uv run nav map build <place>";
    $("#empty").hidden = false;
    renderPanel();
    return;
  }
  const want = decodeURIComponent(location.hash.slice(1));
  openMap(S.maps.some((m) => m.map_id === want) ? want : S.maps[0].map_id);
}

// --- interaction -------------------------------------------------------------------------------------------
$("#maps").addEventListener("change", (e) => { location.hash = e.target.value; });
window.addEventListener("hashchange", () => { const id = decodeURIComponent(location.hash.slice(1)); if (id && id !== S.map?.map_id) openMap(id); });
$("#fit").addEventListener("click", fit);
$("#walk").addEventListener("click", (e) => {
  S.walked = !S.walked;
  e.currentTarget.setAttribute("aria-pressed", S.walked);
  legend();
  draw();
});
const svg = $("#svg");
let drag = null;
svg.addEventListener("pointerdown", (e) => {
  if (e.target.closest(".node, .edge")) return;
  drag = { x: e.clientX, y: e.clientY, tx: S.view.tx, ty: S.view.ty, moved: false };
  svg.setPointerCapture(e.pointerId);
  svg.classList.add("dragging");
});
svg.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  S.view.tx = drag.tx + dx;
  S.view.ty = drag.ty + dy;
  draw();
});
svg.addEventListener("pointerup", () => {
  if (drag && !drag.moved && (S.pin || S.route)) { S.pin = null; draw(); renderPanel(); }  // click on empty space unpins
  drag = null;
  svg.classList.remove("dragging");
});
svg.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = svg.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const k = Math.exp(-e.deltaY * 0.0015), v = S.view;
  v.tx = x - (x - v.tx) * k;
  v.ty = y - (y - v.ty) * k;
  v.s *= k;
  draw();
}, { passive: false });
window.addEventListener("resize", () => S.map && fit());

init();
