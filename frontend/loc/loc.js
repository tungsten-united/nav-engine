// Nav Localize: which node of a map a frame shows, through nav-api (POST /maps/{id}/localize). On site with the
// camera, or at the desk with a recording's video: a walk the map was not built from is the honest test.
"use strict";

const L = { map: null, names: {}, confirmed: null, busy: false, timer: null };
const zoom = zoomer();
const picker = sourcePicker($("#video"), $("#image"), { onChange: () => { $("#placeholder").hidden = !!Frames.kind; } });

function showResult(el) {
  for (const img of $("#result").querySelectorAll("img[src^='blob:']")) URL.revokeObjectURL(img.src);
  $("#result").replaceChildren(el);
}

function renderHistory() {
  History.render($("#history"), L.names);
  const s = History.steady();
  $("#steady").textContent = s ? `steady: ${L.names[s] || s}` : "";
}

async function openMap(id) {
  if (!id) return;
  $("#error").textContent = "";
  try {
    L.map = await navJSON(`/maps/${encodeURIComponent(id)}`);
  } catch (err) {
    $("#error").textContent = err.message;
    return;
  }
  L.names = Object.fromEntries(L.map.nodes.map((n) => [n.id, n.name]));
  L.confirmed = null;
  History.items = [];
  renderHistory();
  $("#expected").replaceChildren(h("option", { value: "" }, "–"),
    ...L.map.nodes.map((n) => h("option", { value: n.id }, n.name)));
  history.replaceState(null, "", `${location.search}#${id}`);
  const refs = L.map.references.length;
  showResult(h("p", { class: refs ? "muted" : "error" }, refs
    ? `${L.map.place}: ${L.map.nodes.length} nodes, ${refs} reference images (${L.map.embedding?.model || "no model"}). No result yet.`
    : `${L.map.place} has no reference images yet: refinish its map on map-api (POST /maps/${id}/refinish).`));
  picker.setPlace(L.map.place, L.map.recordings);
}

async function run() {
  if (L.busy || !L.map) return;
  L.busy = true;
  $("#once").disabled = true;
  $("#error").textContent = "";
  try {
    const blobs = await Frames.burst(+$("#burst").value);
    const heading = Frames.heading();
    const res = await localize(L.map.map_id, blobs, {
      previous: $("#graph").checked ? L.confirmed : null, expected: $("#expected").value || null, heading });
    if (res.status === "confirmed") L.confirmed = res.best;
    History.push(res);
    showResult(resultCard(res, { names: L.names, queries: blobs, onZoom: zoom }));
    renderHistory();
  } catch (err) {
    $("#error").textContent = err.message;
  } finally {
    L.busy = false;
    $("#once").disabled = false;
  }
}

function schedule() {
  clearInterval(L.timer);
  L.timer = $("#auto").checked ? setInterval(run, +$("#every").value * 1000) : null;
}

function status() {
  const hd = Frames.heading();
  const zero = L.map?.frame?.heading_zero_abs_deg;
  $("#heading").textContent = [Frames.kind ? Frames.label() : "no source",
    hd == null ? "no compass" : `compass ${fmtDeg(hd)}${zero != null ? ` (map ${fmtDeg((hd - zero + 360) % 360)})` : ""}`,
    `last confirmed: ${L.names[L.confirmed] || "–"}`].join(" · ");
}

async function init() {
  document.getElementById("top").replaceWith(header("Nav Localize", "/loc/"));
  $("#source").replaceWith(picker.el);
  $("#once").addEventListener("click", run);
  $("#auto").addEventListener("change", schedule);
  $("#every").addEventListener("change", schedule);
  $("#map").addEventListener("change", (e) => openMap(e.target.value));
  healthLine($("#health"));
  setInterval(() => healthLine($("#health")), 30000);
  setInterval(status, 500);
  try {
    await loadMaps($("#map"), decodeURIComponent(location.hash.slice(1)));
    await openMap($("#map").value);
  } catch (err) {
    $("#error").textContent = err.message;
  }
}

init();
