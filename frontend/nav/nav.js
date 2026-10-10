// Nav Navigate: a route over a map's trusted edges (nav-api POST /maps/{id}/route), its spoken instructions, and a
// bare-bones controller to try it: live frames are localized against the next checkpoint; arrival is confirmed by
// hand, or with "auto-confirm" by the node confirmer (arrival.js, the port of nav/navigation/arrival.py): votes over
// consecutive localizations + dead reckoning from the phone's steps + the compass (docs/navigation_stack.md 4.3: a
// place recognised from afar is not reached). The real controller is the app's; this is for checking routes, wording
// and checkpoint recognition.
"use strict";

const BURST = 2;  // frames per /localize call while following a route (every 1 s by default): one confirmer vote
const N = { map: null, names: {}, route: null, hop: 0, confirmed: null, busy: false, timer: null,
            confirmer: null, steps: new ArrivalNav.StepDetector(), stepCount: 0, prevSteps: 0 };
// The step count when the user's node last changed: /localize turns new steps into metres walked from it.
const markNode = () => { N.prevSteps = N.stepCount; };
// Motion is sent with live camera frames only (a recording's video has no live steps).
const liveMotion = () => (Frames.kind === "camera" ? phoneMotion(N.stepCount) : null);
const zoom = zoomer();
const picker = sourcePicker($("#video"), $("#image"), { images: false });

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = N.map?.language || "en";
  speechSynthesis.speak(u);
}

const hopText = (hop) => hop.instruction || hop.steps.map((s) => s.action.replaceAll("_", " ")).join(", then ") || "(no instruction)";
const current = () => N.route?.hops[N.hop] || null;

async function openMap(id) {
  if (!id) return;
  $("#error").textContent = "";
  try {
    N.map = await navJSON(`/maps/${encodeURIComponent(id)}`);
  } catch (err) {
    $("#error").textContent = err.message;
    return;
  }
  N.names = Object.fromEntries(N.map.nodes.map((n) => [n.id, n.name]));
  const opts = () => N.map.nodes.map((n) => h("option", { value: n.id }, n.name));
  $("#start").replaceChildren(...opts());
  $("#goal").replaceChildren(...opts());
  if (N.map.nodes.length > 1) $("#goal").value = N.map.nodes.at(-1).id;
  history.replaceState(null, "", `${location.search}#${id}`);
  N.route = null;
  History.items = [];
  render();
  picker.setPlace(N.map.place, N.map.recordings);
}

async function plan() {
  $("#error").textContent = "";
  try {
    N.route = await navJSON(`/maps/${encodeURIComponent(N.map.map_id)}/route`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: $("#start").value, goal: $("#goal").value, trust: $("#trust").value }) });
  } catch (err) {
    $("#error").textContent = err.message;
    return;
  }
  N.hop = 0;
  N.confirmed = N.route.start;
  markNode();
  History.items = [];
  startConfirmer();
  render();
  if (N.route.found && N.route.hops.length) speak(hopText(N.route.hops[0]));
}

// --- node confirmation (arrival.js) --------------------------------------------------------------------------
// Dead reckoning needs the phone's own steps: only with the camera as the source. A recording's video or an image
// has no live steps, so the confirmer then works without it (and needs every one of the last n votes).
const headingMap = () => {
  const abs = Frames.heading();
  const zero = N.map?.frame?.heading_zero_abs_deg;
  return abs == null ? null : zero == null ? abs : ((abs - zero) % 360 + 360) % 360;
};

function startConfirmer() {
  const hops = N.route?.found ? N.route.hops.slice(N.hop) : [];
  const live = Frames.kind === "camera";
  const tracker = live && hops.length
    ? new ArrivalNav.GraphTracker(hops, { stepLength: N.map?.frame?.live_step_length_m || N.map?.frame?.step_length_m || 0.7 }) : null;
  N.confirmer = new ArrivalNav.NodeConfirmer(hops, tracker);
  showPdr();
}

function showPdr() {
  const c = N.confirmer, s = c?.pdr();
  $("#pdr").textContent = !c || c.done ? "" : s
    ? `${fmtM(Math.max(s.remaining_m, 0))} ± ${fmtM(s.sigma_m)} to ${N.names[c.expected]} · votes ${c.votes.map((v) => v ? v[0] : "·").join("")}`
    : `no steps (${Frames.kind || "no source"}): camera votes only · ${c.votes.map((v) => v ? v[0] : "·").join("")}`;
}

function onConfirmerEvent(ev) {
  if (!ev) return;
  const name = N.names[ev.node] || ev.node;
  if (ArrivalNav.CONFIRM_KINDS.includes(ev.kind)) {
    advance(ev.kind === "passed" ? `Passed ${name}.` : `${name}.`);
  } else if (ev.kind === "uncertain") {
    $("#arrive").textContent = `Not sure you reached ${name}: stop and pan slowly, or confirm by hand.`;
    speak(`Not sure you reached ${name}. Stop and turn slowly, or confirm.`);
  } else if (ev.kind === "off_route") {
    $("#arrive").textContent = `This looks like ${name}, off the route. Locate yourself and plan again.`;
    speak(`This looks like ${name}, which is off the route. Stop, then plan again.`);
  }
  showPdr();
}

addEventListener("devicemotion", (e) => {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null || !N.steps.update(e.timeStamp / 1000, [a.x, a.y, a.z])) return;
  N.stepCount += 1;
  if (N.confirmer?.tracker && $("#autoconfirm").checked) {
    onConfirmerEvent(N.confirmer.onStep(headingMap(), e.timeStamp / 1000));
    showPdr();
  }
});

function render() {
  const r = N.route;
  const list = $("#hops");
  if (!r) {
    $("#summary").textContent = "";
    list.replaceChildren();
  } else if (!r.found) {
    const more = { verified: "observed", observed: "any" }[r.trust];
    $("#summary").replaceChildren(h("span", { class: "error" },
      `No route from ${N.names[r.start]} to ${N.names[r.goal]} over ${r.trust} edges.`),
      more ? ` Try "${more}" (testing only: real users get verified edges).` : "");
    list.replaceChildren();
  } else {
    $("#summary").replaceChildren(r.hops.length ? `${r.hops.length} hops · ${r.length_m == null ? "length unknown" : fmtM(r.length_m)} · over ${r.trust} edges ` : "Already there. ",
      r.hops.length ? h("button", { type: "button", onclick: () => speak(r.hops.map(hopText).join(" Then: ")) }, "Speak all") : null);
    list.replaceChildren(...r.hops.map((hop, i) => h("li", { class: `hop ${i < N.hop ? "done" : i === N.hop ? "current" : ""}` },
      h("div", { class: "hop-head" },
        h("strong", {}, `${i + 1}. ${N.names[hop.source]} → ${N.names[hop.target]}`),
        h("span", { class: "tools" },
          h("span", { class: `chip ${hop.status}` }, hop.status),
          hop.forward ? null : h("span", { class: "chip", title: "Walked against the edge's direction" }, "reverse"),
          h("span", { class: "chip" }, fmtM(hop.length_m)),
          h("span", { class: "chip", title: "Walking direction when leaving, map frame" }, `→ ${fmtDeg(hop.bearing_deg)}`),
          h("button", { type: "button", onclick: () => speak(hopText(hop)) }, "Speak"))),
      h("p", { class: "instr" }, hopText(hop)),
      hop.steps.length ? h("ul", {}, hop.steps.map((s) => h("li", {}, [s.action.replaceAll("_", " "),
        s.distance_m != null ? ` ${fmtM(s.distance_m)}` : "", s.along ? `, ${s.along}` : "", s.until ? `, ${s.until}` : ""].join("")))) : null)));
  }
  const hop = current();
  $("#confirm").disabled = !hop;
  $("#say").disabled = !hop;
  $("#back").disabled = !r?.found || N.hop === 0;
  if (!r?.found) $("#arrive").textContent = r ? "No route." : "Plan a route, then follow it.";
  else if (!hop) $("#arrive").textContent = `Arrived at ${N.names[r.goal]}.`;
  else $("#arrive").textContent = `Hop ${N.hop + 1}/${r.hops.length}: heading to ${N.names[hop.target]}.`;
  $("#arrive").classList.remove("match");
  History.render($("#history"), N.names);
}

// The current hop's target is reached (by hand or by the confirmer): move on and say what's next.
function advance(prefix) {
  const hop = current();
  if (!hop) return;
  N.confirmed = hop.target;
  markNode();
  N.hop += 1;
  render();
  const next = current();
  speak(next ? `${prefix || N.names[hop.target] + "."} ${hopText(next)}` : `You have arrived at ${N.names[hop.target]}.`);
}

function confirmArrival() {
  if (!current()) return;
  N.confirmer?.confirmManual(performance.now() / 1000);  // keeps the confirmer (and its step length) in step
  advance();
  showPdr();
}

function backOne() {
  if (N.hop === 0) return;
  N.hop -= 1;
  N.confirmed = current().source;
  markNode();
  startConfirmer();
  render();
}

async function run() {
  if (N.busy || !N.map) return;
  N.busy = true;
  $("#once").disabled = true;
  $("#error").textContent = "";
  try {
    const blobs = await Frames.burst(BURST);
    const hop = current();
    const res = await localize(N.map.map_id, blobs, {
      previous: N.confirmed, expected: hop?.target || null, heading: Frames.heading(), topK: 3,
      motion: liveMotion(), previousStepCount: N.prevSteps });
    History.push(res);
    if ($("#autoconfirm").checked && N.confirmer && hop) {
      if ((Frames.kind === "camera") !== !!N.confirmer.tracker) startConfirmer();  // the source changed
      const ev = N.confirmer.onLocalization(performance.now() / 1000, res);
      showPdr();
      if (ev) {
        $("#result").replaceChildren(resultCard(res, { names: N.names, compact: true, onZoom: zoom }));
        onConfirmerEvent(ev);
        return res;
      }
    }
    $("#result").replaceChildren(resultCard(res, { names: N.names, compact: true, onZoom: zoom }));
    History.render($("#history"), N.names);
    const auto = $("#autoconfirm").checked;  // with auto-confirm the confirmer's votes (#pdr) are the temporal check
    const steady = auto ? null : History.steady();
    $("#steady").textContent = steady ? `steady: ${N.names[steady]}` : "";
    if (hop && res.status === "confirmed" && res.best === hop.target) {
      $("#arrive").textContent = `${N.names[hop.target]} recognised${steady === hop.target ? " (steady)" : ""}: confirm arrival when you are there.`;
      $("#arrive").classList.add("match");
    } else if (hop) {
      $("#arrive").classList.remove("match");
      $("#arrive").textContent = `Hop ${N.hop + 1}/${N.route.hops.length}: heading to ${N.names[hop.target]}` +
        (res.expected_rank ? ` (rank ${res.expected_rank}, ${fmtScore(res.expected_score)})` : "") + ".";
    }
    return res;
  } catch (err) {
    $("#error").textContent = err.message;
  } finally {
    N.busy = false;
    $("#once").disabled = false;
  }
}

async function locate() {
  if (N.busy || !N.map) return;
  N.busy = true;
  $("#error").textContent = "";
  try {
    const res = await localize(N.map.map_id, await Frames.burst(2), { heading: Frames.heading(), topK: 3 });
    $("#result").replaceChildren(resultCard(res, { names: N.names, compact: true, onZoom: zoom }));
    if (res.best) $("#start").value = res.best;
    $("#summary").textContent = res.best
      ? `Located at ${N.names[res.best]} (${res.status}, score ${fmtScore(res.candidates[0].score)}).` : "Not located.";
  } catch (err) {
    $("#error").textContent = err.message;
  } finally {
    N.busy = false;
  }
}

function schedule() {
  clearInterval(N.timer);
  N.timer = $("#auto").checked ? setInterval(run, +$("#every").value * 1000) : null;
}

async function init() {
  document.getElementById("top").replaceWith(header("Nav Navigate", "/nav/"));
  $("#source").replaceWith(picker.el);
  $("#map").addEventListener("change", (e) => openMap(e.target.value));
  $("#go").addEventListener("click", plan);
  $("#locate").addEventListener("click", locate);
  $("#once").addEventListener("click", run);
  $("#confirm").addEventListener("click", confirmArrival);
  $("#back").addEventListener("click", backOne);
  $("#say").addEventListener("click", () => current() && speak(hopText(current())));
  $("#auto").addEventListener("change", schedule);
  $("#every").addEventListener("change", schedule);
  healthLine($("#health"));
  setInterval(() => healthLine($("#health")), 30000);
  try {
    await loadMaps($("#map"), decodeURIComponent(location.hash.slice(1)));
    await openMap($("#map").value);
  } catch (err) {
    $("#error").textContent = err.message;
  }
}

init();
