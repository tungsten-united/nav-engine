// Live navigation on the phone: steps (StepDetector), dead reckoning along the route (GraphTracker) and node
// confirmation from consecutive nav-api localizations (NodeConfirmer). A 1:1 port of nav/navigation/pdr.py and
// nav/navigation/arrival.py (same constants, same rule order; tests/test_arrival_js.py checks they agree), so the
// app can copy it. Plain script: globals in the browser, module.exports under node.
"use strict";

const ArrivalNav = (() => {
  const wrap = (a) => ((((a + 180) % 360) + 360) % 360) - 180;
  const angleDiff = (a, b) => { let d = (((a - b) % 360) + 360) % 360; return d > 180 ? d - 360 : d; };
  const phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  function erf(x) { // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  // --- pdr.StepDetector: the phone's (Orient app tracker.ts, also its /map recorder) ----------------------------
  const STEP = { lpMs: 60, baseMs: 1500, varMs: 3000, frac: 0.4, floor: 0.1, minGapMs: 400 };
  class StepDetector {
    constructor(c = STEP) { this.c = c; this.last = null; this.lp = 0; this.base = 0; this.variance = 0.04; this.armed = true; this.lastStep = null; }
    // t in seconds, acc = [ax, ay, az] accelerationIncludingGravity (device axes); true on a step
    update(t, acc) {
      const ts = t * 1000, mag = Math.hypot(acc[0], acc[1], acc[2]);
      if (this.last == null) { this.lp = this.base = mag; this.last = ts; return false; }
      const dt = Math.max(ts - this.last, 1), k = (tau) => 1 - Math.exp(-dt / tau);
      this.last = ts;
      this.lp += k(this.c.lpMs) * (mag - this.lp);
      this.base += k(this.c.baseMs) * (this.lp - this.base);
      const d = this.lp - this.base;
      this.variance += k(this.c.varMs) * (d * d - this.variance);
      const hi = Math.max(this.c.floor, this.c.frac * Math.sqrt(this.variance));
      if (this.armed && d > hi) {
        this.armed = false;
        if (this.lastStep == null || ts - this.lastStep > this.c.minGapMs) { this.lastStep = ts; return true; }
      } else if (!this.armed && d < -0.3 * hi) {
        this.armed = true;
      }
      return false;
    }
  }

  // --- pdr.GraphTracker ----------------------------------------------------------------------------------------
  class GraphTracker {
    constructor(route, { stepLength = 0.7, headingOffset = 0, arrivalRadius = 2.0, driftRatio = 0.08, baseSigma = 0.7 } = {}) {
      Object.assign(this, { route, stepLength, headingOffset, arrivalRadius, driftRatio, baseSigma });
      this.hopIndex = 0; this.progress = 0; this.stepsOnHop = 0; this.offCourse = 0;
    }
    get hop() { return this.route[this.hopIndex]; }
    onStep(headingDeg) {
      const b = this.hop.bearing_deg;
      const delta = b == null ? 0 : wrap(headingDeg - this.headingOffset - b);
      this.progress += this.stepLength * Math.max(Math.cos(delta * Math.PI / 180), 0);
      this.stepsOnHop += 1;
      this.offCourse = 0.8 * this.offCourse + 0.2 * delta;
      return this.state();
    }
    state() {
      const remaining = (this.hop.length_m || 0) - this.progress;
      const sigma = this.baseSigma + this.driftRatio * this.progress;
      const p = phi((this.arrivalRadius - remaining) / sigma) - phi((-this.arrivalRadius - remaining) / sigma);
      return { hopIndex: this.hopIndex, progress: this.progress, remaining, sigma, offCourse: this.offCourse, p, stepLength: this.stepLength };
    }
    confirmNode(nodeId) {
      if (nodeId !== this.hop.target) throw new Error(`expected ${this.hop.target}, got ${nodeId}`);
      if (this.stepsOnHop >= 5 && this.progress > 0 && this.hop.length_m) {
        const implied = Math.min(1.0, Math.max(0.25, this.stepLength * this.hop.length_m / this.progress));
        this.stepLength = 0.7 * this.stepLength + 0.3 * implied;
      }
      this.hopIndex = Math.min(this.hopIndex + 1, this.route.length - 1);
      this.progress = 0; this.stepsOnHop = 0; this.offCourse = 0;
    }
  }

  // --- arrival.NodeConfirmer -----------------------------------------------------------------------------------
  // score / margin: the confirmer's own vote thresholds (not nav-api's confirm_score); headingTol: passing only.
  const PARAMS = { k: 2, n: 3, score: 0.30, margin: 0.04, gateP: 0.2, gateSigmas: 2.0, looseFrac: 0.5, headingTol: 90.0 };
  const CONFIRM_KINDS = ["arrived", "passed", "manual"];
  const r2 = (x) => Math.round(x * 100) / 100;
  const r3 = (x) => Math.round(x * 1000) / 1000;

  // node -> the nodes a link joins it to, either direction (arrival.neighbours): map.links, or map.edges before links
  function neighbours(edges) {
    const out = {};
    const add = (a, b) => { (out[a] ||= new Set()).add(b); };
    for (const e of edges || []) { add(e.source, e.target); add(e.target, e.source); }
    return out;
  }

  class NodeConfirmer {
    // route: nav-api RouteHop[] (source, target, length_m, bearing_deg, arrival_deg); tracker: GraphTracker | null;
    // nbrs: neighbours(map.edges) or null
    constructor(route, tracker = null, params = {}, nbrs = null) {
      this.route = route.slice(); this.tracker = tracker; this.p = { ...PARAMS, ...params }; this.nbrs = nbrs || {};
      this.hopIndex = 0; this.votes = []; this.heading = null; this.walkGate = false;
      this.done = !this.route.length; this.uncertainSent = false; this.offRouteSent = false; this.events = [];
    }
    get hop() { return this.done ? null : this.route[this.hopIndex]; }
    get nextHop() { return !this.done && this.hopIndex + 1 < this.route.length ? this.route[this.hopIndex + 1] : null; }
    get previous() { return !this.route.length ? null : this.done ? this.route.at(-1).target : this.route[this.hopIndex].source; }
    get expected() { return this.done ? null : this.route[this.hopIndex].target; }

    pdr() {
      if (!this.tracker || !this.hop || !this.hop.length_m) return null;
      const s = this.tracker.state();
      return { p: r3(s.p), progress_m: r2(s.progress), remaining_m: r2(s.remaining), sigma_m: r2(s.sigma) };
    }
    near(pdr) { return pdr == null || pdr.p >= this.p.gateP || Math.abs(pdr.remaining_m) <= this.p.gateSigmas * pdr.sigma_m; }
    walkedEnough(pdr) { return pdr == null || pdr.progress_m >= (1 - this.p.looseFrac) * this.hop.length_m; }

    onStep(headingMapDeg, t = 0) {
      if (headingMapDeg != null) this.heading = ((headingMapDeg % 360) + 360) % 360;
      if (this.done || !this.tracker) return null;
      this.tracker.onStep(this.heading != null ? this.heading : (this.hop.bearing_deg || 0));
      const pdr = this.pdr();
      if (pdr && !this.uncertainSent && pdr.remaining_m < -this.p.gateSigmas * pdr.sigma_m) {
        this.uncertainSent = true;
        return this.emit("uncertain", this.hop.target, t, { pdr, why: "walked past it, not recognised" });
      }
      return null;
    }

    vote(loc) {
      const hop = this.hop, nxt = this.nextHop;
      if (!hop || !loc.candidates?.length || loc.best == null) return [null, {}];
      const top = loc.candidates[0];
      const clear = top.score >= this.p.score && (loc.margin || 0) >= this.p.margin;
      const ref = top.refs?.[0] || null;
      const info = { best: loc.best, score: top.score, margin: loc.margin, role: ref ? ref.role : null,
                     toward: ref ? ref.toward : null, plausible: top.plausible ?? null };
      if (!clear) return [null, info];
      if (loc.best === hop.target) {
        if (ref && ref.role === "outgoing") return [nxt && ref.toward === nxt.target ? "pass" : null, info];
        return ["arrive", info];
      }
      if (nxt && loc.best === nxt.target) return [ref && (ref.role === "approach" || ref.role === "at") ? "pass" : null, info];
      const onHop = [hop.source, hop.target, ...(nxt ? [nxt.target] : [])];
      if (onHop.includes(loc.best) || onHop.some((x) => this.nbrs[x]?.has(loc.best))) return [null, info];
      return ["elsewhere", info];
    }

    gate(vote, info, pdr) {
      if (info.plausible != null) return info.plausible;  // nav-api's walked-distance gate (steps, no compass)
      if (vote === "arrive") return this.near(pdr) || (info.role === "at" && this.walkedEnough(pdr));
      return this.walkedEnough(pdr);
    }

    onLocalization(t, loc) {
      if (loc.heading_map_deg != null) this.heading = ((loc.heading_map_deg % 360) + 360) % 360;
      if (this.done) return null;
      let [vote, info] = this.vote(loc);
      if (info.plausible != null) this.walkGate = true;
      const pdr = this.pdr();
      if ((vote === "arrive" || vote === "pass") && !this.gate(vote, info, pdr)) vote = "far";
      this.votes.push(vote);
      if (this.votes.length > this.p.n) this.votes.shift();
      const count = (x) => this.votes.filter((v) => v === x).length;
      const nxt = this.nextHop;
      const evidence = { pdr, last: info, votes: this.votes.slice(), heading_map_deg: this.heading };
      const need = this.walkGate || pdr != null ? this.p.k : this.p.n;
      if (count("arrive") >= need) return this.confirm("arrived", t, evidence);
      if (nxt && count("pass") >= need) {
        const leaveDiff = this.heading == null || nxt.bearing_deg == null ? null : Math.abs(angleDiff(this.heading, nxt.bearing_deg));
        evidence.heading_diff_deg = leaveDiff;
        if (leaveDiff == null || leaveDiff <= this.p.headingTol) return this.confirm("passed", t, evidence);
      }
      if (count("elsewhere") >= this.p.k && !this.offRouteSent) {
        this.offRouteSent = true;  // a warning: confirmation goes on
        return this.emit("off_route", info.best ?? null, t, evidence);
      }
      return null;
    }

    confirmManual(t = 0) {
      if (this.done) return null;
      return this.confirm("manual", t, { pdr: this.pdr() });
    }
    reroute(route, tracker = null) { const p = this.p, nb = this.nbrs; Object.assign(this, new NodeConfirmer(route, tracker, p, nb)); }

    emit(kind, node, t, evidence) {
      const ev = { kind, node, hop_index: this.hopIndex, t, evidence };
      this.events.push(ev);
      return ev;
    }
    confirm(kind, t, evidence) {
      const target = this.hop.target;
      const ev = this.emit(kind, target, t, evidence);
      if (this.tracker) { try { this.tracker.confirmNode(target); } catch {} }
      if (this.hopIndex + 1 >= this.route.length) this.done = true; else this.hopIndex += 1;
      this.votes = []; this.uncertainSent = false; this.offRouteSent = false;
      return ev;
    }
  }

  return { StepDetector, GraphTracker, NodeConfirmer, PARAMS, CONFIRM_KINDS, neighbours };
})();

if (typeof module !== "undefined") module.exports = ArrivalNav;
