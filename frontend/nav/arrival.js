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

  // --- pdr.StepDetector ----------------------------------------------------------------------------------------
  const STEP = { gravityTau: 1.0, smoothTau: 0.08, minInterval: 0.4, threshold: 0.15, adaptive: 0.3 };
  class StepDetector {
    constructor(c = STEP) { this.c = c; this.gravity = null; this.v = 0; this.prev = [0, 0]; this.lastT = null; this.lastStepT = -1e9; this.peak = 0; }
    // t in seconds, acc = [ax, ay, az] accelerationIncludingGravity (device axes); true on a step
    update(t, acc) {
      const dt = this.lastT == null ? 0 : Math.max(t - this.lastT, 1e-3);
      this.lastT = t;
      if (!this.gravity) { this.gravity = acc.slice(); return false; }
      const kg = dt / (this.c.gravityTau + dt);
      for (let i = 0; i < 3; i++) this.gravity[i] += kg * (acc[i] - this.gravity[i]);
      const g = Math.hypot(...this.gravity) || 1;
      const vertical = (acc[0] * this.gravity[0] + acc[1] * this.gravity[1] + acc[2] * this.gravity[2]) / g - g;
      this.v += dt / (this.c.smoothTau + dt) * (vertical - this.v);
      const [p2, p1] = this.prev;
      this.prev = [p1, this.v];
      const isPeak = p1 > p2 && p1 >= this.v;
      const threshold = Math.max(this.c.threshold, this.c.adaptive * this.peak);
      if (isPeak && p1 > threshold && t - this.lastStepT >= this.c.minInterval) {
        this.lastStepT = t;
        this.peak = this.peak ? 0.8 * this.peak + 0.2 * p1 : p1;
        return true;
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
  const PARAMS = { k: 2, n: 3, gateP: 0.2, gateSigmas: 2.0, looseFrac: 0.5, headingTol: 45.0 };
  const CONFIRM_KINDS = ["arrived", "passed", "manual"];
  const r2 = (x) => Math.round(x * 100) / 100;
  const r3 = (x) => Math.round(x * 1000) / 1000;

  class NodeConfirmer {
    // route: nav-api RouteHop[] (source, target, length_m, bearing_deg, arrival_deg); tracker: GraphTracker | null
    constructor(route, tracker = null, params = {}) {
      this.route = route.slice(); this.tracker = tracker; this.p = { ...PARAMS, ...params };
      this.hopIndex = 0; this.votes = []; this.heading = null; this.offRoute = false;
      this.done = !this.route.length; this.uncertainSent = false; this.events = [];
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
      if (this.done || this.offRoute || !this.tracker) return null;
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
      if (!loc.candidates?.length || loc.best == null) return [null, {}];
      const top = loc.candidates[0], th = loc.thresholds;
      const clear = top.score >= th.confirm_score && (loc.margin || 0) >= th.confirm_margin;
      const ref = top.refs?.[0] || null;
      const info = { best: loc.best, score: top.score, margin: loc.margin, role: ref ? ref.role : null, toward: ref ? ref.toward : null };
      if (!clear) return [null, info];
      if (loc.best === hop.target) {
        if (ref && ref.role === "outgoing") return [nxt && ref.toward === nxt.target ? "pass" : null, info];
        return ["arrive", info];
      }
      if (nxt && loc.best === nxt.target) return [ref && (ref.role === "approach" || ref.role === "at") ? "pass" : null, info];
      if (loc.best !== hop.source) return ["elsewhere", info];
      return [null, info];
    }

    onLocalization(t, loc) {
      if (loc.heading_map_deg != null) this.heading = ((loc.heading_map_deg % 360) + 360) % 360;
      if (this.done || this.offRoute) return null;
      let [vote, info] = this.vote(loc);
      const pdr = this.pdr();
      if (vote === "arrive" && !(this.near(pdr) || (info.role === "at" && this.walkedEnough(pdr)))) vote = "far";
      else if (vote === "pass" && !this.walkedEnough(pdr)) vote = "far";
      this.votes.push(vote);
      if (this.votes.length > this.p.n) this.votes.shift();
      const count = (x) => this.votes.filter((v) => v === x).length;
      const hop = this.hop, nxt = this.nextHop;
      const evidence = { pdr, last: info, votes: this.votes.slice(), heading_map_deg: this.heading };
      if (count("elsewhere") >= this.p.k) { this.offRoute = true; return this.emit("off_route", info.best ?? null, t, evidence); }
      const need = pdr != null ? this.p.k : this.p.n;
      const absDiff = (a, b) => (a == null || b == null ? null : Math.abs(angleDiff(a, b)));
      const arriveDiff = absDiff(this.heading, hop.arrival_deg ?? null);
      evidence.heading_diff_deg = arriveDiff;
      if (count("arrive") >= need && (arriveDiff == null || arriveDiff <= this.p.headingTol)) return this.confirm("arrived", t, evidence);
      if (nxt && count("pass") >= need) {
        const leaveDiff = absDiff(this.heading, nxt.bearing_deg ?? null);
        evidence.heading_diff_deg = leaveDiff;
        if (leaveDiff == null || leaveDiff <= this.p.headingTol) return this.confirm("passed", t, evidence);
      }
      return null;
    }

    confirmManual(t = 0) {
      if (this.done) return null;
      this.offRoute = false;
      return this.confirm("manual", t, { pdr: this.pdr() });
    }
    reroute(route, tracker = null) { const p = this.p; Object.assign(this, new NodeConfirmer(route, tracker, p)); }

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
      this.votes = []; this.uncertainSent = false;
      return ev;
    }
  }

  return { StepDetector, GraphTracker, NodeConfirmer, PARAMS, CONFIRM_KINDS };
})();

if (typeof module !== "undefined") module.exports = ArrivalNav;
