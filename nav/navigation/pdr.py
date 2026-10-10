"""Pedestrian dead reckoning along a route: live steps + heading -> progress towards the next node.

- `StepDetector`: the phone's causal step detector (the Orient app's tracker.ts, which its /map recorder also
  uses) on |accelerationIncludingGravity|, constant memory. (The map pipeline counts steps offline with a
  band-pass filter, nav/mapping/motion.py; this one can't look ahead.)
- `GraphTracker`: follows a route of `Hop`s (route.py). Only the component of each step along the hop's bearing
  counts as progress; it reports the remaining distance, an uncertainty that grows with the distance walked, and
  the probability of being within `arrival_radius_m` of the next node. It only *predicts* arrival: the caller
  still needs a visual or user confirmation (`confirm_node`), which also re-learns the user's step length.
- `replay`: plays a mapping walk back as a user along the map, to see how early or late the tracker would announce
  each node (`nav map pdr-replay`).

Ported from tungsten-united/demo-repository@c5dc26f (navigation, mapping/src/mapper/pdr.py) by LPA.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nav.navigation.route import Hop, hops
from nav.schemas.map import PlaceMap


@dataclass
class StepDetectorConfig:
    """The phone's step detector: the constants of the Orient app (project-description apps/web/src/motion/
    tracker.ts `STEP`, used by guidance and by its /map recorder), so every live step count agrees."""

    lp_ms: float = 60  # low-pass on |acceleration including gravity|
    base_ms: float = 1500  # slow baseline
    var_ms: float = 3000  # running variance of low-pass minus baseline
    frac: float = 0.4  # a step is a bounce above this x the running std ...
    floor: float = 0.1  # ... and above this many m/s^2
    min_gap_ms: float = 400  # at most 2.5 steps/s


class StepDetector:
    """Feed raw accelerometer samples including gravity (device axes, any rate); `update` is True on the sample where
    a step is recognised. It counts what the phone counts: on Itnig 137 steps, hand sway while standing and talking
    included (a phone step is calibrated per map as `MapFrame.live_step_length_m`, video metres per such step)."""

    def __init__(self, config: StepDetectorConfig | None = None) -> None:
        self.c = config or StepDetectorConfig()
        self.last_ms: float | None = None
        self.lp = 0.0
        self.base = 0.0
        self.var = 0.04
        self.armed = True
        self.last_step_ms: float | None = None

    def update(self, t: float, acc: tuple[float, float, float]) -> bool:
        """`t` in seconds."""
        ts = t * 1000.0
        mag = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
        if self.last_ms is None:
            self.lp = self.base = mag
            self.last_ms = ts
            return False
        dt = max(ts - self.last_ms, 1.0)
        self.last_ms = ts

        def k(tau: float) -> float:
            return 1.0 - math.exp(-dt / tau)

        self.lp += k(self.c.lp_ms) * (mag - self.lp)
        self.base += k(self.c.base_ms) * (self.lp - self.base)
        d = self.lp - self.base
        self.var += k(self.c.var_ms) * (d * d - self.var)
        hi = max(self.c.floor, self.c.frac * math.sqrt(self.var))
        if self.armed and d > hi:
            self.armed = False
            if self.last_step_ms is None or ts - self.last_step_ms > self.c.min_gap_ms:
                self.last_step_ms = ts
                return True
        elif not self.armed and d < -0.3 * hi:
            self.armed = True
        return False


def detect_steps(t_s: np.ndarray, acc: np.ndarray, config: StepDetectorConfig | None = None) -> np.ndarray:
    """Step times (s) over a whole recording: `acc` is [N, 3] accelerationIncludingGravity."""
    det = StepDetector(config)
    ok = ~np.isnan(acc).any(axis=1)
    return np.array([t for t, a in zip(t_s[ok], acc[ok]) if det.update(float(t), tuple(a))])


@dataclass
class TrackerState:
    hop_index: int
    progress_m: float  # walked along the current hop
    remaining_m: float
    sigma_m: float  # 1-sigma uncertainty of the remaining distance
    off_course_deg: float  # walking heading minus the hop's bearing (smoothed)
    arrival_probability: float  # P(within arrival_radius_m of the next node)
    step_length_m: float


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class GraphTracker:
    """Follows a user along a route. `on_step(heading_deg)` per detected step (heading in the map frame:
    live heading - heading_offset_deg), `confirm_node(node_id)` when vision or the user confirms the next node.
    Set `heading_offset_deg` from MapFrame.heading_zero_abs_deg for compass maps, or from a first visual fix."""

    route: list[Hop]
    step_length: float = 0.7
    heading_offset_deg: float = 0.0
    arrival_radius_m: float = 2.0
    drift_ratio: float = 0.08  # PDR error grows ~8% of the distance walked ...
    base_sigma_m: float = 0.7  # ... plus a fixed term
    hop_index: int = 0
    progress_m: float = 0.0
    _steps_on_hop: int = 0
    _off_course: float = 0.0
    history: list[tuple[str, int, float]] = field(default_factory=list)

    @property
    def hop(self) -> Hop:
        return self.route[self.hop_index]

    def on_step(self, heading_deg: float) -> TrackerState:
        bearing = self.hop.bearing_deg
        delta = 0.0 if bearing is None else _wrap(heading_deg - self.heading_offset_deg - bearing)
        # Only the component along the hop counts as progress; sideways steps don't.
        self.progress_m += self.step_length * max(math.cos(math.radians(delta)), 0.0)
        self._steps_on_hop += 1
        self._off_course = 0.8 * self._off_course + 0.2 * delta
        return self.state()

    def state(self) -> TrackerState:
        remaining = (self.hop.length_m or 0.0) - self.progress_m
        sigma = self.base_sigma_m + self.drift_ratio * self.progress_m
        p = _phi((self.arrival_radius_m - remaining) / sigma) - _phi((-self.arrival_radius_m - remaining) / sigma)
        return TrackerState(self.hop_index, self.progress_m, remaining, sigma, self._off_course, p, self.step_length)

    def confirm_node(self, node_id: str) -> None:
        """The next node was confirmed: recalibrate the step length and move on to the next hop."""
        if node_id != self.hop.target:
            raise ValueError(f"expected {self.hop.target}, got {node_id}")
        if self._steps_on_hop >= 5 and self.progress_m > 0 and self.hop.length_m:
            # slow walks while filming measured ~0.3 m per step (Itnig), so the floor is 0.25 m
            implied = float(np.clip(self.step_length * self.hop.length_m / self.progress_m, 0.25, 1.0))
            self.step_length = 0.7 * self.step_length + 0.3 * implied
        self.history.append((node_id, self._steps_on_hop, self.progress_m))
        self.hop_index = min(self.hop_index + 1, len(self.route) - 1)
        self.progress_m, self._steps_on_hop, self._off_course = 0.0, 0, 0.0


# --- replay a mapping walk as a user --------------------------------------------------------------------------
@dataclass
class ReplayRow:
    edge: str
    to: str  # name of the node reached
    length_m: float | None
    steps: int
    predicted_remaining_m: float  # the tracker's remaining distance when the walk actually got there
    sigma_m: float
    arrival_probability: float
    first_alert_s: float | None  # when P > 0.5 first, relative to the real arrival (- = early)
    step_length_after: float


def walk_visits(pmap: PlaceMap, rec_id: str) -> list[tuple[float, str]]:
    """The nodes a recording was at, in order, with the time it got to each (the map's evidence times)."""
    visits = sorted((ev.t_ms, n.id) for n in pmap.nodes for ev in n.evidence if ev.rec_id == rec_id)
    seq: list[tuple[float, str]] = []
    for t, nid in visits:
        if not seq or seq[-1][1] != nid:
            seq.append((t, nid))
    return seq


def walk_route(pmap: PlaceMap, rec_id: str) -> list[tuple[Hop, float]]:
    """The hops a recording walked on the map, with the time it reached each hop's target. A step between two nodes
    with no edge between them is left out (walk_visits has every node)."""
    seq = walk_visits(pmap, rec_id)
    by_pair = {(h.source, h.target): h for e in pmap.edges for h in hops(e)}
    return [(by_pair[(a, b)], t) for (_, a), (t, b) in zip(seq, seq[1:]) if (a, b) in by_pair]


def user_steps(rec_dir: Path, step_scale: float = 1.0) -> np.ndarray:
    """Step times (s, recording clock) of a mapping walk replayed as a user: the causal detector on its IMU, and
    1/step_scale steps where the mapper took one (0.85 = a user with 15% shorter steps)."""
    import pyarrow.parquet as pq

    imu = pq.read_table(rec_dir / "imu.parquet").to_pydict()
    t_s = np.asarray(imu["t_ms"], float) / 1000
    acc = np.stack([np.asarray(imu[k], float) for k in ("ax", "ay", "az")], axis=1)
    steps = detect_steps(t_s, acc)
    events, carry = [], 0.0
    for i, t in enumerate(steps):
        carry += 1.0 / step_scale
        n = int(carry)
        carry -= n
        nxt = steps[i + 1] if i + 1 < len(steps) else t + 0.8
        events += [t + k * (nxt - t) / max(n, 1) for k in range(n)]
    return np.asarray(events)


def map_step_length(pmap: PlaceMap) -> float:
    """The tracker's starting step length: the map's live one (metres per step of this module's causal detector),
    else the mapping pipeline's (per step of its band-pass detector, which counts more), else 0.7 m."""
    f = pmap.frame
    return (f.live_step_length_m or f.step_length_m or 0.7) if f else 0.7


def route_start_ms(pmap: PlaceMap, rec_id: str, route: list[tuple[Hop, float]]) -> float:
    """When the walk left the route's first node (its last view of it before reaching the second)."""
    first = route[0][0]
    left = [ev.t_ms for n in pmap.nodes if n.id == first.source for ev in n.evidence
            if ev.rec_id == rec_id and ev.t_ms < route[0][1]]
    return max(left, default=0.0)


def replay(pmap: PlaceMap, rec_dir: Path, step_scale: float = 1.0, heading_bias_deg: float = 0.0,
           step_length: float | None = None) -> list[ReplayRow]:
    """Replay a recording as a user walking the route it walked. `step_scale` 0.85 = a user with 15% shorter steps;
    `heading_bias_deg` a compass bias. Each node is confirmed when the walk really got there (standing in for the
    visual confirmation)."""
    from nav.mapping.prepare import load_motion_only

    rec_id = rec_dir.name
    route = walk_route(pmap, rec_id)
    if not route:
        return []
    events = user_steps(rec_dir, step_scale)
    track = load_motion_only(rec_id, rec_dir.parent)
    nodes = {n.id: n.name for n in pmap.nodes}
    tracker = GraphTracker([h for h, _ in route], step_length=step_length or map_step_length(pmap))
    rows = []
    ei = int(np.searchsorted(events, route_start_ms(pmap, rec_id, route) / 1000))  # steps from leaving the first node
    for k, (hop, arrival_ms) in enumerate(route):
        first_alert, n_steps = None, 0
        while ei < len(events) and events[ei] * 1000 <= arrival_ms:
            t = events[ei]
            heading = track.at(t * 1000)["heading_deg"]
            state = tracker.on_step((heading or 0.0) + heading_bias_deg)
            n_steps += 1
            if first_alert is None and state.arrival_probability > 0.5:
                first_alert = round(t - arrival_ms / 1000, 1)
            ei += 1
        state = tracker.state()
        rows.append(ReplayRow(hop.edge, nodes.get(hop.target, hop.target), hop.length_m, n_steps,
                              round(state.remaining_m, 2), round(state.sigma_m, 2),
                              round(state.arrival_probability, 2), first_alert, round(tracker.step_length, 3)))
        if k + 1 < len(route):
            tracker.confirm_node(hop.target)
    return rows
