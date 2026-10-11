"""`nav map nav-replay`: play a mapping walk back as a user navigating the route it walked, with the full live loop
(docs/navigation_stack.md 8): the causal step detector feeds the dead-reckoning tracker, the walk's frames are
localized against the map's reference images as nav-api does (localize.py), and the node confirmer (arrival.py)
decides arrivals. The truth is when the walk really reached each node (the map's evidence times).

Live frames are the walk's video at LIVE_FPS (the sharpest of each slot, like a phone sampling the camera), and
frames within `exclude_ms` of a reference image taken from the same walk are left out, so a frame is never matched
against itself. Still optimistic: same light, same viewpoint, same phone as the map. A second walk gives honest
numbers.
"""

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nav.navigation.arrival import CONFIRM_KINDS, ArrivalParams, NodeConfirmer, neighbours
from nav.navigation.localize import RefIndex, localize
from nav.navigation.pdr import GraphTracker, map_step_length, route_start_ms, user_steps, walk_route, walk_visits
from nav.navigation.route import hops
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization, Thresholds


@dataclass
class HopResult:
    hop_index: int
    target: str
    name: str
    length_m: float | None
    arrival_s: float  # when the walk reached it (recording clock)
    kind: str  # arrived | passed | manual (a person had to confirm: missed) | -
    delay_s: float | None  # event time - real arrival (+ = late)
    uncertain: int
    off_route: int
    best_score: float | None  # the target's best score in a frame near the real arrival
    step_length_after: float


@dataclass
class NavReplay:
    rows: list[HopResult]
    frames_used: int
    frames_excluded: int

    @property
    def confirmed(self) -> int:
        return sum(r.kind in ("arrived", "passed") for r in self.rows)

    @property
    def premature(self) -> int:  # confirmed well before getting there: a false confirmation
        return sum(r.delay_s is not None and r.delay_s < -PREMATURE_S and r.kind != "manual" for r in self.rows)


PREMATURE_S = 5.0  # a confirmation this long before the real arrival is a false one
GRACE_S = 10.0  # after the real arrival, wait this long (or until the next node) before a person confirms


LIVE_FPS = 2.0
BURST = 2  # frames per /localize call, as the app sends them ...
EVERY_S = 1.0  # ... every this many seconds


def _live_frames(rec_dir: Path, walk, cache: Path) -> list[dict]:
    """[{t_ms (sensor clock), file}] of the walk's video at LIVE_FPS, cached as JPEGs."""
    from PIL import Image

    from nav.mapping.frames import sample_frames
    from nav.mapping.prepare import _video

    video = _video(rec_dir, walk.manifest)
    video_start = next((e.t_ms for e in walk.events if e.type == "video_start"), 0.0)
    key = hashlib.sha256(json.dumps([LIVE_FPS, video_start, video.stat().st_size]).encode()).hexdigest()[:10]
    folder, meta = cache / f"live-{key}", cache / f"live-{key}.json"
    if not meta.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for sf in sample_frames(video, video_start, LIVE_FPS, 768):
            name = f"{round(sf.t_ms):07d}.jpg"
            Image.fromarray(sf.image).save(folder / name, quality=85)
            rows.append({"t_ms": sf.t_ms, "file": str(folder / name)})
        meta.write_text(json.dumps(rows))
    off = walk.calibration.offset_ms
    return [{**r, "t_ms": r["t_ms"] - off} for r in json.loads(meta.read_text())]


def _live_embeddings(paths: list[Path], model: str, cache: Path) -> np.ndarray:
    from nav.mapping.inference import place_files

    key = hashlib.sha256(json.dumps([model, [str(p) for p in paths]]).encode()).hexdigest()[:10]
    path = cache / f"live-place-{key}.npy"
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(place_files(paths, model), np.float32))
    return np.load(path)


def nav_replay(pmap: PlaceMap, map_dir: Path, rec_dir: Path, cache_root: Path, thresholds: Thresholds,
               params: ArrivalParams | None = None, step_scale: float = 1.0, heading_bias_deg: float = 0.0,
               exclude_ms: float = 400.0, burst: int = BURST, every_s: float = EVERY_S) -> NavReplay | None:
    from nav.mapping.prepare import load_walk
    from nav.mapping.select import SelectParams

    rec_id = rec_dir.name
    route = walk_route(pmap, rec_id)
    if not route or pmap.embedding is None:
        return None
    index = RefIndex.build(pmap, np.load(map_dir / pmap.embedding.file))
    model = pmap.embedding.model
    walk = load_walk(rec_dir, cache_root, SelectParams())
    own_refs = [r.t_ms for r in pmap.references if r.rec_id == rec_id]
    live = _live_frames(rec_dir, walk, cache_root / rec_id)
    rows = [r for r in live if all(abs(r["t_ms"] - t) > exclude_ms for t in own_refs)]
    excluded = len(live) - len(rows)
    emb = _live_embeddings([Path(r["file"]) for r in rows], model, cache_root / rec_id)

    def heading(t_ms: float) -> float | None:
        h = walk.track.at(t_ms)["heading_deg"]
        return None if h is None else (h + heading_bias_deg) % 360

    names = {n.id: n.name for n in pmap.nodes}
    hops = [h for h, _ in route]
    arrivals = [t for _, t in route]
    tracker = GraphTracker(hops, step_length=map_step_length(pmap))
    nb = neighbours(pmap)
    conf = NodeConfirmer(hops, tracker, params, nb)
    start = route_start_ms(pmap, rec_id, route)
    # Every every_s the app grabs `burst` frames: the next usable video frames from that moment (at LIVE_FPS they
    # span (burst - 1) / LIVE_FPS s; the phone's 250 ms apart are closer, so this is a little harsher).
    bursts, i, tick = [], 0, start
    end = rows[-1]["t_ms"] if rows else start
    while tick <= end:
        while i < len(rows) and rows[i]["t_ms"] < tick:
            i += 1
        group = [j for j in range(i, min(i + burst, len(rows))) if rows[j]["t_ms"] <= tick + burst * 1000 / LIVE_FPS]
        if group:
            bursts.append((rows[group[-1]]["t_ms"], "frame", group))
        tick += every_s * 1000
    timeline = sorted([(float(t) * 1000, "step", None) for t in user_steps(rec_dir, step_scale) if t * 1000 >= start]
                      + bursts, key=lambda x: x[0])
    per_hop = [{"kind": "-", "t": None, "uncertain": 0, "off_route": 0, "score": None, "step": None} for _ in hops]
    state = {"offset": 0, "conf": conf}

    def record(ev) -> None:
        if ev is None:
            return
        k = state["offset"] + ev.hop_index
        slot = per_hop[k]
        if ev.kind in CONFIRM_KINDS:
            slot["kind"], slot["t"], slot["step"] = ev.kind, ev.t, state["conf"].tracker.step_length
        elif ev.kind == "uncertain":
            slot["uncertain"] += 1
        elif ev.kind == "off_route":  # a warning (the walk *is* on the route): the confirmer goes on
            slot["off_route"] += 1

    for t_ms, kind, i in timeline:
        c = state["conf"]
        if c.done:
            break
        k = state["offset"] + c.hop_index
        # a person confirms when the confirmer missed it (the walk is GRACE_S past it, or at the next node already)
        deadline = min(arrivals[k] + GRACE_S * 1000, arrivals[k + 1] if k + 1 < len(arrivals) else float("inf"))
        if t_ms > deadline:
            record(c.confirm_manual(t_ms / 1000))
            continue
        if kind == "step":
            record(c.on_step(heading(t_ms), t_ms / 1000))
        else:
            loc = localize(index, emb[i], thresholds, previous=c.previous, expected=c.expected,
                           heading_abs_deg=heading(t_ms), model=model)
            if abs(t_ms - arrivals[k]) <= 3000 and loc.expected_score is not None:
                per_hop[k]["score"] = max(per_hop[k]["score"] or 0.0, loc.expected_score)
            record(c.on_localization(t_ms / 1000, loc))

    out = []
    for k, (hop, arrival_ms) in enumerate(route):
        slot = per_hop[k]
        delay = None if slot["t"] is None else round(slot["t"] - arrival_ms / 1000, 1)
        out.append(HopResult(k, hop.target, names.get(hop.target, hop.target), hop.length_m,
                             round(arrival_ms / 1000, 1), slot["kind"], delay, slot["uncertain"], slot["off_route"],
                             None if slot["score"] is None else round(slot["score"], 3),
                             round(slot["step"] or state["conf"].tracker.step_length, 3)))
    return NavReplay(out, len(rows), excluded)


# --- the orchestrator's loop -----------------------------------------------------------------------------------
# Orient's orchestrator (project-description docs/contracts.md §2) keeps the navigation state; nav-api stays stateless.
# It localizes the user, asks for one route, and follows it: each /localize result is a vote, and the user moves to
# a node ahead on the route (the hop's target, or one further on) once enough results agree, so a node the camera
# missed doesn't stop the route. Lost, it starts over. This is a model of that loop (the orchestrator's
# src/pipeline.rs has the same rules), to tune its parameters on mapping walks.
@dataclass
class LoopParams:
    """Defaults from replaying both Itnig walks (map build 20261010T111114Z): the only settings with no wrong move
    (a node the walk wasn't at); 2 of 3, or a margin of 0.02, moved the user to the next node 10 s early. `elsewhere`
    from confirmed results only and `lost_s` (2026-10-10, build 20261010T131002Z, its four walks at 1.5 and 3.5 fps,
    with and without motion): never worse than an `elsewhere` from any lead and 10 `lost` calls in a row."""

    burst: int = 1  # new frames per /localize call: each frame is sent once, so the votes are over different frames
    k: int = 3  # votes needed ...
    n: int = 4  # ... among the last n calls
    margin: float = 0.04  # a result votes only when its best node leads the second by this much
    lost_margin: float | None = 0.06  # locating, a `lost` one too when it leads by this much (None: never), as the
    #                                   orchestrator's NAV_LOCATE_LOST_MARGIN
    lost_s: float = 6.0  # following, nothing pointing at the route (lost, or a node off it leading) this long:
    #                      start over


def leader(loc: Localization, margin: float, lost_margin: float | None) -> str | None:
    """The node a result points at, if it leads clearly: by `margin`, or by `lost_margin` when every score is low
    (`lost`; None: a lost result points nowhere), and nav-api doesn't mark it farther than the user walked. Live frames
    often score under the lost score while the right node keeps leading (Glories Basement, 2026-10-10: 389 of 886)."""
    if loc.best is None:
        return None
    needed = margin if loc.status != "lost" else lost_margin
    if needed is None or (loc.margin or 0.0) < needed:
        return None
    top = loc.candidates[0] if loc.candidates else None
    return None if top is not None and top.node == loc.best and top.plausible is False else loc.best


def locate_vote(loc: Localization, margin: float, lost_margin: float | None = None) -> str | None:
    """While locating: the node a result votes for, if any. Relocating (sent with the last node and its step count), a
    node farther than the user walked since doesn't vote."""
    return leader(loc, margin, lost_margin)


def follow_vote(loc: Localization, source: str, ahead: list[str], margin: float) -> str:
    """Following from `source`, with `ahead` the route's nodes still to reach (the hop's target first): the node the
    result votes for, "elsewhere" or "none".

    - A node ahead on the route, when it leads: the hop's target, or one further on, because the user walked past a
      node the camera missed. Hop by hop only, a missed node meant starting over (Itnig, 2026-10-10: 8 of 15 restarts
      where another node won were a node further on). nav-api's walked-distance gate (`plausible`) keeps a node seen
      from afar from counting before the user walked that far.
    - "elsewhere": a node off the route, confirmed. An uncertain one is mostly noise (Itnig, 2026-10-10: it dropped the
      route in 15 of 20 restarts); a long stretch of them still starts over (`LoopParams.lost_s`)."""
    lead = leader(loc, margin, None)
    if lead in ahead:
        return lead
    if lead is not None and lead != source and loc.status == "confirmed":
        return "elsewhere"
    return "none"


@dataclass
class LoopEvent:
    kind: str  # located | reached | arrived | lost | no_route
    node: str | None
    t_s: float


class OrchestratorLoop:
    """One session's navigation: locating (no path yet) or following hop `hop` of `path` (route Hops). `path_fn(node,
    t_s)` gives the route from a node to the goal, or None."""

    def __init__(self, goal: str, path_fn, params: LoopParams | None = None):
        self.goal, self.path_fn, self.p = goal, path_fn, params or LoopParams()
        self.node: str | None = None  # the last node reached
        self.node_steps: int | None = None  # the phone's stepCount then: previous_step_count
        self.path: list | None = None
        self.hop = 0
        self.votes: deque = deque(maxlen=self.p.n)
        self.off_hop_since: float | None = None  # following: t_s of the first of the results in a row off the hop
        self.arrived = False

    def request(self) -> dict:
        """What the next /localize call sends besides the frames and their motion. Relocating after lost, the last
        node and its step count go along, so nav-api marks the nodes the user can't have walked to yet."""
        if self.path is None:
            return {"previous": self.node, "expected": None, "previous_step_count": self.node_steps}
        h = self.path[self.hop]
        return {"previous": h.source, "expected": h.target, "previous_step_count": self.node_steps}

    def ahead(self) -> list[str]:
        """The route's nodes still to reach, the hop's target first."""
        return [h.target for h in self.path[self.hop:]]

    def on_localization(self, loc: Localization, step_count: int | None, t_s: float) -> LoopEvent | None:
        if self.arrived:
            return None
        if self.path is None:
            return self._locate(loc, step_count, t_s)
        source, ahead = self.path[self.hop].source, self.ahead()
        vote = follow_vote(loc, source, ahead, self.p.margin)
        self.votes.append(vote)
        # pointing at the route: the hop's source or target leads (not lost), or a node ahead votes
        on_route = (loc.status != "lost" and loc.best in (source, ahead[0])) or vote in ahead
        self.off_hop_since = None if on_route else (t_s if self.off_hop_since is None else self.off_hop_since)
        if vote in ahead and sum(v == vote for v in self.votes) >= self.p.k:
            return self._reach(vote, step_count, t_s)
        off_hop_s = 0.0 if self.off_hop_since is None else t_s - self.off_hop_since
        if sum(v == "elsewhere" for v in self.votes) >= self.p.k or off_hop_s >= self.p.lost_s:
            self.path, self.off_hop_since = None, None
            self.votes.clear()
            return LoopEvent("lost", self.node, t_s)
        return None

    def _locate(self, loc: Localization, step_count: int | None, t_s: float) -> LoopEvent | None:
        vote = loc.best if loc.status == "confirmed" else locate_vote(loc, self.p.margin, self.p.lost_margin)
        self.votes.append(vote)
        if vote is None or (loc.status != "confirmed" and sum(v == vote for v in self.votes) < self.p.k):
            return None
        self.votes.clear()
        return self._located(vote, step_count, t_s)

    def _located(self, node: str, step_count: int | None, t_s: float) -> LoopEvent:
        """The user is at `node`: arrived, or one route from there to the goal."""
        self.node, self.node_steps = node, step_count
        if node == self.goal:
            self.arrived = True
            return LoopEvent("arrived", node, t_s)
        path = self.path_fn(node, t_s)
        if not path:
            return LoopEvent("no_route", node, t_s)
        self.path, self.hop, self.off_hop_since = path, 0, None
        return LoopEvent("located", node, t_s)

    def _reach(self, node: str, step_count: int | None, t_s: float) -> LoopEvent:
        """The user is at `node`, a node ahead on the route: the route goes on from there, past any it skipped."""
        self.node, self.node_steps = node, step_count
        self.votes.clear()
        self.off_hop_since = None
        if node == self.goal:
            self.arrived = True
            return LoopEvent("arrived", node, t_s)
        self.hop = next(k for k in range(self.hop, len(self.path)) if self.path[k].target == node) + 1
        return LoopEvent("reached", node, t_s)


@dataclass
class OrchestratorResult:
    events: list[dict]  # {t_s, kind, node, name, truth, ok}: what the loop decided, and where the walk was
    nodes: list[dict]  # per node of the walk's route after the start: {node, name, arrival_s, reached_s, delay_s}
    evaluations: int
    goal: str

    @property
    def wrong(self) -> list[dict]:  # the loop put the user at a node the walk was not at, nor had just left
        return [e for e in self.events if e["kind"] in MOVES and not e["ok"]]

    @property
    def relocations(self) -> int:
        return sum(e["kind"] == "lost" for e in self.events)

    @property
    def found(self) -> int:
        return sum(n["reached_s"] is not None for n in self.nodes)


MOVES = ("located", "reached", "arrived")
ORCH_FPS = 1.5  # the phone uploads a frame, waits for the 202, then 500 ms: about 1.5 a second (Itnig logs)
TRUTH_S = 3.0  # a node counts as "there" from this long before the walk reached it until the next node
LATE_S = 10.0  # a node the walk left less than this long ago is a late move, not a wrong one
LEAD_S = 5.0  # navigation starts this long before the walk leaves its first node
GOAL_TAIL_S = 6.0  # an episode goes on this long past its goal (or until the walk's next node), to arrive


@dataclass
class Track:
    """A mapping walk as the frames a phone would send (one every 1/fps s, the sharpest of the slot), with their
    place embeddings and the phone's motion at each (its step detector's stepCount, compass, orientation angles)."""

    rec_id: str
    t_ms: np.ndarray  # [F] recording clock
    emb: np.ndarray  # [F, D]
    motions: list  # LiveMotion per frame
    visits: list[tuple[float, str]]  # the nodes the walk was at (walk_visits)


def load_track(pmap: PlaceMap, rec_dir: Path, cache_root: Path, model: str, fps: float = ORCH_FPS,
               step_scale: float = 1.0, exclude_ms: float | None = None) -> Track:
    """`exclude_ms`: leave out frames this close to a reference image the map took from this walk (a frame is never
    matched against itself); None keeps every frame (for an index without this walk's references)."""
    import pyarrow.parquet as pq

    from nav.mapping.prepare import load_walk
    from nav.mapping.select import SelectParams
    from nav.navigation.pdr import StepDetector
    from nav.schemas.navigation import LiveMotion, LiveOrientation

    rec_id = rec_dir.name
    walk = load_walk(rec_dir, cache_root, SelectParams())
    live = _live_frames(rec_dir, walk, cache_root / rec_id)
    emb = _live_embeddings([Path(r["file"]) for r in live], model, cache_root / rec_id)
    own_refs = [r.t_ms for r in pmap.references if r.rec_id == rec_id] if exclude_ms is not None else []
    keep, next_t = [], live[0]["t_ms"] if live else 0.0  # the frame the phone grabs every 1/fps s
    for i, r in enumerate(live):
        if r["t_ms"] >= next_t:
            next_t += 1000 / fps
            if all(abs(r["t_ms"] - t) > exclude_ms for t in own_refs):
                keep.append(i)

    # the phone's sensors along the walk
    imu = pq.read_table(rec_dir / "imu.parquet").to_pydict()
    det, step_times, carry = StepDetector(), [], 0.0
    for t, ax, ay, az in zip(imu["t_ms"], imu["ax"], imu["ay"], imu["az"]):
        if None not in (ax, ay, az) and det.update(t / 1000, (ax, ay, az)):
            carry += 1.0 / step_scale
            while carry >= 1.0:
                step_times.append(t)
                carry -= 1.0
    step_times = np.asarray(step_times)
    ori = pq.read_table(rec_dir / "orientation.parquet").to_pydict()
    ori_t = np.asarray(ori["t_ms"], float)

    def motion_at(t_ms: float) -> LiveMotion:
        i = int(np.clip(np.searchsorted(ori_t, t_ms), 0, len(ori_t) - 1))
        h = ori["heading_deg"][i]
        a, b, g = ori["alpha"][i], ori["beta"][i], ori["gamma"][i]
        recent = int(((step_times > t_ms - 4000) & (step_times <= t_ms)).sum())
        return LiveMotion(step_count=int((step_times <= t_ms).sum()), cadence_hz=recent / 4 if recent >= 3 else 0.0,
                          heading_deg=h,
                          heading_source=None if h is None else "orientation_absolute",
                          orientation=None if None in (a, b, g) else LiveOrientation(alpha=a, beta=b, gamma=g,
                                                                                      absolute=True))

    t_ms = np.asarray([live[i]["t_ms"] for i in keep], float)
    return Track(rec_id, t_ms, emb[keep] if keep else emb[:0], [motion_at(t) for t in t_ms],
                 walk_visits(pmap, rec_id))


@dataclass
class Episode:
    """A user guided to `goal`: the frames the phone sends, in order (from one walk, or pieces of several stitched
    together: episodes.py), and where the user was (`truth`: each node and when they got there, the first at -inf)."""

    t_ms: np.ndarray  # [F] episode clock
    emb: np.ndarray  # [F, D]
    motions: list  # LiveMotion per frame, stepCount counted from the episode's start
    recs: list[str]  # the walk each frame comes from
    keys: list  # a stable id per frame (rec_id, recording t_ms): caches localizations across runs
    truth: list[tuple[str, float]]
    goal: str


def walk_episode(pmap: PlaceMap, track: Track) -> Episode | None:
    """The walk replayed as it went: from just before it left its route's first node to its first visit of the last
    node it walked to, and GOAL_TAIL_S on (a loop that missed the goal is not scored on where the walk went next)."""
    rec_id = track.rec_id
    route = walk_route(pmap, rec_id)
    if not route:
        return None
    start = route_start_ms(pmap, rec_id, route) - LEAD_S * 1000
    goal = route[-1][0].target
    truth = [(route[0][0].source, -float("inf"))]
    for t, n in track.visits:
        if t >= route[0][1] and truth[-1][0] != goal:
            truth.append((n, t))
    after = [t for t, _ in track.visits if t > truth[-1][1]]
    end = min(after[0] if after else float("inf"), truth[-1][1] + GOAL_TAIL_S * 1000)
    sel = np.flatnonzero((track.t_ms >= start) & (track.t_ms < end))
    return Episode(track.t_ms[sel], track.emb[sel], [track.motions[i] for i in sel], [rec_id] * len(sel),
                   [(rec_id, float(track.t_ms[i])) for i in sel], truth, goal)


def run_episode(pmap: PlaceMap, index: RefIndex | dict, episode: Episode, thresholds: Thresholds, make_loop,
                use_motion: bool = True, burst: int = 1, cache: dict | None = None) -> OrchestratorResult:
    """Guide the episode's user with the loop `make_loop(goal, path_fn)` (OrchestratorLoop, belief.BeliefLoop): each
    `burst` frames go to localize() as nav-api would, with what the loop asks for. `index`: one RefIndex, or one per
    rec_id (each walk matched against the others' references). The user follows the instructions: the route from a
    node is the rest of the episode's own route from there (or, where the user isn't, the shortest observed route).
    `cache` keeps localizations across runs of the same episodes (tuning: most calls repeat)."""
    from nav.navigation.route import plan_route

    model = pmap.embedding.model if pmap.embedding else ""
    names = {n.id: n.name for n in pmap.nodes}
    truth, goal = episode.truth, episode.goal
    by_pair = {(h.source, h.target): h for e in pmap.edges for h in hops(e)}

    def truth_at(t_ms: float) -> set[str]:
        return {n for k, (n, t0) in enumerate(truth)
                if t0 - TRUTH_S * 1000 <= t_ms < (truth[k + 1][1] if k + 1 < len(truth) else float("inf"))}

    def path_fn(node: str, t_s: float) -> list | None:
        for k, (n, t0) in enumerate(truth[:-1]):
            if n == node and t0 - TRUTH_S * 1000 <= t_s * 1000 < truth[k + 1][1]:
                rest = [by_pair.get((a, b)) for (a, _), (b, _) in zip(truth[k:], truth[k + 1:])]
                if all(rest):
                    return rest
        return plan_route(pmap, node, goal, "observed")

    loop = make_loop(goal, path_fn)
    thr_key = tuple(thresholds.model_dump().values())
    events, evaluations = [], 0
    for lo in range(0, len(episode.t_ms) - burst + 1, burst):
        group = list(range(lo, lo + burst))
        t_ms = float(episode.t_ms[group[-1]])
        motions = [episode.motions[j] for j in group] if use_motion else None
        req = loop.request()
        steps = motions[-1].step_count if motions else None
        prev_steps = req["previous_step_count"] if use_motion else None
        key = (tuple(episode.keys[j] for j in group), req["previous"], req["expected"], req.get("top_k", 5),
               None if prev_steps is None or steps is None else steps - prev_steps, thr_key)
        loc = cache.get(key) if cache is not None else None
        if loc is None:
            idx = index[episode.recs[group[-1]]] if isinstance(index, dict) else index
            loc = localize(idx, episode.emb[group], thresholds, previous=req["previous"], expected=req["expected"],
                           model=model, motions=motions, previous_step_count=prev_steps, top_k=req.get("top_k", 5))
            if cache is not None:
                cache[key] = loc
        evaluations += 1
        ev = loop.on_localization(loc, steps, t_ms / 1000)
        if ev is not None:
            here = truth_at(t_ms)
            late = ev.node not in here and any(n == ev.node for n in truth_at(t_ms - LATE_S * 1000))
            events.append({"t_s": round(ev.t_s, 1), "kind": ev.kind, "node": ev.node,
                           "name": names.get(ev.node, ev.node), "truth": sorted(here), "late": late,
                           "ok": ev.kind not in MOVES or ev.node in here or late})
        if loop.arrived:
            break

    nodes = []
    for k, (node, t) in enumerate(truth[1:], start=1):
        end = truth[k + 1][1] if k + 1 < len(truth) else float("inf")
        seen = next((e["t_s"] for e in events if e["kind"] in MOVES and e["node"] == node
                     and t - TRUTH_S * 1000 <= e["t_s"] * 1000 < end), None)
        nodes.append({"node": node, "name": names.get(node, node), "arrival_s": round(t / 1000, 1),
                      "reached_s": seen, "delay_s": None if seen is None else round(seen - t / 1000, 1)})
    return OrchestratorResult(events, nodes, evaluations, goal)


def ref_indexes(pmap: PlaceMap, map_dir: Path, holdout: bool, memory: bool = True) -> RefIndex | dict:
    """The map's reference index (with its visual memory, if it has one and `memory`); with `holdout`, one per walk
    of the map without that walk's references and memory frames."""
    from nav.mapping.memory import load_memory

    emb = np.load(map_dir / pmap.embedding.file) if pmap.embedding else np.zeros((0, 0), np.float32)
    mem = load_memory(map_dir, pmap.memory) if memory and pmap.memory else None
    if not holdout:
        return RefIndex.build(pmap, emb, memory=mem)
    return {rec: RefIndex.build(pmap, emb, exclude_rec=rec, memory=mem) for rec in pmap.recordings}


def loop_factory(pmap: PlaceMap, kind: str, params=None):
    """make_loop for run_episode: "belief" (belief.BeliefLoop, BeliefParams) or "votes" (OrchestratorLoop,
    LoopParams)."""
    if kind == "votes":
        return lambda goal, path_fn: OrchestratorLoop(goal, path_fn, params)
    from nav.navigation.belief import BeliefLoop

    return lambda goal, path_fn: BeliefLoop(goal, path_fn, pmap, params)


def orchestrator_replay(pmap: PlaceMap, map_dir: Path, rec_dir: Path, cache_root: Path, thresholds: Thresholds,
                        params=None, use_motion: bool = True, step_scale: float = 1.0,
                        exclude_ms: float = 400.0, fps: float = ORCH_FPS, loop: str = "belief",
                        holdout: bool = False) -> OrchestratorResult | None:
    """The walk replayed as a user guided by the orchestrator's loop (`loop`: "belief" or "votes", with its params)
    to the last node it walked to, who follows the instructions. Frames are the walk's video at `fps`, each sent
    once; with `use_motion` each carries the phone's motion. `holdout`: match against the other walks' references
    only (honest; needs a map of 2+ walks), else against all, leaving out frames within `exclude_ms` of this walk's
    own references."""
    if pmap.embedding is None or not walk_route(pmap, rec_dir.name):
        return None
    track = load_track(pmap, rec_dir, cache_root, pmap.embedding.model, fps, step_scale,
                       None if holdout else exclude_ms)
    episode = walk_episode(pmap, track)
    index = ref_indexes(pmap, map_dir, holdout)
    return run_episode(pmap, index, episode, thresholds, loop_factory(pmap, loop, params), use_motion)
