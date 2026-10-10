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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nav.navigation.arrival import CONFIRM_KINDS, ArrivalParams, NodeConfirmer
from nav.navigation.localize import RefIndex, localize
from nav.navigation.pdr import GraphTracker, map_step_length, route_start_ms, user_steps, walk_route
from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Thresholds


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
    conf = NodeConfirmer(hops, tracker, params)
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
    state = {"offset": 0, "conf": conf}  # after an off_route, a confirmer over the rest of the route

    def record(ev) -> None:
        if ev is None:
            return
        k = state["offset"] + ev.hop_index
        slot = per_hop[k]
        if ev.kind in CONFIRM_KINDS:
            slot["kind"], slot["t"], slot["step"] = ev.kind, ev.t, state["conf"].tracker.step_length
        elif ev.kind == "uncertain":
            slot["uncertain"] += 1
        elif ev.kind == "off_route":  # the walk *is* on the route: relocalize where it is and go on
            slot["off_route"] += 1
            old = state["conf"]
            k = state["offset"] + old.hop_index
            state["offset"] = k
            state["conf"] = NodeConfirmer(hops[k:], GraphTracker(hops[k:], step_length=old.tracker.step_length),
                                          params)

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
@dataclass
class OrchestratorResult:
    nodes: list[dict]  # per route node after the start: {node, name, arrival_s, confirmed_s, delay_s}
    wrong: list[dict]  # confirmations of a node the walk was not at: {t_s, node, truth}
    evaluations: int
    confirmations: int

    @property
    def found(self) -> int:
        return sum(n["delay_s"] is not None for n in self.nodes)


ORCH_FPS = 2.0  # the phone uploads the next frame after the previous 202: about 2 a second
ORCH_BUFFER = 4  # the orchestrator localizes with the session's last 4 frames
TRUTH_S = 3.0  # a node counts as "there" from this long before the walk reached it until the next node


def orchestrator_replay(pmap: PlaceMap, map_dir: Path, rec_dir: Path, cache_root: Path, thresholds: Thresholds,
                        use_motion: bool = True, step_scale: float = 1.0,
                        exclude_ms: float = 400.0) -> OrchestratorResult | None:
    """The walk replayed the way the Orient orchestrator calls nav-api (project-description docs/contracts.md §2):
    one frame at a time into a buffer of the last 4, `localize` on each with `previous` = the last confirmed node,
    and the user's node moves on the first `confirmed`. With `use_motion` each frame carries the phone's motion
    (its step detector's stepCount, compass heading, orientation angles, from the recording) and `previous_step_count`
    is the stepCount when `previous` was confirmed."""
    import pyarrow.parquet as pq

    from nav.mapping.prepare import load_walk
    from nav.mapping.select import SelectParams
    from nav.navigation.pdr import StepDetector
    from nav.schemas.navigation import LiveMotion, LiveOrientation

    rec_id = rec_dir.name
    route = walk_route(pmap, rec_id)
    if not route or pmap.embedding is None:
        return None
    index = RefIndex.build(pmap, np.load(map_dir / pmap.embedding.file))
    model = pmap.embedding.model
    walk = load_walk(rec_dir, cache_root, SelectParams())
    own_refs = [r.t_ms for r in pmap.references if r.rec_id == rec_id]
    live = [r for r in _live_frames(rec_dir, walk, cache_root / rec_id)]
    step = max(1, round(LIVE_FPS / ORCH_FPS))
    rows = [r for r in live[::step] if all(abs(r["t_ms"] - t) > exclude_ms for t in own_refs)]
    emb = _live_embeddings([Path(r["file"]) for r in rows], model, cache_root / rec_id)

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

    names = {n.id: n.name for n in pmap.nodes}
    truth = [(route[0][0].source, route_start_ms(pmap, rec_id, route) - 60_000)] + [(h.target, t) for h, t in route]

    def truth_at(t_ms: float) -> set[str]:
        here = {n for k, (n, t0) in enumerate(truth)
                if t0 - TRUTH_S * 1000 <= t_ms < (truth[k + 1][1] if k + 1 < len(truth) else float("inf"))}
        return here

    previous, prev_steps = None, None
    first_seen: dict[str, float] = {}
    wrong, evaluations, confirmations = [], 0, 0
    for i, r in enumerate(rows):
        lo = max(0, i - ORCH_BUFFER + 1)
        motions = [motion_at(x["t_ms"]) for x in rows[lo:i + 1]] if use_motion else None
        loc = localize(index, emb[lo:i + 1], thresholds, previous=previous, model=model, motions=motions,
                       previous_step_count=prev_steps if use_motion else None)
        evaluations += 1
        if loc.status != "confirmed" or loc.best is None:
            continue
        confirmations += 1
        if loc.best not in truth_at(r["t_ms"]):
            wrong.append({"t_s": round(r["t_ms"] / 1000, 1), "node": loc.best, "name": names.get(loc.best, loc.best),
                          "truth": sorted(truth_at(r["t_ms"]))})
        elif loc.best != previous:
            first_seen.setdefault(loc.best, r["t_ms"])
        if loc.best != previous:  # the step count is kept from when the user's node changed, not re-confirmations
            prev_steps = motions[-1].step_count if motions else None
        previous = loc.best

    nodes = []
    for node, t in truth[1:]:
        seen = first_seen.get(node)
        nodes.append({"node": node, "name": names.get(node, node), "arrival_s": round(t / 1000, 1),
                      "confirmed_s": None if seen is None else round(seen / 1000, 1),
                      "delay_s": None if seen is None else round((seen - t) / 1000, 1)})
    return OrchestratorResult(nodes, wrong, evaluations, confirmations)

