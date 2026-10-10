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
